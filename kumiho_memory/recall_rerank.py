"""Post-recall reranking: cross-encoder relevance, recency decay, MMR diversity.

Layered on top of evidence-grade weighting to turn first-stage hybrid recall
into a properly ordered result set — the reranking stack that peer memory
systems (Zep, mem0) ship and Kumiho was missing.

Design principles (mirrors :mod:`kumiho_memory.evidence_rank`):

* **Relevance dominates.** Search relevance is the base score. Evidence grade
  and recency are *small additive priors* that break ties and demote stale or
  low-trust memories — they never override a strong relevance signal.
* **Every stage no-ops safely.** No timestamp → no recency change. No reranker
  → no cross-encoder change. No evidence → no evidence change. Turning the
  pipeline on can reorder results but never crashes or blanks recall.
* **Deterministic.** No LLM calls, no randomness; the same inputs always
  produce the same ordering (stable sort, greedy MMR).

Order of operations in :func:`rerank`:
``cross-encoder (optional) → +evidence prior → +recency prior → sort → MMR``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from kumiho_memory.evidence_rank import (
    EvidenceRankConfig,
    evidence_badge,
    parse_evidence,
)

#: A reranker scores a query against candidate texts and returns one relevance
#: float per text (higher = more relevant). Backends: a cross-encoder
#: (bge-reranker-v2-m3), Cohere Rerank, or an LLM scorer.
Reranker = Callable[[str, Sequence[str]], Sequence[float]]

_LN2 = math.log(2.0)


@dataclass
class RerankConfig:
    """Tuneable knobs for the post-recall reranking pipeline."""

    #: Exponential recency boost: recent memories get up to ``recency_max_boost``
    #: added to their score, decaying by half every ``recency_half_life_days``.
    recency_enabled: bool = True
    recency_half_life_days: float = 45.0
    recency_max_boost: float = 0.12

    #: Maximal-marginal-relevance diversity. ``mmr_lambda`` in [0,1]: 1.0 is pure
    #: relevance, 0.0 is pure diversity. Suppresses near-duplicate revisions.
    mmr_enabled: bool = True
    mmr_lambda: float = 0.72

    #: Cross-encoder relevance rerank (opt-in — needs a ``Reranker`` and, for the
    #: bundled backend, the optional ``fastembed`` dependency).
    #: Final relevance = ``(1-w)·norm(base) + w·norm(cross_encoder)``.
    cross_encoder_enabled: bool = False
    cross_encoder_weight: float = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_score(mem: Dict[str, Any]) -> Optional[float]:
    """The memory's real relevance score, or ``None`` when it has none.

    ``base_score`` (set by a previous pass) wins over ``score`` for idempotency.
    Bools and non-numeric values count as score-less.
    """
    value = mem.get("base_score", mem.get("score"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _text(mem: Dict[str, Any]) -> str:
    parts = [
        str(mem.get("title", "")),
        str(mem.get("summary", "")),
        str(mem.get("description", "")),
        str(mem.get("content", "")),
    ]
    return " ".join(p for p in parts if p).strip()


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (with or without ``Z``) into aware UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _recency_boost(mem: Dict[str, Any], config: RerankConfig, now: datetime) -> float:
    """Exponential-decay boost from the memory's ``created_at`` age."""
    ts = _parse_ts(mem.get("created_at") or mem.get("created") or mem.get("timestamp"))
    if ts is None:
        return 0.0
    age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
    hl = config.recency_half_life_days
    if hl <= 0:
        return config.recency_max_boost
    return config.recency_max_boost * math.exp(-_LN2 * age_days / hl)


_TOKEN_RE = re.compile(r"[0-9a-z]+|[가-힣]+", re.IGNORECASE)


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def apply_cross_encoder(
    query: str,
    memories: List[Dict[str, Any]],
    reranker: Reranker,
    config: RerankConfig,
) -> List[Dict[str, Any]]:
    """Blend a cross-encoder's relevance into each memory's ``score`` in place.

    Both the base relevance and the cross-encoder score are min-max normalized
    across the candidate set before blending, so the two scales are comparable.
    A reranker that raises is treated as absent (no change).
    """
    if not memories:
        return memories
    texts = [_text(m) for m in memories]
    try:
        ce_scores = list(reranker(query, texts))
    except Exception:
        return memories
    if len(ce_scores) != len(memories):
        return memories

    bases = [(_numeric_score(m) or 0.0) for m in memories]
    base_n = _minmax(bases)
    ce_n = _minmax([float(s) for s in ce_scores])
    w = config.cross_encoder_weight
    for m, b, c, raw in zip(memories, base_n, ce_n, ce_scores):
        m["_cross_encoder_score"] = float(raw)
        m["score"] = (1.0 - w) * b + w * c
        m.pop("base_score", None)  # relevance was replaced; recompute priors on it
    return memories


def mmr_diversify(
    memories: List[Dict[str, Any]],
    config: RerankConfig,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Reorder by Maximal Marginal Relevance to suppress near-duplicates.

    Greedy: repeatedly pick the memory maximizing
    ``λ·rel − (1−λ)·max_sim(candidate, already-selected)`` where similarity is
    token Jaccard over title/summary/description/content. Only memories carrying
    a numeric score participate; score-less ones keep their trailing order.
    """
    scored = [m for m in memories if _numeric_score(m) is not None]
    unscored = [m for m in memories if _numeric_score(m) is None]
    n = len(scored)
    if n <= 2:
        return memories

    rel = _minmax([_numeric_score(m) or 0.0 for m in scored])
    toks = [_tokens(_text(m)) for m in scored]
    lam = config.mmr_lambda
    target = n if limit is None else min(max(limit, 1), n)

    remaining = list(range(n))
    selected: List[int] = []
    # Seed with the most relevant.
    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < target:
        best_i, best_val = None, None
        for i in remaining:
            max_sim = max(_jaccard(toks[i], toks[j]) for j in selected)
            val = lam * rel[i] - (1.0 - lam) * max_sim
            if best_val is None or val > best_val:
                best_i, best_val = i, val
        selected.append(best_i)
        remaining.remove(best_i)

    order = selected + remaining  # any tail beyond `target` keeps relevance order
    return [scored[i] for i in order] + unscored


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    memories: List[Dict[str, Any]],
    *,
    evidence_config: Optional[EvidenceRankConfig] = None,
    config: Optional[RerankConfig] = None,
    reranker: Optional[Reranker] = None,
    limit: Optional[int] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Full post-recall rerank: cross-encoder → +evidence → +recency → sort → MMR.

    Subsumes :func:`kumiho_memory.evidence_rank.apply_evidence_weights` (evidence
    is folded in as one additive prior) so callers apply exactly one reweighting
    pass. Adjusts dicts in place and returns the reordered list. Memories without
    a numeric relevance score are never given a fabricated one and trail the
    scored results in their original order.
    """
    config = config or RerankConfig()
    evidence_config = evidence_config or EvidenceRankConfig()
    if not memories:
        return memories

    now = now or datetime.now(timezone.utc)

    # 1. Cross-encoder relevance (optional) — resets the base relevance.
    ce_active = False
    if config.cross_encoder_enabled and reranker is not None:
        apply_cross_encoder(query, memories, reranker, config)
        ce_active = any(m.get("_cross_encoder_score") is not None for m in memories)

    # 2. Additive priors: evidence grade + recency, applied together so neither
    #    clobbers the other (both would otherwise recompute from base_score).
    levels = [parse_evidence(m, m.get("tags") or ()) for m in memories]
    ev_active = evidence_config.enabled and any(levels)
    rec_enabled = config.recency_enabled and config.recency_max_boost != 0.0
    rec_active = False

    scored: List[Dict[str, Any]] = []
    unscored: List[Dict[str, Any]] = []
    for mem, level in zip(memories, levels):
        base = _numeric_score(mem)
        if base is None:
            unscored.append(mem)
            continue
        mem["base_score"] = base
        ev_delta = (
            evidence_config.weights.get(level, 0.0) if (ev_active and level) else 0.0
        )
        rec_delta = 0.0
        if rec_enabled:
            ts = _parse_ts(
                mem.get("created_at") or mem.get("created") or mem.get("timestamp")
            )
            if ts is not None:
                rec_active = True
                rec_delta = _recency_boost(mem, config, now)
        mem["score"] = base + ev_delta + rec_delta
        if evidence_config.badges:
            badge = evidence_badge(mem, evidence_config)
            if badge:
                mem["evidence_badge"] = badge
        scored.append(mem)

    if not scored:
        return memories

    # 3. Sort by adjusted score ONLY when a signal actually reweighted the set.
    #    With no evidence/recency/cross-encoder the server's relevance order is
    #    authoritative and preserved (back-compat with evidence-only recall).
    if ev_active or rec_active or ce_active:
        scored.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    ordered = scored + unscored

    # 4. Diversity (MMR is itself an intentional reordering; no-ops for <=2).
    if config.mmr_enabled:
        ordered = mmr_diversify(ordered, config, limit)

    memories[:] = ordered
    return memories


# ---------------------------------------------------------------------------
# Optional bundled cross-encoder backend
# ---------------------------------------------------------------------------

def try_fastembed_reranker(
    model_name: str = "BAAI/bge-reranker-base",
) -> Optional[Reranker]:
    """Return a multilingual cross-encoder reranker, or ``None`` if unavailable.

    Uses :mod:`fastembed` (optional dependency, ONNX — no torch). ``model_name``
    must be one of ``fastembed``'s supported cross-encoders (see
    ``TextCrossEncoder.list_supported_models()``); the multilingual options are
    ``BAAI/bge-reranker-base`` (default) and
    ``jinaai/jina-reranker-v2-base-multilingual``. Returns ``None`` when
    fastembed or the model cannot be loaded, so callers can wire it
    unconditionally and simply skip cross-encoder rerank when it is not present.
    """
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except Exception:
        return None
    try:
        encoder = TextCrossEncoder(model_name=model_name)
    except Exception:
        return None

    def _rerank(query: str, texts: Sequence[str]) -> Sequence[float]:
        return list(encoder.rerank(query, list(texts)))

    return _rerank
