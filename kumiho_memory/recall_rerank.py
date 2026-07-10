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
* **Deterministic by default.** recency + MMR + evidence make no LLM calls and
  are fully deterministic. The cross-encoder stage is optional; its reranker may
  be a local cross-encoder (:func:`try_fastembed_reranker`) or the host LLM
  itself (:func:`make_llm_reranker`, reusing the manager's existing adapter — no
  extra API key), both opt-in.

Order of operations in :func:`rerank`:
``cross-encoder (optional) → +evidence prior → +recency prior → sort → MMR``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from kumiho_memory.evidence_rank import (
    EvidenceRankConfig,
    evidence_badge,
    parse_evidence,
)

logger = logging.getLogger(__name__)

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

    #: Event-proximity prior (opt-in, TEMPORAL queries only). Boosts memories
    #: whose semantic ``event_date`` (valid-time) is near the query's reference
    #: time, decaying by half every ``event_proximity_half_life_days``. Distinct
    #: from recency, which measures *storage* age from ``created_at``. Fires only
    #: when the caller passes ``rerank(..., query_time=...)`` — so non-temporal
    #: queries (``query_time=None``) are never reweighted by it. Keep the boost
    #: ``<= recency_max_boost``: the two correlated temporal priors are capped
    #: jointly so time can never outweigh relevance.
    event_proximity_enabled: bool = False
    event_proximity_half_life_days: float = 45.0
    event_proximity_max_boost: float = 0.12

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RerankConfig":
        """Build a config from the ``KUMIHO_RECALL_RERANK`` kill switch env var.

        ``KUMIHO_RECALL_RERANK=0``/``false`` disables the deterministic
        recency + MMR stages (matching the MCP server's convention); every
        other knob keeps its default. The optional cross-encoder stage is
        turned on separately by :func:`resolve_reranker_from_env` once a
        reranker is actually available — so with **no** env vars set this
        returns a plain :class:`RerankConfig`, i.e. exactly the default
        behavior.
        """
        env = os.environ if env is None else env
        if str(env.get("KUMIHO_RECALL_RERANK", "")).strip().lower() in ("0", "false"):
            return cls(recency_enabled=False, mmr_enabled=False)
        return cls()


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


def _pad_iso_date(value: Any) -> str:
    """Pad a partial ISO date (``YYYY`` / ``YYYY-MM``) to ``YYYY-MM-DD``.

    The summarizer stores ``event_date`` at whatever precision it can infer, but
    :func:`_parse_ts` needs a full calendar date. Anything else passes through.
    """
    if not isinstance(value, str):
        return ""
    v = value.strip()
    if re.fullmatch(r"\d{4}", v):
        return f"{v}-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", v):
        return f"{v}-01"
    return v


def _event_proximity_boost(
    mem: Dict[str, Any], config: RerankConfig, query_time: datetime
) -> float:
    """Exponential-decay boost from ``|event_date − query_time|`` (valid-time).

    Distinct from :func:`_recency_boost`, which measures *storage* age from
    ``created_at``. No-ops (``0.0``) when the memory carries no parseable
    ``event_date`` — so legacy/undated revisions are unaffected.
    """
    ts = _parse_ts(_pad_iso_date(mem.get("event_date")))
    if ts is None:
        return 0.0
    gap_days = abs((query_time - ts).total_seconds()) / 86400.0
    hl = config.event_proximity_half_life_days
    if hl <= 0:
        return config.event_proximity_max_boost
    return config.event_proximity_max_boost * math.exp(-_LN2 * gap_days / hl)


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
    query_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Full post-recall rerank: cross-encoder → +evidence → +recency → sort → MMR.

    Subsumes :func:`kumiho_memory.evidence_rank.apply_evidence_weights` (evidence
    is folded in as one additive prior) so callers apply exactly one reweighting
    pass. Adjusts dicts in place and returns the reordered list. Memories without
    a numeric relevance score are never given a fabricated one and trail the
    scored results in their original order.

    ``query_time`` is the reference instant for the optional event-proximity
    prior (:attr:`RerankConfig.event_proximity_enabled`). Pass it ONLY for
    queries with a temporal intent; leaving it ``None`` (the default) keeps the
    event-proximity signal fully dormant, so general recall is never reweighted
    by event dates. ``now`` remains the reference for the storage-recency prior.
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
    # Event-proximity fires only for temporal queries: opt-in config AND a
    # caller-supplied query_time. query_time=None => this prior is a no-op.
    evt_enabled = (
        config.event_proximity_enabled
        and config.event_proximity_max_boost != 0.0
        and query_time is not None
    )
    evt_active = False
    # Recency (storage age) and event-proximity (valid-time) correlate when
    # created_at ≈ event_date, so their sum is capped at the larger single
    # boost — two temporal priors must never jointly outweigh relevance.
    temporal_cap = max(config.recency_max_boost, config.event_proximity_max_boost)

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
        evt_delta = 0.0
        if evt_enabled:
            eb = _event_proximity_boost(mem, config, query_time)
            if eb > 0.0:
                evt_active = True
                evt_delta = eb
        mem["score"] = base + ev_delta + min(rec_delta + evt_delta, temporal_cap)
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
    if ev_active or rec_active or ce_active or evt_active:
        scored.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    ordered = scored + unscored

    # 4. Diversity (MMR is itself an intentional reordering; no-ops for <=2).
    if config.mmr_enabled:
        ordered = mmr_diversify(ordered, config, limit)

    memories[:] = ordered
    return memories


# ---------------------------------------------------------------------------
# Async offload — keep cross-encoder inference off the event loop
# ---------------------------------------------------------------------------

#: Dedicated single-worker executor for cross-encoder reranks.  ONE worker on
#: purpose: inferences stay serialized exactly as they were when :func:`rerank`
#: ran inline (identical results, identical CPU profile, no ONNX thread
#: oversubscription) — the only change is that the event loop is free while a
#: rerank runs, so concurrent recalls can overlap their network I/O (measured
#: 2026-07-10: inline CE collapsed a concurrency-4 harness to ~1 effective).
_RERANK_EXECUTOR: Optional[ThreadPoolExecutor] = None
_RERANK_EXECUTOR_LOCK = threading.Lock()


def _rerank_executor() -> ThreadPoolExecutor:
    global _RERANK_EXECUTOR
    with _RERANK_EXECUTOR_LOCK:
        if _RERANK_EXECUTOR is None:
            _RERANK_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kumiho-rerank",
            )
        return _RERANK_EXECUTOR


async def rerank_async(
    query: str,
    memories: List[Dict[str, Any]],
    *,
    evidence_config: Optional[EvidenceRankConfig] = None,
    config: Optional[RerankConfig] = None,
    reranker: Optional[Reranker] = None,
    limit: Optional[int] = None,
    now: Optional[datetime] = None,
    query_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """:func:`rerank`, offloaded to a worker thread when it would block.

    The fastembed cross-encoder stage is CPU-bound model inference; called
    inline from async recall code it blocks the event loop and serializes
    every concurrent recall.  This wrapper runs the *unchanged* sync
    :func:`rerank` on a dedicated single-worker executor when the reranker is
    tagged offload-safe (``_kumiho_offload_safe``, set by
    :func:`try_fastembed_reranker`), and inline otherwise.  Inline covers two
    deliberate cases: (a) the cross-encoder is dormant — the deterministic
    priors are microseconds and a thread hop would only add latency; (b) the
    reranker is the LLM backend or a user callable — the LLM reranker drives
    the manager's shared async client, and driving that client from a second
    event loop on the worker thread corrupts httpx's loop-bound connection
    pool, so it keeps its pre-0.10.1 inline behavior.  Results are
    byte-identical to the sync call, including the in-place mutation of
    ``memories``.
    """
    cfg = config or RerankConfig()
    call = functools.partial(
        rerank,
        query,
        memories,
        evidence_config=evidence_config,
        config=cfg,
        reranker=reranker,
        limit=limit,
        now=now,
        query_time=query_time,
    )
    offload = (
        cfg.cross_encoder_enabled
        and reranker is not None
        and bool(memories)
        and getattr(reranker, "_kumiho_offload_safe", False)
    )
    if not offload:
        return call()
    return await asyncio.get_running_loop().run_in_executor(
        _rerank_executor(), call,
    )


# ---------------------------------------------------------------------------
# Two-pass focused rerank — re-score with embeddings over title+summary only
# ---------------------------------------------------------------------------

def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Cosine similarity between two float vectors (pure python, no numpy)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _focused_text(mem: Dict[str, Any]) -> str:
    """``title: summary`` scoring text — deliberately narrow.

    Unlike :func:`_text` (used by the cross-encoder/MMR stages), this strips
    description/content *and* structured metadata (implications, events) so
    the score measures only how directly the revision's core claim answers
    the query.
    """
    title = str(mem.get("title", "") or "")
    summary = str(mem.get("summary", "") or "")
    return f"{title}: {summary}" if title else (summary or title or "")


def two_pass_rerank(
    query: str,
    memories: List[Dict[str, Any]],
    embedding_adapter: Optional[Any],
) -> List[Dict[str, Any]]:
    """Re-score recalled memories AND their siblings with focused embeddings.

    Pass 1 (already done upstream): server recall + sibling scoring used
    enriched semantic embeddings (title, summary, implications, events) to
    cast a wide net.

    Pass 2 (this function): re-score every memory and every
    ``sibling_revisions`` entry client-side using *embedding_adapter* on
    **title+summary text only**, replacing the server scores.  Stripping
    implications/events from the scoring signal lets the most *directly*
    relevant revision rank highest.  Primaries and siblings are scored in the
    same batch, so downstream global ranking (``compose_context``) compares
    one consistent cosine scale instead of mixing server relevance (~0-3)
    with sibling cosines (0-1).

    Scores are updated in place — ``score`` on memories (``base_score`` is
    dropped, mirroring :func:`apply_cross_encoder`: the relevance basis was
    replaced) and ``_score`` on siblings — and the same list is returned.
    Ordering is left untouched; callers rank downstream.

    Safe no-op guarantees: returns the input unchanged when *query* is empty,
    *memories* is empty, no ``embedding_adapter`` is configured (debug log),
    or embedding fails (warning log).
    """
    if not memories or not query:
        return memories
    if embedding_adapter is None:
        logger.debug(
            "two_pass_rerank: no embedding adapter configured — skipping"
        )
        return memories

    # Collect scoring targets: every primary memory plus every sibling.
    targets: List[tuple] = []  # (dict, score_key)
    texts: List[str] = []
    for mem in memories:
        targets.append((mem, "score"))
        texts.append(_focused_text(mem))
        for sib in mem.get("sibling_revisions") or []:
            targets.append((sib, "_score"))
            texts.append(_focused_text(sib))

    try:
        embeddings = embedding_adapter.embed([query] + texts)
    except Exception as e:
        logger.warning("two_pass_rerank failed, keeping original scores: %s", e)
        return memories
    if len(embeddings) != len(texts) + 1:
        logger.warning(
            "two_pass_rerank: adapter returned %d vectors for %d texts — "
            "keeping original scores", len(embeddings), len(texts) + 1,
        )
        return memories

    query_vec = embeddings[0]
    scores: List[float] = []
    for (target, key), vec in zip(targets, embeddings[1:]):
        old = target.get(key, 0.0)
        new = _cosine(query_vec, vec)
        target[key] = new
        if key == "score":
            # Relevance basis was replaced; priors must not recompute from
            # the stale server score.
            target.pop("base_score", None)
        scores.append(new)
        logger.debug(
            "Two-pass rerank: %s — %s → %.3f",
            str(target.get("title", "?"))[:50], old, new,
        )

    logger.info(
        "Two-pass rerank: re-scored %d revisions (top: %.3f, bottom: %.3f)",
        len(scores), max(scores) if scores else 0.0,
        min(scores) if scores else 0.0,
    )
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

    # Pure CPU-bound ONNX inference with no event-loop entanglement — safe for
    # rerank_async to move onto its worker thread.  Rerankers WITHOUT this tag
    # (the LLM reranker, arbitrary user callables) stay inline: the LLM
    # reranker drives the manager's shared async client, and running that from
    # a second event loop on the worker thread corrupts httpx's loop-bound
    # connection pool.
    _rerank._kumiho_offload_safe = True  # type: ignore[attr-defined]
    return _rerank


# ---------------------------------------------------------------------------
# LLM reranker — reuse the host LLM (no separate reranker model or API key)
# ---------------------------------------------------------------------------

def _run_coro_sync(make_coro: Callable[[], Any]) -> Any:
    """Run an async factory to completion from sync code.

    The reranker interface is synchronous but LLM adapters are async, and the
    pipeline may run inside an event loop (async recall). If a loop is already
    running we execute the coroutine in a throwaway thread with its own loop;
    otherwise we run it directly.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if not running:
        return asyncio.run(make_coro())
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(make_coro())).result()


def _parse_llm_scores(raw: Any, n: int) -> List[float]:
    """Parse ``n`` relevance floats from an LLM response (array or {"scores":[]})."""
    import json

    text = raw.strip() if isinstance(raw, str) else ""
    arr: Any = None
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            arr = obj
        elif isinstance(obj, dict):
            arr = obj.get("scores")
            if not isinstance(arr, list):
                arr = next((v for v in obj.values() if isinstance(v, list)), None)
    except Exception:
        start, end = text.find("["), text.rfind("]")
        if 0 <= start < end:
            try:
                arr = json.loads(text[start : end + 1])
            except Exception:
                arr = None
    if not isinstance(arr, list) or len(arr) != n:
        raise ValueError("LLM reranker returned unparseable or mismatched scores")
    return [float(x) for x in arr]


def make_llm_reranker(
    adapter: Any,
    model: str,
    *,
    char_limit: int = 600,
    max_tokens: int = 400,
) -> Reranker:
    """Build a :data:`Reranker` that scores relevance with the host LLM.

    Reuses whatever LLM the manager already runs (``summarizer.adapter`` +
    ``light_model``) — **no separate reranker model, download, or API key**.
    One ``chat`` call per rerank scores all candidates. Any failure (LLM error,
    unparseable output) raises, which the pipeline's cross-encoder stage catches
    and treats as a no-op, so recall never breaks.

    This is the wiring for the original "the LLM running Kumiho reranks" design:
    plug the returned callable in as ``reranker`` with
    ``RerankConfig(cross_encoder_enabled=True)``.
    """
    system = (
        "You are a precise search reranker. Given a query and candidate memory "
        "documents, rate how well each document answers the query, from 0.0 "
        "(irrelevant) to 1.0 (directly answers it). Judge relevance only. "
        'Respond with ONLY a JSON object of the form {"scores": [n, n, ...]} '
        "with one number per document, in the given order."
    )

    def _rerank(query: str, texts: Sequence[str]) -> Sequence[float]:
        docs = "\n".join(f"[{i}] {t[:char_limit]}" for i, t in enumerate(texts))
        user = (
            f"Query:\n{query}\n\nDocuments:\n{docs}\n\n"
            f'Return {{"scores": [...]}} with exactly {len(texts)} numbers (0.0-1.0).'
        )
        raw = _run_coro_sync(
            lambda: adapter.chat(
                messages=[{"role": "user", "content": user}],
                model=model,
                system=system,
                max_tokens=max_tokens,
                json_mode=True,
            )
        )
        return _parse_llm_scores(raw, len(texts))

    return _rerank


# ---------------------------------------------------------------------------
# Shared env-based reranker resolution
# ---------------------------------------------------------------------------

def resolve_reranker_from_env(
    adapter: Optional[Any] = None,
    model: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    adapter_factory: Optional[Callable[[], Any]] = None,
) -> Optional[Reranker]:
    """Resolve a :data:`Reranker` from ``KUMIHO_RERANK_*`` env vars, or ``None``.

    Centralizes the reranker wiring that was previously inline in the MCP
    server so **every** construction path (the MCP server, direct
    ``KumihoMemoryManager`` construction, benchmark harnesses, SDK users) gets
    identical behavior. The env conventions mirror the MCP server's:

    * ``KUMIHO_RERANK_CROSS_ENCODER=1`` — local bge cross-encoder via
      :func:`try_fastembed_reranker` (needs the ``fastembed`` extra).
    * ``KUMIHO_RERANK_LLM=1`` — the host LLM itself via
      :func:`make_llm_reranker`, reusing ``adapter`` + ``model`` (no extra
      model download or API key). The cross-encoder wins if both are set.

    ``adapter_factory`` is a lazy alternative to ``adapter``: it is invoked
    (and its errors swallowed with a warning) only when ``KUMIHO_RERANK_LLM``
    actually requests the LLM path.  Pass it when building the adapter is
    itself fallible or expensive — e.g. a summarizer's lazy ``adapter``
    property raises without an API key, and env resolution at manager
    construction must not crash or eagerly build LLM clients that may never
    be used.

    Returns ``None`` when no env var requests a reranker, or when the requested
    backend is unavailable (``fastembed`` missing, or no ``adapter`` for the
    LLM path) — logging a warning so the miswire is visible while recall keeps
    working (every rerank stage no-ops safely). Callers turn the cross-encoder
    stage on (``RerankConfig.cross_encoder_enabled = True``) only when this
    returns a reranker.
    """
    env = os.environ if env is None else env
    truthy = ("1", "true", "yes")

    if str(env.get("KUMIHO_RERANK_CROSS_ENCODER", "")).strip().lower() in truthy:
        reranker = try_fastembed_reranker()
        if reranker is not None:
            logger.info("Cross-encoder recall rerank enabled (fastembed)")
            return reranker
        logger.warning(
            "KUMIHO_RERANK_CROSS_ENCODER set but fastembed/model is "
            "unavailable — install the 'fastembed' extra to enable."
        )

    if str(env.get("KUMIHO_RERANK_LLM", "")).strip().lower() in truthy:
        if adapter is None and adapter_factory is not None:
            try:
                adapter = adapter_factory()
            except Exception as exc:
                logger.warning(
                    "KUMIHO_RERANK_LLM set but the LLM adapter could not be "
                    "built: %s", exc,
                )
        if adapter is not None:
            logger.info(
                "LLM recall rerank enabled (host adapter, model=%s)", model or ""
            )
            return make_llm_reranker(adapter, model or "")
        logger.warning(
            "KUMIHO_RERANK_LLM set but no LLM adapter is configured — set "
            "ANTHROPIC_API_KEY or OPENAI_API_KEY to enable LLM rerank."
        )

    return None
