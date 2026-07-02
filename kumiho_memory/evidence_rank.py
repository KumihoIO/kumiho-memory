"""Evidence-weighted recall reranking + context badges.

Server-side hybrid search ranks by relevance only — a rumor-grade memory
can outrank an official one.  This module adjusts retrieval scores by a
configurable delta per evidence level (deterministic, zero extra LLM
calls, O(k) over retrieved results) and renders grade badges like
``[official]`` when memories are injected into an answering model's
context.

Strict no-op guarantee: when none of the retrieved memories carries an
evidence grade, :func:`apply_evidence_weights` returns its input list
unmodified (same objects, same order, untouched scores) — pre-evidence
deployments see byte-identical behavior.

Score-less memories are never given a fabricated score: ``min_score``
filtering deliberately passes memories whose score is missing or
non-numeric, and inventing ``0.0`` for them would silently flip that
semantics.  Only memories with a real numeric retrieval score
participate in weighting; the rest keep their keys untouched and are
placed after the weighted ones (they carry no relevance ordering of
their own).

Weighting is idempotent: the original retrieval score is preserved in
``base_score`` (a documented public result field) and the adjusted score
is always recomputed from it, so applying the weights twice (e.g. at two
different points of graph-augmented recall) cannot accumulate deltas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from kumiho_memory.evidence import parse_evidence

logger = logging.getLogger(__name__)

#: Default score deltas per evidence level (mild — relevance still
#: dominates; evidence breaks ties and demotes rumors).
DEFAULT_EVIDENCE_WEIGHTS: Dict[str, float] = {
    "official": 0.15,
    "corroborated": 0.08,
    "single_source": 0.0,
    "unverified": -0.10,
}


@dataclass
class EvidenceRankConfig:
    """Tuneable knobs for evidence-weighted recall reranking."""

    enabled: bool = True
    weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EVIDENCE_WEIGHTS)
    )
    badges: bool = True


def _numeric_score(mem: Dict[str, Any]) -> Any:
    """The memory's real retrieval score, or ``None`` when it has none.

    ``base_score`` (set by a previous weighting pass) wins over ``score``
    for idempotency.  Bools and non-numeric values count as score-less.
    """
    value = mem.get("base_score", mem.get("score"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def apply_evidence_weights(
    memories: List[Dict[str, Any]],
    config: EvidenceRankConfig,
) -> List[Dict[str, Any]]:
    """Adjust recall scores by evidence grade and stable-sort descending.

    No LLM calls; adjusts the input dicts **in place** and returns the
    same list.  Returns it completely unmodified when disabled or when no
    memory resolves an evidence level (strict no-op for pre-evidence
    data).

    Only memories carrying a real numeric score are weighted and sorted
    (stable — equal adjusted scores keep retrieval order); score-less
    memories are never given a fabricated score and are placed after the
    weighted ones in their original relative order.  The pre-adjustment
    score is preserved in ``base_score``; the adjusted score is
    recomputed from it on every call, making repeated application
    idempotent.
    """
    if not config.enabled or not memories:
        return memories

    levels = [
        parse_evidence(mem, mem.get("tags") or ()) for mem in memories
    ]
    if not any(levels):
        return memories

    scored: List[Dict[str, Any]] = []
    unscored: List[Dict[str, Any]] = []
    for mem, level in zip(memories, levels):
        base = _numeric_score(mem)
        if base is None:
            unscored.append(mem)
            continue
        mem["base_score"] = base
        delta = config.weights.get(level, 0.0) if level else 0.0
        mem["score"] = base + delta
        scored.append(mem)

    if not scored:
        # Nothing carries a relevance score — no ordering to adjust.
        return memories

    # Python's sort is stable — equal adjusted scores keep base order.
    scored.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    memories[:] = scored + unscored
    return memories


def evidence_badge(mem: Dict[str, Any], config: EvidenceRankConfig) -> str:
    """Return a context badge like ``[official] `` for a memory.

    Empty string when badges are disabled, the memory is ungraded, or the
    grade is ``single_source`` (the neutral default — badging it adds
    noise without signal).
    """
    if not config.badges:
        return ""
    level = parse_evidence(mem, mem.get("tags") or ())
    if not level or level == "single_source":
        return ""
    return f"[{level}] "
