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

Weighting is idempotent: the original retrieval score is preserved in
``_base_score`` and the adjusted score is always recomputed from it, so
applying the weights twice (e.g. before two different caps in
graph-augmented recall) cannot accumulate deltas.
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


def apply_evidence_weights(
    memories: List[Dict[str, Any]],
    config: EvidenceRankConfig,
) -> List[Dict[str, Any]]:
    """Adjust recall scores by evidence grade and stable-sort descending.

    Pure function, no LLM calls.  Returns the input list unmodified when
    disabled or when no memory resolves an evidence level (strict no-op
    for pre-evidence data).  The pre-adjustment score is preserved in
    ``_base_score``; the adjusted score is recomputed from it on every
    call, making repeated application idempotent.
    """
    if not config.enabled or not memories:
        return memories

    levels = [
        parse_evidence(mem, mem.get("tags") or ()) for mem in memories
    ]
    if not any(levels):
        return memories

    for mem, level in zip(memories, levels):
        base = mem.get("_base_score", mem.get("score", 0.0))
        if not isinstance(base, (int, float)):
            base = 0.0
        mem["_base_score"] = base
        delta = config.weights.get(level, 0.0) if level else 0.0
        mem["score"] = base + delta

    # Python's sort is stable — equal adjusted scores keep base order.
    memories.sort(key=lambda m: m.get("score", 0.0), reverse=True)
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
