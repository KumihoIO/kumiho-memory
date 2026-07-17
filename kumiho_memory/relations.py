"""Relational edges between typed ontology nodes: DEPENDS_ON and SUPERSEDES.

Structural edges (DERIVED_FROM / ABOUT / INVOLVES) are deterministic and live
in ``ontology.py``. The two *relational* edges need a source of truth:

- ``decision --DEPENDS_ON--> fact`` uses indices the summarizer emits
  (``decisions[i].based_on`` -> fact positions); no guessing.
- ``decision --SUPERSEDES--> decision`` (and fact->fact) is a belief update:
  a newer node about the *same subject* replacing an older one. Candidates are
  *found* with fulltext search, but the decision to link is made on
  **token-overlap (Jaccard)**, not the search score — search scores are
  corpus-global BM25 (see kumiho-server#28) and would couple this to data
  hygiene. Overlap is corpus-independent.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Minimum token-overlap for two nodes to count as "the same subject".
_SUPERSEDE_JACCARD = 0.6


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.casefold()) if len(t) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def link_depends_on(
    m: Any,
    decision_anchor: Any,
    based_on_indices: List[int],
    fact_anchors: List[Optional[Any]],
    edge_type: str = "DEPENDS_ON",
) -> int:
    """Link a decision to the facts it was based on (summarizer-emitted indices)."""
    edges = 0
    for idx in based_on_indices:
        if 0 <= idx < len(fact_anchors):
            target = fact_anchors[idx]
            if target is not None and m.edge(decision_anchor, target, edge_type):
                edges += 1
    return edges


def link_depends_on_by_overlap(
    m: Any,
    decision_anchor: Any,
    decision_text: str,
    fact_entries: List[Any],
    threshold: float = 0.4,
    edge_type: str = "DEPENDS_ON",
) -> int:
    """Post-hoc grounding when the summarizer emits no ``based_on`` indices.

    The summary schema deliberately omits ``based_on`` in both ontology modes
    (emitting it forced a different structured output on every consolidation —
    measured as a base-recall regression), so the grounding fact is recovered
    the same corpus-independent way SUPERSEDES is: token overlap, scoped to
    the *same consolidation's* facts. Links the single best fact at/above
    *threshold*. ``fact_entries`` are ``(anchor, slug, claim)`` tuples.
    """
    d_tokens = _tokens(decision_text)
    if not d_tokens:
        return 0
    best = None
    best_overlap = 0.0
    for anchor, _slug, claim in fact_entries:
        if anchor is None:
            continue
        overlap = _jaccard(d_tokens, _tokens(claim))
        if overlap > best_overlap:
            best_overlap = overlap
            best = anchor
    if best is not None and best_overlap >= threshold:
        if m.edge(decision_anchor, best, edge_type,
                  {"overlap": f"{best_overlap:.2f}"}):
            return 1
    return 0


def link_supersedes(
    m: Any,
    kind: str,
    space: str,
    self_slug: str,
    anchor: Any,
    text: str,
    project_name: str,
    edge_type: str = "SUPERSEDES",
) -> int:
    """Link *anchor* to a prior same-kind node about the same subject.

    Finds candidates with a scoped, kind-filtered fulltext search, then links
    to the single best *different* item whose text overlaps the new one above
    a token-Jaccard threshold — so the belief-update edge does not depend on
    unstable BM25 scores.
    """
    import kumiho

    new_tokens = _tokens(text)
    if not new_tokens:
        return 0
    try:
        results = kumiho.search(
            text[:150],
            context=f"{project_name}/{space}",
            kind=kind,
            include_revision_metadata=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("supersedes search failed for %s: %s", self_slug, exc)
        return 0

    best_item = None
    best_rev = None
    best_overlap = 0.0
    for r in results or []:
        item = getattr(r, "item", None)
        if item is None:
            continue
        item_kref = getattr(getattr(item, "kref", None), "uri", "") or ""
        # Skip the node we just created (same identity slug).
        if f"/{self_slug}.{kind}" in item_kref:
            continue
        try:
            cand_rev = item.get_latest_revision()
        except Exception:  # noqa: BLE001
            continue
        if cand_rev is None:
            continue
        meta = getattr(cand_rev, "metadata", {}) or {}
        cand_text = meta.get(kind) or meta.get("summary") or meta.get("title") or ""
        overlap = _jaccard(new_tokens, _tokens(cand_text))
        if overlap > best_overlap:
            best_overlap = overlap
            best_item = item
            best_rev = cand_rev

    if best_rev is not None and best_overlap >= _SUPERSEDE_JACCARD:
        # basis labels the heuristic provenance (vs agent-declared belief edges,
        # which record basis: agent); trigger logic + threshold unchanged.
        if m.edge(anchor, best_rev, edge_type,
                  {"reason": "belief update", "basis": "lexical-overlap"}):
            logger.debug("SUPERSEDES: %s replaces %s (overlap=%.2f)",
                         self_slug, getattr(best_item, "kref", "?"), best_overlap)
            # Grounding-staleness ripple (#95): a fact F (best_rev) just got
            # superseded by `anchor` — flag the decisions grounded in F so recall
            # marks them and Dream State can clear them. Only facts carry an
            # incoming DEPENDS_ON (decision->fact), so a decision->decision
            # supersede skips the ripple's wasted get_edges. Best-effort +
            # bounded; see grounding.ripple_grounding_stale.
            if kind == "fact":
                from .grounding import ripple_grounding_stale
                ripple_grounding_stale(
                    best_rev, getattr(getattr(anchor, "kref", None), "uri", ""),
                )
            return 1
    return 0
