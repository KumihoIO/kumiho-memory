"""Relational edges between typed ontology nodes: DEPENDS_ON and SUPERSEDES.

Structural edges (DERIVED_FROM / ABOUT / INVOLVES) are deterministic and live
in ``ontology.py``. The *relational* edges need a source of truth:

- ``decision --DEPENDS_ON--> fact`` uses indices the summarizer emits
  (``decisions[i].based_on`` -> fact positions); no guessing.
- ``decision --SUPERSEDES--> decision`` (and fact->fact) is a belief update:
  a newer node about the *same subject* replacing an older one. Candidates are
  *found* with fulltext search, but the decision to link is made on
  **token-overlap (Jaccard)**, not the search score — search scores are
  corpus-global BM25 (see kumiho-server#28) and would couple this to data
  hygiene. Overlap is corpus-independent.
- ``decision --DERIVED_FROM--> conversation`` (#17) reconnects the two layers
  the same decision can be written into: ``reflect`` stores a capture as a
  ``conversation`` item, ``consolidate`` later projects the same decision into
  a typed ``decision`` node in the same space. Same corpus-independent
  overlap rule as SUPERSEDES; see :func:`link_capture_source`.
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
        from .supersession import supersede_revision
        metadata = {"reason": "belief update", "basis": "lexical-overlap"}
        if edge_type == "SUPERSEDES":
            result = supersede_revision(anchor, best_rev, metadata)
            created = result.created
            if result.error:
                m.supersession_failures = getattr(m, "supersession_failures", 0) + 1
        else:
            created = m.edge(anchor, best_rev, edge_type, metadata)
        if created:
            logger.debug("SUPERSEDES: %s replaces %s (overlap=%.2f)",
                         self_slug, getattr(best_item, "kref", "?"), best_overlap)
            return 1
    return 0


# --------------------------------------------------------------------------- #
# Cross-layer provenance: typed node -> the `conversation` capture of the      #
# same subject (#17)                                                          #
#                                                                             #
# `kumiho_memory_reflect` writes a capture as a `conversation`-kind item and   #
# no typed node. `kumiho_memory_consolidate` later projects                    #
# `knowledge.decisions` into typed `decision`-kind nodes, in the SAME space    #
# agents route decision captures to. The two never met: stacking filters by    #
# kind, so they don't merge, and decompose's structural DERIVED_FROM points    #
# at the consolidated summary revision, not at the earlier capture. Result:    #
# one decision, two unrelated nodes.                                          #
#                                                                             #
# This links them instead of deduplicating them (issue #17, option 2): the     #
# typed node gains a second DERIVED_FROM naming the capture it was also        #
# written from. Purely additive — no item is merged, deprecated or            #
# rewritten, and the edge is the existing provenance edge in its existing      #
# direction (<typed node> --DERIVED_FROM--> conversation), so the ontology     #
# spec's vocabulary is unchanged and the recall reader already traverses it.   #
# --------------------------------------------------------------------------- #

#: Minimum token-overlap for a typed node and a `conversation` capture to count
#: as the same subject. Deliberately the SUPERSEDES bar: both edges answer the
#: same question ("are these two nodes about one subject?"), and the asymmetry
#: of the failure modes points the same way — a missed link leaves the pair as
#: it is today, a wrong link misattributes a decision's provenance to an
#: unrelated memory that recall would then surface as its source.
_CAPTURE_JACCARD = _SUPERSEDE_JACCARD

#: Capture `type` metadata (mcp_tools.MEMORY_TYPES vocabulary, stamped by
#: kumiho.mcp_server.tool_memory_store as `type`) a *decision* link accepts.
#: The space alone doesn't identify the layer — a `preference` or `correction`
#: capture can sit in the decisions space and still share most of a decision's
#: wording. `summary` is included because it is the field's DEFAULT
#: (`cap.get("type", "summary")`), i.e. "the agent didn't say", not "not a
#: decision". Anything else — including an unrecognized value — is skipped:
#: strict allowlist, so an unknown type can never buy a link.
_DECISION_CAPTURE_TYPES = frozenset({"decision", "summary", ""})

#: The kind reflect captures are written as.
_CAPTURE_KIND = "conversation"


def _item_uri(uri: str) -> str:
    """Revision uri -> its item uri (``kref://p/s/n.kind?r=3`` -> without ?r=)."""
    return (uri or "").split("?", 1)[0]


def link_capture_source(
    m: Any,
    space: str,
    anchor: Any,
    text: str,
    project_name: str,
    *,
    exclude_item_uri: str = "",
    accepted_types: frozenset = _DECISION_CAPTURE_TYPES,
    edge_type: str = "DERIVED_FROM",
) -> int:
    """Link a typed node to the ``conversation`` capture of the same subject.

    Finds candidates with the same scoped, kind-filtered fulltext search
    ``link_supersedes`` uses — one search, bounded by the server's result
    page — then links the single best item whose *title* or *summary* overlaps
    *text* above :data:`_CAPTURE_JACCARD`. Title and summary are scored
    separately and the better of the two wins: a capture's title is usually the
    statement itself (high overlap), while its summary is the fuller prose the
    agent wrote (lower overlap for the same subject), so scoring only the
    concatenation would let a long summary dilute a title that matched exactly.
    Each field is judged at the same strict symmetric-Jaccard bar.

    *exclude_item_uri* is the conversation this decomposition is anchored on —
    consolidation's own summary item is a ``conversation`` too, and can live in
    this very space, where it would otherwise match its own decision and draw a
    second provenance edge to a different revision of the node the structural
    edge already covers.

    Returns 1 if an edge was created, 0 otherwise (no candidate, below
    threshold, edge already present, or lookup failure). Never raises: a
    failed lookup degrades to today's behavior.
    """
    import kumiho

    new_tokens = _tokens(text)
    if not new_tokens:
        return 0
    try:
        results = kumiho.search(
            text[:150],
            context=f"{project_name}/{space}",
            kind=_CAPTURE_KIND,
            include_revision_metadata=False,
        )
    except Exception as exc:  # noqa: BLE001 — never block a write on lookup
        logger.debug("capture link search failed in %s: %s", space, exc)
        return 0

    exclude = _item_uri(exclude_item_uri)
    best_rev = None
    best_overlap = 0.0
    for r in results or []:
        item = getattr(r, "item", None)
        if item is None:
            continue
        item_uri = getattr(getattr(item, "kref", None), "uri", "") or ""
        if exclude and _item_uri(item_uri) == exclude:
            continue
        try:
            cand_rev = item.get_latest_revision()
        except Exception:  # noqa: BLE001
            continue
        if cand_rev is None:
            continue
        meta = getattr(cand_rev, "metadata", {}) or {}
        if str(meta.get("type", "")).strip().casefold() not in accepted_types:
            continue
        overlap = max(
            _jaccard(new_tokens, _tokens(str(meta.get("title", "")))),
            _jaccard(new_tokens, _tokens(str(meta.get("summary", "")))),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_rev = cand_rev

    if best_rev is None or best_overlap < _CAPTURE_JACCARD:
        return 0
    # `basis` follows the belief-edge convention (heuristic vs. agent-declared);
    # it is inert on DERIVED_FROM, which the reader never scopes by basis, but
    # it keeps one vocabulary for "how was this edge decided".
    if m.edge(anchor, best_rev, edge_type, {
        "reason": "capture of the same subject",
        "basis": "lexical-overlap",
        "overlap": f"{best_overlap:.2f}",
    }):
        logger.debug("capture link: typed node <- %s (overlap=%.2f)",
                     getattr(getattr(best_rev, "kref", None), "uri", "?"),
                     best_overlap)
        return 1
    return 0
