"""Revision-centric context assembly from recalled memories.

Ported from the LoCoMo benchmark harness (``build_recalled_context`` +
``_collect_top_revisions`` in ``kumiho_eval/common.py``) so context assembly
is an SDK capability instead of harness-side logic.

Recall results are *item-centric*: each memory dict is the item's primary
(published) revision, optionally carrying ``sibling_revisions`` — the other
revisions of a stacked item, each scored against the query (``_score``).
Answering LLMs, however, want the best *revisions* regardless of which item
they hang off.  This module flattens all revisions across all recalled
memories into one pool, ranks them globally by score, caps the pool, and
renders it as plain text.

Design principles (mirrors :mod:`kumiho_memory.recall_rerank`):

* **Siblings subsume the primary.**  When ``sibling_revisions`` is present it
  contains *all* revisions of the item — including the published one — so the
  primary entry is skipped to avoid double-counting.  Its content appears via
  the sibling list when selected.
* **Every stage no-ops safely.**  No scores → original order is preserved.
  ``top_k=0`` → no cap.  Empty input → empty string.
* **Dependency-light.**  Pure-python dict traversal; no numpy, no LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Default global cap on revisions in the composed context.  Bounded so a
#: heavily stacked recall can't balloon the answering context; pass ``0`` for
#: unlimited (the pre-cap behavior).
DEFAULT_CONTEXT_TOP_K = 20

#: Default per-revision cap on raw artifact content in ``"full"`` mode.
DEFAULT_REVISION_CHAR_LIMIT = 8000


def _score_of(value: Any) -> float:
    """Coerce a revision score to float; non-numeric (or bool) counts as 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def collect_top_revisions(
    memories: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Flatten sibling revisions and return the top-*limit* by score.

    When ``sibling_revisions`` exist the primary memory is skipped (it is the
    item-level shell whose recall score is on a different scale; the siblings
    include every revision of the item).  Each returned dict has ``kref``,
    ``title``, ``summary``, and ``_score`` keys.

    Used for question-specific seed selection (e.g. which revisions to
    traverse graph edges from) as well as by :func:`compose_context`.
    """
    candidates: List[Dict[str, Any]] = []
    for mem in memories:
        siblings = mem.get("sibling_revisions") or []
        if siblings:
            for sib in siblings:
                candidates.append({
                    "kref": sib.get("kref", ""),
                    "title": sib.get("title", ""),
                    "summary": sib.get("summary", ""),
                    "_score": _score_of(sib.get("_score", 0.0)),
                })
        else:
            candidates.append({
                "kref": mem.get("kref", ""),
                "title": mem.get("title", ""),
                "summary": mem.get("summary", ""),
                "_score": _score_of(mem.get("score", 0.0)),
            })
    candidates.sort(key=lambda c: c.get("_score", 0.0), reverse=True)
    return candidates[:limit]


def compose_context(
    memories: List[Dict[str, Any]],
    query: str = "",
    *,
    mode: str = "summarized",
    top_k: Optional[int] = None,
    char_limit: int = DEFAULT_REVISION_CHAR_LIMIT,
    fact_budget: int = 2,
) -> str:
    """Build answering-LLM context text from recalled memories.

    **Revision-centric assembly**: flattens all revisions (primary +
    ``sibling_revisions``) from every recalled memory, ranks them globally by
    ``_score``, applies a global *top_k* cap, and renders the survivors.
    Item-level shells are skipped whenever siblings are present — the sibling
    list contains all revisions of the item, so keeping the primary entry too
    would double-count it.

    Parameters
    ----------
    memories:
        Memory dicts as returned by ``recall_memories()``.
    query:
        The original trigger query.  Currently unused by assembly itself
        (scoring happens upstream in recall/rerank); accepted so callers can
        pass their full recall context through one signature.
    mode:
        ``"full"`` includes raw artifact content (lossless, truncated to
        *char_limit* per revision, falling back to title+summary when a
        revision has no content).  ``"summarized"`` (default) renders only
        ``title: summary`` — lossy but cheap.
    top_k:
        Global cap on revisions in the final context.  ``None`` (default)
        uses :data:`DEFAULT_CONTEXT_TOP_K`; ``0`` means unlimited.
    char_limit:
        Per-revision character cap on raw content in ``"full"`` mode.
    """
    full_mode = mode == "full"

    # --- Collect all revisions across all memories into one flat list ---
    all_revisions: List[Dict[str, Any]] = []
    for mem in memories:
        siblings = mem.get("sibling_revisions") or []
        if siblings:
            # Siblings include ALL revisions (including the published/primary
            # one) — upstream reranking decides which are relevant.  Skip the
            # primary entry to avoid double-counting; its data is in the
            # sibling list if selected.
            for sib in siblings:
                all_revisions.append({
                    "title": sib.get("title", ""),
                    "summary": sib.get("summary", ""),
                    "content": sib.get("content", ""),
                    "_score": _score_of(sib.get("_score", 0.0)),
                })
        else:
            # No siblings (non-stacked item or single revision) — use the
            # primary memory directly.
            all_revisions.append({
                "title": mem.get("title", ""),
                "summary": mem.get("summary", ""),
                "content": mem.get("content", ""),
                "_score": _score_of(mem.get("score", 0.0)),
                # Entity-bridge join evidence (graph_augmentation) — kept so
                # the top-K cut below can treat it as additive context.
                "bridge": bool(mem.get("bridge")),
                # Fact-recall leg entries get the same additive treatment.
                "fact_recall": bool(mem.get("fact_recall")),
            })

    # --- Global ranking by score (best revisions first) ---
    # Only reorder when a real score signal exists; an unscored result set
    # keeps the caller's (server relevance) order.
    has_scores = any(r.get("_score", 0.0) > 0 for r in all_revisions)
    if has_scores:
        all_revisions.sort(key=lambda r: r.get("_score", 0.0), reverse=True)

    # --- Apply global top-K cap (0 = unlimited) ---
    # Entity-bridge join evidence rides ON TOP of the cap (max +2): a bridge
    # fact is *additive* multi-hop evidence and must never displace the top-K
    # base revisions a direct answer needs (measured on conv-26: scored
    # bridge facts displacing base hits cost open-domain −0.107). Without
    # bridges this is the exact historical head-slice.
    effective_top_k = DEFAULT_CONTEXT_TOP_K if top_k is None else top_k
    # Additive partition runs UNCONDITIONALLY: conversations own the head of
    # the context; bridge and fact evidence is appended after, never
    # interleaved. This used to run only when the top-K cut fired — with
    # top_k=0 (unlimited) the global score sort let bridge/fact one-liners
    # outrank whole conversations and push the grounding session out of the
    # answering model's attention (measured: LoCoMo-Plus 500-char contexts
    # led by typed one-liners while the answer's session block trailed).
    bridge_revs = [r for r in all_revisions if r.get("bridge")]
    # Fact-recall evidence is additive on the same terms as bridges (its
    # own budget, ``fact_budget`` = the caller's fact_recall_max_results):
    # an answer-shaped claim augments the context, it must never evict
    # the conversations that ground it.
    fact_revs = [r for r in all_revisions
                 if r.get("fact_recall") and not r.get("bridge")]
    regular_revs = [r for r in all_revisions
                    if not r.get("bridge") and not r.get("fact_recall")]
    if effective_top_k > 0:
        regular_revs = regular_revs[:effective_top_k]
        bridge_revs = bridge_revs[:2]
        fact_revs = fact_revs[:fact_budget]
    all_revisions = regular_revs + bridge_revs + fact_revs

    # --- Build text from surviving revisions ---
    texts: List[str] = []
    for rev in all_revisions:
        title = rev.get("title", "")
        summary = rev.get("summary", "")
        content = rev.get("content", "")

        if full_mode and content:
            texts.append(content[:char_limit])
        elif summary:
            texts.append(f"{title}: {summary}" if title else summary)

    return "\n\n".join(texts) if texts else ""
