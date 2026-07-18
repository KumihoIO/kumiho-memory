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

Budget unification (#106): this composer and
``UniversalMemoryManager.build_recalled_context`` (item-centric, MCP ``engage``)
both truncate each section's raw content to the shared
:data:`CONTEXT_BUDGET_CHARS` budget with the shared :data:`TRUNCATION_MARKER`.
For the LoCoMo bench path (this module, ``mode="full"``): sections at or under
the budget are byte-identical to pre-unification output; over-budget sections
are cut to exactly the budget with the marker inside it (previously a bare
``content[:8000]`` slice) — an intentional unification whose bench impact is
measured at the 0.19.0 full-10 RC gate, not claimed neutral.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Default global cap on revisions in the composed context.  Bounded so a
#: heavily stacked recall can't balloon the answering context; pass ``0`` for
#: unlimited (the pre-cap behavior).
DEFAULT_CONTEXT_TOP_K = 20

#: Fallback for the per-section content budget when the env override is unset
#: or malformed.  8000 is the value the LoCoMo benchmark path
#: (``compose_context``, revision-centric, ``recall_mode="full"``) used before
#: unification: sections at or under the budget render byte-identically.
#: Over-budget sections now carry the in-budget truncation marker (previously
#: a bare ``content[:8000]`` slice) — an intentional unification whose bench
#: impact is measured at the 0.19.0 full-10 RC gate, not claimed neutral.
_DEFAULT_CONTEXT_BUDGET_CHARS = 8000


def _resolve_context_budget_chars() -> int:
    """Read the shared per-section content budget from the environment.

    Env override: ``KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS`` (positive int).  A
    missing, blank, non-integer, or non-positive value falls back to
    :data:`_DEFAULT_CONTEXT_BUDGET_CHARS`.
    """
    raw = os.environ.get("KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS")
    if raw is not None and raw.strip():
        try:
            value = int(raw.strip())
        except ValueError:
            logger.warning(
                "KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS=%r is not an integer — "
                "falling back to %d.",
                raw, _DEFAULT_CONTEXT_BUDGET_CHARS,
            )
        else:
            if value > 0:
                return value
            logger.warning(
                "KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS=%d is not positive — "
                "falling back to %d.",
                value, _DEFAULT_CONTEXT_BUDGET_CHARS,
            )
    return _DEFAULT_CONTEXT_BUDGET_CHARS


#: Single source of truth for the per-section raw-content budget, in
#: characters.  BOTH assemblers — :func:`compose_context` (revision-centric,
#: the LoCoMo benchmark path) and
#: :meth:`kumiho_memory.memory_manager.UniversalMemoryManager.build_recalled_context`
#: (item-centric, the MCP ``engage`` path) — truncate each section's raw
#: content to this cap.  Override with ``KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS``.
#: Read at call time (not baked into defaults) so tests can monkeypatch it.
CONTEXT_BUDGET_CHARS = _resolve_context_budget_chars()

#: Backward-compat alias.  Both name the same per-section content budget.
DEFAULT_REVISION_CHAR_LIMIT = CONTEXT_BUDGET_CHARS

#: Marker appended when a section's content is truncated to the budget.  Shared
#: by both assemblers so the truncation policy is identical (marker parity).
TRUNCATION_MARKER = "…[truncated]"


def approx_tokens(text: str) -> int:
    """Rough token estimate for a rendered context: ``len // 4``.

    A deliberately cheap, dependency-free heuristic (the classic ~4 chars/token
    rule) so both assemblers can report an ``approx_tokens`` figure without
    pulling in a tokenizer.  Not exact — a budgeting signal, not a billing one.
    """
    return len(text) // 4


def truncate_section(text: str, limit: Optional[int] = None) -> str:
    """Truncate one section's raw content to *limit* chars, shared by both
    assemblers so the per-section truncation policy is identical.

    *limit* ``None`` resolves to :data:`CONTEXT_BUDGET_CHARS` at call time (so
    a monkeypatched budget is honored); numeric values — including ``0`` — are
    applied as-is (``0`` empties the section, matching the historical
    ``content[:0]`` slice).  Content at or under the budget is returned
    **unchanged** (byte-identical).  When truncation fires the result is
    exactly *limit* chars **total**: the text is cut to
    ``limit - len(TRUNCATION_MARKER)`` and the marker appended, so the budget
    is a hard ceiling.  A limit too small to fit the marker degenerates to a
    bare hard slice (marker omitted) — the ceiling still holds.
    """
    if limit is None:
        limit = CONTEXT_BUDGET_CHARS
    if len(text) <= limit:
        return text
    cut = limit - len(TRUNCATION_MARKER)
    if cut <= 0:
        return text[:limit]
    return text[:cut] + TRUNCATION_MARKER


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
    char_limit: Optional[int] = None,
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
        ``None`` (default) resolves to the shared
        :data:`CONTEXT_BUDGET_CHARS` at call time; numeric values —
        including ``0`` — apply as-is (``0`` empties the section, the
        historical ``content[:0]`` slice semantics).
    """
    full_mode = mode == "full"
    # Shared per-section content budget (monkeypatch-friendly: read now, not
    # baked into the signature default).
    effective_char_limit = (
        CONTEXT_BUDGET_CHARS if char_limit is None else char_limit
    )

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
                    # Dispute / staleness markers are ITEM-level (the recall
                    # marker matches the memory's own kref OR any sibling kref)
                    # so they ride onto every rendered sibling block — a
                    # stacked contested memory must not lose its note.
                    "contested_by": mem.get("contested_by") or [],
                    "grounding_stale": bool(mem.get("grounding_stale")),
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
                # CONTRADICTS marker (graph_augmentation): a one-line "disputed"
                # note is appended to this memory's block so the answering model
                # sees the fact is contested instead of one side unmarked.
                # Carried only on the non-sibling branch, mirroring bridge /
                # fact_recall (those additive markers ride here too).
                "contested_by": mem.get("contested_by") or [],
                # Grounding-staleness marker (#95): a dependent decision whose
                # grounding fact was superseded gets a terse "grounding stale"
                # note, so the answering model weighs it as possibly outdated.
                "grounding_stale": bool(mem.get("grounding_stale")),
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
            entry_text = truncate_section(content, effective_char_limit)
        elif summary:
            entry_text = f"{title}: {summary}" if title else summary
        else:
            continue
        # Contested memories carry a terse "disputed" note on the same block,
        # so the marker travels with the fact it qualifies.
        contested = rev.get("contested_by")
        if contested:
            n = len(contested)
            entry_text += (
                f"\n[contested: disputed by {n} other stored "
                f"memor{'y' if n == 1 else 'ies'}]"
            )
        # Grounding-stale dependents carry a terse note on the same block, so
        # the answering model knows a fact this was based on has been superseded.
        if rev.get("grounding_stale"):
            entry_text += (
                "\n[grounding stale: a fact this was based on was superseded]"
            )
        texts.append(entry_text)

    return "\n\n".join(texts) if texts else ""
