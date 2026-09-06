"""One revision-scoped protocol for replacing facts and decisions.

Edge first, then demote the exact target revision and invalidate its grounding.
Replays repair partial writes even when the edge already exists. This is not a
transaction: callers must surface/retry failures, not mark incomplete work done.
Profile-history SUPERSEDES edges are not belief replacements and stay separate.

Concurrency and recovery contract (kumiho-memory#27; see
``docs/BELIEF_REVISION_CONCURRENCY.md`` for the full statement):

* **Convergent replay.** Repeating the same operation converges: the edge is
  created at most once (existence pre-check), the target is demoted at most once
  (status pre-check), and the grounding ripple is idempotent. Two processes
  replaying the same replacement therefore leave one edge, one demotion.
* **Conflict is preserved, never resolved by arrival order.** A pre-existing
  reverse ``SUPERSEDES`` (target already supersedes source) and a bounded
  ``SUPERSEDES`` cycle are rejected before any write; a reverse edge that
  appears concurrently (observed only after our own edge landed) is reported as
  ``reverse_conflict`` and the demotion is withheld, so neither side is silently
  declared the winner.
* **Scope integrity.** Source and target must share a project scope; a
  cross-scope reference is rejected before any write.
* **Bounded foreground, resumable invalidation.** The ripple is bounded per
  call; when it truncates it persists a durable pending marker on the superseded
  fact so a later maintenance pass (or replay) resumes it. A result distinguishes
  foreground completion from pending asynchronous work.

What this module does NOT provide, and why: true cross-process atomicity /
exactly-once requires a server conditional-write (compare-and-set) or unique
operation-identity primitive. That is upstream work (kumiho-SDKs / server); a
process-local lock is not a distributed guarantee. Until it exists, the harmful
race this module cannot fully exclude is two writers creating *mutually reverse*
edges at the same instant; the post-write reverse re-check below detects and
quarantines it best-effort rather than demoting both.
"""
from dataclasses import dataclass, field
import hashlib
import logging
from typing import Any, Dict, List, Optional

from .grounding import (
    GROUNDING_RIPPLE_PENDING_META,
    ripple_grounding_stale,
)

logger = logging.getLogger(__name__)

#: How deep to walk the target's outgoing SUPERSEDES chain looking for a cycle
#: back to the source. Depth 1 is the direct reverse edge; beyond that catches
#: A->B->C->A. Bounded because the walk costs one ``get_revision`` per hop and a
#: read outage mid-walk fails the operation closed (retryable), never guesses.
SUPERSEDE_CYCLE_MAX_DEPTH = 8


@dataclass
class SupersessionResult:
    """Machine-readable outcome of one replacement attempt.

    The original four counters are unchanged for existing callers. The added
    fields expose conflict state and asynchronous progress: ``complete`` is
    True only when the foreground work finished with nothing left pending.
    """

    created: bool = False
    linked: bool = False
    demoted: bool = False
    stale: int = 0
    error: str = ""
    # --- added (#27), all additive ---
    reverse_conflict: bool = False
    cycle_rejected: bool = False
    ripple_truncated: bool = False
    ripple_pending: bool = False
    op_id: str = ""
    events: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Foreground work done, edge+demotion consistent, nothing pending.

        A conflict (reverse/cycle), an error, or pending ripple work all mean
        the caller must not treat the replacement as fully settled.
        """
        return bool(
            self.linked and not self.error and not self.reverse_conflict
            and not self.cycle_rejected and not self.ripple_pending
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id, "created": self.created, "linked": self.linked,
            "demoted": self.demoted, "stale": self.stale,
            "reverse_conflict": self.reverse_conflict, "cycle_rejected": self.cycle_rejected,
            "ripple_truncated": self.ripple_truncated, "ripple_pending": self.ripple_pending,
            "complete": self.complete, "error": self.error, "events": list(self.events),
        }


def _uri(rev: Any) -> str:
    return getattr(getattr(rev, "kref", None), "uri", "") or ""


def _project_scope(uri: str) -> str:
    """The project segment of a kref, i.e. the first path element.

    ``kref://Project/space/slug.kind?r=1`` -> ``Project``. Empty when the uri
    is not a kref, which the caller treats as "cannot verify scope".
    """
    if not uri.startswith("kref://"):
        return ""
    return uri[len("kref://"):].split("/", 1)[0]


def _op_id(src: str, dst: str) -> str:
    """Stable identity for this replacement (scope + endpoints, base krefs).

    Deterministic across processes and replays so a retry is recognizably the
    same operation. Uses the base krefs (``?r=N`` stripped) because a typed
    node is anchored at one revision and the retry addresses the same belief.
    """
    a, b = src.split("?", 1)[0], dst.split("?", 1)[0]
    scope = _project_scope(src) or _project_scope(dst)
    return hashlib.sha256(f"{scope}\x1f{a}\x1f{b}".encode("utf-8")).hexdigest()[:16]


def _outgoing_supersedes_targets(rev: Any) -> List[str]:
    return [
        getattr(getattr(edge, "target_kref", None), "uri", "") or ""
        for edge in rev.get_edges(edge_type_filter="SUPERSEDES", direction=0)
    ]


def _forms_cycle(target: Any, source_uri: str, *, depth: int) -> bool:
    """True if source is reachable by following SUPERSEDES OUT from target.

    Bounded DFS. Raises on a read outage mid-walk so the caller fails closed
    rather than creating an edge that might complete a cycle it could not see.
    Depth 1 is the direct reverse edge (target already supersedes source).
    """
    try:
        import kumiho
    except Exception:  # noqa: BLE001
        # No SDK to fetch further hops: fall back to the direct reverse check
        # (depth 1), which needs only the target object already in hand.
        return source_uri in _outgoing_supersedes_targets(target)

    base_source = source_uri.split("?", 1)[0]
    seen: set = set()
    frontier: List[Any] = [target]
    for _hop in range(max(1, depth)):
        nxt: List[Any] = []
        for rev in frontier:
            for tgt_uri in _outgoing_supersedes_targets(rev):
                if not tgt_uri or tgt_uri in seen:
                    continue
                if tgt_uri.split("?", 1)[0] == base_source:
                    return True
                seen.add(tgt_uri)
                nxt.append(kumiho.get_revision(tgt_uri))  # may raise -> fail closed
        if not nxt:
            break
        frontier = [r for r in nxt if r is not None]
    return False


def supersede_revision(
    source: Any, target: Any, metadata: Optional[Dict[str, str]] = None,
) -> SupersessionResult:
    """Ensure source SUPERSEDES target; never demote without a confirmed edge.

    Returns new-write counters separately from edge existence, plus conflict and
    async-progress state (:class:`SupersessionResult`). On a metadata failure the
    edge remains useful and a replay retries demotion and the bounded grounding
    ripple. Unrelated metadata and the target's revision identity survive.
    """
    result = SupersessionResult()
    src = _uri(source)
    dst = _uri(target)
    if source is None or target is None or source is target or not src or not dst or src == dst:
        result.error = "Supersession requires two distinct revision references"
        return result
    result.op_id = _op_id(src, dst)

    # Scope integrity: a belief edge never crosses a project boundary. An
    # unverifiable scope (non-kref uri) is allowed through -- the fakes and some
    # legacy krefs are not project-prefixed -- but two DIFFERENT known scopes are
    # rejected before any write.
    src_scope, dst_scope = _project_scope(src), _project_scope(dst)
    if src_scope and dst_scope and src_scope != dst_scope:
        result.error = (
            f"Supersession crosses project scope: {src_scope!r} -> {dst_scope!r}"
        )
        logger.warning("%s", result.error)
        return result

    try:
        # A read outage is NOT an absent edge. Preserve code capture's strict
        # retry contract: surface uncertainty before any duplicate/destructive
        # write rather than completing the commit over a guessed graph state.
        result.linked = dst in _outgoing_supersedes_targets(source)
        # Reject a replacement that would create (or complete) a cycle, of which
        # the direct reverse edge (target already supersedes source) is the
        # depth-1 case. Preserves an unresolved conflict instead of silently
        # accepting a mutually-invalid resolved state.
        if not result.linked and _forms_cycle(target, src, depth=SUPERSEDE_CYCLE_MAX_DEPTH):
            result.cycle_rejected = True
            result.error = "reverse or cyclic SUPERSEDES would invalidate both beliefs"
            result.events.append("cycle_rejected")
            logger.warning("%s (%s -> %s)", result.error, src, dst)
            return result
        if not result.linked:
            if source.create_edge(target, "SUPERSEDES", metadata=metadata or {}) is False:
                raise RuntimeError("edge creation rejected")
            result.created = result.linked = True
            result.events.append("edge_created")
    except Exception as exc:
        result.error = f"Supersession edge failed: {exc}"
        logger.warning("%s", result.error)
        return result

    # Post-write reverse re-check: a concurrent writer may have created the
    # opposite edge between our cycle check and our create. Observing it now
    # means two writers disagree; quarantine (withhold demotion) rather than
    # demote one side. Best-effort -- a read failure here leaves the edge and
    # lets a replay settle it.
    try:
        if src in _outgoing_supersedes_targets(target):
            result.reverse_conflict = True
            result.error = result.error or "concurrent reverse SUPERSEDES detected; demotion withheld"
            result.events.append("reverse_conflict")
            logger.warning("belief-revision conflict: %s <-> %s both supersede", src, dst)
            return result
    except Exception as exc:  # noqa: BLE001
        logger.debug("supersession: reverse re-check failed (%s); proceeding", exc)

    try:
        if (getattr(target, "metadata", {}) or {}).get("status") != "superseded":
            if target.set_attribute("status", "superseded") is False:
                raise RuntimeError("status update rejected")
            result.demoted = True
            result.events.append("target_demoted")
    except Exception as exc:
        result.error = f"Supersession status failed: {exc}"
        logger.warning("%s", result.error)
    # Also retry on an existing edge: a previous process may have stopped
    # between edge creation and ripple. Decisions can ground other decisions.
    result.stale = ripple_grounding_stale(target, src)
    if result.stale:
        result.events.append(f"grounding_rippled:{result.stale}")
    # Truncation leaves a durable pending marker on the fact (grounding module);
    # surface it so the caller knows invalidation is not yet complete and a
    # maintenance/replay pass must finish it.
    pending = str((getattr(target, "metadata", {}) or {}).get(GROUNDING_RIPPLE_PENDING_META, "") or "")
    if pending:
        result.ripple_truncated = True
        result.ripple_pending = True
        result.events.append("ripple_pending")
    return result
