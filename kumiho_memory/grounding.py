"""Grounding-staleness ripple — flag DEPENDS_ON dependents of a superseded fact.

Closes ontology gap G4 / Kumiho paper §15.6 (the deferred feature Atlas shipped
as "Ripple"). ``DEPENDS_ON`` edges ground a decision in the facts it was based
on (``decision --DEPENDS_ON--> fact``); superseding a fact used to trigger
nothing, so recall kept serving the dependent decisions as if their grounding
were intact. This module stamps each such dependent when a ``SUPERSEDES`` edge
lands on the fact, so:

* recall surfaces an additive ``grounding_stale`` marker (``graph_augmentation``
  + ``memory_manager`` read the flag off metadata already fetched — no new
  round-trip), and
* Dream State maintenance re-examines the flag and clears it once grounding is
  re-confirmed (``graph_maintenance``).

Conventions (strict):

* **Metadata is canonical, tag is mirrored/best-effort** — same split as
  :mod:`kumiho_memory.evidence`. The ``grounding_stale`` metadata key is the
  source of truth; the ``grounding:stale`` graph tag is applied best-effort and
  a per-tag failure is tolerated (a reader consults metadata first).
* **gRPC metadata values are strings** — the flag is the literal ``"true"``.
* **Deterministic, keyless, best-effort** — no LLM, every failure is logged and
  swallowed, the ripple never breaks the write it rides on.
* **Bounded fan-out** — at most :data:`RIPPLE_FANOUT_CAP` dependents per
  supersede (a decision-grounding fan-in is normally 0-2; the cap only guards a
  pathological hub). Truncation is logged, never silent.
* **Idempotent** — a dependent already carrying the flag is neither re-stamped
  nor re-tagged, so a re-decompose adds no duplicate tags.

The ripple runs on the WRITE path (inside the bounded decompose worker), NOT on
recall: it costs, per supersede, one ``get_edges`` on the fact plus, per
dependent, one ``get_revision`` and (only when newly stamped) one
``set_metadata`` + one ``tag``. The recall marker reuses already-fetched
metadata and adds zero round-trips.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Canonical metadata flag on a grounding-stale dependent (string ``"true"``;
#: gRPC metadata values are strings).
GROUNDING_STALE_META = "grounding_stale"
#: Companion metadata: kref of the superseding fact whose landing staled this
#: dependent's grounding.
GROUNDING_STALE_SUPERSEDED_BY_META = "grounding_stale_superseded_by"
#: Durable pending-work marker written on a SUPERSEDED FACT when its ripple
#: truncated at the fan-out cap (kumiho-memory#27). Its value is the superseding
#: kref, so a later maintenance pass (or a replay) can resume the invalidation
#: from durable state after a process death, and clear it once no unflagged
#: dependent remains. Never silently dropped: a truncated ripple that leaves
#: this set is discoverable, not a swallowed failure reported as zero.
GROUNDING_RIPPLE_PENDING_META = "grounding_ripple_pending"
#: Companion durable cursor: how many distinct DEPENDS_ON dependents of this
#: fact have already been processed, so a resume continues through the fan-in
#: instead of re-scanning the head. Advances monotonically within one drain;
#: cleared with the pending marker when the fan-in is fully processed. Assumes
#: the server returns this fact's DEPENDS_ON edges in a stable order within a
#: drain (idempotent stamping makes a reprocess harmless if it is not).
GROUNDING_RIPPLE_CURSOR_META = "grounding_ripple_cursor"
#: Mirrored graph tag (metadata is canonical; the tag is best-effort per-tag).
GROUNDING_STALE_TAG = "grounding:stale"
#: The cleared value written by maintenance (metadata is never deleted, so a
#: reader that only sees metadata still reads a definite non-stale state).
_TRUE = "true"
_FALSE = "false"

#: Max DEPENDS_ON dependents examined/stamped per supersede (fan-out guard).
RIPPLE_FANOUT_CAP = 20

# Edge direction constant (mirrors kumiho.INCOMING / graph_maintenance._INCOMING;
# kept literal so the module has no hard kumiho import at load time).
_INCOMING = 1

# NOTE (future work, deliberately NOT implemented): an optional LLM re-grade
# could, at maintenance time, judge whether the superseding fact actually
# changes the dependent decision's basis (vs. a cosmetic revision) and clear or
# keep the flag on that judgment. The keyless deterministic core here is the
# mandatory path (issue #95); the LLM re-grade would slot into
# ``graph_maintenance.GraphMaintainer._clear_stale_grounding`` as an extra,
# opt-in signal — never on the plugin's keyless path.


def is_grounding_stale(meta: Optional[Dict[str, Any]]) -> bool:
    """True if *meta* carries the canonical grounding-stale flag."""
    return str((meta or {}).get(GROUNDING_STALE_META, "")).lower() == _TRUE


def apply_grounding_marker(entry: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    """Stamp the additive grounding-stale recall marker on *entry* from *meta*.

    Reuses metadata already fetched onto the revision the recall path touches
    (mirrors the ``evidence_level`` / ``source`` reads) — zero extra round-trip.
    Purely additive: no score change, no reordering, no removal (mirrors #94's
    ``contested_by``).
    """
    if not is_grounding_stale(meta):
        return
    entry["grounding_stale"] = True
    superseded_by = str((meta or {}).get(GROUNDING_STALE_SUPERSEDED_BY_META, "") or "")
    if superseded_by:
        entry["superseded_by"] = superseded_by


def _default_get_revision(uri: str) -> Any:
    """Resolve a revision through the ambient ``kumiho`` SDK (the write-path
    seam that tests monkeypatch via ``sys.modules['kumiho']``)."""
    import kumiho  # bound at call time
    return kumiho.get_revision(uri)


def ripple_grounding_stale(
    superseded_rev: Any,
    superseding_kref: str,
    *,
    cap: int = RIPPLE_FANOUT_CAP,
    get_revision: Optional[Any] = None,
) -> int:
    """Flag decisions grounded in *superseded_rev* as grounding-stale.

    Finds the revisions with a ``DEPENDS_ON`` edge INTO *superseded_rev* (the
    ``decision --DEPENDS_ON--> fact`` grounding written by ontology decompose)
    and stamps each with ``grounding_stale="true"`` +
    ``grounding_stale_superseded_by=<superseding_kref>`` metadata plus the
    mirrored ``grounding:stale`` tag.

    Best-effort, keyless, deterministic, bounded (``cap``), idempotent (an
    already-stale dependent is skipped, so no re-stamp / duplicate tag). Returns
    the count of dependents newly stamped (0 on any failure).

    ``get_revision`` overrides how a dependent revision is fetched; it defaults
    to the ambient ``kumiho`` SDK (the write path), and the maintenance resume
    pass passes its own injected SDK so a resume runs against the same client
    the sweep uses rather than a process-global one.

    Resumable truncation (kumiho-memory#27): when the distinct DEPENDS_ON
    dependents exceed ``cap`` the ripple stamps ``cap`` of them and writes a
    durable :data:`GROUNDING_RIPPLE_PENDING_META` marker + cursor on
    *superseded_rev*, so the remaining work is discoverable and a later pass --
    :func:`resume_grounding_ripple`, or the maintenance sweep -- finishes it
    after a process death, continuing from the cursor. A run that processes
    every remaining dependent clears the marker. Truncation is therefore never
    a swallowed failure reported as zero affected dependents.
    """
    if superseded_rev is None:
        return 0
    if get_revision is None:
        try:
            import kumiho  # noqa: F401 — availability gate for the default path
        except Exception:  # noqa: BLE001
            return 0
        get_revision = _default_get_revision

    try:
        incoming = superseded_rev.get_edges(
            edge_type_filter="DEPENDS_ON", direction=_INCOMING,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("grounding ripple: get_edges failed: %s", exc)
        return 0

    superseded_uri = getattr(getattr(superseded_rev, "kref", None), "uri", "") or ""

    # Distinct, valid dependents (a DEPENDS_ON whose TARGET is this fact, not
    # self). Built first so truncation is measured against the real fan-in, and
    # the pending marker reflects whether work actually remains.
    candidates: list = []
    seen_src: set = set()
    for edge in incoming or []:
        src_uri = getattr(getattr(edge, "source_kref", None), "uri", "") or ""
        tgt_uri = getattr(getattr(edge, "target_kref", None), "uri", "") or ""
        if tgt_uri and superseded_uri and tgt_uri != superseded_uri:
            continue
        if not src_uri or src_uri == superseded_uri or src_uri in seen_src:
            continue
        seen_src.add(src_uri)
        candidates.append(src_uri)

    # Resume from the durable cursor so a truncated drain continues through the
    # fan-in instead of re-scanning the head. First call (no cursor) starts at 0.
    try:
        cursor = int(str((getattr(superseded_rev, "metadata", {}) or {}).get(
            GROUNDING_RIPPLE_CURSOR_META, "") or "0"))
    except (TypeError, ValueError):
        cursor = 0
    cursor = max(0, min(cursor, len(candidates)))

    stamped = 0
    processed = cursor
    for src_uri in candidates[cursor:cursor + cap]:
        processed += 1
        try:
            dep_rev = get_revision(src_uri)
        except Exception as exc:  # noqa: BLE001
            logger.debug("grounding ripple: get_revision %s failed: %s", src_uri, exc)
            continue
        if dep_rev is None:
            continue
        # Idempotent: a dependent already flagged (canonical metadata) is not
        # re-stamped, so a re-decompose never doubles the tag.
        if is_grounding_stale(getattr(dep_rev, "metadata", {}) or {}):
            continue
        try:
            dep_rev.set_metadata({
                GROUNDING_STALE_META: _TRUE,
                GROUNDING_STALE_SUPERSEDED_BY_META: superseding_kref or "",
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("grounding ripple: set_metadata %s failed: %s", src_uri, exc)
            continue
        try:
            dep_rev.tag(GROUNDING_STALE_TAG)
        except Exception as exc:  # noqa: BLE001
            # Metadata is canonical; a missing mirrored tag is tolerated.
            logger.debug("grounding ripple: tag %s failed: %s", src_uri, exc)
        stamped += 1

    remaining = processed < len(candidates)
    if remaining:
        logger.info(
            "grounding ripple: DEPENDS_ON dependents (%d) exceed cap %d for %s — "
            "%d/%d processed, remainder deferred to a resume pass",
            len(candidates), cap, superseded_uri, processed, len(candidates),
        )
    _persist_ripple_progress(
        superseded_rev, superseding_kref if remaining else "", processed if remaining else 0,
    )
    return stamped


def _persist_ripple_progress(superseded_rev: Any, superseding_kref: str, cursor: int) -> None:
    """Set (remaining) or clear (drained) the durable pending marker + cursor.

    Only writes when the state actually changes, so the common non-truncated
    ripple pays nothing. Best-effort: a write failure leaves the previous marker
    and the next pass re-evaluates -- it never breaks the ripple it rides on.
    """
    try:
        meta = getattr(superseded_rev, "metadata", {}) or {}
        cur_pending = str(meta.get(GROUNDING_RIPPLE_PENDING_META, "") or "")
        cur_cursor = str(meta.get(GROUNDING_RIPPLE_CURSOR_META, "") or "")
    except Exception:  # noqa: BLE001
        return
    want_pending = superseding_kref or ""
    want_cursor = str(cursor) if superseding_kref else ""
    if cur_pending == want_pending and cur_cursor == want_cursor:
        return
    try:
        superseded_rev.set_metadata({
            GROUNDING_RIPPLE_PENDING_META: want_pending,
            GROUNDING_RIPPLE_CURSOR_META: want_cursor,
        })
    except Exception as exc:  # noqa: BLE001
        logger.debug("grounding ripple: progress-marker write failed: %s", exc)


def has_pending_ripple(meta: Optional[Dict[str, Any]]) -> str:
    """The superseding kref of a fact's deferred ripple, or ``""`` if none."""
    return str((meta or {}).get(GROUNDING_RIPPLE_PENDING_META, "") or "")


def resume_grounding_ripple(
    superseded_rev: Any, *, cap: int = RIPPLE_FANOUT_CAP, get_revision: Optional[Any] = None,
) -> int:
    """Resume a truncated ripple from the fact's durable pending marker.

    Reads :data:`GROUNDING_RIPPLE_PENDING_META` off *superseded_rev* and, when
    set, re-runs :func:`ripple_grounding_stale` for it. Because the ripple is
    idempotent (already-stale dependents skipped) each call flags up to ``cap``
    more and re-evaluates truncation, clearing the marker once none remain. A
    fact with no pending marker is a no-op returning 0. Safe to call repeatedly
    and from more than one process; convergent under the same replay contract as
    the write path.
    """
    if superseded_rev is None:
        return 0
    pending = has_pending_ripple(getattr(superseded_rev, "metadata", {}) or {})
    if not pending:
        return 0
    return ripple_grounding_stale(
        superseded_rev, pending, cap=cap, get_revision=get_revision,
    )
