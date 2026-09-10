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
* **Deterministic and keyless** - failed dependents retain pending work;
  failed progress writes are surfaced so callers can replay the partial write.
* **Bounded fan-out** — at most :data:`RIPPLE_FANOUT_CAP` dependents per
  supersede (a decision-grounding fan-in is normally 0-2; the cap only guards a
  pathological hub). Truncation is logged, never silent.
* **Idempotent** — a dependent already carrying the flag is neither re-stamped
  nor re-tagged, so a re-decompose adds no duplicate tags.

The ripple runs on the WRITE path (inside the bounded decompose worker), NOT on
recall: it costs, per supersede, one ``get_edges`` on the fact plus, per
dependent, one ``get_revision`` and (only when newly stamped) one
``set_metadata`` + one ``tag``. Nonempty batches also acknowledge durable
progress before processing and after advancing/clearing the cursor. The recall marker reuses already-fetched
metadata and adds zero round-trips.
"""

from __future__ import annotations

import hashlib
import json
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
#: Cursor counts a successfully processed prefix of a sorted dependency snapshot.
#: Its fingerprint binds both membership and the superseding revision. Changed
#: membership resets the cursor; backend edge ordering cannot skip work.
GROUNDING_RIPPLE_CURSOR_META = "grounding_ripple_cursor"
GROUNDING_RIPPLE_SNAPSHOT_META = "grounding_ripple_snapshot"
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
    the count of dependents newly stamped. Progress-write failures raise so the
    caller cannot report an undiscoverable partial write as complete.

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
    cap = max(1, min(int(cap), RIPPLE_FANOUT_CAP))
    if get_revision is None:
        try:
            import kumiho  # noqa: F401 — availability gate for the default path
        except Exception:  # noqa: BLE001
            _persist_ripple_progress(superseded_rev, superseding_kref, 0)
            return 0
        get_revision = _default_get_revision

    try:
        incoming = superseded_rev.get_edges(
            edge_type_filter="DEPENDS_ON", direction=_INCOMING,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("grounding ripple: get_edges failed: %s", exc)
        _persist_ripple_progress(superseded_rev, superseding_kref, 0)
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

    # Persist a stable snapshot identity, not an unbounded list of references.
    # Insertion/removal/reordering must never turn an ordinal cursor into a skip.
    candidates.sort()
    snapshot = hashlib.sha256(json.dumps(
        [superseding_kref, candidates], ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    meta = getattr(superseded_rev, "metadata", {}) or {}
    try:
        cursor = int(str(meta.get(GROUNDING_RIPPLE_CURSOR_META, "") or "0"))
    except (TypeError, ValueError):
        cursor = 0
    if meta.get(GROUNDING_RIPPLE_SNAPSHOT_META) != snapshot:
        cursor = 0
    cursor = max(0, min(cursor, len(candidates)))

    # Record pending work BEFORE touching dependents. A crash or failed read
    # leaves a retryable prefix; a progress-write failure must reach the caller.
    if candidates:
        _persist_ripple_progress(superseded_rev, superseding_kref, cursor, snapshot)
    stamped = 0
    processed = cursor
    failed_prefix = False
    for src_uri in candidates[cursor:cursor + cap]:
        try:
            dep_rev = get_revision(src_uri)
            if dep_rev is None:
                raise RuntimeError("dependent revision unavailable")
            if not is_grounding_stale(getattr(dep_rev, "metadata", {}) or {}):
                if dep_rev.set_metadata({
                    GROUNDING_STALE_META: _TRUE,
                    GROUNDING_STALE_SUPERSEDED_BY_META: superseding_kref or "",
                }) is False:
                    raise RuntimeError("dependent metadata update rejected")
                stamped += 1
                try:
                    dep_rev.tag(GROUNDING_STALE_TAG)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("grounding ripple: tag %s failed: %s", src_uri, exc)
        except Exception as exc:  # noqa: BLE001
            # Do not advance past failed work. Later successful replays may
            # re-read an already stamped prefix, at most cap revisions per run.
            logger.debug("grounding ripple: dependent %s failed: %s", src_uri, exc)
            failed_prefix = True
            continue
        if not failed_prefix:
            processed += 1

    remaining = processed < len(candidates)
    if remaining:
        logger.info("grounding ripple: %d/%d dependents processed for %s; pending",
                    processed, len(candidates), superseded_uri)
    _persist_ripple_progress(
        superseded_rev, superseding_kref if remaining else "",
        processed if remaining else 0, snapshot if remaining else "",
    )
    return stamped


def _persist_ripple_progress(
    superseded_rev: Any, superseding_kref: str, cursor: int, snapshot: str = "",
) -> None:
    """Persist or clear bounded progress; never hide a missing acknowledgement.

    The supersession caller surfaces failure for replay. An earlier durable
    marker remains discoverable if updating or clearing it fails.
    """
    meta = getattr(superseded_rev, "metadata", {}) or {}
    wanted = {
        GROUNDING_RIPPLE_PENDING_META: superseding_kref or "",
        GROUNDING_RIPPLE_CURSOR_META: str(cursor) if superseding_kref else "",
        GROUNDING_RIPPLE_SNAPSHOT_META: snapshot if superseding_kref else "",
    }
    if all(str(meta.get(key, "") or "") == value for key, value in wanted.items()):
        return
    if superseded_rev.set_metadata(wanted) is False:
        raise RuntimeError("grounding progress metadata update rejected")
    # The SDK returns a fresh Revision, leaving this object's metadata unchanged.
    # Reflect only the acknowledged keys locally for the result/resume observer.
    superseded_rev.metadata.update(wanted)


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
    fact with no pending marker is a no-op returning 0. Replays are idempotent;
    cross-process progress updates remain best-effort without server CAS.
    """
    if superseded_rev is None:
        return 0
    pending = has_pending_ripple(getattr(superseded_rev, "metadata", {}) or {})
    if not pending:
        return 0
    return ripple_grounding_stale(
        superseded_rev, pending, cap=cap, get_revision=get_revision,
    )
