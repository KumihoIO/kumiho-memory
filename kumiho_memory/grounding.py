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


def ripple_grounding_stale(
    superseded_rev: Any,
    superseding_kref: str,
    *,
    cap: int = RIPPLE_FANOUT_CAP,
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
    """
    if superseded_rev is None:
        return 0
    try:
        import kumiho  # noqa: F401 — bound at call time (fake-SDK test seam)
    except Exception:  # noqa: BLE001
        return 0

    try:
        incoming = superseded_rev.get_edges(
            edge_type_filter="DEPENDS_ON", direction=_INCOMING,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("grounding ripple: get_edges failed: %s", exc)
        return 0

    superseded_uri = getattr(getattr(superseded_rev, "kref", None), "uri", "") or ""
    stamped = 0
    examined: set = set()
    for edge in incoming or []:
        src_uri = getattr(getattr(edge, "source_kref", None), "uri", "") or ""
        # INCOMING already scopes edges to those targeting the fact, but be
        # defensive for fakes/servers that ignore the direction filter: only
        # a DEPENDS_ON whose TARGET is this fact grounds a dependent in it.
        tgt_uri = getattr(getattr(edge, "target_kref", None), "uri", "") or ""
        if tgt_uri and superseded_uri and tgt_uri != superseded_uri:
            continue
        if not src_uri or src_uri == superseded_uri or src_uri in examined:
            continue
        if len(examined) >= cap:
            logger.info(
                "grounding ripple: DEPENDS_ON dependents exceed cap %d for %s — "
                "truncating", cap, superseded_uri,
            )
            break
        examined.add(src_uri)
        try:
            dep_rev = kumiho.get_revision(src_uri)
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
    return stamped
