"""Valid-time intervals (ontology gap G8) — additive ``valid_from`` / ``valid_to``
metadata + opt-in "as-of" recall.

Kumiho already carries a *point* of valid time on a memory: ``event_date``
(``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``). Gruber's encoding-bias critique (G8)
is that a belief is usually true over an *interval*, not at an instant — "Alice
worked at Acme from 2019 to 2022" is one fact, not a point event. This module
adds that interval, ADDITIVELY:

* **Write side** — ``valid_from`` / ``valid_to`` are stored as extra revision
  metadata alongside ``event_date``. ``event_date`` is never read or written
  here, and neither is issue #119's ``event_date_confidence`` — the three keys
  are independent.
* **Read side** — an opt-in "as-of" filter (``KUMIHO_MEMORY_AS_OF_RECALL``,
  default OFF) *demotes* (never deletes) facts whose interval excludes the
  requested instant, so a query "as of 2020" ranks facts valid in 2020 ahead of
  ones that only became true later or had already lapsed.

**Byte-identical when OFF.** :func:`apply_as_of_recall` is a strict no-op unless
BOTH the flag is on AND an as-of instant is supplied, and even then it leaves the
list untouched when no memory is actually excluded — so the default recall path
is unchanged.

Interval semantics (half-open dates are padded to whole periods so partial
precision behaves intuitively):

* ``valid_from`` pads to the START of its period (``2020`` → ``2020-01-01``);
  a fact is not yet valid before it.
* ``valid_to`` pads to the END of its period (``2020`` → ``2020-12-31``,
  ``2020-03`` → ``2020-03-31``); a fact is no longer valid after it.
* A missing bound is open (``-inf`` / ``+inf``); a fact with neither bound is
  always valid and is never demoted.
"""

from __future__ import annotations

import calendar
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Env flag gating the as-of recall filter. Default OFF (unset/empty). Any of
#: ``1/true/yes/on`` (case-insensitive) turns it on — the plugin-server flag
#: convention (mirrors ``KUMIHO_DREAM_MAINTAIN_GRAPH``).
AS_OF_RECALL_ENV = "KUMIHO_MEMORY_AS_OF_RECALL"

#: Additive interval metadata keys (distinct from ``event_date`` and from issue
#: #119's ``event_date_confidence``).
VALID_FROM_META = "valid_from"
VALID_TO_META = "valid_to"

#: Additive recall marker stamped on a memory the as-of pass demoted (so a
#: caller can surface "not valid as of <date>" without recomputing the interval).
AS_OF_EXCLUDED_KEY = "as_of_excluded"

# A valid-time bound must be a clean ISO-8601 calendar date at year, month, or
# day precision — the exact grammar ``event_date`` already uses
# (``memory_manager._ISO_EVENT_DATE_RE``). Anything else (prose, timestamps) is
# rejected so a malformed bound never silently narrows recall.
_ISO_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def as_of_recall_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True if the as-of recall flag is set truthy in *env* (defaults to
    ``os.environ``)."""
    source = os.environ if env is None else env
    return str(source.get(AS_OF_RECALL_ENV, "")).strip().casefold() in _TRUTHY


def normalize_valid_date(value: Any) -> str:
    """Return *value* as a clean ISO date string, or ``""`` if it isn't one.

    Mirrors the ``event_date`` acceptance grammar so a caller can validate a
    ``valid_from`` / ``valid_to`` before writing it as metadata.
    """
    if not isinstance(value, str):
        return ""
    v = value.strip()
    return v if _ISO_DATE_RE.match(v) else ""


def _pad_lower(value: Any) -> Optional[date]:
    """Parse a valid-time lower bound, padding partial precision to the START of
    its period (``2020`` → 2020-01-01, ``2020-03`` → 2020-03-01)."""
    v = normalize_valid_date(value)
    if not v:
        return None
    parts = v.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except ValueError:
        return None


def _pad_upper(value: Any) -> Optional[date]:
    """Parse a valid-time upper bound, padding partial precision to the END of
    its period (``2020`` → 2020-12-31, ``2020-03`` → 2020-03-31) so a fact valid
    "through 2020" is not spuriously excluded on 2020-06-01."""
    v = normalize_valid_date(value)
    if not v:
        return None
    parts = v.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 12
        if len(parts) > 2:
            day = int(parts[2])
        else:
            day = calendar.monthrange(year, month)[1]
        return date(year, month, day)
    except ValueError:
        return None


def interval_of(meta: Optional[Dict[str, Any]]) -> Tuple[Optional[date], Optional[date]]:
    """The ``(lower, upper)`` valid-time bounds parsed from *meta*.

    Either or both may be ``None`` (open bound / absent). Reads the same keys off
    a recall entry dict or a raw revision-metadata dict.
    """
    if not meta:
        return (None, None)
    return (_pad_lower(meta.get(VALID_FROM_META)), _pad_upper(meta.get(VALID_TO_META)))


def interval_excludes(meta: Optional[Dict[str, Any]], as_of: date) -> bool:
    """True if the memory's valid-time interval EXCLUDES *as_of*.

    A memory with neither bound (the common case — no interval was ever written)
    is always valid, so this returns ``False`` and the memory is never demoted.
    """
    lower, upper = interval_of(meta)
    if lower is not None and as_of < lower:
        return True
    if upper is not None and as_of > upper:
        return True
    return False


def apply_valid_interval_marker(
    entry: Dict[str, Any], meta: Optional[Dict[str, Any]]
) -> None:
    """Surface ``valid_from`` / ``valid_to`` onto a recall *entry* from *meta*.

    Additive and lossless (mirrors the ``event_date`` / grounding-marker reads in
    ``memory_manager``): only sets a key when the metadata carries a *valid* ISO
    bound, so ungraded/legacy revisions are unchanged and a malformed bound is
    dropped rather than surfaced.
    """
    if not meta:
        return
    vf = normalize_valid_date(meta.get(VALID_FROM_META))
    if vf:
        entry[VALID_FROM_META] = vf
    vt = normalize_valid_date(meta.get(VALID_TO_META))
    if vt:
        entry[VALID_TO_META] = vt


def _as_of_date(as_of: Any) -> Optional[date]:
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    return None


def apply_as_of_recall(
    memories: List[Dict[str, Any]],
    as_of: Any,
    *,
    enabled: bool,
) -> List[Dict[str, Any]]:
    """Demote memories whose valid-time interval excludes *as_of* (opt-in).

    A strict no-op — the list is returned untouched, byte-identical — unless
    *enabled* is true AND *as_of* is a real date/datetime AND at least one memory
    is actually excluded. When it does fire it performs a STABLE partition:
    valid (and interval-less) memories keep their order at the front, excluded
    memories keep their order at the back and are stamped
    ``as_of_excluded=True``. Nothing is deleted (recall-safe soft demotion) and
    no relevance score is altered.
    """
    if not enabled or not memories:
        return memories
    as_of_date = _as_of_date(as_of)
    if as_of_date is None:
        return memories

    included: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for mem in memories:
        if interval_excludes(mem, as_of_date):
            excluded.append(mem)
        else:
            included.append(mem)
    if not excluded:
        # Nothing lapsed/pending as of the query date — leave the list exactly
        # as the reranker ordered it (no marker written on any memory).
        return memories
    for mem in excluded:
        mem[AS_OF_EXCLUDED_KEY] = True
    memories[:] = included + excluded
    logger.debug(
        "as-of recall (%s): demoted %d/%d memories with lapsed/pending valid-time",
        as_of_date.isoformat(), len(excluded), len(memories),
    )
    return memories
