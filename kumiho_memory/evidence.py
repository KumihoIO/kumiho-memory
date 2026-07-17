"""Evidence-level schema convention for memories (metadata + mirrored tags).

Memories carry an *evidence grade* describing how trustworthy they are.
The grade lives in two places:

- revision metadata key ``evidence_level`` (with optional ``source`` and
  ``confidence`` companions) — the canonical value. ``evidence_level`` is a
  PROVENANCE grade; the self-reported ``confidence``/``certainty`` companions
  are a different axis — see :mod:`kumiho_memory.trust_vocab` for the one
  defined mapping between them and its limits.
- mirrored graph tag ``evidence:<level>`` — tags get server-side
  time-range history and can be filtered/resolved, while metadata cannot

When the two diverge (tags are applied per-tag with best-effort
semantics), the metadata value wins — see :func:`parse_evidence`.

Levels, from most to least trustworthy::

    official       explicit operator/ingest flag, never LLM-inferred;
                   pair with the ``published`` tag for deprecation
                   protection
    corroborated   >= N independent agreeing sources, none contradicting
    single_source  identified source, no corroboration
    unverified     everything else

Grades are only stamped when a caller provides one — existing memories
and callers that never mention evidence keep byte-identical behavior.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

OFFICIAL = "official"
CORROBORATED = "corroborated"
SINGLE_SOURCE = "single_source"
UNVERIFIED = "unverified"

#: All valid evidence levels, ordered most → least trustworthy.
EVIDENCE_LEVELS = (OFFICIAL, CORROBORATED, SINGLE_SOURCE, UNVERIFIED)

#: Grade assumed for memories that carry no evidence marking at all.
DEFAULT_EVIDENCE_LEVEL = UNVERIFIED

#: Prefix of the mirrored graph tag (``evidence:official`` etc.).
EVIDENCE_TAG_PREFIX = "evidence:"


def evidence_tag(level: str) -> str:
    """Return the mirrored graph tag for an evidence *level*.

    Parameters
    ----------
    level:
        One of :data:`EVIDENCE_LEVELS`.

    Raises
    ------
    ValueError
        If *level* is not a known evidence level.
    """
    if level not in EVIDENCE_LEVELS:
        raise ValueError(
            f"Unknown evidence level {level!r} — expected one of {EVIDENCE_LEVELS}"
        )
    return f"{EVIDENCE_TAG_PREFIX}{level}"


def parse_evidence(
    meta: Optional[Mapping[str, object]],
    tags: Optional[Iterable[str]] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    """Resolve the evidence level of a memory from metadata and tags.

    The metadata key ``evidence_level`` takes precedence; the mirrored
    ``evidence:<level>`` tag is the fallback (tag application is
    best-effort per-tag, so metadata is the more reliable carrier).
    Unknown values are ignored rather than raised — stored data must
    never make recall fail.

    Parameters
    ----------
    meta:
        Revision metadata mapping (may be ``None``).
    tags:
        Revision tags (may be ``None``).
    default:
        Returned when neither carrier holds a valid level.
    """
    level = str((meta or {}).get("evidence_level", "") or "")
    if level in EVIDENCE_LEVELS:
        return level
    for tag in tags or ():
        if isinstance(tag, str) and tag.startswith(EVIDENCE_TAG_PREFIX):
            candidate = tag[len(EVIDENCE_TAG_PREFIX):]
            if candidate in EVIDENCE_LEVELS:
                return candidate
    return default
