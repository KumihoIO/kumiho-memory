"""Cross-vocabulary trust mapping (ontology G7).

Three trust vocabularies coexist with no defined correspondence:

- ``certainty`` (``low/medium/high``) — on facts (``summarization.py``,
  ``ontology.py``); a SELF-REPORTED strength claim by the writer.
- ``confidence`` (``low/medium/high``) — on code decisions
  (``code_capture.py``); also a SELF-REPORTED strength claim.
- ``evidence_level`` (``official/corroborated/single_source/unverified``,
  ``evidence.py``) — a PROVENANCE grade: how the claim is sourced and
  corroborated, not what the writer feels about it.

The two axes are NOT interchangeable. This module states the one mapping
between them and its limit:

- ``certainty``/``confidence`` do NOT lift ``evidence_level``. A
  "high certainty" fact is still ``unverified`` provenance unless it is
  separately graded — the strength band computed here is a *different*
  metadata field, never fed back into the provenance grade.
- All three normalize onto one coarse ordinal :class:`StrengthBand`
  (``LOW < MEDIUM < HIGH``) that is usable ONLY as a tie-breaker between
  otherwise-equal candidates. The raw 4-level ``evidence_level`` remains
  the authoritative provenance signal; the band collapses ``official`` and
  ``corroborated`` together and does not distinguish them.

Pure and offline: no server, no LLM, no stored-data migration. Nothing
here changes how recall or grading already reads these fields — this is
definition + a helper the ontology spec (:mod:`kumiho_memory.ontology_spec`)
embeds so agents can commit to a shared reading.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional

from .evidence import EVIDENCE_LEVELS

#: Vocabulary names, as they appear as metadata keys in the graph.
CERTAINTY = "certainty"
CONFIDENCE = "confidence"
EVIDENCE_LEVEL = "evidence_level"


class StrengthBand(IntEnum):
    """Coarse ordinal trust band shared by all three vocabularies.

    Ordinal (``LOW < MEDIUM < HIGH``) so it can break ties; not a provenance
    grade and not a probability.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3


#: self-reported value -> band. ``certainty`` and ``confidence`` share this
#: (identical vocabularies): the writer's asserted strength.
_SELF_REPORTED_TO_BAND: Dict[str, StrengthBand] = {
    "low": StrengthBand.LOW,
    "medium": StrengthBand.MEDIUM,
    "high": StrengthBand.HIGH,
}

#: provenance grade -> band, by amount of external support. ``official`` and
#: ``corroborated`` both land in ``HIGH`` — the band is coarse and does not
#: separate operator-blessed from multiply-corroborated; use the raw
#: ``evidence_level`` for that distinction.
_PROVENANCE_TO_BAND: Dict[str, StrengthBand] = {
    "unverified": StrengthBand.LOW,
    "single_source": StrengthBand.MEDIUM,
    "corroborated": StrengthBand.HIGH,
    "official": StrengthBand.HIGH,
}

_VALUE_TO_BAND: Dict[str, Dict[str, StrengthBand]] = {
    CERTAINTY: _SELF_REPORTED_TO_BAND,
    CONFIDENCE: _SELF_REPORTED_TO_BAND,
    EVIDENCE_LEVEL: _PROVENANCE_TO_BAND,
}


def normalize_trust(vocabulary: str, value: str) -> Optional[StrengthBand]:
    """Normalize a ``(vocabulary, value)`` pair to a :class:`StrengthBand`.

    Accepts any of the three vocabularies (``certainty``, ``confidence``,
    ``evidence_level``). Returns ``None`` for an unknown vocabulary or an
    unrecognized value — normalization must never raise on stored data.
    """
    table = _VALUE_TO_BAND.get((vocabulary or "").strip().lower())
    if table is None:
        return None
    return table.get((value or "").strip().lower())


def mapping_as_dict() -> Dict[str, object]:
    """The mapping as plain data for the ontology spec Item to embed."""
    return {
        "axes": {
            "self_reported": {
                "vocabularies": [CERTAINTY, CONFIDENCE],
                "values": ["low", "medium", "high"],
                "meaning": "strength claim asserted by the writer "
                           "(fact certainty; code-decision confidence)",
            },
            "provenance": {
                "vocabularies": [EVIDENCE_LEVEL],
                "values": list(EVIDENCE_LEVELS),
                "meaning": "how the claim is sourced and corroborated "
                           "(graded, never self-reported)",
            },
        },
        "bands": {b.name.lower(): int(b) for b in StrengthBand},
        "value_to_band": {
            vocab: {value: band.name.lower() for value, band in table.items()}
            for vocab, table in _VALUE_TO_BAND.items()
        },
        "limits": (
            "Self-reported certainty/confidence do NOT lift evidence_level: a "
            "high-certainty fact is still `unverified` provenance unless "
            "separately graded. The band is a coarse tie-breaker only; the raw "
            "4-level evidence_level remains the authoritative provenance signal "
            "and is not distinguished for official vs corroborated by the band."
        ),
    }
