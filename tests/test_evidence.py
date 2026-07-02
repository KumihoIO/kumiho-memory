"""Tests for the evidence-level schema helpers."""

import pytest

from kumiho_memory.evidence import (
    CORROBORATED,
    DEFAULT_EVIDENCE_LEVEL,
    EVIDENCE_LEVELS,
    OFFICIAL,
    SINGLE_SOURCE,
    UNVERIFIED,
    evidence_tag,
    parse_evidence,
)


def test_levels_are_complete_and_ordered():
    assert EVIDENCE_LEVELS == (OFFICIAL, CORROBORATED, SINGLE_SOURCE, UNVERIFIED)
    assert DEFAULT_EVIDENCE_LEVEL == UNVERIFIED


def test_evidence_tag_valid_levels():
    assert evidence_tag(OFFICIAL) == "evidence:official"
    assert evidence_tag(CORROBORATED) == "evidence:corroborated"
    assert evidence_tag(SINGLE_SOURCE) == "evidence:single_source"
    assert evidence_tag(UNVERIFIED) == "evidence:unverified"


def test_evidence_tag_rejects_unknown_level():
    with pytest.raises(ValueError, match="Unknown evidence level"):
        evidence_tag("rumor")
    with pytest.raises(ValueError, match="Unknown evidence level"):
        evidence_tag("")


def test_parse_evidence_prefers_metadata():
    """Metadata wins over a diverging mirrored tag."""
    level = parse_evidence(
        {"evidence_level": "official"},
        ["evidence:unverified", "published"],
    )
    assert level == "official"


def test_parse_evidence_falls_back_to_tag():
    assert parse_evidence({}, ["summarized", "evidence:corroborated"]) == "corroborated"
    assert parse_evidence(None, ["evidence:single_source"]) == "single_source"


def test_parse_evidence_ignores_invalid_values():
    """Bad stored data must never raise at recall time."""
    assert parse_evidence({"evidence_level": "rumor"}, ["evidence:bogus"]) is None
    assert parse_evidence({"evidence_level": "rumor"}, ["evidence:official"]) == "official"


def test_parse_evidence_default():
    assert parse_evidence({}, []) is None
    assert parse_evidence(None, None) is None
    assert parse_evidence({}, [], default=UNVERIFIED) == "unverified"
