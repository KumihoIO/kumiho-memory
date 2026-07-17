# -*- coding: utf-8 -*-
"""Canonical relation-predicate registry (predicate_registry).

Folding is pure and offline (no server, no fake kumiho): a raw predicate
resolves to a canonical edge type or the RELATES_TO fallback. These tests pin
the vocabulary contract the ontology write path commits to.
"""
from kumiho_memory.predicate_registry import (
    RELATES_TO,
    canonical_types,
    registry_as_dict,
    resolve_predicate,
)

_REQUIRED = {
    "DEPENDS_ON", "USES", "IMPLEMENTS", "PART_OF", "SUPERSEDES",
    "SUPPORTS", "CAUSES", "CONTRADICTS", "RELATES_TO",
}


def test_required_canonicals_present():
    assert _REQUIRED.issubset(set(canonical_types()))
    assert 8 <= len(canonical_types()) <= 12


def test_canonical_passthrough_not_folded():
    etype, res = resolve_predicate("uses")
    assert etype == "USES"
    assert res.folded is False and res.fallback is False
    # RELATES_TO reached as a real canonical is passthrough, not a fallback.
    etype, res = resolve_predicate("relates_to")
    assert etype == RELATES_TO
    assert res.folded is False and res.fallback is False


def test_synonym_folding():
    assert resolve_predicate("utilizes")[0] == "USES"
    assert resolve_predicate("relies_on")[0] == "DEPENDS_ON"
    assert resolve_predicate("replaces")[0] == "SUPERSEDES"
    assert resolve_predicate("conflicts_with")[0] == "CONTRADICTS"
    _, res = resolve_predicate("utilizes")
    assert res.folded is True and res.fallback is False
    assert res.normalized == "UTILIZES"


def test_case_spacing_separator_variants_fold_identically():
    variants = ["relies on", "Relies-On", "RELIES_ON", "  relies_on  "]
    results = {resolve_predicate(v)[0] for v in variants}
    assert results == {"DEPENDS_ON"}


def test_unknown_predicate_falls_back_to_relates_to():
    etype, res = resolve_predicate("frobnicates")
    assert etype == RELATES_TO
    assert res.fallback is True and res.folded is False
    assert res.normalized == "FROBNICATES"  # token preserved for lossless metadata


def test_cjk_and_unnormalizable_fall_back_not_dropped():
    for pred in ("관련", "依存", "!!!", "   ", ""):
        etype, res = resolve_predicate(pred)
        assert etype == RELATES_TO      # never None -> the caller never drops it
        assert res.fallback is True
        assert res.normalized == ""     # normalizes to nothing; verbatim kept by caller


def test_registry_self_consistency_no_synonym_maps_to_two_canonicals():
    # Import already builds (and validates) the reverse index, but re-derive it
    # here so a future edit that introduces an ambiguous synonym fails loudly.
    from kumiho_memory.predicate_registry import _normalize

    seen = {}
    canonicals = set(canonical_types())
    for canonical, synonyms in registry_as_dict().items():
        for synonym in synonyms:
            tok = _normalize(synonym)
            assert tok, f"{synonym!r} normalizes to nothing"
            assert tok not in canonicals, f"{synonym!r} shadows canonical {tok}"
            assert tok not in seen or seen[tok] == canonical, (
                f"{tok} maps to both {seen.get(tok)} and {canonical}")
            seen[tok] = canonical


def test_registry_as_dict_is_a_copy():
    a = registry_as_dict()
    a["USES"] = ()          # mutate the copy
    a["INJECTED"] = ("x",)
    b = registry_as_dict()
    assert b["USES"]        # module constant untouched
    assert "INJECTED" not in b
