# -*- coding: utf-8 -*-
"""Tests for the ontology spec Item (G2) and the trust-vocabulary mapping (G7).

The spec build is pure (no server). Seeding is exercised against a fake
``kumiho`` SDK installed via ``monkeypatch.setitem`` (never ``sys.modules.pop``)
so the Item/Revision/tag machinery is driven end-to-end.
"""

import json
import sys
import types

import pytest

from kumiho_memory.ontology import OntologySchema
from kumiho_memory.ontology_spec import (
    SPEC_ITEM,
    SPEC_KIND,
    SPEC_SPACE,
    SPEC_TAG,
    build_spec,
    seed_ontology_spec,
)
from kumiho_memory.predicate_registry import (
    RELATES_TO,
    canonical_types,
    registry_as_dict,
)
from kumiho_memory.trust_vocab import (
    CERTAINTY,
    CONFIDENCE,
    EVIDENCE_LEVEL,
    StrengthBand,
    mapping_as_dict,
    normalize_trust,
)

_ITEM_KEY = (f"/CognitiveMemory/{SPEC_SPACE}", SPEC_ITEM, SPEC_KIND)


# ---------------------------------------------------------------------------
# Fake kumiho SDK (Item/Revision/tag machinery)
# ---------------------------------------------------------------------------


class FakeRpcError(Exception):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class FakeStatusCode:
    ALREADY_EXISTS = "ALREADY_EXISTS"
    NOT_FOUND = "NOT_FOUND"


class FakeKref:
    def __init__(self, uri):
        self.uri = uri

    def __str__(self):
        return self.uri


class FakeRevision:
    def __init__(self, item, number, metadata):
        self.item = item
        self.number = number
        self.metadata = dict(metadata or {})
        self.kref = FakeKref(f"{item.kref.uri}?r={number}")
        self.tags = []

    def tag(self, tag_name):
        # A tag resolves to one revision per item: move it off any prior holder.
        for rev in self.item.revisions:
            if rev is not self and tag_name in rev.tags:
                rev.tags.remove(tag_name)
        if tag_name not in self.tags:
            self.tags.append(tag_name)
        self.item.tag_index[tag_name] = self


class FakeItem:
    def __init__(self, kref_uri):
        self.kref = FakeKref(kref_uri)
        self.revisions = []
        self.tag_index = {}

    def create_revision(self, metadata=None):
        rev = FakeRevision(self, len(self.revisions) + 1, metadata)
        self.revisions.append(rev)
        return rev

    def get_revision_by_tag(self, tag):
        return self.tag_index.get(tag)


class FakeProject:
    def __init__(self, name="CognitiveMemory"):
        self.name = name
        self.spaces = set()
        self.items = {}

    def create_space(self, name, parent_path=None):
        if name in self.spaces:
            raise FakeRpcError(FakeStatusCode.ALREADY_EXISTS)
        self.spaces.add(name)

    def create_item(self, item_name, kind, parent_path=None, metadata=None):
        key = (parent_path, item_name, kind)
        if key in self.items:
            raise FakeRpcError(FakeStatusCode.ALREADY_EXISTS)
        base = (parent_path or f"/{self.name}").strip("/")
        item = FakeItem(f"kref://{base}/{item_name}.{kind}")
        self.items[key] = item
        return item

    def get_item(self, item_name, kind, parent_path=None):
        return self.items[(parent_path, item_name, kind)]


def _install_fake_kumiho(monkeypatch, project):
    fake_kumiho = types.ModuleType("kumiho")
    fake_kumiho.get_project = lambda name: project
    fake_grpc = types.ModuleType("grpc")
    fake_grpc.RpcError = FakeRpcError
    fake_grpc.StatusCode = FakeStatusCode
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)
    monkeypatch.setitem(sys.modules, "grpc", fake_grpc)


# ---------------------------------------------------------------------------
# build_spec — round-trips the live registries
# ---------------------------------------------------------------------------


def test_build_spec_embeds_and_round_trips_registry():
    spec = build_spec()
    assert spec["spec_version"] == OntologySchema().version

    reg = spec["relation_registry"]
    assert reg["canonical_types"] == list(canonical_types())
    assert reg["synonyms"] == {c: list(s) for c, s in registry_as_dict().items()}
    assert reg["fallback"]["type"] == RELATES_TO

    # Survives JSON serialization unchanged (metadata carries it as JSON).
    restored = json.loads(json.dumps(spec, ensure_ascii=False))
    assert restored["relation_registry"]["synonyms"] == reg["synonyms"]
    assert restored["relation_registry"]["canonical_types"] == reg["canonical_types"]


def test_build_spec_covers_node_kinds_and_edges():
    spec = build_spec()
    assert set(spec["node_kinds"]) == {
        "entity", "fact", "decision", "event", "action", "question",
    }
    # Entity identity is slug-of-name; the others slug-of-text.
    assert "slug-of-name" in spec["node_kinds"]["entity"]["identity_rule"]
    assert "slug-of-text" in spec["node_kinds"]["fact"]["identity_rule"]
    assert spec["node_kinds"]["fact"]["space"] == OntologySchema().facts_space

    edges = spec["edge_types"]
    for name in ("DERIVED_FROM", "ABOUT", "INVOLVES", "DEPENDS_ON",
                 "SUPERSEDES", "CONTRADICTS"):
        assert name in edges
    # SUPERSEDES documents the dual basis convention (agent-declared preferred,
    # lexical fallback with its threshold) + newest-wins direction.
    assert "agent" in edges["SUPERSEDES"]["basis"]
    assert "lexical-overlap" in edges["SUPERSEDES"]["basis"]
    assert "0.6" in edges["SUPERSEDES"]["basis"]
    assert "newest-wins" in edges["SUPERSEDES"]["direction"]
    # CONTRADICTS documents both dispute bases and the predicate-registry
    # exclusion (entity relation edges are not disputes).
    assert "agent" in edges["CONTRADICTS"]["basis"]
    assert "evidence-assessor" in edges["CONTRADICTS"]["basis"]
    assert "predicate" in edges["CONTRADICTS"]["basis"]
    # DEPENDS_ON documents the grounding-staleness ripple convention.
    assert "grounding_stale" in edges["DEPENDS_ON"]["grounding_staleness"]
    assert "grounding:stale" in edges["DEPENDS_ON"]["grounding_staleness"]


def test_build_spec_embeds_trust_mapping():
    spec = build_spec()
    assert spec["trust_vocabulary"] == mapping_as_dict()


def test_default_schema_version_is_v2():
    # Phase 2 (SUPERSEDES dual-basis, CONTRADICTS, grounding staleness) bumped
    # the contract: re-seeding must mint a new tagged revision.
    assert OntologySchema().version == "kumiho.agent_memory.ontology.v2"


def test_build_spec_honors_schema_version():
    spec = build_spec(OntologySchema(version="kumiho.agent_memory.ontology.v3"))
    assert spec["spec_version"] == "kumiho.agent_memory.ontology.v3"


# ---------------------------------------------------------------------------
# seed_ontology_spec — Item/Revision/tag lifecycle
# ---------------------------------------------------------------------------


def test_fresh_seed_creates_item_and_tagged_revision(monkeypatch):
    project = FakeProject()
    _install_fake_kumiho(monkeypatch, project)

    result = seed_ontology_spec()

    assert result is not None
    assert result.created_item is True
    assert result.created_revision is True
    assert result.version == OntologySchema().version

    item = project.items[_ITEM_KEY]
    assert len(item.revisions) == 1
    tagged = item.get_revision_by_tag(SPEC_TAG)
    assert tagged is item.revisions[0]
    # Content round-trips through the revision metadata.
    content = json.loads(tagged.metadata["content"])
    assert content["relation_registry"]["canonical_types"] == list(canonical_types())
    assert tagged.metadata["spec_version"] == OntologySchema().version


def test_reseed_same_version_is_noop(monkeypatch):
    project = FakeProject()
    _install_fake_kumiho(monkeypatch, project)

    first = seed_ontology_spec()
    second = seed_ontology_spec()

    assert second is not None
    assert second.created_item is False
    assert second.created_revision is False
    assert second.revision_kref == first.revision_kref
    # No second revision was minted.
    assert len(project.items[_ITEM_KEY].revisions) == 1


def test_version_bump_creates_new_revision_and_retags(monkeypatch):
    # A v1-seeded deployment re-seeded at the CURRENT default (v2) exercises
    # the designed bump path: new revision on the same item, tag moves.
    project = FakeProject()
    _install_fake_kumiho(monkeypatch, project)

    seed_ontology_spec(schema=OntologySchema(version="kumiho.agent_memory.ontology.v1"))
    bumped = seed_ontology_spec()

    assert bumped is not None
    assert bumped.created_item is False
    assert bumped.created_revision is True

    item = project.items[_ITEM_KEY]
    assert len(item.revisions) == 2
    tagged = item.get_revision_by_tag(SPEC_TAG)
    assert tagged is item.revisions[1]  # tag moved to the new revision
    assert tagged.metadata["spec_version"] == OntologySchema().version
    assert SPEC_TAG not in item.revisions[0].tags  # moved off the old one


def test_seed_returns_none_when_project_missing(monkeypatch):
    fake_kumiho = types.ModuleType("kumiho")
    fake_kumiho.get_project = lambda name: None
    fake_grpc = types.ModuleType("grpc")
    fake_grpc.RpcError = FakeRpcError
    fake_grpc.StatusCode = FakeStatusCode
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)
    monkeypatch.setitem(sys.modules, "grpc", fake_grpc)

    assert seed_ontology_spec() is None


def test_seed_failures_are_swallowed(monkeypatch):
    def _boom(name):
        raise RuntimeError("backend down")

    fake_kumiho = types.ModuleType("kumiho")
    fake_kumiho.get_project = _boom
    fake_grpc = types.ModuleType("grpc")
    fake_grpc.RpcError = FakeRpcError
    fake_grpc.StatusCode = FakeStatusCode
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)
    monkeypatch.setitem(sys.modules, "grpc", fake_grpc)

    # Best-effort: never raises, returns None.
    assert seed_ontology_spec() is None


# ---------------------------------------------------------------------------
# trust_vocab — the G7 mapping helper
# ---------------------------------------------------------------------------


def test_normalize_trust_covers_all_three_vocabularies():
    assert normalize_trust(CERTAINTY, "low") is StrengthBand.LOW
    assert normalize_trust(CERTAINTY, "medium") is StrengthBand.MEDIUM
    assert normalize_trust(CERTAINTY, "high") is StrengthBand.HIGH

    assert normalize_trust(CONFIDENCE, "low") is StrengthBand.LOW
    assert normalize_trust(CONFIDENCE, "high") is StrengthBand.HIGH

    assert normalize_trust(EVIDENCE_LEVEL, "unverified") is StrengthBand.LOW
    assert normalize_trust(EVIDENCE_LEVEL, "single_source") is StrengthBand.MEDIUM
    assert normalize_trust(EVIDENCE_LEVEL, "corroborated") is StrengthBand.HIGH
    assert normalize_trust(EVIDENCE_LEVEL, "official") is StrengthBand.HIGH


def test_normalize_trust_is_ordinal_for_tiebreaking():
    assert normalize_trust(CERTAINTY, "high") > normalize_trust(CERTAINTY, "low")
    assert int(normalize_trust(EVIDENCE_LEVEL, "official")) == 3
    assert int(normalize_trust(EVIDENCE_LEVEL, "unverified")) == 1


def test_normalize_trust_is_case_and_whitespace_insensitive():
    assert normalize_trust("CERTAINTY", " High ") is StrengthBand.HIGH


def test_normalize_trust_unknown_returns_none():
    assert normalize_trust(CERTAINTY, "bogus") is None
    assert normalize_trust("nonsense", "high") is None
    assert normalize_trust(EVIDENCE_LEVEL, "rumor") is None
    assert normalize_trust(CERTAINTY, "") is None
    assert normalize_trust("", "high") is None


def test_self_reported_does_not_lift_provenance():
    # Different axes: a high self-reported certainty and an unverified
    # provenance normalize independently; the helper never merges them.
    assert normalize_trust(CERTAINTY, "high") is StrengthBand.HIGH
    assert normalize_trust(EVIDENCE_LEVEL, "unverified") is StrengthBand.LOW


def test_mapping_as_dict_agrees_with_helper():
    m = mapping_as_dict()
    assert m["bands"] == {"low": 1, "medium": 2, "high": 3}
    for vocab, table in m["value_to_band"].items():
        for value, band_name in table.items():
            assert normalize_trust(vocab, value).name.lower() == band_name
