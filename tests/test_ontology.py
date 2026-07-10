"""Tests for schema-driven conversation decomposition into a typed graph."""

import sys
import types

import grpc
import pytest

from kumiho_memory.ontology import (
    OntologySchema,
    _mentions,
    _sync_decompose,
    _word_tokens,
)
from kumiho_memory.relations import _jaccard, _tokens, link_supersedes
from kumiho_memory.summarization import _decision_item_schema, build_summary_schema_mode


class _RpcErr(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class _Rev:
    def __init__(self, kref, metadata=None):
        self.kref = types.SimpleNamespace(uri=kref)
        self.metadata = metadata or {}
        self.edges = []  # (target_uri, edge_type, metadata)

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((target.kref.uri, edge_type, metadata or {}))


class _Item:
    def __init__(self, kref):
        self.kref = types.SimpleNamespace(uri=kref)
        self.revisions = []

    def get_latest_revision(self):
        return self.revisions[-1] if self.revisions else None

    def create_revision(self, metadata=None):
        rev = _Rev(f"{self.kref.uri}?r={len(self.revisions) + 1}", metadata)
        self.revisions.append(rev)
        return rev


class _Project:
    def __init__(self, name="Proj"):
        self.name = name
        self.spaces = set()
        self.items = {}  # (space_path, name, kind) -> _Item

    def create_space(self, name, parent_path=None):
        if name in self.spaces:
            raise _RpcErr(grpc.StatusCode.ALREADY_EXISTS)
        self.spaces.add(name)

    def create_item(self, name, kind, parent_path=None, metadata=None):
        key = (parent_path, name, kind)
        if key in self.items:
            raise _RpcErr(grpc.StatusCode.ALREADY_EXISTS)
        item = _Item(f"kref:/{parent_path}/{name}.{kind}")
        self.items[key] = item
        return item

    def get_item(self, name, kind, parent_path=None):
        return self.items[(parent_path, name, kind)]


def _install_kumiho(monkeypatch, project, conv_rev, search_results=None):
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: project
    fake.get_revision = lambda kref: conv_rev
    fake.search = lambda *a, **k: (search_results or [])
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    # relations.py imports kumiho lazily too.
    return fake


def _all_edges(project, conv_rev):
    """Collect (source_uri, edge_type, target_uri) across every created node."""
    edges = []
    for src in [conv_rev] + [it.revisions[-1] for it in project.items.values() if it.revisions]:
        for target_uri, etype, _ in src.edges:
            edges.append((src.kref.uri, etype, target_uri))
    return edges


SUMMARY = {
    "summary": "we discussed the migration",
    "classification": {"entities": ["Redis", "Anthropic"]},
    "knowledge": {
        "facts": [
            {"claim": "Redis 7 supports streams", "certainty": "high"},
            {"claim": "Upstash bills per request", "certainty": "medium"},
        ],
        "decisions": [
            {"decision": "Adopt Redis for the event bus", "reason": "streams + cost",
             "based_on": [0, 1]},
        ],
        "actions": [{"task": "migrate the bus off Upstash", "status": "done"}],
        "open_questions": ["do we need cross-region replication?"],
    },
    "events": [
        {"event": "Cutover to Redis", "when": "last week",
         "participants": ["Anthropic"], "consequence": "lower latency"},
    ],
}


def test_decompose_materializes_typed_nodes(monkeypatch):
    project = _Project("Proj")
    conv = _Rev("kref://Proj/infra/mem.conversation?r=1")
    _install_kumiho(monkeypatch, project, conv)

    stats = _sync_decompose(conv.kref.uri, SUMMARY, "Proj", OntologySchema())

    assert stats["entities"] == 2
    assert stats["facts"] == 2
    assert stats["decisions"] == 1
    assert stats["actions"] == 1
    assert stats["questions"] == 1
    assert stats["events"] == 1
    # Nodes land in kind-specific spaces.
    kinds = {kind for (_sp, _n, kind) in project.items}
    assert kinds == {"entity", "fact", "decision", "action", "event", "question"}
    # Typed nodes carry title/summary so the reader surfaces content, not stubs.
    for (sp, name, kind), item in project.items.items():
        if kind in ("fact", "decision", "event", "action", "question"):
            meta = item.revisions[-1].metadata
            assert meta.get("title") and meta.get("summary")


def test_decompose_wires_structural_and_relational_edges(monkeypatch):
    project = _Project("Proj")
    conv = _Rev("kref://Proj/infra/mem.conversation?r=1")
    _install_kumiho(monkeypatch, project, conv)

    _sync_decompose(conv.kref.uri, SUMMARY, "Proj", OntologySchema())
    edges = _all_edges(project, conv)
    etypes = [e[1] for e in edges]

    # Provenance: every typed node -> conversation.
    assert etypes.count("DERIVED_FROM") == 2 + 1 + 1 + 1 + 1  # facts+dec+action+event+question
    # ABOUT: a fact mentioning "Redis" links to the Redis entity; conversation
    # links to its entities too.
    assert "ABOUT" in etypes
    # INVOLVES: the event's participant (Anthropic) -> entity.
    assert any(et == "INVOLVES" for (_s, et, _t) in edges)
    # DEPENDS_ON: the decision based_on facts [0,1] -> two fact nodes.
    depends = [e for e in edges if e[1] == "DEPENDS_ON"]
    assert len(depends) == 2
    # The DEPENDS_ON targets are the fact nodes.
    fact_krefs = {it.kref.uri for (sp, n, k), it in project.items.items() if k == "fact"}
    assert all(t.split("?r=")[0] in {fk.replace("kref:/", "kref://") for fk in fact_krefs}
               or ".fact" in t for (_s, _e, t) in depends)


def test_supersedes_uses_token_overlap_not_score(monkeypatch):
    project = _Project("Proj")
    conv = _Rev("kref://Proj/x/mem.conversation?r=1")

    # A prior decision node about the same subject, returned by search.
    prior = _Item("kref:/Proj/decisions/use-upstash.decision")
    prior.create_revision({"decision": "Use Upstash for the event bus streams"})
    result = types.SimpleNamespace(item=prior, score=999.0)  # score deliberately huge
    _install_kumiho(monkeypatch, project, conv, search_results=[result])

    new_anchor = _Rev("kref:/Proj/decisions/use-redis.decision?r=1")
    n = link_supersedes(
        object.__new__(_MaterializerStub).__class__ if False else _MaterializerStub(),
        "decision", "decisions", "use-redis", new_anchor,
        "Use Redis for the event bus streams", "Proj",
    )
    # High token overlap ("use ... the event bus streams") -> SUPERSEDES.
    assert n == 1
    assert new_anchor.edges and new_anchor.edges[0][1] == "SUPERSEDES"


def test_supersedes_skips_low_overlap(monkeypatch):
    project = _Project("Proj")
    conv = _Rev("kref://Proj/x/mem.conversation?r=1")
    prior = _Item("kref:/Proj/decisions/hiring.decision")
    prior.create_revision({"decision": "Open two backend roles in Q3"})
    result = types.SimpleNamespace(item=prior, score=999.0)
    _install_kumiho(monkeypatch, project, conv, search_results=[result])

    new_anchor = _Rev("kref:/Proj/decisions/use-redis.decision?r=1")
    n = link_supersedes(
        _MaterializerStub(), "decision", "decisions", "use-redis", new_anchor,
        "Use Redis for the event bus streams", "Proj",
    )
    assert n == 0
    assert not new_anchor.edges


def test_jaccard_is_corpus_independent():
    a = _tokens("Use Redis for the event bus streams")
    b = _tokens("Use Upstash for the event bus streams")
    assert _jaccard(a, b) > 0.6  # same subject
    assert _jaccard(_tokens("hiring plan"), _tokens("redis migration")) == 0.0


def test_mentions_is_token_match_not_substring():
    # Short / ambiguous names must not match inside longer words.
    assert not _mentions(_word_tokens("AI"), _word_tokens("the plan was finalized"))
    assert not _mentions(_word_tokens("IT"), _word_tokens("the credit was applied"))
    assert not _mentions(_word_tokens("US"), _word_tokens("we discussed the census"))
    # Korean: "김" must not match the fused token "김치" (Hangul is space-free).
    assert not _mentions(_word_tokens("김"), _word_tokens("김치를 먹었다"))
    # Real mentions match on token boundaries.
    assert _mentions(_word_tokens("Redis"), _word_tokens("we adopted Redis today"))
    assert _mentions(_word_tokens("Redis Cluster"), _word_tokens("use Redis Cluster now"))
    # Multi-word name must be contiguous.
    assert not _mentions(_word_tokens("Redis Cluster"), _word_tokens("Redis runs the cluster"))


def test_summary_schema_is_identical_in_both_ontology_modes(monkeypatch):
    # The summarizer schema must be byte-identical whether the ontology is on
    # or off — an ontology-gated `based_on` was tried and MEASURED to shift
    # every ontology-on consolidation's structured output (weaker base
    # recall). DEPENDS_ON is derived post-hoc by token overlap instead.
    monkeypatch.setenv("KUMIHO_MEMORY_ONTOLOGY", "0")   # explicit opt-out
    dec_off = _decision_item_schema()
    schema_off = repr(build_summary_schema_mode())

    monkeypatch.setenv("KUMIHO_MEMORY_ONTOLOGY", "1")
    dec_on = _decision_item_schema()
    schema_on = repr(build_summary_schema_mode())

    assert dec_off == dec_on
    assert schema_off == schema_on                      # byte-identical
    assert "based_on" not in schema_on                  # in EITHER mode
    assert sorted(dec_on["properties"]) == ["decision", "reason"]


def test_depends_on_derived_by_overlap_without_based_on():
    # With based_on gone from the schema, the grounding fact is recovered by
    # token overlap against the same consolidation's facts (top-1, >=0.4).
    from kumiho_memory.relations import link_depends_on_by_overlap

    edges = []

    class M:
        def edge(self, s, t, et, md=None):
            edges.append((s, t, et, md))
            return True

    facts = [
        ("F_A", "a", "Acme pet insurance covers Max's vet bills"),
        ("F_B", "b", "Caroline enjoys gardening on weekends"),
    ]
    n = link_depends_on_by_overlap(
        M(), "D",
        "Keep Max on Acme pet insurance (reason: vet bills expensive)", facts,
    )
    assert n == 1
    assert edges[0][1] == "F_A"                 # the overlapping fact wins
    assert edges[0][2] == "DEPENDS_ON"

    edges.clear()                               # nothing above threshold
    n = link_depends_on_by_overlap(M(), "D", "Ship the new landing page", facts)
    assert n == 0 and edges == []


def test_about_edge_for_participant_only_entity(monkeypatch):
    # A fact naming an entity that appears ONLY as an event participant (absent
    # from classification.entities) must still get its ABOUT edge — entities are
    # materialized before fact/decision linking regardless of source order.
    project = _Project("P")
    conv = _Rev("kref://P/x/mem.conversation?r=1")
    _install_kumiho(monkeypatch, project, conv)
    summary = {
        "classification": {"entities": []},
        "knowledge": {
            "facts": [{"claim": "Anthropic approved the migration", "certainty": "high"}],
            "decisions": [], "actions": [], "open_questions": [],
        },
        "events": [{"event": "Kickoff", "participants": ["Anthropic"]}],
    }
    _sync_decompose(conv.kref.uri, summary, "P", OntologySchema())
    edges = _all_edges(project, conv)
    assert any(et == "ABOUT" and "anthropic.entity" in t and ".fact" in s
               for (s, et, t) in edges), edges


class _MaterializerStub:
    """Minimal stand-in exposing the `edge` method link_supersedes calls."""

    def edge(self, source_rev, target_rev, edge_type, metadata=None):
        source_rev.create_edge(target_rev, edge_type, metadata)
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
