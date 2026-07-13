# -*- coding: utf-8 -*-
"""Keyless agent-driven ontology decomposition (ontology._sync_decompose_agent).

Uses in-memory graph fakes (no server) to assert the deterministic write layer:
typed nodes, ABOUT/DERIVED_FROM/relation edges, idempotency, and structural
validation. The live end-to-end proof is scripts/dogfood_ontology_agent.py.
"""
import asyncio
import time

import kumiho
from kumiho_memory import ontology
from kumiho_memory.ontology import (
    _sync_decompose_agent, _predicate_edge_type, OntologySchema,
    decompose_and_link_agent,
)


class _FakeRev:
    def __init__(self, uri):
        self.kref = type("K", (), {"uri": uri})()
        self.edges = []  # (edge_type, target_uri)

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((edge_type, target.kref.uri))

    def get_edges(self, edge_type_filter=None, direction=0):
        out = []
        for et, turi in self.edges:
            if edge_type_filter and et != edge_type_filter:
                continue
            out.append(type("E", (), {"target_kref": type("K", (), {"uri": turi})()})())
        return out


class _FakeItem:
    def __init__(self, uri):
        self._rev = _FakeRev(uri)
        self._created = False

    def get_latest_revision(self):
        return self._rev if self._created else None

    def create_revision(self, metadata=None):
        self._created = True
        return self._rev


class _FakeProject:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def create_space(self, sp):
        pass

    def create_item(self, slug, kind, parent_path=None):  # get-or-create
        key = (parent_path, slug)
        if key not in self._items:
            self._items[key] = _FakeItem(f"kref:/{parent_path}/{slug}.{kind}?r=1")
        return self._items[key]

    def get_item(self, slug, kind, parent_path=None):
        return self._items[(parent_path, slug)]


def _patch(monkeypatch, conv_rev):
    proj = _FakeProject("proj")
    monkeypatch.setattr(kumiho, "get_project", lambda name: proj)
    monkeypatch.setattr(kumiho, "get_revision", lambda kref: conv_rev)
    return proj


_DECOMP = {
    "entities": [
        {"name": "Decision Memory", "type": "system"},
        {"name": "config_from_env", "type": "convention", "aliases": ["config helper"]},
        {"name": "KUMIHO_SERVER_ENDPOINT"},
    ],
    "facts": [
        {"statement": "Toggles use config_from_env", "about": ["config_from_env", "Decision Memory"]},
        {"statement": "Endpoint read from KUMIHO_SERVER_ENDPOINT", "about": ["KUMIHO_SERVER_ENDPOINT"]},
    ],
    "relations": [{"subject": "Decision Memory", "predicate": "uses", "object": "config_from_env"}],
}


def test_predicate_normalization():
    assert _predicate_edge_type("depends on") == "DEPENDS_ON"
    assert _predicate_edge_type("  bad!! ") == "BAD"
    assert _predicate_edge_type("123") is None      # must start with a letter
    assert _predicate_edge_type("") is None


def test_decompose_writes_typed_nodes_and_edges(monkeypatch):
    conv = _FakeRev("kref:/proj/conversations/c.conversation?r=1")
    proj = _patch(monkeypatch, conv)
    stats = _sync_decompose_agent("kref:/proj/conversations/c.conversation?r=1", _DECOMP, "proj", OntologySchema())

    assert stats["entities"] == 3
    assert stats["facts"] == 2
    assert stats["relations"] == 1
    # conversation -> entity ABOUT for all three entities
    assert sum(1 for et, _ in conv.edges if et == "ABOUT") == 3
    # a fact carries DERIVED_FROM -> conversation and ABOUT -> entity
    fact_item = proj._items[("/proj/facts", next(s for (sp, s) in proj._items if sp == "/proj/facts"))]
    kinds = {et for et, _ in fact_item._rev.edges}
    assert "DERIVED_FROM" in kinds and "ABOUT" in kinds
    # the relation became a typed USES edge entity->entity
    dm_key = next(k for k in proj._items if k[0] == "/proj/entities" and "decision-memory" in k[1])
    assert any(et == "USES" for et, _ in proj._items[dm_key]._rev.edges)


def test_idempotent_rerun(monkeypatch):
    conv = _FakeRev("kref:/proj/conversations/c.conversation?r=1")
    _patch(monkeypatch, conv)
    kref = "kref:/proj/conversations/c.conversation?r=1"
    _sync_decompose_agent(kref, _DECOMP, "proj", OntologySchema())
    second = _sync_decompose_agent(kref, _DECOMP, "proj", OntologySchema())
    # nodes still counted (get-or-create) but NO new edges on the re-run
    assert second["edges"] == 0
    assert second["relations"] == 0


def test_structural_validation(monkeypatch):
    conv = _FakeRev("kref:/proj/conversations/c.conversation?r=1")
    _patch(monkeypatch, conv)
    decomp = {
        "entities": [{"name": "A"}, {"name": ""}],           # empty entity dropped
        "facts": [{"statement": ""}, {"statement": "real fact", "about": ["A"]}],  # empty fact dropped
        "relations": [
            {"subject": "A", "predicate": "links", "object": "Ghost"},  # unresolved object -> dropped
            {"subject": "A", "predicate": "123", "object": "A"},        # bad predicate + self -> dropped
        ],
    }
    stats = _sync_decompose_agent("kref:/proj/conversations/c.conversation?r=1", decomp, "proj", OntologySchema())
    assert stats["entities"] == 1
    assert stats["facts"] == 1
    assert stats["relations"] == 0


def test_slow_write_reports_status_not_bare_empty(monkeypatch):
    """A write that outruns the deadline must not report a bare ``{}``.

    The daemon worker keeps writing after the timeout, so an empty ``{}`` reads
    as a no-op when the graph is actually being populated. The wrapper returns a
    status marker instead. Regression for the 25s-bound-vs-cloud-latency bug.
    """
    def _slow(*_a, **_k):
        time.sleep(0.3)
        return {"entities": 1, "facts": 0, "relations": 0, "edges": 1}

    monkeypatch.setattr(ontology, "_sync_decompose_agent", _slow)
    res = asyncio.run(decompose_and_link_agent(
        "kref:/proj/conversations/c.conversation?r=1",
        {"entities": [{"name": "a"}]}, project_name="proj", timeout=0.05,
    ))
    assert res != {}
    assert res.get("status") == "in_progress"


def test_fast_write_returns_real_counts(monkeypatch):
    """Within the deadline, the real per-kind counts flow back unchanged."""
    monkeypatch.setattr(
        ontology, "_sync_decompose_agent",
        lambda *_a, **_k: {"entities": 2, "facts": 1, "relations": 1, "edges": 5},
    )
    res = asyncio.run(decompose_and_link_agent(
        "kref:/proj/conversations/c.conversation?r=1",
        {"entities": [{"name": "a"}]}, project_name="proj",
    ))
    assert res == {"entities": 2, "facts": 1, "relations": 1, "edges": 5}
