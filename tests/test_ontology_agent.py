# -*- coding: utf-8 -*-
"""Keyless agent-driven ontology decomposition (ontology._sync_decompose_agent).

Uses in-memory graph fakes (no server) to assert the deterministic write layer:
typed nodes, ABOUT/DERIVED_FROM/relation edges, idempotency, and structural
validation. The live end-to-end proof is scripts/dogfood_ontology_agent.py.
"""
import asyncio
import sys
import time
import types

from kumiho._text import slugify

from kumiho_memory import ontology
from kumiho_memory.ontology import (
    _sync_decompose_agent, _predicate_edge_type, OntologySchema,
    decompose_and_link_agent,
)


class _FakeRev:
    def __init__(self, uri):
        self.kref = type("K", (), {"uri": uri})()
        self.edges = []  # (edge_type, target_uri)
        self.edge_meta = []  # (edge_type, target_uri, metadata) — for metadata asserts

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((edge_type, target.kref.uri))
        self.edge_meta.append((edge_type, target.kref.uri, dict(metadata or {})))

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
    # ``_sync_decompose_agent`` does ``import kumiho`` at CALL time, binding to
    # whatever object sits at ``sys.modules['kumiho']`` then. Resolve it here —
    # also at call time — so the monkeypatch targets the exact object the code
    # under test will use, regardless of how sibling tests swap that entry.
    import kumiho
    proj = _FakeProject("proj")
    monkeypatch.setattr(kumiho, "get_project", lambda name: proj, raising=False)
    monkeypatch.setattr(kumiho, "get_revision", lambda kref: conv_rev, raising=False)
    # The lexical SUPERSEDES fallback calls kumiho.search; stub it to [] so these
    # (no-supersedes) fixtures don't reach a real backend. Belief-path tests below
    # install a full fake kumiho module with their own search/get_revision.
    monkeypatch.setattr(kumiho, "search", lambda *a, **k: [], raising=False)
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


# --- Predicate registry: folding + RELATES_TO fallback (relations not dropped) ---

_DECOMP_PREDICATES = {
    "entities": [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}],
    "facts": [],
    "relations": [
        {"subject": "Alpha", "predicate": "utilizes", "object": "Beta"},    # fold -> USES
        {"subject": "Beta", "predicate": "frobnicates", "object": "Gamma"},  # unknown -> RELATES_TO
        {"subject": "Gamma", "predicate": "관련", "object": "Alpha"},         # CJK -> RELATES_TO
    ],
}


def _entity_meta(proj, slug):
    return proj._items[("/proj/entities", slug)]._rev.edge_meta


def test_predicate_folding_and_fallback_are_lossless(monkeypatch):
    conv = _FakeRev("kref:/proj/conversations/c.conversation?r=1")
    proj = _patch(monkeypatch, conv)
    stats = _sync_decompose_agent(
        "kref:/proj/conversations/c.conversation?r=1", _DECOMP_PREDICATES, "proj", OntologySchema())

    # No relation is dropped for an unrecognized/CJK predicate.
    assert stats["relations"] == 3

    # Synonym folds onto the canonical edge type; normalized token preserved.
    (et, _turi, md), = _entity_meta(proj, "alpha")
    assert et == "USES"
    assert md["predicate"] == "utilizes"
    assert md["predicate_token"] == "UTILIZES"

    # Unknown predicate -> RELATES_TO, verbatim + normalized token both kept.
    (et, _turi, md), = _entity_meta(proj, "beta")
    assert et == "RELATES_TO"
    assert md["predicate"] == "frobnicates"
    assert md["predicate_token"] == "FROBNICATES"

    # CJK predicate normalizes to nothing -> RELATES_TO, verbatim kept, no token.
    (et, _turi, md), = _entity_meta(proj, "gamma")
    assert et == "RELATES_TO"
    assert md["predicate"] == "관련"
    assert "predicate_token" not in md


def test_relates_to_relations_are_idempotent(monkeypatch):
    conv = _FakeRev("kref:/proj/conversations/c.conversation?r=1")
    _patch(monkeypatch, conv)
    kref = "kref:/proj/conversations/c.conversation?r=1"
    _sync_decompose_agent(kref, _DECOMP_PREDICATES, "proj", OntologySchema())
    second = _sync_decompose_agent(kref, _DECOMP_PREDICATES, "proj", OntologySchema())
    # folded + fallback edges dedupe on re-decompose exactly like canonical ones.
    assert second["relations"] == 0
    assert second["edges"] == 0


# --- Agent-declared belief change: SUPERSEDES / CONTRADICTS -----------------
#
# Uses the sys.modules fake-SDK seam (setitem, never pop) so the belief-target
# resolver's kumiho.get_revision / kumiho.search and the fact-space get_item are
# all under test control.

_CONV_URI = "kref:/proj/conversations/c.conversation?r=1"


def _install_module(monkeypatch, conv_rev, proj=None, search_results=None, revs=None):
    proj = proj or _FakeProject("proj")
    known = {conv_rev.kref.uri: conv_rev}
    known.update(revs or {})
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: proj
    fake.get_revision = lambda kref: known[kref]  # KeyError when unknown -> dropped
    fake.search = lambda *a, **k: (search_results or [])
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return proj


def _fact_rev(proj, statement):
    return proj._items[("/proj/facts", slugify(statement, hash_on_truncate=True))]._rev


def _edges_of(rev, edge_type):
    return [(et, turi, md) for (et, turi, md) in rev.edge_meta if et == edge_type]


def test_agent_supersedes_and_contradicts_land_with_basis_agent(monkeypatch):
    conv = _FakeRev(_CONV_URI)
    proj = _install_module(monkeypatch, conv)
    decomp = {
        "facts": [
            {"statement": "Use Upstash for the event bus"},   # prior belief (in-call)
            {"statement": "Use Redis for the event bus"},      # new belief
            {"statement": "Cache TTL is 30 seconds"},
        ],
        "supersedes": [
            {"statement": "Use Redis for the event bus",
             "replaces": "Use Upstash for the event bus", "reason": "cost"},
        ],
        "contradicts": [
            {"statement": "Cache TTL is 30 seconds",
             "conflicts_with": "Use Upstash for the event bus"},
        ],
    }
    stats = _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())

    assert stats["supersedes"] == 1
    assert stats["contradicts"] == 1

    prior_uri = _fact_rev(proj, "Use Upstash for the event bus").kref.uri

    sup = _edges_of(_fact_rev(proj, "Use Redis for the event bus"), "SUPERSEDES")
    assert len(sup) == 1
    assert sup[0][1] == prior_uri
    assert sup[0][2] == {"basis": "agent", "reason": "cost"}

    con = _edges_of(_fact_rev(proj, "Cache TTL is 30 seconds"), "CONTRADICTS")
    assert len(con) == 1
    assert con[0][1] == prior_uri
    assert con[0][2] == {"basis": "agent"}          # no reason carried -> not stored


def test_belief_target_unresolvable_is_dropped(monkeypatch):
    conv = _FakeRev(_CONV_URI)
    proj = _install_module(monkeypatch, conv)
    decomp = {
        "facts": [{"statement": "New fact A"}],
        "supersedes": [{"statement": "New fact A", "replaces": "some unknown prior"}],
    }
    stats = _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())
    assert stats["supersedes"] == 0
    # Dropped, never mislinked: the source fact carries no SUPERSEDES edge at all
    # (agent declared it, so the lexical heuristic is skipped for it too).
    assert _edges_of(_fact_rev(proj, "New fact A"), "SUPERSEDES") == []


def test_belief_target_own_kref_is_dropped(monkeypatch):
    # Hardening: a kref-path (c) resolution of the fact's OWN revision returns a
    # FRESH object (`is` identity fails), so the self-guard must also compare
    # kref uris — a self-SUPERSEDES edge must never land.
    conv = _FakeRev(_CONV_URI)
    self_slug = slugify("New belief X", hash_on_truncate=True)
    self_kref = f"kref://proj/facts/{self_slug}.fact?r=1"
    alias_rev = _FakeRev(self_kref)   # distinct object, same uri as the anchor
    proj = _install_module(monkeypatch, conv, revs={self_kref: alias_rev})
    decomp = {
        "facts": [{"statement": "New belief X"}],
        "supersedes": [{"statement": "New belief X", "replaces": self_kref}],
    }
    stats = _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())
    assert stats["supersedes"] == 0
    assert _edges_of(_fact_rev(proj, "New belief X"), "SUPERSEDES") == []
    assert alias_rev.edge_meta == []


def test_belief_target_kref_path(monkeypatch):
    conv = _FakeRev(_CONV_URI)
    prior_kref = "kref:/proj/facts/prior-belief.fact?r=7"
    prior_rev = _FakeRev(prior_kref)
    proj = _install_module(monkeypatch, conv, revs={prior_kref: prior_rev})
    decomp = {
        "facts": [{"statement": "New belief X"}],
        "supersedes": [{"statement": "New belief X", "replaces": prior_kref}],
    }
    stats = _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())
    assert stats["supersedes"] == 1
    sup = _edges_of(_fact_rev(proj, "New belief X"), "SUPERSEDES")
    assert len(sup) == 1
    assert sup[0][1] == prior_kref
    assert sup[0][2] == {"basis": "agent"}


def test_belief_target_existing_item_path(monkeypatch):
    conv = _FakeRev(_CONV_URI)
    proj = _FakeProject("proj")
    # An existing fact item from a PRIOR call (get, don't create).
    prior_slug = slugify("Prior belief P", hash_on_truncate=True)
    prior_item = proj.create_item(prior_slug, "fact", parent_path="/proj/facts")
    prior_item.create_revision({"claim": "Prior belief P"})   # marks it materialized
    _install_module(monkeypatch, conv, proj=proj)
    decomp = {
        "facts": [{"statement": "New belief Q"}],
        "supersedes": [{"statement": "New belief Q", "replaces": "Prior belief P"}],
    }
    stats = _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())
    assert stats["supersedes"] == 1
    sup = _edges_of(_fact_rev(proj, "New belief Q"), "SUPERSEDES")
    assert len(sup) == 1
    assert sup[0][1] == prior_item._rev.kref.uri
    assert sup[0][2] == {"basis": "agent"}


def test_heuristic_supersedes_skipped_when_agent_declared(monkeypatch):
    import kumiho_memory.relations as relations
    called_slugs = []

    def _spy(m, kind, space, self_slug, anchor, text, project_name, edge_type="SUPERSEDES"):
        called_slugs.append(self_slug)
        return 0

    monkeypatch.setattr(relations, "link_supersedes", _spy)
    conv = _FakeRev(_CONV_URI)
    _install_module(monkeypatch, conv)
    decomp = {
        "facts": [
            {"statement": "Use Redis for the bus"},
            {"statement": "Cache TTL is 60s"},
        ],
        # Target unresolvable, but the fact is still *declared* by the agent, so
        # the lexical fallback must be skipped for it (agent's declaration wins).
        "supersedes": [{"statement": "Use Redis for the bus", "replaces": "unknown prior"}],
    }
    _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())

    assert slugify("Use Redis for the bus", hash_on_truncate=True) not in called_slugs
    assert slugify("Cache TTL is 60s", hash_on_truncate=True) in called_slugs


def test_agent_belief_edges_idempotent_rerun(monkeypatch):
    conv = _FakeRev(_CONV_URI)
    _install_module(monkeypatch, conv)
    decomp = {
        "facts": [
            {"statement": "Use Upstash for the event bus"},
            {"statement": "Use Redis for the event bus"},
        ],
        "supersedes": [
            {"statement": "Use Redis for the event bus",
             "replaces": "Use Upstash for the event bus"},
        ],
        "contradicts": [
            {"statement": "Use Upstash for the event bus",
             "conflicts_with": "Use Redis for the event bus"},
        ],
    }
    _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())
    second = _sync_decompose_agent(_CONV_URI, decomp, "proj", OntologySchema())
    assert second["supersedes"] == 0
    assert second["contradicts"] == 0
    assert second["edges"] == 0
