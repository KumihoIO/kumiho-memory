# -*- coding: utf-8 -*-
"""Grounding-staleness ripple (kumiho_memory.grounding + write-path wiring, #95).

In-memory graph fakes (no server) assert the deterministic write layer: on a
SUPERSEDES landing on a fact F, the decisions with a DEPENDS_ON edge INTO F are
flagged ``grounding_stale`` (metadata + mirrored tag). Covers the shared helper
directly, both write paths that call it (heuristic ``relations.link_supersedes``
and agent-declared ``ontology._sync_decompose_agent``), the fan-out cap, and
idempotency. The recall marker, context note, and maintenance clear live in
test_graph_augmentation / test_context_compose / test_graph_maintenance.

Fake-SDK seam: ``monkeypatch.setitem(sys.modules, 'kumiho', ...)`` only (never
pop), per repo convention.
"""
import sys
import types

from kumiho._text import slugify

from kumiho_memory.grounding import (
    GROUNDING_STALE_META,
    GROUNDING_STALE_SUPERSEDED_BY_META,
    GROUNDING_STALE_TAG,
    apply_grounding_marker,
    is_grounding_stale,
    ripple_grounding_stale,
)
from kumiho_memory.ontology import OntologySchema, _sync_decompose_agent
from kumiho_memory.relations import link_supersedes

_INCOMING, _OUTGOING, _BOTH = 1, 0, 2


# ---------------------------------------------------------------------------
# In-memory graph fakes
# ---------------------------------------------------------------------------


class _Kref:
    def __init__(self, uri):
        self.uri = uri


class _Edge:
    def __init__(self, source, target, edge_type):
        self.source_kref = _Kref(source)
        self.target_kref = _Kref(target)
        self.edge_type = edge_type


class _Rev:
    """A revision that models incoming/outgoing edges + metadata + tags."""

    def __init__(self, uri, metadata=None, incoming=None, outgoing=None, item=None):
        self.kref = _Kref(uri)
        self.metadata = dict(metadata or {})
        self.tags = []
        self.item = item
        self._incoming = list(incoming or [])   # edges where this rev is target
        self._outgoing = list(outgoing or [])   # edges where this rev is source
        self.set_metadata_calls = 0

    def get_edges(self, edge_type_filter=None, direction=0):
        if direction == _INCOMING:
            pool = self._incoming
        elif direction == _OUTGOING:
            pool = self._outgoing
        else:
            pool = self._incoming + self._outgoing
        return [e for e in pool if not edge_type_filter or e.edge_type == edge_type_filter]

    def create_edge(self, target, edge_type, metadata=None):
        self._outgoing.append(_Edge(self.kref.uri, target.kref.uri, edge_type))

    def set_metadata(self, md):
        self.metadata.update(md)
        self.set_metadata_calls += 1
        return self

    def tag(self, t):
        # Append unconditionally so a double-tag would be observable (the
        # ripple's idempotency must prevent that, not the fake).
        self.tags.append(t)

    def get_item(self):
        return self.item


class _FactItem:
    """A get-or-create fact item; anchor rev is materialized on first write."""

    def __init__(self, uri, rev=None):
        self.kref = _Kref(uri)
        self.deprecated = False
        self._rev = rev

    def get_latest_revision(self):
        return self._rev

    def create_revision(self, metadata=None):
        self._rev = _Rev(f"{self.kref.uri}?r=1", metadata)
        return self._rev


class _Project:
    def __init__(self, name="proj"):
        self.name = name
        self._items = {}   # (parent_path, slug, kind) -> item

    def create_space(self, sp):
        pass

    def create_item(self, slug, kind, parent_path=None):   # get-or-create
        key = (parent_path, slug, kind)
        if key not in self._items:
            self._items[key] = _FactItem(f"kref:/{parent_path}/{slug}.{kind}")
        return self._items[key]

    def get_item(self, slug, kind, parent_path=None):
        return self._items[(parent_path, slug, kind)]

    def preload(self, slug, kind, rev):
        item = _FactItem(f"kref:/{self.name}/{kind}s/{slug}.{kind}", rev=rev)
        rev.item = item
        self._items[(f"/{self.name}/{kind}s", slug, kind)] = item
        return item


def _install(monkeypatch, revs, project=None, conv_rev=None, search_results=None):
    known = dict(revs)
    if conv_rev is not None:
        known[conv_rev.kref.uri] = conv_rev
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: project
    fake.get_revision = lambda kref: known[kref]   # KeyError -> swallowed as failure
    fake.search = lambda *a, **k: (search_results or [])
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return fake


class _Mat:
    """Materializer stub: ``edge`` always links (records nothing needed here)."""

    def edge(self, source, target, edge_type, metadata=None):
        source.create_edge(target, edge_type, metadata)
        return True


_FNEW = "kref:/proj/facts/fnew.fact?r=1"
_FURI = "kref:/proj/facts/f.fact?r=1"


def _fact_with_dependents(dep_uris):
    incoming = [_Edge(u, _FURI, "DEPENDS_ON") for u in dep_uris]
    return _Rev(_FURI, metadata={"claim": "old belief"}, incoming=incoming)


# ---------------------------------------------------------------------------
# Shared helper — small units
# ---------------------------------------------------------------------------


def test_is_grounding_stale_reads_canonical_flag():
    assert is_grounding_stale({GROUNDING_STALE_META: "true"})
    assert is_grounding_stale({GROUNDING_STALE_META: "TRUE"})
    assert not is_grounding_stale({GROUNDING_STALE_META: "false"})
    assert not is_grounding_stale({})
    assert not is_grounding_stale(None)


def test_apply_grounding_marker_is_additive():
    entry = {"kref": "k", "score": 0.5}
    apply_grounding_marker(entry, {
        GROUNDING_STALE_META: "true",
        GROUNDING_STALE_SUPERSEDED_BY_META: _FNEW,
    })
    assert entry["grounding_stale"] is True
    assert entry["superseded_by"] == _FNEW
    # score/kref untouched (purely additive)
    assert entry["score"] == 0.5

    clean = {"kref": "k"}
    apply_grounding_marker(clean, {"evidence_level": "official"})
    assert "grounding_stale" not in clean


# ---------------------------------------------------------------------------
# ripple_grounding_stale — the shared helper
# ---------------------------------------------------------------------------


def test_ripple_flags_all_dependents(monkeypatch):
    d_uris = ["kref:/proj/decisions/d1.decision?r=1",
              "kref:/proj/decisions/d2.decision?r=1"]
    deps = {u: _Rev(u) for u in d_uris}
    F = _fact_with_dependents(d_uris)
    _install(monkeypatch, deps)

    n = ripple_grounding_stale(F, _FNEW)

    assert n == 2
    for d in deps.values():
        assert d.metadata[GROUNDING_STALE_META] == "true"
        assert d.metadata[GROUNDING_STALE_SUPERSEDED_BY_META] == _FNEW
        assert d.tags == [GROUNDING_STALE_TAG]


def test_ripple_ignores_non_depends_on_and_wrong_target(monkeypatch):
    d_uri = "kref:/proj/decisions/d.decision?r=1"
    other = "kref:/proj/decisions/other.decision?r=1"
    F = _Rev(_FURI, incoming=[
        _Edge(d_uri, _FURI, "DEPENDS_ON"),          # a real grounding
        _Edge(other, _FURI, "ABOUT"),               # not a grounding edge
        _Edge("kref:/x.decision?r=1", "kref:/other-fact.fact?r=1", "DEPENDS_ON"),  # targets a different fact
    ])
    deps = {d_uri: _Rev(d_uri), other: _Rev(other)}
    _install(monkeypatch, deps)

    n = ripple_grounding_stale(F, _FNEW)

    assert n == 1
    assert deps[d_uri].metadata.get(GROUNDING_STALE_META) == "true"
    assert not is_grounding_stale(deps[other].metadata)


def test_ripple_respects_fanout_cap(monkeypatch):
    d_uris = [f"kref:/proj/decisions/d{i}.decision?r=1" for i in range(25)]
    deps = {u: _Rev(u) for u in d_uris}
    F = _fact_with_dependents(d_uris)
    _install(monkeypatch, deps)

    n = ripple_grounding_stale(F, _FNEW, cap=20)

    assert n == 20
    stamped = sum(1 for d in deps.values() if is_grounding_stale(d.metadata))
    assert stamped == 20


def test_ripple_is_idempotent_no_restamp_no_duplicate_tag(monkeypatch):
    d_uri = "kref:/proj/decisions/d.decision?r=1"
    dep = _Rev(d_uri)
    F = _fact_with_dependents([d_uri])
    _install(monkeypatch, {d_uri: dep})

    first = ripple_grounding_stale(F, _FNEW)
    second = ripple_grounding_stale(F, _FNEW)

    assert first == 1
    assert second == 0                       # already stale -> not re-stamped
    assert dep.set_metadata_calls == 1        # metadata written exactly once
    assert dep.tags == [GROUNDING_STALE_TAG]  # tag not duplicated


def test_ripple_best_effort_swallows_missing_dependent(monkeypatch):
    # A dependent whose revision can't be fetched is skipped, not raised.
    present = "kref:/proj/decisions/present.decision?r=1"
    missing = "kref:/proj/decisions/missing.decision?r=1"
    F = _fact_with_dependents([missing, present])
    _install(monkeypatch, {present: _Rev(present)})   # 'missing' -> KeyError

    n = ripple_grounding_stale(F, _FNEW)

    assert n == 1  # only the resolvable dependent stamped; the missing one swallowed


def test_ripple_no_edges_is_noop(monkeypatch):
    F = _Rev(_FURI)   # no incoming DEPENDS_ON
    _install(monkeypatch, {})
    assert ripple_grounding_stale(F, _FNEW) == 0


# ---------------------------------------------------------------------------
# Heuristic path: relations.link_supersedes(kind="fact") triggers the ripple
# ---------------------------------------------------------------------------


def test_heuristic_fact_supersede_ripples(monkeypatch):
    d_uri = "kref:/Proj/decisions/keep-upstash.decision?r=1"
    dep = _Rev(d_uri)
    prior_uri = "kref:/Proj/facts/use-upstash.fact?r=1"
    prior = _Rev(prior_uri, metadata={"fact": "Use Upstash for the event bus streams"},
                 incoming=[_Edge(d_uri, prior_uri, "DEPENDS_ON")])
    prior_item = _FactItem("kref:/Proj/facts/use-upstash.fact", rev=prior)
    result = types.SimpleNamespace(item=prior_item, score=999.0)
    _install(monkeypatch, {d_uri: dep}, search_results=[result])

    new_anchor = _Rev("kref:/Proj/facts/use-redis.fact?r=1")
    n = link_supersedes(
        _Mat(), "fact", "facts", "use-redis", new_anchor,
        "Use Redis for the event bus streams", "Proj",
    )

    assert n == 1                                       # SUPERSEDES landed
    assert dep.metadata[GROUNDING_STALE_META] == "true"
    assert dep.metadata[GROUNDING_STALE_SUPERSEDED_BY_META] == new_anchor.kref.uri
    assert dep.tags == [GROUNDING_STALE_TAG]


def test_heuristic_decision_supersede_does_not_ripple(monkeypatch):
    # Decisions carry no incoming DEPENDS_ON; a decision->decision supersede must
    # not even attempt the ripple (kind gate). Prior decision rev has no
    # get_edges-relevant grounding; the dependent must stay untouched.
    prior_uri = "kref:/Proj/decisions/use-upstash.decision?r=1"
    prior = _Rev(prior_uri, metadata={"decision": "Use Upstash for the event bus streams"})
    prior_item = _FactItem("kref:/Proj/decisions/use-upstash.decision", rev=prior)
    result = types.SimpleNamespace(item=prior_item, score=999.0)
    _install(monkeypatch, {}, search_results=[result])

    new_anchor = _Rev("kref:/Proj/decisions/use-redis.decision?r=1")
    n = link_supersedes(
        _Mat(), "decision", "decisions", "use-redis", new_anchor,
        "Use Redis for the event bus streams", "Proj",
    )

    assert n == 1
    # prior rev never gets a get_edges/ripple call — no stamping anywhere.
    assert not is_grounding_stale(prior.metadata)


# ---------------------------------------------------------------------------
# Agent-declared path: ontology._sync_decompose_agent supersedes triggers ripple
# ---------------------------------------------------------------------------

_CONV = "kref:/proj/conversations/c.conversation?r=1"


def test_agent_declared_supersede_ripples(monkeypatch):
    proj = _Project("proj")
    conv = _Rev(_CONV)
    # A prior fact P (from an earlier session) grounded by decision D.
    d_uri = "kref:/proj/decisions/keep-p.decision?r=1"
    dep = _Rev(d_uri)
    prior_slug = slugify("Prior belief P", hash_on_truncate=True)
    prior_uri = f"kref:/proj/facts/{prior_slug}.fact"
    prior_rev = _Rev(prior_uri, metadata={"claim": "Prior belief P"},
                     incoming=[_Edge(d_uri, prior_uri, "DEPENDS_ON")])
    proj.preload(prior_slug, "fact", prior_rev)
    _install(monkeypatch, {d_uri: dep}, project=proj, conv_rev=conv)

    decomp = {
        "facts": [{"statement": "New belief Q"}],
        "supersedes": [{"statement": "New belief Q", "replaces": "Prior belief P"}],
    }
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())

    assert stats["supersedes"] == 1
    # The prior fact's dependent decision is now flagged grounding-stale.
    assert dep.metadata[GROUNDING_STALE_META] == "true"
    # ... pointing at the NEW fact that superseded its grounding.
    new_fact_rev = proj.get_item(
        slugify("New belief Q", hash_on_truncate=True), "fact", "/proj/facts",
    ).get_latest_revision()
    assert dep.metadata[GROUNDING_STALE_SUPERSEDED_BY_META] == new_fact_rev.kref.uri
    assert dep.tags == [GROUNDING_STALE_TAG]


def test_agent_contradicts_does_not_ripple(monkeypatch):
    # CONTRADICTS is not a superseded grounding — the ripple must not fire.
    proj = _Project("proj")
    conv = _Rev(_CONV)
    d_uri = "kref:/proj/decisions/keep-p.decision?r=1"
    dep = _Rev(d_uri)
    prior_slug = slugify("Prior belief P", hash_on_truncate=True)
    prior_uri = f"kref:/proj/facts/{prior_slug}.fact"
    prior_rev = _Rev(prior_uri, metadata={"claim": "Prior belief P"},
                     incoming=[_Edge(d_uri, prior_uri, "DEPENDS_ON")])
    proj.preload(prior_slug, "fact", prior_rev)
    _install(monkeypatch, {d_uri: dep}, project=proj, conv_rev=conv)

    decomp = {
        "facts": [{"statement": "New belief Q"}],
        "contradicts": [{"statement": "New belief Q", "conflicts_with": "Prior belief P"}],
    }
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())

    assert stats["contradicts"] == 1
    assert not is_grounding_stale(dep.metadata)   # contested != grounding-stale
