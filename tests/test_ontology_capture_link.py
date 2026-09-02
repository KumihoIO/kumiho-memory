# -*- coding: utf-8 -*-
"""Cross-layer capture linking (#17, kumiho_memory.relations.link_capture_source).

`reflect` writes a decision as a `conversation`-kind item; `consolidate` later
projects the same decision into a typed `decision` node in the SAME space.
Stacking filters by kind so they never merge, and decompose's structural
DERIVED_FROM points at the consolidated summary — so the same decision sat in
the graph as two unrelated nodes. Decompose now links the typed node to the
capture (additive; no dedup, no merge).

In-memory graph fakes (no server) drive the match rule, the type gate, the
env kill switch, the search-failure fallback, and idempotence across two
consolidations of the same session.
"""
import sys
import types

import pytest

from kumiho._text import slugify

from kumiho_memory.ontology import OntologySchema, _sync_decompose

_DECISIONS_PATH = "/proj/decisions"
_CONV = "kref://proj/conversations/session-1.conversation?r=1"

DECISION_TEXT = "Exclude SUPERSEDES from both MCP edge tools"

#: The reflect capture of the same decision: a different wording of one
#: subject, the way an agent actually writes it.
CAPTURE_TITLE = "Exclude SUPERSEDES from both of the MCP edge tools"

SUMMARY = {
    "summary": "we settled the edge-tool boundary",
    "classification": {"entities": []},
    "knowledge": {
        "facts": [],
        "decisions": [{"decision": DECISION_TEXT, "reason": "belief protocol"}],
        "actions": [],
        "open_questions": [],
    },
    "events": [],
}


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #

class _Rev:
    def __init__(self, uri, metadata=None):
        self.kref = types.SimpleNamespace(uri=uri)
        self.metadata = dict(metadata or {})
        self.edges = []  # (edge_type, target_uri, metadata)

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((edge_type, target.kref.uri, dict(metadata or {})))

    def get_edges(self, edge_type_filter=None, direction=0):
        return [
            types.SimpleNamespace(target_kref=types.SimpleNamespace(uri=turi))
            for et, turi, _md in self.edges
            if not edge_type_filter or et == edge_type_filter
        ]


class _Item:
    def __init__(self, uri, metadata=None, materialized=False):
        self.kref = types.SimpleNamespace(uri=uri)
        self._rev = _Rev(f"{uri}?r=1", metadata)
        self._materialized = materialized

    def get_latest_revision(self):
        return self._rev if self._materialized else None

    def create_revision(self, metadata=None):
        if metadata:
            self._rev.metadata.update(metadata)
        self._materialized = True
        return self._rev


class _Project:
    def __init__(self):
        self.items = {}  # (parent_path, slug) -> _Item

    def create_space(self, sp):
        pass

    def capture(self, title, summary="", memory_type="decision",
                space_path=_DECISIONS_PATH, name="reflect-capture"):
        """Seed a `conversation`-kind reflect capture already in the space."""
        uri = f"kref://proj{space_path}/{name}.conversation"
        item = _Item(uri, {"title": title, "summary": summary or title,
                           "type": memory_type}, materialized=True)
        self.items[(space_path, name)] = item
        return item

    def create_item(self, slug, kind, parent_path=None):  # get-or-create
        key = (parent_path, slug)
        if key not in self.items:
            space = parent_path.strip("/").split("/")[-1]
            self.items[key] = _Item(f"kref://proj/{space}/{slug}.{kind}")
        return self.items[key]

    def get_item(self, slug, kind, parent_path=None):
        return self.items[(parent_path, slug)]


class _SearchResult:
    def __init__(self, item):
        self.item = item
        self.score = 1.0


def _install(monkeypatch, proj, *, conversation_hits=(), raise_search=False):
    """Install a `kumiho` module whose search answers per requested kind."""
    conv = _Rev(_CONV)
    calls = []
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: proj
    fake.get_revision = lambda kref: conv

    def _search(query, context=None, kind=None, **kw):
        calls.append((query, context, kind))
        if raise_search:
            raise RuntimeError("backend down")
        if kind == "conversation":
            return [_SearchResult(i) for i in conversation_hits]
        return []

    fake.search = _search
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return conv, calls


def _decision_rev(proj):
    return proj.items[(_DECISIONS_PATH, slugify(DECISION_TEXT, hash_on_truncate=True))]._rev


def _derived_from_targets(rev):
    return [turi for et, turi, _ in rev.edges if et == "DERIVED_FROM"]


# --------------------------------------------------------------------------- #
# The link                                                                     #
# --------------------------------------------------------------------------- #

def test_typed_decision_links_the_reflect_capture(monkeypatch):
    """The typed node keeps its structural provenance AND gains one naming the
    capture the same decision was already written into."""
    proj = _Project()
    cap = proj.capture(CAPTURE_TITLE)
    _install(monkeypatch, proj, conversation_hits=[cap])

    stats = _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())

    assert stats["decisions"] == 1
    assert stats["capture_links"] == 1
    rev = _decision_rev(proj)
    assert _derived_from_targets(rev) == [_CONV, cap._rev.kref.uri]
    # Edge metadata follows the module's basis convention.
    md = [m for et, turi, m in rev.edges if turi == cap._rev.kref.uri][0]
    assert md["basis"] == "lexical-overlap"
    assert float(md["overlap"]) >= 0.6


def test_link_is_drawn_once_across_two_consolidations(monkeypatch):
    """Consolidation re-runs over the same session; the edge must not double."""
    proj = _Project()
    cap = proj.capture(CAPTURE_TITLE)
    _install(monkeypatch, proj, conversation_hits=[cap])
    schema = OntologySchema()

    first = _sync_decompose(_CONV, SUMMARY, "proj", schema)
    second = _sync_decompose(_CONV, SUMMARY, "proj", schema)

    assert first["capture_links"] == 1
    assert second["capture_links"] == 0  # precheck saw it; nothing new written
    rev = _decision_rev(proj)
    assert _derived_from_targets(rev).count(cap._rev.kref.uri) == 1


def test_one_search_per_decision(monkeypatch):
    """Bounded like the alias resolver: one conversation lookup per decision."""
    proj = _Project()
    cap = proj.capture(CAPTURE_TITLE)
    _, calls = _install(monkeypatch, proj, conversation_hits=[cap])

    _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())

    conv_searches = [c for c in calls if c[2] == "conversation"]
    assert len(conv_searches) == 1
    assert conv_searches[0][1] == "proj/decisions"  # scoped to the space


# --------------------------------------------------------------------------- #
# Conservatism: a missed link beats a wrong one                                #
# --------------------------------------------------------------------------- #

def test_unrelated_capture_below_threshold_is_not_linked(monkeypatch):
    proj = _Project()
    cap = proj.capture("Ship the Redis event bus migration this quarter")
    _install(monkeypatch, proj, conversation_hits=[cap])

    stats = _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())

    assert stats["capture_links"] == 0
    assert _derived_from_targets(_decision_rev(proj)) == [_CONV]


@pytest.mark.parametrize("memory_type", ["preference", "correction", "note"])
def test_non_decision_capture_type_is_skipped(monkeypatch, memory_type):
    """The space doesn't identify the layer: a preference can sit in the
    decisions space and share a decision's wording. Only decision-compatible
    capture types (and the `summary` default) are eligible."""
    proj = _Project()
    cap = proj.capture(CAPTURE_TITLE, memory_type=memory_type)
    _install(monkeypatch, proj, conversation_hits=[cap])

    stats = _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())

    assert stats["capture_links"] == 0


def test_untyped_capture_is_eligible(monkeypatch):
    """`summary` is reflect's DEFAULT type — "the agent didn't say", not "not a
    decision" — so it stays eligible."""
    proj = _Project()
    cap = proj.capture(CAPTURE_TITLE, memory_type="summary")
    _install(monkeypatch, proj, conversation_hits=[cap])

    assert _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())["capture_links"] == 1


def test_anchor_conversation_is_never_self_linked(monkeypatch):
    """Consolidation's own summary is a `conversation` too and can live in the
    decisions space, where it matches its own decision. Linking it would add a
    second provenance edge to a *different revision* of the node the structural
    edge already covers."""
    proj = _Project()
    own = _Item("kref://proj/conversations/session-1.conversation",
                {"title": CAPTURE_TITLE, "summary": CAPTURE_TITLE, "type": "summary"},
                materialized=True)
    own._rev = _Rev("kref://proj/conversations/session-1.conversation?r=7",
                    own._rev.metadata)
    _install(monkeypatch, proj, conversation_hits=[own])

    stats = _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())

    assert stats["capture_links"] == 0
    assert _derived_from_targets(_decision_rev(proj)) == [_CONV]


def test_search_failure_falls_back_to_current_behavior(monkeypatch):
    proj = _Project()
    _install(monkeypatch, proj, raise_search=True)

    stats = _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())

    assert stats["decisions"] == 1  # the write still happened
    assert stats["capture_links"] == 0


# --------------------------------------------------------------------------- #
# Gating                                                                       #
# --------------------------------------------------------------------------- #

def test_default_is_on(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_LINK_CAPTURES", raising=False)
    proj = _Project()
    cap = proj.capture(CAPTURE_TITLE)
    _install(monkeypatch, proj, conversation_hits=[cap])

    assert _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())["capture_links"] == 1


@pytest.mark.parametrize("value", ["0", "false", "OFF", "no"])
def test_env_kill_switch(monkeypatch, value):
    monkeypatch.setenv("KUMIHO_MEMORY_LINK_CAPTURES", value)
    proj = _Project()
    cap = proj.capture(CAPTURE_TITLE)
    _, calls = _install(monkeypatch, proj, conversation_hits=[cap])

    stats = _sync_decompose(_CONV, SUMMARY, "proj", OntologySchema())

    assert stats["capture_links"] == 0
    assert not [c for c in calls if c[2] == "conversation"]  # no lookup at all
    assert _derived_from_targets(_decision_rev(proj)) == [_CONV]
