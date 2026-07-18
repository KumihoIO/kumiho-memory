# -*- coding: utf-8 -*-
"""Valid-time interval WRITE side (ontology G8): the agent decompose path
persists ``valid_from`` / ``valid_to`` as additive fact metadata, validated as
ISO dates, without touching ``event_date`` or #119's ``event_date_confidence``.
"""
import sys
import types

from kumiho._text import slugify

from kumiho_memory.ontology import OntologySchema, _sync_decompose_agent

_CONV = "kref://proj/conversations/c.conversation?r=1"


class _Rev:
    def __init__(self, uri, metadata=None):
        self.kref = types.SimpleNamespace(uri=uri)
        self.metadata = dict(metadata or {})
        self.edges = []

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((edge_type, target.kref.uri))

    def get_edges(self, edge_type_filter=None, direction=0):
        return []


class _Item:
    def __init__(self, uri):
        self._rev = _Rev(uri)
        self._materialized = False

    def get_latest_revision(self):
        return self._rev if self._materialized else None

    def create_revision(self, metadata=None):
        if metadata:
            self._rev.metadata.update(metadata)
        self._materialized = True
        return self._rev


class _Project:
    def __init__(self):
        self.items = {}

    def create_space(self, sp):
        pass

    def create_item(self, slug, kind, parent_path=None):
        key = (parent_path, slug)
        if key not in self.items:
            self.items[key] = _Item(f"kref://proj/{kind}s/{slug}.{kind}?r=1")
        return self.items[key]

    def get_item(self, slug, kind, parent_path=None):
        return self.items[(parent_path, slug)]


def _install(monkeypatch):
    proj = _Project()
    conv = _Rev(_CONV)
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: proj
    fake.get_revision = lambda kref: conv
    fake.search = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return proj


def _fact_meta(proj, statement):
    slug = slugify(statement, hash_on_truncate=True)
    return proj.items[("/proj/facts", slug)]._rev.metadata


def test_valid_interval_written_as_metadata(monkeypatch):
    proj = _install(monkeypatch)
    decomp = {"facts": [{
        "statement": "Alice worked at Acme",
        "valid_from": "2019", "valid_to": "2022-06",
    }]}
    _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    meta = _fact_meta(proj, "Alice worked at Acme")
    assert meta["valid_from"] == "2019"
    assert meta["valid_to"] == "2022-06"


def test_open_ended_interval_only_writes_present_bound(monkeypatch):
    proj = _install(monkeypatch)
    decomp = {"facts": [{"statement": "Bob leads the team", "valid_from": "2023-01"}]}
    _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    meta = _fact_meta(proj, "Bob leads the team")
    assert meta["valid_from"] == "2023-01"
    assert "valid_to" not in meta


def test_invalid_bound_is_dropped(monkeypatch):
    proj = _install(monkeypatch)
    decomp = {"facts": [{"statement": "Carol joined", "valid_from": "last spring"}]}
    _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    meta = _fact_meta(proj, "Carol joined")
    assert "valid_from" not in meta


def test_absent_interval_writes_nothing(monkeypatch):
    proj = _install(monkeypatch)
    decomp = {"facts": [{"statement": "Dave shipped it"}]}
    _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    meta = _fact_meta(proj, "Dave shipped it")
    assert "valid_from" not in meta and "valid_to" not in meta


def test_event_date_and_confidence_keys_untouched(monkeypatch):
    """G8 must not read or write event_date, nor #119's event_date_confidence —
    the interval keys are strictly independent additive metadata."""
    proj = _install(monkeypatch)
    decomp = {"facts": [{
        "statement": "Eve moved teams", "valid_from": "2021",
        # A hostile fact that also carries the sibling keys: they must NOT be
        # copied into the written fact metadata by the valid-time path.
        "event_date": "2021-03-02", "event_date_confidence": "high",
    }]}
    _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    meta = _fact_meta(proj, "Eve moved teams")
    assert meta["valid_from"] == "2021"
    assert "event_date" not in meta
    assert "event_date_confidence" not in meta
