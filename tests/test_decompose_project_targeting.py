# -*- coding: utf-8 -*-
"""Project targeting for keyless agent decompose (issue #136).

``UniversalMemoryManager.memory_decompose`` gains an optional ``project`` that
routes the materialized typed nodes into a DIFFERENT project. These tests pin
the three contract points:

  (a) an explicit, resolvable project materializes into ``/{project}/{space}``
      (not the manager's configured project);
  (b) omitting ``project`` is byte-identical to today — the configured project
      is threaded through unchanged and NO project probe runs;
  (c) an explicit project that is absent / inaccessible falls back to the
      configured project without raising (best-effort contract).

Uses the in-memory graph fakes + the ``sys.modules`` fake-SDK seam
(``monkeypatch.setitem``, never ``pop``) so ``kumiho.get_project`` /
``get_revision`` / ``search`` are all under test control.
"""
import asyncio
import sys
import types

from kumiho_memory.redis_memory import RedisMemoryBuffer
from kumiho_memory.memory_manager import UniversalMemoryManager

from fakes import FakeRedis


# --- in-memory graph fakes (mirror tests/test_ontology_agent.py) -----------


class _FakeRev:
    def __init__(self, uri):
        self.kref = type("K", (), {"uri": uri})()
        self.edges = []

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


def _manager(project="ConfiguredProj"):
    buffer = RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test")
    mgr = UniversalMemoryManager(project=project, redis_buffer=buffer)
    mgr.ontology_enabled = True   # decompose self-gates on this
    return mgr


_CONV = "kref:/x/conversations/c.conversation?r=1"
_PAYLOAD = dict(
    entities=[{"name": "Alpha"}],
    facts=[{"statement": "Alpha exists", "about": ["Alpha"]}],
    relations=[],
)


def _install_kumiho(monkeypatch, *, projects, conv_rev, get_project=None):
    fake = types.ModuleType("kumiho")
    fake.get_project = get_project or (lambda name: projects.get(name))
    fake.get_revision = lambda kref: conv_rev
    fake.search = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return fake


# --- (a) explicit resolvable project routes materialization ----------------


def test_explicit_project_routes_materialization(monkeypatch):
    conv = _FakeRev(_CONV)
    other = _FakeProject("OtherProj")
    configured = _FakeProject("ConfiguredProj")
    _install_kumiho(
        monkeypatch,
        projects={"OtherProj": other, "ConfiguredProj": configured},
        conv_rev=conv,
    )

    mgr = _manager(project="ConfiguredProj")
    res = asyncio.run(mgr.memory_decompose(_CONV, project="OtherProj", **_PAYLOAD))

    stats = res["decomposed"]
    assert stats["entities"] == 1 and stats["facts"] == 1
    # Materialized into /OtherProj/... — the explicit target's spaces.
    assert any(sp == "/OtherProj/entities" for (sp, _slug) in other._items)
    assert any(sp == "/OtherProj/facts" for (sp, _slug) in other._items)
    # The manager's configured project received NOTHING.
    assert configured._items == {}


# --- (b) omitting project is byte-identical to today -----------------------


def test_omitted_is_byte_identical(monkeypatch):
    """No probe; the configured project is threaded through unchanged — and it
    is the SAME thread as explicitly passing the configured project."""
    import kumiho_memory.ontology as ontology

    captured = []

    async def _capture(kref, decomposition, *, project_name, **kw):
        captured.append((kref, project_name, decomposition))
        return {"entities": 0, "facts": 0, "relations": 0, "edges": 0}

    monkeypatch.setattr(ontology, "decompose_and_link_agent", _capture)

    probe_calls = []
    fake = types.ModuleType("kumiho")
    # If _resolve_decompose_project probed on the omitted path, this records it.
    fake.get_project = lambda name: probe_calls.append(name) or object()
    monkeypatch.setitem(sys.modules, "kumiho", fake)

    mgr = _manager(project="ConfiguredProj")

    # Omitted.
    asyncio.run(mgr.memory_decompose(_CONV, **_PAYLOAD))
    # Explicit == configured (must short-circuit the probe too).
    asyncio.run(mgr.memory_decompose(_CONV, project="ConfiguredProj", **_PAYLOAD))

    assert probe_calls == []                    # NEVER probed in either case
    assert len(captured) == 2
    # Both calls thread the identical (kref, project_name, decomposition).
    assert captured[0] == captured[1]
    assert captured[0][1] == "ConfiguredProj"   # == self.project


# --- (c) unresolvable explicit project falls back, never raises ------------


def test_unresolvable_project_falls_back(monkeypatch):
    conv = _FakeRev(_CONV)
    configured = _FakeProject("ConfiguredProj")
    # get_project only knows the configured project; the explicit "Ghost" => None.
    _install_kumiho(
        monkeypatch,
        projects={"ConfiguredProj": configured},
        conv_rev=conv,
    )

    mgr = _manager(project="ConfiguredProj")
    res = asyncio.run(mgr.memory_decompose(_CONV, project="Ghost", **_PAYLOAD))

    # Did not raise; materialized into the CONFIGURED project (fallback).
    stats = res["decomposed"]
    assert stats["entities"] == 1 and stats["facts"] == 1
    assert any(sp == "/ConfiguredProj/entities" for (sp, _slug) in configured._items)


def test_get_project_raising_falls_back(monkeypatch):
    """A raising probe is caught — decompose stays best-effort, never raises."""
    conv = _FakeRev(_CONV)
    configured = _FakeProject("ConfiguredProj")

    def _boom(name):
        if name == "ConfiguredProj":
            return configured
        raise RuntimeError("backend unavailable")

    _install_kumiho(
        monkeypatch,
        projects={},
        conv_rev=conv,
        get_project=_boom,
    )

    mgr = _manager(project="ConfiguredProj")
    res = asyncio.run(mgr.memory_decompose(_CONV, project="Unreachable", **_PAYLOAD))

    stats = res["decomposed"]
    assert stats["entities"] == 1 and stats["facts"] == 1
    assert any(sp == "/ConfiguredProj/entities" for (sp, _slug) in configured._items)
