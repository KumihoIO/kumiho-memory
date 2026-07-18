# -*- coding: utf-8 -*-
"""Write-time entity alias resolution (ontology gap G5, kumiho_memory.ontology).

Before minting a new entity hub, decompose resolves the surface form against
EXISTING graph hubs (normalized name + their stored ``aliases``) so a new
session's "PostgreSQL" reuses the existing "Postgres" hub instead of forking a
duplicate. Flag-gated (KUMIHO_MEMORY_ALIAS_RESOLUTION, default OFF); a lookup
failure falls back to the current create-new behavior and never blocks a write.

In-memory graph fakes (no server) drive the decision table + the bound (one
search per new surface, cached) + the fallback.
"""
import sys
import types

from kumiho._text import slugify

from kumiho_memory import ontology
from kumiho_memory.ontology import OntologySchema, _AliasResolver, _sync_decompose_agent

_CONV = "kref://proj/conversations/c.conversation?r=1"
_ENTITIES_PATH = "/proj/entities"


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #

class _Rev:
    def __init__(self, uri, metadata=None):
        self.kref = types.SimpleNamespace(uri=uri)
        self.metadata = dict(metadata or {})
        self.edges = []          # (edge_type, target_uri, metadata)
        self.meta_updates = []   # set_metadata calls

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((edge_type, target.kref.uri, dict(metadata or {})))

    def get_edges(self, edge_type_filter=None, direction=0):
        out = []
        for et, turi, _md in self.edges:
            if edge_type_filter and et != edge_type_filter:
                continue
            out.append(types.SimpleNamespace(
                target_kref=types.SimpleNamespace(uri=turi)))
        return out

    def set_metadata(self, metadata):
        self.metadata.update(metadata)
        self.meta_updates.append(dict(metadata))


class _Item:
    def __init__(self, uri, metadata=None, materialized=False):
        self.kref = types.SimpleNamespace(uri=uri)
        self._rev = _Rev(uri, metadata)
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
        self.items = {}          # (parent_path, slug) -> _Item
        self.created_slugs = []   # slugs create_item actually minted

    def create_space(self, sp):
        pass

    def preexisting_entity(self, name, aliases=None):
        """Seed an already-materialized entity hub (a prior session's node)."""
        slug = slugify(name, hash_on_truncate=True)
        md = {"display_name": name}
        if aliases:
            md["aliases"] = ", ".join(aliases)
        uri = f"kref://proj/entities/{slug}.entity?r=1"
        self.items[(_ENTITIES_PATH, slug)] = _Item(uri, md, materialized=True)
        return self.items[(_ENTITIES_PATH, slug)]

    def create_item(self, slug, kind, parent_path=None):  # get-or-create
        key = (parent_path, slug)
        if key not in self.items:
            self.items[key] = _Item(f"kref://proj/{parent_path.strip('/').split('/')[-1]}/{slug}.{kind}?r=1")
            if kind == "entity":
                self.created_slugs.append(slug)
        return self.items[key]

    def get_item(self, slug, kind, parent_path=None):
        return self.items[(parent_path, slug)]


class _SearchResult:
    def __init__(self, item):
        self.item = item


def _install(monkeypatch, proj, *, search=None, raise_search=False):
    conv = _Rev(_CONV)
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: proj
    known = {_CONV: conv}

    def _get_revision(kref):
        return known[kref]

    fake.get_revision = _get_revision

    def _search(*a, **k):
        if raise_search:
            raise RuntimeError("backend down")
        return search or []

    fake.search = _search
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return conv


def _enable(monkeypatch, on=True):
    monkeypatch.setenv("KUMIHO_MEMORY_ALIAS_RESOLUTION", "1" if on else "0")


def _entity_items(proj):
    return {slug for (sp, slug) in proj.items if sp == _ENTITIES_PATH}


# --------------------------------------------------------------------------- #
# Decision table                                                              #
# --------------------------------------------------------------------------- #

def test_new_surface_reuses_existing_hub_via_stored_alias(monkeypatch):
    """"PostgreSQL" resolves onto the existing "Postgres" hub (alias match) —
    no duplicate 'postgresql' entity, the fact ABOUT-links the existing hub."""
    _enable(monkeypatch)
    proj = _Project()
    hub = proj.preexisting_entity("Postgres", aliases=["PostgreSQL", "psql"])
    _install(monkeypatch, proj, search=[_SearchResult(hub)])

    decomp = {
        "entities": [{"name": "PostgreSQL", "type": "system"}],
        "facts": [{"statement": "PostgreSQL 16 ships logical replication",
                   "about": ["PostgreSQL"]}],
    }
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())

    assert stats["entities_reused"] == 1
    assert stats["entities"] == 0
    # No new entity item was minted; only the pre-seeded 'postgres' hub exists.
    assert _entity_items(proj) == {slugify("Postgres", hash_on_truncate=True)}
    assert "postgresql" not in _entity_items(proj)
    # The fact's ABOUT edge targets the reused hub.
    fact_slug = slugify("PostgreSQL 16 ships logical replication", hash_on_truncate=True)
    fact_rev = proj.items[("/proj/facts", fact_slug)]._rev
    about = [turi for et, turi, _ in fact_rev.edges if et == "ABOUT"]
    assert about == [hub._rev.kref.uri]


def test_no_match_creates_new_hub(monkeypatch):
    _enable(monkeypatch)
    proj = _Project()
    # Search returns an unrelated hub whose surfaces don't overlap.
    other = proj.preexisting_entity("Kafka")
    _install(monkeypatch, proj, search=[_SearchResult(other)])
    decomp = {"entities": [{"name": "Redis"}], "facts": []}
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    assert stats["entities"] == 1
    assert stats["entities_reused"] == 0
    assert slugify("Redis", hash_on_truncate=True) in proj.created_slugs


def test_exact_slug_is_not_a_reuse_get_or_create_owns_it(monkeypatch):
    """An identical surface (same slug) is the get-or-create path's job — the
    resolver must skip it so behavior stays additive (counted as entities, not
    entities_reused)."""
    _enable(monkeypatch)
    proj = _Project()
    hub = proj.preexisting_entity("Redis")
    _install(monkeypatch, proj, search=[_SearchResult(hub)])
    decomp = {"entities": [{"name": "Redis"}], "facts": []}
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    assert stats["entities_reused"] == 0
    assert stats["entities"] == 1


def test_exact_slug_hub_wins_over_alias_hub(monkeypatch):
    """If a hub with OUR own slug exists, defer to get-or-create (return None)
    even when another hub lists our surface as an alias — never override true
    identity by folding into a different hub."""
    _enable(monkeypatch)
    proj = _Project()
    exact = proj.preexisting_entity("PostgreSQL")                 # slug postgresql
    aliasing = proj.preexisting_entity("Postgres", aliases=["PostgreSQL"])
    # Search returns the aliasing hub FIRST, then the exact hub.
    _install(monkeypatch, proj, search=[_SearchResult(aliasing), _SearchResult(exact)])
    decomp = {"entities": [{"name": "PostgreSQL"}], "facts": []}
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    # Deferred to get-or-create: not counted as a reuse, no alias-fold onto Postgres.
    assert stats["entities_reused"] == 0
    assert aliasing._rev.meta_updates == []


def test_lookup_failure_falls_back_to_create(monkeypatch):
    """A search error never blocks the write: fall back to minting a new hub."""
    _enable(monkeypatch)
    proj = _Project()
    _install(monkeypatch, proj, raise_search=True)
    decomp = {"entities": [{"name": "PostgreSQL"}], "facts": []}
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    assert stats["entities"] == 1
    assert stats["entities_reused"] == 0
    assert slugify("PostgreSQL", hash_on_truncate=True) in proj.created_slugs


def test_flag_off_never_resolves(monkeypatch):
    """With the flag OFF a matching hub in search is ignored — a duplicate hub
    is minted, exactly the pre-G5 behavior."""
    _enable(monkeypatch, on=False)
    proj = _Project()
    hub = proj.preexisting_entity("Postgres", aliases=["PostgreSQL"])
    _install(monkeypatch, proj, search=[_SearchResult(hub)])
    decomp = {"entities": [{"name": "PostgreSQL"}], "facts": []}
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    assert stats["entities_reused"] == 0
    assert stats["entities"] == 1
    assert slugify("PostgreSQL", hash_on_truncate=True) in proj.created_slugs


def test_reuse_appends_new_surface_as_alias(monkeypatch):
    """A new surface not already listed is appended to the reused hub's aliases
    (match here is via our supplied alias 'Postgres' == the hub's name)."""
    _enable(monkeypatch)
    proj = _Project()
    hub = proj.preexisting_entity("Postgres")   # no aliases yet
    _install(monkeypatch, proj, search=[_SearchResult(hub)])
    decomp = {"entities": [{"name": "psql", "aliases": ["Postgres"]}], "facts": []}
    stats = _sync_decompose_agent(_CONV, decomp, "proj", OntologySchema())
    assert stats["entities_reused"] == 1
    # 'psql' (the new surface) is appended; 'Postgres' (already the display) not.
    assert hub._rev.meta_updates, "expected an alias append"
    appended = hub._rev.metadata["aliases"]
    assert "psql" in appended
    assert appended.count("Postgres") == 0  # display name isn't duplicated in aliases


# --------------------------------------------------------------------------- #
# Bound: one lookup per new surface, cached                                    #
# --------------------------------------------------------------------------- #

def test_resolver_caches_one_lookup_per_surface(monkeypatch):
    calls = []
    proj = _Project()
    hub = proj.preexisting_entity("Postgres", aliases=["PostgreSQL"])

    fake = types.ModuleType("kumiho")

    def _search(query, **k):
        calls.append(query)
        return [_SearchResult(hub)]

    fake.search = _search
    monkeypatch.setitem(sys.modules, "kumiho", fake)

    resolver = _AliasResolver("proj", OntologySchema(), enabled=True)
    first = resolver.resolve("PostgreSQL")
    second = resolver.resolve("PostgreSQL")
    assert first is not None and second == first
    assert len(calls) == 1   # cached — no second search for the same surface


def test_resolver_disabled_does_no_lookup(monkeypatch):
    calls = []
    fake = types.ModuleType("kumiho")
    fake.search = lambda *a, **k: calls.append(1) or []
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    resolver = _AliasResolver("proj", OntologySchema(), enabled=False)
    assert resolver.resolve("Anything") is None
    assert calls == []
