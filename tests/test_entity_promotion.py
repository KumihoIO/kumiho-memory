"""Tests for write-time entity promotion (entity Items + ABOUT edges)."""

import asyncio
import sys
import types

import grpc
import pytest

from kumiho_memory.entity_promotion import (
    EntityPromotionConfig,
    _slugify_entity,
    _sync_promote,
    promote_entities,
)


class FakeRpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class FakeRevision:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.edges = []

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((target, edge_type, metadata or {}))


class FakeItem:
    def __init__(self, name):
        self.name = name
        self.revisions = []

    def get_latest_revision(self):
        return self.revisions[-1] if self.revisions else None

    def create_revision(self, metadata=None):
        rev = FakeRevision(metadata)
        self.revisions.append(rev)
        return rev


class FakeProject:
    def __init__(self):
        self.name = "CognitiveMemory"
        self.spaces = set()
        self.items = {}

    def create_space(self, name, parent_path=None):
        if name in self.spaces:
            raise FakeRpcError(grpc.StatusCode.ALREADY_EXISTS)
        self.spaces.add(name)

    def create_item(self, item_name, kind, parent_path=None, metadata=None):
        key = (parent_path, item_name, kind)
        if key in self.items:
            raise FakeRpcError(grpc.StatusCode.ALREADY_EXISTS)
        item = FakeItem(item_name)
        self.items[key] = item
        return item

    def get_item(self, item_name, kind, parent_path=None):
        return self.items[(parent_path, item_name, kind)]


def _install_fake_kumiho(monkeypatch, project, source_revision):
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: project
    fake.get_revision = lambda kref: source_revision
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return fake


def test_slugify_entity_is_deterministic_identity_key():
    assert _slugify_entity("Anthropic AI") == "anthropic-ai"
    assert _slugify_entity("  ACME Corp. ") == "acme-corp"
    assert _slugify_entity("anthropic ai") == "anthropic-ai"
    assert _slugify_entity("") == ""
    assert len(_slugify_entity("x" * 200)) <= 48


def test_sync_promote_creates_items_and_about_edges(monkeypatch):
    project = FakeProject()
    source = FakeRevision()
    _install_fake_kumiho(monkeypatch, project, source)

    touched, edges = _sync_promote(
        "kref://p/s/mem.conversation?r=1",
        # Case variants dedupe to one identity within the batch.
        ["Anthropic", "anthropic", "Redis"],
        "CognitiveMemory",
        EntityPromotionConfig(),
    )

    assert touched == 2
    assert edges == 2
    space = "/CognitiveMemory/entities"
    assert (space, "anthropic", "entity") in project.items
    assert (space, "redis", "entity") in project.items
    # First surface form is preserved as the display name on the anchor.
    anchor = project.items[(space, "anthropic", "entity")].revisions[0]
    assert anchor.metadata["display_name"] == "Anthropic"
    # ABOUT edges go memory revision -> entity anchor.
    assert [e[1] for e in source.edges] == ["ABOUT", "ABOUT"]


def test_second_promotion_is_idempotent_on_items(monkeypatch):
    project = FakeProject()
    first_source = FakeRevision()
    _install_fake_kumiho(monkeypatch, project, first_source)
    cfg = EntityPromotionConfig()

    _sync_promote("kref://p/s/mem.conversation?r=1", ["Anthropic"], "CognitiveMemory", cfg)
    second_source = FakeRevision()
    sys.modules["kumiho"].get_revision = lambda kref: second_source
    _sync_promote("kref://p/s/mem.conversation?r=2", ["anthropic"], "CognitiveMemory", cfg)

    space = "/CognitiveMemory/entities"
    item = project.items[(space, "anthropic", "entity")]
    # One entity item, one anchor revision — but each memory links to it.
    assert len(project.items) == 1
    assert len(item.revisions) == 1
    assert len(first_source.edges) == 1
    assert len(second_source.edges) == 1
    # Both edges target the same anchor: the stable hub for traversal.
    assert first_source.edges[0][0] is item.revisions[0]
    assert second_source.edges[0][0] is item.revisions[0]


def test_max_entities_caps_fanout(monkeypatch):
    project = FakeProject()
    source = FakeRevision()
    _install_fake_kumiho(monkeypatch, project, source)

    names = [f"Entity {i}" for i in range(10)]
    touched, edges = _sync_promote(
        "kref://p/s/mem.conversation?r=1",
        names,
        "CognitiveMemory",
        EntityPromotionConfig(max_entities=3),
    )
    assert touched == 3
    assert edges == 3


def test_promote_entities_disabled_is_a_noop(monkeypatch):
    called = []
    fake = types.ModuleType("kumiho")
    fake.get_project = lambda name: called.append(name)
    monkeypatch.setitem(sys.modules, "kumiho", fake)

    result = asyncio.run(
        promote_entities(
            "kref://p/s/mem.conversation?r=1",
            ["Anthropic"],
            project_name="CognitiveMemory",
            config=EntityPromotionConfig(enabled=False),
        )
    )
    assert result == {"entities": 0, "edges": 0}
    assert not called


def test_promote_entities_async_wrapper_runs_worker(monkeypatch):
    project = FakeProject()
    source = FakeRevision()
    _install_fake_kumiho(monkeypatch, project, source)

    result = asyncio.run(
        promote_entities(
            "kref://p/s/mem.conversation?r=1",
            ["Anthropic", "Redis"],
            project_name="CognitiveMemory",
        )
    )
    assert result["entities"] == 2
    assert result["edges"] == 2


def test_manager_env_kill_switch(monkeypatch):
    """KUMIHO_MEMORY_ENTITY_PROMOTION=0 disables the default-on wiring."""
    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from fakes import FakeRedis

    class StubRedactor:
        def redact(self, text):
            return text

        def reject_credentials(self, text):
            return None

    def build(**kwargs):
        return UniversalMemoryManager(
            redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test"),
            summarizer=object(),
            pii_redactor=StubRedactor(),
            memory_store=None,
            **kwargs,
        )

    monkeypatch.setenv("KUMIHO_MEMORY_ENTITY_PROMOTION", "0")
    assert build().entity_promotion_config is None

    monkeypatch.delenv("KUMIHO_MEMORY_ENTITY_PROMOTION", raising=False)
    manager = build()
    assert manager.entity_promotion_config is not None
    assert manager.entity_promotion_config.enabled

    custom = EntityPromotionConfig(max_entities=2)
    assert build(entity_promotion=custom).entity_promotion_config is custom
    assert build(entity_promotion=False).entity_promotion_config is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
