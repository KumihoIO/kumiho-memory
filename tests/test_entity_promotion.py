"""Tests for write-time entity promotion (entity Items + ABOUT edges)."""

import asyncio
import sys
import types

import grpc
import pytest

from kumiho_memory import entity_promotion
from kumiho_memory.entity_promotion import (
    EntityPromotionConfig,
    _slugify_entity,
    _sync_promote,
    promote_entities,
)


@pytest.fixture(autouse=True)
def _clear_module_caches():
    # entity_promotion caches resolved projects and per-slug locks in module
    # globals; clear them so fakes don't leak across tests.
    entity_promotion._project_cache.clear()
    with entity_promotion._anchor_locks_guard:
        entity_promotion._anchor_locks.clear()
    yield


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


def test_slugify_entity_preserves_non_ascii():
    # Korean/CJK entity names must NOT collapse to "" (they'd be dropped).
    assert _slugify_entity("김철수") == "김철수"
    assert _slugify_entity("株式会社") == "株式会社"
    # Mixed script keeps both.
    assert _slugify_entity("OpenAI 오픈에이아이") == "openai-오픈에이아이"


def test_slugify_entity_hash_suffix_avoids_long_name_collision():
    # Two distinct names sharing a 48-char prefix must NOT collide into one
    # entity — a wrong merge is irreversible (kref identity is permanent).
    a = "x" * 60 + "-alpha"
    b = "x" * 60 + "-beta"
    sa, sb = _slugify_entity(a), _slugify_entity(b)
    assert sa != sb
    assert len(sa) <= 48 and len(sb) <= 48


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


def _build_manager(**kwargs):
    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from fakes import FakeRedis

    class StubRedactor:
        def redact(self, text):
            return text

        def reject_credentials(self, text):
            return None

    return UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test"),
        summarizer=object(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        **kwargs,
    )


def test_ontology_is_opt_in_and_off_by_default(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_ONTOLOGY", raising=False)
    monkeypatch.delenv("KUMIHO_MEMORY_ENTITY_PROMOTION", raising=False)
    # Default: entity promotion off, and entity recall off on the graph config.
    m = _build_manager(graph_augmentation=True)
    assert m.entity_promotion_config is None
    assert m.graph_augmentation_config.entity_recall is False


def test_ontology_switch_enables_write_and_read(monkeypatch):
    monkeypatch.setenv("KUMIHO_MEMORY_ONTOLOGY", "1")
    monkeypatch.delenv("KUMIHO_MEMORY_ENTITY_PROMOTION", raising=False)
    m = _build_manager(graph_augmentation=True)
    assert m.entity_promotion_config is not None  # write on
    assert m.graph_augmentation_config.entity_recall is True  # read on


def test_entity_promotion_env_forces_independently(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_ONTOLOGY", raising=False)
    # Force ON without the ontology switch.
    monkeypatch.setenv("KUMIHO_MEMORY_ENTITY_PROMOTION", "1")
    assert _build_manager().entity_promotion_config is not None
    # Force OFF wins even with ontology on.
    monkeypatch.setenv("KUMIHO_MEMORY_ONTOLOGY", "1")
    monkeypatch.setenv("KUMIHO_MEMORY_ENTITY_PROMOTION", "0")
    assert _build_manager().entity_promotion_config is None


def test_explicit_config_overrides_env(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_ONTOLOGY", raising=False)
    monkeypatch.delenv("KUMIHO_MEMORY_ENTITY_PROMOTION", raising=False)
    custom = EntityPromotionConfig(max_entities=2)
    assert _build_manager(entity_promotion=custom).entity_promotion_config is custom
    assert _build_manager(entity_promotion=False).entity_promotion_config is None


def test_real_sdk_exposes_methods_entity_promotion_calls():
    """Guards against the mocked-suite blind spot: if the real SDK renames
    or reshapes these, the feature silently no-ops in production. Assert the
    contract against the actual classes (no server needed)."""
    import kumiho
    from kumiho.project import Project
    from kumiho.item import Item
    from kumiho.revision import Revision

    assert hasattr(kumiho, "get_project")
    assert hasattr(kumiho, "get_revision")
    for method in ("create_space", "create_item", "get_item"):
        assert callable(getattr(Project, method, None)), f"Project.{method} missing"
    for method in ("get_latest_revision", "create_revision"):
        assert callable(getattr(Item, method, None)), f"Item.{method} missing"
    assert callable(getattr(Revision, "create_edge", None)), "Revision.create_edge missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
