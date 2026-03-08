"""Tests for kumiho_memory.dream_state — DreamState consolidation processor."""

import asyncio
import json
import sys
import tempfile
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from kumiho_memory.dream_state import (
    DreamState,
    DreamStateStats,
    MemoryAssessment,
    _parse_assessments,
)


# ---------------------------------------------------------------------------
# Stubs — lightweight fakes for the kumiho SDK objects
# ---------------------------------------------------------------------------


@dataclass
class FakeKref:
    uri: str


@dataclass
class FakeRevision:
    kref: FakeKref
    item_kref: FakeKref
    metadata: Dict[str, str] = field(default_factory=dict)
    deprecated: bool = False
    created_at: str = "2026-03-08T12:00:00+00:00"


class FakeItem:
    def __init__(self, kref_uri: str, *, has_get_members: bool = False):
        self.kref = FakeKref(kref_uri)
        self._members: list = []
        self._revisions: list = []
        self._latest_rev: Optional[FakeRevision] = None
        self._has_get_members = has_get_members

    def create_revision(self, metadata: dict):
        rev = FakeRevision(
            kref=FakeKref(f"{self.kref.uri}?r={len(self._revisions) + 1}"),
            item_kref=self.kref,
            metadata=metadata,
        )
        self._revisions.append(rev)
        return _RevisionHandle(rev)

    def get_revision_by_tag(self, tag: str):
        if tag == "latest" and self._latest_rev is not None:
            return self._latest_rev
        if self._revisions:
            return self._revisions[-1]
        return None

    def get_revisions(self):
        return list(self._revisions)

    def get_members(self):
        return self._members


class _RevisionHandle:
    def __init__(self, rev: FakeRevision):
        self.kref = rev.kref
        self._artifacts: list = []

    def create_artifact(self, name: str, path: str):
        self._artifacts.append((name, path))


class FakeSpace:
    def __init__(self, path: str):
        self.path = path


class FakeClient:
    """Tracks all mutation calls for assertion."""

    def __init__(self):
        self.deprecated: List[str] = []
        self.tags: List[tuple] = []
        self.metadata_updates: List[tuple] = []
        self.edges: List[tuple] = []
        self._published_krefs: set = set()
        self._items_by_space: Dict[str, List[FakeItem]] = {}

    def set_deprecated(self, kref, value):
        self.deprecated.append((kref.uri if hasattr(kref, "uri") else str(kref), value))

    def tag_revision(self, kref, tag):
        self.tags.append((kref.uri if hasattr(kref, "uri") else str(kref), tag))

    def update_revision_metadata(self, kref, updates):
        self.metadata_updates.append(
            (kref.uri if hasattr(kref, "uri") else str(kref), updates)
        )

    def create_edge(self, source, target, edge_type):
        src = source.kref.uri if hasattr(source, "kref") else str(source)
        tgt = target.kref.uri if hasattr(target, "kref") else str(target)
        self.edges.append((src, tgt, edge_type))

    def has_tag(self, kref, tag):
        uri = kref.uri if hasattr(kref, "uri") else str(kref)
        return uri in self._published_krefs and tag == "published"

    def get_items(self, parent_path="", kind_filter="", include_deprecated=False):
        return self._items_by_space.get(parent_path, [])


def _build_fake_sdk(
    *,
    revisions: Optional[List[FakeRevision]] = None,
    items: Optional[Dict[str, FakeItem]] = None,
    attributes: Optional[Dict[str, Dict[str, str]]] = None,
    client: Optional[FakeClient] = None,
    spaces: Optional[List[FakeSpace]] = None,
    items_by_space: Optional[Dict[str, List[FakeItem]]] = None,
):
    """Create a fake ``kumiho`` module that mimics the real SDK."""
    revisions = revisions or []
    items = items or {}
    attributes = attributes if attributes is not None else {}
    client = client or FakeClient()
    spaces = spaces or [FakeSpace("/CognitiveMemory/personal")]

    if items_by_space:
        client._items_by_space = items_by_space

    sdk = types.ModuleType("kumiho")

    def get_item(kref_uri):
        return items.get(kref_uri)

    def get_project(name):
        proj = types.SimpleNamespace()
        proj.name = name
        proj.get_space = lambda n: types.SimpleNamespace(
            create_item=lambda name, kind: _ensure_item(items, f"kref://{name}/{n}.{kind}")
        )
        proj.create_space = lambda n: types.SimpleNamespace(
            create_item=lambda name, kind: _ensure_item(items, f"kref://{name}/{n}.{kind}")
        )
        proj.get_spaces = lambda recursive=False: list(spaces)
        return proj

    def get_attribute(kref, key):
        return attributes.get(kref, {}).get(key)

    def set_attribute(kref, key, value):
        attributes.setdefault(kref, {})[key] = value

    def get_revision(kref_str):
        for rev in revisions:
            if rev.kref.uri == kref_str:
                return rev
        # Return a simple stub so create_edge doesn't fail
        return types.SimpleNamespace(kref=FakeKref(kref_str))

    def get_client_fn():
        return client

    sdk.get_item = get_item
    sdk.get_project = get_project
    sdk.get_attribute = get_attribute
    sdk.set_attribute = set_attribute
    sdk.get_revision = get_revision
    sdk.get_client = get_client_fn
    sdk.Kref = FakeKref

    return sdk, client, attributes


def _ensure_item(items: dict, kref_uri: str) -> FakeItem:
    if kref_uri not in items:
        items[kref_uri] = FakeItem(kref_uri)
    return items[kref_uri]


class StubSummarizer:
    """A MemorySummarizer stand-in with a controllable adapter."""

    def __init__(self, response: str = "[]"):
        self.adapter = StubAdapter(response)
        self.model = "stub-model"


class StubAdapter:
    """LLMAdapter stand-in that returns pre-configured JSON."""

    def __init__(self, response: str = "[]"):
        self._response = response

    async def chat(self, *, messages, model, system, max_tokens, json_mode=False):
        return self._response


# ---------------------------------------------------------------------------
# Helper to inject the fake SDK
# ---------------------------------------------------------------------------


def _make_dream_state(sdk_module, summarizer=None, **kwargs):
    """Build a DreamState and monkey-patch its kumiho import."""
    ds = DreamState(
        summarizer=summarizer or StubSummarizer(),
        **kwargs,
    )
    # Patch `import kumiho` inside run() by injecting into sys.modules
    sys.modules["kumiho"] = sdk_module
    return ds


def _cleanup_sdk():
    sys.modules.pop("kumiho", None)


def _make_item_with_revision(
    item_kref: str,
    rev_kref: str,
    metadata: Dict[str, str],
    created_at: str = "2026-03-08T12:00:00+00:00",
    deprecated: bool = False,
) -> tuple:
    """Create a FakeItem with a latest revision pre-attached."""
    item = FakeItem(item_kref)
    rev = FakeRevision(
        kref=FakeKref(rev_kref),
        item_kref=FakeKref(item_kref),
        metadata=metadata,
        deprecated=deprecated,
        created_at=created_at,
    )
    item._latest_rev = rev
    return item, rev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_no_revisions():
    """No revisions since last run → report with zeros."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}
    # No items in any space
    sdk, client, attrs = _build_fake_sdk(
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={},
    )

    async def run():
        ds = _make_dream_state(sdk)
        ds._cursor_item_kref = cursor_item.kref.uri
        try:
            report = await ds.run()
            assert report["success"] is True
            assert report["events_processed"] == 0
            assert report["revisions_assessed"] == 0
            assert report["deprecated"] == 0
        finally:
            _cleanup_sdk()

    asyncio.run(run())


def test_run_processes_revisions():
    """Revisions found → assess → apply full pipeline."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/mem1.conversation",
        "kref://CognitiveMemory/personal/mem1.conversation?r=1",
        {"title": "User preference", "summary": "User likes dark mode"},
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.8,
        "should_deprecate": False,
        "deprecation_reason": "",
        "suggested_tags": ["preference", "ui"],
        "metadata_updates": {"topic": "dark-mode"},
        "related_indices": [],
        "relationship_type": "",
    }])

    sdk, client, attrs = _build_fake_sdk(
        revisions=[rev],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": [mem_item]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                assert report["events_processed"] == 1
                assert report["revisions_assessed"] == 1
                assert report["deprecated"] == 0
                assert report["tags_added"] == 2
                assert report["metadata_updated"] == 1
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_load_last_run_at_first_run():
    """Returns None when no timestamp exists."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}
    sdk, _, attrs = _build_fake_sdk(items=items)

    ds = _make_dream_state(sdk)
    try:
        result = ds._load_last_run_at(sdk, cursor_item.kref.uri)
        assert result is None
    finally:
        _cleanup_sdk()


def test_save_and_load_last_run_at():
    """Round-trip timestamp persistence."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}
    sdk, _, attrs = _build_fake_sdk(items=items)

    ds = _make_dream_state(sdk)
    try:
        assert ds._load_last_run_at(sdk, cursor_item.kref.uri) is None

        ds._save_last_run_at(sdk, cursor_item.kref.uri, "2026-03-08T10:00:00+00:00")
        loaded = ds._load_last_run_at(sdk, cursor_item.kref.uri)
        assert loaded == "2026-03-08T10:00:00+00:00"
    finally:
        _cleanup_sdk()


def test_revision_time_filter():
    """Only revisions after last_run_at are collected."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    # Old revision — should be skipped
    old_item, old_rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/old.conversation",
        "kref://CognitiveMemory/personal/old.conversation?r=1",
        {"title": "Old memory", "summary": "Before cutoff"},
        created_at="2026-03-07T08:00:00+00:00",
    )

    # New revision — should be collected
    new_item, new_rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/new.conversation",
        "kref://CognitiveMemory/personal/new.conversation?r=1",
        {"title": "New memory", "summary": "After cutoff"},
        created_at="2026-03-08T14:00:00+00:00",
    )

    sdk, client, attrs = _build_fake_sdk(
        revisions=[old_rev, new_rev],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": [old_item, new_item]},
        # Set last_run_at so old items are filtered out
        attributes={
            cursor_item.kref.uri: {
                "last_run_at": "2026-03-08T10:00:00+00:00",
            }
        },
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.8,
        "should_deprecate": False,
        "deprecation_reason": "",
        "suggested_tags": ["new"],
        "metadata_updates": {},
        "related_indices": [],
        "relationship_type": "",
    }])

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                # Only the new revision should be processed
                assert report["events_processed"] == 1
                assert report["revisions_assessed"] == 1
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_assess_deprecation():
    """LLM recommends deprecation → set_deprecated called."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/old.conversation",
        "kref://CognitiveMemory/personal/old.conversation?r=1",
        {"title": "Old info", "summary": "Outdated data"},
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.1,
        "should_deprecate": True,
        "deprecation_reason": "outdated information",
        "suggested_tags": [],
        "metadata_updates": {},
        "related_indices": [],
        "relationship_type": "",
    }])

    sdk, fake_client, attrs = _build_fake_sdk(
        revisions=[rev],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": [mem_item]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                assert report["deprecated"] == 1
                assert len(fake_client.deprecated) == 1
                assert fake_client.deprecated[0][1] is True
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_assess_tag_updates():
    """LLM recommends tags → tag_revision called."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/work/task.conversation",
        "kref://CognitiveMemory/work/task.conversation?r=1",
        {"title": "CI pipeline", "summary": "Setup GitHub Actions"},
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.9,
        "should_deprecate": False,
        "deprecation_reason": "",
        "suggested_tags": ["ci-cd", "github-actions", "devops"],
        "metadata_updates": {},
        "related_indices": [],
        "relationship_type": "",
    }])

    sdk, fake_client, _ = _build_fake_sdk(
        revisions=[rev],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/work")],
        items_by_space={"/CognitiveMemory/work": [mem_item]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["tags_added"] == 3
                tag_names = [t[1] for t in fake_client.tags]
                assert "ci-cd" in tag_names
                assert "github-actions" in tag_names
                assert "devops" in tag_names
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_assess_relationships():
    """LLM recommends edges → create_edge called."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    kref_a = "kref://CognitiveMemory/work/a.conversation"
    kref_b = "kref://CognitiveMemory/work/b.conversation"

    item_a, rev_a = _make_item_with_revision(
        kref_a, f"{kref_a}?r=1",
        {"title": "Deploy v1", "summary": "First deploy"},
    )
    item_b, rev_b = _make_item_with_revision(
        kref_b, f"{kref_b}?r=1",
        {"title": "Deploy v2", "summary": "Second deploy"},
    )

    llm_response = json.dumps([
        {
            "index": 0,
            "relevance_score": 0.8,
            "should_deprecate": False,
            "deprecation_reason": "",
            "suggested_tags": [],
            "metadata_updates": {},
            "related_indices": [1],
            "relationship_type": "DERIVED_FROM",
        },
        {
            "index": 1,
            "relevance_score": 0.9,
            "should_deprecate": False,
            "deprecation_reason": "",
            "suggested_tags": [],
            "metadata_updates": {},
            "related_indices": [],
            "relationship_type": "",
        },
    ])

    sdk, fake_client, _ = _build_fake_sdk(
        revisions=[rev_a, rev_b],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/work")],
        items_by_space={"/CognitiveMemory/work": [item_a, item_b]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["edges_created"] == 1
                assert len(fake_client.edges) == 1
                _, _, edge_type = fake_client.edges[0]
                assert edge_type == "DERIVED_FROM"
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_dry_run_no_mutations():
    """dry_run=True → no SDK mutation calls."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/dry.conversation",
        "kref://CognitiveMemory/personal/dry.conversation?r=1",
        {"title": "Dry run test", "summary": "Should not mutate"},
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.1,
        "should_deprecate": True,
        "deprecation_reason": "test deprecation",
        "suggested_tags": ["test-tag"],
        "metadata_updates": {"key": "val"},
        "related_indices": [],
        "relationship_type": "",
    }])

    sdk, fake_client, attrs = _build_fake_sdk(
        revisions=[rev],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": [mem_item]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                dry_run=True,
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                assert report["revisions_assessed"] == 1
                # No mutations should have occurred
                assert len(fake_client.deprecated) == 0
                assert len(fake_client.tags) == 0
                assert len(fake_client.metadata_updates) == 0
                assert len(fake_client.edges) == 0
                # Stats should still show 0 since apply was skipped
                assert report["deprecated"] == 0
                assert report["tags_added"] == 0
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_published_never_deprecated():
    """Published items skipped even if LLM says deprecate."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    kref_str = "kref://CognitiveMemory/personal/pub.conversation?r=1"
    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/pub.conversation",
        kref_str,
        {"title": "Published doc", "summary": "Important published data"},
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.1,
        "should_deprecate": True,
        "deprecation_reason": "LLM says deprecate",
        "suggested_tags": [],
        "metadata_updates": {},
        "related_indices": [],
        "relationship_type": "",
    }])

    fake_client = FakeClient()
    # Mark the kref as having the "published" tag
    fake_client._published_krefs.add(kref_str)

    sdk, _, attrs = _build_fake_sdk(
        revisions=[rev],
        items=items,
        client=fake_client,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": [mem_item]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                # Should NOT have deprecated the published item
                assert report["deprecated"] == 0
                assert len(fake_client.deprecated) == 0
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_max_deprecation_guard():
    """Max 50% deprecation circuit breaker."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    # Create 4 items, LLM says deprecate ALL of them
    mem_items = []
    revisions = []
    llm_items = []
    for i in range(4):
        kref_base = f"kref://CognitiveMemory/personal/item{i}.conversation"
        kref_rev = f"{kref_base}?r=1"
        item, rev = _make_item_with_revision(
            kref_base, kref_rev,
            {"title": f"Item {i}", "summary": f"Content {i}"},
        )
        mem_items.append(item)
        revisions.append(rev)
        llm_items.append({
            "index": i,
            "relevance_score": 0.1,
            "should_deprecate": True,
            "deprecation_reason": "not useful",
            "suggested_tags": [],
            "metadata_updates": {},
            "related_indices": [],
            "relationship_type": "",
        })

    llm_response = json.dumps(llm_items)
    sdk, fake_client, _ = _build_fake_sdk(
        revisions=revisions,
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": mem_items},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                # 4 items assessed, max 50% = 2 can be deprecated
                assert report["deprecated"] == 2
                assert len(fake_client.deprecated) == 2
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_report_generation():
    """Report revision + artifact created."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/work/rep.conversation",
        "kref://CognitiveMemory/work/rep.conversation?r=1",
        {"title": "Report test", "summary": "Testing report gen"},
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.7,
        "should_deprecate": False,
        "deprecation_reason": "",
        "suggested_tags": ["testing"],
        "metadata_updates": {},
        "related_indices": [],
        "relationship_type": "",
    }])

    sdk, _, _ = _build_fake_sdk(
        revisions=[rev],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/work")],
        items_by_space={"/CognitiveMemory/work": [mem_item]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                # A report_kref should be returned
                assert "report_kref" in report
                assert report["report_kref"] is not None
                # A revision should have been created on the cursor item
                assert len(cursor_item._revisions) == 1
                rev_meta = cursor_item._revisions[0].metadata
                assert rev_meta["type"] == "dream_state_report"
                assert int(rev_meta["tags_added"]) == 1
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_stacked_revision_detected():
    """A new revision on an existing item (stacked) is picked up."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    # Item already existed with r=1, now has a new stacked revision r=2
    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/existing.conversation",
        "kref://CognitiveMemory/personal/existing.conversation?r=2",
        {"title": "Updated preference", "summary": "User changed to light mode"},
        created_at="2026-03-08T15:00:00+00:00",
    )

    llm_response = json.dumps([{
        "index": 0,
        "relevance_score": 0.9,
        "should_deprecate": False,
        "deprecation_reason": "",
        "suggested_tags": ["preference-updated"],
        "metadata_updates": {},
        "related_indices": [],
        "relationship_type": "",
    }])

    sdk, client, attrs = _build_fake_sdk(
        revisions=[rev],
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": [mem_item]},
        # Last run was before this revision
        attributes={
            cursor_item.kref.uri: {
                "last_run_at": "2026-03-08T10:00:00+00:00",
            }
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(
                sdk,
                summarizer=StubSummarizer(llm_response),
                artifact_root=tmpdir,
            )
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                assert report["events_processed"] == 1
                assert report["revisions_assessed"] == 1
                assert report["tags_added"] == 1
            finally:
                _cleanup_sdk()

        asyncio.run(run())


def test_deprecated_revisions_skipped():
    """Already-deprecated revisions should not be collected."""
    cursor_item = FakeItem("kref://CognitiveMemory/_dream_state.conversation")
    items = {cursor_item.kref.uri: cursor_item}

    mem_item, rev = _make_item_with_revision(
        "kref://CognitiveMemory/personal/dep.conversation",
        "kref://CognitiveMemory/personal/dep.conversation?r=1",
        {"title": "Deprecated", "summary": "Already deprecated"},
        deprecated=True,
    )

    sdk, client, attrs = _build_fake_sdk(
        items=items,
        spaces=[FakeSpace("/CognitiveMemory/personal")],
        items_by_space={"/CognitiveMemory/personal": [mem_item]},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        async def run():
            ds = _make_dream_state(sdk, artifact_root=tmpdir)
            ds._cursor_item_kref = cursor_item.kref.uri
            try:
                report = await ds.run()
                assert report["success"] is True
                assert report["events_processed"] == 0
            finally:
                _cleanup_sdk()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_parse_assessments_valid_json():
    """Direct JSON array should parse correctly."""
    raw = json.dumps([{"index": 0, "relevance_score": 0.5}])
    result = _parse_assessments(raw)
    assert len(result) == 1
    assert result[0]["index"] == 0


def test_parse_assessments_markdown_fenced():
    """JSON inside markdown code fences should be extracted."""
    raw = '```json\n[{"index": 0, "relevance_score": 0.9}]\n```'
    result = _parse_assessments(raw)
    assert len(result) == 1


def test_parse_assessments_invalid_returns_empty():
    """Unparseable text should return an empty list."""
    result = _parse_assessments("This is not JSON at all.")
    assert result == []
