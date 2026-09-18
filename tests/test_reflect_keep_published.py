# -*- coding: utf-8 -*-
"""Reflect is the one writer that opts into ``keep_published`` (KumihoIO/kumiho-SDKs#171).

A reflect capture is a correction the user authorized ("fix it, my favourite
colour is black"), and reflect passes each capture's classification tags
through to the store. When such a capture STACKS onto an item whose current
revision is ``published``, the SDK used to leave the tag where it was, and
recall — which resolves ``published`` before ``latest`` — kept returning the
value the user had just corrected. ``keep_published=True`` moves the tag onto
the stacked revision.

Everything else that writes memory — auto-memorize's background assessment,
consolidation, experience records, pattern candidates — keeps the default,
because nothing should publish on a writer's own initiative.

The keyword only exists from kumiho 0.13.2, and kumiho-memory's floor is
``kumiho>=0.10.7``, so reflect asks the resolved store callable whether it
takes it. These tests run against both shapes.
"""
import asyncio
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import kumiho_memory.mcp_tools as mcp_tools_module
from kumiho_memory.mcp_tools import (
    _accepts_keep_published,
    _keep_published_kwarg,
    tool_memory_add_response,
    tool_memory_consolidate,
    tool_memory_ingest,
    tool_memory_reflect,
)
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

import test_mcp_tools as MT
from fakes import FakeRedis

from kumiho.mcp_server import tool_memory_store as _installed_store

#: Whether the kumiho actually installed here is new enough to take the keyword.
SDK_HAS_KEEP_PUBLISHED = _accepts_keep_published(_installed_store)


CAPTURE = {
    "type": "preference",
    "title": "Favorite color is black",
    "content": "Correction: the user's favorite color is black, not blue.",
    "tags": ["preference", "color", "personal"],
}


def _reflect(capture=CAPTURE, *, captures=None, space_path="personal"):
    """One reflect call through whatever store is currently patched in."""
    ingest = tool_memory_ingest({"user_id": "user-keep-published", "message": "hi"})
    return tool_memory_reflect({
        "session_id": ingest["session_id"],
        "response": "Noted — black it is.",
        "space_path": space_path,
        "captures": captures if captures is not None else [capture],
        "discover_edges": False,
    })


# ---------------------------------------------------------------------------
# The signature probe
# ---------------------------------------------------------------------------


def test_the_probe_reads_the_three_store_shapes():
    def new_sdk(project, *, keep_published=False):
        pass

    def old_sdk(project, *, stack_revisions=True):
        pass

    def forwarding_wrapper(project, **kwargs):
        pass

    assert _accepts_keep_published(new_sdk) is True
    assert _accepts_keep_published(forwarding_wrapper) is True
    assert _accepts_keep_published(old_sdk) is False
    assert _keep_published_kwarg(new_sdk) == {"keep_published": True}
    assert _keep_published_kwarg(old_sdk) == {}


def test_an_unreadable_signature_is_treated_as_not_accepting():
    """Omitting the keyword costs one tag move; guessing costs the whole write."""
    class _Opaque:
        __signature__ = "not a signature"

        def __call__(self, **kwargs):
            return {}

    assert _accepts_keep_published(_Opaque()) is False
    assert _keep_published_kwarg(_Opaque()) == {}


# ---------------------------------------------------------------------------
# Reflect opts in — single capture and batch
# ---------------------------------------------------------------------------


def _new_sdk_store(calls):
    def store(*, keep_published=False, **kwargs):
        calls.append({"keep_published": keep_published, **kwargs})
        return {"revision_kref": f"kref://m/cap/{len(calls)}", "item_kref": "kref://m/item/1"}
    return store


def _new_sdk_batch(calls):
    def store_batch(captures, *, keep_published=False, **kwargs):
        calls.append({"captures": captures, "keep_published": keep_published, **kwargs})
        return {
            "results": [{"revision_kref": f"kref://m/batch/{i}"} for i in range(len(captures))],
            "stored_krefs": [f"kref://m/batch/{i}" for i in range(len(captures))],
            "stacked": 0,
        }
    return store_batch


def test_reflect_single_capture_opts_in():
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store", _new_sdk_store(calls)):
            result = _reflect()
        assert result["captures_stored"] == 1
        assert calls[0]["keep_published"] is True
        # Unchanged alongside it: the capture's own tags still go through, and
        # a routed capture still stacks (keep_published only matters if it does).
        assert calls[0]["tags"] == CAPTURE["tags"]
        assert calls[0]["stack_revisions"] is True
    finally:
        MT._cleanup_manager()


def test_reflect_batch_opts_in():
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store_batch", _new_sdk_batch(calls)), \
                patch("kumiho.batch_create_revisions", create=True):
            result = _reflect(captures=[CAPTURE, dict(CAPTURE, title="Prefers dark mode")])
        assert result["captures_stored"] == 2
        assert len(calls) == 1
        assert calls[0]["keep_published"] is True
        assert calls[0]["stack_revisions"] is True
    finally:
        MT._cleanup_manager()


def test_a_kwargs_forwarding_store_still_receives_it():
    """The suite's own fakes take ``**kwargs``; a forwarder forwards this too."""
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store", MT._fake_store_recorder(calls)):
            _reflect()
        assert calls[0]["keep_published"] is True
    finally:
        MT._cleanup_manager()


# ---------------------------------------------------------------------------
# An older SDK — the keyword is omitted, not forced
# ---------------------------------------------------------------------------


def _old_sdk_store(calls):
    """kumiho < 0.13.2: no ``keep_published``, and ``**kwargs`` would hide the point."""
    def store(*, project, space_path, memory_type, title, summary, assistant_text,
              source_revision_krefs, edge_type, tags, metadata, stack_revisions):
        calls.append({"tags": tags, "stack_revisions": stack_revisions})
        return {"revision_kref": f"kref://m/cap/{len(calls)}", "item_kref": "kref://m/item/1"}
    return store


def _old_sdk_batch(calls):
    def store_batch(captures, *, project, space_path, source_revision_krefs,
                    edge_type, stack_revisions, idempotency_prefix):
        calls.append({"captures": captures, "stack_revisions": stack_revisions})
        return {
            "results": [{"revision_kref": f"kref://m/batch/{i}"} for i in range(len(captures))],
            "stored_krefs": [f"kref://m/batch/{i}" for i in range(len(captures))],
            "stacked": 0,
        }
    return store_batch


def test_an_older_sdk_single_store_is_called_without_the_keyword():
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store", _old_sdk_store(calls)):
            result = _reflect()
        assert result["captures_stored"] == 1  # would be 0 on a TypeError
        assert calls[0]["tags"] == CAPTURE["tags"]
    finally:
        MT._cleanup_manager()


def test_an_older_sdk_batch_store_is_called_without_the_keyword():
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store_batch", _old_sdk_batch(calls)), \
                patch("kumiho.batch_create_revisions", create=True):
            result = _reflect(captures=[CAPTURE, dict(CAPTURE, title="Prefers dark mode")])
        assert result["captures_stored"] == 2
        assert len(calls) == 1
    finally:
        MT._cleanup_manager()


# ---------------------------------------------------------------------------
# Every automated writer keeps the default
# ---------------------------------------------------------------------------


def test_auto_memorize_does_not_opt_in(monkeypatch):
    """``_background_assess`` stores what an assessor decided, not what a user said."""
    from kumiho_memory.memory_manager import MemoryAssessResult
    from test_assessors_evidence import _run_background_assess

    stored, _ = _run_background_assess(
        MemoryAssessResult(
            should_store=True, content="X is corroborated", memory_type="fact",
            evidence_level="corroborated", source="news:bbc", supporting_krefs=[],
        ),
        {"item_kref": "kref://m/new", "revision_kref": "kref://m/new/rev/1"},
        monkeypatch,
    )
    assert "keep_published" not in stored


def test_consolidation_does_not_opt_in():
    fake = FakeRedis()
    stored = {}

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://m/item", "revision_kref": "kref://m/item?r=1"}

    async def retrieve_stub(**kwargs):
        return []

    mcp_tools_module._manager = UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=fake, redis_url="redis://test"),
        summarizer=MT.StubSummarizer(),
        pii_redactor=MT.StubRedactor(),
        memory_store=store_stub,
        memory_retrieve=retrieve_stub,
        consolidation_threshold=2,
        artifact_root=tempfile.mkdtemp(),
    )
    try:
        ingest = tool_memory_ingest({"user_id": "user-consolidate", "message": "I like tea."})
        tool_memory_add_response({"session_id": ingest["session_id"], "response": "Green tea."})
        result = tool_memory_consolidate({
            "session_id": ingest["session_id"],
            "summary": {"title": "Tea", "summary": "User likes green tea.",
                        "classification": {"topics": ["tea"]}},
        })
        assert result["success"] is True, result
        assert "keep_published" not in stored
    finally:
        MT._cleanup_manager()


def test_experience_records_do_not_opt_in():
    from kumiho_memory.experience import record_experience

    payloads = []

    def store(**payload):
        payloads.append(payload)
        return {"revision_kref": "kref://project/experiences/pilot.experience?r=1"}

    manager = SimpleNamespace(project="project", memory_store=store)
    asyncio.run(record_experience(manager, {
        "experience_id": "pilot-run-a", "title": "Try a pilot", "situation": "Small team",
        "goal": "Release on time", "decision": "Run a limited pilot",
        "rationale": "Bound upkeep", "expected_outcome": "Know support cost",
        "alternatives": ["Full rollout"], "applicability_conditions": ["Team stays small"],
    }))
    assert payloads and "keep_published" not in payloads[0]


def test_pattern_candidates_do_not_opt_in(monkeypatch):
    from kumiho_memory.insight_patterns import store_pattern_candidate
    import test_insight_patterns as TIP

    rows = [TIP.row(), TIP.row("beta")]
    req = TIP.request(rows)
    manager, payloads = TIP.manager(monkeypatch, rows)
    result = asyncio.run(store_pattern_candidate(manager, req, TIP.proposal(req), "/p/patterns"))
    assert result["status"] == "stored_proposal"
    assert payloads and "keep_published" not in payloads[0]


# ---------------------------------------------------------------------------
# End to end against the installed SDK's real store
# ---------------------------------------------------------------------------


class _Kref:
    def __init__(self, uri):
        self.uri = uri


class _Revision:
    """A revision under the server's two tag rules.

    A tag lives on one revision per item, so tagging MOVES it; and a published
    revision is frozen and rejects tags applied to it afterwards.
    """

    def __init__(self, item, number):
        self.item = item
        self.number = number
        self.kref = _Kref(f"{item.kref.uri}?r={number}")
        self.tags = {"latest"}
        self.metadata = {}

    @property
    def published(self):
        return "published" in self.tags

    def tag(self, name):
        if self.published:
            raise RuntimeError("PERMISSION_DENIED: a published revision is immutable")
        for other in self.item.revisions:
            other.tags.discard(name)
        self.tags.add(name)

    def create_artifact(self, name, location):
        return MagicMock()

    def create_edge(self, target, edge_type):
        return MagicMock()


class _Item:
    item_name = "favorite-color-3f9a"

    def __init__(self):
        self.kref = _Kref("kref://CognitiveMemory/personal/favorite-color-3f9a.conversation")
        self.revisions = []

    def create_revision(self, metadata=None):
        for prior in self.revisions:
            prior.tags.discard("latest")
        revision = _Revision(self, len(self.revisions) + 1)
        revision.metadata = dict(metadata or {})
        self.revisions.append(revision)
        return revision

    def get_revision_by_tag(self, tag):
        return next((r for r in self.revisions if tag in r.tags), None)


def _reflect_through_the_real_store(item, capture, *, stacks):
    """Reflect into the INSTALLED ``kumiho.mcp_server`` store, graph writes stubbed.

    The similarity gate itself is not under test, so ``stacks`` decides what it
    returns; everything downstream of it is the SDK's own code.
    """
    similar = (item, 0.81, 0.2, 0.3) if stacks else (None, 0.0, 0.0, 0.0)
    with patch("kumiho.mcp_server._ensure_configured", return_value=True), \
            patch("kumiho.mcp_server._get_project_cached", return_value=MagicMock()), \
            patch("kumiho.mcp_server._ensure_space_path",
                  return_value="/CognitiveMemory/personal"), \
            patch("kumiho.mcp_server._find_similar_item", return_value=similar), \
            patch("kumiho.mcp_server._get_or_create_item", return_value=item), \
            patch("kumiho.mcp_server._write_memory_artifact", return_value=""), \
            patch("kumiho.mcp_server._get_or_create_bundle", return_value=MagicMock()):
        return _reflect(capture)


def test_a_correction_lands_where_recall_reads_it():
    """The whole point, against whichever kumiho is installed.

    From 0.13.2 the stacked correction takes ``published`` from the revision it
    supersedes. Before it, the keyword is omitted — reflect still stores, and
    the tag stays put; that is the behaviour this change asks the SDK to fix.
    """
    try:
        MT._install_test_manager()
        item = _Item()
        _reflect_through_the_real_store(item, {
            "type": "preference", "title": "Favorite color is blue",
            "content": "The user's favorite color is blue.",
        }, stacks=False)
        blue = item.revisions[0]
        assert blue.published, "an untagged store publishes its new item's revision"

        result = _reflect_through_the_real_store(item, CAPTURE, stacks=True)
        black = item.revisions[1]
        assert result["captures_stored"] == 1
        assert result["stored_krefs"] == [black.kref.uri]
        assert black.tags >= set(CAPTURE["tags"])

        if SDK_HAS_KEEP_PUBLISHED:
            assert black.published
            assert not blue.published
            assert item.get_revision_by_tag("published") is black
        else:
            assert not black.published
            assert item.get_revision_by_tag("published") is blue
    finally:
        MT._cleanup_manager()


@pytest.mark.skipif(not SDK_HAS_KEEP_PUBLISHED, reason="requires kumiho >= 0.13.2")
def test_an_unpublished_chain_is_left_alone_by_the_opt_in():
    """Nothing is published on the writer's initiative — only a tag moved forward."""
    try:
        MT._install_test_manager()
        item = _Item()
        _reflect_through_the_real_store(item, dict(CAPTURE, tags=["draft"]), stacks=False)
        assert item.get_revision_by_tag("published") is None

        _reflect_through_the_real_store(item, dict(CAPTURE, tags=["draft"]), stacks=True)
        assert item.get_revision_by_tag("published") is None
        assert len(item.revisions) == 2
    finally:
        MT._cleanup_manager()


def test_the_stored_capture_metadata_is_untouched_by_the_opt_in():
    """One keyword, nothing else: the revision the correction wrote is unchanged."""
    try:
        MT._install_test_manager()
        item = _Item()
        _reflect_through_the_real_store(item, dict(CAPTURE, event_date="2026-09-18"),
                                        stacks=False)
        metadata = item.revisions[0].metadata
        assert metadata["memory_type"] == "preference"
        assert metadata["event_date"] == "2026-09-18"
        assert json.loads(json.dumps(metadata))  # plain, serialisable strings
    finally:
        MT._cleanup_manager()
