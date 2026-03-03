"""Tests for kumiho_memory.mcp_tools — MCP tool wrappers for memory operations."""

import asyncio
import json
import os
import tempfile
from unittest.mock import patch

from kumiho_memory.mcp_tools import (
    MEMORY_TOOLS,
    MEMORY_TOOL_HANDLERS,
    tool_chat_add,
    tool_chat_get,
    tool_chat_clear,
    tool_memory_ingest,
    tool_memory_add_response,
    tool_memory_consolidate,
    tool_memory_recall,
    tool_memory_store_execution,
    tool_memory_dream_state,
    _get_manager,
)
from kumiho_memory.mcp_tools import _manager as _initial_manager
import kumiho_memory.mcp_tools as mcp_tools_module

from kumiho_memory.redis_memory import RedisMemoryBuffer
from kumiho_memory.memory_manager import UniversalMemoryManager

from fakes import FakeRedis


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubSummarizer:
    async def summarize_conversation(self, messages, context=None):
        return {
            "type": "summary",
            "title": "Stub summary",
            "summary": "Stub summary text.",
            "classification": {"topics": ["stub"]},
        }

    async def generate_implications(self, messages, context=None):
        return []


class StubRedactor:
    def anonymize_summary(self, summary):
        return summary

    def reject_credentials(self, text):
        pass


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _install_test_manager(tmpdir=None):
    """Create and install a test manager with faked Redis."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://memory/test"}

    async def retrieve_stub(**kwargs):
        return {"revision_krefs": ["kref://memory/test/rev/1"]}

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=store_stub,
        memory_retrieve=retrieve_stub,
        consolidation_threshold=2,
        artifact_root=tmpdir or tempfile.mkdtemp(),
    )
    # Inject into the module singleton
    mcp_tools_module._manager = manager
    return manager, stored


def _cleanup_manager():
    mcp_tools_module._manager = None
    # Reset the recall deduplication timer so tests don't interfere
    mcp_tools_module._recall_cache_time = 0.0


# ---------------------------------------------------------------------------
# Tests — tool registry
# ---------------------------------------------------------------------------


def test_memory_tools_count():
    """Should have 10 tools registered."""
    assert len(MEMORY_TOOLS) == 10


def test_memory_tool_handlers_count():
    """Should have 10 handlers registered."""
    assert len(MEMORY_TOOL_HANDLERS) == 10


def test_all_tools_have_handlers():
    """Every tool in MEMORY_TOOLS must have a corresponding handler."""
    for tool in MEMORY_TOOLS:
        name = tool["name"]
        assert name in MEMORY_TOOL_HANDLERS, f"Missing handler for {name}"


def test_all_tools_have_input_schema():
    """Every tool must have a valid inputSchema."""
    for tool in MEMORY_TOOLS:
        assert "inputSchema" in tool, f"Missing inputSchema for {tool['name']}"
        schema = tool["inputSchema"]
        assert schema.get("type") == "object"


def test_tool_names_are_prefixed():
    """All tool names should start with kumiho_ prefix."""
    for tool in MEMORY_TOOLS:
        assert tool["name"].startswith("kumiho_"), f"Bad prefix: {tool['name']}"


# ---------------------------------------------------------------------------
# Tests — chat memory tools
# ---------------------------------------------------------------------------


def test_chat_add():
    """kumiho_chat_add should add a message to the Redis buffer."""
    try:
        _install_test_manager()
        result = tool_chat_add({
            "session_id": "test-session-1",
            "message": "Hello from MCP",
            "role": "user",
        })
        assert result["success"] is True
        assert result["message_count"] == 1
    finally:
        _cleanup_manager()


def test_chat_get():
    """kumiho_chat_get should retrieve messages."""
    try:
        _install_test_manager()
        # Add a message first
        tool_chat_add({
            "session_id": "test-session-2",
            "message": "Test message",
        })
        result = tool_chat_get({"session_id": "test-session-2"})
        assert result["message_count"] == 1
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Test message"
    finally:
        _cleanup_manager()


def test_chat_clear():
    """kumiho_chat_clear should remove all messages."""
    try:
        _install_test_manager()
        tool_chat_add({
            "session_id": "test-session-3",
            "message": "To be cleared",
        })
        result = tool_chat_clear({"session_id": "test-session-3"})
        assert result["success"] is True
        assert result["cleared_count"] == 1

        # Verify empty
        get_result = tool_chat_get({"session_id": "test-session-3"})
        assert get_result["message_count"] == 0
    finally:
        _cleanup_manager()


# ---------------------------------------------------------------------------
# Tests — memory lifecycle tools
# ---------------------------------------------------------------------------


def test_memory_ingest():
    """kumiho_memory_ingest should buffer message and return context."""
    try:
        _install_test_manager()
        result = tool_memory_ingest({
            "user_id": "user-mcp-1",
            "message": "Remember this preference",
            "context": "personal",
        })
        assert "session_id" in result
        assert "working_memory" in result
        assert "long_term_memory" in result
        assert isinstance(result["working_memory"], list)
    finally:
        _cleanup_manager()


def test_memory_add_response():
    """kumiho_memory_add_response should add assistant message."""
    try:
        _install_test_manager()
        # Ingest first to get a session_id
        ingest = tool_memory_ingest({
            "user_id": "user-mcp-2",
            "message": "Test question",
        })
        result = tool_memory_add_response({
            "session_id": ingest["session_id"],
            "response": "Test answer",
        })
        assert result["success"] is True
        assert result["message_count"] == 2
    finally:
        _cleanup_manager()


def test_memory_consolidate():
    """kumiho_memory_consolidate should summarize and store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _install_test_manager(tmpdir)
            # Need enough messages to consolidate (threshold=2)
            ingest = tool_memory_ingest({
                "user_id": "user-mcp-3",
                "message": "First message",
            })
            tool_memory_add_response({
                "session_id": ingest["session_id"],
                "response": "First response",
            })
            result = tool_memory_consolidate({
                "session_id": ingest["session_id"],
            })
            assert result["success"] is True
            assert "summary" in result
        finally:
            _cleanup_manager()


def test_memory_recall():
    """kumiho_memory_recall should return search results."""
    try:
        _install_test_manager()
        result = tool_memory_recall({
            "query": "user preferences",
            "limit": 3,
        })
        assert "results" in result
        assert "count" in result
        assert result["count"] == 1
        assert result["results"][0] == {"kref": "kref://memory/test/rev/1"}
    finally:
        _cleanup_manager()


def test_memory_recall_with_filters():
    """kumiho_memory_recall should forward space_paths and memory_types."""
    try:
        _install_test_manager()
        result = tool_memory_recall({
            "query": "ssh errors",
            "space_paths": ["CognitiveMemory/work"],
            "memory_types": ["error"],
        })
        assert result["count"] >= 0
    finally:
        _cleanup_manager()


def test_memory_recall_deduplication():
    """Duplicate recall calls within the dedup window should return empty."""
    try:
        _install_test_manager()
        # First call — executes normally
        result1 = tool_memory_recall({"query": "first query"})
        assert result1["count"] == 1

        # Second call within dedup window — should return empty with note
        result2 = tool_memory_recall({"query": "different query"})
        assert result2["count"] == 0
        assert result2["deduplicated"] is True
        assert "Duplicate recall" in result2["note"]
    finally:
        _cleanup_manager()


def test_memory_recall_dedup_expires():
    """After the dedup window expires, a fresh recall should execute."""
    import time as _time

    try:
        _install_test_manager()
        result1 = tool_memory_recall({"query": "query A"})

        # Artificially expire the cache by backdating the timestamp
        mcp_tools_module._recall_cache_time = _time.monotonic() - 10.0

        result2 = tool_memory_recall({"query": "query B"})
        # Should be a fresh call, not the cached object
        assert result2 is not result1
        assert result2["count"] == 1
    finally:
        _cleanup_manager()


# ---------------------------------------------------------------------------
# Tests — tool execution
# ---------------------------------------------------------------------------


def test_memory_store_execution():
    """kumiho_memory_store_execution should store tool result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            manager, stored = _install_test_manager(tmpdir)
            result = tool_memory_store_execution({
                "task": "git status",
                "status": "done",
                "exit_code": 0,
                "stdout": "On branch main",
                "tools": ["shell_exec"],
                "topics": ["git"],
            })
            assert result["success"] is True
            assert result["memory_type"] == "action"
            assert stored.get("memory_type") == "action"
        finally:
            _cleanup_manager()


def test_memory_store_execution_failure():
    """Failed execution should be stored as error type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _install_test_manager(tmpdir)
            result = tool_memory_store_execution({
                "task": "npm test",
                "status": "failed",
                "exit_code": 1,
                "stderr": "3 tests failed",
            })
            assert result["success"] is True
            assert result["memory_type"] == "error"
        finally:
            _cleanup_manager()


# ---------------------------------------------------------------------------
# Tests — dream state
# ---------------------------------------------------------------------------


def test_memory_dream_state_dry_run():
    """kumiho_memory_dream_state with dry_run should assess without mutating."""
    import sys
    import types

    # Create a minimal fake kumiho SDK for dream state
    fake_sdk = types.ModuleType("kumiho")

    class FakeKref:
        def __init__(self, uri):
            self.uri = uri

    class FakeItem:
        def __init__(self, kref_uri):
            self.kref = FakeKref(kref_uri)

    fake_sdk.get_item = lambda kref: FakeItem(kref)
    fake_sdk.get_project = lambda name: types.SimpleNamespace(
        get_space=lambda n: None,
        create_space=lambda n: types.SimpleNamespace(
            create_item=lambda name, kind: FakeItem(f"kref://{name}/{n}.{kind}")
        ),
    )
    fake_sdk.event_stream = lambda **kw: iter([])
    fake_sdk.get_attribute = lambda kref, key: None
    fake_sdk.set_attribute = lambda kref, key, val: None
    fake_sdk.batch_get_revisions = lambda **kw: ([], [])
    fake_sdk.get_client = lambda: None
    fake_sdk.Kref = FakeKref

    sys.modules["kumiho"] = fake_sdk
    try:
        # Patch MemorySummarizer so DreamState doesn't need a real API key
        with patch("kumiho_memory.dream_state.MemorySummarizer", return_value=StubSummarizer()):
            result = tool_memory_dream_state({
                "project": "CognitiveMemory",
                "dry_run": True,
            })
        assert result["success"] is True
        assert result["events_processed"] == 0
    finally:
        sys.modules.pop("kumiho", None)
