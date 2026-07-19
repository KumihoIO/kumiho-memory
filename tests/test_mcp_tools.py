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
    tool_memory_engage,
    tool_memory_reflect,
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


class ErrorSummarizer:
    async def summarize_conversation(self, messages, context=None):
        return {
            "type": "summary",
            "title": "Conversation summary",
            "summary": "I installed 0.4.5 and restarted the setup too.",
            "events": [],
            "implications": [],
            "knowledge": {"facts": [], "decisions": [], "actions": [], "open_questions": []},
            "classification": {"topics": [], "entities": []},
            "error": "The api_key client option must be set",
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
    # Reset the recall deduplication cache so tests don't interfere
    mcp_tools_module._recall_recent.clear()


# ---------------------------------------------------------------------------
# Tests — tool registry
# ---------------------------------------------------------------------------


def test_memory_tools_count():
    """Should have 14 tools registered (10 base + engage + reflect + space_profile + decompose)."""
    assert len(MEMORY_TOOLS) == 14


def test_memory_tool_handlers_count():
    """Should have 14 handlers registered."""
    assert len(MEMORY_TOOL_HANDLERS) == 14


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


def test_memory_consolidate_returns_summary_error_instead_of_storing_fallback():
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def store_stub(**kwargs):
        raise AssertionError("memory_store should not be called on summary failure")

    async def retrieve_stub(**kwargs):
        return []

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=ErrorSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=store_stub,
        memory_retrieve=retrieve_stub,
        consolidation_threshold=2,
        artifact_root=tempfile.mkdtemp(),
    )
    mcp_tools_module._manager = manager

    try:
        ingest = tool_memory_ingest({
            "user_id": "user-mcp-error",
            "message": "I installed 0.4.5 and restarted the setup too.",
        })
        tool_memory_add_response({
            "session_id": ingest["session_id"],
            "response": "Understood.",
        })
        result = tool_memory_consolidate({
            "session_id": ingest["session_id"],
        })
        assert result["success"] is False
        assert "Conversation summarization failed" in result["error"]
        assert "api_key client option must be set" in result["error"]
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
    """An IDENTICAL recall query within the dedup window returns empty."""
    try:
        _install_test_manager()
        # First call — executes normally
        result1 = tool_memory_recall({"query": "same query"})
        assert result1["count"] == 1

        # Same query again within the window — deduplicated
        result2 = tool_memory_recall({"query": "same query"})
        assert result2["count"] == 0
        assert result2["deduplicated"] is True
        assert "Duplicate recall" in result2["note"]
    finally:
        _cleanup_manager()


def test_memory_recall_distinct_queries_not_deduped():
    """DISTINCT queries within the window both execute — the dedup keys off the
    query, not a single global timestamp (regression guard for the singleton-
    lock bug that starved concurrent distinct recalls)."""
    try:
        _install_test_manager()
        r1 = tool_memory_recall({"query": "query A"})
        r2 = tool_memory_recall({"query": "query B"})
        r3 = tool_memory_recall({"query": "query A", "space_paths": ["s/x"]})  # different scope
        assert r1["count"] == 1
        assert r2.get("deduplicated") is not True and r2["count"] == 1
        assert r3.get("deduplicated") is not True and r3["count"] == 1
    finally:
        _cleanup_manager()


def test_memory_recall_dedup_expires():
    """After the dedup window expires, the same query executes again."""
    import time as _time

    try:
        _install_test_manager()
        result1 = tool_memory_recall({"query": "query A"})

        # Backdate every recorded signature so the window has elapsed.
        for sig in list(mcp_tools_module._recall_recent):
            mcp_tools_module._recall_recent[sig] = _time.monotonic() - 10.0

        result2 = tool_memory_recall({"query": "query A"})
        # Should be a fresh call, not deduplicated
        assert result2 is not result1
        assert result2["count"] == 1
    finally:
        _cleanup_manager()


# ---------------------------------------------------------------------------
# Tests — composite tools (engage / reflect)
# ---------------------------------------------------------------------------


def test_memory_engage_returns_context_and_krefs():
    """kumiho_memory_engage should return context, results, and source_krefs."""
    try:
        _install_test_manager()
        result = tool_memory_engage({
            "query": "user preferences",
            "limit": 3,
        })
        assert "context" in result
        assert "results" in result
        assert "source_krefs" in result
        assert result["count"] == 1
        assert result["source_krefs"] == ["kref://memory/test/rev/1"]
    finally:
        _cleanup_manager()


def test_memory_engage_exposes_approx_tokens():
    """engage should surface an additive approx_tokens (chars/4) budget field."""
    from kumiho_memory.context_compose import approx_tokens

    try:
        _install_test_manager()
        result = tool_memory_engage({"query": "user preferences", "limit": 3})
        assert "approx_tokens" in result
        assert result["approx_tokens"] == approx_tokens(result["context"])
    finally:
        _cleanup_manager()


def test_memory_engage_filters_by_min_score():
    """kumiho_memory_engage should drop low-scoring memories before context."""
    try:
        manager, _ = _install_test_manager()

        async def recall_stub(*args, **kwargs):
            return [
                {
                    "kref": "kref://memory/test/low",
                    "summary": "low relevance",
                    "score": 0.2,
                },
                {
                    "kref": "kref://memory/test/high",
                    "summary": "high relevance",
                    "score": 0.9,
                },
            ]

        manager.recall_memories = recall_stub
        result = tool_memory_engage({
            "query": "user preferences",
            "limit": 3,
            "min_score": 0.7,
        })

        assert result["count"] == 1
        assert result["source_krefs"] == ["kref://memory/test/high"]
        assert result["results"][0]["summary"] == "high relevance"
        assert "high relevance" in result["context"]
        assert "low relevance" not in result["context"]
    finally:
        _cleanup_manager()


def test_memory_recall_filters_by_min_score():
    """kumiho_memory_recall should expose the same min_score filter."""
    try:
        manager, _ = _install_test_manager()

        async def recall_stub(*args, **kwargs):
            return [
                {"kref": "kref://memory/test/low", "score": 0.2},
                {"kref": "kref://memory/test/high", "score": 0.9},
            ]

        manager.recall_memories = recall_stub
        result = tool_memory_recall({
            "query": "user preferences",
            "limit": 3,
            "min_score": 0.7,
        })

        assert result["count"] == 1
        assert result["results"][0]["kref"] == "kref://memory/test/high"
    finally:
        _cleanup_manager()


def test_memory_engage_deduplication():
    """Engage dedups an identical repeated query within the window."""
    try:
        _install_test_manager()
        result1 = tool_memory_engage({"query": "same"})
        assert result1["count"] == 1

        result2 = tool_memory_engage({"query": "same"})
        assert result2["count"] == 0
        assert result2["deduplicated"] is True
    finally:
        _cleanup_manager()


def test_memory_engage_and_recall_share_dedup():
    """Engage and recall share the dedup cache: the same query across the two
    tools is deduped, while a distinct query still executes."""
    try:
        _install_test_manager()
        result1 = tool_memory_engage({"query": "shared query"})
        assert result1["count"] == 1

        # Same query via recall — deduped (shared cache)
        result2 = tool_memory_recall({"query": "shared query"})
        assert result2["count"] == 0
        assert result2["deduplicated"] is True

        # A distinct query still executes
        result3 = tool_memory_recall({"query": "other query"})
        assert result3.get("deduplicated") is not True and result3["count"] == 1
    finally:
        _cleanup_manager()


def test_memory_reflect_buffers_response():
    """kumiho_memory_reflect without captures should buffer response only."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({
            "user_id": "user-reflect-1",
            "message": "Test question",
        })
        result = tool_memory_reflect({
            "session_id": ingest["session_id"],
            "response": "Here is my answer.",
        })
        assert result["buffered"] is True
        assert result["captures_stored"] == 0
        assert result["stored_krefs"] == []
    finally:
        _cleanup_manager()


def test_memory_reflect_with_captures():
    """kumiho_memory_reflect with captures should store them."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({
            "user_id": "user-reflect-2",
            "message": "I prefer dark mode",
        })

        # Mock tool_memory_store to avoid needing the full kumiho SDK
        store_calls = []

        def fake_store(**kwargs):
            store_calls.append(kwargs)
            return {
                "revision_kref": f"kref://memory/cap/{len(store_calls)}",
                "item_kref": "kref://memory/item/1",
            }

        with patch("kumiho_memory.mcp_tools.tool_memory_reflect.__module__", "kumiho_memory.mcp_tools"):
            with patch("kumiho.mcp_server.tool_memory_store", fake_store):
                result = tool_memory_reflect({
                    "session_id": ingest["session_id"],
                    "response": "Noted, dark mode it is.",
                    "captures": [
                        {
                            "type": "preference",
                            "title": "Prefers dark mode on Mar 27",
                            "content": "User prefers dark mode for all interfaces.",
                        },
                    ],
                    "source_krefs": ["kref://memory/test/rev/1"],
                })
        assert result["buffered"] is True
        assert result["captures_stored"] == 1
        assert len(result["stored_krefs"]) == 1
        assert len(store_calls) == 1
        assert store_calls[0]["memory_type"] == "preference"
        assert store_calls[0]["source_revision_krefs"] == ["kref://memory/test/rev/1"]
    finally:
        _cleanup_manager()


def _fake_store_recorder(store_calls):
    def fake_store(**kwargs):
        store_calls.append(kwargs)
        return {
            "revision_kref": f"kref://memory/cap/{len(store_calls)}",
            "item_kref": "kref://memory/item/1",
        }
    return fake_store


def _fake_batch_recorder(batch_calls):
    def fake_batch(captures, **kwargs):
        batch_calls.append({"captures": captures, **kwargs})
        return {
            "results": [
                {"revision_kref": f"kref://memory/batch/{i}",
                 "item_kref": "kref://memory/item/b"}
                for i in range(len(captures))
            ],
            "stored_krefs": [f"kref://memory/batch/{i}" for i in range(len(captures))],
            "stacked": 0,
        }
    return fake_batch


def _effective_metadata(store_calls, batch_calls):
    """Per-capture metadata that reached the write layer, regardless of route
    (the per-capture ``tool_memory_store`` loop vs one ``tool_memory_store_batch``)."""
    if batch_calls:
        return [c.get("metadata") for c in batch_calls[0]["captures"]]
    return [k.get("metadata") for k in store_calls]


def test_memory_reflect_capture_stamps_valid_event_date():
    """A valid ISO event_date (full or partial) is passed into revision metadata."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({
            "user_id": "user-reflect-ed",
            "message": "We shipped bge-m3 back in March.",
        })
        store_calls, batch_calls = [], []
        with patch("kumiho.mcp_server.tool_memory_store", _fake_store_recorder(store_calls)), \
                patch("kumiho.mcp_server.tool_memory_store_batch", _fake_batch_recorder(batch_calls)):
            result = tool_memory_reflect({
                "session_id": ingest["session_id"],
                "response": "Noted.",
                "captures": [
                    {
                        "type": "fact",
                        "title": "Shipped bge-m3 in March 2026",
                        "content": "bge-m3 embedding backend shipped.",
                        "event_date": "2026-03-14",
                    },
                    {
                        "type": "fact",
                        "title": "Founded in 2019",
                        "content": "The project was founded.",
                        "event_date": "2019",  # year-only is valid
                    },
                ],
            })
        assert result["captures_stored"] == 2
        assert "dropped_event_dates" not in result
        # Valid dates reach the write layer regardless of route (>=2 -> batch).
        meta = _effective_metadata(store_calls, batch_calls)
        assert meta[0] == {"event_date": "2026-03-14"}
        assert meta[1] == {"event_date": "2019"}
    finally:
        _cleanup_manager()


def test_memory_reflect_capture_drops_invalid_event_date():
    """A malformed/relative event_date is dropped and reported; capture still stored."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({
            "user_id": "user-reflect-ed2",
            "message": "Something happened last Tuesday.",
        })
        store_calls, batch_calls = [], []
        with patch("kumiho.mcp_server.tool_memory_store", _fake_store_recorder(store_calls)), \
                patch("kumiho.mcp_server.tool_memory_store_batch", _fake_batch_recorder(batch_calls)):
            result = tool_memory_reflect({
                "session_id": ingest["session_id"],
                "response": "Noted.",
                "captures": [
                    {
                        "type": "fact",
                        "title": "Relative date capture",
                        "content": "Happened at some point.",
                        "event_date": "last Tuesday",
                    },
                    {
                        "type": "fact",
                        "title": "Wrong separator capture",
                        "content": "Also happened.",
                        "event_date": "2026/03/14",
                    },
                ],
            })
        # Both captures are still stored — reflect never fails over a bad date.
        assert result["captures_stored"] == 2
        meta = _effective_metadata(store_calls, batch_calls)
        assert meta[0] is None
        assert meta[1] is None
        # ...but the drops are reported for the caller.
        dropped = result.get("dropped_event_dates")
        assert dropped is not None and len(dropped) == 2
        assert dropped[0]["event_date"] == "last Tuesday"
        assert dropped[1]["event_date"] == "2026/03/14"
    finally:
        _cleanup_manager()


def test_memory_reflect_capture_without_event_date_unchanged():
    """A capture with no event_date behaves exactly as before (metadata None)."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({
            "user_id": "user-reflect-ed3",
            "message": "No dates here.",
        })
        store_calls = []
        with patch("kumiho_memory.mcp_tools.tool_memory_reflect.__module__", "kumiho_memory.mcp_tools"):
            with patch("kumiho.mcp_server.tool_memory_store", _fake_store_recorder(store_calls)):
                result = tool_memory_reflect({
                    "session_id": ingest["session_id"],
                    "response": "Noted.",
                    "captures": [
                        {
                            "type": "preference",
                            "title": "Prefers concise output",
                            "content": "User likes short answers.",
                        },
                    ],
                })
        assert result["captures_stored"] == 1
        assert "dropped_event_dates" not in result
        assert store_calls[0]["metadata"] is None
    finally:
        _cleanup_manager()


def test_memory_reflect_bulk_routes_to_batch():
    """>=2 captures go through ONE batched write, not the per-capture loop."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({"user_id": "user-reflect-bulk", "message": "bulk"})
        store_calls, batch_calls = [], []
        with patch("kumiho.mcp_server.tool_memory_store", _fake_store_recorder(store_calls)), \
                patch("kumiho.mcp_server.tool_memory_store_batch", _fake_batch_recorder(batch_calls)):
            result = tool_memory_reflect({
                "session_id": ingest["session_id"],
                "response": "ok",
                "captures": [
                    {"type": "fact", "title": "A", "content": "aaa"},
                    {"type": "fact", "title": "B", "content": "bbb"},
                    {"type": "fact", "title": "C", "content": "ccc"},
                ],
            })
        assert len(batch_calls) == 1                    # exactly one batched write
        assert len(batch_calls[0]["captures"]) == 3
        assert store_calls == []                        # per-capture loop NOT used
        assert result["captures_stored"] == 3
    finally:
        _cleanup_manager()


def test_memory_reflect_single_capture_uses_store():
    """A single capture keeps the byte-identical per-capture path (no batch)."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({"user_id": "user-reflect-one", "message": "one"})
        store_calls, batch_calls = [], []
        with patch("kumiho.mcp_server.tool_memory_store", _fake_store_recorder(store_calls)), \
                patch("kumiho.mcp_server.tool_memory_store_batch", _fake_batch_recorder(batch_calls)):
            result = tool_memory_reflect({
                "session_id": ingest["session_id"],
                "response": "ok",
                "captures": [{"type": "fact", "title": "solo", "content": "single"}],
            })
        assert len(store_calls) == 1                     # per-capture path used
        assert batch_calls == []                         # batch NOT used
        assert result["captures_stored"] == 1
    finally:
        _cleanup_manager()


def test_memory_reflect_idempotency_prefix_forces_batch_and_returns_capture_results():
    """A single capture WITH idempotency_prefix takes the batch path and the
    result carries a positional capture_results list (the 0.17.0 bulk contract
    history backfill consumes)."""
    try:
        _install_test_manager()
        ingest = tool_memory_ingest({"user_id": "user-reflect-idem", "message": "x"})
        store_calls, batch_calls = [], []
        with patch("kumiho.mcp_server.tool_memory_store", _fake_store_recorder(store_calls)), \
                patch("kumiho.mcp_server.tool_memory_store_batch", _fake_batch_recorder(batch_calls)):
            result = tool_memory_reflect({
                "session_id": ingest["session_id"],
                "response": "ok",
                "captures": [{"type": "fact", "title": "solo", "content": "single"}],
                "idempotency_prefix": "backfill:sess1:abc123",
            })
        assert len(batch_calls) == 1                         # prefix forces batch even for 1 capture
        assert batch_calls[0]["idempotency_prefix"] == "backfill:sess1:abc123"
        assert store_calls == []
        rows = result.get("capture_results")
        assert isinstance(rows, list) and len(rows) == 1     # positionally aligned
        assert rows[0].get("revision_kref")
    finally:
        _cleanup_manager()


def test_reflect_schema_exposes_idempotency_prefix():
    """batch_capable() in the backfill runner keys off this schema property."""
    from kumiho_memory.mcp_tools import MEMORY_TOOLS
    reflect = next(t for t in MEMORY_TOOLS if t["name"] == "kumiho_memory_reflect")
    assert "idempotency_prefix" in reflect["inputSchema"]["properties"]


def test_decompose_schema_exposes_belief_change_fields():
    """Agents learn the two optional belief-change lists from the schema."""
    from kumiho_memory.mcp_tools import MEMORY_TOOLS
    decompose = next(t for t in MEMORY_TOOLS if t["name"] == "kumiho_memory_decompose")
    props = decompose["inputSchema"]["properties"]
    assert "supersedes" in props and "contradicts" in props
    assert props["supersedes"]["items"]["required"] == ["statement", "replaces"]
    assert props["contradicts"]["items"]["required"] == ["statement", "conflicts_with"]


def test_decompose_schema_exposes_optional_project_target():
    """Agents learn the optional project-targeting field from the schema (#136)."""
    from kumiho_memory.mcp_tools import MEMORY_TOOLS
    decompose = next(t for t in MEMORY_TOOLS if t["name"] == "kumiho_memory_decompose")
    schema = decompose["inputSchema"]
    assert "project" in schema["properties"]
    assert schema["properties"]["project"]["type"] == "string"
    # Additive + backward-compatible: only kref stays required.
    assert schema["required"] == ["kref"]


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

    # Restore (not pop) the displaced entry: a bare pop would delete the key and
    # force the next `import kumiho` to build a fresh module object, breaking
    # identity-based monkeypatching in sibling tests (see test_ontology_agent).
    _missing = object()
    saved = sys.modules.get("kumiho", _missing)
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
        if saved is _missing:
            sys.modules.pop("kumiho", None)
        else:
            sys.modules["kumiho"] = saved


def test_memory_dream_state_accepts_gemini_and_base_url():
    """Dream State should accept gemini plus an OpenAI-compatible base_url."""
    import sys
    import types

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
            create_item=lambda item_name, kind: FakeItem(f"kref://{item_name}/{n}.{kind}")
        ),
    )
    fake_sdk.event_stream = lambda **kw: iter([])
    fake_sdk.get_attribute = lambda kref, key: None
    fake_sdk.set_attribute = lambda kref, key, val: None
    fake_sdk.batch_get_revisions = lambda **kw: ([], [])
    fake_sdk.get_client = lambda: None
    fake_sdk.Kref = FakeKref

    # Restore (not pop) the displaced entry: a bare pop would delete the key and
    # force the next `import kumiho` to build a fresh module object, breaking
    # identity-based monkeypatching in sibling tests (see test_ontology_agent).
    _missing = object()
    saved = sys.modules.get("kumiho", _missing)
    sys.modules["kumiho"] = fake_sdk
    try:
        with patch("kumiho_memory.summarization.MemorySummarizer", return_value=StubSummarizer()) as mock_summarizer:
            result = tool_memory_dream_state({
                "project": "CognitiveMemory",
                "dry_run": True,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "api_key": "gemini-direct-key",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            })
        assert result["success"] is True
        mock_summarizer.assert_called_once_with(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="gemini-direct-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        dream_tool = next(tool for tool in MEMORY_TOOLS if tool["name"] == "kumiho_memory_dream_state")
        provider_schema = dream_tool["inputSchema"]["properties"]["provider"]
        assert "gemini" in provider_schema["enum"]
        assert "base_url" in dream_tool["inputSchema"]["properties"]
    finally:
        if saved is _missing:
            sys.modules.pop("kumiho", None)
        else:
            sys.modules["kumiho"] = saved


# ---------------------------------------------------------------------------
# Tests — evidence-level plumbing (issue #9)
# ---------------------------------------------------------------------------


def test_ingest_and_consolidate_schemas_accept_evidence():
    """Both lifecycle tools expose optional evidence_level/source args."""
    by_name = {tool["name"]: tool for tool in MEMORY_TOOLS}
    for name in ("kumiho_memory_ingest", "kumiho_memory_consolidate"):
        props = by_name[name]["inputSchema"]["properties"]
        assert "evidence_level" in props, f"{name} missing evidence_level"
        assert props["evidence_level"]["enum"] == [
            "official", "corroborated", "single_source", "unverified",
        ]
        assert "source" in props, f"{name} missing source"
        # Optional — must not be required
        required = by_name[name]["inputSchema"].get("required", [])
        assert "evidence_level" not in required
        assert "source" not in required


def test_memory_ingest_forwards_evidence_to_consolidation():
    """Evidence passed to the ingest tool is stamped on the stored memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            manager, stored = _install_test_manager(tmpdir)
            ingest = tool_memory_ingest({
                "user_id": "user-mcp-ev",
                "message": "Acme shipped v2.0",
                "evidence_level": "official",
                "source": "press-release:acme",
            })
            tool_memory_add_response({
                "session_id": ingest["session_id"],
                "response": "Recorded.",
            })
            result = tool_memory_consolidate({
                "session_id": ingest["session_id"],
            })
            assert result["success"] is True
            assert stored["metadata"]["evidence_level"] == "official"
            assert stored["metadata"]["source"] == "press-release:acme"
            assert "evidence:official" in stored["tags"]
        finally:
            _cleanup_manager()


def test_memory_consolidate_accepts_explicit_evidence():
    """Evidence passed directly to the consolidate tool is stamped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            manager, stored = _install_test_manager(tmpdir)
            ingest = tool_memory_ingest({
                "user_id": "user-mcp-ev2",
                "message": "Two outlets report the same numbers",
            })
            tool_memory_add_response({
                "session_id": ingest["session_id"],
                "response": "Recorded.",
            })
            result = tool_memory_consolidate({
                "session_id": ingest["session_id"],
                "evidence_level": "corroborated",
                "source": "news:reuters",
            })
            assert result["success"] is True
            assert stored["metadata"]["evidence_level"] == "corroborated"
            assert stored["metadata"]["source"] == "news:reuters"
            assert "evidence:corroborated" in stored["tags"]
        finally:
            _cleanup_manager()


# ---------------------------------------------------------------------------
# Tests — backend-error surfacing (issue #103, P1-1)
# ---------------------------------------------------------------------------


def test_memory_recall_surfaces_backend_error():
    """A retrieve-backend failure surfaces as an additive backend_error field so
    an empty result isn't misread as 'no memories'."""
    try:
        manager, _ = _install_test_manager()

        async def error_retrieve(**kwargs):
            return {"error": "graph backend unavailable: connection refused"}

        manager.memory_retrieve = error_retrieve
        result = tool_memory_recall({"query": "recall while backend is down"})
        assert result["count"] == 0
        assert result["results"] == []
        assert "backend_error" in result
        assert "graph backend unavailable" in result["backend_error"]
    finally:
        _cleanup_manager()


def test_memory_recall_no_backend_error_on_success():
    """A healthy recall carries NO backend_error field (behavior unchanged)."""
    try:
        _install_test_manager()  # default retrieve_stub succeeds
        result = tool_memory_recall({"query": "healthy recall success"})
        assert result["count"] == 1
        assert "backend_error" not in result
    finally:
        _cleanup_manager()


def test_memory_recall_no_backend_error_on_empty_healthy():
    """An empty-but-healthy recall carries NO backend_error field."""
    try:
        manager, _ = _install_test_manager()

        async def empty_retrieve(**kwargs):
            return {"revision_krefs": []}

        manager.memory_retrieve = empty_retrieve
        result = tool_memory_recall({"query": "empty but healthy"})
        assert result["count"] == 0
        assert result["results"] == []
        assert "backend_error" not in result
    finally:
        _cleanup_manager()


def test_memory_engage_surfaces_backend_error():
    """engage surfaces the same additive backend_error field on failure."""
    try:
        manager, _ = _install_test_manager()

        async def error_retrieve(**kwargs):
            return {"error": "neo4j down"}

        manager.memory_retrieve = error_retrieve
        result = tool_memory_engage({"query": "engage while backend is down"})
        assert result["count"] == 0
        assert "backend_error" in result
        assert "neo4j down" in result["backend_error"]
    finally:
        _cleanup_manager()


def test_memory_engage_no_backend_error_on_success():
    """A healthy engage carries NO backend_error field (behavior unchanged)."""
    try:
        _install_test_manager()  # default retrieve_stub succeeds
        result = tool_memory_engage({"query": "healthy engage success"})
        assert result["count"] == 1
        assert "backend_error" not in result
    finally:
        _cleanup_manager()


# ---------------------------------------------------------------------------
# Tests — race-free lazy singleton init (issue #103, P1-3)
# ---------------------------------------------------------------------------


def test_get_manager_singleton_thread_safe(monkeypatch):
    """Concurrent first callers must all receive the SAME manager instance —
    the double-checked lock must construct exactly once even when many threads
    race through _get_manager at the same time (a slow constructor widens the
    race window that the missing lock would have lost)."""
    import threading
    import time as _time

    # Reset the singleton and swap in a slow fake builder, both via setitem on
    # the module __dict__ so pytest restores them automatically at teardown.
    monkeypatch.setitem(mcp_tools_module.__dict__, "_manager", None)

    build_count = {"n": 0}
    build_lock = threading.Lock()

    def slow_build():
        with build_lock:
            build_count["n"] += 1
        _time.sleep(0.05)  # widen the race window
        return object()

    monkeypatch.setitem(mcp_tools_module.__dict__, "_build_manager", slow_build)

    barrier = threading.Barrier(8)
    results = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()  # release all threads simultaneously
        mgr = mcp_tools_module._get_manager()
        with results_lock:
            results.append(mgr)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert build_count["n"] == 1, "manager was constructed more than once under a race"
    assert len(results) == 8
    assert all(r is results[0] for r in results), "threads saw different manager instances"
