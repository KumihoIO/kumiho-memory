"""Decision Memory isolation guarantees (design criterion 1).

The code domain must be invisible when its gate is off: the manager's
delegation short-circuits without importing the code modules, and the MCP
registry gains no tools.  With the gate on, the surface appears — and the
conversation-domain tool set is unchanged either way.
"""

import asyncio
import sys

from kumiho_memory.code_decisions import code_memory_enabled


def test_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_CODE", raising=False)
    assert code_memory_enabled() is False
    monkeypatch.setenv("KUMIHO_MEMORY_CODE", "1")
    assert code_memory_enabled() is True
    monkeypatch.setenv("KUMIHO_MEMORY_CODE", "0")
    assert code_memory_enabled() is False


def test_manager_code_why_short_circuits_when_gated_off(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_CODE", raising=False)
    # The gated-off path must not import the query engine at all.
    monkeypatch.delitem(sys.modules, "kumiho_memory.code_query", raising=False)

    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from tests.fakes import FakeRedis

    async def _store(**k):
        return {}

    mgr = UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test"),
        memory_store=_store,
    )
    res = asyncio.run(mgr.code_why("why?", file="a.py"))
    assert res["decisions"] == [] and "disabled" in res.get("error", "")
    assert "kumiho_memory.code_query" not in sys.modules  # never imported


def test_mcp_registration_respects_gate(monkeypatch):
    import kumiho_memory.mcp_tools as mt

    tools_before = list(mt.MEMORY_TOOLS)
    handlers_before = dict(mt.MEMORY_TOOL_HANDLERS)
    try:
        # Simulate a gated-off registration pass on a clean registry.
        mt.MEMORY_TOOLS[:] = [t for t in mt.MEMORY_TOOLS if t["name"] != "kumiho_code_why"]
        mt.MEMORY_TOOL_HANDLERS.pop("kumiho_code_why", None)

        monkeypatch.delenv("KUMIHO_MEMORY_CODE", raising=False)
        mt._register_code_memory_tools()
        assert all(t["name"] != "kumiho_code_why" for t in mt.MEMORY_TOOLS)
        assert "kumiho_code_why" not in mt.MEMORY_TOOL_HANDLERS

        monkeypatch.setenv("KUMIHO_MEMORY_CODE", "1")
        mt._register_code_memory_tools()
        assert any(t["name"] == "kumiho_code_why" for t in mt.MEMORY_TOOLS)
        assert "kumiho_code_why" in mt.MEMORY_TOOL_HANDLERS

        schema = next(
            t for t in mt.MEMORY_TOOLS if t["name"] == "kumiho_code_why"
        )["inputSchema"]
        # anyOf enforces the file|question minimum input contract.
        assert {"required": ["file"]} in schema["anyOf"]
        assert {"required": ["question"]} in schema["anyOf"]
    finally:
        mt.MEMORY_TOOLS[:] = tools_before
        mt.MEMORY_TOOL_HANDLERS.clear()
        mt.MEMORY_TOOL_HANDLERS.update(handlers_before)


def test_conversation_tool_set_unchanged_by_gate():
    """The conversation-domain tool names are identical regardless of the
    code gate — the code domain only ever APPENDS its own tools."""
    import kumiho_memory.mcp_tools as mt

    conversation = {
        t["name"] for t in mt.MEMORY_TOOLS if not t["name"].startswith("kumiho_code_")
    }
    assert {
        "kumiho_chat_add", "kumiho_chat_get", "kumiho_chat_clear",
        "kumiho_memory_ingest", "kumiho_memory_recall", "kumiho_memory_engage",
        "kumiho_memory_reflect", "kumiho_memory_consolidate",
    } <= conversation
