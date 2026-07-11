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
    monkeypatch.setenv("KUMIHO_MEMORY_CODE", "true")
    assert code_memory_enabled() is True  # common truthy spellings accepted
    monkeypatch.setenv("KUMIHO_MEMORY_CODE", "YES")
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


def test_consolidation_call_graph_unchanged_until_automine(monkeypatch):
    """Design §2.2c: with AUTOMINE off, consolidation neither imports the
    session-mining module nor calls ``code_mine_session`` — whether the
    master code gate is off or on, the consolidation call graph and stored
    payload shape are identical.  The chain fires only on the double
    opt-in, with the freshly stored revision kref passed in-band."""
    import tempfile

    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from tests.fakes import FakeRedis

    class _Summarizer:
        async def summarize_conversation(self, messages, context=None):
            return {"type": "summary", "title": "T", "summary": "S.",
                    "classification": {"topics": []}}

        async def generate_implications(self, messages, context=None):
            return []

    class _Redactor:
        def anonymize_summary(self, summary):
            return summary

        def reject_credentials(self, text):
            pass

    mine_calls = []

    async def _consolidate_once(tmpdir):
        stored = {}

        async def _store(**kwargs):
            stored.update(kwargs)
            return {"revision_kref": "kref://p/memory/conv@1"}

        mgr = UniversalMemoryManager(
            redis_buffer=RedisMemoryBuffer(client=FakeRedis(),
                                           redis_url="redis://test"),
            summarizer=_Summarizer(),
            pii_redactor=_Redactor(),
            memory_store=_store,
            consolidation_threshold=2,
            artifact_root=tmpdir,
        )

        async def _spy(session_id, **kwargs):
            mine_calls.append((session_id, kwargs))
            return {}

        mgr.code_mine_session = _spy
        ingest = await mgr.ingest_message(user_id="u", message="hello there")
        await mgr.add_assistant_response(
            session_id=ingest["session_id"], response="hi",
        )
        result = await mgr.consolidate_session(session_id=ingest["session_id"])
        assert result["success"] is True
        return stored

    monkeypatch.setenv("KUMIHO_MEMORY_ONTOLOGY", "0")  # minimal store path
    monkeypatch.delitem(sys.modules, "kumiho_memory.code_session", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        # both gates off — the baseline call graph
        monkeypatch.delenv("KUMIHO_MEMORY_CODE", raising=False)
        monkeypatch.delenv("KUMIHO_MEMORY_CODE_AUTOMINE", raising=False)
        base = asyncio.run(_consolidate_once(tmpdir))
        assert not mine_calls
        assert "kumiho_memory.code_session" not in sys.modules

        # master gate on, AUTOMINE off — must be indistinguishable
        monkeypatch.setenv("KUMIHO_MEMORY_CODE", "1")
        gated = asyncio.run(_consolidate_once(tmpdir))
        assert not mine_calls
        assert "kumiho_memory.code_session" not in sys.modules
        assert sorted(gated.keys()) == sorted(base.keys())
        assert gated.get("summary") == base.get("summary")

        # double opt-in — the chain fires once, kref + messages in-band
        monkeypatch.setenv("KUMIHO_MEMORY_CODE_AUTOMINE", "1")
        asyncio.run(_consolidate_once(tmpdir))
        assert len(mine_calls) == 1
        _sid, kwargs = mine_calls[0]
        assert kwargs["conversation_kref"] == "kref://p/memory/conv@1"
        assert kwargs["messages"]  # zero re-reads: buffer handed over in-band


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
