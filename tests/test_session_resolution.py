"""Session identity resolution (issue #3).

The caller of the session-scoped MCP tools is an LLM. Requiring it to
originate and hold a stable opaque ``session_id`` across turns produced two
failure modes: omission (loud, the turn's memory lost to a validation error)
and drift (silent — the conversation buffer fragments into per-typo buckets
that a 1-hour TTL then erases). These tests pin the redesign:

- ONE resolver for every defaulting path: explicit argument, else host env
  (``KUMIHO_SESSION_ID`` / ``CLAUDE_CODE_SESSION_ID``), else the shared
  active-session pointer, else generated — and the env tier WRITES the
  pointer so a sibling server process without the env var converges on the
  same bucket instead of defaulting differently.
- Consolidation clears the active pointer only if it still points at the
  session being consolidated: unconditional, a backfill consolidation
  severed the LIVE conversation's continuity.
- A write that mints a bucket says so (``created_bucket``): it is the one
  observable trace of a drifted id, and it used to be indistinguishable
  from success.
- ``list_sessions`` round-trips ids containing colons — the ids this system
  itself mints contain three.
"""

import asyncio
import fnmatch
from unittest.mock import patch

import pytest

import kumiho_memory.mcp_tools as mcp_tools_module
from kumiho_memory.mcp_tools import (
    MEMORY_TOOLS,
    _resolve_session,
    tool_chat_add,
)
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from test_memory_manager import StubRedactor, StubSummarizer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _PointerBuffer:
    """In-memory stand-in for the pointer/sequence surface of the buffer."""

    def __init__(self, active=None):
        self.active = active
        self.set_calls = []

    async def get_active_session(self, *, context, user_canonical_id):
        return self.active

    async def set_active_session(
        self, *, context, user_canonical_id, session_id, ttl_seconds=86400
    ):
        self.active = session_id
        self.set_calls.append(session_id)

    async def next_session_sequence(self, *, user_canonical_id, date_str):
        return 7

    async def close(self):
        return None


class _FakeRedis:
    """The handful of redis-py methods the buffer paths under test touch."""

    def __init__(self):
        self.kv = {}
        self.lists = {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None):
        self.kv[key] = value

    async def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)
            self.lists.pop(key, None)

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def expire(self, key, ttl):
        return True

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        n = len(items)
        if n == 0:
            return []
        if start < 0:
            start += n
        if end < 0:
            end += n
        start = max(start, 0)
        end = min(end, n - 1)
        if start > end:
            return []
        return items[start:end + 1]

    async def ttl(self, key):
        return 3600 if key in self.lists or key in self.kv else -2

    async def scan(self, cursor, match=None, count=100):
        keys = [k for k in self.lists if match is None or fnmatch.fnmatch(k, match)]
        return 0, keys


def _manager(buffer):
    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )


def _buffer_with_fake_client():
    """A RedisMemoryBuffer wired straight to a fake client.

    ``__new__`` on purpose: ``__init__`` resolves live config (URLs, tenant
    discovery), which a unit test must not touch.
    """
    buf = RedisMemoryBuffer.__new__(RedisMemoryBuffer)
    buf.client = _FakeRedis()
    buf.tenant_id = "tenant-t"
    buf.tenant_hint = ""
    buf.default_ttl = 3600
    return buf


# ---------------------------------------------------------------------------
# The resolver: explicit > host env > pointer > generated
# ---------------------------------------------------------------------------


def test_an_explicit_argument_always_wins_and_needs_no_manager():
    """Backfill and bulk ingest address historical sessions by name; that
    path must not even consult the resolution machinery."""
    session_id, source = asyncio.run(
        _resolve_session({"session_id": "hist-42"}, manager=None)
    )
    assert (session_id, source) == ("hist-42", "argument")


def test_host_env_wins_and_never_touches_the_pointer(monkeypatch):
    """The env tier is deterministic and must leave the shared pointer alone.

    It used to WRITE the pointer, on the theory that a sibling server without
    the env var serves the same conversation. Measured live, siblings share
    spawn context and all carry the env — the env-less processes belong to
    DIFFERENT (Desktop-app) conversations, and the write let any env-bearing
    CLI process repoint a concurrent env-less conversation's continuity to
    its own bucket, splitting that conversation mid-stream (PR #4 review)."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "sess-uuid-1")
    buffer = _PointerBuffer(active="another-conversations-live-session")
    session_id, source = asyncio.run(_manager(buffer).resolve_session_id())
    assert (session_id, source) == ("sess-uuid-1", "host-env")
    assert buffer.active == "another-conversations-live-session"
    assert buffer.set_calls == []


def test_the_env_tier_is_skipped_for_user_scoped_callers(monkeypatch):
    """The env names the HOST conversation, not a user.

    Applied to user-scoped resolution it collapsed every user and context on
    the process into one bucket: ingest(alice) and ingest(bob) shared working
    memory, and consolidation filed bob's turns under alice's space — the
    blocker from the PR #4 review. A user-scoped caller must resolve
    per-(context, user) exactly as on main, env or no env."""
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "host-uuid-1")
    alice_buf = _PointerBuffer()
    alice, source_a = asyncio.run(
        _manager(alice_buf).resolve_session_id(user_id="alice", context="personal")
    )
    bob, source_b = asyncio.run(
        _manager(_PointerBuffer()).resolve_session_id(user_id="bob", context="work")
    )
    assert alice != "host-uuid-1" and bob != "host-uuid-1"
    assert alice != bob
    assert alice.startswith("personal:user-")
    assert bob.startswith("work:user-")
    assert (source_a, source_b) == ("generated", "generated")


def test_claude_code_session_id_is_the_fallback_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "cc-uuid-9")
    buffer = _PointerBuffer()
    session_id, source = asyncio.run(_manager(buffer).resolve_session_id())
    assert (session_id, source) == ("cc-uuid-9", "host-env")


def test_kumiho_session_id_outranks_claude_code(monkeypatch):
    """KUMIHO_SESSION_ID is the explicit contract; the Claude variable is an
    observed inheritance. Explicit beats observed."""
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "cc-uuid-9")
    monkeypatch.setenv("KUMIHO_SESSION_ID", "kumiho-uuid-1")
    buffer = _PointerBuffer()
    session_id, _ = asyncio.run(_manager(buffer).resolve_session_id())
    assert session_id == "kumiho-uuid-1"


def test_env_matching_pointer_does_not_rewrite_it(monkeypatch):
    monkeypatch.setenv("KUMIHO_SESSION_ID", "sess-uuid-1")
    buffer = _PointerBuffer(active="sess-uuid-1")
    session_id, _ = asyncio.run(_manager(buffer).resolve_session_id())
    assert session_id == "sess-uuid-1"
    assert buffer.set_calls == []


def test_the_pointer_is_reused_when_no_env_is_present():
    """This is ingest's pre-existing continuity mechanism — the redesign
    routes the tools through it rather than inventing a second default that
    would split the conversation by construction."""
    buffer = _PointerBuffer(active="mcp:user-abcdef1234:20260730:003")
    session_id, source = asyncio.run(_manager(buffer).resolve_session_id())
    assert session_id == "mcp:user-abcdef1234:20260730:003"
    assert source == "active-pointer"
    assert buffer.set_calls == []


def test_generated_as_the_last_resort_and_registered_on_the_pointer():
    buffer = _PointerBuffer(active=None)
    session_id, source = asyncio.run(_manager(buffer).resolve_session_id())
    assert source == "generated"
    assert session_id.startswith("mcp:user-")
    assert session_id.endswith(":007")  # the fake sequence
    assert buffer.set_calls == [session_id]


def test_generate_session_id_keeps_its_contract():
    """ingest's entry point: same name, same signature, still returns a str."""
    buffer = _PointerBuffer(active="personal:user-aa:20260101:001")
    session_id = asyncio.run(
        _manager(buffer)._generate_session_id("user-x", "personal")
    )
    assert session_id == "personal:user-aa:20260101:001"


# ---------------------------------------------------------------------------
# The pointer: compare-and-delete
# ---------------------------------------------------------------------------


def test_consolidating_a_backfill_id_no_longer_severs_the_live_pointer():
    """clear_active_session was a plain DELETE: consolidating ANY session for
    a (context, user) — a backfill id, a historical fragment — deleted the
    pointer aimed at the CURRENT session, and the next default-resolved call
    minted a fresh bucket mid-conversation."""
    buf = _buffer_with_fake_client()
    asyncio.run(buf.set_active_session(
        context="personal", user_canonical_id="u1", session_id="live-session",
    ))
    asyncio.run(buf.clear_active_session(
        context="personal", user_canonical_id="u1", only_if="old-backfill-id",
    ))
    assert asyncio.run(buf.get_active_session(
        context="personal", user_canonical_id="u1",
    )) == "live-session"


def test_consolidating_the_pointed_session_still_clears_it():
    buf = _buffer_with_fake_client()
    asyncio.run(buf.set_active_session(
        context="personal", user_canonical_id="u1", session_id="live-session",
    ))
    asyncio.run(buf.clear_active_session(
        context="personal", user_canonical_id="u1", only_if="live-session",
    ))
    assert asyncio.run(buf.get_active_session(
        context="personal", user_canonical_id="u1",
    )) is None


def test_clear_without_only_if_keeps_the_old_unconditional_behaviour():
    buf = _buffer_with_fake_client()
    asyncio.run(buf.set_active_session(
        context="personal", user_canonical_id="u1", session_id="live-session",
    ))
    asyncio.run(buf.clear_active_session(
        context="personal", user_canonical_id="u1",
    ))
    assert asyncio.run(buf.get_active_session(
        context="personal", user_canonical_id="u1",
    )) is None


# ---------------------------------------------------------------------------
# The write signal, and listing round-trip
# ---------------------------------------------------------------------------


def test_the_first_write_into_a_session_says_it_minted_the_bucket():
    """A drifted id lands as a fresh bucket looking exactly like a first
    message; a read of a wrong id returns a clean empty. The mint flag on the
    write is the one place the drift is observable."""
    buf = _buffer_with_fake_client()
    first = asyncio.run(buf.add_message(
        project="P", session_id="s1", role="user", content="hi",
    ))
    second = asyncio.run(buf.add_message(
        project="P", session_id="s1", role="assistant", content="hello",
    ))
    assert first["created_bucket"] is True
    assert second["created_bucket"] is False


def test_list_sessions_round_trips_the_ids_this_system_mints():
    """The generated format is '{context}:user-{hash}:{date}:{seq}' — three
    colons. Split-and-take-one truncated it to just the context, so the
    listing could never name a session the system itself created."""
    buf = _buffer_with_fake_client()
    minted = "personal:user-abcdef1234:20260730:001"
    asyncio.run(buf.add_message(
        project="P", session_id=minted, role="user", content="hi",
    ))
    listing = asyncio.run(buf.list_sessions("P"))
    assert listing["sessions"] == [minted]


# ---------------------------------------------------------------------------
# The MCP layer: schemas and handlers
# ---------------------------------------------------------------------------

SESSION_SCOPED_TOOLS = (
    "kumiho_chat_add",
    "kumiho_chat_get",
    "kumiho_chat_clear",
    "kumiho_memory_add_response",
    "kumiho_memory_consolidate",
    "kumiho_memory_reflect",
)


def test_no_tool_requires_session_id_any_more():
    """The argument stays available for backfill; requiring it forced an LLM
    to invent one."""
    for tool in MEMORY_TOOLS:
        required = tool["inputSchema"].get("required") or []
        assert "session_id" not in required, tool["name"]


def test_every_session_scoped_tool_says_omission_is_the_convention():
    """The tool description is the one prose channel that reliably reaches a
    caller (the transport strips descriptions from optional properties), so
    the defaulting contract has to ride there — on every tool, not one."""
    by_name = {t["name"]: t for t in MEMORY_TOOLS}
    for name in SESSION_SCOPED_TOOLS:
        text = by_name[name]["description"]
        assert "session_id is OPTIONAL" in text, name
        assert "KUMIHO_SESSION_ID" in text, name
        assert "session_id_source" in text, name


class _StubManager:
    """Just enough manager for the handlers: records the session id every
    operation actually receives, resolves a fixed identity."""

    project = "P"

    def __init__(self):
        self.seen = {}
        mgr = self

        class _Buf:
            async def add_message(self, *, project, session_id, role, content,
                                  metadata=None):
                mgr.seen["op_session"] = session_id
                return {"success": True, "message_count": 2,
                        "created_bucket": False}

            async def get_messages(self, *, project, session_id, limit=50):
                mgr.seen["op_session"] = session_id
                return {"messages": [], "session_id": session_id,
                        "message_count": 0, "ttl_remaining": 0}

            async def clear_session(self, project, session_id):
                mgr.seen["op_session"] = session_id
                return {"success": True, "cleared_count": 0}

        self.redis_buffer = _Buf()

    async def resolve_session_id(self, user_id=None, context=None):
        return "sess-resolved-9", "host-env"

    async def add_assistant_response(self, *, session_id, response,
                                     channel="unknown"):
        self.seen["op_session"] = session_id
        return {"success": True, "message_count": 1, "created_bucket": True}

    async def consolidate_session(self, *, session_id, evidence_level=None,
                                  source=None):
        self.seen["op_session"] = session_id
        return {"success": False, "error": "No messages to consolidate"}

    async def handle_user_message(self, *, user_id, message, context,
                                  session_id, working_memory_limit,
                                  recall_limit, evidence_level, source):
        self.seen["op_session"] = session_id
        return {"success": True, "session_id": session_id}


@pytest.mark.parametrize(
    "handler_name,args",
    [
        ("tool_chat_add", {"message": "m"}),
        ("tool_chat_get", {}),
        ("tool_chat_clear", {}),
        ("tool_memory_add_response", {"response": "r"}),
        ("tool_memory_consolidate", {}),
        ("tool_memory_reflect", {"response": "r"}),
        ("tool_memory_ingest", {"user_id": "u", "message": "m"}),
    ],
)
def test_every_session_scoped_handler_resolves_and_reports_when_omitted(
    handler_name, args,
):
    """All seven, not one: the review found only chat_add exercised the
    omission path end-to-end, so a handler that forgot to resolve (or passed
    the raw missing arg through) was invisible. Each handler must feed the
    RESOLVED id to its operation and report it in the result — including
    ingest, whose own per-user defaulting would otherwise put an id-omitting
    MCP conversation's user turns and assistant turns in different buckets."""
    stub = _StubManager()
    handler = getattr(mcp_tools_module, handler_name)
    with patch.object(mcp_tools_module, "_manager", stub):
        result = handler(dict(args))
    assert stub.seen["op_session"] == "sess-resolved-9", handler_name
    assert result["session_id"] == "sess-resolved-9", handler_name
    assert result["session_id_source"] == "host-env", handler_name


def test_consolidating_an_identity_less_session_rotates_the_default_pointer():
    """A default-path session has no user metadata. Skipping the pointer
    clear for it (the old identity gate) left the ('mcp','default') pointer
    aimed at a session whose buffer consolidation just cleared — the tool
    path could never rotate to a fresh session. The compare-and-delete makes
    clearing under the default keys safe: a pointer holding some OTHER
    session stays."""
    buffer = RedisMemoryBuffer(client=_FakeRedis(), redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def scenario():
        # The default-path session: buffered turns, no identity metadata.
        session_id, source = await manager.resolve_session_id()
        assert source == "generated"
        await buffer.add_message(
            project=manager.project, session_id=session_id,
            role="user", content="hello",
        )
        await manager.consolidate_session(session_id=session_id)
        return await buffer.get_active_session(
            context="mcp", user_canonical_id="default",
        )

    assert asyncio.run(scenario()) is None

    async def backfill_scenario():
        live, _ = await manager.resolve_session_id()
        await buffer.add_message(
            project=manager.project, session_id="historical-backfill",
            role="user", content="old",
        )
        await manager.consolidate_session(session_id="historical-backfill")
        return await buffer.get_active_session(
            context="mcp", user_canonical_id="default",
        ), live

    pointer_after, live = asyncio.run(backfill_scenario())
    assert pointer_after == live  # the live session's pointer survived


def test_code_mine_session_fails_fast_when_no_id_can_be_resolved():
    """No argument and no host env: the handler must refuse BEFORE the
    manager call. It used to pass '' through, and with ingest_first=True that
    ran a full commit-ingest pre-pass (LLM calls, graph writes) before
    soft-failing inside mine_session — money spent on a call that could never
    mine anything."""
    calls = []

    class _Spy:
        async def code_mine_session(self, *a, **k):
            calls.append((a, k))
            return {}

    with patch.object(mcp_tools_module, "_manager", _Spy()):
        result = mcp_tools_module.tool_code_mine_session({})

    assert calls == []
    assert any("session_id is required" in e for e in result["errors"])


def test_chat_add_resolves_and_reports_when_the_caller_omits_the_id(monkeypatch):
    """End to end through the handler: no session_id argument, env present —
    the write lands in the env session and the result SAYS so."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "sess-env-7")
    buffer = _PointerBuffer()

    class _Buf(_PointerBuffer):
        pass

    manager = _manager(buffer)

    async def add_message(*, project, session_id, role, content, metadata=None):
        return {"success": True, "message_count": 1, "created_bucket": True,
                "recorded": session_id}

    manager.redis_buffer.add_message = add_message
    with patch.object(mcp_tools_module, "_manager", manager):
        result = tool_chat_add({"message": "hello"})

    assert result["session_id"] == "sess-env-7"
    assert result["session_id_source"] == "host-env"
    assert result["recorded"] == "sess-env-7"
    assert result["created_bucket"] is True
