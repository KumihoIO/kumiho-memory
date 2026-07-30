"""Session identity resolution (issue #3).

The caller of the session-scoped MCP tools is an LLM. Requiring it to
originate and hold a stable opaque ``session_id`` across turns produced two
failure modes: omission (loud, the turn's memory lost to a validation error)
and drift (silent — the conversation buffer fragments into per-typo buckets
that a 1-hour TTL then erases). These tests pin the redesign:

- ONE resolver for every defaulting path: explicit argument, else host env
  (``KUMIHO_SESSION_ID`` / ``CLAUDE_CODE_SESSION_ID``, identity-less callers
  only), else the shared active-session pointer, else generated — and the
  env tier never touches the shared pointer: the write it once did let an
  env-bearing CLI process repoint a concurrent env-less (Desktop-app)
  conversation's continuity to its own bucket (see
  test_host_env_wins_and_never_touches_the_pointer).
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
        self, *, context, user_canonical_id, session_id, ttl_seconds=86400,
        nx=False,
    ):
        if nx and self.active is not None:
            return False
        self.active = session_id
        self.set_calls.append(session_id)
        return True if nx else None

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

    async def set(self, key, value, ex=None, nx=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

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

    async def incr(self, key):
        value = int(self.kv.get(key, 0)) + 1
        self.kv[key] = value
        return value

    async def hset(self, key, mapping=None):
        self.kv.setdefault(key, {})
        self.kv[key].update(mapping or {})

    async def hgetall(self, key):
        value = self.kv.get(key)
        return dict(value) if isinstance(value, dict) else {}

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
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-1")
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


def test_claude_code_session_id_is_no_longer_honored(monkeypatch):
    """CLAUDE_CODE_SESSION_ID reaches the server by env inheritance — an
    accident frozen at spawn, and Claude Code rotates its session on /clear
    WITHOUT respawning the server: trusting it silently merged the
    post-/clear conversation into the previous one's bucket (round 8). Only
    the explicit KUMIHO_SESSION_ID contract resolves; the launcher that sets
    it owns rotation correctness (kumiho-plugins#45)."""
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "cc-uuid-9")
    buffer = _PointerBuffer()
    with pytest.raises(ValueError):
        asyncio.run(_manager(buffer).resolve_session_id())


def test_env_matching_pointer_does_not_rewrite_it(monkeypatch):
    monkeypatch.setenv("KUMIHO_SESSION_ID", "sess-uuid-1")
    buffer = _PointerBuffer(active="sess-uuid-1")
    session_id, _ = asyncio.run(_manager(buffer).resolve_session_id())
    assert session_id == "sess-uuid-1"
    assert buffer.set_calls == []


def test_identity_less_env_less_resolution_refuses_loudly():
    """No host identity and no user identity: REFUSE with instructions.

    Every silent default tried here was a cross-conversation merge in
    disguise — the shared ('mcp','default') pointer merged env-less
    conversations across processes (round 4), and the process-scoped id
    that replaced it merged every chat on hosts that keep ONE server
    across conversations, which is exactly the env-less Claude Desktop
    topology (round 6). With no per-conversation signal there is nothing
    correct to default to; the pointer must also stay untouched."""
    buffer = _PointerBuffer(active="somebody-elses-live-session")
    manager = _manager(buffer)
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(manager.resolve_session_id())
    assert "no session identity available" in str(excinfo.value)
    assert "REUSE" in str(excinfo.value)
    assert buffer.set_calls == []
    assert buffer.active == "somebody-elses-live-session"


def test_user_scoped_resolution_reuses_the_pointer():
    """The (context, user) pointer is ingest's pre-existing continuity
    mechanism, and it stays: user identity is a real cross-process key,
    unlike the identity-less default (round 4)."""
    buffer = _PointerBuffer(active="personal:user-abcdef1234:20260730:003")
    session_id, source = asyncio.run(
        _manager(buffer).resolve_session_id(user_id="alice", context="personal")
    )
    assert session_id == "personal:user-abcdef1234:20260730:003"
    assert source == "active-pointer"
    assert buffer.set_calls == []


def test_user_scoped_generation_registers_the_pointer():
    buffer = _PointerBuffer(active=None)
    session_id, source = asyncio.run(
        _manager(buffer).resolve_session_id(user_id="alice", context="personal")
    )
    assert source == "generated"
    assert session_id.startswith("personal:user-")
    assert session_id.endswith(":007")  # the fake sequence
    assert buffer.set_calls == [session_id]


def test_losing_a_concurrent_pointer_claim_adopts_the_winner():
    """Two concurrent identity-less resolutions that both miss the pointer
    each mint an id; a blind SET kept only the last writer, and everything
    written under the loser became unreachable to every later
    default-resolved call — a silent conversation fork (round 3). The
    registration is a compare-and-claim: the loser adopts the winner."""

    class _RacyBuffer(_PointerBuffer):
        """get_active_session misses for the first two callers (modelling
        two resolvers racing past the read), then serves the claimed value."""

        def __init__(self):
            super().__init__()
            self.reads = 0
            self.seq = 0

        async def get_active_session(self, *, context, user_canonical_id):
            self.reads += 1
            if self.reads <= 2:
                return None
            return self.active

        async def next_session_sequence(self, *, user_canonical_id, date_str):
            self.seq += 1
            return self.seq

    buffer = _RacyBuffer()
    manager = _manager(buffer)
    first, first_source = asyncio.run(
        manager.resolve_session_id(user_id="alice", context="personal")
    )
    second, second_source = asyncio.run(
        manager.resolve_session_id(user_id="alice", context="personal")
    )

    assert first_source == "generated"
    assert (second, second_source) == (first, "active-pointer")
    assert buffer.set_calls == [first]  # the loser never overwrote the claim


def test_created_bucket_comes_from_the_atomic_push_not_a_second_read():
    """RPUSH returns the post-push length atomically; a separate LLEN read
    raced concurrent writers, and two first-writers could both observe
    count 2 — neither reporting created_bucket (round 3)."""

    class _LyingLlenRedis(_FakeRedis):
        async def llen(self, key):
            return 99  # any handler still consulting LLEN gets nonsense

    buf = RedisMemoryBuffer.__new__(RedisMemoryBuffer)
    buf.client = _LyingLlenRedis()
    buf.tenant_id = "tenant-t"
    buf.tenant_hint = ""
    buf.default_ttl = 3600

    result = asyncio.run(buf.add_message(
        project="P", session_id="s1", role="user", content="hi",
    ))
    assert result["message_count"] == 1
    assert result["created_bucket"] is True


def test_ingest_treats_a_blank_session_id_as_generate_for_this_user():
    """Main's ingest resolved `session_id or generate(...)` — "" meant
    "make one for this user", and callers pass it deliberately. The loud
    blank rejection belongs to the identity-less tools only (round 3)."""
    buffer = RedisMemoryBuffer(client=_FakeRedis(), redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )
    with patch.object(mcp_tools_module, "_manager", manager):
        result = mcp_tools_module.tool_memory_ingest(
            {"user_id": "alice", "message": "hi", "session_id": ""}
        )
    assert result["session_id"].startswith("personal:user-")
    assert result["session_id_source"] == "generated"


def test_ingest_rejects_a_whitespace_only_session_id():
    """Main auto-generated only for exactly "" — a whitespace-only value
    reached the buffer and raised there, with zero writes. Normalizing any
    blank silently converted that loud error into a successful write to a
    resolved session (round 8)."""
    stub = _StubManager()
    with patch.object(mcp_tools_module, "_manager", stub):
        with pytest.raises(ValueError):
            mcp_tools_module.tool_memory_ingest(
                {"user_id": "alice", "message": "hi", "session_id": "   "}
            )
    assert "op_session" not in stub.seen  # nothing was written


def test_code_mine_omission_defaults_through_the_rotated_generation(monkeypatch):
    """mine_session loads its transcript from the SAME Redis working-memory
    bucket the identity-less tools write, so the raw env value named the
    consolidated, cleared BASE bucket after any rotation — a guaranteed
    spend-then-soft-fail for the tool's zero-argument usage (round 8). The
    omission default must go through the generation-aware resolver."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-9")
    mined = []

    class _Spy:
        async def resolve_session_id(self, user_id=None, context=None):
            return "host-uuid-9:c1", "host-env"

        async def code_mine_session(self, session_id, **kwargs):
            mined.append(session_id)
            return {"mined": session_id}

    with patch.object(mcp_tools_module, "_manager", _Spy()):
        result = mcp_tools_module.tool_code_mine_session({})
    assert mined == ["host-uuid-9:c1"]
    assert result == {"mined": "host-uuid-9:c1"}

    # Provided-but-blank is an error, never reinterpreted — even with env.
    mined.clear()
    with patch.object(mcp_tools_module, "_manager", _Spy()):
        for blank in ("", "   "):
            result = mcp_tools_module.tool_code_mine_session({"session_id": blank})
            assert "errors" in result
    assert mined == []


def test_the_generation_bump_happens_before_the_buffer_clear(monkeypatch):
    """Ordering (round 8): bump BEFORE clear, so a sibling resolving in the
    window derives the next generation and writes into a bucket that
    survives — after-clear bumping stranded such writes in the cleared
    bucket until TTL, unrecoverable by any future consolidation."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-9")

    class _OpOrderRedis(_FakeRedis):
        def __init__(self):
            super().__init__()
            self.ops = []

        async def incr(self, key):
            self.ops.append(("incr", key))
            return await super().incr(key)

        async def delete(self, *keys):
            for key in keys:
                self.ops.append(("delete", key))
            return await super().delete(*keys)

    client = _OpOrderRedis()
    buffer = RedisMemoryBuffer(client=client, redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def scenario():
        await buffer.add_message(
            project=manager.project, session_id="host-uuid-9",
            role="user", content="hello",
        )
        await manager.consolidate_session(session_id="host-uuid-9")

    asyncio.run(scenario())
    incr_at = next(i for i, (op, k) in enumerate(client.ops)
                   if op == "incr" and "session_gen" in k)
    clear_at = next(i for i, (op, k) in enumerate(client.ops)
                    if op == "delete" and ":messages" in k)
    assert incr_at < clear_at


def test_the_ingest_description_tells_the_truth_about_its_default():
    """Ingest resolves per-user (env tier deliberately skipped), so the
    generic note's "defaults to the host session" claim was FALSE on ingest
    and told the caller the tools converge when they structurally cannot —
    user turns and assistant turns of one conversation landed in two buckets
    under the documented convention (round 3, blocker)."""
    by_name = {t["name"]: t for t in MEMORY_TOOLS}
    ingest = by_name["kumiho_memory_ingest"]

    # The tool description carries ingest-accurate text and the echo rule.
    assert "NOT consulted" in ingest["description"]
    assert "pass that id to the other memory tools" in ingest["description"]
    assert "usually best omitted" not in ingest["description"]

    # The schema property agrees.
    prop = ingest["inputSchema"]["properties"]["session_id"]["description"]
    assert "per-user" in prop
    assert "KUMIHO_SESSION_ID" not in prop

    # And the identity-less tools state the STRUCTURAL convergence rule —
    # repeat the semantic user_id, not an opaque uuid — and advertise the
    # identity properties that make it work.
    for name in SESSION_SCOPED_TOOLS:
        assert "same user_id" in by_name[name]["description"], name
        props = by_name[name]["inputSchema"]["properties"]
        assert "user_id" in props and "context" in props, name


def test_repeating_the_ingest_user_id_converges_on_the_same_bucket(monkeypatch):
    """The round-4 gap: convergence between ingest and the identity-less
    tools relied on prose asking the LLM to echo an opaque session uuid —
    the exact burden issue #3 says LLM callers get wrong. The identity
    properties make it structural: the caller repeats the same user_id it
    gave ingest, and resolution lands on the identical (context, user)
    session — even with a host env present."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-1")
    buffer = RedisMemoryBuffer(client=_FakeRedis(), redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )
    with patch.object(mcp_tools_module, "_manager", manager):
        ingest = mcp_tools_module.tool_memory_ingest(
            {"user_id": "alice", "message": "hello"}
        )
        response = mcp_tools_module.tool_memory_add_response(
            {"response": "hi there", "user_id": "alice"}
        )

    assert response["session_id"] == ingest["session_id"]
    assert response["created_bucket"] is False  # same bucket, not a fork


def test_proxy_mode_derives_the_mint_flag_when_the_server_omits_it():
    """An older proxy server returns its payload without created_bucket;
    hosted mode must still carry the drift signal (round 4)."""
    buf = RedisMemoryBuffer.__new__(RedisMemoryBuffer)
    buf.client = None
    buf.tenant_id = "tenant-t"
    buf.tenant_hint = ""
    buf.default_ttl = 3600

    async def fake_proxy(*, action, payload):
        return {"success": True, "message_count": 1}

    buf._proxy_request = fake_proxy
    result = asyncio.run(buf.add_message(
        project="P", session_id="s1", role="user", content="hi",
    ))
    assert result["created_bucket"] is True


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


def test_mcp_ingest_keeps_users_apart_even_under_a_host_env(monkeypatch):
    """The round-2 blocker: routing ingest through the shared resolver
    WITHOUT its identity bypassed the env-tier guard from the MCP side —
    alice's and bob's turns collapsed into the host-env bucket, bob's
    working memory contained alice's messages, and consolidation filed bob
    under alice's space. The handler must thread user_id/context so the
    resolver's user-scoped path applies exactly as it does for library
    callers."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-1")
    buffer = RedisMemoryBuffer(client=_FakeRedis(), redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )
    with patch.object(mcp_tools_module, "_manager", manager):
        alice = mcp_tools_module.tool_memory_ingest(
            {"user_id": "alice", "message": "alice secret plan"}
        )
        bob = mcp_tools_module.tool_memory_ingest(
            {"user_id": "bob", "message": "bob question", "context": "work"}
        )

    assert alice["session_id"] != "host-uuid-1"
    assert bob["session_id"] != "host-uuid-1"
    assert alice["session_id"] != bob["session_id"]
    assert alice["session_id"].startswith("personal:user-")
    assert bob["session_id"].startswith("work:user-")
    bob_texts = [m.get("content", "") for m in bob.get("working_memory", [])]
    assert all("alice" not in t for t in bob_texts)


def test_consolidating_the_env_session_rotates_its_generation(monkeypatch):
    """The env value is stable for the whole host conversation, so it cannot
    rotate itself — without a generation, consolidation left the tool path
    resolving the cleared bucket forever, and a second consolidation
    overwrote the first conversation's artifact (round 5). The counter lives
    in Redis so SIBLING servers derive the same rotated id."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-9")
    client = _FakeRedis()
    buffer = RedisMemoryBuffer(client=client, redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def scenario():
        first, source = await manager.resolve_session_id()
        assert (first, source) == ("host-uuid-9", "host-env")
        await buffer.add_message(
            project=manager.project, session_id=first,
            role="user", content="hello",
        )
        await manager.consolidate_session(session_id=first)
        rotated, rotated_source = await manager.resolve_session_id()
        return rotated, rotated_source

    rotated, source = asyncio.run(scenario())
    assert (rotated, source) == ("host-uuid-9:c1", "host-env")

    # A sibling server (second manager, same Redis) derives the SAME
    # rotated id — the counter is shared, not in-process.
    sibling = UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=client, redis_url="redis://test"),
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )
    sib_id, _ = asyncio.run(sibling.resolve_session_id())
    assert sib_id == "host-uuid-9:c1"


def test_consolidating_an_old_env_generation_does_not_rotate_the_live_one(
    monkeypatch,
):
    """Exact-generation comparison, not a prefix match: a backfill
    consolidation of '{env}:c1' while the live conversation sits on
    '{env}:c2' must not rotate the live session — the same protection
    only_if gives the pointer (round 5)."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-9")
    client = _FakeRedis()
    buffer = RedisMemoryBuffer(client=client, redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def scenario():
        # Live conversation is on generation 2.
        await buffer.bump_session_generation("host-uuid-9")
        await buffer.bump_session_generation("host-uuid-9")
        live, _ = await manager.resolve_session_id()
        assert live == "host-uuid-9:c2"
        # Backfill-consolidate the OLD generation.
        await buffer.add_message(
            project=manager.project, session_id="host-uuid-9:c1",
            role="user", content="old",
        )
        await manager.consolidate_session(session_id="host-uuid-9:c1")
        after, _ = await manager.resolve_session_id()
        return after

    assert asyncio.run(scenario()) == "host-uuid-9:c2"


def test_a_generation_read_flake_is_retried_not_swallowed(monkeypatch):
    """A single Redis flake on the generation read used to resolve the BASE
    env id mid-conversation — the consolidated, cleared bucket — silently,
    with no log at any level (round 7). The read now follows the same
    one-retry discipline as every other session-continuity Redis op."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-9")

    class _FlakyGenBuffer(_PointerBuffer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def get_session_generation(self, base_session_id):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient flake")
            return 1

    buffer = _FlakyGenBuffer()
    session_id, source = asyncio.run(_manager(buffer).resolve_session_id())
    assert (session_id, source) == ("host-uuid-9:c1", "host-env")
    assert buffer.calls == 2  # initial + one retry


def test_a_bump_flake_at_consolidation_is_retried(monkeypatch, caplog):
    """An unrotated generation after a consolidation flake reintroduces the
    cleared-bucket reuse and the artifact overwrite; a debug-only log hid
    the exposure (round 7). One retry, then a WARNING."""
    import logging

    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-9")

    class _FlakyIncrRedis(_FakeRedis):
        def __init__(self):
            super().__init__()
            self.incr_calls = 0

        async def incr(self, key):
            self.incr_calls += 1
            if self.incr_calls == 1:
                raise ConnectionError("transient flake")
            return await super().incr(key)

    client = _FlakyIncrRedis()
    buffer = RedisMemoryBuffer(client=client, redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def scenario():
        await buffer.add_message(
            project=manager.project, session_id="host-uuid-9",
            role="user", content="hello",
        )
        await manager.consolidate_session(session_id="host-uuid-9")
        return await manager.resolve_session_id()

    with caplog.at_level(logging.WARNING, logger="kumiho_memory.memory_manager"):
        rotated, _ = asyncio.run(scenario())

    # The retry succeeded, so the rotation happened and no warning fired.
    assert rotated == "host-uuid-9:c1"
    assert client.incr_calls == 2
    assert not [r for r in caplog.records if "NOT rotated" in r.getMessage()]


def test_reading_the_generation_slides_its_ttl():
    """Bump-only expiry meant a conversation that stayed on one env id for
    more than 24 h after its last consolidation regressed to the base
    '{env}' id — the consolidated, cleared bucket, and the artifact-
    overwrite target the generation exists to protect (round 6). Reading
    must refresh the TTL so the counter lives as long as the conversation."""

    class _ExpireTrackingRedis(_FakeRedis):
        def __init__(self):
            super().__init__()
            self.expired_keys = []

        async def expire(self, key, ttl):
            self.expired_keys.append((key, ttl))
            return True

    buf = RedisMemoryBuffer.__new__(RedisMemoryBuffer)
    buf.client = _ExpireTrackingRedis()
    buf.tenant_id = "tenant-t"
    buf.tenant_hint = ""
    buf.default_ttl = 3600

    # Generation 0: nothing to keep alive, no refresh.
    assert asyncio.run(buf.get_session_generation("host-uuid-9")) == 0
    assert buf.client.expired_keys == []

    asyncio.run(buf.bump_session_generation("host-uuid-9"))
    buf.client.expired_keys.clear()

    # A live generation is refreshed on every read.
    assert asyncio.run(buf.get_session_generation("host-uuid-9")) == 1
    assert len(buf.client.expired_keys) == 1
    assert buf.client.expired_keys[0][1] == 86400


def test_generation_helpers_degrade_when_the_proxy_is_older():
    """An older proxy server does not know the generation actions; the
    helpers must degrade (generation 0, no rotation) instead of raising —
    the same precedent as only_if/nx (round 6)."""
    buf = RedisMemoryBuffer.__new__(RedisMemoryBuffer)
    buf.client = None
    buf.tenant_id = "tenant-t"
    buf.tenant_hint = ""
    buf.default_ttl = 3600

    async def failing_proxy(*, action, payload):
        raise RuntimeError(f"unknown action: {action}")

    buf._proxy_request = failing_proxy
    assert asyncio.run(buf.get_session_generation("host-uuid-9")) == 0
    assert asyncio.run(buf.bump_session_generation("host-uuid-9")) == 1


def test_a_user_scoped_mint_stamps_identity_metadata(monkeypatch):
    """Consolidation derives the storage space from session metadata, and
    only ingest's first-message path wrote it — a session minted by any
    other identity-scoped call (add_response with user_id, etc.)
    consolidated into a topic-hint space instead of the user's own
    (round 5). The mint itself now stamps user_id/context."""
    buffer = RedisMemoryBuffer(client=_FakeRedis(), redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )
    session_id, source = asyncio.run(
        manager.resolve_session_id(user_id="alice", context="personal")
    )
    assert source == "generated"
    metadata = asyncio.run(
        buffer.get_session_metadata(manager.project, session_id)
    )
    assert metadata.get("user_id") == "alice"
    assert metadata.get("context") == "personal"


def test_a_blank_session_id_is_rejected_not_reinterpreted():
    """"" and "   " are provided values, not omissions. Main pushed them to
    the buffer and raised there; silently resolving them to some other live
    session would move data between buckets on a typo."""
    for blank in ("", "   "):
        with pytest.raises(ValueError):
            asyncio.run(_resolve_session({"session_id": blank}, manager=None))


def test_clear_with_only_if_issues_no_delete_when_the_pointer_is_absent():
    """An absent pointer means there is nothing to clear; issuing the DELETE
    anyway could only destroy a pointer some concurrent resolver sets between
    the read and the delete (round 2)."""
    buf = _buffer_with_fake_client()
    deletes = []
    inner_delete = buf.client.delete

    async def counting_delete(*keys):
        deletes.append(keys)
        return await inner_delete(*keys)

    buf.client.delete = counting_delete
    asyncio.run(buf.clear_active_session(
        context="mcp", user_canonical_id="default", only_if="anything",
    ))
    assert deletes == []


def test_proxy_mode_enforces_only_if_client_side():
    """An older proxy server ignores unknown payload fields, so forwarding
    only_if alone silently restored the unconditional delete on hosted
    deployments while the local path was fixed (round 2). The compare must
    happen before the delete is forwarded, using the main-era
    get_active_session action every server implements."""
    buf = RedisMemoryBuffer.__new__(RedisMemoryBuffer)
    buf.client = None
    buf.tenant_id = "tenant-t"
    buf.tenant_hint = ""
    buf.default_ttl = 3600
    requests = []

    async def fake_proxy(*, action, payload):
        requests.append(action)
        if action == "get_active_session":
            return {"session_id": "live-session"}
        return {}

    buf._proxy_request = fake_proxy

    # Mismatch: the delete must never be forwarded.
    asyncio.run(buf.clear_active_session(
        context="c", user_canonical_id="u", only_if="old-backfill-id",
    ))
    assert "clear_active_session" not in requests

    # Match: the delete goes through.
    requests.clear()
    asyncio.run(buf.clear_active_session(
        context="c", user_canonical_id="u", only_if="live-session",
    ))
    assert "clear_active_session" in requests


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
