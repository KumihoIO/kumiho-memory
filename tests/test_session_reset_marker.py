"""A consolidate (or chat_clear) must never make a session look recreated.

``created_bucket: true`` on a write means ONE thing: that session id has no
working-memory buffer *and never had a deliberate reset* -- it was never used
before, or its buffer was lost to idle-TTL expiry.  Before the reset marker,
``consolidate_session`` ended in ``clear_session`` (buffer + metadata deleted),
so the next write into the very same, unchanged session id saw an empty list,
RPUSH returned 1 and the write reported a fresh bucket.  The host bootstrap tells
the assistant to warn the user whenever that flag is set mid-conversation, so every
consolidate produced a false "your session was recreated" alarm.

Contract pinned here: ``reset_session`` (what consolidate and
``kumiho_chat_clear`` now call) leaves a one-shot marker, written before the buffer
is deleted and only if something is discarded; the first write after it reports
``created_bucket: false`` plus an additive ``buffer_reset_by`` and consumes it.  The
marker lives one working-memory TTL and is single-use, so it never masks a genuine
loss.  It uses only existing proxy actions and fields (the deployed control-plane
schema is strict: an unknown field is a 400).  ``clear_session`` is unchanged.
"""

import asyncio
from unittest.mock import patch

import pytest

import kumiho_memory.mcp_tools as mcp_tools_module
from kumiho_memory import redis_memory
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis
from hosted_fakes import FakeMemoryProxy
from test_memory_manager import (
    AGENT_SUMMARY,
    NeverSummarizer,
    StubRedactor,
    _keyless_manager,
)

#: A host conversation id as the Claude Code bootstrap hands it over: passed
#: explicitly on every call, so it never rotates (unlike the env-derived id).
SESSION = "2a59ba77-7be2-4eec-8f37-9fc2fb397771"
TTL = 600


@pytest.fixture(autouse=True)
def _short_ttl(monkeypatch):
    monkeypatch.setenv("KUMIHO_WORKING_MEMORY_TTL", str(TTL))
    monkeypatch.setattr("kumiho_memory.memory_manager._load_default_retrieve", lambda: None)
    monkeypatch.setattr("kumiho_memory.memory_manager._load_default_store", lambda: None)


@pytest.fixture
def env(tmp_path):
    stored = []
    manager, buffer = _keyless_manager(str(tmp_path), stored)
    return manager, buffer, buffer.client


def _reflect(text, session_id=SESSION):
    return mcp_tools_module.tool_memory_reflect(
        {"response": text, "session_id": session_id},
    )


def _consolidate(session_id=SESSION):
    return mcp_tools_module.tool_memory_consolidate(
        {"session_id": session_id, "summary": AGENT_SUMMARY},
    )


def _count(buffer, manager, session_id=SESSION):
    got = asyncio.run(buffer.get_messages(
        project=manager.project, session_id=session_id, limit=50,
    ))
    return got["message_count"]


def _add(buffer, manager, text="hi", session_id=SESSION, role="user"):
    return buffer.add_message(
        project=manager.project, session_id=session_id, role=role, content=text,
    )


# ---------------------------------------------------------------------------
# The reported defect: reflect -> consolidate -> reflect
# ---------------------------------------------------------------------------


def test_reflect_after_consolidate_keeps_the_session_and_reports_no_new_bucket(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        first = _reflect("first turn")
        consolidated = _consolidate()
        assert consolidated["success"] is True, consolidated
        assert consolidated["session_id"] == SESSION
        # consolidate really did empty the buffer -- the session id is the same
        # and the buffer behind it is empty, which is the situation that used to
        # read as a recreated session.
        assert _count(buffer, manager) == 0
        second = _reflect("second turn")
        third = _reflect("third turn")

    assert first["created_bucket"] is True  # genuinely the first write ever
    assert "buffer_reset_by" not in first
    assert second["session_id"] == first["session_id"] == SESSION
    assert second["session_id_source"] == "argument"
    assert second["created_bucket"] is False, second
    assert second["buffer_reset_by"] == "consolidate"
    assert third["created_bucket"] is False
    assert "buffer_reset_by" not in third  # the marker is single-use
    assert _count(buffer, manager) == 2


def test_repeated_consolidates_never_raise_the_alarm(env):
    """The owner saw it on every consolidate, six times in one day."""
    manager, buffer, _fake = env
    flags = []
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn 0")
        for n in range(1, 5):
            assert _consolidate()["success"] is True
            flags.append(_reflect(f"turn {n}")["created_bucket"])
    assert flags == [False, False, False, False]


def test_a_genuinely_new_session_still_reports_a_fresh_bucket(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        fresh = _reflect("hello", session_id="never-seen-before")
        # ...and an unrelated session's consolidate leaves it a fresh one.
        _reflect("elsewhere")
        _consolidate()
        still_fresh = _reflect("hello again", session_id="another-new-id")
    assert fresh["created_bucket"] is True
    assert "buffer_reset_by" not in fresh
    assert still_fresh["created_bucket"] is True
    assert "buffer_reset_by" not in still_fresh


# ---------------------------------------------------------------------------
# TTL expiry must still be detectable
# ---------------------------------------------------------------------------


def test_idle_ttl_expiry_of_a_live_buffer_still_reports_a_fresh_bucket(env):
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("before the pause")
        fake.advance(TTL + 1)
        after = _reflect("after the pause")
    assert after["created_bucket"] is True


def test_consolidate_then_idle_past_the_ttl_is_a_genuine_loss(env):
    """consolidate, then the session idles past the TTL: the marker expires on the
    same clock the buffer would have, so a write this late is a true loss."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        _consolidate()
        fake.advance(TTL + 1)
        after = _reflect("much later")
    assert after["created_bucket"] is True
    assert "buffer_reset_by" not in after


def test_the_marker_is_honoured_for_a_full_ttl(env):
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        _consolidate()
        fake.advance(TTL - 1)
        after = _reflect("back within the TTL")
    assert after["created_bucket"] is False
    assert after["buffer_reset_by"] == "consolidate"


def test_ttl_loss_after_a_reset_is_not_masked_by_the_spent_marker(env):
    """reflect -> consolidate -> reflect (marker spent) -> idle past TTL: the
    post-consolidate buffer is lost to expiry, and that is the true signal."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        _consolidate()
        assert _reflect("resumed")["buffer_reset_by"] == "consolidate"
        fake.advance(TTL + 1)
        after = _reflect("after the pause")
    assert after["created_bucket"] is True
    assert "buffer_reset_by" not in after


def test_a_buffer_lost_without_any_reset_is_not_excused_by_an_old_marker(env):
    """Eviction (or a flush) after the marker was spent must still read as loss,
    even though the marker's own TTL has not elapsed."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        _consolidate()
        assert _reflect("resumed")["created_bucket"] is False
        fake.storage.pop(buffer._session_messages_key(manager.project, SESSION))
        after = _reflect("after the eviction")
    assert after["created_bucket"] is True


def test_consolidating_a_buffer_that_already_idled_out_is_still_a_genuine_loss(env):
    """The keyless consolidate stores the agent's summary even when the buffer
    expired during a long turn.  Nothing was discarded by that consolidate, so it
    must not relabel the earlier TTL loss as "reset by consolidate"."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("before the pause")
        fake.advance(TTL + 1)
        consolidated = _consolidate()
        assert consolidated["success"] is True, consolidated
        assert consolidated["buffer_was_empty"] is True
        after = _reflect("after the pause")
    assert after["created_bucket"] is True
    assert "buffer_reset_by" not in after


def test_a_second_consolidate_with_nothing_new_keeps_the_pending_marker(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        assert _consolidate()["success"] is True
        again = _consolidate()
        assert again["success"] is True and again["buffer_was_empty"] is True
        after = _reflect("next turn")
    assert after["created_bucket"] is False
    assert after["buffer_reset_by"] == "consolidate"


def test_clearing_a_session_nobody_used_is_not_a_reset(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        cleared = mcp_tools_module.tool_chat_clear({"session_id": "never-used"})
        assert cleared["cleared_count"] == 0
        first = _reflect("hello", session_id="never-used")
    assert first["created_bucket"] is True
    assert "buffer_reset_by" not in first


# ---------------------------------------------------------------------------
# chat_clear: a deliberate reset of a continuing session
# ---------------------------------------------------------------------------


def test_chat_clear_is_a_deliberate_reset_not_a_lost_session(env):
    """DECISION: ``kumiho_chat_clear`` keeps the session id and empties its
    buffer exactly as consolidate does, so by the definition of the flag (never
    had a buffer, or lost it to idle expiry) the next write is not a new bucket.
    It is reported as ``buffer_reset_by: "clear"``."""
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        cleared = mcp_tools_module.tool_chat_clear({"session_id": SESSION})
        assert cleared["success"] is True
        assert cleared["cleared_count"] == 1
        assert cleared["session_id"] == SESSION
        after = _reflect("after the clear")
    assert after["created_bucket"] is False
    assert after["buffer_reset_by"] == "clear"


def test_the_low_level_clear_session_is_still_a_full_wipe(env):
    """``clear_session`` keeps its contract (and is what the existing
    TTL-elapsed simulation in the consolidation tests relies on): no marker."""
    manager, buffer, _fake = env

    async def run():
        await _add(buffer, manager, "one")
        cleared = await buffer.clear_session(manager.project, SESSION)
        assert cleared == {"success": True, "cleared_count": 1}
        return await _add(buffer, manager, "two")

    result = asyncio.run(run())
    assert result["created_bucket"] is True
    assert "buffer_reset_by" not in result


# ---------------------------------------------------------------------------
# Concurrency: PR #4 round 3 (the atomic RPUSH count is the mint signal)
# ---------------------------------------------------------------------------


class _YieldingClient:
    """Delegates to a fake Redis, yielding to the loop before every command.

    Concurrent coroutines then interleave command by command, like two clients on
    one Redis server.  ``on_command(n)`` (optional) is called synchronously as the
    n-th command starts -- a deterministic way to launch a second client at an
    exact point of another coroutine's command stream.
    """

    def __init__(self, inner, on_command=None):
        self._inner = inner
        self._on_command = on_command
        self.commands = 0

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        async def command(*args, **kwargs):
            await asyncio.sleep(0)
            self.commands += 1
            if self._on_command is not None:
                self._on_command(self.commands)
            return await attribute(*args, **kwargs)

        return command


def _yielding(buffer, on_command=None):
    client = _YieldingClient(buffer.client, on_command)
    buffer.client = client
    return client


@pytest.mark.parametrize("yielding", [False, True], ids=["atomic-fake", "interleaving-fake"])
def test_two_appends_right_after_consolidate_do_not_both_report_a_fresh_bucket(env, yielding):
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        _consolidate()
    if yielding:
        _yielding(buffer)

    async def race():
        return await asyncio.gather(*[
            _add(buffer, manager, f"racer {n}") for n in range(2)
        ])

    results = asyncio.run(race())
    assert [r["created_bucket"] for r in results] == [False, False], results
    assert sorted(r["message_count"] for r in results) == [1, 2]
    # exactly one writer was first, so exactly one consumed the marker
    assert sum(1 for r in results if r.get("buffer_reset_by")) == 1


def test_many_concurrent_appends_after_consolidate_consume_the_marker_once(env):
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        _consolidate()
    _yielding(buffer)

    async def race():
        return await asyncio.gather(*[
            _add(buffer, manager, f"racer {n}") for n in range(8)
        ])

    results = asyncio.run(race())
    assert not any(r["created_bucket"] for r in results), results
    assert sorted(r["message_count"] for r in results) == list(range(1, 9))
    assert sum(1 for r in results if r.get("buffer_reset_by") == "consolidate") == 1


def test_no_interleaving_of_a_write_with_consolidate_reads_as_a_recreated_session(tmp_path):
    """The marker is written BEFORE the buffer is deleted.  Written after, a write
    landing between the delete and the marker saw an empty buffer and no marker --
    the false alarm, in a window.  The consolidating client is slow (every Redis
    command yields); a second, fast client writes into the same session at every
    position of the consolidate's command stream, and once after it finished."""

    async def scenario(launch_at, tag):
        manager, buffer = _keyless_manager(str(tmp_path / tag), [])
        fake = buffer.client
        fast = RedisMemoryBuffer(client=fake, redis_url="redis://test")
        await _add(buffer, manager, "user turn")
        await _add(buffer, manager, "assistant turn", role="assistant")
        late = []

        async def late_write():
            return await _add(fast, manager, "late write")

        def on_command(n):
            if n == launch_at and not late:
                late.append(asyncio.get_running_loop().create_task(late_write()))

        slow = _yielding(buffer, on_command)
        consolidated = await manager.consolidate_session(
            session_id=SESSION, summary=AGENT_SUMMARY,
        )
        assert consolidated["success"] is True, consolidated
        if not late:  # launched after the consolidate has completely finished
            late.append(asyncio.get_running_loop().create_task(late_write()))
        return await late[0], slow.commands

    _result, total_commands = asyncio.run(scenario(None, "dry"))
    assert total_commands > 5  # the sweep below really spans a command stream
    for launch_at in range(1, total_commands + 2):
        late_result, _ = asyncio.run(scenario(launch_at, f"at{launch_at}"))
        assert late_result["created_bucket"] is False, (launch_at, late_result)


# ---------------------------------------------------------------------------
# The response stays additive
# ---------------------------------------------------------------------------

_BASE_KEYS = {"success", "message_id", "message_count", "created_bucket"}


def test_the_response_shape_only_ever_grows_by_one_field(env):
    manager, buffer, _fake = env

    async def run():
        first = await _add(buffer, manager, "one")
        second = await _add(buffer, manager, "two")
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")
        after_reset = await _add(buffer, manager, "three")
        later = await _add(buffer, manager, "four")
        return first, second, after_reset, later

    first, second, after_reset, later = asyncio.run(run())
    assert set(first) == set(second) == set(later) == _BASE_KEYS
    assert set(after_reset) == _BASE_KEYS | {"buffer_reset_by"}
    assert (first["created_bucket"], second["created_bucket"]) == (True, False)
    assert (after_reset["created_bucket"], later["created_bucket"]) == (False, False)
    assert after_reset["message_count"] == 1  # still a one-message buffer


def test_reset_session_returns_the_clear_session_result_shape(env):
    manager, buffer, _fake = env

    async def run():
        await _add(buffer, manager, "one")
        await _add(buffer, manager, "two")
        return await buffer.reset_session(manager.project, SESSION, reason="clear")

    assert asyncio.run(run()) == {"success": True, "cleared_count": 2}


# ---------------------------------------------------------------------------
# A marker that cannot be written degrades to the old behaviour, never worse
# ---------------------------------------------------------------------------


def test_a_failed_marker_write_does_not_fail_or_skip_the_consolidate(env, caplog):
    manager, buffer, _fake = env
    real_set = buffer.set_session_metadata

    async def flaky_set(project, session_id, metadata, **kwargs):
        if session_id.endswith(redis_memory.RESET_MARKER_SUFFIX):
            raise RuntimeError("proxy refused the write")
        return await real_set(project, session_id, metadata, **kwargs)

    buffer.set_session_metadata = flaky_set
    with patch.object(mcp_tools_module, "_manager", manager):
        _reflect("turn")
        consolidated = _consolidate()
        assert consolidated["success"] is True, consolidated
        assert _count(buffer, manager) == 0  # the buffer was still cleared
        after = _reflect("next")
    assert after["created_bucket"] is True  # pre-marker behaviour, not a crash
    assert "reset marker" in caplog.text


def test_an_unreadable_marker_degrades_to_a_fresh_bucket(env):
    manager, buffer, _fake = env

    async def broken_get(project, session_id):
        raise RuntimeError("redis hiccup")

    async def run():
        await _add(buffer, manager, "one")
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")
        buffer.get_session_metadata = broken_get
        return await _add(buffer, manager, "two")

    assert asyncio.run(run())["created_bucket"] is True


# ---------------------------------------------------------------------------
# Hosted proxy: no new action, no new field, no longer-lived marker than the buffer
# ---------------------------------------------------------------------------

#: The 12 actions kumiho-memory sends (WORK_PLAN P2-min: the session API contract).
PROXY_ACTIONS = {
    "add_message", "get_messages", "set_session_metadata", "get_session_metadata",
    "clear_session", "list_sessions", "next_sequence", "get_active_session",
    "set_active_session", "clear_active_session", "get_session_generation",
    "bump_session_generation",
}
#: Body fields of the deployed control-plane schema (kumiho-control
#: packages/origin/src/app/api/memory/redis/route.ts, ``z.strictObject``): any
#: other key makes the proxy answer 400 invalid_payload.
PROXY_FIELDS = {
    "action", "project", "session_id", "base_session_id", "role", "content",
    "metadata", "limit", "offset", "default_ttl", "user_canonical_id", "date_str",
    "ttl_seconds", "context", "nx", "only_if", "tenant_hint",
}


class _DeployedProxy(FakeMemoryProxy):
    """The deployed control plane: ``add_message`` returns no ``created_bucket``
    (the client derives it from ``message_count``) and the body is strict."""

    def _dispatch(self, state, action, body):
        assert action in PROXY_ACTIONS, f"new proxy action {action!r}"
        unknown = set(body) - PROXY_FIELDS
        assert not unknown, f"{action}: fields the strict schema rejects: {sorted(unknown)}"
        result = super()._dispatch(state, action, body)
        if action == "add_message":
            result.pop("created_bucket", None)
        return result


@pytest.fixture
def proxied(monkeypatch, tmp_path):
    for var in ("UPSTASH_REDIS_URL", "KUMIHO_UPSTASH_REDIS_URL", "KUMIHO_MEMORY_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    fake = _DeployedProxy()
    monkeypatch.setattr(redis_memory.requests, "post", fake)
    monkeypatch.setattr(
        RedisMemoryBuffer, "_get_fresh_token", staticmethod(lambda **kwargs: "tok"),
    )
    buffer = RedisMemoryBuffer(
        proxy_url="https://cp.test.invalid/api/memory/redis",
        tenant_id="tenant-p",
        prefer_discovery=False,
    )
    stored = []

    async def store_stub(**kwargs):
        stored.append(kwargs)
        return {"item_kref": "kref://m/i", "revision_kref": "kref://m/i?r=1"}

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=NeverSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=store_stub,
        consolidation_threshold=2,
        artifact_root=str(tmp_path),
    )
    return manager, buffer, fake


def test_hosted_proxy_consolidate_then_reflect_reports_no_new_bucket(proxied):
    manager, buffer, fake = proxied
    with patch.object(mcp_tools_module, "_manager", manager):
        first = _reflect("turn")
        assert _consolidate()["success"] is True
        second = _reflect("after")
        third = _reflect("again")
        fresh = _reflect("elsewhere", session_id="a-genuinely-new-id")
    assert first["created_bucket"] is True
    assert (second["created_bucket"], second["buffer_reset_by"]) == (False, "consolidate")
    assert third["created_bucket"] is False and "buffer_reset_by" not in third
    assert fresh["created_bucket"] is True
    # every call that reached the proxy used an existing action and field
    assert {c["action"] for c in fake.calls} <= PROXY_ACTIONS


def test_hosted_proxy_marker_carries_the_buffer_ttl_and_other_calls_are_unchanged(proxied):
    manager, buffer, fake = proxied

    async def run():
        await _add(buffer, manager, "one")
        await buffer.set_session_metadata(manager.project, SESSION, {"user_id": "u"})
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")

    asyncio.run(run())
    sets = [c["body"] for c in fake.calls if c["action"] == "set_session_metadata"]
    ordinary = [b for b in sets if not b["session_id"].endswith(redis_memory.RESET_MARKER_SUFFIX)]
    marker = [b for b in sets if b["session_id"].endswith(redis_memory.RESET_MARKER_SUFFIX)]
    # an ordinary metadata write is byte-for-byte what it was (no new field)
    assert ordinary == [{
        "action": "set_session_metadata", "project": manager.project,
        "session_id": SESSION, "metadata": {"user_id": "u"},
    }]
    # the marker asks for the buffer's own TTL, not the proxy's 3600 default
    assert len(marker) == 1 and marker[0]["default_ttl"] == TTL
    # a reset is: look at the buffer, write the marker, then delete the buffer
    reset_calls = [c["action"] for c in fake.calls][-3:]
    assert reset_calls == ["get_messages", "set_session_metadata", "clear_session"]
    # the marker is written before the buffer is deleted
    actions = [c["action"] for c in fake.calls]
    first_marker = next(
        i for i, c in enumerate(fake.calls)
        if c["action"] == "set_session_metadata"
        and c["body"]["session_id"].endswith(redis_memory.RESET_MARKER_SUFFIX)
    )
    assert first_marker < actions.index("clear_session")


def test_hosted_proxy_clear_session_payload_is_unchanged(proxied):
    manager, buffer, fake = proxied
    asyncio.run(buffer.clear_session(manager.project, SESSION))
    (call,) = [c for c in fake.calls if c["action"] == "clear_session"]
    assert call["body"] == {
        "action": "clear_session", "project": manager.project, "session_id": SESSION,
    }
