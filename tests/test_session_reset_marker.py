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
marker is only pending while nothing was written since, so it lives
``max(TTL, 24 h)`` without ever hiding a lost turn.  ``clear_session`` is unchanged.
Races and late turns: ``test_session_reset_marker_races.py``; hosted proxy wire:
``test_session_reset_marker_hosted.py``.
"""

import asyncio
from unittest.mock import patch

import pytest

import kumiho_memory.mcp_tools as mcp_tools_module
from kumiho_memory import redis_memory
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis
from reset_marker_support import (
    MARKER_TTL, SESSION, TTL, _short_ttl, add, consolidate, count, env, reflect,
)
from test_memory_manager import StubRedactor, StubSummarizer
from test_session_resolution import _FakeRedis

_FIXTURES = (_short_ttl, env)  # imported fixtures: pytest collects them from here


# ---------------------------------------------------------------------------
# The reported defect: reflect -> consolidate -> reflect
# ---------------------------------------------------------------------------


def test_reflect_after_consolidate_keeps_the_session_and_reports_no_new_bucket(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        first = reflect("first turn")
        consolidated = consolidate()
        assert consolidated["success"] is True, consolidated
        assert consolidated["session_id"] == SESSION
        # consolidate really did empty the buffer -- the session id is the same
        # and the buffer behind it is empty, which is the situation that used to
        # read as a recreated session.
        assert count(buffer, manager) == 0
        second = reflect("second turn")
        third = reflect("third turn")

    assert first["created_bucket"] is True  # genuinely the first write ever
    assert "buffer_reset_by" not in first
    assert second["session_id"] == first["session_id"] == SESSION
    assert second["session_id_source"] == "argument"
    assert second["created_bucket"] is False, second
    assert second["buffer_reset_by"] == "consolidate"
    assert third["created_bucket"] is False
    assert "buffer_reset_by" not in third  # the marker is single-use
    assert count(buffer, manager) == 2


def test_repeated_consolidates_never_raise_the_alarm(env):
    """The owner saw it on every consolidate, six times in one day."""
    manager, buffer, _fake = env
    flags = []
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn 0")
        for n in range(1, 5):
            assert consolidate()["success"] is True
            flags.append(reflect(f"turn {n}")["created_bucket"])
    assert flags == [False, False, False, False]


def test_a_genuinely_new_session_still_reports_a_fresh_bucket(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        fresh = reflect("hello", session_id="never-seen-before")
        # ...and an unrelated session's consolidate leaves it a fresh one.
        reflect("elsewhere")
        consolidate()
        still_fresh = reflect("hello again", session_id="another-new-id")
    assert fresh["created_bucket"] is True
    assert "buffer_reset_by" not in fresh
    assert still_fresh["created_bucket"] is True
    assert "buffer_reset_by" not in still_fresh


# ---------------------------------------------------------------------------
# TTL: a true loss is still a loss, a pause after a consolidate is not
# ---------------------------------------------------------------------------


def test_idle_ttl_expiry_of_a_live_buffer_still_reports_a_fresh_bucket(env):
    """Real loss, no consolidate involved: the buffer held turns and idled out."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("before the pause")
        fake.advance(TTL + 1)
        after = reflect("after the pause")
    assert after["created_bucket"] is True
    assert "buffer_reset_by" not in after


def test_a_pause_longer_than_the_buffer_ttl_after_a_consolidate_is_not_a_lost_session(env):
    """consolidate -> idle past the buffer TTL -> reflect.  Nothing was written
    after the consolidate, so there was nothing left to lose: the turns are
    consolidated.  This is the flow the reflex nudge recommends ("consolidate
    before a long pause"); it must not raise the alarm either."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        consolidate()
        fake.advance(TTL + 1)
        after = reflect("back after the pause")
    assert after["created_bucket"] is False
    assert after["buffer_reset_by"] == "consolidate"


def test_the_marker_lives_a_day_not_longer(env):
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        consolidate()
        fake.advance(MARKER_TTL - 1)
        within = reflect("just inside the day")
    assert within["created_bucket"] is False

    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn", session_id="other")
        consolidate(session_id="other")
        fake.advance(MARKER_TTL + 1)
        beyond = reflect("a day and a second later", session_id="other")
    assert beyond["created_bucket"] is True
    assert "buffer_reset_by" not in beyond


@pytest.mark.parametrize("buffer_ttl,expected", [(TTL, MARKER_TTL), (172800, 172800)])
def test_the_marker_ttl_is_the_longer_of_the_buffer_ttl_and_a_day(
    buffer_ttl, expected, monkeypatch,
):
    monkeypatch.setenv("KUMIHO_WORKING_MEMORY_TTL", str(buffer_ttl))
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def run():
        await buffer.add_message(project="P", session_id="s", role="user", content="x")
        await buffer.reset_session("P", "s", reason="consolidate")

    asyncio.run(run())
    key = buffer._session_metadata_key("P", "s" + redis_memory.RESET_MARKER_SUFFIX)
    assert fake.ttl_store[key] == expected


def test_ttl_loss_after_a_reset_is_not_masked_by_the_spent_marker(env):
    """reflect -> consolidate -> reflect (marker spent) -> idle past TTL: the
    post-consolidate buffer is lost to expiry, and that is the true signal."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        consolidate()
        assert reflect("resumed")["buffer_reset_by"] == "consolidate"
        fake.advance(TTL + 1)
        after = reflect("after the pause")
    assert after["created_bucket"] is True
    assert "buffer_reset_by" not in after


def test_a_buffer_lost_without_any_reset_is_not_excused_by_an_old_marker(env):
    """Eviction (or a flush) after the marker was spent must still read as loss,
    even though the marker's own TTL has not elapsed."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        consolidate()
        assert reflect("resumed")["created_bucket"] is False
        fake.storage.pop(buffer._session_messages_key(manager.project, SESSION))
        after = reflect("after the eviction")
    assert after["created_bucket"] is True


def test_consolidating_a_buffer_that_already_idled_out_is_still_a_genuine_loss(env):
    """The keyless consolidate stores the agent's summary even when the buffer
    expired during a long turn.  Nothing was discarded by that consolidate, so it
    must not relabel the earlier TTL loss as "reset by consolidate"."""
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("before the pause")
        fake.advance(TTL + 1)
        consolidated = consolidate()
        assert consolidated["success"] is True, consolidated
        assert consolidated["buffer_was_empty"] is True
        after = reflect("after the pause")
    assert after["created_bucket"] is True
    assert "buffer_reset_by" not in after


def test_a_second_consolidate_with_nothing_new_keeps_the_pending_marker(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        assert consolidate()["success"] is True
        again = consolidate()
        assert again["success"] is True and again["buffer_was_empty"] is True
        after = reflect("next turn")
    assert after["created_bucket"] is False
    assert after["buffer_reset_by"] == "consolidate"


def test_clearing_a_session_nobody_used_is_not_a_reset(env):
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        cleared = mcp_tools_module.tool_chat_clear({"session_id": "never-used"})
        assert cleared["cleared_count"] == 0
        first = reflect("hello", session_id="never-used")
    assert first["created_bucket"] is True
    assert "buffer_reset_by" not in first


# ---------------------------------------------------------------------------
# chat_clear: a deliberate reset of a continuing session
# ---------------------------------------------------------------------------


def test_chat_clear_is_a_deliberate_reset_not_a_lost_session(env):
    """DECISION (owner, 2026-11-02): ``kumiho_chat_clear`` keeps the session id
    and empties its buffer exactly as consolidate does, so by the definition of
    the flag the next write is not a new bucket.  ``buffer_reset_by: "clear"``."""
    manager, buffer, _fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        cleared = mcp_tools_module.tool_chat_clear({"session_id": SESSION})
        assert cleared["success"] is True
        assert cleared["cleared_count"] == 1
        assert cleared["session_id"] == SESSION
        after = reflect("after the clear")
    assert after["created_bucket"] is False
    assert after["buffer_reset_by"] == "clear"


def test_the_low_level_clear_session_is_still_a_full_wipe(env):
    """``clear_session`` keeps its contract (and is what the existing
    TTL-elapsed simulation in the consolidation tests relies on): no marker."""
    manager, buffer, _fake = env

    async def run():
        await add(buffer, manager, "one")
        cleared = await buffer.clear_session(manager.project, SESSION)
        assert cleared == {"success": True, "cleared_count": 1}
        return await add(buffer, manager, "two")

    result = asyncio.run(run())
    assert result["created_bucket"] is True
    assert "buffer_reset_by" not in result


# ---------------------------------------------------------------------------
# The id the host derives from its environment: consolidation rotates it
# ---------------------------------------------------------------------------


def _env_manager(monkeypatch):
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-uuid-9")
    buffer = RedisMemoryBuffer(client=_FakeRedis(), redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer, summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(), memory_store=None,
    )
    return manager


def test_the_rotated_env_id_carries_the_reset_marker_forward(monkeypatch):
    """Consolidating an env-derived session rotates its id to ``<id>:c<n>``.  The
    first write on the rotated id is the same conversation continuing, so it
    reports ``created_bucket: false`` + ``buffer_reset_by``, not a fresh bucket."""
    manager = _env_manager(monkeypatch)

    async def scenario():
        first, _ = await manager.resolve_session_id()
        await manager.add_assistant_response(session_id=first, response="hello")
        await manager.consolidate_session(session_id=first)
        rotated, source = await manager.resolve_session_id()
        after = await manager.add_assistant_response(session_id=rotated, response="again")
        return first, rotated, source, after

    first, rotated, source, after = asyncio.run(scenario())
    assert (first, rotated, source) == ("host-uuid-9", "host-uuid-9:c1", "host-env")
    assert after["created_bucket"] is False, after
    assert after["buffer_reset_by"] == "consolidate"


def test_an_explicit_caller_keeps_its_marker_while_the_generation_rotates(monkeypatch):
    """The Claude Code bootstrap passes the host id EXPLICITLY, and that id is
    also the env value: consolidate bumps the generation (the explicit caller
    never follows it) and the caller keeps writing to the same id.  The marker
    therefore stays on the id that was reset, as well as being carried forward."""
    manager = _env_manager(monkeypatch)

    async def scenario():
        await manager.add_assistant_response(session_id="host-uuid-9", response="hello")
        await manager.consolidate_session(session_id="host-uuid-9")
        return await manager.add_assistant_response(
            session_id="host-uuid-9", response="again",
        )

    after = asyncio.run(scenario())
    assert after["created_bucket"] is False, after
    assert after["buffer_reset_by"] == "consolidate"


# ---------------------------------------------------------------------------
# The id space: "<id>:~reset" is reserved for markers
# ---------------------------------------------------------------------------


def test_a_session_id_ending_in_the_reserved_suffix_is_refused_on_write(env):
    manager, buffer, _fake = env
    with pytest.raises(ValueError, match="reserved"):
        asyncio.run(add(buffer, manager, "x", session_id=SESSION + ":~reset"))
    assert redis_memory.RESET_MARKER_SUFFIX == ":~reset"


@pytest.mark.parametrize("session_id", [
    SESSION,
    "km1_0133494e0123456789abcdef01234567_" + "ab" * 32,
    "host-uuid-9:c3",
    "personal:user-ab12cd34ef:20261102:001",
    "~reset",
    "x:~resetx",
])
def test_every_id_the_system_mints_or_hosts_hand_over_is_still_accepted(env, session_id):
    manager, buffer, _fake = env
    assert asyncio.run(add(buffer, manager, "x", session_id=session_id))["created_bucket"] is True


# ---------------------------------------------------------------------------
# The response stays additive
# ---------------------------------------------------------------------------

_BASE_KEYS = {"success", "message_id", "message_count", "created_bucket"}


def test_the_response_shape_only_ever_grows_by_one_field(env):
    manager, buffer, _fake = env

    async def run():
        first = await add(buffer, manager, "one")
        second = await add(buffer, manager, "two")
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")
        after_reset = await add(buffer, manager, "three")
        later = await add(buffer, manager, "four")
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
        await add(buffer, manager, "one")
        await add(buffer, manager, "two")
        return await buffer.reset_session(manager.project, SESSION, reason="clear")

    assert asyncio.run(run()) == {"success": True, "cleared_count": 2}


# ---------------------------------------------------------------------------
# A marker that cannot be written or read degrades to the old behaviour
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
        reflect("turn")
        consolidated = consolidate()
        assert consolidated["success"] is True, consolidated
        assert count(buffer, manager) == 0  # the buffer was still cleared
        after = reflect("next")
    assert after["created_bucket"] is True  # pre-marker behaviour, not a crash
    assert "reset marker" in caplog.text


def test_an_unreadable_marker_degrades_to_a_fresh_bucket(env):
    manager, buffer, _fake = env

    async def broken_get(project, session_id):
        raise RuntimeError("redis hiccup")

    async def run():
        await add(buffer, manager, "one")
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")
        buffer.get_session_metadata = broken_get
        return await add(buffer, manager, "two")

    assert asyncio.run(run())["created_bucket"] is True


def test_an_unreadable_buffer_length_still_leaves_the_marker(env):
    """If the length cannot be read the reset assumes the session was in use: a
    false alarm is the worse error (documented trade-off, see _message_count)."""
    manager, buffer, fake = env
    real_llen = fake.llen
    calls = {"n": 0}

    async def flaky_llen(key):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("redis hiccup")
        return await real_llen(key)

    async def run():
        await add(buffer, manager, "one")
        fake.llen = flaky_llen
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")
        return await add(buffer, manager, "two")

    after = asyncio.run(run())
    assert after["created_bucket"] is False
    assert after["buffer_reset_by"] == "consolidate"
