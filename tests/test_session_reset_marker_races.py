"""Reset-marker races: concurrent writers, turns that land mid-consolidate, and a
marker that is consumed or replaced while someone else is looking at it.

The invariants (see ``test_session_reset_marker.py`` for the basic contract):

* only the writer whose RPUSH returns 1 looks at the marker, so two appends right
  after a consolidate cannot both report a fresh bucket or both consume it
  (PR #4 round 3);
* a turn appended after a consolidate's read was never summarized: on a direct
  Redis it is KEPT (atomic LTRIM), and it is never silently excused -- the
  marker is not written (or is withdrawn) whenever the buffer is not empty;
* a consume cannot overwrite a newer reset (``reset_id`` nonce), and a consume
  that fails removes the marker instead of leaving it pending over a live buffer.
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
    SESSION, _short_ttl, add, consolidate, contents, count, env, read_contents, reflect,
    yielding,
)
from test_memory_manager import AGENT_SUMMARY, NeverSummarizer, StubRedactor, _keyless_manager

_FIXTURES = (_short_ttl, env)  # imported fixtures: pytest collects them from here
MARKER_ID = SESSION + redis_memory.RESET_MARKER_SUFFIX


def _marker(buffer, manager):
    return asyncio.run(buffer.get_session_metadata(manager.project, MARKER_ID))


# ---------------------------------------------------------------------------
# Concurrent writers right after a consolidate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interleaving", [False, True], ids=["atomic-fake", "interleaving-fake"])
def test_two_appends_right_after_consolidate_do_not_both_report_a_fresh_bucket(env, interleaving):
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        consolidate()
    if interleaving:
        yielding(buffer)

    async def race():
        return await asyncio.gather(*[add(buffer, manager, f"racer {n}") for n in range(2)])

    results = asyncio.run(race())
    assert [r["created_bucket"] for r in results] == [False, False], results
    assert sorted(r["message_count"] for r in results) == [1, 2]
    # exactly one writer was first, so exactly one consumed the marker
    assert sum(1 for r in results if r.get("buffer_reset_by")) == 1


def test_many_concurrent_appends_after_consolidate_consume_the_marker_once(env):
    manager, buffer, fake = env
    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        consolidate()
    yielding(buffer)

    async def race():
        return await asyncio.gather(*[add(buffer, manager, f"racer {n}") for n in range(8)])

    results = asyncio.run(race())
    assert not any(r["created_bucket"] for r in results), results
    assert sorted(r["message_count"] for r in results) == list(range(1, 9))
    assert sum(1 for r in results if r.get("buffer_reset_by") == "consolidate") == 1


# ---------------------------------------------------------------------------
# A writer landing anywhere inside a consolidate
# ---------------------------------------------------------------------------


def test_a_write_anywhere_inside_a_consolidate_is_neither_lost_nor_called_a_recreated_session(tmp_path):
    """The marker is written BEFORE the buffer is deleted (else a write between the
    delete and the marker saw an empty buffer and no marker: the false alarm, in a
    window), and a write that lands after the consolidate's read must survive.  The
    consolidating client is slow (every Redis command yields); a second, fast client
    writes into the same session at every position of the consolidate's command
    stream, and once after it finished.  The late turn is always either in the
    stored transcript, or still in the buffer -- never deleted unseen."""

    async def scenario(launch_at, tag):
        stored = []
        manager, buffer = _keyless_manager(str(tmp_path / tag), stored)
        fast = RedisMemoryBuffer(client=buffer.client, redis_url="redis://test")
        await add(buffer, manager, "user turn")
        await add(buffer, manager, "assistant turn", role="assistant")
        late = []

        async def late_write():
            return await add(fast, manager, "late write")

        def on_command(n):
            if n == launch_at and not late:
                late.append(asyncio.get_running_loop().create_task(late_write()))

        slow = yielding(buffer, on_command)
        consolidated = await manager.consolidate_session(
            session_id=SESSION, summary=AGENT_SUMMARY,
        )
        assert consolidated["success"] is True, consolidated
        if not late:  # launched after the consolidate has completely finished
            late.append(asyncio.get_running_loop().create_task(late_write()))
        late_result = await late[0]
        remaining = await buffer.get_messages(
            project=manager.project, session_id=SESSION, limit=50,
        )
        buffered = [m["content"] for m in remaining["messages"]]
        return late_result, stored[0]["user_text"], buffered, slow.commands

    _result, _text, _buffered, total_commands = asyncio.run(scenario(None, "dry"))
    assert total_commands > 5  # the sweep below really spans a command stream
    for launch_at in range(1, total_commands + 2):
        late_result, transcript, buffered, _ = asyncio.run(scenario(launch_at, f"at{launch_at}"))
        assert late_result["created_bucket"] is False, (launch_at, late_result)
        assert "late write" in transcript or "late write" in buffered, (launch_at, buffered)


# ---------------------------------------------------------------------------
# Turns appended while the consolidate was summarizing
# ---------------------------------------------------------------------------


def _manager_with_late_turn(tmp_path, buffer):
    """A manager whose graph write appends a turn, as a concurrent client would."""
    stored = []
    manager = UniversalMemoryManager(
        redis_buffer=buffer, summarizer=NeverSummarizer(), pii_redactor=StubRedactor(),
        memory_store=None, consolidation_threshold=2, artifact_root=str(tmp_path),
    )

    async def store_stub(**kwargs):
        stored.append(kwargs)
        await buffer.add_message(
            project=manager.project, session_id=SESSION, role="user", content="late turn",
        )
        return {"item_kref": "kref://m/i", "revision_kref": "kref://m/i?r=1"}

    manager.memory_store = store_stub
    return manager, stored


def test_a_turn_appended_during_a_consolidate_is_kept_and_not_excused(tmp_path):
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    manager, stored = _manager_with_late_turn(tmp_path, buffer)

    async def run():
        await add(buffer, manager, "user turn")
        await add(buffer, manager, "assistant turn", role="assistant")
        consolidated = await manager.consolidate_session(session_id=SESSION, summary=AGENT_SUMMARY)
        kept = await buffer.get_messages(project=manager.project, session_id=SESSION, limit=50)
        marker = await buffer.get_session_metadata(manager.project, MARKER_ID)
        follow = await add(buffer, manager, "next turn")
        fake.storage.pop(buffer._session_messages_key(manager.project, SESSION))
        lost = await add(buffer, manager, "after a real loss")
        return consolidated, kept, marker, follow, lost

    consolidated, kept, marker, follow, lost = asyncio.run(run())
    assert consolidated["success"] is True
    assert "late turn" not in stored[0]["user_text"]  # it was never summarized...
    assert [m["content"] for m in kept["messages"]] == ["late turn"]  # ...so it was kept
    assert marker == {}  # the buffer is not empty: nothing to excuse
    assert (follow["created_bucket"], follow["message_count"]) == (False, 2)
    assert "buffer_reset_by" not in follow
    assert lost["created_bucket"] is True  # a later real loss is not masked


def test_reset_trims_only_what_the_caller_read(env):
    manager, buffer, _fake = env

    async def run():
        for n in range(3):
            await add(buffer, manager, f"m{n}")
        partial = await buffer.reset_session(
            manager.project, SESSION, reason="consolidate", expected_count=1,
        )
        kept = await read_contents(buffer, manager)
        marker_after_partial = await buffer.get_session_metadata(manager.project, MARKER_ID)
        full = await buffer.reset_session(
            manager.project, SESSION, reason="consolidate", expected_count=2,
        )
        return partial, kept, marker_after_partial, full

    partial, kept, marker_after_partial, full = asyncio.run(run())
    assert partial == {"success": True, "cleared_count": 1}
    assert kept == ["m1", "m2"]  # the oldest one was read; the rest were not
    assert marker_after_partial == {}  # buffer not emptied -> no marker
    assert full == {"success": True, "cleared_count": 2}
    assert contents(buffer, manager) == []
    assert _marker(buffer, manager)["reason"] == "consolidate"


def test_a_turn_landing_between_the_length_check_and_the_trim_withdraws_the_marker(env):
    """The check said "nothing new", the marker was written, and a turn arrived
    before the trim: the buffer is live, so the marker must not stay pending."""
    manager, buffer, fake = env
    real_ltrim = fake.ltrim

    async def ltrim_after_a_late_turn(key, start, end):
        await add(buffer, manager, "slipped in")
        return await real_ltrim(key, start, end)

    async def run():
        await add(buffer, manager, "one")
        await add(buffer, manager, "two")
        fake.ltrim = ltrim_after_a_late_turn
        await buffer.reset_session(
            manager.project, SESSION, reason="consolidate", expected_count=2,
        )
        fake.ltrim = real_ltrim
        marker = await buffer.get_session_metadata(manager.project, MARKER_ID)
        kept = await read_contents(buffer, manager)
        fake.storage.pop(buffer._session_messages_key(manager.project, SESSION))
        lost = await add(buffer, manager, "after a real loss")
        return marker, kept, lost

    marker, kept, lost = asyncio.run(run())
    assert kept == ["slipped in"]
    assert marker == {}
    assert lost["created_bucket"] is True


def test_consolidating_a_buffer_longer_than_the_read_limit_clears_it_all(env):
    """The read returns the last 1000 turns; the older ones were always dropped
    with the buffer.  They must not look like turns appended during the consolidate."""
    manager, buffer, _fake = env

    async def run():
        for n in range(1005):
            await add(buffer, manager, f"m{n}")
        return await manager.consolidate_session(session_id=SESSION, summary=AGENT_SUMMARY)

    assert asyncio.run(run())["success"] is True
    assert count(buffer, manager) == 0
    follow = asyncio.run(add(buffer, manager, "next"))
    assert (follow["created_bucket"], follow["buffer_reset_by"]) == (False, "consolidate")


# ---------------------------------------------------------------------------
# A consume that races a newer reset, or fails
# ---------------------------------------------------------------------------


def test_a_consume_cannot_overwrite_a_newer_reset(env):
    """Writer A takes the marker of reset 1; before it records the consume, a second
    reset clears A's message and writes its own marker.  A's consume must only
    retire reset 1: the next first write still belongs to reset 2."""
    manager, buffer, _fake = env
    real_get = buffer.get_session_metadata
    state = {"fired": False}

    async def get_then_reset(project, session_id):
        marker = await real_get(project, session_id)
        if session_id == MARKER_ID and not state["fired"]:
            state["fired"] = True
            await buffer.reset_session(project, SESSION, reason="clear")
        return marker

    async def run():
        await add(buffer, manager, "one")
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")
        buffer.get_session_metadata = get_then_reset
        writer = await add(buffer, manager, "two")
        buffer.get_session_metadata = real_get
        follow = await add(buffer, manager, "three")
        return writer, follow

    writer, follow = asyncio.run(run())
    assert (writer["created_bucket"], writer["buffer_reset_by"]) == (False, "consolidate")
    assert (follow["created_bucket"], follow["buffer_reset_by"]) == (False, "clear")


def test_a_failed_consume_removes_the_marker_so_it_cannot_excuse_a_later_loss(env):
    manager, buffer, fake = env
    real_set = buffer.set_session_metadata

    async def refuse_consume(project, session_id, metadata, **kwargs):
        if "consumed" in metadata:
            raise RuntimeError("write refused")
        return await real_set(project, session_id, metadata, **kwargs)

    with patch.object(mcp_tools_module, "_manager", manager):
        reflect("turn")
        consolidate()
        buffer.set_session_metadata = refuse_consume
        resumed = reflect("resumed")
        buffer.set_session_metadata = real_set
        assert (resumed["created_bucket"], resumed["buffer_reset_by"]) == (False, "consolidate")
        assert _marker(buffer, manager) == {}
        fake.storage.pop(buffer._session_messages_key(manager.project, SESSION))
        lost = reflect("after an eviction")
    assert lost["created_bucket"] is True
