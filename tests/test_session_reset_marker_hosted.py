"""Reset marker through the hosted control-plane proxy.

The deployed proxy validates every request body against a strict schema and knows
exactly twelve actions, so the marker may use neither a new action nor a new field
(no schema/action change).  It also cannot trim a list, so a turn appended while a
consolidate was summarizing cannot be kept there: it is discarded as it always was,
but it is never excused -- no marker, a WARNING, and the next write says ``true``.
"""

import asyncio
import logging
from unittest.mock import patch

import pytest

import kumiho_memory.mcp_tools as mcp_tools_module
from kumiho_memory import redis_memory
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from hosted_fakes import FakeMemoryProxy
from reset_marker_support import (
    MARKER_TTL, SESSION, _short_ttl, add, consolidate, reflect,
)
from test_memory_manager import AGENT_SUMMARY, NeverSummarizer, StubRedactor

_FIXTURES = (_short_ttl,)  # imported fixture: pytest collects it from here
MARKER_ID = SESSION + redis_memory.RESET_MARKER_SUFFIX

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
        first = reflect("turn")
        assert consolidate()["success"] is True
        second = reflect("after")
        third = reflect("again")
        fresh = reflect("elsewhere", session_id="a-genuinely-new-id")
    assert first["created_bucket"] is True
    assert (second["created_bucket"], second["buffer_reset_by"]) == (False, "consolidate")
    assert third["created_bucket"] is False and "buffer_reset_by" not in third
    assert fresh["created_bucket"] is True
    # every call that reached the proxy used an existing action and field
    assert {c["action"] for c in fake.calls} <= PROXY_ACTIONS


def test_hosted_proxy_marker_outlives_the_buffer_ttl_and_other_calls_are_unchanged(proxied):
    manager, buffer, fake = proxied

    async def run():
        await add(buffer, manager, "one")
        await buffer.set_session_metadata(manager.project, SESSION, {"user_id": "u"})
        await buffer.reset_session(manager.project, SESSION, reason="consolidate")

    asyncio.run(run())
    sets = [c["body"] for c in fake.calls if c["action"] == "set_session_metadata"]
    ordinary = [b for b in sets if b["session_id"] == SESSION]
    marker = [b for b in sets if b["session_id"] == MARKER_ID]
    # an ordinary metadata write is byte-for-byte what it was (no new field)
    assert ordinary == [{
        "action": "set_session_metadata", "project": manager.project,
        "session_id": SESSION, "metadata": {"user_id": "u"},
    }]
    # the marker asks for max(buffer TTL, a day), not the proxy's 3600 default (a
    # memory-only tier clamps it to its own ceiling server-side)
    assert len(marker) == 1 and marker[0]["default_ttl"] == MARKER_TTL
    assert set(marker[0]["metadata"]) == {"reset_id", "reason"}
    # a reset is: look at the buffer, write the marker, then delete the buffer
    reset_calls = [c["action"] for c in fake.calls][-3:]
    assert reset_calls == ["get_messages", "set_session_metadata", "clear_session"]


def test_hosted_proxy_clear_session_payload_is_unchanged(proxied):
    manager, buffer, fake = proxied
    asyncio.run(buffer.clear_session(manager.project, SESSION))
    (call,) = [c for c in fake.calls if c["action"] == "clear_session"]
    assert call["body"] == {
        "action": "clear_session", "project": manager.project, "session_id": SESSION,
    }


def test_hosted_proxy_a_turn_appended_during_a_consolidate_is_never_excused(proxied, caplog):
    """The proxy cannot trim, so the late turn is discarded -- but the marker is
    not written, the loss is logged, and the next write says ``true``."""
    manager, buffer, fake = proxied
    real_store = manager.memory_store

    async def store_and_append(**kwargs):
        result = await real_store(**kwargs)
        await buffer.add_message(
            project=manager.project, session_id=SESSION, role="user", content="late turn",
        )
        return result

    manager.memory_store = store_and_append

    async def run():
        await add(buffer, manager, "user turn")
        await add(buffer, manager, "assistant turn", role="assistant")
        consolidated = await manager.consolidate_session(session_id=SESSION, summary=AGENT_SUMMARY)
        marker = await buffer.get_session_metadata(manager.project, MARKER_ID)
        follow = await add(buffer, manager, "next turn")
        return consolidated, marker, follow

    with caplog.at_level(logging.WARNING, logger="kumiho_memory.session_reset"):
        consolidated, marker, follow = asyncio.run(run())
    assert consolidated["success"] is True
    assert marker == {}
    assert follow["created_bucket"] is True and "buffer_reset_by" not in follow
    assert "discarded 1 turn(s)" in caplog.text
    # a WARNING that describes, and promises nothing (a first write racing the
    # withdrawal could still see the marker)
    assert "will report created_bucket=true" not in caplog.text
    # the late turn was seen before anything was written: no marker went out at all
    assert not [
        c for c in fake.calls
        if c["action"] == "set_session_metadata" and c["body"]["session_id"] == MARKER_ID
    ]


def _seed_buffer(fake, held):
    # the unregistered bearer "tok" is its own tenant in the fake (see FakeMemoryProxy)
    fake._bucket("tok")["messages"][SESSION] = [
        {"role": "user", "content": f"m{n}", "metadata": {}} for n in range(held)
    ]


@pytest.mark.parametrize("held,excused", [(249, True), (250, False)])
def test_hosted_proxy_a_buffer_at_the_server_cap_gets_no_marker(proxied, held, excused):
    """A memory-only tenant's server trims to the last 250 turns on every append, so
    at the cap a late append leaves the count unchanged and cannot be told apart
    from "nothing arrived".  At the cap the reset excuses nothing."""
    manager, buffer, fake = proxied
    _seed_buffer(fake, held)
    asyncio.run(buffer.reset_session(
        manager.project, SESSION, reason="consolidate", expected_count=held,
    ))
    follow = asyncio.run(add(buffer, manager, "next"))
    assert follow["created_bucket"] is (not excused)
    assert ("buffer_reset_by" in follow) is excused


def test_hosted_proxy_a_turn_landing_between_the_check_and_the_delete_withdraws_the_marker(proxied):
    manager, buffer, fake = proxied
    real_clear = buffer.clear_session

    async def clear_after_a_late_turn(project, session_id):
        if session_id == SESSION:
            await buffer.add_message(
                project=project, session_id=SESSION, role="user", content="slipped in",
            )
        return await real_clear(project, session_id)

    async def run():
        await add(buffer, manager, "one")
        await add(buffer, manager, "two")
        buffer.clear_session = clear_after_a_late_turn
        await buffer.reset_session(
            manager.project, SESSION, reason="consolidate", expected_count=2,
        )
        buffer.clear_session = real_clear
        marker = await buffer.get_session_metadata(manager.project, MARKER_ID)
        follow = await add(buffer, manager, "next")
        return marker, follow

    marker, follow = asyncio.run(run())
    assert marker == {}  # withdrawn: a marker over a lost turn would excuse the loss
    assert follow["created_bucket"] is True
