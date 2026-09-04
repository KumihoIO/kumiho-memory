"""``KUMIHO_WORKING_MEMORY_TTL`` and the read-refresh of the session buffer TTL.

Only ``add_message`` used to refresh the TTL; a session that read its buffer
every turn but wrote less than hourly lost it.  Both halves are pinned here.
"""
import asyncio

import pytest

from fakes import FakeRedis
from kumiho_memory.redis_memory import (
    DEFAULT_WORKING_MEMORY_TTL,
    RedisMemoryBuffer,
    resolve_working_memory_ttl,
)


def _buffer(**kwargs):
    return RedisMemoryBuffer(
        client=FakeRedis(), redis_url="redis://test", prefer_discovery=False, **kwargs,
    )


def test_default_ttl_is_an_hour(monkeypatch):
    monkeypatch.delenv("KUMIHO_WORKING_MEMORY_TTL", raising=False)
    assert DEFAULT_WORKING_MEMORY_TTL == 3600
    assert _buffer().default_ttl == 3600


def test_env_knob_sets_the_ttl(monkeypatch):
    monkeypatch.setenv("KUMIHO_WORKING_MEMORY_TTL", "86400")
    assert _buffer().default_ttl == 86400


def test_explicit_argument_beats_the_env(monkeypatch):
    monkeypatch.setenv("KUMIHO_WORKING_MEMORY_TTL", "86400")
    assert _buffer(default_ttl=120).default_ttl == 120


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-5", "1.5"])
def test_unusable_env_values_fall_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv("KUMIHO_WORKING_MEMORY_TTL", bad)
    assert resolve_working_memory_ttl(None) == 3600
    assert _buffer().default_ttl == 3600


def test_reading_a_live_bucket_refreshes_its_ttl(monkeypatch):
    monkeypatch.setenv("KUMIHO_WORKING_MEMORY_TTL", "7200")
    buffer = _buffer()
    fake = buffer.client

    async def run():
        await buffer.add_message(project="p", session_id="s", role="user", content="hi")
        key = buffer._session_messages_key("p", "s")
        assert fake.ttl_store[key] == 7200
        fake.ttl_store[key] = 10  # about to expire
        got = await buffer.get_messages(project="p", session_id="s", limit=10)
        assert got["message_count"] == 1
        assert fake.ttl_store[key] == 7200
        assert got["ttl_remaining"] == 7200

    asyncio.run(run())


def test_reading_a_missing_bucket_mints_nothing():
    buffer = _buffer()
    fake = buffer.client

    async def run():
        got = await buffer.get_messages(project="p", session_id="nope", limit=10)
        assert got["message_count"] == 0
        assert got["ttl_remaining"] == 0
        assert fake.ttl_store == {}

    asyncio.run(run())
