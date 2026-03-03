import asyncio
from unittest.mock import MagicMock, patch

import pytest

from kumiho_memory.redis_memory import RedisDiscoveryError, RedisMemoryBuffer

from fakes import FakeRedis


def test_add_get_clear_messages():
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def run():
        await buffer.add_message(
            project="TestProject",
            session_id="session-001",
            role="user",
            content="Hello",
        )
        await buffer.add_message(
            project="TestProject",
            session_id="session-001",
            role="assistant",
            content="Hi there",
        )
        result = await buffer.get_messages(
            project="TestProject",
            session_id="session-001",
            limit=10,
        )
        assert result["message_count"] == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

        cleared = await buffer.clear_session("TestProject", "session-001")
        assert cleared["cleared_count"] == 2

    asyncio.run(run())


def test_list_sessions_and_sequence():
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def run():
        await buffer.add_message(
            project="TestProject",
            session_id="session-a",
            role="user",
            content="One",
        )
        await buffer.add_message(
            project="TestProject",
            session_id="session-b",
            role="user",
            content="Two",
        )
        sessions = await buffer.list_sessions("TestProject")
        assert set(sessions["sessions"]) == {"session-a", "session-b"}

        seq1 = await buffer.next_session_sequence(
            user_canonical_id="user-1", date_str="20260203"
        )
        seq2 = await buffer.next_session_sequence(
            user_canonical_id="user-1", date_str="20260203"
        )
        assert seq1 == 1
        assert seq2 == 2

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Token auto-refresh tests
# ---------------------------------------------------------------------------


@patch("kumiho_memory.redis_memory.load_bearer_token", return_value=None)
@patch("kumiho_memory.redis_memory.load_firebase_token", return_value="firebase-tok-123")
def test_get_fresh_token_uses_cached_firebase(mock_fb, mock_bearer):
    """_get_fresh_token should prefer the cached Firebase token."""
    token = RedisMemoryBuffer._get_fresh_token()
    assert token == "firebase-tok-123"
    mock_fb.assert_called_once()


@patch("kumiho_memory.redis_memory.load_firebase_token", return_value=None)
@patch("kumiho_memory.redis_memory.load_bearer_token", return_value="bearer-tok-456")
def test_get_fresh_token_falls_back_to_bearer(mock_bearer, mock_fb):
    """When no Firebase token, fall back to bearer token."""
    token = RedisMemoryBuffer._get_fresh_token()
    assert token == "bearer-tok-456"


@patch("kumiho_memory.redis_memory.load_bearer_token", return_value=None)
def test_get_fresh_token_refreshes_when_no_cached(mock_bearer):
    """When no cached tokens exist, ensure_token refreshes and we re-read
    the Firebase token it saved to disk."""
    # First call returns None (no cache), second call returns the refreshed token.
    with patch("kumiho_memory.redis_memory.load_firebase_token", side_effect=[None, "refreshed-fb-tok"]):
        with patch("kumiho.auth_cli.ensure_token", return_value=("cp-tok", "firebase")) as mock_ensure:
            token = RedisMemoryBuffer._get_fresh_token()
            assert token == "refreshed-fb-tok"
            mock_ensure.assert_called_once_with(interactive=False, force_refresh=False)


@patch("kumiho_memory.redis_memory.load_bearer_token", return_value=None)
@patch("kumiho_memory.redis_memory.load_firebase_token", return_value="new-fb-tok")
def test_get_fresh_token_force_refresh(mock_fb, mock_bearer):
    """force_refresh=True should skip cached tokens and call ensure_token,
    then re-read the Firebase token it saved."""
    with patch("kumiho.auth_cli.ensure_token", return_value=("cp-tok", "firebase")) as mock_ensure:
        token = RedisMemoryBuffer._get_fresh_token(force_refresh=True)
        assert token == "new-fb-tok"
        mock_ensure.assert_called_once_with(interactive=False, force_refresh=True)


@patch("kumiho_memory.redis_memory.load_firebase_token", return_value=None)
@patch("kumiho_memory.redis_memory.load_bearer_token", return_value=None)
def test_get_fresh_token_raises_when_nothing_available(mock_bearer, mock_fb):
    """Should raise RedisDiscoveryError when no credentials available."""
    with patch("kumiho.auth_cli.ensure_token", return_value=(None, None)):
        with pytest.raises(RedisDiscoveryError, match="No credentials available"):
            RedisMemoryBuffer._get_fresh_token()


@patch.object(RedisMemoryBuffer, "_get_fresh_token")
def test_proxy_request_retries_on_401(mock_token):
    """_proxy_request should retry once with a force-refreshed token on 401."""
    mock_token.side_effect = ["stale-tok", "fresh-tok"]

    # Build a buffer in proxy-only mode
    buffer = RedisMemoryBuffer(
        client=None,
        redis_url=None,
        prefer_discovery=False,
        proxy_url="https://proxy.example.com/api/memory/redis",
    )

    mock_response_401 = MagicMock()
    mock_response_401.status_code = 401
    mock_response_401.text = "Unauthorized"

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {"success": True}

    with patch("kumiho_memory.redis_memory.requests.post", side_effect=[mock_response_401, mock_response_ok]) as mock_post:
        result = asyncio.run(
            buffer._proxy_request(action="test", payload={"key": "value"})
        )
        assert result == {"success": True}
        assert mock_post.call_count == 2
        # First call with stale token, second with fresh
        assert mock_post.call_args_list[0][1]["headers"]["Authorization"] == "Bearer stale-tok"
        assert mock_post.call_args_list[1][1]["headers"]["Authorization"] == "Bearer fresh-tok"


@patch.object(RedisMemoryBuffer, "_get_fresh_token", return_value="valid-tok")
def test_proxy_request_surfaces_non_auth_errors(mock_token):
    """Non-auth HTTP errors (e.g. 500) should raise immediately without retry."""
    buffer = RedisMemoryBuffer(
        client=None,
        redis_url=None,
        prefer_discovery=False,
        proxy_url="https://proxy.example.com/api/memory/redis",
    )

    mock_response_500 = MagicMock()
    mock_response_500.status_code = 500
    mock_response_500.text = "Internal Server Error"

    with patch("kumiho_memory.redis_memory.requests.post", return_value=mock_response_500) as mock_post:
        with pytest.raises(RedisDiscoveryError, match="Memory proxy error 500"):
            asyncio.run(
                buffer._proxy_request(action="test", payload={})
            )
        # Only one call — no retry for non-auth errors
        assert mock_post.call_count == 1
