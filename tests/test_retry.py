import asyncio
import json
import os
import tempfile

import pytest

from kumiho_memory.retry import (
    FailureClass,
    RetryQueue,
    classify_failure,
    retry_with_backoff,
)


def test_retry_with_backoff_succeeds_first_try():
    """When the callable succeeds immediately, no retries needed."""
    call_count = 0

    async def good_func(**kwargs):
        nonlocal call_count
        call_count += 1
        return {"ok": True}

    async def run():
        result = await retry_with_backoff(good_func, max_retries=3, foo="bar")
        assert result == {"ok": True}
        assert call_count == 1

    asyncio.run(run())


def test_retry_with_backoff_retries_on_transient_error():
    """Should retry on ConnectionError and eventually succeed."""
    call_count = 0

    async def flaky_func(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("connection refused")
        return {"ok": True}

    async def run():
        result = await retry_with_backoff(
            flaky_func, max_retries=3, base_delay=0.01, project="test"
        )
        assert result == {"ok": True}
        assert call_count == 3

    asyncio.run(run())


def test_retry_with_backoff_exhausts_retries():
    """When all retries fail, the last exception should be raised."""
    call_count = 0

    async def always_fail(**kwargs):
        nonlocal call_count
        call_count += 1
        raise TimeoutError("timed out")

    async def run():
        try:
            await retry_with_backoff(
                always_fail, max_retries=2, base_delay=0.01
            )
            assert False, "Should have raised TimeoutError"
        except TimeoutError:
            pass
        assert call_count == 2

    asyncio.run(run())


def test_retry_with_backoff_no_retry_on_value_error():
    """Non-transient errors should not be retried."""
    call_count = 0

    async def bad_input(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    async def run():
        try:
            await retry_with_backoff(bad_input, max_retries=3, base_delay=0.01)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        assert call_count == 1

    asyncio.run(run())


def test_retry_with_backoff_sync_callable():
    """Should work with sync callables too."""
    call_count = 0

    def sync_func(**kwargs):
        nonlocal call_count
        call_count += 1
        return {"sync": True}

    async def run():
        result = await retry_with_backoff(sync_func, max_retries=1)
        assert result == {"sync": True}
        assert call_count == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Failure classification (issue #118)
# ---------------------------------------------------------------------------


class _HttpError(Exception):
    """Exception carrying an HTTP ``status_code`` (openai/anthropic shape)."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class _ResponseError(Exception):
    """Exception carrying a ``response`` with a ``status_code`` (requests shape)."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.response = type("Resp", (), {"status_code": status_code})()


class _AiohttpError(Exception):
    """Exception carrying a ``status`` int (aiohttp shape)."""

    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class _ContentFilterError(Exception):
    """SDK-style content-filter refusal with no status code."""


def test_classify_transient_exception_types():
    assert classify_failure(ConnectionError("x")) == FailureClass.TRANSIENT
    assert classify_failure(TimeoutError("x")) == FailureClass.TRANSIENT
    assert classify_failure(OSError("x")) == FailureClass.TRANSIENT


def test_classify_deterministic_validation_types():
    assert classify_failure(ValueError("bad")) == FailureClass.DETERMINISTIC
    assert classify_failure(TypeError("bad")) == FailureClass.DETERMINISTIC
    # json.JSONDecodeError is a ValueError subclass — malformed model output.
    try:
        json.loads("{not json")
    except json.JSONDecodeError as exc:
        assert classify_failure(exc) == FailureClass.DETERMINISTIC


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 499])
def test_classify_4xx_status_is_deterministic(status):
    assert classify_failure(_HttpError("no", status)) == FailureClass.DETERMINISTIC


@pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
def test_classify_5xx_status_is_transient(status):
    assert classify_failure(_HttpError("boom", status)) == FailureClass.TRANSIENT


@pytest.mark.parametrize("status", [408, 429])
def test_classify_408_and_429_are_transient(status):
    # Request-timeout and rate-limit are 4xx but genuinely transient.
    assert classify_failure(_HttpError("slow", status)) == FailureClass.TRANSIENT


def test_classify_status_via_response_attribute():
    assert classify_failure(_ResponseError("nope", 404)) == FailureClass.DETERMINISTIC
    assert classify_failure(_ResponseError("down", 503)) == FailureClass.TRANSIENT


def test_classify_status_via_status_attribute():
    assert classify_failure(_AiohttpError("nope", 400)) == FailureClass.DETERMINISTIC
    assert classify_failure(_AiohttpError("down", 502)) == FailureClass.TRANSIENT


def test_classify_content_filter_marker_is_deterministic():
    assert classify_failure(_ContentFilterError("blocked")) == FailureClass.DETERMINISTIC
    assert (
        classify_failure(Exception("Request blocked by content filter"))
        == FailureClass.DETERMINISTIC
    )


def test_classify_rate_limit_marker_is_transient():
    assert classify_failure(Exception("RateLimit exceeded, retry later")) == FailureClass.TRANSIENT
    assert classify_failure(Exception("operation timed out")) == FailureClass.TRANSIENT


def test_classify_unknown_default():
    assert classify_failure(RuntimeError("mystery")) == FailureClass.UNKNOWN
    assert classify_failure(Exception("something odd")) == FailureClass.UNKNOWN


def test_classify_status_precedence_over_transient_type():
    """A carried 4xx status wins over a transient-looking base class."""

    class WeirdConnErr(ConnectionError):
        status_code = 400

    assert classify_failure(WeirdConnErr("x")) == FailureClass.DETERMINISTIC


def test_classify_ignores_bool_status():
    """A boolean attribute must not be read as HTTP status 1."""

    class BoolStatus(Exception):
        status_code = True

    # No usable status → falls through to unknown (message has no markers).
    assert classify_failure(BoolStatus("x")) == FailureClass.UNKNOWN


def test_classify_ignores_out_of_range_status():
    class OddStatus(Exception):
        status_code = 12345

    assert classify_failure(OddStatus("x")) == FailureClass.UNKNOWN


# ---------------------------------------------------------------------------
# retry_with_backoff — classification-driven behavior (issue #118)
# ---------------------------------------------------------------------------


def test_retry_deterministic_status_fails_fast():
    """A 4xx-class failure is not retried (fail fast)."""
    call_count = 0

    async def bad(**kwargs):
        nonlocal call_count
        call_count += 1
        raise _HttpError("invalid request", 400)

    async def run():
        with pytest.raises(_HttpError):
            await retry_with_backoff(bad, max_retries=5, base_delay=0.01)
        assert call_count == 1

    asyncio.run(run())


def test_retry_unknown_is_bounded():
    """Unknown failures retry, but only up to unknown_max_retries."""
    call_count = 0

    async def mystery(**kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("who knows")

    async def run():
        with pytest.raises(RuntimeError):
            await retry_with_backoff(
                mystery, max_retries=5, base_delay=0.01, unknown_max_retries=2
            )
        # Bounded to 2 attempts even though max_retries=5.
        assert call_count == 2

    asyncio.run(run())


def test_retry_unknown_can_succeed_within_bound():
    call_count = 0

    async def flaky(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("transient-ish")
        return {"ok": True}

    async def run():
        result = await retry_with_backoff(
            flaky, max_retries=5, base_delay=0.01, unknown_max_retries=3
        )
        assert result == {"ok": True}
        assert call_count == 2

    asyncio.run(run())


def test_retry_transient_uses_full_budget():
    """5xx (transient) uses max_retries, not the unknown bound."""
    call_count = 0

    async def down(**kwargs):
        nonlocal call_count
        call_count += 1
        raise _HttpError("server error", 503)

    async def run():
        with pytest.raises(_HttpError):
            await retry_with_backoff(
                down, max_retries=4, base_delay=0.01, unknown_max_retries=1
            )
        assert call_count == 4

    asyncio.run(run())


def test_retry_does_not_swallow_cancellation():
    """BaseException (e.g. CancelledError) must propagate, never be retried."""
    call_count = 0

    async def cancel_me(**kwargs):
        nonlocal call_count
        call_count += 1
        raise asyncio.CancelledError()

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await retry_with_backoff(cancel_me, max_retries=5, base_delay=0.01)
        assert call_count == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# RetryQueue tests
# ---------------------------------------------------------------------------


def test_retry_queue_enqueue_and_drain():
    """Enqueued items should be readable via drain()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)

        assert queue.count == 0
        queue.enqueue({"project": "test", "title": "item-1"})
        queue.enqueue({"project": "test", "title": "item-2"})
        assert queue.count == 2

        entries = queue.drain()
        assert len(entries) == 2
        assert entries[0]["payload"]["title"] == "item-1"
        assert entries[1]["payload"]["title"] == "item-2"
        assert "timestamp" in entries[0]


def test_retry_queue_clear():
    """clear() should remove all pending items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)
        queue.enqueue({"project": "test"})
        assert queue.count == 1
        queue.clear()
        assert queue.count == 0


def test_retry_queue_flush_success():
    """flush() should replay payloads and remove succeeded items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)
        queue.enqueue({"project": "test", "title": "replay-1"})
        queue.enqueue({"project": "test", "title": "replay-2"})

        replayed = []

        async def store_stub(**kwargs):
            replayed.append(kwargs.get("title"))
            return {"ok": True}

        async def run():
            result = await queue.flush(store_stub)
            assert result == {"succeeded": 2, "failed": 0, "dropped": 0}
            assert queue.count == 0
            assert replayed == ["replay-1", "replay-2"]

        asyncio.run(run())


def test_retry_queue_flush_partial_failure():
    """Items that fail during flush should stay in the queue."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)
        queue.enqueue({"project": "test", "title": "will-succeed"})
        queue.enqueue({"project": "test", "title": "will-fail"})

        async def partial_store(**kwargs):
            if kwargs.get("title") == "will-fail":
                raise ConnectionError("still down")
            return {"ok": True}

        async def run():
            result = await queue.flush(partial_store, max_retries=1)
            assert result["succeeded"] == 1
            assert result["failed"] == 1
            assert queue.count == 1

            # The remaining item should be the failed one
            remaining = queue.drain()
            assert remaining[0]["payload"]["title"] == "will-fail"

        asyncio.run(run())


def test_retry_queue_flush_drops_deterministic_failure():
    """A payload that fails deterministically on replay is dropped, not
    re-queued (issue #118) — re-queuing would replay the same poison payload
    on every future flush forever."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)
        queue.enqueue({"project": "test", "title": "poison"})
        queue.enqueue({"project": "test", "title": "will-succeed"})

        calls = []

        async def store(**kwargs):
            calls.append(kwargs.get("title"))
            if kwargs.get("title") == "poison":
                raise ValueError("schema validation failed")  # deterministic
            return {"ok": True}

        async def run():
            result = await queue.flush(store, max_retries=3)
            assert result == {"succeeded": 1, "failed": 0, "dropped": 1}
            # The poison payload is gone — not left to replay every flush.
            assert queue.count == 0
            # Deterministic fail-fast: the poison item was attempted exactly once.
            assert calls.count("poison") == 1

        asyncio.run(run())


def test_retry_queue_flush_transient_stays_queued():
    """A transient replay failure is still re-queued (not dropped) — only
    deterministic failures are dropped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)
        queue.enqueue({"project": "test", "title": "blip"})

        async def store(**kwargs):
            raise ConnectionError("still down")  # transient

        async def run():
            result = await queue.flush(store, max_retries=1)
            assert result == {"succeeded": 0, "failed": 1, "dropped": 0}
            assert queue.count == 1  # stays for the next flush

        asyncio.run(run())


def test_retry_queue_flush_empty():
    """Flushing an empty queue should return zeros."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)

        async def store_stub(**kwargs):
            return {}

        async def run():
            result = await queue.flush(store_stub)
            assert result == {"succeeded": 0, "failed": 0, "dropped": 0}

        asyncio.run(run())
