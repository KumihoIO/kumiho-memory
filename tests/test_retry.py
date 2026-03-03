import asyncio
import os
import tempfile

from kumiho_memory.retry import RetryQueue, retry_with_backoff


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
            assert result == {"succeeded": 2, "failed": 0}
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


def test_retry_queue_flush_empty():
    """Flushing an empty queue should return zeros."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(tmpdir)

        async def store_stub(**kwargs):
            return {}

        async def run():
            result = await queue.flush(store_stub)
            assert result == {"succeeded": 0, "failed": 0}

        asyncio.run(run())
