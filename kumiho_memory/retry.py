"""Retry logic and persistent queue for failed memory writes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Exception types considered transient (worth retrying).
TRANSIENT_ERRORS: Tuple[type, ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)
try:
    from requests.exceptions import ConnectionError as ReqConnectionError
    from requests.exceptions import Timeout as ReqTimeout

    TRANSIENT_ERRORS = (*TRANSIENT_ERRORS, ReqConnectionError, ReqTimeout)
except ImportError:
    pass


async def retry_with_backoff(
    func: Callable[..., Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Call *func* with exponential backoff on transient errors.

    Parameters
    ----------
    func:
        Sync or async callable to invoke.
    max_retries:
        Total number of attempts (including the first).
    base_delay:
        Initial backoff in seconds; doubled each attempt with jitter.
    max_delay:
        Cap on backoff duration.
    **kwargs:
        Forwarded to *func*.

    Raises
    ------
    The last exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            result = func(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt + 1 >= max_retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.25)
            logger.warning(
                "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay + jitter,
            )
            await asyncio.sleep(delay + jitter)

    raise last_exc  # type: ignore[misc]


class RetryQueue:
    """File-backed queue for failed ``memory_store`` payloads.

    Pending items are stored as JSON-lines in::

        {queue_dir}/pending.jsonl

    Each line is a JSON object with ``timestamp`` and ``payload`` keys.

    Usage::

        queue = RetryQueue("/path/to/queue")
        queue.enqueue(payload)
        # Later…
        results = await queue.flush(store_callable)
    """

    def __init__(self, queue_dir: Optional[str] = None) -> None:
        self.queue_dir = Path(
            queue_dir
            or os.getenv("KUMIHO_RETRY_QUEUE_DIR")
            or os.path.join(os.path.expanduser("~"), ".kumiho", "retry_queue")
        )
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue_file = self.queue_dir / "pending.jsonl"

    @property
    def count(self) -> int:
        """Number of pending items in the queue."""
        if not self._queue_file.exists():
            return 0
        count = 0
        with open(self._queue_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def enqueue(self, payload: Dict[str, Any]) -> None:
        """Append a failed store payload to the queue."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        with open(self._queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info("Enqueued failed memory_store payload (%d pending)", self.count)

    def drain(self) -> List[Dict[str, Any]]:
        """Read and return all pending entries (does not remove them)."""
        if not self._queue_file.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(self._queue_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed retry queue entry")
        return entries

    async def flush(
        self,
        store_callable: Callable[..., Any],
        *,
        max_retries: int = 2,
    ) -> Dict[str, int]:
        """Attempt to replay all queued payloads through *store_callable*.

        Returns a dict with ``succeeded`` and ``failed`` counts.  Items
        that succeed are removed; items that fail again stay in the queue.
        """
        entries = self.drain()
        if not entries:
            return {"succeeded": 0, "failed": 0}

        succeeded = 0
        still_failed: List[Dict[str, Any]] = []

        for entry in entries:
            payload = entry.get("payload", {})
            try:
                await retry_with_backoff(
                    store_callable,
                    max_retries=max_retries,
                    base_delay=0.5,
                    **payload,
                )
                succeeded += 1
            except Exception as exc:
                logger.warning("Retry queue flush failed for entry: %s", exc)
                still_failed.append(entry)

        # Rewrite queue with only the items that still failed
        if still_failed:
            with open(self._queue_file, "w", encoding="utf-8") as f:
                for entry in still_failed:
                    f.write(json.dumps(entry, default=str) + "\n")
        else:
            self.clear()

        return {"succeeded": succeeded, "failed": len(still_failed)}

    def clear(self) -> None:
        """Remove all pending items."""
        if self._queue_file.exists():
            self._queue_file.unlink()
