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


class FailureClass:
    """Three-way classification of a failed call (issue #118).

    - ``TRANSIENT``: timeouts, 5xx, rate limits, connection errors — worth
      retrying with backoff; the same request may well succeed next time.
    - ``DETERMINISTIC``: validation errors, content-filter refusals, 4xx-class
      semantic rejections — the same request will keep failing, so fail fast
      and let the caller park the content instead of retrying forever.
    - ``UNKNOWN``: everything else — retried, but with a tighter bound than
      transient failures.
    """

    TRANSIENT = "transient"
    DETERMINISTIC = "deterministic"
    UNKNOWN = "unknown"


# HTTP status codes that are transient even though they are 4xx.
_TRANSIENT_STATUS = frozenset({408, 429})

# Lowercased substrings in an exception's class name or message that mark a
# deterministic (content-policy / validation) failure.  Kept small: 4xx status
# shapes already catch the common cases; these are a secondary net for SDK
# exceptions that carry no status code.
_DETERMINISTIC_MARKERS = (
    "contentfilter",
    "content_filter",
    "content filter",
    "contentpolicy",
    "content policy",
    "content management policy",
    "invalidrequest",
    "validationerror",
)

# Substrings marking a transient failure when no type/status signal is present.
_TRANSIENT_MARKERS = (
    "ratelimit",
    "rate limit",
    "toomanyrequests",
    "serviceunavailable",
    "timeout",
    "timed out",
    "temporarily unavailable",
)


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Return an HTTP status code carried by *exc*, or ``None``.

    Looks at the common attribute shapes used by ``requests``/``httpx``/
    ``openai``/``anthropic``/``aiohttp`` exceptions.  Only integers in the
    valid HTTP range are accepted so unrelated attributes (e.g. an OS errno)
    are never mistaken for a status.
    """
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, bool):  # bool is an int subclass — reject
            continue
        if isinstance(val, int) and 100 <= val <= 599:
            return val
    response = getattr(exc, "response", None)
    if response is not None:
        val = getattr(response, "status_code", None)
        if isinstance(val, int) and not isinstance(val, bool) and 100 <= val <= 599:
            return val
    return None


def _matches(text: str, markers: Tuple[str, ...]) -> bool:
    return any(m in text for m in markers)


def classify_failure(exc: BaseException) -> str:
    """Classify *exc* as transient / deterministic / unknown (issue #118).

    Classification is by HTTP-status shape first (the most authoritative
    server signal), then exception type, then name/message markers.
    """
    # A present, valid HTTP status is the strongest signal.
    status = _extract_status_code(exc)
    if status is not None:
        if status in _TRANSIENT_STATUS or 500 <= status <= 599:
            return FailureClass.TRANSIENT
        if 400 <= status <= 499:
            return FailureClass.DETERMINISTIC

    # Connection/timeout/OS errors never reached (or lost) the server.
    if isinstance(exc, TRANSIENT_ERRORS):
        return FailureClass.TRANSIENT

    # Validation-shaped errors repeat deterministically.  ``json.JSONDecodeError``
    # and pydantic's ``ValidationError`` are both ``ValueError`` subclasses.
    if isinstance(exc, (ValueError, TypeError)):
        return FailureClass.DETERMINISTIC

    haystack = f"{type(exc).__name__} {exc}".casefold()
    if _matches(haystack, _DETERMINISTIC_MARKERS):
        return FailureClass.DETERMINISTIC
    if _matches(haystack, _TRANSIENT_MARKERS):
        return FailureClass.TRANSIENT

    return FailureClass.UNKNOWN


async def retry_with_backoff(
    func: Callable[..., Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    unknown_max_retries: int = 2,
    **kwargs: Any,
) -> Any:
    """Call *func* with exponential backoff, respecting failure class (#118).

    Failures are classified with :func:`classify_failure`:

    - **transient** — retried up to *max_retries* times (unchanged behavior).
    - **deterministic** — re-raised immediately (fail fast); retrying would
      just re-fail, so the caller can park the content instead.
    - **unknown** — retried, but capped at *unknown_max_retries* attempts.

    Parameters
    ----------
    func:
        Sync or async callable to invoke.
    max_retries:
        Total number of attempts (including the first) for transient failures.
    base_delay:
        Initial backoff in seconds; doubled each attempt with jitter.
    max_delay:
        Cap on backoff duration.
    unknown_max_retries:
        Attempt cap for *unknown*-class failures (bounded retries).
    **kwargs:
        Forwarded to *func*.

    Raises
    ------
    The last exception if all retries are exhausted, or immediately for a
    deterministic failure.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            result = func(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:  # noqa: BLE001 — classified below
            last_exc = exc
            failure_class = classify_failure(exc)
            if failure_class == FailureClass.DETERMINISTIC:
                # Won't succeed on retry — fail fast.
                raise
            # Transient uses the full budget; unknown is bounded tighter.
            effective_max = (
                max_retries
                if failure_class == FailureClass.TRANSIENT
                else min(max_retries, max(1, unknown_max_retries))
            )
            if attempt + 1 >= effective_max:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.25)
            logger.warning(
                "%s error (attempt %d/%d): %s — retrying in %.1fs",
                failure_class.capitalize(),
                attempt + 1,
                effective_max,
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

        Returns a dict with ``succeeded``, ``failed``, and ``dropped`` counts.
        Items that succeed are removed; items that fail *transiently* stay in
        the queue; items that fail *deterministically* are dropped (issue #118)
        — a deterministic payload will re-fail on every future flush, so
        re-queuing it would replay the same poison content forever.  The
        content was already recorded in the failure ledger at the store seam
        that first enqueued it, so dropping it here loses no tracking.
        """
        entries = self.drain()
        if not entries:
            return {"succeeded": 0, "failed": 0, "dropped": 0}

        succeeded = 0
        dropped = 0
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
                if classify_failure(exc) == FailureClass.DETERMINISTIC:
                    dropped += 1
                    logger.warning(
                        "Dropping deterministically-failing retry queue entry "
                        "(it will never succeed): %s",
                        exc,
                    )
                    continue
                logger.warning("Retry queue flush failed for entry: %s", exc)
                still_failed.append(entry)

        # Rewrite queue with only the items that still failed transiently.
        if still_failed:
            with open(self._queue_file, "w", encoding="utf-8") as f:
                for entry in still_failed:
                    f.write(json.dumps(entry, default=str) + "\n")
        else:
            self.clear()

        return {"succeeded": succeeded, "failed": len(still_failed), "dropped": dropped}

    def clear(self) -> None:
        """Remove all pending items."""
        if self._queue_file.exists():
            self._queue_file.unlink()
