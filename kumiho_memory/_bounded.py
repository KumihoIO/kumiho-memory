"""Run a blocking function in a daemon thread against a deadline.

Extracted so the Windows-hang workaround (a synchronous gRPC call can hang
indefinitely, so it must not run on a shared executor thread) lives in one
place instead of being re-derived per call site.

Note on the leak: Python cannot cancel a running thread. On timeout the
worker is abandoned as a daemon (it dies with the process); ``on_timeout``
is returned to the caller. Keep per-call timeouts modest so abandoned
workers can't accumulate.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_bounded_in_thread(
    fn: Callable[[], T],
    *,
    timeout: float,
    label: str = "bounded task",
    on_timeout: Optional[T] = None,
    on_error: Optional[T] = None,
) -> Optional[T]:
    """Await *fn* running in a daemon thread, giving up after *timeout* s.

    Returns ``fn()``'s result, or ``on_timeout`` / ``on_error`` on deadline
    or exception. Never raises — this is for best-effort enrichment that must
    not break its caller.
    """
    result: list = []
    error: list = []
    done = threading.Event()

    def _worker() -> None:
        try:
            result.append(fn())
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            error.append(exc)
            logger.debug("%s failed: %s", label, exc)
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    deadline = time.monotonic() + timeout
    while not done.is_set():
        if time.monotonic() >= deadline:
            logger.debug("%s timed out after %.0fs", label, timeout)
            return on_timeout
        await asyncio.sleep(0.05)

    if error:
        return on_error
    return result[0] if result else on_error
