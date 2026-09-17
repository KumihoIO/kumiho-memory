"""Keep deliberately unpublished records unpublished across kumiho SDK versions.

Until kumiho 0.13.1, ``kumiho.mcp_server.tool_memory_store`` tagged the
revision it created with ``tags or ["published"]``: caller tags *replaced*
``published``. kumiho-memory's experience snapshots and pattern proposals
relied on that. They pass tags without ``published`` and stay unpublished.

kumiho 0.13.2 (KumihoIO/kumiho-SDKs#170) always applies ``published``, after
the caller's tags, and adds a ``publish`` keyword (default ``True``) to opt
out. Those callers now pass ``publish=False``.

kumiho-memory still supports ``kumiho>=0.10.7``, where ``tool_memory_store``
rejects the unknown keyword with ``TypeError``. On those versions the tags
alone already keep the record unpublished, so the keyword is omitted. The
decision is made from the resolved store callable itself, because the store
can be injected (``UniversalMemoryManager(memory_store=...)``), may be async,
and is not necessarily the installed SDK's function.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict

PUBLISH_KEYWORD = "publish"

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _inspect_accepts_publish(store: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(store).parameters.values()
    except (TypeError, ValueError):
        # Not introspectable (e.g. a C-implemented callable). Omitting the
        # keyword cannot raise, and it is the correct call on an older SDK.
        return False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            # ``**kwargs``: a test fake, a mock, or a wrapper that forwards
            # keywords. It declares that it accepts the keyword.
            return True
        if parameter.name == PUBLISH_KEYWORD and parameter.kind in _KEYWORD_KINDS:
            return True
    return False


@functools.lru_cache(maxsize=64)
def _cached_accepts_publish(store: Callable[..., Any]) -> bool:
    return _inspect_accepts_publish(store)


def store_accepts_publish(store: Callable[..., Any]) -> bool:
    """Whether *store* can be called with ``publish=...`` without a ``TypeError``.

    True for kumiho>=0.13.2's ``tool_memory_store`` and for any callable that
    declares ``publish`` or ``**kwargs``. ``inspect.signature`` follows
    ``__wrapped__``, so a ``functools.wraps`` wrapper reports the wrapped
    store's parameters. False for kumiho<=0.13.1 and for a callable whose
    signature cannot be read. The result is cached per callable.
    """
    # A bound method is a new object on every attribute access; key on the
    # function so the cache hits and does not pin the instance.
    key = getattr(store, "__func__", store)
    try:
        return _cached_accepts_publish(key)
    except TypeError:  # unhashable callable: inspect without caching
        return _inspect_accepts_publish(key)


def unpublished_store_payload(store: Callable[..., Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return *payload* for a record that must be stored unpublished.

    Adds ``publish=False`` when *store* accepts it. Otherwise the payload is
    returned unchanged: an SDK without the keyword does not publish a capture
    whose tags omit ``published``.
    """
    if store_accepts_publish(store):
        return {**payload, PUBLISH_KEYWORD: False}
    return payload
