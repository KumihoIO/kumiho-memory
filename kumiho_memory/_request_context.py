"""Per-request tenant context — SDK import with a vendored fallback.

The hosted Claude connector (see ``kumiho-plugins/docs/CLOUD-CONNECTOR-PLAN.md``
§2.1) serves many tenants from ONE process. Everything that used to be read
from ``os.environ`` or ``~/.kumiho`` — the auth token, the tenant id, the
session identity — becomes per-request state carried in a contextvar, because
the alternative (process env mutation) is a cross-tenant leak by construction.

``kumiho.request_context`` is the canonical home (WP-A). This module imports it
when present and otherwise defines the byte-identical fallback, so
``kumiho-memory`` can ship ahead of the SDK release. **Every module in this
package imports the request context through here**, never from ``kumiho``
directly — one import site means one thing to delete when the SDK version pin
can be raised.

Nothing here changes behavior on the stdio path: with no request set,
:func:`current_request` returns ``None`` and :func:`is_hosted` is ``False``.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised by whichever kumiho version is installed
    from kumiho.request_context import (  # type: ignore
        RequestContext,
        current_request,
        hosted_mode,
        request_context,
    )
except ImportError:  # pragma: no cover - vendored fallback, spec §2.1
    import contextvars
    from contextlib import contextmanager
    from dataclasses import dataclass, field
    from typing import Iterator, List, Optional

    @dataclass(frozen=True)
    class RequestContext:  # type: ignore[no-redef]
        tenant_id: str            # UUID from token claims
        user_id: str              # firebase uid (OAuth) or "service:<token_id>" (API key)
        auth_token: str           # the raw bearer/api-key JWT presented by the caller
        context: str = "claude"   # memory context namespace (active-session pointer key)
        session_id: Optional[str] = None
        client_id: Optional[str] = None
        scopes: List[str] = field(default_factory=list)
        tenant_slug: Optional[str] = None
        region_code: Optional[str] = None
        token_id: Optional[str] = None   # jti

    _request_var: "contextvars.ContextVar[Optional[RequestContext]]" = contextvars.ContextVar(
        "kumiho_request", default=None,
    )

    def current_request() -> "Optional[RequestContext]":  # type: ignore[no-redef]
        return _request_var.get()

    @contextmanager
    def request_context(ctx: "RequestContext") -> "Iterator[RequestContext]":  # type: ignore[no-redef]
        token = _request_var.set(ctx)
        try:
            yield ctx
        finally:
            _request_var.reset(token)

    def hosted_mode() -> bool:  # type: ignore[no-redef]
        import os
        return os.environ.get("KUMIHO_MCP_HOSTED", "").strip().lower() in ("1", "true", "yes")


def is_hosted() -> bool:
    """Whether this process must behave as a shared multi-tenant server.

    True when a request context is active OR the operator set
    ``KUMIHO_MCP_HOSTED``. The env alone is enough on purpose: the filesystem
    and ambient-credential rules must hold for background work that runs
    *outside* a request (an eviction sweep, a module imported at startup),
    not only while one is in flight. The stdio plugin sets neither, so its
    behavior is untouched.
    """
    return current_request() is not None or hosted_mode()


def hosted_llm_enabled() -> bool:
    """Whether a hosted deployment may spend LLM calls (``KUMIHO_HOSTED_LLM``).

    Off by default: v1 of the connector is the keyless core (plan §1 decision
    10). The assessors, the summarizer's provider adapter and the LLM reranker
    all stay unbuilt unless an operator opts in per deployment.
    """
    import os
    return os.environ.get("KUMIHO_HOSTED_LLM", "").strip().lower() in ("1", "true", "yes")


def tenant_scope(default: str = "") -> str:
    """The cache-partition key for the active request, or ``default``.

    Every process-global cache in this package that holds tenant-derived data
    is keyed with this. Empty string on the stdio path keeps the single-tenant
    key space identical to what it was before hosted mode existed.
    """
    ctx = current_request()
    return ctx.tenant_id if ctx is not None else default


__all__ = [
    "RequestContext",
    "current_request",
    "hosted_llm_enabled",
    "hosted_mode",
    "is_hosted",
    "request_context",
    "tenant_scope",
]
