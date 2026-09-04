"""Redis-backed working memory buffer for Kumiho AI Cognitive Memory."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import redis.asyncio as redis
import requests

from kumiho._token_loader import load_bearer_token, load_firebase_token
from kumiho.discovery import (
    DEFAULT_CACHE_PATH,
    DEFAULT_CONTROL_PLANE_URL,
    DiscoveryCache,
    DiscoveryManager,
    _DEFAULT_CACHE_KEY,
)

from kumiho_memory._request_context import current_request, hosted_mode, is_hosted

logger = logging.getLogger(__name__)

#: Idle TTL (seconds) of a session's working-memory buffer when neither the
#: constructor nor ``KUMIHO_WORKING_MEMORY_TTL`` says otherwise.  An hour is the
#: shared-Upstash figure; a self-hosted Redis can afford far more, and the
#: Claude plugin sets a day in CE mode.
DEFAULT_WORKING_MEMORY_TTL = 3600
WORKING_MEMORY_TTL_ENV = "KUMIHO_WORKING_MEMORY_TTL"


def resolve_working_memory_ttl(explicit: Optional[int] = None) -> int:
    """The buffer TTL to use: explicit argument, else the env knob, else 3600.

    A value that is not a positive integer is ignored with a warning rather
    than raised: a mistyped knob must not take the whole buffer down, and the
    hour it falls back to is the behaviour every earlier release had.
    """
    if explicit is not None:
        return int(explicit)
    raw = (os.getenv(WORKING_MEMORY_TTL_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_WORKING_MEMORY_TTL
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning(
            "%s=%r is not a positive integer; using %d s",
            WORKING_MEMORY_TTL_ENV, raw, DEFAULT_WORKING_MEMORY_TTL,
        )
        return DEFAULT_WORKING_MEMORY_TTL
    return value

# Context variable for per-request token override.
# When set (e.g. by kumiho-FastAPI), _get_fresh_token() uses this
# instead of the local filesystem cache.
_token_override_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "redis_token_override", default=None,
)

#: Where the dev escape hatch points when nothing else names a Redis.
HOSTED_LOCAL_REDIS_DEFAULT = "redis://127.0.0.1:6379"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def redact_redis_url(url: Optional[str]) -> str:
    """A Redis URL safe to put in a log line.

    An Upstash URL is ``rediss://default:<token>@host:port`` — the token IS
    the credential, and the one place this URL gets logged is a WARNING that
    an operator is meant to find and read. Logs get shipped, pasted into
    tickets and shown in CI output, so the userinfo has to go.

    Everything else is kept, because the point of the message is to tell an
    operator *which* Redis a hosted process is talking to.
    """
    if not url:
        return "<none>"
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url
    userinfo, at, hostpart = rest.rpartition("@")
    if not at:
        return url
    user = userinfo.split(":", 1)[0]
    return f"{scheme}://{user}:***@{hostpart}" if user else f"{scheme}://***@{hostpart}"


def hosted_local_redis_url() -> Optional[str]:
    """Direct Redis URL for the hosted server's dev mode, or ``None``.

    Hosted mode normally has exactly one route to Redis — the control-plane
    proxy — because that is what namespaces keys per tenant and authenticates
    per request. WP-C's ``KUMIHO_MCP_DEV_MODE=ce`` has no control plane at
    all, so without an escape hatch it has no Redis at all, and the hosted
    path cannot be exercised locally against a CE backend.

    So ``KUMIHO_HOSTED_LOCAL_REDIS=1`` opts into a direct connection, taken
    from ``KUMIHO_LOCAL_REDIS_URL``, else ``UPSTASH_REDIS_URL``, else
    :data:`HOSTED_LOCAL_REDIS_DEFAULT`. Keys are still built by the same
    tenant/user-prefixed key methods (``kumiho:memory:{tenant}:…``, and the
    active-session pointer carries ``{context}:{user}``), so two tenants
    sharing one dev Redis stay in separate key spaces exactly as they do
    behind the proxy.

    **It also requires ``KUMIHO_MCP_HOSTED=1``**, and warns rather than acting
    when that is missing. Gating on the coarse process-wide flag rather than
    on ``is_hosted()`` is the point: a request context alone would let a stray
    env var redirect a plugin user's working memory to localhost. Turning this
    on takes saying so twice.
    """
    if not _env_flag("KUMIHO_HOSTED_LOCAL_REDIS"):
        return None
    if not hosted_mode():
        logger.warning(
            "KUMIHO_HOSTED_LOCAL_REDIS is set but KUMIHO_MCP_HOSTED is not — "
            "ignoring it. The direct-Redis escape hatch belongs to the hosted "
            "server's dev mode and never changes local plugin behavior.",
        )
        return None
    return (
        os.getenv("KUMIHO_LOCAL_REDIS_URL", "").strip()
        or os.getenv("UPSTASH_REDIS_URL", "").strip()
        or HOSTED_LOCAL_REDIS_DEFAULT
    )


class RedisDiscoveryError(RuntimeError):
    """Raised when Redis discovery fails and no fallback is available."""


@dataclass(frozen=True)
class RedisDiscoveryResult:
    redis_url: str
    tenant_id: Optional[str] = None
    region_code: Optional[str] = None


class RedisMemoryBuffer:
    """Short-term memory buffer using Upstash Redis.

    Uses the `kumiho:memory:*` namespace to avoid conflicts with event streaming.
    """

    MEMORY_PREFIX = "kumiho:memory"

    def __init__(
        self,
        *,
        redis_url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        tenant_hint: Optional[str] = None,
        tenant_id: Optional[str] = None,
        control_plane_url: Optional[str] = None,
        discovery_timeout: float = 10.0,
        force_refresh: bool = False,
        prefer_discovery: bool = True,
        proxy_url: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.default_ttl = resolve_working_memory_ttl(default_ttl)
        self.tenant_hint = tenant_hint
        self.tenant_id = tenant_id
        # Hosted mode is decided ONCE, at construction, and remembered: a
        # per-tenant manager is built inside a request and then reused by
        # later ones, so a buffer that re-derived "am I hosted?" per call
        # would flip to the single-tenant rules the moment it were touched
        # outside a request (an eviction sweep, a background consolidation).
        self._hosted = is_hosted()
        if self._hosted:
            # Operator-supplied control-plane URL wins in hosted mode: the RS
            # is pointed at a region/staging origin by env, and the compiled-in
            # default would silently send one tenant's traffic to production.
            # Deliberately NOT consulted on the stdio path — that would change
            # today's behavior for anyone who happens to have the var set.
            self.control_plane_url = (
                control_plane_url
                or os.getenv("KUMIHO_CONTROL_PLANE_URL")
                or DEFAULT_CONTROL_PLANE_URL
            )
        else:
            self.control_plane_url = control_plane_url or DEFAULT_CONTROL_PLANE_URL
        self.discovery_timeout = discovery_timeout
        self.force_refresh = force_refresh
        self.proxy_url = proxy_url or os.getenv("KUMIHO_MEMORY_PROXY_URL")

        if not self.tenant_id:
            if self._hosted:
                # The request's tenant, never the machine's cached one:
                # ~/.kumiho holds whichever tenant last ran `kumiho-auth
                # login` on the host, which in a shared server is nobody's.
                ctx = current_request()
                if ctx is not None:
                    self.tenant_id = ctx.tenant_id
            else:
                cached_tenant = self._load_cached_tenant()
                if cached_tenant:
                    self.tenant_id = cached_tenant.get("tenant_id")

        resolved_url = redis_url
        if (
            not resolved_url
            and prefer_discovery
            and client is None
            and not self.proxy_url
            and not self._hosted
        ):
            discovery = self._discover_upstash_url()
            if discovery:
                resolved_url = discovery.redis_url
                if not self.tenant_id:
                    self.tenant_id = discovery.tenant_id

        # Called unconditionally, because it also carries the "you set this but
        # not KUMIHO_MCP_HOSTED, so it is being ignored" warning — which has to
        # reach an operator who set it in the wrong process.
        local_dev_url = hosted_local_redis_url()

        if not resolved_url and not self._hosted:
            # Ambient Redis credentials are a single-tenant convenience. In a
            # shared server they would hand every tenant the SAME database,
            # under keys the proxy namespaces per tenant precisely so that
            # cannot happen — so hosted mode's route to Redis is the
            # control-plane proxy, authenticated per request. (The one way
            # past that is the deliberate dev opt-in below, which still
            # namespaces every key by tenant and user.)
            resolved_url = os.getenv("KUMIHO_UPSTASH_REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")
        elif not resolved_url and self._hosted and local_dev_url and not self.proxy_url:
            # Dev-mode escape hatch (see hosted_local_redis_url). Beaten by an
            # explicitly configured proxy, so a deployment that has a control
            # plane keeps using it even if the flag is left set by accident.
            # WARNING, not INFO: a hosted process talking straight to a Redis
            # is a fact an operator must be able to find in the logs, and this
            # fires once per manager build rather than once per operation.
            resolved_url = local_dev_url
            logger.warning(
                "KUMIHO_HOSTED_LOCAL_REDIS is active: hosted memory is using a "
                "DIRECT Redis connection (%s) instead of the control-plane "
                "proxy. Keys stay namespaced per tenant and user, but the "
                "per-request token is not checked by anything. Development "
                "only — never enable this in a deployment serving real tenants.",
                redact_redis_url(resolved_url),
            )

        # Auto-fallback: when no direct Redis URL is available, use the
        # control-plane memory proxy so clients never need the raw Redis secret.
        if client is None and not resolved_url and not self.proxy_url:
            self.proxy_url = self._build_proxy_url()

        if client is None and not resolved_url and not self.proxy_url:
            raise RedisDiscoveryError(
                "Unable to resolve Upstash Redis URL. "
                "Run 'kumiho-auth login' to enable control-plane discovery, set "
                "UPSTASH_REDIS_URL / KUMIHO_UPSTASH_REDIS_URL, or configure "
                "KUMIHO_MEMORY_PROXY_URL."
            )

        self.redis_url = resolved_url
        self._client_loop = None
        if client is not None:
            # Caller-injected client (e.g. a test double) — never auto-recreate it.
            self._client = client
            self._owns_client = False
        else:
            self._client = (
                redis.from_url(self.redis_url, decode_responses=True) if self.redis_url else None
            )
            self._owns_client = self._client is not None

    @property
    def client(self):
        """The Redis client, rebound to the currently-running event loop.

        ``redis.asyncio`` binds a connection to the loop it is first used on. A
        caller that runs several ``asyncio.run()`` calls in one process (History
        Backfill replays one ``reflect()`` per session, each under its own fresh
        loop) would otherwise reuse a connection tied to an already-closed loop
        and crash with "Event loop is closed" on the 2nd session. When we created
        the client ourselves from ``redis_url`` we cheaply recreate it on loop
        change (``from_url`` is lazy — no connection until first command). A
        caller-injected client is left untouched.
        """
        if self._client is None or not self._owns_client:
            return self._client
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._client
        if self._client_loop is not loop:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            self._client_loop = loop
        return self._client

    @client.setter
    def client(self, value) -> None:
        self._client = value
        self._client_loop = None
        # An explicitly-assigned client is caller-owned (test double, or the
        # backfill guard's fresh per-loop client) — do not auto-rebind it.
        self._owns_client = False

    def _session_messages_key(self, project: str, session_id: str) -> str:
        tenant_prefix = self.tenant_id or self.tenant_hint
        if tenant_prefix:
            return f"{self.MEMORY_PREFIX}:{tenant_prefix}:{project}:sessions:{session_id}:messages"
        return f"{self.MEMORY_PREFIX}:{project}:sessions:{session_id}:messages"

    def _session_metadata_key(self, project: str, session_id: str) -> str:
        tenant_prefix = self.tenant_id or self.tenant_hint
        if tenant_prefix:
            return f"{self.MEMORY_PREFIX}:{tenant_prefix}:{project}:sessions:{session_id}:metadata"
        return f"{self.MEMORY_PREFIX}:{project}:sessions:{session_id}:metadata"

    def _sequence_key(self, user_canonical_id: str, date_str: str) -> str:
        tenant_prefix = self.tenant_id or self.tenant_hint
        if tenant_prefix:
            return f"{self.MEMORY_PREFIX}:{tenant_prefix}:session_seq:{user_canonical_id}:{date_str}"
        return f"{self.MEMORY_PREFIX}:session_seq:{user_canonical_id}:{date_str}"

    def _session_generation_key(self, base_session_id: str) -> str:
        tenant_prefix = self.tenant_id or self.tenant_hint
        if tenant_prefix:
            return f"{self.MEMORY_PREFIX}:{tenant_prefix}:session_gen:{base_session_id}"
        return f"{self.MEMORY_PREFIX}:session_gen:{base_session_id}"

    async def get_session_generation(self, base_session_id: str) -> int:
        """The consolidation generation of a host-session id (0 = never).

        The host env id is stable for a whole conversation, so consolidation
        cannot rotate it the way generated ids rotate — without a generation
        suffix the tool path resolves the consolidated, cleared bucket again
        and a second consolidation overwrites the first conversation's
        artifact (PR #4 review, round 5). The counter lives in Redis rather
        than in-process so sibling server processes derive the SAME rotated
        id and a server restart cannot silently reset the generation.
        """
        if self.client is None:
            try:
                response = await self._proxy_request(
                    action="get_session_generation",
                    payload={"base_session_id": base_session_id},
                )
            except Exception:
                # An older proxy server does not know this action; degrade
                # to generation 0 (the base id) — the pre-existing
                # behaviour, same precedent as only_if/nx (round 6).
                return 0
            try:
                return int(response.get("generation", 0)) if isinstance(response, dict) else 0
            except (TypeError, ValueError):
                return 0

        key = self._session_generation_key(base_session_id)
        value = await self.client.get(key)
        try:
            generation = int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0
        if generation:
            # SLIDING TTL: bump-only expiry meant a conversation that stayed
            # on one env id for >24h after its last consolidation regressed
            # to the base '{env}' id — the consolidated, cleared bucket, and
            # the artifact-overwrite target the generation exists to protect
            # (round 6). Refreshing on read keeps the counter alive for as
            # long as the conversation is.
            await self.client.expire(key, 86400)
        return generation

    async def bump_session_generation(
        self, base_session_id: str, *, ttl_seconds: int = 86400
    ) -> int:
        """Advance the consolidation generation (see get_session_generation)."""
        if self.client is None:
            try:
                response = await self._proxy_request(
                    action="bump_session_generation",
                    payload={
                        "base_session_id": base_session_id,
                        "ttl_seconds": ttl_seconds,
                    },
                )
            except Exception:
                # Older proxy server: rotation silently unavailable on
                # hosted until the server learns the action — documented
                # degradation, same precedent as only_if/nx (round 6).
                return 1
            try:
                return int(response.get("generation", 1)) if isinstance(response, dict) else 1
            except (TypeError, ValueError):
                return 1

        key = self._session_generation_key(base_session_id)
        value = await self.client.incr(key)
        await self.client.expire(key, ttl_seconds)
        return int(value)

    MAX_MESSAGE_SIZE = 64 * 1024  # 64 KiB per message

    async def add_message(
        self,
        *,
        project: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to working memory."""
        if not project or not project.strip():
            raise ValueError("project must be a non-empty string")
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not content:
            raise ValueError("content must be a non-empty string")
        if len(content) > self.MAX_MESSAGE_SIZE:
            raise ValueError(
                f"content exceeds maximum size ({len(content)} > {self.MAX_MESSAGE_SIZE} bytes)"
            )

        if self.client is None:
            result = await self._proxy_request(
                action="add_message",
                payload={
                    "project": project,
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "metadata": metadata or {},
                    "default_ttl": self.default_ttl,
                },
            )
            # An older proxy server returns its payload without the mint
            # flag; derive it from message_count so hosted mode carries the
            # same drift signal as the local path (PR #4 review, round 4).
            if isinstance(result, dict) and "created_bucket" not in result:
                result["created_bucket"] = result.get("message_count") == 1
            return result

        key = self._session_messages_key(project, session_id)
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        # RPUSH atomically returns the post-push length. A separate LLEN read
        # raced concurrent writers: two first-writers could both observe
        # count 2 and NEITHER report created_bucket (PR #4 review, round 3).
        count = await self.client.rpush(key, json.dumps(message))
        await self.client.expire(key, self.default_ttl)

        return {
            "success": True,
            "message_id": f"{session_id}:{count}",
            "message_count": count,
            # A write that MINTS a bucket is the one observable trace of a
            # caller addressing a session that did not exist — a drifted or
            # mistyped session_id lands here looking exactly like a first
            # message (issue #3). Surfacing it lets the caller notice; a read
            # of a wrong id returns a clean empty indistinguishable from a
            # new session, so the write is where the signal has to live.
            "created_bucket": count == 1,
        }

    async def get_messages(
        self,
        *,
        project: str,
        session_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Retrieve recent messages from working memory."""
        if self.client is None:
            return await self._proxy_request(
                action="get_messages",
                payload={
                    "project": project,
                    "session_id": session_id,
                    "limit": limit,
                    "offset": offset,
                },
            )

        key = self._session_messages_key(project, session_id)
        start = -limit - offset
        end = -1 - offset if offset > 0 else -1
        messages = await self.client.lrange(key, start, end)
        parsed = [json.loads(msg) for msg in messages]
        total = await self.client.llen(key)
        if total > 0:
            # SLIDING ON READ, not only on write.  Only add_message refreshed
            # the TTL, so a session that engaged (a read) every turn but
            # reflected (a write) less than once an hour lost its buffer and
            # the next reflect minted a fresh one -- every mid-session
            # ``created_bucket`` in four days of Claude Code transcripts was
            # exactly that gap or a consolidate.  A read of a live bucket now
            # keeps it alive for another full TTL; a missing bucket is left
            # missing (EXPIRE would be a no-op anyway).
            await self.client.expire(key, self.default_ttl)
            ttl = self.default_ttl
        else:
            ttl = await self.client.ttl(key)

        return {
            "messages": parsed,
            "session_id": session_id,
            "message_count": total,
            "ttl_remaining": ttl if ttl > 0 else 0,
        }

    async def set_session_metadata(
        self,
        project: str,
        session_id: str,
        metadata: Dict[str, str],
    ) -> None:
        """Store session-level metadata (e.g. user_id, context)."""
        if self.client is None:
            await self._proxy_request(
                action="set_session_metadata",
                payload={
                    "project": project,
                    "session_id": session_id,
                    "metadata": metadata,
                },
            )
            return

        key = self._session_metadata_key(project, session_id)
        await self.client.hset(key, mapping=metadata)
        await self.client.expire(key, self.default_ttl)

    async def get_session_metadata(
        self,
        project: str,
        session_id: str,
    ) -> Dict[str, str]:
        """Retrieve session-level metadata."""
        if self.client is None:
            result = await self._proxy_request(
                action="get_session_metadata",
                payload={
                    "project": project,
                    "session_id": session_id,
                },
            )
            return result.get("metadata", {})

        key = self._session_metadata_key(project, session_id)
        data = await self.client.hgetall(key)
        return data or {}

    async def clear_session(self, project: str, session_id: str) -> Dict[str, Any]:
        """Clear working memory for a session."""
        if self.client is None:
            return await self._proxy_request(
                action="clear_session",
                payload={"project": project, "session_id": session_id},
            )

        key = self._session_messages_key(project, session_id)
        count = await self.client.llen(key)
        await self.client.delete(key)
        await self.client.delete(self._session_metadata_key(project, session_id))
        return {"success": True, "cleared_count": count}

    async def list_sessions(self, project: str, limit: int = 20) -> Dict[str, Any]:
        """List active sessions with working memory."""
        if self.client is None:
            return await self._proxy_request(
                action="list_sessions",
                payload={"project": project, "limit": limit},
            )

        tenant_prefix = self.tenant_id or self.tenant_hint
        pattern = f"{self.MEMORY_PREFIX}:{project}:sessions:*:messages"
        if tenant_prefix:
            pattern = f"{self.MEMORY_PREFIX}:{tenant_prefix}:{project}:sessions:*:messages"
        cursor = 0
        keys: List[str] = []

        while True:
            cursor, batch = await self.client.scan(cursor, match=pattern, count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        session_ids: List[str] = []
        for key in keys[:limit]:
            # The session component is everything between the 'sessions'
            # segment and the trailing ':messages'. It has to be a join, not
            # parts[idx + 1]: the ids this system MINTS contain three colons
            # ('{context}:user-{hash}:{date}:{seq}'), so taking one segment
            # truncated every generated id and this listing could never
            # round-trip the system's own sessions (issue #3).
            parts = key.split(":")
            if "sessions" in parts:
                idx = parts.index("sessions")
                if len(parts) > idx + 2:
                    session_ids.append(":".join(parts[idx + 1:-1]))

        return {"sessions": session_ids, "total_sessions": len(keys)}

    async def next_session_sequence(
        self,
        *,
        user_canonical_id: str,
        date_str: str,
        ttl_seconds: int = 172800,
    ) -> int:
        """Increment and return the daily session sequence for a user."""
        if self.client is None:
            response = await self._proxy_request(
                action="next_sequence",
                payload={
                    "user_canonical_id": user_canonical_id,
                    "date_str": date_str,
                    "ttl_seconds": ttl_seconds,
                },
            )
            return int(response.get("sequence", 1))

        key = self._sequence_key(user_canonical_id, date_str)
        value = await self.client.incr(key)
        await self.client.expire(key, ttl_seconds)
        return int(value)

    def _active_session_key(self, context: str, user_canonical_id: str) -> str:
        tenant_prefix = self.tenant_id or self.tenant_hint
        if tenant_prefix:
            return f"{self.MEMORY_PREFIX}:{tenant_prefix}:active_session:{context}:{user_canonical_id}"
        return f"{self.MEMORY_PREFIX}:active_session:{context}:{user_canonical_id}"

    async def get_active_session(
        self,
        *,
        context: str,
        user_canonical_id: str,
    ) -> Optional[str]:
        """Return the active session_id for a user/context, or None."""
        if self.client is None:
            response = await self._proxy_request(
                action="get_active_session",
                payload={"context": context, "user_canonical_id": user_canonical_id},
            )
            return response.get("session_id")

        key = self._active_session_key(context, user_canonical_id)
        return await self.client.get(key)

    async def set_active_session(
        self,
        *,
        context: str,
        user_canonical_id: str,
        session_id: str,
        ttl_seconds: int = 86400,
        nx: bool = False,
    ) -> Optional[bool]:
        """Persist the active session_id for a user/context (default TTL 24 h).

        With ``nx=True`` this is a compare-and-claim: the pointer is written
        only if absent, and the return value says whether THIS caller won.
        A blind SET was last-writer-wins — two concurrent resolutions that
        both missed the pointer each minted an id and both registered it,
        and everything written under the loser became unreachable to every
        later default-resolved call (PR #4 review, round 3).
        """
        if self.client is None:
            response = await self._proxy_request(
                action="set_active_session",
                payload={
                    "context": context,
                    "user_canonical_id": user_canonical_id,
                    "session_id": session_id,
                    "ttl_seconds": ttl_seconds,
                    # Older proxy servers ignore unknown fields and perform
                    # the plain SET — the pre-existing behaviour.
                    "nx": nx,
                },
            )
            if nx:
                claimed = response.get("claimed") if isinstance(response, dict) else None
                return True if claimed is None else bool(claimed)
            return None

        key = self._active_session_key(context, user_canonical_id)
        if nx:
            result = await self.client.set(key, session_id, ex=ttl_seconds, nx=True)
            return bool(result)
        await self.client.set(key, session_id, ex=ttl_seconds)
        return None

    async def clear_active_session(
        self,
        *,
        context: str,
        user_canonical_id: str,
        only_if: Optional[str] = None,
    ) -> None:
        """Delete the active session pointer (called after consolidation).

        ``only_if`` makes the delete conditional on the pointer still holding
        that session_id. Without it this was a plain DELETE: consolidating ANY
        session for a (context, user) — a backfill id, a historical fragment —
        severed the LIVE conversation's continuity pointer, so its next
        default-resolved call minted a fresh session mid-conversation
        (issue #3). Pass the id you consolidated; the live pointer survives
        unless it is the one you actually closed out.

        The get/compare/delete pair is not atomic; the race window (another
        writer moving the pointer between the read and the delete) loses a
        pointer that was being replaced anyway, which is the pre-existing
        behaviour, never worse.
        """
        if self.client is None:
            if only_if is not None:
                # The compare must happen CLIENT-side here: an older proxy
                # server ignores unknown payload fields, so forwarding
                # only_if alone silently restored the unconditional delete on
                # hosted deployments while the local path was fixed (PR #4
                # review, round 2). get_active_session is a main-era proxy
                # action every server implements.
                current = await self.get_active_session(
                    context=context, user_canonical_id=user_canonical_id,
                )
                if current != only_if:
                    return
            await self._proxy_request(
                action="clear_active_session",
                payload={
                    "context": context,
                    "user_canonical_id": user_canonical_id,
                    # Newer servers may enforce this atomically as well.
                    "only_if": only_if,
                },
            )
            return

        key = self._active_session_key(context, user_canonical_id)
        if only_if is not None:
            current = await self.client.get(key)
            if current != only_if:
                # Covers BOTH a pointer holding another session AND an
                # absent pointer (None): with nothing to clear, issuing the
                # DELETE anyway could only destroy a pointer some concurrent
                # resolver set between this read and the delete.
                return
        await self.client.delete(key)

    async def close(self) -> None:
        """Close Redis connection."""
        if hasattr(self.client, "close"):
            await self.client.close()

    async def __aenter__(self) -> "RedisMemoryBuffer":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @staticmethod
    def _get_fresh_token(*, force_refresh: bool = False) -> str:
        """Load a bearer token, refreshing from Firebase if expired.

        Mirrors the retry strategy of the gRPC ``_AutoLoginInterceptor``:
        first try the cached token, then call ``ensure_token`` to silently
        refresh through the Firebase refresh-token flow.

        When running inside kumiho-FastAPI (or any server that sets the
        ``_token_override_var`` context variable), the override token is
        returned immediately — no filesystem lookup is needed.

        The hosted connector sets no override; it sets a
        :class:`RequestContext`, and the caller's own bearer token lives on
        it. Resolution happens HERE, per proxy call, rather than being baked
        into the buffer at construction — one buffer serves every request for
        its tenant, and each of those requests carries a different (and
        expiring) token.
        """
        # Server-injected token takes priority (e.g. kumiho-FastAPI Playground).
        override = _token_override_var.get()
        if override:
            return override

        ctx = current_request()
        if ctx is not None and ctx.auth_token:
            return ctx.auth_token

        if is_hosted():
            # No request token and no override, in a process that serves many
            # tenants: the local credential cache belongs to the machine's
            # operator, and using it would run one tenant's memory operation
            # as another identity. Fail instead.
            raise RedisDiscoveryError(
                "No request credentials available for the memory proxy in "
                "hosted mode (KUMIHO_MCP_HOSTED). The caller's token must be "
                "supplied via kumiho.request_context; local ~/.kumiho "
                "credentials are never used here."
            )

        if not force_refresh:
            token = load_firebase_token() or load_bearer_token()
            if token:
                return token

        # Attempt a silent (non-interactive) refresh via the SDK auth module.
        # ensure_token() may return a Control Plane JWT, but the memory
        # proxy validates Firebase ID tokens directly.  We call
        # ensure_token() for its *side-effect* (refreshing + saving creds)
        # and then re-read the Firebase token from disk.
        try:
            from kumiho.auth_cli import ensure_token

            ensure_token(
                interactive=False,
                force_refresh=force_refresh,
            )
            # Re-read the Firebase ID token that ensure_token just saved.
            firebase_token = load_firebase_token()
            if firebase_token:
                return firebase_token
        except Exception as exc:
            logger.debug("Token refresh attempt failed: %s", exc)

        # Last resort: re-read the cache in case another process refreshed it.
        token = load_firebase_token() or load_bearer_token()
        if token:
            return token

        raise RedisDiscoveryError(
            "No credentials available for memory proxy. Run 'kumiho-auth login' first."
        )

    async def _proxy_request(self, *, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.proxy_url:
            raise RedisDiscoveryError("Redis client is not configured and no proxy URL is set.")

        body = {"action": action, **payload}
        if self.tenant_hint:
            body["tenant_hint"] = self.tenant_hint

        def _do_request(bearer: str) -> requests.Response:
            return requests.post(
                self.proxy_url,
                json=body,
                headers={"Authorization": f"Bearer {bearer}"},
                timeout=self.discovery_timeout,
            )

        def _execute() -> Dict[str, Any]:
            token = self._get_fresh_token()
            response = _do_request(token)

            # On auth failure, force-refresh the token and retry once
            # (same pattern as the gRPC _AutoLoginInterceptor).
            if response.status_code in (401, 403):
                logger.debug(
                    "Memory proxy returned %s — refreshing token and retrying",
                    response.status_code,
                )
                token = self._get_fresh_token(force_refresh=True)
                response = _do_request(token)

            if response.status_code >= 400:
                raise RedisDiscoveryError(
                    f"Memory proxy error {response.status_code}: {response.text[:200]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise RedisDiscoveryError("Memory proxy returned invalid JSON") from exc

        return await asyncio.to_thread(_execute)

    def _discover_upstash_url(self) -> Optional[RedisDiscoveryResult]:
        """Attempt to discover a direct Redis URL via the control plane.

        This only succeeds when the discovery response contains the URL
        (e.g. in guardrails or service catalogue).  When it fails the
        caller falls back to env vars and then to the proxy.
        """
        try:
            token = self._get_fresh_token()
        except RedisDiscoveryError:
            return None

        try:
            firebase_token = self._ensure_firebase_token(token)
            manager = DiscoveryManager(
                control_plane_url=self.control_plane_url,
                timeout=self.discovery_timeout,
            )
            record = manager.resolve(
                id_token=firebase_token,
                tenant_hint=self.tenant_hint,
                force_refresh=self.force_refresh,
            )
        except Exception:
            return None

        # The control plane intentionally does NOT expose the raw Redis
        # URL to clients.  If guardrails happen to contain it (e.g. for
        # privileged service accounts) we can use it, otherwise return
        # the tenant info so the caller can fall back to the proxy.
        url = self._extract_redis_url(record.guardrails or {})

        if not url:
            # No direct URL available — store tenant info so proxy
            # mode can set the correct tenant prefix.
            if not self.tenant_id:
                self.tenant_id = record.tenant_id
            return None

        return RedisDiscoveryResult(
            redis_url=url,
            tenant_id=record.tenant_id,
            region_code=record.region.region_code,
        )

    @staticmethod
    def _load_cached_tenant() -> Optional[Dict[str, Optional[str]]]:
        try:
            cache = DiscoveryCache(DEFAULT_CACHE_PATH)
            record = cache.load(_DEFAULT_CACHE_KEY)
        except Exception:
            return None
        if not record:
            return None
        return {
            "tenant_id": record.tenant_id,
            "tenant_name": record.tenant_name,
        }

    def _build_proxy_url(self) -> Optional[str]:
        """Construct the memory proxy URL from the control plane URL.

        The control plane exposes ``/api/memory/redis`` which acts as a
        server-side proxy so clients never need the raw Upstash secret.
        """
        base = self.control_plane_url
        if not base:
            base = os.getenv("KUMIHO_CONTROL_PLANE_URL") or DEFAULT_CONTROL_PLANE_URL
        if not base:
            return None
        return f"{base.rstrip('/')}/api/memory/redis"

    @staticmethod
    def _extract_redis_url(payload: Dict[str, Any]) -> Optional[str]:
        candidates: Iterable[Tuple[str, ...]] = [
            ("upstash_redis_url",),
            ("redis_url",),
            ("upstash", "redis_url"),
            ("upstash", "url"),
            ("upstash", "redis", "url"),
            ("services", "upstash", "redis_url"),
            ("services", "upstash", "url"),
            ("services", "redis", "url"),
            ("services", "redis", "redis_url"),
        ]

        for path in candidates:
            current: Any = payload
            found = True
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    found = False
                    break
                current = current[key]
            if found and isinstance(current, str) and current:
                return current
        return None

    @staticmethod
    def _ensure_firebase_token(token: str) -> str:
        if RedisMemoryBuffer._looks_like_control_plane_token(token):
            firebase = load_firebase_token()
            if not firebase:
                raise RedisDiscoveryError(
                    "Control-plane token detected but no Firebase ID token is available. "
                    "Run 'kumiho-auth login' to refresh credentials."
                )
            return firebase
        return token

    @staticmethod
    def _looks_like_control_plane_token(token: str) -> bool:
        parts = token.split(".")
        if len(parts) < 2:
            return False
        try:
            import base64
            import json

            payload = parts[1]
            padding = "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode((payload + padding).encode("utf-8"))
            claims = json.loads(decoded)
        except Exception:
            return False

        if isinstance(claims, dict):
            if isinstance(claims.get("tenant_id"), str):
                return True
            iss = claims.get("iss")
            if isinstance(iss, str) and iss.startswith("https://control.kumiho.cloud"):
                return True
            aud = claims.get("aud")
            if isinstance(aud, str) and aud.startswith("kumiho-server"):
                return True
        return False
