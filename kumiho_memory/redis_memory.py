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

logger = logging.getLogger(__name__)

# Context variable for per-request token override.
# When set (e.g. by kumiho-FastAPI), _get_fresh_token() uses this
# instead of the local filesystem cache.
_token_override_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "redis_token_override", default=None,
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
        default_ttl: int = 3600,
        tenant_hint: Optional[str] = None,
        tenant_id: Optional[str] = None,
        control_plane_url: Optional[str] = None,
        discovery_timeout: float = 10.0,
        force_refresh: bool = False,
        prefer_discovery: bool = True,
        proxy_url: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.default_ttl = int(default_ttl)
        self.tenant_hint = tenant_hint
        self.tenant_id = tenant_id
        self.control_plane_url = control_plane_url or DEFAULT_CONTROL_PLANE_URL
        self.discovery_timeout = discovery_timeout
        self.force_refresh = force_refresh
        self.proxy_url = proxy_url or os.getenv("KUMIHO_MEMORY_PROXY_URL")

        if not self.tenant_id:
            cached_tenant = self._load_cached_tenant()
            if cached_tenant:
                self.tenant_id = cached_tenant.get("tenant_id")

        resolved_url = redis_url
        if not resolved_url and prefer_discovery and client is None and not self.proxy_url:
            discovery = self._discover_upstash_url()
            if discovery:
                resolved_url = discovery.redis_url
                if not self.tenant_id:
                    self.tenant_id = discovery.tenant_id

        if not resolved_url:
            resolved_url = os.getenv("KUMIHO_UPSTASH_REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")

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
        self.client = client or (
            redis.from_url(self.redis_url, decode_responses=True) if self.redis_url else None
        )

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
            return await self._proxy_request(
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

        key = self._session_messages_key(project, session_id)
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        await self.client.rpush(key, json.dumps(message))
        await self.client.expire(key, self.default_ttl)
        count = await self.client.llen(key)

        return {
            "success": True,
            "message_id": f"{session_id}:{count}",
            "message_count": count,
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
        ttl = await self.client.ttl(key)
        total = await self.client.llen(key)

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
            parts = key.split(":")
            if "sessions" in parts:
                idx = parts.index("sessions")
                if len(parts) > idx + 1:
                    session_ids.append(parts[idx + 1])

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
    ) -> None:
        """Persist the active session_id for a user/context (default TTL 24 h)."""
        if self.client is None:
            await self._proxy_request(
                action="set_active_session",
                payload={
                    "context": context,
                    "user_canonical_id": user_canonical_id,
                    "session_id": session_id,
                    "ttl_seconds": ttl_seconds,
                },
            )
            return

        key = self._active_session_key(context, user_canonical_id)
        await self.client.set(key, session_id, ex=ttl_seconds)

    async def clear_active_session(
        self,
        *,
        context: str,
        user_canonical_id: str,
    ) -> None:
        """Delete the active session pointer (called after consolidation)."""
        if self.client is None:
            await self._proxy_request(
                action="clear_active_session",
                payload={"context": context, "user_canonical_id": user_canonical_id},
            )
            return

        key = self._active_session_key(context, user_canonical_id)
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
        """
        # Server-injected token takes priority (e.g. kumiho-FastAPI Playground).
        override = _token_override_var.get()
        if override:
            return override

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
