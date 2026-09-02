"""Fakes for the hosted (multi-tenant) memory path.

Two of them, and the split matters:

* :class:`FakeMemoryProxy` stands in for the control-plane Redis proxy
  (``POST {cp}/api/memory/redis``). It partitions its state by the **bearer
  token it was handed**, not by anything the caller asserts — so if a request
  ever reached the proxy with the wrong tenant's token, the fake would happily
  serve that tenant's data and the assertions downstream would catch it. A
  fake that partitioned by ``tenant_id`` from the payload could not detect the
  bug it is here to detect.

* :class:`FakeGraph` stands in for the gRPC backend, recording every store and
  answering retrieves per tenant on the same principle.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional


class FakeProxyResponse:
    """The subset of ``requests.Response`` the buffer touches."""

    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeMemoryProxy:
    """In-memory stand-in for the control-plane memory proxy.

    Install with ``monkeypatch.setattr(redis_memory.requests, "post", proxy)``.

    State is partitioned by the tenant the **presented bearer token** resolves
    to — the way the real proxy derives ``tenant_id`` from the JWT's claims and
    ignores anything the client asserts in the body. Tokens are mapped through
    :meth:`register`; an unregistered token is its own tenant, which is the
    convenient default for single-token tests and still keeps two tenants apart.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token_tenants: Dict[str, str] = {}
        # tenant -> {"messages": {...}, "pointers": {...}, ...}
        self._state: Dict[str, Dict[str, Any]] = {}
        #: Every call seen, in order, with the bearer that carried it. The
        #: concurrency test asserts over this as well as over stored data: a
        #: wrong token that writes a key nobody reads back is still a leak.
        self.calls: List[Dict[str, Any]] = []
        self.unauthenticated_calls: List[Dict[str, Any]] = []

    # -- helpers --------------------------------------------------------
    def register(self, token: str, tenant: str) -> None:
        """Bind a bearer token to a tenant, as the control plane's JWT
        verification would. Lets a test rotate a token without splitting the
        tenant's state."""
        with self._lock:
            self._token_tenants[token] = tenant

    def _tenant_for(self, bearer: str) -> str:
        return self._token_tenants.get(bearer, bearer)

    def _bucket(self, tenant: str) -> Dict[str, Any]:
        return self._state.setdefault(tenant, {
            "messages": {},
            "metadata": {},
            "pointers": {},
            "sequences": {},
            "generations": {},
        })

    def tokens_seen(self) -> List[str]:
        with self._lock:
            return sorted({c["bearer"] for c in self.calls})

    def actions_for(self, bearer: str) -> List[str]:
        with self._lock:
            return [c["action"] for c in self.calls if c["bearer"] == bearer]

    def messages(self, token_or_tenant: str, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            tenant = self._tenant_for(token_or_tenant)
            return list(self._bucket(tenant)["messages"].get(session_id, []))

    # -- the requests.post replacement -----------------------------------
    def __call__(self, url, json=None, headers=None, timeout=None, **kwargs):  # noqa: A002
        body = json or {}
        auth = (headers or {}).get("Authorization", "")
        bearer = auth[7:] if auth.startswith("Bearer ") else ""
        action = body.get("action", "")
        with self._lock:
            tenant = self._tenant_for(bearer)
            record = {
                "url": url, "action": action, "bearer": bearer,
                "tenant": tenant, "body": body,
            }
            self.calls.append(record)
            if not bearer:
                self.unauthenticated_calls.append(record)
                return FakeProxyResponse({"error": "unauthenticated"}, status_code=401)
            return FakeProxyResponse(self._dispatch(self._bucket(tenant), action, body))

    def _dispatch(self, state, action: str, body: Dict[str, Any]) -> Dict[str, Any]:
        session_id = body.get("session_id", "")
        if action == "add_message":
            bucket = state["messages"].setdefault(session_id, [])
            bucket.append({
                "role": body.get("role", "user"),
                "content": body.get("content", ""),
                "metadata": body.get("metadata") or {},
            })
            return {
                "success": True,
                "message_count": len(bucket),
                "created_bucket": len(bucket) == 1,
            }
        if action == "get_messages":
            bucket = state["messages"].get(session_id, [])
            limit = int(body.get("limit", 10) or 10)
            return {
                "messages": bucket[-limit:] if limit else list(bucket),
                "session_id": session_id,
                # The real proxy mirrors the local buffer's shape; callers
                # (handle_user_message) index message_count directly.
                "message_count": len(bucket),
                "ttl_remaining": 3600,
            }
        if action == "clear_session":
            state["messages"].pop(session_id, None)
            state["metadata"].pop(session_id, None)
            return {"success": True, "cleared": True}
        if action == "set_session_metadata":
            state["metadata"].setdefault(session_id, {}).update(
                body.get("metadata") or {}
            )
            return {"success": True}
        if action == "get_session_metadata":
            return {"metadata": state["metadata"].get(session_id, {})}
        if action == "get_active_session":
            key = (body.get("context", ""), body.get("user_canonical_id", ""))
            return {"session_id": state["pointers"].get(key)}
        if action == "set_active_session":
            key = (body.get("context", ""), body.get("user_canonical_id", ""))
            if body.get("nx") and key in state["pointers"]:
                return {"claimed": False, "session_id": state["pointers"][key]}
            state["pointers"][key] = body.get("session_id")
            return {"claimed": True}
        if action == "clear_active_session":
            key = (body.get("context", ""), body.get("user_canonical_id", ""))
            state["pointers"].pop(key, None)
            return {"success": True}
        if action == "next_sequence":
            key = (body.get("user_canonical_id", ""), body.get("date_str", ""))
            state["sequences"][key] = state["sequences"].get(key, 0) + 1
            return {"sequence": state["sequences"][key]}
        if action == "get_session_generation":
            return {"generation": state["generations"].get(
                body.get("base_session_id", ""), 0,
            )}
        if action == "bump_session_generation":
            base = body.get("base_session_id", "")
            state["generations"][base] = state["generations"].get(base, 0) + 1
            return {"generation": state["generations"][base]}
        if action == "list_sessions":
            return {"sessions": list(state["messages"].keys())}
        return {"success": True}


class FakeGraph:
    """Per-tenant stand-in for the gRPC store/retrieve pair.

    ``store_for``/``retrieve_for`` hand out callables bound to one tenant, the
    way the RS binds a ``kumiho`` client per request. Anything stored under one
    tenant is invisible to the other — so a manager shared across tenants shows
    up as a retrieve returning the wrong tenant's krefs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stored: Dict[str, List[Dict[str, Any]]] = {}

    def store_for(self, tenant: str):
        async def _store(**payload):
            with self._lock:
                rows = self.stored.setdefault(tenant, [])
                rows.append(payload)
                index = len(rows)
            return {
                "success": True,
                "revision_kref": f"kref://{tenant}/mem/rev/{index}",
                "item_kref": f"kref://{tenant}/mem/item/{index}",
            }
        return _store

    def retrieve_for(self, tenant: str):
        async def _retrieve(**kwargs):
            with self._lock:
                count = len(self.stored.get(tenant, []))
            return {
                "revision_krefs": [
                    f"kref://{tenant}/mem/rev/{i + 1}" for i in range(max(count, 1))
                ],
            }
        return _retrieve

    def rows(self, tenant: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.stored.get(tenant, []))


class SessionFakeRedis:
    """The redis-py surface the session-resolution paths touch.

    ``tests/fakes.FakeRedis`` deliberately omits ``get``/``set``, so a manager
    built on it cannot exercise the active-session pointer at all — the tier
    these tests are mostly about.
    """

    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}
        self.lists: Dict[str, List[Any]] = {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None, nx=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    async def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)
            self.lists.pop(key, None)

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def expire(self, key, ttl):
        return True

    async def incr(self, key):
        value = int(self.kv.get(key, 0)) + 1
        self.kv[key] = value
        return value

    async def hset(self, key, mapping=None):
        self.kv.setdefault(key, {})
        self.kv[key].update(mapping or {})

    async def hgetall(self, key):
        value = self.kv.get(key)
        return dict(value) if isinstance(value, dict) else {}

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        n = len(items)
        if n == 0:
            return []
        if start < 0:
            start += n
        if end < 0:
            end += n
        start = max(start, 0)
        end = min(end, n - 1)
        if start > end:
            return []
        return items[start:end + 1]

    async def ttl(self, key):
        return 3600 if key in self.lists or key in self.kv else -2

    async def scan(self, cursor, match, count=100):
        import fnmatch
        keys = [k for k in {**self.kv, **self.lists} if fnmatch.fnmatch(k, match)]
        return 0, keys

    async def close(self):
        return None


class StubSummarizer:
    """Deterministic summarizer — never touches a provider."""

    light_model = "stub"
    model = "stub"

    async def summarize_conversation(self, messages, **kwargs):
        return {
            "title": "Stub",
            "summary": "stub summary",
            "type": "summary",
            "classification": {"topics": ["stub"], "entities": []},
        }

    async def summarize_for_storage(self, *args, **kwargs):
        return {"summary": "stub summary"}


class StubRedactor:
    def redact(self, text):
        return text

    def reject_credentials(self, text):
        return None


class FakeKumihoClient:
    """The bare minimum ``kumiho.use_client`` needs to accept a binding.

    WP-A's ``_ensure_configured`` raises in hosted mode unless the hosting
    layer has bound a request-scoped client — deliberately, because the
    fallback is ``auto_configure_from_discovery``, which reads the machine's
    ``~/.kumiho`` credentials and installs them process-globally. Hosted tests
    therefore have to enter ``use_client`` the way WP-C's RS does; this is
    that client, carrying its tenant so a mis-bound one is identifiable.
    """

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id
        self.endpoint = f"{tenant_id}.test.invalid:443"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<FakeKumihoClient {self.tenant_id}>"


def bind_request(ctx):
    """Enter ``request_context(ctx)`` AND ``kumiho.use_client(...)`` together.

    One helper because they are one thing: a hosted request is not correctly
    set up with only half of it, and WP-C enters both per request
    (plan §2.4). Returns a context manager.
    """
    from contextlib import ExitStack, contextmanager

    import kumiho

    from kumiho_memory._request_context import request_context

    @contextmanager
    def _bound():
        with ExitStack() as stack:
            stack.enter_context(request_context(ctx))
            use_client = getattr(kumiho, "use_client", None)
            if use_client is not None:
                stack.enter_context(use_client(FakeKumihoClient(ctx.tenant_id)))
            yield ctx

    return _bound()


def make_request_context(
    tenant_id: str,
    *,
    user_id: Optional[str] = None,
    token: Optional[str] = None,
    session_id: Optional[str] = None,
    context: str = "claude",
):
    """A :class:`RequestContext` with the fields the memory path reads."""
    from kumiho_memory._request_context import RequestContext

    return RequestContext(
        tenant_id=tenant_id,
        user_id=user_id or f"user-{tenant_id}",
        auth_token=token or f"token-{tenant_id}",
        context=context,
        session_id=session_id,
    )
