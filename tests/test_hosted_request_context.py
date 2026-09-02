"""Hosted (multi-tenant) behavior — plan §2.3.

The one property every test here defends: **nothing tenant-derived may be
shared through the process.** Not the manager, not the Redis credentials, not
the dedup guard, not the disk. The stdio plugin depends on the opposite —
one process, one tenant, one singleton — so each rule is checked twice: that
it holds under a request context, and that it is absent without one.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

import kumiho_memory.mcp_tools as mcp_tools
from kumiho_memory import redis_memory
from kumiho_memory._request_context import (
    RequestContext,
    current_request,
    hosted_mode,
    is_hosted,
    request_context,
)
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisDiscoveryError, RedisMemoryBuffer

from hosted_fakes import (
    FakeGraph,
    bind_request,
    FakeMemoryProxy,
    SessionFakeRedis,
    StubRedactor,
    StubSummarizer,
    make_request_context,
)


CONTROL_PLANE = "https://cp.test.invalid"
PROXY_URL = f"{CONTROL_PLANE}/api/memory/redis"


@pytest.fixture(autouse=True)
def _clean_process_state(monkeypatch, tmp_path):
    """Reset every process-global this package keeps, before AND after.

    Listed explicitly rather than swept: each entry is a piece of state that
    would otherwise carry one test's tenant into the next, which is the very
    bug class these tests exist to catch — a leaky fixture would hide it.
    """
    import kumiho_memory.entity_promotion as entity_promotion

    def _reset():
        mcp_tools._manager = None
        mcp_tools._tenant_managers.clear()
        mcp_tools._recall_recent.clear()
        mcp_tools._recall_scope_locks.clear()
        entity_promotion._project_cache.clear()
        entity_promotion._anchor_locks.clear()

    _reset()
    # Deterministic environment: the host running the suite may well have real
    # Redis / proxy / control-plane vars set.
    for var in (
        "UPSTASH_REDIS_URL",
        "KUMIHO_UPSTASH_REDIS_URL",
        "KUMIHO_MEMORY_PROXY_URL",
        "KUMIHO_MCP_HOSTED",
        "KUMIHO_HOSTED_LLM",
        "KUMIHO_HOSTED_LOCAL_REDIS",
        "KUMIHO_LOCAL_REDIS_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KUMIHO_CONTROL_PLANE_URL", CONTROL_PLANE)
    monkeypatch.setenv("KUMIHO_MEMORY_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("KUMIHO_RETRY_QUEUE_DIR", str(tmp_path / "retry"))
    monkeypatch.setenv("KUMIHO_FAILURE_LEDGER_DIR", str(tmp_path / "ledger"))
    yield
    _reset()


@pytest.fixture
def proxy(monkeypatch):
    """Route the Redis buffer's HTTP layer at a fake control-plane proxy."""
    fake = FakeMemoryProxy()
    monkeypatch.setattr(redis_memory.requests, "post", fake)
    return fake


@pytest.fixture
def graph():
    return FakeGraph()


@pytest.fixture
def hosted_managers(monkeypatch, graph):
    """Real hosted construction, with the gRPC seam replaced per tenant.

    Wrapping ``_build_hosted_manager`` instead of replacing it keeps the thing
    under test — buffer wiring, keyless summarizer, disabled disk — real; only
    the graph calls are faked, exactly where the RS binds a per-request
    ``kumiho`` client.
    """
    built = {}
    real_build = mcp_tools._build_hosted_manager

    def _build(ctx):
        manager = real_build(ctx)
        manager.memory_store = graph.store_for(ctx.tenant_id)
        manager.memory_retrieve = graph.retrieve_for(ctx.tenant_id)
        built[ctx.tenant_id] = manager
        return manager

    monkeypatch.setattr(mcp_tools, "_build_hosted_manager", _build)
    return built


def _local_manager(**kwargs):
    """A manager over a fake Redis — for resolution tests that need no proxy."""
    buffer = RedisMemoryBuffer(client=SessionFakeRedis(), redis_url="redis://test")
    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The vendored contract (§2.1)
# ---------------------------------------------------------------------------


def test_request_context_carries_the_fields_the_contract_promises():
    """WP-C builds these by name; a rename here is a silent integration break."""
    ctx = RequestContext(tenant_id="t1", user_id="u1", auth_token="tok")
    assert (ctx.tenant_id, ctx.user_id, ctx.auth_token) == ("t1", "u1", "tok")
    assert ctx.context == "claude"       # memory namespace default (§1 decision 10)
    assert ctx.session_id is None
    assert ctx.scopes == []
    for field in ("client_id", "tenant_slug", "region_code", "token_id"):
        assert getattr(ctx, field) is None


def test_request_context_is_frozen_and_scoped_to_the_block():
    ctx = make_request_context("t1")
    with pytest.raises(Exception):
        ctx.tenant_id = "other"  # type: ignore[misc]
    assert current_request() is None
    with request_context(ctx):
        assert current_request() is ctx
    assert current_request() is None


def test_request_context_does_not_leak_into_a_plain_thread():
    """Threads start with a fresh context — the isolation the RS relies on."""
    seen = []
    with request_context(make_request_context("t1")):
        thread = threading.Thread(target=lambda: seen.append(current_request()))
        thread.start()
        thread.join()
    assert seen == [None]


def test_the_vendored_shim_defers_to_the_sdk_when_it_is_installed():
    """The fallback exists to let this package ship ahead of the SDK, not to
    shadow it: with WP-A installed, ``kumiho_memory`` must be using the SDK's
    ContextVar, or the RS would enter one context and the memory layer would
    read another."""
    sdk = pytest.importorskip("kumiho.request_context")
    assert RequestContext is sdk.RequestContext
    assert current_request is sdk.current_request
    assert request_context is sdk.request_context
    assert hosted_mode is sdk.hosted_mode


def test_hosted_tool_calls_need_a_request_scoped_kumiho_client():
    """WP-A's ``_ensure_configured`` refuses to fall back to the machine's
    credentials in hosted mode. Pinned here because it is the contract WP-C
    must satisfy per request, and getting it wrong serves the OPERATOR's graph
    to a remote caller rather than failing."""
    mcp_server = pytest.importorskip("kumiho.mcp_server")
    if not hasattr(mcp_server, "_ensure_configured"):
        pytest.skip("SDK predates the hosted _ensure_configured guard")

    ctx = make_request_context("tenant-a")
    with request_context(ctx):        # request bound, client NOT bound
        try:
            mcp_server._ensure_configured()
        except RuntimeError as exc:
            assert "use_client" in str(exc)
        else:
            pytest.skip("SDK predates the hosted _ensure_configured guard")

    with bind_request(ctx):           # both bound, as the RS does
        assert mcp_server._ensure_configured() is True


def test_hosted_mode_and_is_hosted_read_the_env_and_the_request(monkeypatch):
    assert hosted_mode() is False and is_hosted() is False
    with request_context(make_request_context("t1")):
        assert hosted_mode() is False   # no env var...
        assert is_hosted() is True      # ...but a request is enough
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    assert hosted_mode() is True and is_hosted() is True


# ---------------------------------------------------------------------------
# Per-tenant manager cache (§2.3 item 1)
# ---------------------------------------------------------------------------


def test_get_manager_returns_a_distinct_manager_per_tenant(proxy, hosted_managers):
    with bind_request(make_request_context("tenant-a")):
        a1 = mcp_tools._get_manager()
        a2 = mcp_tools._get_manager()
    with bind_request(make_request_context("tenant-b")):
        b = mcp_tools._get_manager()
    assert a1 is a2, "same tenant must reuse one manager"
    assert a1 is not b, "different tenants must never share a manager"
    assert a1.redis_buffer is not b.redis_buffer
    assert a1.redis_buffer.tenant_id == "tenant-a"
    assert b.redis_buffer.tenant_id == "tenant-b"


def test_get_manager_keys_on_tenant_not_user(proxy, hosted_managers):
    """Two users of one tenant share a manager — the buffer prefix and the
    graph project are tenant-scoped, so a per-user manager would only multiply
    identical objects."""
    with bind_request(make_request_context("tenant-a", user_id="u1")):
        first = mcp_tools._get_manager()
    with bind_request(make_request_context("tenant-a", user_id="u2")):
        second = mcp_tools._get_manager()
    assert first is second


def test_get_manager_without_a_request_is_the_untouched_singleton(monkeypatch):
    """The stdio path must not go near the tenant cache."""
    sentinel = object()
    monkeypatch.setattr(mcp_tools, "_build_manager", lambda: sentinel)
    assert mcp_tools._get_manager() is sentinel
    assert mcp_tools._get_manager() is sentinel
    assert len(mcp_tools._tenant_managers) == 0


def test_hosted_manager_ignores_a_previously_installed_singleton(proxy, hosted_managers):
    """A process that served stdio first must not serve its manager to a
    hosted request — that manager holds the machine's credentials."""
    mcp_tools._manager = object()
    with bind_request(make_request_context("tenant-a")):
        assert mcp_tools._get_manager() is not mcp_tools._manager


def test_tenant_cache_evicts_the_least_recently_used_beyond_max():
    def _never(label):
        return lambda: pytest.fail(f"{label} should still have been cached")

    cache = mcp_tools._TenantManagerCache(max_entries=2, idle_ttl=0)
    a, b, c = object(), object(), object()
    cache.get("a", lambda: a)
    cache.get("b", lambda: b)
    assert cache.get("a", _never("a")) is a          # touching a makes b oldest
    cache.get("c", lambda: c)
    assert len(cache) == 2
    # a and c survived; b was the least recently used and was dropped.
    assert cache.get("a", _never("a")) is a
    assert cache.get("c", _never("c")) is c
    assert cache.get("b", lambda: "rebuilt") == "rebuilt"


def test_tenant_cache_evicts_after_the_idle_ttl():
    clock = [1000.0]
    cache = mcp_tools._TenantManagerCache(max_entries=8, idle_ttl=30.0)
    with patch.object(mcp_tools.time, "monotonic", lambda: clock[0]):
        first = cache.get("a", lambda: "first")
        clock[0] += 29.0
        assert cache.get("a", lambda: "rebuilt") == first, "still inside the TTL"
        clock[0] += 31.0
        assert cache.get("a", lambda: "rebuilt") == "rebuilt", "idle past the TTL"


def test_tenant_cache_hands_every_concurrent_caller_the_same_instance():
    """Losing this makes the whole design pointless: two managers for one
    tenant means two Redis buffers and two ontology caches diverging."""
    cache = mcp_tools._TenantManagerCache()
    builds = []
    barrier = threading.Barrier(16)

    def _factory():
        time.sleep(0.005)          # widen the window a naive impl would race in
        made = object()
        builds.append(made)
        return made

    def _worker():
        barrier.wait()
        return cache.get("tenant-a", _factory)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = [f.result() for f in [pool.submit(_worker) for _ in range(16)]]

    assert len({id(r) for r in results}) == 1, "callers diverged on the manager"


def test_hosted_manager_is_built_without_env_auth_disk_or_llm(proxy, hosted_managers):
    with bind_request(make_request_context("tenant-a")) as ctx:
        manager = mcp_tools._get_manager()

    # No local disk.
    assert manager.hosted is True
    assert manager.artifact_root is None
    assert manager.failure_ledger is None
    assert manager.retry_queue is None
    # No LLM (KUMIHO_HOSTED_LLM unset).
    with pytest.raises(RuntimeError, match="KUMIHO_HOSTED_LLM"):
        _ = manager.summarizer.adapter
    assert manager.auto_assess_fn is None
    assert manager.embedding_adapter is None
    assert manager.sibling_similarity_threshold == 0.0
    assert manager.graph_augmentation_config is None
    # Redis strictly through the tenant-authenticated proxy.
    assert manager.redis_buffer.redis_url is None
    assert manager.redis_buffer.proxy_url == PROXY_URL
    assert manager.redis_buffer.tenant_id == ctx.tenant_id


def test_hosted_llm_opt_in_builds_a_real_summarizer(monkeypatch, proxy, hosted_managers):
    monkeypatch.setenv("KUMIHO_HOSTED_LLM", "1")
    monkeypatch.setenv("KUMIHO_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("KUMIHO_LLM_PROVIDER", "openai")
    with bind_request(make_request_context("tenant-llm")):
        manager = mcp_tools._get_manager()
    assert manager.summarizer.api_key == "sk-test"
    assert manager.summarizer.provider == "openai"


# ---------------------------------------------------------------------------
# Session resolution order and source labels (§2.3 item 2)
# ---------------------------------------------------------------------------


def _resolve(args, manager, **kwargs):
    import asyncio
    return asyncio.run(mcp_tools._resolve_session(args, manager, **kwargs))


def test_explicit_argument_wins_over_the_request_session():
    manager = _local_manager()
    ctx = make_request_context("t1", session_id="from-request")
    with request_context(ctx):
        assert _resolve({"session_id": "from-argument"}, manager) == (
            "from-argument", "argument",
        )


def test_request_session_is_used_and_labelled_request():
    manager = _local_manager()
    with request_context(make_request_context("t1", session_id="req-sess-1")):
        assert _resolve({}, manager) == ("req-sess-1", "request")


def test_request_session_beats_the_process_env(monkeypatch):
    """The env is process-wide; on a shared server it names nobody's
    conversation. Losing this merges every tenant into one bucket."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-env-id")
    manager = _local_manager()
    with request_context(make_request_context("t1", session_id="req-sess-1")):
        assert _resolve({}, manager) == ("req-sess-1", "request")


def test_hosted_never_falls_back_to_the_process_env_session(monkeypatch):
    """With no ctx.session_id the resolver must go to the tenant's POINTER,
    not to the environment's id."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-env-id")
    manager = _local_manager()
    with request_context(make_request_context("t1", user_id="alice")):
        session_id, source = _resolve({}, manager)
    assert source == "generated"
    assert session_id != "host-env-id"
    assert session_id.startswith("claude:user-")


def test_pointer_hit_is_labelled_active_session():
    manager = _local_manager()
    ctx = make_request_context("t1", user_id="alice")
    with request_context(ctx):
        first, first_source = _resolve({}, manager)
        second, second_source = _resolve({}, manager)
    assert first_source == "generated"
    assert (second, second_source) == (first, "active_session")


def test_generated_session_registers_the_pointer_for_ctx_context_and_user():
    """The pointer key is (ctx.context, ctx.user_id) — that is what makes an
    identity-less hosted tool converge with an identity-bearing one."""
    import asyncio

    manager = _local_manager()
    ctx = make_request_context("t1", user_id="alice", context="claude")
    with request_context(ctx):
        session_id, source = _resolve({}, manager)
        pointer = asyncio.run(manager.redis_buffer.get_active_session(
            context="claude", user_canonical_id="alice",
        ))
    assert source == "generated"
    assert pointer == session_id


def test_hosted_defaults_user_and_context_from_the_request():
    """Two different tools, neither naming a user, must land on ONE bucket."""
    manager = _local_manager()
    ctx = make_request_context("t1", user_id="alice", context="claude")
    with request_context(ctx):
        engage_like, _ = _resolve({}, manager)
        reflect_like, source = _resolve({}, manager)
    assert engage_like == reflect_like
    assert source == "active_session"
    assert engage_like.startswith("claude:user-")


def test_blank_session_argument_is_still_rejected_loudly_when_hosted():
    manager = _local_manager()
    with request_context(make_request_context("t1")):
        with pytest.raises(ValueError, match="non-empty"):
            _resolve({"session_id": "   "}, manager)


def test_non_hosted_labels_are_unchanged(monkeypatch):
    """The stdio contract: 'host-env' and 'active-pointer', not the hosted
    spellings. A client keying off these must not see them change."""
    monkeypatch.setenv("KUMIHO_SESSION_ID", "host-env-id")
    manager = _local_manager()
    assert _resolve({}, manager) == ("host-env-id", "host-env")

    monkeypatch.delenv("KUMIHO_SESSION_ID")
    first, first_source = _resolve({"user_id": "alice"}, manager)
    second, second_source = _resolve({"user_id": "alice"}, manager)
    assert first_source == "generated"
    assert (second, second_source) == (first, "active-pointer")


def test_non_hosted_still_refuses_when_there_is_no_identity_at_all():
    manager = _local_manager()
    with pytest.raises(ValueError, match="no session identity available"):
        _resolve({}, manager)


def test_an_explicit_other_user_keeps_the_user_scoping_guard():
    """Bulk ingest on someone else's behalf must NOT be handed the caller's
    request session — that would file their turns into the caller's bucket."""
    manager = _local_manager()
    ctx = make_request_context("t1", user_id="caller", session_id="caller-session")
    with request_context(ctx):
        session_id, source = _resolve({}, manager, user_id="someone-else",
                                      context="personal")
    assert session_id != "caller-session"
    assert source == "generated"
    assert session_id.startswith("personal:user-")


def test_ingest_takes_its_user_from_the_request(proxy, hosted_managers):
    """user_id stops being required when the request is authenticated."""
    with bind_request(make_request_context("tenant-a", user_id="alice")):
        result = mcp_tools.tool_memory_ingest({"message": "hello"})
    assert result["session_id"].startswith("claude:user-")
    assert result["session_id_source"] == "generated"


def test_ingest_without_a_user_or_a_request_fails_before_writing():
    with pytest.raises(ValueError, match="user_id is required"):
        mcp_tools.tool_memory_ingest({"message": "hello"})


# ---------------------------------------------------------------------------
# Dedup guard keying (§2.3 item 3)
# ---------------------------------------------------------------------------


def test_dedup_scope_is_tenant_user_and_session():
    assert mcp_tools._recall_scope({}) == ""
    with request_context(make_request_context("t1", user_id="u1", session_id="s1")):
        assert mcp_tools._recall_scope({}) == "t1\x1eu1\x1es1"
        # An explicit session argument names the conversation being deduped.
        assert mcp_tools._recall_scope({"session_id": "s2"}) == "t1\x1eu1\x1es2"


def test_identical_queries_from_two_tenants_both_execute(proxy, hosted_managers):
    """The guard suppresses a MODEL's duplicate call, not two customers asking
    the same question at the same moment."""
    with bind_request(make_request_context("tenant-a")):
        a = mcp_tools.tool_memory_engage({"query": "what did we decide"})
    with bind_request(make_request_context("tenant-b")):
        b = mcp_tools.tool_memory_engage({"query": "what did we decide"})
    assert a.get("deduplicated") is not True
    assert b.get("deduplicated") is not True, "tenant B was starved by tenant A"


def test_a_true_duplicate_within_one_tenant_is_still_suppressed(proxy, hosted_managers):
    ctx = make_request_context("tenant-a", session_id="s1")
    with request_context(ctx):
        first = mcp_tools.tool_memory_engage({"query": "same"})
        second = mcp_tools.tool_memory_engage({"query": "same"})
    assert first.get("deduplicated") is not True
    assert second["deduplicated"] is True


def test_two_sessions_of_one_user_do_not_dedup_each_other(proxy, hosted_managers):
    with bind_request(make_request_context("tenant-a", session_id="s1")):
        first = mcp_tools.tool_memory_engage({"query": "same"})
    with bind_request(make_request_context("tenant-a", session_id="s2")):
        second = mcp_tools.tool_memory_engage({"query": "same"})
    assert first.get("deduplicated") is not True
    assert second.get("deduplicated") is not True


def test_the_recall_guard_lock_is_shared_only_on_the_stdio_path():
    """A process-wide lock held across a network recall would make every
    tenant queue behind every other one."""
    assert mcp_tools._recall_guard_lock("") is mcp_tools._recall_lock
    a = mcp_tools._recall_guard_lock("tenant-a\x1eu\x1es")
    b = mcp_tools._recall_guard_lock("tenant-b\x1eu\x1es")
    assert a is not b
    assert a is not mcp_tools._recall_lock
    assert mcp_tools._recall_guard_lock("tenant-a\x1eu\x1es") is a


def test_the_scope_lock_table_is_capped_but_never_drops_a_held_lock():
    """A scope is per SESSION, so on a long-lived server this table grows with
    every conversation the process ever serves."""
    cap = mcp_tools._RECALL_SCOPE_LOCK_CAP
    held = mcp_tools._recall_guard_lock("tenant-a\x1eu\x1ekeep-me")
    held.acquire()
    try:
        for i in range(cap + 1):
            mcp_tools._recall_guard_lock(f"tenant-a\x1eu\x1es{i}")
        assert len(mcp_tools._recall_scope_locks) <= cap
        # The in-flight scope kept its lock, so a concurrent caller in that
        # same conversation still serializes behind it.
        assert mcp_tools._recall_guard_lock("tenant-a\x1eu\x1ekeep-me") is held
    finally:
        held.release()


# ---------------------------------------------------------------------------
# No filesystem writes (§2.3 item 5)
# ---------------------------------------------------------------------------


def test_write_artifact_is_a_no_op_and_creates_nothing(tmp_path):
    target = tmp_path / "artifacts"
    with request_context(make_request_context("t1")):
        manager = _local_manager()
        assert manager._write_artifact(
            session_id="s1", content="secret transcript", space_hint="a/b",
        ) == ""
    assert not target.exists()


def test_read_artifact_content_is_disabled_when_hosted(tmp_path):
    """A stored artifact_location is caller-writable data; hosted must not
    turn it into a read of the operator's disk."""
    import asyncio

    planted = tmp_path / "secret.md"
    planted.write_text("operator secret", encoding="utf-8")
    assert asyncio.run(
        UniversalMemoryManager._read_artifact_content(str(planted))
    ) == "operator secret"
    with request_context(make_request_context("t1")):
        assert asyncio.run(
            UniversalMemoryManager._read_artifact_content(str(planted))
        ) == ""


def test_store_attachment_refuses_rather_than_returning_a_server_path(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("hi", encoding="utf-8")
    with request_context(make_request_context("t1")):
        manager = _local_manager()
        with pytest.raises(RuntimeError, match="hosted mode"):
            manager._store_attachment({"path": str(source)})


def test_retry_queue_writes_nothing_when_hosted(tmp_path):
    from kumiho_memory.retry import RetryQueue

    queue_dir = tmp_path / "retry"
    with request_context(make_request_context("t1")):
        queue = RetryQueue(str(queue_dir))
        queue.enqueue({"project": "P", "summary": "tenant content"})
        assert queue.count == 0
        assert queue.drain() == []
    assert not queue_dir.exists(), "hosted mode created a local queue directory"


def test_failure_ledger_is_absent_and_inert_when_hosted(tmp_path):
    from kumiho_memory.failure_ledger import FailureLedger, default_failure_ledger

    ledger_dir = tmp_path / "ledger"
    with request_context(make_request_context("t1")):
        assert default_failure_ledger() is None
        # Even one passed in explicitly must not write.
        ledger = FailureLedger(str(ledger_dir))
        ledger.record_failure("kref://x", "boom")
    assert not ledger_dir.exists()


def test_dream_state_cursor_file_is_not_read_or_written_when_hosted(tmp_path):
    from kumiho_memory.dream_state import DreamState

    root = tmp_path / "artifacts"
    with request_context(make_request_context("t1")):
        ds = DreamState(project="P", artifact_root=str(root))
        ds._save_cursor_local("2026-09-02T00:00:00Z")
        assert ds._load_cursor_local() is None
    assert not root.exists()


def test_a_full_hosted_tool_round_trip_touches_no_disk(tmp_path, proxy, hosted_managers):
    """The sweep: engage + reflect + chat through the real handlers, with every
    local-state root pointed at an empty directory that must stay empty."""
    root = tmp_path / "state"
    root.mkdir()
    os.environ["KUMIHO_MEMORY_ARTIFACT_ROOT"] = str(root / "artifacts")
    os.environ["KUMIHO_RETRY_QUEUE_DIR"] = str(root / "retry")
    os.environ["KUMIHO_FAILURE_LEDGER_DIR"] = str(root / "ledger")

    with bind_request(make_request_context("tenant-a", user_id="alice")):
        mcp_tools.tool_memory_engage({"query": "anything"})
        with patch("kumiho.mcp_server.tool_memory_store",
                   lambda **kw: {"revision_kref": "kref://t/r/1"}):
            reflected = mcp_tools.tool_memory_reflect({
                "response": "done",
                "captures": [{"type": "fact", "title": "t", "content": "c"}],
                "discover_edges": False,
            })
        chat = mcp_tools.tool_chat_get({})

    # The work really happened — otherwise "nothing was written" is trivial.
    assert reflected["captures_stored"] == 1
    assert [m["content"] for m in chat["messages"]] == ["done"]
    assert list(root.iterdir()) == [], f"hosted run wrote to disk: {list(root.iterdir())}"


def test_the_stdio_path_still_writes_its_artifact(tmp_path):
    """The mirror image — without a request context nothing changed."""
    manager = _local_manager(artifact_root=str(tmp_path / "artifacts"))
    path = manager._write_artifact(session_id="s1", content="hello", space_hint="")
    assert path and os.path.isfile(path)


# ---------------------------------------------------------------------------
# Redis credentials and proxy routing (§2.3 item 4)
# ---------------------------------------------------------------------------


def test_token_comes_from_the_request_when_no_override_is_set():
    with request_context(make_request_context("t1", token="tenant-jwt")):
        assert RedisMemoryBuffer._get_fresh_token() == "tenant-jwt"


def test_an_explicit_override_still_outranks_the_request():
    """kumiho-FastAPI sets the override; it must keep winning."""
    token = redis_memory._token_override_var.set("override-jwt")
    try:
        with request_context(make_request_context("t1", token="tenant-jwt")):
            assert RedisMemoryBuffer._get_fresh_token() == "override-jwt"
    finally:
        redis_memory._token_override_var.reset(token)


def test_hosted_without_a_request_token_refuses_local_credentials(monkeypatch):
    """KUMIHO_MCP_HOSTED with no request: the machine's ~/.kumiho identity is
    the operator's, so using it would run a tenant's operation as someone
    else. It must raise, not fall back."""
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    monkeypatch.setattr(redis_memory, "load_firebase_token",
                        lambda *a, **k: pytest.fail("read a local credential file"))
    monkeypatch.setattr(redis_memory, "load_bearer_token",
                        lambda *a, **k: pytest.fail("read a local credential file"))
    with pytest.raises(RedisDiscoveryError, match="hosted mode"):
        RedisMemoryBuffer._get_fresh_token()


def test_hosted_ignores_ambient_upstash_credentials(monkeypatch, proxy):
    """A raw Redis URL in the operator's env is ONE database for everyone; the
    proxy is what namespaces keys per tenant."""
    monkeypatch.setenv("UPSTASH_REDIS_URL", "redis://operator.invalid:6379")
    monkeypatch.setenv("KUMIHO_UPSTASH_REDIS_URL", "redis://operator2.invalid:6379")
    with request_context(make_request_context("t1")):
        buffer = RedisMemoryBuffer(tenant_id="t1", prefer_discovery=False)
    assert buffer.redis_url is None
    assert buffer.client is None
    assert buffer.proxy_url == PROXY_URL


def test_non_hosted_still_uses_ambient_upstash_credentials(monkeypatch):
    monkeypatch.setenv("UPSTASH_REDIS_URL", "redis://local.invalid:6379")
    buffer = RedisMemoryBuffer(prefer_discovery=False)
    assert buffer.redis_url == "redis://local.invalid:6379"


# --- dev-only direct-Redis escape hatch (WP-C's KUMIHO_MCP_DEV_MODE=ce) ----
# Hosted mode has exactly one route to Redis, because that route is what
# authenticates and namespaces. Dev mode has no control plane, so it needs a
# second one — and these tests are mostly about how tightly it is shut.


def test_the_escape_hatch_is_off_by_default(monkeypatch, proxy):
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    with request_context(make_request_context("t1")):
        buffer = RedisMemoryBuffer(tenant_id="t1", prefer_discovery=False)
    assert buffer.redis_url is None
    assert buffer.proxy_url == PROXY_URL


def test_the_escape_hatch_gives_hosted_mode_a_direct_redis(monkeypatch, proxy, caplog):
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    monkeypatch.setenv("KUMIHO_LOCAL_REDIS_URL", "redis://127.0.0.1:6399")
    with caplog.at_level("WARNING"), request_context(make_request_context("t1")):
        buffer = RedisMemoryBuffer(tenant_id="t1", prefer_discovery=False)
    assert buffer.redis_url == "redis://127.0.0.1:6399"
    assert buffer.proxy_url is None, "the proxy must not also be configured"
    # Exactly one warning, at build time: an operator has to be able to find a
    # hosted process talking straight to Redis in the logs, and it must not
    # repeat per operation.
    hatch = [r for r in caplog.records
             if r.levelname == "WARNING"
             and "KUMIHO_HOSTED_LOCAL_REDIS is active" in r.getMessage()]
    assert len(hatch) == 1


def test_the_escape_hatch_url_falls_back_in_order(monkeypatch):
    from kumiho_memory.redis_memory import (
        HOSTED_LOCAL_REDIS_DEFAULT,
        hosted_local_redis_url,
    )

    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    assert hosted_local_redis_url() == HOSTED_LOCAL_REDIS_DEFAULT
    monkeypatch.setenv("UPSTASH_REDIS_URL", "redis://from-upstash:6379")
    assert hosted_local_redis_url() == "redis://from-upstash:6379"
    monkeypatch.setenv("KUMIHO_LOCAL_REDIS_URL", "redis://from-local:6379")
    assert hosted_local_redis_url() == "redis://from-local:6379"


def test_the_escape_hatch_is_ignored_without_the_hosted_env(monkeypatch, caplog):
    """The stdio plugin must be unreachable by this flag."""
    from kumiho_memory.redis_memory import hosted_local_redis_url

    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    monkeypatch.setenv("KUMIHO_LOCAL_REDIS_URL", "redis://127.0.0.1:6399")
    with caplog.at_level("WARNING"):
        assert hosted_local_redis_url() is None
    assert any("ignoring it" in r.getMessage() for r in caplog.records)


def test_a_request_context_alone_does_not_arm_the_escape_hatch(monkeypatch, proxy):
    """is_hosted() is true here (a request is set) but hosted_mode() is not, so
    the buffer still goes through the proxy. Gating on the coarse process-wide
    flag is what keeps a stray env var from redirecting a plugin user's
    working memory to localhost."""
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    monkeypatch.setenv("KUMIHO_LOCAL_REDIS_URL", "redis://127.0.0.1:6399")
    with request_context(make_request_context("t1")):
        buffer = RedisMemoryBuffer(tenant_id="t1", prefer_discovery=False)
    assert buffer.redis_url is None
    assert buffer.proxy_url == PROXY_URL


def test_the_escape_hatch_does_not_change_the_stdio_path(monkeypatch):
    """Without KUMIHO_MCP_HOSTED the plugin resolves exactly as it always has:
    the ambient UPSTASH_REDIS_URL, not the hatch's default."""
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    monkeypatch.setenv("UPSTASH_REDIS_URL", "redis://plugin-redis.invalid:6379")
    buffer = RedisMemoryBuffer(prefer_discovery=False)
    assert buffer.redis_url == "redis://plugin-redis.invalid:6379"


def test_a_configured_proxy_still_beats_the_escape_hatch(monkeypatch, proxy):
    """A deployment that HAS a control plane keeps using it even if the flag
    was left set by accident."""
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    monkeypatch.setenv("KUMIHO_MEMORY_PROXY_URL", PROXY_URL)
    with request_context(make_request_context("t1")):
        buffer = RedisMemoryBuffer(tenant_id="t1", prefer_discovery=False)
    assert buffer.redis_url is None
    assert buffer.proxy_url == PROXY_URL


def test_the_escape_hatch_keeps_keys_namespaced_per_tenant_and_user(monkeypatch):
    """The whole safety argument for the hatch: one dev Redis, still no
    collision between tenants or between users."""
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")

    def _buffer(tenant):
        with request_context(make_request_context(tenant)):
            return RedisMemoryBuffer(client=SessionFakeRedis(), tenant_id=tenant)

    a, b = _buffer("tenant-a"), _buffer("tenant-b")
    assert a._session_messages_key("P", "s1") != b._session_messages_key("P", "s1")
    assert "tenant-a" in a._session_messages_key("P", "s1")
    assert "tenant-b" in b._session_messages_key("P", "s1")
    # Pointers carry the tenant AND (context, user), exactly as behind the proxy.
    assert a._active_session_key("claude", "alice") != a._active_session_key("claude", "bob")
    assert a._active_session_key("claude", "alice") != b._active_session_key("claude", "alice")
    for key in (a._session_metadata_key("P", "s1"),
                a._sequence_key("alice", "20260902"),
                a._session_generation_key("s1"),
                a._active_session_key("claude", "alice")):
        assert "tenant-a" in key


def test_the_escape_hatch_reaches_a_hosted_manager_end_to_end(
    monkeypatch, proxy, hosted_managers,
):
    """What WP-C's dev mode actually needs: a built manager whose buffer talks
    to Redis directly, with no control plane in the picture."""
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    monkeypatch.setenv("KUMIHO_LOCAL_REDIS_URL", "redis://127.0.0.1:6399")
    with bind_request(make_request_context("tenant-dev")):
        manager = mcp_tools._get_manager()
    assert manager.redis_buffer.redis_url == "redis://127.0.0.1:6399"
    assert manager.redis_buffer.proxy_url is None
    assert manager.redis_buffer.tenant_id == "tenant-dev"
    # Still a hosted manager in every other respect.
    assert manager.hosted is True
    assert manager.artifact_root is None
    assert manager.failure_ledger is None
    assert proxy.calls == [], "dev mode must not touch the control plane at all"


def test_hosted_proxy_url_follows_the_control_plane_env(monkeypatch, proxy):
    monkeypatch.setenv("KUMIHO_CONTROL_PLANE_URL", "https://staging.cp.invalid")
    with request_context(make_request_context("t1")):
        buffer = RedisMemoryBuffer(tenant_id="t1", prefer_discovery=False)
    assert buffer.proxy_url == "https://staging.cp.invalid/api/memory/redis"


def test_hosted_takes_its_tenant_from_the_request_not_the_machine_cache(monkeypatch, proxy):
    """~/.kumiho holds whichever tenant last logged in on this host."""
    monkeypatch.setattr(
        RedisMemoryBuffer, "_load_cached_tenant",
        staticmethod(lambda: pytest.fail("read the machine discovery cache")),
    )
    monkeypatch.setattr(
        RedisMemoryBuffer, "_discover_upstash_url",
        lambda self: pytest.fail("ran control-plane discovery"),
    )
    with request_context(make_request_context("tenant-from-request")):
        buffer = RedisMemoryBuffer()
    assert buffer.tenant_id == "tenant-from-request"


def test_every_proxy_call_carries_the_requesting_tenants_bearer(proxy, hosted_managers):
    import asyncio

    with bind_request(make_request_context("tenant-a", token="jwt-a")):
        manager = mcp_tools._get_manager()
        asyncio.run(manager.redis_buffer.add_message(
            project="P", session_id="s-a", role="user", content="a",
        ))
    with bind_request(make_request_context("tenant-b", token="jwt-b")):
        manager_b = mcp_tools._get_manager()
        asyncio.run(manager_b.redis_buffer.add_message(
            project="P", session_id="s-b", role="user", content="b",
        ))

    assert proxy.unauthenticated_calls == []
    by_session = {c["body"].get("session_id"): c["bearer"] for c in proxy.calls}
    assert by_session == {"s-a": "jwt-a", "s-b": "jwt-b"}


def test_the_token_is_resolved_per_call_not_frozen_into_the_buffer(proxy, hosted_managers):
    """A tenant's manager outlives the request that built it, and the bearer
    is a short-lived JWT. Baking it in at construction would send an expired
    (or another user's) token on every later call."""
    import asyncio

    proxy.register("jwt-first", "tenant-a")
    proxy.register("jwt-second", "tenant-a")
    with bind_request(make_request_context("tenant-a", token="jwt-first")):
        manager = mcp_tools._get_manager()
        asyncio.run(manager.redis_buffer.add_message(
            project="P", session_id="s1", role="user", content="one",
        ))
    with bind_request(make_request_context("tenant-a", token="jwt-second")):
        same = mcp_tools._get_manager()
        asyncio.run(same.redis_buffer.add_message(
            project="P", session_id="s1", role="user", content="two",
        ))
    assert same is manager
    assert [c["bearer"] for c in proxy.calls] == ["jwt-first", "jwt-second"]


# ---------------------------------------------------------------------------
# Two tenants, concurrently, through the real handlers (§2.3 item 7)
# ---------------------------------------------------------------------------


TENANTS = (
    {"tenant": "tenant-alpha", "user": "alice", "token": "jwt-alpha"},
    {"tenant": "tenant-beta", "user": "bob", "token": "jwt-beta"},
)


def test_two_tenants_run_engage_reflect_and_chat_concurrently_without_cross_talk(
    proxy, graph, hosted_managers,
):
    """The whole point of the work package, in one test.

    Two tenants drive the real tool handlers at the same time, on the same
    process, through the same module-level caches. Every shared thing is then
    checked for contamination: the manager, the resolved session, the bearer
    on every single proxy call, the working-memory buffer, and the graph.
    """
    for spec in TENANTS:
        proxy.register(spec["token"], spec["tenant"])

    barrier = threading.Barrier(len(TENANTS))
    results = {}
    errors = []

    def _run(spec):
        ctx = make_request_context(
            spec["tenant"], user_id=spec["user"], token=spec["token"],
        )
        try:
            with bind_request(ctx):
                barrier.wait(timeout=30)
                manager = mcp_tools._get_manager()

                engaged = mcp_tools.tool_memory_engage({
                    "query": f"what does {spec['user']} prefer",
                })
                reflected = mcp_tools.tool_memory_reflect({
                    "response": f"noted for {spec['user']}",
                    "captures": [{
                        "type": "preference",
                        "title": f"{spec['user']} pref",
                        "content": f"secret-of-{spec['tenant']}",
                    }],
                    "discover_edges": False,
                })
                chat = mcp_tools.tool_chat_get({})

                results[spec["tenant"]] = {
                    "manager": manager,
                    "engaged": engaged,
                    "reflected": reflected,
                    "chat": chat,
                }
        except BaseException as exc:  # noqa: BLE001 — surfaced by the assert below
            errors.append((spec["tenant"], exc))

    # Patched once around BOTH threads: unittest.mock.patch swaps a module
    # attribute process-wide, so entering it inside each thread would let one
    # thread's __exit__ restore the real function under the other.
    with patch("kumiho.mcp_server.tool_memory_store", _tenant_store_recorder(graph)):
        threads = [threading.Thread(target=_run, args=(spec,)) for spec in TENANTS]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

    assert errors == [], f"tenant thread failed: {errors}"
    assert set(results) == {"tenant-alpha", "tenant-beta"}

    alpha, beta = results["tenant-alpha"], results["tenant-beta"]

    # 1. Distinct manager instances, and distinct buffers under them.
    assert alpha["manager"] is not beta["manager"]
    assert alpha["manager"].redis_buffer is not beta["manager"].redis_buffer
    assert alpha["manager"].redis_buffer.tenant_id == "tenant-alpha"
    assert beta["manager"].redis_buffer.tenant_id == "tenant-beta"

    # 2. Distinct session ids, each reported with its source, and STABLE across
    #    the three calls of one tenant (reflect and chat must join engage's).
    alpha_session = alpha["reflected"]["session_id"]
    beta_session = beta["reflected"]["session_id"]
    assert alpha_session != beta_session
    for tenant in (alpha, beta):
        assert tenant["reflected"]["session_id"] == tenant["chat"]["session_id"]
        assert tenant["reflected"]["session_id_source"] in (
            "generated", "active_session",
        )
        assert tenant["chat"]["session_id_source"] in ("generated", "active_session")

    # 3. Every proxy call carried the right bearer for the session it touched.
    sessions = {alpha_session: "jwt-alpha", beta_session: "jwt-beta"}
    users = {"alice": "jwt-alpha", "bob": "jwt-beta"}
    assert proxy.unauthenticated_calls == []
    for call in proxy.calls:
        body = call["body"]
        session_id = body.get("session_id")
        if session_id in sessions:
            assert call["bearer"] == sessions[session_id], (
                f"{call['action']} for {session_id} used {call['bearer']}"
            )
        user = body.get("user_canonical_id")
        if user in users:
            assert call["bearer"] == users[user], (
                f"{call['action']} for {user} used {call['bearer']}"
            )
    assert proxy.tokens_seen() == ["jwt-alpha", "jwt-beta"]

    # 4. No cross-talk in working memory: each tenant's buffer holds only its
    #    own response, under its own token.
    alpha_msgs = proxy.messages("jwt-alpha", alpha_session)
    beta_msgs = proxy.messages("jwt-beta", beta_session)
    assert [m["content"] for m in alpha_msgs] == ["noted for alice"]
    assert [m["content"] for m in beta_msgs] == ["noted for bob"]
    assert proxy.messages("jwt-alpha", beta_session) == []
    assert proxy.messages("jwt-beta", alpha_session) == []

    # 5. No cross-talk in the graph: each capture landed under its own tenant.
    alpha_rows = graph.rows("tenant-alpha")
    beta_rows = graph.rows("tenant-beta")
    assert [r["summary"] for r in alpha_rows] == ["secret-of-tenant-alpha"]
    assert [r["summary"] for r in beta_rows] == ["secret-of-tenant-beta"]

    # 6. Recall answered from each tenant's own graph. Asserted non-empty
    #    first: an empty result set would satisfy the loop vacuously and hide
    #    exactly the mixing this checks for.
    for tenant, name in ((alpha, "tenant-alpha"), (beta, "tenant-beta")):
        krefs = tenant["engaged"]["source_krefs"]
        assert krefs, f"{name} recalled nothing — the check below proves nothing"
        for kref in krefs:
            assert kref.startswith(f"kref://{name}/"), kref
        assert tenant["reflected"]["captures_stored"] == 1

    # 7. Two live tenants in the cache, one entry each.
    assert len(mcp_tools._tenant_managers) == 2


def _tenant_store_recorder(graph):
    """A ``tool_memory_store`` stand-in that files writes under the AMBIENT
    tenant — so a capture written under the wrong context is recorded under
    the wrong tenant and step 5 above fails."""
    def _store(**kwargs):
        ctx = current_request()
        tenant = ctx.tenant_id if ctx is not None else "<no-request>"
        rows = graph.stored.setdefault(tenant, [])
        rows.append(kwargs)
        return {"revision_kref": f"kref://{tenant}/mem/rev/{len(rows)}"}
    return _store


# ---------------------------------------------------------------------------
# Context propagation across thread / executor boundaries
# ---------------------------------------------------------------------------
# A raw thread and ``loop.run_in_executor`` both start from an EMPTY
# contextvars context (``asyncio.to_thread`` is the exception — it copies).
# Every one of these paths is best-effort and swallows exceptions, so a
# dropped request context does not fail loudly: it just quietly runs the work
# with no tenant, or — on a shared executor thread — with whichever tenant
# happened to touch that worker last. One test per boundary.


def _seen_identity():
    """(tenant_id, redis token override) as observed by the CURRENT context."""
    ctx = current_request()
    return (
        ctx.tenant_id if ctx is not None else None,
        redis_memory._token_override_var.get(),
    )


def test_run_bounded_in_thread_carries_the_request_into_the_worker():
    import asyncio

    from kumiho_memory._bounded import run_bounded_in_thread

    seen = []
    override = redis_memory._token_override_var.set("override-jwt")
    try:
        with request_context(make_request_context("tenant-a")):
            asyncio.run(run_bounded_in_thread(
                lambda: seen.append(_seen_identity()), timeout=5,
            ))
    finally:
        redis_memory._token_override_var.reset(override)
    assert seen == [("tenant-a", "override-jwt")]


def test_run_coro_in_daemon_thread_carries_the_request_into_the_worker():
    from kumiho_memory._bounded import run_coro_in_daemon_thread

    seen = []

    async def _work():
        seen.append(_seen_identity())

    with request_context(make_request_context("tenant-a")):
        thread = run_coro_in_daemon_thread(_work, timeout=5)
    thread.join(timeout=10)
    assert seen == [("tenant-a", None)]


def test_start_context_thread_carries_the_context_a_bare_thread_drops():
    """The helper, and the contrast that motivates it, in one place."""
    from kumiho_memory._bounded import start_context_thread

    seen, bare = [], []
    with request_context(make_request_context("tenant-a")):
        start_context_thread(lambda: seen.append(_seen_identity())).join(timeout=10)
        raw = threading.Thread(target=lambda: bare.append(_seen_identity()))
        raw.start()
        raw.join(timeout=10)
    assert seen == [("tenant-a", None)]
    assert bare == [(None, None)], "a bare thread should still drop the context"


def test_no_module_starts_a_bare_thread_or_a_bare_executor_task():
    """A structural invariant, because the behavioural tests cannot reach
    every site.

    ``graph_augmentation``'s two daemon threads sit behind SDK-backed edge
    creation and graph traversal — driving them in a unit test would take more
    scaffolding than the thing under test. But the rule is mechanical and the
    failure is silent, so it is checked mechanically: in this package a thread
    is started through :func:`start_context_thread`, and work handed to an
    executor is wrapped in a copied context. Anything else drops the tenant.
    """
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "kumiho_memory"
    offenders = []
    for path in sorted(package.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name in ("threading.Thread", "Thread"):
                # _bounded defines the sanctioned wrapper; that one call IS
                # the helper.
                if path.name == "_bounded.py":
                    continue
                offenders.append(f"{path.name}:{node.lineno} bare {name}(...)")
            elif name.endswith("run_in_executor") or name.endswith(".submit"):
                args = [ast.unparse(a) for a in node.args]
                if not any("ctx.run" in a or "copy_context" in a for a in args):
                    offenders.append(
                        f"{path.name}:{node.lineno} {name}(...) without a copied context"
                    )
    assert offenders == [], (
        "start these through kumiho_memory._bounded.start_context_thread, or "
        "pass contextvars.copy_context().run: " + "; ".join(offenders)
    )


def test_code_query_cross_encoder_scoring_runs_under_the_request():
    """A SHARED executor makes this worse than a one-shot thread: without the
    copy, worker state persists between tenants."""
    import asyncio

    from kumiho_memory import code_query

    seen = []

    def _reranker(question, texts):
        seen.append(_seen_identity())
        return [1.0 for _ in texts]

    with request_context(make_request_context("tenant-a")):
        scores = asyncio.run(code_query._ce_scores("why", ["a", "b"], _reranker))
    assert scores == [1.0, 1.0]
    assert seen == [("tenant-a", None)]


def test_recall_rerank_offload_runs_under_the_request():
    import asyncio

    from kumiho_memory.recall_rerank import RerankConfig, rerank_async

    seen = []

    def _reranker(query, texts):
        seen.append(_seen_identity())
        return [1.0 for _ in texts]

    _reranker._kumiho_offload_safe = True  # type: ignore[attr-defined]
    config = RerankConfig(cross_encoder_enabled=True)
    memories = [{"kref": "k1", "title": "a", "summary": "a"},
                {"kref": "k2", "title": "b", "summary": "b"}]

    with request_context(make_request_context("tenant-a")):
        asyncio.run(rerank_async(
            "q", memories, config=config, reranker=_reranker,
        ))
    assert seen == [("tenant-a", None)]


def test_a_second_request_for_a_tenant_reuses_its_manager_and_session(
    proxy, hosted_managers,
):
    """Continuity: the connector's next tool call must land in the SAME
    conversation bucket, via the pointer rather than a new id."""
    first_ctx = make_request_context("tenant-a", user_id="alice", token="jwt-1")
    second_ctx = make_request_context("tenant-a", user_id="alice", token="jwt-2")
    # Both tokens belong to the same tenant, as a refreshed access token would.
    proxy.register("jwt-1", "tenant-a")
    proxy.register("jwt-2", "tenant-a")

    with request_context(first_ctx):
        first_manager = mcp_tools._get_manager()
        first = mcp_tools.tool_chat_add({"message": "one"})
    with request_context(second_ctx):
        second_manager = mcp_tools._get_manager()
        second = mcp_tools.tool_chat_add({"message": "two"})

    assert first_manager is second_manager
    assert first["session_id"] == second["session_id"]
    assert first["session_id_source"] == "generated"
    assert second["session_id_source"] == "active_session"


# ---------------------------------------------------------------------------
# Shared-manager state that is NOT per-request (found by WP-E1 integration)
# ---------------------------------------------------------------------------


def test_backend_error_does_not_cross_between_concurrent_callers():
    """One tenant's manager is shared by all its users AND all its concurrent
    requests, so a recall's failure signal cannot live on the instance.

    ``_last_backend_error`` was a plain attribute — correct while a manager
    served one user one call at a time, which is exactly what stopped being
    true when `_get_manager` started returning a per-TENANT manager. The
    failure mode is not theoretical: the message carries the failing query's
    text and space paths, and the reader is whichever caller happens to look
    next.
    """
    manager = _local_manager()
    started = threading.Barrier(2)
    seen: dict = {}

    def _caller(label: str, error: str) -> None:
        # Exactly the sequence a tool handler runs: reset, fail, read back.
        manager._last_backend_error = None
        started.wait(timeout=5)
        manager._last_backend_error = error
        time.sleep(0.02)  # let the other thread run its own write
        seen[label] = manager._last_backend_error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_caller, "alice", "retrieve failed for alice's query"),
            pool.submit(_caller, "bob", "retrieve failed for bob's query"),
        ]
        for future in futures:
            future.result()

    assert seen["alice"] == "retrieve failed for alice's query"
    assert seen["bob"] == "retrieve failed for bob's query"


def test_backend_error_still_reads_back_on_the_calling_thread():
    """The stdio contract: set it, read it, clear it — same thread, unchanged."""
    manager = _local_manager()
    assert manager._last_backend_error is None
    manager._last_backend_error = "backend down"
    assert manager._last_backend_error == "backend down"
    manager._last_backend_error = None
    assert manager._last_backend_error is None
    # And the slot is released rather than accumulating a None per thread.
    assert manager._backend_errors == {}


def test_a_blank_tenant_id_is_refused_rather_than_sharing_one_manager():
    """A blank tenant is a cache key like any other — and the worst one.

    Every request carrying it would share a manager, a Redis prefix and an
    active-session pointer: the complete version of the collapse the
    per-tenant cache exists to prevent. The hosting layer rejects a token
    with no tenant claim; this is what keeps that a requirement.
    """
    for blank in ("", "   "):
        with request_context(make_request_context(blank)):
            with pytest.raises(ValueError, match="no tenant_id"):
                mcp_tools._get_manager()
    assert len(mcp_tools._tenant_managers) == 0


def test_the_direct_redis_warning_does_not_print_the_credential(monkeypatch, caplog):
    """The dev escape hatch logs which Redis it chose. Upstash URLs are
    ``rediss://default:<token>@host`` — that token is the credential, and this
    WARNING exists to be read, shipped and pasted into tickets."""
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    monkeypatch.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "1")
    monkeypatch.setenv(
        "KUMIHO_LOCAL_REDIS_URL", "rediss://default:SUPERSECRETTOKEN@fake.upstash.io:6379"
    )

    with caplog.at_level("WARNING"):
        with request_context(make_request_context("tenant-a")):
            RedisMemoryBuffer(client=SessionFakeRedis())

    warnings = [r.getMessage() for r in caplog.records
                if "KUMIHO_HOSTED_LOCAL_REDIS is active" in r.getMessage()]
    assert warnings, caplog.records
    assert "SUPERSECRETTOKEN" not in warnings[0]
    # Still identifies the host, which is the whole point of the warning.
    assert "fake.upstash.io:6379" in warnings[0]
    assert "default:***@" in warnings[0]


def test_redact_redis_url_leaves_a_credential_free_url_alone():
    assert redis_memory.redact_redis_url("redis://127.0.0.1:6379") == "redis://127.0.0.1:6379"
    assert redis_memory.redact_redis_url(None) == "<none>"
    assert redis_memory.redact_redis_url("") == "<none>"
    assert (
        redis_memory.redact_redis_url("rediss://:tok@h:1") == "rediss://***@h:1"
    )


def test_the_entity_anchor_lock_table_is_capped():
    """Keyed per ENTITY, not per tenant: uncapped it grows with every entity
    every tenant has ever promoted, for the life of the process. Capped for
    the same reason (and in the same shape) as _recall_scope_locks."""
    import kumiho_memory.entity_promotion as entity_promotion

    with request_context(make_request_context("tenant-a")):
        for i in range(entity_promotion._ANCHOR_LOCK_CAP + 50):
            entity_promotion._anchor_lock(f"entity-{i}")

    assert len(entity_promotion._anchor_locks) <= entity_promotion._ANCHOR_LOCK_CAP
    # A lock a caller is holding survives the sweep.
    with request_context(make_request_context("tenant-b")):
        held = entity_promotion._anchor_lock("in-flight")
        held.acquire()
        try:
            for i in range(entity_promotion._ANCHOR_LOCK_CAP + 50):
                entity_promotion._anchor_lock(f"more-{i}")
            assert entity_promotion._anchor_lock("in-flight") is held
        finally:
            held.release()
