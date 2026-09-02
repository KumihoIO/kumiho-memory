"""MCP tool definitions for kumiho-memory.

This module exposes kumiho-memory functionality as MCP tools.  It is
**auto-discovered** by the core Kumiho MCP server (``kumiho.mcp_server``)
when ``kumiho-memory`` is installed — no manual registration required.

The plugin hook in ``mcp_server.py`` does::

    try:
        from kumiho_memory.mcp_tools import MEMORY_TOOLS, MEMORY_TOOL_HANDLERS
        TOOLS.extend(MEMORY_TOOLS)
        TOOL_HANDLERS.update(MEMORY_TOOL_HANDLERS)
    except ImportError:
        pass

All tool handlers are **synchronous** functions.  The MCP server dispatches
them via ``asyncio.to_thread(handler, args)``, so they run in a thread pool
with no active event loop — ``asyncio.run()`` is safe to use inside them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from kumiho_memory._request_context import current_request, hosted_llm_enabled

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capture vocabulary
# ---------------------------------------------------------------------------
# The memory types a `reflect` capture may carry. One list, because it was
# three: this module's prose advertised ten, `_discover` hardcoded five inline
# as its edge gate, and assessors.py names four in its LLM prompts. Nothing
# validates the field (`cap.get("type", "summary")` accepts anything), so a
# divergence is silent — it just produces memories filed under a type no
# reader branches on.
#
# It has to be a constant rather than only prose because of what a client
# actually receives. Measured against a live host, forwarding this server:
#
#   - property-level `description` is forwarded only for properties that are
#     REQUIRED in the source. Every optional one arrives undocumented, at every
#     nesting depth.
#   - `enum` survives, and survives on OPTIONAL and NESTED properties too:
#     recall's `recall_mode` and store_execution's `status` both arrive with
#     their enum and no description; so does the nested
#     code_capture.decisions[].evidence[].kind.
#   - the tool-level `description` survives intact.
#   - `required` is NOT reliably forwarded. Deliberately vague, because the
#     evidence does not support a rule: reflect, engage, recall, create_item
#     and decompose (five nested arrays) all lose theirs entirely, while
#     create_edge/delete_edge declare
#     `required: [source_kref, target_kref, edge_type]` and arrive with
#     `required: ["edge_type"]`. Do not assume either behaviour.
#
# So `captures`, being optional, arrives with no descriptions anywhere in its
# sub-schema and a caller has to guess the type — which is exactly what
# happened. Anything a caller MUST know therefore belongs in the tool-level
# description or in a structural keyword like `enum`, never in the description
# of an optional property.
MEMORY_TYPES = (
    "decision",
    "preference",
    "fact",
    "correction",
    "architecture",
    "implementation",
    "synthesis",
    "reflection",
    "summary",
    "skill",
)

#: The subset whose captures are worth spending an edge-discovery LLM call on:
#: types that assert a relationship to prior work. A `fact` or `preference`
#: stands alone, so discovery would burn a call to find nothing.
EDGE_DISCOVERY_TYPES = (
    "decision",
    "architecture",
    "implementation",
    "synthesis",
    "reflection",
)


def earns_edge_discovery(capture_type: Optional[str]) -> bool:
    """Whether a capture of this type is worth an edge-discovery LLM call.

    A one-line predicate so the gate is reachable from a test. Inlined in
    `_discover` it was a condition inside a closure inside a handler, which
    meant widening it to every type — spending a call per `fact` to find
    nothing — could not be caught by anything.
    """
    return capture_type in EDGE_DISCOVERY_TYPES


# One block, shared by the IDENTITY-LESS session-scoped tool descriptions,
# because the TOOL description is the only prose channel that reliably reaches
# a caller (see the transport notes above MEMORY_TYPES). The caller is an LLM:
# asking it to originate and hold a stable opaque id across turns is exactly
# what it gets wrong, and a drifted id used to fragment the conversation
# buffer silently (issue #3). Omission is the recommended convention — with
# the ingest exception spelled out, because ingest resolves PER-USER (its env
# tier is deliberately skipped: the cross-user collapse blocker) and can never
# land on the same default as these tools. Without that sentence, a caller
# following the descriptions filed the user turns (ingest) and assistant
# turns (add_response) of ONE conversation into two buckets by construction
# (PR #4 review, round 3). Not appended to ingest, which carries its own
# accurate text.
_SESSION_DEFAULT_NOTE = (
    " session_id is OPTIONAL when the host provides identity: it defaults "
    "to the host session (KUMIHO_SESSION_ID), and "
    "the result reports the session_id used plus its session_id_source. "
    "With no host identity the call fails with instructions — pass "
    "session_id explicitly and REUSE the same value for the conversation. "
    "EXCEPTION: in the kumiho_memory_ingest workflow, pass the same "
    "user_id (and context) you gave ingest — the call then resolves into "
    "that user's session."
)

# Optional identity properties advertised on every identity-less
# session-scoped tool. They make convergence with ingest STRUCTURAL: the
# caller repeats the same user_id — a stable semantic value — and resolution
# lands on the identical (context, user) pointer, instead of prose asking the
# LLM to hold an opaque session uuid across turns (the failure mode issue #3
# exists to eliminate; PR #4 review, round 4).
_SESSION_IDENTITY_PROPS: Dict[str, Any] = {
    "user_id": {
        "type": "string",
        "description": (
            "Optional. In the kumiho_memory_ingest workflow, pass the same "
            "user_id you gave ingest; resolution then lands on that user's "
            "session instead of the conversation default."
        ),
    },
    "context": {
        "type": "string",
        "description": (
            "Optional, used with user_id: the ingest context "
            "(defaults to 'personal', matching ingest)."
        ),
    },
}

# ---------------------------------------------------------------------------
# Lazy singleton manager (stdio) / per-tenant manager cache (hosted)
# ---------------------------------------------------------------------------

_manager: Optional[Any] = None
_manager_lock = threading.Lock()

#: Hosted manager cache bounds (plan §2.3). 256 tenants is roughly a busy
#: App Runner instance's working set; the idle TTL is what actually keeps the
#: cache small, since a connector tenant goes quiet between conversations.
_TENANT_CACHE_MAX = 256
_TENANT_CACHE_IDLE_TTL = 1800.0  # 30 minutes


class _TenantManagerCache:
    """Bounded, thread-safe LRU of ``UniversalMemoryManager`` per tenant.

    A manager is expensive to build (summarizer resolution, buffer setup) and
    cheap to reuse, but it must never be reused ACROSS tenants: it carries the
    Redis buffer whose keys are tenant-prefixed, and the project handle its
    writes go to. So this is deliberately a cache, not a singleton — one entry
    per ``tenant_id``, evicted by idle time first and size second.

    Eviction just drops the reference. A hosted manager's buffer talks to the
    control-plane proxy over stateless HTTP and holds no Redis client
    (``RedisMemoryBuffer.close`` is a no-op there), so there is no socket to
    leak and no async close to schedule from this synchronous path. The one
    exception is the dev escape hatch (``KUMIHO_HOSTED_LOCAL_REDIS``), where
    the buffer does own a client: an evicted one leaves its connection pool to
    the garbage collector, which is fine on a dev box and is one more reason
    that flag is not for production.
    """

    def __init__(
        self,
        *,
        max_entries: int = _TENANT_CACHE_MAX,
        idle_ttl: float = _TENANT_CACHE_IDLE_TTL,
    ) -> None:
        self.max_entries = max_entries
        self.idle_ttl = idle_ttl
        self._lock = threading.Lock()
        # key -> (manager, last_used_monotonic); ordered by recency of use.
        self._entries: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()

    def get(self, key: str, factory) -> Any:
        """Return the cached manager for *key*, building it if needed.

        ``factory`` runs OUTSIDE the lock. Manager construction can touch the
        network (discovery, summarizer probing), and holding a process-wide
        lock across that would let one slow tenant stall every other tenant's
        tool call. The cost is that two concurrent first-calls for the same
        tenant may both build; the loser's manager is discarded and the winner
        is returned, so callers still converge on ONE instance per tenant —
        which is the property the isolation depends on.
        """
        now = time.monotonic()
        with self._lock:
            self._evict_idle(now)
            entry = self._entries.get(key)
            if entry is not None:
                self._entries[key] = (entry[0], now)
                self._entries.move_to_end(key)
                return entry[0]

        manager = factory()

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries[key] = (existing[0], time.monotonic())
                self._entries.move_to_end(key)
                return existing[0]
            self._entries[key] = (manager, time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return manager

    def _evict_idle(self, now: float) -> None:
        """Drop entries unused for longer than the idle TTL. Caller holds the
        lock."""
        if self.idle_ttl <= 0:
            return
        stale = [
            key for key, (_m, last) in self._entries.items()
            if now - last >= self.idle_ttl
        ]
        for key in stale:
            self._entries.pop(key, None)

    # -- test/introspection helpers -------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_tenant_managers = _TenantManagerCache()


def _get_manager():
    """The ``UniversalMemoryManager`` for this call.

    Hosted (a request context is active): one manager per ``tenant_id`` from
    :data:`_tenant_managers`. The tenant is the isolation boundary the whole
    package is built around — buffer key prefix, graph project, active-session
    pointers — so sharing one manager across tenants would defeat every one of
    them at once.

    Otherwise: the pre-existing process singleton, byte-for-byte. Double-checked
    locking makes first-time initialization race-free: the first concurrent
    callers (parallel ``asyncio.to_thread`` dispatch) can otherwise each observe
    ``_manager is None`` and double-create the manager / Redis client.
    Construction stays lazy — the lock is only contended on the very first call
    and is free thereafter.
    """
    global _manager
    ctx = current_request()
    if ctx is not None:
        return _tenant_managers.get(
            ctx.tenant_id, lambda: _build_hosted_manager(ctx),
        )
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is not None:
            return _manager
        _manager = _build_manager()
        return _manager


def _build_hosted_manager(ctx):
    """Construct a per-tenant manager for a hosted request (plan §2.3).

    Deliberately NOT ``_build_manager`` with flags: that function's whole body
    is environment reads, and in a shared server the environment belongs to the
    operator, not the tenant. The differences are the point —

    * Redis: the control-plane proxy only, authenticated per request with
      ``ctx.auth_token`` (resolved at call time inside the buffer, because the
      manager outlives the request that built it).
    * No local artifact root, no retry queue, no failure ledger — nothing this
      process writes to disk (see ``UniversalMemoryManager.hosted``).
    * No LLM assessors, reranker, embedding adapter or provider summarizer
      unless ``KUMIHO_HOSTED_LLM=1``.
    """
    from kumiho_memory import (
        RedisMemoryBuffer,
        UniversalMemoryManager,
        MemorySummarizer,
        PIIRedactor,
    )

    buffer = RedisMemoryBuffer(
        tenant_id=ctx.tenant_id,
        # Never probe the control plane for a raw Redis URL: hosted traffic
        # goes through the proxy, which namespaces keys per tenant.
        prefer_discovery=False,
    )

    if hosted_llm_enabled():
        # Opt-in per deployment. The env config is the operator's, applied to
        # every tenant equally — acceptable only because they asked for it.
        summarizer = MemorySummarizer()
    else:
        summarizer = _hosted_keyless_summarizer()

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=summarizer,
        pii_redactor=PIIRedactor(),
        # Everything below is the explicit "off" of the corresponding
        # _build_manager env switch. Spelled out rather than defaulted so a
        # new opt-in feature there cannot silently become hosted default.
        graph_augmentation=None,
        sibling_similarity_threshold=0.0,
        embedding_adapter=None,
        auto_assess_fn=None,
        evidence_rank=None,
        rerank=None,
        reranker=None,
        retry_queue=None,
        failure_ledger=None,
    )
    logger.debug(
        "Built hosted memory manager for tenant %s (llm=%s)",
        ctx.tenant_id, "on" if hosted_llm_enabled() else "off",
    )
    return manager


def _hosted_keyless_summarizer():
    """A ``MemorySummarizer`` that can never reach a provider.

    Hosted v1 is the keyless core (plan §1 decision 10): the operator's LLM
    key must not be spent on tenant traffic, and there is no per-tenant key
    yet. Rather than invent a new "no LLM" branch through the manager, this
    raises from ``adapter`` — the exact failure the manager already handles
    everywhere it reaches for one (graph-augmented reformulation and sibling
    selection both catch it and fall back, and ``summarize_conversation``
    returns its deterministic fallback), so hosted mode travels a path that
    is already covered by tests rather than a new one.
    """
    from kumiho_memory import MemorySummarizer

    class _KeylessSummarizer(MemorySummarizer):
        @property
        def adapter(self):
            raise RuntimeError(
                "LLM calls are disabled in hosted mode; set KUMIHO_HOSTED_LLM=1 "
                "to enable them for this deployment"
            )

    return _KeylessSummarizer(
        api_key=None, provider="disabled", model="disabled", light_model="disabled",
    )


def _build_manager():
    """Construct a fresh ``UniversalMemoryManager`` from environment config.

    Called exactly once, under ``_manager_lock`` via ``_get_manager``. The
    stdio path only — hosted requests go through
    :func:`_build_hosted_manager`, which reads no environment auth at all.
    """
    from kumiho_memory import (
        RedisMemoryBuffer,
        UniversalMemoryManager,
        MemorySummarizer,
        PIIRedactor,
    )

    graph_config = None
    if os.environ.get("KUMIHO_GRAPH_AUGMENTED_RECALL", "").strip() in ("1", "true"):
        try:
            from kumiho_memory.graph_augmentation import GraphAugmentationConfig
            graph_config = GraphAugmentationConfig()
        except Exception as exc:
            logger.warning(
                "Graph-augmented recall requested but configuration failed: %s. "
                "Falling back to standard recall.",
                exc,
            )

    # Embedding-based sibling filtering (opt-in via env var).
    sibling_threshold = 0.0
    embedding_adapter = None
    raw_threshold = os.environ.get("KUMIHO_SIBLING_SIMILARITY_THRESHOLD", "").strip()
    if raw_threshold:
        try:
            sibling_threshold = float(raw_threshold)
        except ValueError:
            pass
    if sibling_threshold > 0:
        try:
            from kumiho_memory.summarization import OpenAICompatEmbeddingAdapter
            embedding_adapter = OpenAICompatEmbeddingAdapter.create()
            logger.info(
                "Embedding-based sibling filtering enabled (threshold=%.2f)",
                sibling_threshold,
            )
        except Exception as exc:
            logger.warning(
                "Sibling embedding filtering requested but adapter creation failed: %s. "
                "Falling back to BM25-light keyword filtering.",
                exc,
            )
            sibling_threshold = 0.0

    # Background auto-assessor (opt-in via KUMIHO_AUTO_ASSESS=1).
    # Uses the same LLM adapter as the summarizer — model-agnostic.
    # The assessor runs a fast heuristic pre-filter first, then a graph
    # novelty check, and only calls the LLM when both pass.
    auto_assess_fn = None
    if os.environ.get("KUMIHO_AUTO_ASSESS", "").strip() in ("1", "true"):
        try:
            from kumiho_memory.assessors import create_llm_assessor

            _tmp_summarizer = MemorySummarizer()
            _adapter = getattr(_tmp_summarizer, "adapter", None)
            _model = getattr(_tmp_summarizer, "light_model", "")
            if _adapter is not None:
                policy = os.environ.get("KUMIHO_AUTO_ASSESS_POLICY", "").strip() or None
                kwargs = {"model": _model}
                if policy:
                    kwargs["storage_policy"] = policy
                auto_assess_fn = create_llm_assessor(_adapter, **kwargs)
                logger.info(
                    "Background auto-assessor enabled (model=%s)", _model or "<default>"
                )
            else:
                logger.warning(
                    "KUMIHO_AUTO_ASSESS=1 but no LLM adapter detected — "
                    "set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable."
                )
        except Exception as exc:
            logger.warning("Auto-assess setup failed: %s", exc)

    # Evidence-aware assessor (opt-in via KUMIHO_EVIDENCE_ASSESSOR=1).
    # Takes precedence over KUMIHO_AUTO_ASSESS when both are set — it is
    # a strict superset (same pipeline + evidence grading).
    if os.environ.get("KUMIHO_EVIDENCE_ASSESSOR", "").strip() in ("1", "true"):
        try:
            from kumiho_memory.assessors import (
                EvidencePolicy,
                create_evidence_assessor,
            )

            _tmp_summarizer = MemorySummarizer()
            _adapter = getattr(_tmp_summarizer, "adapter", None)
            _model = getattr(_tmp_summarizer, "light_model", "")
            if _adapter is not None:
                try:
                    min_corroboration = max(1, int(
                        os.environ.get("KUMIHO_EVIDENCE_MIN_CORROBORATION", "2")
                    ))
                except ValueError:
                    min_corroboration = 2
                policy_kwargs: Dict[str, Any] = {
                    "min_corroboration": min_corroboration,
                    "create_supports_edges": os.environ.get(
                        "KUMIHO_EVIDENCE_SUPPORTS_EDGES", "",
                    ).strip() in ("1", "true"),
                }
                storage_policy = os.environ.get("KUMIHO_AUTO_ASSESS_POLICY", "").strip()
                if storage_policy:
                    policy_kwargs["storage_policy"] = storage_policy
                if auto_assess_fn is not None:
                    logger.info(
                        "KUMIHO_EVIDENCE_ASSESSOR overrides KUMIHO_AUTO_ASSESS "
                        "(both were set)"
                    )
                auto_assess_fn = create_evidence_assessor(
                    _adapter, model=_model, policy=EvidencePolicy(**policy_kwargs),
                )
                logger.info(
                    "Evidence assessor enabled (model=%s, min_corroboration=%d)",
                    _model or "<default>", min_corroboration,
                )
            else:
                logger.warning(
                    "KUMIHO_EVIDENCE_ASSESSOR=1 but no LLM adapter detected — "
                    "set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable."
                )
        except Exception as exc:
            logger.warning("Evidence assessor setup failed: %s", exc)

    # Evidence-weighted recall reranking — DEFAULT ON (strict no-op when
    # no memory carries an evidence grade).  KUMIHO_EVIDENCE_RERANK acts
    # as a kill switch: "0"/"false" disables.
    evidence_rank = None
    if os.environ.get("KUMIHO_EVIDENCE_RERANK", "").strip().lower() in ("0", "false"):
        try:
            from kumiho_memory.evidence_rank import EvidenceRankConfig
            evidence_rank = EvidenceRankConfig(enabled=False, badges=False)
            logger.info("Evidence-weighted recall reranking disabled via env")
        except Exception as exc:
            logger.warning("Evidence rerank config failed: %s", exc)

    summarizer = MemorySummarizer()

    # Post-recall rerank: recency decay + MMR diversity — DEFAULT ON and
    # conservative (small recency boost; relevance-dominant MMR).
    # KUMIHO_RECALL_RERANK=0/false is the kill switch.  A relevance reranker
    # stage is opt-in, via either:
    #   KUMIHO_RERANK_CROSS_ENCODER=1  -> local bge cross-encoder (needs fastembed)
    #   KUMIHO_RERANK_LLM=1            -> the host LLM itself, reusing the
    #                                     configured summarizer adapter (no extra
    #                                     model/key). Cross-encoder wins if both.
    rerank_config = None
    reranker = None
    try:
        from kumiho_memory.recall_rerank import (
            RerankConfig,
            resolve_reranker_from_env,
        )
        # Shared env resolution — identical wiring for every construction path
        # (see resolve_reranker_from_env / RerankConfig.from_env).
        rerank_config = RerankConfig.from_env()
        if os.environ.get("KUMIHO_RECALL_RERANK", "").strip().lower() in ("0", "false"):
            logger.info("Post-recall rerank (recency/MMR) disabled via env")
        reranker = resolve_reranker_from_env(
            adapter=getattr(summarizer, "adapter", None),
            model=getattr(summarizer, "light_model", "") or "",
        )
        if reranker is not None:
            rerank_config.cross_encoder_enabled = True
    except Exception as exc:
        logger.warning("Post-recall rerank setup failed: %s", exc)

    from kumiho_memory.failure_ledger import default_failure_ledger

    buffer = RedisMemoryBuffer()
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=summarizer,
        pii_redactor=PIIRedactor(),
        graph_augmentation=graph_config,
        sibling_similarity_threshold=sibling_threshold,
        embedding_adapter=embedding_adapter,
        auto_assess_fn=auto_assess_fn,
        evidence_rank=evidence_rank,
        rerank=rerank_config,
        reranker=reranker,
        # Park content that fails deterministically run after run (#118).
        failure_ledger=default_failure_ledger(),
    )
    return manager


def _min_score_from_args(args: Dict[str, Any]) -> Optional[float]:
    """Return optional relevance threshold from args or environment."""
    raw = args.get("min_score")
    if raw is None:
        raw = os.environ.get("CONSTRUCT_MEMORY_MIN_RELEVANCE_SCORE")
    if raw is None:
        raw = os.environ.get("KUMIHO_MEMORY_MIN_RELEVANCE_SCORE")
    if raw in (None, ""):
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return score


def _passes_min_score(memory: Dict[str, Any], min_score: Optional[float]) -> bool:
    if min_score is None:
        return True
    score = memory.get("score")
    if score is None:
        return True
    try:
        return float(score) >= min_score
    except (TypeError, ValueError):
        return True


def _filter_by_min_score(
    memories: List[Dict[str, Any]],
    min_score: Optional[float],
) -> List[Dict[str, Any]]:
    if min_score is None:
        return memories
    return [m for m in memories if _passes_min_score(m, min_score)]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _resolve_session(
    args: Dict[str, Any],
    manager,
    user_id: Optional[str] = None,
    context: Optional[str] = None,
) -> "Tuple[str, str]":
    """The session identity for this call, and where it came from.

    An explicit argument always wins (backfill and bulk ingest genuinely
    address historical sessions). Omitted, the id resolves server-side —
    host env, then the shared active-session pointer, then generated (see
    MemoryManager.resolve_session_id). The caller is an LLM; requiring it to
    originate and hold a stable opaque id across turns is what fragmented
    conversation buffers silently (issue #3), so omission is the recommended
    path and every result reports which id was actually used.

    Under a hosted request context the order is the same shape with the
    request substituted for the process environment (plan §2.3):
    ``argument`` → ``request`` (``ctx.session_id``) → ``active_session``
    (the pointer for ``(ctx.context, ctx.user_id)``) → ``generated``.

    ``user_id``/``context`` MUST be threaded through by identity-bearing
    tools (ingest). Resolving identity-less on their behalf bypassed the
    resolver's user-scoping guard from the MCP side and collapsed every user
    and context served by one process into one bucket — the exact blocker
    the guard exists to prevent (PR #4 review, round 2). Identity-less tools
    pass nothing and share the conversation-scoped default.

    A provided-but-blank session_id is rejected loudly, not silently
    reinterpreted as omission: on main a blank reached the buffer and raised
    there, and "" quietly resolving to some other live session would move
    data between buckets on a typo.

    A coroutine on purpose: each handler awaits it inside the SAME
    ``asyncio.run`` as the operation it precedes. As its own ``asyncio.run``
    it created a second event loop per tool call, and the redis client — one
    per loop — was abandoned unclosed every time (PR #4 review).
    """
    raw = args.get("session_id")
    if raw is None:
        # Generic identity pickup: any session-scoped tool may carry
        # user_id/context (advertised via _SESSION_IDENTITY_PROPS), so a
        # caller in the ingest workflow converges on ingest's per-user
        # session STRUCTURALLY by repeating the same user_id — a stable
        # semantic value — instead of being asked to echo an opaque session
        # uuid across turns, which is what LLM callers get wrong (PR #4
        # review, round 4).
        if user_id is None:
            user_id = args.get("user_id")
        ctx = current_request()
        if ctx is not None:
            # Hosted: the authenticated request supplies the identity the
            # process environment supplies on stdio. Defaulting from ctx here
            # (rather than leaving it None) is what makes the pointer key
            # (ctx.context, ctx.user_id) — an identity-less hosted tool and an
            # identity-bearing one then land on the SAME bucket, which is the
            # convergence issue #3 exists to protect.
            if user_id is None:
                user_id = ctx.user_id or None
            if context is None:
                context = args.get("context") or ctx.context
        elif user_id is not None and context is None:
            # Match ingest's context default so "same user_id" lands on the
            # same (context, user) pointer.
            context = args.get("context", "personal")
        return await manager.resolve_session_id(user_id=user_id, context=context)
    explicit = str(raw)
    if not explicit.strip():
        raise ValueError(
            "session_id, when provided, must be a non-empty string; "
            "omit it entirely for server-side resolution"
        )
    return explicit, "argument"


def _identity_from_args_or_request(
    args: Dict[str, Any],
    *,
    default_context: str = "personal",
) -> "Tuple[Optional[str], str]":
    """``(user_id, context)`` for an identity-BEARING tool.

    The argument always wins; a hosted request fills the gaps with its
    authenticated identity (plan §2.3: "default user_id/context for every tool
    that accepts them = ctx.user_id/ctx.context"). On stdio this is exactly
    ``args["user_id"]`` and ``args.get("context", "personal")`` — the request
    branch is unreachable, so the plugin path is unchanged.
    """
    user_id = args.get("user_id")
    context = args.get("context")
    ctx = current_request()
    if ctx is not None:
        if user_id is None:
            user_id = ctx.user_id or None
        if context is None:
            context = ctx.context
    if context is None:
        context = default_context
    return user_id, context


def _annotate_session(
    result: Dict[str, Any], session_id: str, source: str
) -> Dict[str, Any]:
    """Stamp the resolved identity onto a tool result, never silently."""
    if isinstance(result, dict):
        result["session_id"] = session_id
        result["session_id_source"] = source
    return result


# Each handler runs resolution and its operation inside ONE asyncio.run.
# _resolve_session as its own asyncio.run created a second event loop per
# tool call, and the redis client — one per loop — was abandoned unclosed
# every time (PR #4 review, measured: 2 clients per call, 0 closed).


def tool_chat_add(args: Dict[str, Any]) -> Dict[str, Any]:
    """Add a message to Redis working memory."""
    manager = _get_manager()

    async def _run():
        session_id, source = await _resolve_session(args, manager)
        result = await manager.redis_buffer.add_message(
            project=args.get("project", manager.project),
            session_id=session_id,
            role=args.get("role", "user"),
            content=args["message"],
            metadata=args.get("metadata"),
        )
        return _annotate_session(result, session_id, source)

    return asyncio.run(_run())


def tool_chat_get(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get messages from Redis working memory."""
    manager = _get_manager()

    async def _run():
        session_id, source = await _resolve_session(args, manager)
        result = await manager.redis_buffer.get_messages(
            project=args.get("project", manager.project),
            session_id=session_id,
            limit=args.get("limit", 50),
        )
        return _annotate_session(result, session_id, source)

    return asyncio.run(_run())


def tool_chat_clear(args: Dict[str, Any]) -> Dict[str, Any]:
    """Clear a session's working memory."""
    manager = _get_manager()

    async def _run():
        session_id, source = await _resolve_session(args, manager)
        result = await manager.redis_buffer.clear_session(
            args.get("project", manager.project),
            session_id,
        )
        return _annotate_session(result, session_id, source)

    return asyncio.run(_run())


def tool_memory_ingest(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a user message — buffers in Redis and returns context.

    Resolves through the shared resolver WITH its identity threaded through:
    ingest is the one identity-bearing tool, and resolving on its behalf
    without user_id/context bypassed the resolver's user-scoping guard —
    every user and context served by one process collapsed into one bucket
    (PR #4 review, round 2). Per-(user, context) resolution matches main's
    ingest semantics exactly; the ingest result reports the resolved
    session_id, and a caller that mixes ingest with the identity-less tools
    should echo that id explicitly.
    """
    manager = _get_manager()

    # Main's ingest treated a blank session_id as "generate for this user"
    # (`session_id or _generate_session_id(...)`), and callers exist that
    # pass "" deliberately. Normalize blank to omission for THIS
    # identity-bearing tool instead of raising — the loud rejection is for
    # the identity-less tools, where "" silently resolving to some other
    # live session would move data between buckets (PR #4 review, round 3).
    # Exactly "" and nothing wider (round 8): main's `session_id or
    # generate` auto-generated only for the empty string, while a
    # whitespace-only value reached the buffer and raised there. Normalizing
    # any blank silently CHANGED "  " from a loud error into a successful
    # write — _resolve_session's rejection handles it loudly instead.
    if args.get("session_id") == "":
        args = {k: v for k, v in args.items() if k != "session_id"}

    # Hosted: identity comes from the authenticated request, so user_id stops
    # being a required argument and context defaults to the request's
    # namespace instead of "personal". A caller MAY still name another user
    # (bulk ingest on someone's behalf) and the resolver's user-scoping guard
    # still applies to that case.
    user_id, context = _identity_from_args_or_request(args)
    if user_id is None:
        raise ValueError(
            "user_id is required: pass it explicitly, or call through a "
            "hosted request context that supplies the authenticated user"
        )

    async def _run():
        session_id, source = await _resolve_session(
            args, manager,
            user_id=user_id,
            context=context,
        )
        result = await manager.handle_user_message(
            user_id=user_id,
            message=args["message"],
            context=context,
            session_id=session_id,
            working_memory_limit=args.get("working_memory_limit", 10),
            recall_limit=args.get("recall_limit", 5),
            evidence_level=args.get("evidence_level"),
            source=args.get("source"),
        )
        return _annotate_session(result, session_id, source)

    return asyncio.run(_run())


def tool_memory_add_response(args: Dict[str, Any]) -> Dict[str, Any]:
    """Add an assistant response to the session buffer."""
    manager = _get_manager()

    async def _run():
        session_id, source = await _resolve_session(args, manager)
        result = await manager.add_assistant_response(
            session_id=session_id,
            response=args["response"],
        )
        return _annotate_session(result, session_id, source)

    return asyncio.run(_run())


def tool_memory_consolidate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate a session — summarize, redact PII, store to graph."""
    manager = _get_manager()

    async def _run():
        session_id, source = await _resolve_session(args, manager)
        # Keyless consolidation: forwarded only when supplied, so a manager
        # with the pre-summary signature keeps working on the LLM path.
        keyless: Dict[str, Any] = {}
        if args.get("summary") is not None:
            keyless["summary"] = args.get("summary")
        if args.get("implications") is not None:
            keyless["implications"] = args.get("implications")
        result = await manager.consolidate_session(
            session_id=session_id,
            evidence_level=args.get("evidence_level"),
            source=args.get("source"),
            **keyless,
        )
        return _annotate_session(result, session_id, source)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Recall deduplication
# ---------------------------------------------------------------------------
# Models sometimes generate parallel kumiho_memory_recall calls for the SAME
# query within a single response despite instructions not to.  This lock
# serializes recall calls so the first one executes and any *identical-query*
# duplicate within the dedup window returns an empty result — eliminating the
# duplicate "Retrieved..." output lines.  The dedup keys off the query + scope
# (not just time), so DISTINCT queries — including concurrent ones from parallel
# agents — always execute instead of being starved by one global timestamp.

_recall_lock = threading.Lock()
_RECALL_DEDUP_WINDOW_SECS = 5.0
# Recall signature -> monotonic time it last executed. Only a true duplicate
# (same query + scope) within the window is suppressed.
_recall_recent: Dict[str, float] = {}
# Guards _recall_recent itself. On stdio this is always taken inside
# _recall_lock and is uncontended; hosted callers in different scopes hold
# DIFFERENT critical-section locks, so the shared dict needs its own.
_recall_recent_guard = threading.Lock()
# Per-scope critical-section locks (hosted only), created on demand. Capped
# because a hosted server is long-lived and a scope is per SESSION, not per
# tenant: unbounded, this grows with every conversation the process ever
# serves. Over the cap the unheld locks are dropped — a caller that still
# holds one keeps working (it holds the reference), and the only cost of
# dropping a lock some later caller would have shared is that one duplicate
# recall runs instead of being suppressed, which is the guard's benign
# direction. Correctness of the dedup record itself is _recall_recent's job.
_RECALL_SCOPE_LOCK_CAP = 4096
_recall_scope_locks: Dict[str, threading.Lock] = {}
_recall_scope_locks_guard = threading.Lock()


def _recall_scope(args: Dict[str, Any]) -> str:
    """Who this recall belongs to: "" on stdio, (tenant, user, session) hosted.

    The dedup guard exists to swallow a model's duplicate tool calls *within
    one response*. Process-global, that is exactly wrong on a shared server:
    two tenants asking the same question at the same moment are not
    duplicates, and the second one would silently receive an empty result
    labelled "you already asked this". Scoping restores the guard's actual
    meaning — same conversation, same question, same instant.
    """
    ctx = current_request()
    if ctx is None:
        return ""
    return "\x1e".join((
        ctx.tenant_id or "",
        ctx.user_id or "",
        str(args.get("session_id") or ctx.session_id or ""),
    ))


def _recall_guard_lock(scope: str) -> threading.Lock:
    """The lock serializing recall for *scope*.

    Stdio keeps the single process-wide lock it has always used. Hosted gets
    one lock per scope, because the critical section spans the recall itself
    (network I/O to the graph backend): a shared lock would make every
    tenant's recall wait behind every other tenant's, turning an unrelated
    slow query into a process-wide stall.
    """
    if not scope:
        return _recall_lock
    with _recall_scope_locks_guard:
        lock = _recall_scope_locks.get(scope)
        if lock is None:
            if len(_recall_scope_locks) >= _RECALL_SCOPE_LOCK_CAP:
                for key in [k for k, v in _recall_scope_locks.items()
                            if not v.locked()]:
                    _recall_scope_locks.pop(key, None)
            lock = threading.Lock()
            _recall_scope_locks[scope] = lock
        return lock


def _recall_signature(args: Dict[str, Any], scope: str = "") -> str:
    """Dedup key: the query plus the scope args that determine the result set,
    prefixed by the requesting identity when there is one."""
    signature = "\x1f".join(str(x) for x in (
        args.get("query", ""),
        args.get("space_paths") or "",
        args.get("memory_types") or "",
        args.get("recall_mode") or "",
        bool(args.get("graph_augmented", False)),
    ))
    return f"{scope}\x1d{signature}" if scope else signature


def _recall_is_duplicate(args: Dict[str, Any], now: float, scope: str = "") -> bool:
    """True if an identical recall ran within the dedup window. Prunes expired
    signatures first so the cache stays small."""
    signature = _recall_signature(args, scope)
    with _recall_recent_guard:
        for key in [k for k, t in _recall_recent.items()
                    if now - t >= _RECALL_DEDUP_WINDOW_SECS]:
            _recall_recent.pop(key, None)
        return signature in _recall_recent


def _recall_mark(args: Dict[str, Any], scope: str = "") -> None:
    """Record that this recall just executed."""
    with _recall_recent_guard:
        _recall_recent[_recall_signature(args, scope)] = time.monotonic()


def tool_memory_recall(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search long-term memories by semantic query.

    Includes a deduplication guard: if called more than once within a short
    time window (e.g. parallel tool calls from the model), subsequent calls
    return an empty result with a note instead of hitting the backend again.
    """
    scope = _recall_scope(args)
    with _recall_guard_lock(scope):
        now = time.monotonic()
        if _recall_is_duplicate(args, now, scope):
            logger.warning(
                "kumiho_memory_recall called again with the same query within "
                "%.1fs — returning empty (query=%r)",
                _RECALL_DEDUP_WINDOW_SECS,
                args.get("query", ""),
            )
            return {
                "results": [],
                "count": 0,
                "deduplicated": True,
                "note": (
                    "Duplicate recall — identical query already returned in "
                    "this response. Vary the query or reuse the prior results."
                ),
            }

        manager = _get_manager()
        recall_mode = args.get("recall_mode", manager.recall_mode)
        results = asyncio.run(
            manager.recall_memories(
                args["query"],
                limit=args.get("limit", 5),
                space_paths=args.get("space_paths"),
                memory_types=args.get("memory_types"),
                graph_augmented=args.get("graph_augmented", False),
            )
        )
        results = _filter_by_min_score(results, _min_score_from_args(args))
        result = {"results": results, "count": len(results), "recall_mode": recall_mode}
        # Additive: surface a backend failure so an empty result isn't read as
        # "no memories" when the graph/retrieve backend was actually down.
        backend_error = getattr(manager, "_last_backend_error", None)
        if backend_error:
            result["backend_error"] = backend_error

        _recall_mark(args, scope)
        return result


def tool_memory_store_execution(args: Dict[str, Any]) -> Dict[str, Any]:
    """Store a tool/command execution result as memory."""
    manager = _get_manager()
    return asyncio.run(
        manager.store_tool_execution(
            task=args["task"],
            status=args.get("status", "done"),
            exit_code=args.get("exit_code"),
            duration_ms=args.get("duration_ms"),
            stdout=args.get("stdout", ""),
            stderr=args.get("stderr", ""),
            tools=args.get("tools"),
            topics=args.get("topics"),
            space_hint=args.get("space_hint", ""),
        )
    )


def tool_memory_space_profile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Profile each Space's knowledge dynamics (no LLM)."""
    from kumiho_memory import SpaceProfiler

    profiler = SpaceProfiler(
        project=args.get("project", "CognitiveMemory"),
        window_days=args.get("window_days", 30),
        dry_run=args.get("dry_run", False),
    )
    return asyncio.run(profiler.run())


def tool_memory_dream_state(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a Dream State memory consolidation cycle."""
    from kumiho_memory import DreamState
    from kumiho_memory.summarization import MemorySummarizer

    # Build a summarizer with explicit model config if provided,
    # otherwise let DreamState use its default (env-var based).
    summarizer = None
    provider = args.get("provider")
    model = args.get("model")
    api_key = args.get("api_key")
    base_url = args.get("base_url")
    if provider or model or api_key or base_url:
        # If provider/model were specified but no api_key, inherit the key
        # from the shared manager's summarizer (which resolved it from env
        # vars at startup).  This avoids requiring a separate LLM key config
        # for Dream State — it reuses whatever the MCP server already has.
        if not api_key or not base_url:
            try:
                shared = _get_manager()
                if not api_key:
                    api_key = getattr(shared.summarizer, "api_key", None)
                if not base_url:
                    base_url = getattr(shared.summarizer, "_base_url", None)
            except Exception:
                pass

        summarizer = MemorySummarizer(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    from kumiho_memory.failure_ledger import default_failure_ledger

    ds = DreamState(
        project=args.get("project", "CognitiveMemory"),
        batch_size=args.get("batch_size", 20),
        dry_run=args.get("dry_run", False),
        max_deprecation_ratio=args.get("max_deprecation_ratio", 0.5),
        allow_published_deprecation=args.get("allow_published_deprecation", False),
        extra_instructions=args.get("extra_instructions"),
        # Pass None (not False) when the caller omits it, so DreamState's
        # tri-state sentinel lets KUMIHO_DREAM_MAINTAIN_GRAPH decide.
        maintain_graph=args.get("maintain_graph"),
        maintenance_llm=args.get("maintenance_llm", False),
        code_project=args.get("code_project"),
        summarizer=summarizer,
        # Skip items parked for repeated deterministic failures (#118).
        failure_ledger=default_failure_ledger(),
    )
    return asyncio.run(ds.run())


def tool_memory_discover_edges(args: Dict[str, Any]) -> Dict[str, Any]:
    """Discover and create edges from a memory to related existing memories."""
    manager = _get_manager()
    edges = asyncio.run(
        manager.discover_edges_post_consolidation(
            revision_kref=args["revision_kref"],
            summary=args["summary"],
            max_queries=args.get("max_queries", 5),
            max_edges=args.get("max_edges", 3),
            min_score=args.get("min_score", 0.3),
            edge_type=args.get("edge_type", "REFERENCED"),
            space_paths=args.get("space_paths"),
        )
    )
    return {"edges": edges, "count": len(edges)}


# ---------------------------------------------------------------------------
# Composite tools — engage / reflect
# ---------------------------------------------------------------------------


def tool_memory_engage(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check memory before responding — combines recall + context building.

    Returns pre-built context, raw results, and source krefs for linking.
    Shares the recall deduplication guard with ``tool_memory_recall``.
    """
    scope = _recall_scope(args)
    with _recall_guard_lock(scope):
        now = time.monotonic()
        if _recall_is_duplicate(args, now, scope):
            return {
                "context": "",
                "results": [],
                "source_krefs": [],
                "count": 0,
                "deduplicated": True,
                "note": (
                    "Duplicate recall — identical query already returned in "
                    "this response. Vary the query or reuse the prior results."
                ),
            }

        manager = _get_manager()
        recall_mode = args.get("recall_mode", manager.recall_mode)
        results = asyncio.run(
            manager.recall_memories(
                args["query"],
                limit=args.get("limit", 5),
                space_paths=args.get("space_paths"),
                memory_types=args.get("memory_types"),
                graph_augmented=args.get("graph_augmented", False),
            )
        )
        results = _filter_by_min_score(results, _min_score_from_args(args))
        context = manager.build_recalled_context(
            results, args["query"], recall_mode
        )
        source_krefs = [m["kref"] for m in results if m.get("kref")]
        _drop_sibling_prose(results, recall_mode)

        import json

        from kumiho_memory.context_compose import approx_tokens

        _recall_mark(args, scope)
        engage_result = {
            "context": context,
            "results": results,
            "source_krefs": source_krefs,
            "count": len(results),
            "recall_mode": recall_mode,
            # Additive budgeting signal (chars/4 heuristic) so callers can
            # size the assembled context without a tokenizer.
            "approx_tokens": approx_tokens(context),
        }
        # ``approx_tokens`` sizes the assembled context only; a caller budgeting
        # on it was off by a factor of ~18 (762 reported vs a 56 KB response,
        # measured 2026-07-31). What lands in the caller's context window is the
        # whole envelope, so report that too rather than silently redefining the
        # older field.
        engage_result["approx_payload_tokens"] = approx_tokens(
            json.dumps(engage_result, ensure_ascii=False, default=str)
        )
        # Additive: surface a backend failure so an empty result isn't read as
        # "no memories" when the graph/retrieve backend was actually down.
        backend_error = getattr(manager, "_last_backend_error", None)
        if backend_error:
            engage_result["backend_error"] = backend_error
        return engage_result


def _drop_sibling_prose(results: List[Dict[str, Any]], recall_mode: str) -> None:
    """Summarized engage: keep the sibling COUNT, drop the sibling prose.

    ``sibling_revisions`` exists so ``build_recalled_context`` can rank every
    revision of an item globally and fold the winners into ``context``. Once it
    has run, returning those same summaries again in ``results`` is pure
    duplication -- and it dominates the response: 47,903 of 55,939 chars (86%)
    over 31 siblings on a plain ``limit=5`` engage, measured 2026-07-31. That
    response overran the host's tool-result ceiling and was spilled to a file,
    so a call whose whole purpose is to put memory INTO the context window put
    none of it there.

    The count stays because a silently shortened list reads as "this item has
    one revision". Callers that want the prose ask for ``recall_mode="full"``,
    where siblings are left untouched.
    """
    if recall_mode == "full":
        return
    for mem in results:
        siblings = mem.pop("sibling_revisions", None)
        if siblings:
            mem["sibling_count"] = len(siblings)


def _capture_space(cap: Dict[str, Any], space_path: str) -> str:
    """The space this capture was routed to, or "" when the caller named none.

    Empty means the project root, and the root is not a space: it is where
    every unrouted memory in the project accumulates. Truthiness is therefore
    the stacking test -- stacking searches the capture's space for something to
    merge into, which is only meaningful inside a space the caller chose.
    """
    return (cap.get("space_hint") or "").strip() or (space_path or "").strip()


def tool_memory_reflect(args: Dict[str, Any]) -> Dict[str, Any]:
    """Capture what matters after responding — buffers response + stores facts.

    Combines ``add_assistant_response`` + N stores + optional edge discovery
    into a single call.  The agent provides structured captures — no external
    LLM is needed for fact extraction.
    """
    manager = _get_manager()
    response = args["response"]

    # 1. Resolve identity + buffer response — one event loop for both.
    async def _resolve_and_buffer():
        session_id, session_source = await _resolve_session(args, manager)
        buf = await manager.add_assistant_response(
            session_id=session_id, response=response,
        )
        return session_id, session_source, buf

    session_id, session_source, buf_result = asyncio.run(_resolve_and_buffer())
    buffered = buf_result.get("success", True)

    # 2. Store captures
    captures = args.get("captures") or []
    source_krefs = args.get("source_krefs") or []
    space_path = args.get("space_path", "")
    do_edges = args.get("discover_edges", True)
    idempotency_prefix = args.get("idempotency_prefix", "") or ""
    stored_krefs: List[str] = []
    edges_total = 0
    dropped_event_dates: List[Dict[str, str]] = []
    capture_results: Optional[List[Dict[str, Any]]] = None

    if captures:
        import kumiho as _kumiho
        from kumiho.mcp_server import tool_memory_store

        # Canonical valid-time validator, shared with the summarizer path so
        # keyless (reflect) and LLM (consolidation) writes agree on shape.
        from kumiho_memory.memory_manager import _ISO_EVENT_DATE_RE

        # Valid-time: validate each capture's ISO event_date ONCE up front so the
        # single-write and batched-write paths stamp identical, validated metadata
        # onto the revision (temporal recall anchors on when the event happened,
        # separate from the server-set created_at). A malformed or relative date is
        # dropped and reported — reflect must never fail over a bad date.
        prepared: List[Dict[str, Any]] = []
        for cap in captures:
            cap_metadata: Optional[Dict[str, str]] = None
            raw_event_date = str(cap.get("event_date", "") or "").strip()
            if raw_event_date:
                if _ISO_EVENT_DATE_RE.match(raw_event_date):
                    cap_metadata = {"event_date": raw_event_date}
                else:
                    dropped_event_dates.append({
                        "title": cap.get("title", ""),
                        "event_date": raw_event_date,
                    })
            prepared.append({"cap": cap, "metadata": cap_metadata})

        def _discover(rev_kref: str, cap: Dict[str, Any]) -> None:
            # Edge discovery (best-effort, skipped if no server-side LLM).
            if not (do_edges and rev_kref
                    and earns_edge_discovery(cap.get("type"))):
                return
            nonlocal edges_total
            try:
                edges = asyncio.run(
                    manager.discover_edges_post_consolidation(
                        revision_kref=rev_kref, summary=cap.get("content", ""),
                    )
                )
                edges_total += len(edges)
            except Exception:
                pass  # graceful — edge discovery is supplementary

        # A ≥2-capture reflect (backfill and any bulk write) goes through ONE
        # batched transaction: it removes the neo4j relationship-group deadlock
        # that per-capture writes hit under load and collapses the heaviest
        # create/revision RPCs into one. A single capture — the common live case —
        # keeps the byte-identical per-capture path below. The guard also degrades
        # gracefully if the installed kumiho core predates the batch helper.
        try:
            from kumiho.mcp_server import tool_memory_store_batch
            _has_batch = hasattr(_kumiho, "batch_create_revisions")
        except ImportError:
            _has_batch = False

        # An idempotency_prefix (bulk/backfill resume) forces the batched path even
        # for a single capture, so the caller always gets positional capture_results.
        if (len(prepared) >= 2 or idempotency_prefix) and _has_batch:
            # The batch RPC takes ONE stacking flag for the whole write, so a
            # single unrouted capture disables stacking for its siblings too.
            # That is the safe direction: a missed stack costs one extra item,
            # a wrong stack silently fuses two memories.
            spaces = [_capture_space(p["cap"], space_path) for p in prepared]
            batch_out = tool_memory_store_batch(
                captures=[{
                    "type": p["cap"].get("type", "summary"),
                    "title": p["cap"].get("title", ""),
                    "content": p["cap"].get("content", ""),
                    "tags": p["cap"].get("tags"),
                    "metadata": p["metadata"],
                    "space_hint": space,
                } for p, space in zip(prepared, spaces)],
                project=manager.project,
                space_path=space_path,
                source_revision_krefs=source_krefs if source_krefs else None,
                edge_type="DERIVED_FROM",
                stack_revisions=all(spaces),
                idempotency_prefix=idempotency_prefix,
            )
            # Positionally-aligned per-capture results (each {revision_kref, ...} or
            # {error}) so a bulk caller (history backfill) can map + mark each
            # capture exactly — reflect's flat stored_krefs alone can't attribute a
            # mid-batch failure. Rows the idempotency_prefix replayed as no-ops come
            # back with their existing revision_kref just like fresh writes.
            capture_results = batch_out.get("results") or []
            for p, res in zip(prepared, capture_results):
                rev_kref = (res or {}).get("revision_kref", "")
                if rev_kref:
                    stored_krefs.append(rev_kref)
                    _discover(rev_kref, p["cap"])
        else:
            for p in prepared:
                cap = p["cap"]
                cap_space = _capture_space(cap, space_path)
                store_result = tool_memory_store(
                    project=manager.project,
                    space_path=cap_space,
                    memory_type=cap.get("type", "summary"),
                    title=cap.get("title", ""),
                    summary=cap.get("content", ""),
                    assistant_text=cap.get("content", ""),
                    source_revision_krefs=source_krefs if source_krefs else None,
                    edge_type="DERIVED_FROM",
                    tags=cap.get("tags"),
                    metadata=p["metadata"],
                    stack_revisions=bool(cap_space),
                )
                rev_kref = store_result.get("revision_kref", "")
                if rev_kref:
                    stored_krefs.append(rev_kref)
                _discover(rev_kref, cap)

    result: Dict[str, Any] = {
        "buffered": buffered,
        "captures_stored": len(stored_krefs),
        "edges_discovered": edges_total,
        "stored_krefs": stored_krefs,
        # A response buffered into a bucket this write MINTED means the id
        # did not match any live conversation — the silent-drift signature
        # (issue #3). Surfaced so the caller can react instead of trusting
        # a success that filed the turn somewhere nobody will read.
        "created_bucket": buf_result.get("created_bucket", False),
    }
    if capture_results is not None:
        result["capture_results"] = capture_results
    if dropped_event_dates:
        result["dropped_event_dates"] = dropped_event_dates
    return _annotate_session(result, session_id, session_source)


# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema)
# ---------------------------------------------------------------------------

MEMORY_TOOLS: List[Dict[str, Any]] = [
    # ── Chat memory (Redis working memory) ────────────────────
    {
        "name": "kumiho_chat_add",
        "description": (
            "Add a user or assistant message to Redis working memory for a "
            "conversation session. Returns the current message count."
            + _SESSION_DEFAULT_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project name for Redis key namespace. Defaults to server's configured project.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier.",
                },
                "message": {
                    "type": "string",
                    "description": "Message content.",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "default": "user",
                    "description": "Message role.",
                },
                **_SESSION_IDENTITY_PROPS,
            },
            "required": ["message"],
        },
    },
    {
        "name": "kumiho_chat_get",
        "description": (
            "Retrieve recent messages from Redis working memory for a "
            "session. Returns messages, count, and TTL."
            + _SESSION_DEFAULT_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project name for Redis key namespace. Defaults to server's configured project.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier.",
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Max messages to return.",
                },
                **_SESSION_IDENTITY_PROPS,
            },
        },
    },
    {
        "name": "kumiho_chat_clear",
        "description": (
            "Clear all working memory for a conversation session."
            + _SESSION_DEFAULT_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project name for Redis key namespace. Defaults to server's configured project.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier.",
                },
                **_SESSION_IDENTITY_PROPS,
            },
        },
    },
    # ── Memory lifecycle (orchestrated) ───────────────────────
    {
        "name": "kumiho_memory_ingest",
        "description": (
            "Ingest a user message into AI cognitive memory. Buffers the "
            "message in Redis, recalls relevant long-term memories, and "
            "returns working memory + long-term context. This is the main "
            "entry point for AI agents handling user messages. "
            # Ingest-accurate text, NOT the shared note: ingest resolves
            # per-user, so the generic "defaults to the host session" claim
            # was false here and told the caller the tools converge when they
            # structurally cannot (PR #4 review, round 3).
            "session_id is OPTIONAL: omitted, it defaults to this user's "
            "active-session pointer for (context, user_id), or a new "
            "per-user id is generated — the host session env vars are NOT "
            "consulted for user-scoped ingest. The result reports the "
            "session_id used; pass that id to the other memory tools "
            "(add_response, consolidate, reflect) to keep the conversation "
            "in one bucket."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Stable user identifier.",
                },
                "message": {
                    "type": "string",
                    "description": "User message text.",
                },
                "context": {
                    "type": "string",
                    "default": "personal",
                    "description": (
                        "Memory context: personal, work, etc."
                    ),
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Existing session ID. If omitted, defaults to the "
                        "per-user active-session pointer for (context, "
                        "user_id); a new per-user id is generated when none "
                        "exists. The host session env vars are NOT consulted "
                        "for user-scoped ingest."
                    ),
                },
                "working_memory_limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max working memory messages to return.",
                },
                "recall_limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max long-term memories to recall.",
                },
                "evidence_level": {
                    "type": "string",
                    "enum": ["official", "corroborated", "single_source", "unverified"],
                    "description": (
                        "Evidence grade stamped on the consolidated memory "
                        "(metadata + mirrored evidence:<level> tag). Stashed "
                        "in session metadata and applied at consolidation. "
                        "Omit to leave the memory ungraded."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Source identifier for evidence tracking, e.g. "
                        "'press-release:acme', 'news:reuters', 'chat:user'."
                    ),
                },
            },
            "required": ["user_id", "message"],
        },
    },
    {
        "name": "kumiho_memory_add_response",
        "description": (
            "Add an assistant response to the session buffer in Redis "
            "working memory. Call this after generating your response."
            + _SESSION_DEFAULT_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier from ingest.",
                },
                "response": {
                    "type": "string",
                    "description": "Assistant response text.",
                },
                **_SESSION_IDENTITY_PROPS,
            },
            "required": ["response"],
        },
    },
    {
        "name": "kumiho_memory_consolidate",
        "description": (
            "Consolidate a conversation session into long-term memory. "
            "Keyless when you pass `summary`: write the session summary "
            "yourself from the conversation (or have a subagent write it) "
            "and no external LLM is called. Redacts PII, writes a local "
            "artifact, stores the summary to the Kumiho graph, and clears "
            "the session's working memory. Without `summary` the configured "
            "LLM summarizer runs instead, which needs an API key or "
            "KUMIHO_LLM_BASE_URL and fails keyless."
            + _SESSION_DEFAULT_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier to consolidate.",
                },
                "summary": {
                    "type": ["object", "string"],
                    "description": (
                        "The summary of this session, written by you (or a "
                        "subagent) from the conversation — keyless, no LLM "
                        "call. Plain text, or an object: {type, title, "
                        "summary, events:[{event, when, event_date (ISO), "
                        "participants:[], consequence}], knowledge:{facts:"
                        "[{claim, certainty}], decisions:[{decision, reason}], "
                        "actions:[{task, status}], open_questions:[]}, "
                        "classification:{topics:[], entities:[]}}. Only "
                        "'summary' is required; missing fields default to "
                        "empty; 'type' is one of summary, fact, decision, "
                        "action, reflection, error (default summary). Write "
                        "it for a reader who was not there: "
                        "what was decided and why, durable facts, open items."
                    ),
                },
                "implications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional, with summary: future situations in which "
                        "this conversation would matter, worded differently "
                        "from the summary so vector search bridges to them."
                    ),
                },
                "evidence_level": {
                    "type": "string",
                    "enum": ["official", "corroborated", "single_source", "unverified"],
                    "description": (
                        "Evidence grade stamped on the stored memory "
                        "(metadata + mirrored evidence:<level> tag). "
                        "Overrides any grade stashed at ingest time. "
                        "Omit to leave the memory ungraded."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Source identifier for evidence tracking, e.g. "
                        "'press-release:acme', 'news:reuters', 'chat:user'."
                    ),
                },
                **_SESSION_IDENTITY_PROPS,
            },
        },
    },
    {
        "name": "kumiho_memory_recall",
        "description": (
            "Search long-term memories by semantic query. Returns matching "
            "memories from the Kumiho graph with optional filtering by "
            "space paths and memory types."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max results to return.",
                },
                "min_score": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Optional minimum relevance score. Results with a "
                        "numeric score below this threshold are dropped."
                    ),
                },
                "space_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Restrict search to these space paths "
                        "(e.g. ['CognitiveMemory/personal'])."
                    ),
                },
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by memory type "
                        "(e.g. ['error'], ['action', 'summary'])."
                    ),
                },
                "graph_augmented": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Enable graph-augmented recall: multi-query "
                        "reformulation + edge traversal to discover "
                        "connected memories that vector search alone misses. "
                        "Requires KUMIHO_GRAPH_AUGMENTED_RECALL=1 env var."
                    ),
                },
                "recall_mode": {
                    "type": "string",
                    "enum": ["full", "summarized"],
                    "default": "summarized",
                    "description": (
                        "Context mode: 'full' includes artifact content "
                        "(raw conversation text), 'summarized' returns only "
                        "title + summary. Affects build_recalled_context()."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    # ── Edge discovery ─────────────────────────────────────────
    {
        "name": "kumiho_memory_discover_edges",
        "description": (
            "Discover and create edges from a newly stored memory to "
            "related existing memories. Uses the LLM to generate "
            "'implication queries' (future scenarios where the memory "
            "would be relevant) and links to matching memories. "
            "Best used after kumiho_memory_consolidate or "
            "kumiho_memory_store."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "revision_kref": {
                    "type": "string",
                    "description": (
                        "The kref URI of the revision to discover "
                        "edges for."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Summary text of the memory (used to generate "
                        "implication queries)."
                    ),
                },
                "max_queries": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max implication queries to generate.",
                },
                "max_edges": {
                    "type": "integer",
                    "default": 3,
                    "description": "Max edges to create.",
                },
                "min_score": {
                    "type": "number",
                    "default": 0.3,
                    "description": (
                        "Minimum similarity score for edge candidates."
                    ),
                },
                "edge_type": {
                    "type": "string",
                    "default": "REFERENCED",
                    "description": "Edge type to create (default: REFERENCED).",
                },
                "space_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Restrict search to these space paths. "
                        "Auto-derived from revision_kref if omitted."
                    ),
                },
            },
            "required": ["revision_kref", "summary"],
        },
    },
    # ── Tool execution ────────────────────────────────────────
    {
        "name": "kumiho_memory_store_execution",
        "description": (
            "Store a tool or command execution result as a structured "
            "memory. Successful executions are stored as 'action' type; "
            "failures as 'error' type. Includes stdout/stderr as artifacts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Description of what was executed "
                        "(e.g. 'git push origin main')."
                    ),
                },
                "status": {
                    "type": "string",
                    "default": "done",
                    "enum": ["done", "failed", "error", "blocked"],
                    "description": "Execution outcome.",
                },
                "exit_code": {
                    "type": "integer",
                    "description": "Process exit code (0 = success).",
                },
                "duration_ms": {
                    "type": "integer",
                    "description": "Execution duration in milliseconds.",
                },
                "stdout": {
                    "type": "string",
                    "description": "Captured standard output.",
                },
                "stderr": {
                    "type": "string",
                    "description": "Captured standard error.",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names used.",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Classification topics.",
                },
                "space_hint": {
                    "type": "string",
                    "description": "Space path hint for organisation.",
                },
            },
            "required": ["task"],
        },
    },
    # ── Composite (engage / reflect) ─────────────────────────
    {
        "name": "kumiho_memory_engage",
        "description": (
            "Check memory before responding. Combines recall + context "
            "building into one call. Returns pre-built context string, "
            "raw results, and source_krefs for passing to reflect. "
            "Shares the recall deduplication guard — at most one engage "
            "or recall per response."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language search query derived from the "
                        "user's current message."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max results to return.",
                },
                "min_score": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Optional minimum relevance score. Results with a "
                        "numeric score below this threshold are dropped "
                        "before context is built."
                    ),
                },
                "space_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Restrict search to these space paths "
                        "(e.g. ['CognitiveMemory/personal'])."
                    ),
                },
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by memory type "
                        "(e.g. ['decision', 'preference'])."
                    ),
                },
                "graph_augmented": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Enable graph-augmented recall for indirect or "
                        "chain-of-decision questions."
                    ),
                },
                "recall_mode": {
                    "type": "string",
                    "enum": ["full", "summarized"],
                    "default": "summarized",
                    "description": (
                        "Context mode: 'full' includes artifact content, "
                        "'summarized' returns title + summary only."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "kumiho_memory_reflect",
        "description": (
            "Capture what matters after responding. Buffers the assistant "
            "response and stores structured captures (decisions, preferences, "
            "facts) with provenance links — all in one call. The agent's own "
            "LLM identifies what to remember; no external API key needed. "
            "Pass source_krefs from engage to create DERIVED_FROM edges. "
            # Restated here on purpose: hosts have been observed to forward
            # property descriptions and `required` arrays only for REQUIRED
            # top-level properties, so a caller reading the schema alone sees
            # `captures` completely undocumented. The tool description does
            # survive, so the non-negotiable parts live here too.
            "REQUIRES response. Each capture in `captures` "
            "requires type, title and content."
            + _SESSION_DEFAULT_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier.",
                },
                "response": {
                    "type": "string",
                    "description": "The assistant response text to buffer.",
                },
                "captures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                # `enum`, not prose: a host that strips the
                                # description off this optional sub-schema
                                # still forwards the allowed values, so the
                                # caller stops guessing (see MEMORY_TYPES).
                                "enum": list(MEMORY_TYPES),
                                "description": "Memory type.",
                            },
                            "title": {
                                "type": "string",
                                "description": (
                                    "Short title with absolute dates "
                                    "(e.g. 'Chose gRPC on Mar 27')."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to store.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Classification tags.",
                            },
                            "space_hint": {
                                "type": "string",
                                "description": (
                                    "Space this capture belongs in; overrides "
                                    "top-level space_path. Copy an existing "
                                    "space name exactly — it is used verbatim, "
                                    "so 'Skills' and 'skills' are two spaces. "
                                    "Without it the capture is filed at the "
                                    "project root AND stored as its own item: "
                                    "revision stacking is only attempted "
                                    "inside a space you named, because at the "
                                    "root the similarity search spans every "
                                    "unrouted memory in the project."
                                ),
                            },
                            "event_date": {
                                "type": "string",
                                "description": (
                                    "ISO-8601 calendar date the captured event "
                                    "actually happened (YYYY, YYYY-MM, or "
                                    "YYYY-MM-DD) — valid-time, kept separate "
                                    "from storage time. Lets temporal recall "
                                    "anchor on when it happened rather than "
                                    "when it was written. Omit when unknown; "
                                    "never guess."
                                ),
                            },
                        },
                        "required": ["type", "title", "content"],
                    },
                    "description": (
                        "Structured facts to store. Each capture becomes a "
                        "graph memory with provenance links. Skip for trivial "
                        "exchanges."
                    ),
                },
                "source_krefs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Krefs from engage results — creates DERIVED_FROM "
                        "edges to source memories."
                    ),
                },
                "space_path": {
                    "type": "string",
                    "description": (
                        "Default space path for captures without a "
                        "space_hint."
                    ),
                },
                "discover_edges": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Run edge discovery on stored captures. "
                        "Gracefully skipped if no server-side LLM."
                    ),
                },
                "idempotency_prefix": {
                    "type": "string",
                    "description": (
                        "Optional. Write the captures through one "
                        "BatchCreateRevisions transaction keyed on "
                        "{prefix}:{index}, so re-submitting the same captures "
                        "replays committed rows as a no-op; the result carries a "
                        "positionally-aligned capture_results list ("
                        "{revision_kref} | {error} per capture) for exact "
                        "per-capture mapping. Used by history backfill for "
                        "resumable bulk ingest."
                    ),
                },
                **_SESSION_IDENTITY_PROPS,
            },
            "required": ["response"],
        },
    },
    # ── Maintenance ───────────────────────────────────────────
    {
        "name": "kumiho_memory_dream_state",
        "description": (
            "Run a Dream State memory consolidation cycle. Replays new "
            "events, assesses memories with an LLM, and applies "
            "deprecation, tagging, metadata enrichment, and relationship "
            "linking. Use dry_run=true to preview without mutations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "default": "CognitiveMemory",
                    "description": "Kumiho project to process.",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Preview changes without applying.",
                },
                "batch_size": {
                    "type": "integer",
                    "default": 20,
                    "description": "Memories per LLM assessment batch.",
                },
                "max_deprecation_ratio": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.1,
                    "maximum": 0.9,
                    "description": "Max fraction of memories to deprecate per run (0.1-0.9).",
                },
                "allow_published_deprecation": {
                    "type": "boolean",
                    "default": False,
                    "description": "Allow deprecation of published items (use with caution).",
                },
                "extra_instructions": {
                    "type": "string",
                    "description": (
                        "Deployment policy appended to the assessment system "
                        "prompt under a DEPLOYMENT POLICY section (e.g. "
                        "'Never propose deprecation for memories tagged "
                        "evidence:official'). Falls back to the "
                        "KUMIHO_DREAM_EXTRA_INSTRUCTIONS env var when omitted. "
                        "Cannot weaken hard guardrails (deprecation cap, "
                        "published protection, conservative-KEEP rule)."
                    ),
                },
                "maintain_graph": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Also consolidate the typed graphs (issue #59): merge "
                        "duplicate entities, dedup facts, prune orphans, "
                        "re-grade code-decision evidence from current atoms, "
                        "dedup decisions, and bridge code_decision→entity. "
                        "Keyless and deterministic; runs even with no new "
                        "revisions. Honors dry_run/max_deprecation_ratio. "
                        "Falls back to KUMIHO_DREAM_MAINTAIN_GRAPH."
                    ),
                },
                "maintenance_llm": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "With maintain_graph, also ask the LLM for semantic "
                        "entity-merge pairs the deterministic alias rule can't "
                        "see (applied through the keyless write path). Needs a "
                        "summarizer key."
                    ),
                },
                "code_project": {
                    "type": "string",
                    "description": (
                        "Explicit {repo}-decisions project for the Decision Memory "
                        "maintenance passes. Derived from KUMIHO_MEMORY_DECISIONS "
                        "wiring when omitted."
                    ),
                },
                "provider": {
                    "type": "string",
                    "enum": ["openai", "anthropic", "gemini"],
                    "description": "LLM provider for assessment. Falls back to KUMIHO_LLM_PROVIDER env var if not set.",
                },
                "model": {
                    "type": "string",
                    "description": "LLM model name for assessment. Falls back to KUMIHO_LLM_MODEL env var if not set.",
                },
                "api_key": {
                    "type": "string",
                    "description": "LLM API key. Falls back to KUMIHO_LLM_API_KEY env var if not set.",
                },
                "base_url": {
                    "type": "string",
                    "description": "OpenAI-compatible base URL. Falls back to KUMIHO_LLM_BASE_URL env var if not set.",
                },
            },
        },
    },
    {
        "name": "kumiho_memory_space_profile",
        "description": (
            "Profile each Space's knowledge dynamics: aggregate churn/"
            "evidence/stability signals, classify Spaces as canonical/"
            "working/correspondence, and persist versioned space-profile "
            "items. Pure aggregation — no LLM calls. A space_class Space "
            "attribute pins the label (drift is then reported only). Use "
            "dry_run=true to classify without persisting."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "default": "CognitiveMemory",
                    "description": "Kumiho project to profile.",
                },
                "window_days": {
                    "type": "integer",
                    "default": 30,
                    "description": "Look-back window for the revision-rate signal.",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Classify but do not persist profiles.",
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handler mapping
# ---------------------------------------------------------------------------

MEMORY_TOOL_HANDLERS: Dict[str, Any] = {
    "kumiho_chat_add": tool_chat_add,
    "kumiho_chat_get": tool_chat_get,
    "kumiho_chat_clear": tool_chat_clear,
    "kumiho_memory_ingest": tool_memory_ingest,
    "kumiho_memory_add_response": tool_memory_add_response,
    "kumiho_memory_consolidate": tool_memory_consolidate,
    "kumiho_memory_recall": tool_memory_recall,
    "kumiho_memory_discover_edges": tool_memory_discover_edges,
    "kumiho_memory_engage": tool_memory_engage,
    "kumiho_memory_reflect": tool_memory_reflect,
    "kumiho_memory_store_execution": tool_memory_store_execution,
    "kumiho_memory_dream_state": tool_memory_dream_state,
    "kumiho_memory_space_profile": tool_memory_space_profile,
}


# ---------------------------------------------------------------------------
# Code Decision Memory (opt-in: KUMIHO_MEMORY_DECISIONS=1)
# ---------------------------------------------------------------------------
# Registered only when the gate is on — with the gate off this module is
# byte-identical in behavior to the pre-code-domain version (the conversation
# domain's release safety pin; see docs/DECISION_MEMORY_DESIGN.md §5.5).


def tool_code_why(args: Dict[str, Any]) -> Dict[str, Any]:
    """Recall captured code decisions ("why is this code the way it is?")."""
    manager = _get_manager()
    return asyncio.run(
        manager.code_why(
            args.get("question"),
            file=args.get("file"),
            line=args.get("line"),
            commit=args.get("commit"),
            repo=args.get("repo"),
            limit=args.get("limit", 5),
        )
    )


def tool_code_ingest(args: Dict[str, Any]) -> Dict[str, Any]:
    """Mine a git commit range into decision nodes (LLM-structured, idempotent)."""
    manager = _get_manager()
    return asyncio.run(
        manager.code_ingest(
            args.get("repo_path", "."),
            args.get("rev_range"),
            max_commits=args.get("max_commits"),
            force=args.get("force", False),
        )
    )


def tool_code_capture(args: Dict[str, Any]) -> Dict[str, Any]:
    """Store agent-extracted code decisions (keyless — you did the extraction)."""
    manager = _get_manager()
    return asyncio.run(
        manager.code_capture(
            args.get("decisions") or [],
            repo_path=args.get("repo_path", "."),
            commit_ref=args.get("commit_ref", "HEAD"),
        )
    )


def tool_code_mine_session(args: Dict[str, Any]) -> Dict[str, Any]:
    """Mine an agent session's conversation into the code-decision graph."""
    manager = _get_manager()
    from kumiho_memory.memory_manager import _host_session_env

    async def _run():
        raw = args.get("session_id")
        if raw is not None:
            session_id = str(raw).strip()
            if not session_id:
                # Provided-but-blank is an ERROR, not an omission — main
                # guaranteed a loud failure for it, and reinterpreting it
                # would mine some other session on a typo (round 8).
                return {"errors": [
                    "session_id, when provided, must be a non-empty string; "
                    "omit it to default to the current host session",
                ]}
        else:
            if not _host_session_env():
                # Fail BEFORE the manager call: with ingest_first=True an
                # empty id used to run a full commit-ingest pre-pass (LLM
                # calls, graph writes) and then soft-fail inside
                # mine_session (PR #4 review).
                return {"errors": [
                    "session_id is required: pass it explicitly or set "
                    "KUMIHO_SESSION_ID",
                ]}
            # Through the GENERATION-AWARE resolver, not the raw env value:
            # mine_session loads its transcript from the same Redis
            # working-memory bucket the identity-less tools write, so after
            # any consolidation the raw env id named the consolidated,
            # cleared base bucket while the live segment sat in the rotated
            # one — a guaranteed spend-then-soft-fail (round 8).
            session_id, _source = await manager.resolve_session_id()
        return await manager.code_mine_session(
            session_id,
            conversation_kref=args.get("conversation_kref", ""),
            repo_path=args.get("repo_path", "."),
            ingest_first=args.get("ingest_first", True),
            force=args.get("force", False),
        )

    return asyncio.run(_run())


def tool_memory_decompose(args: Dict[str, Any]) -> Dict[str, Any]:
    """Keyless: store an agent-extracted ontology decomposition of a memory."""
    manager = _get_manager()
    return asyncio.run(
        manager.memory_decompose(
            args.get("kref", ""),
            entities=args.get("entities") or [],
            facts=args.get("facts") or [],
            relations=args.get("relations") or [],
            supersedes=args.get("supersedes") or [],
            contradicts=args.get("contradicts") or [],
            project=args.get("project"),
        )
    )


_CODE_MEMORY_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "kumiho_code_why",
        "description": (
            "Ask why code is the way it is — recall captured decisions "
            "anchored to a file/line/commit, with their rationale and "
            "evidence chain (measurements, review findings). Provide a "
            "question, a repo-relative file path, or both."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural-language why-question.",
                },
                "file": {
                    "type": "string",
                    "description": "Repo-relative path (forward slashes).",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number inside `file` (boosts, never filters).",
                },
                "commit": {
                    "type": "string",
                    "description": "Commit hash prefix to boost.",
                },
                "repo": {
                    "type": "string",
                    "description": "Repo identifier; defaults to the configured repo.",
                },
                "limit": {"type": "integer", "default": 5},
            },
            "anyOf": [{"required": ["file"]}, {"required": ["question"]}],
        },
    },
    {
        "name": "kumiho_code_ingest",
        "description": (
            "Mine a git commit range into decision-memory nodes "
            "(LLM-structured, idempotent — already-captured commits are "
            "skipped without LLM cost). Omit rev_range for incremental mode."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "default": ".",
                    "description": "Path to the git repository.",
                },
                "rev_range": {
                    "type": "string",
                    "description": "e.g. HEAD~30..HEAD; omit = incremental.",
                },
                "max_commits": {"type": "integer", "default": 50},
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Re-capture commits that already carry markers.",
                },
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "kumiho_code_capture",
        "description": (
            "Store a code DECISION you just made or observed — the keyless, "
            "self-contained way to capture the *why* behind code (no LLM API "
            "key: YOU did the extraction, this only stores it, exactly like "
            "kumiho_memory_reflect). Use it right after you commit code that "
            "embodies a real choice — an alternative picked over another, a "
            "default/policy set, a reversal, or a measured trade-off — or "
            "when a decision is settled in conversation. Anchors are unioned "
            "with the commit's real changed files, so listing files is "
            "enough; hallucinated files are dropped. Defaults to HEAD."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "description": "The decisions to store (usually one).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string",
                                      "description": "The concrete choice (e.g. 'single-worker executor'), not a restatement of the diff."},
                            "decision": {"type": "string",
                                         "description": "What was decided, in one or two sentences."},
                            "rationale": {"type": "string",
                                          "description": "Why — the reasoning, constraint, or trade-off."},
                            "why_question": {"type": "string",
                                             "description": "The question a future reader would ask (e.g. 'why not asyncio.to_thread?')."},
                            "symbols": {"type": "array", "items": {"type": "string"},
                                        "description": "Identifiers involved (function/env/class names)."},
                            "files": {"type": "array", "items": {"type": "string"},
                                      "description": "Repo-relative files this decision defines/touches (forward slashes)."},
                            "evidence": {
                                "type": "array",
                                "description": "Verbatim support: measurements, review findings, rejected alternatives.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "kind": {"type": "string",
                                                 "enum": ["measurement", "review_finding", "incident", "benchmark", "constraint", "rejected_alternative"]},
                                        "text": {"type": "string", "description": "Verbatim quote carrying the WHY."},
                                    },
                                    "required": ["kind", "text"],
                                },
                            },
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["title", "decision"],
                    },
                },
                "repo_path": {"type": "string", "default": ".",
                              "description": "Path to the git repository."},
                "commit_ref": {"type": "string", "default": "HEAD",
                               "description": "The commit these decisions belong to (default HEAD — the one you just made)."},
            },
            "required": ["decisions"],
        },
    },
    {
        "name": "kumiho_code_mine_session",
        "description": (
            "Mine the current agent session into the code-decision graph: "
            "enrich commit-mined decisions with conversation-only "
            "alternatives/measurements, capture decisions that never "
            "reached a commit, and bridge decisions to the consolidated "
            "conversation. session_id (a HOST transcript id) is optional — "
            "omitted, it defaults to the current host session "
            "(KUMIHO_SESSION_ID)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The chat session to mine.",
                },
                "conversation_kref": {
                    "type": "string",
                    "description": (
                        "Consolidated revision kref for the bridge edge; "
                        "omit to skip bridging."
                    ),
                },
                "repo_path": {
                    "type": "string",
                    "default": ".",
                    "description": "Path to the git repository.",
                },
                "ingest_first": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Run an incremental commit ingest first so "
                        "enrichment targets exist (marker-skipped, no LLM "
                        "cost for already-captured commits)."
                    ),
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Re-mine a session that already carries a marker.",
                },
            },
            # session_id here names a HOST transcript to mine, so it defaults
            # from the host env only (never the active-session pointer, which
            # holds a working-memory bucket id, a different namespace).
        },
    },
]


def _register_code_memory_tools() -> None:
    """Gate-checked, idempotent registration.

    The gate is read once at import time — long-lived MCP servers restart to
    change it (same contract as the rest of the env-gated wiring).  The
    idempotency guard makes re-invocation (tests, module reloads) safe.
    """
    from kumiho_memory.code_decisions import code_memory_enabled

    if not code_memory_enabled():
        return
    if any(t["name"] == "kumiho_code_why" for t in MEMORY_TOOLS):
        return  # already registered
    MEMORY_TOOLS.extend(_CODE_MEMORY_TOOLS)
    MEMORY_TOOL_HANDLERS["kumiho_code_why"] = tool_code_why
    MEMORY_TOOL_HANDLERS["kumiho_code_ingest"] = tool_code_ingest
    MEMORY_TOOL_HANDLERS["kumiho_code_capture"] = tool_code_capture
    MEMORY_TOOL_HANDLERS["kumiho_code_mine_session"] = tool_code_mine_session


_register_code_memory_tools()


_ONTOLOGY_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "kumiho_memory_decompose",
        "description": (
            "Decompose a stored memory into the typed knowledge graph — KEYLESS "
            "(no LLM API key: YOU already read the conversation, so extract the "
            "structure yourself and pass it, exactly like kumiho_memory_reflect). "
            "Call it after a substantive exchange, using the `kref` returned by "
            "kumiho_memory_consolidate / kumiho_memory_reflect. Pass the entities "
            "(reusable named hubs), facts (claims, each ABOUT some entities), and "
            "entity->entity relations you distilled from the memory's SUMMARY — "
            "not the raw transcript. Keep it lean (a handful of each). Optionally "
            "declare belief changes you observed: `supersedes` (a new fact "
            "replaces a prior one) and `contradicts` (a fact conflicts with "
            "another) — each names a fact from THIS call and its target (a prior "
            "fact's statement or its kref); unresolvable targets are dropped. "
            "Optionally set `project` to materialize the typed nodes into a "
            "different project (defaults to the manager's configured project)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kref": {"type": "string",
                         "description": "The stored memory revision kref the typed nodes anchor to (from consolidate/reflect)."},
                "project": {"type": "string",
                            "description": "Optional target project for the materialized typed nodes; defaults to the manager's configured project. An absent/inaccessible project falls back to the configured one."},
                "entities": {
                    "type": "array",
                    "description": "Reusable named hubs (people, systems, files, concepts).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Canonical name (the entity hub identity)."},
                            "type": {"type": "string", "description": "Optional category (person, system, file, concept...)."},
                            "aliases": {"type": "array", "items": {"type": "string"},
                                        "description": "Other names that resolve to this same entity."},
                        },
                        "required": ["name"],
                    },
                },
                "facts": {
                    "type": "array",
                    "description": "Durable claims stated in the memory (each links ABOUT its entities).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string", "description": "The claim, one sentence."},
                            "about": {"type": "array", "items": {"type": "string"},
                                      "description": "Entity names this fact is about."},
                            "type": {"type": "string", "description": "Optional fact category."},
                            "valid_from": {"type": "string",
                                           "description": "Optional ISO date (YYYY / YYYY-MM / YYYY-MM-DD) the claim STARTED being true — for facts true over an interval."},
                            "valid_to": {"type": "string",
                                         "description": "Optional ISO date the claim STOPPED being true (omit if still true)."},
                        },
                        "required": ["statement"],
                    },
                },
                "relations": {
                    "type": "array",
                    "description": "Typed entity->entity links (subject predicate object).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "Source entity name."},
                            "predicate": {"type": "string", "description": "Relationship, e.g. 'depends on', 'owns', 'supersedes'."},
                            "object": {"type": "string", "description": "Target entity name."},
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                },
                "supersedes": {
                    "type": "array",
                    "description": "Belief updates: a new fact in this call replaces a prior fact.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string", "description": "A fact statement from THIS call (the new belief)."},
                            "replaces": {"type": "string", "description": "The prior fact's statement text OR its kref uri."},
                            "reason": {"type": "string", "description": "Optional why."},
                        },
                        "required": ["statement", "replaces"],
                    },
                },
                "contradicts": {
                    "type": "array",
                    "description": "Conflicts: a fact in this call contradicts another fact.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string", "description": "A fact statement from THIS call."},
                            "conflicts_with": {"type": "string", "description": "The conflicting fact's statement text OR its kref uri."},
                            "reason": {"type": "string", "description": "Optional why."},
                        },
                        "required": ["statement", "conflicts_with"],
                    },
                },
            },
            "required": ["kref"],
        },
    },
]


def _register_ontology_tools() -> None:
    """Register the keyless ontology-decomposition tool (idempotent).

    Ontology is on by default and the manager self-gates, so this is
    unconditional — the tool returns a clear 'enable ontology' error if off.
    """
    if any(t["name"] == "kumiho_memory_decompose" for t in MEMORY_TOOLS):
        return
    MEMORY_TOOLS.extend(_ONTOLOGY_TOOLS)
    MEMORY_TOOL_HANDLERS["kumiho_memory_decompose"] = tool_memory_decompose


_register_ontology_tools()
