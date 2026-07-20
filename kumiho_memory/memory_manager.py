"""Universal memory manager for AI agents."""

from __future__ import annotations

import asyncio
import inspect
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from kumiho_memory.evidence import EVIDENCE_LEVELS, evidence_tag
from kumiho_memory.grounding import apply_grounding_marker
from kumiho_memory.privacy import PIIRedactor
from kumiho_memory.valid_time import (
    apply_as_of_recall,
    apply_valid_interval_marker,
    as_of_recall_enabled,
)
from kumiho_memory.redis_memory import RedisMemoryBuffer
from kumiho_memory.failure_ledger import FailureLedger, content_key
from kumiho_memory.retry import RetryQueue, classify_failure, retry_with_backoff
from kumiho_memory.summarization import LLMAdapter, MemorySummarizer
from kumiho_memory.temporal_guard import classify_event_date, parse_timestamp

logger = logging.getLogger(__name__)


StoreCallable = Callable[..., Any]
RetrieveCallable = Callable[..., Any]

# Short backoff between the two attempts of a session-id Redis call. A single
# Redis flake must not silently fork a new session, so each call is retried
# once before falling back to a fresh sequence (see _generate_session_id).
_SESSION_REDIS_RETRY_BACKOFF = 0.05

# Payload fields that identify *what* is being stored, used to key the failure
# ledger (#118).  Volatile fields (artifact paths, session ids, timestamps) are
# excluded so the same content maps to the same key across runs.
_FAILURE_KEY_FIELDS = (
    "project",
    "memory_type",
    "title",
    "summary",
    "user_text",
    "assistant_text",
)


def _payload_failure_key(payload: Dict[str, Any]) -> str:
    """Derive a stable failure-ledger key for a ``memory_store`` payload.

    Keys on the content-identifying fields so a poison payload maps to the
    same ledger entry each run.  Falls back to a canonical hash of the whole
    payload when none of those fields are populated.
    """
    parts = [
        f"{field_name}={payload[field_name]}"
        for field_name in _FAILURE_KEY_FIELDS
        if payload.get(field_name)
    ]
    if not parts:
        parts.append(json.dumps(payload, sort_keys=True, default=str))
    return content_key(*parts)


@dataclass
class MemoryAssessResult:
    """Result returned by an ``auto_assess_fn`` implementation.

    The callable receives the recent working-memory messages and a list of
    recalled long-term memories, then returns this object so ``MemoryManager``
    can decide whether to persist anything — without depending on a specific
    LLM provider.
    """

    should_store: bool
    """Whether the excerpt contains something new worth persisting."""

    content: str = ""
    """Extracted fact/decision/preference text to store (used as summary)."""

    memory_type: str = "fact"
    """Memory type: ``"fact"``, ``"decision"``, ``"preference"``, or ``"summary"``."""

    reason: str = ""
    """Short explanation (logged for debugging, stored as assistant_text)."""

    tags: List[str] = field(default_factory=list)
    """Extra tags to attach (``"auto-memorized"`` is always appended)."""

    evidence_level: str = ""
    """Optional evidence grade (see :mod:`kumiho_memory.evidence`).

    When set, the manager stamps it as revision metadata plus the mirrored
    ``evidence:<level>`` tag.  Assessors must never emit ``official`` —
    that grade is reserved for explicit operator/ingest flags.
    """

    source: str = ""
    """Optional source identifier for the claim (e.g. ``"news:reuters"``)."""

    supporting_krefs: List[str] = field(default_factory=list)
    """Revision krefs of corroborating memories — the manager creates
    ``SUPPORTS`` edges from the new memory to each after storing."""

    conflicting_krefs: List[str] = field(default_factory=list)
    """Revision krefs of contradicted memories — recorded in metadata as
    ``conflicts_with`` so the disagreement stays visible (the contradicted
    belief itself is never revised at write time)."""

    create_contradicts_edges: bool = True
    """Gate for the ``CONTRADICTS`` edge bridge (threaded from
    ``EvidencePolicy.create_contradicts_edges``; default ON — it is the
    feature).  ``False`` skips the edge bridge only; the ``conflicts_with``
    metadata above is written regardless."""


# Callable protocol: async (messages, recalled_memories) → MemoryAssessResult.
# ``messages`` = recent working-memory dicts (role/content/timestamp).
# ``recalled_memories`` = top-K long-term memory dicts from the graph.
AutoAssessFn = Callable[
    [List[Dict[str, Any]], List[Dict[str, Any]]],
    Awaitable[MemoryAssessResult],
]

# Stopwords to ignore when computing token-overlap relevance scores.
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into about between through after before above below "
    "and or but not no nor so yet both either neither each every all "
    "some any few more most other such than too very also just only "
    "that this these those it its i me my we our you your he him his "
    "she her they them their what which who whom how when where why "
    "if then else while during until again further once here there "
    "up down out off over under re same own".split()
)

# Max total characters of sibling summary text per item (fallback mode).
# ~20K chars ≈ 5K tokens.
_SIBLING_CHAR_BUDGET = 20_000

# If the best keyword-overlap score among siblings exceeds this threshold,
# use keyword-filtered mode (only return strong matches).  Below this,
# fall back to char-budget mode which keeps all siblings that fit.
_SIBLING_STRONG_SCORE = 0.40

# Per-recall cap on sibling-rerank LLM calls (#102). Sibling enrichment issues
# one LLM round-trip per stacked item; unbounded, N stacked items = N calls
# added to answer latency. Beyond the cap, items use the existing deterministic
# in-process fallback instead of the LLM. This is a SAFETY VALVE against
# pathological recalls that stack many items — NOT a relevance-tuning knob. The
# default sits well above any typical/benchmark recall (default recall limit is
# 5; ~10 stacked items is a generous ceiling), so results are byte-identical
# there. <= 0 disables the cap (unlimited), matching the sibling_top_k=0 idiom.
_SIBLING_LLM_CAP = 16
# Bounded concurrency for the capped set: the LLM calls are independent per
# item, so they run concurrently under this semaphore instead of serially.
_SIBLING_LLM_CONCURRENCY = 4

# A stored event_date must be a clean ISO-8601 calendar date (YYYY, YYYY-MM, or
# YYYY-MM-DD). Guards against the summarizer emitting prose ("last week") into the
# structured event_date field despite the prompt asking for normalized ISO.
_ISO_EVENT_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def _tokenize(text: str) -> List[str]:
    """Lowercase split + strip punctuation, filtering stopwords."""
    return [
        tok for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    ]


def _token_overlap_score(query_tokens: List[str], text: str) -> float:
    """BM25-light relevance score between query tokens and a text string.

    Uses TF-IDF-inspired weighting: tokens that appear in the text get a
    score proportional to their frequency, dampened by log to avoid
    over-counting repeated terms.  Returns 0-1 range.
    """
    if not query_tokens or not text:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    text_counts = Counter(text_tokens)
    score = 0.0
    for qt in query_tokens:
        tf = text_counts.get(qt, 0)
        if tf > 0:
            # Dampened term frequency (log(1+tf)) normalized
            score += math.log(1 + tf)
    # Normalize by query length to get 0-1ish range
    return score / (len(query_tokens) + 1)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two float vectors (pure-python fallback)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class UniversalMemoryManager:
    """Orchestrates working memory, summarization, and long-term storage.

    For agent frameworks (OpenClaw, LangChain, etc.) that already have a
    configured LLM, pass the ``llm_adapter`` parameter to reuse it::

        manager = UniversalMemoryManager(llm_adapter=my_agent_adapter)

    This avoids separate LLM configuration for the memory subsystem.
    """

    def __init__(
        self,
        *,
        project: str = "CognitiveMemory",
        consolidation_threshold: int = 50,
        artifact_root: Optional[str] = None,
        llm_adapter: Optional[LLMAdapter] = None,
        redis_buffer: Optional[RedisMemoryBuffer] = None,
        summarizer: Optional[MemorySummarizer] = None,
        pii_redactor: Optional[PIIRedactor] = None,
        memory_store: Optional[StoreCallable] = None,
        memory_retrieve: Optional[RetrieveCallable] = None,
        redis_url: Optional[str] = None,
        tenant_hint: Optional[str] = None,
        retry_queue: Optional[RetryQueue] = None,
        failure_ledger: Optional[FailureLedger] = None,
        store_max_retries: int = 3,
        graph_augmentation: Optional[Any] = None,
        entity_promotion: Optional[Any] = True,
        recall_mode: str = "summarized",
        sibling_strong_score: float = _SIBLING_STRONG_SCORE,
        sibling_char_budget: int = _SIBLING_CHAR_BUDGET,
        sibling_similarity_threshold: float = 0.0,
        sibling_top_k: int = 0,
        sibling_llm_cap: int = _SIBLING_LLM_CAP,
        embedding_adapter: Optional[Any] = None,
        sibling_score_fields: Optional[List[str]] = None,
        auto_assess_fn: Optional[AutoAssessFn] = None,
        auto_assess_min_messages: int = 3,
        auto_assess_window: int = 6,
        auto_assess_timeout: float = 120.0,
        evidence_rank: Optional[Any] = None,
        rerank: Optional[Any] = None,
        reranker: Optional[Any] = None,
        recall_candidate_multiplier: float = 1.0,
    ) -> None:
        self.project = project
        self.consolidation_threshold = consolidation_threshold
        self.artifact_root = artifact_root or os.getenv(
            "KUMIHO_MEMORY_ARTIFACT_ROOT",
            os.path.join(os.path.expanduser("~"), ".kumiho", "artifacts"),
        )

        self.redis_buffer = redis_buffer or RedisMemoryBuffer(
            redis_url=redis_url,
            tenant_hint=tenant_hint,
        )
        if summarizer is not None:
            self.summarizer = summarizer
        elif llm_adapter is not None:
            self.summarizer = MemorySummarizer(adapter=llm_adapter)
        else:
            self.summarizer = MemorySummarizer()
        self.pii_redactor = pii_redactor or PIIRedactor()

        self.memory_store = memory_store if memory_store is not None else _load_default_store()
        self.memory_retrieve = (
            memory_retrieve if memory_retrieve is not None else _load_default_retrieve()
        )
        self.retry_queue = retry_queue
        # Cross-run failure ledger (issue #118): parks content that fails
        # deterministically so it is not re-stored run after run.  Opt-in at
        # the library layer (None = today's behavior); the MCP/CLI entrypoints
        # wire a default ledger so parking is active in production.
        self.failure_ledger = failure_ledger
        self.store_max_retries = store_max_retries
        # Graph-augmented recall: pass a GraphAugmentationConfig for full
        # control, or simply ``True`` for the default config — no boilerplate.
        # Falsy values (None/False/0) read naturally as "disabled".
        if graph_augmentation is True:
            from kumiho_memory.graph_augmentation import GraphAugmentationConfig
            graph_augmentation = GraphAugmentationConfig()
        elif not graph_augmentation:
            graph_augmentation = None
        self.graph_augmentation_config = graph_augmentation

        # Ontology (write-time typed decomposition + structure-aware recall)
        # is ON by default — opt OUT with KUMIHO_MEMORY_ONTOLOGY=0. The flip
        # from opt-in was decided 2026-07-10 on paired same-corpus evidence:
        # the ontology read stack contributes +0.042 overall and the
        # fact-recall leg +0.054 (all five LoCoMo categories up, 23W/4L),
        # with the write side measured byte-identical on the base summary.
        # The switch still controls BOTH the write (decomposition) and the
        # read (entity/fact recall) together, so the graph is only built
        # when something reads it and vice versa.
        ontology_on = os.getenv("KUMIHO_MEMORY_ONTOLOGY", "1").strip() != "0"

        # entity_promotion: the True default sentinel follows the ontology
        # switch; KUMIHO_MEMORY_ENTITY_PROMOTION=1/0 forces it on/off
        # independently; an explicit config/False always overrides.
        ep_env = os.getenv("KUMIHO_MEMORY_ENTITY_PROMOTION", "").strip()
        if ep_env == "0":
            entity_promotion = None
        elif entity_promotion is True:
            from kumiho_memory.entity_promotion import EntityPromotionConfig
            entity_promotion = (
                EntityPromotionConfig() if (ontology_on or ep_env == "1") else None
            )
        elif not entity_promotion:
            entity_promotion = None
        self.entity_promotion_config = entity_promotion

        # Light up the entity-mediated reader when ontology is on (only
        # meaningful when graph augmentation itself is active).
        if self.graph_augmentation_config is not None and ontology_on:
            self.graph_augmentation_config.entity_recall = True
            # Fact-recall leg rides the same switch (facts are the ontology's
            # payload); KUMIHO_MEMORY_FACT_RECALL=0 is the measurement
            # kill-switch for A/B isolation on top of ontology-on.
            if os.getenv("KUMIHO_MEMORY_FACT_RECALL", "").strip() != "0":
                self.graph_augmentation_config.fact_recall = True

        # Registered entity->entity relation-edge traversal (ontology G1 read
        # side). DEFAULT OFF — opt IN with KUMIHO_MEMORY_RELATION_TRAVERSAL=1;
        # it adds get_edges round-trips per anchor (bounded by the
        # relation_traversal_* caps) and awaits pair-measured benchmarks. It
        # extends the entity-mediated reader, so it only fires when
        # entity_recall is also on (ontology default).
        if (
            self.graph_augmentation_config is not None
            and os.getenv("KUMIHO_MEMORY_RELATION_TRAVERSAL", "").strip() == "1"
        ):
            self.graph_augmentation_config.relation_traversal = True

        # As-of recall (ontology G8). DEFAULT OFF — opt IN with
        # KUMIHO_MEMORY_AS_OF_RECALL=1. When on AND the caller passes a
        # query_time, recall demotes (never deletes) facts whose valid-time
        # interval excludes that instant. Read once here (env convention); the
        # flag-OFF path never touches the reranked list, so recall is
        # byte-identical by default. Awaits the same pair-measured gate as the
        # other Phase 3 flips.
        self.as_of_recall_enabled = as_of_recall_enabled()

        # Multi-draw reformulation override (angle-union harvesting for
        # oblique triggers). Applies whenever graph augmentation is active.
        draws_env = os.getenv("KUMIHO_MEMORY_REFORMULATE_DRAWS", "").strip()
        if draws_env.isdigit() and self.graph_augmentation_config is not None:
            draws = max(1, int(draws_env))
            self.graph_augmentation_config.reformulate_draws = draws
            self.graph_augmentation_config.reformulate_max_angles = max(
                self.graph_augmentation_config.reformulate_max_angles,
                2 * draws + 1,
            )

        # When ontology is on, consolidation decomposes the whole conversation
        # into a typed graph (entities + facts + decisions + events + ...),
        # which subsumes plain entity promotion.
        self.ontology_enabled = ontology_on
        self._graph_recall: Optional[Any] = None  # lazy GraphAugmentedRecall
        # Backend-error signal for recall: set by _lightweight_recall when the
        # retrieve callable reports a backend failure, reset at the start of
        # each recall_memories call. The internal recall still returns [] (the
        # established contract); callers (MCP tools) read this AFTER recall to
        # distinguish "no memories" from "backend down" without a type change.
        self._last_backend_error: Optional[str] = None
        self.recall_mode = recall_mode
        self.sibling_strong_score = sibling_strong_score
        self.sibling_char_budget = sibling_char_budget
        self.sibling_similarity_threshold = sibling_similarity_threshold
        self.sibling_top_k = sibling_top_k
        # Per-recall sibling-rerank LLM-call cap (#102). Env override for the
        # safety valve; <= 0 means unlimited. Falls back to the constructor arg
        # (default _SIBLING_LLM_CAP) when the env var is unset or non-integer.
        cap_env = os.getenv("KUMIHO_MEMORY_SIBLING_LLM_CAP", "").strip()
        try:
            self.sibling_llm_cap = int(cap_env) if cap_env else sibling_llm_cap
        except ValueError:
            self.sibling_llm_cap = sibling_llm_cap
        self.embedding_adapter = embedding_adapter
        self.sibling_score_fields = sibling_score_fields
        # Evidence-weighted recall reranking (deterministic, default on;
        # strict no-op when no memory carries an evidence grade).  Falsy
        # non-None values (False/0) read naturally as "disable" — honor
        # that instead of crashing on attribute access.
        if evidence_rank is None:
            from kumiho_memory.evidence_rank import EvidenceRankConfig
            evidence_rank = EvidenceRankConfig()
        elif not evidence_rank:
            from kumiho_memory.evidence_rank import EvidenceRankConfig
            evidence_rank = EvidenceRankConfig(enabled=False, badges=False)
        self.evidence_rank_config = evidence_rank
        # Post-recall reranking: recency decay + MMR diversity (deterministic,
        # default on, conservative) and an optional cross-encoder relevance
        # stage.  None -> defaults; falsy -> everything disabled.  Subsumes
        # evidence weighting on the plain recall path (applied once).
        from kumiho_memory.recall_rerank import (
            RerankConfig,
            resolve_reranker_from_env,
        )
        if rerank is None and reranker is None:
            # No explicit rerank config AND no explicit reranker: honor the
            # KUMIHO_RERANK_* / KUMIHO_RECALL_RERANK env conventions so direct
            # construction (SDK users, benchmark harnesses) gets the same
            # reranker wiring the MCP server does.  With NO env vars set this
            # is exactly RerankConfig() + reranker=None — behavior unchanged.
            rerank = RerankConfig.from_env()
            # Lazy factory: the summarizer's ``adapter`` property builds (and
            # may fail to build, e.g. no API key) a real LLM client — only
            # touch it if KUMIHO_RERANK_LLM actually requests the LLM path.
            reranker = resolve_reranker_from_env(
                adapter_factory=lambda: getattr(self.summarizer, "adapter", None),
                model=getattr(self.summarizer, "light_model", "") or "",
            )
            if reranker is not None:
                rerank.cross_encoder_enabled = True
        elif rerank is None:
            rerank = RerankConfig()
        elif not rerank:
            rerank = RerankConfig(
                recency_enabled=False, mmr_enabled=False, cross_encoder_enabled=False
            )
        self.rerank_config = rerank
        self.reranker = reranker
        # Retrieve-wide-then-trim: when > 1.0, recall over-fetches candidates
        # (ceil(limit * multiplier)), runs the full rerank stack on the wide
        # set, then trims back to the caller's limit — lifting gold-in-context
        # without enlarging the returned result.  1.0 == current behavior.
        try:
            self.recall_candidate_multiplier = max(1.0, float(recall_candidate_multiplier))
        except (TypeError, ValueError):
            self.recall_candidate_multiplier = 1.0
        # Background memory assessor (model-agnostic, optional)
        self.auto_assess_fn: Optional[AutoAssessFn] = auto_assess_fn
        self.auto_assess_min_messages = auto_assess_min_messages
        self.auto_assess_window = auto_assess_window
        # Outer safety cap on a single background assess (LLM + store + bounded
        # edge writes); a hung assessor can't leak a daemon worker forever.
        self.auto_assess_timeout = auto_assess_timeout
        # In-process cursor: message count at last auto-store per session.
        # Resets on process restart (safe — worst case one extra LLM call).
        self._auto_store_cursors: Dict[str, int] = {}

    async def ingest_message(
        self,
        *,
        user_id: str,
        message: str,
        role: str = "user",
        channel: str = "unknown",
        context: str = "personal",
        session_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        evidence_level: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        if evidence_level:
            evidence_tag(evidence_level)  # validate early — raises ValueError
        resolved_session_id = session_id or await self._generate_session_id(user_id, context)

        metadata: Dict[str, Any] = {
            "channel": channel,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if attachments:
            artifact_pointers: List[Dict[str, Any]] = []
            for attachment in attachments:
                pointer = self._store_attachment(attachment, context=context)
                artifact_pointers.append(pointer)
            metadata["attachments"] = artifact_pointers

        result = await self.redis_buffer.add_message(
            project=self.project,
            session_id=resolved_session_id,
            role=role,
            content=message,
            metadata=metadata,
        )

        # Persist user_id and context as session metadata so that
        # consolidate_session can derive the storage space automatically.
        # Evidence grading (evidence_level/source) is stashed the same way
        # and applied to the stored revision at consolidation time — no
        # revision exists to tag during ingest.
        session_meta: Dict[str, str] = {}
        is_first_message = result.get("message_count", 0) == 1
        if is_first_message:
            session_meta.update({"user_id": user_id, "context": context})
        if evidence_level:
            session_meta["evidence_level"] = evidence_level
        if source:
            session_meta["source"] = source
        if session_meta:
            try:
                if not is_first_message:
                    # Partial update (evidence on a later message): merge
                    # with the stored metadata so backends that replace
                    # rather than merge (proxy mode) keep user_id/context.
                    existing = await self.redis_buffer.get_session_metadata(
                        self.project, resolved_session_id,
                    ) or {}
                    session_meta = {**existing, **session_meta}
                await self.redis_buffer.set_session_metadata(
                    self.project,
                    resolved_session_id,
                    session_meta,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to set session metadata for %s: %s — "
                    "consolidation space derivation may fall back to topic hint",
                    resolved_session_id, exc,
                )

        return {
            "success": True,
            "session_id": resolved_session_id,
            "message_count": result["message_count"],
            "attachments": metadata.get("attachments", []),
        }

    async def add_assistant_response(
        self,
        *,
        session_id: str,
        response: str,
        channel: str = "unknown",
    ) -> Dict[str, Any]:
        result = await self.redis_buffer.add_message(
            project=self.project,
            session_id=session_id,
            role="assistant",
            content=response,
            metadata={
                "channel": channel,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        # Fire background memory assessment — non-blocking, only if configured.
        # NOT asyncio.create_task: the MCP runtime dispatches this via
        # asyncio.run (mcp_tools.tool_memory_add_response / _reflect), a
        # one-shot loop that cancels pending tasks on teardown — a detached task
        # is killed before it stores, so evidence grading AND the CONTRADICTS
        # bridge (issue #94) would silently never run for MCP users, the primary
        # deployment (issue #104). A daemon thread with its own loop survives the
        # handler's teardown (same daemon-survival mechanism as
        # run_bounded_in_thread, which decompose/consolidation already rely on)
        # and runs to completion without blocking the tool response; bounded so a
        # hung assessor can't leak a worker. On a genuine long-lived loop the
        # thread still completes independently — reliability, not a behaviour
        # change.
        if self.auto_assess_fn is not None:
            from kumiho_memory._bounded import run_coro_in_daemon_thread

            run_coro_in_daemon_thread(
                lambda: self._background_assess(session_id),
                timeout=self.auto_assess_timeout,
                label=f"auto-assess ({session_id})",
            )
        return {
            "success": True,
            "message_count": result["message_count"],
        }

    async def _background_assess(self, session_id: str) -> None:
        """Background task: decide if recent buffer content is worth storing.

        Fires after each assistant turn via ``asyncio.create_task``.  The
        registered ``auto_assess_fn`` receives the recent working-memory
        window and the top-K recalled long-term memories, then returns a
        :class:`MemoryAssessResult` with a novelty judgment.  When
        ``should_store`` is ``True`` the extracted content is persisted and
        linked to the recalled graph context automatically.

        The cooldown (``auto_assess_min_messages``) prevents redundant LLM
        calls when several responses arrive in quick succession.
        """
        if self.auto_assess_fn is None:
            return
        try:
            # 1. Fetch messages + session metadata in parallel.
            messages_result, session_meta = await asyncio.gather(
                self.redis_buffer.get_messages(
                    project=self.project,
                    session_id=session_id,
                    limit=self.auto_assess_window,
                ),
                self.redis_buffer.get_session_metadata(self.project, session_id),
            )
            messages: List[Dict[str, Any]] = messages_result.get("messages", [])
            total: int = int(messages_result.get("message_count", 0))

            # Derive user space so auto-memorized items land beside the session's
            # other memories — Dream State finds them via graph neighbourhood.
            session_user_id: Optional[str] = session_meta.get("user_id") if session_meta else None
            session_context: Optional[str] = session_meta.get("context") if session_meta else None
            space_path: Optional[str] = None
            if session_user_id:
                space_path = (
                    f"{session_context}/{session_user_id}"
                    if session_context
                    else session_user_id
                )

            # 2. Cooldown: skip if fewer than min_messages since last assess.
            last_cursor = self._auto_store_cursors.get(session_id, 0)
            if total - last_cursor < self.auto_assess_min_messages:
                return

            # 3. Build query from last 3 messages and recall graph context.
            query_parts = [m.get("content", "")[-300:] for m in messages[-3:] if m.get("content")]
            recalled: List[Dict[str, Any]] = []
            if query_parts and self.memory_retrieve:
                query_text = " ".join(query_parts)
                recalled = await self.recall_memories(query_text, limit=5)

            # 4. Invoke the model-agnostic assess function.
            assess_result: MemoryAssessResult = await self.auto_assess_fn(messages, recalled)

            # Always advance cursor so we don't re-examine the same window.
            self._auto_store_cursors[session_id] = total

            if not assess_result.should_store or not assess_result.content.strip():
                return

            # 5. Store in the user's space so Dream State can find and enrich it.
            source_krefs = [r["kref"] for r in recalled if "kref" in r]
            store_payload: Dict[str, Any] = {
                "project": self.project,
                "memory_type": assess_result.memory_type,
                "title": assess_result.content[:80],
                "summary": assess_result.content,
                "user_text": assess_result.content,
                "assistant_text": assess_result.reason,
                "tags": ["auto-memorized"] + assess_result.tags,
                "metadata": {
                    "memory_type": assess_result.memory_type,
                    "auto_memorized": "true",
                    "session_id": session_id,
                },
            }
            if space_path:
                store_payload["space_path"] = space_path
            if session_user_id:
                store_payload["metadata"]["user_id"] = session_user_id
            if source_krefs:
                store_payload["source_revision_krefs"] = source_krefs

            # Evidence grade from the assessor (never "official" — that
            # grade is operator-only; sanitized rather than raised because
            # LLM output must not crash the background task).
            if assess_result.evidence_level:
                level = assess_result.evidence_level
                if level in EVIDENCE_LEVELS and level != "official":
                    store_payload["metadata"]["evidence_level"] = level
                    store_payload["tags"].append(evidence_tag(level))
                else:
                    logger.warning(
                        "auto_assess: ignoring assessor evidence_level %r "
                        "(unknown or operator-only)", level,
                    )
            if assess_result.source:
                store_payload["metadata"]["source"] = assess_result.source
            if assess_result.conflicting_krefs:
                store_payload["metadata"]["conflicts_with"] = ",".join(
                    assess_result.conflicting_krefs
                )

            store_result = await self._store_with_retry(**store_payload)
            logger.debug(
                "auto_assess stored memory for session %s: %s",
                session_id,
                assess_result.content[:80],
            )

            # SUPPORTS edges to corroborating memories.  Requires the new
            # revision kref — skipped silently when the store was queued
            # for retry (no kref exists yet; replay has no edge mechanism).
            new_kref = (store_result or {}).get("revision_kref", "")
            if new_kref and assess_result.supporting_krefs:
                await self._create_support_edges(
                    new_kref, assess_result.supporting_krefs,
                )

            # CONTRADICTS edges bridge the assessor's conflict verdicts into
            # the graph (the ``conflicts_with`` metadata above is untouched —
            # this is purely additive), so recall can surface "this fact is
            # contested" instead of returning one side unmarked. Gated by the
            # policy kill-switch (create_contradicts_edges, default ON),
            # mirroring the SUPPORTS gate.
            if (
                new_kref
                and assess_result.conflicting_krefs
                and getattr(assess_result, "create_contradicts_edges", True)
            ):
                await self._create_contradicts_edges(
                    new_kref, assess_result.conflicting_krefs,
                )
        except Exception as exc:  # pragma: no cover
            logger.debug("_background_assess error for session %s: %s", session_id, exc)

    async def _create_support_edges(
        self,
        revision_kref: str,
        supporting_krefs: List[str],
        timeout: float = 60.0,
    ) -> int:
        """Create ``SUPPORTS`` edges from a new memory to its corroborators.

        Best-effort: each edge failure is logged at debug level and skipped —
        evidence chains are an enrichment, never a store blocker.  Returns
        the number of edges created.

        Runs the synchronous gRPC calls in a bounded daemon thread (see
        ``_bounded.run_bounded_in_thread``): a hung RPC must not strand a
        shared executor thread.
        """
        from kumiho_memory._bounded import run_bounded_in_thread

        def _sync_create() -> int:
            import kumiho

            source_rev = kumiho.get_revision(revision_kref)
            created = 0
            for target_kref in supporting_krefs:
                try:
                    target_rev = kumiho.get_revision(target_kref)
                    source_rev.create_edge(
                        target_rev,
                        "SUPPORTS",
                        metadata={"reason": "evidence corroboration"},
                    )
                    created += 1
                except Exception as exc:
                    logger.debug(
                        "SUPPORTS edge %s -> %s failed: %s",
                        revision_kref, target_kref, exc,
                    )
            return created

        created = await run_bounded_in_thread(
            _sync_create,
            timeout=timeout,
            label=f"SUPPORTS edges ({revision_kref})",
            on_timeout=0,
            on_error=0,
        ) or 0
        if created:
            logger.debug(
                "Created %d SUPPORTS edge(s) from %s", created, revision_kref,
            )
        return created

    async def _create_contradicts_edges(
        self,
        revision_kref: str,
        conflicting_krefs: List[str],
        timeout: float = 60.0,
    ) -> int:
        """Bridge assessor conflicts into ``CONTRADICTS`` graph edges.

        The conflict-side twin of :meth:`_create_support_edges`: one directed
        ``CONTRADICTS`` edge (new revision -> each conflicting revision) with
        ``basis: evidence-assessor``, turning the assessor's ``conflicts_with``
        metadata into graph structure the recall reader can traverse and
        surface as a contested marker.

        Best-effort (each failure logged at debug level and skipped — conflict
        edges are enrichment, never a store blocker) and idempotent: a target
        already linked by ``CONTRADICTS`` is skipped, mirroring
        ``ontology._Materializer.edge``'s precheck so a re-assess of the same
        window can't duplicate.  Runs the synchronous gRPC calls in a bounded
        daemon thread.  Returns the number of edges created.
        """
        from kumiho_memory._bounded import run_bounded_in_thread

        def _sync_create() -> int:
            import kumiho

            source_rev = kumiho.get_revision(revision_kref)
            created = 0
            for target_kref in conflicting_krefs:
                try:
                    # Idempotency precheck (server-side dedupe is NOT assumed):
                    # skip a target already linked by CONTRADICTS. Best-effort —
                    # if the edge read is unsupported/transient, fall through
                    # and create rather than silently dropping the edge.
                    already = False
                    try:
                        for existing in source_rev.get_edges(
                            edge_type_filter="CONTRADICTS", direction=0,
                        ):
                            if getattr(
                                getattr(existing, "target_kref", None), "uri", "",
                            ) == target_kref:
                                already = True
                                break
                    except Exception:  # noqa: BLE001 — no dedup available; create
                        pass
                    if already:
                        continue
                    target_rev = kumiho.get_revision(target_kref)
                    source_rev.create_edge(
                        target_rev,
                        "CONTRADICTS",
                        metadata={"basis": "evidence-assessor"},
                    )
                    created += 1
                except Exception as exc:
                    logger.debug(
                        "CONTRADICTS edge %s -> %s failed: %s",
                        revision_kref, target_kref, exc,
                    )
            return created

        created = await run_bounded_in_thread(
            _sync_create,
            timeout=timeout,
            label=f"CONTRADICTS edges ({revision_kref})",
            on_timeout=0,
            on_error=0,
        ) or 0
        if created:
            logger.debug(
                "Created %d CONTRADICTS edge(s) from %s", created, revision_kref,
            )
        return created

    async def handle_user_message(
        self,
        *,
        user_id: str,
        message: str,
        channel: str = "unknown",
        context: str = "personal",
        session_id: Optional[str] = None,
        working_memory_limit: int = 10,
        recall_limit: int = 5,
        evidence_level: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        ingest_result = await self.ingest_message(
            user_id=user_id,
            message=message,
            role="user",
            channel=channel,
            context=context,
            session_id=session_id,
            evidence_level=evidence_level,
            source=source,
        )
        session_id = ingest_result["session_id"]

        working_memory_result = await self.redis_buffer.get_messages(
            project=self.project,
            session_id=session_id,
            limit=working_memory_limit,
        )

        long_term_memory = await self.recall_memories(message, limit=recall_limit)

        should_consolidate = (
            working_memory_result["message_count"] >= self.consolidation_threshold
        )

        return {
            "session_id": session_id,
            "working_memory": working_memory_result["messages"],
            "long_term_memory": long_term_memory,
            "should_consolidate": should_consolidate,
        }

    async def consolidate_session(
        self,
        *,
        session_id: str,
        space_path: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[str] = None,
        stack_revisions: Optional[bool] = None,
        evidence_level: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Empty string behaves like None (consistent with ingest_message) —
        # otherwise "" would silently cancel an ingest-stashed grade while
        # bypassing both validation and the session-metadata fallback.
        evidence_level = evidence_level or None
        source = source or None
        if evidence_level:
            evidence_tag(evidence_level)  # validate early — raises ValueError
        messages_result = await self.redis_buffer.get_messages(
            project=self.project,
            session_id=session_id,
            limit=1000,
        )
        messages = messages_result["messages"]

        if not messages:
            return {"success": False, "error": "No messages to consolidate"}

        # Resolve storage space.  Priority:
        # 1. Explicit space_path (caller override)
        # 2. user_id + context (caller-provided identity scoping)
        # 3. Session metadata in Redis (auto-stored during ingest)
        # 4. Topic-derived hint (backwards-compatible default)
        resolved_space: Optional[str] = space_path
        session_user_id: Optional[str] = user_id
        session_context: Optional[str] = context  # may be overridden from metadata below
        session_meta: Dict[str, str] = {}
        session_meta_fetched = False
        if not resolved_space and session_user_id:
            resolved_space = (
                f"{context}/{session_user_id}" if context else session_user_id
            )
        if not resolved_space:
            session_meta_fetched = True
            try:
                session_meta = await self.redis_buffer.get_session_metadata(
                    self.project, session_id,
                ) or {}
                session_user_id = session_meta.get("user_id")
                session_context = session_meta.get("context", "") or session_context
                if session_user_id:
                    resolved_space = (
                        f"{session_context}/{session_user_id}"
                        if session_context
                        else session_user_id
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load session metadata for %s: %s — "
                    "falling back to topic-based hint",
                    session_id, exc,
                )

        # Resolve evidence grade.  Explicit args take precedence over
        # session metadata stashed at ingest time; values from Redis are
        # sanitized rather than raised so bad data never blocks
        # consolidation.  The fetch is skipped when the space-resolution
        # block above already loaded the metadata (even if empty) — one
        # roundtrip per consolidation at most.
        resolved_evidence: Optional[str] = evidence_level
        resolved_source: Optional[str] = source
        if (resolved_evidence is None or resolved_source is None) and not session_meta_fetched:
            try:
                session_meta = await self.redis_buffer.get_session_metadata(
                    self.project, session_id,
                ) or {}
            except Exception as exc:
                logger.debug(
                    "Failed to load session evidence metadata for %s: %s",
                    session_id, exc,
                )
        if resolved_evidence is None:
            resolved_evidence = session_meta.get("evidence_level") or None
        if resolved_source is None:
            resolved_source = session_meta.get("source") or None
        if resolved_evidence and resolved_evidence not in EVIDENCE_LEVELS:
            logger.warning(
                "Ignoring unknown evidence_level %r for session %s",
                resolved_evidence, session_id,
            )
            resolved_evidence = None

        # Run summarization (full model) and implications (light model)
        # in parallel — implications don't depend on the summary result.
        # Use return_exceptions so an implication failure doesn't crash
        # the summarizer (implications are non-critical).
        _summary_or_exc, _impl_or_exc = await asyncio.gather(
            self.summarizer.summarize_conversation(messages),
            self.summarizer.generate_implications(messages),
            return_exceptions=True,
        )
        # Summary is critical — propagate its exception.
        if isinstance(_summary_or_exc, BaseException):
            raise _summary_or_exc
        summary_result = _summary_or_exc
        summarization_error = str(summary_result.get("error", "") or "").strip()
        if summarization_error:
            debug_info = summary_result.get("debug", {})
            logger.warning(
                "summarize_conversation failed for %s: %s",
                session_id,
                summarization_error,
            )
            if isinstance(debug_info, dict) and debug_info:
                logger.warning(
                    "summarize_conversation diagnostics for %s: provider=%s model=%s base_url=%s json_mode=%s raw_len=%s raw_preview=%r",
                    session_id,
                    debug_info.get("provider", ""),
                    debug_info.get("model", ""),
                    debug_info.get("base_url", ""),
                    debug_info.get("json_mode", ""),
                    debug_info.get("raw_response_len", 0),
                    debug_info.get("raw_response_preview", ""),
                )
            return {
                "success": False,
                "error": f"Conversation summarization failed: {summarization_error}",
            }
        # Implications are best-effort — fall back to empty list.
        if isinstance(_impl_or_exc, BaseException):
            logger.warning("generate_implications failed: %s", _impl_or_exc)
            implications: list = []
        else:
            implications = _impl_or_exc
        redacted_summary = self.pii_redactor.anonymize_summary(summary_result.get("summary", ""))

        # Append extracted events to the summary text so they are
        # vector-indexed and visible during recall.  The narrative summary
        # captures the high-level arc; events preserve granular incidents
        # (e.g. "phone battery died mid-call → replaced battery") that
        # narrative compression would otherwise drop.
        events = summary_result.get("events", [])
        if events:
            event_lines: List[str] = []
            for ev in events:
                desc = ev.get("event", "")
                when = ev.get("when", "")
                consequence = ev.get("consequence", "")
                if desc:
                    prefix = f"- [{when}] " if when and when.lower() != "unknown" else "- "
                    if consequence:
                        event_lines.append(f"{prefix}{desc} \u2192 {consequence}")
                    else:
                        event_lines.append(f"{prefix}{desc}")
            if event_lines:
                redacted_summary += "\n\nKey events:\n" + "\n".join(event_lines)
                redacted_summary = self.pii_redactor.anonymize_summary(
                    redacted_summary
                )

        # Append knowledge.facts — concrete factual claims extracted from
        # the conversation.  Without this, the stored summary text lacks
        # specific details (names, possessions, places, roles) that are
        # critical for single-hop and multi-hop factual QA.
        knowledge = summary_result.get("knowledge", {})
        facts = knowledge.get("facts", [])
        if facts:
            fact_lines: List[str] = []
            for fact in facts:
                claim = fact.get("claim", "")
                if claim:
                    fact_lines.append(f"- {claim}")
            if fact_lines:
                redacted_summary += "\n\nKey facts:\n" + "\n".join(fact_lines)
                redacted_summary = self.pii_redactor.anonymize_summary(
                    redacted_summary
                )

        # Append knowledge.decisions — decisions with their rationale.
        decisions = knowledge.get("decisions", [])
        if decisions:
            decision_lines: List[str] = []
            for dec in decisions:
                decision_text = dec.get("decision", "")
                reason = dec.get("reason", "")
                if decision_text:
                    if reason:
                        decision_lines.append(f"- {decision_text} (reason: {reason})")
                    else:
                        decision_lines.append(f"- {decision_text}")
            if decision_lines:
                redacted_summary += "\n\nDecisions:\n" + "\n".join(decision_lines)
                redacted_summary = self.pii_redactor.anonymize_summary(
                    redacted_summary
                )

        # Append implications — hypothetical future situations that would
        # only make sense because of what happened in this conversation.
        # Uses *different* vocabulary than the original text, bridging the
        # semantic gap so vector search can match indirect future queries.
        if implications:
            impl_lines = [f"- {imp}" for imp in implications if imp]
            if impl_lines:
                redacted_summary += (
                    "\n\nFuture relevance:\n" + "\n".join(impl_lines)
                )
                redacted_summary = self.pii_redactor.anonymize_summary(
                    redacted_summary
                )

        # --- Extract structured metadata for separate storage ---
        # These become individual Revision node properties in Neo4j,
        # included in SEMANTIC_KEYS for embedding and available for
        # score_fields-based focused scoring.
        structured_metadata: Dict[str, str] = {}

        entities_list = summary_result.get("classification", {}).get("entities", [])
        if entities_list:
            structured_metadata["entities"] = ", ".join(str(e) for e in entities_list)

        if facts:
            fact_claims = [f.get("claim", "") for f in facts if f.get("claim")]
            if fact_claims:
                structured_metadata["facts"] = "; ".join(fact_claims)

        if events:
            event_summaries: List[str] = []
            for ev in events:
                desc = ev.get("event", "")
                when = ev.get("when", "")
                if desc:
                    prefix = f"[{when}] " if when and when.lower() != "unknown" else ""
                    event_summaries.append(f"{prefix}{desc}")
            if event_summaries:
                structured_metadata["events"] = "; ".join(event_summaries)

            # Canonical event_date = the earliest concrete date among this
            # memory's events (valid-time), kept SEPARATE from the server-set
            # created_at (storage time). ISO-8601 sorts chronologically, so
            # min() of the validated dates is the earliest. Surfaced at recall
            # in both modes and usable as an opt-in temporal ranking signal.
            iso_dates = [
                d for ev in events
                if _ISO_EVENT_DATE_RE.match(d := str(ev.get("event_date", "")).strip())
            ]
            if iso_dates:
                canonical_event_date = min(iso_dates)
                structured_metadata["event_date"] = canonical_event_date
                # Valid-time corroboration (#119). The date passed only a FORMAT
                # regex above; a well-formed hallucination would still pollute
                # the event-proximity boost. Cross-check it against the raw
                # transcript and stamp a confidence marker. An absent key means
                # "legacy row" downstream, so this is purely additive — existing
                # stored dates keep the boost unchanged. Unverified dates stay as
                # metadata but recall_rerank skips their boost.
                source_text = "\n".join(
                    str(m.get("content", "")) for m in messages
                )
                reference_ts = (
                    parse_timestamp(messages[-1].get("timestamp"))
                    if messages else None
                )
                structured_metadata["event_date_confidence"] = classify_event_date(
                    canonical_event_date, source_text, reference_ts,
                )

        if decisions:
            dec_texts = [d.get("decision", "") for d in decisions if d.get("decision")]
            if dec_texts:
                structured_metadata["decisions"] = "; ".join(dec_texts)

        if implications:
            structured_metadata["implications"] = "\n".join(implications)

        # Reject credentials before sending to cloud graph (spec §10.4.5)
        self.pii_redactor.reject_credentials(redacted_summary)

        store_result: Dict[str, Any] = {}
        if self.memory_store:
            topics = summary_result.get("classification", {}).get("topics", [])
            user_lines: List[str] = []
            assistant_lines: List[str] = []

            title = summary_result.get("title", "Conversation")
            conversation_markdown = self._build_conversation_markdown(
                messages=messages,
                title=title,
                session_id=session_id,
                summary=redacted_summary,
                topics=topics,
                user_lines_out=user_lines,
                assistant_lines_out=assistant_lines,
            )

            topic_hint = "/".join(topics[:2]) if topics else ""
            artifact_path = self._write_artifact(
                session_id=session_id,
                content=conversation_markdown,
                space_hint=resolved_space or topic_hint,
            )

            # Collect attachment pointers from all messages in the session
            all_attachments: List[Dict[str, Any]] = []
            for msg in messages:
                msg_attachments = (msg.get("metadata") or {}).get("attachments", [])
                all_attachments.extend(msg_attachments)

            payload: Dict[str, Any] = {
                "project": self.project,
                "memory_type": summary_result.get("type", "summary"),
                "title": title,
                "summary": redacted_summary,
                "user_text": "\n".join(user_lines),
                "assistant_text": "\n".join(assistant_lines),
                "artifact_location": artifact_path,
                "artifact_name": "conversation",
                "bundle_name": topics[0] if topics else "",
                "tags": ["summarized", "published"],
                "metadata": {
                    "session_id": session_id,
                    "message_count": str(len(messages)),
                    "topics": ",".join(topics),
                    **structured_metadata,
                },
            }
            # Explicit space_path or user_id-derived space takes precedence;
            # fall back to topic hint for backwards compatibility.
            if resolved_space:
                payload["space_path"] = resolved_space
            else:
                payload["space_hint"] = topic_hint
            if session_user_id:
                payload["metadata"]["user_id"] = session_user_id

            # Evidence grade: canonical metadata key + mirrored graph tag
            # (tags get server-side time-range history).  Only stamped
            # when a grade was provided — unmarked memories keep the
            # existing tag set.  MUST be applied before "published" —
            # the server freezes a revision as immutable once "published"
            # lands, silently rejecting every tag applied afterward.
            if resolved_evidence:
                payload["metadata"]["evidence_level"] = resolved_evidence
                payload["tags"].insert(-1, evidence_tag(resolved_evidence))
            if resolved_source:
                payload["metadata"]["source"] = resolved_source

            if all_attachments:
                payload["metadata"]["attachments"] = all_attachments

            if stack_revisions is not None:
                payload["stack_revisions"] = stack_revisions

            store_result = await self._store_with_retry(**payload)

            # Promote extracted entities to first-class `entity` Items with
            # ABOUT edges from the stored revision. Requires the revision
            # kref, so queued-for-retry stores are skipped (like SUPPORTS
            # edges above, replay has no enrichment mechanism).
            stored_kref = (store_result or {}).get("revision_kref", "")
            # Write-time graph enrichment. This is `await`ed, not fired off with
            # create_task: the MCP runtime dispatches consolidation via
            # asyncio.run (mcp_tools.tool_memory_consolidate), a one-shot loop
            # that cancels pending tasks on teardown — so a detached task's graph
            # writes would land nondeterministically or not at all. Both calls
            # are internally bounded (run_bounded_in_thread) and best-effort, and
            # both branches are reached ONLY on the opt-in path (ontology on, or
            # entity promotion explicitly configured), so the default store pays
            # nothing and only the opt-in path takes the bounded latency.
            if stored_kref and self.ontology_enabled:
                # Full schema-driven decomposition: entities + facts + decisions
                # + events + actions + questions, wired by typed edges. Subsumes
                # plain entity promotion.
                from kumiho_memory.ontology import decompose_and_link

                await decompose_and_link(
                    stored_kref, summary_result, project_name=self.project,
                )
            elif stored_kref and entities_list and self.entity_promotion_config:
                # Lighter entity-only mode: identity-keyed dedup + direct
                # kind="entity" search value, even without full decomposition.
                from kumiho_memory.entity_promotion import promote_entities

                await promote_entities(
                    stored_kref,
                    [str(e) for e in entities_list],
                    project_name=self.project,
                    config=self.entity_promotion_config,
                )

            # Decision Memory session mining chain (double opt-in:
            # KUMIHO_MEMORY_DECISIONS=1 AND KUMIHO_MEMORY_DECISIONS_AUTOMINE=1 — see
            # docs/SESSION_MINING_DESIGN.md §2.2c).  This exact spot is the
            # point: the Redis buffer is still alive, `messages` is already
            # in memory (zero re-reads), and `stored_kref` is in-band — the
            # consolidated revision cannot be re-found later by session_id
            # (search has no metadata filter).  Mirrors the
            # decompose_and_link chain precedent; a mining failure must
            # never break consolidation.
            if stored_kref:
                from kumiho_memory.code_decisions import code_automine_enabled

                if code_automine_enabled():
                    try:
                        await self.code_mine_session(
                            session_id,
                            messages=messages,
                            conversation_kref=stored_kref,
                            ingest_first=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "code session mining failed (non-fatal): %s", exc,
                        )

        await self.redis_buffer.clear_session(self.project, session_id)

        # Clear the active session pointer so the next conversation starts fresh.
        if session_user_id and session_context and hasattr(self.redis_buffer, "clear_active_session"):
            try:
                await self.redis_buffer.clear_active_session(
                    context=session_context,
                    user_canonical_id=session_user_id,
                )
            except Exception as exc:
                logger.debug("clear_active_session failed: %s", exc)

        return {
            "success": True,
            "summary": redacted_summary,
            "store_result": store_result,
        }

    async def store_tool_execution(
        self,
        *,
        task: str,
        status: str = "done",
        exit_code: Optional[int] = None,
        duration_ms: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        tools: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        space_hint: str = "",
        open_questions: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Store a tool execution result as a structured memory.

        For successful executions, stores as ``type: action``.
        For failures (non-zero exit code or ``status`` in
        ``{"failed", "error", "blocked"}``), stores as ``type: error``.

        Parameters
        ----------
        task:
            Description of what was executed (e.g. ``"git push origin main"``).
        status:
            Execution outcome: ``"done"``, ``"failed"``, ``"error"``,
            ``"blocked"``.
        exit_code:
            Process exit code (0 = success).
        duration_ms:
            Execution duration in milliseconds.
        stdout / stderr:
            Captured output (stored locally as artifact, not uploaded).
        tools:
            Tool names used (e.g. ``["shell_exec"]``).
        topics:
            Classification topics (e.g. ``["git", "deployment"]``).
        space_hint:
            Space path hint for organising the memory.
        open_questions:
            Unresolved questions from failed executions.
        derived_from:
            Krefs this execution was derived from.
        """
        if not self.memory_store:
            return {"success": False, "error": "No memory_store configured"}

        is_error = status in ("failed", "error", "blocked") or (
            exit_code is not None and exit_code != 0
        )
        memory_type = "error" if is_error else "action"

        # Build title from task description
        prefix = "Failed" if is_error else "Successfully executed"
        title = f"{prefix}: {task[:60]}"

        # Build summary
        if is_error and stderr:
            summary = f"Attempted '{task}' but failed: {stderr[:200]}"
        elif is_error:
            summary = f"Attempted '{task}' but failed with status '{status}'"
        else:
            summary = f"Executed '{task}' successfully"

        summary = self.pii_redactor.anonymize_summary(summary)

        # Reject credentials before sending to cloud graph (spec §10.4.5)
        self.pii_redactor.reject_credentials(summary)

        # Write execution log as local artifact
        log_content = (
            f"# Tool Execution: {task}\n\n"
            f"**Status:** {status}  \n"
            f"**Exit code:** {exit_code}  \n"
            f"**Duration:** {duration_ms}ms  \n\n"
        )
        if stdout:
            log_content += f"## stdout\n\n```\n{stdout}\n```\n\n"
        if stderr:
            log_content += f"## stderr\n\n```\n{stderr}\n```\n\n"

        safe_name = task.replace(" ", "_").replace("/", "_")[:40]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact_path = self._write_artifact(
            session_id=f"exec_{timestamp}_{safe_name}",
            content=log_content,
            space_hint=space_hint,
        )

        resolved_topics = topics or []
        knowledge: Dict[str, Any] = {
            "actions": [{
                "task": task,
                "status": status,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
            }],
            "facts": [],
            "decisions": [],
            "open_questions": open_questions or [],
        }

        if is_error and stderr:
            knowledge["facts"].append({
                "claim": stderr[:200],
                "certainty": "high",
            })

        payload: Dict[str, Any] = {
            "project": self.project,
            "memory_type": memory_type,
            "title": title,
            "summary": summary,
            "user_text": task,
            "assistant_text": stdout[:500] if stdout else "",
            "artifact_location": artifact_path,
            "artifact_name": "execution_log",
            "bundle_name": resolved_topics[0] if resolved_topics else "",
            "space_hint": space_hint,
            "tags": [memory_type, status, "published"],
            "metadata": {
                "memory_type": memory_type,
                "exit_code": str(exit_code) if exit_code is not None else "",
                "duration_ms": str(duration_ms) if duration_ms is not None else "",
                "topics": ",".join(resolved_topics),
                "tools": ",".join(tools or []),
            },
        }
        if derived_from:
            payload["metadata"]["derived_from"] = derived_from

        store_result = await self._store_with_retry(**payload)
        return {"success": True, "memory_type": memory_type, "store_result": store_result}

    async def _store_with_retry(self, **payload: Any) -> Dict[str, Any]:
        """Call ``memory_store`` with retry + queue fallback.

        1. If a failure ledger is configured and this content is *parked*
           (repeated deterministic failures, issue #118), skip the store —
           it would just fail again — and return ``{"parked": True}``.
        2. Try up to ``store_max_retries`` with exponential backoff.
        3. On success, clear any ledger history for this content.
        4. On failure, classify it and record it in the ledger; then, if a
           ``retry_queue`` is configured, enqueue the payload for later replay,
           otherwise raise.
        """
        if not self.memory_store:
            return {}

        ledger = self.failure_ledger
        key: Optional[str] = None
        if ledger is not None:
            try:
                key = _payload_failure_key(payload)
                if key and ledger.is_parked(key):
                    logger.info(
                        "memory_store skipped — content parked after repeated "
                        "deterministic failures (#118)"
                    )
                    return {"parked": True}
            except Exception as exc:  # noqa: BLE001 — ledger must never break stores
                logger.debug("failure ledger park check failed (ignored): %s", exc)
                key = None

        try:
            result = await retry_with_backoff(
                self.memory_store,
                max_retries=self.store_max_retries,
                **payload,
            )
            if ledger is not None and key:
                try:
                    ledger.record_success(key)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("failure ledger record_success failed (ignored): %s", exc)
            return result
        except Exception as exc:
            if ledger is not None and key:
                try:
                    ledger.record_failure(key, classify_failure(exc))
                except Exception as le:  # noqa: BLE001
                    logger.debug("failure ledger record_failure failed (ignored): %s", le)
            if self.retry_queue is not None:
                self.retry_queue.enqueue(payload)
                logger.warning(
                    "memory_store failed after %d retries — queued for later: %s",
                    self.store_max_retries,
                    exc,
                )
                return {"queued": True, "error": str(exc)}
            raise

    async def flush_retry_queue(self) -> Dict[str, int]:
        """Replay queued ``memory_store`` calls that previously failed.

        Returns ``{"succeeded": N, "failed": M, "dropped": D}``.  Items that
        fail transiently remain in the queue for the next flush; items that
        fail deterministically are dropped (they would never succeed, #118).
        """
        if not self.retry_queue or not self.memory_store:
            return {"succeeded": 0, "failed": 0, "dropped": 0}
        return await self.retry_queue.flush(self.memory_store)

    @staticmethod
    def _build_conversation_markdown(
        *,
        messages: List[Dict[str, Any]],
        title: str,
        session_id: str,
        summary: str,
        topics: List[str],
        user_lines_out: List[str],
        assistant_lines_out: List[str],
    ) -> str:
        """Build a Markdown document from the full interleaved conversation."""
        parts: List[str] = [
            f"# {title}",
            "",
            f"**Session:** `{session_id}`  ",
            f"**Messages:** {len(messages)}  ",
        ]
        if topics:
            parts.append(f"**Topics:** {', '.join(topics)}  ")
        parts.append(f"**Summary:** {summary}")
        parts.extend(["", "---", ""])

        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("content", "")
            timestamp = (
                msg.get("timestamp", "")
                or msg.get("metadata", {}).get("timestamp", "")
            )

            if role == "assistant":
                assistant_lines_out.append(text)
            else:
                user_lines_out.append(text)

            header = f"### {role.capitalize()}"
            if timestamp:
                header += f"  \n<sub>{timestamp}</sub>"
            parts.extend([header, "", text, ""])

        return "\n".join(parts)

    def _store_attachment(
        self, attachment: Dict[str, Any], *, context: str = ""
    ) -> Dict[str, Any]:
        """Copy an attached file into the artifact directory and return a pointer.

        Parameters
        ----------
        attachment:
            Must contain ``path`` (source file).  Optional keys:
            ``content_type`` (MIME), ``description``.
        context:
            Space hint for organising the file inside the artifact tree.

        Returns
        -------
        Artifact pointer dict with ``location``, ``hash``, ``size_bytes``,
        ``content_type``, ``original_name``, and ``description``.
        """
        source = Path(attachment["path"])
        if not source.is_file():
            raise FileNotFoundError(f"Attachment not found: {source}")

        # Determine MIME type
        content_type = attachment.get("content_type")
        if not content_type:
            content_type, _ = mimetypes.guess_type(source.name)
            content_type = content_type or "application/octet-stream"

        # Target directory: {artifact_root}/{project}/attachments/{context}/
        target_dir = Path(self.artifact_root) / self.project / "attachments"
        if context:
            target_dir = target_dir / context
        target_dir.mkdir(parents=True, exist_ok=True)

        # Compute hash before copying (stream-friendly)
        sha = hashlib.sha256()
        size = 0
        with open(source, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha.update(chunk)
                size += len(chunk)
        file_hash = sha.hexdigest()

        # Copy with hash prefix to avoid collisions
        dest = target_dir / f"{file_hash[:12]}_{source.name}"
        shutil.copy2(source, dest)

        return {
            "type": "attachment",
            "original_name": source.name,
            "storage": "local",
            "location": dest.as_uri(),
            "hash": f"sha256:{file_hash}",
            "size_bytes": size,
            "content_type": content_type,
            "description": attachment.get("description", ""),
        }

    def _write_artifact(
        self, *, session_id: str, content: str, space_hint: str = ""
    ) -> str:
        """Write conversation Markdown and return the path.

        Directory layout::

            {artifact_root}/{project}/{space_segments...}/{session}.md
        """
        safe_name = session_id.replace(":", "_").replace("/", "_")
        target_dir = Path(self.artifact_root) / self.project
        if space_hint:
            segments = [seg for seg in space_hint.split("/") if seg.strip()]
            target_dir = target_dir.joinpath(*segments)
        target_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = target_dir / f"{safe_name}.md"
        artifact_path.write_text(content, encoding="utf-8")
        return str(artifact_path)

    async def recall_memories(
        self,
        query: str,
        *,
        limit: int = 5,
        space_paths: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        graph_augmented: bool = False,
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve long-term memories by semantic query.

        Parameters
        ----------
        query:
            Natural-language search query.
        limit:
            Maximum number of results.
        space_paths:
            Restrict search to these space paths (e.g.
            ``["CognitiveMemory/personal"]``).  When ``None``, searches
            all spaces in the project.
        memory_types:
            Filter by memory type (e.g. ``["error"]`` to find past
            mistakes, ``["action", "error"]`` for all tool executions).
            When ``None``, returns all types.
        graph_augmented:
            When ``True`` and a ``GraphAugmentationConfig`` was provided,
            uses multi-query reformulation + graph edge traversal to
            discover connected memories that vector search alone misses.
        query_time:
            Reference instant for the optional event-proximity rerank prior
            (:attr:`RerankConfig.event_proximity_enabled`).  Pass it ONLY for
            queries with a temporal intent; leaving it ``None`` (the default)
            keeps that prior fully dormant, so general recall is unchanged.
            Applies to both the plain and the graph-augmented recall path.
        """
        # Clear any error signal from a previous recall so a later healthy call
        # (even an empty one) never inherits a stale backend_error.
        self._last_backend_error = None
        if graph_augmented and self.graph_augmentation_config is not None:
            gr = self._get_graph_recall()
            if gr is not None:
                # Graph augmentation uses lightweight recall (no siblings/
                # artifacts) internally so reformulated queries don't
                # duplicate expensive work.  Sibling enrichment runs once
                # on the final merged set.
                #
                # Rerank placement (measured, not aesthetic): the
                # cross-encoder and retrieve-wide-then-trim run PER
                # SUB-QUERY inside graph recall (see _graph_base_recall,
                # wired as gr's recall_fn) — each reformulated angle trims
                # its own candidates by relevance TO THAT ANGLE.  Applying
                # them here instead, against the original query on the
                # merged set, evicts multi-hop evidence: a second hop's
                # memory scores low against the full multi-topic question,
                # so a post-merge cross-encoder trim removes exactly what
                # the answer needs (measured: multi-hop 0.299 per-angle vs
                # 0.194 post-merge on the same data and levers).
                memories = await gr.recall(
                    query,
                    limit=limit,
                    space_paths=space_paths,
                    memory_types=memory_types,
                )
                # Final pass over the merged set: deterministic priors only
                # — evidence (recomputed from base_score, so per-angle
                # weighting is never double-counted), recency,
                # event-proximity (temporal queries), MMR diversity.  The
                # cross-encoder is deliberately EXCLUDED here: per-angle
                # relevance was already measured above, and re-scoring
                # against the original query would reintroduce the
                # multi-hop eviction this pipeline just avoided.
                from dataclasses import replace as _dc_replace
                from kumiho_memory.recall_rerank import rerank
                target = self.graph_augmentation_config.max_total or (limit * 3)
                if getattr(self.graph_augmentation_config, "entity_recall", False):
                    # The recall stage appends up to ``entity_recall_reserve``
                    # score-less entity siblings ON TOP of its cap; mirror that
                    # here so the trailing siblings survive this trim too
                    # (rerank keeps unscored entries last, so without the
                    # extension this [:target] slice would delete exactly
                    # them).
                    target += getattr(
                        self.graph_augmentation_config, "entity_recall_reserve", 0,
                    )
                if getattr(self.graph_augmentation_config, "fact_recall", False):
                    # Same on-top mirroring for the fact-recall entries: the
                    # recall stage appends up to ``fact_recall_max_results``
                    # after its cap, so the trim target must grow by the same
                    # amount or this slice would delete exactly them.
                    target += getattr(
                        self.graph_augmentation_config, "fact_recall_max_results", 0,
                    )
                final_cfg = _dc_replace(
                    self.rerank_config, cross_encoder_enabled=False,
                )
                memories = rerank(
                    query,
                    memories,
                    evidence_config=self.evidence_rank_config,
                    config=final_cfg,
                    reranker=None,
                    limit=target,
                    query_time=query_time,
                )
                if len(memories) > target:
                    memories = memories[:target]
                enriched = await self._enrich_with_siblings(memories, query)
                return self._apply_as_of_recall(enriched, query_time)

        base = await self._base_recall(
            query, limit=limit, space_paths=space_paths,
            memory_types=memory_types, query_time=query_time,
        )
        return self._apply_as_of_recall(base, query_time)

    def _apply_as_of_recall(
        self, memories: List[Dict[str, Any]], query_time: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        """Opt-in as-of demotion (ontology G8), applied once at the recall exit.

        A strict no-op unless the ``KUMIHO_MEMORY_AS_OF_RECALL`` flag is on AND a
        ``query_time`` was supplied — so the default recall path is unchanged
        (byte-identical). The instant is the query's temporal reference
        (``query_time``); facts whose valid-time interval excludes it are moved
        after the still-valid results (never deleted).
        """
        return apply_as_of_recall(
            memories, query_time, enabled=self.as_of_recall_enabled,
        )

    async def _graph_base_recall(
        self,
        query: str,
        *,
        limit: int = 5,
        space_paths: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Per-sub-query recall for graph augmentation: widen → rerank → trim.

        The inner ``recall_fn`` for :class:`GraphAugmentedRecall`.  Each
        reformulated angle gets its own retrieve-wide-then-trim and rerank
        stack (cross-encoder included) scored against *that angle's* query —
        so the candidates that survive per-angle trimming are the ones
        relevant to each hop of a multi-topic question.  Running these
        levers once on the merged set against the original query instead
        measurably evicts multi-hop evidence (the second hop scores low
        against the full question).

        No sibling enrichment, no artifacts — the merged set is enriched
        exactly once downstream.
        """
        multiplier = self.recall_candidate_multiplier
        fetch_limit = (
            math.ceil(limit * multiplier) if multiplier > 1.0 else limit
        )
        memories = await self._lightweight_recall(
            query, limit=fetch_limit, space_paths=space_paths,
            memory_types=memory_types,
        )
        if not memories:
            return memories
        # rerank_async: the cross-encoder stage is CPU-bound inference —
        # awaited off-loop so concurrent sub-query recalls keep overlapping
        # their network I/O instead of serializing behind it.
        from kumiho_memory.recall_rerank import rerank_async
        memories = await rerank_async(
            query,
            memories,
            evidence_config=self.evidence_rank_config,
            config=self.rerank_config,
            reranker=self.reranker,
            limit=limit,
        )
        if len(memories) > limit:
            memories = memories[:limit]
        return memories

    async def _lightweight_recall(
        self,
        query: str,
        *,
        limit: int = 5,
        space_paths: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fast recall: server search + basic metadata only.

        Skips sibling enrichment and artifact loading so it can be used
        as the inner ``recall_fn`` for graph-augmented recall without
        duplicating expensive work across reformulated queries.
        """
        if not self.memory_retrieve:
            return []

        kwargs: Dict[str, Any] = {
            "project": self.project,
            "query": query,
            "limit": limit,
        }
        if space_paths:
            kwargs["space_paths"] = space_paths
        if memory_types:
            kwargs["memory_types"] = memory_types

        result = await _maybe_await(self.memory_retrieve, **kwargs)

        if isinstance(result, dict) and "error" in result:
            logger.warning("_lightweight_recall: retrieve returned error: %s", result["error"])
            # Record the backend failure so callers can distinguish it from an
            # empty-but-healthy result. Truncated to keep tool payloads small.
            self._last_backend_error = str(result["error"])[:500]
            return []

        if isinstance(result, dict) and "revision_krefs" in result:
            revision_krefs = result.get("revision_krefs", [])
            item_krefs = result.get("item_krefs", [])
            scores = result.get("scores", [])

            # Fetch basic revision metadata (title, summary, type) —
            # no artifacts, no siblings.
            meta_tasks = [
                self._fetch_revision_metadata(kref, load_artifacts=False)
                for kref in revision_krefs
            ]
            meta_results = await asyncio.gather(
                *meta_tasks, return_exceptions=True,
            )

            enriched: List[Dict[str, Any]] = []
            for i, (kref, meta) in enumerate(zip(revision_krefs, meta_results)):
                if isinstance(meta, BaseException):
                    logger.warning(
                        "Failed to fetch metadata for %s: %s", kref, meta,
                    )
                    meta = {}
                entry: Dict[str, Any] = {"kref": kref}
                if i < len(scores):
                    entry["score"] = scores[i]
                if i < len(item_krefs):
                    entry["_item_kref"] = item_krefs[i]
                entry.update(meta)
                enriched.append(entry)
            return enriched
        if isinstance(result, list):
            return result
        return []

    async def _enrich_with_siblings(
        self,
        memories: List[Dict[str, Any]],
        query: str,
        alt_queries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Add sibling revisions and artifact content to pre-recalled memories.

        Runs once on the final merged set after graph augmentation has
        deduplicated results, so sibling enrichment and LLM reranking
        happen exactly once per unique item.

        ``alt_queries`` carries the reformulated angles of a multi-query
        recall (graph-augmented path).  Sibling selection sees every angle,
        so an item recalled via a reformulation keeps the revisions relevant
        to *that* angle — with only the original query, a multi-topic
        question (multi-hop) selects siblings for its dominant topic and
        drops the other hop's evidence.

        Per-item sibling selection is independent (each item's revisions and
        the shared query/angles are its only inputs), so the enrichment work
        runs concurrently under a bounded semaphore instead of one serial LLM
        round-trip per item (#102). Results are written back to each ``mem`` in
        place and ``memories`` is returned in its original order, so completion
        order does not affect output. Beyond ``sibling_llm_cap`` LLM-eligible
        items the reranker's existing deterministic fallback is used; the cap
        is a safety valve (default well above typical recalls), not a knob.
        """
        _load_arts = self.recall_mode == "full"
        cap = self.sibling_llm_cap

        async def _enrich_one(
            mem: Dict[str, Any], item_kref: str, allow_llm: bool,
        ) -> None:
            # Load artifact for the primary revision if in full mode.
            if _load_arts and "content" not in mem:
                kref = mem.get("kref", "")
                if kref:
                    try:
                        enriched_meta = await self._fetch_revision_metadata(
                            kref, load_artifacts=True,
                        )
                        if "content" in enriched_meta:
                            mem["content"] = enriched_meta["content"]
                            mem["artifact_name"] = enriched_meta.get("artifact_name", "")
                            mem["artifact_location"] = enriched_meta.get("artifact_location", "")
                    except Exception:
                        pass

            primary_score = mem.get("score", 0.0)
            if isinstance(primary_score, bool) or not isinstance(
                primary_score, (int, float)
            ):
                primary_score = 0.0

            siblings = await self._fetch_sibling_revision_summaries(
                item_kref, mem.get("kref", ""), query=query,
                load_artifacts=_load_arts,
                alt_queries=alt_queries,
                primary_score=float(primary_score),
                allow_llm_rerank=allow_llm,
            )
            if siblings:
                mem["sibling_revisions"] = siblings

        # Assign the LLM-call budget deterministically in list order so the
        # cap decision never depends on gather completion order. Items without
        # _item_kref (graph-traversal results) are skipped entirely.
        pending: List[Tuple[Dict[str, Any], str, bool]] = []
        llm_ordinal = 0
        for mem in memories:
            item_kref = mem.pop("_item_kref", "")
            if not item_kref:
                # Graph-augmented results from edge traversal don't
                # have _item_kref — skip sibling enrichment for them.
                continue
            allow_llm = cap <= 0 or llm_ordinal < cap
            llm_ordinal += 1
            pending.append((mem, item_kref, allow_llm))

        if pending:
            sem = asyncio.Semaphore(_SIBLING_LLM_CONCURRENCY)

            async def _bounded(
                mem: Dict[str, Any], item_kref: str, allow_llm: bool,
            ) -> None:
                async with sem:
                    await _enrich_one(mem, item_kref, allow_llm)

            await asyncio.gather(
                *[_bounded(m, ik, al) for (m, ik, al) in pending]
            )

        return memories

    async def _base_recall(
        self,
        query: str,
        *,
        limit: int = 5,
        space_paths: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Core vector/fulltext recall without graph augmentation.

        Performs lightweight recall then enriches with siblings in one
        pass.  Used for non-graph-augmented path.

        Retrieve-wide-then-trim: when ``recall_candidate_multiplier > 1.0``
        the underlying search fetches ``ceil(limit * multiplier)`` candidates,
        the full rerank stack runs on that wide set, and the result is trimmed
        back to ``limit`` *before* the expensive sibling enrichment — so the
        best-reranked candidates survive and only they are enriched.
        """
        multiplier = self.recall_candidate_multiplier
        fetch_limit = (
            math.ceil(limit * multiplier) if multiplier > 1.0 else limit
        )
        memories = await self._lightweight_recall(
            query, limit=fetch_limit, space_paths=space_paths,
            memory_types=memory_types,
        )
        # Post-recall rerank on the plain path: cross-encoder (optional) +
        # evidence prior + recency prior + MMR diversity, applied once.  The
        # graph path applies its own evidence weighting inside
        # GraphAugmentedRecall (before its caps) — never both.  Awaited via
        # rerank_async so the CPU-bound cross-encoder never blocks the loop.
        from kumiho_memory.recall_rerank import rerank_async
        memories = await rerank_async(
            query,
            memories,
            evidence_config=self.evidence_rank_config,
            config=self.rerank_config,
            reranker=self.reranker,
            limit=limit,
            query_time=query_time,
        )
        # Trim the over-fetched candidates back to the caller's limit after
        # reranking has surfaced the strongest to the front (rerank's MMR
        # already front-loads `limit` items when it runs).
        if fetch_limit > limit and len(memories) > limit:
            memories = memories[:limit]
        return await self._enrich_with_siblings(memories, query)

    def _get_graph_recall(self) -> Optional[Any]:
        """Lazily create the GraphAugmentedRecall instance."""
        if self._graph_recall is not None:
            return self._graph_recall
        if self.graph_augmentation_config is None:
            return None
        try:
            from kumiho_memory.graph_augmentation import GraphAugmentedRecall

            # Try to get the LLM adapter for query reformulation.
            # If unavailable (no API key configured), graph-augmented recall
            # still works for edge traversal and semantic fallback — only
            # multi-query reformulation is skipped.
            adapter = None
            model = ""
            try:
                adapter = self.summarizer.adapter
                model = self.summarizer.light_model
            except Exception:
                logger.info(
                    "No LLM adapter available — graph-augmented recall will "
                    "use edge traversal and semantic fallback without "
                    "multi-query reformulation."
                )

            from kumiho_memory.evidence_rank import apply_evidence_weights

            # Opt-in: seed edge traversal from top-scored sibling revisions.
            # The enrichment runs once inside GraphAugmentedRecall (it pops
            # _item_kref), so the manager's post-recall enrichment pass
            # skips those entries instead of re-doing the work.
            sibling_fetch_fn = (
                self._enrich_with_siblings
                if getattr(
                    self.graph_augmentation_config,
                    "sibling_seeded_traversal",
                    False,
                )
                else None
            )

            self._graph_recall = GraphAugmentedRecall(
                adapter=adapter,
                model=model,
                recall_fn=self._graph_base_recall,
                config=self.graph_augmentation_config,
                evidence_rerank_fn=lambda mems: apply_evidence_weights(
                    mems, self.evidence_rank_config,
                ),
                sibling_fetch_fn=sibling_fetch_fn,
            )
            return self._graph_recall
        except Exception as e:
            logger.warning("Failed to initialize GraphAugmentedRecall: %s", e)
            return None

    async def discover_edges_post_consolidation(
        self,
        revision_kref: str,
        summary: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Discover and create edges from a newly stored memory to related ones.

        Delegates to ``GraphAugmentedRecall.discover_edges()``.  Returns an
        empty list when graph augmentation is not configured.
        """
        gr = self._get_graph_recall()
        if gr is None:
            return []
        return await gr.discover_edges(revision_kref, summary, **kwargs)

    def build_recalled_context(
        self,
        memories: List[Dict[str, Any]],
        query: str = "",
        recall_mode: Optional[str] = None,
    ) -> str:
        """Build text context from recalled memories for an answering LLM.

        Parameters
        ----------
        memories:
            List of memory dicts as returned by ``recall_memories()``.
        query:
            The original trigger query.  When provided and an
            ``embedding_adapter`` is configured, sibling revisions are
            filtered by embedding cosine similarity as a second pass.
            Note: server-scored sibling filtering (when
            ``sibling_similarity_threshold > 0`` but no embedding adapter)
            already runs during ``recall_memories()`` — this method
            receives pre-filtered siblings in that case.
        recall_mode:
            ``"full"`` (default) includes artifact content (raw conversation
            text, truncated to the shared
            :data:`kumiho_memory.context_compose.CONTEXT_BUDGET_CHARS` budget,
            with the truncation marker appended when content was cut).
            ``"summarized"`` uses only title + summary — lossy but cheaper.
            Falls back to the instance's ``self.recall_mode`` when ``None``.

        Note: this assembler KEEPS the truncation marker (agents on the MCP
        ``engage`` path genuinely benefit from knowing content was cut, and
        this path is bench-irrelevant).  ``compose_context`` — the bench/SDK
        path — slices silently instead: the marker text measurably made the
        answer model hedge (−0.055 paired F1 on capped contexts, 0.19.0 RC
        gate 2026-07-18).  Do not unify the marker without re-measuring.
        """
        from kumiho_memory import context_compose
        from kumiho_memory.evidence_rank import evidence_badge

        mode = recall_mode or self.recall_mode
        threshold = self.sibling_similarity_threshold

        texts: List[str] = []
        for mem in memories:
            title = mem.get("title", "")
            summary = mem.get("summary", "")
            content = mem.get("content", "")
            badge = evidence_badge(mem, self.evidence_rank_config)
            # Temporal anchor for summarized mode (full mode already carries
            # dates inline in the raw content). Collapses to "" when absent.
            ev_date = mem.get("event_date", "")
            date_prefix = f"[{ev_date}] " if ev_date else ""

            # Surface the extracted atomic facts (attribute→value claims like
            # "Melanie has been married for five years") as a concise, easily
            # parsed block. They are already embedded in the narrative summary,
            # but pulling them out lets the answering LLM read the precise
            # claim directly instead of digging it out of prose — the same
            # profile-style precise recall that dedicated fact stores rely on.
            facts = mem.get("facts", "")
            if isinstance(facts, (list, tuple)):
                facts = "; ".join(
                    f.get("claim", str(f)) if isinstance(f, dict) else str(f)
                    for f in facts
                )
            facts_suffix = f"\nFacts: {facts}" if facts else ""

            if mode == "full" and content:
                texts.append(
                    badge + context_compose.truncate_section(content)
                    + facts_suffix
                )
            elif summary:
                texts.append(
                    (f"{badge}{date_prefix}{title}: {summary}"
                     if title
                     else f"{badge}{date_prefix}{summary}") + facts_suffix
                )

            # Unfold sibling revisions only in full mode.  In summarized
            # mode the primary title+summary is enough — unrolling siblings
            # can balloon context by 10-30x for stacked items.
            if mode == "full":
                siblings = mem.get("sibling_revisions", [])
                if siblings and query and threshold > 0 and self.embedding_adapter is not None:
                    siblings = self._filter_siblings_by_embedding(
                        siblings, query, threshold,
                    )

                for sib in siblings:
                    sib_badge = evidence_badge(sib, self.evidence_rank_config)
                    sib_content = sib.get("content", "")
                    if sib_content:
                        texts.append(
                            sib_badge
                            + context_compose.truncate_section(sib_content)
                        )
                    else:
                        sib_title = sib.get("title", "")
                        sib_summary = sib.get("summary", "")
                        if sib_summary:
                            texts.append(
                                f"{sib_badge}{sib_title}: {sib_summary}"
                                if sib_title
                                else sib_badge + sib_summary
                            )

        return "\n\n".join(texts) if texts else ""

    def compose_context(
        self,
        memories: List[Dict[str, Any]],
        query: str = "",
        *,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        char_limit: Optional[int] = None,
    ) -> str:
        """Revision-centric context assembly (see :mod:`kumiho_memory.context_compose`).

        Flattens primary + ``sibling_revisions`` from every recalled memory
        (skipping the primary shell when siblings exist — they contain all
        revisions), ranks the pool globally by ``_score``, caps it at
        *top_k*, and renders ``"full"`` (raw content) or ``"summarized"``
        (title+summary) text.

        Complements :meth:`build_recalled_context` (item-centric, with
        evidence badges/facts/event dates): use ``compose_context`` when the
        best revisions should compete globally regardless of which item they
        belong to — e.g. heavily stacked items.

        *mode* defaults to the manager's ``recall_mode``; *top_k* ``None``
        uses :data:`kumiho_memory.context_compose.DEFAULT_CONTEXT_TOP_K`
        (pass ``0`` for unlimited).
        """
        from kumiho_memory.context_compose import compose_context

        fact_budget = 2
        if self.graph_augmentation_config is not None and getattr(
            self.graph_augmentation_config, "fact_recall", False,
        ):
            fact_budget = getattr(
                self.graph_augmentation_config, "fact_recall_max_results", 2,
            )
        return compose_context(
            memories,
            query,
            mode=mode or self.recall_mode,
            top_k=top_k,
            char_limit=char_limit,
            fact_budget=fact_budget,
        )

    def rerank_memories(
        self,
        memories: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Two-pass focused rerank of recalled memories and their siblings.

        Re-scores every memory (``score``) and every ``sibling_revisions``
        entry (``_score``) with the configured ``embedding_adapter`` over
        **title+summary text only**, replacing the server scores so the most
        directly relevant revisions rank highest in
        :meth:`compose_context`'s global pool.  Safe no-op (input returned
        unchanged) when no embedding adapter is configured, *query* is
        empty, or embedding fails — see
        :func:`kumiho_memory.recall_rerank.two_pass_rerank`.
        """
        from kumiho_memory.recall_rerank import two_pass_rerank

        return two_pass_rerank(query, memories, self.embedding_adapter)

    _CODE_MEMORY_DISABLED = "code memory is disabled (set KUMIHO_MEMORY_DECISIONS=1)"

    def _code_memory_context(self) -> Optional[Tuple[Any, str, Any, str]]:
        """Gate + shared wiring for the Decision Memory delegations.

        ``(cfg, project_name, adapter, model)`` when ``KUMIHO_MEMORY_DECISIONS=1``,
        else ``None``.  The LLM wiring reuses the summarizer's adapter +
        light model — no separate key.  Lazy import so the conversation
        paths carry zero code-domain imports while the gate is off.
        """
        from kumiho_memory.code_decisions import (
            code_memory_enabled, config_from_env, resolve_project_name,
        )

        if not code_memory_enabled():
            return None
        cfg = config_from_env()
        adapter = getattr(self.summarizer, "adapter", None)
        model = getattr(self.summarizer, "light_model", "") or getattr(
            self.summarizer, "model", "",
        )
        return cfg, resolve_project_name(self.project, cfg), adapter, model

    async def code_why(
        self,
        question: Optional[str] = None,
        *,
        file: Optional[str] = None,
        line: Optional[int] = None,
        commit: Optional[str] = None,
        repo: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Ask why code is the way it is (Decision Memory, opt-in).

        Thin delegation into the code-decision domain — everything is lazy
        and gated behind ``KUMIHO_MEMORY_DECISIONS=1`` so the conversation paths
        carry zero new imports or behavior when the domain is off.  Code
        decisions live in a dedicated ``{project}-decisions`` kumiho project
        (physical isolation from conversation recall); see
        ``docs/DECISION_MEMORY_DESIGN.md``.
        """
        ctx = self._code_memory_context()
        if ctx is None:
            return {
                "decisions": [], "context": "",
                "error": self._CODE_MEMORY_DISABLED,
            }
        from kumiho_memory.code_query import why

        cfg, project_name, _adapter, _model = ctx
        return await why(
            question,
            file=file, line=line, commit=commit, repo=repo, limit=limit,
            project_name=project_name,
            config=cfg,
            reranker=self.reranker,
        )

    async def code_ingest(
        self,
        repo_path: str = ".",
        rev_range: Optional[str] = None,
        *,
        max_commits: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Mine a git commit range into Decision Memory (opt-in, gated).

        Same lazy/gated shape as :meth:`code_why`.
        """
        ctx = self._code_memory_context()
        if ctx is None:
            return {"errors": [self._CODE_MEMORY_DISABLED]}
        from kumiho_memory.code_capture import ingest_repo

        cfg, project_name, adapter, model = ctx
        stats = await ingest_repo(
            repo_path, rev_range,
            project_name=project_name,
            config=cfg, adapter=adapter, model=model,
            force=force, max_commits=max_commits,
            redactor=self.pii_redactor,
        )
        return stats.as_dict()

    async def code_capture(
        self,
        decisions: List[Dict[str, Any]],
        *,
        repo_path: str = ".",
        commit_ref: str = "HEAD",
    ) -> Dict[str, Any]:
        """Store agent-extracted code decisions — **keyless** (opt-in, gated).

        The self-contained capture path: the agent (Claude) reads the diff or
        conversation, extracts the decision, and passes it here — no separate
        LLM key, mirroring :meth:`memory_reflect`.  ``ingest``/``mine_session``
        exist for the detached-hook / batch case that has no agent in the
        loop and therefore does need a model; this is the primary path when
        Claude is present.
        """
        from kumiho_memory.code_decisions import (
            code_memory_enabled, config_from_env, resolve_project_name,
        )

        if not code_memory_enabled():
            return {"errors": ["code memory is disabled (set KUMIHO_MEMORY_DECISIONS=1)"]}
        from kumiho_memory.code_capture import capture_decisions

        cfg = config_from_env()
        stats = await capture_decisions(
            repo_path, decisions or [], commit_ref=commit_ref,
            project_name=resolve_project_name(self.project, cfg), config=cfg,
            redactor=self.pii_redactor,
        )
        return stats.as_dict()

    async def memory_decompose(
        self,
        kref: str,
        *,
        entities: Optional[List[Dict[str, Any]]] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
        relations: Optional[List[Dict[str, Any]]] = None,
        supersedes: Optional[List[Dict[str, Any]]] = None,
        contradicts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Keyless agent-driven ontology decomposition of a stored memory.

        The in-loop agent (Claude) — which already read the conversation —
        extracts entities / facts / relations and passes them; this only
        validates + writes the typed graph (entity/fact nodes, ABOUT /
        DERIVED_FROM / relation edges), reusing the exact deterministic write
        path the LLM decomposition uses.  No external LLM key — mirrors
        :meth:`memory_reflect` / :meth:`code_capture`.  ``kref`` is the stored
        memory revision (from consolidate/reflect) the typed nodes anchor to.
        """
        if not getattr(self, "ontology_enabled", False):
            return {"errors": ["ontology is disabled (set KUMIHO_MEMORY_ONTOLOGY=1)"]}
        if not kref:
            return {"errors": ["kref is required (the stored memory revision to decompose)"]}
        from kumiho_memory.ontology import decompose_and_link_agent

        stats = await decompose_and_link_agent(
            kref,
            {"entities": entities or [], "facts": facts or [], "relations": relations or [],
             "supersedes": supersedes or [], "contradicts": contradicts or []},
            project_name=self.project,
        )
        return {"decomposed": stats, "kref": kref}

    async def code_mine_session(
        self,
        session_id: str,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        conversation_kref: str = "",
        repo_path: str = ".",
        ingest_first: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Mine an agent session into Decision Memory (opt-in, gated).

        Session mining enriches commit-mined decisions with the
        conversation-only cargo (rejected alternatives, measurements in
        their original form), captures decisions that never reached a
        commit, and bridges decisions to the consolidated conversation
        revision.  Same lazy/gated shape as :meth:`code_ingest`; see
        ``docs/SESSION_MINING_DESIGN.md``.

        ``ingest_first`` runs an incremental commit ingest before mining so
        enrichment targets exist ("talk → commit → mine" is the natural
        workflow); already-captured commits are marker-skipped at zero LLM
        cost.
        """
        ctx = self._code_memory_context()
        if ctx is None:
            return {"errors": [self._CODE_MEMORY_DISABLED]}
        from kumiho_memory.code_session import mine_session

        cfg, project_name, adapter, model = ctx
        if ingest_first:
            try:
                await self.code_ingest(repo_path)
            except Exception as exc:  # noqa: BLE001 — enrichment degrades, mining proceeds
                logger.warning("code session mining: pre-ingest failed: %s", exc)
        stats = await mine_session(
            session_id,
            project_name=project_name,
            messages=messages,
            conversation_kref=conversation_kref,
            repo_path=repo_path,
            config=cfg, adapter=adapter, model=model,
            redactor=self.pii_redactor,
            redis_buffer=self.redis_buffer,
            memory_project=self.project,
            force=force,
        )
        return stats.as_dict()

    async def _fetch_revision_metadata(
        self, kref: str, load_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """Fetch revision metadata and optionally raw artifact content.

        The revision metadata contains redacted/sanitized fields (title,
        summary, type).  The source of truth for the full conversation is
        the raw Markdown artifact stored locally.

        When *load_artifacts* is False (e.g. summarized recall mode),
        the expensive file reads are skipped — title + summary is enough.
        """
        try:
            import kumiho

            revision = await asyncio.to_thread(kumiho.get_revision, kref)
            meta = revision.metadata or {}
            entry: Dict[str, Any] = {
                "title": meta.get("title", ""),
                "summary": meta.get("summary", ""),
                # "type" is a server-reserved metadata key stripped from
                # reads; newer writes mirror it as "memory_type".
                "type": meta.get("type") or meta.get("memory_type") or "",
                "space": meta.get("space", ""),
                "created_at": getattr(revision, "created_at", ""),
                "tags": getattr(revision, "tags", []),
            }
            # Evidence grade (issue #9 schema) — only set when the
            # revision carries it, so ungraded results stay unchanged.
            if meta.get("evidence_level"):
                entry["evidence_level"] = meta["evidence_level"]
            if meta.get("source"):
                entry["source"] = meta["source"]
            # Grounding-staleness marker (#95): a directly-recalled dependent
            # whose grounding fact was superseded carries the flag in the
            # metadata already fetched here — additive, zero extra round-trip.
            apply_grounding_marker(entry, meta)
            # Semantic event date (valid-time). Surfaced BEFORE the
            # load_artifacts branch so it reaches summarized recall too —
            # the one mode that is otherwise date-blind (no content loaded).
            if meta.get("event_date"):
                entry["event_date"] = meta["event_date"]
                # Confidence marker (#119): carry it alongside event_date so the
                # event-proximity boost can skip uncorroborated dates. Absent key
                # (legacy rows) is treated as trusted, so the boost is unchanged.
                if meta.get("event_date_confidence"):
                    entry["event_date_confidence"] = meta["event_date_confidence"]
            # Valid-time interval (G8): additive valid_from/valid_to alongside
            # event_date, surfaced only when present + ISO-valid. Feeds the
            # opt-in as-of recall filter (KUMIHO_MEMORY_AS_OF_RECALL); absent on
            # legacy revisions, so recall is unchanged there.
            apply_valid_interval_marker(entry, meta)

            # Read the raw conversation from the local artifact file.
            if load_artifacts:
                try:
                    artifacts = await asyncio.to_thread(revision.get_artifacts)
                    for artifact in artifacts:
                        location = getattr(artifact, "location", "")
                        if not location:
                            continue
                        content = await self._read_artifact_content(location)
                        if content:
                            entry["artifact_name"] = getattr(artifact, "name", "")
                            entry["artifact_location"] = location
                            entry["content"] = content
                            break  # use the first readable artifact
                except Exception as exc:
                    logger.debug("Failed to fetch artifacts for %s: %s", kref, exc)

            return entry
        except Exception as exc:
            logger.debug("Failed to fetch revision %s: %s", kref, exc)
            return {}

    def _filter_siblings_by_embedding(
        self,
        siblings: List[Dict[str, Any]],
        query: str,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Keep only siblings whose embedding similarity to *query* exceeds *threshold*.

        Uses the configured ``embedding_adapter`` to compute cosine similarity.
        Falls back to returning all siblings if embedding fails.
        """
        if not siblings or not query or threshold <= 0 or self.embedding_adapter is None:
            return siblings

        sib_texts = []
        for sib in siblings:
            t = sib.get("title", "")
            s = sib.get("summary", "")
            # Fact-level ranking: fold the extracted atomic facts into the text
            # scored against the query. A revision whose title/summary is off
            # topic but whose facts hold the answer ("Caroline is from Sweden"
            # under a "counseling" summary) would otherwise rank too low to
            # reach the context. Scoring on the facts too lifts the revision
            # that actually contains the queried attribute — the precise,
            # profile-style retrieval that direct single-hop / temporal
            # questions need.
            f = sib.get("facts", "")
            imp = sib.get("implications", "")
            if isinstance(imp, list):
                imp = "; ".join(str(x) for x in imp)
            if imp:
                # Prospective-indexing parity: implications are the write-time
                # answers to oblique future triggers — score on them too.
                f = f"{f}; {imp}" if f else imp
            if isinstance(f, (list, tuple)):
                f = "; ".join(
                    x.get("claim", str(x)) if isinstance(x, dict) else str(x)
                    for x in f
                )
            base = f"{t}: {s}" if t else s
            sib_texts.append(f"{base}\nFacts: {f}" if f else base)

        try:
            all_texts = [query] + sib_texts
            embeddings = self.embedding_adapter.embed(all_texts)
            query_vec = embeddings[0]

            scored_sibs = []
            for i, sib in enumerate(siblings):
                score = _cosine_similarity(query_vec, embeddings[i + 1])
                scored_sibs.append((score, sib))

            # Sort by score descending, apply threshold.
            # Preserve _score on each sibling for downstream global ranking.
            scored_sibs.sort(key=lambda x: x[0], reverse=True)
            kept = [
                {**sib, "_score": score}
                for score, sib in scored_sibs
                if score >= threshold
            ]

            # Apply top-K cap if configured (0 = unlimited)
            if self.sibling_top_k > 0 and len(kept) > self.sibling_top_k:
                kept = kept[: self.sibling_top_k]

            logger.debug(
                "Sibling embedding filter: %d/%d kept (threshold=%.2f, top_k=%d, scores=%s)",
                len(kept), len(siblings), threshold, self.sibling_top_k,
                [f"{s:.3f}" for s, _ in scored_sibs],
            )
            return kept
        except Exception as e:
            logger.warning("Sibling embedding filter failed, keeping all: %s", e)
            return siblings

    async def _rerank_siblings_with_llm(
        self,
        siblings: List[Dict[str, Any]],
        query: str,
        alt_queries: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Use the LLM to select the most relevant siblings.

        Cosine similarity cannot bridge semantic inversion (e.g.
        "dining out a lot" ↔ "meal prepping for healthy lifestyle").
        The LLM CAN reason about these relationships — it understands
        that a broken goal implies the original goal existed.

        Returns selected siblings with ``_score`` set, or ``None`` if
        the LLM is unavailable so the caller can fall back.
        """
        try:
            adapter = self.summarizer.adapter
        except Exception:
            return None

        # Build numbered list of sibling summaries + structured metadata for the LLM.
        lines: List[str] = []
        for i, sib in enumerate(siblings, 1):
            title = sib.get("title", "Untitled")
            summary = sib.get("summary", "")
            # Truncate long summaries — keep enough context for the LLM.
            if len(summary) > 600:
                summary = summary[:600] + "..."
            entry = f"{i}. {title}: {summary}"
            # Append structured metadata — implications are forward-looking
            # statements that directly bridge semantic inversion (e.g.
            # "Evan might discuss guitar practice progress" matches
            # "barely followed through on something huge").
            for field, label in [
                ("implications", "Future scenarios"),
                ("facts", "Key facts"),
                ("entities", "People/things"),
                ("events", "Events"),
            ]:
                val = sib.get(field, "")
                if val:
                    if len(val) > 250:
                        val = val[:250] + "..."
                    entry += f"\n   {label}: {val}"
            lines.append(entry)

        summaries_text = "\n".join(lines)

        system = (
            "You are a memory retrieval specialist. Given a user's message "
            "and a numbered list of stored conversation summaries, identify "
            "which summaries are most relevant to what the user is referring to.\n\n"
            "IMPORTANT: The user may refer to a past conversation INDIRECTLY:\n"
            "- They might describe the OPPOSITE outcome (e.g. 'I've been "
            "dining out a lot' when the stored memory is about 'meal prepping "
            "for a healthier lifestyle')\n"
            "- They might reference a goal they DIDN'T achieve, where the "
            "stored memory is about SETTING that goal\n"
            "- They might use completely different vocabulary for the same "
            "underlying topic\n"
            "- They might describe a consequence instead of the cause\n\n"
            "Think about the underlying topic, goal, habit, or life event "
            "the user is referring to — not just surface-level word matching.\n\n"
            "Return ONLY the numbers of the 1-3 most relevant summaries, "
            "separated by commas. If none are clearly relevant, return 'none'."
        )

        # Reformulated angles (multi-query recall): show the LLM every angle
        # so it selects summaries covering EACH aspect of a multi-topic
        # question, not just its dominant one.
        angles_text = ""
        if alt_queries:
            angle_lines = "\n".join(f"- {a}" for a in alt_queries if a)
            if angle_lines:
                angles_text = (
                    "\nThe message may also be understood from these angles "
                    "(pick summaries relevant to ANY of them):\n"
                    f"{angle_lines}\n"
                )

        user_msg = (
            f"User's message:\n{query}\n"
            f"{angles_text}\n"
            f"Stored conversation summaries:\n{summaries_text}"
        )

        # Diagnostic: how many siblings have structured metadata?
        has_impl = sum(1 for s in siblings if s.get("implications"))
        has_facts = sum(1 for s in siblings if s.get("facts"))
        has_ent = sum(1 for s in siblings if s.get("entities"))
        logger.info(
            "Reranker metadata coverage: %d siblings — "
            "%d with implications, %d with facts, %d with entities",
            len(siblings), has_impl, has_facts, has_ent,
        )

        try:
            raw = await adapter.chat(
                messages=[{"role": "user", "content": user_msg}],
                model=self.summarizer.light_model,
                system=system,
                max_tokens=30,
            )
            text = raw.strip().lower()
            logger.info(
                "LLM sibling reranker response: %r (query: %.60s, %d siblings)",
                text, query, len(siblings),
            )

            if "none" in text:
                return None

            # Parse comma-separated numbers.
            selected_indices: List[int] = []
            for token in text.replace(",", " ").split():
                token = token.strip().rstrip(".")
                if token.isdigit():
                    idx = int(token) - 1  # 1-indexed → 0-indexed
                    if 0 <= idx < len(siblings):
                        selected_indices.append(idx)

            if not selected_indices:
                return None

            # Assign descending scores so first-picked ranks highest.
            result: List[Dict[str, Any]] = []
            for rank, idx in enumerate(selected_indices):
                score = 1.0 - rank * 0.1  # 1.0, 0.9, 0.8, ...
                result.append({**siblings[idx], "_score": score})

            return result

        except Exception as e:
            logger.warning("LLM sibling reranker failed: %s", e)
            return None

    def _rank_siblings_deterministic(
        self,
        siblings: List[Dict[str, Any]],
        query: str,
        current_rev_kref: str = "",
        alt_queries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank siblings by a local, in-process relevance signal.

        The deterministic counterpart to :meth:`_rerank_siblings_with_llm`.
        Prefers embedding cosine when an ``embedding_adapter`` is configured
        (honoring ``sibling_similarity_threshold`` and ``sibling_top_k``);
        otherwise falls back to BM25-light keyword overlap (honoring
        ``sibling_strong_score``, ``sibling_char_budget`` and
        ``sibling_top_k``), exactly the scoring the default keyword path uses.

        Critically it stays **in-process** — no ``ScoreRevisions`` RPC — so it
        avoids the ~8s server round-trip the old fallback incurred.  Each kept
        sibling carries ``_score`` (most-relevant first).

        When ``current_rev_kref`` names the primary (published) revision, that
        revision is always retained even if it scores below the cut: the
        downstream context builder skips the primary item entry whenever the
        sibling list is non-empty, so dropping the primary here would silently
        lose its content.
        """
        if not siblings or not query:
            return siblings

        # Embedding cosine — the strongest local signal, when available.
        if self.embedding_adapter is not None and self.sibling_similarity_threshold > 0:
            ranked = self._filter_siblings_by_embedding(
                siblings, query, self.sibling_similarity_threshold,
            )
            # The embedding filter may drop everything below threshold; only
            # trust it when it kept something, else fall through to keyword.
            if ranked:
                return self._ensure_primary_retained(
                    ranked, siblings, current_rev_kref,
                )

        # BM25-light keyword overlap (same scoring as the default Mode 3 path).
        # With reformulated angles present, a sibling scores as its BEST match
        # across the original query and every angle — a multi-hop question's
        # second hop matches its reformulation even when it barely overlaps
        # the original phrasing.
        angle_tokens = [_tokenize(query)] + [
            _tokenize(a) for a in (alt_queries or []) if a
        ]
        angle_tokens = [t for t in angle_tokens if t]
        for sib in siblings:
            # Facts parity with the embedding/LLM rankers: a revision whose
            # title/summary is off-topic but whose extracted facts hold the
            # answer must rank on its facts here too.
            facts = sib.get("facts", "")
            if isinstance(facts, list):
                facts = "; ".join(str(x) for x in facts)
            imp = sib.get("implications", "")
            if isinstance(imp, list):
                imp = "; ".join(str(x) for x in imp)
            text = f"{sib.get('title', '')} {sib.get('summary', '')} {facts} {imp}"
            sib["_score"] = max(
                (_token_overlap_score(t, text) for t in angle_tokens),
                default=0.0,
            )

        best_score = max((s["_score"] for s in siblings), default=0.0)
        if best_score >= self.sibling_strong_score:
            # Strong lexical signal — keep only meaningful matches, best first.
            kept = sorted(
                [s for s in siblings if s["_score"] >= self.sibling_strong_score],
                key=lambda s: s["_score"], reverse=True,
            )
        else:
            # Weak/no signal — keep all that fit the char budget, chronological
            # for full timeline coverage (mirrors the default keyword path).
            kept = sorted(siblings, key=lambda s: s.get("created_at") or "")
            total = sum(
                len(f"{s.get('title', '')} {s.get('summary', '')}") for s in kept
            )
            if total > self.sibling_char_budget:
                selected: List[Dict[str, Any]] = []
                used = 0
                for sib in kept:
                    chars = len(f"{sib.get('title', '')} {sib.get('summary', '')}")
                    if used + chars > self.sibling_char_budget:
                        continue
                    selected.append(sib)
                    used += chars
                kept = selected

        if self.sibling_top_k > 0 and len(kept) > self.sibling_top_k:
            kept = kept[: self.sibling_top_k]

        return self._ensure_primary_retained(kept, siblings, current_rev_kref)

    @staticmethod
    def _ensure_primary_retained(
        kept: List[Dict[str, Any]],
        all_siblings: List[Dict[str, Any]],
        current_rev_kref: str,
    ) -> List[Dict[str, Any]]:
        """Append the primary revision to *kept* if a filter dropped it.

        The downstream context builder skips the primary item entry when the
        sibling list is non-empty, so a non-empty result that excludes the
        primary would silently lose the primary's content.  No-ops when the
        primary is unknown, already kept, or *kept* is empty (an empty result
        means "use the primary directly" downstream).
        """
        if not current_rev_kref or not kept:
            return kept
        if any(s.get("kref") == current_rev_kref for s in kept):
            return kept
        primary = next(
            (s for s in all_siblings if s.get("kref") == current_rev_kref), None
        )
        if primary is not None:
            kept = kept + [primary]
        return kept

    async def _filter_siblings_by_server_search(
        self,
        siblings: List[Dict[str, Any]],
        query: str,
        item_kref: str,
        current_rev_kref: str = "",
        alt_queries: Optional[List[str]] = None,
        primary_score: float = 0.0,
        allow_llm_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Select relevant siblings: LLM reranker primary, deterministic fallback.

        Hybrid, not either/or.  The LLM reranker stays the **primary** signal:
        it reads every sibling summary and picks the most relevant, resolving
        the semantic inversion (cognitive/goal questions) that cosine and
        keyword scoring fundamentally cannot bridge — its selections rank on
        top with their assigned scores and are never overridden.

        When the LLM returns ``None`` (no pick), errors, or is unavailable, we
        **fall back to a deterministic in-process ranking** of the siblings
        rather than returning ``[]``.  Returning ``[]`` there was the LoCoMo
        single-hop / temporal regression: for direct-fact queries the LLM
        often answers "none" yet the correct sibling is present and cheaply
        rankable locally.  The fallback (:meth:`_rank_siblings_deterministic`)
        prefers local embedding cosine when an adapter is configured, else
        BM25-light keyword overlap — no ``ScoreRevisions`` RPC, so it avoids
        the ~8s server round-trip the old fallback incurred.

        On an LLM *success* a single capped union is applied: the top
        deterministically-scored sibling is appended when (a) the LLM did not
        already pick it and (b) it has a *strong* lexical/semantic score
        (``>= sibling_strong_score``).  This recovers a lexically-obvious
        match the LLM missed on direct-fact queries without flooding context,
        and no-ops on weak-overlap cognitive queries so the LLM's precise
        selection is left intact.
        """
        if not siblings or not query:
            return siblings

        # --- Primary: LLM reranking (semantic inversion) ---
        # Beyond the per-recall LLM-call cap (#102), allow_llm_rerank is False:
        # skip the round-trip and take exactly the same deterministic fallback
        # path below that fires when the LLM returns None / errors.
        llm_result = (
            await self._rerank_siblings_with_llm(
                siblings, query, alt_queries=alt_queries,
            )
            if allow_llm_rerank
            else None
        )
        if llm_result:
            # Bounded union: recover one strong lexical match the LLM missed.
            ranked = self._rank_siblings_deterministic(
                list(siblings), query, current_rev_kref,
                alt_queries=alt_queries,
            )
            if ranked:
                top = ranked[0]
                picked = {s.get("kref") for s in llm_result}
                if (
                    top.get("kref") not in picked
                    and float(top.get("_score", 0.0)) >= self.sibling_strong_score
                ):
                    llm_result = llm_result + [top]
            logger.info(
                "LLM sibling reranker: %d/%d selected (query: %.60s)",
                len(llm_result), len(siblings), query,
            )
            return llm_result

        # --- Fallback: deterministic ranking (LLM None / error / unavailable) ---
        ranked = self._rank_siblings_deterministic(
            siblings, query, current_rev_kref,
            alt_queries=alt_queries,
        )
        # Parity guard: without the LLM's judgment, the published revision
        # must compete at the item's own recall score.  The downstream
        # context builder skips the primary item entry whenever the sibling
        # list is non-empty — so a fallback list that carries the primary at
        # a near-zero keyword score silently DEMOTES the whole item relative
        # to the pre-fallback behavior (empty list -> primary entry ranked at
        # the item's recall score).  That demotion is what halved multi-hop:
        # the second hop's item usually fails the LLM pick (weak lexical
        # overlap with a multi-topic question) and then vanished from the
        # composed context.  Floor the primary revision's _score at the
        # item's recall score to restore the old ranking exactly; extra
        # fallback siblings keep their own scores and can only add below it.
        if ranked and current_rev_kref and primary_score > 0.0:
            for sib in ranked:
                if sib.get("kref") == current_rev_kref:
                    if float(sib.get("_score", 0.0) or 0.0) < primary_score:
                        sib["_score"] = primary_score
                        ranked = sorted(
                            ranked,
                            key=lambda s: float(s.get("_score", 0.0) or 0.0),
                            reverse=True,
                        )
                    break
        logger.info(
            "LLM sibling reranker returned None; deterministic fallback kept "
            "%d/%d siblings (query: %.60s)",
            len(ranked), len(siblings), query,
        )
        return ranked

    async def _fetch_sibling_revision_summaries(
        self,
        item_kref: str,
        current_rev_kref: str,
        query: str = "",
        load_artifacts: bool = True,
        alt_queries: Optional[List[str]] = None,
        primary_score: float = 0.0,
        allow_llm_rerank: bool = True,
    ) -> List[Dict[str, str]]:
        """Fetch title+summary from sibling revisions of a stacked item.

        For items with multiple revisions (conversation progression), this
        returns the summary of every revision *except* the one already
        fetched as the primary result.

        Three-phase selection strategy:

        1. **Embedding mode** — when an ``embedding_adapter`` is configured
           and ``sibling_similarity_threshold > 0``, filter by embedding
           cosine similarity.
        2. **Server-scored mode** — when ``sibling_similarity_threshold > 0``
           but no embedding adapter is available, use the Kumiho server's
           hybrid search (vector + BM25) to score siblings.
        3. **Keyword mode** (default) — BM25-light keyword overlap.  When
           the query has strong overlap (best score ≥ threshold), return only
           strong matches; otherwise keep all that fit within char budget.
        """
        try:
            import kumiho

            item = await asyncio.to_thread(kumiho.get_item, item_kref)
            revisions = await asyncio.to_thread(item.get_revisions)
            if not revisions or len(revisions) <= 1:
                return []

            siblings: List[Dict[str, Any]] = []
            for rev in revisions:
                rev_uri = rev.kref.uri if hasattr(rev.kref, "uri") else str(rev.kref)
                # Do NOT exclude the primary (published) revision — the LLM
                # reranker must see ALL revisions to select the correct one.
                # Previously the primary was excluded here and also skipped
                # in build_recalled_context, which meant the latest/published
                # revision was *never* in the recalled context.
                meta = rev.metadata or {}
                title = meta.get("title", "")
                summary = meta.get("summary", "")
                created_at = getattr(rev, "created_at", "") or ""
                if title or summary:
                    sib_text = f"{title} {summary}".strip()
                    sib_entry: Dict[str, Any] = {
                        "kref": rev_uri,
                        "title": title,
                        "summary": summary,
                        "created_at": created_at,
                        "_chars": len(sib_text),
                    }
                    # Carry structured metadata for LLM reranking.
                    for field in ("facts", "entities", "events", "decisions", "implications"):
                        val = meta.get(field, "")
                        if val:
                            sib_entry[field] = val
                    # Evidence grade so sibling badges have data
                    # (per-revision — stacked siblings may differ).
                    if meta.get("evidence_level"):
                        sib_entry["evidence_level"] = meta["evidence_level"]
                    siblings.append(sib_entry)

            if not siblings:
                return []

            total_siblings = len(siblings)

            # --- Semantic filtering modes (opt-in via sibling_similarity_threshold > 0) ---
            if self.sibling_similarity_threshold > 0 and query:
                if self.embedding_adapter is not None:
                    # Mode 1: Embedding-based cosine similarity (external API)
                    siblings = self._filter_siblings_by_embedding(
                        siblings, query, self.sibling_similarity_threshold,
                    )
                else:
                    # Mode 2: LLM reranker with a deterministic in-process
                    # fallback (no external API, no ScoreRevisions RPC).
                    siblings = await self._filter_siblings_by_server_search(
                        siblings, query, item_kref, current_rev_kref,
                        alt_queries=alt_queries,
                        primary_score=primary_score,
                        allow_llm_rerank=allow_llm_rerank,
                    )

                # Clean up internal keys before loading artifacts.
                for sib in siblings:
                    sib.pop("_chars", None)

                # Load artifact content only when recall_mode is "full".
                # In "summarized" mode, title+summary is sufficient and
                # skipping file reads saves significant I/O.
                if load_artifacts:
                    async def _load_sib_art(sib_dict: Dict[str, Any]) -> None:
                        try:
                            sib_rev = await asyncio.to_thread(
                                kumiho.get_revision, sib_dict["kref"],
                            )
                            artifacts = await asyncio.to_thread(sib_rev.get_artifacts)
                            for art in artifacts:
                                loc = getattr(art, "location", "")
                                if loc:
                                    text = await self._read_artifact_content(loc)
                                    if text:
                                        sib_dict["content"] = text
                                        sib_dict["artifact_location"] = loc
                                        break
                        except Exception:
                            pass

                    await asyncio.gather(*[_load_sib_art(s) for s in siblings])
                return siblings

            # --- Mode 3: BM25-light keyword overlap (default, free) ---
            query_tokens = _tokenize(query) if query else []
            for sib in siblings:
                if query_tokens:
                    text = f"{sib.get('title', '')} {sib.get('summary', '')}"
                    sib["_score"] = _token_overlap_score(query_tokens, text)
                else:
                    sib["_score"] = 0.0

            best_score = max(s["_score"] for s in siblings)

            if best_score >= self.sibling_strong_score:
                # --- Keyword mode: strong signal found ---
                # Return only siblings with meaningful overlap, sorted by
                # score.  This trims noise when there IS a lexical signal.
                strong = sorted(
                    [s for s in siblings if s["_score"] >= self.sibling_strong_score],
                    key=lambda s: s["_score"], reverse=True,
                )
                siblings = strong

                logger.debug(
                    "Sibling keyword mode for %s: %d/%d kept "
                    "(best_score=%.3f, query: %.60s)",
                    item_kref, len(siblings), total_siblings,
                    best_score, query or "<none>",
                )
            else:
                # --- Budget mode: weak/no keyword signal ---
                # Keep all siblings that fit within the char budget,
                # in chronological order for full timeline coverage.
                siblings.sort(key=lambda s: s.get("created_at") or "")
                total_chars = sum(s["_chars"] for s in siblings)

                if total_chars > self.sibling_char_budget:
                    selected: List[Dict[str, Any]] = []
                    budget_used = 0
                    for sib in siblings:
                        if budget_used + sib["_chars"] > self.sibling_char_budget:
                            continue
                        selected.append(sib)
                        budget_used += sib["_chars"]
                    siblings = selected

                    logger.debug(
                        "Sibling budget mode for %s: %d/%d kept "
                        "(%d chars of %d budget, query: %.60s)",
                        item_kref, len(siblings), total_siblings,
                        budget_used, self.sibling_char_budget,
                        query or "<none>",
                    )
                else:
                    logger.debug(
                        "Sibling pass-through for %s: all %d kept "
                        "(%d chars within %d budget)",
                        item_kref, total_siblings, total_chars,
                        self.sibling_char_budget,
                    )

            # Clean up internal keys and load artifact content for
            # surviving siblings so consumers can access full text.
            # Keep _score for downstream global ranking in context builders.
            for sib in siblings:
                sib.pop("_chars", None)

            # Load artifact content only when recall_mode is "full".
            # In "summarized" mode, title+summary is sufficient.
            if load_artifacts:
                async def _load_sibling_artifact(sib_dict: Dict[str, Any]) -> None:
                    try:
                        sib_rev = await asyncio.to_thread(
                            kumiho.get_revision, sib_dict["kref"],
                        )
                        artifacts = await asyncio.to_thread(sib_rev.get_artifacts)
                        for art in artifacts:
                            loc = getattr(art, "location", "")
                            if loc:
                                text = await self._read_artifact_content(loc)
                                if text:
                                    sib_dict["content"] = text
                                    sib_dict["artifact_location"] = loc
                                    break
                    except Exception:
                        pass  # Content stays absent; consumer falls back to summary

                await asyncio.gather(
                    *[_load_sibling_artifact(s) for s in siblings]
                )

            return siblings
        except Exception as exc:
            logger.debug(
                "Failed to fetch sibling revisions for %s: %s",
                item_kref, exc,
            )
            return []

    @staticmethod
    async def _read_artifact_content(location: str) -> str:
        """Read a local artifact file and return its text content."""
        path = Path(location)
        if not path.is_file():
            return ""
        try:
            return await asyncio.to_thread(
                path.read_text, "utf-8",
            )
        except Exception:
            return ""

    async def close(self) -> None:
        await self.redis_buffer.close()

    async def _generate_session_id(self, user_canonical_id: str, context: str) -> str:
        # A single Redis flake must not silently fork a new session: each call
        # is retried once with a short backoff, and if it still fails we log at
        # WARNING (never silent) before falling back. This never raises — the
        # no-raise guarantee callers rely on is preserved.
        async def _redis_retry(op_name, coro_factory, default):
            for attempt in range(2):
                try:
                    return await coro_factory()
                except Exception as exc:
                    if attempt == 0:
                        await asyncio.sleep(_SESSION_REDIS_RETRY_BACKOFF)
                        continue
                    logger.warning(
                        "_generate_session_id: Redis %s failed after retry (%s); "
                        "falling back — session continuity may be affected.",
                        op_name, exc,
                    )
                    return default
            return default

        # Reuse the active session when one exists (persists across restarts within a day).
        if hasattr(self.redis_buffer, "get_active_session"):
            active = await _redis_retry(
                "get_active_session",
                lambda: self.redis_buffer.get_active_session(
                    context=context,
                    user_canonical_id=user_canonical_id,
                ),
                None,
            )
            if active:
                return active

        user_hash = hashlib.sha256(user_canonical_id.encode()).hexdigest()[:10]
        date = datetime.now(timezone.utc).strftime("%Y%m%d")

        sequence = 1
        if hasattr(self.redis_buffer, "next_session_sequence"):
            sequence = await _redis_retry(
                "next_session_sequence",
                lambda: self.redis_buffer.next_session_sequence(
                    user_canonical_id=user_canonical_id,
                    date_str=date,
                ),
                1,
            )

        new_session_id = f"{context}:user-{user_hash}:{date}:{sequence:03d}"

        # Persist as the active session so follow-up restarts reuse it until consolidated.
        if hasattr(self.redis_buffer, "set_active_session"):
            await _redis_retry(
                "set_active_session",
                lambda: self.redis_buffer.set_active_session(
                    context=context,
                    user_canonical_id=user_canonical_id,
                    session_id=new_session_id,
                ),
                None,
            )

        return new_session_id


def get_memory_space(
    channel_type: str,
    *,
    project: str = "CognitiveMemory",
    team_slug: str = "",
    group_id: str = "",
) -> str:
    """Map a channel type to a Kumiho memory space path.

    This enforces session sandboxing so that memories from different
    contexts (personal DMs, team channels, group chats) don't leak
    across boundaries.

    Parameters
    ----------
    channel_type:
        One of ``"personal_dm"``, ``"team_channel"``, ``"group_dm"``.
        Unknown types default to ``"personal"``.
    project:
        Kumiho project name (default ``"CognitiveMemory"``).
    team_slug:
        Team identifier, required when ``channel_type`` is
        ``"team_channel"``.
    group_id:
        Group identifier, required when ``channel_type`` is
        ``"group_dm"``.

    Returns
    -------
    Space path string, e.g. ``"CognitiveMemory/work/team-alpha"``.
    """
    if channel_type == "team_channel":
        slug = team_slug or "default"
        return f"{project}/work/{slug}"
    if channel_type == "group_dm":
        gid = group_id or "default"
        return f"{project}/groups/{gid}"
    # personal_dm and any unknown type
    return f"{project}/personal"


def _load_default_store() -> Optional[StoreCallable]:
    try:
        from kumiho.mcp_server import tool_memory_store  # type: ignore

        return tool_memory_store
    except Exception:
        return None


def _load_default_retrieve() -> Optional[RetrieveCallable]:
    try:
        from kumiho.mcp_server import tool_memory_retrieve  # type: ignore

        return tool_memory_retrieve
    except Exception:
        return None


async def _maybe_await(func: Callable[..., Any], **kwargs: Any) -> Any:
    result = func(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
