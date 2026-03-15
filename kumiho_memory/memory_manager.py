"""Universal memory manager for AI agents."""

from __future__ import annotations

import asyncio
import inspect
import hashlib
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
from typing import Any, Awaitable, Callable, Dict, List, Optional

from kumiho_memory.privacy import PIIRedactor
from kumiho_memory.redis_memory import RedisMemoryBuffer
from kumiho_memory.retry import RetryQueue, retry_with_backoff
from kumiho_memory.summarization import LLMAdapter, MemorySummarizer

logger = logging.getLogger(__name__)


StoreCallable = Callable[..., Any]
RetrieveCallable = Callable[..., Any]


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
        store_max_retries: int = 3,
        graph_augmentation: Optional[Any] = None,
        recall_mode: str = "full",
        sibling_strong_score: float = _SIBLING_STRONG_SCORE,
        sibling_char_budget: int = _SIBLING_CHAR_BUDGET,
        sibling_similarity_threshold: float = 0.0,
        sibling_top_k: int = 0,
        embedding_adapter: Optional[Any] = None,
        sibling_score_fields: Optional[List[str]] = None,
        auto_assess_fn: Optional[AutoAssessFn] = None,
        auto_assess_min_messages: int = 3,
        auto_assess_window: int = 6,
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
        self.store_max_retries = store_max_retries
        self.graph_augmentation_config = graph_augmentation
        self._graph_recall: Optional[Any] = None  # lazy GraphAugmentedRecall
        self.recall_mode = recall_mode
        self.sibling_strong_score = sibling_strong_score
        self.sibling_char_budget = sibling_char_budget
        self.sibling_similarity_threshold = sibling_similarity_threshold
        self.sibling_top_k = sibling_top_k
        self.embedding_adapter = embedding_adapter
        self.sibling_score_fields = sibling_score_fields
        # Background memory assessor (model-agnostic, optional)
        self.auto_assess_fn: Optional[AutoAssessFn] = auto_assess_fn
        self.auto_assess_min_messages = auto_assess_min_messages
        self.auto_assess_window = auto_assess_window
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
    ) -> Dict[str, Any]:
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
        if result.get("message_count", 0) == 1:
            try:
                await self.redis_buffer.set_session_metadata(
                    self.project,
                    resolved_session_id,
                    {"user_id": user_id, "context": context},
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
        if self.auto_assess_fn is not None:
            asyncio.create_task(self._background_assess(session_id))
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

            await self._store_with_retry(**store_payload)
            logger.debug(
                "auto_assess stored memory for session %s: %s",
                session_id,
                assess_result.content[:80],
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("_background_assess error for session %s: %s", session_id, exc)

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
    ) -> Dict[str, Any]:
        ingest_result = await self.ingest_message(
            user_id=user_id,
            message=message,
            role="user",
            channel=channel,
            context=context,
            session_id=session_id,
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
    ) -> Dict[str, Any]:
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
        if not resolved_space and session_user_id:
            resolved_space = (
                f"{context}/{session_user_id}" if context else session_user_id
            )
        if not resolved_space:
            try:
                session_meta = await self.redis_buffer.get_session_metadata(
                    self.project, session_id,
                )
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
            logger.warning(
                "summarize_conversation failed for %s: %s",
                session_id,
                summarization_error,
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

            if all_attachments:
                payload["metadata"]["attachments"] = all_attachments

            if stack_revisions is not None:
                payload["stack_revisions"] = stack_revisions

            store_result = await self._store_with_retry(**payload)

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

        1. Try up to ``store_max_retries`` with exponential backoff.
        2. If all retries fail and a ``retry_queue`` is configured,
           enqueue the payload for later replay.
        3. If no queue is configured, raise the exception.
        """
        if not self.memory_store:
            return {}

        try:
            return await retry_with_backoff(
                self.memory_store,
                max_retries=self.store_max_retries,
                **payload,
            )
        except Exception as exc:
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

        Returns ``{"succeeded": N, "failed": M}``.  Items that still
        fail remain in the queue for the next flush attempt.
        """
        if not self.retry_queue or not self.memory_store:
            return {"succeeded": 0, "failed": 0}
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
        """
        if graph_augmented and self.graph_augmentation_config is not None:
            gr = self._get_graph_recall()
            if gr is not None:
                # Graph augmentation uses lightweight recall (no siblings/
                # artifacts) internally so reformulated queries don't
                # duplicate expensive work.  Sibling enrichment runs once
                # on the final merged set.
                memories = await gr.recall(
                    query,
                    limit=limit,
                    space_paths=space_paths,
                    memory_types=memory_types,
                )
                return await self._enrich_with_siblings(memories, query)

        return await self._base_recall(
            query, limit=limit, space_paths=space_paths,
            memory_types=memory_types,
        )

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
    ) -> List[Dict[str, Any]]:
        """Add sibling revisions and artifact content to pre-recalled memories.

        Runs once on the final merged set after graph augmentation has
        deduplicated results, so sibling enrichment and LLM reranking
        happen exactly once per unique item.
        """
        _load_arts = self.recall_mode == "full"

        for mem in memories:
            item_kref = mem.pop("_item_kref", "")
            if not item_kref:
                # Graph-augmented results from edge traversal don't
                # have _item_kref — skip sibling enrichment for them.
                continue

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

            siblings = await self._fetch_sibling_revision_summaries(
                item_kref, mem.get("kref", ""), query=query,
                load_artifacts=_load_arts,
            )
            if siblings:
                mem["sibling_revisions"] = siblings

        return memories

    async def _base_recall(
        self,
        query: str,
        *,
        limit: int = 5,
        space_paths: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Core vector/fulltext recall without graph augmentation.

        Performs lightweight recall then enriches with siblings in one
        pass.  Used for non-graph-augmented path.
        """
        memories = await self._lightweight_recall(
            query, limit=limit, space_paths=space_paths,
            memory_types=memory_types,
        )
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

            self._graph_recall = GraphAugmentedRecall(
                adapter=adapter,
                model=model,
                recall_fn=self._lightweight_recall,
                config=self.graph_augmentation_config,
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
            text, truncated to 4000 chars).  ``"summarized"`` uses only
            title + summary — lossy but cheaper.  Falls back to the
            instance's ``self.recall_mode`` when ``None``.
        """
        mode = recall_mode or self.recall_mode
        threshold = self.sibling_similarity_threshold

        texts: List[str] = []
        for mem in memories:
            title = mem.get("title", "")
            summary = mem.get("summary", "")
            content = mem.get("content", "")

            if mode == "full" and content:
                texts.append(content[:4000])
            elif summary:
                texts.append(f"{title}: {summary}" if title else summary)

            # Unfold sibling revisions — optionally filtered by relevance.
            siblings = mem.get("sibling_revisions", [])
            if siblings and query and threshold > 0 and self.embedding_adapter is not None:
                siblings = self._filter_siblings_by_embedding(
                    siblings, query, threshold,
                )

            for sib in siblings:
                sib_title = sib.get("title", "")
                sib_summary = sib.get("summary", "")
                sib_content = sib.get("content", "")

                if mode == "full" and sib_content:
                    texts.append(sib_content[:4000])
                elif sib_summary:
                    texts.append(
                        f"{sib_title}: {sib_summary}" if sib_title else sib_summary
                    )

        return "\n\n".join(texts) if texts else ""

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
                "type": meta.get("type", ""),
                "space": meta.get("space", ""),
                "created_at": getattr(revision, "created_at", ""),
                "tags": getattr(revision, "tags", []),
            }

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
            sib_texts.append(f"{t}: {s}" if t else s)

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

        user_msg = (
            f"User's message:\n{query}\n\n"
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

    async def _filter_siblings_by_server_search(
        self,
        siblings: List[Dict[str, Any]],
        query: str,
        item_kref: str,
    ) -> List[Dict[str, Any]]:
        """Filter siblings using LLM reranking.

        The LLM reads all sibling summaries and picks the most relevant
        ones.  This handles semantic inversion (cognitive/goal questions)
        that cosine similarity fundamentally cannot bridge.

        When the LLM reranker returns ``None`` (no relevant siblings or
        adapter unavailable), return an empty list so only the primary
        revision is used.  Dumping all siblings unfiltered into context
        is noisy and wasteful.  The previous cosine-similarity fallback
        was removed because it adds significant latency (~8s via
        ``ScoreRevisions`` RPC) without meaningfully improving on the
        LLM's judgment.
        """
        if not siblings or not query:
            return siblings

        # --- Primary: LLM reranking ---
        llm_result = await self._rerank_siblings_with_llm(siblings, query)
        if llm_result:
            logger.info(
                "LLM sibling reranker: %d/%d selected (query: %.60s)",
                len(llm_result), len(siblings), query,
            )
            return llm_result

        # LLM returned None — no relevant siblings identified.  Return
        # empty so only the primary revision is used.  Dumping all siblings
        # unfiltered would be noisy and defeat the purpose of reranking.
        logger.info(
            "LLM sibling reranker returned None, skipping all %d siblings "
            "(query: %.60s)",
            len(siblings), query,
        )
        return []

    async def _fetch_sibling_revision_summaries(
        self,
        item_kref: str,
        current_rev_kref: str,
        query: str = "",
        load_artifacts: bool = True,
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
                    # Mode 2: Server-scored hybrid search (no external API)
                    siblings = await self._filter_siblings_by_server_search(
                        siblings, query, item_kref,
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
        # Reuse the active session when one exists (persists across restarts within a day).
        if hasattr(self.redis_buffer, "get_active_session"):
            try:
                active = await self.redis_buffer.get_active_session(
                    context=context,
                    user_canonical_id=user_canonical_id,
                )
                if active:
                    return active
            except Exception:
                pass

        user_hash = hashlib.sha256(user_canonical_id.encode()).hexdigest()[:10]
        date = datetime.now(timezone.utc).strftime("%Y%m%d")

        sequence = 1
        if hasattr(self.redis_buffer, "next_session_sequence"):
            try:
                sequence = await self.redis_buffer.next_session_sequence(
                    user_canonical_id=user_canonical_id,
                    date_str=date,
                )
            except Exception:
                sequence = 1

        new_session_id = f"{context}:user-{user_hash}:{date}:{sequence:03d}"

        # Persist as the active session so follow-up restarts reuse it until consolidated.
        if hasattr(self.redis_buffer, "set_active_session"):
            try:
                await self.redis_buffer.set_active_session(
                    context=context,
                    user_canonical_id=user_canonical_id,
                    session_id=new_session_id,
                )
            except Exception:
                pass

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
