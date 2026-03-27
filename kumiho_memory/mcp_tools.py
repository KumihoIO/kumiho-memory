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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton manager
# ---------------------------------------------------------------------------

_manager: Optional[Any] = None


def _get_manager():
    """Lazily create and return a shared ``UniversalMemoryManager``."""
    global _manager
    if _manager is not None:
        return _manager

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

    summarizer = MemorySummarizer()
    buffer = RedisMemoryBuffer()
    _manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=summarizer,
        pii_redactor=PIIRedactor(),
        graph_augmentation=graph_config,
        sibling_similarity_threshold=sibling_threshold,
        embedding_adapter=embedding_adapter,
        auto_assess_fn=auto_assess_fn,
    )
    return _manager


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def tool_chat_add(args: Dict[str, Any]) -> Dict[str, Any]:
    """Add a message to Redis working memory."""
    manager = _get_manager()
    return asyncio.run(
        manager.redis_buffer.add_message(
            project=args.get("project", manager.project),
            session_id=args["session_id"],
            role=args.get("role", "user"),
            content=args["message"],
            metadata=args.get("metadata"),
        )
    )


def tool_chat_get(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get messages from Redis working memory."""
    manager = _get_manager()
    return asyncio.run(
        manager.redis_buffer.get_messages(
            project=args.get("project", manager.project),
            session_id=args["session_id"],
            limit=args.get("limit", 50),
        )
    )


def tool_chat_clear(args: Dict[str, Any]) -> Dict[str, Any]:
    """Clear a session's working memory."""
    manager = _get_manager()
    return asyncio.run(
        manager.redis_buffer.clear_session(
            args.get("project", manager.project),
            args["session_id"],
        )
    )


def tool_memory_ingest(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a user message — buffers in Redis and returns context."""
    manager = _get_manager()
    return asyncio.run(
        manager.handle_user_message(
            user_id=args["user_id"],
            message=args["message"],
            context=args.get("context", "personal"),
            session_id=args.get("session_id"),
            working_memory_limit=args.get("working_memory_limit", 10),
            recall_limit=args.get("recall_limit", 5),
        )
    )


def tool_memory_add_response(args: Dict[str, Any]) -> Dict[str, Any]:
    """Add an assistant response to the session buffer."""
    manager = _get_manager()
    return asyncio.run(
        manager.add_assistant_response(
            session_id=args["session_id"],
            response=args["response"],
        )
    )


def tool_memory_consolidate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate a session — summarize, redact PII, store to graph."""
    manager = _get_manager()
    return asyncio.run(
        manager.consolidate_session(session_id=args["session_id"])
    )


# ---------------------------------------------------------------------------
# Recall deduplication
# ---------------------------------------------------------------------------
# Models sometimes generate parallel kumiho_memory_recall calls within a
# single response despite instructions not to.  This lock serializes recall
# calls so the first one executes and any duplicate within the dedup window
# returns an empty result with a warning — giving the model nothing to
# summarize, which eliminates the duplicate "Retrieved..." output lines.

_recall_lock = threading.Lock()
_recall_cache_time: float = 0.0
_RECALL_DEDUP_WINDOW_SECS = 5.0


def tool_memory_recall(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search long-term memories by semantic query.

    Includes a deduplication guard: if called more than once within a short
    time window (e.g. parallel tool calls from the model), subsequent calls
    return an empty result with a note instead of hitting the backend again.
    """
    global _recall_cache_time

    with _recall_lock:
        now = time.monotonic()
        if now - _recall_cache_time < _RECALL_DEDUP_WINDOW_SECS:
            logger.warning(
                "kumiho_memory_recall called again within %.1fs dedup window "
                "— returning empty (new_query=%r)",
                now - _recall_cache_time,
                args.get("query", ""),
            )
            return {
                "results": [],
                "count": 0,
                "deduplicated": True,
                "note": (
                    "Duplicate recall — results already returned in this "
                    "response. Do not call kumiho_memory_recall more than "
                    "once per response."
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
        result = {"results": results, "count": len(results), "recall_mode": recall_mode}

        _recall_cache_time = time.monotonic()
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

    ds = DreamState(
        project=args.get("project", "CognitiveMemory"),
        batch_size=args.get("batch_size", 20),
        dry_run=args.get("dry_run", False),
        max_deprecation_ratio=args.get("max_deprecation_ratio", 0.5),
        allow_published_deprecation=args.get("allow_published_deprecation", False),
        summarizer=summarizer,
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
    global _recall_cache_time

    with _recall_lock:
        now = time.monotonic()
        if now - _recall_cache_time < _RECALL_DEDUP_WINDOW_SECS:
            return {
                "context": "",
                "results": [],
                "source_krefs": [],
                "count": 0,
                "deduplicated": True,
                "note": (
                    "Duplicate recall within dedup window. "
                    "Results already returned this turn."
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
        context = manager.build_recalled_context(
            results, args["query"], recall_mode
        )
        source_krefs = [m["kref"] for m in results if m.get("kref")]

        _recall_cache_time = time.monotonic()
        return {
            "context": context,
            "results": results,
            "source_krefs": source_krefs,
            "count": len(results),
            "recall_mode": recall_mode,
        }


def tool_memory_reflect(args: Dict[str, Any]) -> Dict[str, Any]:
    """Capture what matters after responding — buffers response + stores facts.

    Combines ``add_assistant_response`` + N stores + optional edge discovery
    into a single call.  The agent provides structured captures — no external
    LLM is needed for fact extraction.
    """
    manager = _get_manager()
    session_id = args["session_id"]
    response = args["response"]

    # 1. Buffer response
    buf_result = asyncio.run(
        manager.add_assistant_response(session_id=session_id, response=response)
    )
    buffered = buf_result.get("success", True)

    # 2. Store captures
    captures = args.get("captures") or []
    source_krefs = args.get("source_krefs") or []
    space_path = args.get("space_path", "")
    do_edges = args.get("discover_edges", True)
    stored_krefs: List[str] = []
    edges_total = 0

    if captures:
        from kumiho.mcp_server import tool_memory_store

        for cap in captures:
            cap_space = cap.get("space_hint", "") or space_path
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
                stack_revisions=True,
            )
            rev_kref = store_result.get("revision_kref", "")
            if rev_kref:
                stored_krefs.append(rev_kref)

            # 3. Edge discovery (best-effort, skipped if no server-side LLM)
            if do_edges and rev_kref and cap.get("type") in (
                "decision", "architecture", "implementation",
                "synthesis", "reflection",
            ):
                try:
                    edges = asyncio.run(
                        manager.discover_edges_post_consolidation(
                            revision_kref=rev_kref,
                            summary=cap.get("content", ""),
                        )
                    )
                    edges_total += len(edges)
                except Exception:
                    pass  # graceful — edge discovery is supplementary

    return {
        "buffered": buffered,
        "captures_stored": len(stored_krefs),
        "edges_discovered": edges_total,
        "stored_krefs": stored_krefs,
    }


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
            },
            "required": ["session_id", "message"],
        },
    },
    {
        "name": "kumiho_chat_get",
        "description": (
            "Retrieve recent messages from Redis working memory for a "
            "session. Returns messages, count, and TTL."
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
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "kumiho_chat_clear",
        "description": "Clear all working memory for a conversation session.",
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
            },
            "required": ["session_id"],
        },
    },
    # ── Memory lifecycle (orchestrated) ───────────────────────
    {
        "name": "kumiho_memory_ingest",
        "description": (
            "Ingest a user message into AI cognitive memory. Buffers the "
            "message in Redis, recalls relevant long-term memories, and "
            "returns working memory + long-term context. This is the main "
            "entry point for AI agents handling user messages."
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
                        "Existing session ID. Auto-generated if omitted."
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
            },
            "required": ["user_id", "message"],
        },
    },
    {
        "name": "kumiho_memory_add_response",
        "description": (
            "Add an assistant response to the session buffer in Redis "
            "working memory. Call this after generating your response."
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
            },
            "required": ["session_id", "response"],
        },
    },
    {
        "name": "kumiho_memory_consolidate",
        "description": (
            "Consolidate a conversation session into long-term memory. "
            "Summarizes the conversation with an LLM, redacts PII, writes "
            "a local artifact, and stores the summary to the Kumiho graph. "
            "The session's working memory is cleared after consolidation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier to consolidate.",
                },
            },
            "required": ["session_id"],
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
                    "default": "full",
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
                    "default": "full",
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
            "Pass source_krefs from engage to create DERIVED_FROM edges."
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
                                "description": (
                                    "Memory type: decision, preference, fact, "
                                    "correction, architecture, implementation, "
                                    "synthesis, reflection, summary, skill."
                                ),
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
                                    "Space path hint for this capture. "
                                    "Overrides top-level space_path."
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
            },
            "required": ["session_id", "response"],
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
}
