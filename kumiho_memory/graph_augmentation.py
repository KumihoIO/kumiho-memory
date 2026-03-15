"""Graph-augmented recall and post-consolidation edge discovery.

Ported from ``kumiho-benchmarks/kumiho_eval/common.py`` and adapted to use
the provider-agnostic ``LLMAdapter`` protocol instead of a hardcoded OpenAI
client.  This module is **optional** — when not enabled, the memory manager
falls back to standard vector/fulltext recall.

The two main capabilities are:

1. **Graph-augmented recall** — multi-query reformulation → parallel recall →
   edge traversal → semantic fallback.  Discovers connected memories that
   vector similarity alone would miss.

2. **Post-consolidation edge discovery** — after storing a memory, generates
   LLM "implication queries" (future scenarios where the memory is relevant)
   and creates edges to matching existing memories.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from kumiho_memory.summarization import (
    LLMAdapter,
    build_string_array_wrapper_schema,
)

logger = logging.getLogger(__name__)

# Type alias for the recall callable injected by the memory manager.
# It must accept (query, *, limit, space_paths, memory_types) and return a
# list of memory dicts.
RecallFn = Callable[..., Coroutine[Any, Any, List[Dict[str, Any]]]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GraphAugmentationConfig:
    """Tuneable knobs for graph-augmented recall."""

    max_hops: int = 1
    edge_types: List[str] = field(default_factory=lambda: [
        "DERIVED_FROM", "DEPENDS_ON", "REFERENCED",
        "CONTAINS", "CREATED_FROM", "SUPERSEDES",
    ])
    top_k_for_traversal: int = 5
    max_total: Optional[int] = None  # Defaults to base_limit * 3
    reformulate_queries: bool = True
    traversal_timeout: int = 30  # seconds; daemon thread timeout for gRPC edge traversal
    edge_creation_timeout: int = 60  # seconds; daemon thread timeout for edge creation


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class GraphAugmentedRecall:
    """Graph-augmented memory recall with optional edge discovery.

    Parameters
    ----------
    adapter:
        Any object implementing the ``LLMAdapter`` protocol (OpenAI-compat,
        Anthropic, custom).
    model:
        Model identifier passed to ``adapter.chat()``.  Typically the
        summarizer's ``light_model`` (e.g. ``gpt-4o-mini``).
    recall_fn:
        Async callable that performs base vector/fulltext recall.  Signature::

            async def recall_fn(query, *, limit, space_paths, memory_types) -> list[dict]

    config:
        Optional configuration overrides.
    """

    def __init__(
        self,
        adapter: Optional[LLMAdapter] = None,
        model: str = "",
        recall_fn: Optional[RecallFn] = None,
        config: Optional[GraphAugmentationConfig] = None,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.recall_fn = recall_fn  # type: ignore[assignment]
        self.config = config or GraphAugmentationConfig()
        self._has_llm = adapter is not None

    # ------------------------------------------------------------------
    # Public API — recall
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        space_paths: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-query reformulation → parallel recall → edge traversal → merge.

        After standard vector recall, follows edges from each recalled memory
        to discover connected memories that vector similarity alone would miss.
        Falls back to multi-hop semantic recall when no graph edges are found.
        """
        base_limit = limit

        # --- Stage 1: Multi-query reformulation (requires LLM) ---
        alt_queries: List[str] = []
        if self.config.reformulate_queries and self._has_llm:
            alt_queries = await self._reformulate_query(query)
        all_queries = [query] + alt_queries

        # --- Stage 2: Parallel recall + merge ---
        recall_tasks = [
            self.recall_fn(
                q, limit=base_limit,
                space_paths=space_paths, memory_types=memory_types,
            )
            for q in all_queries
        ]
        recall_results = await asyncio.gather(*recall_tasks, return_exceptions=True)

        best_by_kref: Dict[str, Dict[str, Any]] = {}
        for result in recall_results:
            if isinstance(result, BaseException):
                logger.debug("Recall query failed: %s", result)
                continue
            for mem in result:
                kref = mem.get("kref", "")
                if not kref:
                    continue
                existing = best_by_kref.get(kref)
                if existing is None or mem.get("score", 0) > existing.get("score", 0):
                    best_by_kref[kref] = mem

        memories = sorted(
            best_by_kref.values(),
            key=lambda m: m.get("score", 0),
            reverse=True,
        )[: base_limit * 2]

        if len(all_queries) > 1:
            logger.info(
                "Multi-query recall: %d queries -> %d unique memories (from %d total)",
                len(all_queries),
                len(memories),
                sum(
                    len(r) for r in recall_results
                    if not isinstance(r, BaseException)
                ),
            )

        if not memories:
            return memories

        seen_krefs: set = {m.get("kref", "") for m in memories if m.get("kref")}
        augmented = list(memories)

        # --- Stage 3: Edge traversal via kumiho SDK ---
        graph_found = await self._traverse_edges(
            memories, seen_krefs, augmented,
        )

        # --- Stage 4: Semantic fallback (when no edges found) ---
        if graph_found == 0 and self.config.max_hops >= 1:
            logger.debug("No graph edges found, falling back to multi-hop semantic recall")
            secondary_terms: List[str] = []
            for mem in memories[: self.config.top_k_for_traversal]:
                title = mem.get("title", "")
                summary = mem.get("summary", "")
                if title:
                    secondary_terms.append(title)
                elif summary:
                    secondary_terms.append(summary[:100])

            if secondary_terms:
                augmented_query = " ".join(secondary_terms)
                hop_memories = await self.recall_fn(
                    augmented_query, limit=base_limit,
                    space_paths=space_paths, memory_types=memory_types,
                )
                for mem in hop_memories:
                    kref = mem.get("kref", "")
                    if kref and kref not in seen_krefs:
                        seen_krefs.add(kref)
                        mem["graph_augmented"] = True
                        mem["hop"] = 1
                        augmented.append(mem)

        if len(augmented) > len(memories):
            logger.info(
                "Graph augmentation: %d base + %d augmented = %d total",
                len(memories), len(augmented) - len(memories), len(augmented),
            )

        # Cap to prevent context noise
        cap = self.config.max_total or (base_limit * 3)
        if len(augmented) > cap:
            logger.info("Capping augmented memories from %d to %d", len(augmented), cap)
            augmented = augmented[:cap]

        return augmented

    # ------------------------------------------------------------------
    # Public API — post-consolidation edge discovery
    # ------------------------------------------------------------------

    async def discover_edges(
        self,
        revision_kref: str,
        summary: str,
        *,
        max_queries: int = 5,
        max_edges: int = 3,
        min_score: float = 0.3,
        edge_type: str = "REFERENCED",
        space_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Discover and create edges from *revision_kref* to related memories.

        Uses the LLM to generate "implication queries" — future scenarios
        where this memory would be relevant — then searches for matching
        existing memories and creates graph edges to the top candidates.

        When *space_paths* is ``None``, the space is auto-derived from
        *revision_kref* to keep edge discovery scoped.
        """
        import kumiho

        if not revision_kref or not summary:
            return []

        if not self._has_llm:
            logger.info(
                "Edge discovery skipped for %s: no LLM adapter configured. "
                "In Claude Code/Cowork the agent can pass pre-generated "
                "implication queries via kumiho_memory_recall instead.",
                revision_kref,
            )
            return []

        # Auto-derive space scope from the kref.
        if space_paths is None:
            space_paths = _derive_space_paths(revision_kref)

        # Step 1: Generate implication queries
        queries = await self._generate_implication_queries(
            summary, max_queries=max_queries,
        )
        if not queries:
            return []

        logger.debug(
            "Edge discovery for %s: generated %d implication queries (space_paths=%s)",
            revision_kref, len(queries), space_paths,
        )

        # Step 2: Parallel recall for each query
        candidates: Dict[str, Dict[str, Any]] = {}

        async def _search_one(q: str) -> List[tuple]:
            try:
                mems = await self.recall_fn(
                    q, limit=3, space_paths=space_paths, memory_types=None,
                )
                return [(q, m) for m in mems]
            except Exception as e:
                logger.debug("Edge discovery recall failed for query %r: %s", q, e)
                return []

        search_results = await asyncio.gather(*[_search_one(q) for q in queries])
        for hits in search_results:
            for q, mem in hits:
                kref = mem.get("kref", "")
                score = mem.get("score", 0.0)
                if not kref or kref == revision_kref:
                    continue
                if score < min_score:
                    continue
                if kref not in candidates or score > candidates[kref]["score"]:
                    candidates[kref] = {
                        "memory": mem,
                        "score": score,
                        "query": q,
                    }

        if not candidates:
            logger.debug("Edge discovery: no candidates above threshold %.2f", min_score)
            return []

        # Step 3: Create edges to top-N candidates
        sorted_candidates = sorted(
            candidates.values(), key=lambda c: c["score"], reverse=True,
        )[:max_edges]

        source_rev = await _get_revision_with_retry(kumiho, revision_kref)
        if source_rev is None:
            return []

        # Edge creation uses synchronous gRPC calls that can hang on Windows.
        # Run in a daemon thread with OS-level timeout.
        timeout = self.config.edge_creation_timeout
        created_edges: List[Dict[str, Any]] = []

        def _sync_create_edges() -> None:
            for cand in sorted_candidates:
                target_kref = cand["memory"].get("kref", "")
                for attempt in range(1, 4):
                    try:
                        target_rev = kumiho.get_revision(target_kref)
                        source_rev.create_edge(
                            target_rev,
                            edge_type,
                            metadata={
                                "reason": f"LLM implication: {cand['query'][:100]}",
                                "score": str(round(cand["score"], 3)),
                            },
                        )
                        created_edges.append({
                            "source": revision_kref,
                            "target": target_kref,
                            "edge_type": edge_type,
                            "query": cand["query"],
                            "score": cand["score"],
                        })
                        logger.debug(
                            "Created edge %s -> %s (type=%s, query=%r, score=%.3f)",
                            revision_kref, target_kref, edge_type,
                            cand["query"][:60], cand["score"],
                        )
                        break
                    except Exception as e:
                        if "RESOURCE_EXHAUSTED" in str(e) and attempt < 3:
                            import time as _time
                            _time.sleep(0.05 * attempt)
                        else:
                            logger.warning(
                                "Failed to create edge %s -> %s: %s",
                                revision_kref, target_kref, e,
                            )
                            break

        done_event = threading.Event()

        def _worker() -> None:
            try:
                _sync_create_edges()
            except Exception as e:
                logger.debug("Edge creation thread error: %s", e)
            finally:
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        deadline = time.monotonic() + timeout
        while not done_event.is_set():
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.5)

        if not done_event.is_set():
            logger.warning(
                "Edge creation timed out after %ds for %s — returning %d edges created so far",
                timeout, revision_kref, len(created_edges),
            )

        return created_edges

    # ------------------------------------------------------------------
    # Private — LLM query generation
    # ------------------------------------------------------------------

    async def _reformulate_query(self, query: str) -> List[str]:
        """Generate 2-3 alternative search queries via the LLM adapter."""
        system = (
            "You generate alternative memory search queries. "
            "Given a conversational message, produce 2-3 short search queries "
            "that capture different semantic angles of what this person might "
            "be referring to from their past. Focus on:\n"
            "- The underlying emotion or concern\n"
            "- A possible causal event that led to this behavior\n"
            "- Related situations or consequences\n"
            "Return ONLY the queries, one per line, no numbering or bullets."
        )
        try:
            raw = await self.adapter.chat(
                messages=[{"role": "user", "content": query}],
                model=self.model,
                system=system,
                max_tokens=100,
            )
            queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in raw.splitlines()
                if line.strip()
            ]
            logger.info(
                "Multi-query reformulation: %d queries from trigger",
                len(queries),
            )
            return queries[:3]
        except Exception as e:
            logger.warning("Query reformulation failed: %s", e)
            return []

    async def _generate_implication_queries(
        self,
        summary: str,
        *,
        max_queries: int = 5,
    ) -> List[str]:
        """Generate search queries for scenarios where this memory is relevant."""
        prompt = (
            f"Given this conversation memory, generate {max_queries} search "
            "queries that would help find this memory in the FUTURE when "
            "someone is in a related situation.\n\n"
            "Think BEYOND the literal content. Consider:\n"
            "- What implicit constraints or decisions were established?\n"
            "- What life situations or problems would this memory be relevant to?\n"
            "- What future scenarios might this memory affect?\n"
            "- What emotional states or challenges connect to this topic?\n\n"
            f"Memory:\n{summary[:2000]}\n\n"
            f"Return ONLY a JSON array of {max_queries} short search queries "
            '(each 3-8 words). No explanation.\n'
            'Example: ["feeling overwhelmed with commitments", '
            '"declining social invitations", "work-life balance stress"]'
        )
        raw = ""
        try:
            raw = await self.adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=200,
                json_mode=build_string_array_wrapper_schema(
                    "kumiho_queries_response",
                    "queries",
                ),
            )
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned)
                cleaned = cleaned.strip()
            queries = json.loads(cleaned)
            if isinstance(queries, dict):
                queries = queries.get("queries", queries.get("items", []))
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if q][:max_queries]
        except Exception as e:
            logger.warning(
                "Failed to generate implication queries: %s (raw=%r)",
                e, raw[:200] if raw else "",
            )
        return []

    # ------------------------------------------------------------------
    # Private — edge traversal
    # ------------------------------------------------------------------

    async def _traverse_edges(
        self,
        memories: List[Dict[str, Any]],
        seen_krefs: set,
        augmented: List[Dict[str, Any]],
    ) -> int:
        """Follow graph edges from top-K memories and append connected nodes.

        Uses a daemon thread with OS-level timeout to prevent gRPC calls from
        hanging indefinitely on Windows (ProactorEventLoop doesn't process
        timer callbacks while ``to_thread`` futures are pending).
        """
        try:
            import kumiho
        except ImportError:
            logger.debug("kumiho SDK not available, skipping edge traversal")
            return 0

        edge_filter = set(self.config.edge_types)
        top_k = self.config.top_k_for_traversal
        timeout = self.config.traversal_timeout

        # Mutable containers shared with the daemon thread.
        graph_augmented_results: List[Dict[str, Any]] = []
        traverse_result: List[int] = []

        def _sync_graph_traverse() -> int:
            """Run all synchronous gRPC calls in a plain thread."""
            found = 0
            for mem in memories[:top_k]:
                kref_str = mem.get("kref", "")
                if not kref_str:
                    continue
                try:
                    rev = kumiho.get_revision(kref_str)
                    edges = rev.get_edges(direction=kumiho.BOTH)
                    for edge in edges:
                        if edge.edge_type not in edge_filter:
                            continue
                        connected_uri = (
                            edge.target_kref.uri
                            if edge.source_kref.uri == kref_str
                            else edge.source_kref.uri
                        )
                        if not connected_uri or connected_uri in seen_krefs:
                            continue
                        seen_krefs.add(connected_uri)
                        try:
                            connected_rev = kumiho.get_revision(connected_uri)
                            graph_augmented_results.append({
                                "kref": connected_uri,
                                "title": connected_rev.metadata.get("title", ""),
                                "summary": connected_rev.metadata.get("summary", ""),
                                "content": connected_rev.metadata.get("content", ""),
                                "score": 0.0,
                                "graph_augmented": True,
                                "edge_type": edge.edge_type,
                                "from_kref": kref_str,
                            })
                            found += 1
                        except Exception as e:
                            logger.debug(
                                "Failed to fetch connected revision %s: %s",
                                connected_uri, e,
                            )
                except Exception as e:
                    logger.debug("Failed to get edges for %s: %s", kref_str, e)
            return found

        done_event = threading.Event()

        def _worker() -> None:
            try:
                traverse_result.append(_sync_graph_traverse())
            except Exception as e:
                logger.debug("Graph traversal thread error: %s", e)
            finally:
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Poll from the event loop — doesn't consume thread pool threads.
        deadline = time.monotonic() + timeout
        while not done_event.is_set():
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.5)

        if done_event.is_set() and traverse_result:
            augmented.extend(graph_augmented_results)
            return traverse_result[0]

        logger.warning(
            "Graph traversal timed out after %ds — falling back to semantic recall",
            timeout,
        )
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_space_paths(revision_kref: str) -> Optional[List[str]]:
    """Extract space path from a kref URI for scoped search."""
    try:
        path_part = revision_kref.split("?")[0]
        if path_part.startswith("kref://"):
            path_part = path_part[len("kref://"):]
        segments = path_part.strip("/").split("/")
        if len(segments) >= 3:
            return ["/".join(segments[:-1])]
    except Exception:
        pass
    return None


async def _get_revision_with_retry(kumiho_mod: Any, kref: str, max_attempts: int = 3) -> Any:
    """Fetch a revision with retry on rate-limit errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return kumiho_mod.get_revision(kref)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_attempts:
                await asyncio.sleep(0.05 * attempt)
            else:
                logger.warning("Failed to get revision %s: %s", kref, e)
                return None
    return None
