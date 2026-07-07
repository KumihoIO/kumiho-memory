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
        "CONTAINS", "CREATED_FROM", "SUPERSEDES", "SUPPORTS",
    ])
    top_k_for_traversal: int = 5
    max_total: Optional[int] = None  # Defaults to base_limit * 3
    reformulate_queries: bool = True
    traversal_timeout: int = 30  # seconds; daemon thread timeout for gRPC edge traversal
    edge_creation_timeout: int = 60  # seconds; daemon thread timeout for edge creation

    #: Enrich the merged base results with sibling revisions *before* edge
    #: traversal, then seed traversal and the semantic fallback from the
    #: top-scored flattened revisions instead of the primary items.  With
    #: revision stacking the primaries are often the same 1-2 items whose
    #: published revision carries few edges — the question-specific sibling
    #: revisions are where post-consolidation edges actually hang.  Costs
    #: sibling enrichment on the merged pool before the final cap (the
    #: enrichment is reused; the manager does not re-enrich), so it is
    #: opt-in.  No-op unless the host injects a ``sibling_fetch_fn``.
    sibling_seeded_traversal: bool = False

    #: Optional usage-accounting hook, called as ``on_llm_usage(phase, info)``
    #: after each internal LLM call.  ``phase`` is ``"recall_reformulation"``
    #: or ``"implication_queries"``; ``info`` carries ``model`` plus
    #: best-effort ``prompt_tokens`` / ``completion_tokens`` /
    #: ``total_tokens`` (zeros when the adapter does not expose a
    #: ``last_usage`` dict — see :class:`kumiho_memory.summarization.LLMAdapter`).
    #: Hook errors are swallowed; recall never breaks on accounting.
    on_llm_usage: Optional[Callable[[str, Dict[str, Any]], None]] = None


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
    sibling_fetch_fn:
        Optional async callable ``(memories, query) -> memories`` that
        attaches ``sibling_revisions`` (scored against *query*) to the merged
        base results.  When provided, it runs before edge traversal so
        traversal and the semantic fallback can seed from the top-scored
        flattened revisions (see
        :attr:`GraphAugmentationConfig.sibling_seeded_traversal`).
    """

    def __init__(
        self,
        adapter: Optional[LLMAdapter] = None,
        model: str = "",
        recall_fn: Optional[RecallFn] = None,
        config: Optional[GraphAugmentationConfig] = None,
        evidence_rerank_fn: Optional[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
        ] = None,
        sibling_fetch_fn: Optional[
            Callable[
                [List[Dict[str, Any]], str],
                Coroutine[Any, Any, List[Dict[str, Any]]],
            ]
        ] = None,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.recall_fn = recall_fn  # type: ignore[assignment]
        self.config = config or GraphAugmentationConfig()
        self._has_llm = adapter is not None
        # Deterministic score adjustment (e.g. evidence weighting) applied
        # before each result cap.  Must be idempotent — it runs at both
        # the multi-query merge slice and the final cap.
        self.evidence_rerank_fn = evidence_rerank_fn
        self.sibling_fetch_fn = sibling_fetch_fn

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
        max_total: Optional[int] = None,
        max_hops: Optional[int] = None,
        edge_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-query reformulation → parallel recall → edge traversal → merge.

        After standard vector recall, follows edges from each recalled memory
        to discover connected memories that vector similarity alone would miss.
        Falls back to multi-hop semantic recall when no graph edges are found.

        ``max_total``, ``max_hops``, and ``edge_types`` are optional per-call
        overrides of the corresponding :class:`GraphAugmentationConfig`
        fields; ``None`` (the default) keeps the configured values.
        """
        base_limit = limit
        effective_max_hops = (
            self.config.max_hops if max_hops is None else max_hops
        )
        effective_edge_types = (
            self.config.edge_types if edge_types is None else edge_types
        )

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

        # Evidence weighting BEFORE the merge slice — an official-grade
        # memory just past the unweighted boundary must survive the cut.
        merged = list(best_by_kref.values())
        if self.evidence_rerank_fn is not None:
            try:
                merged = self.evidence_rerank_fn(merged)
            except Exception as exc:
                logger.debug("evidence_rerank_fn failed at merge slice: %s", exc)
        memories = sorted(
            merged,
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

        # --- Stage 2b: Optional sibling enrichment (before traversal) ---
        # With revision stacking the primaries are often the same 1-2 items;
        # the question-specific revisions (and the edges created for them by
        # post-consolidation discovery) live in the sibling stack.  Attaching
        # scored siblings here lets traversal + fallback seed from the
        # top-scored *revisions* instead of the item shells.
        if self.sibling_fetch_fn is not None:
            try:
                memories = await self.sibling_fetch_fn(memories, query)
            except Exception as exc:
                logger.debug("sibling_fetch_fn failed, seeding from primaries: %s", exc)

        # Mark sibling krefs as seen too, so edge traversal doesn't
        # re-discover revisions we already have (no-op without siblings).
        seen_krefs: set = set()
        for mem in memories:
            kref = mem.get("kref", "")
            if kref:
                seen_krefs.add(kref)
            for sib in mem.get("sibling_revisions") or []:
                sib_kref = sib.get("kref", "")
                if sib_kref:
                    seen_krefs.add(sib_kref)

        augmented = list(memories)

        # --- Stage 3: Edge traversal via kumiho SDK ---
        # Seeds are the top-K scored revisions: flattened siblings when
        # present (question-specific), otherwise the primaries — which for
        # sibling-less memories reduces exactly to the historical
        # ``memories[:top_k]`` (the merged list is already score-descending).
        seed_krefs = _traversal_seed_krefs(
            memories, self.config.top_k_for_traversal,
        )
        graph_found = await self._traverse_edges(
            seed_krefs, seen_krefs, augmented,
            edge_types=effective_edge_types,
        )

        # --- Stage 4: Semantic fallback (when no edges found) ---
        if graph_found == 0 and effective_max_hops >= 1:
            logger.debug("No graph edges found, falling back to multi-hop semantic recall")
            from kumiho_memory.context_compose import collect_top_revisions

            secondary_terms: List[str] = []
            top_revisions = collect_top_revisions(
                memories, self.config.top_k_for_traversal,
            )
            for rev_info in top_revisions:
                title = rev_info.get("title", "")
                summary = rev_info.get("summary", "")
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

        # Evidence weighting BEFORE the final cap, applied to the BASE
        # results only.  Traversal entries carry a placeholder score 0.0
        # meaning "relevance never measured", not "zero relevance" —
        # mixing them into one score axis would let graph noise evict
        # genuinely relevant (e.g. unverified-adjusted) direct hits.  The
        # historical partition [base hits][traversal appended] is kept,
        # so the cap still trims traversal noise first.  Idempotent
        # (recomputes from base_score) with the merge-slice pass.
        if self.evidence_rerank_fn is not None:
            try:
                base_part = [m for m in augmented if not m.get("graph_augmented")]
                graph_part = [m for m in augmented if m.get("graph_augmented")]
                base_part = self.evidence_rerank_fn(base_part)
                augmented = list(base_part) + graph_part
            except Exception as exc:
                logger.debug("evidence_rerank_fn failed at final cap: %s", exc)

        # Cap to prevent context noise
        cap = max_total or self.config.max_total or (base_limit * 3)
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
            self._report_llm_usage("recall_reformulation")
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
            f"Return ONLY a JSON object like "
            f'{{"queries": ["..."]}} with {max_queries} short search queries '
            '(each 3-8 words). No explanation.\n'
            'Example: {"queries": ["feeling overwhelmed with commitments", '
            '"declining social invitations", "work-life balance stress"]}'
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
            self._report_llm_usage("implication_queries")
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

    def _report_llm_usage(self, phase: str) -> None:
        """Invoke the optional usage hook after an internal LLM call.

        Token counts come from the adapter's best-effort ``last_usage``
        attribute (set by the built-in adapters after each ``chat()``); when
        absent the hook still fires with zero counts so callers can at least
        count calls.  Hook errors are logged and swallowed — accounting must
        never break recall.
        """
        hook = getattr(self.config, "on_llm_usage", None)
        if hook is None:
            return
        info: Dict[str, Any] = {
            "model": self.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        usage = getattr(self.adapter, "last_usage", None)
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    info[key] = int(value)
        try:
            hook(phase, info)
        except Exception as exc:
            logger.debug("on_llm_usage hook failed for %s: %s", phase, exc)

    # ------------------------------------------------------------------
    # Private — edge traversal
    # ------------------------------------------------------------------

    async def _traverse_edges(
        self,
        seed_krefs: List[str],
        seen_krefs: set,
        augmented: List[Dict[str, Any]],
        edge_types: Optional[List[str]] = None,
    ) -> int:
        """Follow graph edges from the seed revisions and append connected nodes.

        Uses a daemon thread with OS-level timeout to prevent gRPC calls from
        hanging indefinitely on Windows (ProactorEventLoop doesn't process
        timer callbacks while ``to_thread`` futures are pending).
        """
        try:
            import kumiho
        except ImportError:
            logger.debug("kumiho SDK not available, skipping edge traversal")
            return 0

        edge_filter = set(
            self.config.edge_types if edge_types is None else edge_types
        )
        timeout = self.config.traversal_timeout

        # Mutable containers shared with the daemon thread.
        graph_augmented_results: List[Dict[str, Any]] = []
        traverse_result: List[int] = []

        def _sync_graph_traverse() -> int:
            """Run all synchronous gRPC calls in a plain thread."""
            found = 0
            for kref_str in seed_krefs:
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
                            entry = {
                                "kref": connected_uri,
                                "title": connected_rev.metadata.get("title", ""),
                                "summary": connected_rev.metadata.get("summary", ""),
                                "content": connected_rev.metadata.get("content", ""),
                                "score": 0.0,
                                "graph_augmented": True,
                                "edge_type": edge.edge_type,
                                "from_kref": kref_str,
                            }
                            # Evidence grade so traversal results are
                            # weightable/badgeable rather than always
                            # default-weight.
                            ev = connected_rev.metadata.get("evidence_level", "")
                            if ev:
                                entry["evidence_level"] = ev
                            src = connected_rev.metadata.get("source", "")
                            if src:
                                entry["source"] = src
                            graph_augmented_results.append(entry)
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


def _traversal_seed_krefs(
    memories: List[Dict[str, Any]],
    top_k: int,
) -> List[str]:
    """Top-*top_k* revision krefs to traverse edges from, best score first.

    Flattens ``sibling_revisions`` (skipping the primary shell when siblings
    exist — the sibling list contains all revisions of the item and their
    ``_score`` is on a different scale than the item-level recall ``score``);
    memories without siblings contribute their own kref/score.  For a
    sibling-less result set this reduces exactly to the historical
    ``memories[:top_k]`` seeding: the merged list arrives score-descending
    and the sort here is stable.
    """
    candidates: List[tuple] = []
    for mem in memories:
        siblings = mem.get("sibling_revisions") or []
        if siblings:
            for sib in siblings:
                sib_kref = sib.get("kref", "")
                if sib_kref:
                    candidates.append((sib_kref, _numeric(sib.get("_score"))))
        else:
            kref = mem.get("kref", "")
            if kref:
                candidates.append((kref, _numeric(mem.get("score"))))
    candidates.sort(key=lambda c: c[1], reverse=True)
    return [kref for kref, _ in candidates[:top_k]]


def _numeric(value: Any) -> float:
    """Coerce a score to float; non-numeric (or bool) counts as 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


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
