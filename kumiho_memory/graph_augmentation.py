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
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

from kumiho_memory._bounded import run_bounded_in_thread
from kumiho_memory.grounding import apply_grounding_marker
from kumiho_memory.summarization import (
    LLMAdapter,
    build_string_array_wrapper_schema,
)

logger = logging.getLogger(__name__)

#: CONTRADICTS edge ``basis`` values that mean a fact-level DISPUTE (agent-
#: declared in decompose, or bridged from the evidence assessor's conflict
#: verdicts). Entity->entity CONTRADICTS relation edges from the predicate
#: registry are domain claims ("A contradicts B") — they carry ``predicate``
#: metadata and no basis, and must neither stamp ``contested_by`` nor be
#: surfaced by the dispute path. The SDK Edge object carries ``metadata``
#: (proto map), so this check reuses data already on the fetched edges.
_DISPUTE_BASES = frozenset({"agent", "evidence-assessor"})

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
        # CONTRADICTS is a belief-change edge like SUPERSEDES: the walk surfaces
        # the opposing revision the same way, and additionally derives the
        # additive ``contested_by`` recall marker from these edges (reusing the
        # seed edges the walk already fetches — no extra round-trip). SCOPED to
        # dispute edges only (metadata ``basis`` in _DISPUTE_BASES): the
        # predicate registry also mints entity->entity CONTRADICTS relation
        # edges (domain claims, ``predicate`` metadata, no basis) which are
        # neither disputes nor followed here. See ``_traverse_edges``.
        "CONTRADICTS",
    ])
    # NOTE: "ABOUT" (memory -> entity anchor, from entity_promotion.py) is
    # deliberately NOT in the generic single-hop `edge_types`. Entity anchors
    # carry no content, so a single ABOUT hop dead-ends at an empty stub.
    # Entity recall is instead a dedicated 2-hop walk (memory -> anchor
    # waypoint -> sibling memories) below, gated by `entity_recall`.
    top_k_for_traversal: int = 5

    #: Enable the entity-mediated 2-hop reader: from each recalled memory,
    #: hop through its entity anchors (ABOUT) to *other* memories about the
    #: same entities — relational recall vector similarity can't reach. The
    #: anchors are waypoints only (never returned as results, since they hold
    #: no content). False at the dataclass level (standalone users opt in
    #: explicitly); the manager turns it on together with entity promotion
    #: whenever the ontology is active — which is the DEFAULT since
    #: 2026-07-10 (opt out with KUMIHO_MEMORY_ONTOLOGY=0), based on paired
    #: same-corpus measurements showing the ontology read stack at +0.042.
    entity_recall: bool = False
    #: Cap on entity anchors followed per seed memory (fan-out guard).
    entity_recall_max_entities: int = 6
    #: Cap on sibling memories pulled in per entity anchor (fan-out guard).
    entity_recall_max_siblings: int = 5
    #: Global budget on anchor expansions across ALL seeds (a shared hub is
    #: expanded once, not per seed) so a dense graph can't blow up the walk.
    entity_recall_max_anchor_fetches: int = 24
    #: Cap slots reserved for the (score-less) 2-hop siblings when the merged
    #: result set overflows, so the walk's payload isn't always trimmed first.
    entity_recall_reserve: int = 3
    #: Registered entity->entity relation-edge traversal (ontology G1 read
    #: side). The typed relation edges decompose writes between entity anchors
    #: (USES, DEPENDS_ON, PART_OF, ..., canonicalized by predicate_registry)
    #: are read by NO recall path today; when this is on, the entity-mediated
    #: reader is EXTENDED by one hop across them to reach a NEIGHBOUR entity's
    #: memories:  seed --ABOUT--> anchor --<relation>--> neighbour anchor
    #: <--ABOUT/INVOLVES-- that neighbour's memories (INVOLVES reaches event
    #: nodes, mirroring the direct walk). Reader and writer thus share one
    #: vocabulary (the registry's canonical set) instead of only syntax
    #: (Gruber's G1). DEFAULT OFF pending pair-measured benchmarks: the extra
    #: hop adds one get_edges round-trip per distinct neighbour anchor, bounded
    #: by the relation_traversal_* caps below. Only fires with ``entity_recall``
    #: (it extends that pass); the manager sets the default from
    #: KUMIHO_MEMORY_RELATION_TRAVERSAL=1. Deliberately NOT added to the generic
    #: single-hop ``edge_types`` whitelist — that would hop the content-free
    #: anchor itself (the documented ABOUT dead-end).
    relation_traversal: bool = False
    #: Max relation edges crossed from each origin anchor. RELATES_TO (the
    #: unregistered-predicate fallback bucket, lowest semantic signal) is
    #: followed only after the specific relations, so under this cap the
    #: discriminative edges win the per-anchor budget.
    relation_traversal_max_edges_per_anchor: int = 3
    #: Global cap on distinct neighbour anchors expanded per recall — each is an
    #: extra get_edges round-trip, so this bounds the added latency.
    relation_traversal_max_neighbors: int = 4
    #: Global cap on neighbour memories pulled in via this path per recall. They
    #: ride the score-less sibling reserve (``entity_recall_reserve``), never
    #: evicting or outranking a direct hit, so this bounds the walk's work, not
    #: the final surfaced count. The ubiquitous-neighbour guard reuses
    #: ``entity_bridge_hub_degree_max``.
    relation_traversal_max_results: int = 4
    #: Entity-bridge join (multi-hop): when two or more reformulated angles
    #: have hits ABOUT the same entity, that entity is a *bridge* and its
    #: fact/event nodes are surfaced with a REAL inherited score
    #: (``factor x`` the weaker linking angle's score) — the typed-graph JOIN
    #: that vector recall cannot express. Unlike the one-sided score-less
    #: siblings of the generic walk, a bridge's relevance is vouched for by
    #: BOTH measured angles, so it competes for context like a first-class
    #: hit and survives tight context budgets. Active with ``entity_recall``.
    entity_bridge_max_results: int = 4
    entity_bridge_score_factor: float = 0.9
    #: Skip hub entities in the bridge join: an anchor with more than this
    #: many incoming ABOUT/INVOLVES edges is a ubiquitous entity (e.g. the
    #: conversation's speakers, linked from nearly every memory) — every
    #: angle reaches it, so it "bridges" everything and its facts are
    #: generic, diluting the join. The discriminative low-degree bridges are
    #: the ones that actually connect a multi-hop question's sub-facts.
    #: (Measured: conv-26 bridge join fired on 105/105 questions before this
    #: filter — the two speaker entities bridged every angle pair.)
    entity_bridge_hub_degree_max: int = 12
    #: Fact-recall leg: retrieve typed ``fact`` nodes as first-class recall
    #: candidates alongside conversations. Facts are answer-shaped atomic
    #: claims, so a single semantic hit often IS the answer a single-hop or
    #: temporal question needs. Searched once with the ORIGINAL query (not
    #: the reformulations) via a direct ``kumiho.search`` scoped to the
    #: project's facts space — the retrieve tool is bypassed on purpose:
    #: typed nodes carry no published/latest tag, so it would drop them all.
    #: On servers with the derived-kind fulltext exclusion (kumiho-server#35)
    #: this reach is vector-only — facts stay out of the lexical corpus but
    #: remain semantically recallable, exactly the intended division. Entries
    #: are additive: they ride on top of the recall cap and the context
    #: top-K (never displacing conversation hits), mirroring the measured
    #: bridge-evidence policy. Rides the ontology switch (default ON;
    #: opt out with KUMIHO_MEMORY_ONTOLOGY=0);
    #: KUMIHO_MEMORY_FACT_RECALL=0 is the measurement kill-switch.
    fact_recall: bool = False
    #: Search hits examined per query (top slice of the fact search).
    fact_recall_limit: int = 3
    #: Deeper scan window for space-scoped calls: the shared facts space
    #: mixes every source space's facts, so provenance filtering needs to
    #: look past the top slice to find in-scope candidates (still admits at
    #: most ``fact_recall_limit`` accepted facts).
    fact_recall_scan_limit: int = 24
    #: Entries appended per recall — also the on-top budget mirrored by the
    #: manager trim and the context composer.
    fact_recall_max_results: int = 2
    #: Fact entries score this × the WEAKEST base hit (axis-relative, not
    #: the server's raw fused score — a cross-encoder rerank rewrites base
    #: scores onto a different scale), so a fact never outranks any
    #: conversation on any score axis.
    fact_recall_score_factor: float = 0.9
    max_total: Optional[int] = None  # Defaults to base_limit * 3
    reformulate_queries: bool = True
    #: Number of independent angle-generation draws per recall. Oblique
    #: triggers make single-draw reformulation a per-run coin flip (three
    #: identical LoCoMo-Plus runs: 21/24/23 of 30; union of draws: 27/30) —
    #: extra draws harvest the union for one small-model call each.
    #: KUMIHO_MEMORY_REFORMULATE_DRAWS overrides via the manager wiring.
    reformulate_draws: int = 1
    #: Cap on total distinct angles kept across draws (each angle costs one
    #: recall RPC, so this bounds latency).
    reformulate_max_angles: int = 3
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
        Optional async callable ``(memories, query, alt_queries) -> memories``
        that attaches ``sibling_revisions`` (scored against *query* and every
        reformulated angle in *alt_queries*) to the merged base results.  A
        legacy 2-arg ``(memories, query)`` callable is also accepted.  When
        provided, it runs before edge traversal so traversal and the semantic
        fallback can seed from the top-scored flattened revisions (see
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

        # Per-angle top hits for the entity-bridge join (Stage 3c): angle
        # attribution must be captured here — the merge above collapses it.
        angle_hits: List[List[Tuple[str, float]]] = []
        for result in recall_results:
            if isinstance(result, BaseException):
                continue
            hits = [(m.get("kref", ""), float(m.get("score") or 0.0))
                    for m in result[:3] if m.get("kref")]
            if hits:
                angle_hits.append(hits)

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
                # Pass the reformulated angles: sibling selection must see
                # every angle of a multi-query recall, or a multi-topic
                # question keeps only its dominant topic's revisions (the
                # per-subquery enrichment this pipeline replaced scored
                # siblings against each angle).
                try:
                    memories = await self.sibling_fetch_fn(
                        memories, query, alt_queries,
                    )
                except TypeError:
                    # Host injected a 2-arg fetcher — original contract.
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

        # --- Stage 3b: Entity-bridge join (opt-in, multi-hop) ---
        # The JOIN the typed graph enables: an entity reached (ABOUT) from two
        # or more angles' top hits connects the question's sub-facts. Its
        # fact/event nodes carry a real inherited score, so they land at the
        # top of context — this runs BEFORE the generic walk so bridges also
        # claim seen_krefs first.
        if self.config.entity_recall and len(angle_hits) >= 2:
            bridge_found = await self._entity_bridge_join(
                angle_hits, seen_krefs, augmented,
            )
            if bridge_found:
                logger.info(
                    "Entity bridge join surfaced %d connecting node(s)",
                    bridge_found,
                )

        # --- Stage 3c: Fact-recall leg (opt-in, first-class facts) ---
        # Semantic retrieval of typed fact nodes with the ORIGINAL query.
        # Runs after the bridge join so bridges claim seen_krefs first (their
        # two-angle evidence outranks a single-query match on the same node),
        # and before the score-less 2-hop walk for the same reason.
        # space_paths does NOT disable the leg — facts are project-level
        # distillations, so a space-scoped call still augments from the SAME
        # project's facts space (derived from the scope; cross-project
        # isolation holds). memory_types does disable it: fact nodes carry
        # no memory_type, so a type-filtered call must not surface them.
        if self.config.fact_recall and not memory_types:
            fact_found = await self._fact_recall_leg(
                query, seen_krefs, augmented, space_paths=space_paths,
            )
            if fact_found:
                logger.info(
                    "Fact recall surfaced %d fact node(s)", fact_found,
                )

        # --- Stage 3d: Entity-mediated 2-hop reader (opt-in) ---
        # Reaches memories that share an entity but not vocabulary/embedding
        # neighborhood — the relational-recall payoff of entity promotion.
        entity_found = 0
        if self.config.entity_recall:
            entity_found = await self._traverse_entity_neighbors(
                seed_krefs, seen_krefs, augmented, query=query,
            )
            if entity_found:
                logger.debug("entity recall surfaced %d sibling node(s)", entity_found)

        # --- Stage 4: Semantic fallback (when no real edges found) ---
        # Gate on graph_found ONLY. entity_found must NOT suppress this: the
        # 2-hop siblings are score-less placeholders that trail and get trimmed
        # first, so counting them as "found" would drop the real, scored
        # fallback hits and leave strictly fewer/worse results than with the
        # feature off (a non-monotonic regression).
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
            reserve = self.config.entity_recall_reserve if self.config.entity_recall else 0
            fact_max = (
                self.config.fact_recall_max_results if self.config.fact_recall else 0
            )
            if reserve > 0 or fact_max > 0:
                # The sibling reserve rides ON TOP of the cap instead of inside
                # it: base + edge-traversal entries keep exactly the slots they
                # get with entity recall off (edges are the measured multi-hop
                # signal — LoCoMo conv-26 showed the old in-cap reserve evicted
                # ALL edge entries whenever base filled, trading the proven
                # edge payload for unproven siblings), and up to ``reserve``
                # score-less siblings are appended after. ON output is a strict
                # superset of the OFF output by construction; the manager-side
                # trim extends its target by the same reserve (memory_manager).
                # Default path (entity_recall off) keeps the exact head-slice.
                base = [m for m in augmented if not m.get("graph_augmented")]
                sib = [m for m in augmented
                       if m.get("graph_augmented") and m.get("score") is None]
                # Fact-recall entries get their own on-top budget: they carry
                # real scores but must neither displace edges/bridges from
                # the room nor be displaced by them (both are additive
                # payloads with independently measured policies).
                facts = [m for m in augmented
                         if m.get("graph_augmented") and m.get("score") is not None
                         and m.get("fact_recall")]
                edges = [m for m in augmented
                         if m.get("graph_augmented") and m.get("score") is not None
                         and not m.get("fact_recall")]
                # Bridges carry a real inherited score; plain traversal entries
                # are 0.0 placeholders. Stable sort keeps the 0.0 group in
                # arrival order while bridges jump ahead of it in the room.
                edges.sort(key=lambda m: m.get("score") or 0.0, reverse=True)
                room = max(0, cap - len(base))
                augmented = base + edges[:room] + facts[:fact_max] + sib[:reserve]
            else:
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
        # Event-driven completion (#97): the worker bridges its threading.Event
        # to the running loop via call_soon_threadsafe, so the awaiting
        # coroutine wakes the instant the daemon finishes rather than on a fixed
        # poll cadence. The old 0.5s poll set an artificial ~0.25s-avg latency
        # floor per call even though the work usually finishes in ms.
        loop = asyncio.get_running_loop()
        completed = asyncio.Event()

        def _worker() -> None:
            try:
                _sync_create_edges()
            except Exception as e:
                logger.debug("Edge creation thread error: %s", e)
            finally:
                done_event.set()
                try:
                    loop.call_soon_threadsafe(completed.set)
                except RuntimeError:
                    # Loop already closed (interpreter shutdown) — the awaiting
                    # coroutine is gone, so signalling completion is moot.
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Same deadline as the old poll loop (timeout seconds), but woken by the
        # completion event instead of re-checking every 0.5s.
        try:
            await asyncio.wait_for(completed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

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
        """Generate alternative search queries via the LLM adapter.

        With ``reformulate_draws`` > 1 the generation runs multiple times and
        the union of angles (deduped, capped at ``reformulate_max_angles``)
        is used. Measured motive (LoCoMo-Plus, 2026-07-10): for obliquely
        phrased triggers, single-draw angle generation is a per-run coin
        flip — three identical runs oscillated 21/24/23 of 30, while the
        UNION of their draws passed 27/30. Multi-draw harvests that union
        inside one run at the cost of one extra small-model call per draw.
        """
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
        draws = max(1, int(getattr(self.config, "reformulate_draws", 1)))
        max_angles = int(getattr(self.config, "reformulate_max_angles", 3))
        seen: set = set()
        queries: List[str] = []
        for _draw in range(draws):
            try:
                raw = await self.adapter.chat(
                    messages=[{"role": "user", "content": query}],
                    model=self.model,
                    system=system,
                    max_tokens=100,
                )
                self._report_llm_usage("recall_reformulation")
            except Exception as e:
                logger.warning("Query reformulation failed: %s", e)
                continue
            for line in raw.splitlines():
                q = line.strip().lstrip("0123456789.-) ")
                if not q:
                    continue
                key = " ".join(q.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                queries.append(q)
            if len(queries) >= max_angles:
                break
        logger.info(
            "Multi-query reformulation: %d queries from trigger (%d draw%s)",
            min(len(queries), max_angles), draws, "s" if draws > 1 else "",
        )
        return queries[:max_angles]

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
        # kref -> revisions it CONTRADICTS (either direction), for the additive
        # ``contested_by`` recall marker. Built from the seed edges the walk
        # already fetches below, so surfacing contestation costs no extra
        # round-trip.
        contested_map: Dict[str, List[str]] = {}

        def _mark_contested(a: str, b: str) -> None:
            # Symmetric: a CONTRADICTS edge disputes both endpoints.
            for src, dst in ((a, b), (b, a)):
                bucket = contested_map.setdefault(src, [])
                if dst not in bucket:
                    bucket.append(dst)

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
                        if not connected_uri:
                            continue
                        # Record contestation BEFORE the seen-check so both a
                        # freshly-surfaced opposing revision and one already in
                        # the base results get marked. Scoped to DISPUTE edges
                        # (basis: agent / evidence-assessor): an entity->entity
                        # CONTRADICTS relation edge (predicate metadata, no
                        # basis) is a domain claim — skip it entirely (no
                        # marker, not surfaced).
                        if edge.edge_type == "CONTRADICTS":
                            basis = str(
                                (getattr(edge, "metadata", None) or {})
                                .get("basis", "")
                            )
                            if basis not in _DISPUTE_BASES:
                                continue
                            _mark_contested(kref_str, connected_uri)
                        if connected_uri in seen_krefs:
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
                            # Grounding-staleness marker (#95): a surfaced
                            # dependent decision whose grounding fact was
                            # superseded carries the flag in metadata already
                            # fetched here — additive, zero extra round-trip
                            # (mirrors evidence_level/source above).
                            apply_grounding_marker(entry, connected_rev.metadata)
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
        # Event-driven completion (#97): bridge the daemon's threading.Event to
        # the running loop via call_soon_threadsafe and await it, instead of
        # polling every 0.5s. Traversal finishes in ms against local CE, so the
        # old poll cadence — not graph work — set the recall latency floor.
        loop = asyncio.get_running_loop()
        completed = asyncio.Event()

        def _worker() -> None:
            try:
                traverse_result.append(_sync_graph_traverse())
            except Exception as e:
                logger.debug("Graph traversal thread error: %s", e)
            finally:
                done_event.set()
                try:
                    loop.call_soon_threadsafe(completed.set)
                except RuntimeError:
                    # Loop already closed (interpreter shutdown) — nothing waits.
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Wake the instant the worker completes, or after `timeout` seconds.
        # A raw daemon thread (not an executor future) means the loop is free to
        # service this timer — the same reason the old asyncio.sleep poll worked
        # on the Windows ProactorEventLoop.
        try:
            await asyncio.wait_for(completed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        if done_event.is_set() and traverse_result:
            if contested_map:
                _attach_contested_markers(
                    augmented, graph_augmented_results, contested_map,
                )
            augmented.extend(graph_augmented_results)
            return traverse_result[0]

        logger.warning(
            "Graph traversal timed out after %ds — falling back to semantic recall",
            timeout,
        )
        return 0

    async def _entity_bridge_join(
        self,
        angle_hits: List[List[Tuple[str, float]]],
        seen_krefs: set,
        augmented: List[Dict[str, Any]],
    ) -> int:
        """Typed-graph JOIN for multi-hop: surface the nodes that CONNECT angles.

        A multi-hop question decomposes into angles; the evidence that answers
        it is the pair of facts linked by a shared entity ("who adopted the
        dog" ⋈ "where does that person work" join on the person). For every
        entity anchor reached via ``ABOUT`` from the top hits of two or more
        DIFFERENT angles, surface that anchor's fact/event nodes with a real
        inherited score (``entity_bridge_score_factor`` × the weaker linking
        angle's score). The bridge's relevance is vouched for by both measured
        angles — unlike the generic walk's one-sided score-less siblings — so
        these entries compete for context as first-class hits and survive
        tight context budgets. Fact/event nodes are preferred over whole
        conversations: they are the terse atomic claims a multi-hop answer
        actually needs.
        """
        try:
            import kumiho
        except ImportError:
            logger.debug("kumiho SDK not available, skipping entity bridge")
            return 0

        factor = self.config.entity_bridge_score_factor
        max_results = self.config.entity_bridge_max_results
        hub_max = self.config.entity_bridge_hub_degree_max
        results: List[Dict[str, Any]] = []

        def _kind(kref: str) -> str:
            head = kref.split("?", 1)[0]
            return head.rsplit(".", 1)[-1] if "." in head else ""

        # Terse atomic claims first; whole conversations are the fallback.
        kind_priority = {"fact": 0, "event": 1, "decision": 2}

        def _sync_join() -> int:
            # anchor kref -> {angle index: best linking score}
            anchor_angles: Dict[str, Dict[int, float]] = {}
            for ai, hits in enumerate(angle_hits):
                for kref_str, score in hits:
                    if not kref_str:
                        continue
                    try:
                        rev = kumiho.get_revision(kref_str)
                        for edge in rev.get_edges(direction=kumiho.BOTH):
                            if edge.edge_type != "ABOUT":
                                continue
                            if edge.source_kref.uri != kref_str:
                                continue  # only outgoing memory -> anchor
                            anchor = edge.target_kref.uri
                            best = anchor_angles.setdefault(anchor, {})
                            if score > best.get(ai, 0.0):
                                best[ai] = score
                    except Exception as exc:
                        logger.debug(
                            "entity bridge: edges for %s failed: %s", kref_str, exc,
                        )

            # A bridge = an anchor linked from >=2 distinct angles. Rank by
            # the WEAKER of its two best angle scores: both sides must be
            # genuinely relevant for the join to mean anything.
            bridges: List[Tuple[float, str]] = []
            for anchor, per_angle in anchor_angles.items():
                if len(per_angle) < 2:
                    continue
                top2 = sorted(per_angle.values(), reverse=True)[:2]
                bridges.append((top2[1], anchor))
            bridges.sort(reverse=True)

            found = 0

            def _surface(link_score: float, anchor: str,
                         candidates: List[Tuple[int, str, str]]) -> None:
                nonlocal found
                candidates.sort(key=lambda c: c[0])
                for _prio, src, etype in candidates[:2]:  # <=2 nodes/bridge
                    if found >= max_results:
                        return
                    seen_krefs.add(src)
                    try:
                        src_rev = kumiho.get_revision(src)
                        results.append({
                            "kref": src,
                            "title": src_rev.metadata.get("title", ""),
                            "summary": src_rev.metadata.get("summary", ""),
                            "content": src_rev.metadata.get("content", ""),
                            # Real score, inherited from the weaker angle:
                            # this is measured relevance by proxy, not the
                            # unmeasured placeholder of the generic walk.
                            "score": round(link_score * factor, 6),
                            "graph_augmented": True,
                            "bridge": True,
                            "edge_type": etype,
                            "via_entity": anchor,
                            "hop": 2,
                        })
                        found += 1
                    except Exception as exc:
                        logger.debug(
                            "entity bridge: node %s failed: %s", src, exc,
                        )

            # Discriminative (low-degree) bridges surface first; hub anchors
            # are DEFERRED, not dropped. In a speaker-centric corpus (a
            # 2-person chat) the hub IS the join key — a hard skip cost
            # multi-hop −0.078 on conv-26, while in multi-entity corpora the
            # discriminative bridges are the signal. Preference covers both.
            deferred: List[Tuple[float, str, List[Tuple[int, str, str]]]] = []
            for link_score, anchor in bridges[:8]:
                if found >= max_results:
                    break
                try:
                    anchor_rev = kumiho.get_revision(anchor)
                    incoming = 0
                    candidates: List[Tuple[int, str, str]] = []
                    for edge in anchor_rev.get_edges(direction=kumiho.BOTH):
                        if edge.edge_type not in ("ABOUT", "INVOLVES"):
                            continue
                        if edge.target_kref.uri != anchor:
                            continue  # only incoming node -> anchor
                        incoming += 1
                        src = edge.source_kref.uri
                        if not src or src in seen_krefs:
                            continue
                        candidates.append(
                            (kind_priority.get(_kind(src), 3), src, edge.edge_type),
                        )
                    if incoming > hub_max:
                        deferred.append((link_score, anchor, candidates))
                        continue
                    _surface(link_score, anchor, candidates)
                except Exception as exc:
                    logger.debug(
                        "entity bridge: anchor %s failed: %s", anchor, exc,
                    )
            # Hub fallback: fill any remaining budget from the deferred hubs
            # (edge candidates were already collected during the degree scan).
            for link_score, anchor, candidates in deferred:
                if found >= max_results:
                    break
                _surface(link_score, anchor, candidates)
            return found

        found = await run_bounded_in_thread(
            _sync_join,
            timeout=self.config.traversal_timeout,
            label="entity bridge",
            on_timeout=0,
            on_error=0,
        ) or 0
        if found:
            augmented.extend(results)
        return found

    async def _fact_recall_leg(
        self,
        query: str,
        seen_krefs: set,
        augmented: List[Dict[str, Any]],
        space_paths: Optional[List[str]] = None,
    ) -> int:
        """Surface typed ``fact`` nodes as first-class semantic candidates.

        One direct ``kumiho.search`` with the original query, scoped to the
        project's facts space and kind-filtered to ``fact``. Direct search —
        not the retrieve tool — because typed nodes carry no published/latest
        tag and the tool silently drops untagged items. Entries score
        ``fact_recall_score_factor`` × the WEAKEST base hit — relative to
        the base axis, not the server's, because a cross-encoder rerank
        rewrites base scores onto a different scale and a raw fused score
        would then sort every fact above every conversation. Facts share one
        trailing score (stable sorts keep their search order) and are
        flagged ``fact_recall`` so the cap here and the context composer
        keep them additive (they never evict or outrank conversation hits).
        """
        try:
            import kumiho
        except ImportError:
            logger.debug("kumiho SDK not available, skipping fact recall")
            return 0

        # Scope: fact nodes live in the project's dedicated facts space
        # (ontology.py). Derive the project from the caller's space_paths
        # when given (a scoped call stays inside its own project), else from
        # the recalled memories themselves.
        project = ""
        for sp in space_paths or []:
            head = sp.lstrip("/").split("/", 1)[0].strip()
            if head:
                project = head
                break
        if not project:
            for m in augmented:
                kref = m.get("kref", "")
                if kref.startswith("kref://"):
                    project = kref[len("kref://"):].split("/", 1)[0]
                    break
        if not project:
            return 0

        base_scores = [
            m.get("score") or 0.0
            for m in augmented if not m.get("graph_augmented")
        ]
        floor = min(base_scores) if base_scores else 0.0
        fact_score = round(floor * self.config.fact_recall_score_factor, 6)
        max_results = self.config.fact_recall_max_results
        limit = self.config.fact_recall_limit
        # Provenance scope: the project's facts space is SHARED across every
        # source space (multi-user projects, multi-corpus benchmarks — a
        # LoCoMo-Plus project measured 2,511 mixed facts), so a space-scoped
        # call must only accept facts whose DERIVED_FROM source conversation
        # lives inside the caller's space_paths. Without this, the top-slice
        # is diluted by other spaces' facts: both misses in the 2026-07-10
        # Plus regression check had their exact answer stored as a fact that
        # cross-space crowding kept out of the slice.
        space_prefixes = tuple(
            "kref://" + sp.strip().lstrip("/").rstrip("/") + "/"
            for sp in (space_paths or []) if sp and sp.strip("/")
        )
        scan_limit = limit if not space_prefixes else max(
            limit, self.config.fact_recall_scan_limit,
        )
        # Snapshot for the worker thread: it must not read or mutate the
        # shared set — a timed-out search would otherwise keep claiming
        # krefs that never get emitted (poisoning the 2-hop walk after us).
        known = set(seen_krefs)
        results: List[Dict[str, Any]] = []

        def _from_caller_space(rev) -> bool:
            """True when the fact's source conversation is inside the scope."""
            if not space_prefixes:
                return True
            try:
                for edge in rev.get_edges(direction=kumiho.BOTH):
                    if edge.edge_type != "DERIVED_FROM":
                        continue
                    src = getattr(getattr(edge, "source_kref", None), "uri", "")
                    if src != getattr(getattr(rev, "kref", None), "uri", ""):
                        continue  # only the fact's own outgoing provenance
                    target = getattr(getattr(edge, "target_kref", None), "uri", "")
                    if target.startswith(space_prefixes):
                        return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("fact provenance check failed: %s", exc)
            return False

        def _sync_search() -> int:
            try:
                hits = kumiho.search(
                    query,
                    context=f"{project}/facts",
                    kind="fact",
                    include_revision_metadata=True,
                )
            except Exception as exc:
                logger.debug("fact recall search failed: %s", exc)
                return 0
            found = 0
            accepted_in_slice = 0
            for r in (hits or [])[:scan_limit]:
                if found >= max_results:
                    break
                # Keep the historical semantics for unscoped calls: only the
                # top ``limit`` hits compete. Scoped calls scan deeper, but
                # still admit at most ``limit`` provenance-accepted facts.
                if accepted_in_slice >= limit:
                    break
                item = getattr(r, "item", None)
                if item is None:
                    continue
                try:
                    rev = item.get_latest_revision()
                except Exception:
                    continue
                if rev is None:
                    continue
                if not _from_caller_space(rev):
                    continue
                accepted_in_slice += 1
                kref = getattr(getattr(rev, "kref", None), "uri", "") or ""
                if not kref or kref in known:
                    continue
                meta = getattr(rev, "metadata", {}) or {}
                known.add(kref)
                results.append({
                    "kref": kref,
                    "title": meta.get("title", ""),
                    "summary": meta.get("summary", "") or meta.get("claim", ""),
                    "content": meta.get("claim", ""),
                    "score": fact_score,
                    "graph_augmented": True,
                    "fact_recall": True,
                    "hop": 1,
                })
                found += 1
            return found

        found = await run_bounded_in_thread(
            _sync_search,
            timeout=self.config.traversal_timeout,
            label="fact recall",
            on_timeout=0,
            on_error=0,
        ) or 0
        if found:
            # Claim krefs only for entries that actually get emitted — after
            # the bounded wait, never from the abandonable worker thread.
            seen_krefs.update(e["kref"] for e in results)
            augmented.extend(results)
        return found

    async def _traverse_entity_neighbors(
        self,
        seed_krefs: List[str],
        seen_krefs: set,
        augmented: List[Dict[str, Any]],
        query: str = "",
    ) -> int:
        """Entity-mediated 2-hop walk: memory → entity anchor → sibling memory.

        Hop 1 follows ``ABOUT`` from each seed memory to its entity anchors;
        hop 2 follows ``ABOUT`` *into* each anchor to reach the other memories
        about that entity. Anchors are pure waypoints — they are never
        appended to results (they carry only ``display_name``), so recall
        context is enriched with sibling *memories*, never empty stubs. That
        is the crucial difference from putting ``ABOUT`` in the generic
        single-hop set, which would surface the anchor itself.

        When ``relation_traversal`` is on this pass is EXTENDED by one more
        hop across the typed entity→entity relation edges decompose writes
        (USES, DEPENDS_ON, ..., canonicalized by ``predicate_registry``):
        ``seed → anchor → <relation> → neighbour anchor → that neighbour's
        memories``. The relation edges are read from the SAME canonical set the
        writer folds onto, closing the G1 writer/reader vocabulary split.
        Performance: this adds one ``get_edges`` round-trip per distinct
        neighbour anchor — the ``relation_traversal_*`` caps bound the fan-out,
        which is why the flag defaults OFF pending pair-measured benchmarks.
        """
        try:
            import kumiho
        except ImportError:
            logger.debug("kumiho SDK not available, skipping entity recall")
            return 0

        max_entities = self.config.entity_recall_max_entities
        max_siblings = self.config.entity_recall_max_siblings
        max_anchor_fetches = self.config.entity_recall_max_anchor_fetches
        results: List[Dict[str, Any]] = []

        # Relation-edge traversal (ontology G1 read side), default OFF. Import
        # the registry lazily and only when the flag is on — mirrors this
        # module's import-at-use idiom and keeps the predicate_registry→ontology
        # chain off the default hot path.
        relation_on = getattr(self.config, "relation_traversal", False)
        canonical_set: frozenset = frozenset()
        relates_to = ""
        if relation_on:
            from kumiho_memory.predicate_registry import (
                RELATES_TO as relates_to,
                canonical_types,
            )
            canonical_set = frozenset(canonical_types())
        rel_max_edges = self.config.relation_traversal_max_edges_per_anchor
        rel_max_neighbors = self.config.relation_traversal_max_neighbors
        rel_max_results = self.config.relation_traversal_max_results
        hub_max = self.config.entity_bridge_hub_degree_max

        def _kind(kref: str) -> str:
            head = kref.split("?", 1)[0]
            return head.rsplit(".", 1)[-1] if "." in head else ""

        # Terse atomic claims first (mirrors the bridge join); whole
        # conversations are the fallback when pulling a neighbour's memories.
        kind_priority = {"fact": 0, "event": 1, "decision": 2}

        def _sync_entity_walk() -> int:
            found = 0
            visited_anchors: set = set()  # a shared hub is expanded once, not per seed
            anchor_fetches = 0            # global fan-out budget across all seeds
            # (relation_type, neighbour_uri, origin_anchor, seed_kref) noted
            # during the direct scan; expanded in a second phase so every direct
            # sibling is already claimed (direct provenance wins on overlap).
            relation_candidates: List[Tuple[str, str, str, str]] = []
            for kref_str in seed_krefs:
                if not kref_str:
                    continue
                if anchor_fetches >= max_anchor_fetches:
                    break
                try:
                    rev = kumiho.get_revision(kref_str)
                    # Hop 1: this memory -> entity anchors (memory is source).
                    anchor_uris: List[str] = []
                    for edge in rev.get_edges(direction=kumiho.BOTH):
                        if edge.edge_type != "ABOUT":
                            continue
                        if edge.source_kref.uri == kref_str:
                            anchor_uris.append(edge.target_kref.uri)
                        if len(anchor_uris) >= max_entities:
                            break
                except Exception as exc:
                    logger.debug("entity recall: edges for %s failed: %s", kref_str, exc)
                    continue

                # Hop 2: each anchor -> sibling nodes (anchor is target).
                for anchor_uri in anchor_uris:
                    if not anchor_uri or anchor_uri in visited_anchors:
                        continue  # dedup shared hubs across seeds
                    visited_anchors.add(anchor_uri)
                    if anchor_fetches >= max_anchor_fetches:
                        break
                    anchor_fetches += 1
                    try:
                        anchor_rev = kumiho.get_revision(anchor_uri)
                        siblings = 0
                        rel_here: List[Tuple[str, str]] = []
                        sibs_capped = False
                        for edge in anchor_rev.get_edges(direction=kumiho.BOTH):
                            etype = edge.edge_type
                            # Relation traversal: note the entity->entity relation
                            # edges from this anchor (either direction) for the
                            # neighbour-expansion phase. Flag off ⇒ this block is
                            # never entered and the walk is byte-identical.
                            if relation_on and etype in canonical_set:
                                if edge.source_kref.uri == anchor_uri:
                                    neighbour = edge.target_kref.uri
                                elif edge.target_kref.uri == anchor_uri:
                                    neighbour = edge.source_kref.uri
                                else:
                                    neighbour = ""
                                if neighbour and neighbour != anchor_uri:
                                    rel_here.append((etype, neighbour))
                                continue
                            if sibs_capped:
                                continue  # scanning only for relation edges now
                            # Reach siblings via ABOUT (memories, facts, decisions,
                            # actions) AND INVOLVES (event nodes, which carry the
                            # distilled event_date) — both point *into* the anchor.
                            if etype not in ("ABOUT", "INVOLVES"):
                                continue
                            if edge.target_kref.uri != anchor_uri:
                                continue  # only incoming node -> anchor
                            sib_uri = edge.source_kref.uri
                            if not sib_uri or sib_uri in seen_krefs:
                                continue
                            seen_krefs.add(sib_uri)
                            try:
                                sib_rev = kumiho.get_revision(sib_uri)
                                # Score-less placeholder: this node's relevance to
                                # the actual query was never measured. Omitting
                                # `score` (not score=0.0) makes recall_rerank treat
                                # it as unscored, so it trails the scored hits and
                                # is never evidence-reweighted into evicting one.
                                # For the same reason no evidence_level/source is
                                # copied — that belongs to a matched hit, not a hub
                                # neighbour.
                                entry = {
                                    "kref": sib_uri,
                                    "title": sib_rev.metadata.get("title", ""),
                                    "summary": sib_rev.metadata.get("summary", ""),
                                    "content": sib_rev.metadata.get("content", ""),
                                    "graph_augmented": True,
                                    "edge_type": edge.edge_type,
                                    "via_entity": anchor_uri,
                                    "from_kref": kref_str,
                                    "hop": 2,
                                }
                                results.append(entry)
                                found += 1
                                siblings += 1
                            except Exception as exc:
                                logger.debug("entity recall: sibling %s failed: %s", sib_uri, exc)
                            if siblings >= max_siblings:
                                # The sibling cap must not short-circuit relation
                                # collection — relation edges can arrive AFTER the
                                # cap-filling sibling (server edge order is
                                # arbitrary). Flag off keeps the historical early
                                # exit byte-identical.
                                if not relation_on:
                                    break
                                sibs_capped = True
                        # Queue this anchor's relation neighbours. RELATES_TO is
                        # the fallback bucket for unregistered predicates —
                        # lowest priority, so the specific relations win the
                        # per-anchor budget under the cap.
                        if relation_on and rel_here:
                            specific = [c for c in rel_here if c[0] != relates_to]
                            fallback = [c for c in rel_here if c[0] == relates_to]
                            for etype_c, neighbour_c in (
                                specific + fallback
                            )[:rel_max_edges]:
                                relation_candidates.append(
                                    (etype_c, neighbour_c, anchor_uri, kref_str),
                                )
                    except Exception as exc:
                        logger.debug("entity recall: anchor %s failed: %s", anchor_uri, exc)

            # --- Relation-neighbour expansion (relation_traversal only) ---
            # Second phase, after all direct siblings are claimed: a memory
            # reachable BOTH directly and via a relation edge keeps its direct
            # (hop-2) provenance because the dedup below skips already-seen
            # krefs. Each neighbour anchor is one extra get_edges round-trip;
            # the caps bound the fan-out.
            if relation_on and relation_candidates:
                # Global specific-over-fallback ordering: the per-anchor slice
                # alone lets an EARLIER anchor's RELATES_TO consume the global
                # caps ahead of a later anchor's specific relation. Stable
                # partition — arrival order is preserved within each class.
                relation_candidates.sort(key=lambda c: c[0] == relates_to)
                rel_found = 0
                neighbours_expanded = 0
                for (relation_type, neighbour_uri,
                     origin_anchor, seed_kref) in relation_candidates:
                    if rel_found >= rel_max_results:
                        break
                    if neighbours_expanded >= rel_max_neighbors:
                        break
                    if not neighbour_uri or neighbour_uri in visited_anchors:
                        continue  # already a direct anchor / already expanded
                    # Relation edges join entity anchors; a canonical-typed edge
                    # to a non-entity would be a stray edge whose single hop
                    # dead-ends — require an entity neighbour.
                    if _kind(neighbour_uri) != "entity":
                        continue
                    visited_anchors.add(neighbour_uri)
                    neighbours_expanded += 1
                    try:
                        neighbour_rev = kumiho.get_revision(neighbour_uri)
                    except Exception as exc:
                        logger.debug(
                            "relation recall: neighbour %s failed: %s",
                            neighbour_uri, exc,
                        )
                        continue
                    # One pass over the neighbour's edges: collect its incoming
                    # ABOUT/INVOLVES memories AND count them for the
                    # ubiquitous-entity guard (a hub neighbour's memories are
                    # generic noise — same rationale as the bridge join's hub
                    # cutoff, reusing entity_bridge_hub_degree_max).
                    incoming = 0
                    cands: List[Tuple[int, str, str]] = []
                    try:
                        for edge in neighbour_rev.get_edges(direction=kumiho.BOTH):
                            if edge.edge_type not in ("ABOUT", "INVOLVES"):
                                continue
                            if edge.target_kref.uri != neighbour_uri:
                                continue  # only incoming node -> neighbour anchor
                            incoming += 1
                            sib_uri = edge.source_kref.uri
                            if not sib_uri or sib_uri in seen_krefs:
                                continue
                            cands.append(
                                (kind_priority.get(_kind(sib_uri), 3),
                                 sib_uri, edge.edge_type),
                            )
                    except Exception as exc:
                        logger.debug(
                            "relation recall: neighbour edges %s failed: %s",
                            neighbour_uri, exc,
                        )
                        continue
                    if incoming > hub_max:
                        continue  # ubiquitous neighbour — its memories are generic
                    cands.sort(key=lambda c: c[0])  # fact/event/decision first
                    sib_here = 0
                    for _prio, sib_uri, sib_edge_type in cands:
                        if rel_found >= rel_max_results or sib_here >= max_siblings:
                            break
                        if sib_uri in seen_krefs:
                            continue
                        seen_krefs.add(sib_uri)
                        try:
                            sib_rev = kumiho.get_revision(sib_uri)
                        except Exception as exc:
                            logger.debug(
                                "relation recall: sibling %s failed: %s",
                                sib_uri, exc,
                            )
                            continue
                        # Score-less like the direct siblings (never evicts a
                        # scored hit), plus relation provenance so downstream
                        # context building can show HOW it was reached: the
                        # relation crossed and the intermediate entity.
                        results.append({
                            "kref": sib_uri,
                            "title": sib_rev.metadata.get("title", ""),
                            "summary": sib_rev.metadata.get("summary", ""),
                            "content": sib_rev.metadata.get("content", ""),
                            "graph_augmented": True,
                            "edge_type": sib_edge_type,
                            "via_entity": neighbour_uri,
                            "via_relation": relation_type,
                            "relation_from_entity": origin_anchor,
                            "from_kref": seed_kref,
                            "hop": 3,
                        })
                        found += 1
                        rel_found += 1
                        sib_here += 1
            return found

        found = await run_bounded_in_thread(
            _sync_entity_walk,
            timeout=self.config.traversal_timeout,
            label="entity recall",
            on_timeout=0,
            on_error=0,
        ) or 0
        if found:
            # Order siblings by lexical overlap with the query so the cap's
            # sibling reserve keeps the MOST relevant ones — walk order is
            # seed × anchor × edge arrival, arbitrary w.r.t. the question.
            # Ordering only: the entries stay score-less, so they still trail
            # scored hits and are never evidence-reweighted (see the
            # placeholder note above).
            if query:
                from kumiho_memory.recall_rerank import _jaccard, _tokens

                q_tokens = _tokens(query)
                results.sort(
                    key=lambda m: _jaccard(
                        q_tokens,
                        _tokens(f"{m.get('title', '')} {m.get('summary', '')}"),
                    ),
                    reverse=True,
                )
            augmented.extend(results)
            logger.info("Entity recall: +%d sibling memories via entity anchors", found)
        return found


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attach_contested_markers(
    base: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]],
    contested_map: Dict[str, List[str]],
    limit: int = 3,
) -> None:
    """Stamp a bounded ``contested_by`` field on entries with CONTRADICTS edges.

    Purely additive — no entry removed, no score changed, no reordering. A
    graph-walk entry matches on its own kref; a base memory matches on its own
    kref OR any sibling revision kref (a CONTRADICTS edge hangs off a specific
    revision, which may be a sibling rather than the item shell). ``limit``
    bounds the surfaced disputing krefs so a heavily-contested memory can't
    balloon the entry.
    """
    for entry in graph_results:
        marks = contested_map.get(entry.get("kref", ""))
        if marks:
            entry["contested_by"] = marks[:limit]
    for mem in base:
        marks = list(contested_map.get(mem.get("kref", ""), []))
        for sib in mem.get("sibling_revisions") or []:
            for k in contested_map.get(sib.get("kref", ""), []):
                if k not in marks:
                    marks.append(k)
        if marks:
            mem["contested_by"] = marks[:limit]


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
