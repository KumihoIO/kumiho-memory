"""Dream State graph maintenance — consolidate the *typed* graphs.

The base Dream State pass assesses flat ``conversation`` revisions.  This
module extends the same consolidation cycle to the two typed graphs that
Dream State never touched (issue #59):

* the conversation **ontology** — ``entity`` / ``fact`` nodes and their
  ``ABOUT`` / ``DERIVED_FROM`` / typed-relation edges (``ontology.py``);
* the **Decision Memory** graph — ``code_decision`` / ``code_evidence``
  nodes in the dedicated ``{project}-code`` project (``code_decisions.py``).

Two passes, by the hard keyless constraint (the plugin path must NEVER use
an external LLM key):

* **KEYLESS deterministic** (:meth:`GraphMaintainer.run_keyless`) — every
  judgment is derived from slugs, edges, and stored atoms, so it runs with
  no model:

    - *entity merge* — fold a duplicate entity into the hub that already
      lists it as an alias (slug identity), repointing its edges;
    - *fact dedup* — collapse near-duplicate facts about the same entity
      (token-Jaccard), the belief-revision analog of the write-time
      SUPERSEDES pass;
    - *orphan prune* — deprecate entities with no edges at all;
    - *evidence re-grade* — recompute a ``code_decision``'s
      ``evidence_level`` from its **current** ``MOTIVATED_BY`` atoms via
      the same deterministic :func:`code_capture._evidence_grade`, and
      lift it (``unverified`` → ``corroborated``) when evidence accrued
      after capture.  This closes the LoE auto-upgrade gap flagged in #6:
      the grade was stamped once at capture and never rose;
    - *decision dedup* — sink a near-identical decision under its twin via
      ``SUPERSEDES`` + a ``status=superseded`` demotion the query side
      already honors;
    - *cross-graph bridge* — draw ``ABOUT`` edges from a ``code_decision``
      to the conversation ``entity`` it is about (symbol/slug identity),
      unifying the two graphs at the graph level (#57 "one brain").

* **OPTIONAL LLM** (:meth:`GraphMaintainer.apply_entity_merges`, fed by
  :class:`~kumiho_memory.dream_state.DreamState`) — semantic entity-merge
  suggestions the deterministic alias rule can't see ("Postgres" vs
  "PostgreSQL" with no alias metadata) are *applied through the same
  keyless write path*; only the *suggestion* uses a model.

Safety (reused, not reinvented): ``dry_run`` computes and counts without
writing; a shared deprecation budget (``max_deprecation_ratio``) caps the
destructive passes; every edge write is idempotent via an existing-edge
precheck; ``evidence:official`` grades are never rewritten (operator-owned).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kumiho._text import slugify

from kumiho_memory.code_capture import _evidence_grade
from kumiho_memory.code_decisions import (
    EDGE_MOTIVATED_BY,
    EDGE_SUPERSEDES,
    KIND_DECISION,
)
from kumiho_memory.evidence import (
    CORROBORATED,
    EVIDENCE_LEVELS,
    OFFICIAL,
    SINGLE_SOURCE,
    UNVERIFIED,
    evidence_tag,
    parse_evidence,
)
from kumiho_memory.grounding import (
    GROUNDING_STALE_META,
    GROUNDING_STALE_SUPERSEDED_BY_META,
    GROUNDING_STALE_TAG,
    is_grounding_stale,
)
from kumiho_memory.ontology import OntologySchema, _mentions, _word_tokens
from kumiho_memory.relations import _jaccard, _tokens

logger = logging.getLogger(__name__)

# Evidence grades ranked least → most trustworthy; re-grade lifts only.
_GRADE_RANK: Dict[str, int] = {
    UNVERIFIED: 0,
    SINGLE_SOURCE: 1,
    CORROBORATED: 2,
    OFFICIAL: 3,
}

#: Near-duplicate cut for *facts about the same entity*.  Deliberately high:
#: a wrong merge is unrecoverable (echoing ``get_or_create_decision_item``'s
#: collision guard), so only near-identical claims collapse; distinct facts
#: are left for the SUPERSEDES/query layers.
_FACT_DEDUP_JACCARD = 0.75

#: Near-identical cut for *code decisions*.  Higher still — two decisions are
#: only folded when their prose is almost the same; genuine belief updates
#: (lower overlap) are the write-time SUPERSEDES pass's job, not dedup's.
_DECISION_DEDUP_JACCARD = 0.82

#: Safety valve on the O(n²) dedup scans — a nightly maintenance over a
#: bounded typed graph never approaches this, but a pathological space must
#: not wedge the run.  Truncation is logged (never silent).
_MAX_DEDUP_NODES = 400

#: Embedding-assisted fact-dedup candidate stage (ontology G6, opt-in via
#: ``KUMIHO_DREAM_EMBED_FACT_DEDUP``).  Server-side vector scoring
#: (``score_revisions`` — no client embeddings, no LLM key) WIDENS candidate
#: pairing beyond a single entity's ``ABOUT`` fan-in, catching paraphrase
#: duplicates filed under different entities that the lexical per-entity scan
#: never compares.  The merge itself still routes through the SAME
#: ``_FACT_DEDUP_JACCARD`` confirmation + fact budget + published protection, so
#: the unrecoverable-merge safety threshold is never loosened — embeddings only
#: nominate, they never authorize.  Bounds honor the typed-node vector-crowding
#: constraint (small k, kind=fact filter, capped query fan-out).
_EMBED_FACT_DEDUP_ENV = "KUMIHO_DREAM_EMBED_FACT_DEDUP"
#: Facts used as a vector query per run (each costs one ``score_revisions`` RPC).
_EMBED_MAX_QUERY_FACTS = 50
#: Nearest neighbours examined per query fact — deliberately SMALL (k), so the
#: stage samples the densest duplicate clusters without flooding recall.
_EMBED_TOP_K = 5
#: Server-similarity floor for a nomination.  Only bounds candidate volume — the
#: lexical Jaccard gate is the actual merge authority, so this is intentionally
#: permissive rather than a second safety threshold.
_EMBED_MIN_SCORE = 0.6
#: Server score methods that count as "vector" evidence.  Pure ``fulltext`` is
#: skipped: it is the lexical signal the existing pass already covers, and G6 is
#: specifically the semantic (embedding) leg.
_EMBED_VECTOR_METHODS = frozenset({"vector", "hybrid"})
#: Max revision krefs a single ``score_revisions`` call accepts (server limit).
_SCORE_REVISIONS_MAX = 100

# Edge-direction constants (mirror kumiho.OUTGOING / INCOMING; imported lazily
# via the injected sdk so this module has no hard kumiho import at load time).
_OUTGOING = 0
_INCOMING = 1
_BOTH = 2


@dataclass
class MaintenanceStats:
    """Counters accumulated by a graph-maintenance run."""

    entities_scanned: int = 0
    facts_scanned: int = 0
    decisions_scanned: int = 0
    entities_merged: int = 0
    facts_merged: int = 0
    orphans_pruned: int = 0
    decisions_regraded: int = 0
    decisions_deduped: int = 0
    bridges_created: int = 0
    edges_repointed: int = 0
    #: Grounding-staleness clear pass (#95): flagged DEPENDS_ON dependents whose
    #: grounding was re-confirmed (flag cleared) vs. still stale (flag kept).
    dependents_cleared: int = 0
    dependents_kept: int = 0
    llm_merges: int = 0
    #: LLM-suggested, referentially-valid merges that couldn't run because the
    #: entity deprecation budget was exhausted this run — lets an operator
    #: distinguish "budget starvation" from "the model found nothing to merge".
    llm_merges_skipped: int = 0
    #: Embedding-assisted fact-dedup (G6, opt-in): candidate pairs the server
    #: vector stage nominated, and how many of those actually merged (cleared
    #: the lexical confirmation + budget).  ``embed_facts_merged`` is a subset of
    #: ``facts_merged``.
    embed_fact_candidates: int = 0
    embed_facts_merged: int = 0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entities_scanned": self.entities_scanned,
            "facts_scanned": self.facts_scanned,
            "decisions_scanned": self.decisions_scanned,
            "entities_merged": self.entities_merged,
            "facts_merged": self.facts_merged,
            "orphans_pruned": self.orphans_pruned,
            "decisions_regraded": self.decisions_regraded,
            "decisions_deduped": self.decisions_deduped,
            "bridges_created": self.bridges_created,
            "edges_repointed": self.edges_repointed,
            "dependents_cleared": self.dependents_cleared,
            "dependents_kept": self.dependents_kept,
            "llm_merges": self.llm_merges,
            "llm_merges_skipped": self.llm_merges_skipped,
            "embed_fact_candidates": self.embed_fact_candidates,
            "embed_facts_merged": self.embed_facts_merged,
            "errors": list(self.errors),
        }


def _node_slug(kref_uri: str, kind: str) -> str:
    """``kref://proj/space/<slug>.<kind>[?r=N]`` → ``<slug>`` ('' if unparsable).

    The single identity extractor shared by every pass — a decision's
    ``ABOUT`` target and an entity item must resolve to the same slug string
    for the deterministic joins to line up.
    """
    seg = (kref_uri or "").split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    suffix = f".{kind}"
    return seg[: -len(suffix)] if seg.endswith(suffix) else seg


@dataclass
class _Entity:
    item: Any
    rev: Any
    slug: str
    display: str
    aliases: List[str]
    entity_type: str
    name_tokens: List[str]


class GraphMaintainer:
    """Keyless maintenance of the ontology + Decision Memory graphs.

    One instance per Dream State run.  All passes are best-effort: a failure
    in one node is logged to :class:`MaintenanceStats` and skipped, never
    aborting the run.  ``dry_run`` short-circuits every *write* (reads and
    counting still happen) so the run mutates nothing.  The counts are a
    close per-pass preview; because the passes are sequential and a live run
    hands mutated state to later passes (a merged entity's facts move onto the
    hub before fact-dedup sees them), a downstream pass's dry-run count can
    differ slightly from the eventual live delta — the dry run never
    over-reports destruction, only the exact downstream interaction.
    """

    def __init__(
        self,
        sdk: Any,
        *,
        project: str,
        code_project: Optional[str] = None,
        schema: Optional[OntologySchema] = None,
        dry_run: bool = False,
        max_deprecation_ratio: float = 0.5,
        allow_published_deprecation: bool = False,
    ) -> None:
        self.sdk = sdk
        self.project = project
        self.code_project = code_project
        self.schema = schema or OntologySchema()
        self.dry_run = dry_run
        self.max_deprecation_ratio = max_deprecation_ratio
        # Operator-published nodes are protected from maintenance
        # deprecation/demotion exactly as the flat conversation pass protects
        # them (mirrors DreamState.allow_published_deprecation).
        self.allow_published_deprecation = allow_published_deprecation
        # Per-kind deprecation budgets ("entity", "fact"): a destructive pass
        # may never deprecate more than ``ratio`` of that kind's live nodes
        # (mirrors DreamState's per-batch deprecation cap).  Decision dedup is
        # non-destructive (status demotion, not deprecation) so it takes no
        # budget.
        self._budgets: Dict[str, int] = {}
        self._client = None

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_keyless(self, stats: MaintenanceStats) -> None:
        """Run every deterministic pass in dependency order.

        Order matters: entities are merged *before* the fact and bridge
        passes so those join against the surviving hubs; evidence is
        re-graded *before* decision dedup so the higher-graded twin wins the
        keep/sink decision; and orphan prune runs *last* — after the bridge —
        so a conversation entity a code decision is about (and just linked to)
        is no longer edgeless and is not pruned out from under the bridge.
        """
        entities = self._load_entities()
        stats.entities_scanned = len(entities)
        # Entity-deprecating passes (merge + orphan prune) share one budget
        # sized from the live entity count.
        self._budgets["entity"] = max(1, int(len(entities) * self.max_deprecation_ratio))
        try:
            self._merge_entities(entities, stats)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"merge_entities: {exc}")
        # Surviving hubs (merges deprecated some) for the passes that join
        # against entities.
        survivors = [e for e in entities if not _is_deprecated(e.item)]
        try:
            self._dedup_facts(survivors, stats)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"dedup_facts: {exc}")

        if self.code_project:
            decisions = self._load_decisions()
            stats.decisions_scanned = len(decisions)
            try:
                self._regrade_decisions(decisions, stats)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"regrade_decisions: {exc}")
            try:
                self._dedup_decisions(decisions, stats)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"dedup_decisions: {exc}")
            try:
                self._bridge_decisions(decisions, survivors, stats)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"bridge_decisions: {exc}")

        # Prune last: an entity the bridge just linked now has an edge and is
        # correctly kept (fixes the prune-then-bridge contradiction).
        try:
            self._prune_orphans(survivors, stats)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"prune_orphans: {exc}")

        # Grounding-staleness clear (#95): runs AFTER the destructive passes so
        # it sees their result — a superseding fact that fact-dedup just folded
        # away reads as "gone" and clears its dependents. Non-destructive
        # (un-flag only), so it takes no deprecation budget and needs no
        # code_project.
        try:
            self._clear_stale_grounding(stats)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"clear_stale_grounding: {exc}")

    def apply_entity_merges(
        self, pairs: List[Tuple[str, str]], stats: MaintenanceStats
    ) -> None:
        """Apply LLM-suggested ``(canonical_slug, duplicate_slug)`` merges.

        The *suggestion* is semantic (a model saw two names as the same
        entity); the *application* is the same deterministic, dry-run-safe,
        budgeted :meth:`_merge_pair` the keyless alias rule uses — the model
        never writes.  Suggestions that don't resolve to two distinct live
        entities are dropped (referential integrity).
        """
        entities = self._load_entities()
        by_slug = {e.slug: e for e in entities}
        # Share the entity budget with the keyless pass when it ran; size it
        # here if this is the only pass (setdefault never revives a budget the
        # keyless pass already exhausted).
        self._budgets.setdefault(
            "entity", max(1, int(len(entities) * self.max_deprecation_ratio))
        )
        for canon_slug, dup_slug in pairs:
            canon = by_slug.get(canon_slug)
            dup = by_slug.get(dup_slug)
            if (
                canon is None
                or dup is None
                or canon.slug == dup.slug
                or _is_deprecated(canon.item)
                or _is_deprecated(dup.item)
            ):
                continue  # unresolvable / already merged — referential drop
            # Distinguish budget starvation (observable) from a clean drop:
            # a referentially-valid pair that can't run because the shared
            # entity budget is spent is recorded, not silently lost.
            if self._budgets.get("entity", 0) <= 0:
                stats.llm_merges_skipped += 1
                continue
            if self._merge_pair(canon, dup, stats):
                stats.llm_merges += 1
        if stats.llm_merges_skipped:
            logger.info(
                "maintenance: %d valid LLM entity-merge pair(s) skipped — "
                "entity deprecation budget exhausted this run",
                stats.llm_merges_skipped,
            )

    # ------------------------------------------------------------------
    # (A) Ontology — embedding-assisted fact dedup (G6, opt-in)
    # ------------------------------------------------------------------

    def apply_embedding_fact_dedup(self, stats: MaintenanceStats) -> None:
        """Nominate near-duplicate fact pairs via server vector scoring, then
        merge them through the SAME verification path as the keyless pass.

        The *nomination* is semantic (server embeddings saw two statements as
        near-identical across whatever entities they are filed under); the
        *application* is the identical deterministic, dry-run-safe,
        budget-and-published-protected :meth:`_confirm_and_collapse_fact` the
        keyless per-entity scan uses — embeddings never authorize a merge, and
        the ``_FACT_DEDUP_JACCARD`` safety threshold is unchanged.  Runs after
        :meth:`run_keyless` so it sees already-merged facts as deprecated.
        """
        candidates = self.embedding_fact_candidates(stats)
        if not candidates:
            return
        # Share the fact budget the keyless pass sized+spent; size it here only
        # if this is the sole fact-deprecating pass (setdefault never revives an
        # exhausted budget).
        live_facts = sum(
            1 for f in self._search(self.project, "fact") if not _is_deprecated(f)
        )
        self._budgets.setdefault(
            "fact", max(1, int(live_facts * self.max_deprecation_ratio))
        )
        seen_pairs: set = set()
        for a, b in candidates:
            if self._confirm_and_collapse_fact(a, b, stats, seen_pairs):
                stats.embed_facts_merged += 1

    def embedding_fact_candidates(
        self, stats: MaintenanceStats
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Server-side vector nomination of near-duplicate fact pairs (G6).

        Keyless: ``score_revisions`` ranks each query fact's statement against
        the other live fact revisions using the server's STORED embeddings — no
        client-side embedding, no LLM key.  Returns a de-duplicated list of
        ``(a, b)`` fact dicts (the ``_about_sources`` shape the merge path
        expects).  Bounded on every axis (query fan-out, k, candidate pool) and
        ``kind=fact``-filtered so it never floods vector recall.  No-op when the
        SDK/server exposes no vector scoring.
        """
        score_fn = getattr(self.sdk, "score_revisions", None)
        if not callable(score_fn):
            return []  # server/SDK without vector scoring — silently skip
        facts = [
            f for f in self._search(self.project, "fact") if not _is_deprecated(f)
        ]
        if len(facts) < 2:
            return []
        if len(facts) > _MAX_DEDUP_NODES:
            logger.info(
                "embedding fact dedup: %d live facts exceed cap %d — truncating",
                len(facts), _MAX_DEDUP_NODES,
            )
            facts = facts[:_MAX_DEDUP_NODES]

        entries: List[Dict[str, Any]] = []
        by_uri: Dict[str, Dict[str, Any]] = {}
        for item in facts:
            rev = _latest(item)
            if rev is None:
                continue
            uri = _uri(rev)
            if not uri:
                continue
            meta = _meta(rev)
            claim = str(
                meta.get("claim", "") or meta.get("summary", "") or meta.get("title", "")
            )
            if not claim.strip():
                continue
            entry = {
                "item": item,
                "rev": rev,
                "slug": _node_slug(uri, "fact"),
                "tokens": _tokens(claim),
                "statement": claim,
                "uri": uri,
            }
            entries.append(entry)
            by_uri[uri] = entry
        if len(entries) < 2:
            return []

        all_uris = [e["uri"] for e in entries]
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        seen: set = set()
        for q in entries[:_EMBED_MAX_QUERY_FACTS]:
            others = [u for u in all_uris if u != q["uri"]][:_SCORE_REVISIONS_MAX]
            if not others:
                continue
            try:
                scored = score_fn(q["statement"], others)
            except Exception as exc:  # noqa: BLE001 — a scoring failure is a no-op
                logger.debug("embedding fact dedup: score_revisions failed: %s", exc)
                continue
            taken = 0
            for sr in scored or []:
                if taken >= _EMBED_TOP_K:
                    break
                if str(sr.get("score_method", "")).lower() not in _EMBED_VECTOR_METHODS:
                    continue
                try:
                    score = float(sr.get("score", 0.0))
                except (TypeError, ValueError):
                    continue
                if score < _EMBED_MIN_SCORE:
                    continue
                taken += 1
                other = by_uri.get(str(sr.get("kref", "")))
                if other is None or other["slug"] == q["slug"]:
                    continue
                key = tuple(sorted((q["slug"], other["slug"])))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((q, other))
        stats.embed_fact_candidates += len(pairs)
        return pairs

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self.sdk.get_client()
        return self._client

    def _search(self, project: str, kind: str) -> List[Any]:
        try:
            return list(
                self.sdk.item_search(
                    context_filter=project, name_filter="", kind_filter=kind
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("maintenance: item_search %s/%s failed: %s", project, kind, exc)
            return []

    def _load_entities(self) -> List[_Entity]:
        out: List[_Entity] = []
        for item in self._search(self.project, "entity"):
            if _is_deprecated(item):
                continue
            rev = _latest(item)
            if rev is None:
                continue
            meta = _meta(rev)
            slug = _node_slug(_uri(item), "entity")
            if not slug:
                continue
            display = str(meta.get("display_name", "") or meta.get("title", "") or slug)
            aliases = [a.strip() for a in str(meta.get("aliases", "")).split(",") if a.strip()]
            out.append(
                _Entity(
                    item=item,
                    rev=rev,
                    slug=slug,
                    display=display,
                    aliases=aliases,
                    entity_type=str(meta.get("entity_type", "")),
                    name_tokens=_word_tokens(display),
                )
            )
        return out

    def _load_decisions(self) -> List[Any]:
        return [
            item
            for item in self._search(self.code_project or "", KIND_DECISION)
            if not _is_deprecated(item)
        ]

    def _take(self, kind: str) -> bool:
        """Consume one unit of *kind*'s deprecation budget; ``False`` when
        exhausted (the caller must then skip the deprecation)."""
        remaining = self._budgets.get(kind, 0)
        if remaining <= 0:
            return False
        self._budgets[kind] = remaining - 1
        return True

    def _protected(self, rev: Any) -> bool:
        """True if *rev* is operator-``published`` and must not be deprecated
        or demoted by maintenance.

        Mirrors the flat pass's hard guard (``_apply_actions`` refuses to
        deprecate a ``published`` revision unless the operator opts in).  #59
        widened the set of nodes Dream State can deprecate; this carries the
        protection across to the typed graphs.
        """
        if self.allow_published_deprecation:
            return False
        return "published" in _safe_tags(rev)

    # ------------------------------------------------------------------
    # (A) Ontology — entity merge
    # ------------------------------------------------------------------

    def _merge_entities(self, entities: List[_Entity], stats: MaintenanceStats) -> None:
        """Fold an entity into the hub that lists it as an alias.

        Deterministic signal only: hub *A* whose ``aliases`` metadata names a
        string that slugs to entity *B*'s slug is asserting "B is me".  The
        alias-lister is canonical; *B*'s edges move to *A* and *B* is
        deprecated.  Exact-slug duplicates cannot occur (get-or-create dedups
        at write time), so alias identity is the only cross-session collision
        left for a keyless pass to catch.
        """
        by_slug = {e.slug: e for e in entities}
        merged: set = set()
        for canon in entities:
            if canon.slug in merged:
                continue
            # Index walk (not a snapshot iterator): _merge_pair folds the
            # absorbed dup's aliases into canon.aliases, so a transitive chain
            # A→B→C collapses in ONE pass instead of one merge per run.
            i = 0
            while i < len(canon.aliases):
                alias = canon.aliases[i]
                i += 1
                dup_slug = slugify(alias, hash_on_truncate=True)
                if not dup_slug or dup_slug == canon.slug:
                    continue
                dup = by_slug.get(dup_slug)
                if dup is None or dup.slug in merged or dup.slug == canon.slug:
                    continue
                if self._merge_pair(canon, dup, stats):
                    merged.add(dup.slug)

    def _merge_pair(
        self, canon: _Entity, dup: _Entity, stats: MaintenanceStats
    ) -> bool:
        """Repoint *dup*'s edges onto *canon*, fold aliases, deprecate *dup*.

        Returns ``True`` when the merge was applied (or, in ``dry_run``, would
        have been).  Consumes one unit of the entity deprecation budget.
        """
        # A published duplicate is operator-owned — never fold it away (the
        # check precedes any edge repointing so a protected node is left whole).
        if self._protected(dup.rev):
            return False
        if not self._take("entity"):
            return False
        # Move both directions so nothing about the duplicate is orphaned:
        # facts/conversations pointing *at* dup, and dup's own typed relations.
        # A dup<->canon edge is dropped, not repointed — it would become a
        # canon->canon self-loop.
        canon_uri = _uri(canon.rev)
        for edge in _edges(dup.rev, None, _INCOMING):
            src_uri = _edge_src(edge)
            if src_uri == canon_uri:
                continue
            src = self.sdk.get_revision(src_uri)
            if src is not None and self._edge_once(src, canon.rev, edge.edge_type, _edge_meta(edge)):
                stats.edges_repointed += 1
        for edge in _edges(dup.rev, None, _OUTGOING):
            dst_uri = _edge_dst(edge)
            if dst_uri == canon_uri:
                continue
            tgt = self.sdk.get_revision(dst_uri)
            if tgt is not None and self._edge_once(canon.rev, tgt, edge.edge_type, _edge_meta(edge)):
                stats.edges_repointed += 1
        # Fold identity: dup's display name + aliases become canon's aliases.
        folded = _dedup_keep_order(canon.aliases + [dup.display] + dup.aliases)
        folded = [a for a in folded if slugify(a, hash_on_truncate=True) != canon.slug]
        updates = {"aliases": ", ".join(folded)}
        if not canon.entity_type and dup.entity_type:
            updates["entity_type"] = dup.entity_type
        if not self.dry_run:
            try:
                self._get_client().update_revision_metadata(canon.rev.kref, updates)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"merge fold {dup.slug}->{canon.slug}: {exc}")
        canon.aliases = folded
        self._deprecate(dup.item, stats, f"merge {dup.slug}->{canon.slug}")
        stats.entities_merged += 1
        return True

    # ------------------------------------------------------------------
    # (A) Ontology — fact dedup (per shared entity)
    # ------------------------------------------------------------------

    def _dedup_facts(self, entities: List[_Entity], stats: MaintenanceStats) -> None:
        """Collapse near-duplicate facts about the *same* entity.

        Scoping the O(n²) compare to one entity's ``ABOUT`` fan-in is both the
        bound and the meaning: "facts about the same entity" (issue A).  The
        keeper is the better-connected fact (more edges), tie-broken by slug
        for total determinism; the loser is deprecated.
        """
        # Size from LIVE facts only (symmetry with the entity budget), so the
        # "half your facts" cap tracks the live set regardless of how much
        # historical deprecation a space has accrued.
        live_facts = sum(
            1 for f in self._search(self.project, "fact") if not _is_deprecated(f)
        )
        self._budgets["fact"] = max(1, int(live_facts * self.max_deprecation_ratio))
        seen_pairs: set = set()
        for ent in entities:
            facts = self._about_sources(ent.rev, "fact")
            stats.facts_scanned += len(facts)
            for i in range(len(facts)):
                a = facts[i]
                if _is_deprecated(a["item"]):
                    continue
                for j in range(i + 1, len(facts)):
                    b = facts[j]
                    if _is_deprecated(b["item"]):
                        continue
                    self._confirm_and_collapse_fact(a, b, stats, seen_pairs)

    def _confirm_and_collapse_fact(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        stats: MaintenanceStats,
        seen_pairs: set,
    ) -> bool:
        """The single fact-merge verification+application path.

        A candidate pair — whether nominated by the per-entity lexical scan or
        the opt-in embedding stage — merges ONLY when it clears the
        ``_FACT_DEDUP_JACCARD`` lexical confirmation (the unrecoverable-merge
        safety threshold), and even then through ``_collapse_node``, which
        enforces published protection and the per-kind fact deprecation budget.
        Returns ``True`` when a merge was applied (or, in ``dry_run``, would
        have been).  Idempotent per pair via *seen_pairs*.
        """
        key = tuple(sorted((a["slug"], b["slug"])))
        if key in seen_pairs:
            return False
        seen_pairs.add(key)
        if a["slug"] == b["slug"]:
            return False
        if _is_deprecated(a["item"]) or _is_deprecated(b["item"]):
            return False
        if _jaccard(a["tokens"], b["tokens"]) < _FACT_DEDUP_JACCARD:
            return False
        # Keeper selection needs the edge count; compute it lazily so the
        # embedding stage only pays the round-trip for a confirmed pair.
        for entry in (a, b):
            if "edges" not in entry:
                entry["edges"] = len(_edges(entry["rev"], None, _BOTH))
        keeper, loser = _pick_keeper(a, b)
        if self._collapse_node(keeper, loser, stats):
            stats.facts_merged += 1
            return True
        return False

    # ------------------------------------------------------------------
    # (A) Ontology — orphan prune
    # ------------------------------------------------------------------

    def _prune_orphans(self, entities: List[_Entity], stats: MaintenanceStats) -> None:
        """Deprecate entities with no edges at all.

        Conservative: only *zero-edge* hubs (no ABOUT in, no relations out) —
        get-or-create resurrects the slug byte-identically if a later
        conversation mentions it, so the prune is reversible in effect.
        """
        for ent in entities:
            if _is_deprecated(ent.item):
                continue  # just merged away
            if _edges(ent.rev, None, _BOTH):
                continue
            if self._protected(ent.rev):
                continue  # operator-published — never pruned
            if not self._take("entity"):
                break
            self._deprecate(ent.item, stats, f"orphan {ent.slug}")
            stats.orphans_pruned += 1

    # ------------------------------------------------------------------
    # (A) Ontology — grounding-staleness clear (#95)
    # ------------------------------------------------------------------

    def _clear_stale_grounding(self, stats: MaintenanceStats) -> None:
        """Re-examine flagged DEPENDS_ON dependents; clear when re-grounded.

        A ``decision`` flagged ``grounding_stale`` (its DEPENDS_ON fact was
        superseded, stamped by :func:`grounding.ripple_grounding_stale`) is
        cleared when the write-time ripple's premise no longer holds — EITHER:

        1. the superseding fact's revision no longer exists (deleted, or its
           item deprecated — e.g. fact-dedup folded it away), OR
        2. the dependent now has its OWN ``DEPENDS_ON`` edge to the superseding
           fact (its grounding was updated to the new belief).

        Otherwise the flag stays: the decision is still grounded in a superseded
        fact and awaits re-grading. Non-destructive (un-flag only), keyless,
        deterministic. Metadata is canonical (set ``"false"`` — never deleted,
        so a metadata-only reader sees a definite non-stale state); the mirrored
        ``grounding:stale`` tag is removed best-effort.

        (An optional LLM re-grade — does the superseding fact actually change
        the decision's basis? — is future work and would slot in here as an
        opt-in signal; the plugin's keyless path never uses it.)
        """
        scanned = 0
        for item in self._search(self.project, "decision"):
            if _is_deprecated(item):
                continue
            rev = _latest(item)
            if rev is None:
                continue
            meta = _meta(rev)
            if not is_grounding_stale(meta):
                continue
            if scanned >= _MAX_DEDUP_NODES:
                logger.info(
                    "maintenance: grounding-clear scan hit cap %d", _MAX_DEDUP_NODES,
                )
                break
            scanned += 1
            superseding_kref = str(
                meta.get(GROUNDING_STALE_SUPERSEDED_BY_META, "") or ""
            )
            if not self._grounding_reconfirmed(rev, superseding_kref):
                stats.dependents_kept += 1
                continue
            if not self.dry_run:
                try:
                    # Un-flag in place: metadata is canonical, so set "false"
                    # (never delete) and drop the now-stale superseded_by pointer.
                    self._get_client().update_revision_metadata(rev.kref, {
                        GROUNDING_STALE_META: "false",
                        GROUNDING_STALE_SUPERSEDED_BY_META: "",
                    })
                    try:
                        self._get_client().untag_revision(rev.kref, GROUNDING_STALE_TAG)
                    except Exception:  # noqa: BLE001
                        pass  # untag is best-effort cleanup (tag is mirrored)
                except Exception as exc:  # noqa: BLE001
                    stats.errors.append(f"clear grounding {_uri(item)}: {exc}")
                    continue
            stats.dependents_cleared += 1

    def _grounding_reconfirmed(self, rev: Any, superseding_kref: str) -> bool:
        """True if a flagged dependent's grounding is re-confirmed (clearable)."""
        if not superseding_kref:
            # No pointer to check against — can't confirm re-grounding, so keep.
            return False
        # (1) superseding fact gone: deleted or its item deprecated (folded away
        # by fact-dedup this run). The SDK RAISES grpc NOT_FOUND for a deleted
        # revision (it does not return None), so ONLY a definitive NOT_FOUND (or
        # a None from a fake) counts as "gone" — a transient RPC error must not
        # clear a still-valid flag (keep it; re-examine next run).
        try:
            sup_rev = self.sdk.get_revision(superseding_kref)
        except Exception as exc:  # noqa: BLE001
            if _is_not_found(exc):
                return True
            logger.debug(
                "grounding clear: get_revision %s failed (kept): %s",
                superseding_kref, exc,
            )
        else:
            if sup_rev is None:
                return True
            sup_item = _item_of(sup_rev)
            if sup_item is not None and _is_deprecated(sup_item):
                return True
        # (2) dependent re-grounded onto the superseding fact itself. Compare on
        # the base kref (typed nodes are anchored at one revision, so the ?r=N
        # is stable — but strip it defensively). Independent of (1), so it still
        # clears even when the superseding fact's existence couldn't be checked.
        base = superseding_kref.split("?", 1)[0]
        for edge in _edges(rev, "DEPENDS_ON", _OUTGOING):
            if _edge_dst(edge).split("?", 1)[0] == base:
                return True
        return False

    # ------------------------------------------------------------------
    # (B) Decision Memory — evidence re-grade (headline)
    # ------------------------------------------------------------------

    def _regrade_decisions(self, decisions: List[Any], stats: MaintenanceStats) -> None:
        """Lift a decision's ``evidence_level`` from its CURRENT atoms.

        Non-destructive metadata update.  For each decision, recompute the
        deterministic grade from the ``evidence_kind`` of every atom it is
        ``MOTIVATED_BY`` today; if that outranks the stored grade, lift it
        (and mirror the ``evidence:<level>`` tag).  Only lifts — evidence
        accrues, and a downgrade could fight an operator ``official`` flag,
        which is never touched.
        """
        for item in decisions:
            rev = _latest(item)
            if rev is None:
                continue
            meta = _meta(rev)
            # Read BOTH carriers: an operator may set the grade via the
            # mirrored evidence:<level> TAG alone (metadata unset).  Ignoring
            # tags would misread `old` too low and let a lift overwrite an
            # `official` grade or downgrade a tag-only `corroborated` one.
            old = parse_evidence(meta, tags=_safe_tags(rev), default=UNVERIFIED) or UNVERIFIED
            if old == OFFICIAL:
                continue  # operator-owned; never auto-rewritten
            atoms: List[Dict[str, str]] = []
            for edge in _edges(rev, EDGE_MOTIVATED_BY, _OUTGOING):
                ev = self.sdk.get_revision(_edge_dst(edge))
                if ev is None:
                    continue
                atoms.append({"kind": str(_meta(ev).get("evidence_kind", "constraint"))})
            new = _evidence_grade(atoms)
            if _GRADE_RANK.get(new, 0) <= _GRADE_RANK.get(old, 0):
                continue  # lift-only; never downgrade
            if not self.dry_run:
                try:
                    self._get_client().update_revision_metadata(
                        rev.kref, {"evidence_level": new}
                    )
                    self._get_client().tag_revision(rev.kref, evidence_tag(new))
                    # Drop the now-stale lower-grade tag so a tag-only consumer
                    # doesn't see two evidence:* tags on one revision.
                    if old in EVIDENCE_LEVELS and old != new:
                        try:
                            self._get_client().untag_revision(rev.kref, evidence_tag(old))
                        except Exception:  # noqa: BLE001
                            pass  # untag is best-effort cleanup, not required
                except Exception as exc:  # noqa: BLE001
                    stats.errors.append(f"regrade {_uri(item)}: {exc}")
                    continue
            stats.decisions_regraded += 1

    # ------------------------------------------------------------------
    # (B) Decision Memory — decision dedup
    # ------------------------------------------------------------------

    def _dedup_decisions(self, decisions: List[Any], stats: MaintenanceStats) -> None:
        """Sink a near-identical decision under its twin.

        Not a merge: the loser is kept but marked ``status=superseded`` with a
        ``SUPERSEDES`` edge from the keeper, so the query side's existing
        demotion (a superseded decision sinks within its factual tier) handles
        it — no node is destroyed, matching capture's reversible split
        philosophy.  Keeper = higher evidence grade, then more edges, then
        smaller slug.
        """
        nodes: List[Dict[str, Any]] = []
        for item in decisions[:_MAX_DEDUP_NODES]:
            rev = _latest(item)
            if rev is None:
                continue
            meta = _meta(rev)
            # Exclude decisions already sunk in a prior run: re-admitting a
            # superseded node lets a later evidence lift flip the keeper choice
            # and sink its former keeper too — a mutual SUPERSEDES cycle that
            # buries a well-evidenced decision.  Once sunk, stays sunk.
            if str(meta.get("status", "active")) == "superseded":
                continue
            text = f"{meta.get('title', '')} {meta.get('decision', '')}"
            nodes.append(
                {
                    "item": item,
                    "rev": rev,
                    "slug": _node_slug(_uri(item), KIND_DECISION),
                    "tokens": _tokens(text),
                    # Grade reads both carriers (metadata + evidence:<level> tag)
                    # so a tag-only grade still wins the keeper choice.
                    "grade": _GRADE_RANK.get(
                        parse_evidence(meta, tags=_safe_tags(rev), default=UNVERIFIED)
                        or UNVERIFIED, 0
                    ),
                    "edges": len(_edges(rev, None, _BOTH)),
                }
            )
        if len(decisions) > _MAX_DEDUP_NODES:
            logger.info(
                "maintenance: decision dedup scanned %d/%d (cap %d)",
                _MAX_DEDUP_NODES, len(decisions), _MAX_DEDUP_NODES,
            )
        sunk: set = set()
        for i in range(len(nodes)):
            a = nodes[i]
            if a["slug"] in sunk:
                continue
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if b["slug"] in sunk or a["slug"] in sunk:
                    continue
                if _jaccard(a["tokens"], b["tokens"]) < _DECISION_DEDUP_JACCARD:
                    continue
                keeper, loser = _pick_decision_keeper(a, b)
                if self._sink_decision(keeper, loser, stats):
                    sunk.add(loser["slug"])

    def _sink_decision(
        self, keeper: Dict[str, Any], loser: Dict[str, Any], stats: MaintenanceStats
    ) -> bool:
        # A published loser is operator-owned — never demote it, even though
        # demotion is reversible (mirrors the deprecation protection).
        if self._protected(loser["rev"]):
            return False
        if self.dry_run:
            stats.decisions_deduped += 1
            return True
        ok = self._edge_once(keeper["rev"], loser["rev"], EDGE_SUPERSEDES, {"reason": "dedup"})
        try:
            loser["rev"].set_attribute("status", "superseded")
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"sink {loser['slug']}: {exc}")
            return False
        if ok:
            stats.decisions_deduped += 1
            return True
        # Edge already existed but status now demoted — still progress.
        stats.decisions_deduped += 1
        return True

    # ------------------------------------------------------------------
    # (C) Cross-graph bridge — code_decision --ABOUT--> entity
    # ------------------------------------------------------------------

    def _bridge_decisions(
        self, decisions: List[Any], entities: List[_Entity], stats: MaintenanceStats
    ) -> None:
        """Link each code decision to the conversation entity it is about.

        Deterministic join across the project boundary (the same
        cross-project pattern as ``DISCUSSED_IN``): a decision's ``symbols``
        that slug-match an entity, or an entity whose name tokens occur in the
        decision's title/text, yields a single ``ABOUT`` edge — the join the
        entity-bridge reader needs to hop from a code decision out to sibling
        conversation memories about the same thing.
        """
        # Never bridge onto an entity a prior pass deprecated (belt-and-braces;
        # orphan prune runs after the bridge, but a merge earlier this run
        # already removed some).
        entities = [e for e in entities if not _is_deprecated(e.item)]
        if not entities:
            return
        by_slug = {e.slug: e for e in entities}
        for item in decisions:
            if _is_deprecated(item):
                continue
            rev = _latest(item)
            if rev is None:
                continue
            meta = _meta(rev)
            targets: Dict[str, _Entity] = {}
            for sym in str(meta.get("symbols", "")).split(","):
                s = slugify(sym.strip(), hash_on_truncate=True)
                if s and s in by_slug:
                    targets[s] = by_slug[s]
            text_tokens = _word_tokens(f"{meta.get('title', '')} {meta.get('decision', '')}")
            for ent in entities:
                if ent.slug in targets:
                    continue
                if ent.name_tokens and _mentions(ent.name_tokens, text_tokens):
                    targets[ent.slug] = ent
            for slug, ent in targets.items():
                if self._edge_once(rev, ent.rev, self.schema.about_edge, {"entity": slug}):
                    stats.bridges_created += 1

    # ------------------------------------------------------------------
    # Shared write helpers (all honor dry_run)
    # ------------------------------------------------------------------

    def _about_sources(self, entity_rev: Any, kind: str) -> List[Dict[str, Any]]:
        """The *kind*-typed nodes with an ``ABOUT`` edge into *entity_rev*."""
        out: List[Dict[str, Any]] = []
        for edge in _edges(entity_rev, self.schema.about_edge, _INCOMING):
            uri = _edge_src(edge)
            if f".{kind}" not in uri:
                continue
            src = self.sdk.get_revision(uri)
            if src is None:
                continue
            meta = _meta(src)
            claim = str(meta.get("claim", "") or meta.get("summary", "") or meta.get("title", ""))
            out.append(
                {
                    "item": _item_of(src),
                    "rev": src,
                    "slug": _node_slug(uri, kind),
                    "tokens": _tokens(claim),
                    "edges": len(_edges(src, None, _BOTH)),
                }
            )
        return out

    def _collapse_node(
        self, keeper: Dict[str, Any], loser: Dict[str, Any], stats: MaintenanceStats
    ) -> bool:
        """Repoint *loser*'s edges onto *keeper* and deprecate *loser*.

        Returns ``False`` (no-op) when the loser can't actually be
        deprecated — a published/protected loser, or a rev with no resolvable
        item — so the caller never counts a merge that left the loser live.
        """
        loser_item = loser.get("item")
        if loser_item is None or self._protected(loser["rev"]):
            return False
        if not self._take("fact"):
            return False
        keeper_uri = _uri(keeper["rev"])
        for edge in _edges(loser["rev"], None, _INCOMING):
            src_uri = _edge_src(edge)
            if src_uri == keeper_uri:
                continue
            src = self.sdk.get_revision(src_uri)
            if src is not None and self._edge_once(src, keeper["rev"], edge.edge_type, _edge_meta(edge)):
                stats.edges_repointed += 1
        for edge in _edges(loser["rev"], None, _OUTGOING):
            dst_uri = _edge_dst(edge)
            if dst_uri == keeper_uri:
                continue
            tgt = self.sdk.get_revision(dst_uri)
            if tgt is not None and self._edge_once(keeper["rev"], tgt, edge.edge_type, _edge_meta(edge)):
                stats.edges_repointed += 1
        self._deprecate(loser_item, stats, f"dedup {loser['slug']}->{keeper['slug']}")
        return True

    def _edge_once(
        self, source_rev: Any, target_rev: Any, edge_type: str, metadata: Dict[str, str]
    ) -> bool:
        """Idempotent edge create (existing-edge precheck), honoring dry_run.

        Mirrors ``ontology._Materializer.edge``: never duplicates an edge on a
        re-run, and on a read failure falls through to create rather than
        silently dropping.
        """
        if source_rev is None or target_rev is None:
            return False
        target_uri = _uri(target_rev)
        if target_uri:
            for existing in _edges(source_rev, edge_type, _OUTGOING):
                if _edge_dst(existing) == target_uri:
                    return False
        if self.dry_run:
            return True
        try:
            source_rev.create_edge(target_rev, edge_type, metadata=metadata or {})
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("maintenance: edge %s failed: %s", edge_type, exc)
            return False

    def _deprecate(self, item: Any, stats: MaintenanceStats, reason: str) -> None:
        """Deprecate an item (respecting dry_run); the caller owns the stat
        counter, so this only performs the write."""
        if self.dry_run:
            return
        try:
            item.set_deprecated(True)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"deprecate {reason}: {exc}")


# ---------------------------------------------------------------------------
# Duck-typed SDK accessors (tolerate fakes + partial objects)
# ---------------------------------------------------------------------------


def _latest(item: Any) -> Any:
    try:
        return item.get_latest_revision()
    except Exception:  # noqa: BLE001
        return None


def _meta(rev: Any) -> Dict[str, Any]:
    return dict(getattr(rev, "metadata", {}) or {})


def _safe_tags(rev: Any) -> List[str]:
    """A revision's graph tags from the construction-time snapshot when
    present (``_cached_tags``), else the ``tags`` attribute — tolerating
    fakes and RPC failures.

    Reads the snapshot first because the SDK's ``Revision.tags`` property
    auto-refreshes via a *blocking* gRPC call once stale (same guard as
    ``dream_state._safe_policy_tags``); the snapshot is fresh enough for a
    protection/grade decision within one run.
    """
    try:
        tags = getattr(rev, "_cached_tags", None)
        if tags is None:
            tags = getattr(rev, "tags", []) or []
        return [t for t in (tags or []) if isinstance(t, str)]
    except Exception:  # noqa: BLE001
        return []


def _uri(obj: Any) -> str:
    return getattr(getattr(obj, "kref", None), "uri", "") or ""


def _is_deprecated(item: Any) -> bool:
    return bool(getattr(item, "deprecated", False))


def _is_not_found(exc: Exception) -> bool:
    """True if *exc* is a gRPC NOT_FOUND — a definitive "revision deleted".

    Distinguishes a deleted revision (clear the stale flag) from a transient RPC
    error (keep it). Tolerant of non-grpc environments (returns False)."""
    code = getattr(exc, "code", None)
    if not callable(code):
        return False
    try:
        import grpc

        return code() == grpc.StatusCode.NOT_FOUND
    except Exception:  # noqa: BLE001
        return False


def _item_of(rev: Any) -> Any:
    getter = getattr(rev, "get_item", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001
            return None
    return getattr(rev, "item", None)


def _edges(rev: Any, edge_type: Optional[str], direction: int) -> List[Any]:
    try:
        return list(rev.get_edges(edge_type_filter=edge_type, direction=direction)) or []
    except Exception:  # noqa: BLE001
        return []


def _edge_src(edge: Any) -> str:
    return getattr(getattr(edge, "source_kref", None), "uri", "") or ""


def _edge_dst(edge: Any) -> str:
    return getattr(getattr(edge, "target_kref", None), "uri", "") or ""


def _edge_meta(edge: Any) -> Dict[str, str]:
    return dict(getattr(edge, "metadata", {}) or {})


def _dedup_keep_order(values: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for v in values:
        v = (v or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _pick_keeper(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Deterministic keeper for two near-duplicate facts: more edges wins,
    tie-broken by the lexicographically smaller slug."""
    if a["edges"] != b["edges"]:
        return (a, b) if a["edges"] > b["edges"] else (b, a)
    return (a, b) if a["slug"] <= b["slug"] else (b, a)


def _pick_decision_keeper(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Keeper for two near-identical decisions: higher evidence grade, then
    more edges, then smaller slug — total order, so re-runs are stable."""
    for key in ("grade", "edges"):
        if a[key] != b[key]:
            return (a, b) if a[key] > b[key] else (b, a)
    return (a, b) if a["slug"] <= b["slug"] else (b, a)
