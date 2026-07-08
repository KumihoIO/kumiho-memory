"""Schema-driven decomposition of a consolidated conversation into a typed graph.

The summarizer already extracts a conversation into typed elements — entities,
facts, decisions, events, actions, open questions. Historically those were
flattened into comma-joined metadata strings on a single ``conversation`` node,
discarding the structure. This module *materializes* each element as its own
kind-typed Item connected by typed edges, so the graph gains real graph-native
structure (typed nodes + typed relations) that can be queried by relationship.

An :class:`OntologySchema` — the "schema that dictates how items are created" —
declares the node kinds, their spaces, and the edge rules. The decomposer reads
the schema and applies it; the schema is versioned so it can evolve (and later
be persisted as a policy Item without touching this code).

Node kinds (materialized from the extraction):

    entity   ← classification.entities  (+ event.participants)   [global, deduped]
    fact     ← knowledge.facts     {claim, certainty}
    decision ← knowledge.decisions {decision, reason}
    event    ← events              {event, when, event_date, consequence}
    action   ← knowledge.actions   {task, status}
    question ← knowledge.open_questions

Structural edges (this module; deterministic, no extra LLM):

    <typed node>  --DERIVED_FROM-->  conversation        (provenance)
    fact|decision|action  --ABOUT-->  entity             (name appears in text)
    event  --INVOLVES-->  entity                         (from participants)

Relational edges (``relations`` module extends this):

    decision  --DEPENDS_ON-->  fact                      (summarizer-emitted)
    decision  --SUPERSEDES-->  decision                  (subject match)

Every typed node carries ``title``/``summary`` so the graph-augmented reader
surfaces it as content (not a bare stub) when it hops through a shared entity.

Best-effort: failures are logged and swallowed, and the whole batch runs in a
bounded daemon thread (``_bounded.run_bounded_in_thread``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kumiho._text import slugify

from ._bounded import run_bounded_in_thread

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.casefold())


def _mentions(name_tokens: List[str], text_tokens: List[str]) -> bool:
    """True if *name_tokens* occurs as a contiguous run inside *text_tokens*.

    Token equality — not substring — so short, ambiguous names ("AI", "IT",
    "US") and space-free scripts don't draw false ABOUT edges: Hangul fuses
    particles onto a token, so the token "김치" never matches the name token
    "김". Conservative for CJK (it can miss a real mention when a morpheme is
    glued on) rather than inventing wrong edges — a real tokenizer (the repo's
    ko-dic) would recover those; tracked as follow-up.
    """
    n = len(name_tokens)
    if n == 0 or n > len(text_tokens):
        return False
    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i:i + n] == name_tokens:
            return True
    return False


@dataclass
class OntologySchema:
    """Declares the node kinds, their spaces, and the structural edge rules.

    Versioned so the ontology can evolve; the decomposer is schema-driven so
    changing this object changes what gets written, not the code below.
    """

    version: str = "kumiho.agent_memory.ontology.v1"

    #: extraction field -> (item kind, space under the project root)
    entities_space: str = "entities"
    facts_space: str = "facts"
    decisions_space: str = "decisions"
    events_space: str = "events"
    actions_space: str = "actions"
    questions_space: str = "questions"

    provenance_edge: str = "DERIVED_FROM"
    about_edge: str = "ABOUT"
    involves_edge: str = "INVOLVES"

    #: Upper bounds per conversation so one chatty extraction can't explode.
    max_per_kind: int = 20

    #: Kinds whose text is scanned for entity mentions to draw ABOUT edges.
    about_source_kinds: Tuple[str, ...] = ("fact", "decision", "action")


def _title_of(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text[:limit]


class _Materializer:
    """Get-or-create typed Items + typed edges against one project.

    Reuses the entity-promotion identity rules (Unicode-aware slug,
    hash-on-truncate, per-slug idempotence) so an ``entity`` written here is
    the same node ``entity_promotion`` would create.
    """

    def __init__(self, project, project_name: str, schema: OntologySchema):
        import kumiho  # noqa: F401 - imported for side-effect availability

        self.project = project
        self.project_name = project_name
        self.schema = schema
        self._ensured_spaces: set = set()
        self._node_cache: Dict[Tuple[str, str], Any] = {}  # (space, slug) -> anchor rev

    def _ensure_space(self, space: str) -> str:
        import grpc

        space_path = f"/{self.project_name}/{space}"
        if space not in self._ensured_spaces:
            try:
                self.project.create_space(space)
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
                    raise
            self._ensured_spaces.add(space)
        return space_path

    def node(self, space: str, kind: str, slug: str, metadata: Dict[str, str]) -> Optional[Any]:
        """Get-or-create the item, return its anchor revision (created once)."""
        import grpc

        cache_key = (space, slug)
        cached = self._node_cache.get(cache_key)
        if cached is not None:
            return cached

        space_path = self._ensure_space(space)
        try:
            try:
                item = self.project.create_item(slug, kind, parent_path=space_path)
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
                    raise
                item = self.project.get_item(slug, kind, parent_path=space_path)
            anchor = item.get_latest_revision()
            if anchor is None:
                anchor = item.create_revision(metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ontology: node %s/%s (%s) failed: %s", space, slug, kind, exc)
            return None
        self._node_cache[cache_key] = anchor
        return anchor

    def edge(self, source_rev, target_rev, edge_type: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        if source_rev is None or target_rev is None:
            return False
        try:
            source_rev.create_edge(target_rev, edge_type, metadata=metadata or {})
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("ontology: edge %s failed: %s", edge_type, exc)
            return False


def _sync_decompose(
    conversation_kref: str,
    summary: Dict[str, Any],
    project_name: str,
    schema: OntologySchema,
) -> Dict[str, int]:
    import kumiho

    project = kumiho.get_project(project_name)
    if project is None:
        return {}
    try:
        conv_rev = kumiho.get_revision(conversation_kref)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ontology: conversation %s unavailable: %s", conversation_kref, exc)
        return {}

    m = _Materializer(project, project_name, schema)
    stats: Dict[str, int] = {"entities": 0, "facts": 0, "decisions": 0,
                             "events": 0, "actions": 0, "questions": 0, "edges": 0}

    # --- Entities (deduped hubs; name -> anchor) ---
    entity_anchors: Dict[str, Any] = {}  # slug -> anchor rev
    entity_display: Dict[str, str] = {}  # slug -> display name
    entity_tokens: Dict[str, List[str]] = {}  # slug -> name word-tokens

    def _ensure_entity(name: str) -> Optional[str]:
        name = (name or "").strip()
        if not name:
            return None
        slug = slugify(name, hash_on_truncate=True)
        if not slug:
            return None
        if slug not in entity_anchors:
            anchor = m.node(schema.entities_space, "entity", slug,
                            {"display_name": name, "promoted_from": conversation_kref})
            if anchor is None:
                return None
            entity_anchors[slug] = anchor
            entity_display[slug] = name
            entity_tokens[slug] = _word_tokens(name)
            stats["entities"] += 1
        return slug

    classification = summary.get("classification") or {}
    for name in (classification.get("entities") or [])[: schema.max_per_kind]:
        _ensure_entity(str(name))
    # Materialize event participants up-front too. The events loop runs *after*
    # fact/decision ABOUT-linking below, so a fact naming a participant-only
    # entity (one absent from classification.entities) would otherwise never get
    # its ABOUT edge. The entity hub set must be order-independent.
    for ev in (summary.get("events") or [])[: schema.max_per_kind]:
        for part in (ev.get("participants") or []):
            _ensure_entity(str(part))

    def _link_about(source_rev, text: str) -> None:
        """ABOUT edges: draw one when an entity's *name tokens* occur as a
        contiguous run in *text* — token equality, not substring (see
        :func:`_mentions`), so ambiguous short names and space-free scripts
        don't spawn false edges."""
        toks = _word_tokens(text)
        for slug, anchor in entity_anchors.items():
            if _mentions(entity_tokens[slug], toks):
                if m.edge(source_rev, anchor, schema.about_edge, {"entity": slug}):
                    stats["edges"] += 1

    # --- Facts (fact_anchors is index-aligned for decision.based_on) ---
    knowledge = summary.get("knowledge") or {}
    fact_anchors: List[Optional[Any]] = []
    fact_entries: List[Tuple[Any, str, str]] = []  # (anchor, slug, claim)
    for fact in (knowledge.get("facts") or [])[: schema.max_per_kind]:
        claim = str(fact.get("claim", "")).strip()
        if not claim:
            fact_anchors.append(None)
            continue
        slug = slugify(claim, hash_on_truncate=True)
        anchor = m.node(schema.facts_space, "fact", slug, {
            "title": _title_of(claim), "summary": claim, "claim": claim,
            "certainty": str(fact.get("certainty", "")),
        })
        fact_anchors.append(anchor)
        if anchor is None:
            continue
        stats["facts"] += 1
        if m.edge(anchor, conv_rev, schema.provenance_edge):
            stats["edges"] += 1
        _link_about(anchor, claim)
        fact_entries.append((anchor, slug, claim))

    # --- Decisions ---
    decision_entries: List[Tuple[Any, str, str, List[int]]] = []  # (anchor, slug, text, based_on)
    for dec in (knowledge.get("decisions") or [])[: schema.max_per_kind]:
        text = str(dec.get("decision", "")).strip()
        if not text:
            continue
        reason = str(dec.get("reason", ""))
        slug = slugify(text, hash_on_truncate=True)
        body = f"{text} (reason: {reason})" if reason else text
        anchor = m.node(schema.decisions_space, "decision", slug, {
            "title": _title_of(text), "summary": body, "decision": text, "reason": reason,
        })
        if anchor is None:
            continue
        stats["decisions"] += 1
        if m.edge(anchor, conv_rev, schema.provenance_edge):
            stats["edges"] += 1
        _link_about(anchor, f"{text} {reason}")
        based_on = [int(i) for i in (dec.get("based_on") or [])
                    if isinstance(i, (int, float))]
        decision_entries.append((anchor, slug, text, based_on))

    # --- Events (participants -> INVOLVES entity) ---
    for ev in (summary.get("events") or [])[: schema.max_per_kind]:
        text = str(ev.get("event", "")).strip()
        if not text:
            continue
        slug = slugify(text, hash_on_truncate=True)
        when = str(ev.get("when", ""))
        anchor = m.node(schema.events_space, "event", slug, {
            "title": _title_of(text), "summary": (f"[{when}] {text}" if when else text),
            "event": text, "when": when, "event_date": str(ev.get("event_date", "")),
            "consequence": str(ev.get("consequence", "")),
        })
        if anchor is None:
            continue
        stats["events"] += 1
        if m.edge(anchor, conv_rev, schema.provenance_edge):
            stats["edges"] += 1
        for part in (ev.get("participants") or []):
            slug_e = _ensure_entity(str(part))
            if slug_e and m.edge(anchor, entity_anchors[slug_e], schema.involves_edge, {"entity": slug_e}):
                stats["edges"] += 1

    # --- Actions ---
    for act in (knowledge.get("actions") or [])[: schema.max_per_kind]:
        task = str(act.get("task", "")).strip()
        if not task:
            continue
        slug = slugify(task, hash_on_truncate=True)
        anchor = m.node(schema.actions_space, "action", slug, {
            "title": _title_of(task), "summary": task, "task": task,
            "status": str(act.get("status", "")),
        })
        if anchor is None:
            continue
        stats["actions"] += 1
        if m.edge(anchor, conv_rev, schema.provenance_edge):
            stats["edges"] += 1
        _link_about(anchor, task)

    # --- Open questions ---
    for q in (knowledge.get("open_questions") or [])[: schema.max_per_kind]:
        text = str(q).strip()
        if not text:
            continue
        slug = slugify(text, hash_on_truncate=True)
        anchor = m.node(schema.questions_space, "question", slug, {
            "title": _title_of(text), "summary": text, "question": text,
        })
        if anchor is None:
            continue
        stats["questions"] += 1
        if m.edge(anchor, conv_rev, schema.provenance_edge):
            stats["edges"] += 1

    # --- Conversation -> entity ABOUT edges ---
    # So the entity-mediated reader can hop from a recalled conversation to its
    # entities and out to sibling memories / facts / decisions about them
    # (mirrors what entity_promotion does in the lighter mode).
    for slug, anchor in entity_anchors.items():
        if m.edge(conv_rev, anchor, schema.about_edge, {"entity": slug}):
            stats["edges"] += 1

    # --- Relational edges (DEPENDS_ON via emitted indices, SUPERSEDES via
    #     subject overlap) ---
    from .relations import link_depends_on, link_supersedes

    for anchor, slug, text, based_on in decision_entries:
        stats["edges"] += link_depends_on(m, anchor, based_on, fact_anchors)
        stats["edges"] += link_supersedes(
            m, "decision", schema.decisions_space, slug, anchor, text, project_name,
        )
    for anchor, slug, claim in fact_entries:
        stats["edges"] += link_supersedes(
            m, "fact", schema.facts_space, slug, anchor, claim, project_name,
        )

    return stats


async def decompose_and_link(
    conversation_kref: str,
    summary: Dict[str, Any],
    *,
    project_name: str,
    schema: Optional[OntologySchema] = None,
    timeout: float = 25.0,
) -> Dict[str, int]:
    """Decompose *summary* into a typed graph anchored on *conversation_kref*.

    Best-effort and bounded; returns per-kind counts (all zero on
    timeout/error). Runs off the caller's event loop.
    """
    if not conversation_kref or not summary:
        return {}
    sch = schema or OntologySchema()
    result = await run_bounded_in_thread(
        lambda: _sync_decompose(conversation_kref, summary, project_name, sch),
        timeout=timeout,
        label=f"ontology decomposition ({conversation_kref})",
        on_timeout={},
        on_error={},
    )
    stats = result or {}
    if stats:
        logger.debug("ontology decomposition of %s: %s", conversation_kref, stats)
    return stats
