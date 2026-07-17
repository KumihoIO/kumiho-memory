"""The ontology as a versioned, fetchable policy Item (ontology G2).

:class:`~kumiho_memory.ontology.OntologySchema` declares the node kinds and
edge rules in code; its docstring promised persistence "as a policy Item
without touching this code". This module discharges that promise: it BUILDS a
serializable spec — node kinds with natural-language definitions, edge
semantics and direction, the canonical relation registry, and the trust-
vocabulary mapping — and SEEDS it as one tagged revision that any agent can
fetch (``kumiho_get_revision_by_tag`` on the :data:`SPEC_TAG` tag) and commit
to. Ontological commitment stops being "read the Python source".

Seeding is idempotent and exercises the Item/Revision machinery on purpose:

- fresh   -> create the item + a revision, move :data:`SPEC_TAG` onto it;
- re-seed at the same ``spec_version`` -> no-op (no new revision);
- version bump -> a NEW revision on the SAME item, tag moves to it.

Structured content lives in the revision's ``content`` metadata as JSON
(metadata values must be strings), mirroring how ``space_profiler`` persists a
versioned policy item. Best-effort: any failure is logged and swallowed
(returns ``None``), never fatal — the same contract onboarding writes use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import trust_vocab
from .ontology import OntologySchema
from .predicate_registry import RELATES_TO, canonical_types, registry_as_dict

logger = logging.getLogger(__name__)

#: Default project the spec is seeded into (the memory root, as used by
#: skill ingestion and the space profiler).
DEFAULT_PROJECT = "CognitiveMemory"

#: Dedicated policy space / item / kind for the spec.
SPEC_SPACE = "ontology"
SPEC_ITEM = "spec"
SPEC_KIND = "policy"

#: Tag the canonical revision carries, so ``get_revision_by_tag`` resolves it.
#: Named analogously to the ``published`` tag skill ingestion moves onto its
#: canonical revision.
SPEC_TAG = "ontology.spec"


def build_spec(schema: Optional[OntologySchema] = None) -> Dict[str, Any]:
    """Build the serializable ontology spec from the live code + registries.

    Sourced from :class:`OntologySchema` (spaces, edge names, version), the
    predicate registry (canonical relation types + synonyms + fallback), and
    :mod:`kumiho_memory.trust_vocab` (the trust mapping) so the persisted spec
    can never drift from what the writer actually applies.
    """
    sch = schema or OntologySchema()
    return {
        "spec_version": sch.version,
        "description": (
            "The shared semantic contract for Kumiho's typed memory graph: "
            "node kinds, edge semantics, the canonical relation registry, and "
            "the trust-vocabulary mapping. Agents commit to this."
        ),
        "node_kinds": _node_kinds(sch),
        "edge_types": _edge_types(sch),
        "relation_registry": _relation_registry(),
        "trust_vocabulary": trust_vocab.mapping_as_dict(),
    }


def _node_kinds(sch: OntologySchema) -> Dict[str, Any]:
    """The six node kinds: definition, space, metadata fields, identity rule.

    Metadata fields and identity rules mirror what ``ontology.py``
    materializes; identity is a Unicode-aware slug (hash-on-truncate) of the
    node's defining text (``slug-of-name`` for entities, ``slug-of-text`` for
    the rest).
    """
    slug_of_name = "slug-of-name (Unicode-aware, hash-on-truncate)"
    slug_of_text = "slug-of-text (Unicode-aware, hash-on-truncate)"
    return {
        "entity": {
            "definition": "A named thing (person, system, org, concept) that "
                          "facts/decisions/actions are about; deduped into a "
                          "global hub across sessions.",
            "space": sch.entities_space,
            "metadata_fields": ["display_name", "promoted_from",
                                "entity_type", "aliases"],
            "identity_rule": slug_of_name,
        },
        "fact": {
            "definition": "A claim asserted in conversation.",
            "space": sch.facts_space,
            "metadata_fields": ["title", "summary", "claim", "certainty",
                                "fact_type"],
            "identity_rule": slug_of_text,
        },
        "decision": {
            "definition": "A choice that was made, with its reason.",
            "space": sch.decisions_space,
            "metadata_fields": ["title", "summary", "decision", "reason"],
            "identity_rule": slug_of_text,
        },
        "event": {
            "definition": "Something that happened, optionally dated.",
            "space": sch.events_space,
            "metadata_fields": ["title", "summary", "event", "when",
                                "event_date", "consequence"],
            "identity_rule": slug_of_text,
        },
        "action": {
            "definition": "A task with a status.",
            "space": sch.actions_space,
            "metadata_fields": ["title", "summary", "task", "status"],
            "identity_rule": slug_of_text,
        },
        "question": {
            "definition": "An open question left unresolved.",
            "space": sch.questions_space,
            "metadata_fields": ["title", "summary", "question"],
            "identity_rule": slug_of_text,
        },
    }


def _edge_types(sch: OntologySchema) -> Dict[str, Any]:
    """Structural + relational edges with semantics and direction.

    SUPERSEDES carries its current basis explicitly (lexical token-overlap,
    newest-wins) so agents can tell a heuristic belief-update edge from a
    semantic one.
    """
    return {
        sch.provenance_edge: {
            "semantics": "provenance: the typed node was extracted from the "
                         "conversation it points at.",
            "direction": "<typed node> --DERIVED_FROM--> conversation",
        },
        sch.about_edge: {
            "semantics": "the source node mentions / is about the target "
                         "entity (token-run mention match, conservative).",
            "direction": "fact|decision|action|conversation --ABOUT--> entity",
        },
        sch.involves_edge: {
            "semantics": "an event involves an entity participant.",
            "direction": "event --INVOLVES--> entity",
        },
        "DEPENDS_ON": {
            "semantics": "a decision is grounded in the fact it was based on.",
            "direction": "decision --DEPENDS_ON--> fact",
        },
        "SUPERSEDES": {
            "semantics": "a newer node replaces an older same-subject node "
                         "(belief update).",
            "direction": "newer --SUPERSEDES--> older (newest-wins)",
            "basis": "lexical token-Jaccard overlap >= 0.6 between the two "
                     "nodes' text; corpus-independent, no semantic conflict "
                     "check.",
        },
    }


def _relation_registry() -> Dict[str, Any]:
    """The canonical entity->entity relation vocabulary, embedded from
    :mod:`kumiho_memory.predicate_registry`."""
    return {
        "canonical_types": list(canonical_types()),
        "synonyms": {canonical: list(synonyms)
                     for canonical, synonyms in registry_as_dict().items()},
        "fallback": {
            "type": RELATES_TO,
            "rule": "an unregistered or unnormalizable predicate folds onto "
                    "RELATES_TO; the verbatim predicate is preserved in edge "
                    "metadata and never dropped.",
        },
    }


@dataclass
class SeedResult:
    """Outcome of a :func:`seed_ontology_spec` call.

    ``created_revision`` is ``False`` when a re-seed at the same version was a
    no-op; ``revision_kref`` then points at the existing tagged revision.
    """

    item_kref: str
    revision_kref: str
    version: str
    created_item: bool
    created_revision: bool


def _get_or_create_item(project: Any, grpc: Any, project_name: str):
    """Get-or-create the spec item (and its space), mirroring the ontology
    materializer's ALREADY_EXISTS handling (``ontology.py``)."""
    space_path = f"/{project_name}/{SPEC_SPACE}"
    try:
        project.create_space(SPEC_SPACE)
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise
    try:
        item = project.create_item(SPEC_ITEM, SPEC_KIND, parent_path=space_path)
        return item, True
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise
        return project.get_item(SPEC_ITEM, SPEC_KIND, parent_path=space_path), False


def seed_ontology_spec(
    project_name: str = DEFAULT_PROJECT,
    schema: Optional[OntologySchema] = None,
) -> Optional[SeedResult]:
    """Idempotently seed the ontology spec as a tagged policy revision.

    Re-seeding at the same ``spec_version`` is a no-op; a version bump creates
    a new revision on the same item and moves :data:`SPEC_TAG` onto it (the
    server holds a tag on a single revision per item). Best-effort: returns
    ``None`` on any failure, logging it — never raises.
    """
    import grpc
    import kumiho

    sch = schema or OntologySchema()
    version = sch.version
    try:
        project = kumiho.get_project(project_name)
        if project is None:
            return None

        item, created_item = _get_or_create_item(project, grpc, project_name)

        existing = item.get_revision_by_tag(SPEC_TAG)
        if existing is not None and \
                (getattr(existing, "metadata", {}) or {}).get("spec_version") == version:
            return SeedResult(
                item_kref=str(item.kref),
                revision_kref=str(existing.kref),
                version=version,
                created_item=created_item,
                created_revision=False,
            )

        content = build_spec(sch)
        revision = item.create_revision(metadata={
            "type": "ontology_spec",
            "title": f"Kumiho agent-memory ontology spec ({version})",
            "summary": "Node kinds, edge semantics, relation registry, and "
                       "trust-vocabulary mapping the typed graph commits to.",
            "spec_version": version,
            "content": json.dumps(content, ensure_ascii=False),
        })
        revision.tag(SPEC_TAG)  # server moves the tag off any prior revision
        return SeedResult(
            item_kref=str(item.kref),
            revision_kref=str(revision.kref),
            version=version,
            created_item=created_item,
            created_revision=True,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort seed, never fatal
        logger.warning("ontology spec seed failed: %s", exc)
        return None
