"""Entity promotion — first-class entity Items with ``ABOUT`` edges.

The summarizer already extracts ``classification.entities`` for every
consolidated memory, but historically those names were flattened into a
comma-joined revision-metadata string and stopped being graph citizens:
no identity, no dedup, no traversal. ``Anthropic``, ``anthropic`` and
``Anthropic AI`` would drift apart as strings, and "what do I know about
X" could only ever be a fuzzy-text question.

This module promotes each extracted entity to an Item of kind
``entity`` in a per-project entities Space:

    kref://<project>/<entities_space>/<slug>.entity

- **Identity-keyed idempotence** — the item name is a deterministic slug
  of the entity name, and creation is get-or-create, so re-encounters
  resolve to the same node instead of minting variants. Unlike revision
  stacking (which keys on a search-score threshold), this dedup does not
  depend on corpus-wide ranking stability. The slug is Unicode-aware
  (Korean/CJK names survive) and hash-suffixed on truncation so two long
  names sharing a prefix can't collide into one entity.
- **Anchor revision** — each entity Item carries a single anchor revision
  (r1) holding ``display_name``. ``ABOUT`` edges always target the anchor,
  so traversal has one stable hub per entity. Anchor creation is
  serialized per-slug within the process to avoid two concurrent
  promotions minting two anchors.
- **``ABOUT`` edges** — memory revision → entity anchor.

The flattened ``entities`` metadata string is still written by the
consolidation path — the server aggregates it into item ``_search_text``
(fulltext prospect indexing), so removing it would regress text recall.

Everything here is best-effort enrichment: failures are logged and
swallowed, never blocking a store, and the blocking SDK calls run in a
bounded daemon thread (see ``_bounded.run_bounded_in_thread``).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from kumiho._text import slugify

from ._bounded import run_bounded_in_thread

logger = logging.getLogger(__name__)

# Per-slug locks serialize anchor get-or-create across the daemon threads a
# single process spawns, so concurrent promotions of the same *new* entity
# don't each create an anchor. (Cross-process races still need server-side
# get-or-create — tracked as a follow-up.)
_anchor_locks_guard = threading.Lock()
_anchor_locks: Dict[str, threading.Lock] = {}

# Cache resolved Project handles by name to avoid a ListProjects per store.
_project_cache: Dict[str, Any] = {}


def _slugify_entity(name: str) -> str:
    """Deterministic, Unicode-aware slug used as the entity Item name."""
    return slugify(name, hash_on_truncate=True)


def _anchor_lock(slug: str) -> threading.Lock:
    with _anchor_locks_guard:
        lock = _anchor_locks.get(slug)
        if lock is None:
            lock = threading.Lock()
            _anchor_locks[slug] = lock
        return lock


@dataclass
class EntityPromotionConfig:
    """Configuration for write-time entity promotion."""

    enabled: bool = True

    #: Space (under the project root) that holds all entity Items.
    entities_space: str = "entities"

    #: Edge type from a memory revision to an entity anchor revision.
    edge_type: str = "ABOUT"

    #: Upper bound on entities promoted per memory — keeps a chatty
    #: extraction from fanning out into dozens of writes.
    max_entities: int = 8

    #: Deadline for the whole promotion batch (daemon-thread poll). Kept
    #: modest since this runs on the consolidation path.
    timeout: float = 15.0


def _resolve_project(project_name: str):
    import kumiho

    cached = _project_cache.get(project_name)
    if cached is not None:
        return cached
    project = kumiho.get_project(project_name)
    if project is not None:
        _project_cache[project_name] = project
    return project


def _get_or_create_entity_item(project, space_path: str, slug: str):
    """create-or-get an ``entity`` Item (mirrors mcp_server._get_or_create_item,
    kept local to avoid importing the heavy mcp_server module)."""
    import grpc

    try:
        return project.create_item(slug, "entity", parent_path=space_path)
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise
        return project.get_item(slug, "entity", parent_path=space_path)


def _sync_promote(
    revision_kref: str,
    entity_names: List[str],
    project_name: str,
    config: EntityPromotionConfig,
) -> Tuple[int, int]:
    """Blocking worker: get-or-create entity items and link them.

    Returns ``(entities_touched, edges_created)``.
    """
    import grpc
    import kumiho

    project = _resolve_project(project_name)
    if project is None:
        logger.debug("entity promotion: project %s not found", project_name)
        return (0, 0)

    # Ensure the entities space exists (idempotent).
    space_path = f"/{project_name}/{config.entities_space}"
    try:
        project.create_space(config.entities_space)
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise

    try:
        source_rev = kumiho.get_revision(revision_kref)
    except Exception as exc:  # noqa: BLE001
        logger.debug("entity promotion: source revision %s unavailable: %s", revision_kref, exc)
        return (0, 0)

    # Deterministic dedup within the batch, preserving first surface form.
    seen: Dict[str, str] = {}
    for raw in entity_names:
        name = str(raw).strip()
        if not name:
            continue
        slug = _slugify_entity(name)
        if slug and slug not in seen:
            seen[slug] = name
        if len(seen) >= config.max_entities:
            break

    touched = 0
    edges = 0
    for slug, raw_name in seen.items():
        try:
            item = _get_or_create_entity_item(project, space_path, slug)

            # Serialize anchor creation per slug so concurrent promotions of
            # the same new entity converge on one anchor revision.
            with _anchor_lock(slug):
                anchor = item.get_latest_revision()
                if anchor is None:
                    anchor = item.create_revision(
                        metadata={"display_name": raw_name, "promoted_from": revision_kref}
                    )
            touched += 1

            source_rev.create_edge(anchor, config.edge_type, metadata={"entity": slug})
            edges += 1
        except Exception as exc:  # noqa: BLE001 - per-entity failures never block the rest
            logger.debug(
                "entity promotion: %s -> %s failed: %s", revision_kref, slug, exc
            )
    return (touched, edges)


async def promote_entities(
    revision_kref: str,
    entity_names: List[str],
    *,
    project_name: str,
    config: Optional[EntityPromotionConfig] = None,
) -> Dict[str, Any]:
    """Promote *entity_names* to entity Items linked from *revision_kref*.

    Best-effort and bounded: runs the blocking SDK calls in a daemon thread
    polled against ``config.timeout`` so a hung RPC cannot strand the event
    loop or a shared executor thread.
    """
    cfg = config or EntityPromotionConfig()
    if not cfg.enabled or not entity_names or not revision_kref:
        return {"entities": 0, "edges": 0}

    outcome = await run_bounded_in_thread(
        lambda: _sync_promote(revision_kref, list(entity_names), project_name, cfg),
        timeout=cfg.timeout,
        label=f"entity promotion ({revision_kref})",
        on_timeout=(0, 0),
        on_error=(0, 0),
    )
    touched, edges = outcome or (0, 0)
    if touched:
        logger.debug(
            "entity promotion: %d entities, %d ABOUT edges from %s",
            touched, edges, revision_kref,
        )
    return {"entities": touched, "edges": edges}
