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
  depend on corpus-wide ranking stability.
- **Anchor revision** — each entity Item carries a single anchor revision
  (r1) holding ``display_name`` (the first raw surface form). ``ABOUT``
  edges from memory revisions always target the anchor so traversal has
  one stable hub per entity.
- **``ABOUT`` edges** — memory revision → entity anchor. Recall traversal
  (graph_augmentation) walks these both ways: memory → entity → sibling
  memories about the same entity, which is multi-hop signal that pure
  vector similarity misses.

The flattened ``entities`` metadata string is still written by the
consolidation path — the server aggregates it into item ``_search_text``
(fulltext prospect indexing), so removing it would regress text recall.

Everything here is best-effort enrichment: failures are logged and
swallowed, never blocking a store. The synchronous gRPC calls run in a
daemon thread polled against a deadline (see graph_augmentation.py for
the Windows-hang rationale).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LEN = 48


def _slugify_entity(name: str) -> str:
    """Deterministic slug used as the entity Item name (identity key)."""
    base = name.lower().strip()
    base = _SLUG_PATTERN.sub("-", base).strip("-")
    return base[:_MAX_SLUG_LEN].strip("-")


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

    #: Deadline for the whole promotion batch (daemon-thread poll).
    timeout: float = 60.0

    #: Extra metadata stamped on every ABOUT edge.
    edge_metadata: Dict[str, str] = field(default_factory=dict)


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

    project = kumiho.get_project(project_name)
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
            try:
                item = project.create_item(slug, "entity", parent_path=space_path)
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
                    raise
                item = project.get_item(slug, "entity", parent_path=space_path)

            anchor = item.get_latest_revision()
            if anchor is None:
                anchor = item.create_revision(
                    metadata={"display_name": raw_name, "promoted_from": revision_kref}
                )
            touched += 1

            edge_metadata = {"entity": slug, **config.edge_metadata}
            source_rev.create_edge(anchor, config.edge_type, metadata=edge_metadata)
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

    Best-effort and bounded: runs the blocking SDK calls in a daemon
    thread polled against ``config.timeout`` so a hung RPC cannot strand
    the event loop or a shared executor thread.
    """
    cfg = config or EntityPromotionConfig()
    if not cfg.enabled or not entity_names or not revision_kref:
        return {"entities": 0, "edges": 0}

    result: List[Tuple[int, int]] = []
    done_event = threading.Event()

    def _worker() -> None:
        try:
            result.append(
                _sync_promote(revision_kref, list(entity_names), project_name, cfg)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("entity promotion failed for %s: %s", revision_kref, exc)
        finally:
            done_event.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    deadline = time.monotonic() + cfg.timeout
    while not done_event.is_set():
        if time.monotonic() >= deadline:
            logger.debug(
                "entity promotion timed out after %.0fs for %s", cfg.timeout, revision_kref
            )
            return {"entities": 0, "edges": 0, "timed_out": True}
        await asyncio.sleep(0.05)

    touched, edges = result[0] if result else (0, 0)
    if touched:
        logger.debug(
            "entity promotion: %d entities, %d ABOUT edges from %s",
            touched, edges, revision_kref,
        )
    return {"entities": touched, "edges": edges}
