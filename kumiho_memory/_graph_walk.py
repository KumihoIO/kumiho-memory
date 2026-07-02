"""Shared paged walkers over the Kumiho graph (spaces and items).

Extracted from ``DreamState`` so ``SpaceProfiler`` can reuse the same
bounded-page enumeration without duplicating the fallback logic.  The
behavior is identical to the original DreamState methods.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def list_project_spaces(
    project: Any,
    project_name: str,
    page_size: int = 100,
) -> List[Any]:
    """Enumerate project spaces without relying on one recursive RPC."""
    root_path = f"/{project_name}"
    discovered: List[Any] = []
    seen_paths = set()
    pending_paths = [root_path]

    while pending_paths:
        parent_path = pending_paths.pop(0)
        cursor: Optional[str] = None

        while True:
            try:
                page = project.get_spaces(
                    parent_path=parent_path,
                    recursive=False,
                    page_size=page_size,
                    cursor=cursor,
                )
            except TypeError:
                # Older SDK stubs/tests only support the legacy recursive API.
                spaces = list(project.get_spaces(recursive=True))
                logger.info(
                    "Using legacy recursive space enumeration for project %s",
                    project_name,
                )
                return spaces
            except Exception as exc:
                raise RuntimeError(
                    "Failed to list child spaces under "
                    f"'{parent_path}' (cursor={cursor or '-'})"
                ) from exc

            children = list(page)
            for space in children:
                path = getattr(space, "path", "")
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                discovered.append(space)
                pending_paths.append(path)

            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                break

    return discovered


def list_space_items(
    sdk: Any,
    space_path: str,
    *,
    kind_filter: str = "",
    page_size: int = 100,
    include_deprecated: bool = False,
) -> List[Any]:
    """List items in a space in bounded pages to avoid RPC deadlines."""
    client = sdk.get_client()
    collected: List[Any] = []
    cursor: Optional[str] = None

    while True:
        try:
            page = client.get_items(
                parent_path=space_path,
                kind_filter=kind_filter,
                page_size=page_size,
                cursor=cursor,
                include_deprecated=include_deprecated,
            )
        except TypeError:
            page = client.get_items(
                parent_path=space_path,
                kind_filter=kind_filter,
                include_deprecated=include_deprecated,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to list items in "
                f"'{space_path}' (cursor={cursor or '-'})"
            ) from exc

        collected.extend(list(page))

        cursor = getattr(page, "next_cursor", None)
        if not cursor:
            break

    return collected
