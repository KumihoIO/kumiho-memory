"""One revision-scoped protocol for replacing facts and decisions.

Edge first, then demote the exact target revision and invalidate its grounding.
Replays repair partial writes even when the edge already exists. This is not a
transaction: callers must surface/retry failures, not mark incomplete work done.
Profile-history SUPERSEDES edges are not belief replacements and stay separate.
"""
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from .grounding import ripple_grounding_stale

logger = logging.getLogger(__name__)


@dataclass
class SupersessionResult:
    created: bool = False
    linked: bool = False
    demoted: bool = False
    stale: int = 0
    error: str = ""


def supersede_revision(
    source: Any, target: Any, metadata: Optional[Dict[str, str]] = None,
) -> SupersessionResult:
    """Ensure source SUPERSEDES target; never demote without a confirmed edge.

Returns new-write counters separately from edge existence. On a metadata
failure the edge remains useful and a replay retries demotion and the bounded
grounding ripple. Unrelated metadata and the target's revision identity survive.
"""
    result = SupersessionResult()
    src = getattr(getattr(source, "kref", None), "uri", "")
    dst = getattr(getattr(target, "kref", None), "uri", "")
    if source is None or target is None or source is target or not src or not dst or src == dst:
        result.error = "Supersession requires two distinct revision references"
        return result
    try:
        try:
            result.linked = any(
                getattr(getattr(edge, "target_kref", None), "uri", "") == dst
                for edge in source.get_edges(edge_type_filter="SUPERSEDES", direction=0)
            )
        except Exception:
            # Match existing writers: a failed precheck may still create. Only
            # a successful create (or an observed edge) authorizes demotion.
            pass
        if not result.linked:
            source.create_edge(target, "SUPERSEDES", metadata=metadata or {})
            result.created = result.linked = True
    except Exception as exc:
        result.error = f"Supersession edge failed: {exc}"
        logger.warning("%s", result.error)
        return result

    try:
        if (getattr(target, "metadata", {}) or {}).get("status") != "superseded":
            if target.set_attribute("status", "superseded") is False:
                raise RuntimeError("status update rejected")
            result.demoted = True
    except Exception as exc:
        result.error = f"Supersession status failed: {exc}"
        logger.warning("%s", result.error)
    # Also retry on an existing edge: a previous process may have stopped
    # between edge creation and ripple. Decisions can ground other decisions.
    result.stale = ripple_grounding_stale(target, src)
    return result
