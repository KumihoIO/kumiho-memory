"""Space-level knowledge profiles — observe each Space's dynamics.

A collection's observed dynamics are themselves a signal about what kind
of knowledge lives in it: a Space whose contents churn fast, carry low
evidence grades, and rarely stabilize is probably a claims/requests
collection — not established concepts — and extraction/consolidation
strategy should adapt per Space instead of applying one global policy.

``SpaceProfiler`` aggregates per-Space statistics from existing SDK
queries (no kumiho-server changes), classifies each Space, and persists
the result as a ``kind="space-profile"`` Item — one per Space, revised
each run, so the profile itself is versioned memory and ``SUPERSEDES``
chains show profile drift.

Labels
------

- ``canonical`` — low churn, high stability (published-heavy, old
  medians) → established concepts
- ``working`` — moderate churn, mixed evidence → active projects/notes
- ``correspondence`` — high churn, low stability, fast supersession →
  claims / requests / responses

A Space owner can pin the label via the ``space_class`` Space attribute;
the profiler then only reports drift instead of relabeling.

Consumers (:func:`get_space_profile`): the evidence assessor, Dream
State ``extra_instructions``, and recall can adapt per-space strategy by
reading the latest profile revision.

Note: true "latest tag-move frequency" is not enumerable client-side
(the SDK exposes point-in-time tag resolution, not tag-move events), so
churn uses revision-creation frequency as the proxy — valid because the
``latest`` tag moves on every ``create_revision``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kumiho_memory import _graph_walk as _walk
from kumiho_memory.evidence import parse_evidence

logger = logging.getLogger(__name__)

CANONICAL = "canonical"
WORKING = "working"
CORRESPONDENCE = "correspondence"

#: All valid space-class labels.
SPACE_CLASSES = (CANONICAL, WORKING, CORRESPONDENCE)

#: Weight of each evidence level when scoring a Space's evidence quality.
_EVIDENCE_QUALITY = {
    "official": 1.0,
    "corroborated": 0.7,
    "single_source": 0.3,
    "unverified": 0.0,
}


@dataclass
class SpaceSignals:
    """Raw per-Space statistics computed from existing SDK queries.

    Stability-family signals (``published_share``,
    ``median_revision_age_days``, ``evidence_histogram``) describe the
    **live** (non-deprecated) revisions only — dead history must not
    make a churned space look established.  Churn-family counts include
    deprecated revisions because historical stacking is exactly what
    churn measures.
    """

    items_count: int = 0
    revisions_count: int = 0
    live_revisions_count: int = 0
    revisions_per_item_mean: float = 0.0
    revision_rate_per_day: float = 0.0
    supersedes_edge_count: int = 0
    supersedes_max_depth: int = 0
    evidence_histogram: Dict[str, int] = field(default_factory=dict)
    deprecated_items: int = 0
    deprecated_revisions: int = 0
    deprecation_ratio: float = 0.0
    published_share: float = 0.0
    median_revision_age_days: float = 0.0
    window_start: str = ""
    window_end: str = ""


@dataclass
class SpaceProfile:
    """Classification result for one Space."""

    space_path: str
    signals: SpaceSignals
    scores: Dict[str, float]
    label: str
    pinned: bool = False
    previous_label: Optional[str] = None
    observed_label: Optional[str] = None
    """Signals-derived label, kept even when a pin overrides ``label`` —
    pinned spaces report pin/observation disagreement as drift."""
    empty: bool = False
    """True when the space had zero observed revisions — the profile is
    not persisted (no data is not the same as observed-moderate-churn)."""


def classify(signals: SpaceSignals) -> Tuple[Dict[str, float], str]:
    """Score and label a Space from its signals (pure, no I/O).

    Returns ``(scores, observed_label)``.  Override/pin handling lives
    in the profiler so the observed label is always computed.

    Scores (each 0..1):

    - ``churn`` — revision stacking depth, revision rate in the window,
      and SUPERSEDES chain depth
    - ``evidence`` — quality-weighted share of graded live revisions
      (ungraded revisions count as 0, i.e. unverified-equivalent)
    - ``stability`` — published share and median revision age of live
      revisions

    Label thresholds (documented, deliberately coarse):

    - ``canonical`` — stability >= 0.6, churn <= 0.4, and (when any
      revision carries a grade) evidence >= 0.3 — a stable but
      low-evidence graded corpus is ``working``, per the issue's
      "low churn, high evidence" definition; ungraded corpora are not
      penalized for predating the evidence schema
    - ``correspondence`` — churn >= 0.6 and stability <= 0.4
    - ``working`` — everything else
    """
    churn = min(
        1.0,
        0.5 * min(1.0, signals.revisions_per_item_mean / 5.0)
        + 0.3 * min(1.0, signals.revision_rate_per_day / 3.0)
        + 0.2 * min(1.0, signals.supersedes_max_depth / 5.0),
    )

    graded_count = sum(signals.evidence_histogram.values())
    graded_quality = sum(
        _EVIDENCE_QUALITY.get(level, 0.0) * count
        for level, count in signals.evidence_histogram.items()
    )
    evidence = (
        graded_quality / signals.live_revisions_count
        if signals.live_revisions_count
        else 0.0
    )

    stability = min(
        1.0,
        0.6 * signals.published_share
        + 0.4 * min(1.0, signals.median_revision_age_days / 30.0),
    )

    scores = {
        "churn": round(churn, 4),
        "evidence": round(evidence, 4),
        "stability": round(stability, 4),
    }

    # Thresholds compare the ROUNDED scores — the persisted values must
    # never contradict the documented label decision (float epsilon at
    # the 0.3 boundary would otherwise label 0.2999... as working while
    # persisting evidence=0.3).
    if (
        scores["stability"] >= 0.6
        and scores["churn"] <= 0.4
        and (graded_count == 0 or scores["evidence"] >= 0.3)
    ):
        label = CANONICAL
    elif scores["churn"] >= 0.6 and scores["stability"] <= 0.4:
        label = CORRESPONDENCE
    else:
        label = WORKING
    return scores, label


def _parse_created_at(value: Any) -> Optional[datetime]:
    """Defensive ISO timestamp parse (same posture as Dream State)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError):
        return None


def _item_name_and_kind(item: Any) -> Tuple[str, str]:
    """Extract ``(name, kind)`` from an item's kref URI, best-effort."""
    uri = ""
    kref = getattr(item, "kref", None)
    if kref is not None:
        uri = getattr(kref, "uri", "") or str(kref)
    last = uri.rsplit("/", 1)[-1]
    if "." in last:
        name, kind = last.split(".", 1)
        return name, kind
    return last, ""


def _snapshot_tags(rev: Any) -> List[str]:
    """A revision's tags without triggering the SDK's auto-refresh RPC.

    ``Revision.tags`` is a property that refreshes over gRPC once >5s
    stale; the construction-time ``_cached_tags`` snapshot is fresh
    enough for signal aggregation (same posture as Dream State).
    """
    try:
        tags = getattr(rev, "_cached_tags", None)
        if tags is None:
            tags = getattr(rev, "tags", None)
        return [t for t in (tags or []) if isinstance(t, str)]
    except Exception:
        return []


def _pick_latest_revision(revisions: List[Any]) -> Optional[Any]:
    """The item's newest revision, robust to server ordering.

    The real server returns revisions newest-FIRST (``ORDER BY number
    DESC``) — naively taking the last element walks SUPERSEDES chains
    from the oldest revision, which has no outgoing edges.  Prefer the
    ``latest`` flag, then the highest ``number``, then the last element.
    """
    if not revisions:
        return None
    for rev in revisions:
        if getattr(rev, "latest", False) is True:
            return rev
    numbered = [
        rev for rev in revisions
        if isinstance(getattr(rev, "number", None), int)
        and not isinstance(getattr(rev, "number", None), bool)
    ]
    if numbered:
        return max(numbered, key=lambda rev: rev.number)
    return revisions[-1]


class SpaceProfiler:
    """Aggregate per-Space signals and persist versioned profiles.

    Pure aggregation — no LLM involved.  Mirrors the ``DreamState``
    construction/run conventions (env-tuned page sizes, ``dry_run``,
    counters dict result).

    Parameters
    ----------
    project:
        Kumiho project name (default ``CognitiveMemory``).
    profile_kind:
        Item kind used for persisted profiles (default ``space-profile``).
    profile_item_name:
        Item name for the per-Space profile item (default
        ``_space_profile``).
    window_days:
        Look-back window for the revision-rate signal.
    max_supersedes_depth:
        Bound on SUPERSEDES chain walking per item (latest revision
        only, to bound RPC count).
    dry_run:
        Compute and classify but do not persist profiles.
    """

    def __init__(
        self,
        *,
        project: str = "CognitiveMemory",
        profile_kind: str = "space-profile",
        profile_item_name: str = "_space_profile",
        window_days: int = 30,
        max_supersedes_depth: int = 10,
        dry_run: bool = False,
    ) -> None:
        self.project = project
        self.profile_kind = profile_kind
        self.profile_item_name = profile_item_name
        self.window_days = max(1, window_days)
        self.max_supersedes_depth = max(0, max_supersedes_depth)
        self.dry_run = dry_run

        import os

        self.space_page_size = max(
            1,
            int(os.getenv("KUMIHO_DREAM_STATE_SPACE_PAGE_SIZE", "100")),
        )
        self.item_page_size = max(
            1,
            int(os.getenv("KUMIHO_DREAM_STATE_ITEM_PAGE_SIZE", "100")),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Profile every Space in the project.

        Returns a counters dict: ``spaces_profiled``, per-label counts,
        drift records, and errors.
        """
        import kumiho

        start = time.monotonic()
        result: Dict[str, Any] = {
            "success": True,
            "spaces_profiled": 0,
            "labels": {},
            "drift": [],
            "profiles": {},
            "errors": [],
            "dry_run": self.dry_run,
        }

        try:
            project = kumiho.get_project(self.project)
            if project is None:
                raise RuntimeError(f"Project '{self.project}' not found")

            spaces = await asyncio.to_thread(
                _walk.list_project_spaces,
                project, self.project, self.space_page_size,
            )
            # Keep the Space handles: attribute reads (space_class pin)
            # must go through Space.get_attribute — bare paths fail the
            # module-level Kref validation.  The project root has no
            # Space handle (None).
            space_handles: Dict[str, Any] = {f"/{self.project}": None}
            space_paths = [f"/{self.project}"]
            for space in spaces:
                path = getattr(space, "path", "")
                if path and path not in space_handles:
                    space_handles[path] = space
                    space_paths.append(path)

            for space_path in space_paths:
                try:
                    profile = await asyncio.to_thread(
                        self._profile_space,
                        kumiho, space_path, space_handles.get(space_path),
                    )
                except Exception as exc:
                    logger.warning(
                        "SpaceProfiler: failed to profile %s: %s",
                        space_path, exc,
                    )
                    result["errors"].append(f"{space_path}: {exc}")
                    continue

                result["spaces_profiled"] += 1
                result["profiles"][space_path] = {
                    "label": profile.label,
                    "observed_label": profile.observed_label,
                    "pinned": profile.pinned,
                    "empty": profile.empty,
                    "scores": profile.scores,
                }
                if profile.empty:
                    # No live observations — nothing persisted, no label
                    # tallied ("no data" is not a classification).
                    continue
                result["labels"][profile.label] = (
                    result["labels"].get(profile.label, 0) + 1
                )
                if profile.pinned:
                    # Pinned spaces relabel nothing — they report
                    # pin/observation disagreement as drift instead.
                    if (
                        profile.observed_label
                        and profile.observed_label != profile.label
                    ):
                        result["drift"].append({
                            "space_path": space_path,
                            "from": profile.label,
                            "to": profile.observed_label,
                            "pinned": True,
                        })
                elif (
                    profile.previous_label
                    and profile.previous_label != profile.label
                ):
                    result["drift"].append({
                        "space_path": space_path,
                        "from": profile.previous_label,
                        "to": profile.label,
                        "pinned": False,
                    })

        except Exception as exc:
            logger.exception("SpaceProfiler run failed")
            result["success"] = False
            result["errors"].append(str(exc))

        result["duration_ms"] = int((time.monotonic() - start) * 1000)
        return result

    # ------------------------------------------------------------------
    # Signal collection
    # ------------------------------------------------------------------

    def _profile_space(
        self,
        sdk: Any,
        space_path: str,
        space_handle: Optional[Any] = None,
    ) -> SpaceProfile:
        """Collect signals, classify, and (unless dry_run) persist."""
        signals = self.collect_signals(sdk, space_path)
        override = self._read_override(sdk, space_path, space_handle)
        previous_label = self._read_previous_label(sdk, space_path)
        scores, observed_label = classify(signals)
        pinned = override is not None
        label = override if pinned else observed_label

        profile = SpaceProfile(
            space_path=space_path,
            signals=signals,
            scores=scores,
            label=label,
            pinned=pinned,
            previous_label=previous_label,
            observed_label=observed_label,
            empty=signals.revisions_count == 0,
        )

        if not self.dry_run and not profile.empty:
            self._persist_profile(sdk, space_path, profile)
        return profile

    def collect_signals(self, sdk: Any, space_path: str) -> SpaceSignals:
        """Aggregate raw statistics for one Space (synchronous RPCs)."""
        now = datetime.now(timezone.utc)
        window_start = now.timestamp() - self.window_days * 86400

        signals = SpaceSignals(
            window_start=datetime.fromtimestamp(
                window_start, tz=timezone.utc,
            ).isoformat(),
            window_end=now.isoformat(),
        )

        items = _walk.list_space_items(
            sdk,
            space_path,
            kind_filter="",
            page_size=self.item_page_size,
            include_deprecated=True,
        )

        ages_days: List[float] = []
        revisions_in_window = 0
        published_count = 0

        for item in items:
            name, kind = _item_name_and_kind(item)
            # Self-measurement exclusion: never count profile items or the
            # Dream State cursor item toward a space's own signals.
            if kind == self.profile_kind or name in (
                self.profile_item_name, "_dream_state",
            ):
                continue

            signals.items_count += 1
            item_deprecated = bool(getattr(item, "deprecated", False))
            if item_deprecated:
                signals.deprecated_items += 1

            try:
                revisions = list(item.get_revisions() or [])
            except Exception as exc:
                logger.debug(
                    "SpaceProfiler: get_revisions failed for %s: %s",
                    name, exc,
                )
                continue

            signals.revisions_count += len(revisions)
            for rev in revisions:
                # Item-level deprecation does NOT cascade to revision
                # flags on the server, but a deprecated item's revisions
                # are dead knowledge (search excludes them) — treat them
                # as deprecated for the live partition.
                deprecated = item_deprecated or bool(
                    getattr(rev, "deprecated", False)
                )
                if deprecated:
                    signals.deprecated_revisions += 1

                created = _parse_created_at(getattr(rev, "created_at", None))
                if created is not None and created.timestamp() >= window_start:
                    revisions_in_window += 1

                if deprecated:
                    # Dead history counts toward churn only — it must not
                    # inflate stability/evidence for the live knowledge.
                    continue

                signals.live_revisions_count += 1
                if getattr(rev, "published", False):
                    published_count += 1

                meta = dict(getattr(rev, "metadata", {}) or {})
                level = parse_evidence(meta, _snapshot_tags(rev))
                if level:
                    signals.evidence_histogram[level] = (
                        signals.evidence_histogram.get(level, 0) + 1
                    )

                if created is not None:
                    ages_days.append(
                        max(0.0, (now - created).total_seconds() / 86400.0)
                    )

            # The server returns revisions newest-first — never assume
            # list position encodes recency.
            latest_rev = _pick_latest_revision(revisions)
            if latest_rev is not None and self.max_supersedes_depth > 0:
                depth = self._supersedes_depth(sdk, latest_rev)
                if depth > 0:
                    signals.supersedes_edge_count += 1
                    signals.supersedes_max_depth = max(
                        signals.supersedes_max_depth, depth,
                    )

        if signals.items_count:
            signals.revisions_per_item_mean = (
                signals.revisions_count / signals.items_count
            )
        if signals.revisions_count:
            signals.deprecation_ratio = (
                signals.deprecated_revisions / signals.revisions_count
            )
        if signals.live_revisions_count:
            signals.published_share = (
                published_count / signals.live_revisions_count
            )
        signals.revision_rate_per_day = revisions_in_window / self.window_days
        if ages_days:
            signals.median_revision_age_days = round(
                statistics.median(ages_days), 2,
            )
        return signals

    def _supersedes_depth(self, sdk: Any, rev: Any) -> int:
        """Length of the SUPERSEDES chain from *rev*, bounded and
        best-effort (0 on any failure — fakes without edges are fine)."""
        depth = 0
        current = rev
        seen: set = set()
        while depth < self.max_supersedes_depth:
            kref = getattr(getattr(current, "kref", None), "uri", "")
            if kref in seen:
                break  # cycle guard
            seen.add(kref)
            try:
                # SDK signature: get_edges(edge_type_filter: Optional[str])
                edges = current.get_edges(edge_type_filter="SUPERSEDES")
            except TypeError:
                try:
                    edges = current.get_edges()
                except Exception:
                    break
            except Exception:
                break

            next_rev = None
            for edge in edges or []:
                if getattr(edge, "edge_type", "") != "SUPERSEDES":
                    continue
                target = getattr(edge, "target_kref", None)
                target_uri = getattr(target, "uri", "") if target else ""
                if not target_uri or target_uri == kref:
                    continue
                try:
                    next_rev = sdk.get_revision(target_uri)
                except Exception:
                    next_rev = None
                break
            if next_rev is None:
                break
            depth += 1
            current = next_rev
        return depth

    # ------------------------------------------------------------------
    # Override / previous label / persistence
    # ------------------------------------------------------------------

    def _read_override(
        self,
        sdk: Any,
        space_path: str,
        space_handle: Optional[Any] = None,
    ) -> Optional[str]:
        """Read the ``space_class`` Space attribute (manual pin).

        MUST go through the Space handle when available: the SDK's
        module-level ``get_attribute`` wraps its argument in ``Kref``,
        whose validator rejects bare space paths (no ``kref://`` scheme)
        — ``Space.get_attribute`` deliberately bypasses that validation.
        The module-level call remains only as a fallback for handle-less
        paths (project root) and legacy stubs.
        """
        value = None
        if space_handle is not None:
            try:
                value = space_handle.get_attribute("space_class")
            except Exception as exc:
                logger.debug(
                    "SpaceProfiler: space_class read via handle failed "
                    "for %s: %s", space_path, exc,
                )
        if value is None:
            try:
                value = sdk.get_attribute(space_path, "space_class")
            except Exception:
                return None
        return value if value in SPACE_CLASSES else None

    def _profile_kref(self, space_path: str) -> str:
        return (
            f"kref://{space_path.strip('/')}"
            f"/{self.profile_item_name}.{self.profile_kind}"
        )

    def _read_previous_label(
        self, sdk: Any, space_path: str,
    ) -> Optional[str]:
        """Label from the existing profile item's latest revision."""
        try:
            item = sdk.get_item(self._profile_kref(space_path))
            if item is None:
                return None
            rev = item.get_revision_by_tag("latest")
            if rev is None:
                return None
            meta = dict(getattr(rev, "metadata", {}) or {})
            label = meta.get("label", "")
            return label if label in SPACE_CLASSES else None
        except Exception:
            return None

    def _persist_profile(
        self, sdk: Any, space_path: str, profile: SpaceProfile,
    ) -> None:
        """Create one revision on the per-Space profile item.

        Revision metadata is ``Dict[str, str]`` — structured values are
        JSON-serialized strings.  Best-effort: persistence failures are
        logged and swallowed (the run result still carries the profile).
        """
        try:
            item = self._get_or_create_profile_item(sdk, space_path)
            if item is None:
                return
            prev_rev = None
            try:
                prev_rev = item.get_revision_by_tag("latest")
            except Exception:
                prev_rev = None

            revision = item.create_revision(metadata={
                "type": "space_profile",
                "title": f"Space profile: {space_path}",
                "summary": (
                    f"label={profile.label} pinned={profile.pinned} "
                    f"churn={profile.scores.get('churn')} "
                    f"evidence={profile.scores.get('evidence')} "
                    f"stability={profile.scores.get('stability')}"
                ),
                "label": profile.label,
                "observed_label": profile.observed_label or "",
                "pinned": "true" if profile.pinned else "false",
                "previous_label": profile.previous_label or "",
                "scores": json.dumps(profile.scores),
                "signals": json.dumps(asdict(profile.signals)),
                "window_start": profile.signals.window_start,
                "window_end": profile.signals.window_end,
            })

            # Dogfood revision-centric memory: profile drift is itself a
            # SUPERSEDES chain.
            if prev_rev is not None:
                try:
                    revision.create_edge(prev_rev, "SUPERSEDES")
                except Exception as exc:
                    logger.debug(
                        "SpaceProfiler: SUPERSEDES edge failed for %s: %s",
                        space_path, exc,
                    )
        except Exception as exc:
            logger.warning(
                "SpaceProfiler: failed to persist profile for %s: %s",
                space_path, exc,
            )

    def _get_or_create_profile_item(
        self, sdk: Any, space_path: str,
    ) -> Optional[Any]:
        try:
            item = sdk.get_item(self._profile_kref(space_path))
            if item is not None:
                return item
        except Exception:
            pass

        # Create in the owning space.  Root-space items (directly under
        # the project) are skipped when no space handle is resolvable.
        rel = space_path.strip("/")
        if rel.startswith(self.project):
            rel = rel[len(self.project):].strip("/")
        try:
            project = sdk.get_project(self.project)
            if not rel:
                logger.debug(
                    "SpaceProfiler: no space handle for project root %s — "
                    "skipping profile persistence", space_path,
                )
                return None
            space = project.get_space(rel)
            return space.create_item(self.profile_item_name, self.profile_kind)
        except Exception as exc:
            logger.warning(
                "SpaceProfiler: could not create profile item in %s: %s",
                space_path, exc,
            )
            return None


def get_space_profile(
    project: str,
    space_path: str,
    *,
    profile_kind: str = "space-profile",
    profile_item_name: str = "_space_profile",
) -> Optional[SpaceProfile]:
    """Read the latest persisted profile for a Space.

    The single read-side API that per-space strategy consumers (evidence
    assessor, Dream State policy, recall) use.  Returns ``None`` when no
    profile exists or it cannot be parsed.
    """
    import kumiho

    # space_path may be project-qualified ("/CognitiveMemory/facts") or
    # relative ("facts") — the project argument qualifies the latter.
    rel = space_path.strip("/")
    proj = project.strip("/")
    if rel != proj and not rel.startswith(f"{proj}/"):
        rel = f"{proj}/{rel}" if rel else proj
    kref = f"kref://{rel}/{profile_item_name}.{profile_kind}"
    try:
        item = kumiho.get_item(kref)
        if item is None:
            return None
        rev = item.get_revision_by_tag("latest")
        if rev is None:
            return None
        meta = dict(getattr(rev, "metadata", {}) or {})
        label = meta.get("label", "")
        if label not in SPACE_CLASSES:
            return None
        try:
            scores = json.loads(meta.get("scores", "") or "{}")
        except (ValueError, TypeError):
            scores = {}
        signals = SpaceSignals()
        try:
            raw_signals = json.loads(meta.get("signals", "") or "{}")
            for key, value in raw_signals.items():
                if hasattr(signals, key):
                    setattr(signals, key, value)
        except (ValueError, TypeError):
            pass
        observed = meta.get("observed_label", "") or None
        return SpaceProfile(
            space_path=space_path,
            signals=signals,
            scores=scores if isinstance(scores, dict) else {},
            label=label,
            pinned=meta.get("pinned", "") == "true",
            previous_label=meta.get("previous_label", "") or None,
            observed_label=observed if observed in SPACE_CLASSES else None,
        )
    except Exception as exc:
        logger.debug(
            "get_space_profile failed for %s: %s", space_path, exc,
        )
        return None
