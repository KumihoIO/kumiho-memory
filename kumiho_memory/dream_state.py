"""Dream State — scheduled memory consolidation processor.

The Dream State runs periodically (e.g. nightly at 3 AM) to:

1. Query revisions created or updated since the last run.
2. Fetch full revision data for changed memories.
3. Inspect bundles for new conversation groupings.
4. Use an LLM to assess each memory: deprecate low-value ones,
   enrich metadata / tags, and suggest relationships.
5. Apply the assessed changes to the Kumiho graph.
6. Persist the timestamp and generate a Markdown report.

Usage::

    from kumiho_memory import DreamState

    ds = DreamState(project="CognitiveMemory")
    report = await ds.run()
    print(report)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from kumiho_memory.summarization import LLMAdapter, MemorySummarizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryAssessment:
    """LLM-produced assessment for a single memory revision."""

    revision_kref: str
    relevance_score: float
    should_deprecate: bool
    deprecation_reason: str = ""
    suggested_tags: List[str] = field(default_factory=list)
    metadata_updates: Dict[str, str] = field(default_factory=dict)
    related_memories: List[Tuple[str, str]] = field(default_factory=list)
    """List of ``(target_revision_kref, edge_type)`` tuples."""


@dataclass
class DreamStateStats:
    """Counters accumulated during a single Dream State run."""

    events_processed: int = 0
    revisions_assessed: int = 0
    deprecated: int = 0
    metadata_updated: int = 0
    tags_added: int = 0
    edges_created: int = 0
    last_cursor: Optional[str] = None  # Kept for backward-compat in report dict
    duration_ms: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ASSESSMENT_SYSTEM_PROMPT = """\
You are a memory consolidation agent performing "Dream State" processing.
You will receive an array of memories (each with an index, title, summary,
type, tags, and metadata).  For **each** memory return a JSON object with
the following fields:

1. index (int): The memory's index in the input array.
2. relevance_score (float 0.0-1.0): How useful is this memory for future
   interactions?
3. should_deprecate (bool): True if the memory should be deprecated.
4. deprecation_reason (str): Why (empty string if keeping).
5. suggested_tags (List[str]): Additional tags for better retrieval.
6. metadata_updates (Dict[str, str]): Metadata key/value corrections or
   enrichments.  Return ``{}`` if none.
7. related_indices (List[int]): Indices of related memories in THIS batch.
8. relationship_type (str): Edge type for related memories — one of
   DERIVED_FROM, REFERENCED, DEPENDS_ON, SUPERSEDES.  Empty string if none.

Return ONLY a JSON array of objects (one per memory).

Guidelines:
- Be conservative: when in doubt, KEEP the memory.
- Deprecate ONLY if the memory is: a near-duplicate of another memory in
  this batch, clearly superseded by newer information, trivially obvious,
  or contains no actionable information.
- Tags should aid retrieval: topic keywords, action types, entity names,
  project identifiers.
- Suggest relationships for memories that reference the same topic,
  project, or decision chain.
"""


def _parse_assessments(raw: str) -> List[Dict[str, Any]]:
    """Best-effort parse of LLM JSON output."""
    # Try direct parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "assessments" in parsed:
            return parsed["assessments"]
        return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON array from markdown fences
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort — look for bare array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# DreamState
# ---------------------------------------------------------------------------


class DreamState:
    """Scheduled memory consolidation processor.

    Parameters
    ----------
    project:
        Kumiho project name (default ``CognitiveMemory``).
    summarizer:
        Existing :class:`MemorySummarizer` to reuse for LLM calls.
    llm_adapter:
        Raw :class:`LLMAdapter` — a ``MemorySummarizer`` is built around it.
    artifact_root:
        Local directory for writing report artifacts.
    cursor_item_name:
        Item name used to persist the run timestamp and
        Dream State reports (default ``_dream_state``).
    batch_size:
        Number of memories to assess per LLM call.
    dry_run:
        If *True*, assess but do **not** mutate anything in Kumiho.
    max_deprecation_ratio:
        Maximum fraction of memories that may be deprecated per run.
        Must be between 0.1 and 0.9 (default 0.5).
    allow_published_deprecation:
        If *True*, published items may be deprecated. Use with caution.
        When relaxed, a warning is logged and recorded in the audit report.
    kind_filter:
        Item kind to process (default ``conversation``).  Set to empty
        string to process all item kinds.
    """

    def __init__(
        self,
        *,
        project: str = "CognitiveMemory",
        summarizer: Optional[MemorySummarizer] = None,
        llm_adapter: Optional[LLMAdapter] = None,
        artifact_root: Optional[str] = None,
        cursor_item_name: str = "_dream_state",
        batch_size: int = 20,
        dry_run: bool = False,
        max_deprecation_ratio: float = 0.5,
        allow_published_deprecation: bool = False,
        kind_filter: str = "conversation",
        # Legacy parameters — accepted but ignored for backward compatibility
        routing_key_filter: str = "revision.*",
        event_timeout: float = 10.0,
    ) -> None:
        self.project = project
        self.cursor_item_name = cursor_item_name
        self.batch_size = batch_size
        self.kind_filter = kind_filter
        self.dry_run = dry_run

        if not (0.1 <= max_deprecation_ratio <= 0.9):
            raise ValueError(
                f"max_deprecation_ratio must be between 0.1 and 0.9, "
                f"got {max_deprecation_ratio}"
            )
        self.max_deprecation_ratio = max_deprecation_ratio
        self.allow_published_deprecation = allow_published_deprecation

        import os

        self.artifact_root = artifact_root or os.getenv(
            "KUMIHO_MEMORY_ARTIFACT_ROOT",
            os.path.join(os.path.expanduser("~"), ".kumiho", "artifacts"),
        )

        if summarizer is not None:
            self.summarizer = summarizer
        elif llm_adapter is not None:
            self.summarizer = MemorySummarizer(adapter=llm_adapter)
        else:
            self.summarizer = MemorySummarizer()

        # Will be resolved lazily on first run.
        self._cursor_item_kref: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Execute a full Dream State cycle.

        Returns a report dict with counters and the final cursor.
        """
        import kumiho

        start = time.monotonic()
        stats = DreamStateStats()
        run_started_at = datetime.now(timezone.utc).isoformat()

        try:
            # 1. Ensure cursor item exists
            cursor_kref = self._ensure_cursor_item(kumiho)

            # 2. Load last_run_at timestamp
            last_run_at = self._load_last_run_at(kumiho, cursor_kref)

            # 3. Collect revisions created/updated since last run
            revisions = await asyncio.to_thread(
                self._collect_revisions, kumiho, last_run_at
            )
            stats.events_processed = len(revisions)
            if not revisions:
                logger.info(
                    "Dream State: no new revisions since %s",
                    last_run_at or "beginning",
                )
                stats.duration_ms = int((time.monotonic() - start) * 1000)
                # Still save timestamp so next run skips this window
                self._save_last_run_at(kumiho, cursor_kref, run_started_at)
                return self._build_result(stats, report_kref=None)

            # 4. Inspect bundles (from revision item krefs)
            bundle_context = self._inspect_bundles_from_revisions(
                kumiho, revisions
            )

            # 5. Assess in batches
            all_assessments: List[MemoryAssessment] = []
            for i in range(0, len(revisions), self.batch_size):
                batch = revisions[i : i + self.batch_size]
                assessments = await self._assess_batch(batch, bundle_context)
                all_assessments.extend(assessments)

            stats.revisions_assessed = len(all_assessments)

            # 6. Apply actions
            self._apply_actions(kumiho, all_assessments, stats)

            # 7. Save last_run_at
            self._save_last_run_at(kumiho, cursor_kref, run_started_at)

            # 8. Generate report
            stats.duration_ms = int((time.monotonic() - start) * 1000)
            report_kref = self._generate_report(
                kumiho, cursor_kref, stats, all_assessments
            )

            return self._build_result(stats, report_kref=report_kref)

        except Exception as exc:
            stats.errors.append(str(exc))
            stats.duration_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Dream State run failed")
            return {
                "success": False,
                "error": str(exc),
                **self._stats_dict(stats),
            }

    # ------------------------------------------------------------------
    # Timestamp management (replaces event-stream cursor)
    # ------------------------------------------------------------------

    def _ensure_cursor_item(self, sdk: Any) -> str:
        """Return the kref of the ``_dream_state`` item, creating it if
        necessary."""
        if self._cursor_item_kref is not None:
            return self._cursor_item_kref

        kref_uri = f"kref://{self.project}/{self.cursor_item_name}.conversation"
        try:
            item = sdk.get_item(kref_uri)
            if item is not None:
                self._cursor_item_kref = item.kref.uri
                return self._cursor_item_kref
        except Exception:
            pass

        # Create the item — first ensure parent space exists.
        try:
            project = sdk.get_project(self.project)
            if project is None:
                raise RuntimeError(
                    f"Project '{self.project}' does not exist"
                )

            try:
                space = project.get_space(self.cursor_item_name)
            except Exception:
                space = None
            if space is None:
                space = project.create_space(self.cursor_item_name)

            item = space.create_item(self.cursor_item_name, "conversation")
            self._cursor_item_kref = item.kref.uri
        except Exception:
            # Fallback: item might already exist (race)
            try:
                item = sdk.get_item(kref_uri)
                self._cursor_item_kref = item.kref.uri
            except Exception as inner:
                raise RuntimeError(
                    f"Failed to ensure cursor item: {inner}"
                ) from inner

        return self._cursor_item_kref  # type: ignore[return-value]

    def _load_last_run_at(self, sdk: Any, cursor_kref: str) -> Optional[str]:
        """Read the last-saved run timestamp (ISO format)."""
        try:
            return sdk.get_attribute(cursor_kref, "last_run_at")
        except Exception:
            return None

    def _save_last_run_at(
        self, sdk: Any, cursor_kref: str, run_at: str
    ) -> None:
        """Persist the run timestamp."""
        sdk.set_attribute(cursor_kref, "last_run_at", run_at)

    # ------------------------------------------------------------------
    # Revision collection (replaces event stream)
    # ------------------------------------------------------------------

    def _collect_revisions(
        self, sdk: Any, last_run_at: Optional[str]
    ) -> list:
        """Enumerate all spaces in the project, list items, and collect
        latest revisions that were created after *last_run_at*.

        This replaces the old event-stream approach which suffered from
        gRPC DEADLINE_EXCEEDED errors and cursor issues.  Direct revision
        queries are reliable and catch both new items and stacked revisions
        on existing items.
        """
        try:
            project = sdk.get_project(self.project)
            if project is None:
                logger.warning("Project '%s' not found", self.project)
                return []
        except Exception as exc:
            logger.warning("Failed to get project '%s': %s", self.project, exc)
            return []

        # Parse the cutoff timestamp
        cutoff: Optional[datetime] = None
        if last_run_at:
            try:
                cutoff = datetime.fromisoformat(last_run_at)
                # Ensure timezone-aware
                if cutoff.tzinfo is None:
                    cutoff = cutoff.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid last_run_at timestamp '%s', processing all",
                    last_run_at,
                )

        # Enumerate all spaces recursively
        try:
            spaces = project.get_spaces(recursive=True)
        except Exception as exc:
            logger.warning("Failed to enumerate spaces: %s", exc)
            return []

        collected: list = []
        cursor_item_kref = self._cursor_item_kref

        for space in spaces:
            try:
                kind_arg = self.kind_filter if self.kind_filter else ""
                items = sdk.get_client().get_items(
                    parent_path=space.path,
                    kind_filter=kind_arg,
                    include_deprecated=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to list items in space '%s': %s",
                    space.path, exc,
                )
                continue

            for item in items:
                # Skip the _dream_state cursor item itself
                item_kref = item.kref.uri if hasattr(item, "kref") else ""
                if cursor_item_kref and item_kref == cursor_item_kref:
                    continue

                try:
                    # Get the latest revision
                    rev = item.get_revision_by_tag("latest")
                    if rev is None:
                        continue
                except Exception:
                    # No 'latest' tag — try getting all revisions
                    try:
                        revs = item.get_revisions()
                        if not revs:
                            continue
                        rev = revs[-1]  # Most recent
                    except Exception:
                        continue

                # Skip deprecated revisions
                if getattr(rev, "deprecated", False):
                    continue

                # Filter by created_at timestamp
                if cutoff is not None and rev.created_at:
                    try:
                        rev_time = datetime.fromisoformat(rev.created_at)
                        if rev_time.tzinfo is None:
                            rev_time = rev_time.replace(tzinfo=timezone.utc)
                        if rev_time <= cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass  # Can't parse — include it to be safe

                collected.append(rev)

        logger.info(
            "Dream State: collected %d revisions since %s",
            len(collected),
            last_run_at or "beginning",
        )
        return collected

    # ------------------------------------------------------------------
    # Bundle inspection
    # ------------------------------------------------------------------

    def _inspect_bundles_from_revisions(
        self, sdk: Any, revisions: list
    ) -> Dict[str, list]:
        """For any bundle items among the collected revisions, fetch members."""
        bundles: Dict[str, list] = {}
        for rev in revisions:
            item_kref = ""
            try:
                item_kref = rev.item_kref.uri if hasattr(rev, "item_kref") else ""
            except Exception:
                continue
            if ".bundle" not in item_kref:
                continue
            if item_kref in bundles:
                continue
            try:
                bundle = sdk.get_item(item_kref)
                if bundle is not None and hasattr(bundle, "get_members"):
                    bundles[item_kref] = bundle.get_members()
            except Exception as exc:
                logger.warning("Failed to inspect bundle %s: %s", item_kref, exc)

        return bundles

    # ------------------------------------------------------------------
    # LLM assessment
    # ------------------------------------------------------------------

    async def _assess_batch(
        self,
        revisions: list,
        bundle_context: Dict[str, list],
    ) -> List[MemoryAssessment]:
        """Send a batch of revisions to the LLM for assessment."""
        if not revisions:
            return []

        # Build the user prompt — serialise each revision to JSON-like text.
        memories: List[Dict[str, Any]] = []
        kref_by_index: Dict[int, str] = {}

        for idx, rev in enumerate(revisions):
            meta = dict(getattr(rev, "metadata", {}) or {})
            entry: Dict[str, Any] = {
                "index": idx,
                "kref": rev.kref.uri if hasattr(rev, "kref") else str(rev),
                "title": meta.get("title", ""),
                "summary": meta.get("summary", ""),
                "type": meta.get("type", meta.get("memory_type", "")),
                "tags": meta.get("tags", ""),
                "topics": meta.get("topics", ""),
            }
            kref_by_index[idx] = entry["kref"]
            memories.append(entry)

        # Include bundle context if available.
        bundle_info = ""
        if bundle_context:
            parts = []
            for bkref, members in bundle_context.items():
                member_strs = []
                for m in members:
                    mkref = m.item_kref.uri if hasattr(m, "item_kref") else str(m)
                    member_strs.append(mkref)
                parts.append(f"Bundle {bkref}: members={member_strs}")
            bundle_info = "\n\nBundle groupings:\n" + "\n".join(parts)

        user_prompt = (
            "Assess the following memories:\n\n"
            + json.dumps(memories, indent=2, default=str)
            + bundle_info
        )

        try:
            raw = await self.summarizer.adapter.chat(
                messages=[{"role": "user", "content": user_prompt}],
                model=self.summarizer.model,
                system=_ASSESSMENT_SYSTEM_PROMPT,
                max_tokens=2048,
                json_mode=True,
            )
        except Exception as exc:
            logger.warning("LLM assessment failed: %s", exc)
            return []

        parsed = _parse_assessments(raw)

        # Convert to MemoryAssessment objects.
        assessments: List[MemoryAssessment] = []
        for item in parsed:
            idx = item.get("index", -1)
            rev_kref = kref_by_index.get(idx, "")
            if not rev_kref:
                continue

            related: List[Tuple[str, str]] = []
            rel_type = item.get("relationship_type", "")
            for rel_idx in item.get("related_indices", []):
                target = kref_by_index.get(rel_idx)
                if target and target != rev_kref:
                    related.append((target, rel_type or "REFERENCED"))

            assessments.append(
                MemoryAssessment(
                    revision_kref=rev_kref,
                    relevance_score=float(item.get("relevance_score", 0.5)),
                    should_deprecate=bool(item.get("should_deprecate", False)),
                    deprecation_reason=item.get("deprecation_reason", ""),
                    suggested_tags=list(item.get("suggested_tags", [])),
                    metadata_updates=dict(item.get("metadata_updates", {})),
                    related_memories=related,
                )
            )

        return assessments

    # ------------------------------------------------------------------
    # Apply actions
    # ------------------------------------------------------------------

    def _apply_actions(
        self,
        sdk: Any,
        assessments: List[MemoryAssessment],
        stats: DreamStateStats,
    ) -> None:
        """Apply the LLM-recommended changes to the Kumiho graph."""
        if self.dry_run:
            logger.info("Dry run — skipping %d actions", len(assessments))
            return

        if not assessments:
            return

        client = sdk.get_client()

        # Safety: cap deprecation per run (spec §9.4.4).
        deprecation_limit = max(1, int(len(assessments) * self.max_deprecation_ratio))
        deprecations_done = 0

        for assessment in assessments:
            kref_str = assessment.revision_kref
            try:
                kref = sdk.Kref(kref_str)
            except Exception:
                stats.errors.append(f"Invalid kref: {kref_str}")
                continue

            # --- Deprecate ---
            if assessment.should_deprecate:
                try:
                    is_published = client.has_tag(kref, "published")
                    if is_published and not self.allow_published_deprecation:
                        logger.info(
                            "Skipping deprecation of published revision %s",
                            kref_str,
                        )
                    elif deprecations_done >= deprecation_limit:
                        logger.info(
                            "Deprecation limit reached (%d/%d), skipping %s",
                            deprecations_done,
                            deprecation_limit,
                            kref_str,
                        )
                    else:
                        if is_published:
                            logger.warning(
                                "Published protection RELAXED — deprecating published revision %s",
                                kref_str,
                            )
                        client.set_deprecated(kref, True)
                        stats.deprecated += 1
                        deprecations_done += 1
                except Exception as exc:
                    stats.errors.append(f"deprecate {kref_str}: {exc}")

            # --- Tags ---
            for tag in assessment.suggested_tags:
                try:
                    client.tag_revision(kref, tag)
                    stats.tags_added += 1
                except Exception as exc:
                    stats.errors.append(f"tag {kref_str} '{tag}': {exc}")

            # --- Metadata updates ---
            if assessment.metadata_updates:
                try:
                    client.update_revision_metadata(
                        kref, assessment.metadata_updates
                    )
                    stats.metadata_updated += 1
                except Exception as exc:
                    stats.errors.append(f"metadata {kref_str}: {exc}")

            # --- Relationships / edges ---
            for target_kref_str, edge_type in assessment.related_memories:
                try:
                    target_kref = sdk.Kref(target_kref_str)
                    # create_edge needs Revision objects; fetch them.
                    source_rev = sdk.get_revision(kref_str)
                    target_rev = sdk.get_revision(target_kref_str)
                    if source_rev and target_rev:
                        client.create_edge(
                            source_rev, target_rev, edge_type
                        )
                        stats.edges_created += 1
                except Exception as exc:
                    stats.errors.append(
                        f"edge {kref_str} → {target_kref_str}: {exc}"
                    )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        sdk: Any,
        cursor_kref: str,
        stats: DreamStateStats,
        assessments: List[MemoryAssessment],
    ) -> Optional[str]:
        """Create a report revision + artifact on the cursor item."""
        now_iso = datetime.now(timezone.utc).isoformat()
        markdown = self._build_report_markdown(
            stats, assessments, now_iso,
            allow_published_deprecation=self.allow_published_deprecation,
        )

        # Write artifact to local storage.
        safe_ts = now_iso.replace(":", "").replace("-", "").split(".")[0]
        artifact_dir = (
            Path(self.artifact_root)
            / self.project
            / self.cursor_item_name
            / "reports"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"dream_state_{safe_ts}.md"
        artifact_path.write_text(markdown, encoding="utf-8")

        # Create revision with metadata.
        try:
            item = sdk.get_item(cursor_kref)
            if item is None:
                return None

            revision = item.create_revision(
                metadata={
                    "type": "dream_state_report",
                    "events_processed": str(stats.events_processed),
                    "revisions_assessed": str(stats.revisions_assessed),
                    "deprecated": str(stats.deprecated),
                    "metadata_updated": str(stats.metadata_updated),
                    "tags_added": str(stats.tags_added),
                    "edges_created": str(stats.edges_created),
                    "cursor": stats.last_cursor or "",
                    "run_at": now_iso,
                    "duration_ms": str(stats.duration_ms),
                },
            )
            revision.create_artifact("report", str(artifact_path))
            return revision.kref.uri
        except Exception as exc:
            logger.warning("Failed to create report revision: %s", exc)
            stats.errors.append(f"report: {exc}")
            return None

    @staticmethod
    def _build_report_markdown(
        stats: DreamStateStats,
        assessments: List[MemoryAssessment],
        timestamp: str,
        *,
        allow_published_deprecation: bool = False,
    ) -> str:
        """Build a Markdown report of the Dream State run."""
        parts: List[str] = [
            f"# Dream State Report — {timestamp}",
            "",
            f"**Events processed:** {stats.events_processed}  ",
            f"**Memories assessed:** {stats.revisions_assessed}  ",
            f"**Duration:** {stats.duration_ms}ms",
            "",
        ]

        if allow_published_deprecation:
            parts.extend([
                "**WARNING:** Published protection was relaxed for this run "
                "(`allow_published_deprecation=true`).  ",
                "",
            ])

        parts.extend([
            "---",
            "",
            "## Actions Taken",
            "",
        ])

        # Deprecated
        deprecated = [a for a in assessments if a.should_deprecate]
        parts.append(f"### Deprecated ({stats.deprecated})")
        parts.append("")
        if deprecated:
            for a in deprecated:
                parts.append(
                    f"- `{a.revision_kref}` — {a.deprecation_reason or 'no reason given'}"
                )
        else:
            parts.append("_None_")
        parts.append("")

        # Metadata Updated
        updated = [a for a in assessments if a.metadata_updates]
        parts.append(f"### Metadata Updated ({stats.metadata_updated})")
        parts.append("")
        if updated:
            for a in updated:
                changes = ", ".join(
                    f"{k}={v}" for k, v in a.metadata_updates.items()
                )
                parts.append(f"- `{a.revision_kref}` — {changes}")
        else:
            parts.append("_None_")
        parts.append("")

        # Tags Added
        tagged = [a for a in assessments if a.suggested_tags]
        parts.append(f"### Tags Added ({stats.tags_added})")
        parts.append("")
        if tagged:
            for a in tagged:
                parts.append(
                    f"- `{a.revision_kref}` — {', '.join(a.suggested_tags)}"
                )
        else:
            parts.append("_None_")
        parts.append("")

        # Relationships Created
        related = [a for a in assessments if a.related_memories]
        parts.append(f"### Relationships Created ({stats.edges_created})")
        parts.append("")
        if related:
            for a in related:
                for target, etype in a.related_memories:
                    parts.append(
                        f"- `{a.revision_kref}` → `{target}` ({etype})"
                    )
        else:
            parts.append("_None_")
        parts.append("")

        # Errors
        if stats.errors:
            parts.append(f"### Errors ({len(stats.errors)})")
            parts.append("")
            for err in stats.errors:
                parts.append(f"- {err}")
            parts.append("")

        # Cursor
        parts.extend([
            "---",
            "",
            "## Cursor",
            "",
            f"`{stats.last_cursor or 'N/A'}`",
            "",
        ])

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        stats: DreamStateStats,
        *,
        report_kref: Optional[str],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"success": True}
        result.update(DreamState._stats_dict(stats))
        if report_kref:
            result["report_kref"] = report_kref
        return result

    @staticmethod
    def _stats_dict(stats: DreamStateStats) -> Dict[str, Any]:
        return {
            "events_processed": stats.events_processed,
            "revisions_assessed": stats.revisions_assessed,
            "deprecated": stats.deprecated,
            "metadata_updated": stats.metadata_updated,
            "tags_added": stats.tags_added,
            "edges_created": stats.edges_created,
            "cursor": stats.last_cursor,
            "duration_ms": stats.duration_ms,
            "errors": stats.errors,
        }
