"""Generic SKILL.md ingest pipeline — parse and ingest into Kumiho graph.

Parses any SKILL.md file (YAML frontmatter + ``##`` section headers) and
ingests non-inline sections into ``CognitiveMemory/Skills`` (or a custom
space) as versioned skill items.  Supports standalone reference docs too.

**Parsing conventions:**

- YAML frontmatter between ``---`` fences provides skill-level metadata
  (``name``, ``description``, ``tags``).
- Each ``##`` header starts a new section.
- Sections containing ``<!-- inline -->`` anywhere in their body are
  **kept inline** and skipped during ingestion.
- Everything else becomes a skill item in the graph.

**Ingestion guarantees:**

- Idempotent — re-ingesting creates a **new revision** on the existing
  item rather than duplicating it.
- Each revision stores the section content in metadata (``content`` key)
  as a fallback and attaches the source file as an artifact.
- ``revision.tag("published")`` marks the new revision as canonical
  (server auto-moves the tag from any previous revision on the same item).

Usage::

    # Python API
    from kumiho_memory.skill_ingest import parse_skill, ingest_skill, ingest_file

    parsed = parse_skill("path/to/SKILL.md")
    results = ingest_skill("path/to/SKILL.md")
    result  = ingest_file("path/to/creative-memory.md", item_name="creative-memory")

    # CLI (via kumiho-memory entry point)
    python -m kumiho_memory ingest-skill path/to/SKILL.md --list
    python -m kumiho_memory ingest-skill path/to/SKILL.md --dry-run
    python -m kumiho_memory ingest-skill path/to/references/ --batch
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SkillSection:
    """A single ``##``-delimited section from a SKILL.md file."""

    name: str
    """Slugified header (e.g. ``"store-link-protocol"``)."""

    title: str
    """Original header text."""

    content: str
    """Full markdown content of the section (including the header line)."""

    inline: bool
    """``True`` if the section contains ``<!-- inline -->``."""

    level: int
    """Heading level (2 for ``##``, 3 for ``###``, etc.)."""

    line_start: int
    """1-based line number where this section starts in the source file."""


@dataclass
class ParsedSkill:
    """Result of parsing a SKILL.md file."""

    name: str
    """From frontmatter ``name`` field."""

    description: str
    """From frontmatter ``description`` field."""

    tags: list[str]
    """From frontmatter ``tags`` field (defaults to ``[]``)."""

    preamble: str
    """Content before the first ``##`` header (after frontmatter)."""

    sections: list[SkillSection] = field(default_factory=list)
    """All ``##``-level sections in order."""

    source_path: Optional[Path] = None
    """Absolute path to the source SKILL.md file."""


# ---------------------------------------------------------------------------
# Slugify helper
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Convert a heading into a URL/item-name-safe slug.

    >>> _slugify("Store & Link Protocol (mandatory)")
    'store-link-protocol-mandatory'
    """
    return _SLUG_RE.sub("-", text.lower()).strip("-")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_INLINE_MARKER = "<!-- inline -->"


def parse_skill(path: str | Path) -> ParsedSkill:
    """Parse a SKILL.md file into structured sections.

    Args:
        path: Path to a SKILL.md (or any markdown with YAML frontmatter).

    Returns:
        A :class:`ParsedSkill` with metadata and sections.
    """
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # --- Parse YAML frontmatter ---
    fm_name = ""
    fm_description = ""
    fm_tags: list[str] = []

    if lines and lines[0].strip() == "---":
        end_idx = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = i
                break
        if end_idx is not None:
            fm_lines = lines[1:end_idx]
            fm_name, fm_description, fm_tags = _parse_frontmatter(fm_lines)
            lines = lines[end_idx + 1 :]  # remainder after frontmatter
            # Adjust line offset for section line_start
            body_line_offset = end_idx + 1  # 0-based offset into original file
        else:
            body_line_offset = 0
    else:
        body_line_offset = 0

    # --- Split into sections ---
    sections: list[SkillSection] = []
    preamble_lines: list[str] = []
    current_header: Optional[tuple[int, str, int, int]] = None  # (level, title, start_idx, file_line)
    current_body: list[str] = []

    # Track whether the previous line contained <!-- inline -->.
    # When we hit a ## heading, this tells us the marker was placed
    # immediately before the heading (the conventional placement).
    prev_line_inline = False

    def _flush_section() -> None:
        nonlocal current_header, current_body
        if current_header is None:
            return
        level, title, _start_idx, file_line, was_pre_inline = current_header
        body_text = "".join(current_body)
        # Reconstruct full section content (header + body)
        header_line = "#" * level + " " + title + "\n"
        full_content = header_line + body_text
        # A section is inline if:
        # - <!-- inline --> appears anywhere in the body, OR
        # - <!-- inline --> appeared on the line immediately before the heading
        is_inline = _INLINE_MARKER in body_text or was_pre_inline
        sections.append(
            SkillSection(
                name=_slugify(title),
                title=title,
                content=full_content.rstrip("\n"),
                inline=is_inline,
                level=level,
                line_start=file_line,
            )
        )
        current_header = None
        current_body = []

    for idx, line in enumerate(lines):
        file_line = body_line_offset + idx + 1  # 1-based
        stripped = line.strip()
        m = _HEADING_RE.match(line.rstrip("\n"))
        if m and len(m.group(1)) == 2:  # Only ## headings start sections
            _flush_section()
            current_header = (len(m.group(1)), m.group(2).strip(), idx, file_line, prev_line_inline)
            current_body = []
        elif current_header is not None:
            current_body.append(line)
        else:
            preamble_lines.append(line)
        # Update prev_line_inline for the NEXT iteration.
        # Skip blank lines so that "<!-- inline -->\n\n## Heading" still works.
        if stripped:
            prev_line_inline = _INLINE_MARKER in stripped

    _flush_section()

    return ParsedSkill(
        name=fm_name,
        description=fm_description,
        tags=fm_tags,
        preamble="".join(preamble_lines).strip(),
        sections=sections,
        source_path=path,
    )


def _parse_frontmatter(fm_lines: list[str]) -> tuple[str, str, list[str]]:
    """Extract name, description, and tags from simple YAML frontmatter lines."""
    name = ""
    description = ""
    tags: list[str] = []

    for line in fm_lines:
        stripped = line.strip()
        if stripped.startswith("name:"):
            name = stripped[len("name:") :].strip().strip("\"'")
        elif stripped.startswith("description:"):
            description = stripped[len("description:") :].strip().strip("\"'")
        elif stripped.startswith("tags:"):
            # Inline list: tags: [a, b, c] or tags: ["a", "b"]
            rest = stripped[len("tags:") :].strip()
            if rest.startswith("["):
                rest = rest.strip("[]")
                tags = [t.strip().strip("\"'") for t in rest.split(",") if t.strip()]
            # YAML list continuation (- item) handled below
        elif stripped.startswith("- ") and not name:
            # Skip list items that aren't tags
            pass
        elif stripped.startswith("- "):
            # Tag list items
            tags.append(stripped[2:].strip().strip("\"'"))

    return name, description, tags


# ---------------------------------------------------------------------------
# Ingestion — SKILL.md sections
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Result of ingesting a single section or file."""

    item_name: str
    item_kref: str
    revision_kref: str
    artifact_kref: Optional[str] = None
    created_new_item: bool = False


def ingest_skill(
    path: str | Path,
    *,
    project: str = "CognitiveMemory",
    space_name: str = "Skills",
    section_filter: Optional[str] = None,
    dry_run: bool = False,
) -> list[IngestResult]:
    """Parse a SKILL.md and ingest all non-inline sections into the graph.

    Args:
        path: Path to the SKILL.md file.
        project: Kumiho project name.
        space_name: Space within the project (created if missing).
        section_filter: If set, only ingest the section with this slug name.
        dry_run: If ``True``, parse and log but don't call the Kumiho API.

    Returns:
        List of :class:`IngestResult` — one per ingested section.
    """
    parsed = parse_skill(path)
    source_file = parsed.source_path or Path(path).resolve()

    # Filter to non-inline sections
    to_ingest = [s for s in parsed.sections if not s.inline]
    if section_filter:
        to_ingest = [s for s in to_ingest if s.name == section_filter]

    if not to_ingest:
        logger.info("No sections to ingest (all inline or filter matched nothing)")
        return []

    if dry_run:
        results: list[IngestResult] = []
        for section in to_ingest:
            logger.info(
                "[DRY RUN] Would ingest: %s (%d chars)",
                section.name,
                len(section.content),
            )
            results.append(
                IngestResult(
                    item_name=section.name,
                    item_kref=f"kref://{project}/{space_name}/{section.name}.skill",
                    revision_kref=f"kref://{project}/{space_name}/{section.name}.skill?r=<new>",
                    created_new_item=False,
                )
            )
        return results

    import grpc
    import kumiho

    space = _ensure_space(kumiho, grpc, project, space_name)
    results = []

    for section in to_ingest:
        result = _ingest_section(
            kumiho=kumiho,
            grpc=grpc,
            space=space,
            project=project,
            space_name=space_name,
            section=section,
            skill_name=parsed.name,
            skill_description=parsed.description,
            skill_tags=parsed.tags,
            source_file=source_file,
        )
        results.append(result)
        logger.info("Ingested: %s → %s", section.name, result.revision_kref)

    return results


def _ensure_space(kumiho: Any, grpc: Any, project_name: str, space_name: str) -> Any:
    """Get or create the target space."""
    project = kumiho.get_project(project_name)
    if project is None:
        raise RuntimeError(f"Project '{project_name}' does not exist")

    try:
        return project.get_space(space_name)
    except grpc.RpcError:
        logger.info("Creating space: %s/%s", project_name, space_name)
        return project.create_space(space_name)


def _ingest_section(
    *,
    kumiho: Any,
    grpc: Any,
    space: Any,
    project: str,
    space_name: str,
    section: SkillSection,
    skill_name: str,
    skill_description: str,
    skill_tags: list[str],
    source_file: Path,
) -> IngestResult:
    """Ingest a single section as an item + revision + artifact."""
    kref_uri = f"kref://{project}/{space_name}/{section.name}.skill"
    created_new = False

    # 1. Find existing item or create new
    try:
        item = kumiho.get_item(kref_uri)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            item = space.create_item(section.name, "skill")
            created_new = True
        else:
            raise

    # 2. Create revision (all metadata values must be str)
    metadata = {
        "title": section.title,
        "summary": f"Skill: {section.title} — {skill_description}",
        "tags": json.dumps(skill_tags + ["skill"]),
        "agent_compat": json.dumps(["claude", "zeroclaw", "openclaw"]),
        "source_skill": skill_name,
        "content": section.content,
    }
    revision = item.create_revision(metadata=metadata)

    # 3. Attach source file as artifact
    artifact = revision.create_artifact(
        name=f"{section.name}.md",
        location=str(source_file),
        metadata={
            "section": section.name,
            "line_start": str(section.line_start),
        },
    )

    # 4. Tag as published (server auto-moves from previous revision)
    revision.tag("published")

    return IngestResult(
        item_name=section.name,
        item_kref=str(item.kref),
        revision_kref=str(revision.kref),
        artifact_kref=str(artifact.kref) if artifact else None,
        created_new_item=created_new,
    )


# ---------------------------------------------------------------------------
# Ingestion — standalone file (reference doc)
# ---------------------------------------------------------------------------


def ingest_file(
    path: str | Path,
    *,
    item_name: Optional[str] = None,
    project: str = "CognitiveMemory",
    space_name: str = "Skills",
    tags: Optional[list[str]] = None,
    dry_run: bool = False,
) -> IngestResult:
    """Ingest a standalone markdown file as a single skill item.

    Args:
        path: Path to the markdown file.
        item_name: Item name (default: filename stem, slugified).
        project: Kumiho project name.
        space_name: Space within the project.
        tags: Additional tags (``["skill"]`` is always included).
        dry_run: Parse and log only.

    Returns:
        :class:`IngestResult` for the ingested item.
    """
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")

    if item_name is None:
        item_name = _slugify(path.stem)

    # Extract title from first # heading
    title = item_name
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:
            title = m.group(2).strip()
            break

    # Extract summary from first non-empty paragraph
    summary = _extract_first_paragraph(text)

    all_tags = list(tags or []) + ["skill"]

    if dry_run:
        logger.info(
            "[DRY RUN] Would ingest file: %s as %s (%d chars)",
            path.name,
            item_name,
            len(text),
        )
        return IngestResult(
            item_name=item_name,
            item_kref=f"kref://{project}/{space_name}/{item_name}.skill",
            revision_kref=f"kref://{project}/{space_name}/{item_name}.skill?r=<new>",
            created_new_item=False,
        )

    import grpc
    import kumiho

    space = _ensure_space(kumiho, grpc, project, space_name)
    kref_uri = f"kref://{project}/{space_name}/{item_name}.skill"
    created_new = False

    try:
        item = kumiho.get_item(kref_uri)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            item = space.create_item(item_name, "skill")
            created_new = True
        else:
            raise

    revision = item.create_revision(metadata={
        "title": title,
        "summary": summary or f"Reference: {title}",
        "tags": json.dumps(all_tags),
        "agent_compat": json.dumps(["claude", "zeroclaw", "openclaw"]),
        "content": text,
    })

    artifact = revision.create_artifact(
        name=path.name,
        location=str(path),
    )

    revision.tag("published")

    return IngestResult(
        item_name=item_name,
        item_kref=str(item.kref),
        revision_kref=str(revision.kref),
        artifact_kref=str(artifact.kref) if artifact else None,
        created_new_item=created_new,
    )


# ---------------------------------------------------------------------------
# Batch ingestion — directory of reference docs
# ---------------------------------------------------------------------------


def ingest_batch(
    directory: str | Path,
    *,
    project: str = "CognitiveMemory",
    space_name: str = "Skills",
    tags: Optional[list[str]] = None,
    dry_run: bool = False,
) -> list[IngestResult]:
    """Ingest all ``.md`` files in a directory as standalone skill items.

    Args:
        directory: Path to directory containing markdown files.
        project: Kumiho project name.
        space_name: Space within the project.
        tags: Additional tags for all items.
        dry_run: Parse and log only.

    Returns:
        List of :class:`IngestResult` — one per file.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        logger.info("No .md files found in %s", directory)
        return []

    results = []
    for md_file in md_files:
        result = ingest_file(
            md_file,
            project=project,
            space_name=space_name,
            tags=tags,
            dry_run=dry_run,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_first_paragraph(text: str) -> str:
    """Extract the first non-empty, non-heading paragraph from markdown."""
    lines = text.splitlines()
    paragraph: list[str] = []
    in_frontmatter = False

    for line in lines:
        stripped = line.strip()

        # Skip frontmatter
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue

        # Skip headings and horizontal rules
        if stripped.startswith("#") or stripped == "---":
            if paragraph:
                break
            continue

        # Skip empty lines
        if not stripped:
            if paragraph:
                break
            continue

        paragraph.append(stripped)

    return " ".join(paragraph)[:300] if paragraph else ""
