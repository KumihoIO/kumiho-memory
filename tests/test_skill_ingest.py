"""Tests for kumiho_memory.skill_ingest — SKILL.md parsing and graph ingestion."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from kumiho_memory.skill_ingest import (
    IngestResult,
    ParsedSkill,
    SkillSection,
    _extract_first_paragraph,
    _parse_frontmatter,
    _slugify,
    ingest_batch,
    ingest_file,
    ingest_skill,
    parse_skill,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SKILL = """\
---
name: test-skill
description: A test skill for unit tests
---

# Test Skill

Preamble text here.

## Section One

Content for section one.

## Section Two

Content for section two.
"""

SKILL_WITH_INLINE = """\
---
name: inline-test
description: Tests inline marker handling
tags: [memory, test]
---

# Inline Test

<!-- inline -->
## Hard Constraints

These constraints are inline and should NOT be ingested.

<!-- inline -->
## Session Bootstrap

Also inline — skip during ingestion.

## Discoverable Section

This section has no inline marker and SHOULD be ingested.

## Another Discoverable

Also ingestible.
"""

SKILL_WITH_PRE_INLINE = """\
---
name: pre-inline
description: Tests inline marker before heading
---

# Pre-inline Test

<!-- inline -->

## Marked Before Heading

This section has the inline marker on the line before the heading.

## Normal Section

This should be ingested.
"""

REFERENCE_DOC = """\
# Creative Memory

Creative memory records what was produced and links it to decisions.

## When to Capture

After writing a deliverable file.

## Capture Flow

Run after delivering the file.
"""

REFERENCE_WITH_FRONTMATTER = """\
---
name: privacy-rules
description: Privacy and trust guidelines
tags: [privacy, compliance]
---

# Privacy & Trust

## What stays local

Full conversation transcripts stay local.
"""


# ---------------------------------------------------------------------------
# Tests — _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_simple(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        assert _slugify("Store & Link Protocol (mandatory)") == "store-link-protocol-mandatory"

    def test_already_slugified(self):
        assert _slugify("already-slugified") == "already-slugified"

    def test_numbers_preserved(self):
        assert _slugify("Step 2 — Context Load") == "step-2-context-load"

    def test_strips_leading_trailing(self):
        assert _slugify("  --hello-- ") == "hello"

    def test_empty_string(self):
        assert _slugify("") == ""

    def test_consecutive_specials(self):
        assert _slugify("foo!!!bar???baz") == "foo-bar-baz"


# ---------------------------------------------------------------------------
# Tests — _parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_basic(self):
        lines = ["name: test-skill\n", "description: A test skill\n"]
        name, desc, tags = _parse_frontmatter(lines)
        assert name == "test-skill"
        assert desc == "A test skill"
        assert tags == []

    def test_with_inline_tags(self):
        lines = [
            "name: tagged\n",
            "description: Tagged skill\n",
            'tags: [memory, "test"]\n',
        ]
        name, desc, tags = _parse_frontmatter(lines)
        assert name == "tagged"
        assert tags == ["memory", "test"]

    def test_with_list_tags(self):
        lines = [
            "name: listed\n",
            "description: Listed tags\n",
            "tags:\n",
            "- alpha\n",
            "- beta\n",
        ]
        name, desc, tags = _parse_frontmatter(lines)
        assert name == "listed"
        assert tags == ["alpha", "beta"]

    def test_quoted_values(self):
        lines = ['name: "quoted-name"\n', "description: 'quoted desc'\n"]
        name, desc, tags = _parse_frontmatter(lines)
        assert name == "quoted-name"
        assert desc == "quoted desc"

    def test_empty(self):
        name, desc, tags = _parse_frontmatter([])
        assert name == ""
        assert desc == ""
        assert tags == []


# ---------------------------------------------------------------------------
# Tests — _extract_first_paragraph
# ---------------------------------------------------------------------------


class TestExtractFirstParagraph:
    def test_simple(self):
        text = "# Heading\n\nFirst paragraph here.\n\nSecond paragraph."
        assert _extract_first_paragraph(text) == "First paragraph here."

    def test_with_frontmatter(self):
        text = "---\nname: test\n---\n\n# Heading\n\nThe paragraph.\n"
        assert _extract_first_paragraph(text) == "The paragraph."

    def test_multiline_paragraph(self):
        text = "# Heading\n\nLine one.\nLine two.\n\nNext paragraph."
        assert _extract_first_paragraph(text) == "Line one. Line two."

    def test_empty(self):
        assert _extract_first_paragraph("") == ""

    def test_only_headings(self):
        assert _extract_first_paragraph("# One\n## Two\n### Three") == ""

    def test_truncates_at_300(self):
        long_line = "x" * 400
        text = f"# Heading\n\n{long_line}\n"
        result = _extract_first_paragraph(text)
        assert len(result) == 300


# ---------------------------------------------------------------------------
# Tests — parse_skill
# ---------------------------------------------------------------------------


class TestParseSkill:
    def test_minimal(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(MINIMAL_SKILL)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            assert parsed.name == "test-skill"
            assert parsed.description == "A test skill for unit tests"
            assert parsed.tags == []
            assert "Preamble text" in parsed.preamble
            assert len(parsed.sections) == 2
            assert parsed.sections[0].name == "section-one"
            assert parsed.sections[0].title == "Section One"
            assert "Content for section one" in parsed.sections[0].content
            assert parsed.sections[0].inline is False
            assert parsed.sections[0].level == 2
            assert parsed.sections[1].name == "section-two"
            assert parsed.source_path == Path(path).resolve()
        finally:
            os.unlink(path)

    def test_inline_marker_in_body(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(SKILL_WITH_INLINE)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            assert parsed.name == "inline-test"
            assert parsed.tags == ["memory", "test"]

            inline_sections = [s for s in parsed.sections if s.inline]
            graph_sections = [s for s in parsed.sections if not s.inline]

            assert len(inline_sections) == 2
            assert inline_sections[0].name == "hard-constraints"
            assert inline_sections[1].name == "session-bootstrap"

            assert len(graph_sections) == 2
            assert graph_sections[0].name == "discoverable-section"
            assert graph_sections[1].name == "another-discoverable"
        finally:
            os.unlink(path)

    def test_inline_marker_before_heading(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(SKILL_WITH_PRE_INLINE)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            inline = [s for s in parsed.sections if s.inline]
            graph = [s for s in parsed.sections if not s.inline]

            assert len(inline) == 1
            assert inline[0].name == "marked-before-heading"

            assert len(graph) == 1
            assert graph[0].name == "normal-section"
        finally:
            os.unlink(path)

    def test_no_frontmatter(self):
        text = "# No Frontmatter\n\n## Section A\n\nContent A.\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            assert parsed.name == ""
            assert parsed.description == ""
            assert len(parsed.sections) == 1
            assert parsed.sections[0].name == "section-a"
        finally:
            os.unlink(path)

    def test_subsections_not_split(self):
        """### headings should be part of the parent ## section, not split."""
        text = """\
---
name: sub-test
description: Test subsections
---

## Parent Section

Intro text.

### Subsection A

Sub content A.

### Subsection B

Sub content B.

## Next Section

Next content.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            assert len(parsed.sections) == 2
            parent = parsed.sections[0]
            assert parent.name == "parent-section"
            assert "### Subsection A" in parent.content
            assert "### Subsection B" in parent.content
            assert parsed.sections[1].name == "next-section"
        finally:
            os.unlink(path)

    def test_line_numbers(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(MINIMAL_SKILL)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            # Sections should have reasonable 1-based line numbers
            for section in parsed.sections:
                assert section.line_start > 0
            # Section One comes before Section Two
            assert parsed.sections[0].line_start < parsed.sections[1].line_start
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests — parse_skill on real SKILL.md files
# ---------------------------------------------------------------------------


class TestParseRealSkills:
    """Parse the actual plugin SKILL.md files to ensure they're valid."""

    # Navigate from kumiho-SDKs/python/kumiho-memory/tests/ up to repo root
    _REPO_ROOT = Path(__file__).resolve().parents[4]  # KumihoIO/
    CLAUDE_SKILL = _REPO_ROOT / "kumiho-plugins" / "claude" / "skills" / "kumiho-memory" / "SKILL.md"
    ZEROCLAW_SKILL = _REPO_ROOT / "kumiho-plugins" / "zeroclaw" / "SKILL.md"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[4] / "kumiho-plugins" / "claude" / "skills" / "kumiho-memory" / "SKILL.md").exists(),
        reason="Claude SKILL.md not found in repo",
    )
    def test_parse_claude_skill(self):
        parsed = parse_skill(self.CLAUDE_SKILL)
        assert parsed.name == "kumiho-memory"
        assert len(parsed.sections) > 0

        inline = [s for s in parsed.sections if s.inline]
        graph = [s for s in parsed.sections if not s.inline]
        assert len(inline) > 0, "Claude SKILL.md should have inline sections"
        assert len(graph) > 0, "Claude SKILL.md should have graph-ingestible sections"

        # Two Reflexes should be inline
        reflex_sections = [s for s in parsed.sections if "reflex" in s.name]
        assert len(reflex_sections) == 1
        assert reflex_sections[0].inline is True

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[4] / "kumiho-plugins" / "zeroclaw" / "SKILL.md").exists(),
        reason="ZeroClaw SKILL.md not found in repo",
    )
    def test_parse_zeroclaw_skill(self):
        parsed = parse_skill(self.ZEROCLAW_SKILL)
        assert parsed.name == "kumiho-memory"
        assert len(parsed.sections) > 0

        # ZeroClaw has no inline markers — all sections are graph-ingestible
        # (ZeroClaw doesn't use <!-- inline --> convention)
        # Just verify it parses without error and has the Two Reflexes section
        reflex_sections = [s for s in parsed.sections if "reflex" in s.name]
        assert len(reflex_sections) == 1


# ---------------------------------------------------------------------------
# Tests — ingest_skill (dry run)
# ---------------------------------------------------------------------------


class TestIngestSkillDryRun:
    def test_dry_run_returns_results(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(SKILL_WITH_INLINE)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path, dry_run=True)
            # Only non-inline sections
            assert len(results) == 2
            assert results[0].item_name == "discoverable-section"
            assert results[1].item_name == "another-discoverable"
            for r in results:
                assert "CognitiveMemory/Skills" in r.item_kref
                assert "?r=<new>" in r.revision_kref
        finally:
            os.unlink(path)

    def test_dry_run_all_inline(self):
        text = """\
---
name: all-inline
description: Everything is inline
---

<!-- inline -->
## Section A

Content A.

<!-- inline -->
## Section B

Content B.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path, dry_run=True)
            assert results == []
        finally:
            os.unlink(path)

    def test_section_filter(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(SKILL_WITH_INLINE)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path, section_filter="discoverable-section", dry_run=True)
            assert len(results) == 1
            assert results[0].item_name == "discoverable-section"
        finally:
            os.unlink(path)

    def test_section_filter_no_match(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(SKILL_WITH_INLINE)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path, section_filter="nonexistent", dry_run=True)
            assert results == []
        finally:
            os.unlink(path)

    def test_custom_project_and_space(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(MINIMAL_SKILL)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path, project="CustomProject", space_name="CustomSpace", dry_run=True)
            assert len(results) == 2
            assert "CustomProject/CustomSpace" in results[0].item_kref
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests — ingest_file (dry run)
# ---------------------------------------------------------------------------


class TestIngestFileDryRun:
    def test_dry_run(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(REFERENCE_DOC)
            f.flush()
            path = f.name

        try:
            stem = _slugify(Path(path).stem)
            result = ingest_file(path, dry_run=True)
            assert result.item_name == stem
            assert "CognitiveMemory/Skills" in result.item_kref
        finally:
            os.unlink(path)

    def test_custom_item_name(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(REFERENCE_DOC)
            f.flush()
            path = f.name

        try:
            result = ingest_file(path, item_name="creative-memory", dry_run=True)
            assert result.item_name == "creative-memory"
            assert "creative-memory" in result.item_kref
        finally:
            os.unlink(path)

    def test_with_tags(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(REFERENCE_DOC)
            f.flush()
            path = f.name

        try:
            result = ingest_file(path, tags=["creative", "cowork"], dry_run=True)
            assert result.item_name is not None
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests — ingest_batch (dry run)
# ---------------------------------------------------------------------------


class TestIngestBatchDryRun:
    def test_batch_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some reference docs
            for name, content in [
                ("creative-memory.md", REFERENCE_DOC),
                ("privacy.md", REFERENCE_WITH_FRONTMATTER),
            ]:
                (Path(tmpdir) / name).write_text(content, encoding="utf-8")

            results = ingest_batch(tmpdir, dry_run=True)
            assert len(results) == 2
            names = {r.item_name for r in results}
            assert "creative-memory" in names
            assert "privacy" in names

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = ingest_batch(tmpdir, dry_run=True)
            assert results == []

    def test_not_a_directory(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Not a directory"):
                ingest_batch(path, dry_run=True)
        finally:
            os.unlink(path)

    def test_sorted_alphabetically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["zebra.md", "alpha.md", "middle.md"]:
                (Path(tmpdir) / name).write_text(f"# {name}\n\nContent.\n", encoding="utf-8")

            results = ingest_batch(tmpdir, dry_run=True)
            assert [r.item_name for r in results] == ["alpha", "middle", "zebra"]


# ---------------------------------------------------------------------------
# Tests — ingest with mocked Kumiho SDK
# ---------------------------------------------------------------------------


class FakeKref:
    def __init__(self, uri):
        self.uri = uri

    def __str__(self):
        return self.uri


class FakeArtifact:
    def __init__(self, name, revision_kref):
        self.kref = FakeKref(f"{revision_kref}/artifact/{name}")


class FakeRevision:
    def __init__(self, item_kref, metadata=None):
        self.kref = FakeKref(f"{item_kref}?r=rev-1")
        self.metadata = metadata or {}
        self._tags = []
        self._artifacts = []

    def tag(self, tag_name):
        self._tags.append(tag_name)

    def create_artifact(self, name, location, metadata=None):
        art = FakeArtifact(name, str(self.kref))
        self._artifacts.append(art)
        return art


class FakeItem:
    def __init__(self, kref_uri):
        self.kref = FakeKref(kref_uri)
        self._revisions = []

    def create_revision(self, metadata=None):
        rev = FakeRevision(str(self.kref), metadata)
        self._revisions.append(rev)
        return rev


class FakeSpace:
    def __init__(self, project_name, space_name):
        self.project_name = project_name
        self.space_name = space_name
        self._items = {}

    def create_item(self, name, kind):
        kref_uri = f"kref://{self.project_name}/{self.space_name}/{name}.{kind}"
        item = FakeItem(kref_uri)
        self._items[name] = item
        return item


class FakeRpcError(Exception):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class FakeStatusCode:
    NOT_FOUND = "NOT_FOUND"


class FakeGrpc:
    RpcError = FakeRpcError
    StatusCode = FakeStatusCode


class TestIngestSkillWithMock:
    """Test actual ingestion logic with mocked Kumiho SDK."""

    def _setup_mocks(self, monkeypatch, existing_items=None):
        """Set up fake kumiho and grpc modules."""
        import sys
        import types

        fake_grpc = types.ModuleType("grpc")
        fake_grpc.RpcError = FakeRpcError
        fake_grpc.StatusCode = FakeStatusCode

        fake_kumiho = types.ModuleType("kumiho")
        space = FakeSpace("CognitiveMemory", "Skills")
        items = existing_items or {}

        def fake_get_item(kref_uri):
            if kref_uri in items:
                return items[kref_uri]
            raise FakeRpcError(FakeStatusCode.NOT_FOUND)

        def fake_get_project(name):
            project = types.SimpleNamespace()
            project.get_space = lambda n: space
            return project

        fake_kumiho.get_item = fake_get_item
        fake_kumiho.get_project = fake_get_project

        monkeypatch.setitem(sys.modules, "grpc", fake_grpc)
        monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)

        return space

    def test_ingest_creates_items(self, monkeypatch):
        space = self._setup_mocks(monkeypatch)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(MINIMAL_SKILL)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path)
            assert len(results) == 2
            assert results[0].item_name == "section-one"
            assert results[0].created_new_item is True
            assert "rev-1" in results[0].revision_kref
            assert results[1].item_name == "section-two"
            assert results[1].created_new_item is True
        finally:
            os.unlink(path)

    def test_ingest_stacks_on_existing_item(self, monkeypatch):
        existing = FakeItem("kref://CognitiveMemory/Skills/section-one.skill")
        existing_items = {"kref://CognitiveMemory/Skills/section-one.skill": existing}
        space = self._setup_mocks(monkeypatch, existing_items=existing_items)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(MINIMAL_SKILL)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path)
            assert len(results) == 2
            # Existing item — stacked revision, not new
            assert results[0].item_name == "section-one"
            assert results[0].created_new_item is False
            # New item
            assert results[1].item_name == "section-two"
            assert results[1].created_new_item is True
        finally:
            os.unlink(path)

    def test_ingest_file_creates_item(self, monkeypatch):
        self._setup_mocks(monkeypatch)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(REFERENCE_DOC)
            f.flush()
            path = f.name

        try:
            result = ingest_file(path, item_name="creative-memory")
            assert result.item_name == "creative-memory"
            assert result.created_new_item is True
            assert result.artifact_kref is not None
        finally:
            os.unlink(path)

    def test_ingest_revision_metadata(self, monkeypatch):
        space = self._setup_mocks(monkeypatch)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(SKILL_WITH_INLINE)
            f.flush()
            path = f.name

        try:
            results = ingest_skill(path)
            # Check that items were created in the space
            assert len(space._items) == 2
            # Check revision metadata
            for item_name, item in space._items.items():
                assert len(item._revisions) == 1
                rev = item._revisions[0]
                assert "published" in rev._tags
                assert rev.metadata.get("source_skill") == "inline-test"
                assert json.loads(rev.metadata["tags"]) == ["memory", "test", "skill"]
                assert json.loads(rev.metadata["agent_compat"]) == ["claude", "zeroclaw", "openclaw"]
                assert len(rev._artifacts) == 1
        finally:
            os.unlink(path)

    def test_ingest_batch_with_mock(self, monkeypatch):
        self._setup_mocks(monkeypatch)

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc-a.md").write_text("# Doc A\n\nContent A.\n", encoding="utf-8")
            (Path(tmpdir) / "doc-b.md").write_text("# Doc B\n\nContent B.\n", encoding="utf-8")

            results = ingest_batch(tmpdir)
            assert len(results) == 2
            assert all(r.created_new_item for r in results)
            assert all(r.artifact_kref is not None for r in results)


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            assert parsed.name == ""
            assert parsed.sections == []
        finally:
            os.unlink(path)

    def test_only_frontmatter(self):
        text = "---\nname: empty\ndescription: No sections\n---\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            assert parsed.name == "empty"
            assert parsed.sections == []
        finally:
            os.unlink(path)

    def test_horizontal_rule_not_confused_with_frontmatter(self):
        text = "# Title\n\n---\n\n## Section\n\nContent.\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            parsed = parse_skill(path)
            assert parsed.name == ""
            assert len(parsed.sections) == 1
        finally:
            os.unlink(path)

    def test_dataclass_defaults(self):
        section = SkillSection(name="test", title="Test", content="# Test", inline=False, level=2, line_start=1)
        assert section.name == "test"

        parsed = ParsedSkill(name="n", description="d", tags=[], preamble="p")
        assert parsed.sections == []
        assert parsed.source_path is None

        result = IngestResult(item_name="i", item_kref="k", revision_kref="r")
        assert result.artifact_kref is None
        assert result.created_new_item is False
