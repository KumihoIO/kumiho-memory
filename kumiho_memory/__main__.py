"""CLI entry point for kumiho-memory.

Usage::

    # Run Dream State (reads config from ~/.kumiho/preferences.json)
    python -m kumiho_memory dream

    # Dry run (assess but don't mutate)
    python -m kumiho_memory dream --dry-run

    # Custom project and batch size
    python -m kumiho_memory dream --project MyProject --batch-size 10

    # Ingest a SKILL.md into the graph
    python -m kumiho_memory ingest-skill path/to/SKILL.md --list
    python -m kumiho_memory ingest-skill path/to/SKILL.md --dry-run
    python -m kumiho_memory ingest-skill path/to/SKILL.md

    # Ingest a standalone reference doc
    python -m kumiho_memory ingest-skill path/to/creative-memory.md --item-name creative-memory

    # Batch ingest a directory of reference docs
    python -m kumiho_memory ingest-skill path/to/references/ --batch

    # Via the installed console script
    kumiho-memory dream
    kumiho-memory ingest-skill path/to/SKILL.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("kumiho_memory")


def _load_preferences() -> dict:
    """Load ~/.kumiho/preferences.json if it exists."""
    prefs_path = Path.home() / ".kumiho" / "preferences.json"
    if prefs_path.exists():
        try:
            return json.loads(prefs_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _configure_llm_from_prefs(prefs: dict, section: str = "dreamState") -> None:
    """Set KUMIHO_LLM_* env vars from preferences.json if not already set.

    The setup wizard stores shared direct-provider config plus per-feature models::

        {
            "llm": { "provider": "gemini", "apiKey": "...", "baseUrl": "..." },
            "dreamState": { "model": { "provider": "gemini", "model": "..." } }
        }

    We merge the shared ``llm`` block first, then let the section-specific
    model config override it before translating to the env vars that
    MemorySummarizer reads.
    """
    model_cfg = {
        **prefs.get("llm", {}),
        **prefs.get(section, {}).get("model", {}),
    }
    if not model_cfg:
        return

    mapping = {
        "provider": "KUMIHO_LLM_PROVIDER",
        "model": "KUMIHO_LLM_MODEL",
        "apiKey": "KUMIHO_LLM_API_KEY",
        "baseUrl": "KUMIHO_LLM_BASE_URL",
    }
    for key, env_var in mapping.items():
        val = model_cfg.get(key)
        if val and not os.environ.get(env_var):
            os.environ[env_var] = val


def cmd_ingest_skill(args: argparse.Namespace) -> int:
    """Ingest a SKILL.md or reference doc(s) into the Kumiho graph."""
    from kumiho_memory.skill_ingest import (
        ingest_batch,
        ingest_file,
        ingest_skill,
        parse_skill,
    )

    target = Path(args.path)

    # --batch: ingest all .md files in a directory
    if args.batch:
        if not target.is_dir():
            print(f"ERROR: --batch requires a directory, got: {target}", file=sys.stderr)
            return 1
        results = ingest_batch(
            target,
            project=args.project,
            space_name=args.space,
            tags=args.tags,
            dry_run=args.dry_run,
        )
        for r in results:
            tag = "[NEW]" if r.created_new_item else "[REV]"
            print(f"  {tag} {r.item_name} → {r.revision_kref}")
        print(f"\nIngested {len(results)} files from {target}")
        return 0

    # --item-name: ingest a standalone file as a single skill item
    if args.item_name:
        if not target.is_file():
            print(f"ERROR: file not found: {target}", file=sys.stderr)
            return 1
        result = ingest_file(
            target,
            item_name=args.item_name,
            project=args.project,
            space_name=args.space,
            tags=args.tags,
            dry_run=args.dry_run,
        )
        tag = "[NEW]" if result.created_new_item else "[REV]"
        print(f"  {tag} {result.item_name} → {result.revision_kref}")
        return 0

    # Default: parse a SKILL.md and ingest sections
    if not target.is_file():
        print(f"ERROR: file not found: {target}", file=sys.stderr)
        return 1

    # --list: show sections without ingesting
    if args.list:
        parsed = parse_skill(target)
        print(f"Skill: {parsed.name}")
        print(f"Description: {parsed.description}")
        print(f"Tags: {parsed.tags}")
        print(f"\nSections ({len(parsed.sections)}):\n")
        for s in parsed.sections:
            marker = "[inline]" if s.inline else "[graph] "
            print(f"  {marker} {s.name}: {s.title} ({len(s.content)} chars, line {s.line_start})")
        graph_count = sum(1 for s in parsed.sections if not s.inline)
        inline_count = sum(1 for s in parsed.sections if s.inline)
        print(f"\n{graph_count} discoverable, {inline_count} inline")
        return 0

    results = ingest_skill(
        target,
        project=args.project,
        space_name=args.space,
        section_filter=args.section,
        dry_run=args.dry_run,
    )
    for r in results:
        tag = "[NEW]" if r.created_new_item else "[REV]"
        print(f"  {tag} {r.item_name} → {r.revision_kref}")
    action = "Would ingest" if args.dry_run else "Ingested"
    print(f"\n{action} {len(results)} sections into {args.project}/{args.space}")
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    """Run a Dream State consolidation cycle."""
    from kumiho_memory import DreamState

    prefs = _load_preferences()
    _configure_llm_from_prefs(prefs, "dreamState")

    ds = DreamState(
        project=args.project,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        max_deprecation_ratio=args.max_deprecation_ratio,
        allow_published_deprecation=args.allow_published_deprecation,
    )

    result = asyncio.run(ds.run())

    # Print summary
    print(json.dumps(result, indent=2, default=str))
    errors = result.get("errors", [])
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kumiho-memory",
        description="Kumiho Memory CLI - standalone memory operations",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command")

    # -- ingest-skill subcommand --
    ingest = sub.add_parser(
        "ingest-skill",
        help="Ingest a SKILL.md or reference docs into the Kumiho graph",
        description="Parse a SKILL.md file and ingest non-inline sections as "
        "versioned skill items in CognitiveMemory/Skills. Also supports "
        "standalone reference docs and batch directory ingestion.",
    )
    ingest.add_argument(
        "path",
        help="Path to SKILL.md file, standalone .md file, or directory (with --batch)",
    )
    ingest.add_argument(
        "--project",
        default="CognitiveMemory",
        help="Kumiho project name (default: CognitiveMemory)",
    )
    ingest.add_argument(
        "--space",
        default="Skills",
        help="Space within the project (default: Skills)",
    )
    ingest.add_argument(
        "--section",
        default=None,
        help="Only ingest the section with this slug name",
    )
    ingest.add_argument(
        "--item-name",
        default=None,
        help="Ingest as a standalone file with this item name (skips section parsing)",
    )
    ingest.add_argument(
        "--batch",
        action="store_true",
        help="Ingest all .md files in the given directory",
    )
    ingest.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Additional tags for ingested items",
    )
    ingest.add_argument(
        "--list",
        action="store_true",
        help="List sections without ingesting",
    )
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview ingestion without making changes",
    )

    # -- dream subcommand --
    dream = sub.add_parser(
        "dream",
        help="Run Dream State memory consolidation",
        description="Run a Dream State cycle: assess recent memories, "
        "deprecate stale ones, enrich metadata, create edges.",
    )
    dream.add_argument(
        "--project",
        default="CognitiveMemory",
        help="Kumiho project name (default: CognitiveMemory)",
    )
    dream.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Memories per LLM assessment batch (default: 20)",
    )
    dream.add_argument(
        "--dry-run",
        action="store_true",
        help="Assess but do not mutate anything",
    )
    dream.add_argument(
        "--max-deprecation-ratio",
        type=float,
        default=0.5,
        help="Max fraction of memories to deprecate per run (default: 0.5)",
    )
    dream.add_argument(
        "--allow-published-deprecation",
        action="store_true",
        help="Allow deprecation of published items (use with caution)",
    )

    parsed = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if parsed.command == "ingest-skill":
        return cmd_ingest_skill(parsed)
    elif parsed.command == "dream":
        return cmd_dream(parsed)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
