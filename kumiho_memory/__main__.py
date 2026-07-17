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


def _print_ingest_results(results: list) -> None:
    """Print one line per result plus a quarantine summary (issue #100)."""
    for r in results:
        marker = "[NEW]" if r.created_new_item else "[REV]"
        quarantine = " [QUARANTINED]" if getattr(r, "quarantined", False) else ""
        print(f"  {marker} {r.item_name} → {r.revision_kref}{quarantine}")

    flagged = [r for r in results if getattr(r, "quarantined", False)]
    if flagged:
        print(
            f"\n  WARNING: {len(flagged)} section(s) quarantined - stored for audit "
            "but withheld from agent_compat/published (not agent-consumable)."
        )
        for r in flagged:
            print(f"    - {r.item_name}: {', '.join(r.quarantine_reasons)}")
        print(
            "  Review, then clear with: "
            "kumiho-memory ingest-skill --clear-quarantine <revision_kref>"
        )


def cmd_ingest_skill(args: argparse.Namespace) -> int:
    """Ingest a SKILL.md or reference doc(s) into the Kumiho graph."""
    from kumiho_memory.skill_ingest import (
        clear_quarantine,
        ingest_batch,
        ingest_file,
        ingest_skill,
        parse_skill,
    )

    # --clear-quarantine: operator vouches for a flagged revision after review.
    if args.clear_quarantine:
        result = clear_quarantine(args.clear_quarantine)
        status = "[CLEARED]" if result.cleared else "[SKIP]"
        print(f"  {status} {result.revision_kref} — {result.detail}")
        return 0 if result.cleared else 1

    if not args.path:
        print("ERROR: path is required (or use --clear-quarantine <kref>)", file=sys.stderr)
        return 1

    target = Path(args.path)

    # Seed the ontology spec alongside skill ingestion at onboarding — the
    # spec is a policy Item agents commit to, seeded like the skills it sits
    # next to. Idempotent (re-seed at same version is a no-op) and best-effort
    # (logged, never fatal); skipped on preview-only runs (--list / --dry-run).
    if not args.dry_run and not args.list:
        from kumiho_memory.ontology_spec import seed_ontology_spec

        seeded = seed_ontology_spec(project_name=args.project)
        if seeded is not None:
            logger.debug(
                "ontology spec %s: %s (v=%s)",
                "seeded" if seeded.created_revision else "current",
                seeded.revision_kref,
                seeded.version,
            )

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
            evidence_level=args.evidence_level,
        )
        _print_ingest_results(results)
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
            evidence_level=args.evidence_level,
        )
        _print_ingest_results([result])
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
        evidence_level=args.evidence_level,
    )
    _print_ingest_results(results)
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
        extra_instructions=args.policy,
    )

    result = asyncio.run(ds.run())

    # Print summary
    print(json.dumps(result, indent=2, default=str))
    errors = result.get("errors", [])
    return 1 if errors else 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Run a SpaceProfiler pass (pure aggregation, no LLM)."""
    from kumiho_memory import SpaceProfiler

    profiler = SpaceProfiler(
        project=args.project,
        window_days=args.window_days,
        dry_run=args.dry_run,
    )

    result = asyncio.run(profiler.run())

    print(json.dumps(result, indent=2, default=str))
    errors = result.get("errors", [])
    return 1 if errors else 0


def cmd_code_ingest(args: argparse.Namespace) -> int:
    """Mine a git commit range into Decision Memory (opt-in domain)."""
    from kumiho_memory.code_decisions import (
        code_memory_enabled, config_from_env, resolve_project_name,
    )

    if not code_memory_enabled():
        print(json.dumps({
            "errors": ["code memory is disabled — set KUMIHO_MEMORY_CODE=1"],
        }))
        return 1

    from kumiho_memory.code_capture import ingest_repo
    from kumiho_memory.summarization import MemorySummarizer

    summarizer = MemorySummarizer()
    cfg = config_from_env()
    stats = asyncio.run(ingest_repo(
        args.repo_path,
        args.range,
        project_name=resolve_project_name(args.project, cfg),
        config=cfg,
        adapter=summarizer.adapter,
        model=summarizer.light_model,
        force=args.force,
        max_commits=args.max_commits,
    ))
    print(json.dumps(stats.as_dict(), indent=2, default=str))
    return 1 if (stats.errors or stats.failed_commits) else 0


def cmd_code_mine_session(args: argparse.Namespace) -> int:
    """Mine an agent session's transcript into Decision Memory (opt-in).

    The loop-closing command: the plugin SessionEnd worker hands a Claude
    Code transcript path here.  ``ingest_first`` runs an incremental commit
    ingest so enrichment targets exist ("talk -> commit -> mine").
    """
    from kumiho_memory.code_decisions import (
        code_memory_enabled, config_from_env, resolve_project_name,
    )

    if not code_memory_enabled():
        print(json.dumps({
            "errors": ["code memory is disabled — set KUMIHO_MEMORY_CODE=1"],
        }))
        return 1

    from kumiho_memory.code_capture import ingest_repo
    from kumiho_memory.code_session import mine_session, parse_claude_transcript
    from kumiho_memory.privacy import PIIRedactor
    from kumiho_memory.summarization import MemorySummarizer

    messages = None
    if args.transcript:
        messages = parse_claude_transcript(args.transcript)
        if not messages:
            print(json.dumps({
                "errors": [f"no messages parsed from transcript {args.transcript!r}"],
            }))
            return 0

    summarizer = MemorySummarizer()
    cfg = config_from_env()
    project_name = resolve_project_name(args.project, cfg)

    async def _run():
        if args.ingest_first:
            try:
                await ingest_repo(
                    args.repo, None, project_name=project_name, config=cfg,
                    adapter=summarizer.adapter, model=summarizer.light_model,
                    max_commits=args.max_commits or 20,
                )
            except Exception as exc:  # noqa: BLE001 — enrichment degrades, mining proceeds
                logger.warning("pre-ingest failed: %s", exc)
        return await mine_session(
            args.session_id,
            project_name=project_name, messages=messages,
            conversation_kref=args.conversation_kref or "", repo_path=args.repo,
            config=cfg, adapter=summarizer.adapter, model=summarizer.light_model,
            redactor=PIIRedactor(), force=args.force,
        )

    stats = asyncio.run(_run())
    print(json.dumps(stats.as_dict(), indent=2, default=str))
    return 1 if stats.errors else 0


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
        nargs="?",
        default=None,
        help="Path to SKILL.md file, standalone .md file, or directory (with --batch)",
    )
    ingest.add_argument(
        "--clear-quarantine",
        default=None,
        metavar="KREF",
        help="Clear quarantine on a flagged skill revision after human review "
        "(re-applies the agent_compat + published consumable markers). "
        "Takes a revision kref; no path argument needed.",
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
        "--evidence-level",
        default=None,
        choices=["official", "corroborated", "single_source", "unverified"],
        help="Evidence grade stamped as revision metadata + mirrored "
        "evidence:<level> tag (default: no grade)",
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
    dream.add_argument(
        "--policy",
        default=None,
        help="Deployment policy text appended to the assessment prompt "
        "(overrides KUMIHO_DREAM_EXTRA_INSTRUCTIONS; pass '' to disable it)",
    )

    # -- profile subcommand --
    profile = sub.add_parser(
        "profile",
        help="Profile each Space's knowledge dynamics (SpaceProfiler)",
        description="Aggregate per-Space churn/evidence/stability signals, "
        "classify each Space (canonical/working/correspondence), and persist "
        "versioned space-profile items. Pure aggregation — no LLM.",
    )
    profile.add_argument(
        "--project",
        default="CognitiveMemory",
        help="Kumiho project name (default: CognitiveMemory)",
    )
    profile.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Look-back window for the revision-rate signal (default: 30)",
    )
    profile.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify but do not persist profiles",
    )

    # -- code-ingest subcommand (Decision Memory, opt-in) --
    code_ingest = sub.add_parser(
        "code-ingest",
        help="Mine git commits into Decision Memory (KUMIHO_MEMORY_CODE=1)",
        description="LLM-structure a git commit range into decision nodes "
        "with git anchors and evidence chains. Idempotent: already-captured "
        "commits are skipped without LLM cost. Omit --range for incremental "
        "mode.",
    )
    code_ingest.add_argument(
        "repo_path",
        nargs="?",
        default=".",
        help="Path to the git repository (default: current directory)",
    )
    code_ingest.add_argument(
        "--range",
        default=None,
        help="Rev range, e.g. HEAD~30..HEAD (omit = incremental)",
    )
    code_ingest.add_argument(
        "--project",
        default="CognitiveMemory",
        help="Memory project; decisions go to '{project}-code' (default: CognitiveMemory)",
    )
    code_ingest.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Cap on commits enumerated (default: config.max_commits)",
    )
    code_ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-capture commits that already carry completion markers",
    )

    # -- code-mine-session subcommand (Decision Memory Phase 2, opt-in) --
    code_mine = sub.add_parser(
        "code-mine-session",
        help="Mine an agent session transcript into Decision Memory",
        description="Enrich commit-mined decisions with conversation-only "
        "alternatives/measurements, capture decisions that never reached a "
        "commit, and bridge them to the conversation. Idempotent: a "
        "completed session is marker-skipped at zero LLM cost.",
    )
    code_mine.add_argument(
        "session_id",
        help="The chat session id (marker identity)",
    )
    code_mine.add_argument(
        "--transcript",
        default=None,
        help="Path to a Claude Code transcript JSONL (the session's messages)",
    )
    code_mine.add_argument(
        "--repo",
        default=".",
        help="Path to the git repository (default: current directory)",
    )
    code_mine.add_argument(
        "--project",
        default="CognitiveMemory",
        help="Memory project; decisions go to '{project}-code' (default: CognitiveMemory)",
    )
    code_mine.add_argument(
        "--conversation-kref",
        default=None,
        help="Consolidated conversation revision kref for the DISCUSSED_IN bridge",
    )
    code_mine.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Cap on commits for the ingest_first pre-pass (default: 20)",
    )
    code_mine.add_argument(
        "--no-ingest-first",
        dest="ingest_first",
        action="store_false",
        help="Skip the incremental commit ingest that seeds enrichment targets",
    )
    code_mine.add_argument(
        "--force",
        action="store_true",
        help="Re-mine a session that already carries a completion marker",
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
    elif parsed.command == "profile":
        return cmd_profile(parsed)
    elif parsed.command == "code-ingest":
        return cmd_code_ingest(parsed)
    elif parsed.command == "code-mine-session":
        return cmd_code_mine_session(parsed)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
