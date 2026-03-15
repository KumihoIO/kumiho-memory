"""CLI entry point for kumiho-memory.

Usage::

    # Run Dream State (reads config from ~/.kumiho/preferences.json)
    python -m kumiho_memory dream

    # Dry run (assess but don't mutate)
    python -m kumiho_memory dream --dry-run

    # Custom project and batch size
    python -m kumiho_memory dream --project MyProject --batch-size 10

    # Via the installed console script
    kumiho-memory dream
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

    if parsed.command == "dream":
        return cmd_dream(parsed)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
