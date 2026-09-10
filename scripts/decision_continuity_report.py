# -*- coding: utf-8 -*-
"""Emit the deterministic-tier decision-continuity report (#26).

Runs every scenario in ``tests/fixtures/decision_continuity_v1.json`` in the
memory (B) and no-memory (A) arms through the real MCP tool handlers over the
in-memory SDK fake, and writes:

* ``decision_continuity_report.json`` — machine-readable, one ``run_id``
  linking fixture version/sha256, product version/git sha, per-scenario
  checks, outcomes and metrics;
* ``decision_continuity_summary.md`` — the short readable summary.

Offline, keyless, deterministic. Unlike the other ``scripts/dogfood_*``
harnesses this one needs no server and no key, so it is safe to run anywhere
the unit suite runs. It is NOT evidence that a model decides better; the
agent tier lives in kumihoclouds/kumiho-benchmarks ``decision_bench``.

    python scripts/decision_continuity_report.py --out results/continuity
    python scripts/decision_continuity_report.py --only explicit_correction_replaces_en
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))  # the harness lives with the tests it backs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / "results" / "continuity"), help="output directory")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--fixture", default="", help="fixture path (default: tests/fixtures/decision_continuity_v1.json)")
    args = ap.parse_args(argv)

    import continuity_harness as H

    fixture_path = Path(args.fixture) if args.fixture else H.FIXTURE_PATH
    fixture = H.load_fixture(fixture_path)
    only = [x.strip() for x in args.only.split(",") if x.strip()] or None
    workdir = Path(tempfile.mkdtemp(prefix="continuity-report-"))
    records = H.run_all(fixture, workdir=workdir, only=only)
    report = H.build_report(records, fixture_path)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision_continuity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    summary = H.render_summary(report)
    (out / "decision_continuity_summary.md").write_text(summary, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
    print(summary)
    failed = [s for s in report["scenarios"] if not s["passed"]]
    print(f"wrote {out / 'decision_continuity_report.json'} ({len(report['scenarios'])} rows, {len(failed)} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
