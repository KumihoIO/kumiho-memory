# -*- coding: utf-8 -*-
"""Decision Memory live dogfood gate (docs/DECISION_MEMORY_DESIGN.md §7.2).

Manual, paid, against a live kumiho-server CE — NOT part of CI.  Mines this
repo's own recent history and asks the three questions an agent would ask,
machine-judged (no human in the loop: verbatim evidence makes substring
matching honest).

Success criterion 5 of issue #43: all three queries must surface a decision
derived from the expected commit, with the expected evidence terms.

Usage (env: KUMIHO_MEMORY_CODE=1, an LLM key, and a reachable CE)::

    python scripts/dogfood_code_memory.py --preflight   # 1-commit dry run
    python scripts/dogfood_code_memory.py               # full gate
    python scripts/dogfood_code_memory.py --keep        # keep project after
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROJECT = "dogfood-code-memory"

CASES = [
    {
        "name": "(a) single-worker executor",
        "query": dict(
            question="why is rerank_async a single-worker executor?",
            file="python/kumiho-memory/kumiho_memory/recall_rerank.py",
            line=420,
        ),
        "expect_commit": "cfec845",
        "expect_evidence": ["adversarial", "concurrency"],
    },
    {
        "name": "(b) ontology default ON",
        "query": dict(
            question="why is KUMIHO_MEMORY_ONTOLOGY default ON?",
            file="python/kumiho-memory/kumiho_memory/memory_manager.py",
        ),
        "expect_commit": "e52e5df",
        "expect_evidence": ["paired", "+0.042"],
    },
    {
        "name": "(c) unconditional additive partition",
        "query": dict(
            question="why is the additive partition unconditional?",
            file="python/kumiho-memory/kumiho_memory/context_compose.py",
        ),
        "expect_commit": "10f113e",
        "expect_evidence": ["displaced", "typed one-liners", "grounding session"],
    },
]


def _judge(case, result) -> dict:
    """Machine judgment: a top-3 decision derived from the expected commit
    AND any expected-evidence term appearing in its evidence/rationale."""
    top = result.get("decisions", [])[:3]
    hit_decision = None
    for d in top:
        shas = [c.get("sha", "") for c in d.get("commits", [])]
        shas.append(d.get("kref", ""))
        anchors_sha = [a.get("commit", "") for a in d.get("anchors", [])]
        blob = " ".join(shas + anchors_sha)
        if case["expect_commit"] in blob:
            hit_decision = d
            break
    evidence_hit = False
    matched_term = ""
    if hit_decision is not None:
        ev_blob = " ".join(
            [e.get("statement", "") for e in hit_decision.get("evidence", [])]
            + [hit_decision.get("rationale", ""), hit_decision.get("decision", "")]
        ).lower()
        for term in case["expect_evidence"]:
            if term.lower() in ev_blob:
                evidence_hit = True
                matched_term = term
                break
    return {
        "case": case["name"],
        "commit_hit": hit_decision is not None,
        "evidence_hit": evidence_hit,
        "matched_term": matched_term,
        "top_titles": [d.get("title", "") for d in top],
        "passed": hit_decision is not None and evidence_hit,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true",
                        help="1-commit dry run (paid-run preflight rule)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the dogfood project after the gate")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="query-only against an existing dogfood project")
    parser.add_argument("--max-commits", type=int, default=40)
    args = parser.parse_args()

    os.environ["KUMIHO_MEMORY_CODE"] = "1"

    from kumiho_memory.code_decisions import CodeMemoryConfig
    from kumiho_memory.code_capture import ingest_repo
    from kumiho_memory.code_query import why
    from kumiho_memory.summarization import MemorySummarizer

    summarizer = MemorySummarizer()
    cfg = CodeMemoryConfig(repo="kumiho-SDKs")

    if not args.skip_ingest:
        n = 1 if args.preflight else args.max_commits
        print(f"[1/3] ingest: {REPO} (newest {n} commits) -> project {PROJECT!r}")
        stats = await ingest_repo(
            REPO, None,
            project_name=PROJECT, config=cfg,
            adapter=summarizer.adapter, model=summarizer.light_model,
            max_commits=n,
        )
        print(json.dumps(stats.as_dict(), indent=2, default=str))
        if args.preflight:
            ok = stats.llm_calls >= 1 and not stats.errors
            print(f"PREFLIGHT {'OK — LLM fired, JSON parsed' if ok else 'FAILED'}")
            return 0 if ok else 1
        if stats.errors:
            print("INGEST ERRORS — aborting gate")
            return 1
        if stats.decisions + stats.skipped_marker == 0:
            print("no decisions captured — aborting gate")
            return 1

    print("[2/3] gate queries")
    verdicts = []
    for case in CASES:
        result = await why(
            project_name=PROJECT, config=cfg, limit=5, **case["query"],
        )
        verdict = _judge(case, result)
        verdicts.append(verdict)
        icon = "PASS" if verdict["passed"] else "FAIL"
        print(f"  [{icon}] {verdict['case']}  commit_hit={verdict['commit_hit']} "
              f"evidence_hit={verdict['evidence_hit']} ({verdict['matched_term']})")
        for t in verdict["top_titles"]:
            print(f"        - {t}")

    passed = sum(1 for v in verdicts if v["passed"])
    print(f"[3/3] GATE: {passed}/3")

    if not args.keep:
        try:
            import kumiho

            project = kumiho.get_project(PROJECT)
            if project is not None:
                kumiho.get_client().delete_project(
                    getattr(project, "id", None) or getattr(project, "project_id", None),
                    force=True,
                )
                print(f"cleaned up project {PROJECT!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"cleanup failed (manual delete needed): {exc}")

    return 0 if passed == 3 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
