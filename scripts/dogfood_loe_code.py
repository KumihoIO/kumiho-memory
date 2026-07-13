# -*- coding: utf-8 -*-
"""Dogfood: capture the §6 (LoE ranking) decision via the KEYLESS code path
and inspect whether the code Decision Memory ontology-izes well.

Runs the working-tree (#6) code against the local CE. Proves:
  - a code_decision node is created with the NEW evidence_level metadata
    (self-graded 'corroborated' because it carries a measurement atom),
  - IMPLEMENTED_IN edges -> code anchors (the changed files),
  - MOTIVATED_BY edges -> code_evidence atoms.
No LLM key.
"""
import asyncio, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KUMIHO_MEMORY_CODE"] = "1"

import kumiho
from kumiho_memory.code_capture import capture_decisions
from kumiho_memory.code_decisions import (
    CodeMemoryConfig, KIND_DECISION, KIND_EVIDENCE,
    EDGE_IMPLEMENTED_IN, EDGE_MOTIVATED_BY,
)

REPO = "G:/git/KumihoIO/kumiho-SDKs"
PROJECT = "loe-code-dogfood"
KEEP = "--keep" in sys.argv

DECISION = {
    "title": "Level-of-Evidence ranking in code_why",
    "decision": ("code_capture stamps a deterministic evidence_level from a "
                 "decision's evidence atoms; code_why folds the evidence delta "
                 "into the probabilistic ranking slot."),
    "rationale": ("Code decisions had no evidence grade, so code_why ranked a "
                  "thin commit-message decision equal to a measured one. Keyless "
                  "deterministic grading + a no-op-when-ungraded delta (mirroring "
                  "the #12 recall reranker) surfaces well-substantiated decisions "
                  "first, with no LLM."),
    "why_question": "Why does code_why rank well-evidenced decisions above thin ones?",
    "confidence": "high",
    "anchors": [
        {"file": "python/kumiho-memory/kumiho_memory/code_capture.py"},
        {"file": "python/kumiho-memory/kumiho_memory/code_query.py"},
    ],
    "symbols": ["_evidence_grade", "_sort_candidates"],
    "evidence": [
        {"kind": "measurement", "text": "Full suite 536 passed / 2 pre-existing "
         "env-drift failures; 3 new tests pass (evidence_grade mapping, "
         "corroborated>unverified tie-break, ungraded no-op)."},
        {"kind": "rejected_alternative", "text": "Considered threading "
         "EvidenceRankConfig into why() for tunable weights; rejected as "
         "over-plumbing for a cheap wiring — reused DEFAULT_EVIDENCE_WEIGHTS."},
        {"kind": "constraint", "text": "official is never auto-assigned; "
         "evidence.py reserves it for an explicit operator flag."},
    ],
}


async def main():
    proj = kumiho.get_project(PROJECT) or kumiho.create_project(PROJECT)
    cfg = CodeMemoryConfig(project=PROJECT, repo="kumiho-SDKs")

    stats = await capture_decisions(
        REPO, [DECISION], commit_ref="HEAD", project_name=PROJECT, config=cfg,
    )
    sd = stats.as_dict() if hasattr(stats, "as_dict") else vars(stats)
    print("CAPTURE stats:", {k: sd[k] for k in ("decisions", "evidence", "anchors", "edges") if k in sd})
    if sd.get("errors"):
        print("  errors:", sd["errors"])

    print("\n=== code Decision Memory graph ===")
    decs = list(kumiho.item_search(context_filter=PROJECT, name_filter="", kind_filter=KIND_DECISION))
    ok = False
    for it in decs:
        rev = it.get_latest_revision()
        md = dict(getattr(rev, "metadata", {}) or {})
        impl = list(rev.get_edges(edge_type_filter=EDGE_IMPLEMENTED_IN, direction=0))
        motv = list(rev.get_edges(edge_type_filter=EDGE_MOTIVATED_BY, direction=0))
        anchors = [e.target_kref.uri.split("/")[-1] for e in impl]
        print(f"code_decision: {md.get('title')!r}")
        print(f"  evidence_level : {md.get('evidence_level')}   <-- §6 (measurement atom -> corroborated)")
        print(f"  confidence     : {md.get('confidence')}")
        print(f"  IMPLEMENTED_IN : {len(impl)} anchors -> {anchors}")
        print(f"  MOTIVATED_BY   : {len(motv)} evidence atoms")
        ok = (md.get("evidence_level") == "corroborated" and len(impl) >= 2 and len(motv) >= 3)

    n_ev = len(list(kumiho.item_search(context_filter=PROJECT, name_filter="", kind_filter=KIND_EVIDENCE)))
    print(f"\ncode_evidence nodes in project: {n_ev}")
    print("\nDOGFOOD", "PASS" if ok else "FAIL")

    if not KEEP:
        try:
            client = kumiho.get_client()
            pid = getattr(proj, "id", None) or getattr(proj, "project_id", None)
            client.delete_project(pid, force=True)
            print(f"cleaned up project {PROJECT!r}")
        except Exception as e:
            print(f"cleanup failed (manual delete needed): {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
