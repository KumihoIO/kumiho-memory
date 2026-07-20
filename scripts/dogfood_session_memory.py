# -*- coding: utf-8 -*-
"""Session mining live dogfood gate (docs/SESSION_MINING_DESIGN.md §7.2).

Manual, paid, against a live kumiho-server CE — NOT part of CI.  Proves
ENRICHMENT specifically: a synthetic session about the real cfec845 offload
decision carries a rejected alternative the commit message does NOT contain,
and after mining, why() must return that alternative with session provenance
while the commit-derived decision stays byte-identical.

Machine-judged (verbatim evidence makes substring matching honest):

  [2] enrich, don't create        (both correlation failure directions)
  [3] additive constitution       (before/after snapshot byte-identical)
  [4] why() surfaces the alternative with session:* provenance, no ghost
      sha, and the bridge kref on the enriched decision (§7.2: session A
      is mined with the conversation kref in-band)
  [5] standalone control          (origin=session, match=="semantic")
  [6] idempotent re-mine          (zero LLM calls, zero new nodes)
  [7] bridge-only reconciliation  (DISCUSSED_IN resolves to the kref)

Usage (env: KUMIHO_MEMORY_DECISIONS=1, an LLM key, and a reachable CE)::

    python scripts/dogfood_session_memory.py --preflight   # 1-chunk dry run
    python scripts/dogfood_session_memory.py               # full gate
    python scripts/dogfood_session_memory.py --keep        # keep project after
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROJECT = "dogfood-session-memory"

#: The sentence that exists ONLY in the conversation — the commit message of
#: cfec845 never says this.  Its survival through mining and back out of
#: why() is the entire point of Phase 2.
ALT_QUOTE = (
    "we considered asyncio.to_thread and rejected it because the default "
    "executor is shared - a 32-thread pool oversubscribes the cross-encoder"
)

SESSION_A = "dogfood-session-enrich"
SESSION_B = "dogfood-session-standalone"


def _messages_enrich(sha7: str):
    ts = "2026-07-11T10:{m:02d}:00+00:00"
    return [
        {"role": "user", "timestamp": ts.format(m=0), "content":
         "the CE rerank is blocking the event loop under the locomo harness"},
        {"role": "assistant", "timestamp": ts.format(m=1), "content":
         "two options: asyncio.to_thread, or a dedicated executor"},
        {"role": "user", "timestamp": ts.format(m=2), "content": ALT_QUOTE},
        {"role": "assistant", "timestamp": ts.format(m=3), "content":
         "agreed - a dedicated single-worker ThreadPoolExecutor keeps "
         "inference serialized in "
         "python/kumiho-memory/kumiho_memory/recall_rerank.py. "
         f"committing as {sha7}"},
        {"role": "user", "timestamp": ts.format(m=4), "content": "yes, go ahead"},
    ]


def _messages_standalone():
    ts = "2026-07-11T11:{m:02d}:00+00:00"
    return [
        {"role": "user", "timestamp": ts.format(m=0), "content":
         "should we migrate the embedding backend to bge-m3 now?"},
        {"role": "assistant", "timestamp": ts.format(m=1), "content":
         "we considered migrating now and deferred it because the release "
         "cycle comes first - the LoCoMo gate must not absorb a backend "
         "change mid-flight"},
        {"role": "user", "timestamp": ts.format(m=2), "content":
         "agreed, decided: defer the bge-m3 migration until after the "
         "release cycle"},
    ]


def _conversation_rev(project, slug, title):
    """Get-or-create a synthetic consolidated-conversation revision (the
    bridge target a real consolidation would have produced)."""
    try:
        project.create_space("conversations")
    except Exception:  # noqa: BLE001 — exists
        pass
    try:
        item = project.create_item(
            slug, "memory", parent_path=f"/{PROJECT}/conversations",
        )
    except Exception:  # noqa: BLE001 — exists from a previous run
        item = project.get_item(
            slug, "memory", parent_path=f"/{PROJECT}/conversations",
        )
    return item.get_latest_revision() or item.create_revision(
        metadata={"title": title},
    )


def _find_cfec845_decision(result_decisions):
    for d in result_decisions:
        shas = [c.get("sha", "") for c in d.get("commits", [])]
        anchors_sha = [a.get("commit", "") for a in d.get("anchors", [])]
        if any(s.startswith("cfec845") for s in shas + anchors_sha):
            return d
    return None


async def _snapshot_cfec845(project_name, cfg):
    """The commit-mined cfec845 decision's revision metadata + evidence
    source_refs, straight off the graph (the additive-constitution baseline)."""
    from kumiho_memory.code_query import why

    r = await why(
        question="why is the rerank offloaded to a dedicated executor?",
        file="python/kumiho-memory/kumiho_memory/recall_rerank.py",
        project_name=project_name, config=cfg, limit=5,
    )
    d = _find_cfec845_decision(r.get("decisions", []))
    if d is None:
        return None
    import kumiho

    rev = kumiho.get_revision(d["kref"])
    item = rev.get_item()
    return {
        "kref": d["kref"],
        "metadata": dict(getattr(rev, "metadata", {}) or {}),
        "revision_count": len(item.get_revisions() or [])
        if hasattr(item, "get_revisions") else None,
        "commit_evidence_refs": sorted(
            e.get("source_ref", "") for e in d.get("evidence", [])
            if str(e.get("source_ref", "")).startswith("commit:")
        ),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true",
                        help="mine session A only, judge LLM/JSON health")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    os.environ["KUMIHO_MEMORY_DECISIONS"] = "1"

    from kumiho_memory.code_decisions import CodeMemoryConfig
    from kumiho_memory.code_capture import _run_git, ingest_repo
    from kumiho_memory.code_query import why
    from kumiho_memory.code_session import mine_session
    from kumiho_memory.privacy import PIIRedactor
    from kumiho_memory.summarization import MemorySummarizer

    summarizer = MemorySummarizer()
    adapter, model = summarizer.adapter, summarizer.light_model
    cfg = CodeMemoryConfig(repo="kumiho-SDKs")
    redactor = PIIRedactor()

    sha_full = _run_git(REPO, "rev-parse", "cfec845").strip()
    sha7 = sha_full[:7]

    # [0] commit-first: enrichment targets must exist
    print(f"[0] ingest {REPO} (ensuring cfec845 is mined) -> {PROJECT!r}")
    stats0 = await ingest_repo(
        REPO, f"{sha_full}~1..{sha_full}",
        project_name=PROJECT, config=cfg, adapter=adapter, model=model,
    )
    print(json.dumps(stats0.as_dict(), indent=2, default=str))
    if stats0.errors:
        print("ingest errors — aborting")
        return 1

    before = await _snapshot_cfec845(PROJECT, cfg)
    if before is None:
        print("cfec845 decision not found after ingest — aborting")
        return 1
    print(f"    before-snapshot: {before['kref']}")

    # conversation revision for session A — §7.2 mines A with the kref
    # in-band (the AUTOMINE chain's shape) so [4] can assert the bridge.
    import kumiho

    conv_a_kref = _conversation_rev(
        kumiho.get_project(PROJECT), "dogfood-conv-a", "the rerank offload chat",
    ).kref.uri

    # [1]+[2] mine the enrichment session
    print("[2] mine session A (enrichment)")
    stats = await mine_session(
        SESSION_A, project_name=PROJECT,
        messages=_messages_enrich(sha7), repo_path=REPO,
        config=cfg, adapter=adapter, model=model, redactor=redactor,
        conversation_kref=conv_a_kref,
    )
    print(json.dumps(stats.as_dict(), indent=2, default=str))
    if args.preflight:
        ok = stats.llm_calls >= 1 and not stats.errors
        print(f"PREFLIGHT {'OK — LLM fired, JSON parsed' if ok else 'FAILED'}")
        return 0 if ok else 1

    checks = {}
    checks["[2] enrich-not-create"] = (
        stats.decisions_enriched >= 1 and stats.decisions_created == 0
    )

    # [3] additive constitution
    after = await _snapshot_cfec845(PROJECT, cfg)
    checks["[3] additive-constitution"] = (
        after is not None
        and after["kref"] == before["kref"]
        and after["metadata"] == before["metadata"]
        and after["revision_count"] == before["revision_count"]
        and after["commit_evidence_refs"] == before["commit_evidence_refs"]
    )

    # [4] the alternative comes back out with session provenance
    print("[4] why('why not asyncio.to_thread for the rerank offload?')")
    r = await why(
        question="why not asyncio.to_thread for the rerank offload?",
        file="python/kumiho-memory/kumiho_memory/recall_rerank.py",
        project_name=PROJECT, config=cfg, limit=5,
    )
    top3 = r.get("decisions", [])[:3]
    hit = None
    for d in top3:
        for e in d.get("evidence", []):
            if (
                e.get("kind") == "rejected_alternative"
                and "default executor is shared" in e.get("statement", "")
                and str(e.get("source_ref", "")).startswith("session:")
            ):
                hit = d
                break
        if hit:
            break
    no_ghosts = all(
        c.get("sha") for d in r.get("decisions", []) for c in d.get("commits", [])
    )
    checks["[4] alternative-with-session-provenance"] = hit is not None
    checks["[4] bridge-on-enriched"] = (
        hit is not None
        and (hit.get("conversation") or {}).get("kref") == conv_a_kref
    )
    checks["[4] no-ghost-commits"] = no_ghosts
    for d in top3:
        print(f"    - {d.get('title', '')} (origin={d.get('origin')})")

    # [5] standalone control: a decision that never reached a commit
    print("[5] mine session B (standalone control)")
    stats_b = await mine_session(
        SESSION_B, project_name=PROJECT,
        messages=_messages_standalone(), repo_path=REPO,
        config=cfg, adapter=adapter, model=model, redactor=redactor,
    )
    print(json.dumps(stats_b.as_dict(), indent=2, default=str))
    rb = await why(
        question="why was the bge-m3 migration deferred?",
        project_name=PROJECT, config=cfg, limit=5,
    )
    top3_b = rb.get("decisions", [])[:3]
    checks["[5] standalone-capture"] = stats_b.decisions_created >= 1 and any(
        d.get("origin") == "session"
        and d.get("match") == "semantic"
        and "bge" in (d.get("title", "") + d.get("decision", "")).lower()
        for d in top3_b
    )
    for d in top3_b:
        print(f"    - {d.get('title', '')} (origin={d.get('origin')})")

    # [6] idempotent re-mine
    print("[6] re-mine session A (idempotency)")
    stats2 = await mine_session(
        SESSION_A, project_name=PROJECT,
        messages=_messages_enrich(sha7), repo_path=REPO,
        config=cfg, adapter=adapter, model=model, redactor=redactor,
        conversation_kref=conv_a_kref,
    )
    checks["[6] idempotent"] = (
        stats2.skipped_marker and stats2.llm_calls == 0
        and stats2.evidence_added == 0 and stats2.decisions_created == 0
    )

    # [7] bridge-only reconciliation: a conversation revision materializes
    # later; the pass backfills DISCUSSED_IN with zero LLM calls.
    print("[7] bridge-only reconciliation")
    conv_kref = _conversation_rev(
        kumiho.get_project(PROJECT), "dogfood-conv-b", "the bge-m3 deferral chat",
    ).kref.uri
    stats3 = await mine_session(
        SESSION_B, project_name=PROJECT,
        messages=_messages_standalone(), repo_path=REPO,
        config=cfg, adapter=adapter, model=model, redactor=redactor,
        conversation_kref=conv_kref,
    )
    rb2 = await why(
        question="why was the bge-m3 migration deferred?",
        project_name=PROJECT, config=cfg, limit=5,
    )
    bridged = any(
        (d.get("conversation") or {}).get("kref") == conv_kref
        for d in rb2.get("decisions", [])[:3]
    )
    checks["[7] bridge-resolves"] = stats3.llm_calls == 0 and bridged

    print()
    passed = 0
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        passed += 1 if ok else 0
    print(f"GATE: {passed}/{len(checks)}")

    if not args.keep:
        try:
            project = kumiho.get_project(PROJECT)
            if project is not None:
                kumiho.get_client().delete_project(
                    getattr(project, "id", None) or getattr(project, "project_id", None),
                    force=True,
                )
                print(f"cleaned up project {PROJECT!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"cleanup failed (manual delete needed): {exc}")

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
