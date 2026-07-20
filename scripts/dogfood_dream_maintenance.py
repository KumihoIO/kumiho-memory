# -*- coding: utf-8 -*-
"""Live dogfood: keyless Dream State graph maintenance against the local CE.

Seeds the two typed graphs with the exact conditions issue #59 targets, then
runs the KEYLESS deterministic maintenance (no LLM key) and proves:

  A. duplicate entity folded into its alias hub  (measurable node reduction),
     near-duplicate facts about one entity collapsed;
  B. a code_decision whose evidence GREW after capture is re-graded
     unverified -> corroborated from its CURRENT MOTIVATED_BY atoms;
  C. the code_decision is bridged to the conversation entity it is about
     (cross-project ABOUT edge — "one brain" at the graph level);
  and the whole pass is idempotent (a re-run changes nothing).

Mirrors scripts/dogfood_ontology_agent.py + scripts/dogfood_loe_code.py.
"""
import asyncio, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KUMIHO_MEMORY_ONTOLOGY"] = "1"
os.environ["KUMIHO_MEMORY_DECISIONS"] = "1"

import grpc
import kumiho
from kumiho._text import slugify
from kumiho_memory.ontology import decompose_and_link_agent
from kumiho_memory.graph_maintenance import GraphMaintainer, MaintenanceStats
from kumiho_memory.code_decisions import (
    KIND_DECISION, KIND_EVIDENCE, EDGE_MOTIVATED_BY, EDGE_SUPERSEDES,
    decision_slug, evidence_slug,
)

PROJECT = "dream-maint-dogfood"
CODE_PROJECT = PROJECT + "-decisions"
KEEP = "--keep" in sys.argv


def ensure_space(proj, sp):
    try:
        proj.create_space(sp)
    except grpc.RpcError as e:
        if e.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise


def get_or_create_item(proj, slug, kind, parent):
    try:
        return proj.create_item(slug, kind, parent_path=parent)
    except grpc.RpcError as e:
        if e.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise
        return proj.get_item(slug, kind, parent_path=parent)


def count(project, kind):
    return len(list(kumiho.item_search(context_filter=project, name_filter="", kind_filter=kind)))


def out_edges(rev, etype):
    return [e.target_kref.uri for e in rev.get_edges(edge_type_filter=etype, direction=0)]


def seed_ontology():
    """Seed entities (incl. an alias-duplicate pair) + duplicate facts."""
    proj = kumiho.get_project(PROJECT) or kumiho.create_project(PROJECT)
    ensure_space(proj, "conversations")
    conv = get_or_create_item(proj, "session-1", "conversation", f"/{PROJECT}/conversations")
    conv_rev = conv.get_latest_revision() or conv.create_revision(
        metadata={"title": "seed session", "summary": "dogfood"})

    decomp = {
        "entities": [
            # hub that explicitly lists the variant as an alias …
            {"name": "PostgreSQL", "type": "system", "aliases": ["Postgres"]},
            # … and the variant, stored as its own node (different slug)
            {"name": "Postgres", "type": "system"},
            {"name": "config_from_env", "type": "convention"},
            {"name": "bge-m3", "type": "model"},
        ],
        "facts": [
            {"statement": "bge-m3 supports a context window of 8192 tokens",
             "about": ["bge-m3"]},
            # near-duplicate of the above (one trailing word) about the same entity
            {"statement": "bge-m3 supports a context window of 8192 tokens today",
             "about": ["bge-m3"]},
            {"statement": "Postgres provides MVCC isolation", "about": ["Postgres"]},
        ],
    }
    asyncio.run(decompose_and_link_agent(conv_rev.kref.uri, decomp, project_name=PROJECT))
    return proj


def seed_stale_decision():
    """A decision stamped 'unverified' at capture whose evidence later grew:
    a measurement atom now MOTIVATED_BY it, but the grade was never lifted —
    the exact LoE auto-upgrade gap (#6)."""
    proj = kumiho.get_project(CODE_PROJECT) or kumiho.create_project(CODE_PROJECT)
    ensure_space(proj, "decisions")
    ensure_space(proj, "evidence")

    dslug = decision_slug("Switch embedding backend to bge-m3", "2026-01-05")
    d_item = get_or_create_item(proj, dslug, KIND_DECISION, f"/{CODE_PROJECT}/decisions")
    d_rev = d_item.get_latest_revision() or d_item.create_revision(metadata={
        "title": "Switch embedding backend to bge-m3",
        "decision": "move OpenAI 3-small -> bge-m3 for retrieval, resolved via config_from_env",
        "symbols": "config_from_env,embedding_backend",
        "evidence_level": "unverified",   # <-- stamped before evidence existed
        "status": "active",
    })

    eslug = evidence_slug("bge-m3 lifted LoCoMo recall by 6 points")
    e_item = get_or_create_item(proj, eslug, KIND_EVIDENCE, f"/{CODE_PROJECT}/evidence")
    e_rev = e_item.get_latest_revision() or e_item.create_revision(metadata={
        "statement": "bge-m3 lifted LoCoMo recall by 6 points",
        "evidence_kind": "measurement",   # <-- corroborating evidence, added later
    })
    # link only if not already linked (idempotent seed)
    if e_rev.kref.uri not in out_edges(d_rev, EDGE_MOTIVATED_BY):
        d_rev.create_edge(e_rev, EDGE_MOTIVATED_BY, metadata={})
    return proj, d_item


def grade_of(item):
    return dict(item.get_latest_revision().metadata or {}).get("evidence_level")


def main():
    print(f"=== seeding (project={PROJECT!r}, code={CODE_PROJECT!r}) ===")
    conv_proj = seed_ontology()
    code_proj, dec_item = seed_stale_decision()

    e0 = count(PROJECT, "entity")
    f0 = count(PROJECT, "fact")
    print(f"BEFORE: entities={e0} facts={f0} decision_grade={grade_of(dec_item)!r}")

    # --- dry run first: counts, but no mutation ---
    dry = MaintenanceStats()
    GraphMaintainer(kumiho, project=PROJECT, code_project=CODE_PROJECT,
                    dry_run=True).run_keyless(dry)
    print("DRY-RUN would:", {k: dry.as_dict()[k] for k in
          ("entities_merged", "facts_merged", "decisions_regraded", "bridges_created")})
    assert grade_of(dec_item) == "unverified", "dry run must not mutate the grade"
    assert count(PROJECT, "entity") == e0, "dry run must not deprecate entities"

    # --- live run ---
    s1 = MaintenanceStats()
    GraphMaintainer(kumiho, project=PROJECT, code_project=CODE_PROJECT).run_keyless(s1)
    print("LIVE run:", s1.as_dict())

    e1 = count(PROJECT, "entity")
    f1 = count(PROJECT, "fact")
    g1 = grade_of(dec_item)
    bridges = out_edges(dec_item.get_latest_revision(), "ABOUT")
    print(f"AFTER: entities={e1} (was {e0}) facts={f1} (was {f0}) grade={g1!r}")
    print(f"decision ABOUT bridges -> {[b.split('/')[-1] for b in bridges]}")

    # --- idempotent re-run ---
    s2 = MaintenanceStats()
    GraphMaintainer(kumiho, project=PROJECT, code_project=CODE_PROJECT).run_keyless(s2)
    print("RE-RUN (should be all zero for merged/regraded/bridged):", {k: s2.as_dict()[k] for k in
          ("entities_merged", "facts_merged", "decisions_regraded", "bridges_created")})

    checks = {
        "entity merged (node reduction)": e1 == e0 - 1 and s1.entities_merged == 1,
        "fact deduped (node reduction)": f1 == f0 - 1 and s1.facts_merged == 1,
        "evidence re-graded unverified->corroborated": g1 == "corroborated"
                                                        and s1.decisions_regraded == 1,
        "cross-graph bridge >=1": s1.bridges_created >= 1 and len(bridges) >= 1,
        "idempotent re-run": (s2.entities_merged == 0 and s2.facts_merged == 0
                              and s2.decisions_regraded == 0 and s2.bridges_created == 0),
    }
    print("\n=== checks ===")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(checks.values())
    print("\nDOGFOOD", "PASS" if ok else "FAIL")

    if not KEEP:
        client = kumiho.get_client()
        for proj in (conv_proj, code_proj):
            try:
                pid = getattr(proj, "id", None) or getattr(proj, "project_id", None)
                client.delete_project(pid, force=True)
            except Exception as e:
                print(f"cleanup failed for {getattr(proj, 'name', '?')}: {e}")
        print("cleaned up projects")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
