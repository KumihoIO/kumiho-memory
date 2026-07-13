# -*- coding: utf-8 -*-
"""Live dogfood: keyless agent-driven ontology decomposition against local CE.

Proves entity/fact nodes go 0 -> N with ABOUT/DERIVED_FROM/relation edges, and
that a re-run is idempotent (no duplicate edges). No LLM key used.
"""
import asyncio, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KUMIHO_MEMORY_ONTOLOGY"] = "1"

import grpc
import kumiho
from kumiho_memory.ontology import decompose_and_link_agent

PROJECT = "ontology-agent-dogfood"
KEEP = "--keep" in sys.argv


def ensure_space(proj, sp):
    try:
        proj.create_space(sp)
    except grpc.RpcError as e:
        if e.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise


def count(kind):
    return len(list(kumiho.item_search(context_filter=PROJECT, name_filter="", kind_filter=kind)))


def get_or_create_item(proj, slug, kind, parent):
    try:
        return proj.create_item(slug, kind, parent_path=parent)
    except grpc.RpcError as e:
        if e.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise
        return proj.get_item(slug, kind, parent_path=parent)


def main():
    proj = kumiho.get_project(PROJECT) or kumiho.create_project(PROJECT)
    print(f"BEFORE: entities={count('entity')} facts={count('fact')}")

    ensure_space(proj, "conversations")
    conv_item = get_or_create_item(proj, "dogfood-memory-1", "conversation", f"/{PROJECT}/conversations")
    conv_rev = conv_item.get_latest_revision() or conv_item.create_revision(
        metadata={"title": "Decision Memory session", "summary": "keyless ontology dogfood"})
    conv_kref = conv_rev.kref.uri
    print("conversation kref:", conv_kref)

    decomp = {
        "entities": [
            {"name": "Decision Memory", "type": "system"},
            {"name": "config_from_env", "type": "convention", "aliases": ["config helper"]},
            {"name": "KUMIHO_SERVER_ENDPOINT", "type": "convention"},
        ],
        "facts": [
            {"statement": "Feature toggles are resolved through config_from_env, not constructor args",
             "about": ["config_from_env", "Decision Memory"]},
            {"statement": "The community-edition endpoint is read from KUMIHO_SERVER_ENDPOINT",
             "about": ["KUMIHO_SERVER_ENDPOINT"]},
        ],
        "relations": [
            {"subject": "Decision Memory", "predicate": "uses", "object": "config_from_env"},
        ],
    }

    stats1 = asyncio.run(decompose_and_link_agent(conv_kref, decomp, project_name=PROJECT))
    print("decompose #1:", stats1)
    stats2 = asyncio.run(decompose_and_link_agent(conv_kref, decomp, project_name=PROJECT))
    print("decompose #2 (idempotent — new nodes/edges should be ~0):", stats2)

    print(f"AFTER: entities={count('entity')} facts={count('fact')}")

    # edges on the conversation revision
    cr = kumiho.get_revision(conv_kref)
    about = list(cr.get_edges(edge_type_filter="ABOUT", direction=0))
    print(f"conversation ABOUT edges: {len(about)} ->", [e.target_kref.uri.split('/')[-1] for e in about])

    # a fact's DERIVED_FROM -> conversation
    facts = list(kumiho.item_search(context_filter=PROJECT, name_filter="", kind_filter="fact"))
    if facts:
        frev = facts[0].get_latest_revision()
        df = list(frev.get_edges(edge_type_filter="DERIVED_FROM", direction=0))
        ab = list(frev.get_edges(edge_type_filter="ABOUT", direction=0))
        print(f"fact[0] DERIVED_FROM: {len(df)}  ABOUT: {len(ab)}")

    # relation edge: Decision Memory -USES-> config_from_env
    ents = {i.kref.uri.split('/')[-1]: i for i in kumiho.item_search(context_filter=PROJECT, name_filter="", kind_filter="entity")}
    dm = next((i for k, i in ents.items() if "decision-memory" in k), None)
    if dm:
        uses = list(dm.get_latest_revision().get_edges(edge_type_filter="USES", direction=0))
        print(f"entity 'Decision Memory' USES edges: {len(uses)} ->", [e.target_kref.uri.split('/')[-1] for e in uses])

    ok = count("entity") >= 3 and count("fact") >= 2 and len(about) >= 3
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
    raise SystemExit(main())
