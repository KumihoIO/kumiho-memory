# -*- coding: utf-8 -*-
"""Byte-diff verification harness for the recall hot-path perf work (#97, #102).

Both perf fixes are RESULT-INVARIANT: they change *when* work completes
(event-driven traversal wait) and *how* it is scheduled (bounded-concurrency
sibling enrichment), never *what* recall returns. This harness proves that
empirically: it runs recall via ``UniversalMemoryManager`` against the local
Community Edition and writes a canonical JSON dump with all volatile (timing)
fields stripped. Two runs — pre-change vs post-change, or twice on the same
code — must produce byte-identical files. Any diff is a real regression.

To keep the gate meaningful the manager is built with a DETERMINISTIC config:
graph-augmented recall + sibling enrichment are both ON (so the changed code
paths actually run), but there is NO LLM adapter, so query reformulation and
the LLM sibling reranker are skipped in favour of their deterministic
fallbacks. Recall then depends only on the CE's deterministic vector/BM25
scores and the stored corpus — reproducible across runs. (The #102 cap default
is well above any typical recall, so it is never hit here regardless.)

Run it with the cloud token unset so it targets local CE, and with the
worktree on PYTHONPATH so the patched kumiho_memory is the one under test::

    env -u KUMIHO_AUTH_TOKEN \
        PYTHONPATH=<worktree>/python/kumiho-memory \
        <venv python> scripts/recall_bytediff.py \
            --project BenchmarkEval --out pre.json

    # ... apply the change / switch branch, same command with --out post.json
    diff pre.json post.json        # must be empty

The orchestrator runs this after the ON benchmark arm finishes (both share the
local CE); this module only ships the harness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Em-dashes / non-ASCII summaries: keep printing safe on legacy Windows codepages.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# --- Guard: refuse to run against the cloud (mirrors kumiho-benchmarks run.py) ---
if os.environ.get("KUMIHO_AUTH_TOKEN") and "--allow-token" not in sys.argv and "--compare" not in sys.argv:
    sys.exit(
        "KUMIHO_AUTH_TOKEN is set — this would run against the cloud, not local CE.\n"
        "Unset it (recommended: `env -u KUMIHO_AUTH_TOKEN ...`) or pass "
        "--allow-token to override."
    )


# Keys whose values are per-run wall-clock artifacts, never part of the recall
# result. Stripped before serialization so timing noise can't mask (or fake) a
# byte diff. Stored content timestamps (created_at, event_date) are STABLE
# across runs and are deliberately kept — they are part of the answer.
_VOLATILE_SUFFIXES = ("_ms", "_seconds", "_secs", "_ns")
_VOLATILE_SUBSTRINGS = ("latency", "elapsed", "duration", "took", "wall_clock")


def _is_volatile_key(key: str) -> bool:
    k = key.lower()
    if k.endswith(_VOLATILE_SUFFIXES):
        return True
    return any(sub in k for sub in _VOLATILE_SUBSTRINGS)


#: Recall scores carry a recency prior computed from wall-clock time, so two
#: runs minutes apart drift at ~1e-6 (measured pre-vs-pre on identical code).
#: Round score-like floats to 4 decimals: drift vanishes, while any real
#: ranking-affecting change (>=1e-4 score movement, or a reorder — list order
#: is compared exactly) still fails the diff.
_SCORE_KEY_SUBSTRINGS = ("score",)
_SCORE_DECIMALS = 4


def _strip_volatile(obj, _key: str = ""):
    """Recursively drop volatile (timing) keys and round score floats."""
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v, k)
            for k, v in obj.items()
            if not _is_volatile_key(k)
        }
    if isinstance(obj, list):
        return [_strip_volatile(v, _key) for v in obj]
    if isinstance(obj, float) and any(s in _key.lower() for s in _SCORE_KEY_SUBSTRINGS):
        return round(obj, _SCORE_DECIMALS)
    return obj


def default_queries() -> list:
    """~20 deterministic probe queries.

    A fixed list (not randomized, not time-seeded) so the harness output depends
    only on the corpus and the code under test. Broad enough to surface stacked
    items (exercising sibling enrichment) and graph-connected memories
    (exercising edge traversal) in a typical benchmark project.
    """
    return [
        "what did the user decide",
        "recent changes and updates",
        "preferences and settings",
        "past problems and how they were resolved",
        "goals and plans mentioned earlier",
        "people and organizations discussed",
        "technical details and configuration",
        "reasons behind a choice",
        "something that was postponed or cancelled",
        "a habit or routine that changed",
        "work and projects in progress",
        "questions left open",
        "constraints and requirements",
        "events with a specific date",
        "tools and systems in use",
        "a decision that was later reversed",
        "health, food, or lifestyle notes",
        "travel or location references",
        "feedback given or received",
        "the most important takeaway",
    ]


def _load_queries(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not all(isinstance(q, str) for q in data):
        raise SystemExit(f"--queries file {path!r} must be a JSON list of strings")
    return data


class _NoLLMSummarizer:
    """Summarizer stub with no usable LLM adapter.

    Recall only reads ``.adapter`` / ``.light_model``. With ``adapter=None``,
    query reformulation is skipped and the LLM sibling reranker falls through to
    its deterministic in-process fallback — making recall reproducible.
    """

    adapter = None
    light_model = ""


def _build_manager(project: str, *, graph: bool, sibling_threshold: float):
    from kumiho_memory import RedisMemoryBuffer, UniversalMemoryManager

    graph_config = None
    if graph:
        from kumiho_memory.graph_augmentation import GraphAugmentationConfig
        graph_config = GraphAugmentationConfig()

    return UniversalMemoryManager(
        project=project,
        redis_buffer=RedisMemoryBuffer(),
        summarizer=_NoLLMSummarizer(),
        graph_augmentation=graph_config,
        sibling_similarity_threshold=sibling_threshold,
    )


async def _run_all(manager, queries, *, limit: int, graph: bool) -> list:
    results = []
    for q in queries:
        mems = await manager.recall_memories(
            q, limit=limit, graph_augmented=graph,
        )
        results.append({"query": q, "memories": _strip_volatile(mems)})
    return results


def compare_dumps(path_a: str, path_b: str, tolerance: float = 1e-3) -> list:
    """Structural comparison: everything must match exactly EXCEPT score-like
    floats, which must agree within *tolerance*.

    Recall scores carry a wall-clock recency prior, so exact float equality is
    unattainable across runs even on identical code (measured drift ~1e-6);
    rounding merely moves the boundary. Order, keys, krefs, and content are
    compared exactly — those are the result under test.
    Returns a list of difference descriptions (empty = equivalent).
    """
    diffs: list = []

    def walk(a, b, path):
        if type(a) is not type(b):
            diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
            return
        if isinstance(a, dict):
            if set(a) != set(b):
                diffs.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
                return
            for k in a:
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            if len(a) != len(b):
                diffs.append(f"{path}: length {len(a)} != {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, float):
            key = path.rsplit(".", 1)[-1]
            if any(s in key.lower() for s in _SCORE_KEY_SUBSTRINGS):
                if abs(a - b) > tolerance:
                    diffs.append(f"{path}: score {a} vs {b} (>{tolerance})")
            elif a != b:
                diffs.append(f"{path}: {a} != {b}")
        elif a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")

    with open(path_a, encoding="utf-8") as fh:
        da = json.load(fh)
    with open(path_b, encoding="utf-8") as fh:
        db = json.load(fh)
    walk(da, db, "$")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", nargs=2, metavar=("PRE", "POST"),
                    help="compare two dumps (exact structure/order/content, "
                         "score floats within tolerance) and exit")
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="score tolerance for --compare (default 1e-3)")
    ap.add_argument("--project", help="Kumiho project name to recall against")
    ap.add_argument("--out", help="path to write the canonical JSON dump")
    ap.add_argument("--queries", help="JSON file: list of query strings (default: 20 built-in probes)")
    ap.add_argument("--limit", type=int, default=5, help="recall limit per query (default 5)")
    ap.add_argument("--no-graph", action="store_true", help="disable graph-augmented recall")
    ap.add_argument("--sibling-threshold", type=float, default=0.30,
                    help="sibling_similarity_threshold; >0 exercises sibling enrichment (default 0.30)")
    ap.add_argument("--allow-token", action="store_true", help="allow running with KUMIHO_AUTH_TOKEN set")
    ap.add_argument("--allow-no-stacked", action="store_true",
                    help="do not fail when no stacked/sibling-bearing results surfaced "
                         "(a green diff over such a corpus never exercises the #102 path)")
    args = ap.parse_args()

    if args.compare:
        problems = compare_dumps(args.compare[0], args.compare[1], args.tolerance)
        for p in problems[:40]:
            print(f"[bytediff] DIFF {p}", file=sys.stderr)
        if problems:
            print(f"[bytediff] GATE FAILED: {len(problems)} difference(s)", file=sys.stderr)
            return 1
        print("[bytediff] GATE PASSED: structurally identical "
              f"(scores within {args.tolerance})", file=sys.stderr)
        return 0
    if not args.project or not args.out:
        ap.error("--project and --out are required unless --compare is used")

    graph = not args.no_graph
    queries = _load_queries(args.queries) if args.queries else default_queries()

    # Transparency: confirm which kumiho_memory is under test (worktree vs the
    # editable install pointing at the root checkout).
    import kumiho_memory
    print(f"[bytediff] kumiho_memory: {kumiho_memory.__file__}", file=sys.stderr)
    print(f"[bytediff] project={args.project!r} queries={len(queries)} "
          f"limit={args.limit} graph={graph} sibling_threshold={args.sibling_threshold}",
          file=sys.stderr)

    manager = _build_manager(
        args.project, graph=graph, sibling_threshold=args.sibling_threshold,
    )
    results = asyncio.run(
        _run_all(manager, queries, limit=args.limit, graph=graph)
    )

    payload = {
        "project": args.project,
        "limit": args.limit,
        "graph_augmented": graph,
        "sibling_similarity_threshold": args.sibling_threshold,
        "query_count": len(queries),
        "results": results,
    }
    # sort_keys canonicalizes dict key order; list order (the recall ranking) is
    # preserved — it is a result under test. default=str tolerates datetimes.
    text = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False, default=str,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"[bytediff] wrote {args.out} ({len(text)} chars)", file=sys.stderr)

    # Non-vacuous guard: a green diff proves nothing for the sibling-enrichment
    # path (#102) if no stacked results ever surfaced. Fail loudly instead of
    # passing silently over a corpus with no revision-stacked items.
    stacked = sum(
        1
        for entry in results
        for mem in entry["memories"]
        if mem.get("sibling_revisions")
    )
    print(f"[bytediff] sibling-bearing results: {stacked}", file=sys.stderr)
    if stacked == 0 and not args.allow_no_stacked:
        print(
            "[bytediff] FAIL: zero sibling-bearing results — the sibling "
            "enrichment path was never exercised; pick a corpus with stacked "
            "items or pass --allow-no-stacked to accept a partial gate.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
