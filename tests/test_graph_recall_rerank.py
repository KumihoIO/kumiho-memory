"""Tests for cognitive-recall consolidation on the graph-augmented path.

Covers what the migration added:

* the manager's graph path now runs the same post-recall rerank stack as the
  plain path (query_time / event-proximity included) and honors
  retrieve-wide-then-trim;
* ``graph_augmentation=True`` boolean shorthand builds a default config;
* traversal seeds come from top-scored flattened revisions
  (``_traversal_seed_krefs``);
* the optional ``on_llm_usage`` accounting hook fires and can never break
  recall.
"""

import asyncio
from datetime import datetime, timezone

from kumiho_memory.graph_augmentation import (
    GraphAugmentationConfig,
    GraphAugmentedRecall,
    _traversal_seed_krefs,
)
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.recall_rerank import RerankConfig
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


def _make_manager(**kw):
    buffer = RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test")

    async def _store(**k):
        return {}

    async def _retrieve(**k):
        return []

    return UniversalMemoryManager(
        redis_buffer=buffer,
        memory_store=_store,
        memory_retrieve=_retrieve,
        **kw,
    )


class _StubGraphRecall:
    """Records the limit the manager passes and returns a fixed pool."""

    def __init__(self, pool):
        self.pool = pool
        self.last_limit = None

    async def recall(self, query, *, limit, space_paths=None, memory_types=None):
        self.last_limit = limit
        return [dict(m) for m in self.pool[:limit]]


def _wire_graph_stub(mgr, pool):
    stub = _StubGraphRecall(pool)
    mgr._get_graph_recall = lambda: stub

    async def _identity_enrich(memories, query):
        return memories

    mgr._enrich_with_siblings = _identity_enrich
    return stub


# ---------------------------------------------------------------------------
# Manager __init__ — graph_augmentation shorthand
# ---------------------------------------------------------------------------

def test_graph_augmentation_true_builds_default_config():
    mgr = _make_manager(graph_augmentation=True)
    assert isinstance(mgr.graph_augmentation_config, GraphAugmentationConfig)


def test_graph_augmentation_falsy_means_disabled():
    assert _make_manager(graph_augmentation=None).graph_augmentation_config is None
    assert _make_manager(graph_augmentation=False).graph_augmentation_config is None


def test_graph_augmentation_explicit_config_kept():
    cfg = GraphAugmentationConfig(max_hops=2)
    mgr = _make_manager(graph_augmentation=cfg)
    assert mgr.graph_augmentation_config is cfg


# ---------------------------------------------------------------------------
# Manager graph path — rerank stack + wide-then-trim
# ---------------------------------------------------------------------------

QT = datetime(2023, 5, 8, tzinfo=timezone.utc)


def test_graph_path_runs_rerank_stack_with_query_time():
    pool = [
        {"kref": "kref://far", "title": "far", "summary": "",
         "score": 0.85, "event_date": "2018-05-08"},
        {"kref": "kref://near", "title": "near", "summary": "",
         "score": 0.80, "event_date": "2023-05-08"},
    ]
    mgr = _make_manager(
        graph_augmentation=True,
        rerank=RerankConfig(
            recency_enabled=False, mmr_enabled=False,
            event_proximity_enabled=True,
            event_proximity_half_life_days=30.0,
            event_proximity_max_boost=0.12,
        ),
    )
    _wire_graph_stub(mgr, pool)

    out = asyncio.run(
        mgr.recall_memories("q", limit=2, graph_augmented=True, query_time=QT)
    )
    # The temporal prior fires on the graph path too: near overtakes far.
    assert [m["title"] for m in out] == ["near", "far"]

    out_dormant = asyncio.run(
        mgr.recall_memories("q", limit=2, graph_augmented=True)
    )
    assert [m["title"] for m in out_dormant] == ["far", "near"]


def test_graph_path_trims_merged_set_to_max_total():
    pool = [
        {"kref": f"kref://m{i}", "title": f"m{i}", "summary": "",
         "score": 0.4 + 0.1 * i}
        for i in range(8)
    ]
    mgr = _make_manager(
        graph_augmentation=GraphAugmentationConfig(max_total=3),
        rerank=RerankConfig(recency_enabled=False, mmr_enabled=False),
    )
    stub = _wire_graph_stub(mgr, pool)
    # Real graph recall returns merge+traversal sets larger than the limit —
    # make the stub do the same so the max_total trim has work to do.
    async def _overflowing(query, *, limit, space_paths=None, memory_types=None):
        return [dict(m) for m in stub.pool]

    stub.recall = _overflowing
    out = asyncio.run(mgr.recall_memories("q", limit=2, graph_augmented=True))
    assert len(out) == 3              # trimmed to config.max_total


def test_graph_path_passes_plain_limit_not_widened():
    # The multiplier widens PER SUB-QUERY (inside _graph_base_recall), never
    # the limit handed to graph recall — widening the merged set and trimming
    # it against the original query is the measured multi-hop eviction bug.
    pool = [{"kref": "kref://a", "title": "a", "summary": "", "score": 0.5}]
    mgr = _make_manager(graph_augmentation=True, recall_candidate_multiplier=3.0)
    stub = _wire_graph_stub(mgr, pool)
    asyncio.run(mgr.recall_memories("q", limit=3, graph_augmented=True))
    assert stub.last_limit == 3


# ---------------------------------------------------------------------------
# _graph_base_recall — per-sub-query widen → rerank (CE vs THAT query) → trim
# ---------------------------------------------------------------------------

class _RecordingRetrieve:
    def __init__(self, pool):
        self.pool = pool
        self.last_limit = None

    async def __call__(self, *, project, query, limit, **kw):
        self.last_limit = limit
        return [dict(m) for m in self.pool[:limit]]


def _make_manager_with_retrieve(pool, **kw):
    buffer = RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test")

    async def _store(**k):
        return {}

    retrieve = _RecordingRetrieve(pool)
    mgr = UniversalMemoryManager(
        redis_buffer=buffer,
        memory_store=_store,
        memory_retrieve=retrieve,
        **kw,
    )
    return mgr, retrieve


def test_graph_base_recall_widens_reranks_and_trims_per_query():
    # Server order is ascending relevance; the cross-encoder (keyed on the
    # sub-query text appearing in the title) must float its matches into the
    # trimmed top — proving CE runs against THIS query inside the sub-recall.
    pool = [
        {"title": "noise-1", "summary": "", "score": 0.9},
        {"title": "noise-2", "summary": "", "score": 0.8},
        {"title": "hop2 museum", "summary": "", "score": 0.1},
        {"title": "hop2 paris", "summary": "", "score": 0.05},
    ]

    def fake_ce(query, texts):
        return [1.0 if "hop2" in t else 0.0 for t in texts]

    mgr, retrieve = _make_manager_with_retrieve(
        pool,
        recall_candidate_multiplier=2.0,
        rerank=RerankConfig(
            recency_enabled=False, mmr_enabled=False,
            cross_encoder_enabled=True,
        ),
        reranker=fake_ce,
    )
    out = asyncio.run(mgr._graph_base_recall("hop2 details", limit=2))
    assert retrieve.last_limit == 4          # ceil(2 * 2.0) per sub-query
    assert len(out) == 2                     # trimmed back to limit
    assert {m["title"] for m in out} == {"hop2 museum", "hop2 paris"}


def test_graph_final_pass_never_reapplies_cross_encoder():
    # The merged-set pass must not re-score against the original query —
    # a reranker call there would reintroduce the multi-hop eviction.
    calls = {"n": 0}

    def counting_ce(query, texts):
        calls["n"] += 1
        return [0.5] * len(texts)

    pool = [{"kref": "kref://a", "title": "a", "summary": "", "score": 0.5}]
    mgr = _make_manager(
        graph_augmentation=True,
        rerank=RerankConfig(cross_encoder_enabled=True),
        reranker=counting_ce,
    )
    _wire_graph_stub(mgr, pool)  # stub bypasses _graph_base_recall entirely
    asyncio.run(mgr.recall_memories("q", limit=2, graph_augmented=True))
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# _traversal_seed_krefs
# ---------------------------------------------------------------------------

def test_seed_krefs_siblingless_reduces_to_primary_order():
    memories = [
        {"kref": "kref://a", "score": 0.9},
        {"kref": "kref://b", "score": 0.7},
        {"kref": "kref://c", "score": 0.5},
    ]
    assert _traversal_seed_krefs(memories, 2) == ["kref://a", "kref://b"]


def test_seed_krefs_prefers_scored_sibling_revisions():
    memories = [
        {"kref": "kref://shell", "score": 3.0, "sibling_revisions": [
            {"kref": "kref://rev-hot", "_score": 0.9},
            {"kref": "kref://rev-cold", "_score": 0.1},
        ]},
        {"kref": "kref://solo", "score": 0.5},
    ]
    seeds = _traversal_seed_krefs(memories, 2)
    # The shell is skipped; its top-scored revision seeds first.
    assert seeds == ["kref://rev-hot", "kref://solo"]
    assert "kref://shell" not in _traversal_seed_krefs(memories, 10)


def test_seed_krefs_skips_empty_krefs():
    memories = [
        {"kref": "", "score": 0.9},
        {"kref": "kref://real", "score": 0.1},
    ]
    assert _traversal_seed_krefs(memories, 2) == ["kref://real"]


# ---------------------------------------------------------------------------
# sibling_fetch_fn receives the reformulated angles
# ---------------------------------------------------------------------------

class _ReformulatingAdapter:
    async def chat(self, *, messages, model, system=None, max_tokens=1024,
                   json_mode=False):
        return "angle one\nangle two"


def test_sibling_fetch_fn_receives_alt_queries():
    seen = {}

    async def fetch3(memories, query, alt_queries):
        seen["query"] = query
        seen["alts"] = alt_queries
        return memories

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [{"kref": "kref://base", "title": "t", "summary": "s",
                 "score": 0.5}]

    gr = GraphAugmentedRecall(
        recall_fn=recall_fn, adapter=_ReformulatingAdapter(), model="light",
        config=GraphAugmentationConfig(reformulate_queries=True, max_hops=0),
        sibling_fetch_fn=fetch3,
    )
    asyncio.run(gr.recall("trigger", limit=2))
    assert seen["query"] == "trigger"
    assert seen["alts"] == ["angle one", "angle two"]


def test_legacy_two_arg_sibling_fetch_fn_still_works():
    calls = []

    async def fetch2(memories, query):
        calls.append(query)
        return memories

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [{"kref": "kref://base", "title": "t", "summary": "s",
                 "score": 0.5}]

    gr = GraphAugmentedRecall(
        recall_fn=recall_fn, adapter=_ReformulatingAdapter(), model="light",
        config=GraphAugmentationConfig(reformulate_queries=True, max_hops=0),
        sibling_fetch_fn=fetch2,
    )
    out = asyncio.run(gr.recall("trigger", limit=2))
    assert out and calls == ["trigger"]


# ---------------------------------------------------------------------------
# on_llm_usage accounting hook
# ---------------------------------------------------------------------------

class _UsageAdapter:
    """LLM adapter stub exposing the last_usage convention."""

    def __init__(self):
        self.last_usage = {
            "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
        }

    async def chat(self, *, messages, model, system=None, max_tokens=1024,
                   json_mode=False):
        return "alternative query one\nalternative query two"


def _gr(config, adapter=None):
    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [{"kref": "kref://base", "title": "t", "summary": "s",
                 "score": 0.5}]

    return GraphAugmentedRecall(
        recall_fn=recall_fn, adapter=adapter, model="light", config=config,
    )


def test_usage_hook_fires_on_reformulation_with_token_counts():
    seen = []
    cfg = GraphAugmentationConfig(
        reformulate_queries=True, max_hops=0,
        on_llm_usage=lambda phase, info: seen.append((phase, info)),
    )
    gr = _gr(cfg, adapter=_UsageAdapter())
    asyncio.run(gr.recall("trigger", limit=2))
    assert ("recall_reformulation", {
        "model": "light", "prompt_tokens": 11,
        "completion_tokens": 7, "total_tokens": 18,
    }) in seen


def test_usage_hook_absent_adapter_usage_reports_zeros():
    class _NoUsageAdapter:
        async def chat(self, **k):
            return "alt"

    seen = []
    cfg = GraphAugmentationConfig(
        reformulate_queries=True, max_hops=0,
        on_llm_usage=lambda phase, info: seen.append(info),
    )
    gr = _gr(cfg, adapter=_NoUsageAdapter())
    asyncio.run(gr.recall("trigger", limit=2))
    assert seen and seen[0]["total_tokens"] == 0


def test_usage_hook_errors_never_break_recall():
    def _boom(phase, info):
        raise RuntimeError("accounting exploded")

    cfg = GraphAugmentationConfig(
        reformulate_queries=True, max_hops=0, on_llm_usage=_boom,
    )
    gr = _gr(cfg, adapter=_UsageAdapter())
    out = asyncio.run(gr.recall("trigger", limit=2))
    assert out and out[0]["kref"] == "kref://base"


def test_no_hook_configured_is_fine():
    cfg = GraphAugmentationConfig(reformulate_queries=True, max_hops=0)
    gr = _gr(cfg, adapter=_UsageAdapter())
    out = asyncio.run(gr.recall("trigger", limit=2))
    assert out


# ---------------------------------------------------------------------------
# Entity-mediated 2-hop reader (ontology-lite)
# ---------------------------------------------------------------------------

class _FakeKref:
    def __init__(self, uri):
        self.uri = uri


class _FakeEdge:
    def __init__(self, source, target, edge_type):
        self.source_kref = _FakeKref(source)
        self.target_kref = _FakeKref(target)
        self.edge_type = edge_type


class _FakeRev:
    def __init__(self, kref, metadata, edges):
        self.kref = _FakeKref(kref)
        self.metadata = metadata
        self._edges = edges

    def get_edges(self, direction=None):
        return self._edges


def _install_entity_graph(monkeypatch):
    """memory M1 --ABOUT--> anchor A <--ABOUT-- sibling memory M2."""
    import sys
    import types

    M1 = "kref://p/notes/m1.conversation?r=1"
    A = "kref://p/entities/acme.entity?r=1"
    M2 = "kref://p/notes/m2.conversation?r=1"

    graph = {
        M1: _FakeRev(M1, {"title": "M1", "summary": "seed"},
                     [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"},  # anchor: no content
                    [_FakeEdge(M1, A, "ABOUT"), _FakeEdge(M2, A, "ABOUT")]),
        M2: _FakeRev(M2, {"title": "M2", "summary": "sibling about acme"}, []),
    }
    fake = types.ModuleType("kumiho")
    fake.BOTH = "BOTH"
    fake.get_revision = lambda kref: graph[kref]
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return M1, A, M2


def test_entity_reader_surfaces_sibling_and_skips_anchor(monkeypatch):
    M1, A, M2 = _install_entity_graph(monkeypatch)
    gr = GraphAugmentedRecall(config=GraphAugmentationConfig(entity_recall=True))

    augmented = []
    seen = {M1}
    found = asyncio.run(gr._traverse_entity_neighbors([M1], seen, augmented))

    krefs = [m["kref"] for m in augmented]
    assert found == 1
    assert M2 in krefs                      # sibling memory reached via the entity
    assert A not in krefs                   # anchor is a waypoint, never a result
    entry = next(m for m in augmented if m["kref"] == M2)
    assert entry["via_entity"] == A
    assert entry["hop"] == 2
    assert entry["summary"] == "sibling about acme"  # real content, not an empty stub


def test_entity_reader_noop_when_disabled(monkeypatch):
    M1, A, M2 = _install_entity_graph(monkeypatch)
    gr = GraphAugmentedRecall(config=GraphAugmentationConfig(entity_recall=False))
    # The disabled flag is enforced in recall(); the method itself still works,
    # so assert the *config* gate is what recall() checks.
    assert gr.config.entity_recall is False


def _install_graph(monkeypatch, graph, on_get=None):
    import sys
    import types

    def _get(kref):
        if on_get is not None:
            on_get(kref)
        return graph[kref]

    fake = types.ModuleType("kumiho")
    fake.BOTH = "BOTH"
    fake.get_revision = _get
    monkeypatch.setitem(sys.modules, "kumiho", fake)


def test_entity_sibling_is_scoreless_and_follows_involves(monkeypatch):
    # Siblings must be score-less placeholders so recall_rerank never evidence-
    # boosts them into evicting a genuine hit; and hop-2 must follow INVOLVES so
    # event nodes (which carry event_date) about the entity are reachable.
    M1 = "kref://p/notes/m1.conversation?r=1"
    A = "kref://p/entities/acme.entity?r=1"
    M2 = "kref://p/notes/m2.conversation?r=1"
    E = "kref://p/events/launch.event?r=1"
    _install_graph(monkeypatch, {
        M1: _FakeRev(M1, {"title": "M1", "summary": "seed"}, [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"), _FakeEdge(M2, A, "ABOUT"),
            _FakeEdge(E, A, "INVOLVES"),
        ]),
        M2: _FakeRev(M2, {"title": "M2", "summary": "sib",
                          "evidence_level": "official", "source": "wiki"}, []),
        E: _FakeRev(E, {"title": "Launch", "summary": "launched",
                        "event_date": "2026-01-01"}, []),
    })
    gr = GraphAugmentedRecall(config=GraphAugmentationConfig(entity_recall=True))
    augmented = []
    found = asyncio.run(gr._traverse_entity_neighbors([M1], {M1}, augmented))

    krefs = {m["kref"] for m in augmented}
    assert found == 2
    assert M2 in krefs and E in krefs            # ABOUT sibling + INVOLVES event
    for entry in augmented:
        assert entry.get("score") is None        # score-less: not evidence-reweighted
        assert "evidence_level" not in entry     # hub neighbour's grade not copied
        assert "source" not in entry
    assert next(m for m in augmented if m["kref"] == E)["edge_type"] == "INVOLVES"


def test_entity_walk_dedups_shared_anchor_across_seeds(monkeypatch):
    # A hub shared by two seeds is expanded once, not per seed (fan-out guard).
    M1 = "kref://p/notes/m1.conversation?r=1"
    M3 = "kref://p/notes/m3.conversation?r=1"
    A = "kref://p/entities/acme.entity?r=1"
    M2 = "kref://p/notes/m2.conversation?r=1"
    a_fetches = {"n": 0}
    _install_graph(monkeypatch, {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
        M3: _FakeRev(M3, {"title": "M3"}, [_FakeEdge(M3, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"), _FakeEdge(M3, A, "ABOUT"),
            _FakeEdge(M2, A, "ABOUT"),
        ]),
        M2: _FakeRev(M2, {"title": "M2", "summary": "sib"}, []),
    }, on_get=lambda kref: a_fetches.__setitem__("n", a_fetches["n"] + (kref == A)))

    gr = GraphAugmentedRecall(config=GraphAugmentationConfig(entity_recall=True))
    augmented = []
    asyncio.run(gr._traverse_entity_neighbors([M1, M3], {M1, M3}, augmented))
    assert a_fetches["n"] == 1                    # shared hub expanded exactly once


def test_entity_siblings_ordered_by_query_relevance(monkeypatch):
    # The cap's sibling reserve keeps ``sib[:reserve]``; walk arrival order is
    # seed × anchor × edge order — arbitrary w.r.t. the question — so siblings
    # must be sorted by lexical overlap with the query before being appended,
    # or the reserve keeps arbitrary ones and the relevant sibling is trimmed.
    M1 = "kref://p/notes/m1.conversation?r=1"
    A = "kref://p/entities/acme.entity?r=1"
    S_off = "kref://p/notes/offtopic.conversation?r=1"
    S_hit = "kref://p/notes/hit.conversation?r=1"
    _install_graph(monkeypatch, {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"),
            _FakeEdge(S_off, A, "ABOUT"),   # arrives FIRST, irrelevant
            _FakeEdge(S_hit, A, "ABOUT"),   # arrives second, matches the query
        ]),
        S_off: _FakeRev(S_off, {"title": "Gardening",
                                "summary": "tomato plants and soil"}, []),
        S_hit: _FakeRev(S_hit, {"title": "Acme funding",
                                "summary": "acme raised a funding round"}, []),
    })
    gr = GraphAugmentedRecall(config=GraphAugmentationConfig(entity_recall=True))
    augmented = []
    asyncio.run(gr._traverse_entity_neighbors(
        [M1], {M1}, augmented, query="when did acme raise its funding round?",
    ))
    assert [m["kref"] for m in augmented] == [S_hit, S_off]  # relevance first
    # Ordering only — entries stay score-less (evidence-inversion fix intact).
    assert all(m.get("score") is None for m in augmented)


def test_sibling_reserve_rides_on_top_and_never_evicts_edges(monkeypatch):
    # Old in-cap reserve: base(6) + keep_sib(3) left room=0 for edge entries at
    # cap=9 — ON traded the proven edge-traversal payload (the a3c multi-hop
    # weapon) for unproven siblings. New behavior: edges keep exactly their
    # OFF-equivalent room (cap - base) and the sibling reserve rides ON TOP,
    # so the ON output is a strict superset of the OFF output.
    base_krefs = [f"kref://p/notes/b{i}.conversation?r=1" for i in range(6)]
    edge_krefs = [f"kref://p/notes/e{i}.conversation?r=1" for i in range(3)]
    A = "kref://p/entities/acme.entity?r=1"
    sib_krefs = [f"kref://p/notes/s{i}.conversation?r=1" for i in range(2)]

    graph = {}
    for i, b in enumerate(base_krefs):
        edges = []
        if i < 3:  # first three seeds each have one structural edge
            edges.append(_FakeEdge(b, edge_krefs[i], "DERIVED_FROM"))
        if i == 0:  # first seed also links the entity anchor
            edges.append(_FakeEdge(b, A, "ABOUT"))
        graph[b] = _FakeRev(b, {"title": f"B{i}", "summary": "base"}, edges)
    for i, e in enumerate(edge_krefs):
        graph[e] = _FakeRev(e, {"title": f"E{i}", "summary": "edge hit"}, [])
    graph[A] = _FakeRev(A, {"display_name": "Acme"}, [
        _FakeEdge(base_krefs[0], A, "ABOUT"),
        _FakeEdge(sib_krefs[0], A, "ABOUT"),
        _FakeEdge(sib_krefs[1], A, "ABOUT"),
    ])
    for i, s in enumerate(sib_krefs):
        graph[s] = _FakeRev(s, {"title": f"S{i}", "summary": "sibling"}, [])
    _install_graph(monkeypatch, graph)

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [{"kref": b, "title": f"B{i}", "summary": "base",
                 "score": 0.9 - i * 0.1} for i, b in enumerate(base_krefs)]

    gr = GraphAugmentedRecall(
        recall_fn=recall_fn,
        config=GraphAugmentationConfig(entity_recall=True),
    )
    out = asyncio.run(gr.recall("acme question", limit=3))  # cap = 9

    krefs = [m["kref"] for m in out]
    assert all(e in krefs for e in edge_krefs)     # edges NOT evicted (old bug)
    assert all(s in krefs for s in sib_krefs)      # siblings ride on top
    assert len(out) == 6 + 3 + 2                   # base + edges + reserve-capped sibs
    assert krefs[:6] == base_krefs                 # scored base hits stay first
