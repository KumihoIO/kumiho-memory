"""Tests for evidence-weighted recall reranking + badges (issue #12)."""

import asyncio

from kumiho_memory.evidence_rank import (
    DEFAULT_EVIDENCE_WEIGHTS,
    EvidenceRankConfig,
    apply_evidence_weights,
    evidence_badge,
)
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


def _mem(kref, score, level=None, tags=None):
    m = {"kref": kref, "score": score, "title": kref, "summary": "s"}
    if level:
        m["evidence_level"] = level
    if tags:
        m["tags"] = tags
    return m


# ---------------------------------------------------------------------------
# apply_evidence_weights
# ---------------------------------------------------------------------------


def test_official_outranks_higher_relevance_rumor():
    """official at 0.50 beats unverified at 0.60 after weighting
    (0.50+0.15=0.65 > 0.60-0.10=0.50)."""
    memories = [
        _mem("kref://rumor", 0.60, "unverified"),
        _mem("kref://official", 0.50, "official"),
    ]
    out = apply_evidence_weights(memories, EvidenceRankConfig())
    assert [m["kref"] for m in out] == ["kref://official", "kref://rumor"]
    assert out[0]["score"] == 0.65
    assert out[0]["base_score"] == 0.50
    assert out[1]["score"] == 0.50


def test_strict_noop_when_no_evidence():
    """No memory graded -> same objects, same order, untouched scores."""
    memories = [_mem("kref://a", 0.9), _mem("kref://b", 0.5)]
    snapshot = [dict(m) for m in memories]
    out = apply_evidence_weights(memories, EvidenceRankConfig())
    assert out is memories
    assert out == snapshot
    assert "base_score" not in out[0]


def test_disabled_is_noop():
    memories = [_mem("kref://a", 0.5, "official")]
    out = apply_evidence_weights(memories, EvidenceRankConfig(enabled=False))
    assert out is memories
    assert out[0]["score"] == 0.5


def test_idempotent_double_application():
    """Applying twice (merge slice + final cap) must not accumulate."""
    memories = [_mem("kref://a", 0.5, "official"), _mem("kref://b", 0.6)]
    cfg = EvidenceRankConfig()
    once = apply_evidence_weights(memories, cfg)
    score_after_once = [m["score"] for m in once]
    twice = apply_evidence_weights(once, cfg)
    assert [m["score"] for m in twice] == score_after_once


def test_level_from_mirrored_tag_only():
    """Grade parsed from the evidence:<level> tag when metadata absent."""
    memories = [
        _mem("kref://tagged", 0.5, tags=["published", "evidence:corroborated"]),
        _mem("kref://plain", 0.55),
    ]
    out = apply_evidence_weights(memories, EvidenceRankConfig())
    assert out[0]["kref"] == "kref://tagged"  # 0.58 > 0.55
    assert abs(out[0]["score"] - 0.58) < 1e-9


def test_stable_sort_preserves_ties():
    memories = [
        _mem("kref://first", 0.5, "single_source"),
        _mem("kref://second", 0.5),
    ]
    out = apply_evidence_weights(memories, EvidenceRankConfig())
    assert [m["kref"] for m in out] == ["kref://first", "kref://second"]


def test_custom_weights():
    cfg = EvidenceRankConfig(weights={"unverified": -0.5})
    memories = [_mem("kref://a", 0.6, "unverified"), _mem("kref://b", 0.2)]
    out = apply_evidence_weights(memories, cfg)
    assert out[0]["kref"] == "kref://b"
    assert abs(out[1]["score"] - 0.1) < 1e-9


def test_default_weights_shape():
    assert DEFAULT_EVIDENCE_WEIGHTS == {
        "official": 0.15,
        "corroborated": 0.08,
        "single_source": 0.0,
        "unverified": -0.10,
    }


# ---------------------------------------------------------------------------
# evidence_badge
# ---------------------------------------------------------------------------


def test_badges():
    cfg = EvidenceRankConfig()
    assert evidence_badge(_mem("k", 0.5, "official"), cfg) == "[official] "
    assert evidence_badge(_mem("k", 0.5, "corroborated"), cfg) == "[corroborated] "
    assert evidence_badge(_mem("k", 0.5, "unverified"), cfg) == "[unverified] "
    assert evidence_badge(_mem("k", 0.5, "single_source"), cfg) == ""
    assert evidence_badge(_mem("k", 0.5), cfg) == ""
    assert evidence_badge(
        _mem("k", 0.5, "official"), EvidenceRankConfig(badges=False),
    ) == ""


# ---------------------------------------------------------------------------
# Manager integration
# ---------------------------------------------------------------------------


def _make_manager(retrieve_stub, **kwargs):
    class _StubSummarizer:
        async def summarize_conversation(self, messages, context=None):
            return {}

        async def generate_implications(self, messages, context=None):
            return []

    return UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test"),
        summarizer=_StubSummarizer(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
        **kwargs,
    )


def test_recall_memories_applies_weights():
    """Plain recall path reorders graded results."""
    async def retrieve_stub(**kwargs):
        return [
            _mem("kref://rumor", 0.60, "unverified"),
            _mem("kref://official", 0.50, "official"),
        ]

    manager = _make_manager(retrieve_stub)

    async def run():
        results = await manager.recall_memories("query")
        assert [m["kref"] for m in results] == ["kref://official", "kref://rumor"]

    asyncio.run(run())


def test_recall_memories_noop_without_evidence():
    """Ungraded results keep the server order (back-compat)."""
    expected = [_mem("kref://a", 0.4), _mem("kref://b", 0.9)]

    async def retrieve_stub(**kwargs):
        return [dict(m) for m in expected]

    manager = _make_manager(retrieve_stub)

    async def run():
        results = await manager.recall_memories("query")
        assert [m["kref"] for m in results] == ["kref://a", "kref://b"]

    asyncio.run(run())


def test_recall_memories_rerank_disabled():
    async def retrieve_stub(**kwargs):
        return [
            _mem("kref://rumor", 0.60, "unverified"),
            _mem("kref://official", 0.50, "official"),
        ]

    manager = _make_manager(
        retrieve_stub, evidence_rank=EvidenceRankConfig(enabled=False),
    )

    async def run():
        results = await manager.recall_memories("query")
        assert [m["kref"] for m in results] == ["kref://rumor", "kref://official"]

    asyncio.run(run())


def test_build_recalled_context_renders_badges():
    async def retrieve_stub(**kwargs):
        return []

    manager = _make_manager(retrieve_stub)
    memories = [
        _mem("kref://official", 0.5, "official"),
        _mem("kref://plain", 0.4),
    ]
    ctx = manager.build_recalled_context(memories, recall_mode="summarized")
    assert "[official] kref://official: s" in ctx
    assert "[official] kref://plain" not in ctx
    assert "kref://plain: s" in ctx


def test_build_recalled_context_no_badges_when_disabled():
    async def retrieve_stub(**kwargs):
        return []

    manager = _make_manager(
        retrieve_stub, evidence_rank=EvidenceRankConfig(badges=False),
    )
    memories = [_mem("kref://official", 0.5, "official")]
    ctx = manager.build_recalled_context(memories, recall_mode="summarized")
    assert "[official]" not in ctx


def test_build_recalled_context_prepends_event_date_in_summarized():
    # In summarized mode (no artifact content) the event_date anchor is the only
    # way a temporal question sees a date — assert it is surfaced, and only for
    # memories that carry one.
    async def retrieve_stub(**kwargs):
        return []

    manager = _make_manager(retrieve_stub)
    dated = _mem("kref://dated", 0.5)
    dated["event_date"] = "2023-05-08"
    undated = _mem("kref://undated", 0.4)
    ctx = manager.build_recalled_context([dated, undated], recall_mode="summarized")
    assert "[2023-05-08] kref://dated: s" in ctx    # anchor surfaced
    assert "[2023-05-08] kref://undated" not in ctx  # not on the undated memory
    assert "kref://undated: s" in ctx                # undated memory unaffected


# ---------------------------------------------------------------------------
# Graph-augmented path
# ---------------------------------------------------------------------------


def test_graph_recall_weights_before_final_cap():
    """An official memory beyond the unweighted cap boundary survives."""
    from kumiho_memory.graph_augmentation import (
        GraphAugmentationConfig,
        GraphAugmentedRecall,
    )

    # 4 results, cap max_total=3: official is last by base score and would
    # be cut without weighting; with weighting it sorts first.
    base_results = [
        _mem("kref://u1", 0.60, "unverified"),
        _mem("kref://u2", 0.59, "unverified"),
        _mem("kref://u3", 0.58, "unverified"),
        _mem("kref://official", 0.55, "official"),
    ]

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [dict(m) for m in base_results]

    cfg = GraphAugmentationConfig(
        reformulate_queries=False, max_total=3, max_hops=0,
    )
    rank_cfg = EvidenceRankConfig()
    gr = GraphAugmentedRecall(
        recall_fn=recall_fn,
        config=cfg,
        evidence_rerank_fn=lambda mems: apply_evidence_weights(mems, rank_cfg),
    )

    async def run():
        results = await gr.recall("query", limit=2)
        krefs = [m["kref"] for m in results]
        assert "kref://official" in krefs
        assert krefs[0] == "kref://official"  # 0.70 beats 0.50/0.49/0.48
        assert len(results) <= 3

    asyncio.run(run())


def test_graph_recall_unweighted_order_unchanged():
    """Without evidence data the graph path returns base order (no-op)."""
    from kumiho_memory.graph_augmentation import (
        GraphAugmentationConfig,
        GraphAugmentedRecall,
    )

    base_results = [
        _mem("kref://a", 0.9),
        _mem("kref://b", 0.5),
    ]

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [dict(m) for m in base_results]

    rank_cfg = EvidenceRankConfig()
    gr = GraphAugmentedRecall(
        recall_fn=recall_fn,
        config=GraphAugmentationConfig(reformulate_queries=False, max_hops=0),
        evidence_rerank_fn=lambda mems: apply_evidence_weights(mems, rank_cfg),
    )

    async def run():
        results = await gr.recall("query", limit=2)
        assert [m["kref"] for m in results] == ["kref://a", "kref://b"]
        assert results[0]["score"] == 0.9

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Review-hardening tests (adversarial review round 1)
# ---------------------------------------------------------------------------


def test_scoreless_memories_never_get_fabricated_scores():
    """min_score deliberately passes score-less memories — weighting must
    not invent score=0.0 for them just because a batch-mate is graded."""
    scoreless_graded = {"kref": "kref://g", "title": "g", "summary": "s",
                        "evidence_level": "official"}
    scoreless_plain = {"kref": "kref://p", "title": "p", "summary": "s"}
    scored = _mem("kref://s", 0.5, "corroborated")

    out = apply_evidence_weights(
        [scoreless_graded, scoreless_plain, scored], EvidenceRankConfig(),
    )
    assert "score" not in scoreless_graded
    assert "base_score" not in scoreless_graded
    assert "score" not in scoreless_plain
    # scored memories sort first; score-less keep relative order after
    assert [m["kref"] for m in out] == ["kref://s", "kref://g", "kref://p"]


def test_non_numeric_scores_left_untouched():
    """A string score from a JSON backend must not be coerced to 0.0."""
    stringly = {"kref": "kref://str", "score": "0.85"}
    graded = _mem("kref://g", 0.5, "official")
    out = apply_evidence_weights([stringly, graded], EvidenceRankConfig())
    assert stringly["score"] == "0.85"
    assert "base_score" not in stringly
    assert out[0]["kref"] == "kref://g"


def test_all_scoreless_batch_keeps_order():
    memories = [
        {"kref": "kref://a", "evidence_level": "official"},
        {"kref": "kref://b"},
    ]
    out = apply_evidence_weights(memories, EvidenceRankConfig())
    assert [m["kref"] for m in out] == ["kref://a", "kref://b"]
    assert all("score" not in m for m in out)


def test_manager_accepts_falsy_evidence_rank():
    """evidence_rank=False reads as 'disable' — must not crash recall."""
    async def retrieve_stub(**kwargs):
        return [
            _mem("kref://rumor", 0.60, "unverified"),
            _mem("kref://official", 0.50, "official"),
        ]

    manager = _make_manager(retrieve_stub, evidence_rank=False)

    async def run():
        results = await manager.recall_memories("query")
        # disabled -> retrieval order preserved
        assert [m["kref"] for m in results] == ["kref://rumor", "kref://official"]

    asyncio.run(run())
    ctx = manager.build_recalled_context(
        [_mem("kref://official", 0.5, "official")], recall_mode="summarized",
    )
    assert "[official]" not in ctx


def test_graph_traversal_entries_not_mixed_into_weighted_sort():
    """Traversal placeholder scores (0.0 = 'unmeasured') must not compete
    with adjusted base scores: an unverified direct hit stays ahead of
    traversal noise, and the cap trims traversal entries first."""
    from kumiho_memory.graph_augmentation import (
        GraphAugmentationConfig,
        GraphAugmentedRecall,
    )

    base_results = [
        _mem("kref://b1", 0.42),
        _mem("kref://b2", 0.30),
        _mem("kref://b3", 0.08, "unverified"),  # adjusts to -0.02
    ]

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [dict(m) for m in base_results]

    rank_cfg = EvidenceRankConfig()
    gr = GraphAugmentedRecall(
        recall_fn=recall_fn,
        config=GraphAugmentationConfig(
            reformulate_queries=False, max_total=4, max_hops=1,
        ),
        evidence_rerank_fn=lambda mems: apply_evidence_weights(mems, rank_cfg),
    )

    async def fake_traverse(memories, seen_krefs, augmented):
        for kref in ("kref://trav1", "kref://trav2"):
            augmented.append({
                "kref": kref, "title": "t", "summary": "s",
                "score": 0.0, "graph_augmented": True,
                "edge_type": "REFERENCED", "from_kref": "kref://b1",
            })
        return 2

    gr._traverse_edges = fake_traverse

    async def run():
        results = await gr.recall("query", limit=2)
        krefs = [m["kref"] for m in results]
        # cap=4: all three base hits survive (incl. the -0.02 unverified
        # DIRECT hit); exactly one traversal entry is trimmed — noise is
        # cut first, never a measured hit.
        assert krefs[:3] == ["kref://b1", "kref://b2", "kref://b3"]
        assert krefs[3] == "kref://trav1"
        assert len(results) == 4

    asyncio.run(run())


def test_graded_traversal_entry_does_not_outrank_base_hits():
    """An official-graded traversal entry (placeholder 0.0) must not be
    presented above genuinely relevant direct hits."""
    from kumiho_memory.graph_augmentation import (
        GraphAugmentationConfig,
        GraphAugmentedRecall,
    )

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [dict(_mem("kref://b1", 0.14)), dict(_mem("kref://b2", 0.12, "single_source"))]

    rank_cfg = EvidenceRankConfig()
    gr = GraphAugmentedRecall(
        recall_fn=recall_fn,
        config=GraphAugmentationConfig(reformulate_queries=False, max_hops=1),
        evidence_rerank_fn=lambda mems: apply_evidence_weights(mems, rank_cfg),
    )

    async def fake_traverse(memories, seen_krefs, augmented):
        augmented.append({
            "kref": "kref://trav-official", "title": "t", "summary": "s",
            "score": 0.0, "graph_augmented": True,
            "evidence_level": "official",
            "edge_type": "SUPPORTS", "from_kref": "kref://b1",
        })
        return 1

    gr._traverse_edges = fake_traverse

    async def run():
        results = await gr.recall("query", limit=2)
        krefs = [m["kref"] for m in results]
        assert krefs[0] == "kref://b1"
        assert krefs[-1] == "kref://trav-official"

    asyncio.run(run())
