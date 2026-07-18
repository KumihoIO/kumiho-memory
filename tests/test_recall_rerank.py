"""Tests for the post-recall reranking pipeline (recency, MMR, cross-encoder)."""

from datetime import datetime, timedelta, timezone

import pytest

from kumiho_memory.evidence_rank import EvidenceRankConfig
from kumiho_memory.recall_rerank import (
    RerankConfig,
    apply_cross_encoder,
    mmr_diversify,
    rerank,
    _jaccard,
    _minmax,
    _parse_ts,
    _recency_boost,
    _tokens,
)

NOW = datetime(2026, 7, 4, tzinfo=timezone.utc)


def _mem(**kw):
    m = {"title": "", "summary": "", "score": 1.0}
    m.update(kw)
    return m


# ---------------- helpers ----------------

def test_parse_ts_variants():
    assert _parse_ts("2026-07-04T00:00:00Z") == NOW
    assert _parse_ts("2026-07-04T00:00:00+00:00") == NOW
    assert _parse_ts("2026-07-04") is not None  # date-only, midnight UTC
    assert _parse_ts("") is None
    assert _parse_ts(None) is None
    assert _parse_ts("not-a-date") is None


def test_jaccard_and_minmax():
    assert _jaccard(set(), {"a"}) == 0.0
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert _minmax([]) == []
    assert _minmax([5.0, 5.0]) == [0.0, 0.0]
    assert _minmax([0.0, 1.0, 2.0]) == [0.0, 0.5, 1.0]


def test_tokens_handles_korean_and_latin():
    t = _tokens("neo4j 메모리 검색")
    assert "neo4j" in t and "메모리" in t and "검색" in t


# ---------------- recency ----------------

def test_recency_recent_beats_old():
    fresh = _mem(created_at="2026-07-04T00:00:00Z")
    old = _mem(created_at="2026-01-01T00:00:00Z")
    cfg = RerankConfig()
    assert _recency_boost(fresh, cfg, NOW) > _recency_boost(old, cfg, NOW)
    # fresh gets ~full boost, old decayed well below it
    assert _recency_boost(fresh, cfg, NOW) == pytest.approx(cfg.recency_max_boost, abs=1e-9)


def test_recency_no_timestamp_is_zero():
    assert _recency_boost(_mem(), RerankConfig(), NOW) == 0.0


def test_recency_reorders_ties_in_composite():
    # Equal relevance; recency should surface the newer memory.
    a = _mem(title="a", score=0.5, created_at="2026-01-01T00:00:00Z")
    b = _mem(title="b", score=0.5, created_at="2026-07-01T00:00:00Z")
    out = rerank("q", [a, b], config=RerankConfig(mmr_enabled=False), now=NOW)
    assert out[0]["title"] == "b"


# ---------------- MMR ----------------

def test_mmr_separates_near_duplicates():
    # Three items; #1 and #2 are near-identical, #3 is distinct but lower rel.
    dup1 = _mem(title="deploy rollback ecs", score=1.0)
    dup2 = _mem(title="deploy rollback ecs task", score=0.95)
    diff = _mem(title="korean tokenizer morpheme", score=0.90)
    out = mmr_diversify([dup1, dup2, diff], RerankConfig(mmr_lambda=0.5), limit=3)
    # The distinct doc should be promoted above the near-duplicate.
    titles = [m["title"] for m in out]
    assert titles[0] == "deploy rollback ecs"
    assert titles.index("korean tokenizer morpheme") < titles.index("deploy rollback ecs task")


def test_mmr_noop_for_tiny_lists():
    a, b = _mem(title="a"), _mem(title="b")
    assert mmr_diversify([a, b], RerankConfig(), None) == [a, b]


def test_mmr_keeps_scoreless_at_tail():
    s = _mem(title="scored", score=1.0)
    u = _mem(title="unscored")
    u.pop("score")
    out = mmr_diversify([s, _mem(title="s2", score=0.9), _mem(title="s3", score=0.8), u], RerankConfig(), None)
    assert out[-1]["title"] == "unscored"


# ---------------- cross-encoder ----------------

def test_cross_encoder_reorders_by_reranker():
    a = _mem(title="alpha", score=0.9)
    b = _mem(title="bravo", score=0.1)
    # Reranker strongly prefers bravo.
    def rr(query, texts):
        return [0.0 if "alpha" in t else 1.0 for t in texts]
    cfg = RerankConfig(cross_encoder_enabled=True, cross_encoder_weight=1.0, mmr_enabled=False, recency_enabled=False)
    out = rerank("q", [a, b], config=cfg, reranker=rr, now=NOW)
    assert out[0]["title"] == "bravo"
    assert out[0]["_cross_encoder_score"] == 1.0


def test_cross_encoder_failure_is_ignored():
    a = _mem(title="a", score=0.9)
    def boom(query, texts):
        raise RuntimeError("model down")
    out = apply_cross_encoder("q", [a], boom, RerankConfig())
    assert out[0]["score"] == 0.9  # unchanged


# ---------------- composite ----------------

def test_rerank_noop_when_no_signals():
    # No evidence, recency off, MMR off, no reranker → order unchanged.
    a = _mem(title="a", score=0.9)
    b = _mem(title="b", score=0.8)
    cfg = RerankConfig(recency_enabled=False, mmr_enabled=False)
    out = rerank("q", [a, b], config=cfg, now=NOW)
    assert [m["title"] for m in out] == ["a", "b"]


def test_rerank_evidence_still_applies():
    # Lower-relevance but official memory should be liftable by evidence.
    weak_official = _mem(title="official", score=0.80, evidence_level="official")
    strong_unverified = _mem(title="unverified", score=0.86, evidence_level="unverified")
    cfg = RerankConfig(recency_enabled=False, mmr_enabled=False)
    ev = EvidenceRankConfig()
    out = rerank("q", [strong_unverified, weak_official], evidence_config=ev, config=cfg, now=NOW)
    # 0.80 + 0.15 (official) = 0.95 > 0.86 - 0.10 (unverified) = 0.76
    assert out[0]["title"] == "official"


def test_rerank_scoreless_memories_trail():
    scored = _mem(title="scored", score=0.5)
    orphan = _mem(title="orphan")
    orphan.pop("score")
    out = rerank("q", [orphan, scored], config=RerankConfig(mmr_enabled=False), now=NOW)
    assert out[0]["title"] == "scored"
    assert out[-1]["title"] == "orphan"


def test_rerank_empty_list():
    assert rerank("q", [], config=RerankConfig()) == []


# ---------------- LLM reranker ----------------

class _FakeAdapter:
    """Async chat adapter returning a canned response (records the call)."""

    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def chat(self, *, messages, model, system="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.last_model = model
        return self.response


def test_parse_llm_scores_variants():
    from kumiho_memory.recall_rerank import _parse_llm_scores
    assert _parse_llm_scores('{"scores": [0.1, 0.9]}', 2) == [0.1, 0.9]
    assert _parse_llm_scores("[0.2, 0.8]", 2) == [0.2, 0.8]
    assert _parse_llm_scores('here: {"scores":[1,0]} ok', 2) == [1.0, 0.0]
    with pytest.raises(ValueError):
        _parse_llm_scores("nonsense", 2)
    with pytest.raises(ValueError):
        _parse_llm_scores("[0.1]", 2)  # wrong length


def test_llm_reranker_reorders_from_sync_context():
    from kumiho_memory.recall_rerank import make_llm_reranker
    adapter = _FakeAdapter('{"scores": [0.0, 1.0]}')
    rr = make_llm_reranker(adapter, "light-model")
    a = _mem(title="alpha", score=0.9)
    b = _mem(title="bravo", score=0.1)
    cfg = RerankConfig(cross_encoder_enabled=True, cross_encoder_weight=1.0,
                       mmr_enabled=False, recency_enabled=False)
    out = rerank("q", [a, b], config=cfg, reranker=rr, now=NOW)
    assert out[0]["title"] == "bravo"
    assert adapter.calls == 1 and adapter.last_model == "light-model"


def test_llm_reranker_works_inside_running_loop():
    # Simulates the async recall path: the sync reranker must bridge to the
    # already-running loop via a worker thread.
    import asyncio
    from kumiho_memory.recall_rerank import make_llm_reranker

    async def run():
        adapter = _FakeAdapter('{"scores": [1.0, 0.0]}')
        rr = make_llm_reranker(adapter, "m")
        return rr("q", ["doc a", "doc b"])

    scores = asyncio.run(run())
    assert scores == [1.0, 0.0]


def test_llm_reranker_failure_is_a_noop():
    from kumiho_memory.recall_rerank import make_llm_reranker
    bad = _FakeAdapter("not json at all")
    rr = make_llm_reranker(bad, "m")
    a = _mem(title="a", score=0.9)
    b = _mem(title="b", score=0.8)
    cfg = RerankConfig(cross_encoder_enabled=True, mmr_enabled=False, recency_enabled=False)
    out = rerank("q", [a, b], config=cfg, reranker=rr, now=NOW)
    assert [m["title"] for m in out] == ["a", "b"]  # unchanged


def test_default_reranker_model_is_a_known_fastembed_id():
    # Guards against a model-name typo silently disabling the cross-encoder
    # (try_fastembed_reranker swallows the load error and returns None).
    import inspect

    from kumiho_memory.recall_rerank import try_fastembed_reranker

    default = inspect.signature(try_fastembed_reranker).parameters["model_name"].default
    assert default in {
        "BAAI/bge-reranker-base",
        "jinaai/jina-reranker-v2-base-multilingual",
        "Xenova/ms-marco-MiniLM-L-6-v2",
        "Xenova/ms-marco-MiniLM-L-12-v2",
    }, f"default reranker model not a supported fastembed id: {default}"


# ---------------- event-proximity (event_date valid-time) ----------------

from kumiho_memory.recall_rerank import _event_proximity_boost, _pad_iso_date

QT = datetime(2023, 5, 8, tzinfo=timezone.utc)  # a temporal query's reference time


def test_pad_iso_date_variants():
    assert _pad_iso_date("2023") == "2023-01-01"
    assert _pad_iso_date("2023-05") == "2023-05-01"
    assert _pad_iso_date("2023-05-07") == "2023-05-07"
    assert _pad_iso_date("  2023-05-07  ") == "2023-05-07"
    assert _pad_iso_date("last week") == "last week"  # unparseable → passthrough
    assert _pad_iso_date(None) == ""


def test_event_proximity_boost_decays_with_gap():
    cfg = RerankConfig(event_proximity_half_life_days=30.0, event_proximity_max_boost=0.12)
    near = _event_proximity_boost({"event_date": "2023-05-08"}, cfg, QT)  # 0-day gap
    far = _event_proximity_boost({"event_date": "2023-01-08"}, cfg, QT)   # 120-day gap
    assert near == pytest.approx(0.12)         # zero gap → full boost
    assert 0.0 < far < near                    # farther event → smaller boost
    assert _event_proximity_boost({"event_date": "2023"}, cfg, QT) > 0.0  # partial precision
    assert _event_proximity_boost({"title": "x"}, cfg, QT) == 0.0         # no event_date → no-op


def test_event_proximity_default_off_is_noop():
    # Feature disabled: event_date + query_time present but base order preserved.
    a = _mem(title="a", score=0.9, event_date="2023-05-08")
    b = _mem(title="b", score=0.8, event_date="2023-05-08")
    cfg = RerankConfig(recency_enabled=False, mmr_enabled=False)  # proximity defaults off
    out = rerank("q", [a, b], config=cfg, now=NOW, query_time=QT)
    assert [m["title"] for m in out] == ["a", "b"]


def test_event_proximity_requires_query_time():
    # Enabled but query_time=None (a non-temporal query) → dormant, input order kept.
    far_high = _mem(title="far_high", score=0.90, event_date="2018-01-01")
    near_low = _mem(title="near_low", score=0.80, event_date="2023-05-08")
    cfg = RerankConfig(recency_enabled=False, mmr_enabled=False,
                       event_proximity_enabled=True)
    out = rerank("q", [far_high, near_low], config=cfg, now=NOW, query_time=None)
    assert [m["title"] for m in out] == ["far_high", "near_low"]  # gate closed


def test_event_proximity_reorders_when_temporal():
    # Enabled + query_time supplied: the near-dated memory overtakes on the prior.
    a = _mem(title="near", score=0.80, event_date="2023-05-08")  # 0 gap → +0.12
    b = _mem(title="far", score=0.85, event_date="2018-05-08")   # ~5y gap → ~0
    cfg = RerankConfig(recency_enabled=False, mmr_enabled=False,
                       event_proximity_enabled=True,
                       event_proximity_half_life_days=30.0,
                       event_proximity_max_boost=0.12)
    out = rerank("q", [a, b], config=cfg, now=NOW, query_time=QT)
    assert [m["title"] for m in out] == ["near", "far"]  # 0.80+0.12 > 0.85


def test_event_proximity_undated_memory_unaffected():
    # A memory with no event_date gets no temporal prior even when enabled — the
    # regression guard for non-temporal / undated results in a temporal query.
    dated = _mem(title="dated", score=0.80, event_date="2023-05-08")
    undated = _mem(title="undated", score=0.85)
    cfg = RerankConfig(recency_enabled=False, mmr_enabled=False,
                       event_proximity_enabled=True, event_proximity_half_life_days=30.0)
    out = rerank("q", [dated, undated], config=cfg, now=NOW, query_time=QT)
    assert [m["title"] for m in out] == ["dated", "undated"]  # 0.92 > 0.85
    assert undated["score"] == pytest.approx(0.85)            # undated prior = 0


def test_temporal_priors_capped_jointly():
    # recency + event-proximity both maxed must not exceed the single-signal cap.
    m = _mem(title="m", score=0.5, event_date="2026-07-04",
             created_at="2026-07-04T00:00:00Z")
    cfg = RerankConfig(mmr_enabled=False,
                       recency_enabled=True, recency_max_boost=0.12,
                       event_proximity_enabled=True, event_proximity_max_boost=0.12)
    out = rerank("q", [m], config=cfg, now=NOW, query_time=NOW)
    assert out[0]["score"] == pytest.approx(0.5 + 0.12)  # capped at 0.12, not 0.24


# --------- event_date confidence guard (#119, hallucinated-date guard) ---------

def test_unverified_confidence_gets_no_boost():
    # A well-formed but uncorroborated date must not move ranking.
    cfg = RerankConfig(event_proximity_half_life_days=30.0, event_proximity_max_boost=0.12)
    mem = {"event_date": "2023-05-08", "event_date_confidence": "unverified"}
    assert _event_proximity_boost(mem, cfg, QT) == 0.0


@pytest.mark.parametrize("confidence", ["verified", "derived", None])
def test_verified_derived_and_absent_boost_is_byte_identical(confidence):
    # Regression guard: the boost for verified / derived / absent-key rows must
    # equal the pre-#119 value (which had no confidence key at all).
    cfg = RerankConfig(event_proximity_half_life_days=30.0, event_proximity_max_boost=0.12)
    legacy = {"event_date": "2023-05-08"}  # no confidence key = pre-#119 row
    baseline = _event_proximity_boost(legacy, cfg, QT)
    assert baseline > 0.0
    guarded = {"event_date": "2023-05-08"}
    if confidence is not None:
        guarded["event_date_confidence"] = confidence
    assert _event_proximity_boost(guarded, cfg, QT) == baseline


def test_unverified_date_does_not_reorder_but_verified_does():
    # Two memories, identical except confidence: the verified near-dated one
    # overtakes on the temporal prior; the unverified one never does.
    near_verified = _mem(title="near_verified", score=0.80, event_date="2023-05-08",
                         event_date_confidence="verified")
    far = _mem(title="far", score=0.85, event_date="2018-05-08")
    cfg = RerankConfig(recency_enabled=False, mmr_enabled=False,
                       event_proximity_enabled=True,
                       event_proximity_half_life_days=30.0,
                       event_proximity_max_boost=0.12)
    out = rerank("q", [near_verified, far], config=cfg, now=NOW, query_time=QT)
    assert [m["title"] for m in out] == ["near_verified", "far"]  # 0.92 > 0.85

    # Same shape, but the near date is unverified → no boost → order preserved.
    near_unverified = _mem(title="near_unverified", score=0.80, event_date="2023-05-08",
                           event_date_confidence="unverified")
    far2 = _mem(title="far2", score=0.85, event_date="2018-05-08")
    out2 = rerank("q", [near_unverified, far2], config=cfg, now=NOW, query_time=QT)
    assert [m["title"] for m in out2] == ["far2", "near_unverified"]  # 0.85 > 0.80


# ---------------- env-based reranker resolution (shared helper) ----------------

import kumiho_memory.recall_rerank as rr_mod
from kumiho_memory.recall_rerank import resolve_reranker_from_env


def test_from_env_default_is_plain_config():
    cfg = RerankConfig.from_env(env={})
    assert cfg.recency_enabled and cfg.mmr_enabled
    assert cfg.cross_encoder_enabled is False


def test_from_env_kill_switch_disables_recency_and_mmr():
    cfg = RerankConfig.from_env(env={"KUMIHO_RECALL_RERANK": "0"})
    assert not cfg.recency_enabled and not cfg.mmr_enabled
    assert RerankConfig.from_env(env={"KUMIHO_RECALL_RERANK": "false"}).mmr_enabled is False


def test_resolve_reranker_env_unset_returns_none():
    assert resolve_reranker_from_env(env={}) is None


def test_resolve_reranker_cross_encoder_unavailable_is_noop(monkeypatch):
    # fastembed missing → try_fastembed_reranker returns None → safe no-op.
    monkeypatch.setattr(rr_mod, "try_fastembed_reranker", lambda *a, **k: None)
    out = resolve_reranker_from_env(env={"KUMIHO_RERANK_CROSS_ENCODER": "1"})
    assert out is None


def test_resolve_reranker_cross_encoder_mocked(monkeypatch):
    sentinel = lambda q, texts: [1.0] * len(texts)
    monkeypatch.setattr(rr_mod, "try_fastembed_reranker", lambda *a, **k: sentinel)
    out = resolve_reranker_from_env(env={"KUMIHO_RERANK_CROSS_ENCODER": "1"})
    assert out is sentinel


def test_resolve_reranker_llm_path(monkeypatch):
    adapter = _FakeAdapter('{"scores": [1.0, 0.0]}')
    out = resolve_reranker_from_env(
        adapter=adapter, model="m", env={"KUMIHO_RERANK_LLM": "1"},
    )
    assert callable(out)
    assert out("q", ["a", "b"]) == [1.0, 0.0]  # exercises the wired reranker


def test_resolve_reranker_llm_without_adapter_is_noop():
    out = resolve_reranker_from_env(env={"KUMIHO_RERANK_LLM": "1"})
    assert out is None


def test_resolve_reranker_cross_encoder_wins_over_llm(monkeypatch):
    sentinel = lambda q, texts: [0.5] * len(texts)
    monkeypatch.setattr(rr_mod, "try_fastembed_reranker", lambda *a, **k: sentinel)
    adapter = _FakeAdapter('{"scores": [1.0]}')
    out = resolve_reranker_from_env(
        adapter=adapter, model="m",
        env={"KUMIHO_RERANK_CROSS_ENCODER": "1", "KUMIHO_RERANK_LLM": "1"},
    )
    assert out is sentinel  # cross-encoder preferred when both set


# ---------------- rerank_async (event-loop offload) ----------------


def _ce_memories():
    return [
        _mem(title="a", summary="alpha", score=0.2),
        _mem(title="b", summary="bravo", score=0.9),
        _mem(title="c", summary="charlie", score=0.5),
    ]


def test_rerank_async_matches_sync_with_cross_encoder():
    import asyncio
    import copy

    from kumiho_memory.recall_rerank import rerank_async

    reranker = lambda q, texts: [float(len(t)) for t in texts]
    reranker._kumiho_offload_safe = True  # exercise the executor path
    cfg = RerankConfig(cross_encoder_enabled=True, recency_enabled=False)

    sync_in = _ce_memories()
    async_in = copy.deepcopy(sync_in)
    sync_out = rerank("q", sync_in, config=cfg, reranker=reranker, now=NOW)
    async_out = asyncio.run(
        rerank_async("q", async_in, config=cfg, reranker=reranker, now=NOW)
    )
    assert async_out == sync_out
    assert async_out is async_in  # in-place semantics preserved through the thread


def test_rerank_async_inline_when_cross_encoder_dormant():
    import asyncio
    import copy

    import kumiho_memory.recall_rerank as rr

    sync_in = _ce_memories()
    async_in = copy.deepcopy(sync_in)
    cfg = RerankConfig(recency_enabled=False)  # cross_encoder_enabled=False
    sync_out = rerank("q", sync_in, config=cfg, now=NOW)
    async_out = asyncio.run(rerank_async_call(rr, "q", async_in, cfg))
    assert async_out == sync_out


async def rerank_async_call(rr, query, memories, cfg):
    return await rr.rerank_async(query, memories, config=cfg, now=NOW)


def test_rerank_async_does_not_block_event_loop():
    import asyncio
    import time

    from kumiho_memory.recall_rerank import rerank_async

    def slow_reranker(q, texts):
        time.sleep(0.3)  # simulates CPU-bound cross-encoder inference
        return [1.0] * len(texts)

    slow_reranker._kumiho_offload_safe = True

    cfg = RerankConfig(cross_encoder_enabled=True, recency_enabled=False)

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        t = asyncio.ensure_future(ticker())
        try:
            await rerank_async(
                "q", _ce_memories(), config=cfg, reranker=slow_reranker, now=NOW,
            )
        finally:
            t.cancel()
        return ticks

    # Inline execution would freeze the loop for the full 0.3s (ticks == 0);
    # the executor offload must leave it running.
    assert asyncio.run(scenario()) >= 3


def test_rerank_async_thread_routing_by_offload_tag(monkeypatch):
    # The executor hop must happen EXACTLY when the reranker is tagged
    # offload-safe: dormant configs and untagged rerankers (the LLM backend,
    # user callables — they may drive loop-bound clients) stay on the caller's
    # thread; the tagged fastembed cross-encoder moves off it.
    import asyncio
    import threading

    import kumiho_memory.recall_rerank as rr

    seen = {}
    real_rerank = rr.rerank

    def recording_rerank(*a, **kw):
        seen["thread"] = threading.get_ident()
        return real_rerank(*a, **kw)

    monkeypatch.setattr(rr, "rerank", recording_rerank)
    caller = threading.get_ident()

    ce = lambda q, texts: [1.0] * len(texts)
    dormant = RerankConfig(recency_enabled=False)
    enabled = RerankConfig(cross_encoder_enabled=True, recency_enabled=False)

    # (a) dormant cross-encoder -> inline, no thread hop
    asyncio.run(rr.rerank_async("q", _ce_memories(), config=dormant, now=NOW))
    assert seen["thread"] == caller

    # (b) enabled but untagged reranker (LLM-like) -> still inline
    asyncio.run(
        rr.rerank_async("q", _ce_memories(), config=enabled, reranker=ce, now=NOW)
    )
    assert seen["thread"] == caller

    # (c) enabled + tagged -> offloaded to the worker thread
    ce._kumiho_offload_safe = True
    asyncio.run(
        rr.rerank_async("q", _ce_memories(), config=enabled, reranker=ce, now=NOW)
    )
    assert seen["thread"] != caller
