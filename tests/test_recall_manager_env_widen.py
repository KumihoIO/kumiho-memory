"""Manager-level tests for:

* Task 2 — env-based reranker resolution reaching direct construction
  (KUMIHO_RERANK_CROSS_ENCODER=1 works without going through the MCP server).
* Task 3 — retrieve-wide-then-trim (recall_candidate_multiplier) and the
  optional query_time plumbing into the rerank stack.
"""

import asyncio
from datetime import datetime, timezone

import kumiho_memory.recall_rerank as rr_mod
from kumiho_memory.recall_rerank import RerankConfig
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


class _StubSummarizer:
    """Summarizer with no LLM adapter (env resolution sees adapter=None)."""

    light_model = "light-model"


class _RecordingRetrieve:
    """Async retrieve stub returning a list, honoring + recording ``limit``."""

    def __init__(self, pool):
        self.pool = pool
        self.last_limit = None

    async def __call__(self, *, project, query, limit, **kw):
        self.last_limit = limit
        return [dict(m) for m in self.pool[:limit]]


def _make_manager(retrieve=None, **kw):
    buffer = RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test")

    async def _store(**k):
        return {"item_kref": "kref://x"}

    if retrieve is None:
        async def retrieve(**k):
            return []

    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=_StubSummarizer(),
        memory_store=_store,
        memory_retrieve=retrieve,
        **kw,
    )


# --------------------------------------------------------------------------
# Task 2 — env resolution on the direct construction path
# --------------------------------------------------------------------------

def test_env_unset_leaves_defaults_unchanged(monkeypatch):
    monkeypatch.delenv("KUMIHO_RERANK_CROSS_ENCODER", raising=False)
    monkeypatch.delenv("KUMIHO_RERANK_LLM", raising=False)
    monkeypatch.delenv("KUMIHO_RECALL_RERANK", raising=False)
    mgr = _make_manager()
    assert mgr.reranker is None
    assert mgr.rerank_config.cross_encoder_enabled is False
    assert mgr.rerank_config.recency_enabled and mgr.rerank_config.mmr_enabled


def test_env_cross_encoder_unavailable_is_safe_noop(monkeypatch):
    monkeypatch.setattr(rr_mod, "try_fastembed_reranker", lambda *a, **k: None)
    monkeypatch.setenv("KUMIHO_RERANK_CROSS_ENCODER", "1")
    mgr = _make_manager()
    assert mgr.reranker is None
    assert mgr.rerank_config.cross_encoder_enabled is False


def test_env_cross_encoder_mocked_activates_config(monkeypatch):
    sentinel = lambda q, texts: [1.0] * len(texts)
    monkeypatch.setattr(rr_mod, "try_fastembed_reranker", lambda *a, **k: sentinel)
    monkeypatch.setenv("KUMIHO_RERANK_CROSS_ENCODER", "1")
    mgr = _make_manager()
    assert mgr.reranker is sentinel
    assert mgr.rerank_config.cross_encoder_enabled is True


def test_env_resolution_never_builds_or_crashes_on_the_adapter(monkeypatch):
    # The summarizer's lazy ``adapter`` property builds a real LLM client and
    # raises without an API key.  Env resolution at construction must touch
    # it only when KUMIHO_RERANK_LLM requests it — and swallow its errors.
    class _RaisingAdapterSummarizer:
        light_model = "m"
        built = 0

        @property
        def adapter(self):
            _RaisingAdapterSummarizer.built += 1
            raise RuntimeError("no api key configured")

    monkeypatch.delenv("KUMIHO_RERANK_CROSS_ENCODER", raising=False)
    monkeypatch.delenv("KUMIHO_RERANK_LLM", raising=False)
    mgr = _make_manager()  # default env: property must never be touched
    buffer = RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test")

    async def _store(**k):
        return {}

    async def _retrieve(**k):
        return []

    mgr = UniversalMemoryManager(
        redis_buffer=buffer, summarizer=_RaisingAdapterSummarizer(),
        memory_store=_store, memory_retrieve=_retrieve,
    )
    assert _RaisingAdapterSummarizer.built == 0

    monkeypatch.setenv("KUMIHO_RERANK_LLM", "1")
    mgr = UniversalMemoryManager(
        redis_buffer=buffer, summarizer=_RaisingAdapterSummarizer(),
        memory_store=_store, memory_retrieve=_retrieve,
    )  # property raises inside the factory — construction survives
    assert _RaisingAdapterSummarizer.built == 1
    assert mgr.reranker is None
    assert mgr.rerank_config.cross_encoder_enabled is False


def test_explicit_reranker_bypasses_env(monkeypatch):
    # When the caller passes a reranker explicitly, env is not consulted.
    monkeypatch.setenv("KUMIHO_RERANK_CROSS_ENCODER", "1")
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        return lambda q, t: []

    monkeypatch.setattr(rr_mod, "try_fastembed_reranker", _boom)
    explicit = lambda q, t: [0.0] * len(t)
    mgr = _make_manager(reranker=explicit)
    assert mgr.reranker is explicit
    assert called["n"] == 0  # env resolver never ran


# --------------------------------------------------------------------------
# Task 3 — retrieve-wide-then-trim
# --------------------------------------------------------------------------

def _pool():
    # Ascending scores so the strongest are LAST in server order — only
    # reranking (not input order) can float them into the trimmed top.
    return [
        {"title": f"m{i}", "summary": "", "score": 0.40 + 0.10 * i}
        for i in range(6)
    ]


def test_default_multiplier_fetches_exactly_limit():
    retrieve = _RecordingRetrieve(_pool())
    mgr = _make_manager(retrieve)  # recall_candidate_multiplier defaults to 1.0
    out = asyncio.run(mgr.recall_memories("q", limit=3))
    assert retrieve.last_limit == 3          # no over-fetch
    assert len(out) <= 3


def test_multiplier_widens_fetch_and_trims_output():
    retrieve = _RecordingRetrieve(_pool())
    mgr = _make_manager(retrieve, recall_candidate_multiplier=2.0)
    out = asyncio.run(mgr.recall_memories("q", limit=3))
    assert retrieve.last_limit == 6          # ceil(3 * 2.0)
    assert len(out) == 3                     # trimmed back to the caller's limit


def test_best_reranked_survive_the_trim():
    retrieve = _RecordingRetrieve(_pool())
    mgr = _make_manager(retrieve, recall_candidate_multiplier=2.0)
    out = asyncio.run(mgr.recall_memories("q", limit=3))
    titles = {m["title"] for m in out}
    # The three highest-scored candidates (m3/m4/m5) — last in server order —
    # survive the widen-then-trim because rerank front-loads them.
    assert titles == {"m3", "m4", "m5"}


def test_multiplier_below_one_is_clamped():
    retrieve = _RecordingRetrieve(_pool())
    mgr = _make_manager(retrieve, recall_candidate_multiplier=0.5)
    asyncio.run(mgr.recall_memories("q", limit=3))
    assert retrieve.last_limit == 3          # clamped to >= 1.0, no under-fetch


# --------------------------------------------------------------------------
# Task 3 (optional) — query_time reaches the event-proximity rerank stage
# --------------------------------------------------------------------------

QT = datetime(2023, 5, 8, tzinfo=timezone.utc)


def _temporal_retrieve():
    async def retrieve(**k):
        return [
            {"title": "far", "summary": "", "score": 0.85, "event_date": "2018-05-08"},
            {"title": "near", "summary": "", "score": 0.80, "event_date": "2023-05-08"},
        ]
    return retrieve


def _temporal_cfg():
    return RerankConfig(
        recency_enabled=False, mmr_enabled=False,
        event_proximity_enabled=True, event_proximity_half_life_days=30.0,
        event_proximity_max_boost=0.12,
    )


def test_query_time_activates_event_proximity():
    mgr = _make_manager(_temporal_retrieve(), rerank=_temporal_cfg())
    out = asyncio.run(mgr.recall_memories("q", limit=2, query_time=QT))
    # near (0.80 + 0.12) overtakes far (0.85) once the temporal prior fires.
    assert [m["title"] for m in out] == ["near", "far"]


def test_query_time_none_keeps_prior_dormant():
    mgr = _make_manager(_temporal_retrieve(), rerank=_temporal_cfg())
    out = asyncio.run(mgr.recall_memories("q", limit=2))  # query_time defaults None
    assert [m["title"] for m in out] == ["far", "near"]  # input order preserved
