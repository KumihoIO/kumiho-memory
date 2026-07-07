"""Tests for the two-pass focused rerank (kumiho_memory.recall_rerank.two_pass_rerank).

Ported from the LoCoMo harness's ``rerank_memories`` with one deliberate
upgrade these tests pin: primaries and siblings are re-scored in the SAME
batch so downstream global ranking (compose_context) compares one cosine
scale instead of mixing server relevance with sibling cosines.
"""

from kumiho_memory.recall_rerank import two_pass_rerank


class FakeEmbeddingAdapter:
    """Deterministic embeddings: exact-text lookup, unknown text → [0, 1]."""

    def __init__(self, mapping=None, fail=False, short=False):
        self.mapping = mapping or {}
        self.fail = fail
        self.short = short
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding backend down")
        vecs = [self.mapping.get(t, [0.0, 1.0]) for t in texts]
        return vecs[:-1] if self.short else vecs


def _fixture():
    """One primary aligned with the query, one sibling aligned, one not."""
    mem = {
        "title": "prim", "summary": "sp", "score": 2.7, "base_score": 2.7,
        "sibling_revisions": [
            {"title": "hit", "summary": "sh", "_score": 0.1},
            {"title": "miss", "summary": "sm", "_score": 0.99},
        ],
    }
    adapter = FakeEmbeddingAdapter({
        "q": [1.0, 0.0],
        "prim: sp": [1.0, 0.0],   # cos 1.0
        "hit: sh": [1.0, 1.0],    # cos ~0.707
        "miss: sm": [0.0, 1.0],   # cos 0.0
    })
    return [mem], adapter


def test_rescores_primaries_and_siblings_on_one_scale():
    memories, adapter = _fixture()
    out = two_pass_rerank("q", memories, adapter)
    assert out is memories  # in-place, same list returned
    mem = out[0]
    assert mem["score"] == 1.0
    sib_scores = {s["title"]: s["_score"] for s in mem["sibling_revisions"]}
    assert abs(sib_scores["hit"] - 0.7071) < 1e-3
    assert sib_scores["miss"] == 0.0  # server's 0.99 replaced


def test_base_score_is_dropped_from_primaries():
    memories, adapter = _fixture()
    out = two_pass_rerank("q", memories, adapter)
    # The relevance basis was replaced — priors must not recompute from the
    # stale server score (mirrors apply_cross_encoder).
    assert "base_score" not in out[0]


def test_no_adapter_is_a_noop():
    memories, _ = _fixture()
    out = two_pass_rerank("q", memories, None)
    assert out is memories
    assert out[0]["score"] == 2.7
    assert out[0]["sibling_revisions"][1]["_score"] == 0.99


def test_empty_query_or_memories_is_a_noop():
    memories, adapter = _fixture()
    assert two_pass_rerank("", memories, adapter) is memories
    assert adapter.calls == 0
    assert two_pass_rerank("q", [], adapter) == []
    assert adapter.calls == 0


def test_embed_failure_keeps_original_scores():
    memories, _ = _fixture()
    out = two_pass_rerank("q", memories, FakeEmbeddingAdapter(fail=True))
    assert out[0]["score"] == 2.7
    assert out[0]["sibling_revisions"][0]["_score"] == 0.1


def test_vector_count_mismatch_keeps_original_scores():
    memories, _ = _fixture()
    adapter = FakeEmbeddingAdapter(short=True)
    out = two_pass_rerank("q", memories, adapter)
    assert out[0]["score"] == 2.7


def test_ordering_is_left_to_downstream():
    # two_pass_rerank replaces scores but never reorders — compose_context /
    # rerank() rank downstream.
    memories = [
        {"title": "a", "summary": "sa", "score": 0.1},
        {"title": "b", "summary": "sb", "score": 0.9},
    ]
    adapter = FakeEmbeddingAdapter({
        "q": [1.0, 0.0], "a: sa": [1.0, 0.0], "b: sb": [0.0, 1.0],
    })
    out = two_pass_rerank("q", memories, adapter)
    assert [m["title"] for m in out] == ["a", "b"]
    assert out[0]["score"] == 1.0 and out[1]["score"] == 0.0


def test_manager_rerank_memories_noop_without_embedding_adapter():
    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from fakes import FakeRedis

    async def _store(**k):
        return {}

    async def _retrieve(**k):
        return []

    mgr = UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://t"),
        memory_store=_store,
        memory_retrieve=_retrieve,
    )
    memories = [{"title": "a", "summary": "sa", "score": 0.3}]
    out = mgr.rerank_memories(memories, "q")
    assert out is memories and out[0]["score"] == 0.3
