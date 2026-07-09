"""Tests for revision-centric context assembly (kumiho_memory.context_compose).

Semantics ported from the LoCoMo harness's ``build_recalled_context`` /
``_collect_top_revisions`` — these tests pin the parity contract: sibling
lists subsume the primary, global score ranking is conditional on a real
signal, and the top-k cap treats 0 as unlimited.
"""

import asyncio

from kumiho_memory.context_compose import (
    DEFAULT_CONTEXT_TOP_K,
    collect_top_revisions,
    compose_context,
)


def _mem(title, summary, score=0.0, siblings=None, content=""):
    m = {
        "kref": f"kref://{title}",
        "title": title,
        "summary": summary,
        "score": score,
        "content": content,
    }
    if siblings is not None:
        m["sibling_revisions"] = siblings
    return m


def _sib(title, summary, score=0.0, content=""):
    return {
        "kref": f"kref://{title}",
        "title": title,
        "summary": summary,
        "_score": score,
        "content": content,
    }


# ---------------------------------------------------------------------------
# compose_context
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_string():
    assert compose_context([]) == ""
    assert compose_context([_mem("a", "")]) == ""  # no summary, no content


def test_summarized_mode_renders_title_colon_summary():
    out = compose_context([_mem("a", "sa"), _mem("b", "sb")])
    assert out == "a: sa\n\nb: sb"


def test_summary_without_title_renders_bare():
    out = compose_context([_mem("", "just a summary")])
    assert out == "just a summary"


def test_siblings_subsume_the_primary():
    # The primary shell must NOT appear when siblings exist — the sibling
    # list contains all revisions of the item, including the published one.
    mem = _mem("shell", "shell summary", score=9.9,
               siblings=[_sib("rev1", "s1", 0.5), _sib("rev2", "s2", 0.9)])
    out = compose_context([mem])
    assert "shell summary" not in out
    assert out == "rev2: s2\n\nrev1: s1"  # sibling _score ordering


def test_global_ranking_across_items():
    # Revisions compete globally, regardless of which item they hang off.
    m1 = _mem("i1", "", siblings=[_sib("low", "sl", 0.2), _sib("high", "sh", 0.95)])
    m2 = _mem("solo", "ss", score=0.5)
    out = compose_context([m1, m2])
    assert out.split("\n\n") == ["high: sh", "solo: ss", "low: sl"]


def test_unscored_input_preserves_caller_order():
    # No real score signal anywhere → server relevance order is kept.
    out = compose_context([_mem("first", "s1"), _mem("second", "s2")])
    assert out == "first: s1\n\nsecond: s2"


def test_top_k_caps_the_pool_and_zero_means_unlimited():
    mems = [_mem(f"m{i}", f"s{i}", score=1.0 - i * 0.1) for i in range(6)]
    capped = compose_context(mems, top_k=2)
    assert capped == "m0: s0\n\nm1: s1"
    unlimited = compose_context(mems, top_k=0)
    assert len(unlimited.split("\n\n")) == 6


def test_bridge_evidence_rides_on_top_of_the_cap():
    # Entity-bridge join evidence is ADDITIVE: it must never displace the
    # top-K base revisions (measured: displacement cost open-domain −0.107
    # on conv-26), and it survives the cut even when its score trails.
    mems = [_mem(f"m{i}", f"s{i}", score=1.0 - i * 0.1) for i in range(6)]
    bridge = _mem("bridge-fact", "joined evidence", score=0.05)
    bridge["bridge"] = True
    out = compose_context(mems + [bridge], top_k=5)
    blocks = out.split("\n\n")
    assert len(blocks) == 6                      # 5 base + 1 additive bridge
    assert blocks[:5] == [f"m{i}: s{i}" for i in range(5)]  # base untouched
    assert blocks[5] == "bridge-fact: joined evidence"
    # Without bridges the historical head-slice is byte-identical.
    assert compose_context(mems, top_k=5).split("\n\n") == [
        f"m{i}: s{i}" for i in range(5)
    ]


def test_top_k_none_uses_module_default():
    n = DEFAULT_CONTEXT_TOP_K + 5
    mems = [_mem(f"m{i}", f"s{i}", score=1.0) for i in range(n)]
    out = compose_context(mems)  # top_k=None
    assert len(out.split("\n\n")) == DEFAULT_CONTEXT_TOP_K


def test_full_mode_uses_content_with_char_limit_and_summary_fallback():
    with_content = _mem("a", "sa", content="X" * 50)
    without_content = _mem("b", "sb")
    out = compose_context([with_content, without_content],
                          mode="full", char_limit=10)
    assert out == "X" * 10 + "\n\nb: sb"


def test_summarized_mode_ignores_content():
    out = compose_context([_mem("a", "sa", content="RAW")], mode="summarized")
    assert out == "a: sa"


def test_bool_scores_do_not_count_as_signal():
    # A bool score must not trigger the sort (parity with recall_rerank's
    # score coercion rules).
    out = compose_context([_mem("first", "s1", score=True),
                           _mem("second", "s2", score=False)])
    assert out == "first: s1\n\nsecond: s2"


# ---------------------------------------------------------------------------
# collect_top_revisions
# ---------------------------------------------------------------------------

def test_collect_flattens_and_skips_primary_shell():
    mem = _mem("shell", "ss", score=9.9,
               siblings=[_sib("r1", "s1", 0.3), _sib("r2", "s2", 0.8)])
    top = collect_top_revisions([mem], limit=10)
    krefs = [c["kref"] for c in top]
    assert krefs == ["kref://r2", "kref://r1"]  # score-desc, no shell


def test_collect_siblingless_memory_contributes_itself():
    top = collect_top_revisions([_mem("solo", "ss", score=0.7)], limit=10)
    assert top == [{
        "kref": "kref://solo", "title": "solo", "summary": "ss", "_score": 0.7,
    }]


def test_collect_respects_limit_globally():
    m1 = _mem("i1", "", siblings=[_sib("a", "", 0.9), _sib("b", "", 0.1)])
    m2 = _mem("c", "sc", score=0.5)
    top = collect_top_revisions([m1, m2], limit=2)
    assert [c["kref"] for c in top] == ["kref://a", "kref://c"]


# ---------------------------------------------------------------------------
# Manager convenience method
# ---------------------------------------------------------------------------

def test_manager_compose_context_defaults_to_recall_mode():
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
        recall_mode="full",
    )
    mems = [_mem("a", "sa", content="RAWCONTENT")]
    # recall_mode="full" flows through as the default mode.
    assert mgr.compose_context(mems) == "RAWCONTENT"
    # Explicit mode overrides the manager's recall_mode.
    assert mgr.compose_context(mems, mode="summarized") == "a: sa"
