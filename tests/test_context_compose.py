"""Tests for revision-centric context assembly (kumiho_memory.context_compose).

Semantics ported from the LoCoMo harness's ``build_recalled_context`` /
``_collect_top_revisions`` — these tests pin the parity contract: sibling
lists subsume the primary, global score ranking is conditional on a real
signal, and the top-k cap treats 0 as unlimited.
"""

import asyncio

import kumiho_memory.context_compose as cc
from kumiho_memory.context_compose import (
    CONTEXT_BUDGET_CHARS,
    DEFAULT_CONTEXT_TOP_K,
    DEFAULT_REVISION_CHAR_LIMIT,
    TRUNCATION_MARKER,
    approx_tokens,
    collect_top_revisions,
    compose_context,
    truncate_section,
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
    # A limit below the marker width degenerates to a bare hard slice (origin
    # behavior preserved, budget ceiling honored); the content-less memory
    # falls back to title+summary.
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


def test_fact_recall_evidence_rides_on_top_of_the_cap():
    # Fact-recall entries are additive on the same terms as bridge evidence:
    # their own +2 budget after the top-K head slice, never displacing it.
    mems = [_mem(f"m{i}", f"s{i}", score=1.0 - i * 0.1) for i in range(6)]
    facts = []
    for i in range(3):                                # 3 offered, budget keeps 2
        f = _mem(f"fact{i}", f"claim{i}", score=0.05)
        f["fact_recall"] = True
        facts.append(f)
    out = compose_context(mems + facts, top_k=5)
    blocks = out.split("\n\n")
    assert len(blocks) == 7                           # 5 base + 2 additive facts
    assert blocks[:5] == [f"m{i}: s{i}" for i in range(5)]   # base untouched
    assert blocks[5] == "fact0: claim0" and blocks[6] == "fact1: claim1"


def test_bridge_and_fact_budgets_are_independent():
    # A node that is both bridge and fact-recall counts once (as bridge);
    # each additive class keeps its own +2 budget on top of the head slice.
    mems = [_mem(f"m{i}", f"s{i}", score=1.0 - i * 0.1) for i in range(6)]
    bridge = _mem("bridge-fact", "joined evidence", score=0.05)
    bridge["bridge"] = True
    bridge["fact_recall"] = True                      # dual-flagged: bridge wins
    fact = _mem("fact0", "claim0", score=0.04)
    fact["fact_recall"] = True
    out = compose_context(mems + [bridge, fact], top_k=5)
    blocks = out.split("\n\n")
    assert len(blocks) == 7                           # 5 base + 1 bridge + 1 fact
    assert blocks[:5] == [f"m{i}: s{i}" for i in range(5)]
    assert "bridge-fact: joined evidence" in blocks[5:]
    assert "fact0: claim0" in blocks[5:]


def test_fact_budget_kwarg_mirrors_config():
    # fact_recall_max_results is documented as mirrored by the composer —
    # the manager passes it through as ``fact_budget``.
    mems = [_mem(f"m{i}", f"s{i}", score=1.0 - i * 0.1) for i in range(6)]
    facts = []
    for i in range(3):
        f = _mem(f"fact{i}", f"claim{i}", score=0.05)
        f["fact_recall"] = True
        facts.append(f)
    out = compose_context(mems + facts, top_k=5, fact_budget=3)
    blocks = out.split("\n\n")
    assert len(blocks) == 8                           # 5 base + all 3 facts
    assert blocks[5:] == [f"fact{i}: claim{i}" for i in range(3)]


# ---------------------------------------------------------------------------
# Contested marker (CONTRADICTS edges, threaded from graph_augmentation)
# ---------------------------------------------------------------------------

def test_contested_memory_gets_disputed_note():
    contested = _mem("c", "X is true", score=0.5)
    contested["contested_by"] = ["kref://other"]
    out = compose_context([contested])
    assert out == "c: X is true\n[contested: disputed by 1 other stored memory]"


def test_contested_note_pluralizes_and_is_bounded_by_marker():
    contested = _mem("c", "X is true", score=0.5)
    contested["contested_by"] = ["kref://a", "kref://b"]
    out = compose_context([contested])
    assert "[contested: disputed by 2 other stored memories]" in out


def test_no_contested_note_without_marker():
    # Regression: an ordinary memory renders exactly as before — the note is
    # strictly additive.
    out = compose_context([_mem("c", "X is true", score=0.5)])
    assert out == "c: X is true"
    assert "contested" not in out


def test_contested_note_in_full_mode():
    contested = _mem("c", "sum", score=0.5, content="the raw content")
    contested["contested_by"] = ["kref://other"]
    out = compose_context([contested], mode="full")
    assert out.startswith("the raw content")
    assert "[contested: disputed by 1 other stored memory]" in out


def test_contested_note_survives_sibling_branch():
    # A STACKED contested memory (sibling_revisions present) must not lose its
    # note: the item-level marker rides onto the rendered sibling blocks.
    contested = _mem("c", "X is true", score=0.5,
                     siblings=[_sib("c-r2", "X is true (rev 2)", score=0.9)])
    contested["contested_by"] = ["kref://other"]
    out = compose_context([contested])
    assert "c-r2: X is true (rev 2)" in out
    assert "[contested: disputed by 1 other stored memory]" in out


def test_grounding_note_survives_sibling_branch():
    stale = _mem("d", "keep Upstash", score=0.5,
                 siblings=[_sib("d-r2", "keep Upstash (rev 2)", score=0.9)])
    stale["grounding_stale"] = True
    out = compose_context([stale])
    assert "d-r2: keep Upstash (rev 2)" in out
    assert "[grounding stale: a fact this was based on was superseded]" in out


# ---------------------------------------------------------------------------
# Grounding-staleness note (#95) — additive, mirrors the contested note
# ---------------------------------------------------------------------------

def test_grounding_stale_memory_gets_note():
    stale = _mem("d", "keep Upstash", score=0.5)
    stale["grounding_stale"] = True
    out = compose_context([stale])
    assert out == "d: keep Upstash\n[grounding stale: a fact this was based on was superseded]"


def test_no_grounding_note_without_flag():
    # Strictly additive: an ordinary memory renders exactly as before.
    out = compose_context([_mem("d", "keep Upstash", score=0.5)])
    assert out == "d: keep Upstash"
    assert "grounding stale" not in out


def test_grounding_note_in_full_mode():
    stale = _mem("d", "sum", score=0.5, content="the raw decision")
    stale["grounding_stale"] = True
    out = compose_context([stale], mode="full")
    assert out.startswith("the raw decision")
    assert "[grounding stale: a fact this was based on was superseded]" in out


# ---------------------------------------------------------------------------
# Unified context budget (#106) — one source of truth + token approx + marker
# ---------------------------------------------------------------------------

def _make_full_manager():
    """A manager wired for context assembly (full recall mode, no backend)."""
    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from fakes import FakeRedis

    async def _store(**k):
        return {}

    async def _retrieve(**k):
        return []

    return UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://t"),
        memory_store=_store,
        memory_retrieve=_retrieve,
        recall_mode="full",
    )


# --- Single source of truth: env override -----------------------------------

def test_env_override_sets_budget(monkeypatch):
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS", "1234")
    assert cc._resolve_context_budget_chars() == 1234


def test_env_override_malformed_and_nonpositive_fall_back(monkeypatch):
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS", "not-an-int")
    assert cc._resolve_context_budget_chars() == cc._DEFAULT_CONTEXT_BUDGET_CHARS
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS", "-5")
    assert cc._resolve_context_budget_chars() == cc._DEFAULT_CONTEXT_BUDGET_CHARS
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_BUDGET_CHARS", "   ")
    assert cc._resolve_context_budget_chars() == cc._DEFAULT_CONTEXT_BUDGET_CHARS


def test_default_budget_matches_bench_path_historical_value():
    # Unified default = the value the LoCoMo (compose_context, full) path used
    # before unification — byte-identical for sections at or under the budget;
    # over-budget sections carry the in-budget marker (measured at the 0.19.0
    # full-10 RC gate, not claimed neutral).
    assert cc._DEFAULT_CONTEXT_BUDGET_CHARS == 8000
    assert DEFAULT_REVISION_CHAR_LIMIT == CONTEXT_BUDGET_CHARS


# --- Single source of truth: both assemblers honor the constant -------------

def test_both_assemblers_read_the_shared_budget_constant(monkeypatch):
    # Monkeypatch the one constant; BOTH assemblers must truncate at it.
    monkeypatch.setattr(cc, "CONTEXT_BUDGET_CHARS", 20)
    content = "Y" * 40
    # Budget-inclusive: cut to (limit - marker) then append → exactly 20 chars.
    expected = "Y" * (20 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER
    assert len(expected) == 20

    # Bench path: revision-centric compose_context (char_limit=None → constant).
    bench = compose_context([_mem("a", "sa", score=0.5, content=content)],
                            mode="full")
    assert bench == expected

    # MCP engage path: item-centric build_recalled_context.
    mgr = _make_full_manager()
    item = mgr.build_recalled_context([_mem("a", "sa", content=content)],
                                      recall_mode="full")
    assert item == expected


# --- Shared per-section truncation policy: marker parity ---------------------

def test_truncation_marker_parity_across_assemblers(monkeypatch):
    monkeypatch.setattr(cc, "CONTEXT_BUDGET_CHARS", 16)
    content = "Z" * 30
    bench = compose_context([_mem("a", "sa", score=0.5, content=content)],
                            mode="full")
    mgr = _make_full_manager()
    item = mgr.build_recalled_context([_mem("a", "sa", content=content)],
                                      recall_mode="full")
    # Identical marker, identical cut point → identical section output, and
    # the budget is a hard ceiling (marker included in the 16 chars).
    assert bench.endswith(TRUNCATION_MARKER)
    assert item.endswith(TRUNCATION_MARKER)
    assert bench == item == "Z" * (16 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER
    assert len(bench) == 16


def test_truncate_section_policy():
    m = TRUNCATION_MARKER
    assert truncate_section("short", 100) == "short"    # under budget: unchanged
    assert truncate_section("abcdef", 6) == "abcdef"    # exact fit: no marker
    # Truncation is budget-inclusive: marker fits INSIDE the limit, so the
    # result is exactly `limit` chars — the budget is a hard ceiling.
    out = truncate_section("a" * 30, 20)
    assert out == "a" * (20 - len(m)) + m
    assert len(out) == 20
    # Limit too small to fit the marker: bare hard slice, ceiling still holds.
    assert truncate_section("abcdef", 3) == "abc"
    # 0 passes through like the historical content[:0] slice (NOT unlimited).
    assert truncate_section("abcdef", 0) == ""


def test_truncate_section_none_limit_uses_constant(monkeypatch):
    monkeypatch.setattr(cc, "CONTEXT_BUDGET_CHARS", 14)
    out = truncate_section("x" * 30)
    assert out == "x" * (14 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER
    assert len(out) == 14


def test_char_limit_zero_empties_content_origin_semantics():
    # F3 regression: numeric char_limit passes straight through to slicing —
    # 0 empties the section exactly like origin's content[:0] (NOT unlimited).
    out = compose_context([_mem("a", "sa", score=0.5, content="RAW" * 10)],
                          mode="full", char_limit=0)
    assert out == ""


# --- Token approximation -----------------------------------------------------

def test_approx_tokens_is_chars_over_four():
    assert approx_tokens("") == 0
    assert approx_tokens("abcd") == 1
    assert approx_tokens("a" * 41) == 10
    # Applies to either assembler's rendered string output.
    bench = compose_context([_mem("a", "sa")])   # "a: sa"
    assert approx_tokens(bench) == len(bench) // 4


# --- Behavior guard: bench assembler byte-identical under budget -------------

def test_bench_assembler_byte_identical_under_budget():
    # Under-budget raw content is returned unchanged (no marker) — the LoCoMo
    # benchmark path (compose_context, recall_mode="full") is byte-identical
    # to pre-#106 for sections AT OR UNDER the budget.  Over-budget sections
    # carry the in-budget truncation marker — an intentional unification whose
    # bench impact is measured at the 0.19.0 full-10 RC gate.
    content = "hello world"          # 11 chars, far under the 8000 default
    out = compose_context([_mem("a", "sa", score=0.5, content=content)],
                          mode="full")
    assert out == content
    assert TRUNCATION_MARKER not in out
    # Same when the budget is passed explicitly.
    out2 = compose_context([_mem("a", "sa", score=0.5, content=content)],
                           mode="full", char_limit=CONTEXT_BUDGET_CHARS)
    assert out2 == content
