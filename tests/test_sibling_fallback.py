"""Tests for hybrid sibling selection: LLM reranker primary + deterministic
fallback (the LoCoMo single-hop / temporal regression fix).

The reranker must stay the primary signal (it resolves semantic inversion),
but when it returns None / errors / is unavailable the manager must fall back
to a deterministic in-process ranking instead of dropping every sibling.
"""

import asyncio

from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class _RerankAdapter:
    """Async chat adapter returning a canned response (or raising)."""

    def __init__(self, response=None, raises=False):
        self.response = response
        self.raises = raises
        self.calls = 0

    async def chat(self, *, messages, model, system="", max_tokens=1024, **kw):
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm down")
        return self.response


class _SummarizerWithAdapter:
    """Minimal summarizer exposing an ``adapter`` + ``light_model``."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.light_model = "light-model"


class _SummarizerNoAdapter:
    """Summarizer with NO adapter attribute — the LLM path is unavailable."""

    light_model = "light-model"


class _FakeEmbedder:
    """Deterministic embedder: relevant texts map to [1,0], others to [0,1]."""

    def embed(self, texts):
        out = []
        for t in texts:
            out.append([1.0, 0.0] if "swim" in t.lower() else [0.0, 1.0])
        return out


def _make_manager(summarizer, **kw):
    buffer = RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test")

    async def _store(**k):
        return {"item_kref": "kref://x"}

    async def _retrieve(**k):
        return []

    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=summarizer,
        memory_store=_store,
        memory_retrieve=_retrieve,
        **kw,
    )


def _sib(kref, title, summary, created_at="2023-01-01"):
    text = f"{title} {summary}"
    return {
        "kref": kref,
        "title": title,
        "summary": summary,
        "created_at": created_at,
        "_chars": len(text),
    }


# Strong lexical match to "swimming pool"; the rest are unrelated.
def _strong():
    return _sib("a", "Pool day", "went swimming at the pool with the kids")


def _weak(kref="b"):
    return _sib(kref, "Cooking", "made pasta for dinner tonight")


QUERY = "swimming pool"


# --------------------------------------------------------------------------
# LLM primary — success paths (behavior preserved)
# --------------------------------------------------------------------------

def test_llm_picks_returned_unchanged_when_no_strong_deterministic():
    # Query with no lexical overlap → union no-ops → LLM pick returned as-is.
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="1")))
    sibs = [_weak("b"), _weak("c")]
    out = asyncio.run(
        mgr._filter_siblings_by_server_search(sibs, "philosophy of mind", "item")
    )
    assert len(out) == 1
    assert out[0]["kref"] == "b"
    assert out[0]["_score"] == 1.0


def test_llm_union_recovers_strong_match_it_missed():
    # LLM picks the weak sibling (index 2); the strong lexical match (index 1)
    # is appended by the single capped union.
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="2")))
    sibs = [_strong(), _weak("b")]
    out = asyncio.run(mgr._filter_siblings_by_server_search(sibs, QUERY, "item"))
    krefs = [s["kref"] for s in out]
    assert krefs == ["b", "a"]  # LLM pick first, unioned strong match appended


def test_llm_union_noops_without_strong_signal():
    # No strong lexical sibling → union adds nothing, LLM selection intact.
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="1")))
    sibs = [_weak("b"), _weak("c")]
    out = asyncio.run(mgr._filter_siblings_by_server_search(sibs, QUERY, "item"))
    assert [s["kref"] for s in out] == ["b"]


# --------------------------------------------------------------------------
# Deterministic fallback — the regression fix (None / raise / unavailable)
# --------------------------------------------------------------------------

def test_llm_none_falls_back_not_empty():
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="none")))
    sibs = [_strong(), _weak("b")]
    out = asyncio.run(mgr._filter_siblings_by_server_search(sibs, QUERY, "item"))
    assert out, "fallback must not return [] when siblings are present"
    assert out[0]["kref"] == "a"  # strong lexical match ranked first


def test_llm_raise_falls_back_not_empty():
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(raises=True)))
    sibs = [_strong(), _weak("b")]
    out = asyncio.run(mgr._filter_siblings_by_server_search(sibs, QUERY, "item"))
    assert [s["kref"] for s in out] == ["a"]


def test_llm_unavailable_falls_back_not_empty():
    mgr = _make_manager(_SummarizerNoAdapter())
    sibs = [_strong(), _weak("b")]
    out = asyncio.run(mgr._filter_siblings_by_server_search(sibs, QUERY, "item"))
    assert out and out[0]["kref"] == "a"


def test_weak_signal_fallback_keeps_all_within_budget():
    # No lexical signal at all — fallback must keep siblings (budget mode),
    # never blank them.  This is the core anti-regression property.
    mgr = _make_manager(_SummarizerNoAdapter())
    sibs = [_weak("b"), _weak("c")]
    out = asyncio.run(
        mgr._filter_siblings_by_server_search(sibs, "unrelated topic", "item")
    )
    assert {s["kref"] for s in out} == {"b", "c"}


# --------------------------------------------------------------------------
# Cross-repo contract: the primary revision must survive a non-empty result
# --------------------------------------------------------------------------

def test_fallback_retains_primary_revision():
    # Strong match is a sibling; the primary (kref "p") is unrelated and would
    # be dropped by strong-keyword mode.  It must be re-appended so downstream
    # context assembly (which skips the primary item entry) keeps its content.
    mgr = _make_manager(_SummarizerNoAdapter())
    primary = _weak("p")
    sibs = [_strong(), primary]
    out = asyncio.run(
        mgr._filter_siblings_by_server_search(sibs, QUERY, "item", "p")
    )
    krefs = {s["kref"] for s in out}
    assert "a" in krefs and "p" in krefs


# --------------------------------------------------------------------------
# Fallback parity: the published revision competes at the item's recall score
# (the multi-hop regression — a weak-scored fallback list silently DEMOTED
# the whole item because downstream context assembly skips the primary item
# entry whenever siblings are non-empty)
# --------------------------------------------------------------------------

def test_fallback_floors_primary_at_item_recall_score():
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="none")))
    primary = _weak("p")  # near-zero keyword overlap with QUERY
    sibs = [_weak("b"), primary]
    out = asyncio.run(
        mgr._filter_siblings_by_server_search(
            sibs, QUERY, "item", "p", primary_score=0.83,
        )
    )
    by_kref = {s["kref"]: s for s in out}
    assert by_kref["p"]["_score"] == 0.83  # floored at the item's score
    assert out[0]["kref"] == "p"           # and re-ranked to the front


def test_fallback_floor_keeps_higher_deterministic_score():
    # When the primary already outsores the floor, it keeps its own score.
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="none")))
    primary = _strong()  # strong lexical match, kref "a"
    out = asyncio.run(
        mgr._filter_siblings_by_server_search(
            [primary, _weak("b")], QUERY, "item", "a", primary_score=0.01,
        )
    )
    assert out[0]["kref"] == "a"
    assert out[0]["_score"] > 0.01  # own keyword score, not the tiny floor


def test_fallback_no_floor_when_primary_unknown_or_scoreless():
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="none")))
    sibs = [_weak("b"), _weak("c")]
    out = asyncio.run(
        mgr._filter_siblings_by_server_search(sibs, QUERY, "item")
    )  # no current_rev_kref, no primary_score
    assert all(s.get("_score", 0.0) < 0.5 for s in out)


def test_llm_success_never_floored():
    # LoCoMo-Plus protection: on LLM success the selection (and its scores)
    # are untouched — the floor applies only to the deterministic fallback.
    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="2")))
    primary = _weak("p")
    sibs = [_weak("b"), primary]
    out = asyncio.run(
        mgr._filter_siblings_by_server_search(
            sibs, "philosophy of mind", "item", "p", primary_score=0.9,
        )
    )
    assert [s["kref"] for s in out] == ["p"]
    assert out[0]["_score"] == 1.0  # LLM-assigned, not the 0.9 floor


def test_fallback_end_to_end_multihop_item_survives_context():
    # Regression scenario in miniature: item A's siblings carry LLM-style
    # scores (1.0/0.9); item B's LLM said "none" so it fell back.  B's
    # primary must still reach the composed context — pre-fix its near-zero
    # keyword _score buried it and the hop-2 fact vanished.
    from kumiho_memory.context_compose import compose_context

    mgr = _make_manager(_SummarizerWithAdapter(_RerankAdapter(response="none")))
    b_primary = _sib("bp", "Trip", "visited museum in Paris with sister")
    b_sibs = asyncio.run(
        mgr._filter_siblings_by_server_search(
            [b_primary, _weak("b2")], QUERY, "item-b", "bp", primary_score=0.8,
        )
    )
    memories = [
        {"kref": "a", "title": "A", "summary": "sa", "score": 0.7,
         "sibling_revisions": [
             {"kref": "a1", "title": "A1", "summary": "swim class", "_score": 1.0},
             {"kref": "a2", "title": "A2", "summary": "pool party", "_score": 0.9},
         ]},
        {"kref": "b", "title": "B", "summary": "sb", "score": 0.8,
         "sibling_revisions": b_sibs},
    ]
    ctx = compose_context(memories, QUERY, top_k=3)
    assert "visited museum in Paris" in ctx  # hop-2 evidence present


# --------------------------------------------------------------------------
# Multi-angle sibling selection (reformulated queries reach the selectors)
# --------------------------------------------------------------------------

def test_deterministic_scores_max_over_angles():
    # The sibling matches only the ALT query — with angles it lands in the
    # strong branch; without them it would be weak-signal budget mode.
    mgr = _make_manager(_SummarizerNoAdapter())
    hop2 = _sib("h2", "Museum", "visited museum exhibits in Paris")
    out = mgr._rank_siblings_deterministic(
        [hop2, _weak("b")], "swimming pool",
        alt_queries=["museum exhibits Paris"],
    )
    assert out[0]["kref"] == "h2"
    assert out[0]["_score"] >= mgr.sibling_strong_score


class _PromptCapturingAdapter(_RerankAdapter):
    def __init__(self, response="none"):
        super().__init__(response=response)
        self.last_user_msg = ""

    async def chat(self, *, messages, model, system="", max_tokens=1024, **kw):
        self.last_user_msg = messages[0]["content"]
        return await super().chat(
            messages=messages, model=model, system=system,
            max_tokens=max_tokens, **kw,
        )


def test_llm_prompt_includes_angles():
    adapter = _PromptCapturingAdapter(response="1")
    mgr = _make_manager(_SummarizerWithAdapter(adapter))
    asyncio.run(
        mgr._filter_siblings_by_server_search(
            [_weak("b")], QUERY, "item",
            alt_queries=["angle one", "angle two"],
        )
    )
    assert "angle one" in adapter.last_user_msg
    assert "angle two" in adapter.last_user_msg


def test_llm_prompt_no_angle_block_without_alts():
    adapter = _PromptCapturingAdapter(response="1")
    mgr = _make_manager(_SummarizerWithAdapter(adapter))
    asyncio.run(
        mgr._filter_siblings_by_server_search([_weak("b")], QUERY, "item")
    )
    assert "angles" not in adapter.last_user_msg


# --------------------------------------------------------------------------
# No-op guards
# --------------------------------------------------------------------------

def test_empty_siblings_noop():
    mgr = _make_manager(_SummarizerNoAdapter())
    assert asyncio.run(mgr._filter_siblings_by_server_search([], QUERY, "item")) == []


def test_empty_query_noop_returns_input():
    mgr = _make_manager(_SummarizerNoAdapter())
    sibs = [_strong()]
    out = asyncio.run(mgr._filter_siblings_by_server_search(sibs, "", "item"))
    assert out == sibs


# --------------------------------------------------------------------------
# Deterministic ranker directly
# --------------------------------------------------------------------------

def test_rank_keyword_orders_by_score():
    mgr = _make_manager(_SummarizerNoAdapter())
    strong = _sib("a", "Pool", "swimming pool swimming pool")  # heavy overlap
    lighter = _sib("b", "Pool", "swimming pool")               # lighter, still strong
    out = mgr._rank_siblings_deterministic([lighter, strong], QUERY)
    assert [s["kref"] for s in out] == ["a", "b"]
    assert out[0]["_score"] > out[1]["_score"]


def test_rank_top_k_cap():
    mgr = _make_manager(_SummarizerNoAdapter(), sibling_top_k=1)
    sibs = [_sib("a", "Pool", "swimming pool"), _sib("b", "Swim", "swimming pool")]
    out = mgr._rank_siblings_deterministic(sibs, QUERY)
    assert len(out) == 1


def test_rank_embedding_mode_used_when_adapter_present():
    mgr = _make_manager(
        _SummarizerNoAdapter(),
        embedding_adapter=_FakeEmbedder(),
        sibling_similarity_threshold=0.3,
    )
    strong = _sib("a", "Pool", "went swimming at the pool")  # embeds to [1,0]
    weak = _sib("b", "Cooking", "made pasta")                # embeds to [0,1]
    out = mgr._rank_siblings_deterministic([strong, weak], QUERY)
    assert [s["kref"] for s in out] == ["a"]  # only the above-threshold sibling
