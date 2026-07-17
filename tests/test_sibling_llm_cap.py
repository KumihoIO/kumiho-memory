"""#102 — bounded + parallelized sibling-enrichment LLM calls.

``_enrich_with_siblings`` used to loop recalled memories serially, issuing one
LLM sibling-rerank round-trip per stacked item. These tests pin the two fixes:

1. A per-recall LLM-call cap (``sibling_llm_cap`` / ``KUMIHO_MEMORY_SIBLING_LLM_CAP``):
   items beyond the cap take the EXISTING deterministic fallback instead of the
   LLM — the LLM is simply not consulted for them.
2. ``asyncio.gather`` with bounded concurrency: per-item selection is
   independent, so gathering changes nothing semantically. Results are written
   back by identity/index, never by completion order.

No server: the manager is built with stub store/retrieve, and the sibling
fetch is monkeypatched with an in-process fake.
"""

import asyncio

from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _RecordingAdapter:
    """LLM adapter whose ``chat`` records how many times it was called."""

    def __init__(self, reply="none"):
        self.reply = reply
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return self.reply


class _FakeSummarizer:
    def __init__(self, adapter):
        self.adapter = adapter
        self.light_model = "light"


def _manager(summarizer=None):
    async def _retrieve(**kwargs):
        return {}

    return UniversalMemoryManager(
        project="p",
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test"),
        summarizer=summarizer or _FakeSummarizer(_RecordingAdapter()),
        memory_store=None,
        memory_retrieve=_retrieve,
    )


def _siblings():
    return [
        {"kref": "kref://p/i/1.conversation?r=1", "title": "tea", "summary": "user likes green tea", "created_at": "2026-01-01"},
        {"kref": "kref://p/i/1.conversation?r=2", "title": "coffee", "summary": "user switched to coffee", "created_at": "2026-01-02"},
    ]


# ---------------------------------------------------------------------------
# (1) The cap routes over-budget items to the EXISTING deterministic fallback
# ---------------------------------------------------------------------------

def test_allow_llm_rerank_false_skips_llm_and_uses_deterministic_fallback():
    adapter = _RecordingAdapter(reply="1")  # would pick sibling #1 if consulted
    mgr = _manager(_FakeSummarizer(adapter))
    query = "what does the user drink"

    # Expected fallback result computed independently (primary_score=0 and empty
    # current_rev_kref so the primary-score floor block is inert).
    expected = mgr._rank_siblings_deterministic(list(_siblings()), query, "")

    out = asyncio.run(
        mgr._filter_siblings_by_server_search(
            _siblings(), query, "kref://p/i/1", "",
            allow_llm_rerank=False,
        )
    )

    assert adapter.calls == 0, "LLM must not be consulted beyond the cap"
    assert [(s["kref"], s["_score"]) for s in out] == [
        (s["kref"], s["_score"]) for s in expected
    ]


def test_allow_llm_rerank_true_consults_llm():
    # Same inputs, cap NOT hit: the LLM IS consulted (default behavior). With a
    # "none" reply it still lands on the identical deterministic fallback, so the
    # only observable difference from the capped path is the round-trip itself.
    adapter = _RecordingAdapter(reply="none")
    mgr = _manager(_FakeSummarizer(adapter))
    query = "what does the user drink"
    expected = mgr._rank_siblings_deterministic(list(_siblings()), query, "")

    out = asyncio.run(
        mgr._filter_siblings_by_server_search(
            _siblings(), query, "kref://p/i/1", "",
            allow_llm_rerank=True,
        )
    )

    assert adapter.calls == 1, "LLM must be consulted within the cap"
    assert [(s["kref"], s["_score"]) for s in out] == [
        (s["kref"], s["_score"]) for s in expected
    ]


# ---------------------------------------------------------------------------
# (2) The cap is enforced per recall, in deterministic list order
# ---------------------------------------------------------------------------

def _memories(n):
    return [
        {"kref": f"kref://p/m{i}?r=1", "score": 0.5, "_item_kref": f"item{i}"}
        for i in range(n)
    ]


def _patch_fetch_record_flags(mgr, flags):
    async def _fake_fetch(item_kref, current_rev_kref, query="", load_artifacts=True,
                          alt_queries=None, primary_score=0.0, allow_llm_rerank=True):
        flags[item_kref] = allow_llm_rerank
        return []
    mgr._fetch_sibling_revision_summaries = _fake_fetch


def test_cap_enforced_items_beyond_cap_get_deterministic_fallback():
    mgr = _manager()
    mgr.sibling_llm_cap = 2
    flags = {}
    _patch_fetch_record_flags(mgr, flags)

    asyncio.run(mgr._enrich_with_siblings(_memories(4), "q"))

    assert flags == {
        "item0": True, "item1": True,   # within cap -> LLM allowed
        "item2": False, "item3": False,  # beyond cap -> deterministic fallback
    }


def test_cap_zero_means_unlimited():
    mgr = _manager()
    mgr.sibling_llm_cap = 0  # <= 0 disables the cap (sibling_top_k=0 idiom)
    flags = {}
    _patch_fetch_record_flags(mgr, flags)

    asyncio.run(mgr._enrich_with_siblings(_memories(5), "q"))

    assert all(flags[f"item{i}"] is True for i in range(5))


def test_default_cap_does_not_trip_for_typical_recall():
    mgr = _manager()  # default cap = 16
    flags = {}
    _patch_fetch_record_flags(mgr, flags)

    asyncio.run(mgr._enrich_with_siblings(_memories(10), "q"))

    assert all(flags[f"item{i}"] is True for i in range(10))


def test_env_override_sets_cap(monkeypatch):
    monkeypatch.setenv("KUMIHO_MEMORY_SIBLING_LLM_CAP", "1")
    mgr = _manager()
    assert mgr.sibling_llm_cap == 1
    flags = {}
    _patch_fetch_record_flags(mgr, flags)
    asyncio.run(mgr._enrich_with_siblings(_memories(3), "q"))
    assert flags == {"item0": True, "item1": False, "item2": False}


# ---------------------------------------------------------------------------
# (3) gather preserves index/identity assignment despite shuffled completions
# ---------------------------------------------------------------------------

def test_gather_assigns_results_by_identity_not_completion_order():
    mgr = _manager()
    mems = _memories(4)
    item_krefs = [m["_item_kref"] for m in mems]  # captured before pop

    async def _fake_fetch(item_kref, current_rev_kref, query="", load_artifacts=True,
                          alt_queries=None, primary_score=0.0, allow_llm_rerank=True):
        # Earlier items sleep LONGER, so completion order is the reverse of
        # submission order — the assignment must still track identity.
        idx = int(item_kref.replace("item", ""))
        await asyncio.sleep((4 - idx) * 0.02)
        return [{"kref": f"sib-of-{item_kref}"}]

    mgr._fetch_sibling_revision_summaries = _fake_fetch

    out = asyncio.run(mgr._enrich_with_siblings(mems, "q"))

    for mem, item_kref in zip(out, item_krefs):
        assert mem["sibling_revisions"] == [{"kref": f"sib-of-{item_kref}"}]


# ---------------------------------------------------------------------------
# (4) gather output equals a serial reference on a fixed fake
# ---------------------------------------------------------------------------

def test_gather_equals_serial_reference():
    query = "q"

    async def _fake_fetch(item_kref, current_rev_kref, query="", load_artifacts=True,
                          alt_queries=None, primary_score=0.0, allow_llm_rerank=True):
        idx = int(item_kref.replace("item", ""))
        # Shuffled completion delays: interleave the finish order.
        await asyncio.sleep(((idx * 7) % 5) * 0.01)
        return [{"kref": f"sib-{item_kref}", "_score": 1.0 - idx * 0.1}]

    # --- gather path (production _enrich_with_siblings) ---
    mgr = _manager()
    mgr._fetch_sibling_revision_summaries = _fake_fetch
    gathered = asyncio.run(mgr._enrich_with_siblings(_memories(6), query))
    gathered_view = [(m["kref"], m.get("sibling_revisions")) for m in gathered]

    # --- serial reference: same fake, one item at a time, in list order ---
    async def _serial():
        mems = _memories(6)
        for mem in mems:
            item_kref = mem.pop("_item_kref", "")
            if not item_kref:
                continue
            sibs = await _fake_fetch(item_kref, mem.get("kref", ""), query=query)
            if sibs:
                mem["sibling_revisions"] = sibs
        return mems

    serial = asyncio.run(_serial())
    serial_view = [(m["kref"], m.get("sibling_revisions")) for m in serial]

    assert gathered_view == serial_view


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
