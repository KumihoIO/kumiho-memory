"""Every explicit recall observes current data, even immediately after another."""

import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kumiho_memory import mcp_tools
from kumiho_memory.evidence_rank import EvidenceRankConfig
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.recall_timing import timed


@pytest.fixture
def manager(monkeypatch):
    mgr = SimpleNamespace(
        recall_mode="summarized", sibling_similarity_threshold=0,
        evidence_rank_config=EvidenceRankConfig(), _last_backend_error=None,
        rows=[{"kref": "kref://memory/preferences/example.fact?r=1",
               "title": "Preference", "summary": "Initial preference", "score": 0.9}],
    )

    @timed("search")
    async def recall(*args, **kwargs):
        return copy.deepcopy(mgr.rows[:kwargs["limit"]])

    mgr.recall_memories = AsyncMock(side_effect=recall)
    mgr.build_recalled_context = lambda *a, **k: UniversalMemoryManager.build_recalled_context(mgr, *a, **k)
    monkeypatch.setattr(mcp_tools, "_get_manager", lambda: mgr)
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_OPT_ENABLED", "0")
    return mgr


@pytest.mark.parametrize("tool", [mcp_tools.tool_memory_engage, mcp_tools.tool_memory_recall])
def test_three_immediate_identical_calls_all_retrieve(manager, tool):
    responses = [tool({"query": "preference"}) for _ in range(3)]
    assert manager.recall_memories.await_count == 3
    for result in responses:
        assert result["count"] == 1
        assert result.get("deduplicated") is not True
        if tool is mcp_tools.tool_memory_engage:
            assert result["context"]
            assert "search" in result["timing_ms"]


def test_immediate_recall_observes_correction_and_deletion(manager):
    before = mcp_tools.tool_memory_engage({"query": "preference"})
    manager.rows[0].update(kref="kref://memory/preferences/example.fact?r=2", summary="Corrected preference")
    corrected = mcp_tools.tool_memory_engage({"query": "preference"})
    manager.rows.clear()
    deleted = mcp_tools.tool_memory_engage({"query": "preference"})
    assert "Initial preference" in before["context"]
    assert "Corrected preference" in corrected["context"]
    assert corrected["source_krefs"] == ["kref://memory/preferences/example.fact?r=2"]
    assert deleted["count"] == 0
    assert deleted["context"] == ""
    assert "search" in deleted["timing_ms"]
    assert manager.recall_memories.await_count == 3


def test_immediate_retry_after_backend_failure_retrieves(manager):
    rows = manager.rows
    manager.rows = []
    manager._last_backend_error = "temporary retrieval failure"
    failed = mcp_tools.tool_memory_engage({"query": "preference"})
    manager.rows = rows
    manager._last_backend_error = None
    recovered = mcp_tools.tool_memory_engage({"query": "preference"})
    assert failed["backend_error"] == "temporary retrieval failure"
    assert recovered["count"] == 1
    assert "backend_error" not in recovered
    assert manager.recall_memories.await_count == 2


def test_changed_limit_and_score_are_honored_immediately(manager):
    manager.rows.append({"kref": "kref://memory/preferences/other.fact?r=1",
                         "title": "Other", "summary": "Other preference", "score": 0.7})
    first = mcp_tools.tool_memory_engage({"query": "preference", "limit": 1})
    wider = mcp_tools.tool_memory_engage({"query": "preference", "limit": 2})
    filtered = mcp_tools.tool_memory_engage({"query": "preference", "limit": 2, "min_score": 0.8})
    assert [r["count"] for r in (first, wider, filtered)] == [1, 2, 1]
    assert manager.recall_memories.await_count == 3


def test_concurrent_identical_calls_each_return_results(manager):
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: mcp_tools.tool_memory_engage({"query": "preference"}), range(3)))
    assert [r["count"] for r in results] == [1, 1, 1]
    assert manager.recall_memories.await_count == 3
