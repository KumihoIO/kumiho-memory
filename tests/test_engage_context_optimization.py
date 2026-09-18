"""Judged delivery through ``kumiho_memory_engage``: the real engage path.

The manager is the strict SimpleNamespace the insight engage tests use (any
unexpected lookup, store or session call fails), and the SDK's ``evaluate`` is
monkeypatched, so nothing here touches a graph or a network.
"""

import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import kumiho
from kumiho_memory import context_optimization as ctxopt
from kumiho_memory import mcp_tools
from kumiho_memory.evidence_rank import EvidenceRankConfig
from kumiho_memory.memory_manager import UniversalMemoryManager

from test_context_optimization import Judgment, Result


QUERY = "what did we decide about deployment?"


def row(index, **overrides):
    result = {
        "kref": "kref://CognitiveMemory/decisions/item-%d?r=1" % index,
        "title": "memory %d" % index,
        "summary": "stored memory %d" % index,
        "type": "decision",
        "score": 0.9 - index / 100.0,
        "created_at": "2026-09-18T10:00:00Z",
    }
    result.update(overrides)
    return result


@pytest.fixture
def manager(monkeypatch):
    mgr = SimpleNamespace(
        recall_mode="summarized",
        sibling_similarity_threshold=0,
        evidence_rank_config=EvidenceRankConfig(),
        _last_backend_error=None,
    )
    mgr.rows = [row(index) for index in range(12)]
    mgr.recall_memories = AsyncMock(
        side_effect=lambda *a, **k: copy.deepcopy(mgr.rows),
    )
    mgr.build_recalled_context = Mock(side_effect=lambda *a, **k:
        UniversalMemoryManager.build_recalled_context(mgr, *a, **k))
    monkeypatch.setattr(mcp_tools, "_get_manager", lambda: mgr)
    mcp_tools._recall_recent.clear()
    ctxopt._backoff_until.clear()
    yield mgr
    mcp_tools._recall_recent.clear()
    ctxopt._backoff_until.clear()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_OPT_ENABLED", "1")


def judge(verdicts):
    """Install an ``evaluate`` that answers *verdicts* in fragment order."""
    calls = []

    def evaluate(query, fragments, questions, **kwargs):
        calls.append({
            "query": query, "fragments": fragments,
            "questions": questions, **kwargs,
        })
        return Result(fragments=[
            Judgment(fragment["id"], relevance=rel, evidence=ev)
            for fragment, (rel, ev) in zip(fragments, verdicts)
        ])

    evaluate.calls = calls
    return evaluate


def install(monkeypatch, evaluate):
    monkeypatch.setattr(kumiho, "evaluate", evaluate, raising=False)
    return evaluate


def engage(**overrides):
    return mcp_tools.tool_memory_engage({"query": QUERY, **overrides})


def recall_limit(manager):
    return manager.recall_memories.await_args.kwargs["limit"]


# ---------------------------------------------------------------------------
# Disabled — today's engage, byte for byte
# ---------------------------------------------------------------------------


def test_disabled_recalls_the_callers_limit_and_reports_nothing(manager, monkeypatch):
    install(monkeypatch, Mock(side_effect=AssertionError("must not evaluate")))
    result = engage(limit=3)

    assert recall_limit(manager) == 3
    assert "optimization" not in result
    assert result["count"] == 12


def test_disabled_response_is_unchanged(manager, monkeypatch):
    """The feature adds exactly one field, and only when it is configured on."""
    baseline = engage()
    mcp_tools._recall_recent.clear()
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_OPT_ENABLED", "0")
    again = engage()

    assert again == baseline


# ---------------------------------------------------------------------------
# Enabled — the judgment decides delivery
# ---------------------------------------------------------------------------


def test_enabled_widens_the_recall(manager, enabled, monkeypatch):
    install(monkeypatch, judge([(0.9, 0.9)] * 12))
    engage(limit=5)

    manager.recall_memories.assert_awaited_once_with(
        QUERY, limit=50, space_paths=None, memory_types=None,
        graph_augmented=False,
    )


def test_a_caller_limit_above_the_pool_still_wins(manager, enabled, monkeypatch):
    install(monkeypatch, judge([(0.9, 0.9)] * 12))
    engage(limit=64)

    assert recall_limit(manager) == 64


def test_delivery_is_the_kept_memories_only(manager, enabled, monkeypatch):
    verdicts = [(0.1, 0.1)] * 12
    verdicts[2] = (0.9, 0.9)
    verdicts[9] = (0.8, 0.7)
    install(monkeypatch, judge(verdicts))

    result = engage(limit=5)

    assert result["optimization"] == {"status": "applied", "candidates": 12}
    assert result["count"] == 2
    assert result["source_krefs"] == [
        "kref://CognitiveMemory/decisions/item-2?r=1",
        "kref://CognitiveMemory/decisions/item-9?r=1",
    ]
    assert [m["title"] for m in result["results"]] == ["memory 2", "memory 9"]
    assert "stored memory 2" in result["context"]
    assert "stored memory 0" not in result["context"]
    assert result["approx_tokens"] == len(result["context"]) // 4


def test_nothing_relevant_delivers_nothing(manager, enabled, monkeypatch):
    """A pool with nothing relevant is an empty answer, not an error."""
    install(monkeypatch, judge([(0.0, 0.0)] * 12))

    result = engage(limit=5)

    assert result["context"] == ""
    assert result["count"] == 0
    assert result["results"] == []
    assert result["source_krefs"] == []
    assert result["optimization"] == {"status": "applied", "candidates": 12}


def test_the_judged_request_carries_no_krefs(manager, enabled, monkeypatch):
    evaluate = install(monkeypatch, judge([(0.9, 0.9)] * 12))
    engage(limit=5)

    sent = evaluate.calls[0]
    assert sent["rubric_version"] == ctxopt.RUBRIC_VERSION
    assert [f["id"] for f in sent["fragments"]] == [
        "c%02d" % i for i in range(1, 13)
    ]
    assert "kref" not in repr(sent["fragments"])


def test_min_score_filters_before_the_judgment(manager, enabled, monkeypatch):
    manager.rows = [row(0, score=0.2), row(1, score=0.9)]
    evaluate = install(monkeypatch, judge([(0.9, 0.9)] * 2))

    result = engage(limit=5, min_score=0.7)

    assert [f["metadata"]["title"] for f in evaluate.calls[0]["fragments"]] == [
        "memory 1",
    ]
    assert result["optimization"]["candidates"] == 1
    assert result["count"] == 1


def test_insight_synthesis_sees_the_kept_memories(manager, enabled, monkeypatch):
    verdicts = [(0.1, 0.1)] * 12
    verdicts[4] = (0.9, 0.9)
    install(monkeypatch, judge(verdicts))

    result = engage(limit=5, include_insights=True)

    assert result["insight_brief"]["source_krefs"] == [
        "kref://CognitiveMemory/decisions/item-4?r=1",
    ]
    sources = result["synthesis_request"]["sources"]
    assert [s["kref"] for s in sources] == [
        "kref://CognitiveMemory/decisions/item-4?r=1",
    ]


# ---------------------------------------------------------------------------
# Fallback — today's delivery, whatever went wrong
# ---------------------------------------------------------------------------


def _fallback_result(monkeypatch, evaluate, **overrides):
    install(monkeypatch, evaluate)
    return engage(limit=5, **overrides)


@pytest.mark.parametrize("status", [
    "not_entitled", "over_limit", "provider_unavailable", "invalid_request",
])
def test_each_non_ok_status_delivers_the_caller_limit_prefix(
    manager, enabled, monkeypatch, status,
):
    result = _fallback_result(
        monkeypatch, lambda *a, **k: Result(status=status),
    )

    assert result["optimization"] == {
        "status": "fallback", "candidates": 12, "reason": status,
    }
    assert result["count"] == 5
    assert [m["title"] for m in result["results"]] == [
        "memory %d" % i for i in range(5)
    ]


def test_an_exception_delivers_the_caller_limit_prefix(manager, enabled, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("provider exploded")

    result = _fallback_result(monkeypatch, boom)

    assert result["optimization"]["status"] == "fallback"
    assert result["optimization"]["reason"] == "error"
    assert result["count"] == 5


def test_an_sdk_without_evaluate_delivers_the_caller_limit_prefix(
    manager, enabled, monkeypatch,
):
    monkeypatch.delattr(kumiho, "evaluate", raising=False)

    result = engage(limit=5)

    assert result["optimization"] == {
        "status": "fallback", "candidates": 12, "reason": "sdk_unavailable",
    }
    assert result["count"] == 5


def test_fallback_still_widened_the_recall(manager, enabled, monkeypatch):
    """The first failure pays for one wide recall; the back-off pays for none."""
    _fallback_result(monkeypatch, lambda *a, **k: Result(status="not_entitled"))
    assert recall_limit(manager) == 50


# ---------------------------------------------------------------------------
# Back-off — the engage path returns to exactly today's recall
# ---------------------------------------------------------------------------


def test_a_backed_off_scope_does_not_widen_and_does_not_evaluate(
    manager, enabled, monkeypatch,
):
    ctxopt._backoff_until[""] = time.monotonic() + 600.0
    install(monkeypatch, Mock(side_effect=AssertionError("must not evaluate")))

    result = engage(limit=3)

    assert recall_limit(manager) == 3
    assert result["optimization"] == {
        "status": "fallback", "candidates": 12, "reason": "backoff",
    }
    assert result["count"] == 3


def test_not_entitled_backs_the_scope_off_for_the_next_call(
    manager, enabled, monkeypatch,
):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(kwargs)
        return Result(status="not_entitled")

    install(monkeypatch, refuse)
    engage(limit=5)
    mcp_tools._recall_recent.clear()
    second = engage(limit=5)

    assert len(calls) == 1, "the second call must not ask again"
    assert recall_limit(manager) == 5, "and must not widen the recall"
    assert second["optimization"]["reason"] == "backoff"


def test_the_backoff_is_per_requesting_identity(manager, enabled, monkeypatch):
    """A hosted process serves many tenants; one refusal is not all of them."""
    from kumiho_memory._request_context import request_context
    from hosted_fakes import make_request_context

    entitled = judge([(0.9, 0.9)] * 12)

    def evaluate(query, fragments, questions, **kwargs):
        ctx = mcp_tools.current_request()
        if ctx is not None and ctx.tenant_id == "tenant-a":
            return Result(status="not_entitled")
        return entitled(query, fragments, questions, **kwargs)

    install(monkeypatch, evaluate)

    with request_context(make_request_context("tenant-a", session_id="s-1")):
        refused = engage(limit=5)
        mcp_tools._recall_recent.clear()
        again = engage(limit=5)
    with request_context(make_request_context("tenant-b", session_id="s-2")):
        mcp_tools._recall_recent.clear()
        other = engage(limit=5)

    assert refused["optimization"]["reason"] == "not_entitled"
    assert again["optimization"]["reason"] == "backoff"
    assert other["optimization"]["status"] == "applied"
    assert len(entitled.calls) == 1, "only tenant-b's judged request ran"


# ---------------------------------------------------------------------------
# tool_memory_recall is not part of this
# ---------------------------------------------------------------------------


def test_recall_is_untouched(manager, enabled, monkeypatch):
    install(monkeypatch, Mock(side_effect=AssertionError("must not evaluate")))

    result = mcp_tools.tool_memory_recall({"query": QUERY, "limit": 3})

    assert recall_limit(manager) == 3
    assert "optimization" not in result
    assert result["count"] == 12
