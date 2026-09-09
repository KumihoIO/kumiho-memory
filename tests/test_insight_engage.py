"""Opt-in insight integration: the real context assembler, one scoped recall."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kumiho_memory import mcp_tools
from kumiho_memory._request_context import current_request, request_context
from kumiho_memory.evidence_rank import EvidenceRankConfig
from kumiho_memory.memory_manager import UniversalMemoryManager
from hosted_fakes import make_request_context


KREF = "kref://CognitiveMemory/decisions/deploy.decision?r=1"
FACT = "kref://CognitiveMemory/facts/team.fact?r=2"


def decision(**overrides):
    result = {
        "kref": KREF,
        "title": "Deployment choice",
        "summary": "Use managed hosting because the team has no operator.",
        "type": "decision",
        "score": 0.9,
        "evidence_level": "single_source",
        "grounding_stale": True,
        "superseded_by": FACT,
    }
    result.update(overrides)
    return result


@pytest.fixture
def manager(monkeypatch):
    # Only the two existing engage operations are available. Any accidental
    # extra lookup, store, provider or session call fails on the strict surface.
    mgr = SimpleNamespace(
        recall_mode="summarized",
        sibling_similarity_threshold=0,
        evidence_rank_config=EvidenceRankConfig(),
        _last_backend_error=None,
    )
    mgr.rows = [decision()]
    mgr.recall_memories = AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(mgr.rows))
    mgr.build_recalled_context = Mock(side_effect=lambda *a, **k:
        UniversalMemoryManager.build_recalled_context(mgr, *a, **k))
    monkeypatch.setattr(mcp_tools, "_get_manager", lambda: mgr)
    mcp_tools._recall_recent.clear()
    yield mgr
    mcp_tools._recall_recent.clear()


def engage(**overrides):
    return mcp_tools.tool_memory_engage({
        "query": "Should we revisit deployment with our new team?",
        "include_insights": True,
        **overrides,
    })


def test_opt_in_preserves_existing_fields_and_has_no_extra_calls(manager):
    baseline = engage(include_insights=False)
    mcp_tools._recall_recent.clear()
    with_brief = engage()
    assert with_brief["insight_brief"]["status"] == "ready"
    assert with_brief["insight_brief"]["candidates"][0]["kind"] == "changed_premise"
    for key, value in baseline.items():
        if key != "approx_payload_tokens":
            assert with_brief[key] == value
    assert manager.recall_memories.await_count == 2
    assert manager.build_recalled_context.call_count == 2
    assert "session_id" not in with_brief
    assert "session_id_source" not in with_brief


@pytest.mark.parametrize("flag", [False, None, "true", 1])
def test_only_explicit_boolean_true_builds_a_brief(manager, monkeypatch, flag):
    import kumiho_memory.insight as insight
    forbidden = Mock(side_effect=AssertionError("builder must be opt-in"))
    monkeypatch.setattr(insight, "build_insight_brief", forbidden)
    result = engage(include_insights=flag)
    assert "insight_brief" not in result
    forbidden.assert_not_called()


def test_omitting_flag_retains_default_response(manager):
    result = mcp_tools.tool_memory_engage({"query": "deployment"})
    assert "insight_brief" not in result


def test_same_filters_and_one_recall_feed_the_brief(manager):
    manager.rows = [decision(score=0.9), decision(
        kref="kref://CognitiveMemory/other/private.fact?r=1",
        title="EXCLUDED", summary="EXCLUDED", score=0.1,
    )]
    result = engage(limit=2, min_score=0.7,
                    space_paths=["CognitiveMemory/decisions"],
                    memory_types=["decision"], graph_augmented=True)
    manager.recall_memories.assert_awaited_once_with(
        "Should we revisit deployment with our new team?", limit=2,
        space_paths=["CognitiveMemory/decisions"], memory_types=["decision"],
        graph_augmented=True,
    )
    assert "EXCLUDED" not in json.dumps(result)
    assert result["insight_brief"]["source_krefs"] == [KREF]
    assert FACT in result["insight_brief"]["candidates"][0]["missing_source_krefs"]


def test_sibling_evidence_is_read_before_summary_payload_drops_it(manager):
    sibling = decision(kref=KREF.replace("?r=1", "?r=2"), summary="A changed sibling rationale.")
    manager.rows = [decision(type="summary", sibling_revisions=[sibling])]
    result = engage(recall_mode="summarized")
    assert "sibling_revisions" not in result["results"][0]
    assert result["results"][0]["sibling_count"] == 1
    brief = result["insight_brief"]
    assert sibling["kref"] in brief["source_krefs"]
    assert "A changed sibling rationale." in json.dumps(brief)


def test_missing_link_remains_missing_instead_of_becoming_inspected_evidence(manager):
    result = engage()
    brief = result["insight_brief"]
    assert FACT not in brief["source_krefs"]
    assert FACT in brief["candidates"][0]["missing_source_krefs"]
    assert manager.recall_memories.await_count == 1


def test_backend_error_is_incomplete_even_with_partial_decision_evidence(manager):
    manager._last_backend_error = "retrieve unavailable"
    result = engage()
    assert result["backend_error"] == "retrieve unavailable"
    assert result["insight_brief"]["status"] == "retrieval_incomplete"
    assert result["insight_brief"]["candidates"]


def test_empty_backend_failure_differs_from_no_candidates(manager):
    manager.rows = []
    result = engage()
    assert result["insight_brief"]["status"] == "insufficient_evidence"
    mcp_tools._recall_recent.clear()
    manager._last_backend_error = "retrieve unavailable"
    result = engage()
    assert result["insight_brief"]["status"] == "retrieval_incomplete"
    assert result["insight_brief"]["candidates"] == []


def test_shared_dedup_still_blocks_extra_retrieval_and_false_empty_brief(manager):
    first = engage()
    assert first["insight_brief"]["status"] == "ready"
    duplicate = engage()
    assert duplicate["deduplicated"] is True
    assert "insight_brief" not in duplicate
    recall = mcp_tools.tool_memory_recall({"query": "Should we revisit deployment with our new team?"})
    assert recall["deduplicated"] is True
    assert manager.recall_memories.await_count == 1


def test_enabling_flag_after_same_recall_does_not_bypass_dedup(manager):
    engage(include_insights=False)
    result = engage()
    assert result["deduplicated"] is True
    assert "insight_brief" not in result
    assert manager.recall_memories.await_count == 1


def test_payload_estimate_includes_brief(manager):
    result = engage()
    estimated = result.pop("approx_payload_tokens")
    assert estimated == len(json.dumps(result, ensure_ascii=False, default=str)) // 4
    assert estimated > result["approx_tokens"]


def test_tenant_requests_do_not_reuse_briefs_or_dedup_each_other(manager):
    async def scoped_recall(*args, **kwargs):
        tenant = current_request().tenant_id
        return [decision(kref=f"kref://{tenant}/decisions/deploy.decision?r=1",
                         title=tenant, summary=f"Decision owned by {tenant}",
                         superseded_by="")]
    manager.recall_memories.side_effect = scoped_recall
    with request_context(make_request_context("tenant-a", session_id="s")):
        a = engage()
    with request_context(make_request_context("tenant-b", session_id="s")):
        b = engage()
    assert "tenant-a" in json.dumps(a["insight_brief"])
    assert "tenant-a" not in json.dumps(b["insight_brief"])
    assert "tenant-b" in json.dumps(b["insight_brief"])
    assert "tenant-b" not in json.dumps(a["insight_brief"])
    assert manager.recall_memories.await_count == 2


def test_schema_and_transport_description_expose_opt_in_contract():
    tool = next(t for t in mcp_tools.MEMORY_TOOLS if t["name"] == "kumiho_memory_engage")
    schema = tool["inputSchema"]
    assert schema["properties"]["include_insights"]["type"] == "boolean"
    assert schema["properties"]["include_insights"]["default"] is False
    assert "include_insights" not in schema["required"]
    # Optional property descriptions can be stripped by hosts; this contract
    # must survive in the tool-level description.
    assert "include_insights=true" in tool["description"]
    assert "hypotheses, not established conclusions" in tool["description"]


def test_incomplete_status_is_inside_serialized_brief_budget(manager):
    from kumiho_memory.insight import MAX_BRIEF_CHARS
    # Unicode escapes make this a near-budget envelope, not a prose-only cap.
    related = [f"kref://p/facts/{i}.fact?r=1" for i in range(3)]
    manager.rows = [decision(title="가" * 120, summary="가" * 240,
                             grounding_stale=False, contested_by=related)]
    manager.rows.extend(decision(kref=ref, type="fact", title="가" * 120,
                                 summary="가" * 240, grounding_stale=False)
                        for ref in related)
    manager._last_backend_error = "partial retrieval"
    result = engage(query="Q" * 485)
    assert result["insight_brief"]["status"] == "retrieval_incomplete"
    assert len(json.dumps(result["insight_brief"])) <= MAX_BRIEF_CHARS
