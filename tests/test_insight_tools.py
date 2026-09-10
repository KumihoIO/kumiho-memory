"""Offline MCP lifecycle tests with no provider, graph writes mocked at boundary."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kumiho_memory import insight_tools, mcp_tools
from kumiho_memory.experience import normalize_experience
from kumiho_memory.insight_patterns import prepare_pattern_request

REF = "kref://project/experiences/run.experience?r=1"
PATTERN = "kref://project/patterns/lesson.pattern_candidate?r=1"


def extraction():
    return {"experience_id": "run-one", "title": "Pilot decision", "situation": "Small team",
            "goal": "On-time delivery", "decision": "Pilot first", "rationale": "Limit support cost",
            "expected_outcome": "Measure support", "decision_state": "proposed"}


def row():
    record = {**normalize_experience(extraction()), "recorded_at": "2026-09-09T00:00:00Z"}
    return {"kref": REF, "metadata": {"experience_record": json.dumps(record)}, "tags": []}


def proposal():
    return {"kind": "conditional_lesson", "title": "Pilot can expose support costs",
            "hypothesis": "A small pilot may expose support costs before rollout.",
            "applicability_conditions": ["Small team"], "counterexamples": [], "source_krefs": [REF]}


@pytest.fixture
def manager(monkeypatch):
    mgr = SimpleNamespace(project="project", memory_store=AsyncMock(return_value={"revision_kref": REF}))
    monkeypatch.setattr(mcp_tools, "_get_manager", lambda: mgr)
    return mgr


@pytest.fixture
def graph(monkeypatch):
    import kumiho
    rows = {REF: row()}
    def read(ref):
        data = rows[ref]
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=data["metadata"], tags=data.get("tags", []), deprecated=data.get("revision_deprecated", False))
    monkeypatch.setattr(kumiho, "get_revision", read)
    def read_item(ref):
        data = next(row for key, row in rows.items() if key.split("?")[0] == ref)
        if data.get("item_inaccessible"):
            raise RuntimeError("Item denied")
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=data.get("item_markers", {}), deprecated=data.get("deprecated", False))
    monkeypatch.setattr(kumiho, "get_item", read_item)
    return rows


def test_registry_has_six_explicit_tools_and_correct_annotations():
    assert len(insight_tools.INSIGHT_TOOLS) == 6
    writes = {"record_experience", "record_outcome", "store_pattern"}
    for definition in insight_tools.INSIGHT_TOOLS:
        name = definition["name"]
        assert name in mcp_tools.MEMORY_TOOL_HANDLERS
        assert sum(t["name"] == name for t in mcp_tools.MEMORY_TOOLS) == 1
        assert definition["annotations"]["readOnlyHint"] == (name.removeprefix("kumiho_memory_") not in writes)
        assert definition["inputSchema"]["additionalProperties"] is False


def test_record_tool_writes_unpublished_proposal(manager):
    result = insight_tools.tool_record_experience({"record": extraction(), "space_path": "/project/experiences"})
    assert result["record"]["decision_state"] == "proposed"
    payload = manager.memory_store.call_args.kwargs
    assert "proposal" in payload["tags"]
    assert "published" not in payload["tags"]
    assert "evidence:unverified" in payload["tags"]


def test_outcome_tool_keeps_observation_separate(manager, graph):
    result = insight_tools.tool_record_outcome({"experience_kref": REF, "outcome": {
        "observed_outcome": "Support exceeded budget", "observed_at": "2026-09-08T00:00:00Z",
        "outcome_status": "failure", "acceptance": "accepted"}})
    assert result["record"]["outcome_status"] == "failure"
    assert result["record"]["acceptance"] == "accepted"
    assert manager.memory_store.call_args.kwargs["source_revision_krefs"] == [REF]


def test_prepare_is_explicit_read_only_and_never_constructs_dream(manager, graph, monkeypatch):
    from kumiho_memory.dream_state import DreamState
    monkeypatch.setattr(DreamState, "__init__", lambda *a, **k: pytest.fail("Constructed DreamState provider cycle"))
    result = insight_tools.tool_prepare_patterns({"source_krefs": [REF], "space_paths": ["/project/experiences"]})
    assert result["status"] == "ready"
    assert result["source_krefs"] == [REF]
    manager.memory_store.assert_not_called()


@pytest.mark.parametrize("args", [
    {"source_krefs": [], "space_paths": ["/project"]},
    {"source_krefs": [REF] * 13, "space_paths": ["/project"]},
    {"source_krefs": [REF], "space_paths": []},
    {"source_krefs": [REF], "space_paths": ["project"]},
    {"source_krefs": [REF], "space_paths": ["/other"]},
    {"source_krefs": [REF.replace("?r=1", "")], "space_paths": ["/project"]},
    {"source_krefs": [REF], "space_paths": ["/project/team"]},
])
def test_prepare_rejects_invalid_bounds_scope_before_any_write(manager, graph, args):
    result = insight_tools.tool_prepare_patterns(args)
    assert result["error_type"] == "invalid_input_or_source"
    manager.memory_store.assert_not_called()


def test_store_pattern_revalidates_sources_and_keeps_proposal(manager, graph):
    manager.memory_store.return_value = {"revision_kref": PATTERN}
    request = insight_tools.tool_prepare_patterns({"source_krefs": [REF], "space_paths": ["/project"]})
    result = insight_tools.tool_store_pattern({"request": request, "candidate": proposal(), "space_path": "/project/patterns"})
    assert result["status"] == "stored_proposal"
    payload = manager.memory_store.call_args.kwargs
    assert payload["assistant_text"] or payload["user_text"]
    assert "published" not in payload["tags"]
    assert payload["metadata"]["evidence_level"] == "unverified"


def test_store_pattern_rejects_changed_source(manager, graph):
    request = prepare_pattern_request([graph[REF]], space_paths=["/project"])
    graph[REF]["metadata"]["grounding_stale"] = "true"
    result = insight_tools.tool_store_pattern({"request": request, "candidate": proposal(), "space_path": "/project/patterns"})
    assert "error" in result
    manager.memory_store.assert_not_called()


def test_bad_arguments_do_not_resolve_manager(monkeypatch):
    monkeypatch.setattr(insight_tools, "_manager", lambda: pytest.fail("manager constructed"))
    assert "error" in insight_tools.tool_record_experience({"unexpected": 1})
    assert "error" in insight_tools.tool_record_outcome({})
    assert "error" in insight_tools.tool_store_pattern({})
    assert "error" in insight_tools.tool_prepare_patterns({})
    assert "error" in insight_tools.tool_check_pattern({})


def test_validate_response_is_structural_and_never_resolves_manager(monkeypatch):
    import kumiho_memory.insight_synthesis as synthesis
    monkeypatch.setattr(insight_tools, "_manager", lambda: pytest.fail("manager constructed"))
    expected = {"valid": True, "validation_scope": "structural_only", "semantic_support_verified": False}
    monkeypatch.setattr(synthesis, "validate_insight_response", lambda req, response: expected)
    result = insight_tools.tool_validate_insight_response({"request": {}, "response": {}})
    assert result == expected


def test_backend_exception_does_not_leak_transport_secrets(manager):
    manager.memory_store.side_effect = RuntimeError("Bearer " + "private" * 10)
    result = insight_tools.tool_record_experience({"record": extraction()})
    assert result["error_type"] == "operation_failed"
    assert "Bearer" not in json.dumps(result)


def test_credential_input_rejected_without_write(manager):
    data = extraction()
    data["rationale"] = "sk-" + "a" * 32
    result = insight_tools.tool_record_experience({"record": data})
    assert "error" in result
    manager.memory_store.assert_not_called()


def test_backend_value_error_does_not_leak_credentials(manager):
    manager.memory_store.side_effect = ValueError("Bearer " + "private" * 10)
    result = insight_tools.tool_record_experience({"record": extraction()})
    assert "Bearer" not in json.dumps(result)


def test_large_revision_number_remains_an_identifier():
    record = extraction()
    record["source_krefs"] = [REF.replace("?r=1", "?r=1234567890")]
    assert normalize_experience(record)["source_krefs"] == record["source_krefs"]


def persist_pattern_fixture(manager, graph):
    manager.memory_store.return_value = {"revision_kref": PATTERN}
    request = insight_tools.tool_prepare_patterns({"source_krefs": [REF], "space_paths": ["/project"]})
    result = insight_tools.tool_store_pattern({"request": request, "candidate": proposal(), "space_path": "/project/patterns"})
    assert result["status"] == "stored_proposal"
    payload = manager.memory_store.call_args.kwargs
    graph[PATTERN] = {"kref": PATTERN, "metadata": payload["metadata"], "tags": payload["tags"]}
    manager.memory_store.reset_mock()


def test_check_pattern_is_reviewable_but_semantic_applicability_unknown(manager, graph):
    persist_pattern_fixture(manager, graph)
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert result["status"] == "reviewable"
    assert result["applicability"] == "unknown"
    assert result["semantic_support_verified"] is False
    manager.memory_store.assert_not_called()


def test_check_pattern_changed_premise_is_stale(manager, graph):
    persist_pattern_fixture(manager, graph)
    graph[REF]["metadata"]["grounding_stale"] = "true"
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert result["status"] == "stale"
    assert result["stale_sources"][0]["kref"] == REF
    manager.memory_store.assert_not_called()


def test_check_pattern_missing_evidence_is_unknown(manager, graph):
    persist_pattern_fixture(manager, graph)
    del graph[REF]
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert result["status"] == "unknown"
    assert result["missing_source_krefs"] == [REF]
    manager.memory_store.assert_not_called()


def test_check_pattern_does_not_widen_scope_for_source(manager, graph):
    persist_pattern_fixture(manager, graph)
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project/patterns"]})
    assert result["status"] == "unknown"
    assert result["missing_source_krefs"] == [REF]


def test_check_pattern_deprecated_proposal_is_stale(manager, graph):
    persist_pattern_fixture(manager, graph)
    graph[PATTERN]["tags"].append("deprecated")
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert result["status"] == "stale"
    assert any(s["kref"] == PATTERN for s in result["stale_sources"])


def test_check_pattern_tampered_candidate_rejected(manager, graph):
    persist_pattern_fixture(manager, graph)
    data = json.loads(graph[PATTERN]["metadata"]["pattern_candidate"])
    data["hypothesis"] = "Tampered"
    graph[PATTERN]["metadata"]["pattern_candidate"] = json.dumps(data)
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert "error" in result
    manager.memory_store.assert_not_called()


def test_check_pattern_item_level_changed_premise_is_stale(manager, graph):
    persist_pattern_fixture(manager, graph)
    graph[REF]["item_markers"] = {"grounding_stale": "true"}
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert result["status"] == "stale"
    assert any(s["kref"] == REF for s in result["stale_sources"])
    assert "grounding_stale" not in graph[REF]["metadata"]


def test_check_pattern_unreadable_item_state_is_unknown(manager, graph):
    persist_pattern_fixture(manager, graph)
    graph[REF]["item_inaccessible"] = True
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert result["status"] == "unknown"
    assert REF in result["missing_source_krefs"]


def test_prepare_item_state_failure_is_closed(manager, graph):
    graph[REF]["item_inaccessible"] = True
    result = insight_tools.tool_prepare_patterns({"source_krefs": [REF], "space_paths": ["/project"]})
    assert "error" in result
    manager.memory_store.assert_not_called()


def test_store_rechecks_item_marker_changes(manager, graph):
    request = insight_tools.tool_prepare_patterns({"source_krefs": [REF], "space_paths": ["/project"]})
    graph[REF]["item_markers"] = {"contested_by": json.dumps(["kref://project/facts/change.fact?r=1"])}
    result = insight_tools.tool_store_pattern({"request": request, "candidate": proposal(), "space_path": "/project/patterns"})
    assert "error" in result
    manager.memory_store.assert_not_called()


def test_check_pattern_deprecated_revision_with_active_item_is_stale(manager, graph):
    persist_pattern_fixture(manager, graph)
    graph[REF]["revision_deprecated"] = True
    assert not graph[REF].get("deprecated", False)
    result = insight_tools.tool_check_pattern({"pattern_kref": PATTERN, "space_paths": ["/project"]})
    assert result["status"] == "stale"
    assert any(s["kref"] == REF and s["marker_scope"] == "revision" for s in result["stale_sources"])
    manager.memory_store.assert_not_called()


def test_store_rejects_deprecated_revision_with_active_item(manager, graph):
    request = insight_tools.tool_prepare_patterns({"source_krefs": [REF], "space_paths": ["/project"]})
    graph[REF]["revision_deprecated"] = True
    result = insight_tools.tool_store_pattern({"request": request, "candidate": proposal(), "space_path": "/project/patterns"})
    assert "error" in result
    manager.memory_store.assert_not_called()
