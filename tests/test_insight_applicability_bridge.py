"""PR31 applicability metadata must survive PR32's host insight/lifecycle paths."""
import asyncio
import json

import pytest

from kumiho_memory.applicability import (
    CLAIM_ORIGINS, DECISION_STATES, apply_applicability_marker, applicability_notes,
)
from kumiho_memory.context_compose import compose_context
from kumiho_memory.experience import experience_from_memory, normalize_experience
from kumiho_memory.insight import build_insight_brief
from kumiho_memory.insight_patterns import assess_pattern_applicability
from kumiho_memory.insight_synthesis import prepare_insight_request
from test_insight_recall import experience, pattern, setup

REF = "kref://p/decisions/old.decision?r=1"


def source(**fields):
    return {"kref": REF, "title": "Old choice", "summary": "Use a small pilot", **fields}


@pytest.mark.parametrize("markers,reason", [
    ({"status": "superseded"}, "superseded"),
    ({"status": " SUPERSEDED "}, "superseded"),
    ({"superseded": True}, "superseded"),
    ({"superseded": "true"}, "superseded"),
    ({"decision_state": "contested"}, "contested"),
    ({"status": "contested"}, "contested"),
    ({"contested": True}, "contested"),
])
def test_standalone_current_markers_reach_packet_and_health(markers, reason):
    row = source(**markers)
    packet = prepare_insight_request("Does the historical choice still apply?", [row])
    evidence = packet["sources"][0]
    for key, value in markers.items():
        assert evidence[key] == value
    leads = packet["review_brief"]["candidates"]
    assert leads and leads[0]["marker_scope"] == "revision"
    assert leads[0]["evidence"][0]["provenance"][reason] is True
    health = assess_pattern_applicability({"source_krefs": [REF]}, [{"kref": REF, "metadata": markers}])
    assert health["status"] == "stale"
    assert health["stale_sources"] == [{"kref": REF, "reason": reason, "marker_scope": "revision"}]
    assert health["applicability"] == "unknown"


@pytest.mark.parametrize("markers,reason", [
    ({"status": "superseded"}, "superseded"),
    ({"superseded": True}, "superseded"),
    ({"decision_state": "contested"}, "contested"),
])
@pytest.mark.parametrize("same_ref", [True, False])
def test_parent_warning_never_becomes_sibling_revision_provenance(markers, reason, same_ref):
    sibling = source(status="active")
    parent = source(**markers, sibling_revisions=[sibling])
    if not same_ref:
        parent["kref"] = REF.replace("?r=1", "?r=2")
    request = prepare_insight_request("Can this still inform the choice?", [parent])
    row = request["sources"][0]
    assert row["status"] == "active"
    assert not row.get("superseded") and not row.get("contested")
    for key, value in markers.items():
        assert row["item_markers"][key] == value
    lead = request["review_brief"]["candidates"][0]
    assert lead["marker_scope"] == "item"
    assert lead["evidence"][0]["provenance"][reason] is False
    assert "another sibling" in lead["observation"]


@pytest.mark.parametrize("markers,reason", [
    ({"status": "superseded"}, "superseded"),
    ({"superseded": True}, "superseded"),
    ({"decision_state": "contested"}, "contested"),
])
def test_canonical_item_state_survives_graph_read_and_synthesis(monkeypatch, markers, reason):
    row = experience(item_markers=markers)
    manager, _, _ = setup(monkeypatch, [row])
    from kumiho_memory.insight_recall import recall_learned_sources
    result = asyncio.run(recall_learned_sources(manager, "pilot"))
    packet = prepare_insight_request("pilot", result["results"])
    recalled = packet["sources"][0]
    assert recalled["item_markers"][reason] is True
    assert not recalled.get(reason)
    assert not recalled.get("grounding_stale")
    summary = json.loads(recalled["summary"])
    assert summary["source_health_details"]["stale_sources"] == [
        {"kref": row["kref"], "reason": reason, "marker_scope": "item"}]
    assert packet["review_brief"]["candidates"][0]["marker_scope"] == "item"


def test_superseded_pattern_source_is_not_relabelled_as_pattern_grounding(monkeypatch):
    row = experience()
    proposal = pattern([row])
    row["metadata"]["status"] = "superseded"
    manager, _, _ = setup(monkeypatch, [row, proposal])
    from kumiho_memory.insight_recall import recall_learned_sources
    result = asyncio.run(recall_learned_sources(manager, "pilot"))
    request = prepare_insight_request("pilot", result["results"])
    recalled = next(r for r in request["sources"] if r["type"] == "pattern_candidate")
    assert not recalled.get("grounding_stale") and not recalled.get("superseded")
    summary = json.loads(recalled["summary"])
    assert summary["source_health"] == "stale"
    assert summary["source_health_details"]["stale_sources"] == [
        {"kref": row["kref"], "reason": "superseded", "marker_scope": "revision"}]


def test_legacy_proposed_external_canonical_hash_still_decodes():
    # Captured from pre-integration PR32 normalize_experience, not recomputed
    # from today's implementation: alias rewriting would break this identity.
    record = {
        "schema": "kumiho.experience.v1", "record_type": "experience", "experience_id": "legacy-v1",
        "title": "Prior choice", "situation": "Small team", "goal": "Release",
        "decision": "Try pilot", "rationale": "Observe cost", "expected_outcome": "Know cost",
        "origin": "external", "decision_state": "proposed", "acceptance": "unknown",
        "alternatives": [], "applicability_conditions": [], "observed_at": "", "observed_outcome": "",
        "outcome_status": "unknown", "source_krefs": [],
        "record_id": "0630ccebcb74a44ae9527ce2222c1dde224ba0542eea7fb1e67fa9a96c6324f2",
        "recorded_at": "2026-09-09T00:00:00+00:00",
    }
    decoded = experience_from_memory({"kref": "kref://p/experiences/legacy.experience?r=1",
        "metadata": {"experience_record": json.dumps(record)}})
    assert decoded is not None and decoded["record_id"] == record["record_id"]
    assert decoded["origin"] == "external" and decoded["decision_state"] == "proposed"
    projected = {}
    apply_applicability_marker(projected, decoded)
    assert projected == {"origin": "external", "decision_state": "proposed"}
    assert "not an accepted decision" in applicability_notes(projected)


@pytest.mark.parametrize("origin", CLAIM_ORIGINS)
@pytest.mark.parametrize("state", DECISION_STATES)
def test_shared_vocab_retains_explicit_labels_through_experience_and_recall(origin, state):
    record = normalize_experience({
        "experience_id": "union", "title": "Choice", "situation": "Small team", "goal": "Release",
        "decision": "Pilot", "rationale": "Observe", "expected_outcome": "Known cost",
        "origin": origin, "decision_state": state,
    })
    projected = {}
    apply_applicability_marker(projected, record)
    assert projected.get("origin", "unknown") == origin
    assert projected.get("decision_state", "unknown") == state


@pytest.mark.parametrize("state,word", [
    ("proposal", "proposal"), ("proposed", "proposal"),
    ("contested", "contested"), ("superseded", "superseded"), ("rejected", "rejected"),
])
def test_ordinary_context_qualifies_standalone_states(state, word):
    assert "[" + word + ":" in compose_context([source(decision_state=state)], mode="summarized")
    if state in ("proposal", "proposed", "rejected"):
        # Non-acceptance is not falsification of a historical observation.
        health = assess_pattern_applicability({"source_krefs": [REF]}, [source(decision_state=state)])
        assert health["status"] == "reviewable" and health["applicability"] == "unknown"


def test_shared_vocab_is_advertised_by_reflect_and_experience_tools():
    from kumiho_memory.mcp_tools import MEMORY_TOOLS
    tools = {t["name"]: t for t in MEMORY_TOOLS}
    reflect = tools["kumiho_memory_reflect"]["inputSchema"]["properties"]["captures"]["items"]["properties"]
    experience_schema = tools["kumiho_memory_record_experience"]["inputSchema"]["properties"]["record"]["properties"]
    for schema in (reflect, experience_schema):
        assert set(schema["origin"]["enum"]) == set(CLAIM_ORIGINS)
        assert set(schema["decision_state"]["enum"]) == set(DECISION_STATES)


@pytest.mark.parametrize("markers,reason", [
    ({"status": "superseded"}, "superseded"),
    ({"decision_state": "contested"}, "contested"),
])
def test_direct_brief_keeps_canonical_item_warning_separate(markers, reason):
    brief = build_insight_brief("Does the old choice apply?", [source(status="active", item_markers=markers)])
    lead = brief["candidates"][0]
    assert lead["marker_scope"] == "item"
    assert lead["evidence"][0]["provenance"][reason] is False
    assert lead["evidence"][0]["provenance"]["status"] == "active"
