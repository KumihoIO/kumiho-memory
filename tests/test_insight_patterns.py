"""Offline lifecycle tests: lineage, source freshness, scope, and explicit writes."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from kumiho_memory.experience import normalize_experience
from kumiho_memory.insight_patterns import (
    MAX_REQUEST_CHARS, MAX_SOURCES, assess_pattern_applicability,
    prepare_pattern_request, store_pattern_candidate, validate_pattern_candidate,
)


def row(name="alpha", *, experience_id=None, summary=None, **metadata):
    record = normalize_experience({
        "experience_id": experience_id or f"event-{name}", "title": f"Pilot {name}",
        "situation": "A small team faced an uncertain deployment.",
        "goal": "Deliver a reliable release", "alternatives": ["Broad rollout", "Small pilot"],
        "decision": "Run a small pilot", "rationale": summary or "Observe operating cost before commitment.",
        "applicability_conditions": ["Small team", "Uncertain operating cost"],
        "expected_outcome": "Lower rollout risk", "source_krefs": [], "origin": "user",
        "decision_state": "accepted",
    })
    record["recorded_at"] = "2026-09-09T01:00:00+00:00"
    return {"kref": f"kref://p/experiences/{name}.experience?r=1",
            "metadata": {"experience_record": json.dumps(record), **metadata}, "tags": []}


def request(rows=None):
    return prepare_pattern_request(rows if rows is not None else [row(), row("beta")],
                                   space_paths=["/p/experiences"])


def proposal(req, **overrides):
    return {"kind": "recurring_pattern", "title": "Pilot before commitment",
            "hypothesis": "A small pilot may help this team test operating assumptions.",
            "applicability_conditions": ["The operating cost remains uncertain"],
            "counterexamples": [], "source_krefs": req["source_krefs"], **overrides}


def manager(monkeypatch, rows):
    import kumiho
    indexed = {r["kref"]: r for r in rows}
    def get_revision(ref):
        source = indexed[ref]
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=source["metadata"],
                               tags=source.get("tags", []))
    monkeypatch.setattr(kumiho, "get_revision", get_revision)
    def get_item(ref):
        source = next(r for k, r in indexed.items() if k.split("?", 1)[0] == ref)
        return SimpleNamespace(kref=SimpleNamespace(uri=ref),
                               metadata=source.get("item_markers", {}), deprecated=False)
    monkeypatch.setattr(kumiho, "get_item", get_item)
    payloads = []
    def store(**payload):
        payloads.append(payload)
        return {"revision_kref": "kref://p/patterns/pilot.pattern_candidate?r=1",
                "edges_created": payload["source_revision_krefs"]}
    return SimpleNamespace(project="p", memory_store=store), payloads


def test_prepare_is_pure_bounded_and_contains_only_canonical_scoped_sources():
    rows = [row(), row("beta"), {"kref": "kref://p/experiences/raw.memory?r=1", "content": "RAW"},
            {**row("other"), "kref": "kref://elsewhere/private/other.experience?r=1"}]
    before = copy.deepcopy(rows)
    req = request(rows)
    assert req["status"] == "ready"
    assert len(req["sources"]) == 2
    assert rows == before
    assert request(rows) == req
    assert "RAW" not in json.dumps(req)
    assert "elsewhere" not in json.dumps(req)
    assert req["sources"][0]["text"]["expected_outcome"] == "Lower rollout risk"
    assert req["sources"][0]["text"]["observed_outcome"] == ""


def test_prepare_requires_explicit_scope_and_never_uses_prefix_collision():
    assert prepare_pattern_request([row()])["status"] == "invalid_scope"
    assert prepare_pattern_request([row()], space_paths=["/p/../other"])["status"] == "invalid_scope"
    assert prepare_pattern_request([row()], space_paths=["/p/experience"])["sources"] == []


def test_prepare_caps_large_unicode_records_with_exact_snapshot_alignment():
    rows = [row(str(i), summary="가" * 1900) for i in range(100)]
    req = request(rows)
    assert len(json.dumps(req)) <= MAX_REQUEST_CHARS
    assert len(req["sources"]) <= MAX_SOURCES
    assert req["truncated"] is True
    assert req["source_krefs"] == [s["kref"] for s in req["sources"]]
    assert all(s["snippet_truncated"] for s in req["sources"])


def test_candidate_is_forced_to_inferred_unverified_proposal():
    req = request()
    result = validate_pattern_candidate(req, proposal(req, evidence_level="official", origin="user",
                                                     decision_state="accepted", inferred=False))
    assert result["inferred"] is True
    assert result["origin"] == "agent"
    assert result["decision_state"] == "proposal"
    assert result["evidence_level"] == "unverified"
    assert result["corroboration"] == "not_established"
    assert result["counterexample_status"] == "unknown"
    assert result["applicability"] == "unknown"


@pytest.mark.parametrize("change", [
    {"source_krefs": ["kref://p/experiences/unknown.experience?r=1"]},
    {"source_krefs": ["kref://p/experiences/alpha.experience"]},
    {"applicability_conditions": []}, {"counterexamples": None},
    {"title": ""}, {"hypothesis": "x" * 1601}, {"kind": "proven_belief"},
])
def test_invalid_host_proposals_are_rejected(change):
    req = request()
    with pytest.raises(ValueError):
        validate_pattern_candidate(req, proposal(req, **change))


def test_revision_duplicates_and_same_experience_identity_do_not_count_as_recurrence():
    a = row()
    revised = {**row("beta"), "kref": a["kref"].replace("?r=1", "?r=2")}
    req = request([a, revised])
    with pytest.raises(ValueError, match="distinct experiences"):
        validate_pattern_candidate(req, proposal(req))
    req = request([a, row("beta", experience_id="event-alpha")])
    with pytest.raises(ValueError, match="distinct experiences"):
        validate_pattern_candidate(req, proposal(req))


def test_shared_lineage_is_never_reported_as_corroboration():
    rows = [row(), row("beta")]
    # Canonical experience source provenance can overlap; number of experience
    # items still never upgrades evidence or establishes independence.
    req = request(rows)
    result = validate_pattern_candidate(req, proposal(req))
    assert result["corroboration"] == "not_established"
    assert "not independent corroboration" in req["host_instruction"]


def test_one_experience_can_support_conditional_lesson_only():
    req = request([row()])
    with pytest.raises(ValueError):
        validate_pattern_candidate(req, proposal(req))
    result = validate_pattern_candidate(req, proposal(req, kind="conditional_lesson"))
    assert result["kind"] == "conditional_lesson"


def test_modified_snapshot_fails_validation():
    req = request()
    req["sources"][0]["text"]["decision"] = "Different decision"
    with pytest.raises(ValueError, match="modified"):
        validate_pattern_candidate(req, proposal(req))


def test_explicit_store_revalidates_sources_and_preserves_exact_lineage(monkeypatch):
    rows = [row(), row("beta")]
    req = request(rows)
    target, payloads = manager(monkeypatch, rows)
    result = asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    assert result["status"] == "stored_proposal"
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["source_revision_krefs"] == req["source_krefs"]
    assert payload["edge_type"] == "DERIVED_FROM"
    assert payload["memory_item_kind"] == "pattern_candidate"
    assert "published" not in payload["tags"]
    assert payload["stack_revisions"] is False
    stored = json.loads(payload["metadata"]["pattern_candidate"])
    assert stored["evidence_level"] == "unverified"
    assert stored["decision_state"] == "proposal"


def test_forged_rehashed_snapshot_cannot_bypass_backend_revalidation(monkeypatch):
    from kumiho_memory.insight_patterns import _digest
    rows = [row(), row("beta")]
    req = request(rows)
    req["sources"][0]["text"]["decision"] = "Forged source content"
    req["snapshot_id"] = _digest(req["sources"])
    target, payloads = manager(monkeypatch, rows)
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    assert payloads == []


@pytest.mark.parametrize("source_change", ["missing", "changed", "stale", "contested", "superseded"])
def test_source_changes_fail_closed_before_any_write(monkeypatch, source_change):
    rows = [row(), row("beta")]
    req = request(rows)
    if source_change == "missing":
        rows.pop()
    elif source_change == "changed":
        rows[0] = row(summary="A different reason")
    else:
        key = {"stale": "grounding_stale", "contested": "contested_by", "superseded": "superseded_by"}[source_change]
        rows[0]["metadata"][key] = "true" if source_change == "stale" else [rows[1]["kref"]]
    target, payloads = manager(monkeypatch, rows)
    with pytest.raises(ValueError):
        asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    assert payloads == []


def test_scope_escape_and_credentials_cannot_write(monkeypatch):
    rows = [row(), row("beta")]
    req = request(rows)
    target, payloads = manager(monkeypatch, rows)
    with pytest.raises(ValueError):
        asyncio.run(store_pattern_candidate(target, req, proposal(req), "/other/patterns"))
    with pytest.raises(ValueError):
        asyncio.run(store_pattern_candidate(target, req, proposal(req, hypothesis="sk-proj-" + "x" * 40), "/p/patterns"))
    assert payloads == []


def test_pii_redaction_does_not_rewrite_pinned_revision_identifiers(monkeypatch):
    rows = [row(), row("beta")]
    rows[0]["kref"] = rows[0]["kref"].replace("?r=1", "?r=1234567890")
    req = request(rows)
    target, payloads = manager(monkeypatch, rows)
    asyncio.run(store_pattern_candidate(target, req, proposal(req, hypothesis="Contact alice@example.com to review."), "/p/patterns"))
    payload = payloads[0]
    assert "alice@example.com" not in payload["summary"]
    assert payload["source_revision_krefs"][0].endswith("?r=1234567890")


def test_missing_superseded_disputed_and_clean_sources_remain_epistemically_distinct():
    rows = [row(), row("beta")]
    req = request(rows)
    candidate = validate_pattern_candidate(req, proposal(req))
    assert assess_pattern_applicability(candidate, rows)["status"] == "reviewable"
    assert assess_pattern_applicability(candidate, rows)["applicability"] == "unknown"
    assert assess_pattern_applicability(candidate, rows[:1])["status"] == "unknown"
    rows[0]["metadata"]["grounding_stale"] = "true"
    result = assess_pattern_applicability(candidate, rows[:1])
    assert result["status"] == "stale"
    assert result["missing_source_krefs"] == [rows[1]["kref"]]


def test_store_error_never_returns_success(monkeypatch):
    rows = [row(), row("beta")]
    req = request(rows)
    target, _ = manager(monkeypatch, rows)
    target.memory_store = lambda **payload: {"error": "backend unavailable"}
    assert asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))["status"] == "store_failed"


def test_outcomes_are_exposed_as_observations_and_never_count_as_new_experiences():
    from kumiho_memory.experience import _normalize_outcome
    experience = row()
    observed = _normalize_outcome(experience["kref"], {
        "observed_outcome": "Pilot exposed an unexpected operating cost",
        "observed_at": "2026-09-08T10:00:00+00:00", "outcome_status": "mixed",
        "acceptance": "unknown", "origin": "user", "source_krefs": [],
    })
    observed["recorded_at"] = "2026-09-09T01:00:00+00:00"
    outcome = {"kref": "kref://p/experiences/result.experience?r=1",
               "metadata": {"experience_record": json.dumps(observed)}}
    req = prepare_pattern_request([experience], [outcome], space_paths=["/p/experiences"])
    assert req["sources"][1]["experience_kref"] == experience["kref"]
    assert req["sources"][1]["text"]["outcome_status"] == "mixed"
    with pytest.raises(ValueError, match="distinct experiences"):
        validate_pattern_candidate(req, proposal(req))
    assert validate_pattern_candidate(req, proposal(req, kind="conditional_lesson"))["source_krefs"] == req["source_krefs"]


def test_shared_original_provenance_retains_lineage_without_claiming_independence():
    from kumiho_memory.experience import experience_from_memory
    rows = [row(), row("beta")]
    lineage = "kref://p/reports/shared.memory?r=1"
    for source in rows:
        record = experience_from_memory(source)
        payload = {key: value for key, value in record.items()
                   if key not in ("schema", "record_type", "record_id", "recorded_at", "kref")}
        payload["source_krefs"] = [lineage]
        normalized = normalize_experience(payload)
        normalized["recorded_at"] = record["recorded_at"]
        source["metadata"]["experience_record"] = json.dumps(normalized)
    req = request(rows)
    assert all(source["source_krefs"] == [lineage] for source in req["sources"])
    assert validate_pattern_candidate(req, proposal(req))["corroboration"] == "not_established"


def test_preexisting_stale_source_can_be_previewed_but_cannot_be_stored(monkeypatch):
    rows = [row(grounding_stale="true"), row("beta")]
    req = request(rows)
    assert req["status"] == "ready"
    target, payloads = manager(monkeypatch, rows)
    with pytest.raises(ValueError, match="stale"):
        asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    assert payloads == []


@pytest.mark.parametrize("bad", [None, [], "text", {"x": float("nan")}, {"x": 1 << 150},
                                  {"x": [[[[[[[[[["deep"]]]]]]]]]]}])
def test_malformed_host_payload_fails_with_validation_error(bad):
    with pytest.raises(ValueError):
        validate_pattern_candidate(request(), bad)


def test_unicode_candidate_is_bounded_in_actual_json_encoding():
    req = request()
    with pytest.raises(ValueError, match="encoded size"):
        validate_pattern_candidate(req, proposal(req, hypothesis="가" * 1600))


def test_malformed_source_identity_cannot_raise_an_unhandled_type_error():
    from kumiho_memory.insight_patterns import _digest
    req = request()
    req["sources"][0]["experience_id"] = {"bad": "shape"}
    req["snapshot_id"] = _digest(req["sources"])
    with pytest.raises(ValueError, match="identity"):
        validate_pattern_candidate(req, proposal(req))


def test_best_effort_graph_edge_failure_is_reported_without_losing_metadata_lineage(monkeypatch):
    rows = [row(), row("beta")]
    req = request(rows)
    target, _ = manager(monkeypatch, rows)
    target.memory_store = lambda **payload: {
        "revision_kref": "kref://p/patterns/pilot.pattern_candidate?r=1", "edges_created": []}
    result = asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    assert result["status"] == "stored_proposal"
    assert result["lineage_status"] == "metadata_only"
    assert result["graph_links_verified"] is False


def test_backend_wrong_scope_or_kind_never_reports_success(monkeypatch):
    rows = [row(), row("beta")]
    req = request(rows)
    target, _ = manager(monkeypatch, rows)
    target.memory_store = lambda **payload: {"revision_kref": "kref://other/patterns/pilot.fact?r=1"}
    result = asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    assert result["status"] == "store_failed"
    assert result["partial_write_possible"] is True


def test_stored_pattern_decoder_preserves_proposal_status_and_rejects_tampering(monkeypatch):
    from kumiho_memory.insight_patterns import pattern_from_memory
    rows = [row(), row("beta")]
    req = request(rows)
    target, payloads = manager(monkeypatch, rows)
    result = asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    stored = {"kref": result["revision_kref"], "metadata": payloads[0]["metadata"]}
    decoded = pattern_from_memory(stored)
    assert decoded["decision_state"] == "proposal"
    assert decoded["evidence_level"] == "unverified"
    assert decoded["source_krefs"] == req["source_krefs"]
    raw = json.loads(stored["metadata"]["pattern_candidate"])
    raw["hypothesis"] = "Modified text"
    stored["metadata"]["pattern_candidate"] = json.dumps(raw)
    assert pattern_from_memory(stored) is None


def test_decoder_rejects_forged_promoted_status_even_with_matching_hash():
    from kumiho_memory.insight_patterns import _digest, pattern_from_memory
    req = request()
    candidate = validate_pattern_candidate(req, proposal(req))
    candidate["evidence_level"] = "official"
    candidate["candidate_id"] = _digest({k: v for k, v in candidate.items() if k != "candidate_id"})
    assert pattern_from_memory({"kref": "kref://p/patterns/x.pattern_candidate?r=1",
                                "metadata": {"pattern_candidate": json.dumps(candidate)}}) is None


def test_deprecated_pattern_node_is_not_revived_by_healthy_sources(monkeypatch):
    from kumiho_memory.insight_patterns import pattern_from_memory
    rows = [row(), row("beta")]
    req = request(rows)
    target, payloads = manager(monkeypatch, rows)
    result = asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    stored = {"kref": result["revision_kref"], "metadata": payloads[0]["metadata"], "tags": ["deprecated"]}
    decoded = pattern_from_memory(stored)
    assessment = assess_pattern_applicability(decoded, rows)
    assert assessment["status"] == "stale"
    assert assessment["stale_sources"][0]["reason"] == "pattern_superseded"


def test_item_markers_preserve_scope_without_borrowing_revision_provenance():
    source = row()
    source["item_markers"] = {"grounding_stale": "true", "origin": "operator", "evidence_level": "official"}
    req = request([source])
    snapshot = req["sources"][0]
    assert snapshot["source_state"]["grounding_stale"] is True
    assert snapshot["source_state"]["marker_scopes"]["grounding_stale"] == "item"
    assert snapshot["origin"] == "user"
    assert snapshot["evidence_level"] == "unverified"
    candidate = validate_pattern_candidate(req, proposal(req, kind="conditional_lesson"))
    assessment = assess_pattern_applicability(candidate, [source])
    assert assessment["status"] == "stale"
    assert assessment["stale_sources"][0]["marker_scope"] == "item"


def test_item_marker_change_invalidates_snapshot_before_store(monkeypatch):
    import kumiho_memory.experience as experience_module
    rows = [row(), row("beta")]
    req = request(rows)
    rows[0]["item_markers"] = {"contested_by": [rows[1]["kref"]]}
    async def fresh_sources(*args, **kwargs):
        return rows
    monkeypatch.setattr(experience_module, "validate_source_refs", fresh_sources)
    target, payloads = manager(monkeypatch, rows)
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(store_pattern_candidate(target, req, proposal(req), "/p/patterns"))
    assert payloads == []


def test_revision_and_item_markers_are_identified_as_both():
    source = row(grounding_stale="true")
    source["item_markers"] = {"grounding_stale": "true"}
    req = request([source])
    assert req["sources"][0]["source_state"]["marker_scopes"]["grounding_stale"] == "both"


@pytest.mark.parametrize("markers, reason", [
    ({"deprecated": True}, "superseded"),
    ({"as_of_excluded": "true"}, "as_of_excluded"),
    ({"contested_by": '["kref://p/facts/opposite.fact?r=1"]'}, "contested"),
])
def test_sdk_item_marker_encodings_are_recognized(markers, reason):
    source = row()
    source["item_markers"] = markers
    req = request([source])
    candidate = validate_pattern_candidate(req, proposal(req, kind="conditional_lesson"))
    health = assess_pattern_applicability(candidate, [source])
    assert health["status"] == "stale"
    assert health["stale_sources"][0]["reason"] == reason
    assert health["stale_sources"][0]["marker_scope"] == "item"


def test_sdk_revision_deprecation_does_not_require_item_deprecation():
    source = row()
    source["revision_deprecated"] = True
    source["item_markers"] = {}
    req = request([source])
    candidate = validate_pattern_candidate(req, proposal(req, kind="conditional_lesson"))
    health = assess_pattern_applicability(candidate, [source])
    assert health["status"] == "stale"
    assert health["stale_sources"][0]["reason"] == "superseded"
    assert health["stale_sources"][0]["marker_scope"] == "revision"


def test_sdk_false_deprecated_flag_does_not_shadow_explicit_metadata_deprecation():
    source = row(deprecated="true")
    source["revision_deprecated"] = False
    req = request([source])
    assert req["sources"][0]["source_state"]["superseded"] is True
