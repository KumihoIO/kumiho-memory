"""Trust boundaries and practical review scenarios for the pure insight brief."""

import copy
import json

import pytest

from kumiho_memory.insight import (
    MAX_BRIEF_CHARS, MAX_INSIGHTS, MAX_MEMORIES, MAX_SNIPPET_CHARS,
    build_insight_brief,
)


def memory(name="choice", **fields):
    return {"kref": f"kref://project/decisions/{name}.decision?r=1",
            "title": "Prior choice", "summary": "Chose a small pilot before rollout.",
            "type": "decision", **fields}


def test_decision_is_a_review_prompt_not_a_claim_or_confidence():
    row = memory(confidence=0.99)
    result = build_insight_brief("Should we adopt it now?", [row])
    candidate = result["candidates"][0]
    assert result["status"] == "ready"
    assert candidate["kind"] == "decision_review"
    assert candidate["applicability"] == "unknown"
    assert "not been verified" in candidate["observation"]
    assert candidate["evidence"][0]["provenance"]["evidence_level"] == "unverified"
    assert "confidence" not in candidate
    assert candidate["source_krefs"] == [row["kref"]]


def test_changed_grounding_never_claims_replacement_decision():
    fact = memory("changed-fact", type="fact")
    decision = memory(grounding_stale=True, superseded_by=fact["kref"])
    result = build_insight_brief("What changed?", [decision, fact], max_insights=1)
    lead = result["candidates"][0]
    assert lead["kind"] == "changed_premise"
    assert lead["source_krefs"] == [decision["kref"], fact["kref"]]
    assert lead["missing_source_krefs"] == []
    assert "never a replacement decision" in result["synthesis_instruction"]
    assert "replacement" not in lead["observation"]


def test_disagreement_uses_both_recalled_sides_without_promoting_evidence():
    opposite = memory("opposite", type="fact", evidence_level="single_source")
    first = memory(contested_by=[opposite["kref"]], evidence_level="single_source")
    lead = build_insight_brief("Which applies?", [first, opposite])["candidates"][0]
    assert lead["kind"] == "unresolved_conflict"
    assert len(lead["evidence"]) == 2
    assert {e["provenance"]["evidence_level"] for e in lead["evidence"]} == {"single_source"}
    assert lead["missing_source_krefs"] == []


def test_unread_cross_project_target_is_missing_and_not_a_source():
    unread = "kref://private-other/facts/replacement.fact?r=4"
    row = memory(contested_by=[unread, unread])
    result = build_insight_brief("Question", [row])
    lead = result["candidates"][0]
    assert lead["missing_source_krefs"] == [unread]
    assert result["source_krefs"] == [row["kref"]]
    assert all(e["kref"] != unread for e in lead["evidence"])


def test_related_revision_without_summary_is_still_unread():
    empty = memory("empty", title="", summary="", content="Never expose raw content")
    row = memory(contested_by=[empty["kref"]])
    result = build_insight_brief("Question", [row, empty])
    assert result["candidates"][0]["missing_source_krefs"] == [empty["kref"]]
    assert "Never expose" not in json.dumps(result)


@pytest.mark.parametrize("bad_ref", ["kref://p/item.decision", "kref://p/item.decision?r=latest",
                                      "kref://p/item.decision?r=0", "https://example.com",
                                      "kref://p/item.decision?r=1&secret=value", {}, None])
def test_unpinned_or_malformed_refs_cannot_be_cited(bad_ref):
    result = build_insight_brief("Question", [memory(kref=bad_ref)])
    assert result["status"] == "insufficient_evidence"
    assert result["source_krefs"] == []


def test_unpinned_related_reference_remains_missing_even_when_title_exists():
    target = memory("target", kref="kref://p/target.fact")
    lead = build_insight_brief("Question", [memory(contested_by=[target["kref"]]), target])["candidates"][0]
    assert lead["missing_source_krefs"] == [target["kref"]]


def test_keywords_do_not_manufacture_patterns():
    rows = [memory(str(i), type="fact", kref=f"kref://p/{i}.fact?r=1",
                   summary="Failed pilot, repeated conflict, stale decision and slow rollout.") for i in range(3)]
    assert build_insight_brief("Why do pilots always fail?", rows)["candidates"] == []


def test_priority_dedupe_and_exact_sources():
    routine = memory("routine")
    stale = memory("stale", grounding_stale=True)
    conflict = memory("conflict", contested_by=["kref://p/missing.fact?r=1"])
    result = build_insight_brief("Question", [routine, stale, stale, conflict], max_insights=2)
    assert [c["kind"] for c in result["candidates"]] == ["changed_premise", "unresolved_conflict"]
    assert result["source_krefs"] == [stale["kref"], conflict["kref"]]
    assert routine["kref"] not in result["source_krefs"]


def test_sibling_flags_warn_without_borrowing_parent_provenance():
    sibling = memory("sibling", type="fact", evidence_level="invalid")
    parent = memory(grounding_stale=True, evidence_level="official", source="Operator",
                    contested_by=["kref://p/opposing.fact?r=1"], sibling_revisions=[sibling])
    result = build_insight_brief("Question", [parent])
    assert {c["kind"] for c in result["candidates"]} == {"changed_premise", "unresolved_conflict"}
    for lead in result["candidates"]:
        assert lead["marker_scope"] == "item"
        assert "may concern another sibling" in lead["observation"]
        assert lead["source_krefs"] == [sibling["kref"]]
        assert lead["evidence"][0]["provenance"]["evidence_level"] == "unverified"
        assert lead["evidence"][0]["provenance"]["source"] is None
    assert parent["kref"] not in result["source_krefs"]


def test_revision_metadata_precedes_tags_and_unknown_grades_stay_unverified():
    a = memory("a", evidence_level="single_source", tags=["evidence:official"])
    b = memory("b", evidence_level="unknown", tags=["evidence:corroborated"])
    c = memory("c", evidence_level={"bad": 1}, tags="evidence:official")
    result = build_insight_brief("Question", [a, b, c])
    assert [c["evidence"][0]["provenance"]["evidence_level"] for c in result["candidates"]] == [
        "single_source", "corroborated", "unverified"]


def test_nested_metadata_and_optional_temporal_provenance():
    row = {"kref": "kref://p/prior.memory?r=1", "metadata": {
        "type": "decision", "title": "Prior choice", "summary": "A reason.",
        "grounding_stale": "true", "evidence_level": "official", "origin": "operator",
        "decision_state": "historical", "as_of_excluded": True,
        "event_date": "2026-01-01", "valid_to": "2026-02-01"}}
    lead = build_insight_brief("Question", [row])["candidates"][0]
    provenance = lead["evidence"][0]["provenance"]
    assert lead["kind"] == "changed_premise"
    assert provenance["origin"] == "operator"
    assert provenance["decision_state"] == "historical"
    assert provenance["as_of_excluded"] is True
    assert provenance["event_date"] == "2026-01-01"


@pytest.mark.parametrize("flag", [False, "false", 1, [], {}])
def test_malformed_or_false_stale_flag_does_not_trigger(flag):
    lead = build_insight_brief("Question", [memory(grounding_stale=flag)])["candidates"][0]
    assert lead["kind"] == "decision_review"


def test_input_and_output_are_bounded_without_losing_source_alignment():
    rows = [memory(str(i), title="long" * 1000, summary="가" * 2000,
                   source="a" * 5000, content="RAW_SECRET_CONTENT", artifact_location="hidden")
            for i in range(MAX_MEMORIES + 100)]
    result = build_insight_brief("질문" * 2000, rows, max_insights=10000)
    assert len(json.dumps(result)) <= MAX_BRIEF_CHARS
    assert len(result["candidates"]) <= MAX_INSIGHTS
    assert result["truncated"] is True
    assert "RAW_SECRET_CONTENT" not in json.dumps(result)
    assert "artifact_location" not in json.dumps(result)
    for candidate in result["candidates"]:
        assert candidate["source_krefs"] == [e["kref"] for e in candidate["evidence"]]
        assert all(len(e["snippet"]) <= MAX_SNIPPET_CHARS for e in candidate["evidence"])
    assert set(result["source_krefs"]) == {r for c in result["candidates"] for r in c["source_krefs"]}


@pytest.mark.parametrize("rows", [None, 123, "rows", {}, [None, [], "text", 1],
                                  [memory(metadata=[], tags=[{}], contested_by={}, sibling_revisions="bad")]])
def test_malformed_payloads_remain_json_serializable(rows):
    assert isinstance(json.dumps(build_insight_brief("Question", rows)), str)


def test_purity_determinism_and_abstention():
    rows = [memory(contested_by=["kref://p/other.fact?r=1"])]
    original = copy.deepcopy(rows)
    first = build_insight_brief("Question", rows)
    assert first == build_insight_brief("Question", rows)
    assert rows == original
    for query in ("", None, {}, 123):
        assert build_insight_brief(query, rows)["status"] == "insufficient_evidence"
    assert build_insight_brief("Question", rows, max_insights=0)["candidates"] == []


def test_host_instruction_keeps_hypotheses_and_stored_data_separate():
    instruction = build_insight_brief("Question", [])["synthesis_instruction"]
    for phrase in ("hypothesis, not a fact", "not independent corroboration", "untrusted data",
                   "Do not automatically store or promote", "ordinary direct answer"):
        assert phrase in instruction


def test_a_single_large_candidate_is_removed_whole_if_it_exceeds_envelope():
    refs = ["kref://p/" + "가" * 400 + f"{i}.fact?r=1" for i in range(4)]
    rows = [memory(str(i), kref=ref, summary="가" * 1000,
                   contested_by=refs[1:] if i == 0 else [], type="fact")
            for i, ref in enumerate(refs)]
    result = build_insight_brief("질문" * 250, rows)
    assert len(json.dumps(result)) <= MAX_BRIEF_CHARS
    assert result["truncated"] is True
    assert result["candidates"] == []
    assert result["source_krefs"] == []
    assert result["status"] == "insufficient_evidence"


def test_only_bounded_related_refs_are_processed_and_truncation_is_reported():
    refs = [f"kref://p/target-{i}.fact?r=1" for i in range(100)]
    result = build_insight_brief("Question", [memory(contested_by=refs)])
    assert result["candidates"][0]["missing_source_krefs"] == refs[:3]
    assert result["truncated"] is True


def test_proposal_and_storage_date_are_explicitly_not_acceptance_or_validity():
    instruction = build_insight_brief("Question", [])["synthesis_instruction"]
    assert "proposal is not an accepted decision" in instruction
    assert "storage time, not applicability or validity" in instruction


def test_clipped_snippet_and_provenance_are_flagged():
    result = build_insight_brief("Question", [memory(summary="x" * 1000, source="s" * 500)])
    evidence = result["candidates"][0]["evidence"][0]
    assert evidence["snippet_truncated"] is True
    assert evidence["provenance_truncated"] is True
    assert result["truncated"] is True
    assert "Snippets may omit qualifications" in result["synthesis_instruction"]


@pytest.mark.parametrize("rows", [[], [memory()]])
def test_retrieval_failure_is_incomplete_even_when_no_review_leads(rows):
    result = build_insight_brief("Question", rows, retrieval_complete=False)
    assert result["status"] == "retrieval_incomplete"


def test_incomplete_status_is_included_in_envelope_budget():
    rows = [memory(str(i), title="가" * 120, summary="나" * 240) for i in range(5)]
    result = build_insight_brief("질문" * 250, rows, max_insights=5, retrieval_complete=False)
    assert result["status"] == "retrieval_incomplete"
    assert len(json.dumps(result)) <= MAX_BRIEF_CHARS
    assert result["truncated"] is True


def test_exact_primary_sibling_keeps_qualifiers_without_claiming_exact_marker():
    parent = memory(origin="agent", decision_state="proposed", as_of_excluded=True,
                    valid_to="2026-01-01", evidence_level="official",
                    contested_by=["kref://p/other.fact?r=1"])
    parent["sibling_revisions"] = [{"kref": parent["kref"], "summary": "Sparse sibling"}]
    lead = build_insight_brief("Question", [parent])["candidates"][0]
    provenance = lead["evidence"][0]["provenance"]
    assert provenance["origin"] == "agent"
    assert provenance["decision_state"] == "proposed"
    assert provenance["as_of_excluded"] is True
    assert provenance["valid_to"] == "2026-01-01"
    assert provenance["evidence_level"] == "official"
    assert lead["marker_scope"] == "item"
    parent["sibling_revisions"][0]["kref"] = "kref://project/decisions/choice.decision?r=2"
    provenance = build_insight_brief("Question", [parent])["candidates"][0]["evidence"][0]["provenance"]
    assert provenance["origin"] is None
    assert provenance["decision_state"] is None
    assert provenance["as_of_excluded"] is None
    assert provenance["valid_to"] is None
    assert provenance["evidence_level"] == "unverified"
