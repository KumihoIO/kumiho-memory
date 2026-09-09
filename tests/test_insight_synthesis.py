import importlib.util
import json
from pathlib import Path

import pytest

from kumiho_memory.insight_synthesis import (
    prepare_baseline_request, prepare_insight_request, validate_insight_response,
)

REF = "kref://test/decisions/queue.decision?r=1"
OTHER = "kref://test/facts/staff.fact?r=2"


def memory(**kwargs):
    return {"kref": REF, "title": "Queue choice", "summary": "We kept Redis because one person operates the service.", **kwargs}


def response(mode="hypothesis", ref=REF):
    return {"mode": mode, "answer": "Check whether staffing still limits operations.",
            "source_krefs": [ref], "hypotheses": [{
                "statement": "Staffing may still favor a single service.", "source_krefs": [ref],
                "conditions": ["The operator count is unchanged."],
                "alternative_explanation": "Integration cost may dominate instead.",
                "verification_step": "Estimate the additional weekly on-call work.",
                "caveats": ["No measured outcome is included."],
            }] if mode == "hypothesis" else []}


def test_no_candidates_is_not_a_synthesis_gate():
    request = prepare_insight_request("What changed?", [memory(kref=OTHER, type="fact", summary="The team now has five operators.")])
    assert request["status"] == "ready"
    assert request["review_brief"]["candidates"] == []
    assert "no candidates does not" in request["instructions"]
    assert validate_insight_response(request, response("direct", OTHER))["valid"]


def test_source_parity_and_fingerprints():
    request = prepare_insight_request("Should we add a service?", [memory()], current_context="We want to ship soon.", goals=["Low maintenance"])
    baseline = prepare_baseline_request(request)
    for key in ("sources", "source_krefs", "query", "current_context", "goals", "budget", "output_contract"):
        assert request[key] == baseline[key]
    assert "review_brief" not in baseline
    assert request["snapshot_fingerprint"] != baseline["snapshot_fingerprint"]
    assert prepare_insight_request("Should we add a service?", [memory()], current_context="We want to ship soon.", goals=["Low maintenance"])["snapshot_fingerprint"] == request["snapshot_fingerprint"]
    assert validate_insight_response(request, response())["valid"]


def test_references_require_included_pinned_revision_even_when_linked():
    request = prepare_insight_request("Why?", [memory(contested_by=[OTHER])])
    checked = validate_insight_response(request, response(ref=OTHER))
    assert not checked["valid"]
    assert checked["unsupported_source_krefs"] == [OTHER]
    checked = validate_insight_response(request, response(ref=REF.split("?")[0]))
    assert not checked["valid"]


def test_semantic_support_is_never_asserted():
    request = prepare_insight_request("Why?", [memory()])
    answer = response()
    answer["hypotheses"][0]["statement"] = "The moon is made of cheese."
    checked = validate_insight_response(request, answer)
    assert checked["valid"]  # Membership is not entailment.
    assert checked["semantic_support_verified"] is False
    assert checked["validation_scope"] == "structural_only"


def test_snapshot_tampering_rejected():
    request = prepare_insight_request("Why?", [memory()])
    request["sources"][0]["summary"] = "A fabricated source."
    assert not validate_insight_response(request, response())["valid"]


@pytest.mark.parametrize("field", ["conditions", "alternative_explanation", "verification_step", "caveats", "statement", "source_krefs"])
def test_hypothesis_contract_fields_required(field):
    request = prepare_insight_request("Why?", [memory()])
    answer = response()
    answer["hypotheses"][0].pop(field)
    assert not validate_insight_response(request, answer)["valid"]


def test_direct_and_clarify_allowed_without_sources():
    request = prepare_insight_request("What is 2+2?", [])
    assert request["status"] == "no_sources"
    for mode in ("direct", "clarify"):
        assert validate_insight_response(request, {"mode": mode, "answer": "4", "source_krefs": [], "hypotheses": []})["valid"]


def test_source_budget_and_no_raw_artifacts():
    request = prepare_insight_request("Why?", [memory(summary="a" * 20000, content="private raw artifact", ignored="nope")], max_source_chars=512)
    assert request["budget"]["source_chars"] <= 512
    assert request["budget"]["source_text_truncated"]
    assert "content" not in request["sources"][0]
    assert "ignored" not in request["sources"][0]
    assert request["sources"][0]["summary"]


def test_sibling_scope_and_duplicate_identity():
    request = prepare_insight_request("Why?", [memory(grounding_stale=True, sibling_revisions=[{"kref": OTHER, "summary": "A different revision"}]), memory(kref=OTHER)])
    assert request["source_krefs"] == [OTHER]
    assert request["sources"][0]["item_markers"]["grounding_stale"] is True
    assert "grounding_stale" not in request["sources"][0]
    assert "title" not in request["sources"][0]


def test_privacy_precedes_truncation_and_each_atom_is_screened():
    secret = "sk-proj-" + "A" * 24
    request = prepare_insight_request("email me at person@example.com", [memory(summary="a" * 10000 + " " + secret, title="Safe title", origin=secret, tags=[secret, "fact"], content=secret)], current_context=secret, goals=[secret, "Email second@example.org"])
    encoded = json.dumps(request)
    assert secret not in encoded
    assert "person@example.com" not in encoded
    assert "second@example.org" not in encoded
    assert request["sources"][0]["summary"] == ""
    assert request["sources"][0]["title"] == "Safe title"
    assert request["budget"]["credential_atoms_dropped"] == 5


def test_credential_ref_is_dropped():
    secret_ref = "kref://test/sk-proj-" + "A" * 24 + ".fact?r=1"
    request = prepare_insight_request("Why?", [memory(kref=secret_ref)])
    assert request["sources"] == []


def test_unstructured_unsupported_citation_detected():
    request = prepare_insight_request("Why?", [memory()])
    answer = response()
    answer["answer"] += " " + OTHER
    assert validate_insight_response(request, answer)["unsupported_source_krefs"] == [OTHER]


def test_incomplete_retrieval_is_preserved():
    request = prepare_insight_request("Why?", [memory()], retrieval_complete=False)
    assert request["status"] == "retrieval_incomplete"
    assert request["review_brief"]["status"] == "retrieval_incomplete"


def _harness():
    spec = importlib.util.spec_from_file_location("evaluate_insight", Path(__file__).resolve().parents[1] / "scripts/evaluate_insight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_requires_external_labels_and_source_parity():
    request = prepare_insight_request("Why?", [memory()])
    case = {"id": "sample", "request": request, "baseline_request": prepare_baseline_request(request), "baseline_response": response("direct"), "insight_response": response()}
    harness = _harness()
    report = harness.evaluate_cases([case])
    assert report["scored_pairs"] == 0
    assert report["arms"]["insight"]["mean_usefulness"] is None
    case["labels"] = {"reviewer": "independent-reviewer", "independent": True, "blind": True, "insight_warranted": True, "baseline": {"usefulness": 2, "grounding": 4, "unsupported_claims": 0}, "insight": {"usefulness": 3, "grounding": 3, "unsupported_claims": 1}}
    report = harness.evaluate_cases([case])
    assert report["paired"]["usefulness"]["wins"] == 1
    assert report["paired"]["grounding"]["regressions"] == 1
    assert report["arms"]["baseline"]["mode_label_agreement"] == 0
    assert report["arms"]["insight"]["mode_label_agreement"] == 1
    case["baseline_request"]["sources"] = []
    assert harness.evaluate_cases([case])["scored_pairs"] == 0


def test_synthetic_fixtures_are_behavior_expectations_not_quality_scores():
    path = Path(__file__).parent / "fixtures/insight_scenarios.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 7
    for case in cases:
        assert case["expected_behavior"]
        request = prepare_insight_request(case["query"], case["memories"], current_context=case.get("current_context", ""))
        assert prepare_baseline_request(request)["sources"] == request["sources"]


@pytest.mark.parametrize("mutation", [None, {"sources": None}, {"source_krefs": [{}]}, {"sources": [{"kref": []}]}])
def test_malformed_requests_return_errors(mutation):
    request = prepare_insight_request("Why?", [memory()])
    if mutation is not None:
        request.update(mutation)
    else:
        request = None
    assert not validate_insight_response(request, response())["valid"]


def test_prose_citation_punctuation_is_not_an_unsupported_ref():
    request = prepare_insight_request("Why?", [memory()])
    answer = response()
    answer["answer"] += " (" + REF + ")."
    assert validate_insight_response(request, answer)["valid"]


def test_contract_mutation_does_not_change_future_requests():
    request = prepare_insight_request("Why?", [memory()])
    request["output_contract"]["mode"] = "broken"
    assert prepare_insight_request("Why?", [memory()])["output_contract"]["mode"] != "broken"


def test_sibling_parent_leads_keep_item_scope_and_sanitized_evidence():
    secret = "sk-proj-" + "A" * 24
    request = prepare_insight_request("Why?", [memory(
        summary=secret, origin="user", grounding_stale=True,
        contested_by=[REF], grounding_stale_superseded_by=REF,
        sibling_revisions=[{"kref": OTHER, "title": "Safe sibling", "summary": "Its own evidence"}],
    )])
    leads = request["review_brief"]["candidates"]
    assert {lead["kind"] for lead in leads} == {"changed_premise", "unresolved_conflict"}
    assert all(lead["marker_scope"] == "item" for lead in leads)
    assert all(lead["source_krefs"] == [OTHER] for lead in leads)
    assert all(lead["evidence"][0]["provenance"]["origin"] is None for lead in leads)
    assert all(lead["evidence"][0]["snippet"] == "Its own evidence" for lead in leads)
    assert secret not in json.dumps(request)
