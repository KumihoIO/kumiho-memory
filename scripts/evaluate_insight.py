"""Describe paired captured host responses; never generate or auto-grade answers.

Usage: python scripts/evaluate_insight.py --input private-captures.json --output report.json
Input: {"cases": [{"id": str, "request": enriched_request,
 "baseline_request": baseline_request, "baseline_response": object,
 "insight_response": object, "labels": {"reviewer": str, "independent": true,
 "blind": true, "insight_warranted": bool,
 "baseline": {"usefulness": 0..4, "grounding": 0..4, "unsupported_claims": int},
 "insight": {"usefulness": 0..4, "grounding": 0..4, "unsupported_claims": int}},
 "timing_ms": {"baseline": number, "insight": number}}]}
Labels must be assigned after blinded independent review of source text and both
answers. The harness checks label shape, not reviewer independence or honesty.
Unlabelled captures get structural results only. No keyword quality scoring.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kumiho_memory.insight_synthesis import validate_insight_response


def _chars(value):
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _response(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _labels_valid(labels):
    if not isinstance(labels, dict) or labels.get("independent") is not True or labels.get("blind") is not True or not isinstance(labels.get("reviewer"), str) or not labels["reviewer"].strip() or type(labels.get("insight_warranted")) is not bool:
        return False
    for arm in ("baseline", "insight"):
        score = labels.get(arm)
        if not isinstance(score, dict):
            return False
        if any(type(score.get(key)) is not int or not 0 <= score[key] <= 4 for key in ("usefulness", "grounding")):
            return False
        if type(score.get("unsupported_claims")) is not int or score["unsupported_claims"] < 0:
            return False
    return True


def evaluate_cases(cases):
    """Produce descriptive metrics; small convenience samples do not generalize."""
    rows = []
    for case in cases:
        request = case["request"]
        baseline = case["baseline_request"]
        parity_fields = ("query", "current_context", "goals", "sources", "source_krefs", "output_contract", "budget", "status")
        parity = all(request.get(field) == baseline.get(field) for field in parity_fields)
        validation = {
            "baseline": validate_insight_response(baseline, case.get("baseline_response")),
            "insight": validate_insight_response(request, case.get("insight_response")),
        }
        labels = case.get("labels")
        # Invalid outputs remain in scored comparisons: excluding them would
        # inflate quality. Only invalid snapshots/source parity disqualify a pair.
        snapshot_ok = all(not any("request " in error for error in arm["errors"]) for arm in validation.values())
        scored = parity and snapshot_ok and _labels_valid(labels)
        row = {
            "id": case.get("id", str(len(rows))), "source_parity": parity,
            "scored": scored, "validation": validation,
            "request_chars": {"baseline": _chars(baseline), "insight": _chars(request)},
            "guidance_overhead_chars": _chars(request) - _chars(baseline),
            "response_chars": {arm: _chars(case.get(arm + "_response")) for arm in ("baseline", "insight")},
        }
        if scored:
            row["scores"] = {arm: labels[arm] for arm in ("baseline", "insight")}
            row["usefulness_delta"] = labels["insight"]["usefulness"] - labels["baseline"]["usefulness"]
            row["grounding_delta"] = labels["insight"]["grounding"] - labels["baseline"]["grounding"]
            row["non_hypothesis_mode_matches_label"] = {arm: ((_response(case.get(arm + "_response")).get("mode") in ("direct", "clarify")) == (not labels["insight_warranted"])) for arm in ("baseline", "insight")}
        timing = case.get("timing_ms", {})
        if isinstance(timing, dict) and all(type(timing.get(arm)) in (int, float) and math.isfinite(timing[arm]) and timing[arm] >= 0 for arm in ("baseline", "insight")):
            row["timing_ms"] = timing
        rows.append(row)
    scored = [row for row in rows if row["scored"]]
    report = {
        "schema_version": 1, "cases": len(rows), "scored_pairs": len(scored),
        "source_parity_failures": sum(not row["source_parity"] for row in rows),
        "quality_basis": "externally supplied blinded independent rubric labels; independence is asserted by caller, not verified",
        "limitations": ["Descriptive convenience-sample comparison, not a benchmark or causal estimate.",
                       "Citation membership validation does not verify semantic support.",
                       "Same captured source text and source budget; enriched guidance has additional overhead.",
                       "Mode-label agreement is only a hypothesis-vs-direct/clarify format proxy, not correct abstention or usefulness; direct answers can contain useful insight.",
                       "No model calls, keyword quality scoring, or semantic grading are performed by this script."],
        "mean_guidance_overhead_chars": mean([row["guidance_overhead_chars"] for row in rows]) if rows else None,
        "arms": {}, "paired": {}, "results": rows,
    }
    for arm in ("baseline", "insight"):
        report["arms"][arm] = {
            "structurally_valid": sum(row["validation"][arm]["valid"] for row in rows),
            "unsupported_reference_occurrences": sum(len(row["validation"][arm]["unsupported_source_krefs"]) for row in rows),
            "mean_usefulness": mean([row["scores"][arm]["usefulness"] for row in scored]) if scored else None,
            "mean_grounding": mean([row["scores"][arm]["grounding"] for row in scored]) if scored else None,
            "unsupported_claims_reviewed": sum(row["scores"][arm]["unsupported_claims"] for row in scored) if scored else None,
            "mode_label_agreement": mean([row["non_hypothesis_mode_matches_label"][arm] for row in scored]) if scored else None,
        }
    for metric in ("usefulness", "grounding"):
        deltas = [row[metric + "_delta"] for row in scored]
        report["paired"][metric] = {"wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas), "regressions": sum(d < 0 for d in deltas), "mean_delta": mean(deltas) if deltas else None}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    report = evaluate_cases(payload["cases"])
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
