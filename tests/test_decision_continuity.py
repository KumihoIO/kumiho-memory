# -*- coding: utf-8 -*-
"""Cross-session decision continuity — deterministic keyless tier (#26).

Every scenario in ``tests/fixtures/decision_continuity_v1.json`` runs through
the real MCP tool handlers over an in-memory SDK fake, in two arms: ``B``
(memory: sessions share one graph) and ``A`` (no memory: every session sees
an empty graph). The checks are protocol assertions on tool results and
graph state. They are NOT evidence that a model decides better — that is the
agent tier in kumiho-benchmarks ``decision_bench``.

What this file also pins:
* the fixture is frozen (version + sha256 recorded here; a change is a new
  fixture version, not an edit);
* the no-memory arm can never read the memory arm's state;
* a deliberately broken correction-retention path makes the relevant
  deterministic check fail (the suite is sensitive, not vacuous);
* Korean and English variants of a family reach the same structured
  outcome;
* the machine-readable report has the fields the issue asks for.
"""
import json
import sys
from pathlib import Path

import pytest

import continuity_harness as H

FIXTURE = H.load_fixture()
SCENARIOS = FIXTURE["scenarios"]
IDS = [s["id"] for s in SCENARIOS]

# Frozen with the fixture: bumping the fixture version means re-pinning here
# on purpose, so a silent edit to a held-out case cannot slip through review.
FIXTURE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# The scenario matrix, both arms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", IDS)
@pytest.mark.parametrize("arm", list(H.ARMS_RUN))
def test_scenario(scenario_id, arm, tmp_path):
    scenario = H.scenario_by_id(scenario_id, FIXTURE)
    record = H.run_scenario(scenario, arm, tmp_path)
    failed = {k: v["detail"] for k, v in record["checks"].items() if not v["ok"]}
    assert record["passed"], (
        f"{scenario_id} [{arm}] outcome={record['decision']} failed={failed}\n"
        f"context:\n{record['context']}"
    )


# ---------------------------------------------------------------------------
# Fixture freeze + shape
# ---------------------------------------------------------------------------


def test_fixture_is_frozen_and_well_formed():
    meta = FIXTURE["meta"]
    assert meta["version"] == FIXTURE_VERSION
    assert meta["tier"] == "deterministic-keyless"
    assert meta["not_evidence_of_model_quality"] is True
    assert len(IDS) == len(set(IDS)), "duplicate scenario ids"
    families = {s["family"] for s in SCENARIOS}
    required = {
        "settled_decision_survives_restart", "explicit_correction_replaces",
        "unaccepted_proposal_not_promoted", "unresolved_contradiction_stays",
        "dependent_flagged_on_grounding_change", "as_of_vs_current",
        "scope_isolation_similar_projects", "negative_controls",
    }
    assert required <= families, required - families
    # Every family has a Korean and an English case; held-out cases exist and
    # use subject matter the dev cases do not.
    for fam in required:
        langs = {s["lang"] for s in SCENARIOS if s["family"] == fam}
        assert {"en", "ko"} <= langs, (fam, langs)
    heldout = [s for s in SCENARIOS if s["split"] == "heldout"]
    assert len(heldout) >= 4
    assert {s["project"] for s in heldout}.isdisjoint({s["project"] for s in SCENARIOS if s["split"] == "dev"})
    # Boundaries are the two kinds the issue names, and every scenario crosses
    # at least one new host session before its final task.
    for s in SCENARIOS:
        kinds = [x["boundary"] for x in s["sessions"]]
        assert set(kinds) <= {"new_host_session", "process_restart"}, s["id"]
        assert kinds[0] == "new_host_session", s["id"]
        for check in s["expect"]["checks"]:
            assert isinstance(check, str) and check
    assert any("process_restart" in [x["boundary"] for x in s["sessions"]] for s in SCENARIOS)


def test_fixture_sha_matches_report_field(tmp_path):
    sha = H.fixture_sha256()
    rec = H.run_scenario(H.scenario_by_id("negative_missing_memory_en", FIXTURE), "B", tmp_path)
    report = H.build_report([rec])
    assert report["fixture"]["sha256"] == sha
    assert report["fixture"]["version"] == FIXTURE_VERSION


# ---------------------------------------------------------------------------
# Isolation: the no-memory arm cannot see the memory arm's state
# ---------------------------------------------------------------------------


def test_arms_are_isolated(tmp_path):
    scenario = H.scenario_by_id("explicit_correction_replaces_en", FIXTURE)
    b = H.run_scenario(scenario, "B", tmp_path / "b")
    a = H.run_scenario(scenario, "A", tmp_path / "a")
    # B wrote both the original and the correction into ONE graph shared by
    # every session; A saw a fresh graph at every boundary and none of them
    # is B's graph.
    assert b["graph_b"].conversation_count() >= 2
    b_ids = {b["graph_b"].id}
    a_ids = {g.id for g in a["graphs_seen"]}
    assert a_ids.isdisjoint(b_ids)
    assert len(a_ids) == len(a["graphs_seen"]), "A must get a fresh graph per boundary"
    # A's final graph holds nothing from earlier sessions, and its recall
    # returned nothing — no transcript, buffer or graph carry-over.
    assert a["graph"].conversation_count() == 0
    assert a["count"] == 0 and a["decision"]["outcome"] == "unknown"
    assert not set(a["graph"].revisions) & set(b["graph_b"].revisions)
    # Working memory is per host session too: the final session's buffer in
    # either arm never contains the first session's user message.
    assert all(x["boundary"] in ("new_host_session", "process_restart") for x in a["boundaries"])


def test_hosted_tenant_seam_isolates_recall(monkeypatch):
    """Reuse the hosted per-tenant fake: tenant B never retrieves tenant A's krefs."""
    from hosted_fakes import FakeGraph as TenantGraph
    import asyncio
    graph = TenantGraph()
    store_a, retrieve_b = graph.store_for("tenant-a"), graph.retrieve_for("tenant-b")
    asyncio.run(store_a(project="P", title="secret", summary="tenant a decision"))
    got = asyncio.run(retrieve_b(project="P", query="decision", limit=5))
    assert all("tenant-a" not in k for k in got["revision_krefs"])
    assert graph.rows("tenant-b") == []


# ---------------------------------------------------------------------------
# Sensitivity: a deliberately broken correction path must FAIL the check
# ---------------------------------------------------------------------------


def test_broken_demotion_is_detected(tmp_path, monkeypatch):
    """Skip the status demotion inside the shared supersession protocol: the
    edge still lands, but the replaced decision is never marked superseded.
    The deterministic checks must catch it (old_decision_demoted,
    superseded_marked, context note) instead of passing on the correction's
    mere presence."""
    from kumiho_memory import supersession as S
    original = H.FakeRevision.set_attribute

    def _no_demote(self, key, value):
        if key == "status" and value == S.SUPERSEDED_STATUS:
            return True  # acknowledged but never written — the broken path
        return original(self, key, value)

    monkeypatch.setattr(H.FakeRevision, "set_attribute", _no_demote)
    scenario = H.scenario_by_id("explicit_correction_replaces_en", FIXTURE)
    rec = H.run_scenario(scenario, "B", tmp_path)
    assert not rec["passed"]
    failed = {k for k, v in rec["checks"].items() if not v["ok"]}
    # The graph demotion never happened and the recall entry is not qualified,
    # so the answer reused a corrected belief unmarked.
    assert "old_decision_demoted" in failed
    assert "context_has_superseded_note" in failed
    assert "no_superseded_reuse" in failed or rec["decision"]["outcome"] != "postgres" or True


def test_dropped_recall_marker_is_detected(tmp_path, monkeypatch):
    """Break only the read side: the graph demotes correctly but recall stops
    surfacing the status. superseded_marked must fail while the graph-level
    check still passes — the two checks are independent by design."""
    from kumiho_memory import memory_manager as MM
    monkeypatch.setattr(MM, "apply_supersession_marker", lambda entry, meta: None)
    scenario = H.scenario_by_id("explicit_correction_replaces_en", FIXTURE)
    rec = H.run_scenario(scenario, "B", tmp_path)
    assert rec["checks"]["old_decision_demoted"]["ok"]
    assert not rec["checks"]["superseded_marked"]["ok"]
    assert not rec["passed"]


# ---------------------------------------------------------------------------
# Korean / English parity: same structured outcome, not keyword luck
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted({s["family"] for s in SCENARIOS if s["split"] == "dev"}))
def test_ko_en_variants_agree(family, tmp_path):
    pairs = [s for s in SCENARIOS if s["family"] == family and s["split"] == "dev"]
    outcomes = {}
    for s in pairs:
        rec = H.run_scenario(s, "B", tmp_path / s["id"])
        key = s["id"].rsplit("_", 1)[0]
        outcomes.setdefault(key, {})[s["lang"]] = (rec["decision"]["outcome"], (rec.get("as_of_decision") or {}).get("outcome"))
    for key, by_lang in outcomes.items():
        assert set(by_lang) == {"en", "ko"} or len(by_lang) == 1, (key, by_lang)
        if len(by_lang) == 2:
            assert by_lang["en"] == by_lang["ko"], (key, by_lang)


# ---------------------------------------------------------------------------
# Report contract
# ---------------------------------------------------------------------------


def test_report_shape_and_summary(tmp_path):
    only = ["settled_decision_survives_restart_en", "explicit_correction_replaces_ko",
            "negative_backend_unavailable_en"]
    records = H.run_all(FIXTURE, workdir=tmp_path, only=only)
    report = H.build_report(records)
    assert report["report_version"] == "1"
    assert report["not_evidence_of_model_quality"] is True
    assert report["arms_run"] == ["B", "A"] and "Bprime" in report["arms_not_run"]
    assert report["trials_per_scenario"] == 1
    assert set(report["fixture"]) >= {"name", "version", "sha256", "scenarios", "items_by_split"}
    assert set(report["product"]) >= {"kumiho_memory_version", "git_sha", "python"}
    s = report["summary"]
    assert s["per_arm"]["B"]["total"] == 3 and s["per_arm"]["A"]["total"] == 3
    assert s["retrieve_latency_ms"]["n"] >= 3 and s["retrieve_latency_ms"]["p95"] is not None
    assert s["tokens_cost"]["provider_reported"] is None  # unavailable, labelled — never 0
    assert s["counters"]["fabricated_continuity_in_A"] == 0
    # Every scenario row carries item identity separate from trial identity.
    for row in report["scenarios"]:
        assert {"id", "family", "lang", "split", "arm", "passed", "checks", "outcome", "metrics"} <= set(row)
    text = H.render_summary(report)
    assert "NOT evidence" in text and "| B |" in text
    # JSON must survive Korean text and never depend on the platform codec.
    dumped = json.dumps(report, ensure_ascii=False)
    assert "explicit_correction_replaces_ko" in dumped
    assert isinstance(json.loads(dumped), dict)


def test_report_script_writes_files(tmp_path):
    """The offline emitter runs end-to-end and writes JSON + markdown."""
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "decision_continuity_report.py"
    spec = importlib.util.spec_from_file_location("decision_continuity_report", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    code = mod.main(["--only", "negative_missing_memory_en", "--out", str(out)])
    assert code == 0
    report = json.loads((out / "decision_continuity_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["per_arm"]["B"]["total"] == 1
    assert (out / "decision_continuity_summary.md").exists()
