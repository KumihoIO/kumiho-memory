"""Offline boundary tests: explicit observations, append-only lineage and privacy."""
import asyncio
from functools import wraps
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kumiho_memory.experience import (
    experience_from_memory, normalize_experience, record_experience,
    record_outcome, scoped_space, validate_source_refs,
)
from kumiho_memory.privacy import CredentialDetectedError

def run_async(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


REF = "kref://project/experiences/pilot.experience?r=1"
SOURCE = "kref://project/decisions/pilot.decision?r=1"


def experience(**kwargs):
    return dict(experience_id="pilot-run-a", title="Try a pilot", situation="Small team",
                goal="Release on time", decision="Run a limited pilot", rationale="Bound upkeep",
                expected_outcome="Know support cost", alternatives=["Full rollout"],
                applicability_conditions=["Team size stays small"], **kwargs)


@pytest.fixture
def manager():
    return SimpleNamespace(project="project", memory_store=AsyncMock(return_value={"revision_kref": REF}))


def stored_row(record):
    snapshot = {**normalize_experience(record), "recorded_at": "2026-09-09T00:00:00+00:00"}
    return {"kref": REF, "metadata": {"experience_record": json.dumps(snapshot)}}


def mock_revisions(monkeypatch, rows):
    import kumiho
    def read(ref):
        row = rows[ref]
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=row["metadata"], tags=[], deprecated=row.get("revision_deprecated", False))
    monkeypatch.setattr(kumiho, "get_revision", read)
    def read_item(ref):
        row = next(row for key, row in rows.items() if key.split("?")[0] == ref)
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=row.get("item_markers", {}), deprecated=False)
    monkeypatch.setattr(kumiho, "get_item", read_item)


def test_normalize_distinguishes_acceptance_and_result():
    result = normalize_experience(experience(acceptance="accepted", decision_state="accepted"))
    assert result["outcome_status"] == "unknown"
    assert result["observed_at"] == ""
    assert result["record_id"] == normalize_experience(experience(acceptance="accepted", decision_state="accepted"))["record_id"]
    other = experience(acceptance="accepted", decision_state="accepted")
    other["experience_id"] = "pilot-run-b"
    assert result["record_id"] != normalize_experience(other)["record_id"]


@pytest.mark.parametrize("update", [
    {"observed_outcome": "It worked"}, {"observed_at": "2026-09-09T00:00:00Z"},
    {"outcome_status": "success"}, {"observed_outcome": "It worked", "observed_at": "2026-09-09"},
    {"source_krefs": [SOURCE.split("?")[0]]}, {"confidence": 0.99},
    {"alternatives": ["x"] * 13}, {"rationale": "x" * 2001},
    {"experience_id": ""},
])
def test_invalid_extractions_fail_closed(update):
    record = experience()
    record.update(update)
    with pytest.raises(ValueError):
        normalize_experience(record)


@run_async
@pytest.mark.parametrize("field", ["title", "rationale", "alternatives", "applicability_conditions", "experience_id"])
async def test_credentials_in_any_atom_prevent_all_writes(manager, field):
    record = experience()
    secret = "sk-" + "a" * 32
    record[field] = [secret] if isinstance(record.get(field), list) else secret
    with pytest.raises(CredentialDetectedError):
        await record_experience(manager, record)
    manager.memory_store.assert_not_called()


@run_async
async def test_store_redacts_every_prose_atom_and_does_not_stack(manager):
    record = experience()
    record["title"] = "Ask person@example.com"
    record["alternatives"] = ["Email person@example.com"]
    result = await record_experience(manager, record)
    payload = manager.memory_store.call_args.kwargs
    assert "person@example.com" not in json.dumps(payload)
    assert "[email]" in payload["title"]
    assert payload["memory_item_kind"] == "experience"
    assert payload["stack_revisions"] is False
    assert result["idempotent"] is False
    assert result["belief_promoted"] is False
    assert payload["metadata"]["evidence_level"] == "unverified"
    assert result["record"]["recorded_at"] and result["record"]["observed_at"] == ""


@run_async
async def test_outcome_is_separate_pinned_observation(manager, monkeypatch):
    original = stored_row(experience())
    before = copy.deepcopy(original)
    mock_revisions(monkeypatch, {REF: original})
    result = await record_outcome(manager, REF, {
        "observed_outcome": "Support exceeded two days", "observed_at": "2026-09-08T18:00:00+09:00",
        "outcome_status": "failure", "acceptance": "accepted", "origin": "user",
    })
    payload = manager.memory_store.call_args.kwargs
    assert original == before
    assert payload["source_revision_krefs"] == [REF]
    assert payload["memory_type"] == "outcome"
    assert result["record"]["outcome_status"] == "failure"
    assert result["record"]["acceptance"] == "accepted"
    assert result["record"]["observed_at"] == "2026-09-08T09:00:00+00:00"
    assert result["record"]["recorded_at"] != result["record"]["observed_at"]
    decoded = experience_from_memory({"kref": REF.replace("pilot.", "outcome."), "metadata": payload["metadata"]})
    assert decoded["record_type"] == "outcome"


@run_async
async def test_source_scope_rejects_cross_project_before_fetch(manager, monkeypatch):
    import kumiho
    read = AsyncMock()
    monkeypatch.setattr(kumiho, "get_revision", read)
    def read_item(ref):
        row = next(row for key, row in rows.items() if key.split("?")[0] == ref)
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=row.get("item_markers", {}), deprecated=False)
    monkeypatch.setattr(kumiho, "get_item", read_item)
    with pytest.raises(ValueError):
        await record_experience(manager, experience(source_krefs=[SOURCE.replace("project/", "other/")]))
    read.assert_not_called()
    manager.memory_store.assert_not_called()


@run_async
async def test_inaccessible_source_prevents_write(manager, monkeypatch):
    mock_revisions(monkeypatch, {})
    with pytest.raises(ValueError, match="inaccessible"):
        await record_experience(manager, experience(source_krefs=[SOURCE]))
    manager.memory_store.assert_not_called()


@run_async
async def test_source_narrow_scope_rejects_sibling_prefix(manager, monkeypatch):
    mock_revisions(monkeypatch, {})
    with pytest.raises(ValueError, match="requested spaces"):
        await validate_source_refs(manager, [SOURCE], space_paths=["project/decision"])


@run_async
async def test_nonexperience_parent_rejected(manager, monkeypatch):
    mock_revisions(monkeypatch, {SOURCE: {"metadata": {"summary": "Some claim"}}})
    with pytest.raises(ValueError, match="canonical"):
        await record_outcome(manager, SOURCE, {"observed_outcome": "Worked", "observed_at": "2026-09-09T00:00:00Z"})
    manager.memory_store.assert_not_called()


def test_read_snapshot_roundtrip_and_tampering():
    row = stored_row(experience())
    assert experience_from_memory(row)["experience_id"] == "pilot-run-a"
    record = json.loads(row["metadata"]["experience_record"])
    record["decision"] = "Tampered decision"
    row["metadata"]["experience_record"] = json.dumps(record)
    assert experience_from_memory(row) is None
    assert experience_from_memory({"kref": REF, "summary": "An experience"}) is None


def test_space_paths_are_project_scoped(manager):
    assert scoped_space(manager) == "/project/experiences"
    assert scoped_space(manager, "project/team") == "/project/team"
    assert scoped_space(manager, "team") == "/project/team"
    for path in ("/other/team", "project/../other", "project//team"):
        with pytest.raises(ValueError):
            scoped_space(manager, path)


@run_async
async def test_hosted_missing_context_fails_closed(manager, monkeypatch):
    monkeypatch.setenv("KUMIHO_MCP_HOSTED", "1")
    with pytest.raises(ValueError, match="active tenant"):
        await record_experience(manager, experience())
    manager.memory_store.assert_not_called()


def test_identifier_pii_rejected_instead_of_corrupting_reference():
    with pytest.raises(ValueError, match="personal information"):
        normalize_experience(experience(source_krefs=["kref://project/team/a@example.com.fact?r=1"]))


def test_source_paths_cannot_traverse():
    with pytest.raises(ValueError):
        normalize_experience(experience(source_krefs=["kref://project/../other/a.fact?r=1"]))


@run_async
async def test_hosted_manager_tenant_mismatch_rejected(manager, monkeypatch):
    from kumiho_memory import experience as module
    manager.redis_buffer = SimpleNamespace(tenant_id="tenant-a")
    monkeypatch.setattr(module, "current_request", lambda: SimpleNamespace(tenant_id="tenant-b"))
    with pytest.raises(ValueError, match="tenant"):
        await record_experience(manager, experience())
    manager.memory_store.assert_not_called()


@run_async
async def test_backend_error_does_not_claim_success(manager):
    manager.memory_store.return_value = {"error": "backend error"}
    with pytest.raises(RuntimeError, match="partial write"):
        await record_experience(manager, experience())


@run_async
async def test_same_report_stable_deduplication_key(manager):
    first = await record_experience(manager, experience())
    second = await record_experience(manager, experience())
    assert first["deduplication_key"] == second["deduplication_key"]
    assert manager.memory_store.call_count == 2  # append semantics are explicit


def test_redacted_record_decodes_without_hash_drift():
    data = experience()
    data["rationale"] = "Contact person@example.com about support"
    row = stored_row(data)
    assert experience_from_memory(row)["rationale"] == "Contact [email] about support"


@run_async
async def test_source_item_markers_do_not_replace_revision_provenance(manager, monkeypatch):
    rows = {SOURCE: {"metadata": {"origin": "user", "evidence_level": "single_source"},
                     "item_markers": {"grounding_stale": "true", "origin": "agent", "evidence_level": "official"}}}
    mock_revisions(monkeypatch, rows)
    result = (await validate_source_refs(manager, [SOURCE]))[0]
    assert result["metadata"]["origin"] == "user"
    assert result["metadata"]["evidence_level"] == "single_source"
    assert result["item_markers"] == {"grounding_stale": "true"}


@run_async
async def test_source_item_identity_mismatch_fails_closed(manager, monkeypatch):
    import kumiho
    mock_revisions(monkeypatch, {SOURCE: {"metadata": {}}})
    monkeypatch.setattr(kumiho, "get_item", lambda ref: SimpleNamespace(kref=SimpleNamespace(uri="kref://other/x.fact"), metadata={}))
    with pytest.raises(ValueError, match="item state is inaccessible"):
        await validate_source_refs(manager, [SOURCE])


@run_async
async def test_sdk_revision_deprecated_preserved_independently(manager, monkeypatch):
    rows = {SOURCE: {"metadata": {"origin": "user", "deprecated": "false"}, "revision_deprecated": True}}
    mock_revisions(monkeypatch, rows)
    result = (await validate_source_refs(manager, [SOURCE]))[0]
    assert result["revision_deprecated"] is True
    assert result["metadata"]["deprecated"] == "false"
    assert not result["item_markers"].get("deprecated", False)


@run_async
async def test_sdk_revision_deprecated_malformed_is_inaccessible(manager, monkeypatch):
    mock_revisions(monkeypatch, {SOURCE: {"metadata": {}, "revision_deprecated": "unknown"}})
    with pytest.raises(ValueError, match="inaccessible"):
        await validate_source_refs(manager, [SOURCE])


@pytest.mark.parametrize("target", [
    "/project/experiences/sk-proj-" + "a" * 24,
    "/project/person@example.com",
])
@run_async
async def test_target_space_identifiers_reject_secrets_and_pii_before_write(manager, target):
    with pytest.raises(ValueError):
        await record_experience(manager, experience(), space_path=target)
    manager.memory_store.assert_not_called()


@run_async
async def test_project_identifier_credential_rejected_before_write(manager):
    manager.project = "sk-proj-" + "a" * 24
    with pytest.raises(CredentialDetectedError):
        await record_experience(manager, experience())
    manager.memory_store.assert_not_called()


@run_async
async def test_source_scope_identifier_credential_rejected_before_sdk_read(manager, monkeypatch):
    import kumiho
    monkeypatch.setattr(kumiho, "get_revision", lambda ref: pytest.fail("SDK read attempted"))
    with pytest.raises(CredentialDetectedError):
        await validate_source_refs(manager, [SOURCE], space_paths=["/project/sk-proj-" + "a" * 24])
