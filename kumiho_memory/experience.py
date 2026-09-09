"""Host-extracted experience snapshots and separate outcome observations.

These records preserve reports, not established beliefs. No model call or belief
promotion occurs here. Stores append unpublished snapshots; record_id permits
content deduplication by consumers but is NOT a backend idempotency guarantee.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from datetime import datetime, timezone
from typing import Any

from ._request_context import current_request, is_hosted
from .privacy import PIIRedactor

SCHEMA = "kumiho.experience.v1"
MAX_TEXT = 2000
MAX_LIST = 12
MAX_RECORD_CHARS = 24000
_REF = re.compile(r"kref://([^/\s?#]+)/([^\s?#]+)\?r=[1-9][0-9]*")


def _text(value: Any, name: str, *, required: bool = False, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} must be a string of at most {limit} characters")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{name} is required")
    return value


def _enum(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Invalid {name}")
    return value


def _texts(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_LIST:
        raise ValueError(f"{name} must be a list of at most {MAX_LIST} strings")
    return list(dict.fromkeys(_text(v, name, required=True) for v in value))


def _refs(value: Any) -> list[str]:
    refs = _texts(value, "source_krefs")
    if any(len(ref) > 512 or not _REF.fullmatch(ref)
           or any(part in ("", ".", "..") for part in ref.removeprefix("kref://").split("?")[0].split("/"))
           for ref in refs):
        raise ValueError("source_krefs must contain revision-pinned krefs")
    redactor = PIIRedactor()
    for ref in refs:
        redactor.reject_credentials(ref)
        if redactor.anonymize_summary(ref.split("?", 1)[0]) != ref.split("?", 1)[0]:
            raise ValueError("Identifiers must not contain personal information")
    return refs


def _timestamp(value: Any, name: str) -> str:
    value = _text(value, name, required=True, limit=64)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} must be an ISO timestamp") from None
    if dt.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return dt.astimezone(timezone.utc).isoformat()


def sanitize_atoms(value: Any, *, _depth: int = 0) -> Any:
    """Screen each full string for credentials BEFORE PII rewriting.

    A credential rejects the record atomically; no prefix truncation or partial
    write can bypass screening. Dictionary keys are schema-owned by callers.
    """
    if _depth > 8:
        raise ValueError("Record nesting exceeds limit")
    redactor = PIIRedactor()
    if isinstance(value, str):
        redactor.reject_credentials(value)
        return redactor.anonymize_summary(value)
    if isinstance(value, list):
        return [sanitize_atoms(v, _depth=_depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_atoms(v, _depth=_depth + 1) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError("Unsupported record value")


def _finish(record: dict) -> dict:
    # References and timestamps are identifiers, so reject secrets but do not
    # apply phone-number PII rewriting to revision numbers or timestamp digits.
    identifiers = {key: record[key] for key in (
        "source_krefs", "experience_kref", "observed_at", "experience_id"
    ) if key in record}
    safe = sanitize_atoms({k: v for k, v in record.items() if k not in identifiers})
    redactor = PIIRedactor()
    for key, value in identifiers.items():
        for atom in value if isinstance(value, list) else [value]:
            redactor.reject_credentials(atom)
            pii_atom = atom.split("?", 1)[0] if key in ("source_krefs", "experience_kref") else atom
            if key != "observed_at" and redactor.anonymize_summary(pii_atom) != pii_atom:
                raise ValueError("Identifiers must not contain personal information")
    safe.update(identifiers)
    canonical = json.dumps(safe, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if len(canonical) > MAX_RECORD_CHARS:
        raise ValueError("Experience exceeds total size limit")
    safe["record_id"] = hashlib.sha256(canonical.encode()).hexdigest()
    return safe


def normalize_experience(record: dict) -> dict:
    """Validate and sanitize an explicit host extraction, without inference."""
    allowed = {"experience_id", "title", "situation", "goal", "alternatives", "decision", "rationale",
               "applicability_conditions", "expected_outcome", "decision_state", "origin",
               "source_krefs", "observed_outcome", "observed_at", "outcome_status", "acceptance"}
    if not isinstance(record, dict) or set(record) - allowed:
        raise ValueError("Unknown experience fields")
    experience_id = _text(record.get("experience_id", ""), "experience_id", required=True, limit=128)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", experience_id):
        raise ValueError("experience_id must be a stable opaque identifier")
    result = {"schema": SCHEMA, "record_type": "experience", "experience_id": experience_id}
    for key in ("title", "situation", "goal", "decision", "rationale", "expected_outcome"):
        result[key] = _text(record.get(key, ""), key, required=True,
                            limit=160 if key == "title" else MAX_TEXT)
    for key in ("alternatives", "applicability_conditions"):
        result[key] = _texts(record.get(key, []), key)
    result["decision_state"] = _enum(record.get("decision_state", "unknown"), "decision_state",
                                     ("proposed", "accepted", "rejected", "unknown"))
    result["origin"] = _enum(record.get("origin", "unknown"), "origin", ("user", "agent", "external", "unknown"))
    result["source_krefs"] = _refs(record.get("source_krefs", []))
    result.update(_observation(record, required=False))
    return _finish(result)


def _observation(record: dict, *, required: bool) -> dict:
    observed = _text(record.get("observed_outcome", ""), "observed_outcome", required=required)
    date = record.get("observed_at", "")
    status = _enum(record.get("outcome_status", "unknown"), "outcome_status",
                   ("success", "failure", "mixed", "unknown"))
    if bool(observed) != bool(date) or (status != "unknown" and not observed):
        raise ValueError("Observed outcomes require observed_at and descriptive evidence")
    return {"observed_outcome": observed,
            "observed_at": _timestamp(date, "observed_at") if date else "",
            "outcome_status": status,
            "acceptance": _enum(record.get("acceptance", "unknown"), "acceptance",
                                ("accepted", "rejected", "unknown"))}


def _screen_identifier(value: str) -> None:
    """Reject secrets/PII in graph identifiers instead of changing identity."""
    redactor = PIIRedactor()
    redactor.reject_credentials(value)
    if redactor.anonymize_summary(value) != value:
        raise ValueError("Graph identifiers must not contain personal information")


def _scope_guard(manager: Any) -> str:
    project = getattr(manager, "project", "")
    if not isinstance(project, str) or len(project) > 128 or not re.fullmatch(r"[^/\s?#]+", project):
        raise ValueError("A concrete manager project is required")
    _screen_identifier(project)
    ctx = current_request()
    if is_hosted() and (ctx is None or not ctx.tenant_id):
        raise ValueError("Hosted experience operations require an active tenant")
    buffer_tenant = getattr(getattr(manager, "redis_buffer", None), "tenant_id", None)
    if ctx is not None and buffer_tenant != ctx.tenant_id:
        raise ValueError("Manager tenant does not match active request")
    return project


def scoped_space(manager: Any, path: str | None = None) -> str:
    project = _scope_guard(manager)
    if path is None:
        return f"/{project}/experiences"
    path = _text(path, "space_path", required=True, limit=512)
    _screen_identifier(path)
    parts = path.strip("/").split("/")
    if any(not p or p in (".", "..") or re.search(r"[\s?#\\]", p) for p in parts):
        raise ValueError("Invalid space path")
    if path.startswith("/") and parts[0] != project:
        raise ValueError("Space lies outside manager project")
    if parts[0] != project:
        parts.insert(0, project)
    return "/" + "/".join(parts)


async def validate_source_refs(manager: Any, refs: list[str], *, space_paths: list[str] | None = None) -> list[dict]:
    """Read pinned source metadata in current auth context, failing closed."""
    project = _scope_guard(manager)
    refs = _refs(refs)
    spaces = [scoped_space(manager, p).strip("/") for p in space_paths] if space_paths is not None else None
    if spaces == []:
        raise ValueError("Source scope must not be empty")
    for ref in refs:
        match = _REF.fullmatch(ref)
        if match is None or match.group(1) != project:
            raise ValueError("Source lies outside manager project")
        parent = ref.removeprefix("kref://").split("?")[0].rsplit("/", 1)[0]
        if spaces is not None and not any(parent == p or parent.startswith(p + "/") for p in spaces):
            raise ValueError("Source lies outside requested spaces")
    import kumiho
    rows = []
    for ref in refs:
        try:
            rev = await asyncio.to_thread(kumiho.get_revision, ref)
            actual = getattr(getattr(rev, "kref", None), "uri", "")
            meta = getattr(rev, "metadata", None)
            revision_deprecated = getattr(rev, "deprecated", False)
            if type(revision_deprecated) is not bool:
                raise ValueError("Invalid revision deprecated state")
            if actual != ref or not isinstance(meta, dict):
                raise ValueError("Invalid revision response")
        except Exception:
            raise ValueError("Source revision is inaccessible") from None
        tags = getattr(rev, "tags", [])
        if not isinstance(tags, (list, tuple)) or len(tags) > 128:
            raise ValueError("Invalid or excessive source tags")
        # Revision provenance is kept separate; current item-level
        # markers can apply conservatively across siblings, never overwrite it.
        item_uri = ref.split("?", 1)[0]
        try:
            item = await asyncio.to_thread(kumiho.get_item, item_uri)
            if getattr(getattr(item, "kref", None), "uri", "") != item_uri:
                raise ValueError("Invalid item response")
            item_meta = getattr(item, "metadata", None)
            if not isinstance(item_meta, dict):
                raise ValueError("Invalid item metadata")
            markers = {}
            for key in ("grounding_stale", "grounding_stale_superseded_by", "contested_by",
                        "superseded_by", "decision_state", "as_of_excluded", "tags", "deprecated"):
                if key not in item_meta:
                    continue
                value = item_meta[key]
                if (isinstance(value, str) and len(value) <= 8192) or type(value) is bool:
                    markers[key] = value
                elif isinstance(value, list) and len(value) <= 32 and all(isinstance(v, str) and len(v) <= 512 for v in value):
                    markers[key] = list(value)
                else:
                    raise ValueError("Invalid or excessive item markers")
            deprecated = getattr(item, "deprecated", False)
            if type(deprecated) is not bool:
                raise ValueError("Invalid item deprecated state")
            if deprecated:
                markers["deprecated"] = True
        except Exception:
            raise ValueError("Current source item state is inaccessible") from None
        rows.append({"kref": ref, "metadata": dict(meta), "tags": list(tags), "revision_deprecated": revision_deprecated, "item_markers": markers})
    return rows


def experience_from_memory(row: dict) -> dict | None:
    """Decode a canonical snapshot only; malformed/legacy prose is not evidence."""
    if not isinstance(row, dict) or not _REF.fullmatch(str(row.get("kref", ""))):
        return None
    meta = row.get("metadata", row)
    if not isinstance(meta, dict) or not row["kref"].split("?")[0].endswith(".experience"):
        return None
    raw = meta.get("experience_record")
    if not isinstance(raw, str) or len(raw) > MAX_RECORD_CHARS + 1024:
        return None
    try:
        record = json.loads(raw)
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            return None
        record_id = record.pop("record_id")
        recorded_at = _timestamp(record.pop("recorded_at"), "recorded_at")
        kind = record.pop("record_type")
        record.pop("schema")
        if kind == "experience":
            normalized = normalize_experience(record)
        elif kind == "outcome":
            parent = record.pop("experience_kref")
            normalized = _normalize_outcome(parent, record)
        else:
            return None
        if normalized["record_id"] != record_id:
            return None
        return {**normalized, "recorded_at": recorded_at, "kref": row["kref"]}
    except (ValueError, TypeError, KeyError):
        return None


def _normalize_outcome(experience_kref: str, outcome: dict) -> dict:
    _refs([experience_kref])
    if not isinstance(outcome, dict) or set(outcome) - {
        "observed_outcome", "observed_at", "outcome_status", "acceptance", "origin", "source_krefs"
    }:
        raise ValueError("Unknown outcome fields")
    return _finish({"schema": SCHEMA, "record_type": "outcome", "experience_kref": experience_kref,
                    **_observation(outcome, required=True),
                    "origin": _enum(outcome.get("origin", "unknown"), "origin", ("user", "agent", "external", "unknown")),
                    "source_krefs": _refs(outcome.get("source_krefs", []))})


async def _store(manager: Any, record: dict, space: str) -> dict:
    if not callable(getattr(manager, "memory_store", None)):
        raise ValueError("No memory_store configured")
    record = {**record, "recorded_at": datetime.now(timezone.utc).isoformat()}
    title = record.get("title") or "Outcome observation"
    summary = record.get("observed_outcome") or record.get("decision", "")
    refs = list(dict.fromkeys(([record["experience_kref"]] if "experience_kref" in record else []) + record["source_krefs"]))
    payload = {"project": _scope_guard(manager), "space_path": space,
               "memory_item_kind": "experience", "memory_type": record["record_type"],
               "title": title, "summary": summary, "user_text": json.dumps(record, ensure_ascii=False),
               "metadata": {"experience_record": json.dumps(record, ensure_ascii=False),
                            "record_id": record["record_id"], "schema": SCHEMA,
                            "origin": record["origin"], "evidence_level": "unverified",
                            "decision_state": record.get("decision_state", "unknown")},
               "source_revision_krefs": refs, "edge_type": "DERIVED_FROM",
               "tags": ["experience", "evidence:unverified"] + (["proposal"] if record.get("decision_state") == "proposed" else []), "stack_revisions": False}
    result = await asyncio.to_thread(manager.memory_store, **payload)
    if inspect.isawaitable(result):
        result = await result
    if (not isinstance(result, dict) or result.get("error")
            or not _REF.fullmatch(str(result.get("revision_kref", "")))
            or not result["revision_kref"].startswith(f"kref://{manager.project}/")
            or not result["revision_kref"].split("?")[0].endswith(".experience")):
        raise RuntimeError("Experience store failed; a partial write may require reconciliation")
    return {**result, "record": record, "idempotent": False,
            "deduplication_key": record["record_id"], "belief_promoted": False}


async def record_experience(manager: Any, record: dict, *, space_path: str | None = None) -> dict:
    normalized = normalize_experience(record)
    space = scoped_space(manager, space_path)
    await validate_source_refs(manager, normalized["source_krefs"])
    return await _store(manager, normalized, space)


async def record_outcome(manager: Any, experience_kref: str, outcome: dict) -> dict:
    normalized = _normalize_outcome(experience_kref, outcome)
    rows = await validate_source_refs(manager, [experience_kref])
    parent = experience_from_memory(rows[0])
    if parent is None or parent["record_type"] != "experience":
        raise ValueError("Parent must be a canonical experience snapshot")
    await validate_source_refs(manager, normalized["source_krefs"])
    space = "/" + experience_kref.removeprefix("kref://").split("?")[0].rsplit("/", 1)[0]
    return await _store(manager, normalized, scoped_space(manager, space))
