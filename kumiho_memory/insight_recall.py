"""Explicit opt-in discovery of canonical experiences and pattern proposals.

Two kind-specific discovery calls, at most three selected records, and at most
six referenced health records are read. No provider, artifact, or graph scan is
introduced here. SDK discovery may perform its own bounded fallback calls.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
from typing import Any

from .experience import (
    _refs, _scope_guard, experience_from_memory, sanitize_atoms, scoped_space,
    validate_source_refs,
)
from .insight_patterns import assess_pattern_applicability, pattern_from_memory
from .privacy import PIIRedactor

MAX_LEARNED_SOURCES = 3
MAX_HEALTH_REFS = 6
MAX_RESULT_CHARS = 24000
KINDS = ("experience", "pattern_candidate")


def _pinned(ref: Any, kind: str, scopes: list[str]) -> bool:
    if not isinstance(ref, str) or len(ref) > 512 or not re.fullmatch(r"kref://[^\s?#%\\]+\?r=[1-9][0-9]*", ref):
        return False
    item = ref.split("?", 1)[0]
    if not item.endswith("." + kind):
        return False
    parent = "/" + item[7:].rsplit("/", 1)[0]
    return any(parent == scope or parent.startswith(scope + "/") for scope in scopes)


def _view(record: dict) -> tuple[dict, bool]:
    """Sanitize full canonical text before building a bounded, noncanonical view."""
    scalar_fields = (
        "title", "situation", "goal", "decision", "rationale", "expected_outcome",
        "observed_outcome", "outcome_status", "acceptance", "decision_state",
        "origin", "hypothesis", "kind", "record_type",
    )
    list_fields = ("alternatives", "applicability_conditions", "counterexamples")
    values = {key: record[key] for key in scalar_fields + list_fields if key in record}
    values = sanitize_atoms(values)
    bounded = {}
    truncated = False
    for key, value in values.items():
        if isinstance(value, str):
            bounded[key] = value[:400]
            truncated |= len(value) > 400
        elif isinstance(value, list):
            bounded[key] = [entry[:160] for entry in value[:5] if isinstance(entry, str)]
            truncated |= len(value) > 5 or any(isinstance(entry, str) and len(entry) > 160 for entry in value)
    # Keep validated observation time and exact lineage refs as identifiers.
    for key in ("observed_at", "experience_kref", "source_krefs", "experience_id", "record_id", "candidate_id"):
        if key in record:
            if key == "source_krefs":
                _refs(record[key])
            elif key == "experience_kref":
                _refs([record[key]])
            bounded[key] = record[key]
    return bounded, truncated


async def recall_learned_sources(
    manager: Any, query: str, *, limit: int = 3,
    space_paths: list[str] | None = None,
    memory_types: list[str] | None = None, min_score: float | None = None,
) -> dict:
    """Return canonical learned records under the current manager auth/scope.

    Requires a configured kind-aware retrieval callable. No broad fallback is
    attempted if its signature rejects memory_item_kind. Results supplement an
    opt-in synthesis snapshot; they do not change ordinary recall or its ranking.
    Counters describe this layer, not SDK-internal transport calls.
    """
    result = {"status": "no_sources", "results": [], "retrieval_calls": 0,
              "source_reads": 0, "health_source_reads": 0,
              "validation_scope": "canonical_records_and_source_health_only",
              "semantic_support_verified": False}
    errors = []
    try:
        project = _scope_guard(manager)
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("A nonempty bounded query is required")
        privacy = PIIRedactor()
        privacy.reject_credentials(query)
        safe_query = privacy.anonymize_summary(query)
        if type(limit) is not int or not 1 <= limit <= MAX_LEARNED_SOURCES:
            raise ValueError("limit must be 1 to 3")
        if space_paths is not None and (not isinstance(space_paths, list) or not 1 <= len(space_paths) <= 8):
            raise ValueError("space_paths must contain 1 to 8 project scopes")
        scopes = [scoped_space(manager, path) for path in space_paths] if space_paths is not None else ["/" + project]
        for scope in scopes:
            privacy.reject_credentials(scope)
            if privacy.anonymize_summary(scope) != scope:
                raise ValueError("Scope identifiers must not contain personal information")
        if memory_types is not None and (not isinstance(memory_types, list) or len(memory_types) > 12
                or any(not isinstance(t, str) or not t or len(t) > 64 for t in memory_types)):
            raise ValueError("memory_types must be a bounded list")
        threshold = 0.0 if min_score is None else min_score
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("min_score must be between zero and one")
        allowed_types = set(memory_types) if memory_types is not None else None
        retrieve = getattr(manager, "memory_retrieve", None)
        if not callable(retrieve):
            raise ValueError("Kind-aware memory retrieval is unavailable")
    except ValueError:
        return {**result, "status": "unavailable", "backend_error": "Invalid query/scope or unavailable scoped retrieval"}

    if allowed_types is not None and not allowed_types.intersection({"experience", "outcome", "pattern_candidate"}):
        return result
    groups: list[list[str]] = []
    discovered_scores: dict[str, float | None] = {}
    for kind in KINDS:
        if allowed_types is not None and not allowed_types.intersection(
                {"experience", "outcome"} if kind == "experience" else {"pattern_candidate"}):
            continue
        try:
            result["retrieval_calls"] += 1
            found = await asyncio.to_thread(
                retrieve, project=project, query=safe_query, limit=limit,
                space_paths=scopes, memory_item_kind=kind,
                include_revision_metadata=True,
            )
            found = await found if inspect.isawaitable(found) else found
            if not isinstance(found, dict) or found.get("error"):
                raise ValueError("Discovery failed")
            refs = found.get("revision_krefs")
            if not isinstance(refs, list):
                raise ValueError("Unsupported discovery result")
            scores = found.get("scores")
            accepted = []
            for index, ref in enumerate(refs[:limit]):
                if not _pinned(ref, kind, scopes):
                    continue
                raw_score = (scores[index] if isinstance(scores, list) and len(scores) == len(refs)
                             else scores.get(ref) if isinstance(scores, dict) else None)
                score = float(raw_score) if type(raw_score) in (int, float) and math.isfinite(raw_score) else None
                if threshold > 0 and score is None:
                    errors.append("Discovery scores unavailable for a positive relevance threshold")
                    continue
                if score is not None and score < threshold:
                    continue
                discovered_scores[ref] = score
                accepted.append(ref)
            groups.append(list(dict.fromkeys(accepted)))
        except Exception:
            groups.append([])
            errors.append("A kind-specific discovery failed; no broad fallback was attempted")
    # Round-robin gives both kinds a chance within the single shared cap.
    selected = []
    for index in range(limit):
        for group in groups:
            if index < len(group) and group[index] not in selected:
                selected.append(group[index])
    selected = selected[:limit]
    health_cache: dict[str, dict | None] = {}
    for ref in selected:
        result["source_reads"] += 1
        try:
            row = (await validate_source_refs(manager, [ref], space_paths=scopes))[0]
            if ref.split("?", 1)[0].endswith(".experience"):
                record = experience_from_memory(row)
                kind = "experience"
            else:
                record = pattern_from_memory(row)
                kind = "pattern_candidate"
            if record is None:
                raise ValueError("Not a canonical learned record")
            record_type = record["record_type"] if kind == "experience" else "pattern_candidate"
            if allowed_types is not None and record_type not in allowed_types:
                continue
            view, truncated = _view(record)
            own_health = assess_pattern_applicability({"source_krefs": [ref]}, [row])
            health = own_health
            if kind == "pattern_candidate":
                health_rows = []
                for source in record["source_krefs"]:
                    if source not in health_cache and len(health_cache) < MAX_HEALTH_REFS:
                        health_cache[source] = None
                        result["health_source_reads"] += 1
                        try:
                            health_cache[source] = (await validate_source_refs(manager, [source], space_paths=scopes))[0]
                        except ValueError:
                            pass
                    if health_cache.get(source) is not None:
                        health_rows.append(health_cache[source])
                health = assess_pattern_applicability(record, health_rows)
                if own_health["status"] == "stale":
                    health["status"] = "stale"
                    health.setdefault("stale_sources", []).extend(own_health.get("stale_sources", []))
            # Health is also in prose: a downstream whitelist must not silently
            # strip the only warning that a recalled proposal is stale/unknown.
            summary = json.dumps({"source_health": health["status"], "applicability": "unknown",
                                  "view_truncated": truncated, "record": view},
                                 ensure_ascii=False, separators=(",", ":"))
            packet = {"kref": ref, "title": view.get("title") or "Outcome observation",
                      "summary": summary, "type": record_type, "origin": record.get("origin", "unknown"),
                      "decision_state": record.get("decision_state", "unknown"),
                      "evidence_level": "unverified", "source_health": health,
                      "canonical_record": view, "snapshot_truncated": truncated,
                      "conditions": view.get("applicability_conditions", []),
                      "outcome": view.get("observed_outcome", "")}
            if discovered_scores.get(ref) is not None:
                packet["score"] = discovered_scores[ref]
            if health["status"] == "stale":
                packet["grounding_stale"] = True
            if len(json.dumps({**result, "results": result["results"] + [packet]}, ensure_ascii=False)) > MAX_RESULT_CHARS:
                errors.append("Learned source output budget exhausted")
                break
            result["results"].append(packet)
        except Exception:
            errors.append("A selected learned record was inaccessible or noncanonical")
    result["status"] = "partial" if errors else "ready" if result["results"] else "no_sources"
    if errors:
        result["backend_error"] = "; ".join(dict.fromkeys(errors))
    return result
