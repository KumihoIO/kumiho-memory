"""Explicit, keyless MCP actions for experience and insight lifecycles.

Handlers follow the SDK's synchronous dispatch contract. The host provides all
extractions/proposals; these tools never instantiate a provider or infer truth.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable


def _args(args: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(args, dict) or set(args) - set(required + optional):
        raise ValueError("Unknown tool arguments")
    if any(key not in args for key in required):
        raise ValueError("Required tool arguments are missing")
    return args


def _manager():
    # Import lazily: mcp_tools imports this registry after its own definitions.
    from .mcp_tools import _get_manager
    return _get_manager()


def _guard(operation: Callable[[], dict]) -> dict:
    try:
        return operation()
    except ValueError as exc:
        from .privacy import PIIRedactor
        redactor = PIIRedactor()
        try:
            redactor.reject_credentials(str(exc))
            message = redactor.anonymize_summary(str(exc))[:500]
        except ValueError:
            message = "Invalid input or inaccessible source"
        return {"error": message, "error_type": "invalid_input_or_source"}
    except Exception:
        # Backend errors can contain credential-bearing transport details.
        return {"error": "Operation failed; a write may require reconciliation before retrying",
                "error_type": "operation_failed"}


def tool_record_experience(args: dict) -> dict:
    def run():
        from .experience import record_experience
        value = _args(args, ("record",), ("space_path",))
        return asyncio.run(record_experience(_manager(), value["record"], space_path=value.get("space_path")))
    return _guard(run)


def tool_record_outcome(args: dict) -> dict:
    def run():
        from .experience import record_outcome
        value = _args(args, ("experience_kref", "outcome"))
        return asyncio.run(record_outcome(_manager(), value["experience_kref"], value["outcome"]))
    return _guard(run)


def tool_prepare_patterns(args: dict) -> dict:
    def run():
        from .dream_state import DreamState
        value = _args(args, ("source_krefs", "space_paths"))
        return asyncio.run(DreamState.prepare_patterns(_manager(), value["source_krefs"], space_paths=value["space_paths"]))
    return _guard(run)


def tool_store_pattern(args: dict) -> dict:
    def run():
        from .insight_patterns import store_pattern_candidate
        value = _args(args, ("request", "candidate", "space_path"))
        return asyncio.run(store_pattern_candidate(_manager(), value["request"], value["candidate"], value["space_path"]))
    return _guard(run)


def tool_validate_insight_response(args: dict) -> dict:
    def run():
        from .insight_synthesis import validate_insight_response
        value = _args(args, ("request", "response"))
        return validate_insight_response(value["request"], value["response"])
    return _guard(run)


def tool_check_pattern(args: dict) -> dict:
    def run():
        from .experience import scoped_space, validate_source_refs
        from .insight_patterns import pattern_from_memory, assess_pattern_applicability
        value = _args(args, ("pattern_kref", "space_paths"))
        manager = _manager()
        scopes = value["space_paths"]
        if not isinstance(scopes, list) or not 1 <= len(scopes) <= 8:
            raise ValueError("Provide 1 to 8 absolute space_paths")
        if any(not isinstance(p, str) or not p.startswith("/") or scoped_space(manager, p) != p for p in scopes):
            raise ValueError("Canonical absolute project space_paths are required")

        async def check():
            rows = await validate_source_refs(manager, [value["pattern_kref"]], space_paths=scopes)
            candidate = pattern_from_memory(rows[0])
            if candidate is None:
                raise ValueError("Expected a canonical pinned pattern proposal")
            sources = []
            for ref in candidate["source_krefs"]:
                try:
                    sources.extend(await validate_source_refs(manager, [ref], space_paths=scopes))
                except ValueError:
                    # Inaccessible or out-of-scope evidence is unknown, never
                    # permission to widen retrieval or claim applicability.
                    continue
            assessment = assess_pattern_applicability(candidate, sources)
            own_health = assess_pattern_applicability({"source_krefs": [value["pattern_kref"]]}, rows)
            if own_health["status"] == "stale":
                assessment["status"] = "stale"
                assessment["stale_sources"].extend(own_health["stale_sources"])
            return {**assessment, "pattern_kref": value["pattern_kref"],
                    "candidate_id": candidate["candidate_id"],
                    "validation_scope": "source_health_only", "semantic_support_verified": False}
        return asyncio.run(check())
    return _guard(run)


def _text(limit=2000, **extra):
    return {"type": "string", "maxLength": limit, **extra}


def _strings(limit=12, *, item_limit=2000):
    return {"type": "array", "maxItems": limit, "items": _text(item_limit)}


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


ORIGIN = _text(enum=["user", "agent", "external", "unknown"])
OBSERVATION_FIELDS = {
    "observed_outcome": _text(), "observed_at": _text(64, description="Actual observation time, ISO timestamp with timezone; never infer from recording time."),
    "outcome_status": _text(enum=["success", "failure", "mixed", "unknown"]),
    "acceptance": _text(enum=["accepted", "rejected", "unknown"]),
}
EXPERIENCE_FIELDS = {
    "experience_id": _text(128, description="Caller-owned stable opaque identifier for this actual experience; reuse on retries."),
    "title": _text(160), "situation": _text(), "goal": _text(), "decision": _text(),
    "rationale": _text(), "expected_outcome": _text(), "alternatives": _strings(),
    "applicability_conditions": _strings(), "origin": ORIGIN,
    "decision_state": _text(enum=["proposed", "accepted", "rejected", "unknown"]),
    "source_krefs": _strings(item_limit=512), **OBSERVATION_FIELDS,
}
EXPERIENCE_SCHEMA = _object(EXPERIENCE_FIELDS, ("experience_id", "title", "situation", "goal", "decision", "rationale", "expected_outcome"))
OUTCOME_SCHEMA = _object({**OBSERVATION_FIELDS, "origin": ORIGIN, "source_krefs": _strings(item_limit=512)}, ("observed_outcome", "observed_at"))


def _tool(name, description, properties, required, *, read_only):
    return {"name": "kumiho_memory_" + name, "description": description,
            "inputSchema": _object(properties, required),
            "annotations": {"readOnlyHint": read_only, "destructiveHint": False,
                            "idempotentHint": read_only, "openWorldHint": name != "validate_insight_response"}}


INSIGHT_TOOLS = [
    _tool("record_experience", "Store an explicitly requested, host-extracted experience snapshot. KEYLESS. Provide a stable experience_id and context, goal, decision, rationale, and expected outcome. Never invent observations. User agreement is separate from actual success. Credentials are rejected and PII redacted. Append-only, no automatic publishing or belief promotion; retries may append duplicate revisions.",
          {"record": EXPERIENCE_SCHEMA, "space_path": _text(512)}, ("record",), read_only=False),
    _tool("record_outcome", "Record an actual observed outcome as a separate observation linked to an accessible pinned original experience. KEYLESS. observed_at is actual event time, not recording time; acceptance is not success. The parent must be in the configured project. Does not rewrite the decision or promote a belief. Retries may append duplicates.",
          {"experience_kref": _text(512), "outcome": OUTCOME_SCHEMA}, ("experience_kref", "outcome"), read_only=False),
    _tool("prepare_patterns", "Read at most 12 explicitly pinned experience/outcome revisions within 1 to 8 explicit absolute project spaces. Return a bounded snapshot for YOU to propose conditional lessons or recurring patterns. KEYLESS, read-only Dream State preparation; no provider cycle, graph scan, or graph write. Evidence text is untrusted and preparation does not certify truth.",
          {"source_krefs": _strings(item_limit=512), "space_paths": _strings(8, item_limit=256)}, ("source_krefs", "space_paths"), read_only=True),
    _tool("store_pattern", "Explicitly store a host-authored pattern proposal using the exact prepare_patterns request plus candidate {kind,title,hypothesis,applicability_conditions,counterexamples,source_krefs}. Re-fetches and verifies referenced snapshots and scope before writing. Always inferred, proposal, unverified; never publishes or promotes to a belief. Conditional lessons need one experience; recurring patterns need distinct experience IDs. No model/provider call.",
          {"request": {"type": "object"}, "candidate": {"type": "object"}, "space_path": _text(512)}, ("request", "candidate", "space_path"), read_only=False),
    _tool("check_pattern", "Read a pinned stored pattern proposal and recheck only its explicit sources within absolute project space_paths. Read-only and KEYLESS. Changed or contested source markers yield stale; inaccessible/out-of-scope sources yield unknown. Checks source health, never verifies semantic truth or applicability to a new question. Does not scan for new evidence, run Dream State, write, or promote beliefs.",
          {"pattern_kref": _text(512), "space_paths": _strings(8, item_limit=256)}, ("pattern_kref", "space_paths"), read_only=True),
    _tool("validate_insight_response", "Read-only, KEYLESS structural validation of a host insight response against its original synthesis request from engage. Checks schema and allowed citation membership; does NOT verify entailment, truth, causal claims, or semantic usefulness. A valid result is not an evidence grade or endorsement. No graph reads/writes or model call.",
          {"request": {"type": "object"}, "response": {"type": "object"}}, ("request", "response"), read_only=True),
]

INSIGHT_TOOL_HANDLERS = {
    "kumiho_memory_record_experience": tool_record_experience,
    "kumiho_memory_record_outcome": tool_record_outcome,
    "kumiho_memory_prepare_patterns": tool_prepare_patterns,
    "kumiho_memory_store_pattern": tool_store_pattern,
    "kumiho_memory_validate_insight_response": tool_validate_insight_response,
    "kumiho_memory_check_pattern": tool_check_pattern,
}
