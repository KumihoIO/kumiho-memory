# -*- coding: utf-8 -*-
"""Applicability, origin, and uncertainty in recalled memory (kumiho-memory#28).

Covers the two axes #28 adds — claim origin/actor and decision acceptance
state — end to end: the reflect write stamps them, recall surfaces them
additively, both context assemblers render the qualifier, the qualifier survives
truncation and limit=1, an agent proposal is never promoted by repeated recall,
and provenance-grade separation is preserved (high certainty stays unverified).
"""
import sys
import types
from unittest.mock import patch

import pytest

from kumiho_memory import applicability as A
from kumiho_memory import context_compose as CC
from kumiho_memory.applicability import (
    apply_applicability_marker,
    applicability_notes,
    normalize_decision_state,
    normalize_origin,
)


# ---------------------------------------------------------------------------
# Vocabulary + marker units
# ---------------------------------------------------------------------------


def test_origin_normalization_defaults_to_unknown():
    assert normalize_origin("user") == "user"
    assert normalize_origin("AGENT") == "agent"
    assert normalize_origin("nonsense") == "unknown"
    assert normalize_origin(None) == "unknown"


def test_decision_state_normalization_defaults_to_unknown():
    assert normalize_decision_state("proposal") == "proposal"
    assert normalize_decision_state("Accepted") == "accepted"
    assert normalize_decision_state("") == "unknown"


def test_marker_is_additive_and_skips_unknown():
    entry = {"kref": "k", "score": 0.5}
    apply_applicability_marker(entry, {"origin": "agent", "decision_state": "proposal"})
    assert entry["origin"] == "agent" and entry["decision_state"] == "proposal"
    assert entry["score"] == 0.5  # untouched

    clean = {"kref": "k"}
    apply_applicability_marker(clean, {"origin": "unknown", "decision_state": "unknown"})
    assert "origin" not in clean and "decision_state" not in clean  # unknown → unstamped

    legacy = {"kref": "k"}
    apply_applicability_marker(legacy, {})  # legacy revision, no fields
    assert "origin" not in legacy and "decision_state" not in legacy


def test_notes_flag_proposal_and_agent_only():
    assert "proposal" in applicability_notes({"decision_state": "proposal"})
    assert "agent-asserted" in applicability_notes({"origin": "agent"})
    # accepted / user / imported / observed / unknown add no note
    assert applicability_notes({"decision_state": "accepted", "origin": "user"}) == ""
    assert applicability_notes({"origin": "imported"}) == ""
    assert applicability_notes({}) == ""


# ---------------------------------------------------------------------------
# Rendering: qualifier travels with the block, and survives truncation
# ---------------------------------------------------------------------------


def test_compose_context_renders_proposal_and_agent_notes():
    mems = [{
        "kref": "kref://p/notes/x.conversation?r=1",
        "title": "Event bus", "summary": "Consider Kafka for replay.",
        "score": 0.9, "decision_state": "proposal", "origin": "agent",
    }]
    text = CC.compose_context(mems, mode="summarized")
    assert "[proposal:" in text
    assert "[origin: agent-asserted" in text


def test_qualifier_survives_tight_char_budget():
    # A tight per-section budget truncates the content but the note is appended
    # AFTER truncation, so a material qualification is never silently dropped.
    long_summary = "x" * 5000
    mems = [{
        "kref": "kref://p/notes/x.conversation?r=1",
        "title": "T", "summary": long_summary,
        "score": 0.9, "decision_state": "proposal",
    }]
    text = CC.compose_context(mems, mode="full", char_limit=20)
    assert "[proposal:" in text


def test_qualifier_survives_limit_one_on_a_contested_pair():
    # limit=1 collapses to a single block; its contested + proposal notes stay.
    mems = [{
        "kref": "kref://p/notes/a.conversation?r=1",
        "title": "A", "summary": "one side", "score": 0.9,
        "contested_by": ["kref://p/notes/b.conversation?r=1"],
        "decision_state": "proposal",
    }]
    text = CC.compose_context(mems, mode="summarized", top_k=1)
    assert "[contested:" in text and "[proposal:" in text


def test_notes_ride_on_stacked_sibling_revisions():
    mems = [{
        "kref": "kref://p/notes/x.conversation?r=2",
        "title": "primary", "summary": "primary",
        "decision_state": "proposal", "origin": "agent",
        "sibling_revisions": [
            {"kref": "kref://p/notes/x.conversation?r=1", "title": "old", "summary": "old rev", "_score": 0.5},
        ],
    }]
    text = CC.compose_context(mems, mode="summarized")
    assert "[proposal:" in text  # item-level markers ride onto the sibling block


# ---------------------------------------------------------------------------
# Write path: reflect stamps the axes; recall surfaces them
# ---------------------------------------------------------------------------


def test_reflect_stamps_origin_and_state_into_store_metadata():
    from kumiho_memory import mcp_tools as M
    # Reuse the manager installer + store recorder from test_mcp_tools.
    import test_mcp_tools as MT
    MT._install_test_manager()
    try:
        ingest = M.tool_memory_ingest({"user_id": "u-appl", "message": "hi"})
        store_calls = []
        with patch("kumiho.mcp_server.tool_memory_store", MT._fake_store_recorder(store_calls)):
            M.tool_memory_reflect({
                "session_id": ingest["session_id"],
                "response": "ok",
                "captures": [{
                    "type": "decision", "title": "Maybe Kafka",
                    "content": "I would consider Kafka.",
                    "origin": "agent", "decision_state": "proposal",
                }],
                "discover_edges": False,
            })
        assert len(store_calls) == 1
        md = store_calls[0]["metadata"]
        assert md["origin"] == "agent" and md["decision_state"] == "proposal"
    finally:
        MT._cleanup_manager()


def test_reflect_drops_unknown_axis_values():
    from kumiho_memory import mcp_tools as M
    import test_mcp_tools as MT
    MT._install_test_manager()
    try:
        ingest = M.tool_memory_ingest({"user_id": "u-appl2", "message": "hi"})
        store_calls = []
        with patch("kumiho.mcp_server.tool_memory_store", MT._fake_store_recorder(store_calls)):
            M.tool_memory_reflect({
                "session_id": ingest["session_id"], "response": "ok",
                "captures": [{"type": "fact", "title": "t", "content": "c",
                              "origin": "bogus", "decision_state": "bogus"}],
                "discover_edges": False,
            })
        md = store_calls[0]["metadata"]
        # A bad value is not stamped (None metadata or a dict without the keys).
        assert md is None or ("origin" not in md and "decision_state" not in md)
    finally:
        MT._cleanup_manager()


# ---------------------------------------------------------------------------
# A proposal is never promoted by repeated recall / self-citation
# ---------------------------------------------------------------------------


def test_recall_surfaces_and_never_promotes_a_proposal():
    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    import sys as _sys
    sys.path.insert(0, "tests")
    from fakes import FakeRedis

    # A fake graph that returns one proposal-typed revision from retrieve.
    class _Rev:
        def __init__(self):
            self.kref = types.SimpleNamespace(uri="kref://p/decisions/maybe.decision?r=1")
            self.metadata = {"title": "Maybe Kafka", "summary": "consider kafka",
                             "type": "decision", "origin": "agent",
                             "decision_state": "proposal"}
            self.created_at = ""
            self.tags = []
        def get_artifacts(self):
            return []
    rev = _Rev()
    fake = types.ModuleType("kumiho")
    fake.get_revision = lambda uri: rev
    prev = sys.modules.get("kumiho")
    sys.modules["kumiho"] = fake
    try:
        async def retrieve(**kw):
            return {"revision_krefs": [rev.kref.uri], "scores": [0.9]}
        mgr = UniversalMemoryManager(
            project="P",
            redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://t"),
            memory_retrieve=retrieve, entity_promotion=False,
        )
        import asyncio
        # Recall it three times: the state stays "proposal" every time — nothing
        # promotes it, and the entry is marked as such.
        for _ in range(3):
            out = asyncio.run(mgr.recall_memories("kafka", limit=3))
            assert out and out[0].get("decision_state") == "proposal"
            assert out[0].get("origin") == "agent"
        ctx = mgr.build_recalled_context(out, "kafka", "summarized")
        assert "[proposal:" in ctx and "[origin: agent-asserted" in ctx
    finally:
        if prev is None:
            sys.modules.pop("kumiho", None)
        else:
            sys.modules["kumiho"] = prev


# ---------------------------------------------------------------------------
# Provenance-grade separation preserved (high certainty stays unverified)
# ---------------------------------------------------------------------------


def test_high_certainty_does_not_lift_provenance():
    from kumiho_memory.trust_vocab import normalize_trust, StrengthBand
    # certainty=high is HIGH on its own band, but evidence_level is a separate
    # axis: a high-certainty claim is still `unverified` provenance.
    assert normalize_trust("certainty", "high") == StrengthBand.HIGH
    assert normalize_trust("evidence_level", "unverified") == StrengthBand.LOW
    # And the applicability axis is independent of both.
    entry = {}
    apply_applicability_marker(entry, {"origin": "agent", "certainty": "high",
                                       "evidence_level": "unverified"})
    assert entry.get("origin") == "agent"
    assert "evidence_level" not in entry  # marker never touches provenance
