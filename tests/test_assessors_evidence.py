"""Tests for the evidence-aware assessor (Level-of-Evidence epic, issue #10)."""

import asyncio
import json
import sys
import tempfile
import types

from kumiho_memory.assessors import (
    EvidencePolicy,
    create_evidence_assessor,
    grade_evidence,
)
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


class FakeAdapter:
    """LLMAdapter stub returning a canned JSON response."""

    def __init__(self, response):
        self.response = response if isinstance(response, str) else json.dumps(response)
        self.last_system = None
        self.last_messages = None

    async def chat(self, messages, model="", system="", max_tokens=0):
        self.last_system = system
        self.last_messages = messages
        return self.response


def _memories():
    """Five recalled memories with varied grades and sources."""
    return [
        {"kref": "kref://m/1", "title": "Official statement", "summary": "X is true",
         "source": "press-release:acme", "evidence_level": "official",
         "tags": ["published", "evidence:official"], "score": 0.5},
        {"kref": "kref://m/2", "title": "Reuters report", "summary": "X is true",
         "source": "news:reuters", "evidence_level": "single_source",
         "tags": ["evidence:single_source"], "score": 0.4},
        {"kref": "kref://m/3", "title": "AP report", "summary": "X is true",
         "source": "news:ap", "evidence_level": "single_source",
         "tags": ["evidence:single_source"], "score": 0.3},
        {"kref": "kref://m/4", "title": "Chat note", "summary": "X may be true",
         "tags": [], "score": 0.2},
        {"kref": "kref://m/5", "title": "Same-outlet dup", "summary": "X is true",
         "source": "news:reuters", "tags": [], "score": 0.1},
    ]


# ---------------------------------------------------------------------------
# grade_evidence — pure policy rules
# ---------------------------------------------------------------------------


def test_grade_official_pinning_blocks_promotion():
    """Contradicting an official memory -> unverified + conflict recorded."""
    grade = grade_evidence(_memories(), agrees_with=[2, 3], contradicts=[1],
                           policy=EvidencePolicy())
    assert grade["evidence_level"] == "unverified"
    assert grade["pinned"] is True
    assert grade["conflicting_krefs"] == ["kref://m/1"]
    assert grade["supporting_krefs"] == []


def test_grade_two_distinct_sources_promote():
    grade = grade_evidence(_memories(), agrees_with=[2, 3], contradicts=[],
                           policy=EvidencePolicy())
    assert grade["evidence_level"] == "corroborated"
    assert grade["memory_type"] == "fact"
    assert set(grade["supporting_krefs"]) == {"kref://m/2", "kref://m/3"}


def test_grade_same_source_twice_does_not_promote():
    """Two agreeing memories from the SAME source are one source."""
    grade = grade_evidence(_memories(), agrees_with=[2, 5], contradicts=[],
                           policy=EvidencePolicy())
    assert grade["evidence_level"] is None
    assert grade["has_agreement"] is True


def test_grade_contradiction_blocks_promotion():
    """A non-official contradiction still blocks corroboration."""
    grade = grade_evidence(_memories(), agrees_with=[2, 3], contradicts=[4],
                           policy=EvidencePolicy())
    assert grade["evidence_level"] is None
    assert grade["pinned"] is False
    assert grade["conflicting_krefs"] == ["kref://m/4"]


def test_grade_sourceless_agreement_does_not_promote():
    grade = grade_evidence(_memories(), agrees_with=[4], contradicts=[],
                           policy=EvidencePolicy())
    assert grade["evidence_level"] is None


def test_grade_malformed_indices_are_dropped():
    grade = grade_evidence(_memories(), agrees_with=[0, 99, "x", 2, 2, None],
                           contradicts="not-a-list", policy=EvidencePolicy())
    # Only index 2 survives -> one source -> no promotion, no crash
    assert grade["evidence_level"] is None
    assert grade["conflicting_krefs"] == []


def test_grade_min_corroboration_configurable():
    grade = grade_evidence(_memories(), agrees_with=[2], contradicts=[],
                           policy=EvidencePolicy(min_corroboration=1))
    assert grade["evidence_level"] == "corroborated"


# ---------------------------------------------------------------------------
# create_evidence_assessor — end-to-end with a fake adapter
# ---------------------------------------------------------------------------


def _assess(adapter_response, recalled, policy=None):
    adapter = FakeAdapter(adapter_response)
    assessor = create_evidence_assessor(
        adapter, policy=policy or EvidencePolicy(), skip_heuristic=True,
    )
    messages = [{"role": "user", "content": "Acme earnings claim " * 10}]
    return asyncio.run(assessor(messages, recalled)), adapter


def test_assessor_official_pinning():
    result, _ = _assess(
        {"should_store": True, "content": "Contradicting claim",
         "memory_type": "fact", "reason": "conflict",
         "agrees_with": [], "contradicts": [1], "source": "news:blog"},
        _memories(),
    )
    assert result.should_store is True
    assert result.evidence_level == "unverified"
    assert result.conflicting_krefs == ["kref://m/1"]


def test_assessor_corroboration_promotes_to_fact():
    result, _ = _assess(
        {"should_store": True, "content": "X is true",
         "memory_type": "summary", "reason": "corroborated",
         "agrees_with": [2, 3], "contradicts": [], "source": "news:bbc"},
        _memories(),
        policy=EvidencePolicy(create_supports_edges=True),
    )
    assert result.evidence_level == "corroborated"
    assert result.memory_type == "fact"  # forced by promotion
    assert set(result.supporting_krefs) == {"kref://m/2", "kref://m/3"}


def test_assessor_supports_edges_off_by_default():
    result, _ = _assess(
        {"should_store": True, "content": "X", "memory_type": "fact",
         "reason": "", "agrees_with": [2, 3], "contradicts": [], "source": "s"},
        _memories(),
    )
    assert result.evidence_level == "corroborated"
    assert result.supporting_krefs == []


def test_assessor_single_source():
    result, _ = _assess(
        {"should_store": True, "content": "New claim", "memory_type": "fact",
         "reason": "", "agrees_with": [], "contradicts": [],
         "source": "news:reuters"},
        _memories(),
    )
    assert result.evidence_level == "single_source"
    assert result.source == "news:reuters"


def test_assessor_no_source_defaults_unverified():
    result, _ = _assess(
        {"should_store": True, "content": "Rumor", "memory_type": "fact",
         "reason": "", "agrees_with": [], "contradicts": [], "source": ""},
        _memories(),
    )
    assert result.evidence_level == "unverified"


def test_assessor_never_emits_official():
    """Even a hostile LLM response cannot mint an official grade — the
    grade comes from grade_evidence, never from the model's own fields."""
    result, _ = _assess(
        {"should_store": True, "content": "X", "memory_type": "fact",
         "reason": "", "agrees_with": [], "contradicts": [],
         "source": "", "evidence_level": "official"},
        _memories(),
    )
    assert result.evidence_level == "unverified"


def test_assessor_invalid_json_is_safe():
    result, _ = _assess("this is not json {", _memories())
    assert result.should_store is False
    assert "json parse error" in result.reason


def test_assessor_prompt_shows_grades_and_sources():
    _, adapter = _assess(
        {"should_store": False, "content": "", "memory_type": "fact",
         "reason": "", "agrees_with": [], "contradicts": [], "source": ""},
        _memories(),
    )
    prompt = adapter.last_messages[0]["content"]
    assert "1. [official | press-release:acme]" in prompt
    assert "agrees_with" in adapter.last_system


# ---------------------------------------------------------------------------
# _background_assess integration — stamping + SUPPORTS edges
# ---------------------------------------------------------------------------


class _EdgeFakeRevision:
    def __init__(self, kref):
        self.kref = kref
        self.edges = []

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((str(target.kref), edge_type))


def _run_background_assess(assess_result, store_result, monkeypatch):
    """Drive _background_assess with a canned assessor and record effects."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}
    revisions = {}

    fake_kumiho = types.ModuleType("kumiho")

    def get_revision(kref):
        revisions.setdefault(kref, _EdgeFakeRevision(kref))
        return revisions[kref]

    fake_kumiho.get_revision = get_revision
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return store_result

    async def retrieve_stub(**kwargs):
        return [
            {"kref": "kref://m/2", "title": "R", "summary": "X",
             "source": "news:reuters", "score": 0.4, "tags": []},
        ]

    async def assess_fn(messages, recalled):
        return assess_result

    class _StubSummarizer:
        async def summarize_conversation(self, messages, context=None):
            return {}

        async def generate_implications(self, messages, context=None):
            return []

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=_StubSummarizer(),
        memory_store=store_stub,
        memory_retrieve=retrieve_stub,
        auto_assess_fn=assess_fn,
        auto_assess_min_messages=1,
    )

    async def run():
        ingest = await manager.ingest_message(
            user_id="user-bg", message="Claim about Acme earnings.",
        )
        await manager._background_assess(ingest["session_id"])

    asyncio.run(run())
    return stored, revisions


def test_background_assess_stamps_evidence(monkeypatch):
    from kumiho_memory.memory_manager import MemoryAssessResult

    stored, revisions = _run_background_assess(
        MemoryAssessResult(
            should_store=True, content="X is corroborated",
            memory_type="fact", evidence_level="corroborated",
            source="news:bbc",
            supporting_krefs=["kref://m/2", "kref://m/3"],
        ),
        {"item_kref": "kref://m/new", "revision_kref": "kref://m/new/rev/1"},
        monkeypatch,
    )
    assert stored["metadata"]["evidence_level"] == "corroborated"
    assert stored["metadata"]["source"] == "news:bbc"
    assert "evidence:corroborated" in stored["tags"]
    # SUPPORTS edges created from the new revision to both corroborators
    new_rev = revisions["kref://m/new/rev/1"]
    assert ("kref://m/2", "SUPPORTS") in new_rev.edges
    assert ("kref://m/3", "SUPPORTS") in new_rev.edges


def test_background_assess_rejects_official_from_assessor(monkeypatch):
    from kumiho_memory.memory_manager import MemoryAssessResult

    stored, _ = _run_background_assess(
        MemoryAssessResult(
            should_store=True, content="Hostile", memory_type="fact",
            evidence_level="official",
        ),
        {"item_kref": "kref://m/new", "revision_kref": "kref://m/new/rev/1"},
        monkeypatch,
    )
    assert "evidence_level" not in stored["metadata"]
    assert not any(t.startswith("evidence:") for t in stored["tags"])


def test_background_assess_records_conflicts(monkeypatch):
    from kumiho_memory.memory_manager import MemoryAssessResult

    stored, _ = _run_background_assess(
        MemoryAssessResult(
            should_store=True, content="Conflicting claim", memory_type="fact",
            evidence_level="unverified",
            conflicting_krefs=["kref://m/1"],
        ),
        {"item_kref": "kref://m/new", "revision_kref": "kref://m/new/rev/1"},
        monkeypatch,
    )
    assert stored["metadata"]["evidence_level"] == "unverified"
    assert stored["metadata"]["conflicts_with"] == "kref://m/1"


def test_background_assess_queued_store_skips_edges(monkeypatch):
    """No revision_kref (retry-queued store) -> SUPPORTS edges skipped."""
    from kumiho_memory.memory_manager import MemoryAssessResult

    stored, revisions = _run_background_assess(
        MemoryAssessResult(
            should_store=True, content="X", memory_type="fact",
            evidence_level="corroborated",
            supporting_krefs=["kref://m/2"],
        ),
        {"queued": True},
        monkeypatch,
    )
    assert stored["metadata"]["evidence_level"] == "corroborated"
    assert revisions == {}  # no edge creation attempted
