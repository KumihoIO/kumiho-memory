"""Tests for the judged-delivery layer (``kumiho_memory.context_optimization``).

No network, no graph: the evaluation client is injected and the clock is
passed in, so the back-off windows are exercised without sleeping.
"""

import json

import pytest

from kumiho_memory import context_optimization as ctxopt
from kumiho_memory.context_optimization import (
    ContextOptimizationPolicy,
    EvaluationUnavailable,
    RUBRIC_VERSION,
    SdkEvaluationClient,
    build_fragments,
    is_backed_off,
    optimize_recall,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class Answer:
    """A NoulAnswer stand-in: one probability."""

    def __init__(self, noul):
        self.noul = noul


class Judgment:
    """A FragmentEvaluation stand-in."""

    def __init__(self, fragment_id, relevance=None, evidence=None, error="",
                 answers=None):
        self.fragment_id = fragment_id
        self.error = error
        if answers is None:
            answers = {}
            if relevance is not None:
                answers["is_relevant"] = Answer(relevance)
            if evidence is not None:
                answers["contains_answer_evidence"] = Answer(evidence)
        self.answers = answers


class Result:
    """An EvaluationResult stand-in."""

    def __init__(self, status="ok", fragments=()):
        self.status = status
        self.fragments = list(fragments)


class RecordingClient:
    """Records every request; returns *result* or raises *error*."""

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else Result()
        self.error = error
        self.calls = []

    def evaluate(self, query, fragments, questions, *, timeout_ms):
        self.calls.append({
            "query": query,
            "fragments": fragments,
            "questions": questions,
            "timeout_ms": timeout_ms,
        })
        if self.error is not None:
            raise self.error
        return self.result


class RpcCode:
    def __init__(self, name):
        self.name = name


class RpcError(Exception):
    """A grpc.RpcError stand-in: the status code hangs off ``code()``."""

    def __init__(self, name):
        super().__init__(name)
        self._code = RpcCode(name)

    def code(self):
        return self._code


ENABLED = ContextOptimizationPolicy(enabled=True)


def memories(count, *, summary="a stored memory about the query subject"):
    return [
        {
            "kref": "kref://memory/proj/space/item-%d/rev/1" % index,
            "title": "memory %d" % index,
            "summary": summary,
            "type": "fact",
            "space": "Personal/Preferences",
            "created_at": "2026-09-18T11:22:33Z",
            "score": 0.5,
        }
        for index in range(count)
    ]


@pytest.fixture(autouse=True)
def _clear_backoff():
    ctxopt._backoff_until.clear()
    yield
    ctxopt._backoff_until.clear()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_defaults_are_off():
    """An unset environment leaves engage exactly as it was."""
    policy = ContextOptimizationPolicy.from_env({})
    assert policy.enabled is False
    assert policy.candidates == 50
    assert policy.summary_chars == 600
    assert policy.relevance_min == 0.45
    assert policy.evidence_min == 0.5
    assert policy.timeout_ms == 4000


def test_policy_reads_every_knob():
    policy = ContextOptimizationPolicy.from_env({
        "KUMIHO_MEMORY_CONTEXT_OPT_ENABLED": "1",
        "KUMIHO_MEMORY_CONTEXT_OPT_CANDIDATES": "32",
        "KUMIHO_MEMORY_CONTEXT_OPT_SUMMARY_CHARS": "800",
        "KUMIHO_MEMORY_CONTEXT_OPT_RELEVANCE_MIN": "0.6",
        "KUMIHO_MEMORY_CONTEXT_OPT_EVIDENCE_MIN": "0.7",
        "KUMIHO_MEMORY_CONTEXT_OPT_TIMEOUT_MS": "9000",
    })
    assert policy == ContextOptimizationPolicy(
        enabled=True, candidates=32, summary_chars=800,
        relevance_min=0.6, evidence_min=0.7, timeout_ms=9000,
    )


@pytest.mark.parametrize("value", ["yes", "on", "TRUE", "1"])
def test_policy_enabled_accepts_the_usual_spellings(value):
    assert ContextOptimizationPolicy.from_env(
        {"KUMIHO_MEMORY_CONTEXT_OPT_ENABLED": value},
    ).enabled is True


def test_policy_enabled_rejects_nonsense_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        policy = ContextOptimizationPolicy.from_env(
            {"KUMIHO_MEMORY_CONTEXT_OPT_ENABLED": "maybe"},
        )
    assert policy.enabled is False
    assert "ENABLED" in caplog.text


@pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "-5", "4.5"])
def test_policy_invalid_ints_fall_back_to_the_default(raw):
    """A malformed knob must not silently reshape the request."""
    policy = ContextOptimizationPolicy.from_env({
        "KUMIHO_MEMORY_CONTEXT_OPT_CANDIDATES": raw,
        "KUMIHO_MEMORY_CONTEXT_OPT_SUMMARY_CHARS": raw,
        "KUMIHO_MEMORY_CONTEXT_OPT_TIMEOUT_MS": raw,
    })
    assert policy.candidates == 50
    assert policy.summary_chars == 600
    assert policy.timeout_ms == 4000


def test_policy_invalid_int_warns_once_per_variable(caplog):
    with caplog.at_level("WARNING"):
        ContextOptimizationPolicy.from_env(
            {"KUMIHO_MEMORY_CONTEXT_OPT_TIMEOUT_MS": "soon"},
        )
    warnings = [r for r in caplog.records if "TIMEOUT_MS" in r.getMessage()]
    assert len(warnings) == 1


def test_policy_clamps_candidates_to_the_request_cap(caplog):
    """The Evaluate RPC accepts at most 64 fragments."""
    with caplog.at_level("WARNING"):
        policy = ContextOptimizationPolicy.from_env(
            {"KUMIHO_MEMORY_CONTEXT_OPT_CANDIDATES": "500"},
        )
    assert policy.candidates == 64
    assert "CANDIDATES" in caplog.text


@pytest.mark.parametrize("raw", ["abc", "-0.1", "1.5", "42"])
def test_policy_invalid_thresholds_fall_back_to_the_default(raw, caplog):
    with caplog.at_level("WARNING"):
        policy = ContextOptimizationPolicy.from_env({
            "KUMIHO_MEMORY_CONTEXT_OPT_RELEVANCE_MIN": raw,
            "KUMIHO_MEMORY_CONTEXT_OPT_EVIDENCE_MIN": raw,
        })
    assert policy.relevance_min == 0.45
    assert policy.evidence_min == 0.5
    assert "RELEVANCE_MIN" in caplog.text


def test_policy_accepts_the_unit_interval_bounds():
    policy = ContextOptimizationPolicy.from_env({
        "KUMIHO_MEMORY_CONTEXT_OPT_RELEVANCE_MIN": "0",
        "KUMIHO_MEMORY_CONTEXT_OPT_EVIDENCE_MIN": "1",
    })
    assert (policy.relevance_min, policy.evidence_min) == (0.0, 1.0)


def test_policy_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("KUMIHO_MEMORY_CONTEXT_OPT_ENABLED", "1")
    assert ContextOptimizationPolicy.from_env().enabled is True


# ---------------------------------------------------------------------------
# Fragments — privacy and shape
# ---------------------------------------------------------------------------


def test_fragments_carry_no_kref_space_or_tags():
    """The graph address of a memory is not part of the question being asked."""
    pool = memories(3)
    for mem in pool:
        mem["tags"] = ["published", "evidence:official"]
        mem["user_id"] = "user-42"
        mem["tenant_id"] = "tenant-7"
    request = json.dumps(build_fragments(pool, ENABLED))
    for forbidden in (
        "kref://", "item-0", "Personal/Preferences", "published",
        "evidence:official", "user-42", "tenant-7",
    ):
        assert forbidden not in request, forbidden
    for fragment in build_fragments(pool, ENABLED):
        assert set(fragment) == {"id", "text", "metadata"}
        assert set(fragment["metadata"]) <= {"title", "type", "date"}


def test_fragment_ids_are_opaque_and_positional():
    fragments = build_fragments(memories(3), ENABLED)
    assert [f["id"] for f in fragments] == ["c01", "c02", "c03"]


def test_fragment_text_is_the_truncated_summary():
    """Judging the first 600 chars raised precision 0.88 -> 0.95 and cut
    evaluation tokens 36%; delivery still uses the full memory."""
    pool = memories(1, summary="x" * 5000)
    fragment = build_fragments(pool, ENABLED)[0]
    assert fragment["text"] == "x" * 600
    assert pool[0]["summary"] == "x" * 5000, "the stored memory is untouched"


def test_fragment_metadata_omits_empty_values():
    fragment = build_fragments(
        [{"summary": "text", "title": "", "type": None, "created_at": ""}],
        ENABLED,
    )[0]
    assert fragment["metadata"] == {}


def test_fragment_date_is_the_iso_day():
    fragment = build_fragments(memories(1), ENABLED)[0]
    assert fragment["metadata"]["date"] == "2026-09-18"


def test_fragment_summary_chars_follows_the_policy():
    policy = ContextOptimizationPolicy(enabled=True, summary_chars=10)
    fragment = build_fragments(memories(1, summary="y" * 50), policy)[0]
    assert fragment["text"] == "y" * 10


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


def test_question_texts_are_the_measured_rubric():
    client = RecordingClient()
    optimize_recall("q", memories(1), limit=5, policy=ENABLED, client=client)
    assert client.calls[0]["questions"] == [
        {
            "id": "is_relevant",
            "type": "noul",
            "instructions":
                "Does memory {fragment} address the subject of the query?",
        },
        {
            "id": "contains_answer_evidence",
            "type": "noul",
            "instructions": (
                "Does memory {fragment} state information usable in a direct "
                "answer to the query?"
            ),
        },
    ]


def test_request_carries_the_query_and_timeout():
    client = RecordingClient()
    policy = ContextOptimizationPolicy(enabled=True, timeout_ms=1500)
    optimize_recall("what is my favourite colour", memories(2),
                    limit=5, policy=policy, client=client)
    call = client.calls[0]
    assert call["query"] == "what is my favourite colour"
    assert call["timeout_ms"] == 1500
    assert len(call["fragments"]) == 2


# ---------------------------------------------------------------------------
# Keep rule
# ---------------------------------------------------------------------------


def _judged(pairs, status="ok"):
    return Result(status=status, fragments=[
        Judgment("c%02d" % (index + 1), relevance=rel, evidence=ev)
        for index, (rel, ev) in enumerate(pairs)
    ])


def test_keeps_only_what_passes_both_questions():
    pool = memories(4)
    client = RecordingClient(_judged([
        (0.9, 0.9),   # keep
        (0.9, 0.2),   # relevant but answers nothing
        (0.1, 0.9),   # evidence about another subject
        (0.8, 0.6),   # keep
    ]))
    outcome = optimize_recall("q", pool, limit=5, policy=ENABLED, client=client)
    assert outcome.status == "applied"
    assert outcome.candidates == 4
    assert outcome.reason == ""
    assert outcome.memories == [pool[0], pool[3]], "pipeline order preserved"


def test_thresholds_are_inclusive_at_the_boundary():
    pool = memories(3)
    client = RecordingClient(_judged([
        (0.45, 0.5),     # exactly on both thresholds — kept
        (0.4499, 0.9),   # just under relevance
        (0.9, 0.4999),   # just under evidence
    ]))
    outcome = optimize_recall("q", pool, limit=5, policy=ENABLED, client=client)
    assert outcome.memories == [pool[0]]


def test_delivery_is_not_padded_to_the_limit():
    """One relevant memory in a pool of fifty delivers one memory."""
    pool = memories(50)
    verdicts = [(0.0, 0.0)] * 50
    verdicts[30] = (0.95, 0.95)
    outcome = optimize_recall(
        "q", pool, limit=5, policy=ENABLED, client=RecordingClient(_judged(verdicts)),
    )
    assert outcome.memories == [pool[30]]


def test_zero_kept_is_a_valid_outcome():
    """A pool with nothing relevant delivers nothing, and is still 'applied'."""
    pool = memories(6)
    outcome = optimize_recall(
        "q", pool, limit=5, policy=ENABLED,
        client=RecordingClient(_judged([(0.1, 0.1)] * 6)),
    )
    assert outcome.memories == []
    assert outcome.status == "applied"
    assert outcome.candidates == 6


def test_delivery_can_exceed_the_callers_limit():
    """The delivered count is the judgment's, not the recall limit's."""
    pool = memories(12)
    outcome = optimize_recall(
        "q", pool, limit=5, policy=ENABLED,
        client=RecordingClient(_judged([(0.9, 0.9)] * 12)),
    )
    assert outcome.memories == pool


# ---------------------------------------------------------------------------
# Unjudged fragments — today's rule, per memory
# ---------------------------------------------------------------------------


def test_partial_result_keeps_unjudged_memories_within_the_limit():
    pool = memories(8)
    result = Result(status="partial", fragments=[
        Judgment("c01", relevance=0.1, evidence=0.1),   # judged out
        Judgment("c02", relevance=0.9, evidence=0.9),   # judged in
        # c03..c08 unjudged: kept while inside the caller's limit of 3.
    ])
    outcome = optimize_recall(
        "q", pool, limit=3, policy=ENABLED, client=RecordingClient(result),
    )
    assert outcome.status == "applied"
    assert outcome.memories == [pool[1], pool[2]]


def test_per_fragment_error_falls_back_to_position():
    pool = memories(4)
    result = Result(fragments=[
        Judgment("c01", error="provider refused"),
        Judgment("c02", relevance=0.9, evidence=0.9),
        Judgment("c03", error="provider refused"),
        Judgment("c04", relevance=0.9, evidence=0.9),
    ])
    outcome = optimize_recall(
        "q", pool, limit=2, policy=ENABLED, client=RecordingClient(result),
    )
    # c01 is inside the limit and kept; c03 is outside it and dropped.
    assert outcome.memories == [pool[0], pool[1], pool[3]]


def test_missing_answer_is_treated_as_unjudged():
    pool = memories(3)
    result = Result(fragments=[
        Judgment("c01", relevance=0.9),               # no evidence answer
        Judgment("c02", evidence=0.9),                # no relevance answer
        Judgment("c03", answers={"is_relevant": object()}),  # not a noul
    ])
    outcome = optimize_recall(
        "q", pool, limit=1, policy=ENABLED, client=RecordingClient(result),
    )
    assert outcome.memories == [pool[0]]


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------


def test_exception_falls_back_to_the_caller_limit_prefix():
    pool = memories(20)
    outcome = optimize_recall(
        "q", pool, limit=5, policy=ENABLED,
        client=RecordingClient(error=ValueError("boom")),
    )
    assert outcome.status == "fallback"
    assert outcome.reason == "error"
    assert outcome.candidates == 20
    assert outcome.memories == pool[:5]


def test_missing_sdk_function_is_feature_unavailable(monkeypatch):
    """An SDK without ``evaluate`` must degrade, not raise."""
    import kumiho

    monkeypatch.delattr(kumiho, "evaluate", raising=False)
    pool = memories(9)
    outcome = optimize_recall("q", pool, limit=5, policy=ENABLED)
    assert outcome.status == "fallback"
    assert outcome.reason == "sdk_unavailable"
    assert outcome.memories == pool[:5]


def test_default_client_calls_the_sdk_with_the_rubric(monkeypatch):
    import kumiho

    seen = {}

    def fake_evaluate(query, fragments, questions, **kwargs):
        seen.update(kwargs, query=query, fragments=fragments, questions=questions)
        return Result(fragments=[Judgment("c01", relevance=0.9, evidence=0.9)])

    monkeypatch.setattr(kumiho, "evaluate", fake_evaluate, raising=False)
    pool = memories(1)
    outcome = optimize_recall(
        "q", pool, limit=5,
        policy=ContextOptimizationPolicy(enabled=True, timeout_ms=2500),
    )
    assert outcome.status == "applied"
    assert seen["rubric_version"] == RUBRIC_VERSION
    assert seen["timeout_ms"] == 2500


def test_sdk_client_raises_evaluation_unavailable_without_evaluate(monkeypatch):
    import kumiho

    monkeypatch.delattr(kumiho, "evaluate", raising=False)
    with pytest.raises(EvaluationUnavailable):
        SdkEvaluationClient().evaluate("q", [], [], timeout_ms=1000)


@pytest.mark.parametrize("status", [
    "not_entitled", "over_limit", "provider_unavailable", "invalid_request",
])
def test_non_ok_status_falls_back_with_that_reason(status):
    pool = memories(20)
    outcome = optimize_recall(
        "q", pool, limit=5, policy=ENABLED,
        client=RecordingClient(Result(status=status)),
    )
    assert outcome.status == "fallback"
    assert outcome.reason == status
    assert outcome.memories == pool[:5]


def test_unrecognised_status_falls_back():
    outcome = optimize_recall(
        "q", memories(7), limit=5, policy=ENABLED,
        client=RecordingClient(Result(status="teapot")),
    )
    assert outcome.reason == "unknown_status"
    assert len(outcome.memories) == 5


def test_empty_recall_needs_no_request():
    client = RecordingClient()
    outcome = optimize_recall("q", [], limit=5, policy=ENABLED, client=client)
    assert outcome.memories == []
    assert outcome.candidates == 0
    assert outcome.reason == "no_candidates"
    assert client.calls == []


def test_fallback_prefix_is_shorter_than_the_limit_when_recall_was():
    pool = memories(2)
    outcome = optimize_recall(
        "q", pool, limit=5, policy=ENABLED,
        client=RecordingClient(error=RuntimeError("boom")),
    )
    assert outcome.memories == pool


# ---------------------------------------------------------------------------
# Back-off
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _fail(scope, clock, *, status=None, error=None):
    return optimize_recall(
        "q", memories(10), limit=5, policy=ENABLED, scope=scope, clock=clock,
        client=RecordingClient(
            Result(status=status) if status else None, error=error,
        ),
    )


def test_not_entitled_backs_off_for_ten_minutes():
    clock = Clock()
    _fail("tenant-a", clock, status="not_entitled")

    assert is_backed_off("tenant-a", now=clock.now) is True
    assert is_backed_off("tenant-a", now=clock.now + 599.0) is True
    assert is_backed_off("tenant-a", now=clock.now + 600.0) is False


@pytest.mark.parametrize("status", ["over_limit", "provider_unavailable"])
def test_transient_status_backs_off_for_one_minute(status):
    clock = Clock()
    _fail("tenant-a", clock, status=status)

    assert is_backed_off("tenant-a", now=clock.now + 59.0) is True
    assert is_backed_off("tenant-a", now=clock.now + 60.0) is False


@pytest.mark.parametrize("code,seconds", [
    ("UNIMPLEMENTED", 600.0),
    ("PERMISSION_DENIED", 600.0),
])
def test_grpc_codes_that_mean_not_here_back_off(code, seconds):
    clock = Clock()
    outcome = _fail("tenant-a", clock, error=RpcError(code))

    assert outcome.reason == code.casefold()
    assert is_backed_off("tenant-a", now=clock.now + seconds - 1.0) is True
    assert is_backed_off("tenant-a", now=clock.now + seconds) is False


def test_deadline_exceeded_does_not_back_off():
    """A slow call is a one-off; the next query may well be entitled and fast."""
    clock = Clock()
    outcome = _fail("tenant-a", clock, error=RpcError("DEADLINE_EXCEEDED"))

    assert outcome.reason == "timeout"
    assert is_backed_off("tenant-a", now=clock.now) is False


def test_invalid_request_does_not_back_off():
    clock = Clock()
    _fail("tenant-a", clock, status="invalid_request")
    assert is_backed_off("tenant-a", now=clock.now) is False


def test_backoff_is_keyed_by_requesting_identity():
    """One unentitled tenant must not disable the feature for the others."""
    clock = Clock()
    _fail("tenant-a", clock, status="not_entitled")

    assert is_backed_off("tenant-a", now=clock.now) is True
    assert is_backed_off("tenant-b", now=clock.now) is False

    client = RecordingClient(_judged([(0.9, 0.9)] * 10))
    outcome = optimize_recall(
        "q", memories(10), limit=5, policy=ENABLED, scope="tenant-b",
        clock=clock, client=client,
    )
    assert outcome.status == "applied"
    assert len(client.calls) == 1


def test_backed_off_scope_sends_no_request():
    clock = Clock()
    _fail("tenant-a", clock, status="not_entitled")

    client = RecordingClient()
    pool = memories(10)
    outcome = optimize_recall(
        "q", pool, limit=5, policy=ENABLED, scope="tenant-a", clock=clock,
        client=client,
    )
    assert client.calls == []
    assert outcome.status == "fallback"
    assert outcome.reason == "backoff"
    assert outcome.memories == pool[:5]


def test_the_feature_resumes_after_the_window():
    clock = Clock()
    _fail("tenant-a", clock, status="over_limit")

    clock.now += 61.0
    client = RecordingClient(_judged([(0.9, 0.9)] * 10))
    outcome = optimize_recall(
        "q", memories(10), limit=5, policy=ENABLED, scope="tenant-a",
        clock=clock, client=client,
    )
    assert outcome.status == "applied"
    assert len(client.calls) == 1


def test_expired_windows_do_not_accumulate():
    clock = Clock()
    for index in range(5):
        _fail("tenant-%d" % index, clock, status="over_limit")
    clock.now += 61.0
    _fail("tenant-late", clock, status="over_limit")

    assert list(ctxopt._backoff_until) == ["tenant-late"]
