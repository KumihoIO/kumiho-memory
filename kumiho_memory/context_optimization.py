"""Judged delivery for ``kumiho_memory_engage``: what to hand over, not what ranks.

Engage recalls ``limit`` memories and delivers all of them, so search rank is
also the delivery decision.  Measured on 40 queries x 50 candidates, rank
cannot carry that: 61 of 100 necessary memories sat outside the top 5 and the
search score separated necessary from unnecessary at AUC 0.67.  Judging the
same 50 candidates in ONE batched evaluation request separates them at AUC
0.97 — necessary-memory recall 0.55 -> 0.95, precision 0.38 -> 0.95, delivered
count dynamic (0-43, median 3), and nothing delivered for a query whose pool
held nothing relevant.

So this module widens the recall, asks the Kumiho server's ``Evaluate`` RPC two
questions about each candidate, and delivers the memories that pass.  Policy,
fragment building and assembly live here; keys, tier gating and metering live
in the server.

**What leaves the process.**  One fragment per candidate carrying the memory's
summary (first :attr:`ContextOptimizationPolicy.summary_chars` characters) and
title / type / date as metadata.  Nothing else: no kref, no space path, no
tags, no tenant or user id.  The opaque fragment ids (``c01``, ``c02``, ...)
are positional and local to one call.  Judging the first 600 characters was
measured against judging the whole summary: same recall, precision 0.88 ->
0.95, 36% fewer evaluation tokens.  Delivery always uses the full memory.

**Availability.**  ``Evaluate`` is a paid Kumiho Cloud capability.  On a
self-hosted CE server, an older SDK, or a tier without the entitlement the
feature is simply off: every failure path delivers the first ``limit``
candidates — today's behaviour — and nothing raises out of engage.  After a
verdict that will not change soon (no entitlement, no RPC) the scope backs off
for ten minutes; after a transient one (over limit, provider unavailable) for
one minute.  The back-off is per requesting identity, because the hosted
connector serves many tenants from one process.

Out of scope here: supersession.  The judge keeps superseded "latest version"
memories (measured), which is a deterministic problem solved deterministically
elsewhere.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)

#: Rubric identity sent with every request, so server-side caching and the
#: measurements that justify the thresholds below refer to the same questions.
#: A module constant, not an env var: changing the questions without changing
#: this would silently reuse judgments of different questions.
RUBRIC_VERSION = "ctxopt-v3"

#: Env prefix for every knob in :class:`ContextOptimizationPolicy`.
ENV_PREFIX = "KUMIHO_MEMORY_CONTEXT_OPT_"

#: Fragments the ``Evaluate`` RPC accepts in one request.
MAX_CANDIDATES = 64

DEFAULT_CANDIDATES = 50
DEFAULT_SUMMARY_CHARS = 600
DEFAULT_RELEVANCE_MIN = 0.45
DEFAULT_EVIDENCE_MIN = 0.5
DEFAULT_TIMEOUT_MS = 4000

RELEVANCE_QUESTION = "is_relevant"
EVIDENCE_QUESTION = "contains_answer_evidence"

#: The rubric.  ``{fragment}`` is substituted server-side with each fragment's
#: label.  Two questions, because the two more that were measured never changed
#: a delivery for the better: a premise-conflict question moved nothing, and a
#: prompt-injection question fired on memories that legitimately record rules
#: for the assistant — every memory it dropped was one the answer needed.
QUESTIONS: Tuple[Dict[str, str], ...] = (
    {
        "id": RELEVANCE_QUESTION,
        "type": "noul",
        "instructions": "Does memory {fragment} address the subject of the query?",
    },
    {
        "id": EVIDENCE_QUESTION,
        "type": "noul",
        "instructions": (
            "Does memory {fragment} state information usable in a direct "
            "answer to the query?"
        ),
    },
)

#: Statuses whose fragment judgments are usable.  ``partial`` carries fewer
#: judgments than fragments; the unjudged ones fall back per-memory.
_USABLE_STATUSES = frozenset({"ok", "partial"})
#: Non-ok statuses this module recognises; anything else reports
#: ``unknown_status``.
_KNOWN_STATUSES = frozenset({
    "not_entitled", "over_limit", "provider_unavailable", "invalid_request",
})
#: Statuses and gRPC codes that mean "not available on this deployment".
_ENTITLEMENT_REASONS = frozenset({
    "not_entitled", "unimplemented", "permission_denied",
})
#: Statuses that mean "not available right now".
_TRANSIENT_REASONS = frozenset({"over_limit", "provider_unavailable"})

#: Back-off after a verdict that will not change soon, in seconds.
ENTITLEMENT_BACKOFF_SECS = 600.0
#: Back-off after a transient failure, in seconds.
TRANSIENT_BACKOFF_SECS = 60.0

STATUS_APPLIED = "applied"
STATUS_FALLBACK = "fallback"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"", "0", "false", "no", "off"})


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _env_flag(source: Mapping[str, str], key: str, default: bool) -> bool:
    """A boolean env value; anything unrecognised falls back with one warning."""
    raw = str(source.get(ENV_PREFIX + key, "")).strip().casefold()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    logger.warning(
        "%s%s=%r is not a boolean — falling back to %s.",
        ENV_PREFIX, key, raw, default,
    )
    return default


def _env_int(source: Mapping[str, str], key: str, default: int) -> int:
    """A positive int from the environment.

    Mirrors ``context_compose._resolve_context_budget_chars``: a missing or
    blank value is the default silently; a non-integer or non-positive one is
    the default with one warning naming the variable.
    """
    raw = source.get(ENV_PREFIX + key)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        logger.warning(
            "%s%s=%r is not an integer — falling back to %d.",
            ENV_PREFIX, key, raw, default,
        )
        return default
    if value <= 0:
        logger.warning(
            "%s%s=%d is not positive — falling back to %d.",
            ENV_PREFIX, key, value, default,
        )
        return default
    return value


def _env_unit_float(source: Mapping[str, str], key: str, default: float) -> float:
    """A threshold in ``0.0..1.0``; anything else is the default with a warning.

    Noul answers are probabilities, so a threshold outside the unit interval is
    a typo rather than a preference — one would deliver everything, the other
    nothing.
    """
    raw = source.get(ENV_PREFIX + key)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        logger.warning(
            "%s%s=%r is not a number — falling back to %s.",
            ENV_PREFIX, key, raw, default,
        )
        return default
    if not 0.0 <= value <= 1.0:
        logger.warning(
            "%s%s=%s is outside 0.0..1.0 — falling back to %s.",
            ENV_PREFIX, key, value, default,
        )
        return default
    return value


def _clamp_candidates(value: int) -> int:
    """Hold the candidate pool at the request cap the server enforces."""
    if value > MAX_CANDIDATES:
        logger.warning(
            "%sCANDIDATES=%d exceeds the %d-fragment request cap — using %d.",
            ENV_PREFIX, value, MAX_CANDIDATES, MAX_CANDIDATES,
        )
        return MAX_CANDIDATES
    return value


@dataclass(frozen=True)
class ContextOptimizationPolicy:
    """How wide to judge, on how much text, and how hard to pass.

    Thresholds are the measured operating point of :data:`RUBRIC_VERSION`
    (necessary-memory recall 0.95 at precision 0.95); moving them trades one
    for the other.
    """

    enabled: bool = False
    candidates: int = DEFAULT_CANDIDATES
    summary_chars: int = DEFAULT_SUMMARY_CHARS
    relevance_min: float = DEFAULT_RELEVANCE_MIN
    evidence_min: float = DEFAULT_EVIDENCE_MIN
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None,
    ) -> "ContextOptimizationPolicy":
        """Read the policy from *env* (defaults to ``os.environ``).

        Every knob is ``KUMIHO_MEMORY_CONTEXT_OPT_`` + the field name in upper
        case.  Off by default: an unset environment produces the disabled
        policy, and engage then runs exactly as it did before this feature.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            enabled=_env_flag(source, "ENABLED", False),
            candidates=_clamp_candidates(
                _env_int(source, "CANDIDATES", DEFAULT_CANDIDATES),
            ),
            summary_chars=_env_int(
                source, "SUMMARY_CHARS", DEFAULT_SUMMARY_CHARS,
            ),
            relevance_min=_env_unit_float(
                source, "RELEVANCE_MIN", DEFAULT_RELEVANCE_MIN,
            ),
            evidence_min=_env_unit_float(
                source, "EVIDENCE_MIN", DEFAULT_EVIDENCE_MIN,
            ),
            timeout_ms=_env_int(source, "TIMEOUT_MS", DEFAULT_TIMEOUT_MS),
        )


# ---------------------------------------------------------------------------
# Back-off, per requesting identity
# ---------------------------------------------------------------------------
# Keyed by the scope the recall dedup guard uses (``mcp_tools._recall_scope``):
# "" on stdio, (tenant, user, session) hosted. A process-global back-off would
# let one unentitled tenant turn the feature off for every other tenant the
# hosted connector serves.

_backoff_until: Dict[str, float] = {}
_backoff_guard = threading.Lock()


def is_backed_off(scope: str, *, now: Optional[float] = None) -> bool:
    """Whether *scope* is inside a back-off window at *now*.

    *now* is a monotonic reading; tests pass their own clock rather than
    sleeping.  An elapsed window is dropped on the way past, so a scope that
    recovers leaves nothing behind.
    """
    moment = time.monotonic() if now is None else now
    with _backoff_guard:
        until = _backoff_until.get(scope)
        if until is None:
            return False
        if moment >= until:
            del _backoff_until[scope]
            return False
        return True


def _start_backoff(scope: str, seconds: float, reason: str, now: float) -> None:
    """Stop trying for *scope* for *seconds*, and say why once."""
    with _backoff_guard:
        for key in [k for k, until in _backoff_until.items() if now >= until]:
            del _backoff_until[key]
        _backoff_until[scope] = now + seconds
    logger.warning(
        "context optimization unavailable (%s) — not retrying for %.0fs.",
        reason, seconds,
    )


def _backoff_seconds(reason: str) -> float:
    """How long *reason* suppresses further attempts; 0 means keep trying."""
    if reason in _ENTITLEMENT_REASONS:
        return ENTITLEMENT_BACKOFF_SECS
    if reason in _TRANSIENT_REASONS:
        return TRANSIENT_BACKOFF_SECS
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation client
# ---------------------------------------------------------------------------


class EvaluationUnavailable(RuntimeError):
    """This install cannot evaluate at all — the SDK has no ``evaluate``."""


class EvaluationClient(Protocol):
    """One batched judgment request.  Injectable so tests need no network."""

    def evaluate(
        self,
        query: str,
        fragments: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        *,
        timeout_ms: int,
    ) -> Any:
        ...


class SdkEvaluationClient:
    """The default client: ``kumiho.evaluate``, resolved at call time.

    Never at import time — ``evaluate`` arrives in a later SDK than this
    package's dependency floor, and an install without it must degrade to the
    old delivery rather than fail to import.
    """

    def evaluate(
        self,
        query: str,
        fragments: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        *,
        timeout_ms: int,
    ) -> Any:
        try:
            import kumiho
        except ImportError as exc:  # pragma: no cover - kumiho is a dependency
            raise EvaluationUnavailable("sdk_unavailable") from exc
        evaluate = getattr(kumiho, "evaluate", None)
        if evaluate is None:
            raise EvaluationUnavailable("sdk_unavailable")
        return evaluate(
            query,
            fragments,
            questions,
            rubric_version=RUBRIC_VERSION,
            timeout_ms=timeout_ms,
        )


def _grpc_reason(exc: BaseException) -> str:
    """A short reason for a raised evaluation failure.

    gRPC status codes are read off the exception rather than by importing
    ``grpc``: ``UNIMPLEMENTED`` is a CE or pre-Evaluate server, and
    ``PERMISSION_DENIED`` is a tier without the entitlement — both mean "stop
    asking for a while".  Everything else, timeouts included, is a one-off.
    """
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            return "error"
    name = getattr(code, "name", None)
    if not isinstance(name, str):
        return "error"
    lowered = name.casefold()
    if lowered in ("unimplemented", "permission_denied"):
        return lowered
    if lowered == "deadline_exceeded":
        return "timeout"
    return "error"


# ---------------------------------------------------------------------------
# Fragments
# ---------------------------------------------------------------------------


def fragment_id(position: int) -> str:
    """The opaque local id of the candidate at *position* (0-based).

    Positional and per-call.  Never the kref: the judged text is the memory's
    own summary, and the graph address of that memory is not part of the
    question being asked.
    """
    return "c%02d" % (position + 1)


def build_fragments(
    memories: Sequence[Dict[str, Any]],
    policy: ContextOptimizationPolicy,
) -> List[Dict[str, Any]]:
    """The judged view of *memories*, in pipeline order.

    Title, type and ISO day of ``created_at`` travel as metadata (empty values
    omitted); the text is the summary cut to ``policy.summary_chars``.  Nothing
    else is included — see the module docstring.
    """
    fragments: List[Dict[str, Any]] = []
    for position, mem in enumerate(memories):
        pairs = (
            ("title", mem.get("title")),
            ("type", mem.get("type")),
            ("date", str(mem.get("created_at") or "")[:10]),
        )
        metadata = {
            key: str(value).strip()
            for key, value in pairs
            if str(value or "").strip()
        }
        fragments.append({
            "id": fragment_id(position),
            "text": str(mem.get("summary") or "")[:policy.summary_chars],
            "metadata": metadata,
        })
    return fragments


# ---------------------------------------------------------------------------
# Judgment
# ---------------------------------------------------------------------------


def _noul(answers: Any, question_id: str) -> Optional[float]:
    """The noul of one answer, or ``None`` when it is absent or not a noul."""
    get = getattr(answers, "get", None)
    if not callable(get):
        return None
    value = getattr(get(question_id), "noul", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _verdict(judgment: Any) -> Optional[Tuple[float, float]]:
    """Both nouls for one fragment, or ``None`` when it was not judged."""
    if judgment is None:
        return None
    error = getattr(judgment, "error", "")
    if error:
        return None
    answers = getattr(judgment, "answers", None)
    relevance = _noul(answers, RELEVANCE_QUESTION)
    evidence = _noul(answers, EVIDENCE_QUESTION)
    if relevance is None or evidence is None:
        return None
    return relevance, evidence


def apply_keep_rule(
    memories: Sequence[Dict[str, Any]],
    result: Any,
    *,
    limit: int,
    policy: ContextOptimizationPolicy,
) -> List[Dict[str, Any]]:
    """The memories to deliver, in pipeline order.

    A judged memory is kept when it is about the query's subject *and* states
    something usable in the answer.  An unjudged one (``partial`` result,
    per-fragment error, missing answer) is treated exactly as it is today: kept
    if it is within the caller's own ``limit`` positions, dropped otherwise.
    Nothing pads the result — delivering zero memories is a valid outcome, and
    the measured point of the feature.
    """
    by_id: Dict[str, Any] = {}
    for judgment in getattr(result, "fragments", None) or []:
        key = str(getattr(judgment, "fragment_id", "") or "")
        if key:
            by_id[key] = judgment

    kept: List[Dict[str, Any]] = []
    for position, mem in enumerate(memories):
        verdict = _verdict(by_id.get(fragment_id(position)))
        if verdict is None:
            if position < limit:
                kept.append(mem)
            continue
        relevance, evidence = verdict
        if relevance >= policy.relevance_min and evidence >= policy.evidence_min:
            kept.append(mem)
    return kept


@dataclass(frozen=True)
class OptimizationOutcome:
    """What engage should deliver, and whether a judgment decided it."""

    memories: List[Dict[str, Any]]
    status: str
    candidates: int
    reason: str = ""


def _fallback(
    memories: Sequence[Dict[str, Any]], limit: int, reason: str,
) -> OptimizationOutcome:
    """Today's delivery: the first *limit* recalled candidates."""
    return OptimizationOutcome(
        memories=list(memories[:limit]),
        status=STATUS_FALLBACK,
        candidates=len(memories),
        reason=reason,
    )


def optimize_recall(
    query: str,
    memories: Sequence[Dict[str, Any]],
    *,
    limit: int,
    policy: ContextOptimizationPolicy,
    scope: str = "",
    client: Optional[EvaluationClient] = None,
    clock: Callable[[], float] = time.monotonic,
) -> OptimizationOutcome:
    """Decide which of *memories* to deliver for *query*.

    *limit* is the caller's own limit — the size of today's delivery, and the
    fallback everywhere this feature cannot decide.  This function never
    raises: every failure is an outcome with ``status="fallback"`` and a short
    reason, because a memory tool that fails to answer because its optimizer
    failed is worse than one that answers the old way.
    """
    candidates = list(memories)
    now = clock()
    if is_backed_off(scope, now=now):
        return _fallback(candidates, limit, "backoff")
    if not candidates:
        return _fallback(candidates, limit, "no_candidates")

    fragments = build_fragments(candidates, policy)
    questions = [dict(question) for question in QUESTIONS]
    try:
        result = (client or SdkEvaluationClient()).evaluate(
            query, fragments, questions, timeout_ms=policy.timeout_ms,
        )
    except EvaluationUnavailable:
        return _fallback(candidates, limit, "sdk_unavailable")
    except Exception as exc:
        reason = _grpc_reason(exc)
        seconds = _backoff_seconds(reason)
        if seconds:
            _start_backoff(scope, seconds, reason, now)
        else:
            logger.debug("context optimization failed (%s): %s", reason, exc)
        return _fallback(candidates, limit, reason)

    status = str(getattr(result, "status", "") or "")
    if status not in _USABLE_STATUSES:
        reason = status if status in _KNOWN_STATUSES else "unknown_status"
        seconds = _backoff_seconds(reason)
        if seconds:
            _start_backoff(scope, seconds, reason, now)
        return _fallback(candidates, limit, reason)

    kept = apply_keep_rule(candidates, result, limit=limit, policy=policy)
    logger.debug(
        "context optimization delivered %d of %d judged candidates.",
        len(kept), len(candidates),
    )
    return OptimizationOutcome(
        memories=kept, status=STATUS_APPLIED, candidates=len(candidates),
    )
