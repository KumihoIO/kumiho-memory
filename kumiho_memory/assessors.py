"""Built-in AutoAssessFn implementations for kumiho-memory.

Implements a two-stage pipeline:

Stage 1 — Algorithm (free, instant)
    Signal-word heuristic + minimum content length + hard-skip patterns
    for trivial messages (greetings, acknowledgements, pure code).
    No LLM call is made when Stage 1 fails.

Stage 2 — Graph novelty check (free, uses recall results)
    If the top recalled memory already has cosine similarity > threshold,
    the content is already stored — skip the LLM call.

Stage 3 — LLM judgment (only when Stages 1 & 2 pass)
    A configurable policy instruction (``storage_policy``) acts as the
    "rule book".  The LLM sees the message window + existing memories
    and decides whether to persist, what to extract, and how to type it.

Usage::

    from kumiho_memory.assessors import create_llm_assessor
    from kumiho_memory import AnthropicAdapter

    adapter = AnthropicAdapter(api_key=..., model="claude-haiku-4-5-20251001")
    assessor = create_llm_assessor(adapter)

    manager = UniversalMemoryManager(auto_assess_fn=assessor)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional

from kumiho_memory.evidence import (
    CORROBORATED,
    SINGLE_SOURCE,
    UNVERIFIED,
    parse_evidence,
)
from kumiho_memory.memory_manager import AutoAssessFn, MemoryAssessResult

logger = logging.getLogger(__name__)


# ── Store-signal patterns ──────────────────────────────────────────────────────
# Presence of any of these suggests a message may contain something worth storing.
_STORE_SIGNALS: List[str] = [
    r"\b(prefer|preference|i\s+like|i\s+love|i\s+hate|i\s+dislike|i\s+always\s+use)\b",
    r"\b(always|never|usually|typically|tend\s+to)\b",
    r"\b(decided|decision|chose|choosing|going\s+with|let'?s\s+(use|go\s+with|switch\s+to))\b",
    r"\b(remember(\s+this)?|important|key\s+(point|thing|detail|fact))\b",
    r"\b(my\s+(workflow|setup|config|configuration|approach|style|rule|convention|stack))\b",
    r"\b(we\s+(agreed|decided|chose|picked|settled\s+on))\b",
    r"\b(don'?t\s+(ever|forget|use|do)|avoid|should\s+(always|never))\b",
    r"\b(i\s+use|i\s+rely\s+on|we\s+use|our\s+(team|project)\s+uses)\b",
]

# ── Hard-skip patterns ────────────────────────────────────────────────────────
# Messages matching these are trivially uninteresting; skip without LLM call.
_SKIP_PATTERNS: List[str] = [
    # Short acknowledgements / filler
    r"^(ok|okay|sure|yes|no|nope|yep|yup|got\s+it|sounds\s+good|perfect|great|awesome|cool"
    r"|thanks?|thank\s+you|ty|cheers|acknowledged?|understood?|will\s+do|done|makes?\s+sense"
    r"|of\s+course|absolutely|exactly|right|correct|agreed)[.!?\s]*$",
    # "Here is / Let me / I'll..." short openers (assistant scaffolding)
    r"^(let\s+me|here'?s|here\s+is|i'll|i\s+will|sure[,!]?\s+here)[^.!?]{0,60}[.!]?\s*$",
    # Pure code fence content (skip — ephemeral, not a memory candidate)
    r"^```[\s\S]{0,2000}```\s*$",
]

_STORE_RE = re.compile("|".join(_STORE_SIGNALS), re.IGNORECASE)
_SKIP_RE = re.compile("|".join(_SKIP_PATTERNS), re.IGNORECASE | re.DOTALL)

# Minimum combined character length across the message window.
# Anything shorter is almost certainly not a memory candidate.
_MIN_COMBINED_CHARS: int = 60

# Cosine similarity threshold above which a recalled memory is considered
# near-duplicate — skip the LLM call and return should_store=False.
_DUPLICATE_SCORE_THRESHOLD: float = 0.90

# Default storage policy instruction injected into the LLM system prompt.
# Callers can override this via the ``storage_policy`` parameter.
DEFAULT_STORAGE_POLICY: str = """Store ONLY content that is concretely useful in a future conversation:
- Explicit user preferences ("I prefer X over Y", "I always use Z")
- Decisions with clear rationale ("We chose gRPC because...")
- Stable facts about the user or project (tech stack, team structure, constraints)
- Corrections ("Actually the right approach is...", "Never do X because...")

Do NOT store:
- Transactional exchanges (greetings, "okay", "thanks")
- Ephemeral task output (code, data, tables) — these are not reusable memories
- Content already present in the existing memories list
- Vague statements with no actionable specificity"""


# ── Pre-filter algorithm ───────────────────────────────────────────────────────

def heuristic_prefilter(messages: List[Dict[str, Any]]) -> bool:
    """Fast O(n) pre-filter; no LLM call.

    Returns ``True`` if the message window looks like a memory candidate
    worth passing to the LLM assessor.  Returns ``False`` to skip entirely.

    Stages
    ------
    1. Minimum combined content length.
    2. Hard-skip patterns on the last message.
    3. Store-signal pattern match across the window.
    """
    relevant = [m for m in messages[-4:] if m.get("content")]
    if not relevant:
        return False

    combined = " ".join(m["content"] for m in relevant)

    # 1. Too short to be interesting
    if len(combined.strip()) < _MIN_COMBINED_CHARS:
        return False

    # 2. Last message matches a hard-skip pattern
    last_content = relevant[-1]["content"].strip()
    if _SKIP_RE.match(last_content):
        return False

    # 3. Store-signal present in combined window
    return bool(_STORE_RE.search(combined))


# ── LLM assessor factory ───────────────────────────────────────────────────────

def create_llm_assessor(
    adapter: Any,
    model: str = "",
    *,
    storage_policy: str = DEFAULT_STORAGE_POLICY,
    duplicate_score_threshold: float = _DUPLICATE_SCORE_THRESHOLD,
    skip_heuristic: bool = False,
) -> AutoAssessFn:
    """Create a two-stage :data:`AutoAssessFn` backed by any ``LLMAdapter``.

    Parameters
    ----------
    adapter:
        Any ``LLMAdapter`` (``AnthropicAdapter``, ``OpenAICompatAdapter``, etc.).
        The callable is model-agnostic — the adapter handles provider differences.
    model:
        Model identifier string passed to ``adapter.chat()``.  Defaults to the
        adapter's own default when empty.  Prefer a fast/cheap model (e.g.
        ``claude-haiku-4-5-20251001``) since this runs after every turn.
    storage_policy:
        Plain-text instruction block that defines WHAT to store and what to
        skip.  Defaults to :data:`DEFAULT_STORAGE_POLICY`.  Override to
        customise the memory policy for your application.
    duplicate_score_threshold:
        Cosine similarity cutoff above which a recalled memory is treated as a
        near-duplicate and the LLM call is skipped.  Default ``0.90``.
    skip_heuristic:
        When ``True``, bypass Stage 1 (heuristic pre-filter) and always invoke
        the LLM.  Useful for testing or when you want exhaustive coverage.

    Returns
    -------
    An ``async (messages, recalled) → MemoryAssessResult`` callable.
    """

    system_prompt = (
        "You are a memory relevance judge for an AI assistant's long-term memory system.\n\n"
        "Storage policy (follow exactly):\n"
        f"{storage_policy}\n\n"
        "You will receive:\n"
        "- EXISTING MEMORIES: summaries of what is already stored\n"
        "- RECENT CONVERSATION: the last few turns of dialogue\n\n"
        "Respond with ONLY valid JSON, no markdown fences:\n"
        '{"should_store": true/false, "content": "...", '
        '"memory_type": "fact|decision|preference|summary", "reason": "..."}\n\n'
        "Rules:\n"
        "- content must be a self-contained statement (no pronouns without context)\n"
        "- content must NOT duplicate anything already in EXISTING MEMORIES\n"
        "- if should_store is false, content may be empty string"
    )

    async def _assess(
        messages: List[Dict[str, Any]],
        recalled: List[Dict[str, Any]],
    ) -> MemoryAssessResult:
        # ── Stage 1: heuristic pre-filter (free, instant) ────────────────────
        if not skip_heuristic and not heuristic_prefilter(messages):
            logger.debug("auto_assess: heuristic skip")
            return MemoryAssessResult(should_store=False, reason="heuristic skip")

        # ── Stage 2: graph novelty check (free, uses recall scores) ──────────
        if recalled:
            top_score = recalled[0].get("score", 0.0)
            if isinstance(top_score, (int, float)) and top_score >= duplicate_score_threshold:
                logger.debug(
                    "auto_assess: near-duplicate skip (score=%.3f >= %.3f)",
                    top_score,
                    duplicate_score_threshold,
                )
                return MemoryAssessResult(
                    should_store=False,
                    reason=f"near-duplicate in graph (score={top_score:.3f})",
                )

        # ── Stage 3: LLM novelty judgment ─────────────────────────────────────
        # Build the message window (last 6 turns, truncated per message)
        window_lines: List[str] = []
        for m in messages[-6:]:
            role = m.get("role", "?").upper()
            content = (m.get("content") or "")[:600]
            window_lines.append(f"{role}: {content}")
        window_text = "\n".join(window_lines)

        # Build existing-memories context
        memory_lines: List[str] = []
        for mem in recalled[:5]:
            title = mem.get("title", "")
            summary = mem.get("summary", "")
            line = f"- {title}: {summary}" if title else f"- {summary}"
            if line.strip() != "-":
                memory_lines.append(line)
        memory_context = "\n".join(memory_lines) or "(none)"

        user_msg = (
            f"EXISTING MEMORIES:\n{memory_context}\n\n"
            f"RECENT CONVERSATION:\n{window_text}"
        )

        try:
            raw: str = await adapter.chat(
                messages=[{"role": "user", "content": user_msg}],
                model=model,
                system=system_prompt,
                max_tokens=250,
            )
            # Strip markdown code fences if the model wraps JSON anyway
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)

            data: Dict[str, Any] = json.loads(cleaned)
            result = MemoryAssessResult(
                should_store=bool(data.get("should_store", False)),
                content=str(data.get("content", "")),
                memory_type=str(data.get("memory_type", "fact")),
                reason=str(data.get("reason", "")),
            )
            logger.debug(
                "auto_assess LLM: should_store=%s type=%s reason=%s",
                result.should_store,
                result.memory_type,
                result.reason[:80],
            )
            return result

        except json.JSONDecodeError as exc:
            logger.warning("auto_assess: LLM returned invalid JSON: %s | raw=%r", exc, raw[:200])
            return MemoryAssessResult(should_store=False, reason=f"json parse error: {exc}")
        except Exception as exc:
            logger.warning("auto_assess: LLM call failed: %s", exc)
            return MemoryAssessResult(should_store=False, reason=f"assessor error: {exc}")

    return _assess


# ── Evidence-aware assessor (Level-of-Evidence epic) ───────────────────────────

@dataclass
class EvidencePolicy:
    """Policy knobs for :func:`create_evidence_assessor`.

    Attributes
    ----------
    min_corroboration:
        Minimum number of agreeing memories with *distinct*, non-empty
        ``source`` values required to promote a claim to ``corroborated``.
    official_tags:
        Tags that mark a recalled memory as pinned — a claim contradicting
        such a memory is stored as ``unverified`` with the conflict noted,
        and the pinned belief stands.  Default: ``evidence:official`` only.
        The bare ``published`` tag is deliberately NOT included — this
        codebase stamps ``published`` on virtually every stored revision
        as its currency tag, so pinning on it would fire for ordinary
        memories.  Deployments that use ``published`` as a curated marker
        can add it: ``official_tags=frozenset({"evidence:official",
        "published"})``.
    create_supports_edges:
        When ``True``, corroborating revision krefs are returned on the
        assess result so the manager creates ``SUPPORTS`` edges after the
        store completes.
    create_contradicts_edges:
        Kill-switch for the ``CONTRADICTS`` edge bridge (default ``True`` —
        it is the feature). When ``False`` the manager skips the edge
        bridge; the ``conflicts_with`` metadata is untouched either way
        (it predates the bridge and stays the canonical conflict record).
    storage_policy:
        WHAT-to-store instruction block (same contract as
        :func:`create_llm_assessor`).
    duplicate_score_threshold:
        Near-duplicate recall score above which the LLM call is skipped.
    """

    min_corroboration: int = 2
    official_tags: FrozenSet[str] = frozenset({"evidence:official"})
    create_supports_edges: bool = False
    create_contradicts_edges: bool = True
    storage_policy: str = DEFAULT_STORAGE_POLICY
    duplicate_score_threshold: float = _DUPLICATE_SCORE_THRESHOLD


def grade_evidence(
    recalled: List[Dict[str, Any]],
    agrees_with: List[int],
    contradicts: List[int],
    policy: EvidencePolicy,
) -> Dict[str, Any]:
    """Apply the evidence policy to LLM agree/contradict judgments.

    Pure function (no LLM, no I/O) so the policy rules are unit-testable
    in isolation.  ``agrees_with`` / ``contradicts`` are 1-based indices
    into *recalled*; out-of-range or malformed entries are dropped.

    Rules, in order:

    1. **Official pinning** — the claim contradicts a memory carrying an
       ``official``-grade tag (``policy.official_tags``) → grade
       ``unverified``, conflict recorded, pinned belief untouched.
    2. **Corroboration** — ≥ ``max(1, min_corroboration)`` agreeing
       memories with distinct non-empty sources and zero contradictions →
       ``corroborated`` (``memory_type`` forced to ``"fact"``).
    3. **Single source** — an agreeing memory or claim source is
       identified, no corroboration — resolved by the caller from the
       claim's own source (this function reports ``has_agreement``).
    4. Default — ``unverified``.

    Returns a dict: ``evidence_level``, ``memory_type`` (or ``None``),
    ``supporting_krefs``, ``conflicting_krefs``, ``pinned``,
    ``has_agreement``.
    """
    def _clean(indices: Any) -> List[int]:
        if not isinstance(indices, list):
            return []
        out: List[int] = []
        for i in indices:
            # bool is an int subclass — a JSON `true` must not silently
            # become index 1.
            if (
                isinstance(i, int)
                and not isinstance(i, bool)
                and 1 <= i <= len(recalled)
                and i not in out
            ):
                out.append(i)
        return out

    agrees = _clean(agrees_with)
    conflicts = _clean(contradicts)

    def _mem(i: int) -> Dict[str, Any]:
        return recalled[i - 1]

    def _krefs(indices: List[int]) -> List[str]:
        return [
            str(_mem(i).get("kref", "")) for i in indices if _mem(i).get("kref")
        ]

    # Rule 1 — official pinning
    for i in conflicts:
        tags = set(_mem(i).get("tags") or ())
        if tags & policy.official_tags or parse_evidence(_mem(i), tags) == "official":
            return {
                "evidence_level": UNVERIFIED,
                "memory_type": None,
                "supporting_krefs": [],
                "conflicting_krefs": _krefs(conflicts),
                "pinned": True,
                "has_agreement": bool(agrees),
            }

    # Rule 2 — corroboration (distinct non-empty sources, no contradiction)
    if not conflicts and agrees:
        distinct_sources = {
            str(_mem(i).get("source", "")).strip()
            for i in agrees
            if str(_mem(i).get("source", "")).strip()
        }
        # Clamp: promotion always needs at least one identified source —
        # min_corroboration=0 must not mint "corroborated" from nothing.
        if len(distinct_sources) >= max(1, policy.min_corroboration):
            return {
                "evidence_level": CORROBORATED,
                "memory_type": "fact",
                "supporting_krefs": _krefs(agrees),
                "conflicting_krefs": [],
                "pinned": False,
                "has_agreement": True,
            }

    # Rules 3/4 — resolved by the caller (depends on the claim's own source)
    return {
        "evidence_level": None,
        "memory_type": None,
        "supporting_krefs": [],
        "conflicting_krefs": _krefs(conflicts),
        "pinned": False,
        "has_agreement": bool(agrees and not conflicts),
    }


def create_evidence_assessor(
    adapter: Any,
    model: str = "",
    *,
    policy: Optional[EvidencePolicy] = None,
    skip_heuristic: bool = False,
) -> AutoAssessFn:
    """Create an evidence-aware :data:`AutoAssessFn` (screened revision).

    Same three-stage pipeline as :func:`create_llm_assessor`, but the LLM
    additionally judges which existing memories the new claim agrees with
    or contradicts, and a deterministic policy pass
    (:func:`grade_evidence`) turns that into an evidence grade:

    - claims contradicting ``official``/``published`` memories are stored
      as ``unverified`` with the conflict recorded — the pinned belief is
      never revised at write time
    - claims corroborated by ≥ N distinctly-sourced memories are promoted
      to ``corroborated`` facts (optionally linked with ``SUPPORTS`` edges)
    - the assessor never emits ``official`` — that grade is operator-only

    Parameters
    ----------
    adapter:
        Any ``LLMAdapter``; prefer a fast model (runs after every turn).
    model:
        Model identifier passed to ``adapter.chat()``.
    policy:
        :class:`EvidencePolicy` instance; defaults to ``EvidencePolicy()``.
    skip_heuristic:
        Bypass the Stage-1 heuristic pre-filter (testing / exhaustive mode).
    """
    policy = policy or EvidencePolicy()

    system_prompt = (
        "You are a memory relevance and evidence judge for an AI assistant's "
        "long-term memory system.\n\n"
        "Storage policy (follow exactly):\n"
        f"{policy.storage_policy}\n\n"
        "You will receive:\n"
        "- EXISTING MEMORIES: a NUMBERED list of stored memories, each with "
        "its evidence grade and source when known\n"
        "- RECENT CONVERSATION: the last few turns of dialogue\n\n"
        "Respond with ONLY valid JSON, no markdown fences:\n"
        '{"should_store": true/false, "content": "...", '
        '"memory_type": "fact|decision|preference|summary", "reason": "...", '
        '"agrees_with": [1, 3], "contradicts": [2], "source": ""}\n\n'
        "Rules:\n"
        "- content must be a self-contained statement (no pronouns without context)\n"
        "- content must NOT duplicate anything already in EXISTING MEMORIES\n"
        "- agrees_with: numbers of existing memories asserting the SAME claim\n"
        "- contradicts: numbers of existing memories asserting the OPPOSITE\n"
        "- source: where the claim comes from if identifiable (e.g. "
        '"news:reuters", "press-release:acme", "chat:user"), else empty\n'
        "- if should_store is false, content may be empty string"
    )

    async def _assess(
        messages: List[Dict[str, Any]],
        recalled: List[Dict[str, Any]],
    ) -> MemoryAssessResult:
        # ── Stage 1: heuristic pre-filter ────────────────────────────────────
        if not skip_heuristic and not heuristic_prefilter(messages):
            logger.debug("evidence_assess: heuristic skip")
            return MemoryAssessResult(should_store=False, reason="heuristic skip")

        # ── Stage 2: graph novelty check ─────────────────────────────────────
        if recalled:
            top_score = recalled[0].get("score", 0.0)
            if isinstance(top_score, (int, float)) and top_score >= policy.duplicate_score_threshold:
                logger.debug(
                    "evidence_assess: near-duplicate skip (score=%.3f)", top_score,
                )
                return MemoryAssessResult(
                    should_store=False,
                    reason=f"near-duplicate in graph (score={top_score:.3f})",
                )

        # ── Stage 3: LLM judgment with evidence context ──────────────────────
        window_lines: List[str] = []
        for m in messages[-6:]:
            role = m.get("role", "?").upper()
            content = (m.get("content") or "")[:600]
            window_lines.append(f"{role}: {content}")
        window_text = "\n".join(window_lines)

        shown = recalled[:5]
        memory_lines: List[str] = []
        for idx, mem in enumerate(shown, start=1):
            grade = parse_evidence(mem, mem.get("tags") or ()) or "ungraded"
            src = str(mem.get("source", "")).strip() or "unknown source"
            title = mem.get("title", "")
            summary = mem.get("summary", "")
            body = f"{title}: {summary}" if title else str(summary)
            memory_lines.append(f"{idx}. [{grade} | {src}] {body}")
        memory_context = "\n".join(memory_lines) or "(none)"

        user_msg = (
            f"EXISTING MEMORIES:\n{memory_context}\n\n"
            f"RECENT CONVERSATION:\n{window_text}"
        )

        # Pre-bind so the JSONDecodeError handler below can safely log it
        # even when adapter.chat() itself raises JSONDecodeError.
        raw: str = ""
        try:
            raw = await adapter.chat(
                messages=[{"role": "user", "content": user_msg}],
                model=model,
                system=system_prompt,
                max_tokens=350,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)

            data: Dict[str, Any] = json.loads(cleaned)

            grade = grade_evidence(
                shown,
                data.get("agrees_with", []),
                data.get("contradicts", []),
                policy,
            )

            claim_source = str(data.get("source", "") or "").strip()[:200]
            level = grade["evidence_level"]
            if level is None:
                # Rules 3/4: identified source → single_source; else unverified.
                level = SINGLE_SOURCE if claim_source else UNVERIFIED

            result = MemoryAssessResult(
                should_store=bool(data.get("should_store", False)),
                content=str(data.get("content", "")),
                memory_type=grade["memory_type"] or str(data.get("memory_type", "fact")),
                reason=str(data.get("reason", "")),
                evidence_level=level,
                source=claim_source,
                supporting_krefs=(
                    grade["supporting_krefs"] if policy.create_supports_edges else []
                ),
                # conflicting_krefs is NOT gated here — it also feeds the
                # conflicts_with metadata, which the edge kill-switch must not
                # touch. The bridge gate rides separately on the result.
                conflicting_krefs=grade["conflicting_krefs"],
                create_contradicts_edges=policy.create_contradicts_edges,
            )
            logger.debug(
                "evidence_assess: should_store=%s level=%s pinned=%s reason=%s",
                result.should_store,
                result.evidence_level,
                grade["pinned"],
                result.reason[:80],
            )
            return result

        except json.JSONDecodeError as exc:
            logger.warning(
                "evidence_assess: LLM returned invalid JSON: %s | raw=%r",
                exc, raw[:200],
            )
            return MemoryAssessResult(should_store=False, reason=f"json parse error: {exc}")
        except Exception as exc:
            logger.warning("evidence_assess: LLM call failed: %s", exc)
            return MemoryAssessResult(should_store=False, reason=f"assessor error: {exc}")

    return _assess
