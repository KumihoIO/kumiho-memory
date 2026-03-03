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
from typing import Any, Dict, List, Optional

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
