"""Pre-LLM redaction on the conversation-consolidation path (#138).

The consolidation path used to hand RAW buffered messages to both
``summarize_conversation`` and ``generate_implications``; redaction only ran on
what came BACK.  These tests pin the closed leg: what the model SEES is already
screened, while the local artifact keeps the verbatim transcript.

Secret-shaped strings are assembled at RUNTIME by concatenation so no literal
credential shape ever lands in the repository.
"""

import asyncio
import os
import tempfile

from kumiho_memory.memory_manager import (
    UniversalMemoryManager,
    _redact_messages_for_llm,
)
from kumiho_memory.privacy import PIIRedactor
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


# --- planted needles, built at runtime (never a literal in the repo) -------
PLANTED_EMAIL = "alice.roberts" + "@" + "example.com"
PLANTED_CREDENTIAL = "sk-" + ("T" * 10) + ("9" * 14)


class RecordingSummarizer:
    """Records exactly what each of the two LLM entry points received."""

    def __init__(self, summary="User discussed the rollout."):
        self.summary = summary
        self.summarize_calls = []
        self.implication_calls = []

    async def summarize_conversation(self, messages, context=None):
        self.summarize_calls.append([dict(m) for m in messages])
        return {
            "type": "summary",
            "title": "Recorded summary",
            "summary": self.summary,
            "events": [],
            "implications": [],
            "knowledge": {
                "facts": [], "decisions": [], "actions": [], "open_questions": [],
            },
            "classification": {"topics": ["rollout"], "entities": []},
        }

    async def generate_implications(self, messages, context=None):
        self.implication_calls.append([dict(m) for m in messages])
        return []

    def seen_text(self):
        """All message content observed across BOTH calls, concatenated."""
        parts = []
        for call in self.summarize_calls + self.implication_calls:
            for msg in call:
                parts.append(str(msg.get("content", "")))
        return "\n".join(parts)


class ExplodingRedactor(PIIRedactor):
    """Real redactor that blows up on a marked line (pre-LLM inputs only)."""

    def anonymize_summary(self, summary):
        if "BOOM" in summary:
            raise RuntimeError("redactor exploded")
        return super().anonymize_summary(summary)


def _make_manager(tmpdir, summarizer, redactor, stored):
    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://memory/item"}

    return UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test"),
        summarizer=summarizer,
        pii_redactor=redactor,
        memory_store=store_stub,
        consolidation_threshold=2,
        artifact_root=tmpdir,
    )


async def _ingest_and_consolidate(manager, user_message, assistant_message):
    ingest = await manager.ingest_message(
        user_id="user-1", message=user_message, context="personal",
    )
    session_id = ingest["session_id"]
    await manager.add_assistant_response(
        session_id=session_id, response=assistant_message,
    )
    result = await manager.consolidate_session(session_id=session_id)
    return session_id, result


# --------------------------------------------------------------------------
# (a) neither needle is ever observed by EITHER LLM entry point
# --------------------------------------------------------------------------

def test_planted_needles_never_reach_summarizer_or_implications():
    summarizer = RecordingSummarizer()
    stored = {}
    user_message = f"Ping me at {PLANTED_EMAIL} about the rollout."
    assistant_message = f"Noted. Use the key\n{PLANTED_CREDENTIAL}\nfor staging."

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_manager(tmpdir, summarizer, PIIRedactor(), stored)
        session_id, result = asyncio.run(
            _ingest_and_consolidate(manager, user_message, assistant_message)
        )

        assert result["success"] is True

        # BOTH entry points were exercised...
        assert len(summarizer.summarize_calls) == 1
        assert len(summarizer.implication_calls) == 1

        # ...and neither of them ever saw either needle.
        seen = summarizer.seen_text()
        assert PLANTED_EMAIL not in seen
        assert PLANTED_CREDENTIAL not in seen

        # PII is redacted IN PLACE; the credential LINE is dropped whole —
        # exactly the code_capture / code_session policy.
        assert "[email]" in seen
        assert "[redacted]" in seen
        # Structure survives: surrounding prose and both roles are intact.
        assert "about the rollout." in seen
        assert "for staging." in seen
        roles = [
            m.get("role") for m in summarizer.summarize_calls[0]
        ]
        assert roles == [
            m.get("role") for m in summarizer.implication_calls[0]
        ]
        assert "user" in roles and "assistant" in roles

        # (b) the LOCAL artifact still holds the ORIGINAL raw transcript.
        artifact_path = stored.get("artifact_location", "")
        assert artifact_path and os.path.isfile(artifact_path)
        content = open(artifact_path, encoding="utf-8").read()
        assert PLANTED_EMAIL in content
        assert PLANTED_CREDENTIAL in content


# --------------------------------------------------------------------------
# the copy must not mutate the caller's list
# --------------------------------------------------------------------------

def test_redaction_copies_and_never_mutates_the_original():
    messages = [
        {"role": "user", "content": f"mail {PLANTED_EMAIL}", "timestamp": "t0",
         "metadata": {"attachments": []}},
        {"role": "assistant", "content": PLANTED_CREDENTIAL, "timestamp": "t1"},
    ]
    original = [dict(m) for m in messages]

    redacted = _redact_messages_for_llm(PIIRedactor(), messages)

    assert messages == original          # untouched, in place
    assert redacted is not messages
    assert redacted[0] is not messages[0]
    # structure preserved (role/order/timestamp/metadata), content only changed
    assert [m["role"] for m in redacted] == ["user", "assistant"]
    assert redacted[0]["timestamp"] == "t0"
    assert redacted[0]["metadata"] == {"attachments": []}
    assert PLANTED_EMAIL not in redacted[0]["content"]
    assert redacted[1]["content"] == "[redacted]"


def test_redaction_is_a_noop_without_a_redactor():
    messages = [{"role": "user", "content": PLANTED_EMAIL}]
    assert _redact_messages_for_llm(None, messages) is messages


# --------------------------------------------------------------------------
# (c) the post-LLM anonymize_summary layer is unchanged
# --------------------------------------------------------------------------

def test_post_llm_anonymize_summary_layer_unchanged():
    # The model's OUTPUT carries PII of its own — the second layer must still
    # scrub it, independently of the new pre-LLM leg.
    summarizer = RecordingSummarizer(summary=f"Reach the user at {PLANTED_EMAIL}.")
    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_manager(tmpdir, summarizer, PIIRedactor(), stored)
        _, result = asyncio.run(
            _ingest_and_consolidate(manager, "Plain question.", "Plain answer.")
        )

        assert result["success"] is True
        assert PLANTED_EMAIL not in result["summary"]
        assert "[email]" in result["summary"]


# --------------------------------------------------------------------------
# (d) a redaction failure must not crash consolidation
# --------------------------------------------------------------------------

def test_redaction_failure_does_not_crash_consolidation():
    summarizer = RecordingSummarizer()
    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_manager(tmpdir, summarizer, ExplodingRedactor(), stored)
        _, result = asyncio.run(
            _ingest_and_consolidate(manager, "BOOM goes the redactor.", "Okay.")
        )

        assert result["success"] is True
        # fails CLOSED: the line the redactor choked on is dropped, not leaked
        seen = summarizer.seen_text()
        assert "BOOM" not in seen
        assert "[redacted]" in seen
