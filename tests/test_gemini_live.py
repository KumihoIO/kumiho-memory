import asyncio
import os

import pytest

from kumiho_memory.summarization import MemorySummarizer


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _live_gemini_config():
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("KUMIHO_LLM_API_KEY")
    )
    enabled = os.getenv("KUMIHO_RUN_LIVE_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    model = os.getenv("KUMIHO_GEMINI_MODEL", "gemini-2.5-flash")
    return enabled, api_key, model


@pytest.mark.skipif(
    not _live_gemini_config()[0] or not _live_gemini_config()[1],
    reason="Live Gemini test requires KUMIHO_RUN_LIVE_TESTS=1 and a Gemini API key",
)
def test_live_gemini_structured_summary_and_implications():
    enabled, api_key, model = _live_gemini_config()
    assert enabled
    assert api_key

    summarizer = MemorySummarizer(
        provider="gemini",
        model=model,
        light_model=model,
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "On March 15, 2026 at 9:30 PM KST, I rewrote the arXiv "
                "endorsement email draft for Yifei and tightened the AI "
                "responsiveness wording."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Noted. The revised email emphasizes faster response times "
                "and keeps the endorsement request concise."
            ),
        },
    ]

    async def run_probe():
        summary = await summarizer.summarize_conversation(messages, strict=True)
        implications = await summarizer.generate_implications(messages)
        return summary, implications

    summary, implications = asyncio.run(run_probe())

    assert summary["type"] in {"summary", "reflection", "fact", "decision", "action"}
    assert isinstance(summary["title"], str) and summary["title"].strip()
    assert isinstance(summary["summary"], str) and summary["summary"].strip()
    assert isinstance(summary["events"], list)
    assert isinstance(summary["knowledge"]["facts"], list)
    assert isinstance(summary["classification"]["topics"], list)
    assert implications
    assert all(isinstance(item, str) and item.strip() for item in implications)
