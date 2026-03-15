import asyncio
import os

import pytest

from kumiho_memory.summarization import MemorySummarizer


def _live_openai_config():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("KUMIHO_LLM_API_KEY")
    enabled = os.getenv("KUMIHO_RUN_LIVE_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    model = os.getenv("KUMIHO_OPENAI_MODEL", "gpt-5-mini")
    return enabled, api_key, model


@pytest.mark.skipif(
    not _live_openai_config()[0] or not _live_openai_config()[1],
    reason="Live OpenAI test requires KUMIHO_RUN_LIVE_TESTS=1 and OPENAI_API_KEY",
)
def test_live_openai_structured_summary_handles_longer_transcripts():
    enabled, api_key, model = _live_openai_config()
    assert enabled
    assert api_key

    summarizer = MemorySummarizer(
        provider="openai",
        model=model,
        light_model=model,
        api_key=api_key,
    )

    messages = []
    for i in range(1, 11):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"On March {i}, 2026, I updated Kumiho memory integration task {i}: "
                    f"fixed OpenAI OAuth handling, checked Gemini structured output, "
                    f"revised arXiv email draft {i}, adjusted Dream State page size, "
                    f"logged provider/model/base_url diagnostics, and noted that Yifei "
                    f"needed a clearer endorsement email. I also recorded budget figure "
                    f"{i * 111}, device model X{i}, and deployment note batch-{i}."
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Understood. I summarized task {i}, captured the dates, provider "
                    f"changes, budget figure {i * 111}, device model X{i}, deployment "
                    f"note batch-{i}, and the action to keep the endorsement request "
                    f"concise while emphasizing faster AI responsiveness."
                ),
            }
        )

    async def run_probe():
        summary = await summarizer.summarize_conversation(messages, strict=True)
        implications = await summarizer.generate_implications(messages)
        return summary, implications

    summary, implications = asyncio.run(run_probe())

    assert summary["type"] in {"summary", "reflection", "fact", "decision", "action"}
    assert isinstance(summary["title"], str) and summary["title"].strip()
    assert isinstance(summary["summary"], str) and summary["summary"].strip()
    assert isinstance(summary["events"], list) and summary["events"]
    assert isinstance(summary["knowledge"]["facts"], list)
    assert isinstance(summary["classification"]["topics"], list)
    assert implications
    assert all(isinstance(item, str) and item.strip() for item in implications)
