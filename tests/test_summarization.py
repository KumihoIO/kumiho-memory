import asyncio
import json
import os
from unittest.mock import patch

from kumiho_memory.summarization import MemorySummarizer


class StubAdapter:
    """Minimal LLMAdapter that returns a canned response."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def chat(self, *, messages, model, system="", max_tokens=1024, json_mode=False):
        return self._response


class StubResponsesClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.responses = self
        self.chat = type("ChatNamespace", (), {
            "completions": type("CompletionNamespace", (), {
                "create": self._unexpected_chat_completion_call,
            })()
        })()
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type("StubResponse", (), {"output_text": self.output_text, "output": []})()

    async def _unexpected_chat_completion_call(self, **kwargs):
        raise AssertionError(f"chat.completions.create should not be called: {kwargs}")


def test_summarize_conversation_with_stubbed_llm():
    canned = json.dumps({
        "type": "summary",
        "title": "Async preference",
        "summary": "User prefers async communication.",
        "knowledge": {"facts": [], "decisions": [], "actions": [], "open_questions": []},
        "classification": {"topics": ["communication"], "entities": []},
    })
    summarizer = MemorySummarizer(adapter=StubAdapter(canned), model="stub")

    messages = [
        {"role": "user", "content": "I prefer async communication."},
        {"role": "assistant", "content": "Understood."},
    ]

    result = asyncio.run(summarizer.summarize_conversation(messages))
    assert result["type"] == "summary"
    assert "communication" in result["summary"].lower()


def test_extract_topics_with_stubbed_llm():
    summarizer = MemorySummarizer(adapter=StubAdapter("memory, agents, redis"), model="stub")

    topics = asyncio.run(summarizer.extract_topics("Memory systems for agents."))
    assert topics == ["memory", "agents", "redis"]


def test_custom_adapter_protocol():
    """Verify any object with a chat() method works as an adapter."""

    class GeminiStub:
        async def chat(self, *, messages, model, system="", max_tokens=1024, json_mode=False):
            return json.dumps({
                "type": "fact",
                "title": "Gemini works",
                "summary": "Custom adapters are supported.",
                "knowledge": {"facts": [], "decisions": [], "actions": [], "open_questions": []},
                "classification": {"topics": ["testing"], "entities": []},
            })

    summarizer = MemorySummarizer(adapter=GeminiStub(), model="gemini-2.0-flash")
    assert summarizer.provider == "custom"

    messages = [{"role": "user", "content": "Does Gemini work?"}]
    result = asyncio.run(summarizer.summarize_conversation(messages))
    assert result["type"] == "fact"
    assert result["classification"]["topics"] == ["testing"]


def test_openai_compat_adapter_uses_responses_api_for_codex_models():
    from kumiho_memory.summarization import OpenAICompatAdapter

    client = StubResponsesClient('{"summary":"Codex summary"}')
    adapter = OpenAICompatAdapter(client)

    result = asyncio.run(
        adapter.chat(
            messages=[{"role": "user", "content": "Summarize this"}],
            model="gpt-5-codex",
            system="Return JSON",
            json_mode=True,
        )
    )

    assert result == '{"summary":"Codex summary"}'
    assert client.last_kwargs["model"] == "gpt-5-codex"
    assert "System: Return JSON" in client.last_kwargs["input"]
    assert "User: Summarize this" in client.last_kwargs["input"]


def test_memory_summarizer_adapter_uses_late_kumiho_llm_env():
    from kumiho_memory.summarization import OpenAICompatAdapter

    with patch.dict(os.environ, {}, clear=True):
        summarizer = MemorySummarizer(provider="openai", model="gpt-5-codex")
        stub_adapter = StubAdapter("{}")

        with patch.object(OpenAICompatAdapter, "create", return_value=stub_adapter) as create:
            os.environ["KUMIHO_LLM_API_KEY"] = "late-key"
            os.environ["KUMIHO_LLM_BASE_URL"] = "http://localhost:11434/v1"
            assert summarizer.adapter is stub_adapter

    create.assert_called_once_with(
        api_key="late-key",
        base_url="http://localhost:11434/v1",
    )


def test_normalize_summary_marks_missing_summary_as_error():
    result = MemorySummarizer._normalize_summary(
        {
            "type": "summary",
            "title": "Missing body",
            "knowledge": {},
            "classification": {},
        },
        fallback_messages=[{"role": "user", "content": "We moved the launch to April 15."}],
    )

    assert result["summary"] == "We moved the launch to April 15."
    assert result["error"] == "Summarizer response did not include a non-empty summary"


def test_fallback_summary_ignores_injected_kumiho_blocks():
    messages = [
        {"role": "user", "content": "I replaced the phone battery last week."},
        {
            "role": "assistant",
            "content": (
                "<kumiho_memory>\n"
                "1. Previous note\n"
                "</kumiho_memory>\n"
            ),
        },
    ]

    assert (
        MemorySummarizer._fallback_summary_text(messages)
        == "I replaced the phone battery last week."
    )


def test_fallback_summary_skips_non_user_assistant_messages():
    messages = [
        {"role": "system", "content": "<kumiho_memory>Injected context</kumiho_memory>"},
        {"role": "assistant", "content": "We decided to postpone the migration until April."},
    ]

    assert (
        MemorySummarizer._fallback_summary_text(messages)
        == "We decided to postpone the migration until April."
    )
