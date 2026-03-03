import asyncio
import json

from kumiho_memory.summarization import MemorySummarizer


class StubAdapter:
    """Minimal LLMAdapter that returns a canned response."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def chat(self, *, messages, model, system="", max_tokens=1024, json_mode=False):
        return self._response


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
