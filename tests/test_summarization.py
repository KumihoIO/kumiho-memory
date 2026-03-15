import asyncio
import json
import os
import logging
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


class StubChatCompletionsClient:
    def __init__(
        self,
        *,
        fail_on_param: str | None = None,
        empty_content: bool = False,
        parsed_payload=None,
        refusal=None,
    ) -> None:
        self.responses = type("ResponsesNamespace", (), {
            "create": self._unexpected_responses_call,
        })()
        self.chat = type("ChatNamespace", (), {
            "completions": type("CompletionNamespace", (), {
                "create": self._create_chat_completion,
            })()
        })()
        self.fail_on_param = fail_on_param
        self.empty_content = empty_content
        self.parsed_payload = parsed_payload
        self.refusal = refusal
        self.calls = []

    async def _create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_param and self.fail_on_param in kwargs:
            raise ValueError(
                f"Unsupported parameter: '{self.fail_on_param}' is not supported with this model. "
                "Use 'max_completion_tokens' instead."
            )
        response_format = kwargs.get("response_format", {})
        schema = response_format.get("json_schema", {}).get("schema", {})
        schema_type = schema.get("type")
        properties = schema.get("properties", {})
        if "queries" in properties:
            content = '{"queries":["one","two"]}'
        elif "implications" in properties:
            content = '{"implications":["one","two"]}'
        elif schema_type == "array":
            content = '["one","two"]'
        elif schema_type == "object":
            content = '{"summary":"Structured JSON"}'
        else:
            content = kwargs.get("messages", [{}])[-1].get("content", "")
        if self.empty_content:
            content = None
        return type("StubResponse", (), {
            "choices": [type("StubChoice", (), {
                "message": type("StubMessage", (), {
                    "content": (
                        f"ok:{content}"
                        if isinstance(content, str) and not content.startswith("{") and not content.startswith("[")
                        else content
                    ),
                    "parsed": self.parsed_payload,
                    "refusal": self.refusal,
                    "tool_calls": [],
                })(),
                "finish_reason": "stop",
            })()]
        })()

    async def _unexpected_responses_call(self, **kwargs):
        raise AssertionError(f"responses.create should not be called: {kwargs}")


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


def test_generate_implications_accepts_wrapped_array_object():
    summarizer = MemorySummarizer(
        adapter=StubAdapter('{"implications":["future travel planning","budget pressure"]}'),
        model="stub",
        light_model="stub-light",
    )

    implications = asyncio.run(
        summarizer.generate_implications(
            [{"role": "user", "content": "I need to save more for trips."}],
        )
    )

    assert implications == ["future travel planning", "budget pressure"]


def test_generate_implications_accepts_fenced_wrapped_object():
    summarizer = MemorySummarizer(
        adapter=StubAdapter(
            '```json\n{"implications":["future travel planning","budget pressure"]}\n```'
        ),
        model="stub",
        light_model="stub-light",
    )

    implications = asyncio.run(
        summarizer.generate_implications(
            [{"role": "user", "content": "I need to save more for trips."}],
        )
    )

    assert implications == ["future travel planning", "budget pressure"]


def test_summarize_conversation_includes_debug_payload_on_invalid_json():
    summarizer = MemorySummarizer(
        adapter=StubAdapter("not valid json at all"),
        provider="openai",
        model="gpt-5-mini",
    )

    result = asyncio.run(
        summarizer.summarize_conversation(
            [{"role": "user", "content": "Summarize this please."}],
        )
    )

    assert result["error"] == "No valid JSON found in summarizer response"
    assert result["debug"]["provider"] == "openai"
    assert result["debug"]["model"] == "gpt-5-mini"
    assert result["debug"]["raw_response_len"] == len("not valid json at all")
    assert result["debug"]["raw_response_preview"] == "not valid json at all"


def test_summarize_conversation_logs_diagnostics_on_invalid_json(caplog):
    summarizer = MemorySummarizer(
        adapter=StubAdapter("not valid json at all"),
        provider="openai",
        model="gpt-5-mini",
    )

    with caplog.at_level(logging.WARNING, logger="kumiho_memory.summarization"):
        asyncio.run(
            summarizer.summarize_conversation(
                [{"role": "user", "content": "Summarize this please."}],
            )
        )

    assert "summarize_conversation failed: No valid JSON found in summarizer response" in caplog.text
    assert "provider=openai" in caplog.text
    assert "model=gpt-5-mini" in caplog.text
    assert "raw_preview='not valid json at all'" in caplog.text


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


def test_openai_compat_adapter_uses_max_completion_tokens_for_gpt5_chat_models():
    from kumiho_memory.summarization import OpenAICompatAdapter, build_summary_schema_mode

    client = StubChatCompletionsClient()
    adapter = OpenAICompatAdapter(client)

    result = asyncio.run(
        adapter.chat(
            messages=[{"role": "user", "content": "Summarize this"}],
            model="gpt-5.4",
            max_tokens=321,
            json_mode=build_summary_schema_mode(),
        )
    )

    assert result == '{"summary":"Structured JSON"}'
    assert client.calls[0]["max_completion_tokens"] == 321
    assert client.calls[0]["reasoning_effort"] == "none"
    assert "max_tokens" not in client.calls[0]
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "summary" in schema["properties"]


def test_openai_compat_adapter_uses_json_schema_for_array_mode():
    from kumiho_memory.summarization import OpenAICompatAdapter, build_string_array_wrapper_schema

    client = StubChatCompletionsClient()
    adapter = OpenAICompatAdapter(client)

    result = asyncio.run(
        adapter.chat(
            messages=[{"role": "user", "content": "Return items"}],
            model="gpt-5-mini",
            max_tokens=123,
            json_mode=build_string_array_wrapper_schema("kumiho_queries_response", "queries"),
        )
    )

    assert result == '{"queries":["one","two"]}'
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert client.calls[0]["reasoning_effort"] == "minimal"
    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["queries"]["type"] == "array"


def test_openai_compat_adapter_uses_json_object_for_non_native_schema_mode():
    from kumiho_memory.summarization import OpenAICompatAdapter, build_summary_schema_mode

    client = StubChatCompletionsClient()
    adapter = OpenAICompatAdapter(
        client,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    asyncio.run(
        adapter.chat(
            messages=[{"role": "user", "content": "Summarize this"}],
            model="gemini-2.5-flash",
            max_tokens=321,
            json_mode=build_summary_schema_mode(),
        )
    )

    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_summarize_conversation_accepts_fenced_json_object():
    canned = (
        "```json\n"
        "{"
        '"type":"summary",'
        '"title":"Async preference",'
        '"summary":"User prefers async communication.",'
        '"events":[],'
        '"knowledge":{"facts":[],"decisions":[],"actions":[],"open_questions":[]},'
        '"classification":{"topics":["communication"],"entities":[]}'
        "}\n"
        "```"
    )
    summarizer = MemorySummarizer(adapter=StubAdapter(canned), model="stub")

    result = asyncio.run(
        summarizer.summarize_conversation(
            [{"role": "user", "content": "I prefer async communication."}],
        )
    )

    assert result["type"] == "summary"
    assert result["summary"] == "User prefers async communication."


def test_openai_compat_adapter_retries_with_max_completion_tokens_when_max_tokens_is_rejected():
    from kumiho_memory.summarization import OpenAICompatAdapter

    client = StubChatCompletionsClient(fail_on_param="max_tokens")
    adapter = OpenAICompatAdapter(client)

    result = asyncio.run(
        adapter.chat(
            messages=[{"role": "user", "content": "Summarize this"}],
            model="gpt-4o",
            max_tokens=222,
        )
    )

    assert result == "ok:Summarize this"
    assert len(client.calls) == 2
    assert client.calls[0]["max_tokens"] == 222
    assert client.calls[1]["max_completion_tokens"] == 222


def test_openai_compat_adapter_uses_parsed_payload_when_content_is_empty():
    from kumiho_memory.summarization import OpenAICompatAdapter, build_summary_schema_mode

    client = StubChatCompletionsClient(
        empty_content=True,
        parsed_payload={"summary": "Structured JSON via parsed"},
    )
    adapter = OpenAICompatAdapter(client)

    result = asyncio.run(
        adapter.chat(
            messages=[{"role": "user", "content": "Summarize this"}],
            model="gpt-5-mini",
            max_tokens=111,
            json_mode=build_summary_schema_mode(),
        )
    )

    assert result == '{"summary": "Structured JSON via parsed"}'


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
