"""LLM-based summarization for memory consolidation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger(__name__)

JsonMode = Union[bool, str, Dict[str, Any]]


def _strict_object_schema(
    properties: Dict[str, Any],
    *,
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required or list(properties.keys()),
    }


def _json_schema_mode(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "schema": schema,
    }


def build_string_array_wrapper_schema(name: str, field_name: str) -> Dict[str, Any]:
    return _json_schema_mode(
        name,
        _strict_object_schema({
            field_name: {
                "type": "array",
                "items": {"type": "string"},
            },
        }),
    )


def build_summary_schema_mode() -> Dict[str, Any]:
    return _json_schema_mode(
        "kumiho_summary_response",
        _strict_object_schema({
            "type": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "events": {
                "type": "array",
                "items": _strict_object_schema({
                    "event": {"type": "string"},
                    "when": {"type": "string"},
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "consequence": {"type": "string"},
                }),
            },
            "knowledge": _strict_object_schema({
                "facts": {
                    "type": "array",
                    "items": _strict_object_schema({
                        "claim": {"type": "string"},
                        "certainty": {"type": "string"},
                    }),
                },
                "decisions": {
                    "type": "array",
                    "items": _strict_object_schema({
                        "decision": {"type": "string"},
                        "reason": {"type": "string"},
                    }),
                },
                "actions": {
                    "type": "array",
                    "items": _strict_object_schema({
                        "task": {"type": "string"},
                        "status": {"type": "string"},
                    }),
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }),
            "classification": _strict_object_schema({
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }),
        }),
    )


# ---------------------------------------------------------------------------
# LLM Adapter protocol — implement this to add any provider
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMAdapter(Protocol):
    """Interface for LLM providers.

    Implement this protocol to plug in any LLM provider (Gemini, Mistral,
    Ollama, Cohere, etc.).  Built-in adapters are provided for
    OpenAI-compatible and Anthropic APIs.

    Example custom adapter::

        class MyAdapter:
            async def chat(self, *, messages, model, system="",
                           max_tokens=1024, json_mode=False) -> str:
                # call your LLM here
                return response_text
    """

    async def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        system: str = "",
        max_tokens: int = 1024,
        json_mode: JsonMode = False,
    ) -> str:
        """Send a chat request and return the raw response text."""
        ...


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


class OpenAICompatAdapter:
    """Adapter for OpenAI and any OpenAI-compatible API.

    Works with: OpenAI, Azure OpenAI, Google Gemini (via OpenAI compat),
    Ollama, vLLM, LiteLLM, Together, Groq, Mistral, and others.

    Usage::

        # Standard OpenAI
        adapter = OpenAICompatAdapter.create(api_key="sk-...")

        # Gemini via OpenAI-compatible endpoint
        adapter = OpenAICompatAdapter.create(
            api_key="your-gemini-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # Local Ollama
        adapter = OpenAICompatAdapter.create(
            base_url="http://localhost:11434/v1",
        )
    """

    def __init__(self, client: Any, *, base_url: Optional[str] = None) -> None:
        self._client = client
        self._base_url = (base_url or "").rstrip("/")

    @staticmethod
    def _json_preview(value: Any, *, limit: int = 240) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, default=str)
            except Exception:
                text = repr(value)
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    @classmethod
    def _coerce_message_content(cls, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)
                    continue
                if isinstance(item, dict):
                    maybe_text = item.get("text")
                    if isinstance(maybe_text, str) and maybe_text:
                        text_parts.append(maybe_text)
            return "\n".join(part for part in text_parts if part).strip()
        return ""

    def _extract_chat_message_text(
        self,
        *,
        response: Any,
        model: str,
        json_mode: JsonMode,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            logger.warning(
                "chat.completions returned no choices | model=%s base_url=%s json_mode=%s reasoning_effort=%s",
                model,
                self._base_url or "",
                MemorySummarizer._describe_json_mode(json_mode),
                reasoning_effort or "",
            )
            return ""

        choice = choices[0]
        message = getattr(choice, "message", None)
        content = self._coerce_message_content(getattr(message, "content", None))
        if content:
            return content

        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            try:
                return json.dumps(parsed, default=str)
            except Exception:
                pass

        logger.warning(
            "chat.completions returned empty message content | model=%s base_url=%s json_mode=%s reasoning_effort=%s finish_reason=%s refusal=%r parsed_type=%s parsed_preview=%r content_type=%s content_preview=%r tool_calls=%s",
            model,
            self._base_url or "",
            MemorySummarizer._describe_json_mode(json_mode),
            reasoning_effort or "",
            getattr(choice, "finish_reason", ""),
            getattr(message, "refusal", None),
            type(parsed).__name__ if parsed is not None else "",
            self._json_preview(parsed),
            type(getattr(message, "content", None)).__name__,
            self._json_preview(getattr(message, "content", None)),
            len(getattr(message, "tool_calls", None) or []),
        )
        return ""

    @staticmethod
    def _prefers_max_completion_tokens(model: str) -> bool:
        lowered = model.strip().lower()
        return bool(re.match(r"^(gpt-5|o\d)", lowered))

    @staticmethod
    def _supports_alternate_token_param(exc: Exception, *, failed_param: str) -> bool:
        message = str(exc).lower()
        return (
            "unsupported parameter" in message and
            failed_param in message
        )

    @staticmethod
    def _default_reasoning_effort(model: str, json_mode: JsonMode) -> Optional[str]:
        if not json_mode:
            return None
        lowered = model.strip().lower()
        if lowered.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.4")):
            return "none"
        if lowered.startswith("gpt-5"):
            return "minimal"
        return None

    @staticmethod
    def _json_mode_kind(json_mode: JsonMode) -> Optional[str]:
        if isinstance(json_mode, dict):
            return "schema"
        if json_mode == "array":
            return "array"
        if json_mode:
            return "object"
        return None

    def _is_native_openai(self) -> bool:
        return not self._base_url or "api.openai.com" in self._base_url.lower()

    def _response_format(self, json_mode: JsonMode) -> Optional[Dict[str, Any]]:
        mode_kind = self._json_mode_kind(json_mode)
        if not mode_kind:
            return None

        if self._is_native_openai():
            if isinstance(json_mode, dict):
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_mode["name"],
                        "strict": True,
                        "schema": json_mode["schema"],
                    },
                }
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": f"kumiho_{mode_kind}_response",
                    "strict": True,
                    "schema": {"type": mode_kind},
                },
            }

        if mode_kind == "object":
            return {"type": "json_object"}

        return None

    @classmethod
    def create(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "OpenAICompatAdapter":
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install kumiho-memory[openai]"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return cls(AsyncOpenAI(**kwargs), base_url=base_url)

    async def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        system: str = "",
        max_tokens: int = 1024,
        json_mode: JsonMode = False,
    ) -> str:
        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        if "codex" in model.lower():
            prompt_parts: List[str] = []
            for msg in full_messages:
                role = str(msg.get("role", "user")).strip().capitalize()
                content = str(msg.get("content", "")).strip()
                if content:
                    prompt_parts.append(f"{role}: {content}")

            response = await self._client.responses.create(
                model=model,
                input="\n\n".join(prompt_parts),
                max_output_tokens=max_tokens,
            )

            output_text = getattr(response, "output_text", "") or ""
            if output_text:
                return output_text

            output = getattr(response, "output", None) or []
            text_parts: List[str] = []
            for item in output:
                for content in getattr(item, "content", None) or []:
                    text = getattr(content, "text", None)
                    if text:
                        text_parts.append(str(text))
            joined = "\n".join(part for part in text_parts if part).strip()
            if not joined:
                logger.warning(
                    "responses.create returned no output text | model=%s base_url=%s output_items=%s",
                    model,
                    self._base_url or "",
                    len(output),
                )
            return joined

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": full_messages,
        }
        reasoning_effort = self._default_reasoning_effort(model, json_mode)
        token_param = (
            "max_completion_tokens"
            if self._prefers_max_completion_tokens(model)
            else "max_tokens"
        )
        kwargs[token_param] = max_tokens
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response_format = self._response_format(json_mode)
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            alt_param = "max_tokens" if token_param == "max_completion_tokens" else "max_completion_tokens"
            if not self._supports_alternate_token_param(exc, failed_param=token_param):
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop(token_param, None)
            retry_kwargs[alt_param] = max_tokens
            response = await self._client.chat.completions.create(**retry_kwargs)
        return self._extract_chat_message_text(
            response=response,
            model=model,
            json_mode=json_mode,
            reasoning_effort=reasoning_effort,
        )


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def create(cls, *, api_key: Optional[str] = None) -> "AnthropicAdapter":
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required: pip install kumiho-memory[anthropic]"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        return cls(AsyncAnthropic(**kwargs))

    async def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        system: str = "",
        max_tokens: int = 1024,
        json_mode: JsonMode = False,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""


# ---------------------------------------------------------------------------
# Embedding adapter protocol — opt-in for embedding-based sibling filtering
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Interface for text embedding providers.

    Implement this protocol to plug in any embedding provider.  Used by
    the memory manager for embedding-based sibling relevance filtering.
    """

    def embed(
        self,
        texts: List[str],
        *,
        model: str = "",
    ) -> List[List[float]]:
        """Embed a batch of texts and return a list of float vectors."""
        ...


class OpenAICompatEmbeddingAdapter:
    """Embedding adapter for OpenAI and compatible APIs.

    Usage::

        adapter = OpenAICompatEmbeddingAdapter.create(api_key="sk-...")
        vectors = adapter.embed(["hello world", "test"])
    """

    def __init__(self, client: Any, default_model: str = "text-embedding-3-small") -> None:
        self._client = client
        self._default_model = default_model

    @classmethod
    def create(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> "OpenAICompatEmbeddingAdapter":
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install kumiho-memory[openai]"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return cls(OpenAI(**kwargs), default_model=model)

    def embed(
        self,
        texts: List[str],
        *,
        model: str = "",
    ) -> List[List[float]]:
        resolved_model = model or self._default_model
        resp = self._client.embeddings.create(input=texts, model=resolved_model)
        return [item.embedding for item in resp.data]


# ---------------------------------------------------------------------------
# Provider / model defaults
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {
        "model": "gpt-4o",
        "light_model": "gpt-4o-mini",
    },
    "anthropic": {
        "model": "claude-sonnet-4-5-20241022",
        "light_model": "claude-haiku-4-20250414",
    },
}

# Key-prefix heuristics used when the caller only supplies an API key.
_KEY_PREFIX_MAP = {
    "sk-": "openai",
}


def _detect_provider(api_key: str) -> str:
    """Best-effort provider detection from API key prefix."""
    for prefix, provider in _KEY_PREFIX_MAP.items():
        if api_key.startswith(prefix):
            return provider
    # Anthropic keys don't have a stable prefix; default to openai if unsure.
    return "openai"


class MemorySummarizer:
    """Summarize conversations into structured memory.

    Supports any LLM provider through the adapter pattern:

    1. **Built-in providers** (``provider="openai"`` or ``"anthropic"``):
       Auto-configured from env vars or explicit arguments.

    2. **OpenAI-compatible APIs** (Gemini, Ollama, Groq, Together, vLLM…):
       Set ``base_url`` to point at the compatible endpoint.

    3. **Fully custom providers**: Pass any object implementing the
       ``LLMAdapter`` protocol via the ``adapter`` parameter.

    Configuration priority (highest → lowest):

    1. Explicit ``adapter`` — bypasses all other config.
    2. Constructor arguments (``api_key``, ``provider``, ``model``,
       ``base_url``).
    3. Unified env vars: ``KUMIHO_LLM_API_KEY``, ``KUMIHO_LLM_PROVIDER``,
       ``KUMIHO_LLM_MODEL``, ``KUMIHO_LLM_BASE_URL``.
    4. Provider-specific env vars: ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``.
    5. Built-in defaults per provider.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        light_model: Optional[str] = None,
        base_url: Optional[str] = None,
        adapter: Optional[LLMAdapter] = None,
        client: Optional[Any] = None,
    ) -> None:
        # --- Fast path: caller provides a ready adapter ---
        if adapter is not None:
            self._adapter: Optional[LLMAdapter] = adapter
            self.provider = provider or "custom"
            self.model = model or "default"
            self.light_model = light_model or model or "default"
            self.api_key = api_key
            self._client = None
            self._base_url = base_url
            return

        # --- Resolve configuration (lightweight, no SDK init) ---
        resolved_key = api_key or os.getenv("KUMIHO_LLM_API_KEY")

        resolved_base_url = (
            base_url
            or os.getenv("KUMIHO_LLM_BASE_URL", "").strip()
            or None
        )

        resolved_provider = (
            (provider or "").strip().lower()
            or os.getenv("KUMIHO_LLM_PROVIDER", "").strip().lower()
            or self._provider_from_env_keys()
            or (_detect_provider(resolved_key) if resolved_key else "openai")
        )

        defaults = PROVIDER_DEFAULTS.get(resolved_provider, {})

        self.provider = resolved_provider
        self.api_key = resolved_key
        self.model = (
            model
            or os.getenv("KUMIHO_LLM_MODEL", "").strip()
            or defaults.get("model", "gpt-5.3-codex-spark" if "codex" in resolved_provider else "gpt-4o")
        )
        default_light_model = (
            self.model if "codex" in self.model.lower()
            else defaults.get("light_model", self.model)
        )
        self.light_model = (
            light_model
            or os.getenv("KUMIHO_LLM_LIGHT_MODEL", "").strip()
            or default_light_model
        )

        # --- Defer adapter construction until first use ---
        self._adapter = None
        self._client = client
        self._base_url = resolved_base_url

    @property
    def adapter(self) -> LLMAdapter:
        """Lazily build the LLM adapter on first use."""
        if self._adapter is not None:
            return self._adapter

        if self._client is not None:
            if self.provider == "anthropic":
                self._adapter = AnthropicAdapter(self._client)
            else:
                self._adapter = OpenAICompatAdapter(self._client)
        else:
            self._adapter = self._build_adapter(
                self.provider, self.api_key, self._base_url,
            )
        return self._adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def summarize_conversation(
        self,
        messages: List[Dict[str, Any]],
        *,
        context: Optional[str] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Summarize conversation into structured memory."""

        system_prompt = self._system_prompt()
        conversation_text = self._format_messages(messages)
        user_prompt = self._user_prompt(conversation_text, context=context)
        json_mode = build_summary_schema_mode()
        raw = ""

        try:
            raw = await self.adapter.chat(
                messages=[{"role": "user", "content": user_prompt}],
                model=self.model,
                system=system_prompt,
                max_tokens=2560,
                json_mode=json_mode,
            )
            result = self._parse_json(raw)
        except Exception as exc:
            diagnostics = self._build_diagnostics(
                model=self.model,
                json_mode=json_mode,
                raw_response=raw,
            )
            self._log_llm_failure(
                "summarize_conversation",
                error=exc,
                diagnostics=diagnostics,
            )
            if strict:
                raise
            return self._fallback_summary(messages, error=str(exc), debug=diagnostics)

        return self._normalize_summary(result, fallback_messages=messages)

    async def extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        prompt = (
            "Extract 3-5 concise topic keywords from this text.\n\n"
            f"{text}\n\nTopics (comma-separated):"
        )
        response = await self.adapter.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.light_model,
            max_tokens=60,
        )
        topics = [item.strip().lower() for item in response.split(",") if item.strip()]
        return topics

    async def generate_implications(
        self,
        messages: List[Dict[str, Any]],
        *,
        context: Optional[str] = None,
    ) -> List[str]:
        """Generate prospective implications using the light model.

        Runs independently of summarization so it can be called in parallel.
        Returns 3-5 hypothetical future situations where this conversation
        would be the missing context, using *different* vocabulary than the
        original text to bridge semantic gaps in vector search.
        """
        conversation_text = self._format_messages(messages)
        prompt = (
            "Read this conversation and imagine someone months later says or "
            "does something that ONLY makes sense because of what happened here.\n"
            "Generate 3-5 short descriptions of those future situations.\n"
            "Rules:\n"
            "- Use DIFFERENT vocabulary than the original conversation\n"
            "- Focus on downstream behavioral changes, anxieties, preferences, "
            "or habits that would result from these events\n"
            "- Each description should be 1 sentence, max 20 words\n"
            "- Return a JSON array of strings, nothing else\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            "JSON array:"
        )

        raw = ""
        json_mode = build_string_array_wrapper_schema(
            "kumiho_implications_response",
            "implications",
        )
        try:
            raw = await self.adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.light_model,
                max_tokens=512,
                json_mode=json_mode,
            )
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                implications = parsed.get("implications")
                if isinstance(implications, list):
                    return [str(item).strip() for item in implications if item][:5]
                for val in parsed.values():
                    if isinstance(val, list):
                        return [str(item).strip() for item in val if item][:5]
        except Exception as exc:
            self._log_llm_failure(
                "generate_implications",
                error=exc,
                diagnostics=self._build_diagnostics(
                    model=self.light_model,
                    json_mode=json_mode,
                    raw_response=raw,
                ),
            )

        return []

    # ------------------------------------------------------------------
    # Adapter construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_adapter(
        provider: str,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> LLMAdapter:
        resolved_base_url = (
            base_url
            or os.getenv("KUMIHO_LLM_BASE_URL", "").strip()
            or None
        )
        resolved_key = api_key or os.getenv("KUMIHO_LLM_API_KEY")

        if provider == "anthropic" and not resolved_base_url:
            key = resolved_key or os.getenv("ANTHROPIC_API_KEY")
            return AnthropicAdapter.create(api_key=key)

        # Default to OpenAI-compatible (covers openai, gemini, ollama, etc.)
        key = resolved_key or os.getenv("OPENAI_API_KEY")
        return OpenAICompatAdapter.create(api_key=key, base_url=resolved_base_url)

    @staticmethod
    def _provider_from_env_keys() -> str:
        """Detect provider from which provider-specific env var is set."""
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return ""

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a memory extraction AI. Analyze conversations and extract structured knowledge.\n"
            "Output JSON in this exact schema:\n"
            "{\n"
            '  "type": "summary | fact | decision | action | reflection | error",\n'
            '  "title": "One-line summary (max 12 words)",\n'
            '  "summary": "Comprehensive summary (5-10 sentences) preserving ALL concrete details: dates, times, names, places, numbers, amounts, brands, model names, titles, roles, preferences, and outcomes",\n'
            '  "events": [\n'
            "    {\n"
            '      "event": "Specific incident or action that happened",\n'
            '      "when": "Date or time when it occurred (e.g. \'7 May 2023\', \'June 2023\', \'2022\', \'last week\') — use exact dates from the conversation when available",\n'
            '      "participants": ["person1"],\n'
            '      "consequence": "What changed as a result (behavioral change, decision, outcome)"\n'
            "    }\n"
            "  ],\n"
            '  "knowledge": {\n'
            '    "facts": [{"claim": "Specific factual claim with concrete detail", "certainty": "low | medium | high"}],\n'
            '    "decisions": [{"decision": "...", "reason": "..."}],\n'
            '    "actions": [{"task": "...", "status": "open | done | blocked"}],\n'
            '    "open_questions": ["..."]\n'
            "  },\n"
            '  "classification": {\n'
            '    "topics": ["topic1", "topic2"],\n'
            '    "entities": ["person1", "project1"]\n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- The summary should capture enough context that a future reader can answer specific factual questions without reading the full conversation\n"
            "- PRESERVE ALL DATES, TIMESTAMPS, AND TEMPORAL MARKERS mentioned in the conversation. Include them in the summary text and in each event's 'when' field. If a message is prefixed with a date like '[7 May 2023]', that is the date of the event.\n"
            "- Extract ALL specific events, incidents, and behavioral changes mentioned — even seemingly minor ones (accidents, purchases, lifestyle changes, health events, equipment failures)\n"
            "- Each event must be a concrete thing that happened, not a general topic or theme\n"
            "- Always include the consequence: what changed, what was decided, or what behavior resulted\n"
            "- Always include the 'when' field for each event — use exact dates from the conversation when available, otherwise use relative dates or 'unknown'\n"
            "- Extract ALL factual claims into knowledge.facts — include every piece of concrete information: names, places, brands, model numbers, job titles, relationships, hobbies, preferences, possessions, plans, amounts, measurements. If someone mentions owning a specific car, phone, or tool, that's a fact. If they mention where they work, live, or travel, that's a fact.\n"
            "- Extract decisions with their rationale into knowledge.decisions\n"
            "- Redact PII using placeholders like [EMAIL], [PHONE]\n"
            "- Focus on knowledge, not verbatim quotes"
        )

    @staticmethod
    def _format_messages(messages: List[Dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", ""))
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _user_prompt(conversation_text: str, *, context: Optional[str]) -> str:
        context_line = f"\nContext: {context}\n" if context else ""
        return (
            "Conversation to summarize:\n\n"
            f"{conversation_text}\n"
            f"{context_line}\n"
            "Extract structured knowledge in JSON format."
        )

    def _build_diagnostics(
        self,
        *,
        model: str,
        json_mode: JsonMode,
        raw_response: str = "",
    ) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": model,
            "base_url": self._base_url or "",
            "json_mode": self._describe_json_mode(json_mode),
            "raw_response_len": len(raw_response or ""),
            "raw_response_preview": self._preview_text(raw_response),
        }

    def _log_llm_failure(
        self,
        operation: str,
        *,
        error: Exception,
        diagnostics: Dict[str, Any],
    ) -> None:
        logger.warning(
            "%s failed: %s | provider=%s model=%s base_url=%s json_mode=%s raw_len=%s raw_preview=%r",
            operation,
            error,
            diagnostics.get("provider", ""),
            diagnostics.get("model", ""),
            diagnostics.get("base_url", ""),
            diagnostics.get("json_mode", ""),
            diagnostics.get("raw_response_len", 0),
            diagnostics.get("raw_response_preview", ""),
        )

    @staticmethod
    def _describe_json_mode(json_mode: JsonMode) -> str:
        if isinstance(json_mode, dict):
            schema = json_mode.get("schema", {})
            properties = schema.get("properties", {})
            property_names = ",".join(sorted(str(key) for key in properties.keys()))
            return f"schema:{json_mode.get('name', '')}:type={schema.get('type', '')}:props={property_names}"
        if json_mode == "array":
            return "array"
        if json_mode:
            return "object"
        return "none"

    @staticmethod
    def _preview_text(text: str, *, limit: int = 240) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    # ------------------------------------------------------------------
    # JSON parsing & fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        raise ValueError("No valid JSON found in summarizer response")

    @staticmethod
    def _normalize_summary(result: Dict[str, Any], *, fallback_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = dict(result)
        summary.setdefault("type", "summary")
        summary.setdefault("title", "Conversation summary")
        raw_summary = summary.get("summary")
        if isinstance(raw_summary, str) and raw_summary.strip():
            summary["summary"] = raw_summary.strip()
        else:
            summary["summary"] = MemorySummarizer._fallback_summary_text(fallback_messages)
            summary.setdefault(
                "error",
                "Summarizer response did not include a non-empty summary",
            )
        summary.setdefault("knowledge", {})
        summary.setdefault("classification", {})
        summary.setdefault("events", [])
        summary.setdefault("implications", [])
        summary["knowledge"].setdefault("facts", [])
        summary["knowledge"].setdefault("decisions", [])
        summary["knowledge"].setdefault("actions", [])
        summary["knowledge"].setdefault("open_questions", [])
        summary["classification"].setdefault("topics", [])
        summary["classification"].setdefault("entities", [])
        return summary

    @staticmethod
    def _fallback_summary(
        messages: List[Dict[str, Any]],
        *,
        error: str,
        debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = MemorySummarizer._fallback_summary_text(messages)
        result = {
            "type": "summary",
            "title": "Conversation summary",
            "summary": text,
            "events": [],
            "implications": [],
            "knowledge": {"facts": [], "decisions": [], "actions": [], "open_questions": []},
            "classification": {"topics": [], "entities": []},
            "error": error,
        }
        if debug:
            result["debug"] = debug
        return result

    @staticmethod
    def _fallback_summary_text(messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return "No conversation content available."

        def _strip_injected_blocks(text: str) -> str:
            cleaned = text
            for tag in ("kumiho_memory", "kumiho_project", "kumiho_instructions"):
                cleaned = re.sub(
                    rf"<{tag}>.*?</{tag}>",
                    "",
                    cleaned,
                    flags=re.DOTALL | re.IGNORECASE,
                )
            return cleaned.strip()

        preferred_roles = {"user", "assistant"}
        for msg in reversed(messages):
            role = str(msg.get("role", "")).strip().lower()
            if role and role not in preferred_roles:
                continue
            snippet = _strip_injected_blocks(str(msg.get("content") or ""))
            if len(snippet) > 500:
                snippet = f"{snippet[:497]}..."
            if snippet:
                return snippet

        return "Conversation summary unavailable."
