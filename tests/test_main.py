"""Tests for the kumiho-memory CLI preference loading helpers."""

import os

from kumiho_memory.__main__ import _configure_llm_from_prefs


def test_configure_llm_from_prefs_merges_shared_llm_and_section_model(monkeypatch):
    monkeypatch.delenv("KUMIHO_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KUMIHO_LLM_MODEL", raising=False)
    monkeypatch.delenv("KUMIHO_LLM_API_KEY", raising=False)
    monkeypatch.delenv("KUMIHO_LLM_BASE_URL", raising=False)

    _configure_llm_from_prefs(
        {
            "llm": {
                "provider": "gemini",
                "apiKey": "gemini-direct-key",
                "baseUrl": "https://generativelanguage.googleapis.com/v1beta/openai/",
            },
            "dreamState": {
                "model": {
                    "model": "gemini-2.5-flash-lite",
                },
            },
        },
        "dreamState",
    )

    assert os.environ["KUMIHO_LLM_PROVIDER"] == "gemini"
    assert os.environ["KUMIHO_LLM_MODEL"] == "gemini-2.5-flash-lite"
    assert os.environ["KUMIHO_LLM_API_KEY"] == "gemini-direct-key"
    assert os.environ["KUMIHO_LLM_BASE_URL"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
