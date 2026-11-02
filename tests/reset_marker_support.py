"""Shared fixtures and helpers for the session reset-marker tests.

Role: one definition of the setup the three ``test_session_reset_marker*`` modules
share -- a keyless manager over a fake Redis with a short buffer TTL, thin wrappers
around the MCP tool handlers, and a client wrapper that makes every Redis command
yield to the event loop so concurrent coroutines interleave command by command.
Contract: nothing here asserts; the modules import what they use (fixtures
included: pytest collects a fixture that is in the importing module's namespace).
"""

import asyncio

import pytest

import kumiho_memory.mcp_tools as mcp_tools_module

from test_memory_manager import AGENT_SUMMARY, _keyless_manager

#: A host conversation id as the Claude Code bootstrap hands it over: passed
#: explicitly on every call, so it never rotates (unlike the env-derived id).
SESSION = "2a59ba77-7be2-4eec-8f37-9fc2fb397771"
TTL = 600
#: The reset marker outlives a short buffer TTL (RESET_MARKER_MIN_TTL).
MARKER_TTL = 86400


@pytest.fixture(autouse=True)
def _short_ttl(monkeypatch):
    monkeypatch.setenv("KUMIHO_WORKING_MEMORY_TTL", str(TTL))
    monkeypatch.setattr("kumiho_memory.memory_manager._load_default_retrieve", lambda: None)
    monkeypatch.setattr("kumiho_memory.memory_manager._load_default_store", lambda: None)


@pytest.fixture
def env(tmp_path):
    stored = []
    manager, buffer = _keyless_manager(str(tmp_path), stored)
    return manager, buffer, buffer.client


def reflect(text, session_id=SESSION):
    return mcp_tools_module.tool_memory_reflect(
        {"response": text, "session_id": session_id},
    )


def consolidate(session_id=SESSION):
    return mcp_tools_module.tool_memory_consolidate(
        {"session_id": session_id, "summary": AGENT_SUMMARY},
    )


def count(buffer, manager, session_id=SESSION):
    got = asyncio.run(buffer.get_messages(
        project=manager.project, session_id=session_id, limit=50,
    ))
    return got["message_count"]


async def read_contents(buffer, manager, session_id=SESSION):
    got = await buffer.get_messages(
        project=manager.project, session_id=session_id, limit=50,
    )
    return [m["content"] for m in got["messages"]]


def contents(buffer, manager, session_id=SESSION):
    return asyncio.run(read_contents(buffer, manager, session_id))


def add(buffer, manager, text="hi", session_id=SESSION, role="user"):
    return buffer.add_message(
        project=manager.project, session_id=session_id, role=role, content=text,
    )


class YieldingClient:
    """Delegates to a fake Redis, yielding to the loop before every command.

    Concurrent coroutines then interleave command by command, like two clients on
    one Redis server.  ``on_command(n)`` (optional) is called synchronously as the
    n-th command starts -- a deterministic way to launch a second client at an
    exact point of another coroutine's command stream.
    """

    def __init__(self, inner, on_command=None):
        self._inner = inner
        self._on_command = on_command
        self.commands = 0

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        async def command(*args, **kwargs):
            await asyncio.sleep(0)
            self.commands += 1
            if self._on_command is not None:
                self._on_command(self.commands)
            return await attribute(*args, **kwargs)

        return command


def yielding(buffer, on_command=None):
    client = YieldingClient(buffer.client, on_command)
    buffer.client = client
    return client
