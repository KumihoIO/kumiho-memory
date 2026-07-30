import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


@pytest.fixture(autouse=True)
def _no_inherited_host_session(monkeypatch):
    """Strip host-session env vars from every test by default.

    resolve_session_id prefers KUMIHO_SESSION_ID / CLAUDE_CODE_SESSION_ID,
    and CLAUDE_CODE_SESSION_ID is genuinely present when the suite runs
    inside a Claude Code session — which is exactly how this suite is usually
    run. Without this, any test of the pointer/generated tiers silently
    exercises the env tier instead, and the suite passes or fails depending
    on who launched pytest. Tests that want the env tier set it explicitly
    via monkeypatch.
    """
    monkeypatch.delenv("KUMIHO_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
