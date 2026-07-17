"""Regression test for issue #104: background auto-assess must survive the MCP
dispatch pattern.

MCP tool handlers run each call under its own ``asyncio.run`` (see
``mcp_tools.tool_memory_add_response`` / ``tool_memory_reflect``), a one-shot
loop that cancels pending tasks on teardown.  The old code scheduled the
assessor with a detached ``asyncio.create_task``, so under MCP the task was
killed before it stored anything — evidence grading AND the CONTRADICTS bridge
(#94) silently never ran for MCP users, the primary deployment.

The fix runs ``_background_assess`` in a daemon thread that owns its own event
loop (``kumiho_memory._bounded.run_coro_in_daemon_thread``), which is
independent of the handler's loop and therefore survives its teardown.

The test drives BOTH dispatch shapes so the guarantee is demonstrated, not
assumed: the MCP shape (``asyncio.run`` per call) and a long-lived loop.  Both
must land the store AND the CONTRADICTS edge.  It is deterministic — it joins
the named daemon worker so the whole assess -> store -> bridge chain has
finished before asserting (the bridge's own bounded edge thread is awaited
inside ``_background_assess``, so joining the outer worker covers it).
"""

import asyncio
import sys
import threading
import types

from kumiho_memory.memory_manager import MemoryAssessResult, UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer

from fakes import FakeRedis


class _StubSummarizer:
    async def summarize_conversation(self, messages, context=None):
        return {}

    async def generate_implications(self, messages, context=None):
        return []


class _EdgeFakeRevision:
    def __init__(self, kref):
        self.kref = kref
        self.edges = []

    def get_edges(self, edge_type_filter=None, direction=0):
        return []

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((str(target.kref), edge_type))


def _make_manager(monkeypatch, stored, revisions):
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    # sys.modules seam for the CONTRADICTS bridge's ``import kumiho``.
    fake_kumiho = types.ModuleType("kumiho")

    def get_revision(kref):
        revisions.setdefault(kref, _EdgeFakeRevision(kref))
        return revisions[kref]

    fake_kumiho.get_revision = get_revision
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://m/new", "revision_kref": "kref://m/new/rev/1"}

    async def retrieve_stub(**kwargs):
        return [{"kref": "kref://m/1", "title": "R", "summary": "X",
                 "score": 0.4, "tags": []}]

    async def assess_fn(messages, recalled):
        # Real assess is async I/O + an LLM round-trip; the await here is what
        # made the detached create_task die before finishing under MCP.
        await asyncio.sleep(0.01)
        return MemoryAssessResult(
            should_store=True, content="Conflicting claim", memory_type="fact",
            evidence_level="unverified", conflicting_krefs=["kref://m/1"],
        )

    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=_StubSummarizer(),
        memory_store=store_stub,
        memory_retrieve=retrieve_stub,
        auto_assess_fn=assess_fn,
        auto_assess_min_messages=1,
    )


def _join_assess_workers(timeout=10.0):
    """Wait for every surviving background-assess daemon worker to finish."""
    for t in list(threading.enumerate()):
        if t.name.startswith("auto-assess"):
            t.join(timeout)


def _assert_assess_landed(stored, revisions):
    assert stored, "store must be called by the surviving background assess"
    new_rev = revisions.get("kref://m/new/rev/1")
    assert new_rev is not None, "new revision must have been fetched for bridging"
    assert ("kref://m/1", "CONTRADICTS") in new_rev.edges, (
        "the CONTRADICTS bridge (#94) must land even under MCP dispatch"
    )


def test_mcp_dispatch_background_assess_completes(monkeypatch):
    """MCP shape: asyncio.run per handler call, loop torn down on return."""
    stored, revisions = {}, {}
    manager = _make_manager(monkeypatch, stored, revisions)

    # Mirror mcp_tools: each tool call is its own asyncio.run.
    ingest = asyncio.run(manager.ingest_message(user_id="u", message="claim"))
    session_id = ingest["session_id"]
    asyncio.run(
        manager.add_assistant_response(session_id=session_id, response="resp")
    )
    # The add_response handler's loop is now closed. The daemon worker owns its
    # own loop and must still complete the full assess -> store -> bridge chain.
    _join_assess_workers()

    _assert_assess_landed(stored, revisions)


def _bridged(revisions):
    rev = revisions.get("kref://m/new/rev/1")
    return rev is not None and ("kref://m/1", "CONTRADICTS") in rev.edges


def test_long_lived_loop_background_assess_completes(monkeypatch):
    """Control: on a long-lived loop (one that stays alive after the turn) the
    assess completes.  This is the baseline the MCP case is contrasted against —
    it holds whether the assessor is scheduled via a detached create_task (old)
    or a daemon worker (fixed), because the loop is never torn down mid-flight.
    """
    stored, revisions = {}, {}
    manager = _make_manager(monkeypatch, stored, revisions)

    async def run():
        ingest = await manager.ingest_message(user_id="u", message="claim")
        await manager.add_assistant_response(
            session_id=ingest["session_id"], response="resp"
        )
        # Long-lived loop: keep it alive so the scheduled work can finish.
        for _ in range(500):
            if _bridged(revisions):
                break
            await asyncio.sleep(0.01)

    asyncio.run(run())
    _join_assess_workers()

    _assert_assess_landed(stored, revisions)
