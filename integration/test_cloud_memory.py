"""Explicit opt-in Cloud contract tests; run separately from unit SDK fakes.

Only synthetic data in a freshly-created, owned project. No LLM provider key,
local login, CE fallback, production project reuse, or token output is allowed.
"""
import asyncio
import os
import time
import uuid
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

pytestmark = [pytest.mark.live, pytest.mark.cloud]


@pytest.fixture(scope="module")
def cloud(tmp_path_factory):
    if os.getenv("KUMIHO_RUN_CLOUD_TESTS") != "1":
        pytest.skip("Cloud tests require explicit KUMIHO_RUN_CLOUD_TESTS=1")
    token = os.getenv("KUMIHO_AUTH_TOKEN", "").strip()
    if not token:
        pytest.fail("KUMIHO_AUTH_TOKEN is required; live validation must not silently skip")

    root = tmp_path_factory.mktemp("cloud-auth")
    with pytest.MonkeyPatch.context() as env:
        env.setenv("KUMIHO_CONFIG_DIR", str(root))
        env.setenv("KUMIHO_MCP_HOSTED", "1")
        env.setenv("KUMIHO_HOSTED_LLM", "0")
        env.setenv("KUMIHO_HOSTED_LOCAL_REDIS", "0")
        env.setenv("KUMIHO_MEMORY_PROXY_URL", "https://control.kumiho.cloud/api/memory/redis")
        env.setenv("KUMIHO_CONTROL_PLANE_URL", "https://control.kumiho.cloud")
        import kumiho
        from kumiho.discovery import DiscoveryManager
        from kumiho.request_context import RequestContext, request_context

        cache = root / "discovery.json"
        record = DiscoveryManager(
            control_plane_url="https://control.kumiho.cloud", cache_path=cache, timeout=15,
        ).resolve(id_token=token, force_refresh=True)
        target = record.region.grpc_authority or record.region.server_url
        host = urlparse(target if "://" in target else "https://" + target).hostname
        assert host and host.endswith(".kumiho.cloud"), "Discovery did not return a Cloud endpoint"
        client = kumiho.client_from_discovery(
            id_token=token, control_plane_url="https://control.kumiho.cloud", cache_path=str(cache),
        )
        run_id = uuid.uuid4().hex
        name = "memory-ci-" + run_id
        ctx = RequestContext(
            tenant_id=record.tenant_id, user_id="memory-ci-" + run_id,
            auth_token=token, context="cloud-contract-test",
        )
        with kumiho.use_client(client), request_context(ctx):
            project = client.create_project(name, metadata={"memory_ci_owner": run_id})
            try:
                yield SimpleNamespace(project=project, client=client, run_id=run_id)
            finally:
                # Re-read and validate exact ownership before cleanup. Archive
                # only; never force-delete a protected/referenced project.
                owned = client.get_project(name)
                assert owned is not None and owned.project_id == project.project_id
                assert owned.metadata.get("memory_ci_owner") == run_id
                result = client.delete_project(project.project_id, force=False)
                assert result.success, "Synthetic project cleanup failed"
                archived = client.get_project(name, include_deprecated=True)
                assert archived is None or archived.deprecated


def test_working_memory_session_isolation_and_cleanup(cloud):
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    buffer = RedisMemoryBuffer(default_ttl=300, prefer_discovery=False)
    assert buffer.client is None, "Live test must use the authenticated Cloud proxy"
    project = cloud.project.name
    one, two = "one-" + cloud.run_id, "two-" + cloud.run_id

    async def check():
        try:
            first = await buffer.add_message(project=project, session_id=one, role="user", content="Synthetic amber otter")
            second = await buffer.add_message(project=project, session_id=two, role="user", content="Synthetic cobalt fox")
            assert first["created_bucket"] and second["created_bucket"]
            a = await buffer.get_messages(project=project, session_id=one)
            b = await buffer.get_messages(project=project, session_id=two)
            assert [m["content"] for m in a["messages"]] == ["Synthetic amber otter"]
            assert [m["content"] for m in b["messages"]] == ["Synthetic cobalt fox"]
        finally:
            try:
                await buffer.clear_session(project=project, session_id=one)
            finally:
                await buffer.clear_session(project=project, session_id=two)
                await buffer.close()
        assert (await buffer.get_messages(project=project, session_id=one))["message_count"] == 0
        assert (await buffer.get_messages(project=project, session_id=two))["message_count"] == 0

    asyncio.run(check())


def test_store_manager_recall_space_isolation_and_belief_replacement(cloud):
    import kumiho
    from kumiho.mcp_server import tool_memory_store
    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from kumiho_memory.summarization import MemorySummarizer
    from kumiho_memory.supersession import supersede_revision

    project = cloud.project
    stored = []
    for space, text in (("alpha", "Synthetic amber otter uses a copper compass"),
                        ("beta", "Synthetic amber otter uses a silver compass")):
        result = tool_memory_store(
            project=project.name, space_path=space, title=text, summary=text,
            user_text=text, memory_type="fact", stack_revisions=False,
        )
        assert not result.get("error"), result
        stored.append(result["revision_kref"])
    manager = UniversalMemoryManager(
        project=project.name, redis_buffer=RedisMemoryBuffer(prefer_discovery=False),
        summarizer=MemorySummarizer(adapter=_NoLLM()), entity_promotion=False,
    )
    deadline = time.monotonic() + 45
    while True:
        recalled = asyncio.run(manager.recall_memories(
            "Synthetic amber otter copper compass", space_paths=[project.name + "/alpha"], limit=5,
        ))
        assert not manager._last_backend_error, "Cloud recall reported a backend failure"
        refs = {entry["kref"] for entry in recalled}
        assert stored[1] not in refs, "Recall crossed the requested space boundary"
        if stored[0] in refs:
            break
        assert time.monotonic() < deadline, "Stored memory was not searchable within 45 seconds"
        time.sleep(2)

    # Real revision edges + status + dependent invalidation, verified by fresh
    # server reads, not the SDK's mutated in-memory revision object.
    old = project.create_item("prior", "fact").create_revision(metadata={"status": "active", "sentinel": "keep"})
    new = project.create_item("replacement", "fact").create_revision(metadata={"status": "active"})
    dep = project.create_item("grounded", "decision").create_revision(metadata={"status": "active"})
    dep.create_edge(old, "DEPENDS_ON")
    first = supersede_revision(new, old, {"basis": "cloud-contract-test"})
    assert first.created and first.demoted and first.stale == 1 and not first.error
    reread = kumiho.get_revision(old.kref.uri)
    assert reread.metadata["status"] == "superseded" and reread.metadata["sentinel"] == "keep"
    dependent = kumiho.get_revision(dep.kref.uri)
    assert dependent.metadata["grounding_stale"] == "true"
    assert dependent.metadata["grounding_stale_superseded_by"] == new.kref.uri
    second = supersede_revision(kumiho.get_revision(new.kref.uri), reread)
    assert second.linked and not second.created and not second.error
    edges = kumiho.get_revision(new.kref.uri).get_edges(edge_type_filter="SUPERSEDES", direction=0)
    assert sum(e.target_kref.uri == old.kref.uri for e in edges) == 1


class _NoLLM:
    async def chat(self, **kwargs):
        raise AssertionError("Cloud contract tests must not call an LLM provider")
