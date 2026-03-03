import asyncio
import hashlib
import os
import tempfile

from kumiho_memory.memory_manager import UniversalMemoryManager, get_memory_space
from kumiho_memory.redis_memory import RedisMemoryBuffer
from kumiho_memory.retry import RetryQueue

from fakes import FakeRedis


class StubSummarizer:
    async def summarize_conversation(self, messages, context=None):
        return {
            "type": "summary",
            "title": "Stub summary",
            "summary": "User likes tea.",
            "classification": {"topics": ["tea"]},
        }

    async def generate_implications(self, messages, context=None):
        return []


class StubRedactor:
    def anonymize_summary(self, summary):
        return summary.replace("tea", "[topic]")

    def reject_credentials(self, text):
        pass


def test_memory_manager_consolidation_calls_store():
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://memory/item"}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=store_stub,
            consolidation_threshold=2,
            artifact_root=tmpdir,
        )

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-1",
                message="I like tea.",
                context="personal",
            )
            session_id = ingest["session_id"]
            await manager.add_assistant_response(
                session_id=session_id,
                response="Green tea is best.",
            )
            result = await manager.consolidate_session(session_id=session_id)
            assert result["success"] is True
            assert "[topic]" in result["summary"]
            assert stored.get("project") == manager.project

            # Verify artifact file was written
            artifact_path = stored.get("artifact_location", "")
            assert artifact_path, "artifact_location must be set"
            assert os.path.isfile(artifact_path)
            assert artifact_path.endswith(".md")

            # Verify space-based subdirectory structure:
            # StubSummarizer returns topics=["tea"], so space_hint="tea"
            # Expected path: {tmpdir}/CognitiveMemory/tea/{session}.md
            expected_dir = os.path.join(tmpdir, "CognitiveMemory", "tea")
            assert os.path.dirname(artifact_path) == expected_dir

            content = open(artifact_path, encoding="utf-8").read()
            assert "# Stub summary" in content
            assert "I like tea." in content
            assert "Green tea is best." in content
            assert session_id in content

            # Verify artifact_name
            assert stored.get("artifact_name") == "conversation"

        asyncio.run(run())


def test_handle_user_message_flags_consolidation():
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        consolidation_threshold=1,
    )

    async def run():
        context = await manager.handle_user_message(
            user_id="user-2",
            message="Remember this.",
            context="work",
        )
        assert context["should_consolidate"] is True

    asyncio.run(run())


# ---------------------------------------------------------------------------
# recall_memories / memory_retrieve tests
# ---------------------------------------------------------------------------


def test_recall_memories_with_revision_krefs():
    """When memory_retrieve returns a dict with revision_krefs, recall_memories
    should enrich them with scores and revision metadata."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def retrieve_stub(**kwargs):
        assert kwargs["project"] == "CognitiveMemory"
        assert kwargs["query"] == "tea preferences"
        assert kwargs["limit"] == 3
        return {
            "item_krefs": ["kref://memory/item/1"],
            "revision_krefs": [
                "kref://memory/item/1/rev/1",
                "kref://memory/item/1/rev/2",
            ],
            "scores": [0.95, 0.82],
        }

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
    )

    async def run():
        results = await manager.recall_memories("tea preferences", limit=3)
        assert len(results) == 2
        # Results now include scores and any metadata fetched from revisions.
        assert results[0]["kref"] == "kref://memory/item/1/rev/1"
        assert results[0]["score"] == 0.95
        assert results[1]["kref"] == "kref://memory/item/1/rev/2"
        assert results[1]["score"] == 0.82

    asyncio.run(run())


def test_recall_memories_with_list_result():
    """When memory_retrieve returns a plain list, recall_memories should pass
    it through unchanged."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    expected = [
        {"title": "Tea preferences", "summary": "User likes green tea."},
        {"title": "Coffee chat", "summary": "User dislikes instant coffee."},
    ]

    async def retrieve_stub(**kwargs):
        return list(expected)

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
    )

    async def run():
        results = await manager.recall_memories("beverages")
        assert results == expected

    asyncio.run(run())


def test_recall_memories_with_no_retriever():
    """When memory_retrieve is None, recall_memories should return an empty
    list without errors."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=None,
    )

    async def run():
        results = await manager.recall_memories("anything")
        assert results == []

    asyncio.run(run())


def test_recall_memories_with_sync_callable():
    """memory_retrieve can be a synchronous function — _maybe_await should
    handle it correctly."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    def sync_retrieve(**kwargs):
        return {
            "revision_krefs": ["kref://memory/sync/1"],
        }

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=sync_retrieve,
    )

    async def run():
        results = await manager.recall_memories("sync query")
        assert len(results) == 1
        assert results[0] == {"kref": "kref://memory/sync/1"}

    asyncio.run(run())


def test_recall_memories_with_unexpected_return():
    """When memory_retrieve returns something unexpected (e.g. a string),
    recall_memories should return an empty list."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def bad_retrieve(**kwargs):
        return "unexpected string"

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=bad_retrieve,
    )

    async def run():
        results = await manager.recall_memories("query")
        assert results == []

    asyncio.run(run())


def test_handle_user_message_includes_long_term_memory():
    """handle_user_message should include long_term_memory from recall_memories
    in its response."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def retrieve_stub(**kwargs):
        return {
            "revision_krefs": ["kref://memory/item/tea/rev/1"],
        }

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
        consolidation_threshold=100,
    )

    async def run():
        result = await manager.handle_user_message(
            user_id="user-3",
            message="What tea do I like?",
            context="personal",
        )
        assert "long_term_memory" in result
        assert len(result["long_term_memory"]) == 1
        assert result["long_term_memory"][0] == {
            "kref": "kref://memory/item/tea/rev/1"
        }
        # Working memory should also be present
        assert "working_memory" in result
        assert len(result["working_memory"]) >= 1
        assert result["should_consolidate"] is False

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Artifact attachment tests
# ---------------------------------------------------------------------------


def test_ingest_message_with_attachment():
    """Attachments should be copied to the artifact directory and returned
    as artifact pointers in the ingest result and message metadata."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake image file
        src_file = os.path.join(tmpdir, "screenshot.png")
        file_content = b"\x89PNG\r\n\x1a\nfake-image-data"
        with open(src_file, "wb") as f:
            f.write(file_content)

        expected_hash = hashlib.sha256(file_content).hexdigest()

        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=None,
            artifact_root=tmpdir,
        )

        async def run():
            result = await manager.ingest_message(
                user_id="user-1",
                message="Here is a screenshot",
                context="personal",
                attachments=[
                    {"path": src_file, "description": "Dashboard screenshot"},
                ],
            )
            assert result["success"] is True
            assert len(result["attachments"]) == 1

            pointer = result["attachments"][0]
            assert pointer["type"] == "attachment"
            assert pointer["original_name"] == "screenshot.png"
            assert pointer["storage"] == "local"
            assert pointer["hash"] == f"sha256:{expected_hash}"
            assert pointer["size_bytes"] == len(file_content)
            assert pointer["content_type"] == "image/png"
            assert pointer["description"] == "Dashboard screenshot"

            # Verify the file was actually copied
            # location is a file:// URI — extract the path portion
            loc = pointer["location"]
            assert loc.startswith("file:///")
            # The copied file should exist under attachments/personal/
            artifact_dir = os.path.join(tmpdir, "CognitiveMemory", "attachments", "personal")
            assert os.path.isdir(artifact_dir)
            copied_files = os.listdir(artifact_dir)
            assert len(copied_files) == 1
            assert copied_files[0].endswith("_screenshot.png")

        asyncio.run(run())


def test_ingest_message_without_attachment():
    """When no attachments are provided, the result should have an empty list
    and message metadata should not contain an attachments key."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def run():
        result = await manager.ingest_message(
            user_id="user-1",
            message="Just text, no files",
            context="personal",
        )
        assert result["success"] is True
        assert result["attachments"] == []

    asyncio.run(run())


def test_attachment_missing_file_raises():
    """Passing a non-existent file path should raise FileNotFoundError."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=None,
            artifact_root=tmpdir,
        )

        async def run():
            try:
                await manager.ingest_message(
                    user_id="user-1",
                    message="Broken attachment",
                    attachments=[{"path": "/nonexistent/file.jpg"}],
                )
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass

        asyncio.run(run())


def test_consolidation_carries_attachment_pointers():
    """Attachments from ingested messages should appear in the store payload
    metadata during consolidation."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://memory/item"}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source file
        src_file = os.path.join(tmpdir, "notes.pdf")
        with open(src_file, "wb") as f:
            f.write(b"%PDF-1.4 fake-pdf-content")

        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=store_stub,
            consolidation_threshold=2,
            artifact_root=tmpdir,
        )

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-1",
                message="See attached notes",
                context="work",
                attachments=[
                    {"path": src_file, "content_type": "application/pdf"},
                ],
            )
            session_id = ingest["session_id"]

            await manager.add_assistant_response(
                session_id=session_id,
                response="I see the notes.",
            )

            result = await manager.consolidate_session(session_id=session_id)
            assert result["success"] is True

            # Verify attachments carried through to store payload metadata
            meta = stored.get("metadata", {})
            assert "attachments" in meta
            assert len(meta["attachments"]) == 1
            assert meta["attachments"][0]["original_name"] == "notes.pdf"
            assert meta["attachments"][0]["content_type"] == "application/pdf"
            assert meta["attachments"][0]["hash"].startswith("sha256:")

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Tool execution memory tests
# ---------------------------------------------------------------------------


def test_store_tool_execution_success():
    """Successful tool execution should be stored with type 'action'."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://memory/exec/1"}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=store_stub,
            artifact_root=tmpdir,
        )

        async def run():
            result = await manager.store_tool_execution(
                task="git commit -m 'Fix auth bug'",
                status="done",
                exit_code=0,
                duration_ms=450,
                stdout="[main 7f3a9b] Fix auth bug\n 1 file changed",
                tools=["shell_exec"],
                topics=["git", "version-control"],
                space_hint="work/project-alpha",
            )
            assert result["success"] is True
            assert result["memory_type"] == "action"
            assert stored["memory_type"] == "action"
            assert stored["title"].startswith("Successfully executed")
            assert "action" in stored["tags"]
            assert "done" in stored["tags"]
            assert "published" in stored["tags"]
            assert stored["metadata"]["exit_code"] == "0"
            assert stored["metadata"]["tools"] == "shell_exec"

            # Verify artifact was written
            assert os.path.isfile(stored["artifact_location"])

        asyncio.run(run())


def test_store_tool_execution_failure():
    """Failed tool execution should be stored with type 'error'."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://memory/exec/2"}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=store_stub,
            artifact_root=tmpdir,
        )

        async def run():
            result = await manager.store_tool_execution(
                task="git push origin main",
                status="failed",
                exit_code=128,
                stderr="git@github.com: Permission denied (publickey).",
                tools=["shell_exec"],
                topics=["git", "ssh"],
                open_questions=["Which SSH key should be used?"],
            )
            assert result["success"] is True
            assert result["memory_type"] == "error"
            assert stored["memory_type"] == "error"
            assert stored["title"].startswith("Failed")
            assert "Permission denied" in stored["summary"]
            assert "error" in stored["tags"]
            assert "published" in stored["tags"]
            assert stored["metadata"]["exit_code"] == "128"

        asyncio.run(run())


def test_store_tool_execution_no_store():
    """When no memory_store is configured, should return error dict."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )
    # Force-clear the store in case _load_default_store found a real one
    manager.memory_store = None

    async def run():
        result = await manager.store_tool_execution(
            task="some command",
            status="done",
        )
        assert result["success"] is False

    asyncio.run(run())


def test_store_tool_execution_nonzero_exit_is_error():
    """A non-zero exit code should produce type 'error' even if status is 'done'."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=store_stub,
            artifact_root=tmpdir,
        )

        async def run():
            result = await manager.store_tool_execution(
                task="npm test",
                status="done",
                exit_code=1,
                stderr="3 tests failed",
            )
            assert result["memory_type"] == "error"
            assert stored["memory_type"] == "error"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# recall_memories with space_paths and memory_types tests
# ---------------------------------------------------------------------------


def test_recall_memories_with_space_paths():
    """space_paths should be forwarded to the retriever."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    received = {}

    async def retrieve_stub(**kwargs):
        received.update(kwargs)
        return {"revision_krefs": ["kref://memory/scoped/1"]}

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
    )

    async def run():
        results = await manager.recall_memories(
            "project requirements",
            space_paths=["CognitiveMemory/work/team-alpha"],
        )
        assert len(results) == 1
        assert received["space_paths"] == ["CognitiveMemory/work/team-alpha"]

    asyncio.run(run())


def test_recall_memories_with_memory_types():
    """memory_types should be forwarded to the retriever."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    received = {}

    async def retrieve_stub(**kwargs):
        received.update(kwargs)
        return {"revision_krefs": ["kref://memory/error/1"]}

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
    )

    async def run():
        results = await manager.recall_memories(
            "git push ssh",
            memory_types=["error"],
        )
        assert len(results) == 1
        assert received["memory_types"] == ["error"]

    asyncio.run(run())


def test_recall_memories_without_filters():
    """When no filters are provided, space_paths and memory_types should not
    be in the kwargs sent to the retriever."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    received = {}

    async def retrieve_stub(**kwargs):
        received.update(kwargs)
        return []

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
    )

    async def run():
        await manager.recall_memories("anything")
        assert "space_paths" not in received
        assert "memory_types" not in received

    asyncio.run(run())


# ---------------------------------------------------------------------------
# get_memory_space tests
# ---------------------------------------------------------------------------


def test_get_memory_space_personal():
    assert get_memory_space("personal_dm") == "CognitiveMemory/personal"


def test_get_memory_space_team():
    assert get_memory_space(
        "team_channel", team_slug="team-alpha"
    ) == "CognitiveMemory/work/team-alpha"


def test_get_memory_space_group():
    assert get_memory_space(
        "group_dm", group_id="abc123"
    ) == "CognitiveMemory/groups/abc123"


def test_get_memory_space_unknown_defaults_personal():
    assert get_memory_space("smoke_signal") == "CognitiveMemory/personal"


def test_get_memory_space_custom_project():
    assert get_memory_space(
        "team_channel", project="MyProject", team_slug="ops"
    ) == "MyProject/work/ops"


# ---------------------------------------------------------------------------
# Retry + queue integration tests
# ---------------------------------------------------------------------------


def test_consolidation_retries_on_transient_error():
    """consolidate_session should retry on transient errors before succeeding."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    call_count = 0
    stored = {}

    async def flaky_store(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("transient")
        stored.update(kwargs)
        return {"item_kref": "kref://retry/1"}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=flaky_store,
            consolidation_threshold=2,
            artifact_root=tmpdir,
            store_max_retries=3,
        )

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-retry",
                message="Test retry",
                context="personal",
            )
            await manager.add_assistant_response(
                session_id=ingest["session_id"],
                response="OK",
            )
            result = await manager.consolidate_session(session_id=ingest["session_id"])
            assert result["success"] is True
            assert stored.get("project") == "CognitiveMemory"
            assert call_count == 2  # failed once, succeeded on retry

        asyncio.run(run())


def test_consolidation_queues_on_persistent_failure():
    """When all retries fail and a queue is configured, the payload should be
    enqueued instead of raising."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def always_fail(**kwargs):
        raise ConnectionError("down")

    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(os.path.join(tmpdir, "queue"))

        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=always_fail,
            consolidation_threshold=2,
            artifact_root=tmpdir,
            store_max_retries=2,
            retry_queue=queue,
        )

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-queue",
                message="Queue me",
                context="personal",
            )
            await manager.add_assistant_response(
                session_id=ingest["session_id"],
                response="Noted",
            )
            result = await manager.consolidate_session(session_id=ingest["session_id"])
            # Should still succeed (locally) — payload queued
            assert result["success"] is True
            assert result["store_result"].get("queued") is True

            # Queue should have one pending item
            assert queue.count == 1
            entries = queue.drain()
            assert entries[0]["payload"]["project"] == "CognitiveMemory"

        asyncio.run(run())


def test_flush_retry_queue_replays_items():
    """flush_retry_queue should replay queued items through memory_store."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    replayed = []

    async def good_store(**kwargs):
        replayed.append(kwargs.get("title"))
        return {"ok": True}

    with tempfile.TemporaryDirectory() as tmpdir:
        queue = RetryQueue(os.path.join(tmpdir, "queue"))
        queue.enqueue({"project": "test", "title": "queued-1"})
        queue.enqueue({"project": "test", "title": "queued-2"})

        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=good_store,
            retry_queue=queue,
        )

        async def run():
            result = await manager.flush_retry_queue()
            assert result == {"succeeded": 2, "failed": 0}
            assert queue.count == 0
            assert replayed == ["queued-1", "queued-2"]

        asyncio.run(run())


def test_consolidation_raises_without_queue():
    """When retries fail and no queue is configured, the error should propagate."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    async def always_fail(**kwargs):
        raise ConnectionError("no queue fallback")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=StubSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=always_fail,
            consolidation_threshold=2,
            artifact_root=tmpdir,
            store_max_retries=1,
            retry_queue=None,
        )

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-noq",
                message="No queue",
                context="personal",
            )
            await manager.add_assistant_response(
                session_id=ingest["session_id"],
                response="Nope",
            )
            try:
                await manager.consolidate_session(session_id=ingest["session_id"])
                assert False, "Should have raised ConnectionError"
            except ConnectionError:
                pass

        asyncio.run(run())
