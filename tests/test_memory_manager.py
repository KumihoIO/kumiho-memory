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


class ErrorSummarizer:
    async def summarize_conversation(self, messages, context=None):
        return {
            "type": "summary",
            "title": "Conversation summary",
            "summary": "I installed 0.4.5 and restarted the setup too.",
            "events": [],
            "implications": [],
            "knowledge": {"facts": [], "decisions": [], "actions": [], "open_questions": []},
            "classification": {"topics": [], "entities": []},
            "error": (
                "The api_key client option must be set either by passing "
                "api_key to the client or by setting the OPENAI_API_KEY "
                "environment variable"
            ),
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
            # session metadata (user_id + context) resolves the space to
            # personal/user-1 — the topic hint is only a fallback.
            # Expected path: {tmpdir}/CognitiveMemory/personal/user-1/{session}.md
            expected_dir = os.path.join(tmpdir, "CognitiveMemory", "personal", "user-1")
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


def test_consolidation_returns_error_without_storing_raw_fallback_summary():
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    store_calls = 0

    async def store_stub(**kwargs):
        nonlocal store_calls
        store_calls += 1
        return {"item_kref": "kref://memory/item"}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UniversalMemoryManager(
            redis_buffer=buffer,
            summarizer=ErrorSummarizer(),
            pii_redactor=StubRedactor(),
            memory_store=store_stub,
            consolidation_threshold=2,
            artifact_root=tmpdir,
        )

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-error",
                message="I installed 0.4.5 and restarted the setup too.",
                context="personal",
            )
            session_id = ingest["session_id"]
            await manager.add_assistant_response(
                session_id=session_id,
                response="Understood.",
            )

            result = await manager.consolidate_session(session_id=session_id)

            assert result["success"] is False
            assert "Conversation summarization failed" in result["error"]
            assert "api_key client option must be set" in result["error"]
            assert store_calls == 0

            working = await buffer.get_messages(
                project=manager.project,
                session_id=session_id,
                limit=10,
            )
            assert working["message_count"] == 2

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


# ---------------------------------------------------------------------------
# Failure ledger / parking at the store seam (issue #118)
# ---------------------------------------------------------------------------


def _ledger_manager(memory_store, ledger, *, store_max_retries=2, retry_queue=None):
    buffer = RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test")
    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=memory_store,
        failure_ledger=ledger,
        retry_queue=retry_queue,
        store_max_retries=store_max_retries,
    )


_POISON_PAYLOAD = {
    "project": "CognitiveMemory",
    "memory_type": "fact",
    "title": "Poison title",
    "summary": "This content deterministically fails.",
}


def test_store_skips_parked_content():
    """A parked payload is not sent to memory_store; returns {'parked': True}."""
    from kumiho_memory.failure_ledger import FailureLedger
    from kumiho_memory.memory_manager import _payload_failure_key

    calls = []

    async def store(**kwargs):
        calls.append(kwargs)
        return {"revision_kref": "kref://x"}

    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2)
        key = _payload_failure_key(_POISON_PAYLOAD)
        ledger.record_failure(key, "deterministic")
        ledger.record_failure(key, "deterministic")  # now parked
        assert ledger.is_parked(key) is True

        manager = _ledger_manager(store, ledger)

        async def run():
            result = await manager._store_with_retry(**_POISON_PAYLOAD)
            assert result == {"parked": True}
            assert calls == []  # store never attempted

        asyncio.run(run())


def test_store_records_deterministic_failure_and_parks():
    """Two deterministic store failures park the content; the 3rd is skipped."""
    from kumiho_memory.failure_ledger import FailureLedger
    from kumiho_memory.memory_manager import _payload_failure_key

    attempts = []

    async def store(**kwargs):
        attempts.append(kwargs.get("title"))
        raise ValueError("schema validation failed")  # deterministic

    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2)
        queue = RetryQueue(os.path.join(tmp, "queue"))
        manager = _ledger_manager(store, ledger, retry_queue=queue)
        key = _payload_failure_key(_POISON_PAYLOAD)

        async def run():
            # 1st failure: recorded, queued, not yet parked.
            r1 = await manager._store_with_retry(**_POISON_PAYLOAD)
            assert r1.get("queued") is True
            assert ledger.is_parked(key) is False
            # 2nd failure: reaches threshold → parked.
            r2 = await manager._store_with_retry(**_POISON_PAYLOAD)
            assert r2.get("queued") is True
            assert ledger.is_parked(key) is True
            # 3rd call: skipped entirely (store not attempted again).
            before = len(attempts)
            r3 = await manager._store_with_retry(**_POISON_PAYLOAD)
            assert r3 == {"parked": True}
            assert len(attempts) == before  # deterministic ValueError → 1 call each

        asyncio.run(run())


def test_store_transient_failures_never_park():
    """Repeated transient (ConnectionError) failures never park the content."""
    from kumiho_memory.failure_ledger import FailureLedger
    from kumiho_memory.memory_manager import _payload_failure_key

    async def store(**kwargs):
        raise ConnectionError("network down")  # transient

    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2)
        queue = RetryQueue(os.path.join(tmp, "queue"))
        manager = _ledger_manager(store, ledger, retry_queue=queue, store_max_retries=1)
        key = _payload_failure_key(_POISON_PAYLOAD)

        async def run():
            for _ in range(4):
                await manager._store_with_retry(**_POISON_PAYLOAD)
            assert ledger.is_parked(key) is False
            entry = ledger.get(key)
            assert entry["last_error_class"] == "transient"

        asyncio.run(run())


def test_store_success_clears_ledger_history():
    """A successful store clears any prior failure history for the content."""
    from kumiho_memory.failure_ledger import FailureLedger
    from kumiho_memory.memory_manager import _payload_failure_key

    outcome = {"fail": True}

    async def store(**kwargs):
        if outcome["fail"]:
            raise ValueError("deterministic once")
        return {"revision_kref": "kref://ok"}

    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=3)
        queue = RetryQueue(os.path.join(tmp, "queue"))
        manager = _ledger_manager(store, ledger, retry_queue=queue)
        key = _payload_failure_key(_POISON_PAYLOAD)

        async def run():
            await manager._store_with_retry(**_POISON_PAYLOAD)
            assert ledger.get(key)["attempts"] == 1
            outcome["fail"] = False
            result = await manager._store_with_retry(**_POISON_PAYLOAD)
            assert result == {"revision_kref": "kref://ok"}
            assert ledger.get(key) is None  # history cleared

        asyncio.run(run())


def test_store_without_ledger_is_unchanged():
    """No ledger configured → behavior identical to before (#118 additive)."""
    calls = []

    async def store(**kwargs):
        calls.append(kwargs)
        return {"revision_kref": "kref://x"}

    manager = _ledger_manager(store, None)

    async def run():
        result = await manager._store_with_retry(**_POISON_PAYLOAD)
        assert result == {"revision_kref": "kref://x"}
        assert len(calls) == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Evidence-level schema tests (issue #9)
# ---------------------------------------------------------------------------


def _make_evidence_manager(buffer, stored, tmpdir):
    async def store_stub(**kwargs):
        stored.update(kwargs)
        return {"item_kref": "kref://memory/item"}

    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=store_stub,
        consolidation_threshold=2,
        artifact_root=tmpdir,
    )


def test_consolidation_stamps_evidence_from_ingest():
    """evidence_level/source stashed at ingest survive to the store payload
    as metadata keys plus the mirrored evidence:<level> tag."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_evidence_manager(buffer, stored, tmpdir)

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-ev1",
                message="Acme announced record earnings.",
                context="personal",
                evidence_level="official",
                source="press-release:acme",
            )
            await manager.add_assistant_response(
                session_id=ingest["session_id"],
                response="Noted.",
            )
            result = await manager.consolidate_session(session_id=ingest["session_id"])
            assert result["success"] is True
            assert stored["metadata"]["evidence_level"] == "official"
            assert stored["metadata"]["source"] == "press-release:acme"
            assert "evidence:official" in stored["tags"]
            assert "summarized" in stored["tags"]
            assert "published" in stored["tags"]
            # The server freezes a revision as immutable once "published"
            # is applied — any tag applied after it is silently dropped.
            # The evidence tag MUST be ordered before "published".
            assert (
                stored["tags"].index("evidence:official")
                < stored["tags"].index("published")
            )

        asyncio.run(run())


def test_consolidation_explicit_evidence_overrides_session():
    """An explicit consolidate_session arg wins over ingest-time metadata."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_evidence_manager(buffer, stored, tmpdir)

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-ev2",
                message="Some rumor.",
                context="personal",
                evidence_level="unverified",
            )
            await manager.add_assistant_response(
                session_id=ingest["session_id"],
                response="Noted.",
            )
            await manager.consolidate_session(
                session_id=ingest["session_id"],
                evidence_level="corroborated",
                source="news:reuters",
            )
            assert stored["metadata"]["evidence_level"] == "corroborated"
            assert stored["metadata"]["source"] == "news:reuters"
            assert "evidence:corroborated" in stored["tags"]
            assert "evidence:unverified" not in stored["tags"]

        asyncio.run(run())


def test_consolidation_without_evidence_is_unchanged():
    """No evidence provided -> no evidence keys, tag set untouched."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_evidence_manager(buffer, stored, tmpdir)

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-ev3",
                message="I like tea.",
                context="personal",
            )
            await manager.add_assistant_response(
                session_id=ingest["session_id"],
                response="Green tea is best.",
            )
            await manager.consolidate_session(session_id=ingest["session_id"])
            assert stored["tags"] == ["summarized", "published"]
            assert "evidence_level" not in stored["metadata"]
            assert "source" not in stored["metadata"]

        asyncio.run(run())


def test_ingest_rejects_unknown_evidence_level():
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")

    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def run():
        try:
            await manager.ingest_message(
                user_id="user-ev4",
                message="Whatever.",
                evidence_level="rumor",
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    asyncio.run(run())


def test_fetch_revision_metadata_exposes_evidence(monkeypatch):
    """Round-trip read side: recall entries surface evidence_level/source
    metadata and the mirrored tag; ungraded revisions gain no new keys."""
    import sys
    import types

    class FakeRevision:
        def __init__(self, metadata, tags):
            self.metadata = metadata
            self.tags = tags
            self.created_at = "2026-07-02T00:00:00Z"

    revisions = {
        "kref://memory/item/rev/graded": FakeRevision(
            {"title": "Graded", "summary": "S", "evidence_level": "official",
             "source": "press-release:acme"},
            ["published", "evidence:official"],
        ),
        "kref://memory/item/rev/plain": FakeRevision(
            {"title": "Plain", "summary": "S"},
            ["published"],
        ),
    }

    fake_kumiho = types.ModuleType("kumiho")
    fake_kumiho.get_revision = lambda kref: revisions[kref]
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)

    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    async def run():
        graded = await manager._fetch_revision_metadata(
            "kref://memory/item/rev/graded", load_artifacts=False,
        )
        assert graded["evidence_level"] == "official"
        assert graded["source"] == "press-release:acme"
        assert "evidence:official" in graded["tags"]

        plain = await manager._fetch_revision_metadata(
            "kref://memory/item/rev/plain", load_artifacts=False,
        )
        assert "evidence_level" not in plain
        assert "source" not in plain

    asyncio.run(run())


def test_recall_memories_exposes_evidence_via_public_api(monkeypatch):
    """Full round-trip through the PUBLIC recall path: memory_retrieve
    returns krefs, recall_memories() results expose the evidence grade."""
    import sys
    import types

    class FakeRevision:
        def __init__(self, metadata, tags):
            self.metadata = metadata
            self.tags = tags
            self.created_at = "2026-07-02T00:00:00Z"

    revisions = {
        "kref://memory/item/rev/official": FakeRevision(
            {"title": "Official", "summary": "S", "evidence_level": "official",
             "source": "press-release:acme"},
            ["published", "evidence:official"],
        ),
    }

    fake_kumiho = types.ModuleType("kumiho")
    fake_kumiho.get_revision = lambda kref: revisions[kref]
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)

    async def retrieve_stub(**kwargs):
        return {
            "revision_krefs": ["kref://memory/item/rev/official"],
            "scores": [0.9],
        }

    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=retrieve_stub,
    )

    async def run():
        results = await manager.recall_memories("acme earnings")
        assert len(results) == 1
        assert results[0]["evidence_level"] == "official"
        assert results[0]["source"] == "press-release:acme"
        assert "evidence:official" in results[0]["tags"]

    asyncio.run(run())


def test_consolidation_empty_string_evidence_behaves_like_none():
    """evidence_level='' must not cancel the ingest-stashed grade —
    it behaves like None and the session fallback still applies."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_evidence_manager(buffer, stored, tmpdir)

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-ev5",
                message="Official statement.",
                context="personal",
                evidence_level="official",
                source="press-release:acme",
            )
            await manager.add_assistant_response(
                session_id=ingest["session_id"],
                response="Noted.",
            )
            await manager.consolidate_session(
                session_id=ingest["session_id"],
                evidence_level="",
                source="",
            )
            assert stored["metadata"]["evidence_level"] == "official"
            assert stored["metadata"]["source"] == "press-release:acme"
            assert "evidence:official" in stored["tags"]

        asyncio.run(run())


def test_evidence_on_later_message_preserves_session_identity():
    """Evidence provided on a non-first message must not wipe the
    user_id/context stashed at message 1 (merge, not replace)."""
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_evidence_manager(buffer, stored, tmpdir)

        async def run():
            ingest = await manager.ingest_message(
                user_id="user-ev6",
                message="First message, no evidence.",
                context="personal",
            )
            session_id = ingest["session_id"]
            await manager.ingest_message(
                user_id="user-ev6",
                message="Second message with evidence.",
                context="personal",
                session_id=session_id,
                evidence_level="single_source",
                source="news:reuters",
            )
            meta = await buffer.get_session_metadata(manager.project, session_id)
            assert meta["user_id"] == "user-ev6"
            assert meta["context"] == "personal"
            assert meta["evidence_level"] == "single_source"
            assert meta["source"] == "news:reuters"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Backend-error signal (issue #103, P1-1)
#
# recall_memories keeps its List return (the established contract — [] on
# failure), but records a backend-error summary on the manager so callers can
# distinguish "backend down" from "no memories". The signal is reset per call.
# ---------------------------------------------------------------------------


def _make_manager(memory_retrieve):
    fake = FakeRedis()
    buffer = RedisMemoryBuffer(client=fake, redis_url="redis://test")
    return UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        memory_retrieve=memory_retrieve,
    )


def test_recall_records_backend_error_signal_on_failure():
    """A retrieve backend error is recorded on the manager while recall still
    returns [] (internal contract unchanged)."""
    async def error_retrieve(**kwargs):
        return {"error": "graph backend unavailable: connection refused"}

    manager = _make_manager(error_retrieve)

    async def run():
        results = await manager.recall_memories("anything")
        assert results == []
        assert manager._last_backend_error is not None
        assert "graph backend unavailable" in manager._last_backend_error

    asyncio.run(run())


def test_recall_no_backend_error_signal_on_empty_healthy():
    """An empty-but-healthy retrieve leaves the signal cleared."""
    async def empty_retrieve(**kwargs):
        return {"revision_krefs": []}

    manager = _make_manager(empty_retrieve)

    async def run():
        results = await manager.recall_memories("nothing here")
        assert results == []
        assert manager._last_backend_error is None

    asyncio.run(run())


def test_recall_backend_error_signal_cleared_on_next_success():
    """A failure sets the signal; a subsequent healthy recall clears it (the
    reset at the top of recall_memories prevents a stale error leaking)."""
    state = {"fail": True}

    async def flaky_retrieve(**kwargs):
        if state["fail"]:
            return {"error": "boom"}
        return {"revision_krefs": ["kref://memory/ok/1"]}

    manager = _make_manager(flaky_retrieve)

    async def run():
        await manager.recall_memories("q1")
        assert manager._last_backend_error is not None

        state["fail"] = False
        results = await manager.recall_memories("q2")
        assert len(results) == 1
        assert manager._last_backend_error is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Session-id Redis flake: retry-then-fallback, never silent (issue #103, P1-2)
# ---------------------------------------------------------------------------


class _FlakySessionBuffer:
    """A redis buffer whose session ops always raise — exercises the
    retry-then-fallback path (all attempts fail -> WARNING + fallback)."""

    def __init__(self):
        self.calls = {
            "get_active_session": 0,
            "next_session_sequence": 0,
            "set_active_session": 0,
        }

    async def get_active_session(self, *, context, user_canonical_id):
        self.calls["get_active_session"] += 1
        raise ConnectionError("redis flake: get_active_session")

    async def next_session_sequence(self, *, user_canonical_id, date_str):
        self.calls["next_session_sequence"] += 1
        raise ConnectionError("redis flake: next_session_sequence")

    async def set_active_session(self, *, context, user_canonical_id, session_id):
        self.calls["set_active_session"] += 1
        raise ConnectionError("redis flake: set_active_session")

    async def close(self):
        return None


class _RecoverOnRetryBuffer:
    """get_active_session fails once, then succeeds on the retry — exercises
    the 'retry once before falling back' success path (no warning)."""

    def __init__(self):
        self.get_calls = 0

    async def get_active_session(self, *, context, user_canonical_id):
        self.get_calls += 1
        if self.get_calls == 1:
            raise ConnectionError("transient flake")
        return "personal:user-abc:20260101:007"

    async def close(self):
        return None


def test_generate_session_id_retries_then_warns_on_redis_flake(caplog):
    """A persistent Redis flake must NOT silently fork: each op retries once,
    then logs at WARNING and falls back to a well-formed session id."""
    import logging

    buffer = _FlakySessionBuffer()
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    with caplog.at_level(logging.WARNING, logger="kumiho_memory.memory_manager"):
        session_id = asyncio.run(
            manager._generate_session_id("user-canonical-x", "personal")
        )

    # No-raise guarantee preserved + fallback continuity id is well-formed.
    assert session_id.startswith("personal:user-")
    assert session_id.endswith(":001")  # next_session_sequence fell back to 1

    # Each flaky op was attempted twice (initial + one retry) before fallback.
    assert buffer.calls["get_active_session"] == 2
    assert buffer.calls["next_session_sequence"] == 2
    assert buffer.calls["set_active_session"] == 2

    # The fallback is logged loudly, not silent.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    messages = [r.getMessage() for r in warnings]
    assert any("_generate_session_id" in m for m in messages)
    assert any("get_active_session" in m for m in messages)


def test_generate_session_id_recovers_on_retry_without_warning(caplog):
    """When the retry succeeds, the active session is reused and NO warning is
    logged for that op — the retry-before-fallback path works."""
    import logging

    buffer = _RecoverOnRetryBuffer()
    manager = UniversalMemoryManager(
        redis_buffer=buffer,
        summarizer=StubSummarizer(),
        pii_redactor=StubRedactor(),
        memory_store=None,
    )

    with caplog.at_level(logging.WARNING, logger="kumiho_memory.memory_manager"):
        session_id = asyncio.run(
            manager._generate_session_id("user-x", "personal")
        )

    # Reused the active session returned by the successful retry.
    assert session_id == "personal:user-abc:20260101:007"
    assert buffer.get_calls == 2  # failed once, retried, succeeded

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any(
        "get_active_session" in r.getMessage() for r in warnings
    )
