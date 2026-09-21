"""Scheduling changes must preserve evidence, context isolation and ordering."""
import asyncio
from contextvars import ContextVar
from types import SimpleNamespace
import threading
from unittest.mock import AsyncMock

import pytest

from kumiho_memory.memory_manager import _maybe_await, _metadata_scope, _coalesce_metadata
from kumiho_memory.graph_augmentation import GraphAugmentedRecall, GraphAugmentationConfig
from kumiho_memory.recall_timing import engage_timing, timed, _current


def test_sync_retrieval_overlaps_and_copies_request_context():
    tenant = ContextVar('test_tenant')
    barrier = threading.Barrier(2, timeout=3)
    def search(**kwargs):
        barrier.wait()
        return tenant.get(), kwargs['query']
    async def run():
        tenant.set('tenant-a')
        return await asyncio.gather(_maybe_await(search, query='a'), _maybe_await(search, query='b'))
    assert asyncio.run(run()) == [('tenant-a', 'a'), ('tenant-a', 'b')]


def test_async_and_awaitable_returning_search():
    async def value(**kwargs):
        return kwargs
    async def run():
        assert await _maybe_await(value, query='x') == {'query': 'x'}
        assert await _maybe_await(lambda **kw: value(**kw), query='y') == {'query': 'y'}
    asyncio.run(run())


def test_metadata_singleflight_copies_values_and_expires_between_recalls():
    class Reader:
        calls = 0
        @_coalesce_metadata
        async def fetch(self, kref, load_artifacts=True):
            self.calls += 1
            await asyncio.sleep(0)
            return {'summary': 'red', 'tags': ['current']}
        @_metadata_scope
        async def recall(self):
            a, b = await asyncio.gather(self.fetch('r1', False), self.fetch('r1', False))
            a['tags'].append('changed')
            assert b['tags'] == ['current']
            return b
    reader = Reader()
    asyncio.run(reader.recall())
    assert reader.calls == 1
    asyncio.run(reader.recall())
    assert reader.calls == 2


def test_metadata_does_not_cache_errors_or_artifacts():
    class Reader:
        calls = 0
        @_coalesce_metadata
        async def fetch(self, kref, load_artifacts=True):
            self.calls += 1
            return {} if self.calls == 1 else {'summary': 'latest'}
        @_metadata_scope
        async def recall(self):
            assert await self.fetch('r1', False) == {}
            assert await self.fetch('r1', False) == {'summary': 'latest'}
            await self.fetch('r1', True)
            await self.fetch('r1', True)
    reader = Reader()
    asyncio.run(reader.recall())
    assert reader.calls == 4


def test_concurrent_recalls_do_not_share_tenant_metadata():
    tenant = ContextVar('tenant')
    class Reader:
        @_coalesce_metadata
        async def fetch(self, kref, load_artifacts=True):
            await asyncio.sleep(0)
            return {'summary': tenant.get()}
        @_metadata_scope
        async def recall(self, name):
            tenant.set(name)
            return await self.fetch('same-kref', False)
    async def run():
        reader = Reader()
        return await asyncio.gather(reader.recall('a'), reader.recall('b'))
    assert asyncio.run(run()) == [{'summary': 'a'}, {'summary': 'b'}]


def test_original_search_overlaps_reformulation_and_merge_order_is_stable():
    async def run():
        primary_started = asyncio.Event()
        reformulation_done = asyncio.Event()
        async def recall(query, **kwargs):
            if query == 'original':
                primary_started.set()
                await reformulation_done.wait()
            return [{'kref': query, 'score': 1}]
        graph = GraphAugmentedRecall(adapter=SimpleNamespace(), model='test', recall_fn=recall,
            config=GraphAugmentationConfig(max_hops=0, entity_recall=False, fact_recall=False))
        async def reformulate(query):
            await asyncio.wait_for(primary_started.wait(), 2)
            reformulation_done.set()
            return ['alternate']
        graph._reformulate_query = reformulate
        graph._traverse_edges = AsyncMock(return_value=0)
        result = await graph.recall('original')
        assert [r['kref'] for r in result] == ['original', 'alternate']
    asyncio.run(run())


def test_failed_reformulation_does_not_orphan_primary():
    async def run():
        stopped = asyncio.Event()
        started = asyncio.Event()
        async def recall(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        graph = GraphAugmentedRecall(adapter=SimpleNamespace(), model='test', recall_fn=recall)
        async def reformulate(query):
            await started.wait()
            raise ValueError('failed')
        graph._reformulate_query = reformulate
        with pytest.raises(ValueError):
            await graph.recall('original')
        assert stopped.is_set()
    asyncio.run(run())


def test_timing_scope_is_reset_and_payload_accounts_for_diagnostics():
    @timed('evaluation')
    def evaluate():
        return None
    @engage_timing
    def engage():
        evaluate()
        return {'count': 0}
    result = engage()
    assert set(result['timing_ms']) == {'evaluation', 'total'}
    assert result['timing_ms']['total'] >= result['timing_ms']['evaluation'] >= 0
    assert result['approx_payload_tokens'] > 0
    assert _current.get() is None
    @engage_timing
    def failed():
        raise RuntimeError()
    with pytest.raises(RuntimeError):
        failed()
    assert _current.get() is None


def test_resolved_metadata_avoids_rpc_and_preserves_belief_markers(monkeypatch):
    import kumiho
    from kumiho_memory.memory_manager import UniversalMemoryManager
    monkeypatch.setattr(kumiho, 'get_revision', lambda *a: pytest.fail('duplicate RPC'))
    manager = object.__new__(UniversalMemoryManager)
    result = asyncio.run(manager._fetch_revision_metadata('kref://p/s/a.kind?r=1', False,
        resolved={'metadata': {'title':'old preference', 'summary':'black',
            'memory_type':'preference', 'status':'superseded',
            'superseded_by':'kref://p/s/a.kind?r=2'},
            'created_at':'2026-09-01T00:00:00Z', 'tags':['preference']}))
    assert result['summary'] == 'black'
    assert result['status'] == 'superseded'
    assert result['superseded'] is True
    assert result['tags'] == ['preference']
