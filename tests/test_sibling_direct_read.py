"""Eliminating GetItem must preserve the existing revision list and fallback."""

import asyncio
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import kumiho

from kumiho_memory.memory_manager import _get_sibling_revisions
from test_sibling_llm_cap import _manager


ITEM = "kref://p/nested/space/item.conversation"


@pytest.mark.parametrize("selector", ["", "?r=2", "?r=2&a=notes.txt"])
def test_direct_read_uses_same_revision_rpc_without_get_item(monkeypatch, selector):
    revisions = [object(), object()]
    client = SimpleNamespace(get_revisions=Mock(return_value=revisions))
    monkeypatch.setattr(kumiho, "get_client", lambda: client)
    legacy = Mock(side_effect=AssertionError("GetItem must not run"))
    monkeypatch.setattr(kumiho, "get_item", legacy)
    assert _get_sibling_revisions(ITEM + selector) is revisions
    client.get_revisions.assert_called_once_with(kumiho.Kref(ITEM))
    legacy.assert_not_called()


@pytest.mark.parametrize("missing", ["get_client", "Kref", "get_revisions"])
def test_missing_capability_uses_legacy_item_path(monkeypatch, missing):
    client = SimpleNamespace(get_revisions=Mock())
    monkeypatch.setattr(kumiho, "get_client", lambda: client)
    if missing == "get_revisions":
        del client.get_revisions
    else:
        monkeypatch.delattr(kumiho, missing)
    revisions = [object()]
    item = SimpleNamespace(get_revisions=Mock(return_value=revisions))
    legacy = Mock(return_value=item)
    monkeypatch.setattr(kumiho, "get_item", legacy)
    assert _get_sibling_revisions(ITEM) is revisions
    legacy.assert_called_once_with(ITEM)
    item.get_revisions.assert_called_once_with()


def test_direct_read_failure_does_not_replay_via_legacy(monkeypatch):
    client = SimpleNamespace(get_revisions=Mock(side_effect=PermissionError("denied")))
    monkeypatch.setattr(kumiho, "get_client", lambda: client)
    legacy = Mock(side_effect=AssertionError("must not retry permission errors"))
    monkeypatch.setattr(kumiho, "get_item", legacy)
    with pytest.raises(PermissionError):
        _get_sibling_revisions(ITEM)
    legacy.assert_not_called()


def test_invalid_selector_is_validated_before_removal(monkeypatch):
    client = SimpleNamespace(get_revisions=Mock())
    monkeypatch.setattr(kumiho, "get_client", lambda: client)
    with pytest.raises(ValueError):
        _get_sibling_revisions(ITEM + "?r=../../secret")
    client.get_revisions.assert_not_called()


def test_workers_keep_the_request_client_and_do_not_cache_between_calls(monkeypatch):
    tenant = ContextVar("sibling_tenant")
    versions = {"a": ["a1"], "b": ["b1"]}
    monkeypatch.setattr(kumiho, "get_client", lambda: SimpleNamespace(
        get_revisions=lambda kref: list(versions[tenant.get()]),
    ))

    async def read(name):
        tenant.set(name)
        return await asyncio.to_thread(_get_sibling_revisions, ITEM)

    async def run():
        assert await asyncio.gather(read("a"), read("b")) == [["a1"], ["b1"]]
        versions["a"] = ["a2"]
        assert await read("a") == ["a2"]

    asyncio.run(run())


def test_direct_and_legacy_reads_produce_identical_sibling_evidence(monkeypatch):
    revisions = [SimpleNamespace(
        kref=kumiho.Kref(ITEM + f"?r={n}"), created_at=f"2026-01-0{n}",
        metadata={"title": f"memory {n}", "summary": f"revision {n} history",
                  "facts": f"fact {n}", "evidence_level": "single_source"},
    ) for n in (3, 2, 1)]
    client = SimpleNamespace(get_revisions=Mock(return_value=revisions))
    monkeypatch.setattr(kumiho, "get_client", lambda: client)
    monkeypatch.setattr(kumiho, "get_item", lambda _: SimpleNamespace(
        get_revisions=lambda: revisions,
    ))
    manager = _manager()
    manager.sibling_similarity_threshold = 0
    direct = asyncio.run(manager._fetch_sibling_revision_summaries(
        ITEM, ITEM + "?r=3", load_artifacts=False,
    ))
    monkeypatch.delattr(kumiho, "get_client")
    legacy = asyncio.run(manager._fetch_sibling_revision_summaries(
        ITEM, ITEM + "?r=3", load_artifacts=False,
    ))
    assert direct == legacy
    assert len(direct) == 3
    assert {r["kref"] for r in direct} == {r.kref.uri for r in revisions}
