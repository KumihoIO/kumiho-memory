"""Experience snapshots and pattern proposals stay unpublished on every kumiho SDK.

Until kumiho 0.13.1, ``tool_memory_store`` tagged a new revision
``tags or ["published"]``, so a caller's tags replaced ``published``. Both
callers here pass tags without ``published`` and relied on that. kumiho 0.13.2
(KumihoIO/kumiho-SDKs#170) publishes every stored revision and adds a
``publish`` keyword to opt out, which older SDKs reject with ``TypeError``.
The callers pass ``publish=False`` only to a store that accepts it.
"""
import asyncio
import functools
import gc
import inspect
import json
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kumiho_memory import _store_compat
from kumiho_memory._store_compat import store_accepts_publish, unpublished_store_payload
from kumiho_memory.experience import normalize_experience, record_experience, record_outcome
from kumiho_memory.insight_patterns import prepare_pattern_request, store_pattern_candidate

EXPERIENCE_REF = "kref://project/experiences/pilot.experience?r=1"
PATTERN_REF = "kref://project/patterns/pilot.pattern_candidate?r=1"


# ---------------------------------------------------------------------------
# Store shapes
# ---------------------------------------------------------------------------
#
# The two SDK shapes carry the keyword parameters ``tool_memory_store`` really
# has, and no ``**kwargs``: an unknown keyword raises TypeError, as on the SDK.


def _recorded(calls, kwargs, revision_kref):
    calls.append(kwargs)
    return {"revision_kref": revision_kref, "edges_created": kwargs.get("source_revision_krefs") or []}


def sdk_0131_store(calls, revision_kref):
    """``tool_memory_store`` as of kumiho 0.13.1: no ``publish`` keyword."""
    def tool_memory_store(project="CognitiveMemory", space_path="", space_hint="", policy_kref=None,
                          memory_item_kind="conversation", bundle_name="", memory_type="summary",
                          title="", summary="", user_text="", assistant_text="",
                          artifact_location="", artifact_name="chat_io", tags=None,
                          source_revision_krefs=None, metadata=None, edge_type="DERIVED_FROM",
                          stack_revisions=True):
        return _recorded(calls, {k: v for k, v in locals().items() if k not in ("calls", "revision_kref")},
                         revision_kref)
    return tool_memory_store


def sdk_0132_store(calls, revision_kref):
    """``tool_memory_store`` as of kumiho 0.13.2: ``publish`` appended last."""
    def tool_memory_store(project="CognitiveMemory", space_path="", space_hint="", policy_kref=None,
                          memory_item_kind="conversation", bundle_name="", memory_type="summary",
                          title="", summary="", user_text="", assistant_text="",
                          artifact_location="", artifact_name="chat_io", tags=None,
                          source_revision_krefs=None, metadata=None, edge_type="DERIVED_FROM",
                          stack_revisions=True, publish=True):
        return _recorded(calls, {k: v for k, v in locals().items() if k not in ("calls", "revision_kref")},
                         revision_kref)
    return tool_memory_store


def async_old_store(calls, revision_kref):
    async def store(*, project, space_path, memory_item_kind, memory_type, title, summary,
                    user_text, assistant_text="", metadata, source_revision_krefs, edge_type,
                    tags, stack_revisions):
        return _recorded(calls, dict(locals()), revision_kref)
    return store


def async_new_store(calls, revision_kref):
    async def store(*, project, space_path, memory_item_kind, memory_type, title, summary,
                    user_text, assistant_text="", metadata, source_revision_krefs, edge_type,
                    tags, stack_revisions, publish=True):
        return _recorded(calls, dict(locals()), revision_kref)
    return store


NEW_SHAPES = [sdk_0132_store, async_new_store]
OLD_SHAPES = [sdk_0131_store, async_old_store]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_the_sdk_shapes_are_told_apart():
    assert store_accepts_publish(sdk_0132_store([], ""))
    assert not store_accepts_publish(sdk_0131_store([], ""))
    assert store_accepts_publish(async_new_store([], ""))
    assert not store_accepts_publish(async_old_store([], ""))


def test_a_store_taking_kwargs_or_a_mock_receives_the_keyword():
    def fake(**payload):
        return payload

    assert store_accepts_publish(fake)
    assert store_accepts_publish(MagicMock())
    assert store_accepts_publish(AsyncMock())


def test_a_wrapper_reports_the_store_it_wraps():
    old = sdk_0131_store([], "")

    @functools.wraps(old)
    def wrapper(*args, **kwargs):
        return old(*args, **kwargs)

    assert not store_accepts_publish(wrapper)


def test_an_uninspectable_store_is_not_given_the_keyword(monkeypatch):
    def store(**payload):
        return payload

    def unreadable(_callable, *args, **kwargs):
        raise ValueError("no signature found")

    monkeypatch.setattr(_store_compat.inspect, "signature", unreadable)
    assert not store_accepts_publish(store)
    assert unpublished_store_payload(store, {"tags": ["x"]}) == {"tags": ["x"]}


def test_an_unhashable_callable_is_inspected_without_the_cache():
    class Store:
        __hash__ = None

        def __call__(self, *, tags=None, publish=True):
            return tags

    assert store_accepts_publish(Store())


def test_the_cache_does_not_keep_a_bound_store_alive():
    class Manager:
        def memory_store(self, *, tags=None, publish=True):
            return tags

    manager = Manager()
    assert store_accepts_publish(manager.memory_store)
    assert store_accepts_publish(manager.memory_store)  # cached per function
    alive = weakref.ref(manager)
    del manager
    gc.collect()
    assert alive() is None


def test_payload_is_not_mutated():
    payload = {"tags": ["experience"]}
    assert unpublished_store_payload(sdk_0132_store([], ""), payload) == {"tags": ["experience"], "publish": False}
    assert payload == {"tags": ["experience"]}
    assert unpublished_store_payload(sdk_0131_store([], ""), payload) is payload


# ---------------------------------------------------------------------------
# Experience snapshots and outcome observations
# ---------------------------------------------------------------------------


def experience():
    return {"experience_id": "pilot-run-a", "title": "Try a pilot", "situation": "Small team",
            "goal": "Release on time", "decision": "Run a limited pilot", "rationale": "Bound upkeep",
            "expected_outcome": "Know support cost", "decision_state": "proposed"}


def serve_experience(monkeypatch):
    """Let ``validate_source_refs`` read back the stored experience snapshot."""
    import kumiho
    snapshot = {**normalize_experience(experience()), "recorded_at": "2026-09-09T00:00:00+00:00"}
    metadata = {"experience_record": json.dumps(snapshot)}
    monkeypatch.setattr(kumiho, "get_revision", lambda ref: SimpleNamespace(
        kref=SimpleNamespace(uri=ref), metadata=metadata, tags=[], deprecated=False))
    monkeypatch.setattr(kumiho, "get_item", lambda ref: SimpleNamespace(
        kref=SimpleNamespace(uri=ref), metadata={}, deprecated=False))


@pytest.mark.parametrize("shape", NEW_SHAPES, ids=lambda s: s.__name__)
def test_experience_passes_publish_false_to_a_store_that_accepts_it(shape, monkeypatch):
    calls = []
    manager = SimpleNamespace(project="project", memory_store=shape(calls, EXPERIENCE_REF))
    asyncio.run(record_experience(manager, experience()))
    serve_experience(monkeypatch)
    asyncio.run(record_outcome(manager, EXPERIENCE_REF, {
        "observed_outcome": "Support took two days", "observed_at": "2026-09-10T00:00:00Z",
        "outcome_status": "mixed"}))

    assert [call["publish"] for call in calls] == [False, False]
    assert all("published" not in call["tags"] for call in calls)
    assert "proposal" in calls[0]["tags"]


@pytest.mark.parametrize("shape", OLD_SHAPES, ids=lambda s: s.__name__)
def test_experience_omits_publish_for_a_store_without_it(shape, monkeypatch):
    calls = []
    manager = SimpleNamespace(project="project", memory_store=shape(calls, EXPERIENCE_REF))
    result = asyncio.run(record_experience(manager, experience()))
    serve_experience(monkeypatch)
    asyncio.run(record_outcome(manager, EXPERIENCE_REF, {
        "observed_outcome": "Support took two days", "observed_at": "2026-09-10T00:00:00Z",
        "outcome_status": "mixed"}))

    assert result["revision_kref"] == EXPERIENCE_REF
    assert len(calls) == 2
    assert all("publish" not in call and "published" not in call["tags"] for call in calls)


def test_experience_passes_publish_false_to_an_injected_async_mock():
    manager = SimpleNamespace(project="project",
                              memory_store=AsyncMock(return_value={"revision_kref": EXPERIENCE_REF}))
    asyncio.run(record_experience(manager, experience()))
    payload = manager.memory_store.call_args.kwargs
    assert payload["publish"] is False
    assert "published" not in payload["tags"]


# ---------------------------------------------------------------------------
# Pattern proposals
# ---------------------------------------------------------------------------


def pattern_source(name):
    record = normalize_experience({
        "experience_id": f"event-{name}", "title": f"Pilot {name}", "situation": "Uncertain deployment",
        "goal": "Reliable release", "decision": "Run a small pilot", "rationale": "Observe cost first",
        "expected_outcome": "Lower rollout risk", "decision_state": "accepted", "origin": "user",
    })
    record["recorded_at"] = "2026-09-09T01:00:00+00:00"
    return {"kref": f"kref://project/experiences/{name}.experience?r=1",
            "metadata": {"experience_record": json.dumps(record)}, "tags": []}


def pattern_inputs(monkeypatch):
    import kumiho
    rows = {row["kref"]: row for row in (pattern_source("alpha"), pattern_source("beta"))}
    monkeypatch.setattr(kumiho, "get_revision", lambda ref: SimpleNamespace(
        kref=SimpleNamespace(uri=ref), metadata=rows[ref]["metadata"], tags=[], deprecated=False))
    monkeypatch.setattr(kumiho, "get_item", lambda ref: SimpleNamespace(
        kref=SimpleNamespace(uri=ref), metadata={}, deprecated=False))
    request = prepare_pattern_request(list(rows.values()), space_paths=["/project/experiences"])
    candidate = {"kind": "recurring_pattern", "title": "Pilot before commitment",
                 "hypothesis": "A small pilot may test operating assumptions.",
                 "applicability_conditions": ["Operating cost is uncertain"],
                 "counterexamples": [], "source_krefs": request["source_krefs"]}
    return request, candidate


@pytest.mark.parametrize("shape", NEW_SHAPES, ids=lambda s: s.__name__)
def test_pattern_passes_publish_false_to_a_store_that_accepts_it(shape, monkeypatch):
    request, candidate = pattern_inputs(monkeypatch)
    calls = []
    manager = SimpleNamespace(project="project", memory_store=shape(calls, PATTERN_REF))
    result = asyncio.run(store_pattern_candidate(manager, request, candidate, "/project/patterns"))

    assert result["status"] == "stored_proposal"
    (call,) = calls
    assert call["publish"] is False
    assert "published" not in call["tags"]


@pytest.mark.parametrize("shape", OLD_SHAPES, ids=lambda s: s.__name__)
def test_pattern_omits_publish_for_a_store_without_it(shape, monkeypatch):
    request, candidate = pattern_inputs(monkeypatch)
    calls = []
    manager = SimpleNamespace(project="project", memory_store=shape(calls, PATTERN_REF))
    result = asyncio.run(store_pattern_candidate(manager, request, candidate, "/project/patterns"))

    assert result["status"] == "stored_proposal"
    (call,) = calls
    assert "publish" not in call
    assert "published" not in call["tags"]


# ---------------------------------------------------------------------------
# The installed kumiho SDK's own tool_memory_store
# ---------------------------------------------------------------------------
#
# These run the real store function with only its graph I/O replaced, so the
# same assertions hold on whichever SDK is installed: 0.13.1 (tags replace
# "published") and 0.13.2 (publish=False withholds it).

_SDK_SEAMS = ("_ensure_configured", "_get_project_cached", "_ensure_space_path",
              "_get_or_create_item", "_write_memory_artifact", "_get_or_create_bundle")


class _Revision:
    def __init__(self, item, number):
        self.kref = SimpleNamespace(uri=f"{item.kref.uri}?r={number}")
        self.tags_applied = []

    def tag(self, tag):
        self.tags_applied.append(tag)

    def create_artifact(self, name, location):
        return SimpleNamespace(kref=SimpleNamespace(uri=f"{self.kref.uri}&a={name}"))

    def create_edge(self, target, edge_type):
        return SimpleNamespace(target_kref=SimpleNamespace(uri=target.kref.uri))


class _Item:
    def __init__(self, space, name, kind):
        self.item_name = name
        self.kref = SimpleNamespace(uri=f"kref:/{space}/{name}.{kind}")
        self.revisions = []

    def create_revision(self, metadata=None):
        self.revisions.append(_Revision(self, len(self.revisions) + 1))
        return self.revisions[-1]


@pytest.fixture
def sdk_store(monkeypatch):
    mcp_server = pytest.importorskip("kumiho.mcp_server")
    missing = [name for name in _SDK_SEAMS if not hasattr(mcp_server, name)]
    if missing:
        pytest.skip(f"installed kumiho.mcp_server lacks {missing}")
    items = []

    def get_or_create_item(project, space_path, item_name, kind):
        items.append(_Item(space_path, item_name, kind))
        return items[-1]

    monkeypatch.setattr(mcp_server, "_ensure_configured", lambda: True)
    monkeypatch.setattr(mcp_server, "_get_project_cached", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(mcp_server, "_ensure_space_path", lambda project, path: "/" + path.strip("/"))
    monkeypatch.setattr(mcp_server, "_get_or_create_item", get_or_create_item)
    monkeypatch.setattr(mcp_server, "_write_memory_artifact", lambda **kwargs: "")
    monkeypatch.setattr(mcp_server, "_get_or_create_bundle", MagicMock())
    return SimpleNamespace(store=mcp_server.tool_memory_store, items=items)


def _applied_tags(items):
    return [tag for item in items for revision in item.revisions for tag in revision.tags_applied]


def test_helper_matches_the_installed_sdk_signature(sdk_store):
    declared = "publish" in inspect.signature(sdk_store.store).parameters
    assert store_accepts_publish(sdk_store.store) is declared


def test_installed_sdk_leaves_an_experience_unpublished(sdk_store, monkeypatch):
    manager = SimpleNamespace(project="project", memory_store=sdk_store.store)
    result = asyncio.run(record_experience(manager, experience()))
    serve_experience(monkeypatch)
    asyncio.run(record_outcome(manager, result["revision_kref"], {
        "observed_outcome": "Support took two days", "observed_at": "2026-09-10T00:00:00Z",
        "outcome_status": "mixed"}))

    assert len(sdk_store.items) == 2
    assert _applied_tags(sdk_store.items) == [
        "experience", "evidence:unverified", "proposal",
        "experience", "evidence:unverified",
    ]


def test_installed_sdk_leaves_a_pattern_proposal_unpublished(sdk_store, monkeypatch):
    request, candidate = pattern_inputs(monkeypatch)
    manager = SimpleNamespace(project="project", memory_store=sdk_store.store)
    result = asyncio.run(store_pattern_candidate(manager, request, candidate, "/project/patterns"))

    assert result["status"] == "stored_proposal"
    assert result["graph_links_verified"] is True
    assert _applied_tags(sdk_store.items) == ["pattern-candidate", "inferred", "proposal", "evidence:unverified"]
