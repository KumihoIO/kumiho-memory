# -*- coding: utf-8 -*-
"""A reflect capture can name the memory it corrects (KumihoIO/kumiho-SDKs#172).

"No, my favourite colour is black" is a correction, and until now the store had
to *guess* whether it restated something already held: a similarity score over
the capture's space, calibrated at 0.58-0.68 for a restated same-subject
capture against a strong gate of 0.75. A miss files the correction as a second
memory and leaves the stale one exactly where recall reads it.

A capture that carries ``revises`` does not guess. The kref names the item, the
new text becomes that item's next revision in that item's own space, and the
``published`` tag — the one recall resolves first — moves with it when the
memory being corrected had one. Reflect then records the replacement in the
graph through the package's own belief-revision protocol: SUPERSEDES edge,
target demoted, grounding staleness rippled.

``item_kref`` only exists from kumiho 0.13.2, and kumiho-memory's floor is
``kumiho>=0.10.7``, so reflect asks the resolved store callable whether it
declares the parameter. These tests run against both shapes.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from kumiho_memory.mcp_tools import (
    MEMORY_TOOLS,
    _accepts_item_kref,
    _item_kref_for,
    tool_memory_ingest,
    tool_memory_reflect,
)

import test_mcp_tools as MT

from kumiho.mcp_server import tool_memory_store as _installed_store

#: Whether the kumiho actually installed here is new enough to be told what it
#: is revising. Assertions that depend on it are skipped, never inverted.
SDK_HAS_ITEM_KREF = _accepts_item_kref(_installed_store)

ITEM = "kref://CognitiveMemory/personal/favorite-color-3f9a.conversation"

CAPTURE = {
    "type": "preference",
    "title": "Favorite color is black",
    "content": "Correction: the user's favorite color is black, not blue.",
    "tags": ["preference", "color", "personal"],
    "revises": ITEM,
}


def _reflect(capture=CAPTURE, *, captures=None, space_path="personal"):
    """One reflect call through whatever store is currently patched in."""
    ingest = tool_memory_ingest({"user_id": "user-revises", "message": "hi"})
    return tool_memory_reflect({
        "session_id": ingest["session_id"],
        "response": "Noted — black it is.",
        "space_path": space_path,
        "captures": captures if captures is not None else [capture],
        "discover_edges": False,
    })


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def test_the_capture_schema_advertises_revises():
    reflect = next(t for t in MEMORY_TOOLS if t["name"] == "kumiho_memory_reflect")
    field = reflect["inputSchema"]["properties"]["captures"]["items"]["properties"]["revises"]
    assert field["type"] == "string"
    assert field["description"] == (
        "kref of the memory this capture corrects or updates; the capture "
        "becomes that memory's new revision."
    )
    # Optional: a correction is a special case, not the shape of every capture.
    assert reflect["inputSchema"]["properties"]["captures"]["items"]["required"] == [
        "type", "title", "content",
    ]


# ---------------------------------------------------------------------------
# The signature probe
# ---------------------------------------------------------------------------


def test_the_probe_reads_the_three_store_shapes():
    def new_sdk(project, *, item_kref=""):
        pass

    def old_sdk(project, *, stack_revisions=True):
        pass

    def forwarding_wrapper(project, **kwargs):
        pass

    assert _accepts_item_kref(new_sdk) is True
    assert _accepts_item_kref(old_sdk) is False
    # A **kwargs catch-all says nothing about the store behind it, and a store
    # that swallows the keyword without acting on it would file the correction
    # as a brand-new memory instead of revising the one the user named.
    assert _accepts_item_kref(forwarding_wrapper) is False
    assert _item_kref_for(new_sdk, ITEM) == {"item_kref": ITEM}
    assert _item_kref_for(old_sdk, ITEM) == {}
    assert _item_kref_for(forwarding_wrapper, ITEM) == {}


def test_a_capture_that_revises_nothing_never_sends_the_keyword():
    def new_sdk(project, *, item_kref=""):
        pass

    assert _item_kref_for(new_sdk, "") == {}


def test_an_unreadable_signature_is_treated_as_not_accepting():
    """The capture is stored either way; saying no costs only the old behaviour."""
    class _Opaque:
        __signature__ = "not a signature"

        def __call__(self, **kwargs):
            return {}

    assert _accepts_item_kref(_Opaque()) is False
    assert _item_kref_for(_Opaque(), ITEM) == {}


def test_a_wrapper_is_read_as_itself_not_as_what_it_wraps():
    """``follow_wrapped=False``: a ``functools.wraps`` wrapper over the new SDK
    advertises the wrapped signature, but only the wrapper is actually called."""
    import functools

    def new_sdk(project, *, item_kref=""):
        pass

    @functools.wraps(new_sdk)
    def narrowing_wrapper(project):
        pass

    assert _accepts_item_kref(narrowing_wrapper) is False


# ---------------------------------------------------------------------------
# Reflect passes the target — single capture and batch
# ---------------------------------------------------------------------------


def _new_sdk_store(calls):
    def store(*, item_kref="", **kwargs):
        calls.append({"item_kref": item_kref, **kwargs})
        return {
            "revision_kref": f"kref://m/cap/{len(calls)}",
            "item_kref": ITEM,
            "stacked": True,
            "previous_revision_kref": f"{ITEM}?r=1",
        }
    return store


def _new_sdk_batch(calls):
    def store_batch(captures, *, item_kref="", **kwargs):
        calls.append({"captures": captures, **kwargs})
        return {
            "results": [
                {"revision_kref": f"kref://m/batch/{i}",
                 "previous_revision_kref": f"{ITEM}?r=1"}
                for i in range(len(captures))
            ],
            "stored_krefs": [f"kref://m/batch/{i}" for i in range(len(captures))],
            "stacked": len(captures),
        }
    return store_batch


def test_reflect_single_capture_names_the_target():
    try:
        MT._install_test_manager()
        calls, superseded = [], []
        with patch("kumiho.mcp_server.tool_memory_store", _new_sdk_store(calls)), \
                patch("kumiho_memory.mcp_tools._record_supersession",
                      lambda new, prev: superseded.append((new, prev))):
            result = _reflect()
        assert result["captures_stored"] == 1
        assert calls[0]["item_kref"] == ITEM
        # Unchanged alongside it: the capture's own tags still go through.
        assert calls[0]["tags"] == CAPTURE["tags"]
        assert superseded == [("kref://m/cap/1", f"{ITEM}?r=1")]
    finally:
        MT._cleanup_manager()


def test_reflect_batch_names_the_target_on_that_capture_only():
    try:
        MT._install_test_manager()
        calls, superseded = [], []
        plain = {"type": "fact", "title": "Unrelated", "content": "Something else."}
        with patch("kumiho.mcp_server.tool_memory_store", _new_sdk_store([])), \
                patch("kumiho.mcp_server.tool_memory_store_batch", _new_sdk_batch(calls)), \
                patch("kumiho.batch_create_revisions", create=True), \
                patch("kumiho_memory.mcp_tools._record_supersession",
                      lambda new, prev: superseded.append((new, prev))):
            result = _reflect(captures=[CAPTURE, plain])
        assert result["captures_stored"] == 2
        assert len(calls) == 1
        sent = calls[0]["captures"]
        assert sent[0]["item_kref"] == ITEM
        # The key is absent, not empty: a capture that revises nothing must not
        # look to the SDK like one aimed at a kref it cannot resolve.
        assert "item_kref" not in sent[1]
        # Only the revising row supersedes anything.
        assert superseded == [("kref://m/batch/0", f"{ITEM}?r=1")]
    finally:
        MT._cleanup_manager()


def test_a_capture_without_revises_is_stored_exactly_as_before():
    try:
        MT._install_test_manager()
        calls, superseded = [], []
        with patch("kumiho.mcp_server.tool_memory_store", _new_sdk_store(calls)), \
                patch("kumiho_memory.mcp_tools._record_supersession",
                      lambda new, prev: superseded.append((new, prev))):
            result = _reflect({"type": "fact", "title": "T", "content": "C"})
        assert result["captures_stored"] == 1
        assert calls[0]["item_kref"] == ""  # the SDK default, never sent
        assert calls[0]["stack_revisions"] is True
        assert calls[0]["space_path"] == "personal"
        assert superseded == []
    finally:
        MT._cleanup_manager()


# ---------------------------------------------------------------------------
# An older SDK — the keyword is omitted, not forced
# ---------------------------------------------------------------------------


def _old_sdk_store(calls):
    """kumiho < 0.13.2: no ``item_kref``, and ``**kwargs`` would hide the point."""
    def store(*, project, space_path, memory_type, title, summary, assistant_text,
              source_revision_krefs, edge_type, tags, metadata, stack_revisions):
        calls.append({"tags": tags, "stack_revisions": stack_revisions})
        return {"revision_kref": f"kref://m/cap/{len(calls)}", "item_kref": ITEM}
    return store


def _old_sdk_batch(calls):
    def store_batch(captures, *, project, space_path, source_revision_krefs,
                    edge_type, stack_revisions, idempotency_prefix):
        calls.append({"captures": captures, "stack_revisions": stack_revisions})
        return {
            "results": [{"revision_kref": f"kref://m/batch/{i}"} for i in range(len(captures))],
            "stored_krefs": [f"kref://m/batch/{i}" for i in range(len(captures))],
            "stacked": 0,
        }
    return store_batch


def test_an_older_sdk_single_store_is_called_without_the_keyword():
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store", _old_sdk_store(calls)):
            result = _reflect()
        assert result["captures_stored"] == 1  # would be 0 on a TypeError
        assert calls[0]["tags"] == CAPTURE["tags"]
    finally:
        MT._cleanup_manager()


def test_an_older_sdk_batch_store_is_called_without_the_key():
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store", _old_sdk_store([])), \
                patch("kumiho.mcp_server.tool_memory_store_batch", _old_sdk_batch(calls)), \
                patch("kumiho.batch_create_revisions", create=True):
            result = _reflect(captures=[CAPTURE, dict(CAPTURE, title="Prefers dark mode")])
        assert result["captures_stored"] == 2
        assert len(calls) == 1
        assert all("item_kref" not in cap for cap in calls[0]["captures"])
    finally:
        MT._cleanup_manager()


def test_a_kwargs_forwarding_store_is_not_given_the_keyword():
    """The suite's own fakes take ``**kwargs``; they still must not be told."""
    try:
        MT._install_test_manager()
        calls = []
        with patch("kumiho.mcp_server.tool_memory_store", MT._fake_store_recorder(calls)):
            result = _reflect()
        assert result["captures_stored"] == 1
        assert "item_kref" not in calls[0]
    finally:
        MT._cleanup_manager()


# ---------------------------------------------------------------------------
# End to end against the installed SDK's real store
# ---------------------------------------------------------------------------


class _Kref:
    def __init__(self, uri):
        self.uri = uri


class _Revision:
    """A revision under the server's two tag rules.

    A tag lives on one revision per item, so tagging MOVES it; and a published
    revision is frozen and rejects tags applied to it afterwards.
    """

    def __init__(self, item, number):
        self.item = item
        self.number = number
        self.kref = _Kref(f"{item.kref.uri}?r={number}")
        self.tags = {"latest"}
        self.metadata = {}
        self.edges = []

    @property
    def published(self):
        return "published" in self.tags

    def tag(self, name):
        if self.published:
            raise RuntimeError("PERMISSION_DENIED: a published revision is immutable")
        for other in self.item.revisions:
            other.tags.discard(name)
        self.tags.add(name)

    def create_artifact(self, name, location):
        return MagicMock()

    def create_edge(self, target, edge_type, metadata=None):
        self.edges.append((edge_type, target.kref.uri, dict(metadata or {})))
        return MagicMock()

    def get_edges(self, edge_type_filter=None, direction=None):
        return [
            MagicMock(target_kref=_Kref(uri))
            for etype, uri, _md in self.edges
            if edge_type_filter in (None, etype)
        ]

    def set_attribute(self, name, value):
        self.metadata[name] = value
        return True


class _Item:
    item_name = "favorite-color-3f9a"

    def __init__(self):
        self.kref = _Kref(ITEM)
        self.revisions = []

    def create_revision(self, metadata=None):
        for prior in self.revisions:
            prior.tags.discard("latest")
        revision = _Revision(self, len(self.revisions) + 1)
        revision.metadata = dict(metadata or {})
        self.revisions.append(revision)
        return revision

    def get_revision_by_tag(self, tag):
        return next((r for r in self.revisions if tag in r.tags), None)


def _reflect_through_the_real_store(item, capture, *, stacks=False):
    """Reflect into the INSTALLED ``kumiho.mcp_server`` store, graph I/O stubbed.

    ``stacks`` drives the similarity gate for captures that do NOT name a
    target; a capture that carries ``revises`` never reaches the gate.
    """
    similar = (item, 0.81, 0.2, 0.3) if stacks else (None, 0.0, 0.0, 0.0)

    def _revision_by_uri(uri):
        return next((r for r in item.revisions if r.kref.uri == uri), None)

    with patch("kumiho.mcp_server._ensure_configured", return_value=True), \
            patch("kumiho.mcp_server._get_project_cached", return_value=MagicMock()), \
            patch("kumiho.mcp_server._ensure_space_path",
                  return_value="/CognitiveMemory/personal"), \
            patch("kumiho.mcp_server._find_similar_item", return_value=similar), \
            patch("kumiho.mcp_server._get_or_create_item", return_value=item), \
            patch("kumiho.mcp_server._write_memory_artifact", return_value=""), \
            patch("kumiho.mcp_server._get_or_create_bundle", return_value=MagicMock()), \
            patch("kumiho.get_item", return_value=item, create=True), \
            patch("kumiho.get_revision", side_effect=_revision_by_uri, create=True):
        return _reflect(capture)


def _seed(item, *, published=True):
    """One ordinary capture, so the item has a revision to correct."""
    _reflect_through_the_real_store(item, {
        "type": "preference", "title": "Favorite color is blue",
        "content": "The user's favorite color is blue.",
        **({} if published else {"tags": ["draft"]}),
    })
    return item.revisions[0]


@pytest.mark.skipif(not SDK_HAS_ITEM_KREF, reason="requires kumiho >= 0.13.2")
def test_a_correction_becomes_the_named_memorys_next_revision():
    try:
        MT._install_test_manager()
        item = _Item()
        blue = _seed(item)
        assert blue.published, "an untagged store publishes its new item's revision"

        result = _reflect_through_the_real_store(item, CAPTURE)
        assert len(item.revisions) == 2, "the correction revised, it did not duplicate"
        black = item.revisions[1]
        assert result["stored_krefs"] == [black.kref.uri] == [f"{ITEM}?r=2"]
        assert black.tags >= set(CAPTURE["tags"])
        # The tag recall resolves first moved with the correction.
        assert item.get_revision_by_tag("published") is black
        assert not blue.published
        # And placement came from the item, not from space_path="personal".
        assert black.metadata["space"] == "/CognitiveMemory/personal"
    finally:
        MT._cleanup_manager()


@pytest.mark.skipif(not SDK_HAS_ITEM_KREF, reason="requires kumiho >= 0.13.2")
def test_a_revision_kref_names_the_item_that_owns_it():
    """Callers hand back what engage showed them, selectors and all."""
    try:
        MT._install_test_manager()
        item = _Item()
        blue = _seed(item)
        _reflect_through_the_real_store(item, dict(CAPTURE, revises=blue.kref.uri))
        assert len(item.revisions) == 2
        assert item.get_revision_by_tag("published") is item.revisions[1]
    finally:
        MT._cleanup_manager()


@pytest.mark.skipif(not SDK_HAS_ITEM_KREF, reason="requires kumiho >= 0.13.2")
def test_correcting_an_unpublished_memory_publishes_nothing():
    """The tag is carried forward, never created."""
    try:
        MT._install_test_manager()
        item = _Item()
        _seed(item, published=False)
        assert item.get_revision_by_tag("published") is None

        _reflect_through_the_real_store(item, dict(CAPTURE, tags=["draft"]))
        assert len(item.revisions) == 2
        assert item.get_revision_by_tag("published") is None
    finally:
        MT._cleanup_manager()


@pytest.mark.skipif(not SDK_HAS_ITEM_KREF, reason="requires kumiho >= 0.13.2")
def test_the_replacement_is_recorded_in_the_graph():
    """The published tag is what recall reads; SUPERSEDES is what the graph knows."""
    try:
        MT._install_test_manager()
        item = _Item()
        blue = _seed(item)
        _reflect_through_the_real_store(item, CAPTURE)
        black = item.revisions[1]

        assert ("SUPERSEDES", blue.kref.uri, {"reason": "belief update",
                                              "basis": "agent"}) in black.edges
        assert blue.metadata["status"] == "superseded"
    finally:
        MT._cleanup_manager()


@pytest.mark.skipif(not SDK_HAS_ITEM_KREF, reason="requires kumiho >= 0.13.2")
def test_a_failed_supersession_still_reports_the_stored_capture():
    """Bookkeeping is best-effort — the memory is already written and tagged."""
    try:
        MT._install_test_manager()
        item = _Item()
        _seed(item)
        with patch("kumiho_memory.supersession.supersede_revision",
                   side_effect=RuntimeError("graph down")):
            result = _reflect_through_the_real_store(item, CAPTURE)
        assert result["captures_stored"] == 1
        assert item.get_revision_by_tag("published") is item.revisions[1]
    finally:
        MT._cleanup_manager()


def test_an_ordinary_capture_is_untouched_by_all_of_this():
    """No ``revises``: the 1.5.0 path, on either SDK — a new item, published."""
    try:
        MT._install_test_manager()
        item = _Item()
        _reflect_through_the_real_store(item, {
            "type": "preference", "title": "Favorite color is blue",
            "content": "The user's favorite color is blue.",
            "event_date": "2026-09-18",
        })
        assert len(item.revisions) == 1
        metadata = item.revisions[0].metadata
        assert item.revisions[0].published
        assert metadata["memory_type"] == "preference"
        assert metadata["event_date"] == "2026-09-18"
        assert json.loads(json.dumps(metadata))  # plain, serialisable strings
    finally:
        MT._cleanup_manager()


def test_an_old_sdk_accepts_the_field_and_stores_normally():
    """kumiho 0.13.1 knows nothing of ``item_kref``; reflect must not break.

    The capture is stored through the ordinary path — a TypeError, an error
    result or a dropped capture would all show up here.
    """
    if SDK_HAS_ITEM_KREF:
        pytest.skip("requires kumiho < 0.13.2")
    try:
        MT._install_test_manager()
        item = _Item()
        result = _reflect_through_the_real_store(item, CAPTURE)
        assert result["captures_stored"] == 1
        assert len(item.revisions) == 1
        assert item.revisions[0].tags >= set(CAPTURE["tags"])
    finally:
        MT._cleanup_manager()
