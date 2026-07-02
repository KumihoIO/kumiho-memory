"""Tests for kumiho_memory.space_profiler — per-Space knowledge profiles."""

import asyncio
import json
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from kumiho_memory.space_profiler import (
    CANONICAL,
    CORRESPONDENCE,
    SPACE_CLASSES,
    WORKING,
    SpaceProfiler,
    SpaceSignals,
    classify,
    get_space_profile,
)


# ---------------------------------------------------------------------------
# Fakes — mirror the test_dream_state pattern
# ---------------------------------------------------------------------------


@dataclass
class FakeKref:
    uri: str


class FakeRevision:
    def __init__(self, kref, metadata=None, *, created_at="",
                 published=False, deprecated=False):
        self.kref = FakeKref(kref)
        self.metadata = metadata or {}
        self.created_at = created_at
        self.published = published
        self.deprecated = deprecated
        self._edges: list = []
        self.tagged: list = []
        self.edges_created: list = []

    def get_edges(self, edge_type_filter=None):
        return list(self._edges)

    def create_edge(self, target, edge_type, metadata=None):
        self.edges_created.append(
            (getattr(getattr(target, "kref", None), "uri", ""), edge_type)
        )


class FakeItem:
    def __init__(self, kref_uri, *, deprecated=False):
        self.kref = FakeKref(kref_uri)
        self.deprecated = deprecated
        self._revisions: List[FakeRevision] = []

    def get_revisions(self):
        return list(self._revisions)

    def get_revision_by_tag(self, tag):
        if self._revisions:
            return self._revisions[-1]
        return None

    def create_revision(self, metadata=None):
        rev = FakeRevision(
            f"{self.kref.uri}?r={len(self._revisions) + 1}",
            metadata=dict(metadata or {}),
        )
        self._revisions.append(rev)
        return rev


class FakeSpaceHandle:
    def __init__(self, items: Dict[str, FakeItem]):
        self._items = items

    def create_item(self, name, kind):
        # The profiler only creates its own profile item here.
        item = FakeItem(f"kref://created/{name}.{kind}")
        self._items[f"kref://created/{name}.{kind}"] = item
        self._items["__last_created__"] = item
        return item


class FakeSpace:
    def __init__(self, path):
        self.path = path


class FakeClient:
    def __init__(self, items_by_space):
        self._items_by_space = items_by_space

    def get_items(self, parent_path="", kind_filter="",
                  include_deprecated=False, page_size=None, cursor=None):
        if page_size is not None or cursor is not None:
            raise TypeError("legacy stub — no pagination")
        items = self._items_by_space.get(parent_path, [])
        if not include_deprecated:
            items = [i for i in items if not getattr(i, "deprecated", False)]
        return list(items)


def _build_fake_sdk(
    *,
    items_by_space: Dict[str, List[FakeItem]],
    spaces: List[FakeSpace],
    items_by_kref: Optional[Dict[str, FakeItem]] = None,
    attributes: Optional[Dict[str, Dict[str, str]]] = None,
    revisions_by_kref: Optional[Dict[str, FakeRevision]] = None,
):
    items_by_kref = items_by_kref if items_by_kref is not None else {}
    attributes = attributes if attributes is not None else {}
    revisions_by_kref = revisions_by_kref or {}
    client = FakeClient(items_by_space)

    sdk = types.ModuleType("kumiho")

    def get_project(name):
        proj = types.SimpleNamespace()
        proj.name = name
        proj.get_spaces = lambda recursive=False: list(spaces)
        proj.get_space = lambda rel: FakeSpaceHandle(items_by_kref)
        return proj

    sdk.get_project = get_project
    sdk.get_client = lambda: client
    sdk.get_item = lambda kref: items_by_kref.get(kref)
    sdk.get_attribute = lambda kref, key: attributes.get(kref, {}).get(key)
    sdk.get_revision = lambda kref: revisions_by_kref.get(kref)
    return sdk, items_by_kref


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _stable_item(kref, revision_count=1, age_days=90.0):
    """A published, old, single-revision item (canonical profile)."""
    item = FakeItem(kref)
    for i in range(revision_count):
        item._revisions.append(FakeRevision(
            f"{kref}?r={i + 1}",
            metadata={"evidence_level": "official"},
            created_at=_iso_days_ago(age_days),
            published=True,
        ))
    return item


def _churny_item(kref, revision_count=8):
    """A fast-stacking, unpublished, ungraded item (correspondence)."""
    item = FakeItem(kref)
    for i in range(revision_count):
        item._revisions.append(FakeRevision(
            f"{kref}?r={i + 1}",
            metadata={},
            created_at=_iso_days_ago(float(revision_count - i)),
            published=False,
        ))
    return item


# ---------------------------------------------------------------------------
# classify — pure threshold function
# ---------------------------------------------------------------------------


def test_classify_stable_published_space_is_canonical():
    signals = SpaceSignals(
        items_count=10, revisions_count=12,
        revisions_per_item_mean=1.2, revision_rate_per_day=0.1,
        supersedes_max_depth=0,
        evidence_histogram={"official": 10},
        published_share=0.9, median_revision_age_days=60.0,
    )
    scores, label, pinned = classify(signals)
    assert label == CANONICAL
    assert pinned is False
    assert scores["churn"] <= 0.4
    assert scores["stability"] >= 0.6


def test_classify_high_churn_unpublished_space_is_correspondence():
    signals = SpaceSignals(
        items_count=5, revisions_count=40,
        revisions_per_item_mean=8.0, revision_rate_per_day=5.0,
        supersedes_max_depth=6,
        evidence_histogram={},
        published_share=0.0, median_revision_age_days=2.0,
    )
    scores, label, pinned = classify(signals)
    assert label == CORRESPONDENCE
    assert scores["churn"] >= 0.6
    assert scores["stability"] <= 0.4


def test_classify_mixed_space_is_working():
    signals = SpaceSignals(
        items_count=5, revisions_count=15,
        revisions_per_item_mean=3.0, revision_rate_per_day=1.0,
        supersedes_max_depth=1,
        evidence_histogram={"single_source": 5},
        published_share=0.5, median_revision_age_days=10.0,
    )
    _, label, _ = classify(signals)
    assert label == WORKING


def test_classify_override_wins_and_pins():
    signals = SpaceSignals()  # would classify canonical-ish (all zeros -> working)
    scores, label, pinned = classify(signals, override="correspondence")
    assert label == CORRESPONDENCE
    assert pinned is True
    # invalid override ignored
    _, label, pinned = classify(signals, override="bogus")
    assert label in SPACE_CLASSES
    assert pinned is False


def test_classify_empty_space_is_working():
    _, label, _ = classify(SpaceSignals())
    assert label == WORKING


# ---------------------------------------------------------------------------
# collect_signals
# ---------------------------------------------------------------------------


def test_collect_signals_counts_and_histogram():
    space_path = "/CognitiveMemory/facts"
    items = [
        _stable_item("kref://CognitiveMemory/facts/a.conversation"),
        _churny_item("kref://CognitiveMemory/facts/b.conversation", 3),
    ]
    deprecated_item = FakeItem(
        "kref://CognitiveMemory/facts/dead.conversation", deprecated=True,
    )
    deprecated_item._revisions.append(FakeRevision(
        "kref://CognitiveMemory/facts/dead.conversation?r=1",
        created_at=_iso_days_ago(5), deprecated=True,
    ))
    items.append(deprecated_item)

    sdk, _ = _build_fake_sdk(
        items_by_space={space_path: items},
        spaces=[FakeSpace(space_path)],
    )
    profiler = SpaceProfiler(dry_run=True)
    signals = profiler.collect_signals(sdk, space_path)

    assert signals.items_count == 3
    assert signals.revisions_count == 5
    assert signals.deprecated_items == 1
    assert signals.deprecated_revisions == 1
    assert signals.evidence_histogram == {"official": 1}
    assert signals.published_share == 1 / 5
    assert signals.deprecation_ratio == 1 / 5
    assert signals.revisions_per_item_mean == 5 / 3
    assert signals.median_revision_age_days > 0


def test_collect_signals_excludes_profiler_and_cursor_items():
    """Self-measurement exclusion: profile items and the Dream State
    cursor never count toward a space's own signals."""
    space_path = "/CognitiveMemory/facts"
    profile_item = FakeItem(
        "kref://CognitiveMemory/facts/_space_profile.space-profile",
    )
    profile_item._revisions.append(FakeRevision(
        "kref://CognitiveMemory/facts/_space_profile.space-profile?r=1",
        created_at=_iso_days_ago(0.1),
    ))
    cursor_item = FakeItem(
        "kref://CognitiveMemory/facts/_dream_state.conversation",
    )
    cursor_item._revisions.append(FakeRevision(
        "kref://CognitiveMemory/facts/_dream_state.conversation?r=1",
        created_at=_iso_days_ago(0.1),
    ))

    sdk, _ = _build_fake_sdk(
        items_by_space={space_path: [profile_item, cursor_item]},
        spaces=[FakeSpace(space_path)],
    )
    profiler = SpaceProfiler(dry_run=True)
    signals = profiler.collect_signals(sdk, space_path)
    assert signals.items_count == 0
    assert signals.revisions_count == 0


def test_supersedes_chain_depth_bounded():
    space_path = "/CognitiveMemory/claims"
    item = FakeItem("kref://CognitiveMemory/claims/x.conversation")
    revs = {}
    prev_uri = None
    for i in range(4):
        uri = f"kref://CognitiveMemory/claims/x.conversation?r={i + 1}"
        rev = FakeRevision(uri, created_at=_iso_days_ago(4 - i))
        if prev_uri:
            edge = types.SimpleNamespace(
                edge_type="SUPERSEDES", target_kref=FakeKref(prev_uri),
            )
            rev._edges.append(edge)
        revs[uri] = rev
        prev_uri = uri
    item._revisions = [revs[f"kref://CognitiveMemory/claims/x.conversation?r={i + 1}"] for i in range(4)]

    sdk, _ = _build_fake_sdk(
        items_by_space={space_path: [item]},
        spaces=[FakeSpace(space_path)],
        revisions_by_kref=revs,
    )
    profiler = SpaceProfiler(dry_run=True, max_supersedes_depth=2)
    signals = profiler.collect_signals(sdk, space_path)
    assert signals.supersedes_edge_count == 1
    assert signals.supersedes_max_depth == 2  # bounded below true depth 3


# ---------------------------------------------------------------------------
# run — persistence, override, drift, dry_run
# ---------------------------------------------------------------------------


def _run_profiler(sdk, **kwargs):
    profiler = SpaceProfiler(**kwargs)
    sys.modules["kumiho"] = sdk
    try:
        return asyncio.run(profiler.run())
    finally:
        sys.modules.pop("kumiho", None)


def test_run_persists_profile_revision():
    space_path = "/CognitiveMemory/facts"
    items_by_kref: Dict[str, FakeItem] = {}
    sdk, created = _build_fake_sdk(
        items_by_space={
            space_path: [_stable_item("kref://CognitiveMemory/facts/a.conversation")],
        },
        spaces=[FakeSpace(space_path)],
        items_by_kref=items_by_kref,
    )

    result = _run_profiler(sdk)
    assert result["success"] is True
    assert result["spaces_profiled"] == 2  # project root + facts
    assert result["profiles"][space_path]["label"] == CANONICAL

    profile_item = created.get("__last_created__")
    assert profile_item is not None
    rev = profile_item._revisions[-1]
    assert rev.metadata["label"] == CANONICAL
    assert rev.metadata["type"] == "space_profile"
    json.loads(rev.metadata["scores"])  # valid JSON
    parsed_signals = json.loads(rev.metadata["signals"])
    assert parsed_signals["items_count"] == 1


def test_run_supersedes_edge_links_profile_drift():
    """A pre-existing profile revision gets a SUPERSEDES edge from the
    new one — profile drift is itself a SUPERSEDES chain."""
    space_path = "/CognitiveMemory/facts"
    profile_kref = "kref://CognitiveMemory/facts/_space_profile.space-profile"
    existing = FakeItem(profile_kref)
    existing._revisions.append(FakeRevision(
        f"{profile_kref}?r=1",
        metadata={"label": CORRESPONDENCE},
        created_at=_iso_days_ago(1),
    ))

    sdk, _ = _build_fake_sdk(
        items_by_space={
            space_path: [_stable_item("kref://CognitiveMemory/facts/a.conversation")],
        },
        spaces=[FakeSpace(space_path)],
        items_by_kref={profile_kref: existing},
    )

    result = _run_profiler(sdk)
    # canonical now, correspondence before -> drift reported
    drift = [d for d in result["drift"] if d["space_path"] == space_path]
    assert drift and drift[0]["from"] == CORRESPONDENCE
    assert drift[0]["to"] == CANONICAL
    new_rev = existing._revisions[-1]
    assert new_rev.metadata["previous_label"] == CORRESPONDENCE
    assert (f"{profile_kref}?r=1", "SUPERSEDES") in new_rev.edges_created


def test_run_override_pins_label():
    space_path = "/CognitiveMemory/facts"
    sdk, _ = _build_fake_sdk(
        items_by_space={
            space_path: [_stable_item("kref://CognitiveMemory/facts/a.conversation")],
        },
        spaces=[FakeSpace(space_path)],
        attributes={space_path: {"space_class": CORRESPONDENCE}},
    )

    result = _run_profiler(sdk)
    assert result["profiles"][space_path]["label"] == CORRESPONDENCE
    assert result["profiles"][space_path]["pinned"] is True


def test_run_dry_run_does_not_persist():
    space_path = "/CognitiveMemory/facts"
    items_by_kref: Dict[str, FakeItem] = {}
    sdk, created = _build_fake_sdk(
        items_by_space={
            space_path: [_churny_item("kref://CognitiveMemory/facts/b.conversation")],
        },
        spaces=[FakeSpace(space_path)],
        items_by_kref=items_by_kref,
    )

    result = _run_profiler(sdk, dry_run=True)
    assert result["dry_run"] is True
    assert result["spaces_profiled"] == 2
    assert "__last_created__" not in created  # nothing persisted


# ---------------------------------------------------------------------------
# get_space_profile — read-side API
# ---------------------------------------------------------------------------


def test_get_space_profile_round_trip(monkeypatch):
    profile_kref = "kref://CognitiveMemory/facts/_space_profile.space-profile"
    item = FakeItem(profile_kref)
    item._revisions.append(FakeRevision(
        f"{profile_kref}?r=1",
        metadata={
            "label": CANONICAL,
            "pinned": "false",
            "previous_label": WORKING,
            "scores": json.dumps({"churn": 0.1, "evidence": 0.8, "stability": 0.9}),
            "signals": json.dumps({"items_count": 7, "published_share": 0.9}),
        },
    ))

    fake_kumiho = types.ModuleType("kumiho")
    fake_kumiho.get_item = lambda kref: item if kref == profile_kref else None
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)

    profile = get_space_profile("CognitiveMemory", "/CognitiveMemory/facts")
    assert profile is not None
    assert profile.label == CANONICAL
    assert profile.previous_label == WORKING
    assert profile.scores["stability"] == 0.9
    assert profile.signals.items_count == 7


def test_get_space_profile_missing_returns_none(monkeypatch):
    fake_kumiho = types.ModuleType("kumiho")
    fake_kumiho.get_item = lambda kref: None
    monkeypatch.setitem(sys.modules, "kumiho", fake_kumiho)
    assert get_space_profile("CognitiveMemory", "/CognitiveMemory/nope") is None
