# -*- coding: utf-8 -*-
"""Concurrency and recovery contract for belief replacement (kumiho-memory#27).

Deterministic barriers and fault injection over the in-memory graph fakes from
``test_grounding_ripple`` / ``test_supersession`` — no sleeps, no server. These
assert the *added* guarantees on top of the 1.4.1 sequential-replay behaviour:

* two workers that both observe a missing edge and both write still converge to
  one demotion (idempotent replay);
* a bounded ``SUPERSEDES`` cycle (direct reverse and A->B->C->A) is rejected
  before any write, preserving an unresolved conflict;
* a reverse edge that appears only after our own write is detected and the
  demotion withheld (quarantine, not a silent winner);
* a cross-project reference is rejected before any write;
* a truncated grounding ripple leaves a durable, resumable pending marker, and
  a resume pass (and the maintenance sweep) drains it after a process death;
* the result object distinguishes foreground completion from pending work.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from kumiho_memory.grounding import (
    GROUNDING_RIPPLE_PENDING_META,
    GROUNDING_STALE_META,
    has_pending_ripple,
    resume_grounding_ripple,
    ripple_grounding_stale,
)
from kumiho_memory.supersession import (
    SUPERSEDE_CYCLE_MAX_DEPTH,
    SupersessionResult,
    supersede_revision,
)


# ---------------------------------------------------------------------------
# Graph fakes (mirror test_grounding_ripple, with a shared revision registry so
# the cycle walk's kumiho.get_revision resolves multi-hop chains)
# ---------------------------------------------------------------------------


class _Kref:
    def __init__(self, uri):
        self.uri = uri


class _Edge:
    def __init__(self, source, target, edge_type, metadata=None):
        self.source_kref = _Kref(source)
        self.target_kref = _Kref(target)
        self.edge_type = edge_type
        self.metadata = dict(metadata or {})


class _Rev:
    def __init__(self, uri, metadata=None, incoming=None, outgoing=None):
        self.kref = _Kref(uri)
        self.metadata = dict(metadata or {})
        self.tags = []
        self._incoming = list(incoming or [])
        self._outgoing = list(outgoing or [])
        self.fail_edge = False
        self.set_metadata_calls = 0

    def get_edges(self, edge_type_filter=None, direction=0):
        pool = {1: self._incoming, 0: self._outgoing}.get(direction, self._incoming + self._outgoing)
        return [e for e in pool if not edge_type_filter or e.edge_type == edge_type_filter]

    def create_edge(self, target, edge_type, metadata=None):
        if self.fail_edge:
            return False
        e = _Edge(self.kref.uri, target.kref.uri, edge_type, metadata)
        self._outgoing.append(e)
        target._incoming.append(e)
        return e

    def set_metadata(self, md):
        self.metadata.update(md)
        self.set_metadata_calls += 1
        return self

    def set_attribute(self, key, value):
        self.metadata[key] = value
        return True

    def tag(self, t):
        if t not in self.tags:
            self.tags.append(t)


def _install(monkeypatch, revs):
    reg = {r.kref.uri: r for r in revs}
    fake = types.ModuleType("kumiho")
    fake.get_revision = lambda uri: reg[uri]
    fake.INCOMING, fake.OUTGOING, fake.BOTH = 1, 0, 2
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return reg


# ---------------------------------------------------------------------------
# Idempotent convergence under contention
# ---------------------------------------------------------------------------


def test_two_workers_observe_missing_edge_then_both_write_converge(monkeypatch):
    """Both writers see no edge and create one; the graph converges to a single
    demotion, one live SUPERSEDES, and a consistent result on the replay."""
    new = _Rev("kref://p/facts/new.fact?r=1")
    old = _Rev("kref://p/facts/old.fact?r=1", {"status": "active"})
    _install(monkeypatch, [new, old])

    r1 = supersede_revision(new, old, {"basis": "agent"})
    r2 = supersede_revision(new, old, {"basis": "agent"})  # the racing replay

    assert r1.created and r1.demoted and r1.complete
    assert r2.linked and not r2.created and not r2.demoted and r2.complete
    assert old.metadata["status"] == "superseded"
    assert len(new.get_edges("SUPERSEDES", 0)) == 1
    assert r1.op_id == r2.op_id  # same operation identity across the two writers


def test_distinct_sources_replace_same_target_are_both_recorded(monkeypatch):
    """Two different beliefs replacing the same prior one both link and demote it
    once — convergent, not a conflict (a conflict is a *reverse* pair)."""
    a = _Rev("kref://p/facts/a.fact?r=1")
    b = _Rev("kref://p/facts/b.fact?r=1")
    old = _Rev("kref://p/facts/old.fact?r=1", {"status": "active"})
    _install(monkeypatch, [a, b, old])

    ra = supersede_revision(a, old)
    rb = supersede_revision(b, old)

    assert ra.demoted and rb.linked and not rb.reverse_conflict
    assert old.metadata["status"] == "superseded"
    assert {e.source_kref.uri for e in old.get_edges("SUPERSEDES", 1)} == {a.kref.uri, b.kref.uri}


# ---------------------------------------------------------------------------
# Conflict preservation: reverse and cyclic replacements
# ---------------------------------------------------------------------------


def test_direct_reverse_edge_is_rejected_before_any_write(monkeypatch):
    first = _Rev("kref://p/facts/first.fact?r=1", {"status": "active"})
    second = _Rev("kref://p/facts/second.fact?r=1", {"status": "active"})
    second.create_edge(first, "SUPERSEDES")  # second already supersedes first
    _install(monkeypatch, [first, second])

    result = supersede_revision(first, second)  # would make first<->second mutual

    assert result.cycle_rejected and result.error and not result.created
    assert not result.complete
    assert first.get_edges("SUPERSEDES", 0) == []
    assert second.metadata["status"] == "active"  # neither demoted


def test_three_node_cycle_is_rejected(monkeypatch):
    # a->b->c already exist; c->a would close the cycle a->b->c->a.
    a = _Rev("kref://p/facts/a.fact?r=1", {"status": "active"})
    b = _Rev("kref://p/facts/b.fact?r=1", {"status": "active"})
    c = _Rev("kref://p/facts/c.fact?r=1", {"status": "active"})
    a.create_edge(b, "SUPERSEDES")
    b.create_edge(c, "SUPERSEDES")
    _install(monkeypatch, [a, b, c])

    result = supersede_revision(c, a)  # c supersedes a -> closes the cycle

    assert result.cycle_rejected and not result.created
    assert c.get_edges("SUPERSEDES", 0) == []


def test_long_chain_within_depth_is_not_a_false_cycle(monkeypatch):
    # A straight chain a->b->c->d (no cycle): superseding d with a NEW node must
    # be allowed — the walk from the target d finds no path back to the source.
    nodes = [_Rev(f"kref://p/facts/n{i}.fact?r=1", {"status": "active"}) for i in range(4)]
    for x, y in zip(nodes, nodes[1:]):
        x.create_edge(y, "SUPERSEDES")
    src = _Rev("kref://p/facts/src.fact?r=1")
    _install(monkeypatch, nodes + [src])

    result = supersede_revision(src, nodes[-1])  # src supersedes the chain tail
    assert result.created and result.demoted and not result.cycle_rejected


def test_concurrent_reverse_edge_after_write_is_quarantined(monkeypatch):
    """A reverse edge that appears only AFTER our create is detected on the
    post-write re-check; the demotion is withheld rather than picking a winner."""
    src = _Rev("kref://p/facts/src.fact?r=1")
    tgt = _Rev("kref://p/facts/tgt.fact?r=1", {"status": "active"})
    _install(monkeypatch, [src, tgt])

    real_create = src.create_edge

    def _create_then_race(target, edge_type, metadata=None):
        edge = real_create(target, edge_type, metadata)
        # a concurrent worker creates the opposite edge right after ours lands
        tgt.create_edge(src, "SUPERSEDES")
        return edge

    monkeypatch.setattr(src, "create_edge", _create_then_race)
    result = supersede_revision(src, tgt)

    assert result.created and result.reverse_conflict and not result.demoted
    assert not result.complete
    assert tgt.metadata["status"] == "active"  # neither side silently demoted


# ---------------------------------------------------------------------------
# Scope integrity + read-outage fail-closed
# ---------------------------------------------------------------------------


def test_cross_project_reference_is_rejected(monkeypatch):
    src = _Rev("kref://ProjectA/facts/x.fact?r=1")
    tgt = _Rev("kref://ProjectB/facts/y.fact?r=1", {"status": "active"})
    _install(monkeypatch, [src, tgt])

    result = supersede_revision(src, tgt)

    assert result.error and "scope" in result.error and not result.created
    assert tgt.metadata["status"] == "active"


def test_cycle_walk_read_outage_fails_closed(monkeypatch):
    """A read failure mid cycle-walk must not create the edge on a guess."""
    src = _Rev("kref://p/facts/src.fact?r=1")
    tgt = _Rev("kref://p/facts/tgt.fact?r=1", {"status": "active"})
    tgt.create_edge(_Rev("kref://p/facts/mid.fact?r=1"), "SUPERSEDES")  # one hop out
    fake = types.ModuleType("kumiho")

    def _boom(uri):
        raise RuntimeError("graph backend unavailable")

    fake.get_revision = _boom
    fake.INCOMING, fake.OUTGOING, fake.BOTH = 1, 0, 2
    monkeypatch.setitem(sys.modules, "kumiho", fake)

    result = supersede_revision(src, tgt)
    assert result.error and not result.created
    assert src.get_edges("SUPERSEDES", 0) == []


# ---------------------------------------------------------------------------
# Resumable / discoverable truncated grounding invalidation
# ---------------------------------------------------------------------------


def _fact_with_n_dependents(n, reg):
    fact = _Rev("kref://p/facts/f.fact?r=1", {"status": "active"})
    deps = []
    for i in range(n):
        d = _Rev(f"kref://p/decisions/d{i}.decision?r=1")
        fact._incoming.append(_Edge(d.kref.uri, fact.kref.uri, "DEPENDS_ON"))
        deps.append(d)
        reg[d.kref.uri] = d
    reg[fact.kref.uri] = fact
    return fact, deps


def test_truncated_ripple_persists_pending_marker(monkeypatch):
    reg = {}
    fact, deps = _fact_with_n_dependents(25, reg)
    fake = types.ModuleType("kumiho")
    fake.get_revision = lambda uri: reg[uri]
    monkeypatch.setitem(sys.modules, "kumiho", fake)

    stamped = ripple_grounding_stale(fact, "kref://p/facts/new.fact?r=1", cap=20)

    assert stamped == 20
    assert has_pending_ripple(fact.metadata) == "kref://p/facts/new.fact?r=1"
    flagged = sum(1 for d in deps if d.metadata.get(GROUNDING_STALE_META) == "true")
    assert flagged == 20


def test_resume_drains_pending_ripple_then_clears_marker(monkeypatch):
    reg = {}
    fact, deps = _fact_with_n_dependents(25, reg)
    fake = types.ModuleType("kumiho")
    fake.get_revision = lambda uri: reg[uri]
    monkeypatch.setitem(sys.modules, "kumiho", fake)

    ripple_grounding_stale(fact, "kref://p/facts/new.fact?r=1", cap=20)  # 20/25, pending
    assert has_pending_ripple(fact.metadata)

    resumed = resume_grounding_ripple(fact, cap=20)  # the remaining 5

    assert resumed == 5
    assert all(d.metadata.get(GROUNDING_STALE_META) == "true" for d in deps)
    assert has_pending_ripple(fact.metadata) == ""  # marker cleared, backlog drained


def test_untruncated_ripple_leaves_no_pending_marker(monkeypatch):
    reg = {}
    fact, deps = _fact_with_n_dependents(3, reg)
    fake = types.ModuleType("kumiho")
    fake.get_revision = lambda uri: reg[uri]
    monkeypatch.setitem(sys.modules, "kumiho", fake)

    ripple_grounding_stale(fact, "kref://p/facts/new.fact?r=1", cap=20)
    assert has_pending_ripple(fact.metadata) == ""
    assert fact.set_metadata_calls == 2  # pending before processing, cleared after success


def test_resume_is_noop_without_a_pending_marker(monkeypatch):
    reg = {}
    fact, _ = _fact_with_n_dependents(1, reg)
    fake = types.ModuleType("kumiho")
    fake.get_revision = lambda uri: reg[uri]
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    assert resume_grounding_ripple(fact) == 0


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


def test_result_distinguishes_complete_from_pending():
    complete = SupersessionResult(created=True, linked=True, demoted=True)
    assert complete.complete
    pending = SupersessionResult(created=True, linked=True, demoted=True, ripple_pending=True)
    assert not pending.complete
    conflict = SupersessionResult(created=True, linked=True, reverse_conflict=True)
    assert not conflict.complete
    errored = SupersessionResult(linked=True, error="boom")
    assert not errored.complete
    d = pending.as_dict()
    assert d["ripple_pending"] is True and d["complete"] is False and "op_id" in d


def test_cycle_depth_constant_is_documented_and_bounded():
    assert 1 <= SUPERSEDE_CYCLE_MAX_DEPTH <= 32


@pytest.mark.parametrize("existing_edge", [False, True])
def test_reverse_recheck_outage_withholds_demotion_and_repairs_on_replay(monkeypatch, existing_edge):
    src = _Rev("kref://p/facts/new.fact?r=1")
    old = _Rev("kref://p/facts/old.fact?r=1", {"status": "active"})
    _install(monkeypatch, [src, old])
    if existing_edge:
        src.create_edge(old, "SUPERSEDES")
    original = old.get_edges
    reads = 0

    def outage(**kwargs):
        nonlocal reads
        reads += 1
        if existing_edge or reads == 2:
            raise RuntimeError("reverse read unavailable")
        return original(**kwargs)

    monkeypatch.setattr(old, "get_edges", outage)
    result = supersede_revision(src, old)
    assert result.linked and result.error and not result.demoted and not result.complete
    assert old.metadata["status"] == "active"
    monkeypatch.setattr(old, "get_edges", original)
    assert supersede_revision(src, old).complete
    assert len(src._outgoing) == 1


@pytest.mark.parametrize("failure", ["fetch", "missing", "metadata", "negative_ack"])
def test_failed_dependent_is_not_skipped_by_durable_cursor(monkeypatch, failure):
    reg = {}
    fact, deps = _fact_with_n_dependents(3, reg)
    _install(monkeypatch, [fact, *deps])
    original = deps[0].set_metadata
    def fetch(uri):
        if uri == deps[0].kref.uri:
            if failure == "fetch":
                raise RuntimeError("temporary read failure")
            if failure == "missing":
                return None
        return reg[uri]
    if failure == "metadata":
        monkeypatch.setattr(deps[0], "set_metadata", lambda md: (_ for _ in ()).throw(RuntimeError("write failed")))
    if failure == "negative_ack":
        monkeypatch.setattr(deps[0], "set_metadata", lambda md: False)
    assert ripple_grounding_stale(fact, "kref://p/facts/new.fact?r=1", cap=2, get_revision=fetch) == 1
    assert has_pending_ripple(fact.metadata)
    assert fact.metadata["grounding_ripple_cursor"] == "0"
    monkeypatch.setattr(deps[0], "set_metadata", original)
    for _ in range(2):
        resume_grounding_ripple(fact, cap=2, get_revision=reg.__getitem__)
    assert all(d.metadata.get(GROUNDING_STALE_META) == "true" for d in deps)
    assert not has_pending_ripple(fact.metadata)


@pytest.mark.parametrize("change", ["reorder", "insert", "remove"])
def test_changed_dependency_snapshot_cannot_skip_remaining_work(monkeypatch, change):
    reg = {}
    fact, deps = _fact_with_n_dependents(3, reg)
    _install(monkeypatch, [fact, *deps])
    ripple_grounding_stale(fact, "kref://p/facts/new.fact?r=1", cap=2, get_revision=reg.__getitem__)
    if change == "reorder":
        fact._incoming.reverse()
    elif change == "insert":
        added = _Rev("kref://p/decisions/a.decision?r=1")
        reg[added.kref.uri] = added
        deps.append(added)
        fact._incoming.insert(0, _Edge(added.kref.uri, fact.kref.uri, "DEPENDS_ON"))
    else:
        fact._incoming.pop(0)
    reads = []
    def fetch(uri):
        reads.append(uri)
        return reg[uri]
    for _ in range(3):
        before = len(reads)
        resume_grounding_ripple(fact, cap=2, get_revision=fetch)
        assert len(reads) - before <= 2
    assert all(d.metadata.get(GROUNDING_STALE_META) == "true" for d in deps)
    assert not has_pending_ripple(fact.metadata)


def test_crash_during_first_batch_leaves_durable_resume_marker(monkeypatch):
    reg = {}
    fact, deps = _fact_with_n_dependents(2, reg)
    _install(monkeypatch, [fact, *deps])
    def crash(uri):
        raise KeyboardInterrupt("process stopped")
    with pytest.raises(KeyboardInterrupt):
        ripple_grounding_stale(fact, "kref://p/facts/new.fact?r=1", get_revision=crash)
    assert has_pending_ripple(fact.metadata)
    assert resume_grounding_ripple(fact, get_revision=reg.__getitem__) == 2
    assert not has_pending_ripple(fact.metadata)


@pytest.mark.parametrize("stage", ["initial", "clear"])
def test_progress_write_failure_requires_supersession_replay(monkeypatch, stage):
    reg = {}
    old, deps = _fact_with_n_dependents(1, reg)
    new = _Rev("kref://p/facts/new.fact?r=1")
    _install(monkeypatch, [old, new, *deps])
    original = old.set_metadata
    def reject(md):
        if bool(md[GROUNDING_RIPPLE_PENDING_META]) == (stage == "initial"):
            return False
        return original(md)
    monkeypatch.setattr(old, "set_metadata", reject)
    result = supersede_revision(new, old)
    assert result.error and not result.complete
    if stage == "clear":
        assert has_pending_ripple(old.metadata)
    monkeypatch.setattr(old, "set_metadata", original)
    assert supersede_revision(new, old).complete
    assert deps[0].metadata[GROUNDING_STALE_META] == "true"
    assert len(new._outgoing) == 1


def test_grounding_edge_outage_remains_discoverable(monkeypatch):
    reg = {}
    old, deps = _fact_with_n_dependents(1, reg)
    new = _Rev("kref://p/facts/new.fact?r=1")
    _install(monkeypatch, [old, new, *deps])
    original = old.get_edges
    def outage(**kwargs):
        if kwargs.get("edge_type_filter") == "DEPENDS_ON":
            raise RuntimeError("grounding read unavailable")
        return original(**kwargs)
    monkeypatch.setattr(old, "get_edges", outage)
    result = supersede_revision(new, old)
    assert result.ripple_pending and not result.complete
    monkeypatch.setattr(old, "get_edges", original)
    assert resume_grounding_ripple(old) == 1


def test_progress_tracks_sdk_fresh_returned_revision(monkeypatch):
    reg = {}
    fact, deps = _fact_with_n_dependents(3, reg)
    _install(monkeypatch, [fact, *deps])
    persisted = dict(fact.metadata)
    def fresh_update(md):
        persisted.update(md)
        return _Rev(fact.kref.uri, persisted)
    monkeypatch.setattr(fact, "set_metadata", fresh_update)
    assert ripple_grounding_stale(fact, "kref://p/facts/new.fact?r=1", cap=2) == 2
    assert fact.metadata["grounding_ripple_cursor"] == "2"
    assert has_pending_ripple(fact.metadata) == has_pending_ripple(persisted)
    assert resume_grounding_ripple(fact, cap=2) == 1
    assert not has_pending_ripple(fact.metadata) and not has_pending_ripple(persisted)
