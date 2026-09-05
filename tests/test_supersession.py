"""Cross-writer replacement invariants, including partial-write recovery."""
import sys
from types import SimpleNamespace

import pytest

from kumiho_memory.supersession import supersede_revision
from kumiho_memory.code_capture import IngestStats, _supersede_pass
from kumiho_memory.code_decisions import CodeMemoryConfig
from kumiho_memory.graph_maintenance import GraphMaintainer, MaintenanceStats
from test_grounding_ripple import _Rev, _Edge


class Revision(_Rev):
    fail_status = False
    fail_edge = False

    def set_attribute(self, key, value):
        if self.fail_status:
            raise RuntimeError("injected status failure")
        self.metadata[key] = value
        return True

    def create_edge(self, target, edge_type, metadata=None):
        if self.fail_edge:
            raise RuntimeError("injected edge failure")
        super().create_edge(target, edge_type, metadata)


@pytest.fixture
def graph(monkeypatch):
    new = Revision("kref://test/new.fact?r=1")
    old = Revision("kref://test/old.fact?r=1", {"status": "active", "keep": "provenance"})
    dep = Revision("kref://test/dependent.decision?r=1")
    old._incoming.append(_Edge(dep.kref.uri, old.kref.uri, "DEPENDS_ON"))
    revisions = {r.kref.uri: r for r in (new, old, dep)}
    monkeypatch.setitem(sys.modules, "kumiho", SimpleNamespace(
        get_revision=lambda uri: revisions[uri], INCOMING=1,
    ))
    return new, old, dep


def test_revision_identity_metadata_ripple_and_replay(graph):
    new, old, dep = graph
    first = supersede_revision(new, old, {"basis": "agent"})
    assert first.created and first.demoted and first.stale == 1 and not first.error
    assert old.metadata == {"status": "superseded", "keep": "provenance"}
    assert old.kref.uri.endswith("old.fact?r=1")
    assert dep.metadata["grounding_stale_superseded_by"] == new.kref.uri
    again = supersede_revision(new, old)
    assert again.linked and not again.created and not again.demoted and again.stale == 0
    assert len(new._outgoing) == 1


def test_failed_edge_never_demotes_or_invalidates(graph):
    new, old, dep = graph
    new.fail_edge = True
    result = supersede_revision(new, old)
    assert result.error and not result.linked
    assert old.metadata["status"] == "active" and dep.metadata == {}


def test_replay_repairs_status_failure_without_duplicate_edge(graph):
    new, old, dep = graph
    old.fail_status = True
    assert supersede_revision(new, old).error
    old.fail_status = False
    repaired = supersede_revision(new, old)
    assert not repaired.error and repaired.demoted and not repaired.created
    assert len(new._outgoing) == 1


def test_replay_repairs_interrupted_ripple(graph, monkeypatch):
    new, old, dep = graph
    with monkeypatch.context() as scoped:
        scoped.setattr(dep, "set_metadata", lambda md: (_ for _ in ()).throw(RuntimeError("offline")))
        assert supersede_revision(new, old).stale == 0
    assert supersede_revision(new, old).stale == 1
    assert len(new._outgoing) == 1


def test_self_reference_is_rejected(graph):
    new, old, dep = graph
    alias = Revision(new.kref.uri)
    assert supersede_revision(new, alias).error
    assert not new._outgoing and alias.metadata == {}


def test_explicit_negative_status_ack_is_not_success(graph, monkeypatch):
    new, old, dep = graph
    monkeypatch.setattr(old, "set_attribute", lambda *args: False)
    assert supersede_revision(new, old).error
    assert old.metadata["status"] == "active"


@pytest.mark.parametrize("failed_edge", [False, True])
def test_maintenance_demotes_only_after_edge_and_ripples(graph, failed_edge):
    new, old, dep = graph
    new.fail_edge = failed_edge
    worker = GraphMaintainer(sys.modules["kumiho"], project="test")
    stats = MaintenanceStats()
    ok = worker._sink_decision({"rev": new}, {"rev": old, "slug": "old"}, stats)
    assert ok is not failed_edge
    assert (old.metadata["status"] == "superseded") is not failed_edge
    assert bool(dep.metadata.get("grounding_stale")) is not failed_edge


def test_code_capture_ripples_decision_dependencies(graph):
    new, old, dep = graph
    old.metadata.update(title="Use bounded workers for capture", decided_at="2026-01-01T00:00:00Z")
    anchor = Revision("kref://test/worker.code_anchor?r=1", incoming=[
        _Edge(old.kref.uri, "kref://test/worker.code_anchor?r=1", "IMPLEMENTED_IN"),
    ])
    meta = {"title": "Use bounded workers for capture", "decided_at": "2026-01-02T00:00:00Z"}
    stats = IngestStats()
    _supersede_pass(None, CodeMemoryConfig(), new, meta, [anchor], "", stats)
    assert stats.superseded == 1 and stats.edges == 1
    assert dep.metadata["grounding_stale"] == "true"
    _supersede_pass(None, CodeMemoryConfig(), new, meta, [anchor], "", stats)
    assert stats.superseded == 1 and stats.edges == 1


def test_agent_decompose_reports_partial_status_failure_and_repairs(monkeypatch):
    from kumiho._text import slugify
    from kumiho_memory.ontology import _sync_decompose_agent, OntologySchema
    from test_grounding_ripple import _Project, _install, _CONV
    project = _Project("proj")
    old = Revision("kref://proj/facts/prior.fact?r=1", {"status": "active"})
    project.preload(slugify("Prior belief", hash_on_truncate=True), "fact", old)
    _install(monkeypatch, {}, project=project, conv_rev=Revision(_CONV))
    declaration = {"facts": [{"statement": "New belief"}],
                   "supersedes": [{"statement": "New belief", "replaces": "Prior belief"}]}
    old.fail_status = True
    first = _sync_decompose_agent(_CONV, declaration, "proj", OntologySchema())
    assert first["supersession_failures"] == 1 and first["supersedes"] == 1
    old.fail_status = False
    second = _sync_decompose_agent(_CONV, declaration, "proj", OntologySchema())
    assert second.get("supersession_failures", 0) == 0 and second["supersedes"] == 0
    assert old.metadata["status"] == "superseded"
