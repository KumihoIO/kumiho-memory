"""Adversarial review: successful paths must not hide destructive fallbacks."""
import sys
from types import SimpleNamespace

import pytest

from kumiho._text import slugify
from kumiho_memory.ontology import OntologySchema, _sync_decompose_agent
from kumiho_memory.relations import link_supersedes
from kumiho_memory.supersession import supersede_revision
from test_grounding_ripple import _Rev, _Edge, _Project, _FactItem, _Mat, _install, _CONV


def test_unreadable_existing_edge_never_creates_a_duplicate(monkeypatch):
    old = _Rev("kref://proj/facts/old.fact?r=1", {"status": "active"})
    new = _Rev("kref://proj/facts/new.fact?r=1", outgoing=[
        _Edge("kref://proj/facts/new.fact?r=1", old.kref.uri, "SUPERSEDES"),
    ])
    _install(monkeypatch, {})
    monkeypatch.setattr(new, "get_edges", lambda **kw: (_ for _ in ()).throw(RuntimeError("read unavailable")))
    result = supersede_revision(new, old)
    assert result.error
    assert not result.created and not result.demoted
    assert len(new._outgoing) == 1 and old.metadata["status"] == "active"


@pytest.mark.parametrize("resolvable", [True, False])
def test_explicit_contradiction_never_lets_overlap_pick_a_winner(monkeypatch, resolvable):
    prior_text = "Use Upstash for the event bus streams"
    new_text = "Use Redis for the event bus streams"
    prior = _Rev("kref://proj/facts/prior.fact?r=1", {"summary": prior_text, "status": "active"})
    dep = _Rev("kref://proj/decisions/dependent.decision?r=1")
    prior._incoming = [_Edge(dep.kref.uri, prior.kref.uri, "DEPENDS_ON")]
    project = _Project("proj")
    item = project.preload(slugify(prior_text, hash_on_truncate=True), "fact", prior)
    _install(monkeypatch, {dep.kref.uri: dep}, project=project, conv_rev=_Rev(_CONV),
             search_results=[SimpleNamespace(item=item)])
    stats = _sync_decompose_agent(_CONV, {
        "facts": [{"statement": new_text}],
        "contradicts": [{"statement": new_text,
                        "conflicts_with": prior_text if resolvable else "unavailable target"}],
    }, "proj", OntologySchema())
    new = project.get_item(slugify(new_text, hash_on_truncate=True), "fact", "/proj/facts").get_latest_revision()
    assert stats["contradicts"] == int(resolvable)
    assert new.get_edges(edge_type_filter="SUPERSEDES", direction=0) == []
    assert prior.metadata["status"] == "active"
    assert dep.metadata == {}


def test_replaying_an_old_fact_cannot_supersede_its_replacement(monkeypatch):
    old_text = "Use Upstash for the event bus streams"
    new_text = "Use Redis for the event bus streams"
    old = _Rev("kref://proj/facts/old.fact?r=1", {"summary": old_text, "status": "active"})
    new = _Rev("kref://proj/facts/new.fact?r=1", {"summary": new_text, "status": "active"})
    _install(monkeypatch, {})
    assert supersede_revision(new, old).demoted
    sys.modules["kumiho"].search = lambda *a, **kw: [SimpleNamespace(item=_FactItem("kref://proj/facts/new.fact", new))]
    assert link_supersedes(_Mat(), "fact", "facts", "old", old, old_text, "proj") == 0
    assert new.metadata["status"] == "active"
    assert old.get_edges(edge_type_filter="SUPERSEDES", direction=0) == []


def test_explicit_negative_edge_ack_cannot_demote(monkeypatch):
    old = _Rev("kref://proj/facts/old.fact?r=1", {"status": "active"})
    new = _Rev("kref://proj/facts/new.fact?r=1")
    _install(monkeypatch, {})
    monkeypatch.setattr(new, "create_edge", lambda *a, **kw: False)
    result = supersede_revision(new, old)
    assert result.error and not result.linked and not result.demoted
    assert old.metadata["status"] == "active"


def test_reverse_edge_blocks_demotion_even_before_status_was_repaired(monkeypatch):
    first = _Rev("kref://proj/facts/first.fact?r=1", {"status": "active"})
    second = _Rev("kref://proj/facts/second.fact?r=1", {"status": "active"}, outgoing=[
        _Edge("kref://proj/facts/second.fact?r=1", first.kref.uri, "SUPERSEDES"),
    ])
    _install(monkeypatch, {})
    result = supersede_revision(first, second)
    assert result.error and not result.created and not result.demoted
    assert first._outgoing == [] and second.metadata["status"] == "active"


def test_contradiction_protects_both_facts_in_the_same_call(monkeypatch):
    texts = ["Use Upstash for the event bus streams", "Use Redis for the event bus streams"]
    project = _Project("proj")
    fake = _install(monkeypatch, {}, project=project, conv_rev=_Rev(_CONV))
    fake.search = lambda *a, **kw: [SimpleNamespace(item=item)
                                  for (_, _, kind), item in project._items.items() if kind == "fact"]
    stats = _sync_decompose_agent(_CONV, {
        "facts": [{"statement": t} for t in texts],
        "contradicts": [{"statement": texts[1], "conflicts_with": texts[0]}],
    }, "proj", OntologySchema())
    assert stats["contradicts"] == 1
    for (_, _, kind), item in project._items.items():
        if kind == "fact":
            rev = item.get_latest_revision()
            assert rev.get_edges(edge_type_filter="SUPERSEDES", direction=0) == []
            assert rev.metadata.get("status") != "superseded"


@pytest.mark.parametrize("unreadable", ["source", "target"])
def test_code_capture_propagates_uncertain_replacement_for_retry(monkeypatch, unreadable):
    from kumiho_memory.code_capture import IngestStats, _supersede_pass
    from kumiho_memory.code_decisions import CodeMemoryConfig

    old = _Rev("kref://proj/decisions/old.code_decision?r=1", {
        "title": "Use bounded workers for capture", "decided_at": "2026-01-01T00:00:00Z", "status": "active",
    })
    new = _Rev("kref://proj/decisions/new.code_decision?r=1", {
        "title": "Use bounded workers for capture", "decided_at": "2026-01-02T00:00:00Z", "status": "active",
    })
    anchor_uri = "kref://proj/anchors/worker.code_anchor?r=1"
    anchor = _Rev(anchor_uri, incoming=[_Edge(old.kref.uri, anchor_uri, "IMPLEMENTED_IN")])
    fake = _install(monkeypatch, {old.kref.uri: old})
    fake.INCOMING = 1
    monkeypatch.setattr(new if unreadable == "source" else old, "get_edges",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("read unavailable")))
    stats = IngestStats()
    # _sync_write_decision propagates this; _sync_write_commit writes its
    # completion marker only after every decision stage returns successfully.
    with pytest.raises(RuntimeError, match="read unavailable"):
        _supersede_pass(None, CodeMemoryConfig(), new, new.metadata, [anchor], "", stats)
    assert stats.edges == stats.superseded == 0
    assert old.metadata["status"] == "active" and new._outgoing == []
