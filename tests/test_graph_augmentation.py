"""Flag-gated traversal of registered entity→entity relation edges (G1 read side).

The relation edges decompose writes between entity anchors (USES, DEPENDS_ON,
..., canonicalized by ``predicate_registry``) were read by no recall path; this
extends the entity-mediated reader to cross them, behind
``relation_traversal`` (env ``KUMIHO_MEMORY_RELATION_TRAVERSAL``, DEFAULT OFF).

Uses the same fake-kumiho seam as ``test_graph_recall_rerank`` — a plain
``sys.modules['kumiho']`` stub with ``get_revision``/``get_edges`` — so no
server is needed. ``monkeypatch.setitem`` (never pop) per repo convention.
"""

import asyncio

from kumiho_memory.graph_augmentation import (
    GraphAugmentationConfig,
    GraphAugmentedRecall,
)


# ---------------------------------------------------------------------------
# Fake-kumiho graph seam (mirrors test_graph_recall_rerank)
# ---------------------------------------------------------------------------

class _FakeKref:
    def __init__(self, uri):
        self.uri = uri


class _FakeEdge:
    def __init__(self, source, target, edge_type):
        self.source_kref = _FakeKref(source)
        self.target_kref = _FakeKref(target)
        self.edge_type = edge_type


class _FakeRev:
    def __init__(self, kref, metadata, edges):
        self.kref = _FakeKref(kref)
        self.metadata = metadata
        self._edges = edges

    def get_edges(self, direction=None):
        return self._edges


def _install_graph(monkeypatch, graph, on_get=None):
    import sys
    import types

    def _get(kref):
        if on_get is not None:
            on_get(kref)
        return graph[kref]

    fake = types.ModuleType("kumiho")
    fake.BOTH = "BOTH"
    fake.get_revision = _get
    monkeypatch.setitem(sys.modules, "kumiho", fake)


# Krefs: entity anchors end in ``.entity`` (the reader parses the kref kind to
# refuse hopping through a non-entity neighbour).
M1 = "kref://p/notes/m1.conversation?r=1"
A = "kref://p/entities/acme.entity?r=1"        # seed's entity anchor
N = "kref://p/entities/redis.entity?r=1"        # neighbour entity anchor
NMEM = "kref://p/notes/redis-note.conversation?r=1"


def _seed_to_neighbour_graph(relation_type="USES", direction="out",
                             neighbour=N, extra_anchor_edges=None,
                             neighbour_edges=None):
    """M1 --ABOUT--> A --<relation>--> N <--ABOUT-- NMEM (default)."""
    if direction == "out":
        rel_edge = _FakeEdge(A, neighbour, relation_type)   # A is subject
    else:
        rel_edge = _FakeEdge(neighbour, A, relation_type)   # A is object
    anchor_edges = [_FakeEdge(M1, A, "ABOUT"), rel_edge]
    anchor_edges += (extra_anchor_edges or [])
    n_edges = neighbour_edges if neighbour_edges is not None else [
        _FakeEdge(NMEM, neighbour, "ABOUT"),
    ]
    return {
        M1: _FakeRev(M1, {"title": "M1", "summary": "seed"},
                     [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, anchor_edges),
        neighbour: _FakeRev(neighbour, {"display_name": "Redis"}, n_edges),
        NMEM: _FakeRev(NMEM, {"title": "Redis note",
                              "summary": "redis powers the buffer"}, []),
    }


def _run(monkeypatch, graph, *, relation_traversal, query="", seen=None,
         on_get=None, **cfg_kw):
    _install_graph(monkeypatch, graph, on_get=on_get)
    gr = GraphAugmentedRecall(config=GraphAugmentationConfig(
        entity_recall=True, relation_traversal=relation_traversal, **cfg_kw,
    ))
    augmented = []
    seen = set(seen if seen is not None else {M1})
    found = asyncio.run(
        gr._traverse_entity_neighbors([M1], seen, augmented, query=query)
    )
    return found, augmented


# ---------------------------------------------------------------------------
# Flag OFF ⇒ byte-identical (the new code path is never entered)
# ---------------------------------------------------------------------------

def test_relation_traversal_off_never_enters_relation_path(monkeypatch):
    # The regression guard: with the flag off (the default), the relation edge
    # is ignored, the neighbour's memory is NOT surfaced, and the neighbour
    # anchor is NEVER fetched — proving the new round-trips don't happen.
    fetches = []
    graph = _seed_to_neighbour_graph()
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=False,
        on_get=lambda kref: fetches.append(kref),
    )
    assert found == 0
    assert augmented == []
    assert N not in fetches            # neighbour anchor never fetched
    assert NMEM not in fetches         # nor its memory


def test_relation_traversal_off_is_default():
    assert GraphAugmentationConfig().relation_traversal is False


# ---------------------------------------------------------------------------
# Flag ON ⇒ neighbour reachable through BOTH edge directions, with provenance
# ---------------------------------------------------------------------------

def test_relation_traversal_reaches_neighbour_via_outgoing_edge(monkeypatch):
    graph = _seed_to_neighbour_graph(relation_type="USES", direction="out")
    found, augmented = _run(monkeypatch, graph, relation_traversal=True)
    assert found == 1
    entry = augmented[0]
    assert entry["kref"] == NMEM
    # Provenance: the relation crossed + the intermediate entity + the seed.
    assert entry["via_relation"] == "USES"
    assert entry["via_entity"] == N               # entity the sibling is about
    assert entry["relation_from_entity"] == A     # intermediate (origin) anchor
    assert entry["from_kref"] == M1
    assert entry["edge_type"] == "ABOUT"          # memory -> neighbour anchor
    assert entry["hop"] == 3
    # Score-less like the direct siblings: never evicts/outranks a direct hit.
    assert entry.get("score") is None


def test_relation_traversal_reaches_neighbour_via_incoming_edge(monkeypatch):
    # The relation edge points INTO the anchor (anchor is the object) — the
    # reader must follow it in reverse to reach the neighbour all the same.
    graph = _seed_to_neighbour_graph(relation_type="DEPENDS_ON", direction="in")
    found, augmented = _run(monkeypatch, graph, relation_traversal=True)
    assert found == 1
    assert augmented[0]["kref"] == NMEM
    assert augmented[0]["via_relation"] == "DEPENDS_ON"


def test_sibling_cap_does_not_short_circuit_relation_collection(monkeypatch):
    # Regression (confirmed bug): the cap-filling sibling used to break out of
    # the whole edge scan, so a relation edge arriving AFTER it was never
    # collected — reachability depended on server edge order. Both permutations
    # must reach the same nodes.
    S1 = "kref://p/notes/s1.conversation?r=1"

    def _graph(anchor_edges):
        return {
            M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
            A: _FakeRev(A, {"display_name": "Acme"}, anchor_edges),
            N: _FakeRev(N, {"display_name": "Redis"},
                        [_FakeEdge(NMEM, N, "ABOUT")]),
            NMEM: _FakeRev(NMEM, {"title": "Redis note", "summary": "s"}, []),
            S1: _FakeRev(S1, {"title": "S1", "summary": "s"}, []),
        }

    siblings_first = [
        _FakeEdge(M1, A, "ABOUT"),
        _FakeEdge(S1, A, "ABOUT"),      # fills max_siblings=1
        _FakeEdge(A, N, "USES"),        # used to be unreachable (post-break)
    ]
    relation_first = [
        _FakeEdge(M1, A, "ABOUT"),
        _FakeEdge(A, N, "USES"),
        _FakeEdge(S1, A, "ABOUT"),
    ]
    reached = []
    for edges in (siblings_first, relation_first):
        found, augmented = _run(
            monkeypatch, _graph(edges), relation_traversal=True,
            entity_recall_max_siblings=1,
        )
        reached.append({m["kref"] for m in augmented})
    assert reached[0] == reached[1]     # order-independent reachability
    assert NMEM in reached[0]           # relation neighbour's memory reached
    assert S1 in reached[0]             # direct sibling (cap slot) kept too


def test_relation_traversal_follows_involves_into_neighbour(monkeypatch):
    # A neighbour's memories are reached via ABOUT *and* INVOLVES (event nodes).
    E = "kref://p/events/launch.event?r=1"
    graph = _seed_to_neighbour_graph(neighbour_edges=[
        _FakeEdge(E, N, "INVOLVES"),
    ])
    graph[E] = _FakeRev(E, {"title": "Launch", "summary": "launched",
                            "event_date": "2026-01-01"}, [])
    found, augmented = _run(monkeypatch, graph, relation_traversal=True)
    assert found == 1
    assert augmented[0]["kref"] == E
    assert augmented[0]["edge_type"] == "INVOLVES"


# ---------------------------------------------------------------------------
# RELATES_TO is the lowest-priority fallback bucket
# ---------------------------------------------------------------------------

def test_relates_to_is_lowest_priority(monkeypatch):
    # The anchor has a specific relation (USES→N) and a RELATES_TO (→R). With
    # the per-anchor edge budget = 1, the specific relation wins the slot and
    # the RELATES_TO neighbour is never expanded — even though its edge is
    # listed FIRST (proving the ordering, not arrival order, decides).
    R = "kref://p/entities/misc.entity?r=1"
    RMEM = "kref://p/notes/misc-note.conversation?r=1"
    graph = _seed_to_neighbour_graph(extra_anchor_edges=[])
    # Rebuild A's edges so RELATES_TO arrives before USES.
    graph[A] = _FakeRev(A, {"display_name": "Acme"}, [
        _FakeEdge(M1, A, "ABOUT"),
        _FakeEdge(A, R, "RELATES_TO"),   # fallback, listed first
        _FakeEdge(A, N, "USES"),          # specific, listed second
    ])
    graph[R] = _FakeRev(R, {"display_name": "Misc"}, [_FakeEdge(RMEM, R, "ABOUT")])
    graph[RMEM] = _FakeRev(RMEM, {"title": "Misc", "summary": "misc"}, [])
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=True,
        relation_traversal_max_edges_per_anchor=1,
    )
    krefs = [m["kref"] for m in augmented]
    assert krefs == [NMEM]        # USES neighbour surfaced
    assert RMEM not in krefs      # RELATES_TO neighbour dropped by the budget


def test_relates_to_still_followed_when_budget_allows(monkeypatch):
    # RELATES_TO is deprioritized, not excluded: with room in the budget it is
    # still crossed (a relation is never dropped, mirroring the writer).
    R = "kref://p/entities/misc.entity?r=1"
    RMEM = "kref://p/notes/misc-note.conversation?r=1"
    graph = _seed_to_neighbour_graph()
    graph[A] = _FakeRev(A, {"display_name": "Acme"}, [
        _FakeEdge(M1, A, "ABOUT"),
        _FakeEdge(A, R, "RELATES_TO"),
        _FakeEdge(A, N, "USES"),
    ])
    graph[R] = _FakeRev(R, {"display_name": "Misc"}, [_FakeEdge(RMEM, R, "ABOUT")])
    graph[RMEM] = _FakeRev(RMEM, {"title": "Misc", "summary": "misc"}, [])
    found, augmented = _run(monkeypatch, graph, relation_traversal=True)
    krefs = {m["kref"] for m in augmented}
    assert krefs == {NMEM, RMEM}
    assert next(m for m in augmented if m["kref"] == RMEM)["via_relation"] == "RELATES_TO"


def test_global_caps_prefer_specific_over_earlier_relates_to(monkeypatch):
    # Per-anchor priority alone is not enough: an EARLIER anchor's RELATES_TO
    # must not consume the global neighbour budget ahead of a LATER anchor's
    # specific relation — candidates are partitioned specific-first globally
    # (stable, arrival order kept within each class) before the caps apply.
    A2 = "kref://p/entities/second.entity?r=1"
    R = "kref://p/entities/misc.entity?r=1"
    RMEM = "kref://p/notes/misc-note.conversation?r=1"
    graph = {
        M1: _FakeRev(M1, {"title": "M1"}, [
            _FakeEdge(M1, A, "ABOUT"),      # A expanded first
            _FakeEdge(M1, A2, "ABOUT"),     # A2 second
        ]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"),
            _FakeEdge(A, R, "RELATES_TO"),  # earlier anchor, fallback bucket
        ]),
        A2: _FakeRev(A2, {"display_name": "Second"}, [
            _FakeEdge(M1, A2, "ABOUT"),
            _FakeEdge(A2, N, "USES"),       # later anchor, specific relation
        ]),
        R: _FakeRev(R, {"display_name": "Misc"}, [_FakeEdge(RMEM, R, "ABOUT")]),
        RMEM: _FakeRev(RMEM, {"title": "Misc", "summary": "s"}, []),
        N: _FakeRev(N, {"display_name": "Redis"}, [_FakeEdge(NMEM, N, "ABOUT")]),
        NMEM: _FakeRev(NMEM, {"title": "Redis note", "summary": "s"}, []),
    }
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=True,
        relation_traversal_max_neighbors=1,
    )
    assert [m["kref"] for m in augmented] == [NMEM]   # specific wins globally
    assert augmented[0]["via_relation"] == "USES"


# ---------------------------------------------------------------------------
# Hard caps: edges-per-anchor, neighbours-total, results-total
# ---------------------------------------------------------------------------

def test_cap_max_neighbours_total(monkeypatch):
    # Five USES neighbours, each with one memory; the neighbour cap bounds how
    # many distinct anchors are expanded per recall.
    anchor_edges = [_FakeEdge(M1, A, "ABOUT")]
    graph = {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
    }
    for i in range(5):
        ni = f"kref://p/entities/n{i}.entity?r=1"
        mi = f"kref://p/notes/n{i}-mem.conversation?r=1"
        anchor_edges.append(_FakeEdge(A, ni, "USES"))
        graph[ni] = _FakeRev(ni, {"display_name": f"N{i}"}, [_FakeEdge(mi, ni, "ABOUT")])
        graph[mi] = _FakeRev(mi, {"title": f"N{i} mem", "summary": "s"}, [])
    graph[A] = _FakeRev(A, {"display_name": "Acme"}, anchor_edges)
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=True,
        relation_traversal_max_edges_per_anchor=5,
        relation_traversal_max_neighbors=2,
        relation_traversal_max_results=10,
    )
    assert found == 2             # only two neighbours expanded


def test_cap_max_results_total(monkeypatch):
    # One neighbour with five memories; the results cap bounds the total pulled
    # in via this path, regardless of a generous per-anchor sibling cap.
    n_edges = []
    graph = {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"},
                    [_FakeEdge(M1, A, "ABOUT"), _FakeEdge(A, N, "USES")]),
    }
    for i in range(5):
        mi = f"kref://p/notes/rn{i}.conversation?r=1"
        n_edges.append(_FakeEdge(mi, N, "ABOUT"))
        graph[mi] = _FakeRev(mi, {"title": f"RN{i}", "summary": "s"}, [])
    graph[N] = _FakeRev(N, {"display_name": "Redis"}, n_edges)
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=True,
        relation_traversal_max_results=2,
    )
    assert found == 2


def test_cap_max_edges_per_anchor(monkeypatch):
    # Three USES edges from the anchor but a per-anchor budget of 1 → at most
    # one neighbour queued from this anchor.
    anchor_edges = [_FakeEdge(M1, A, "ABOUT")]
    graph = {M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")])}
    for i in range(3):
        ni = f"kref://p/entities/e{i}.entity?r=1"
        mi = f"kref://p/notes/e{i}-mem.conversation?r=1"
        anchor_edges.append(_FakeEdge(A, ni, "USES"))
        graph[ni] = _FakeRev(ni, {"display_name": f"E{i}"}, [_FakeEdge(mi, ni, "ABOUT")])
        graph[mi] = _FakeRev(mi, {"title": f"E{i} mem", "summary": "s"}, [])
    graph[A] = _FakeRev(A, {"display_name": "Acme"}, anchor_edges)
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=True,
        relation_traversal_max_edges_per_anchor=1,
        relation_traversal_max_neighbors=10,
        relation_traversal_max_results=10,
    )
    assert found == 1


# ---------------------------------------------------------------------------
# Ubiquitous-neighbour guard (reuses entity_bridge_hub_degree_max)
# ---------------------------------------------------------------------------

def test_ubiquitous_neighbour_is_skipped(monkeypatch):
    # A hub neighbour (incoming ABOUT count above the cutoff) is skipped — its
    # memories are generic noise. A discriminative low-degree neighbour on the
    # same anchor is surfaced.
    HUB = "kref://p/entities/hub.entity?r=1"
    LOW = "kref://p/entities/low.entity?r=1"
    LOWMEM = "kref://p/notes/low-mem.conversation?r=1"
    hub_mems = [f"kref://p/notes/h{i}.conversation?r=1" for i in range(3)]
    graph = {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"),
            _FakeEdge(A, HUB, "USES"),
            _FakeEdge(A, LOW, "USES"),
        ]),
        HUB: _FakeRev(HUB, {"display_name": "Hub"},
                      [_FakeEdge(m, HUB, "ABOUT") for m in hub_mems]),  # 3 incoming
        LOW: _FakeRev(LOW, {"display_name": "Low"},
                      [_FakeEdge(LOWMEM, LOW, "ABOUT")]),               # 1 incoming
        LOWMEM: _FakeRev(LOWMEM, {"title": "Low mem", "summary": "s"}, []),
    }
    for m in hub_mems:
        graph[m] = _FakeRev(m, {"title": "h", "summary": "s"}, [])
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=True,
        entity_bridge_hub_degree_max=2,   # hub has 3 incoming > 2 → skipped
        relation_traversal_max_neighbors=10,
        relation_traversal_max_results=10,
    )
    krefs = [m["kref"] for m in augmented]
    assert krefs == [LOWMEM]                 # discriminative neighbour only
    assert all(m not in krefs for m in hub_mems)


# ---------------------------------------------------------------------------
# Dedup: reachable both directly and via relation → direct provenance wins
# ---------------------------------------------------------------------------

def test_dedup_direct_provenance_wins(monkeypatch):
    # SHARED is both a direct sibling of A (hop 2) and a memory of neighbour N
    # (would be hop 3). It must appear ONCE, with the direct provenance — the
    # relation phase runs after all direct siblings are claimed.
    SHARED = "kref://p/notes/shared.conversation?r=1"
    NONLY = "kref://p/notes/nonly.conversation?r=1"
    graph = {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"),
            _FakeEdge(SHARED, A, "ABOUT"),   # SHARED is a DIRECT sibling of A
            _FakeEdge(A, N, "USES"),
        ]),
        N: _FakeRev(N, {"display_name": "Redis"}, [
            _FakeEdge(SHARED, N, "ABOUT"),   # SHARED also a memory of N
            _FakeEdge(NONLY, N, "ABOUT"),
        ]),
        SHARED: _FakeRev(SHARED, {"title": "Shared", "summary": "s"}, []),
        NONLY: _FakeRev(NONLY, {"title": "N only", "summary": "s"}, []),
    }
    found, augmented = _run(monkeypatch, graph, relation_traversal=True)
    by_kref = {}
    for m in augmented:
        by_kref.setdefault(m["kref"], []).append(m)
    assert len(by_kref[SHARED]) == 1              # appears exactly once
    assert by_kref[SHARED][0]["hop"] == 2         # direct provenance wins
    assert "via_relation" not in by_kref[SHARED][0]
    assert NONLY in by_kref and by_kref[NONLY][0]["hop"] == 3


def test_stray_non_canonical_edge_type_not_followed(monkeypatch):
    # An entity->entity edge whose type is OUTSIDE the registry's canonical set
    # (e.g. OWNS, written by pre-registry code) is not traversed — the reader's
    # vocabulary is exactly predicate_registry.canonical_types().
    graph = _seed_to_neighbour_graph(relation_type="OWNS", direction="out")
    found, augmented = _run(monkeypatch, graph, relation_traversal=True)
    assert found == 0 and augmented == []


def test_non_entity_neighbour_not_followed(monkeypatch):
    # A canonical-typed edge to a non-entity kref (a stray conversation) is a
    # dead-end stub — the reader refuses to hop it.
    STRAY = "kref://p/notes/stray.conversation?r=1"
    graph = {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"),
            _FakeEdge(A, STRAY, "USES"),
        ]),
        STRAY: _FakeRev(STRAY, {"title": "Stray", "summary": "s"}, []),
    }
    found, augmented = _run(monkeypatch, graph, relation_traversal=True)
    assert found == 0 and augmented == []


# ---------------------------------------------------------------------------
# End-to-end recall(): relation sibling rides on top, never evicts direct hits
# ---------------------------------------------------------------------------

def test_recall_surfaces_relation_sibling_on_top(monkeypatch):
    # Through the full recall() pipeline: the base hit stays first and scored,
    # the relation-linked neighbour memory rides on top as a score-less sibling.
    _install_graph(monkeypatch, _seed_to_neighbour_graph())

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [{"kref": M1, "title": "M1", "summary": "seed", "score": 0.9}]

    gr = GraphAugmentedRecall(
        recall_fn=recall_fn,
        config=GraphAugmentationConfig(entity_recall=True, relation_traversal=True),
    )
    out = asyncio.run(gr.recall("acme uses redis?", limit=3))
    krefs = [m["kref"] for m in out]
    assert krefs[0] == M1                          # scored base hit stays first
    assert NMEM in krefs                           # relation sibling on top
    entry = next(m for m in out if m["kref"] == NMEM)
    assert entry["via_relation"] == "USES" and entry["hop"] == 3
    assert entry.get("score") is None


def test_recall_relation_off_is_inert(monkeypatch):
    # With the flag off, the same graph yields only the base hit — the relation
    # neighbour is invisible to recall().
    _install_graph(monkeypatch, _seed_to_neighbour_graph())

    async def recall_fn(query, *, limit, space_paths=None, memory_types=None):
        return [{"kref": M1, "title": "M1", "summary": "seed", "score": 0.9}]

    gr = GraphAugmentedRecall(
        recall_fn=recall_fn,
        config=GraphAugmentationConfig(entity_recall=True, relation_traversal=False),
    )
    out = asyncio.run(gr.recall("acme uses redis?", limit=3))
    assert [m["kref"] for m in out] == [M1]


# ---------------------------------------------------------------------------
# Env-flag wiring through the manager (default OFF; opt in with =1)
# ---------------------------------------------------------------------------

def _build_manager(**kwargs):
    from kumiho_memory.memory_manager import UniversalMemoryManager
    from kumiho_memory.redis_memory import RedisMemoryBuffer
    from fakes import FakeRedis

    class StubRedactor:
        def redact(self, text):
            return text

        def reject_credentials(self, text):
            return None

    return UniversalMemoryManager(
        redis_buffer=RedisMemoryBuffer(client=FakeRedis(), redis_url="redis://test"),
        summarizer=object(),
        pii_redactor=StubRedactor(),
        memory_store=None,
        **kwargs,
    )


def test_relation_traversal_env_default_off_and_opt_in(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_RELATION_TRAVERSAL", raising=False)
    # Ontology on by default lights up entity_recall, but relation traversal
    # stays OFF until explicitly opted in.
    m = _build_manager(graph_augmentation=True)
    assert m.graph_augmentation_config.entity_recall is True
    assert m.graph_augmentation_config.relation_traversal is False

    monkeypatch.setenv("KUMIHO_MEMORY_RELATION_TRAVERSAL", "1")
    m = _build_manager(graph_augmentation=True)
    assert m.graph_augmentation_config.relation_traversal is True

    # Any non-"1" value (including "0") leaves it off.
    monkeypatch.setenv("KUMIHO_MEMORY_RELATION_TRAVERSAL", "0")
    m = _build_manager(graph_augmentation=True)
    assert m.graph_augmentation_config.relation_traversal is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
