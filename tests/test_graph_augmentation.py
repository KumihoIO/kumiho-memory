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
import itertools

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
    def __init__(self, source, target, edge_type, metadata=None):
        self.source_kref = _FakeKref(source)
        self.target_kref = _FakeKref(target)
        self.edge_type = edge_type
        self.metadata = dict(metadata or {})  # mirrors kumiho.edge.Edge.metadata


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
# #105/F3 — entity->entity belief-TYPED relation edges are domain claims, not
# belief edges: CONTRADICTS/SUPERSEDES are canonical relation predicates, so
# they DO hang off anchors, and without dispute basis they must neither ride
# the belief class nor the belief alphabet ("C..." < "DEPENDS_ON"/"USES") into
# the relation caps. Demoted below the specifics, above RELATES_TO.
# ---------------------------------------------------------------------------

def test_entity_relation_contradicts_does_not_outcompete_specifics(monkeypatch):
    # A carries an entity->entity CONTRADICTS domain claim (predicate metadata,
    # NO basis) plus DEPENDS_ON and USES. With the per-anchor edge budget = 2,
    # the two positive specifics win the slots and the CONTRADICTS neighbour is
    # never expanded — in BOTH arrival orders (the class demotion, not the
    # alphabet or arrival order, decides: "CONTRADICTS" < "DEPENDS_ON" < "USES"
    # lexicographically, so an undemoted CONTRADICTS would always win).
    CE = "kref://p/entities/rival.entity?r=1"
    CEM = "kref://p/notes/rival-note.conversation?r=1"
    D = "kref://p/entities/postgres.entity?r=1"
    DMEM = "kref://p/notes/postgres-note.conversation?r=1"
    rel_edges = [
        _FakeEdge(A, CE, "CONTRADICTS", {"predicate": "conflicts_with"}),
        _FakeEdge(A, D, "DEPENDS_ON"),
        _FakeEdge(A, N, "USES"),
    ]
    for order in (rel_edges, list(reversed(rel_edges))):
        graph = {
            M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
            A: _FakeRev(A, {"display_name": "Acme"},
                        [_FakeEdge(M1, A, "ABOUT")] + order),
            CE: _FakeRev(CE, {"display_name": "Rival"},
                         [_FakeEdge(CEM, CE, "ABOUT")]),
            CEM: _FakeRev(CEM, {"title": "Rival note", "summary": "r"}, []),
            D: _FakeRev(D, {"display_name": "Postgres"},
                        [_FakeEdge(DMEM, D, "ABOUT")]),
            DMEM: _FakeRev(DMEM, {"title": "Pg note", "summary": "d"}, []),
            N: _FakeRev(N, {"display_name": "Redis"},
                        [_FakeEdge(NMEM, N, "ABOUT")]),
            NMEM: _FakeRev(NMEM, {"title": "Redis note", "summary": "n"}, []),
        }
        found, augmented = _run(
            monkeypatch, graph, relation_traversal=True,
            relation_traversal_max_edges_per_anchor=2,
        )
        krefs = {m["kref"] for m in augmented}
        assert krefs == {DMEM, NMEM}   # the positive specifics win the budget
        assert CEM not in krefs        # domain claim starved by the cap


def test_entity_relation_contradicts_still_followed_when_budget_allows(monkeypatch):
    # Demoted, not excluded (mirrors the RELATES_TO rule): with room in the
    # per-anchor budget the CONTRADICTS domain claim is still crossed, and it
    # still outranks the RELATES_TO fallback.
    CE = "kref://p/entities/rival.entity?r=1"
    CEM = "kref://p/notes/rival-note.conversation?r=1"
    R = "kref://p/entities/misc.entity?r=1"
    RMEM = "kref://p/notes/misc-note.conversation?r=1"
    graph = {
        M1: _FakeRev(M1, {"title": "M1"}, [_FakeEdge(M1, A, "ABOUT")]),
        A: _FakeRev(A, {"display_name": "Acme"}, [
            _FakeEdge(M1, A, "ABOUT"),
            _FakeEdge(A, R, "RELATES_TO"),   # fallback, listed first
            _FakeEdge(A, CE, "CONTRADICTS", {"predicate": "conflicts_with"}),
            _FakeEdge(A, N, "USES"),
        ]),
        CE: _FakeRev(CE, {"display_name": "Rival"},
                     [_FakeEdge(CEM, CE, "ABOUT")]),
        CEM: _FakeRev(CEM, {"title": "Rival note", "summary": "r"}, []),
        R: _FakeRev(R, {"display_name": "Misc"},
                    [_FakeEdge(RMEM, R, "ABOUT")]),
        RMEM: _FakeRev(RMEM, {"title": "Misc", "summary": "m"}, []),
        N: _FakeRev(N, {"display_name": "Redis"},
                    [_FakeEdge(NMEM, N, "ABOUT")]),
        NMEM: _FakeRev(NMEM, {"title": "Redis note", "summary": "n"}, []),
    }
    found, augmented = _run(
        monkeypatch, graph, relation_traversal=True,
        relation_traversal_max_edges_per_anchor=2,
    )
    krefs = {m["kref"] for m in augmented}
    assert krefs == {NMEM, CEM}    # USES first, then the domain claim
    assert RMEM not in krefs       # fallback still last, starved by the cap
    assert next(m for m in augmented if m["kref"] == CEM)[
        "via_relation"] == "CONTRADICTS"


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


# ---------------------------------------------------------------------------
# CONTRADICTS: first-class edge (whitelist) + contested_by recall marker
# ---------------------------------------------------------------------------

# Krefs for the contested-marker walk. The seed is a conversation; the opposing
# nodes are facts (kind is irrelevant to the walk, but mirrors real usage).
CS = "kref://p/notes/cs.conversation?r=1"
CT = "kref://p/facts/ct.fact?r=1"


def _traverse(monkeypatch, graph, seeds, base, seen=None, **cfg_kw):
    """Drive _traverse_edges against the fake graph seam."""
    _install_graph(monkeypatch, graph)
    gr = GraphAugmentedRecall(config=GraphAugmentationConfig(**cfg_kw))
    augmented = list(base)  # shallow copy: entry dicts are shared with `base`
    seen = set(seen if seen is not None else {m["kref"] for m in base})
    found = asyncio.run(gr._traverse_edges(seeds, seen, augmented))
    return found, augmented


def test_default_edge_types_include_contradicts():
    # Whitelist: CONTRADICTS is traversable/visible like SUPERSEDES.
    assert "CONTRADICTS" in GraphAugmentationConfig().edge_types


def test_contradicts_edge_surfaces_opposing_and_marks_both(monkeypatch):
    # Outgoing CONTRADICTS (seed -> target, basis: agent): the opposing revision
    # is surfaced by the walk (like SUPERSEDES) AND both endpoints gain
    # `contested_by`.
    graph = {
        CS: _FakeRev(CS, {"title": "S", "summary": "X is true"},
                     [_FakeEdge(CS, CT, "CONTRADICTS", {"basis": "agent"})]),
        CT: _FakeRev(CT, {"title": "T", "summary": "X is false"}, []),
    }
    base = [{"kref": CS, "title": "S", "summary": "X is true", "score": 0.5}]
    _found, augmented = _traverse(monkeypatch, graph, [CS], base)

    surfaced = [m for m in augmented if m["kref"] == CT]
    assert len(surfaced) == 1
    assert surfaced[0]["edge_type"] == "CONTRADICTS"
    assert surfaced[0]["contested_by"] == [CS]
    # The seed result entry is marked as contested by the opposing revision.
    assert base[0]["contested_by"] == [CT]


def test_contradicts_edge_incoming_direction_also_marks(monkeypatch):
    # Incoming CONTRADICTS (target -> seed, basis: evidence-assessor): both
    # directions and both dispute bases must mark.
    graph = {
        CS: _FakeRev(CS, {"title": "S", "summary": "X is true"},
                     [_FakeEdge(CT, CS, "CONTRADICTS",
                                {"basis": "evidence-assessor"})]),
        CT: _FakeRev(CT, {"title": "T", "summary": "X is false"}, []),
    }
    base = [{"kref": CS, "title": "S", "summary": "X is true", "score": 0.5}]
    _found, augmented = _traverse(monkeypatch, graph, [CS], base)

    assert base[0]["contested_by"] == [CT]
    surfaced = [m for m in augmented if m["kref"] == CT]
    assert surfaced and surfaced[0]["contested_by"] == [CS]


def test_entity_relation_contradicts_is_not_a_dispute(monkeypatch):
    # An entity->entity CONTRADICTS relation edge (predicate registry: carries
    # `predicate` metadata, NO basis) is a domain claim — it must neither stamp
    # contested_by nor surface the other entity via the dispute path.
    ea = "kref://p/entities/theory-a.entity?r=1"
    eb = "kref://p/entities/theory-b.entity?r=1"
    graph = {
        ea: _FakeRev(ea, {"display_name": "Theory A"},
                     [_FakeEdge(ea, eb, "CONTRADICTS",
                                {"predicate": "conflicts_with"})]),
        eb: _FakeRev(eb, {"display_name": "Theory B"}, []),
    }
    base = [{"kref": ea, "title": "Theory A", "summary": "", "score": 0.5}]
    _found, augmented = _traverse(monkeypatch, graph, [ea], base)

    assert "contested_by" not in base[0]
    assert not [m for m in augmented if m["kref"] == eb]   # not surfaced
    assert _found == 0


def test_basisless_contradicts_is_not_a_dispute(monkeypatch):
    # Defensive: a CONTRADICTS edge with NO metadata at all (unknown writer) is
    # not treated as a dispute either — only the two known bases mark.
    graph = {
        CS: _FakeRev(CS, {"title": "S", "summary": "X"},
                     [_FakeEdge(CS, CT, "CONTRADICTS")]),
        CT: _FakeRev(CT, {"title": "T", "summary": "not X"}, []),
    }
    base = [{"kref": CS, "title": "S", "summary": "X", "score": 0.5}]
    _found, augmented = _traverse(monkeypatch, graph, [CS], base)

    assert "contested_by" not in base[0]
    assert not [m for m in augmented if m["kref"] == CT]


def test_contested_by_is_bounded(monkeypatch):
    # A heavily-contested seed surfaces at most the first 3 disputing krefs.
    targets = [f"kref://p/facts/t{i}.fact?r=1" for i in range(5)]
    graph = {
        CS: _FakeRev(CS, {"title": "S", "summary": "X"},
                     [_FakeEdge(CS, t, "CONTRADICTS", {"basis": "agent"})
                      for t in targets]),
    }
    for t in targets:
        graph[t] = _FakeRev(t, {"title": t, "summary": "not X"}, [])
    base = [{"kref": CS, "title": "S", "summary": "X", "score": 0.5}]
    _traverse(monkeypatch, graph, [CS], base)

    assert base[0]["contested_by"] == targets[:3]


def test_no_contested_marker_without_contradicts(monkeypatch):
    # A non-CONTRADICTS edge (SUPPORTS) surfaces the node but adds no marker.
    graph = {
        CS: _FakeRev(CS, {"title": "S", "summary": "X"},
                     [_FakeEdge(CS, CT, "SUPPORTS")]),
        CT: _FakeRev(CT, {"title": "T", "summary": "also X"}, []),
    }
    base = [{"kref": CS, "title": "S", "summary": "X", "score": 0.5}]
    _found, augmented = _traverse(monkeypatch, graph, [CS], base)

    assert "contested_by" not in base[0]
    surfaced = [m for m in augmented if m["kref"] == CT]
    assert surfaced and "contested_by" not in surfaced[0]


def test_contested_marker_matches_sibling_revision_kref(monkeypatch):
    # The CONTRADICTS edge hangs off a sibling revision; the item-level result
    # entry is still marked (the marker matches own kref OR any sibling kref).
    sib = "kref://p/notes/cs.conversation?r=2"
    graph = {
        sib: _FakeRev(sib, {"title": "S", "summary": "X"},
                      [_FakeEdge(sib, CT, "CONTRADICTS", {"basis": "agent"})]),
        CT: _FakeRev(CT, {"title": "T", "summary": "not X"}, []),
    }
    base = [{
        "kref": CS, "title": "S", "summary": "X", "score": 0.5,
        "sibling_revisions": [{"kref": sib, "title": "S", "summary": "X"}],
    }]
    _traverse(monkeypatch, graph, [sib], base, seen={CS, sib})

    assert base[0]["contested_by"] == [CT]


# ---------------------------------------------------------------------------
# Grounding-staleness recall marker (#95): a surfaced dependent whose grounding
# fact was superseded carries an additive grounding_stale flag from metadata the
# walk already fetches — mirrors the contested_by marker (zero extra round-trip).
# ---------------------------------------------------------------------------

GF = "kref://p/facts/gf.fact?r=1"           # a superseded fact (the seed)
GD = "kref://p/decisions/gd.decision?r=1"    # decision grounded in GF (DEPENDS_ON)
GNEW = "kref://p/facts/gnew.fact?r=1"        # the fact that superseded GF


def test_grounding_stale_dependent_gets_recall_marker(monkeypatch):
    # From the superseded fact seed, the walk hops the incoming DEPENDS_ON to the
    # dependent decision and reads its grounding_stale metadata onto the entry.
    graph = {
        GF: _FakeRev(GF, {"title": "F", "summary": "old belief"},
                     [_FakeEdge(GD, GF, "DEPENDS_ON")]),
        GD: _FakeRev(GD, {"title": "D", "summary": "grounded decision",
                          "grounding_stale": "true",
                          "grounding_stale_superseded_by": GNEW}, []),
    }
    base = [{"kref": GF, "title": "F", "summary": "old belief", "score": 0.5}]
    _found, augmented = _traverse(monkeypatch, graph, [GF], base)

    surfaced = [m for m in augmented if m["kref"] == GD]
    assert len(surfaced) == 1
    assert surfaced[0]["edge_type"] == "DEPENDS_ON"
    assert surfaced[0]["grounding_stale"] is True
    assert surfaced[0]["superseded_by"] == GNEW


def test_dependent_without_stale_flag_gets_no_marker(monkeypatch):
    # An intact dependent (no grounding_stale metadata) surfaces unmarked.
    graph = {
        GF: _FakeRev(GF, {"title": "F", "summary": "belief"},
                     [_FakeEdge(GD, GF, "DEPENDS_ON")]),
        GD: _FakeRev(GD, {"title": "D", "summary": "grounded decision"}, []),
    }
    base = [{"kref": GF, "title": "F", "summary": "belief", "score": 0.5}]
    _found, augmented = _traverse(monkeypatch, graph, [GF], base)

    surfaced = [m for m in augmented if m["kref"] == GD]
    assert surfaced and "grounding_stale" not in surfaced[0]


# ---------------------------------------------------------------------------
# #97 — event-driven traversal completion (no 0.5s poll floor)
# ---------------------------------------------------------------------------

def _run_traverse_with_delay(monkeypatch, *, worker_delay, timeout):
    """Drive ``_traverse_edges`` over a trivial one-seed graph whose
    ``get_revision`` blocks for ``worker_delay`` s inside the daemon worker.

    Returns ``(found, elapsed_seconds, augmented)``.
    """
    import time as _time

    seed = "kref://p/notes/seed.conversation?r=1"
    graph = {seed: _FakeRev(seed, {"title": "S", "summary": "s"}, [])}

    def _delay(_kref):
        _time.sleep(worker_delay)

    _install_graph(monkeypatch, graph, on_get=_delay)
    gr = GraphAugmentedRecall(
        config=GraphAugmentationConfig(traversal_timeout=timeout),
    )
    augmented = []
    start = _time.monotonic()
    found = asyncio.run(gr._traverse_edges([seed], set(), augmented))
    elapsed = _time.monotonic() - start
    return found, elapsed, augmented


def test_traversal_wait_wakes_promptly_when_worker_finishes_fast(monkeypatch):
    # Worker finishes in ~50ms; the event-driven wait must return well under the
    # old 0.5s poll cadence, which floored every recall at ~0.5s here regardless
    # of how fast the traversal actually was.
    found, elapsed, _ = _run_traverse_with_delay(
        monkeypatch, worker_delay=0.05, timeout=30,
    )
    assert found == 0  # seed has no matching edges -> nothing surfaced
    assert elapsed < 0.45, (
        f"event-driven wait took {elapsed:.3f}s — should be far below the "
        f"old 0.5s poll floor"
    )


def test_traversal_times_out_and_returns_empty(monkeypatch):
    # Worker blocks longer than the timeout: current semantics preserved —
    # return 0 (empty), leave `augmented` untouched, and give up at ~timeout
    # rather than waiting out the full worker duration.
    found, elapsed, augmented = _run_traverse_with_delay(
        monkeypatch, worker_delay=0.4, timeout=0.1,
    )
    assert found == 0
    assert augmented == []
    assert elapsed < 0.35, (
        f"timeout wait took {elapsed:.3f}s — should be ~= the 0.1s timeout"
    )


# ---------------------------------------------------------------------------
# #105 — deterministic belief-safety-first traversal contract
#
# Same graph, edges delivered in every (sampled) arrival order ⇒ byte-identical
# traversal results AND markers. Belief edges (CONTRADICTS/SUPERSEDES) are
# processed before positive edges within each budget window, so a contradiction
# can never be crowded out of a fan-out / result / sibling cap by luck. Mirrors
# the #90 sibling-cap regression-test pattern, generalized to the whole reader.
# ---------------------------------------------------------------------------

def _sample_perms(seq, cap=24):
    """All permutations of *seq* (≤ ``cap``), else a deterministic subsample
    that always includes the identity and reversed orders."""
    perms = list(itertools.permutations(seq))
    if len(perms) <= cap:
        return perms
    step = max(1, len(perms) // cap)
    sub = perms[::step][:cap]
    if perms[-1] not in sub:      # keep a fully-reversed arrival order too
        sub[-1] = perms[-1]
    return sub


# --- (a) generic walk: belief edges surface first, stable within a class -----

PW_S = "kref://p/notes/pw-seed.conversation?r=1"
PW_CT = "kref://p/facts/pw-a.fact?r=1"     # CONTRADICTS (dispute basis)
PW_SUP = "kref://p/facts/pw-b.fact?r=1"    # SUPERSEDES
PW_REF = "kref://p/facts/pw-c.fact?r=1"    # REFERENCED (positive)
PW_SPT = "kref://p/facts/pw-d.fact?r=1"    # SUPPORTS (positive)


def test_generic_walk_is_arrival_order_invariant(monkeypatch):
    # The seed's four whitelisted edges, delivered in all 24 arrival orders,
    # must yield an identical surfaced sequence and identical contested marker.
    base_edges = [
        _FakeEdge(PW_S, PW_SUP, "SUPERSEDES"),
        _FakeEdge(PW_S, PW_REF, "REFERENCED"),
        _FakeEdge(PW_S, PW_SPT, "SUPPORTS"),
        _FakeEdge(PW_S, PW_CT, "CONTRADICTS", {"basis": "agent"}),
    ]
    # Belief class first (CONTRADICTS < SUPERSEDES by edge_type), then the
    # positives (REFERENCED < SUPPORTS), each tiebroken by (edge_type, uri).
    expected_seq = [
        (PW_CT, "CONTRADICTS"), (PW_SUP, "SUPERSEDES"),
        (PW_REF, "REFERENCED"), (PW_SPT, "SUPPORTS"),
    ]
    signatures = set()
    for perm in _sample_perms(base_edges):
        graph = {PW_S: _FakeRev(PW_S, {"title": "S", "summary": "seed"},
                                list(perm))}
        for k in (PW_CT, PW_SUP, PW_REF, PW_SPT):
            graph[k] = _FakeRev(k, {"title": k, "summary": ""}, [])
        base = [{"kref": PW_S, "title": "S", "summary": "seed", "score": 0.5}]
        _found, augmented = _traverse(monkeypatch, graph, [PW_S], base)
        seq = tuple((m["kref"], m["edge_type"])
                    for m in augmented if m.get("graph_augmented"))
        signatures.add((seq, tuple(base[0].get("contested_by", ()))))

    assert len(signatures) == 1, "arrival order changed the traversal result"
    (seq, contested), = signatures
    assert seq == tuple(expected_seq)
    assert contested == (PW_CT,)   # marker recorded regardless of arrival order


# --- (b) entity-neighbor sibling cap: which siblings survive is deterministic -

EN_M1 = "kref://p/notes/en-m1.conversation?r=1"   # seed
EN_A = "kref://p/entities/en-a.entity?r=1"
EN_S1 = "kref://p/notes/en-s1.conversation?r=1"
EN_S2 = "kref://p/notes/en-s2.conversation?r=1"
EN_S3 = "kref://p/notes/en-s3.conversation?r=1"


def test_entity_sibling_cap_is_arrival_order_invariant(monkeypatch):
    # A has three sibling ABOUT edges but max_siblings=2. In every arrival order
    # the SAME two siblings survive (lexicographically smallest source uris) —
    # arrival order no longer decides who gets starved (the #90 class of bug).
    anchor_incoming = [
        _FakeEdge(EN_M1, EN_A, "ABOUT"),   # the seed itself (already seen)
        _FakeEdge(EN_S1, EN_A, "ABOUT"),
        _FakeEdge(EN_S2, EN_A, "ABOUT"),
        _FakeEdge(EN_S3, EN_A, "ABOUT"),
    ]
    signatures = set()
    for perm in _sample_perms(anchor_incoming):
        graph = {
            EN_M1: _FakeRev(EN_M1, {"title": "M1"},
                            [_FakeEdge(EN_M1, EN_A, "ABOUT")]),
            EN_A: _FakeRev(EN_A, {"display_name": "A"}, list(perm)),
            EN_S1: _FakeRev(EN_S1, {"title": "S1", "summary": "s1"}, []),
            EN_S2: _FakeRev(EN_S2, {"title": "S2", "summary": "s2"}, []),
            EN_S3: _FakeRev(EN_S3, {"title": "S3", "summary": "s3"}, []),
        }
        _install_graph(monkeypatch, graph)
        gr = GraphAugmentedRecall(config=GraphAugmentationConfig(
            entity_recall=True, entity_recall_max_siblings=2,
        ))
        augmented = []
        asyncio.run(gr._traverse_entity_neighbors(
            [EN_M1], {EN_M1}, augmented, query="",
        ))
        signatures.add(tuple((m["kref"], m["edge_type"]) for m in augmented))

    assert len(signatures) == 1, "arrival order changed which siblings survived"
    (seq,) = signatures
    assert seq == ((EN_S1, "ABOUT"), (EN_S2, "ABOUT"))


# --- (b') entity-bridge join: which connecting nodes surface is deterministic -

BJ_M1 = "kref://p/notes/bj-m1.conversation?r=1"
BJ_M2 = "kref://p/notes/bj-m2.conversation?r=1"
BJ_X = "kref://p/entities/bj-x.entity?r=1"
BJ_F1 = "kref://p/facts/bj-f1.fact?r=1"
BJ_F2 = "kref://p/facts/bj-f2.fact?r=1"
BJ_F3 = "kref://p/facts/bj-f3.fact?r=1"


def test_entity_bridge_candidates_are_arrival_order_invariant(monkeypatch):
    # X bridges two angles; it has three incoming fact candidates but only two
    # surface (<=2 nodes/bridge). In every arrival order the SAME two facts win
    # (lexicographically smallest uris), not whichever the server listed first.
    x_incoming = [
        _FakeEdge(BJ_M1, BJ_X, "ABOUT"),   # the two angle hits (already seen)
        _FakeEdge(BJ_M2, BJ_X, "ABOUT"),
        _FakeEdge(BJ_F1, BJ_X, "ABOUT"),
        _FakeEdge(BJ_F2, BJ_X, "ABOUT"),
        _FakeEdge(BJ_F3, BJ_X, "ABOUT"),
    ]
    signatures = set()
    for perm in _sample_perms(x_incoming):
        graph = {
            BJ_M1: _FakeRev(BJ_M1, {"title": "M1"},
                            [_FakeEdge(BJ_M1, BJ_X, "ABOUT")]),
            BJ_M2: _FakeRev(BJ_M2, {"title": "M2"},
                            [_FakeEdge(BJ_M2, BJ_X, "ABOUT")]),
            BJ_X: _FakeRev(BJ_X, {"display_name": "X"}, list(perm)),
            BJ_F1: _FakeRev(BJ_F1, {"title": "F1", "summary": "f1"}, []),
            BJ_F2: _FakeRev(BJ_F2, {"title": "F2", "summary": "f2"}, []),
            BJ_F3: _FakeRev(BJ_F3, {"title": "F3", "summary": "f3"}, []),
        }
        _install_graph(monkeypatch, graph)
        gr = GraphAugmentedRecall(
            config=GraphAugmentationConfig(entity_recall=True),
        )
        augmented = []
        angle_hits = [[(BJ_M1, 0.9)], [(BJ_M2, 0.8)]]
        asyncio.run(gr._entity_bridge_join(
            angle_hits, {BJ_M1, BJ_M2}, augmented,
        ))
        signatures.add(tuple(m["kref"] for m in augmented))

    assert len(signatures) == 1, "arrival order changed the surfaced bridges"
    (seq,) = signatures
    assert seq == (BJ_F1, BJ_F2)


# --- (c) tight budget: belief edges always survive the cap first -------------

TB_S = "kref://p/notes/tb-seed.conversation?r=1"
TB_CT = "kref://p/facts/tb-ct.fact?r=1"    # CONTRADICTS (dispute basis)
TB_SUP = "kref://p/facts/tb-sup.fact?r=1"   # SUPERSEDES
TB_P1 = "kref://p/facts/tb-p1.fact?r=1"     # SUPPORTS (positive)
TB_P2 = "kref://p/facts/tb-p2.fact?r=1"     # REFERENCED (positive)


def test_tight_budget_never_starves_belief_edges(monkeypatch):
    # base_limit=1 ⇒ cap=3 ⇒ one base seed + room for only TWO graph entries.
    # The seed has two belief + two positive edges. Under arrival order the
    # survivors used to be luck; now the two belief edges ALWAYS take the room
    # and the positives are the ones crowded out — belief-safety first.
    seed_edges = [
        _FakeEdge(TB_S, TB_CT, "CONTRADICTS", {"basis": "agent"}),
        _FakeEdge(TB_S, TB_SUP, "SUPERSEDES"),
        _FakeEdge(TB_S, TB_P1, "SUPPORTS"),
        _FakeEdge(TB_S, TB_P2, "REFERENCED"),
    ]

    async def _recall_fn(q, *, limit, space_paths=None, memory_types=None):
        return [{"kref": TB_S, "title": "S", "summary": "seed", "score": 0.9}]

    survivors_seen = set()
    for perm in _sample_perms(seed_edges):
        graph = {TB_S: _FakeRev(TB_S, {"title": "S", "summary": "seed"},
                                list(perm))}
        for k, s in ((TB_CT, "ct"), (TB_SUP, "sup"), (TB_P1, "p1"),
                     (TB_P2, "p2")):
            graph[k] = _FakeRev(k, {"title": k, "summary": s}, [])
        _install_graph(monkeypatch, graph)
        gr = GraphAugmentedRecall(
            recall_fn=_recall_fn, config=GraphAugmentationConfig(),
        )
        result = asyncio.run(gr.recall("q", limit=1))
        survivors = frozenset(
            m["kref"] for m in result if m.get("graph_augmented")
        )
        survivors_seen.add(survivors)

    assert survivors_seen == {frozenset({TB_CT, TB_SUP})}, (
        "a tight budget let arrival order decide which edges survived"
    )


# --- (F4) cap truncation × marker identity: jointly pinned ------------------

MK_S = "kref://p/notes/mk-seed.conversation?r=1"
MK_CT = "kref://p/facts/mk-ct.fact?r=1"        # CONTRADICTS (dispute basis)
MK_SUP = "kref://p/facts/mk-sup.fact?r=1"       # SUPERSEDES
MK_DEP = "kref://p/decisions/mk-dep.decision?r=1"  # DEPENDS_ON, grounding-stale
MK_NEW = "kref://p/facts/mk-new.fact?r=1"       # what superseded DEP's ground
MK_R = "kref://p/facts/mk-r.fact?r=1"           # REFERENCED (truncated)
MK_P = "kref://p/facts/mk-p.fact?r=1"           # SUPPORTS (truncated)


def test_cap_truncation_and_markers_are_jointly_order_invariant(monkeypatch):
    # The spec's marker_completeness clause, pinned WITH truncation: run the
    # full recall path under a cap that cuts two positive edges, across permuted
    # edge arrival orders, and assert the surviving entries AND both marker
    # kinds (contested_by, grounding_stale/superseded_by) are byte-identical.
    seed_edges = [
        _FakeEdge(MK_S, MK_CT, "CONTRADICTS", {"basis": "agent"}),
        _FakeEdge(MK_S, MK_SUP, "SUPERSEDES"),
        _FakeEdge(MK_DEP, MK_S, "DEPENDS_ON"),   # incoming: dependent decision
        _FakeEdge(MK_S, MK_R, "REFERENCED"),
        _FakeEdge(MK_S, MK_P, "SUPPORTS"),
    ]

    async def _recall_fn(q, *, limit, space_paths=None, memory_types=None):
        # Fresh dict per call: markers are stamped onto result entries, so a
        # shared dict would leak contested_by between permutation runs.
        return [{"kref": MK_S, "title": "S", "summary": "seed", "score": 0.9}]

    signatures = set()
    for perm in _sample_perms(seed_edges):
        graph = {MK_S: _FakeRev(MK_S, {"title": "S", "summary": "seed"},
                                list(perm))}
        for k in (MK_CT, MK_SUP, MK_R, MK_P):
            graph[k] = _FakeRev(k, {"title": k, "summary": "x"}, [])
        graph[MK_DEP] = _FakeRev(MK_DEP, {
            "title": "D", "summary": "grounded decision",
            "grounding_stale": "true",
            "grounding_stale_superseded_by": MK_NEW,
        }, [])
        _install_graph(monkeypatch, graph)
        gr = GraphAugmentedRecall(
            recall_fn=_recall_fn, config=GraphAugmentationConfig(),
        )
        result = asyncio.run(gr.recall("q", limit=1, max_total=4))
        signatures.add(tuple(
            (m["kref"], m.get("edge_type", ""),
             tuple(m.get("contested_by", ())),
             m.get("grounding_stale"), m.get("superseded_by", ""))
            for m in result
        ))

    assert len(signatures) == 1, (
        "cap truncation or markers varied with edge arrival order"
    )
    (sig,) = signatures
    # Survivors: base seed + belief edges + the smallest positive (DEPENDS_ON);
    # REFERENCED and SUPPORTS are the ones deterministically truncated.
    assert [e[0] for e in sig] == [MK_S, MK_CT, MK_SUP, MK_DEP]
    by_kref = {e[0]: e for e in sig}
    # Contested marker: recorded from the FULL list, present on both endpoints.
    assert by_kref[MK_S][2] == (MK_CT,)
    assert by_kref[MK_CT][2] == (MK_S,)
    # Grounding-staleness marker rides the surviving dependent entry.
    assert by_kref[MK_DEP][3] is True
    assert by_kref[MK_DEP][4] == MK_NEW


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
