# -*- coding: utf-8 -*-
"""Embedding-assisted fact dedup (ontology gap G6, opt-in).

Server-side vector scoring (``score_revisions`` — keyless, no client embeddings,
no LLM) WIDENS candidate pairing beyond one entity's ABOUT fan-in. The merge
itself still routes through the SAME lexical-Jaccard confirmation + fact budget +
published protection as the keyless per-entity scan, so the unrecoverable-merge
safety threshold is never loosened. Reuses the in-memory graph fakes from
``test_graph_maintenance``.
"""
import asyncio

from kumiho_memory.graph_maintenance import (
    GraphMaintainer,
    MaintenanceStats,
    _EMBED_TOP_K,
)
from kumiho_memory.dream_state import DreamState

from test_graph_maintenance import FakeGraph


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _rev_uri(item):
    return item.get_latest_revision().kref.uri


def _vec(item, score):
    return {"kref": _rev_uri(item), "score": score, "score_method": "vector"}


def _sdk_with_scores(graph, score_map, *, record=None):
    """graph.sdk() augmented with a fake ``score_revisions`` driven by
    ``score_map`` (query statement -> list of scored dicts). ``record`` collects
    every kref the maintainer asks to score (for the kind-filter/bound asserts)."""
    mod = graph.sdk()

    def score_revisions(query, krefs, score_fields=None):
        if record is not None:
            record.extend(krefs)
        return score_map.get(query, [])

    mod.score_revisions = score_revisions
    return mod


def _maintainer(sdk, **kw):
    return GraphMaintainer(sdk, project="Mem", code_project=None, **kw)


# --------------------------------------------------------------------------- #
# candidate generation: bounds + kind filter                                  #
# --------------------------------------------------------------------------- #

def test_candidates_only_score_facts_and_dedupe_pairs():
    g = FakeGraph()
    # Non-fact nodes must never be scored (kind=fact filter).
    g.entity("Mem", "Alpha")
    g.entity("Mem", "Beta")
    f1 = g.fact("Mem", "The database uses connection pooling")
    f2 = g.fact("Mem", "Database uses connection pooling")   # near-dup of f1
    f3 = g.fact("Mem", "Kafka handles the event stream")     # unrelated

    scores = {
        "The database uses connection pooling": [_vec(f2, 0.92), _vec(f3, 0.10)],
        "Database uses connection pooling": [_vec(f1, 0.92)],
        "Kafka handles the event stream": [],
    }
    scored_krefs: list = []
    sdk = _sdk_with_scores(g, scores, record=scored_krefs)

    stats = MaintenanceStats()
    pairs = _maintainer(sdk).embedding_fact_candidates(stats)

    # One de-duplicated pair {f1, f2}; (f1,f3) below the score floor.
    assert len(pairs) == 1
    slugs = {pairs[0][0]["slug"], pairs[0][1]["slug"]}
    assert slugs == {f1.slug, f2.slug}
    assert stats.embed_fact_candidates == 1
    # Kind filter: every kref handed to score_revisions is a FACT revision.
    assert scored_krefs, "expected some scoring"
    assert all(".fact" in k for k in scored_krefs)
    assert not any(".entity" in k for k in scored_krefs)


def test_candidates_respect_small_k_per_query():
    g = FakeGraph()
    q = g.fact("Mem", "anchor fact zero")
    neighbours = [g.fact("Mem", f"distinct neighbour number {i}") for i in range(_EMBED_TOP_K + 3)]
    # All neighbours score high for the anchor's query (but are lexically
    # distinct, so they'd never merge — this isolates the k bound).
    scores = {"anchor fact zero": [_vec(n, 0.9) for n in neighbours]}
    sdk = _sdk_with_scores(g, scores)
    stats = MaintenanceStats()
    pairs = _maintainer(sdk).embedding_fact_candidates(stats)
    # Only the anchor's query yields hits; capped at k.
    assert len(pairs) == _EMBED_TOP_K


def test_fulltext_method_is_not_a_vector_nomination():
    g = FakeGraph()
    f1 = g.fact("Mem", "The database uses connection pooling")
    f2 = g.fact("Mem", "Database uses connection pooling")
    scores = {
        "The database uses connection pooling": [
            {"kref": _rev_uri(f2), "score": 0.99, "score_method": "fulltext"},
        ],
    }
    sdk = _sdk_with_scores(g, scores)
    stats = MaintenanceStats()
    assert _maintainer(sdk).embedding_fact_candidates(stats) == []
    assert stats.embed_fact_candidates == 0


def test_no_score_revisions_is_a_noop():
    g = FakeGraph()
    g.fact("Mem", "one fact")
    g.fact("Mem", "two fact")
    sdk = g.sdk()  # no score_revisions attribute
    stats = MaintenanceStats()
    m = _maintainer(sdk)
    assert m.embedding_fact_candidates(stats) == []
    m.apply_embedding_fact_dedup(stats)   # must not raise
    assert stats.embed_facts_merged == 0


# --------------------------------------------------------------------------- #
# the merge routes through the SAME verification layer (assert invoked)        #
# --------------------------------------------------------------------------- #

def test_vector_similar_but_lexically_distinct_is_NOT_merged(monkeypatch):
    """The safety threshold is not weakened: a paraphrase the embedding stage
    nominates but that fails the lexical Jaccard confirmation never merges — and
    the confirmation path is provably invoked."""
    g = FakeGraph()
    f_a = g.fact("Mem", "Alice manages the payroll system")
    f_b = g.fact("Mem", "Compensation processing is Alices responsibility")
    scores = {"Alice manages the payroll system": [_vec(f_b, 0.95)]}
    sdk = _sdk_with_scores(g, scores)

    calls = []
    orig = GraphMaintainer._confirm_and_collapse_fact

    def spy(self, a, b, stats, seen):
        calls.append(tuple(sorted((a["slug"], b["slug"]))))
        return orig(self, a, b, stats, seen)

    monkeypatch.setattr(GraphMaintainer, "_confirm_and_collapse_fact", spy)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    # verification layer invoked on the nominated pair...
    assert tuple(sorted((f_a.slug, f_b.slug))) in calls
    # ...and it refused the merge (Jaccard below _FACT_DEDUP_JACCARD).
    assert stats.embed_facts_merged == 0
    assert stats.facts_merged == 0
    assert not f_a.deprecated and not f_b.deprecated


def test_cross_entity_lexical_duplicate_is_merged():
    """The G6 win: two near-identical facts filed under DIFFERENT entities — the
    per-entity scan never pairs them — converge via the embedding nomination,
    then merge through the unchanged confirmation."""
    g = FakeGraph()
    e1 = g.entity("Mem", "ServiceA")
    e2 = g.entity("Mem", "ServiceB")
    f1 = g.fact("Mem", "The cache uses an LRU eviction policy")
    f2 = g.fact("Mem", "Cache uses an LRU eviction policy")
    g.link(f1, e1, "ABOUT")     # filed under different entities
    g.link(f2, e2, "ABOUT")
    scores = {"The cache uses an LRU eviction policy": [_vec(f2, 0.93)]}
    sdk = _sdk_with_scores(g, scores)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    assert stats.embed_facts_merged == 1
    assert stats.facts_merged == 1
    # exactly one of the pair was soft-deprecated (the loser)
    assert (f1.deprecated ^ f2.deprecated)


# --------------------------------------------------------------------------- #
# distinguishing-asymmetry guard (2026-07-19 live gate-check finding)          #
#                                                                              #
# High token-Jaccard overlap is not sufficient evidence of duplication when   #
# the sentences flip polarity (negation) or state a different named day —    #
# a real local-CE gate-check merged both before this guard existed.          #
# --------------------------------------------------------------------------- #

def test_asymmetric_negation_blocks_merge_english():
    """'happy' vs 'NOT happy' — near-identical tokens, opposite meaning."""
    g = FakeGraph()
    f_a = g.fact("Mem", "The user is happy with the results of the recent product launch")
    f_b = g.fact("Mem", "The user is not happy with the results of the recent product launch")
    scores = {f_a.get_latest_revision().metadata["claim"]: [_vec(f_b, 0.95)]}
    sdk = _sdk_with_scores(g, scores)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    assert stats.embed_facts_merged == 0
    assert not f_a.deprecated and not f_b.deprecated


def test_asymmetric_negation_blocks_merge_korean():
    """'satisfied' vs 'NOT satisfied' — Korean negation is a verb-ending
    change, not a standalone inserted word; the guard checks raw text so it
    still catches this even though the 2-char negation tokens are dropped by
    the length-filtered tokenizer before Jaccard ever sees them."""
    g = FakeGraph()
    f_a = g.fact("Mem", "사용자는 이번 프로젝트 진행 상황에 만족하고 있다")
    f_b = g.fact("Mem", "사용자는 이번 프로젝트 진행 상황에 만족하고 있지 않다")
    scores = {f_a.get_latest_revision().metadata["claim"]: [_vec(f_b, 0.95)]}
    sdk = _sdk_with_scores(g, scores)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    assert stats.embed_facts_merged == 0
    assert not f_a.deprecated and not f_b.deprecated


def test_differing_weekday_blocks_merge():
    """Same meeting, different day — near-identical tokens, different fact."""
    g = FakeGraph()
    f_a = g.fact("Mem", "The user's weekly team status meeting has been scheduled for 3pm this Friday in the main conference room")
    f_b = g.fact("Mem", "The user's weekly team status meeting has been scheduled for 3pm this Monday in the main conference room")
    scores = {f_a.get_latest_revision().metadata["claim"]: [_vec(f_b, 0.95)]}
    sdk = _sdk_with_scores(g, scores)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    assert stats.embed_facts_merged == 0
    assert not f_a.deprecated and not f_b.deprecated


def test_symmetric_negation_does_not_block_merge():
    """Both statements negated (same polarity) is not an asymmetry — a true
    near-verbatim restatement that happens to share a negation word must
    still merge; the guard is asymmetric-only, not a blanket negation ban."""
    g = FakeGraph()
    f_a = g.fact("Mem", "The user is not satisfied with the current pricing plan")
    f_b = g.fact("Mem", "User is not satisfied with the current pricing plan")
    scores = {f_a.get_latest_revision().metadata["claim"]: [_vec(f_b, 0.95)]}
    sdk = _sdk_with_scores(g, scores)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    assert stats.embed_facts_merged == 1
    assert (f_a.deprecated ^ f_b.deprecated)


def test_reordered_duplicate_without_markers_still_merges():
    """Regression pin: the guard must not interfere with an ordinary
    near-verbatim duplicate that contains neither negation nor weekday
    markers (the true-positive case from the live gate-check)."""
    g = FakeGraph()
    f_a = g.fact("Mem", "사용자는 아침 회의보다 저녁 회의를 더 선호한다")
    f_b = g.fact("Mem", "사용자는 저녁 회의를 아침 회의보다 더 선호한다")
    scores = {f_a.get_latest_revision().metadata["claim"]: [_vec(f_b, 0.95)]}
    sdk = _sdk_with_scores(g, scores)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    assert stats.embed_facts_merged == 1
    assert (f_a.deprecated ^ f_b.deprecated)


def test_keyless_per_entity_path_also_gets_the_guard():
    """The guard lives in the shared confirmation gate, so the ALWAYS-
    available keyless per-entity scan (run_keyless, no opt-in flag) is
    protected too, not just the embedding-assisted stage."""
    g = FakeGraph()
    e1 = g.entity("Mem", "Feature")
    f_a = g.fact("Mem", "The feature is enabled for all users")
    f_b = g.fact("Mem", "The feature is not enabled for all users")
    g.link(f_a, e1, "ABOUT")
    g.link(f_b, e1, "ABOUT")
    sdk = g.sdk()

    stats = MaintenanceStats()
    _maintainer(sdk).run_keyless(stats)

    assert stats.facts_merged == 0
    assert not f_a.deprecated and not f_b.deprecated


# --------------------------------------------------------------------------- #
# protection + caps carried over unchanged                                    #
# --------------------------------------------------------------------------- #

def test_published_loser_is_protected():
    g = FakeGraph()
    keeper = g.fact("Mem", "The queue drains in FIFO order")
    loser = g.fact("Mem", "Queue drains in FIFO order")
    # Give the keeper an extra edge so _pick_keeper deterministically keeps it,
    # and publish the loser so it must survive.
    g.link(keeper, g.entity("Mem", "Queue"), "ABOUT")
    g.tag(loser, "published")
    scores = {"The queue drains in FIFO order": [_vec(loser, 0.95)]}
    sdk = _sdk_with_scores(g, scores)

    stats = MaintenanceStats()
    _maintainer(sdk).apply_embedding_fact_dedup(stats)

    assert stats.embed_facts_merged == 0
    assert not loser.deprecated   # published node untouched


def test_fact_deprecation_budget_caps_embedding_merges():
    g = FakeGraph()
    # Two independent lexical-duplicate clusters.
    a1 = g.fact("Mem", "The alpha metric increased last quarter")
    a2 = g.fact("Mem", "Alpha metric increased last quarter")
    b1 = g.fact("Mem", "The beta rollout completed on schedule")
    b2 = g.fact("Mem", "Beta rollout completed on schedule")
    scores = {
        "The alpha metric increased last quarter": [_vec(a2, 0.95)],
        "The beta rollout completed on schedule": [_vec(b2, 0.95)],
    }
    sdk = _sdk_with_scores(g, scores)
    # 4 live facts * 0.25 -> budget 1: only one cluster may collapse.
    stats = MaintenanceStats()
    _maintainer(sdk, max_deprecation_ratio=0.25).apply_embedding_fact_dedup(stats)
    assert stats.embed_facts_merged == 1


def test_dry_run_counts_but_does_not_deprecate():
    g = FakeGraph()
    f1 = g.fact("Mem", "The token bucket refills every second")
    f2 = g.fact("Mem", "Token bucket refills every second")
    scores = {"The token bucket refills every second": [_vec(f2, 0.95)]}
    sdk = _sdk_with_scores(g, scores)
    stats = MaintenanceStats()
    _maintainer(sdk, dry_run=True).apply_embedding_fact_dedup(stats)
    assert stats.embed_facts_merged == 1
    assert not f1.deprecated and not f2.deprecated   # nothing mutated


# --------------------------------------------------------------------------- #
# DreamState wiring: the flag routes _maintain through the embedding pass      #
# --------------------------------------------------------------------------- #

def _dream(**kw):
    # No summarizer key needed: _maintain only touches the keyless maintainer.
    return DreamState(project="Mem", maintain_graph=True, **kw)


def test_dream_state_flag_on_invokes_embedding_pass(monkeypatch):
    g = FakeGraph()
    called = []
    monkeypatch.setattr(
        GraphMaintainer, "apply_embedding_fact_dedup",
        lambda self, stats: called.append(True),
    )
    ds = _dream(embed_fact_dedup=True)
    asyncio.run(ds._maintain(g.sdk()))
    assert called == [True]


def test_dream_state_flag_off_skips_embedding_pass(monkeypatch):
    g = FakeGraph()
    called = []
    monkeypatch.setattr(
        GraphMaintainer, "apply_embedding_fact_dedup",
        lambda self, stats: called.append(True),
    )
    ds = _dream(embed_fact_dedup=False)
    asyncio.run(ds._maintain(g.sdk()))
    assert called == []


def test_dream_state_embed_flag_reads_env(monkeypatch):
    monkeypatch.setenv("KUMIHO_DREAM_EMBED_FACT_DEDUP", "1")
    assert _dream().embed_fact_dedup is True
    monkeypatch.setenv("KUMIHO_DREAM_EMBED_FACT_DEDUP", "0")
    assert _dream().embed_fact_dedup is False
