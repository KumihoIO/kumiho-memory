# -*- coding: utf-8 -*-
"""Keyless graph maintenance (kumiho_memory.graph_maintenance).

In-memory graph fakes (no server) assert the deterministic maintenance layer:
evidence re-grade, cross-graph bridge, entity merge, fact dedup, decision
dedup, orphan prune, dry-run safety, idempotency, and the deprecation cap.
The live end-to-end proof is scripts/dogfood_dream_maintenance.py.
"""
import asyncio
import types

from kumiho._text import slugify
from kumiho_memory.dream_state import DreamState
from kumiho_memory.graph_maintenance import GraphMaintainer, MaintenanceStats, _node_slug

_OUTGOING, _INCOMING, _BOTH = 0, 1, 2


# ---------------------------------------------------------------------------
# In-memory graph fakes
# ---------------------------------------------------------------------------


class FakeKref:
    def __init__(self, uri):
        self.uri = uri


class FakeEdge:
    def __init__(self, source, target, edge_type, metadata):
        self.source_kref = FakeKref(source)
        self.target_kref = FakeKref(target)
        self.edge_type = edge_type
        self.metadata = dict(metadata or {})


class FakeRev:
    def __init__(self, graph, item, uri, metadata):
        self._graph = graph
        self.item = item
        self.kref = FakeKref(uri)
        self.metadata = dict(metadata)
        self.tags = []

    def get_edges(self, edge_type_filter=None, direction=0):
        out = []
        for e in self._graph.edges:
            if edge_type_filter and e["type"] != edge_type_filter:
                continue
            hit = (
                (direction == _OUTGOING and e["source"] == self.kref.uri)
                or (direction == _INCOMING and e["target"] == self.kref.uri)
                or (direction == _BOTH and self.kref.uri in (e["source"], e["target"]))
            )
            if hit:
                out.append(FakeEdge(e["source"], e["target"], e["type"], e["metadata"]))
        return out

    def create_edge(self, target_rev, edge_type, metadata=None):
        self._graph.edges.append({
            "source": self.kref.uri,
            "target": target_rev.kref.uri,
            "type": edge_type,
            "metadata": dict(metadata or {}),
        })

    def set_attribute(self, key, value):
        self.metadata[key] = value

    def get_item(self):
        return self.item


class FakeItem:
    def __init__(self, graph, project, kind, slug, metadata):
        self.project = project
        self.kind = kind
        self.slug = slug
        self.deprecated = False
        uri = f"kref://{project}/{kind}s/{slug}.{kind}"
        self.kref = FakeKref(uri)
        self._rev = FakeRev(graph, self, uri + "?r=1", metadata)

    def get_latest_revision(self):
        return self._rev

    def set_deprecated(self, status):
        self.deprecated = status


class FakeClient:
    def __init__(self, graph):
        self._graph = graph
        self.metadata_updates = []
        self.tags = []
        self.untags = []

    def update_revision_metadata(self, kref, metadata):
        rev = self._graph.rev_by_uri(kref.uri)
        if rev is not None:
            rev.metadata.update(metadata)
        self.metadata_updates.append((kref.uri, dict(metadata)))
        return rev

    def tag_revision(self, kref, tag):
        rev = self._graph.rev_by_uri(kref.uri)
        if rev is not None and tag not in rev.tags:
            rev.tags.append(tag)
        self.tags.append((kref.uri, tag))

    def untag_revision(self, kref, tag):
        rev = self._graph.rev_by_uri(kref.uri)
        if rev is not None and tag in rev.tags:
            rev.tags.remove(tag)
        self.untags.append((kref.uri, tag))

    def has_tag(self, kref, tag):
        rev = self._graph.rev_by_uri(kref.uri)
        return rev is not None and tag in rev.tags

    def get_items(self, parent_path="", kind_filter="", page_size=None,
                  cursor=None, include_deprecated=False):
        return []  # no conversation items → DreamState.run hits no-revisions


class FakeGraph:
    def __init__(self):
        self.items = []
        self.edges = []
        self.client = FakeClient(self)

    # --- builders ---
    def add(self, project, kind, slug, metadata):
        item = FakeItem(self, project, kind, slug, metadata)
        self.items.append(item)
        return item

    def entity(self, project, name, aliases=None, entity_type=""):
        md = {"display_name": name, "entity_type": entity_type}
        if aliases:
            md["aliases"] = ", ".join(aliases)
        return self.add(project, "entity", slugify(name, hash_on_truncate=True), md)

    def fact(self, project, claim):
        return self.add(project, "fact", slugify(claim, hash_on_truncate=True),
                        {"claim": claim, "summary": claim, "title": claim[:80]})

    def decision(self, project, title, decision, evidence_level="unverified",
                 symbols="", status="active"):
        return self.add(project, "code_decision",
                        slugify(f"{title} 20260101", hash_on_truncate=True), {
                            "title": title, "decision": decision,
                            "evidence_level": evidence_level, "symbols": symbols,
                            "status": status,
                        })

    def evidence(self, project, statement, kind):
        return self.add(project, "code_evidence", slugify(statement, hash_on_truncate=True),
                        {"statement": statement, "evidence_kind": kind})

    def tag(self, item, *tags):
        """Attach graph tags to an item's latest revision (e.g. 'published',
        'evidence:official')."""
        rev = item.get_latest_revision()
        for t in tags:
            if t not in rev.tags:
                rev.tags.append(t)
        return item

    def tags_of(self, item):
        return list(item.get_latest_revision().tags)

    def link(self, src_item, tgt_item, edge_type, metadata=None):
        self.edges.append({
            "source": src_item.get_latest_revision().kref.uri,
            "target": tgt_item.get_latest_revision().kref.uri,
            "type": edge_type,
            "metadata": dict(metadata or {}),
        })

    # --- queries ---
    def rev_by_uri(self, uri):
        for it in self.items:
            if it.get_latest_revision().kref.uri == uri:
                return it.get_latest_revision()
        return None

    def item_search(self, context_filter="", name_filter="", kind_filter=""):
        return [
            it for it in self.items
            if it.project == context_filter
            and it.kind == kind_filter
            and not it.deprecated
        ]

    def sdk(self):
        mod = types.ModuleType("kumiho_fake")
        mod.item_search = self.item_search
        mod.get_revision = self.rev_by_uri
        mod.get_client = lambda: self.client
        return mod

    # --- assertion helpers ---
    def live(self, kind, project):
        return [it for it in self.items if it.kind == kind and it.project == project
                and not it.deprecated]

    def has_edge(self, src_rev_uri, tgt_rev_uri, edge_type):
        return any(
            e["source"] == src_rev_uri and e["target"] == tgt_rev_uri and e["type"] == edge_type
            for e in self.edges
        )


def _maintainer(graph, project="Mem", code_project="Mem-code", **kw):
    return GraphMaintainer(graph.sdk(), project=project, code_project=code_project, **kw)


# ---------------------------------------------------------------------------
# (B) Evidence re-grade — the headline
# ---------------------------------------------------------------------------


def test_evidence_regrade_lifts_unverified_to_corroborated():
    """A decision stamped 'unverified' at capture that later gained a
    measurement atom is re-graded to 'corroborated' from its CURRENT atoms."""
    g = FakeGraph()
    dec = g.decision("Mem-code", "Use bge-m3", "switch embedding backend",
                     evidence_level="unverified")
    meas = g.evidence("Mem-code", "recall +6pts on LoCoMo", "measurement")
    g.link(dec, meas, "MOTIVATED_BY")

    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)

    assert stats.decisions_regraded == 1
    assert dec.get_latest_revision().metadata["evidence_level"] == "corroborated"
    assert ("evidence:corroborated" in
            [t for _, t in g.client.tags])


def test_evidence_regrade_idempotent_and_never_downgrades():
    """Already-corroborated stays put (idempotent); a decision whose atoms
    only justify single_source is never downgraded from corroborated."""
    g = FakeGraph()
    dec = g.decision("Mem-code", "Keep cosine sibling", "hybrid ranking",
                     evidence_level="corroborated")
    # only a constraint atom now (would grade single_source) — must NOT lower
    g.link(dec, g.evidence("Mem-code", "official reserved for operators", "constraint"),
           "MOTIVATED_BY")

    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)

    assert stats.decisions_regraded == 0
    assert dec.get_latest_revision().metadata["evidence_level"] == "corroborated"


def test_evidence_regrade_never_touches_official():
    g = FakeGraph()
    dec = g.decision("Mem-code", "Pin schema", "operator decision",
                     evidence_level="official")
    g.link(dec, g.evidence("Mem-code", "benchmark win", "benchmark"), "MOTIVATED_BY")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.decisions_regraded == 0
    assert dec.get_latest_revision().metadata["evidence_level"] == "official"


# ---------------------------------------------------------------------------
# (C) Cross-graph bridge
# ---------------------------------------------------------------------------


def test_bridge_by_symbol_slug():
    """A code_decision whose `symbols` name an entity slug gets an ABOUT edge
    to that conversation entity (cross-project, deterministic)."""
    g = FakeGraph()
    ent = g.entity("Mem", "config_from_env", entity_type="convention")
    dec = g.decision("Mem-code", "Resolve toggles via env",
                     "feature toggles read from config_from_env",
                     symbols="config_from_env,resolve_project_name")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.bridges_created == 1
    assert g.has_edge(dec.get_latest_revision().kref.uri,
                      ent.get_latest_revision().kref.uri, "ABOUT")


def test_bridge_by_name_mention():
    """No symbol match, but the entity's name appears in the decision text."""
    g = FakeGraph()
    ent = g.entity("Mem", "Decision Memory")
    dec = g.decision("Mem-code", "Isolate code nodes",
                     "Decision Memory lives in a separate project", symbols="")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.bridges_created == 1


def test_bridge_skips_when_no_entity_match():
    g = FakeGraph()
    g.entity("Mem", "Postgres")
    g.decision("Mem-code", "Add retry", "wrap RPCs in backoff", symbols="_retry")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.bridges_created == 0


# ---------------------------------------------------------------------------
# (A) Entity merge — alias-driven
# ---------------------------------------------------------------------------


def test_entity_alias_merge_reduces_nodes_and_repoints():
    """Hub listing a variant as an alias absorbs the variant entity; the
    variant's incoming fact edge is repointed onto the hub."""
    g = FakeGraph()
    hub = g.entity("Mem", "PostgreSQL", aliases=["Postgres"])
    dup = g.entity("Mem", "Postgres")
    fact = g.fact("Mem", "Postgres uses MVCC for isolation")
    g.link(fact, dup, "ABOUT", {"entity": dup.slug})

    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)

    assert stats.entities_merged == 1
    assert dup.deprecated is True
    assert hub.deprecated is False
    # fact now ABOUT the hub
    assert g.has_edge(fact.get_latest_revision().kref.uri,
                      hub.get_latest_revision().kref.uri, "ABOUT")
    # measurable reduction: 1 live entity left
    assert len(g.live("entity", "Mem")) == 1


def test_entity_merge_folds_aliases():
    g = FakeGraph()
    hub = g.entity("Mem", "Kubernetes", aliases=["k8s"])
    g.entity("Mem", "k8s")
    _maintainer(g).run_keyless(MaintenanceStats())
    folded = hub.get_latest_revision().metadata.get("aliases", "")
    assert "k8s" in folded


# ---------------------------------------------------------------------------
# (A) Fact dedup — per shared entity
# ---------------------------------------------------------------------------


def test_fact_dedup_collapses_near_duplicates():
    g = FakeGraph()
    ent = g.entity("Mem", "bge-m3")
    f1 = g.fact("Mem", "bge-m3 supports a context window of 8192 tokens")
    f2 = g.fact("Mem", "bge-m3 supports a context window of 8192 tokens now")
    g.link(f1, ent, "ABOUT", {"entity": ent.slug})
    g.link(f2, ent, "ABOUT", {"entity": ent.slug})

    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)

    assert stats.facts_merged == 1
    assert len(g.live("fact", "Mem")) == 1


def test_fact_dedup_keeps_distinct_facts():
    g = FakeGraph()
    ent = g.entity("Mem", "bge-m3")
    g.link(g.fact("Mem", "bge-m3 has an 8192 token context window"), ent, "ABOUT")
    g.link(g.fact("Mem", "bge-m3 costs about twelve dollars per million tokens"),
           ent, "ABOUT")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.facts_merged == 0
    assert len(g.live("fact", "Mem")) == 2


# ---------------------------------------------------------------------------
# (B) Decision dedup — non-destructive status demotion
# ---------------------------------------------------------------------------


def test_decision_dedup_sinks_twin_via_supersedes():
    g = FakeGraph()
    keep = g.decision("Mem-code", "Adopt hybrid retrieval",
                      "combine BM25 and dense retrieval for recall",
                      evidence_level="corroborated")
    dupe = g.decision("Mem-code", "Adopt hybrid retrieval search",
                      "combine BM25 and dense retrieval for recall",
                      evidence_level="unverified")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)

    assert stats.decisions_deduped == 1
    # non-destructive: both items still live, loser demoted
    assert keep.deprecated is False and dupe.deprecated is False
    assert dupe.get_latest_revision().metadata["status"] == "superseded"
    assert g.has_edge(keep.get_latest_revision().kref.uri,
                      dupe.get_latest_revision().kref.uri, "SUPERSEDES")


# ---------------------------------------------------------------------------
# (A) Orphan prune
# ---------------------------------------------------------------------------


def test_orphan_prune_only_zero_edge_entities():
    g = FakeGraph()
    orphan = g.entity("Mem", "OrphanThing")
    linked = g.entity("Mem", "LinkedThing")
    g.link(g.fact("Mem", "LinkedThing matters for recall"), linked, "ABOUT")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.orphans_pruned == 1
    assert orphan.deprecated is True
    assert linked.deprecated is False


# ---------------------------------------------------------------------------
# Safety: dry_run, idempotency, cap
# ---------------------------------------------------------------------------


def test_dry_run_makes_no_mutations():
    g = FakeGraph()
    hub = g.entity("Mem", "PostgreSQL", aliases=["Postgres"])
    dup = g.entity("Mem", "Postgres")
    dec = g.decision("Mem-code", "Use bge-m3", "switch backend", evidence_level="unverified")
    g.link(dec, g.evidence("Mem-code", "measured +6", "measurement"), "MOTIVATED_BY")
    ent = g.entity("Mem", "config_from_env")
    g.decision("Mem-code", "env toggles", "read config_from_env", symbols="config_from_env")

    before_edges = len(g.edges)
    stats = MaintenanceStats()
    _maintainer(g, dry_run=True).run_keyless(stats)

    # counts computed...
    assert stats.entities_merged >= 1
    assert stats.decisions_regraded == 1
    assert stats.bridges_created >= 1
    # ...but nothing mutated
    assert dup.deprecated is False
    assert dec.get_latest_revision().metadata["evidence_level"] == "unverified"
    assert len(g.edges) == before_edges
    assert g.client.metadata_updates == []


def test_idempotent_rerun():
    g = FakeGraph()
    hub = g.entity("Mem", "PostgreSQL", aliases=["Postgres"])
    g.entity("Mem", "Postgres")
    dec = g.decision("Mem-code", "Use bge-m3", "switch backend", evidence_level="unverified")
    g.link(dec, g.evidence("Mem-code", "measured +6", "measurement"), "MOTIVATED_BY")
    ent = g.entity("Mem", "config_from_env")
    g.decision("Mem-code", "env toggles", "read config_from_env", symbols="config_from_env")

    m1 = _maintainer(g)
    s1 = MaintenanceStats()
    m1.run_keyless(s1)
    assert s1.entities_merged == 1 and s1.decisions_regraded == 1 and s1.bridges_created == 1

    # fresh maintainer (fresh budgets) over the mutated graph — no new work
    s2 = MaintenanceStats()
    _maintainer(g).run_keyless(s2)
    assert s2.entities_merged == 0
    assert s2.decisions_regraded == 0
    assert s2.bridges_created == 0


def test_deprecation_cap_limits_merges():
    """max_deprecation_ratio caps how many entities a run may deprecate."""
    g = FakeGraph()
    # 10 hubs each with a distinct alias-duplicate → 10 candidate merges
    for i in range(10):
        g.entity("Mem", f"Hub{i}", aliases=[f"Alias{i}"])
        g.entity("Mem", f"Alias{i}")
    stats = MaintenanceStats()
    _maintainer(g, max_deprecation_ratio=0.2).run_keyless(stats)
    # budget = int(20 live entities * 0.2) = 4
    assert stats.entities_merged == 4


# ---------------------------------------------------------------------------
# Optional-LLM entity merge applies through the keyless path
# ---------------------------------------------------------------------------


def test_apply_entity_merges_uses_keyless_path():
    g = FakeGraph()
    canon = g.entity("Mem", "PostgreSQL")
    dup = g.entity("Mem", "Postgres")
    g.link(g.fact("Mem", "Postgres is relational"), dup, "ABOUT")

    m = _maintainer(g)
    stats = MaintenanceStats()
    # LLM said: fold 'postgres' into 'postgresql'
    m.apply_entity_merges([(canon.slug, dup.slug)], stats)

    assert stats.llm_merges == 1
    assert dup.deprecated is True
    assert len(g.live("entity", "Mem")) == 1


def test_apply_entity_merges_drops_unknown_slugs():
    g = FakeGraph()
    canon = g.entity("Mem", "PostgreSQL")
    m = _maintainer(g)
    stats = MaintenanceStats()
    m.apply_entity_merges([(canon.slug, "does-not-exist"), ("ghost", "phantom")], stats)
    assert stats.llm_merges == 0
    assert canon.deprecated is False


# ---------------------------------------------------------------------------
# Code passes skipped when no code project
# ---------------------------------------------------------------------------


def test_no_code_project_skips_decision_passes():
    g = FakeGraph()
    g.decision("Mem-code", "orphan decision", "x", evidence_level="unverified")
    stats = MaintenanceStats()
    GraphMaintainer(g.sdk(), project="Mem", code_project=None).run_keyless(stats)
    assert stats.decisions_scanned == 0
    assert stats.decisions_regraded == 0


def test_node_slug_helper():
    assert _node_slug("kref://P/entities/config-from-env.entity?r=3", "entity") == "config-from-env"
    assert _node_slug("kref://P/decisions/x.code_decision", "code_decision") == "x"
    assert _node_slug("", "entity") == ""


# ---------------------------------------------------------------------------
# Adversarial-review regression tests
# ---------------------------------------------------------------------------


def test_published_entity_never_merged():
    """A `published` duplicate is operator-owned — never folded away."""
    g = FakeGraph()
    g.entity("Mem", "PostgreSQL", aliases=["Postgres"])
    dup = g.tag(g.entity("Mem", "Postgres"), "published")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.entities_merged == 0
    assert dup.deprecated is False
    # opt-in override still allows it
    g2 = FakeGraph()
    g2.entity("Mem", "PostgreSQL", aliases=["Postgres"])
    dup2 = g2.tag(g2.entity("Mem", "Postgres"), "published")
    _maintainer(g2, allow_published_deprecation=True).run_keyless(MaintenanceStats())
    assert dup2.deprecated is True


def test_published_fact_never_deduped():
    g = FakeGraph()
    ent = g.entity("Mem", "bge-m3")
    f1 = g.fact("Mem", "bge-m3 supports a context window of 8192 tokens")
    f2 = g.tag(g.fact("Mem", "bge-m3 supports a context window of 8192 tokens now"),
               "published")
    g.link(f1, ent, "ABOUT")
    g.link(f2, ent, "ABOUT")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    # f2 is published; if the keeper heuristic picks f2 as loser it must be
    # skipped. Neither fact should be deprecated when the loser is published.
    assert f2.deprecated is False


def test_published_orphan_never_pruned():
    g = FakeGraph()
    orphan = g.tag(g.entity("Mem", "Sacred"), "published")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.orphans_pruned == 0
    assert orphan.deprecated is False


def test_published_decision_never_sunk():
    g = FakeGraph()
    g.decision("Mem-code", "Adopt hybrid retrieval",
               "combine BM25 and dense retrieval", evidence_level="corroborated")
    loser = g.tag(g.decision("Mem-code", "Adopt hybrid retrieval search",
                             "combine BM25 and dense retrieval",
                             evidence_level="unverified"), "published")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    # published loser must keep active status
    assert loser.get_latest_revision().metadata["status"] == "active"


def test_regrade_respects_official_tag_without_metadata():
    """Grade set via the evidence:official TAG alone (metadata unset) must
    still block a lift — parse_evidence reads both carriers."""
    g = FakeGraph()
    dec = g.decision("Mem-code", "Pin schema", "operator call", evidence_level="")
    g.tag(dec, "evidence:official")
    g.link(dec, g.evidence("Mem-code", "benchmark win", "benchmark"), "MOTIVATED_BY")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.decisions_regraded == 0
    assert dec.get_latest_revision().metadata.get("evidence_level", "") == ""


def test_regrade_never_downgrades_tag_only_grade():
    """A tag-only 'corroborated' whose atoms now grade single_source must not
    be lowered."""
    g = FakeGraph()
    dec = g.decision("Mem-code", "Keep cosine", "hybrid ranking", evidence_level="")
    g.tag(dec, "evidence:corroborated")
    g.link(dec, g.evidence("Mem-code", "a stated constraint", "constraint"), "MOTIVATED_BY")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.decisions_regraded == 0


def test_regrade_untags_stale_lower_grade_on_lift():
    g = FakeGraph()
    dec = g.decision("Mem-code", "Use bge-m3", "switch backend",
                     evidence_level="single_source")
    g.tag(dec, "evidence:single_source")
    g.link(dec, g.evidence("Mem-code", "measured +6", "measurement"), "MOTIVATED_BY")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.decisions_regraded == 1
    tags = g.tags_of(dec)
    assert "evidence:corroborated" in tags
    assert "evidence:single_source" not in tags  # stale tag removed


def test_decision_dedup_idempotent_across_regrade_flip():
    """Correctness-F1: once a decision is sunk, a later evidence lift on it
    must NOT flip the keeper and create a mutual SUPERSEDES cycle."""
    g = FakeGraph()
    a = g.decision("Mem-code", "Adopt hybrid retrieval",
                   "combine BM25 and dense retrieval for recall",
                   evidence_level="unverified")
    b = g.decision("Mem-code", "Adopt hybrid retrieval search",
                   "combine BM25 and dense retrieval for recall",
                   evidence_level="unverified")
    # Run 1: equal grade → slug tiebreak picks a keeper; the other is sunk.
    _maintainer(g).run_keyless(MaintenanceStats())
    sunk = [d for d in (a, b) if d.get_latest_revision().metadata["status"] == "superseded"]
    kept = [d for d in (a, b) if d.get_latest_revision().metadata["status"] == "active"]
    assert len(sunk) == 1 and len(kept) == 1
    loser = sunk[0]

    # The sunk loser now accrues a measurement atom → regrade lifts it.
    g.link(loser, g.evidence("Mem-code", "measured +7 on LoCoMo", "measurement"),
           "MOTIVATED_BY")
    s2 = MaintenanceStats()
    _maintainer(g).run_keyless(s2)

    # Run 2 must NOT sink the former keeper: no cycle, only one superseded.
    assert s2.decisions_deduped == 0
    still_superseded = [d for d in (a, b)
                        if d.get_latest_revision().metadata["status"] == "superseded"]
    assert len(still_superseded) == 1


def test_transitive_alias_chain_collapses_in_one_pass():
    """Correctness-F2: A→B→C alias chain folds fully in a single run."""
    g = FakeGraph()
    g.entity("Mem", "Alpha", aliases=["Beta"])
    g.entity("Mem", "Beta", aliases=["Gamma"])
    g.entity("Mem", "Gamma")
    stats = MaintenanceStats()
    # ratio 0.9 so the deprecation budget (2) admits both merges — isolates the
    # structural one-pass fix from the orthogonal budget cap.
    _maintainer(g, max_deprecation_ratio=0.9).run_keyless(stats)
    assert stats.entities_merged == 2  # both Beta and Gamma folded in one pass
    assert len(g.live("entity", "Mem")) == 1


def test_bridge_keeps_decision_referenced_orphan():
    """Correctness-F3: an otherwise-orphan entity a decision is about gets a
    bridge edge (prune runs last) and is NOT pruned."""
    g = FakeGraph()
    ent = g.entity("Mem", "widget_factory")   # zero conversation edges
    dec = g.decision("Mem-code", "Build widgets", "use the widget_factory helper",
                     symbols="widget_factory")
    stats = MaintenanceStats()
    _maintainer(g).run_keyless(stats)
    assert stats.bridges_created == 1
    assert stats.orphans_pruned == 0          # the bridge gave it an edge
    assert ent.deprecated is False


def test_llm_merge_budget_starvation_is_observable():
    """Integration-F3: a valid LLM pair dropped for budget is counted."""
    g = FakeGraph()
    # exhaust the entity budget with keyless alias merges, then feed an LLM pair
    g.entity("Mem", "Hub", aliases=["HubAlias"])
    g.entity("Mem", "HubAlias")
    canon = g.entity("Mem", "PostgreSQL")
    dup = g.entity("Mem", "Postgres")
    m = _maintainer(g, max_deprecation_ratio=0.1)  # budget = int(4*0.1)=1 wait -> max(1,..)=1
    stats = MaintenanceStats()
    m.run_keyless(stats)   # consumes the single budget unit on Hub/HubAlias
    m.apply_entity_merges([(canon.slug, dup.slug)], stats)
    assert stats.llm_merges == 0
    assert stats.llm_merges_skipped == 1
    assert dup.deprecated is False


# ---------------------------------------------------------------------------
# DreamState.run() wiring (issue #59 integration)
# ---------------------------------------------------------------------------


def _run_sdk(graph, cursor_uri="kref://Mem/_dream_state.conversation"):
    """A kumiho-module stand-in that drives DreamState.run() into the
    no-revisions branch while backing the typed-graph passes with *graph*."""
    class _Kref:
        def __init__(self, uri):
            self.uri = uri

    class _RevHandle:
        def __init__(self, uri):
            self.kref = _Kref(uri)

        def create_artifact(self, name, path):
            return None

    class _CursorItem:
        def __init__(self, uri):
            self.kref = _Kref(uri)
            self.revisions = []

        def create_revision(self, metadata=None):
            h = _RevHandle(f"{self.kref.uri}?r={len(self.revisions) + 1}")
            self.revisions.append(metadata or {})
            return h

    cursor = _CursorItem(cursor_uri)

    class _Project:
        name = "Mem"

        def get_spaces(self, **kw):
            return []  # no child spaces → only the project root is walked

    sdk = types.ModuleType("kumiho_fake_run")
    sdk.get_project = lambda name: _Project()
    sdk.get_client = lambda: graph.client
    sdk.get_item = lambda uri: cursor if uri == cursor_uri else None
    sdk.get_attribute = lambda kref, key: None
    sdk.set_attribute = lambda kref, key, value: None
    sdk.item_search = graph.item_search
    sdk.get_revision = graph.rev_by_uri
    sdk.Kref = _Kref
    return sdk, cursor


def test_run_surfaces_nonzero_maintenance_in_result_and_report(tmp_path):
    """Integration-F4: run() with a seeded graph carries NON-zero maintenance
    stats into result['maintenance'] and the report metadata."""
    g = FakeGraph()
    g.entity("Mem", "PostgreSQL", aliases=["Postgres"])
    g.entity("Mem", "Postgres")
    sdk, cursor = _run_sdk(g)

    ds = DreamState(project="Mem", summarizer=_stub_summarizer(),
                    maintain_graph=True, code_project=None,
                    artifact_root=str(tmp_path))
    ds._cursor_item_kref = cursor.kref.uri

    import sys
    saved = sys.modules.get("kumiho")
    sys.modules["kumiho"] = sdk
    try:
        report = asyncio.run(ds.run())
    finally:
        if saved is not None:
            sys.modules["kumiho"] = saved
        else:
            sys.modules.pop("kumiho", None)

    assert report["success"] is True
    assert report["events_processed"] == 0
    assert report["maintenance"]["entities_merged"] == 1
    # the report revision metadata mirrors the non-zero count
    assert cursor.revisions and cursor.revisions[0]["entities_merged"] == "1"
    assert cursor.revisions[0]["maintain_graph_active"] == "true"


def test_maintain_never_aborts_the_run(monkeypatch):
    """Integration-F1: a maintenance failure is recorded, not raised — it must
    not discard the flat pass's report/cursor."""
    import kumiho_memory.graph_maintenance as gm

    def boom(self, stats):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(gm.GraphMaintainer, "run_keyless", boom)
    ds = DreamState(summarizer=_stub_summarizer(), maintain_graph=True)
    result = asyncio.run(ds._maintain(types.SimpleNamespace()))
    assert any("kaboom" in e for e in result["errors"])


def _stub_summarizer():
    class _Adapter:
        async def chat(self, **kw):
            return '{"merges": []}'

    return types.SimpleNamespace(adapter=_Adapter(), model="stub")
