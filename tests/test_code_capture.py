"""Tests for the Decision Memory capture adapter (P2).

Real ``git`` subprocess against a synthetic repo in ``tmp_path`` (the
pipeline shells out, so the tests do too); the LLM is a canned-JSON stub and
the kumiho SDK is faked at the module seam.  Covers: prefilter asymmetry,
packet budgets + rationale-carrier preservation, anchor hallucination
defense + stat fallback, low-confidence drops, idempotent marker skips
(zero LLM calls on re-run), and the marker-written-last crash property.
"""

import asyncio
import subprocess
import sys
import types

from kumiho_memory.code_decisions import CodeMemoryConfig
from kumiho_memory.code_capture import (
    CommitInfo,
    IngestStats,
    build_packet,
    capture_decisions,
    derive_repo_id,
    enumerate_commits,
    ingest_repo,
    prefilter,
    validate_decisions,
    _truncate_file_diff,
    _evidence_grade,
)


def test_evidence_grade_from_atoms():
    """§6: a code decision's Level-of-Evidence grade is derived
    deterministically (keyless) from its evidence atoms."""
    assert _evidence_grade([{"kind": "measurement"}]) == "corroborated"
    assert _evidence_grade(
        [{"kind": "review_finding"}, {"kind": "constraint"}]
    ) == "corroborated"
    assert _evidence_grade([{"kind": "benchmark"}]) == "corroborated"
    assert _evidence_grade([{"kind": "incident"}]) == "corroborated"
    assert _evidence_grade([{"kind": "constraint"}]) == "single_source"
    assert _evidence_grade([{"kind": "rejected_alternative"}]) == "single_source"
    assert _evidence_grade([{}]) == "single_source"   # default kind = constraint
    assert _evidence_grade([]) == "unverified"
    assert _evidence_grade(None) == "unverified"


def test_run_git_is_bounded_by_timeout(monkeypatch):
    """Every git subprocess must pass a timeout — the keyless capture resolves
    git OUTSIDE the write bound, so an unbounded git hang would hang the whole
    tool indefinitely (observed as a multi-minute no-op)."""
    from kumiho_memory import code_capture as cc

    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(stdout="ok")

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    assert cc._run_git(".", "rev-parse", "HEAD") == "ok"
    assert captured.get("timeout") == cc._GIT_TIMEOUT
    # a real git hang now surfaces as TimeoutExpired, which the callers already
    # convert into "git resolution failed" / a repo-id fallback (not a hang)
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=cc._GIT_TIMEOUT)

    monkeypatch.setattr(cc.subprocess, "run", hang)
    assert derive_repo_id(".")  # falls back to the dir name, does not raise


# ---------------- synthetic git repo ----------------


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    (repo / "a.py").write_text("# choose executor\nX = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m",
         "feat: offload rerank to executor\n\nInline CE blocked the loop; "
         "measured: concurrency-4 collapsed to ~1 effective.")
    (repo / "a.py").write_text("# ONE worker on purpose\nX = 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fix: single worker executor\n\n"
         "Adversarial review found cross-loop pool corruption with N workers.")
    return repo


# ---------------- enumeration / prefilter / packet ----------------


def test_enumerate_commits_fields(tmp_path):
    repo = _make_repo(tmp_path)
    commits = enumerate_commits(str(repo), None, 10)
    assert len(commits) == 2
    newest = commits[0]
    assert newest.subject.startswith("fix: single worker")
    assert "Adversarial review" in newest.body
    assert newest.hash and newest.author_date


def test_derive_repo_id_falls_back_to_dirname(tmp_path):
    repo = _make_repo(tmp_path)
    assert derive_repo_id(str(repo)) == "repo"


def test_prefilter_asymmetry():
    keep, _ = prefilter(CommitInfo("h", "a", "d", "merge branch x", "", parents=["1", "2"]))
    assert not keep  # merge without body: certain noise
    keep, _ = prefilter(CommitInfo("h", "a", "d", "merge branch x",
                                   "squash body carries rationale", parents=["1", "2"]))
    assert keep      # squash-merge WITH body: the only rationale carrier
    keep, _ = prefilter(CommitInfo("h", "a", "d", "chore: bump version", ""))
    assert not keep
    keep, _ = prefilter(CommitInfo("h", "a", "d", "chore: align __version__",
                                   "note the reformulate-draws knob"))
    assert keep      # chore WITH body passes — type is never a criterion


def test_truncate_preserves_hunks_and_comments():
    lines = (
        ["@@ -1,9 +1,9 @@"]
        + [f"+x{i} = {i}" for i in range(30)]
        + ["+# ONE worker on purpose — rationale comment"]
        + [f"-y{i} = {i}" for i in range(30)]
    )
    kept = _truncate_file_diff(lines, budget=10)
    text = "\n".join(kept)
    assert "@@ -1,9 +1,9 @@" in text                      # hunk header survives
    assert "ONE worker on purpose" in text                # comment tier first
    assert "[...truncated" in text                        # disclosure marker
    assert len(kept) <= 12


def test_build_packet_is_message_first(tmp_path):
    repo = _make_repo(tmp_path)
    commits = enumerate_commits(str(repo), None, 1)
    c = commits[0]
    c.files = ["a.py"]
    packet = build_packet(str(repo), c, CodeMemoryConfig())
    assert packet.index("subject:") < packet.index("changed files:")
    assert "- a.py" in packet
    assert "ONE worker on purpose" in packet  # comment carried as evidence


# ---------------- validation ----------------


def _decision(**kw):
    base = {
        "title": "Use a single-worker executor",
        "decision": "One worker",
        "rationale": "serialize inference",
        "why_question": "why single worker?",
        "symbols": ["rerank_async"],
        "evidence": [{"kind": "measurement", "text": "collapsed to ~1", "source_ref": "commit:x"}],
        "anchors": [{"file": "a.py", "line_start": 1, "line_end": 9, "role": "primary"}],
        "supersedes_hint": "",
        "confidence": "high",
    }
    base.update(kw)
    return base


def test_validate_drops_hallucinated_anchor_with_stat_fallback():
    c = CommitInfo("h", "a", "2026-07-10T00:00:00+09:00", "s", "b",
                   files=["real.py", "other.py"])
    out = validate_decisions(
        c, [_decision(anchors=[{"file": "ghost.py", "line_start": 1,
                                "line_end": 2, "role": "primary"}])],
        CodeMemoryConfig(),
    )
    assert len(out) == 1
    files = {a["file"] for a in out[0]["anchors"]}
    assert files == {"real.py", "other.py"}  # fallback to ground truth
    assert all(a["role"] == "touched" for a in out[0]["anchors"])


def test_validate_drops_low_confidence_without_evidence():
    c = CommitInfo("h", "a", "d", "s", "b", files=["a.py"])
    out = validate_decisions(
        c, [_decision(confidence="low", evidence=[])], CodeMemoryConfig(),
    )
    assert out == []
    out = validate_decisions(
        c, [_decision(confidence="low")], CodeMemoryConfig(),
    )
    assert len(out) == 1  # low WITH evidence survives


def test_validate_caps_anchors():
    c = CommitInfo("h", "a", "d", "s", "b", files=[f"f{i}.py" for i in range(20)])
    anchors = [{"file": f"f{i}.py", "line_start": 1, "line_end": 2, "role": "touched"}
               for i in range(20)]
    out = validate_decisions(c, [_decision(anchors=anchors)], CodeMemoryConfig())
    assert len(out[0]["anchors"]) == CodeMemoryConfig().max_anchors_per_decision


# ---------------- ingest: idempotency + marker-last (faked SDK) ----------------


class _StubAdapter:
    """Canned-JSON LLM stub counting calls."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def chat(self, *, messages, model, system="", max_tokens=1024, json_mode=False):
        self.calls += 1
        return self.payload


class _MemRev:
    def __init__(self, uri, metadata):
        self.kref = types.SimpleNamespace(uri=uri)
        self.metadata = dict(metadata)
        self.edges = []

    def get_edges(self, edge_type_filter=None, direction=0):
        out = []
        for e in self.edges:
            if edge_type_filter and e.edge_type != edge_type_filter:
                continue
            if direction == 0 and e.source_kref.uri != self.kref.uri:
                continue
            if direction == 1 and e.target_kref.uri != self.kref.uri:
                continue
            out.append(e)
        return out

    def create_edge(self, target, edge_type, metadata=None):
        e = types.SimpleNamespace(
            edge_type=edge_type,
            source_kref=self.kref,
            target_kref=target.kref,
            metadata=dict(metadata or {}),
        )
        self.edges.append(e)
        target.edges.append(e)
        _FAKE.edges.append(e)
        return e

    def get_item(self):
        return _FAKE.item_of[self.kref.uri]


class _MemItem:
    def __init__(self, slug, kind, project):
        self.slug, self.kind = slug, kind
        self.kref = types.SimpleNamespace(uri=f"kref://{project}/{kind}/{slug}")
        self.revisions = []

    def get_latest_revision(self):
        return self.revisions[-1] if self.revisions else None

    def create_revision(self, metadata=None, number=0):
        rev = _MemRev(f"{self.kref.uri}@{len(self.revisions) + 1}", metadata or {})
        self.revisions.append(rev)
        _FAKE.revs[rev.kref.uri] = rev
        _FAKE.item_of[rev.kref.uri] = self
        return rev


class _MemProject:
    def __init__(self, name):
        self.name = name
        self.items = {}
        self.spaces = set()

    def create_space(self, name):
        self.spaces.add(name)

    def create_item(self, slug, kind, parent_path=""):
        key = (slug, kind)
        if key in self.items:
            import grpc
            exc = grpc.RpcError()
            exc.code = lambda: grpc.StatusCode.ALREADY_EXISTS
            raise exc
        item = _MemItem(slug, kind, self.name)
        self.items[key] = item
        return item

    def get_item(self, slug, kind, parent_path=""):
        item = self.items.get((slug, kind))
        if item is None:
            raise KeyError(slug)
        return item


class _FakeState:
    def __init__(self):
        self.projects = {}
        self.revs = {}
        self.item_of = {}
        self.edges = []
        self.embedding_texts = []


_FAKE = _FakeState()


def _install_fake_kumiho(monkeypatch):
    global _FAKE
    _FAKE = _FakeState()
    fake = types.ModuleType("kumiho")
    fake.OUTGOING, fake.INCOMING, fake.BOTH = 0, 1, 2

    def get_project(name):
        return _FAKE.projects.get(name)

    def create_project(name):
        p = _MemProject(name)
        _FAKE.projects[name] = p
        return p

    def get_revision(uri):
        return _FAKE.revs[uri]

    class _Client:
        def create_revision(self, item_kref, metadata=None, number=0, embedding_text=""):
            _FAKE.embedding_texts.append(embedding_text)
            for item in list(_FAKE.projects.values())[0].items.values() if _FAKE.projects else []:
                pass
            # resolve item by kref uri
            for p in _FAKE.projects.values():
                for item in p.items.values():
                    if item.kref.uri == getattr(item_kref, "uri", item_kref):
                        return item.create_revision(metadata=metadata)
            raise KeyError(item_kref)

    fake.get_project = get_project
    fake.create_project = create_project
    fake.get_revision = get_revision
    fake.get_client = lambda: _Client()
    monkeypatch.setitem(sys.modules, "kumiho", fake)
    return fake


def _payload_for(commits):
    import json as _json

    return _json.dumps({
        "commits": [
            {
                "hash": c.hash,
                "decisions": [{
                    "title": "Use a single-worker executor",
                    "decision": "Run CE on one worker",
                    "rationale": "inline CE blocked the loop",
                    "why_question": "why is the executor single-worker?",
                    "symbols": ["rerank_async"],
                    "evidence": [{"kind": "measurement",
                                  "text": "concurrency-4 collapsed to ~1 effective",
                                  "source_ref": f"commit:{c.hash[:7]}"}],
                    "anchors": [{"file": "a.py", "line_start": 1,
                                 "line_end": 2, "role": "primary"}],
                    "supersedes_hint": "",
                    "confidence": "high",
                }],
            }
            for c in commits
        ],
    })


def test_ingest_writes_and_is_idempotent(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    commits = enumerate_commits(str(repo), None, 10)
    adapter = _StubAdapter(_payload_for(commits))
    cfg = CodeMemoryConfig(repo="testrepo")

    stats = asyncio.run(ingest_repo(
        str(repo), None,
        project_name="p-code", config=cfg, adapter=adapter, model="stub",
    ))
    assert stats.errors == []
    assert stats.decisions >= 1
    assert stats.evidence >= 1
    assert adapter.calls >= 1
    # embedding_text went through the client-level path (§0-5), why-first
    assert any(t.startswith("why is the executor single-worker?")
               for t in _FAKE.embedding_texts)
    # markers exist for both commits
    project = _FAKE.projects["p-code"]
    markers = [i for (s, k), i in project.items.items() if k == "code_commit"]
    assert len(markers) == len(commits)

    # Second run: marker skip means ZERO new LLM calls and no new decisions
    calls_before = adapter.calls
    decisions_before = stats.decisions
    stats2 = asyncio.run(ingest_repo(
        str(repo), None,
        project_name="p-code", config=cfg, adapter=adapter, model="stub",
    ))
    assert adapter.calls == calls_before
    assert stats2.skipped_marker == len(commits)
    assert stats2.decisions == 0 and decisions_before >= 1


def test_marker_written_last_on_crash(tmp_path, monkeypatch):
    """Crash injection: if the decision write fails, no marker may exist —
    the commit must be retried on the next run."""
    repo = _make_repo(tmp_path)
    fake = _install_fake_kumiho(monkeypatch)
    commits = enumerate_commits(str(repo), None, 1)
    adapter = _StubAdapter(_payload_for(commits))
    cfg = CodeMemoryConfig(repo="testrepo")

    # Sabotage the client-level revision write (decision node creation).
    class _Boom:
        def create_revision(self, *a, **kw):
            raise RuntimeError("injected crash")

    fake.get_client = lambda: _Boom()

    stats = asyncio.run(ingest_repo(
        str(repo), "HEAD~1..HEAD", project_name="p-code",
        config=cfg, adapter=adapter, model="stub",
    ))
    assert stats.failed_commits  # reported, not swallowed
    project = _FAKE.projects.get("p-code")
    markers = [
        i for (s, k), i in (project.items.items() if project else [])
        if k == "code_commit" and i.get_latest_revision() is not None
    ]
    assert markers == []  # no completed marker -> retry on next run


def test_ingest_without_adapter_reports_error(tmp_path):
    repo = _make_repo(tmp_path)
    stats = asyncio.run(ingest_repo(
        str(repo), None, project_name="p-code",
        config=CodeMemoryConfig(), adapter=None, model="",
    ))
    assert stats.errors and "adapter" in stats.errors[0]


# ---------------- review-fix regression tests ----------------


def test_rev_range_injection_rejected(tmp_path):
    from kumiho_memory.code_capture import _validate_rev_range
    import pytest as _pytest

    assert _validate_rev_range("HEAD~30..HEAD") == "HEAD~30..HEAD"
    assert _validate_rev_range("v1.0.0...main") == "v1.0.0...main"
    for bad in ("--output=/tmp/pwn", "-p", "--upload-pack=rm", "; rm -rf"):
        with _pytest.raises(ValueError):
            _validate_rev_range(bad)


def test_prefilter_runs_after_files_loaded(tmp_path, monkeypatch):
    """A bodyless short-subject commit WITH a real diff must survive: the
    trivial-subject rule reads commit.files, which is only valid after the
    changed-file load (reviewed-and-confirmed ordering defect)."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fix rerank")  # 2 words, no body, real diff

    _install_fake_kumiho(monkeypatch)
    commits = enumerate_commits(str(repo), None, 5)
    adapter = _StubAdapter(_payload_for(commits))
    stats = asyncio.run(ingest_repo(
        str(repo), None, project_name="p-code",
        config=CodeMemoryConfig(repo="r"), adapter=adapter, model="stub",
    ))
    assert stats.skipped_prefilter == 0
    assert adapter.calls >= 1  # it reached the LLM


def test_supersede_pass_three_signals_and_inplace_demotion(monkeypatch):
    """3-signal confluence + in-place demotion: the SAME revision the edges
    are pinned to must carry status=superseded afterwards (the two-krefs
    identity split was the review's confirmed critical)."""
    import kumiho_memory.code_capture as cc

    fake = _install_fake_kumiho(monkeypatch)
    project = fake.create_project("p-code")
    for s in ("decisions", "anchors", "commits", "evidence"):
        project.create_space(s)

    cfg = CodeMemoryConfig(repo="r")
    # old decision anchored to a.py, decided earlier (different tz offset —
    # +14:00 makes the raw string LARGER than the -05:00 new date, which
    # would flip a naive string comparison)
    old_item = project.create_item("old-dec", "code_decision")
    old_rev = old_item.create_revision(metadata={
        "title": "Use inline call",
        "decision": "Call the cross encoder inline on the loop",
        "decided_at": "2026-07-10T23:00:00+14:00",  # = 09:00 UTC
        "status": "active",
    })
    anchor_item = project.create_item("anchor-a", "code_anchor")
    anchor_rev = anchor_item.create_revision(metadata={"repo": "r", "path": "a.py"})
    old_rev.create_edge(anchor_rev, "IMPLEMENTED_IN", metadata={})

    new_item = project.create_item("new-dec", "code_decision")
    new_rev = new_item.create_revision(metadata={
        "title": "Use offloaded call",
        "decision": "Call the cross encoder inline on the loop via executor",
        "decided_at": "2026-07-10T10:00:00-05:00",  # = 15:00 UTC (LATER)
        "status": "active",
    })

    # set_attribute support on the fake
    def set_attribute(self, key, value):
        self.metadata[key] = value
        return True
    _MemRev.set_attribute = set_attribute

    stats = IngestStats()
    cc._supersede_pass(
        project, cfg, new_rev, new_rev.metadata, [anchor_rev], "", stats,
    )
    # linked (jaccard high, shared anchor, old(09:00Z) < new(15:00Z))
    sup_edges = [e for e in new_rev.edges if e.edge_type == "SUPERSEDES"]
    assert len(sup_edges) == 1 and sup_edges[0].target_kref.uri == old_rev.kref.uri
    # demotion happened IN PLACE on the edge-pinned revision
    assert old_rev.metadata["status"] == "superseded"
    assert old_item.get_latest_revision() is old_rev  # no new revision created

    # time-order guard: swapping direction must NOT link (old > new)
    stats2 = IngestStats()
    cc._supersede_pass(
        project, cfg, old_rev, old_rev.metadata, [anchor_rev], "", stats2,
    )
    back_edges = [e for e in old_rev.edges
                  if e.edge_type == "SUPERSEDES" and e.source_kref.uri == old_rev.kref.uri]
    assert back_edges == []


def test_supersede_requires_jaccard_confluence(monkeypatch):
    """Shared anchor alone must not link — the Jaccard signal gates it."""
    import kumiho_memory.code_capture as cc

    fake = _install_fake_kumiho(monkeypatch)
    project = fake.create_project("p-code")
    old_item = project.create_item("old2", "code_decision")
    old_rev = old_item.create_revision(metadata={
        "title": "Completely unrelated topic",
        "decision": "Cache invalidation policy for artifacts",
        "decided_at": "2026-07-01T00:00:00+00:00",
        "status": "active",
    })
    anchor_item = project.create_item("anchor-b", "code_anchor")
    anchor_rev = anchor_item.create_revision(metadata={"repo": "r", "path": "b.py"})
    old_rev.create_edge(anchor_rev, "IMPLEMENTED_IN", metadata={})

    new_item = project.create_item("new2", "code_decision")
    new_rev = new_item.create_revision(metadata={
        "title": "Executor offload",
        "decision": "Run reranks on a dedicated executor",
        "decided_at": "2026-07-10T00:00:00+00:00",
        "status": "active",
    })
    stats = IngestStats()
    cc._supersede_pass(
        project, CodeMemoryConfig(repo="r"), new_rev, new_rev.metadata,
        [anchor_rev], "", stats,
    )
    assert all(e.edge_type != "SUPERSEDES" for e in new_rev.edges)
    assert old_rev.metadata["status"] == "active"


def test_rewrite_convergence_no_duplicates(tmp_path, monkeypatch):
    """Rebase simulation: amending a commit changes its sha; re-ingest must
    converge on the SAME decision node (sha-free identity) — no duplicates."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")

    commits = enumerate_commits(str(repo), None, 10)
    adapter = _StubAdapter(_payload_for(commits))
    asyncio.run(ingest_repo(str(repo), None, project_name="p-code",
                            config=cfg, adapter=adapter, model="stub"))
    decisions_before = len([
        i for (s, k), i in _FAKE.projects["p-code"].items.items()
        if k == "code_decision"
    ])

    # rewrite history: amend HEAD (same author date, new sha). The committer
    # date is forced forward — an amend within the same second would
    # otherwise reproduce the identical sha (flaky in fast batch runs).
    import os as _os
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--amend", "-q", "--no-edit"],
        check=True, capture_output=True, text=True,
        env={**_os.environ, "GIT_COMMITTER_DATE": "2030-01-01T00:00:00+00:00"},
    )
    commits2 = enumerate_commits(str(repo), None, 10)
    assert commits2[0].hash != commits[0].hash  # sha changed
    adapter2 = _StubAdapter(_payload_for(commits2))
    asyncio.run(ingest_repo(str(repo), None, project_name="p-code",
                            config=cfg, adapter=adapter2, model="stub"))
    decisions_after = len([
        i for (s, k), i in _FAKE.projects["p-code"].items.items()
        if k == "code_decision"
    ])
    assert decisions_after == decisions_before  # converged, not duplicated


def test_force_deprecates_then_rewrites(tmp_path, monkeypatch):
    """--force (design §4.6): the commit's stale decisions are deprecated
    (in-place status too), then re-mining writes a fresh revision and
    restores converged items."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    commits = enumerate_commits(str(repo), None, 10)
    adapter = _StubAdapter(_payload_for(commits))

    # deprecated 플래그를 지원하도록 페이크 확장
    _MemItem.deprecated = False
    def set_deprecated(self, status):
        self.deprecated = bool(status)
    _MemItem.set_deprecated = set_deprecated
    def set_attribute(self, key, value):
        self.metadata[key] = value
        return True
    _MemRev.set_attribute = set_attribute

    asyncio.run(ingest_repo(str(repo), None, project_name="p-code",
                            config=cfg, adapter=adapter, model="stub"))
    project = _FAKE.projects["p-code"]
    dec_items = [i for (s, k), i in project.items.items() if k == "code_decision"]
    revs_before = {i.slug: len(i.revisions) for i in dec_items}
    assert dec_items and all(not i.deprecated for i in dec_items)

    stats = asyncio.run(ingest_repo(str(repo), None, project_name="p-code",
                                    config=cfg, adapter=adapter, model="stub",
                                    force=True))
    assert stats.deprecated >= 1                     # pre-pass retired old gen
    for i in dec_items:
        assert not i.deprecated                      # converged -> restored
        assert len(i.revisions) > revs_before[i.slug]  # fresh revision written
        assert i.revisions[-1].metadata.get("status") == "active"


def test_capture_decisions_is_keyless(tmp_path, monkeypatch):
    """Agent-driven capture: structured decisions from the agent are written
    with NO adapter/LLM (the keyless reflect pattern for code). Anchors union
    with the commit's real changed files; hallucinated files drop."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")

    # what Claude would pass after committing — no LLM produced this
    decisions = [{
        "title": "Use a single-worker executor",
        "decision": "run the CE rerank on one dedicated worker",
        "rationale": "a shared pool oversubscribes the cross-encoder",
        "why_question": "why not the default executor?",
        "symbols": ["rerank_async"],
        "files": ["a.py", "ghost_never_changed.py"],   # ghost must drop
        "evidence": [{"kind": "rejected_alternative",
                      "text": "the default executor is shared - 32-thread oversubscription"}],
        "confidence": "high",
    }]

    # NOTE: no adapter, no model — the whole point
    stats = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-code", config=cfg,
    ))
    assert stats.decisions == 1 and not stats.errors
    assert stats.evidence == 1

    import kumiho
    project = kumiho.get_project("p-code")
    d_item = next(i for (s, k), i in project.items.items() if k == "code_decision")
    rev = d_item.get_latest_revision()
    files = set(rev.metadata["files"].split(","))
    assert "a.py" in files                      # real changed file anchored
    assert "ghost_never_changed.py" not in files  # hallucinated file dropped
    # committed under the real HEAD commit (sha-anchored provenance)
    assert rev.metadata["commit_hash"]


def test_capture_decisions_empty_and_bad_ref(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    s1 = asyncio.run(capture_decisions(str(repo), [], project_name="p-code", config=cfg))
    assert s1.errors and "no decisions" in s1.errors[0]
    s2 = asyncio.run(capture_decisions(
        str(repo), [{"title": "x", "decision": "y"}],
        commit_ref="does-not-exist-ref", project_name="p-code", config=cfg,
    ))
    assert s2.errors  # unresolvable ref -> loud error, no write
