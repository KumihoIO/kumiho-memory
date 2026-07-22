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


class _FakePipe:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _StuckProc:
    """A git process whose pipes never close — kill() doesn't release them.

    Models the Windows failure mode of KumihoIO/kumiho-SDKs#79: the child is
    stuck in uninterruptible kernel I/O (or a descendant inherited its pipe
    handles), so TerminateProcess doesn't close the pipes and any
    timeout-less communicate() blocks forever.
    """

    def __init__(self):
        self.killed = False
        self.timeouts_seen = []
        self.returncode = None
        self.stdout = _FakePipe()
        self.stderr = _FakePipe()

    def communicate(self, timeout=None):
        self.timeouts_seen.append(timeout)
        raise subprocess.TimeoutExpired(cmd="git", timeout=timeout or 0)

    def kill(self):
        self.killed = True


class _SlowDrainProc(_StuckProc):
    """Times out once, then drains after kill() — the grace-success path."""

    def communicate(self, timeout=None):
        self.timeouts_seen.append(timeout)
        if len(self.timeouts_seen) == 1:
            raise subprocess.TimeoutExpired(cmd="git", timeout=timeout or 0)
        return ("partial-out", "partial-err")


def test_run_git_kill_path_is_bounded(monkeypatch):
    """#79: every wait in _run_git must carry an explicit bound — INCLUDING
    the post-kill() pipe drain. subprocess.run's own timeout handling is not
    enough: its Windows TimeoutExpired path calls kill() then a timeout-less
    communicate(), which blocks forever while anything holds the pipes
    (observed as a 30-minute kumiho_code_capture hang)."""
    from kumiho_memory import code_capture as cc

    procs = []

    def fake_popen(*args, **kwargs):
        proc = _StuckProc()
        procs.append(proc)
        return proc

    monkeypatch.setattr(cc.subprocess, "Popen", fake_popen)

    try:
        cc._run_git(".", "rev-parse", "HEAD")
        raise AssertionError("expected TimeoutExpired")
    except subprocess.TimeoutExpired:
        pass

    (proc,) = procs
    assert proc.killed
    # both waits — initial and post-kill grace — were explicitly bounded
    assert proc.timeouts_seen == [cc._GIT_TIMEOUT, cc._GIT_KILL_GRACE]
    # abandon path closed our pipe ends so reader threads exit promptly
    assert proc.stdout.closed and proc.stderr.closed

    # callers already convert TimeoutExpired into "git resolution failed" /
    # a repo-id fallback (not a hang)
    assert derive_repo_id(".")  # falls back to the dir name, does not raise


def test_run_git_timeout_attaches_drained_output(monkeypatch):
    """Parity with subprocess.run: when the post-kill grace drain succeeds,
    the raised TimeoutExpired carries what the child wrote before dying —
    callers diagnosing WHY git stalled get real bytes, not None."""
    from kumiho_memory import code_capture as cc

    procs = []

    def fake_popen(*args, **kwargs):
        proc = _SlowDrainProc()
        procs.append(proc)
        return proc

    monkeypatch.setattr(cc.subprocess, "Popen", fake_popen)

    try:
        cc._run_git(".", "rev-parse", "HEAD")
        raise AssertionError("expected TimeoutExpired")
    except subprocess.TimeoutExpired as exc:
        assert exc.output == "partial-out"
        assert exc.stderr == "partial-err"

    (proc,) = procs
    assert proc.killed
    assert proc.timeouts_seen == [cc._GIT_TIMEOUT, cc._GIT_KILL_GRACE]


def test_run_git_kills_child_on_interrupt(monkeypatch):
    """Parity with subprocess.run's bare-except: an interrupt (or any other
    non-timeout escape) mid-communicate() must kill the child before
    propagating — otherwise a live git process leaks with open pipes."""
    from kumiho_memory import code_capture as cc

    class _InterruptProc:
        def __init__(self):
            self.killed = False
            self.returncode = None

        def communicate(self, timeout=None):
            raise KeyboardInterrupt()

        def kill(self):
            self.killed = True

    procs = []

    def fake_popen(*args, **kwargs):
        proc = _InterruptProc()
        procs.append(proc)
        return proc

    monkeypatch.setattr(cc.subprocess, "Popen", fake_popen)

    try:
        cc._run_git(".", "rev-parse", "HEAD")
        raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass

    (proc,) = procs
    assert proc.killed  # the child was killed, not leaked


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
        project_name="p-decisions", config=cfg, adapter=adapter, model="stub",
    ))
    assert stats.errors == []
    assert stats.decisions >= 1
    assert stats.evidence >= 1
    assert adapter.calls >= 1
    # embedding_text went through the client-level path (§0-5), why-first
    assert any(t.startswith("why is the executor single-worker?")
               for t in _FAKE.embedding_texts)
    # markers exist for both commits
    project = _FAKE.projects["p-decisions"]
    markers = [i for (s, k), i in project.items.items() if k == "code_commit"]
    assert len(markers) == len(commits)

    # Second run: marker skip means ZERO new LLM calls and no new decisions
    calls_before = adapter.calls
    decisions_before = stats.decisions
    stats2 = asyncio.run(ingest_repo(
        str(repo), None,
        project_name="p-decisions", config=cfg, adapter=adapter, model="stub",
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
        str(repo), "HEAD~1..HEAD", project_name="p-decisions",
        config=cfg, adapter=adapter, model="stub",
    ))
    assert stats.failed_commits  # reported, not swallowed
    project = _FAKE.projects.get("p-decisions")
    markers = [
        i for (s, k), i in (project.items.items() if project else [])
        if k == "code_commit" and i.get_latest_revision() is not None
    ]
    assert markers == []  # no completed marker -> retry on next run


def test_ingest_without_adapter_reports_error(tmp_path):
    repo = _make_repo(tmp_path)
    stats = asyncio.run(ingest_repo(
        str(repo), None, project_name="p-decisions",
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
        str(repo), None, project_name="p-decisions",
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
    project = fake.create_project("p-decisions")
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
    project = fake.create_project("p-decisions")
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
    asyncio.run(ingest_repo(str(repo), None, project_name="p-decisions",
                            config=cfg, adapter=adapter, model="stub"))
    decisions_before = len([
        i for (s, k), i in _FAKE.projects["p-decisions"].items.items()
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
    asyncio.run(ingest_repo(str(repo), None, project_name="p-decisions",
                            config=cfg, adapter=adapter2, model="stub"))
    decisions_after = len([
        i for (s, k), i in _FAKE.projects["p-decisions"].items.items()
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

    asyncio.run(ingest_repo(str(repo), None, project_name="p-decisions",
                            config=cfg, adapter=adapter, model="stub"))
    project = _FAKE.projects["p-decisions"]
    dec_items = [i for (s, k), i in project.items.items() if k == "code_decision"]
    revs_before = {i.slug: len(i.revisions) for i in dec_items}
    assert dec_items and all(not i.deprecated for i in dec_items)

    stats = asyncio.run(ingest_repo(str(repo), None, project_name="p-decisions",
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
        project_name="p-decisions", config=cfg,
    ))
    assert stats.decisions == 1 and not stats.errors
    assert stats.evidence == 1

    import kumiho
    project = kumiho.get_project("p-decisions")
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
    s1 = asyncio.run(capture_decisions(str(repo), [], project_name="p-decisions", config=cfg))
    assert s1.errors and "no decisions" in s1.errors[0]
    s2 = asyncio.run(capture_decisions(
        str(repo), [{"title": "x", "decision": "y"}],
        commit_ref="does-not-exist-ref", project_name="p-decisions", config=cfg,
    ))
    assert s2.errors  # unresolvable ref -> loud error, no write


# ---------------- privacy boundary (issue #99: P0-1) ----------------
#
# The commit-mining path must cross into the cloud graph through the SAME
# per-atom PII/credential door as the session path (code_session): a
# credential-bearing atom is DROPPED (counted), PII is redacted in place, and
# clean content is stored byte-identical.  Screening defaults ON — the entry
# points construct a PIIRedactor when none is passed, so these tests exercise
# the default (keyless) posture with no explicit redactor.

_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"          # matches PIIRedactor aws_access_key
_SK_KEY = "sk-abcdefghij0123456789ABCDEF"  # matches api_key_generic
_EMAIL = "alice@example.com"               # matches PIIRedactor email
_PHONE = "415-555-0199"                    # matches PIIRedactor phone


def _make_repo_secret_subject(tmp_path):
    """A one-commit repo whose commit SUBJECT carries a leaked key — the
    'password in the commit message' case from the issue.  The subject is
    stored verbatim on the marker and injected into the decision embedding, so
    it leaves for the graph independently of the mined decision text."""
    repo = tmp_path / "repo_sec"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    (repo / "a.py").write_text("X = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m",
         f"fix: rotate leaked key {_AWS_KEY}\n\nrationale body, no secrets here")
    return repo


def _payload_one(commit_hash, *, evidence, **overrides):
    """A single-commit / single-decision canned LLM payload with caller-chosen
    evidence atoms (the ingest-path secret injection point)."""
    import json as _json

    dec = {
        "title": "Load the signing key from the environment",
        "decision": "read the key from env instead of a literal",
        "rationale": "hardcoded secrets leak into history",
        "why_question": "why env not a literal?",
        "symbols": ["load_key"],
        "evidence": evidence,
        "anchors": [{"file": "a.py", "line_start": 1, "line_end": 2,
                     "role": "primary"}],
        "supersedes_hint": "",
        "confidence": "high",
    }
    dec.update(overrides)
    return _json.dumps({"commits": [{"hash": commit_hash, "decisions": [dec]}]})


class _CapturingAdapter(_StubAdapter):
    """Records the prompt text sent to the LLM so tests can assert on the
    PRE-LLM packet — the exact bytes that would transit the provider."""

    def __init__(self, payload):
        super().__init__(payload)
        self.prompts = []

    async def chat(self, *, messages, model, system="", max_tokens=1024,
                   json_mode=False):
        self.prompts.append(messages[0]["content"])
        return await super().chat(
            messages=messages, model=model, system=system,
            max_tokens=max_tokens, json_mode=json_mode,
        )


def _make_repo_pii_diff(tmp_path):
    """A two-commit repo whose HEAD carries PII in the subject/body AND a
    credential + PII inside the diff — the 'raw packet transits the provider'
    case from issue #117."""
    repo = tmp_path / "repo_pii"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    (repo / "cfg.py").write_text("OWNER = 'x'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed config module\n\nbaseline body")
    (repo / "cfg.py").write_text(
        f"OWNER = '{_EMAIL}'\n"
        f'API_KEY = "{_SK_KEY}"\n'
        f"PHONE = '{_PHONE}'\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m",
         f"fix: page {_EMAIL} on failure\n\n"
         f"escalation contact is {_PHONE}, no other notes")
    return repo


def test_ingest_anonymizes_packet_before_llm(tmp_path, monkeypatch):
    """Issue #117: the per-commit packet is anonymized BEFORE it reaches the
    LLM adapter (parity with the session path).  PII is redacted in place, a
    credential-bearing line is dropped to [redacted], and diff STRUCTURE (hunk
    headers, file paths, labels) survives so extraction quality is preserved."""
    repo = _make_repo_pii_diff(tmp_path)
    _install_fake_kumiho(monkeypatch)
    commits = enumerate_commits(str(repo), "HEAD~1..HEAD", 1)
    adapter = _CapturingAdapter(_payload_for(commits))
    cfg = CodeMemoryConfig(repo="testrepo")

    stats = asyncio.run(ingest_repo(
        str(repo), "HEAD~1..HEAD", project_name="p-decisions",
        config=cfg, adapter=adapter, model="stub",
    ))
    assert adapter.calls >= 1 and adapter.prompts
    sent = "\n".join(adapter.prompts)   # everything the provider would see

    # content anonymized: no raw PII or key transits the LLM
    assert _EMAIL not in sent
    assert _PHONE not in sent
    assert _SK_KEY not in sent
    # PII redacted IN PLACE (descriptors present, in subject/body AND diff)
    assert "[email]" in sent and "[phone]" in sent
    # credential-bearing line DROPPED to the placeholder (counted)
    assert "[redacted]" in sent
    assert stats.credentials_dropped >= 1
    # structure preserved: the model still has hunk headers, paths, and labels
    assert "subject:" in sent
    assert "changed files:" in sent and "- cfg.py" in sent
    assert "--- diff: cfg.py ---" in sent
    assert "@@" in sent


def test_ingest_screens_credential_in_commit_message(tmp_path, monkeypatch):
    """(a) A secret in the commit MESSAGE must not reach the graph: the marker
    subject is blanked and the decision embedding carries no key."""
    repo = _make_repo_secret_subject(tmp_path)
    _install_fake_kumiho(monkeypatch)
    commits = enumerate_commits(str(repo), None, 10)
    adapter = _StubAdapter(_payload_for(commits))  # clean decision prose
    cfg = CodeMemoryConfig(repo="testrepo")

    stats = asyncio.run(ingest_repo(
        str(repo), None, project_name="p-decisions",
        config=cfg, adapter=adapter, model="stub",
    ))
    assert stats.credentials_dropped >= 1
    assert stats.decisions >= 1  # the clean decision still stored (surgical)
    # the key appears nowhere that leaves for the graph
    assert all(_AWS_KEY not in t for t in _FAKE.embedding_texts)
    project = _FAKE.projects["p-decisions"]
    markers = [i for (s, k), i in project.items.items() if k == "code_commit"]
    assert markers
    for m in markers:
        # F4: the dropped subject is the literal placeholder, never "" — an
        # empty embedding_text makes write_revision fall back to embedding
        # ALL metadata (hash/author/bookkeeping vector pollution).
        assert m.get_latest_revision().metadata.get("subject") == "[redacted]"
    # the marker embedding went through the client-level (explicit-text) path:
    # the fake only records embedding_texts on that path, so the placeholder's
    # presence proves the embed-all fallback was NOT taken.
    assert "[redacted]" in _FAKE.embedding_texts


def test_ingest_drops_credential_evidence_keeps_clean(tmp_path, monkeypatch):
    """(b) A secret quoted from a diff excerpt into evidence DROPS (counted);
    the clean sibling evidence in the same decision survives byte-identical."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    commits = enumerate_commits(str(repo), "HEAD~1..HEAD", 1)
    clean_text = "the loop collapsed to ~1 effective worker"
    adapter = _StubAdapter(_payload_one(commits[0].hash, evidence=[
        {"kind": "constraint",
         "text": f"the token was hardcoded as {_SK_KEY} in the diff",
         "source_ref": "commit:x"},
        {"kind": "measurement", "text": clean_text, "source_ref": "commit:x"},
    ]))
    cfg = CodeMemoryConfig(repo="testrepo")

    stats = asyncio.run(ingest_repo(
        str(repo), "HEAD~1..HEAD", project_name="p-decisions",
        config=cfg, adapter=adapter, model="stub",
    ))
    assert stats.credentials_dropped == 1
    assert stats.evidence == 1     # only the clean atom was written
    assert stats.decisions == 1    # the decision itself survived
    project = _FAKE.projects["p-decisions"]
    ev_items = [i for (s, k), i in project.items.items() if k == "code_evidence"]
    statements = [i.get_latest_revision().metadata.get("statement", "")
                  for i in ev_items]
    assert statements == [clean_text]                 # clean stored verbatim
    assert all(_SK_KEY not in t for t in _FAKE.embedding_texts)


def test_capture_screens_agent_supplied_credential(tmp_path, monkeypatch):
    """(c) Keyless agent path: a credential in an agent-supplied evidence entry
    is dropped even though the CONTENT trust model is otherwise unchanged."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    clean_text = "the default executor is shared across the process"
    decisions = [{
        "title": "Load the key from env",
        "decision": "read the API key from the environment",
        "rationale": "never commit secrets",
        "why_question": "why env not literal?",
        "symbols": ["load_key"],
        "files": ["a.py"],
        "evidence": [
            {"kind": "constraint",
             "text": 'was hardcoded: api_key = "supersecretvalue123"'},
            {"kind": "rejected_alternative", "text": clean_text},
        ],
        "confidence": "high",
    }]
    stats = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert stats.credentials_dropped == 1
    assert stats.evidence == 1 and stats.decisions == 1
    import kumiho
    project = kumiho.get_project("p-decisions")
    ev_items = [i for (s, k), i in project.items.items() if k == "code_evidence"]
    statements = [i.get_latest_revision().metadata.get("statement", "")
                  for i in ev_items]
    assert statements == [clean_text]
    assert all("supersecretvalue123" not in s for s in statements)


def test_capture_redacts_pii_in_place_not_dropped(tmp_path, monkeypatch):
    """PII (emails) is REDACTED in place, never dropped: the decision and its
    evidence still store, with the address replaced by the [email] descriptor."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    decisions = [{
        "title": "Notify the owner",
        "decision": "email the module owner alice@example.com on failure",
        "rationale": "the owner triages incidents",
        "why_question": "who is paged?",
        "symbols": ["notify"],
        "files": ["a.py"],
        "evidence": [{"kind": "constraint",
                      "text": "escalate to bob@example.com within the hour"}],
        "confidence": "high",
    }]
    stats = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert stats.credentials_dropped == 0            # PII is not a credential
    assert stats.decisions == 1 and stats.evidence == 1  # nothing dropped
    import kumiho
    project = kumiho.get_project("p-decisions")
    d_item = next(i for (s, k), i in project.items.items() if k == "code_decision")
    dmeta = d_item.get_latest_revision().metadata
    assert "alice@example.com" not in dmeta["decision"]
    assert "[email]" in dmeta["decision"]
    ev_item = next(i for (s, k), i in project.items.items() if k == "code_evidence")
    stmt = ev_item.get_latest_revision().metadata["statement"]
    assert "bob@example.com" not in stmt and "[email]" in stmt


def test_capture_clean_content_stored_byte_identical(tmp_path, monkeypatch):
    """Clean content is stored byte-identical — the screen is a no-op on text
    with no PII/credential, and drops nothing."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    decision_text = "run the CE rerank on one dedicated worker"
    rationale_text = "a shared pool oversubscribes the cross-encoder"
    evidence_text = "the default executor is shared across the process"
    decisions = [{
        "title": "Use a single-worker executor",
        "decision": decision_text,
        "rationale": rationale_text,
        "why_question": "why not the default executor?",
        "symbols": ["rerank_async"],
        "files": ["a.py"],
        "evidence": [{"kind": "rejected_alternative", "text": evidence_text}],
        "confidence": "high",
    }]
    stats = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert stats.credentials_dropped == 0
    assert stats.decisions == 1 and stats.evidence == 1
    import kumiho
    project = kumiho.get_project("p-decisions")
    d_item = next(i for (s, k), i in project.items.items() if k == "code_decision")
    dmeta = d_item.get_latest_revision().metadata
    assert dmeta["decision"] == decision_text
    assert dmeta["rationale"] == rationale_text
    ev_item = next(i for (s, k), i in project.items.items() if k == "code_evidence")
    assert ev_item.get_latest_revision().metadata["statement"] == evidence_text


def test_all_evidence_dropped_decision_survives(tmp_path, monkeypatch):
    """All-atoms-dropped policy (parity with the session path): a
    high-confidence decision whose ONLY evidence is a credential still stores —
    its anchors carry it — only the evidence atom is dropped, not the decision."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    decisions = [{
        "title": "Rotate the signing key",
        "decision": "move signing to KMS",
        "rationale": "manual keys leak",
        "why_question": "why KMS?",
        "symbols": ["sign"],
        "files": ["a.py"],
        "evidence": [{"kind": "constraint",
                      "text": 'was hardcoded: secret = "abcdefgh12345678"'}],
        "confidence": "high",
    }]
    stats = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert stats.credentials_dropped == 1
    assert stats.evidence == 0     # the only atom dropped
    assert stats.decisions == 1    # decision SURVIVES (anchor-carried)
    assert not stats.errors


def test_validate_decisions_credential_in_prose_drops_decision(monkeypatch):
    """A credential in the decision's OWN prose drops the WHOLE decision
    (matches code_session.validate_session_decisions), counted once."""
    from kumiho_memory.privacy import PIIRedactor

    c = CommitInfo("h", "a", "2026-07-10T00:00:00+09:00", "s", "b",
                   files=["a.py"])
    stats = IngestStats()
    out = validate_decisions(
        c,
        [_decision(decision=f'api_key = "{_SK_KEY}"'),
         _decision(title="Clean decision")],
        CodeMemoryConfig(), redactor=PIIRedactor(), stats=stats,
    )
    assert len(out) == 1                       # only the clean decision remains
    assert out[0]["title"] == "Clean decision"
    assert stats.credentials_dropped == 1


def test_validate_decisions_credential_symbol_dropped():
    """F1: symbols reach metadata AND the embedding text — a credential-
    bearing symbol entry drops (counted); identifiers are NOT PII-redacted
    (rewriting them would corrupt the correlation coordinates)."""
    from kumiho_memory.privacy import PIIRedactor

    c = CommitInfo("h", "a", "2026-07-10T00:00:00+09:00", "s", "b",
                   files=["a.py"])
    stats = IngestStats()
    out = validate_decisions(
        c, [_decision(symbols=["rerank_async", _AWS_KEY])],
        CodeMemoryConfig(), redactor=PIIRedactor(), stats=stats,
    )
    assert out[0]["symbols"] == ["rerank_async"]  # key dropped, clean kept
    assert stats.credentials_dropped == 1


def test_capture_credential_source_ref_replaced_not_dropped(tmp_path, monkeypatch):
    """F2: a credential in evidence source_ref (model/agent free text) swaps
    to the deterministic commit default — the atom itself survives."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    clean_text = "the default executor is shared across the process"
    decisions = [{
        "title": "Use a single-worker executor",
        "decision": "run the CE rerank on one dedicated worker",
        "rationale": "a shared pool oversubscribes the cross-encoder",
        "why_question": "why not the default executor?",
        "symbols": ["rerank_async"],
        "files": ["a.py"],
        "evidence": [{"kind": "constraint", "text": clean_text,
                      "source_ref": f"see {_SK_KEY}"}],
        "confidence": "high",
    }]
    stats = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert stats.credentials_dropped == 1   # the source_ref content, counted
    assert stats.evidence == 1              # the atom itself survived
    import kumiho
    project = kumiho.get_project("p-decisions")
    ev_item = next(i for (s, k), i in project.items.items() if k == "code_evidence")
    emeta = ev_item.get_latest_revision().metadata
    assert emeta["statement"] == clean_text
    assert _SK_KEY not in emeta["source_ref"]
    assert emeta["source_ref"].startswith("commit:")  # deterministic default


def test_validate_decisions_stats_none_does_not_crash():
    """F3: a direct call without stats must not AttributeError on the
    credential counter — the gate constructs a throwaway IngestStats."""
    from kumiho_memory.privacy import PIIRedactor

    c = CommitInfo("h", "a", "2026-07-10T00:00:00+09:00", "s", "b",
                   files=["a.py"])
    out = validate_decisions(
        c, [_decision(decision=f'api_key = "{_SK_KEY}"')],
        CodeMemoryConfig(), redactor=PIIRedactor(),  # stats omitted
    )
    assert out == []  # credential decision still dropped, no crash


# ---------------- overall capture deadline + partial success (#98) ----------
#
# The keyless capture tool must return before the MCP client's request timeout
# fires (a live capture hit -32001 against the cloud).  capture_decisions bounds
# the WHOLE tool with an overall deadline and checkpoints per decision: on
# deadline it returns a PARTIAL result naming what landed + what's pending +
# how to finish.  A retry with the same args must NOT duplicate (get-or-create
# nodes + existence-checked edges).


class _FakeClock:
    """A frozen monotonic clock the test advances explicitly, so a deadline
    can be tripped at an exact point in the batch with no wall-clock races."""

    def __init__(self, t=0.0):
        self.t = t

    def monotonic(self):
        return self.t


def _two_distinct_decisions():
    """Two decisions with disjoint titles + evidence so 'which landed' is
    unambiguous when a deadline cuts the batch in half."""
    return [
        {"title": "Decision ONE single worker",
         "decision": "run the CE rerank on one dedicated worker",
         "rationale": "a shared pool oversubscribes the cross-encoder",
         "why_question": "why one worker?",
         "symbols": ["rerank_async"],
         "files": ["a.py"],
         "evidence": [{"kind": "measurement",
                       "text": "concurrency-4 collapsed to ~1 effective (decision one)"}],
         "confidence": "high"},
        {"title": "Decision TWO cache policy",
         "decision": "invalidate the cache on every write",
         "rationale": "readers saw stale rows after a write",
         "why_question": "why invalidate eagerly?",
         "symbols": ["cache_put"],
         "files": ["a.py"],
         "evidence": [{"kind": "constraint",
                       "text": "stale reads observed after write (decision two)"}],
         "confidence": "high"},
    ]


def _count_items(project, kind):
    return [i for (s, k), i in project.items.items()
            if k == kind and i.get_latest_revision() is not None]


async def _run_synchronously(fn, *, timeout, label="", on_timeout=None,
                             on_error=None):
    """Drop-in for run_bounded_in_thread that runs fn INLINE (no daemon
    thread, no clock of its own).  Used only by the deadline tests so the
    fake clock's sole consumer is capture_decisions' own deadline math —
    the real bounded runner is exercised by the other tests."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 — mirror the helper's swallow-and-report
        return on_error


def test_capture_deadline_env_parsing(monkeypatch):
    """The overall budget comes from KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE
    (the deprecated KUMIHO_MEMORY_CODE_CAPTURE_DEADLINE is still honored as a
    fallback); blanks/garbage/non-positive fall back to the (sub-MCP-timeout)
    default."""
    import kumiho_memory.code_capture as cc

    monkeypatch.delenv("KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE", raising=False)
    monkeypatch.delenv("KUMIHO_MEMORY_CODE_CAPTURE_DEADLINE", raising=False)
    assert cc._capture_deadline() == cc._CAPTURE_DEADLINE_DEFAULT
    assert cc._CAPTURE_DEADLINE_DEFAULT < 60.0  # comfortably under the MCP timeout

    monkeypatch.setenv("KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE", "30")
    assert cc._capture_deadline() == 30.0
    monkeypatch.setenv("KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE", "12.5")
    assert cc._capture_deadline() == 12.5
    for bad in ("", "   ", "abc", "0", "-5"):
        monkeypatch.setenv("KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE", bad)
        assert cc._capture_deadline() == cc._CAPTURE_DEADLINE_DEFAULT

    # Legacy fallback: the deprecated CODE name is read only when the new name
    # is unset, and the new name wins when both are present.
    monkeypatch.delenv("KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE", raising=False)
    monkeypatch.setenv("KUMIHO_MEMORY_CODE_CAPTURE_DEADLINE", "22")
    assert cc._capture_deadline() == 22.0
    monkeypatch.setenv("KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE", "33")
    assert cc._capture_deadline() == 33.0  # new name wins over legacy


def test_capture_generous_deadline_full_success_unchanged(tmp_path, monkeypatch):
    """A generous deadline is a no-op: every decision + the marker land, and
    the result carries NO partial fields (fast-path shape unchanged)."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    monkeypatch.setenv("KUMIHO_MEMORY_DECISIONS_CAPTURE_DEADLINE", "600")

    stats = asyncio.run(capture_decisions(
        str(repo), _two_distinct_decisions(), commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert stats.partial is False
    assert stats.decisions == 2 and stats.evidence == 2 and not stats.errors
    d = stats.as_dict()
    for k in ("partial", "landed_krefs", "pending_decisions", "resume"):
        assert k not in d  # omitted on the non-partial path

    import kumiho
    project = kumiho.get_project("p-decisions")
    assert len(_count_items(project, "code_decision")) == 2
    assert len(_count_items(project, "code_commit")) == 1  # marker written


def test_capture_idempotent_double_run_no_duplicates(tmp_path, monkeypatch):
    """Idempotency proof: capturing the SAME commit+decisions twice produces an
    IDENTICAL graph — no new items, revisions, or edges.  This is the retry-
    safety the partial result promises (get-or-create nodes, edge prechecks)."""
    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    decisions = _two_distinct_decisions()

    s1 = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert s1.partial is False and s1.decisions == 2 and not s1.errors

    import kumiho
    project = kumiho.get_project("p-decisions")
    items_before = {(s, k): len(i.revisions) for (s, k), i in project.items.items()}
    edges_before = len(_FAKE.edges)
    revs_before = len(_FAKE.revs)

    # second run, byte-identical args → converge, add nothing
    s2 = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert s2.decisions == 0 and s2.evidence == 0  # nothing newly written
    assert not s2.errors and s2.partial is False

    items_after = {(s, k): len(i.revisions) for (s, k), i in project.items.items()}
    assert items_after == items_before          # no new items, no new revisions
    assert len(_FAKE.edges) == edges_before      # no duplicate edges
    assert len(_FAKE.revs) == revs_before        # no duplicate revisions


def test_capture_partial_on_deadline_mid_batch(tmp_path, monkeypatch):
    """Deadline reached BETWEEN decisions: decision one lands whole, decision
    two is pending, and the marker is withheld — never a half-written decision.
    The result reports landed krefs, the pending title, and an idempotent-retry
    resume token."""
    import kumiho_memory.code_capture as cc

    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")

    clock = _FakeClock(0.0)
    monkeypatch.setattr(cc.time, "monotonic", clock.monotonic)
    # Inline the bounded runner so the fake clock's ONLY consumer is the
    # deadline math in capture_decisions (no daemon-thread poll race).
    monkeypatch.setattr(cc, "run_bounded_in_thread", _run_synchronously)

    orig_write = cc._sync_write_decision
    seen = {"n": 0}

    def jump_after_first(*a, **kw):
        payload = orig_write(*a, **kw)     # decision fully written first
        seen["n"] += 1
        if seen["n"] == 1:
            clock.t = 10_000.0             # now blow past the 45s deadline
        return payload

    monkeypatch.setattr(cc, "_sync_write_decision", jump_after_first)

    stats = asyncio.run(capture_decisions(
        str(repo), _two_distinct_decisions(), commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))

    d = stats.as_dict()
    assert d["partial"] is True
    assert len(d["landed_krefs"]) == 1 and d["landed_krefs"][0]
    assert d["pending_decisions"] == ["Decision TWO cache policy"]
    assert "commit_ref=" in d["resume"] and "idempotent" in d["resume"]
    assert stats.decisions == 1 and stats.evidence == 1

    import kumiho
    project = kumiho.get_project("p-decisions")
    dec_items = _count_items(project, "code_decision")
    assert len(dec_items) == 1                       # only decision one landed
    ev_texts = [i.get_latest_revision().metadata["statement"]
                for i in _count_items(project, "code_evidence")]
    assert ev_texts == ["concurrency-4 collapsed to ~1 effective (decision one)"]
    assert _count_items(project, "code_commit") == []  # marker withheld (partial)
    # decision one is WHOLE — its anchor + evidence edges are present
    drev = dec_items[0].get_latest_revision()
    etypes = {e.edge_type for e in drev.edges}
    assert "IMPLEMENTED_IN" in etypes and "MOTIVATED_BY" in etypes


def test_capture_partial_then_retry_completes_without_duplicates(tmp_path, monkeypatch):
    """The resume contract end-to-end: a deadline-truncated capture followed by
    a re-call with the SAME args finishes the batch and writes the marker, with
    NO duplicate of the decision that already landed."""
    import kumiho_memory.code_capture as cc

    repo = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    cfg = CodeMemoryConfig(repo="testrepo")
    decisions = _two_distinct_decisions()

    clock = _FakeClock(0.0)
    monkeypatch.setattr(cc.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cc, "run_bounded_in_thread", _run_synchronously)

    orig_write = cc._sync_write_decision
    state = {"n": 0, "jump": True}

    def jump_after_first(*a, **kw):
        payload = orig_write(*a, **kw)
        state["n"] += 1
        if state["jump"] and state["n"] == 1:
            clock.t = 10_000.0
        return payload

    monkeypatch.setattr(cc, "_sync_write_decision", jump_after_first)

    # first pass — deadline truncates after decision one
    s1 = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert s1.partial is True and s1.pending_decisions == ["Decision TWO cache policy"]

    import kumiho
    project = kumiho.get_project("p-decisions")
    assert len(_count_items(project, "code_decision")) == 1

    # retry with the SAME args, deadline no longer trips
    clock.t = 0.0
    state["jump"] = False
    s2 = asyncio.run(capture_decisions(
        str(repo), decisions, commit_ref="HEAD",
        project_name="p-decisions", config=cfg,
    ))
    assert s2.partial is False and not s2.errors
    # decision two written now; decision one converged (no new decision node)
    assert s2.decisions == 1

    dec_items = _count_items(project, "code_decision")
    assert len(dec_items) == 2                        # both decisions, no dup
    assert len(_count_items(project, "code_commit")) == 1  # marker now written
    # no decision node was duplicated — each carries exactly one revision
    assert all(len(i.revisions) == 1 for i in dec_items)
