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
    build_packet,
    derive_repo_id,
    enumerate_commits,
    ingest_repo,
    prefilter,
    validate_decisions,
    _truncate_file_diff,
)


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
