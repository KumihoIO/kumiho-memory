"""Session mining (Decision Memory Phase 2) — docs/SESSION_MINING_DESIGN.md.

Unit coverage for the deterministic core (salience/packet/parser/validate),
the correlation matrix (enrich vs standalone — conjunction, never lexical
alone), the additive-enrichment constitution, dedup layers, idempotency,
crash safety, force, and the conversation bridge.  No LLM, server, Redis, or
network — everything is stubbed; git is a local throwaway repo.
"""

import asyncio
import json
import subprocess
import sys
import types

from kumiho_memory.code_decisions import (
    CodeMemoryConfig,
    EDGE_DERIVED_FROM,
    EDGE_DISCUSSED_IN,
    EDGE_IMPLEMENTED_IN,
    EDGE_MOTIVATED_BY,
    KIND_COMMIT,
    KIND_DECISION,
    KIND_EVIDENCE,
    KIND_SESSION,
    session_slug,
)
from kumiho_memory.code_session import (
    SessionMineStats,
    _salience,
    build_chunks,
    correlate,
    mine_session,
    parse_conversation_markdown,
    select_messages,
    validate_session_decisions,
)


# ---------------- git fixture ----------------


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
    (repo / "rerank.py").write_text("# one worker on purpose\nX = 1\n",
                                    encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m",
         "fix: offload rerank to a dedicated single-worker executor\n\n"
         "Inline CE blocked the loop under the harness.")
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True)
    return repo, out.stdout.strip()


# ---------------- fake kumiho (mirrors test_code_capture) ----------------


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

    def set_attribute(self, key, value):
        self.metadata[key] = value
        return True

    def get_item(self):
        return _FAKE.item_of[self.kref.uri]


class _MemItem:
    def __init__(self, slug, kind, project):
        self.slug, self.kind = slug, kind
        self.kref = types.SimpleNamespace(uri=f"kref://{project}/{kind}/{slug}")
        self.revisions = []
        self.deprecated = False

    def get_latest_revision(self):
        return self.revisions[-1] if self.revisions else None

    def create_revision(self, metadata=None, number=0):
        rev = _MemRev(f"{self.kref.uri}@{len(self.revisions) + 1}", metadata or {})
        self.revisions.append(rev)
        _FAKE.revs[rev.kref.uri] = rev
        _FAKE.item_of[rev.kref.uri] = self
        return rev

    def set_deprecated(self, status):
        self.deprecated = bool(status)


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
        def create_revision(self, item_kref, metadata=None, number=0,
                            embedding_text=""):
            _FAKE.embedding_texts.append(embedding_text)
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


class _StubAdapter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def chat(self, *, messages, model, system="", max_tokens=1024,
                   json_mode=False):
        self.calls += 1
        return self.payload


class _StubRedis:
    def __init__(self, messages):
        self._messages = messages

    async def get_messages(self, project, session_id, limit=1000):
        return {"messages": list(self._messages)}


def _msg(role, content, ts="2026-07-11T10:00:00+00:00"):
    return {"role": role, "content": content, "timestamp": ts}


def _cfg(**kw):
    return CodeMemoryConfig(repo="repo", **kw)


# ---------------- salience / selection / chunks ----------------


def test_salience_decision_words_fire():
    m = {"role": "assistant", "content":
         "we decided to use a dedicated executor instead of to_thread"}
    assert _salience(m, 0) >= 3


def test_salience_korean_lexicon():
    m = {"role": "user", "content": "draws 가설은 기각, 가산 원칙 채택으로 결정"}
    assert _salience(m, 0) >= 3


def test_salience_stack_trace_penalized():
    trace = "\n".join(["  at 0x7f3a9b2c%d ()" % i for i in range(12)])
    assert _salience({"role": "tool", "content": trace}, 0) < 2


def test_selection_keeps_assent_neighbor_and_frame():
    messages = [_msg("user", f"filler {i}") for i in range(10)]
    messages[5] = _msg("assistant",
                       "let's go with a dedicated executor in `rerank.py` "
                       "instead of to_thread")
    messages[6] = _msg("user", "yes do it")
    from kumiho_memory.code_session import _normalize_messages

    norm = _normalize_messages(messages)
    kept = select_messages(norm, _cfg())
    kept_idx = {m["index"] for m in kept if m["index"] >= 0}
    assert {5, 6} <= kept_idx           # signal + assent witness
    assert {0, 1, 7, 8, 9} <= kept_idx  # frame: first 2 + last 3
    assert any(m["index"] < 0 and "elided" in m["content"] for m in kept)


def test_selection_truncates_long_messages_head_tail():
    long = "decided: keep this line\n" + ("x" * 5000) + "\ntail sentence kept"
    from kumiho_memory.code_session import _normalize_messages

    norm = _normalize_messages([_msg("assistant", long)])
    kept = select_messages(norm, _cfg())
    body = next(m["content"] for m in kept if m["index"] >= 0)
    assert len(body) <= 800 + len("\n[...]\n")
    assert body.startswith("decided: keep this line")
    assert body.endswith("tail sentence kept")
    assert "[...]" in body


def test_chunks_frame_and_no_message_split():
    from kumiho_memory.code_session import _normalize_messages

    norm = _normalize_messages(
        [_msg("user", f"decided option {i} instead of the other") for i in range(6)]
    )
    packets = build_chunks("s1", select_messages(norm, _cfg()), _cfg())
    assert packets and packets[0].startswith("session s1, chunk 1/")
    assert "[m0 user 2026-07-11T10:00:00+00:00]" in packets[0]


def test_chunks_cap_keeps_high_density():
    cfg = _cfg(session_chunk_chars=400, session_max_chunks=2)
    msgs = []
    for i in range(12):
        text = ("decided measured 12% instead of rather than rejected"
                if i >= 8 else "hello filler chatter nothing")
        msgs.append(_msg("user", text * 3))
    from kumiho_memory.code_session import _normalize_messages

    selected = select_messages(_normalize_messages(msgs), cfg)
    packets = build_chunks("s1", selected, cfg)
    assert len(packets) <= 2
    assert any("decided" in p for p in packets)


def test_selection_is_deterministic():
    msgs = [_msg("user", "we decided to go with plan B instead of A"),
            _msg("assistant", "measured 42 ms on the bench")]
    from kumiho_memory.code_session import _normalize_messages

    a = build_chunks("s", select_messages(_normalize_messages(msgs), _cfg()), _cfg())
    b = build_chunks("s", select_messages(_normalize_messages(msgs), _cfg()), _cfg())
    assert a == b


# ---------------- artifact parser (golden round-trip) ----------------


def test_artifact_parser_golden_roundtrip():
    from kumiho_memory.memory_manager import UniversalMemoryManager as MemoryManager

    messages = [
        {"role": "user", "content": "why is the rerank slow?",
         "timestamp": "2026-07-10T14:00:00+00:00"},
        {"role": "assistant",
         "content": "two options:\n- to_thread\n- dedicated executor",
         "timestamp": "2026-07-10T14:01:00+00:00", "metadata": {}},
        {"role": "user", "content": "go with the dedicated executor"},
    ]
    md = MemoryManager._build_conversation_markdown(
        messages=messages, title="T", session_id="s1", summary="sum",
        topics=["perf"], user_lines_out=[], assistant_lines_out=[],
    )
    parsed = parse_conversation_markdown(md)
    assert [m["role"] for m in parsed] == ["user", "assistant", "user"]
    assert parsed[0]["timestamp"] == "2026-07-10T14:00:00+00:00"
    assert parsed[1]["content"] == "two options:\n- to_thread\n- dedicated executor"
    assert parsed[2]["timestamp"] == ""


# ---------------- validate ----------------


def _candidate(**kw):
    d = {
        "title": "Use a dedicated single-worker executor",
        "decision": "Run the CE rerank on a dedicated single-worker executor",
        "rationale": "the default executor is shared",
        "why_question": "why a dedicated executor?",
        "symbols": ["rerank_async"],
        "files": [],
        "mentioned_commits": [],
        "alternatives": [],
        "evidence": [],
        "settled_by_message": 1,
        "status_hint": "unknown",
        "confidence": "high",
    }
    d.update(kw)
    return d


def _validate(cands, packets, repo_path=".", tracked=None, cfg=None,
              redactor=None):
    stats = SessionMineStats()
    out = validate_session_decisions(
        cands, packets=packets, repo_path=repo_path,
        tracked_files=tracked or set(), config=cfg or _cfg(),
        redactor=redactor, stats=stats,
    )
    return out, stats


def test_validate_verbatim_normalized_pass_and_fabrication_drop():
    packet = "[m1 user] we  rejected   to_thread because the pool is shared"
    ok = _candidate(evidence=[{"kind": "constraint",
                               "text": "we rejected to_thread because the pool is shared",
                               "text_en": "", "message_index": 1}])
    fab = _candidate(title="Other decision entirely about caching",
                     decision="use LRU cache for embeddings",
                     evidence=[{"kind": "constraint",
                                "text": "quantum flux capacitor overheated badly",
                                "text_en": "", "message_index": 1}])
    out, stats = _validate([ok, fab], [packet])
    assert out[0]["evidence"]                  # normalized substring match
    assert not out[1]["evidence"]              # pure fabrication dropped
    assert stats.evidence_dropped_verbatim == 1


def test_validate_containment_relaxation_boundary():
    packet = ("[m1 user] the default executor is shared and a 32-thread pool "
              "oversubscribes the cross-encoder model badly")
    # paraphrase reusing >=60% of the quote's tokens
    near = _candidate(alternatives=[{
        "option": "to_thread", "verdict": "rejected",
        "quote": "the default executor is shared and oversubscribes the cross-encoder",
        "quote_en": "", "message_index": 1}])
    far = _candidate(title="B", alternatives=[{
        "option": "to_thread", "verdict": "rejected",
        "quote": "completely different words about unrelated topics entirely here",
        "quote_en": "", "message_index": 1}])
    out, stats = _validate([near, far], [packet])
    assert out[0]["alternatives"]
    assert not out[1]["alternatives"]


def test_validate_sha_and_file_ground_truth(tmp_path):
    repo, sha = _make_repo(tmp_path)
    packet = f"[m1 user] committed as {sha[:7]} touching rerank.py and ghost.py"
    cand = _candidate(
        mentioned_commits=[sha[:7], "deadbeef99", "not-a-sha"],
        files=["rerank.py", "ghost.py"],
    )
    out, _ = _validate([cand], [packet], repo_path=str(repo),
                       tracked={"rerank.py"})
    assert list(out[0]["verified_commits"].values()) == [sha]
    assert out[0]["files"] == ["rerank.py"]   # ghost dropped -> anchor story shrinks


def test_validate_low_confidence_without_payload_dropped():
    empty = _candidate(confidence="low")
    with_alt = _candidate(
        title="B", confidence="low",
        alternatives=[{"option": "x", "verdict": "rejected",
                       "quote": "we rejected x because it is slow",
                       "quote_en": "", "message_index": 1}],
    )
    packet = "[m1 user] we rejected x because it is slow"
    out, _ = _validate([empty, with_alt], [packet])
    assert [d["title"] for d in out] == ["B"]


def test_validate_credential_atom_drops_but_session_survives():
    from kumiho_memory.privacy import PIIRedactor

    packet = ('[m1 user] set api_key = "sk-abcdefghijklmnopqrstuvwx" then '
              "we rejected to_thread because the pool is shared")
    cand = _candidate(
        evidence=[
            {"kind": "constraint",
             "text": 'set api_key = "sk-abcdefghijklmnopqrstuvwx" then',
             "text_en": "", "message_index": 1},
            {"kind": "constraint",
             "text": "we rejected to_thread because the pool is shared",
             "text_en": "", "message_index": 1},
        ],
    )
    out, stats = _validate([cand], [packet], redactor=PIIRedactor())
    assert len(out) == 1                       # the SESSION survives
    assert len(out[0]["evidence"]) == 1        # only the credential atom died
    assert stats.credentials_dropped == 1


def test_validate_credential_symbol_dropped_session_parity():
    """Issue #99 F1 parity: symbols reach metadata AND the embedding text on
    the session path too — a credential-bearing symbol entry drops (counted),
    clean identifiers pass untouched (no PII redaction for identifiers)."""
    from kumiho_memory.privacy import PIIRedactor

    packet = "[m1 user] we use rerank_async for the offload"
    cand = _candidate(symbols=["rerank_async", "AKIAIOSFODNN7EXAMPLE"])
    out, stats = _validate([cand], [packet], redactor=PIIRedactor())
    assert len(out) == 1                       # the decision survives
    assert out[0]["symbols"] == ["rerank_async"]
    assert stats.credentials_dropped == 1


# ---------------- correlate (the matrix) ----------------


def _seed_commit_decision(project, repo, sha, *, title, decision, symbols,
                          decided_at, file="rerank.py"):
    """Wire a commit-mined decision exactly as code_capture leaves it."""
    from kumiho_memory.code_decisions import (
        anchor_slug, commit_slug, decision_slug,
    )

    d_item = project.create_item(decision_slug(title, decided_at),
                                 KIND_DECISION, "")
    d_rev = d_item.create_revision({
        "title": title, "decision": decision, "symbols": symbols,
        "decided_at": decided_at, "status": "active", "commit_hash": sha,
    })
    try:
        a_item = project.create_item(anchor_slug(repo, file), "code_anchor", "")
        a_rev = a_item.create_revision({"repo": repo, "path": file})
    except Exception:  # shared anchor hub — second decision reuses it
        a_item = project.get_item(anchor_slug(repo, file), "code_anchor")
        a_rev = a_item.get_latest_revision()
    d_rev.create_edge(a_rev, EDGE_IMPLEMENTED_IN, {"commit_hash": sha})
    m_item = project.create_item(commit_slug(repo, sha), KIND_COMMIT, "")
    m_rev = m_item.create_revision({
        "repo": repo, "hash": sha, "subject": title, "decisions_count": "1",
    })
    d_rev.create_edge(m_rev, EDGE_DERIVED_FROM, {})
    return d_rev


def test_correlate_sha_path_needs_sanity_floor(monkeypatch, tmp_path):
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    _seed_commit_decision(project, "repo", sha,
                          title="Use a dedicated single-worker executor",
                          decision="run rerank on one dedicated worker executor",
                          symbols="rerank_async",
                          decided_at="2026-07-10T12:00:00+00:00")
    cfg = _cfg()
    hit = correlate(project, cfg, "repo",
                    _candidate(verified_commits={sha[:7]: sha}),
                    "2026-07-11T12:00:00+00:00")
    assert hit is not None and hit["correlation"] == "sha"

    # a misquoted sha whose decision shares no vocabulary must NOT merge
    miss = correlate(project, cfg, "repo",
                     _candidate(title="Migrate embeddings to bge-m3",
                                decision="switch embedding backend entirely",
                                symbols=["bge"],
                                verified_commits={sha[:7]: sha}),
                     "2026-07-11T12:00:00+00:00")
    assert miss is None


def test_correlate_anchor_needs_conjunction_and_window(monkeypatch, tmp_path):
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    _seed_commit_decision(project, "repo", sha,
                          title="Use a dedicated single-worker executor",
                          decision="run the CE rerank on a dedicated single-worker executor",
                          symbols="rerank_async",
                          decided_at="2026-07-10T12:00:00+00:00")
    cfg = _cfg()
    base = _candidate(files=["rerank.py"], verified_commits={})

    # anchor + lex + symbol overlap -> ENRICH
    assert correlate(project, cfg, "repo", dict(base),
                     "2026-07-11T12:00:00+00:00") is not None
    # symbol removed and lex < blind 0.5 -> STANDALONE
    nosym = dict(base, symbols=["unrelated_symbol"],
                 title="Dedicated executor for reranking work",
                 decision="use one dedicated executor for the heavy rerank")
    hit = correlate(project, cfg, "repo", nosym, "2026-07-11T12:00:00+00:00")
    if hit is not None:  # only via the strong-lexical escape hatch
        assert hit["overlap"] >= cfg.correlate_jaccard_blind
    # outside the 14-day window -> STANDALONE (same file, different era)
    assert correlate(project, cfg, "repo", dict(base),
                     "2026-09-01T12:00:00+00:00") is None
    # lexical similarity ALONE (no sha, no anchor) can never merge
    assert correlate(project, cfg, "repo",
                     _candidate(files=[], verified_commits={}),
                     "2026-07-11T12:00:00+00:00") is None


def test_correlate_multiple_targets_picks_best_single(monkeypatch, tmp_path):
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    _seed_commit_decision(project, "repo", sha,
                          title="Use a dedicated single-worker executor",
                          decision="run the CE rerank on a dedicated single-worker executor",
                          symbols="rerank_async",
                          decided_at="2026-07-10T12:00:00+00:00")
    _seed_commit_decision(project, "repo", "f" * 40,
                          title="Cap executor queue depth",
                          decision="bound the dedicated executor queue for rerank",
                          symbols="rerank_async",
                          decided_at="2026-07-09T12:00:00+00:00",
                          file="rerank.py")
    hit = correlate(project, _cfg(), "repo",
                    _candidate(files=["rerank.py"]),
                    "2026-07-11T12:00:00+00:00")
    assert hit is not None
    assert "single-worker" in hit["rev"].metadata["title"]  # best lex, exactly one


# ---------------- end-to-end mining (fake graph + stub LLM) ----------------


def _payload(decisions):
    return json.dumps({"decisions": decisions})


def _session_messages(sha=None):
    quote = ("we considered asyncio.to_thread and rejected it because the "
             "default executor is shared")
    msgs = [
        _msg("user", "the CE rerank is blocking the event loop under the harness"),
        _msg("assistant", "two options: asyncio.to_thread, or a dedicated executor"),
        _msg("user", quote),
        _msg("assistant",
             "agreed — dedicated single-worker executor in rerank.py"
             + (f", committing as {sha[:7]}" if sha else "")),
    ]
    return msgs, quote


def _enrich_payload(quote, sha):
    return _payload([{
        "title": "Use a dedicated single-worker executor",
        "decision": "run the CE rerank on a dedicated single-worker executor",
        "rationale": "the default executor is shared",
        "why_question": "why not asyncio.to_thread for the rerank offload?",
        "symbols": ["rerank_async"],
        "files": ["rerank.py"],
        "mentioned_commits": [sha[:7]],
        "alternatives": [{
            "option": "asyncio.to_thread", "verdict": "rejected",
            "quote": quote, "quote_en": "", "message_index": 2,
        }],
        "evidence": [],
        "settled_by_message": 3,
        "status_hint": "committed",
        "confidence": "high",
    }])


def _mine(repo, adapter, session_id="s1", **kw):
    return asyncio.run(mine_session(
        session_id, project_name="p-code", repo_path=str(repo),
        config=_cfg(), adapter=adapter, model="stub", **kw,
    ))


def test_enrichment_is_additive_and_idempotent(monkeypatch, tmp_path):
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    target = _seed_commit_decision(
        project, "repo", sha,
        title="Use a dedicated single-worker executor",
        decision="run the CE rerank on a dedicated single-worker executor",
        symbols="rerank_async", decided_at="2026-07-11T09:00:00+00:00",
    )
    meta_before = dict(target.metadata)
    revcount_before = len(target.get_item().revisions)

    msgs, quote = _session_messages(sha)
    adapter = _StubAdapter(_enrich_payload(quote, sha))
    stats = _mine(repo, adapter, messages=msgs, conversation_kref="")

    # -- both failure directions at once: enrich, don't create
    assert stats.decisions_enriched == 1 and stats.decisions_created == 0
    # -- constitution: the target decision is byte-identical
    assert target.metadata == meta_before
    assert len(target.get_item().revisions) == revcount_before
    # -- the session-only alternative landed as a MOTIVATED_BY evidence atom
    ev_edges = target.get_edges(edge_type_filter=EDGE_MOTIVATED_BY, direction=0)
    assert len(ev_edges) == 1
    ev = kumiho.get_revision(ev_edges[0].target_kref.uri)
    assert ev.metadata["evidence_kind"] == "rejected_alternative"
    assert ev.metadata["alternative"] == "asyncio.to_thread"
    assert ev.metadata["source_ref"].startswith("session:s1#m")
    assert ev_edges[0].metadata["correlation"] == "sha"
    # -- provenance hub: enriched decision + new evidence hang on the marker
    marker = project.get_item(session_slug("repo", "s1"), KIND_SESSION)
    assert marker.get_latest_revision() is not None
    # -- idempotency: complete marker -> zero LLM, zero new nodes
    edges_before = len(_FAKE.edges)
    stats2 = _mine(repo, adapter, messages=msgs)
    assert stats2.skipped_marker and stats2.llm_calls == 0
    assert len(_FAKE.edges) == edges_before


def test_standalone_capture_origin_session(monkeypatch, tmp_path):
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)

    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [
        _msg("user", "should we migrate embeddings to bge-m3 now?"),
        _msg("assistant", quote),
        _msg("user", "yes, agreed"),
    ]
    adapter = _StubAdapter(_payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer the bge-m3 migration until after the release cycle",
        "rationale": "release cycle first",
        "why_question": "why was the bge-m3 migration deferred?",
        "symbols": [], "files": ["rerank.py"], "mentioned_commits": [],
        "alternatives": [{
            "option": "migrate now", "verdict": "deferred",
            "quote": quote, "quote_en": "", "message_index": 1,
        }],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))
    stats = _mine(repo, adapter, messages=msgs)
    assert stats.decisions_created == 1 and stats.decisions_enriched == 0

    import kumiho

    project = kumiho.get_project("p-code")
    d_item = next(i for (s, k), i in project.items.items()
                  if k == KIND_DECISION)
    rev = d_item.get_latest_revision()
    assert rev.metadata["origin"] == "session"
    assert rev.metadata["session_id"] == "s1"
    assert rev.metadata["commit_hash"] == ""
    assert rev.metadata["decided_at"] == "2026-07-11T10:00:00+00:00"
    # anchor: role=mentioned on the ls-files-verified path
    a_edges = rev.get_edges(edge_type_filter=EDGE_IMPLEMENTED_IN, direction=0)
    assert a_edges and a_edges[0].metadata["role"] == "mentioned"
    # rejected alternative joins the embedding (doc2query in reverse)
    assert any("Rejected alternatives" in t for t in _FAKE.embedding_texts)
    # provenance: decision -> session marker
    der = rev.get_edges(edge_type_filter=EDGE_DERIVED_FROM, direction=0)
    assert der and "session" in der[0].target_kref.uri


def test_session_marker_credential_line_uses_redacted_placeholder(
        monkeypatch, tmp_path):
    """Issue #117 part 2 / F4 analog (PR #111): when the first user line bears a
    credential it is dropped, and the resulting session_line — which feeds the
    session MARKER's embedding_text — must be the '[redacted]' placeholder, NOT
    '' (an empty embedding_text makes write_revision fall back to embedding ALL
    metadata: hash/author/bookkeeping vector pollution)."""
    from kumiho_memory.privacy import PIIRedactor

    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)

    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [
        _msg("user", "should we migrate embeddings to bge-m3 now? set "
                     'api_key = "sk-abcdefghij0123456789ABCDEF" first'),
        _msg("assistant", quote),
        _msg("user", "yes, agreed"),
    ]
    adapter = _StubAdapter(_payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer the bge-m3 migration until after the release cycle",
        "rationale": "release cycle first",
        "why_question": "why was the bge-m3 migration deferred?",
        "symbols": [], "files": ["rerank.py"], "mentioned_commits": [],
        "alternatives": [{
            "option": "migrate now", "verdict": "deferred",
            "quote": quote, "quote_en": "", "message_index": 1,
        }],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))
    stats = _mine(repo, adapter, messages=msgs, redactor=PIIRedactor())
    assert stats.credentials_dropped >= 1

    import kumiho

    project = kumiho.get_project("p-code")
    marker = project.get_item(session_slug("repo", "s1"), KIND_SESSION)
    assert marker.get_latest_revision() is not None
    # The marker embedding_text is EXACTLY the placeholder — proving the
    # dropped session_line took the client-level (explicit-text) write path,
    # NOT write_revision's embed-all-metadata fallback (which appends nothing
    # here).  The decision embedding merely CONTAINS the placeholder substring,
    # so an exact-match count isolates the marker write.
    assert _FAKE.embedding_texts.count("[redacted]") >= 1


def test_evidence_dedup_slug_convergence_and_near_dup(monkeypatch, tmp_path):
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    target = _seed_commit_decision(
        project, "repo", sha,
        title="Use a dedicated single-worker executor",
        decision="run the CE rerank on a dedicated single-worker executor",
        symbols="rerank_async", decided_at="2026-07-11T09:00:00+00:00",
    )
    # pre-existing commit evidence NODE (from some other commit's mining),
    # not yet attached to this decision — the session repeats it verbatim.
    stmt = "inline CE blocked the loop under the harness"
    from kumiho_memory.code_decisions import evidence_slug

    ev_item = project.create_item(evidence_slug(stmt), KIND_EVIDENCE, "")
    ev_rev = ev_item.create_revision({
        "statement": stmt, "evidence_kind": "incident",
        "source_ref": f"commit:{sha[:12]}",
    })
    commit_meta_before = dict(ev_rev.metadata)

    msgs, quote = _session_messages(sha)
    msgs.append(_msg("assistant", stmt))  # session repeats the commit sentence
    payload = json.loads(_enrich_payload(quote, sha))
    payload["decisions"][0]["evidence"] = [
        # layer 1: identical sentence -> slug convergence, metadata untouched
        {"kind": "incident", "text": stmt, "text_en": "", "message_index": 4},
        # layer 2: near-duplicate of the rejected-alternative quote -> cut
        {"kind": "constraint",
         "text": "we considered asyncio.to_thread and rejected it because "
                 "the default executor is shared",
         "text_en": "", "message_index": 2},
    ]
    stats = _mine(repo, _StubAdapter(json.dumps(payload)), messages=msgs)

    assert stats.decisions_enriched == 1
    # layer 1: the identical sentence converged on the existing node —
    # its metadata is untouched (source_ref still commit:*), only the
    # missing MOTIVATED_BY edge was added
    assert ev_rev.metadata == commit_meta_before
    # the alternative was the only NEW node; the near-dup restatement of
    # the alternative's quote was cut (layer 2)
    assert stats.evidence_added == 1
    assert stats.evidence_dropped_dup == 1
    ev_edges = target.get_edges(edge_type_filter=EDGE_MOTIVATED_BY, direction=0)
    assert len(ev_edges) == 2  # session alternative + converged commit atom
    targets = {e.target_kref.uri for e in ev_edges}
    assert ev_rev.kref.uri in targets


def test_bridge_and_bridge_only_reconciliation(monkeypatch, tmp_path):
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    # a conversation revision in a DIFFERENT project (cross-project kref)
    conv_project = kumiho.create_project("p")
    conv_item = conv_project.create_item("conv-1", "memory", "")
    conv_rev = conv_item.create_revision({"title": "the chat"})

    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [_msg("user", "migrate now?"), _msg("assistant", quote),
            _msg("user", "yes, agreed")]
    adapter = _StubAdapter(_payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer the bge-m3 migration until after the release cycle",
        "rationale": "release first",
        "why_question": "why deferred?",
        "symbols": [], "files": [], "mentioned_commits": [],
        "alternatives": [{"option": "migrate now", "verdict": "deferred",
                          "quote": quote, "quote_en": "", "message_index": 1}],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))

    # mined WITHOUT a kref: no bridge yet
    stats = _mine(repo, adapter, messages=msgs)
    assert stats.bridged == 0

    # kref arrives later -> bridge-only pass, zero LLM
    stats2 = _mine(repo, adapter, messages=msgs,
                   conversation_kref=conv_rev.kref.uri)
    assert stats2.skipped_marker and stats2.llm_calls == 0
    assert stats2.bridged == 1
    project = kumiho.get_project("p-code")
    d_item = next(i for (s, k), i in project.items.items() if k == KIND_DECISION)
    edges = d_item.get_latest_revision().get_edges(
        edge_type_filter=EDGE_DISCUSSED_IN, direction=0,
    )
    assert len(edges) == 1
    assert edges[0].target_kref.uri == conv_rev.kref.uri
    assert edges[0].metadata["session_id"] == "s1"
    # marker remembers the kref now
    marker = project.get_item(session_slug("repo", "s1"), KIND_SESSION)
    assert marker.get_latest_revision().metadata["conversation_kref"] == conv_rev.kref.uri
    # re-run with same kref: fully idempotent
    stats3 = _mine(repo, adapter, messages=msgs,
                   conversation_kref=conv_rev.kref.uri)
    assert stats3.skipped_marker and stats3.bridged == 0


def test_marker_written_last_on_crash(monkeypatch, tmp_path):
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)

    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [_msg("user", "migrate?"), _msg("assistant", quote),
            _msg("user", "yes, agreed")]
    adapter = _StubAdapter(_payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer until after the release cycle",
        "rationale": "release first", "why_question": "why deferred?",
        "symbols": [], "files": [], "mentioned_commits": [],
        "alternatives": [{"option": "now", "verdict": "deferred",
                          "quote": quote, "quote_en": "", "message_index": 1}],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))

    # crash injection: the SESSION marker item creation explodes
    original = _MemProject.create_item

    def exploding(self, slug, kind, parent_path=""):
        if kind == KIND_SESSION:
            raise RuntimeError("boom")
        return original(self, slug, kind, parent_path)

    monkeypatch.setattr(_MemProject, "create_item", exploding)
    stats = _mine(repo, adapter, messages=msgs)
    assert stats.errors  # write failed, loudly
    import kumiho

    project = kumiho.get_project("p-code")
    assert (session_slug("repo", "s1"), KIND_SESSION) not in project.items

    # next run retries cleanly and completes
    monkeypatch.setattr(_MemProject, "create_item", original)
    stats2 = _mine(repo, adapter, messages=msgs)
    assert not stats2.errors
    assert (session_slug("repo", "s1"), KIND_SESSION) in project.items


def test_remine_on_message_growth(monkeypatch, tmp_path):
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)

    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [_msg("user", "migrate?"), _msg("assistant", quote),
            _msg("user", "yes, agreed")]
    adapter = _StubAdapter(_payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer until after the release cycle",
        "rationale": "release first", "why_question": "why deferred?",
        "symbols": [], "files": [], "mentioned_commits": [],
        "alternatives": [{"option": "now", "verdict": "deferred",
                          "quote": quote, "quote_en": "", "message_index": 1}],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))
    _mine(repo, adapter, messages=msgs)
    calls_after_first = adapter.calls

    # session grew past the delta -> full re-mine (slug convergence dedups)
    grown = msgs + [_msg("user", f"more chatter {i}") for i in range(12)]
    stats = _mine(repo, adapter, messages=grown)
    assert not stats.skipped_marker
    assert adapter.calls > calls_after_first


def test_force_deprecates_only_this_sessions_decisions(monkeypatch, tmp_path):
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    commit_decision = _seed_commit_decision(
        project, "repo", sha,
        title="Use a dedicated single-worker executor",
        decision="run the CE rerank on a dedicated single-worker executor",
        symbols="rerank_async", decided_at="2026-07-11T09:00:00+00:00",
    )

    quote = ("we considered asyncio.to_thread and rejected it because the "
             "default executor is shared")
    standalone_quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs, _ = _session_messages(sha)
    msgs.append(_msg("assistant", standalone_quote))
    payload = json.loads(_enrich_payload(quote, sha))
    payload["decisions"].append({
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer the bge-m3 migration until after the release cycle",
        "rationale": "release first", "why_question": "why deferred?",
        "symbols": [], "files": [], "mentioned_commits": [],
        "alternatives": [{"option": "now", "verdict": "deferred",
                          "quote": standalone_quote, "quote_en": "",
                          "message_index": 4}],
        "evidence": [], "settled_by_message": 4,
        "status_hint": "uncommitted", "confidence": "high",
    })
    adapter = _StubAdapter(json.dumps(payload))
    stats = _mine(repo, adapter, messages=msgs)
    assert stats.decisions_enriched == 1 and stats.decisions_created == 1

    # force re-mine: the session-origin decision cycles through deprecation
    # and comes back active; the ENRICHED commit decision is untouched.
    stats2 = _mine(repo, adapter, messages=msgs, force=True)
    assert stats2.deprecated == 1
    assert commit_decision.metadata.get("status") == "active"
    assert not commit_decision.get_item().deprecated
    d_item = next(i for (s, k), i in project.items.items()
                  if k == KIND_DECISION and i.get_latest_revision()
                  .metadata.get("origin") == "session")
    assert not d_item.deprecated                    # restored on convergence
    assert d_item.get_latest_revision().metadata["status"] == "active"
    assert len(d_item.revisions) >= 2               # fresh revision written


def test_no_transcript_is_a_loud_error(monkeypatch, tmp_path):
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    stats = asyncio.run(mine_session(
        "s-gone", project_name="p-code", repo_path=str(repo),
        config=_cfg(), adapter=_StubAdapter("{}"), model="stub",
        redis_buffer=_StubRedis([]),
    ))
    assert stats.errors and "cannot be mined" in stats.errors[0]


def test_redis_source_loads_messages(monkeypatch, tmp_path):
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [_msg("user", "migrate?"), _msg("assistant", quote),
            _msg("user", "yes, agreed")]
    adapter = _StubAdapter(_payload([]))
    stats = asyncio.run(mine_session(
        "s1", project_name="p-code", repo_path=str(repo),
        config=_cfg(), adapter=adapter, model="stub",
        redis_buffer=_StubRedis(msgs), memory_project="p",
    ))
    assert stats.source == "redis"
    assert stats.messages_seen == 3
    assert stats.llm_calls == 1  # zero decisions is a valid answer


def test_gate_off_manager_returns_error(monkeypatch):
    monkeypatch.delenv("KUMIHO_MEMORY_CODE", raising=False)
    from kumiho_memory.code_decisions import code_automine_enabled, code_memory_enabled

    assert not code_memory_enabled()
    assert not code_automine_enabled()
    # AUTOMINE alone must not open the chain (double opt-in)
    monkeypatch.setenv("KUMIHO_MEMORY_CODE_AUTOMINE", "1")
    assert not code_automine_enabled()
    monkeypatch.setenv("KUMIHO_MEMORY_CODE", "1")
    assert code_automine_enabled()


def test_resolve_tracked_path_unique_suffix():
    """Models abbreviate repo-relative paths (live-measured) — a UNIQUE
    suffix match resolves, ambiguity or absence drops."""
    from kumiho_memory.code_session import _resolve_tracked_path

    tracked = {
        "python/kumiho-memory/kumiho_memory/recall_rerank.py",
        "python/kumiho-memory/tests/test_recall_rerank.py",
        "a/dup.py", "b/dup.py",
    }
    assert (_resolve_tracked_path("kumiho_memory/recall_rerank.py", tracked)
            == "python/kumiho-memory/kumiho_memory/recall_rerank.py")
    assert (_resolve_tracked_path("recall_rerank.py", tracked)
            == "python/kumiho-memory/kumiho_memory/recall_rerank.py")
    assert (_resolve_tracked_path("python/kumiho-memory/kumiho_memory/recall_rerank.py",
                                  tracked)
            == "python/kumiho-memory/kumiho_memory/recall_rerank.py")
    assert _resolve_tracked_path("dup.py", tracked) == ""      # ambiguous
    assert _resolve_tracked_path("ghost.py", tracked) == ""    # absent
    assert _resolve_tracked_path("", tracked) == ""


def test_chunk_cap_pins_frame_chunks():
    """§2.3: over the chunk cap, the first and last chunks carry the session
    frame (goal statement / closing agreement) and are pinned — density
    eviction competes only over the middle.  Whole-chunk drop of the
    low-density opening chunk was a reviewed spec violation."""
    cfg = _cfg(session_chunk_chars=120, session_max_chunks=2)
    texts = (
        ["opening goal statement, low salience filler chatter here"]
        + [f"decided measured 12% instead of rather than rejected option {i}"
           for i in range(6)]
        + ["closing agreement, also low salience"]
    )
    selected = [
        {"index": i, "role": "user", "timestamp": "", "content": t,
         "score": 5 if 0 < i < 7 else 0}
        for i, t in enumerate(texts)
    ]
    packets = build_chunks("s1", selected, cfg)
    assert len(packets) == 2
    assert "opening goal statement" in packets[0]   # frame pinned
    assert "closing agreement" in packets[-1]       # frame pinned


def _bge_payload(quote):
    return _payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer the bge-m3 migration until after the release cycle",
        "rationale": "release first", "why_question": "why deferred?",
        "symbols": [], "files": [], "mentioned_commits": [],
        "alternatives": [{"option": "migrate now", "verdict": "deferred",
                          "quote": quote, "quote_en": "", "message_index": 1}],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }])


class _StubSessionRedactor:
    """Deterministic PIIRedactor stand-in: rewrites the email, raises on
    the key pattern (privacy.reject_credentials raises; verified)."""

    def anonymize_summary(self, text):
        return text.replace("bob@example.com", "[EMAIL]")

    def reject_credentials(self, text):
        if "sk-" in text:
            raise ValueError("credential detected")


def test_session_line_is_redacted_before_storage(monkeypatch, tmp_path):
    """§5.1: session_line feeds STORED embedding_text (marker + standalone
    decisions) — the raw first user turn must never reach the store."""
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)

    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [_msg("user", "bob@example.com asked: migrate the embeddings now?"),
            _msg("assistant", quote),
            _msg("user", "yes, agreed")]
    stats = _mine(repo, _StubAdapter(_bge_payload(quote)), messages=msgs,
                  redactor=_StubSessionRedactor())
    assert stats.decisions_created == 1
    assert _FAKE.embedding_texts
    assert all("bob@example.com" not in t for t in _FAKE.embedding_texts)
    assert any("[EMAIL]" in t for t in _FAKE.embedding_texts)


def test_session_line_with_credential_is_dropped(monkeypatch, tmp_path):
    """A pasted key in the first user turn drops the session_line atom
    (per-atom screen) — the session itself still mines."""
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)

    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [_msg("user", "my key is sk-abcdefghijklmnop — also, migrate now?"),
            _msg("assistant", quote),
            _msg("user", "yes, agreed")]
    stats = _mine(repo, _StubAdapter(_bge_payload(quote)), messages=msgs,
                  redactor=_StubSessionRedactor())
    assert stats.decisions_created == 1          # the SESSION survives
    assert stats.credentials_dropped >= 1
    assert all("sk-" not in t for t in _FAKE.embedding_texts)


def test_correlate_anchored_dead_zone_regression(monkeypatch, tmp_path):
    """S2 regression: lex moved to FULL prose (honest pairs ~0.26), so the
    anchored floor moved to the same measured basis — a same-decision pair
    below the draft's 0.35 floor must still enrich via the anchor + symbol
    conjunction instead of splitting into a duplicate standalone."""
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    target = _seed_commit_decision(
        project, "repo", sha,
        title="offload fastembed cross-encoder rerank",
        decision="offload the cross-encoder rerank processing to a dedicated "
                 "single-worker executor instead of blocking the event loop",
        symbols="rerank_async", decided_at="2026-07-10T12:00:00+00:00",
    )
    target.metadata["rationale"] = "inline CE blocked the loop under the harness"
    target.metadata["why_question"] = "why is the rerank offloaded?"

    cand = _candidate(
        title="Use a Dedicated ThreadPoolExecutor for Inference Serialization",
        decision="Commit to using a dedicated single-worker "
                 "ThreadPoolExecutor for inference serialization.",
        rationale="The default executor oversubscribes the cross-encoder "
                  "with a shared 32-thread pool, causing issues with the "
                  "event loop.",
        why_question="Why did we choose a dedicated ThreadPoolExecutor "
                     "instead of asyncio.to_thread?",
        symbols=["rerank_async"],
        files=["rerank.py"],
        verified_commits={},
    )
    # Pin the scenario inside the dead zone the draft floor created.
    from kumiho_memory.code_session import _lex_text
    from kumiho_memory.relations import _jaccard, _tokens

    lex = _jaccard(_tokens(_lex_text(cand)), _tokens(_lex_text(target.metadata)))
    assert 0.20 <= lex < 0.35, f"pair drifted out of the dead zone: {lex:.2f}"
    hit = correlate(project, _cfg(), "repo", cand, "2026-07-11T12:00:00+00:00")
    assert hit is not None and hit["correlation"] == "anchored"


def test_correlate_lex_uses_full_prose(monkeypatch, tmp_path):
    """Dogfood-calibrated: a live same-decision pair measured 0.14 on
    title+decision but 0.26 on full prose — rationale carries the shared
    why-vocabulary, so correlation must read it."""
    repo, sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    import kumiho

    project = kumiho.create_project("p-code")
    # commit side, styled like the real cfec845 extraction
    from kumiho_memory.code_decisions import commit_slug, decision_slug

    d_item = project.create_item(
        decision_slug("offload fastembed cross-encoder rerank",
                      "2026-07-10T12:00:00+00:00"), KIND_DECISION, "")
    d_rev = d_item.create_revision({
        "title": "offload fastembed cross-encoder rerank",
        "decision": "offload the cross-encoder rerank processing to a "
                    "dedicated single-worker executor instead of blocking "
                    "the event loop",
        "rationale": "inline CE blocked the loop under the harness",
        "why_question": "why is the rerank offloaded?",
        "symbols": "", "decided_at": "2026-07-10T12:00:00+00:00",
        "status": "active",
    })
    m_item = project.create_item(commit_slug("repo", sha), KIND_COMMIT, "")
    m_rev = m_item.create_revision({"repo": "repo", "hash": sha,
                                    "decisions_count": "1"})
    d_rev.create_edge(m_rev, EDGE_DERIVED_FROM, {})

    # session side, styled like the real live extraction (different title)
    cand = _candidate(
        title="Use a Dedicated ThreadPoolExecutor for Inference Serialization",
        decision="Commit to using a dedicated single-worker "
                 "ThreadPoolExecutor for inference serialization.",
        rationale="The default executor oversubscribes the cross-encoder "
                  "with a shared 32-thread pool, causing issues with the "
                  "event loop.",
        why_question="Why did we choose a dedicated ThreadPoolExecutor "
                     "instead of asyncio.to_thread?",
        verified_commits={sha[:7]: sha},
    )
    hit = correlate(project, _cfg(), "repo", cand, "2026-07-11T12:00:00+00:00")
    assert hit is not None and hit["correlation"] == "sha"


def test_delta_remine_refreshes_marker_message_count(monkeypatch, tmp_path):
    """A growth-triggered re-mine MUST rewrite the marker with the new
    message_count — otherwise the stale count keeps tripping the delta
    threshold and every subsequent mine re-pays full LLM cost forever."""
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    quote = "defer the bge-m3 migration because the release cycle comes first"
    base = [_msg("user", "migrate?"), _msg("assistant", quote),
            _msg("user", "yes, agreed")]
    adapter = _StubAdapter(_payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer until after the release cycle",
        "rationale": "release first", "why_question": "why deferred?",
        "symbols": [], "files": [], "mentioned_commits": [],
        "alternatives": [{"option": "now", "verdict": "deferred",
                          "quote": quote, "quote_en": "", "message_index": 1}],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))
    _mine(repo, adapter, messages=base)                       # marker mc=3
    grown = base + [_msg("user", f"more chatter {i}") for i in range(12)]
    _mine(repo, adapter, messages=grown)                      # re-mine (mc->15)
    calls = adapter.calls
    # third mine of the SAME grown session must now converge to skip
    stats3 = _mine(repo, adapter, messages=grown)
    assert stats3.skipped_marker
    assert adapter.calls == calls                             # zero new LLM

    import kumiho
    project = kumiho.get_project("p-code")
    marker = project.get_item(session_slug("repo", "s1"), KIND_SESSION)
    assert marker.get_latest_revision().metadata["message_count"] == str(len(grown))


def test_force_does_not_self_correlate_anchored_session_decision(monkeypatch, tmp_path):
    """--force pre-pass deprecates this session's decisions; an ANCHORED
    standalone one must not then rediscover its own deprecated self via its
    IMPLEMENTED_IN edge and enrich onto it (the ENRICH branch never
    un-deprecates → permanent retirement).  It must fall back to standalone
    and be restored."""
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    quote = "we deferred the rewrite because the executor pool is shared"
    msgs = [_msg("user", "should we rewrite rerank.py?"),
            _msg("assistant", quote),
            _msg("user", "yes, decided: keep the single-worker executor")]
    adapter = _StubAdapter(_payload([{
        "title": "Keep the single-worker executor in rerank.py",
        "decision": "keep the dedicated single-worker executor for the rerank",
        "rationale": "the default executor pool is shared",
        "why_question": "why keep the single-worker executor?",
        "symbols": ["rerank_async"], "files": ["rerank.py"],
        "mentioned_commits": [],
        "alternatives": [{"option": "rewrite", "verdict": "deferred",
                          "quote": quote, "quote_en": "", "message_index": 1}],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))
    stats = _mine(repo, adapter, messages=msgs)               # anchored standalone
    assert stats.decisions_created == 1

    stats2 = _mine(repo, adapter, messages=msgs, force=True)
    # went back to standalone (not enriched onto its deprecated self)
    assert stats2.decisions_created == 1 and stats2.decisions_enriched == 0

    import kumiho
    project = kumiho.get_project("p-code")
    d_item = next(i for (s, k), i in project.items.items()
                  if k == KIND_DECISION and i.get_latest_revision()
                  .metadata.get("origin") == "session")
    assert not d_item.deprecated                              # restored
    assert d_item.get_latest_revision().metadata["status"] == "active"


class _FailingAdapter:
    """LLM stub whose structuring call always raises."""
    def __init__(self):
        self.calls = 0

    async def chat(self, *, messages, model, system="", max_tokens=1024,
                   json_mode=False):
        self.calls += 1
        raise RuntimeError("model unavailable")


def test_chunk_structuring_failure_withholds_marker_for_retry(monkeypatch, tmp_path):
    """A failed LLM chunk is a FAILURE, not a zero-decision verdict: the
    session marker is withheld so the next run re-mines instead of
    permanently skipping the un-judged chunk."""
    repo, _sha = _make_repo(tmp_path)
    _install_fake_kumiho(monkeypatch)
    quote = "defer the bge-m3 migration because the release cycle comes first"
    msgs = [_msg("user", "migrate?"), _msg("assistant", quote),
            _msg("user", "yes, agreed")]

    stats = _mine(repo, _FailingAdapter(), messages=msgs)
    assert any("structuring failed" in e for e in stats.errors)
    assert not stats.skipped_marker

    import kumiho
    project = kumiho.get_project("p-code")
    # no complete marker was written -> the session can be retried
    marker = project.items.get((session_slug("repo", "s1"), KIND_SESSION))
    assert marker is None or marker.get_latest_revision() is None

    # a subsequent run with a working adapter re-mines and completes
    good = _StubAdapter(_payload([{
        "title": "Defer the bge-m3 embedding migration",
        "decision": "defer until after the release cycle",
        "rationale": "release first", "why_question": "why deferred?",
        "symbols": [], "files": [], "mentioned_commits": [],
        "alternatives": [{"option": "now", "verdict": "deferred",
                          "quote": quote, "quote_en": "", "message_index": 1}],
        "evidence": [], "settled_by_message": 2,
        "status_hint": "uncommitted", "confidence": "high",
    }]))
    stats2 = _mine(repo, good, messages=msgs)
    assert not stats2.skipped_marker and stats2.decisions_created == 1
    marker = project.get_item(session_slug("repo", "s1"), KIND_SESSION)
    assert marker.get_latest_revision() is not None


def test_parse_claude_transcript_jsonl(tmp_path):
    """The plugin SessionEnd input surface: Claude Code transcript JSONL ->
    mine_session messages (text kept, tool_use marked, tool_result/system-
    reminder dropped, timestamps preserved)."""
    from kumiho_memory.code_session import parse_claude_transcript

    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join([
        json.dumps({"timestamp": "2026-07-11T10:00:00Z",
                    "message": {"role": "user", "content": "why single-worker executor?"}}),
        json.dumps({"message": {"role": "assistant", "content": [
            {"type": "text", "text": "because the pool is shared"},
            {"type": "tool_use", "name": "Bash"},
            {"type": "tool_result", "content": "verbose noise dropped"},
        ]}}),
        json.dumps({"message": {"role": "user", "content": "<system-reminder>ignore me</system-reminder>"}}),
        json.dumps({"type": "summary", "role": "system", "content": "not a turn"}),
        "not json at all",
        json.dumps({"message": {"role": "user", "content": "yes, decided"}}),
    ]), encoding="utf-8")

    msgs = parse_claude_transcript(str(path))
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["timestamp"] == "2026-07-11T10:00:00Z"
    assert "because the pool is shared" in msgs[1]["content"]
    assert "*[tool: Bash]*" in msgs[1]["content"]
    assert "verbose noise" not in msgs[1]["content"]     # tool_result dropped
    assert msgs[2]["content"] == "yes, decided"          # system-reminder skipped


def test_parse_claude_transcript_missing_file():
    from kumiho_memory.code_session import parse_claude_transcript
    assert parse_claude_transcript("/nonexistent/transcript.jsonl") == []
