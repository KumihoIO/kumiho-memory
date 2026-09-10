# -*- coding: utf-8 -*-
"""Cross-session decision-continuity harness (kumiho-memory#26, deterministic tier).

Drives the REAL ``kumiho_memory.mcp_tools`` handlers — ingest, reflect,
decompose, consolidate, engage — across genuinely separate sessions, over an
in-memory stand-in for the ``kumiho`` SDK. No LLM key, no network, no server.

What is real here: the tool handlers, ``UniversalMemoryManager``, the Redis
working-memory buffer (over ``tests/fakes.FakeRedis``), the ontology write
path, the shared supersession protocol, grounding ripple, valid-time
demotion, graph-augmented recall and context composition. What is faked: the
graph server (``FakeGraph`` — items, revisions, edges, a token-overlap search)
and the summarizer/redactor (stubs; every path exercised is keyless).

Arms
----
``B`` (memory)   — every session shares ONE long-term graph.
``A`` (no memory) — every session gets an EMPTY graph; the final task sees
                    nothing the earlier sessions wrote. Working memory (Redis)
                    still survives a *process restart* in both arms, because a
                    restart is not a new conversation.
``Bprime`` (decoy) is an agent-tier arm: a generic authoritative brief cannot
change a scripted agent, so this tier records it as not run.

Session boundaries
------------------
``new_host_session`` — new session id, new Redis, new manager process.
``process_restart``  — same session id and Redis, new manager process (the
                       ``mcp_tools`` module singleton is reset and its recall
                       dedup cache cleared, as a freshly spawned server would
                       start).

This tier produces *protocol assertions*: it proves what the memory layer
returns and marks, never that a model makes better decisions. The scripted
agent reads structured outcomes (``option:*`` tags, or a scenario's declared
per-language signal mapping for typed facts) — no free-text keyword judging.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
import types
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import kumiho._text  # noqa: F401  (real submodules must be importable behind the fake)
from kumiho._text import slugify

from kumiho_memory import mcp_tools as mcp_tools_module
from kumiho_memory.memory_manager import UniversalMemoryManager
from kumiho_memory.redis_memory import RedisMemoryBuffer
from kumiho_memory.mcp_tools import (
    tool_memory_consolidate,
    tool_memory_decompose,
    tool_memory_engage,
    tool_memory_ingest,
    tool_memory_reflect,
)

from fakes import FakeRedis
from hosted_fakes import StubSummarizer
from kumiho_memory.privacy import PIIRedactor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "decision_continuity_v1.json"

OUTGOING, INCOMING, BOTH = 0, 1, 2
ARMS_RUN = ("B", "A")
ARMS_NOT_RUN = {"Bprime": "agent tier only — a decoy brief cannot change a scripted agent"}

_TOK = re.compile(r"\w+", re.UNICODE)

#: Function words carry no topic; a naive token-overlap fake would otherwise
#: "recall" an unrelated memory because both mention "the" (the negative
#: control). The real recall path is embedding/BM25 scored and never does.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "use", "used", "using", "which", "what", "how",
    "for", "with", "that", "this", "these", "those", "of", "to", "in", "on",
    "and", "or", "no", "not", "we", "our", "it", "its", "as", "at", "by",
    "should", "have", "has", "had", "will", "now",
})


def _tokens(text: str) -> set:
    toks = {t for t in _TOK.findall((text or "").casefold()) if len(t) >= 2}
    content = {t for t in toks if t not in _STOPWORDS and not (t.isascii() and len(t) < 3)}
    return content or toks


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _project_of(kref: str) -> str:
    if not kref.startswith("kref://"):
        return ""
    return kref[len("kref://"):].split("/", 1)[0]


# ---------------------------------------------------------------------------
# In-memory SDK fake
# ---------------------------------------------------------------------------


class FakeKref:
    def __init__(self, uri: str) -> None:
        self.uri = uri

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeKref({self.uri!r})"


class FakeEdge:
    def __init__(self, source: str, target: str, edge_type: str, metadata=None) -> None:
        self.source_kref = FakeKref(source)
        self.target_kref = FakeKref(target)
        self.edge_type = edge_type
        self.metadata = dict(metadata or {})


class FakeRevision:
    """A revision with metadata, tags, a timestamp and both edge directions."""

    def __init__(self, uri: str, metadata: Dict[str, Any], item: "FakeItem", created_at: str,
                 tags: Optional[List[str]] = None) -> None:
        self.kref = FakeKref(uri)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.tags: List[str] = list(tags or [])
        self.item = item
        self.created_at = created_at
        self.deprecated = False
        self._incoming: List[FakeEdge] = []
        self._outgoing: List[FakeEdge] = []

    # --- read side (grounding ripple, augmentation walk, materializer) ---
    def get_edges(self, edge_type_filter=None, direction=OUTGOING):
        if direction == INCOMING:
            pool = self._incoming
        elif direction == OUTGOING:
            pool = self._outgoing
        else:
            pool = self._incoming + self._outgoing
        return [e for e in pool if not edge_type_filter or e.edge_type == edge_type_filter]

    def get_item(self):
        return self.item

    def get_artifacts(self):
        return []

    # --- write side (supersession protocol, ripple, materializer) ---
    def create_edge(self, target, edge_type, metadata=None):
        edge = FakeEdge(self.kref.uri, target.kref.uri, edge_type, metadata)
        self._outgoing.append(edge)
        target._incoming.append(edge)
        return edge

    def set_metadata(self, md):
        self.metadata.update(md)
        return self

    def set_attribute(self, key, value):
        self.metadata[key] = value
        return True

    def tag(self, t):
        if t not in self.tags:
            self.tags.append(t)


class FakeItem:
    def __init__(self, graph: "FakeGraph", project: str, parent_path: str, slug: str, kind: str) -> None:
        self.graph = graph
        self.project = project
        self.parent_path = parent_path  # "/{project}/{space}" or "/{project}"
        self.slug = slug
        self.kind = kind
        self.kref = FakeKref(f"kref://{parent_path.strip('/')}/{slug}.{kind}")
        self.revisions: List[FakeRevision] = []
        self.deprecated = False

    def get_latest_revision(self):
        return self.revisions[-1] if self.revisions else None

    def create_revision(self, metadata=None, tags=None):
        uri = f"{self.kref.uri}?r={len(self.revisions) + 1}"
        rev = FakeRevision(uri, metadata or {}, self, self.graph.tick(), tags)
        self.revisions.append(rev)
        self.graph.revisions[uri] = rev
        return rev

    def set_deprecated(self, flag: bool = True):
        self.deprecated = bool(flag)


class FakeProject:
    def __init__(self, graph: "FakeGraph", name: str) -> None:
        self.graph = graph
        self.name = name
        self.items: Dict[Tuple[str, str, str], FakeItem] = {}
        self.spaces: set = set()

    def create_space(self, path: str):
        self.spaces.add(path)

    def create_item(self, slug: str, kind: str, parent_path: Optional[str] = None):
        parent = parent_path or f"/{self.name}"
        key = (parent, slug, kind)
        item = self.items.get(key)
        if item is None:
            item = FakeItem(self.graph, self.name, parent, slug, kind)
            self.items[key] = item
        return item

    def get_item(self, slug: str, kind: str, parent_path: Optional[str] = None):
        return self.items[(parent_path or f"/{self.name}", slug, kind)]


class FakeGraph:
    """The long-term store behind one arm. Stateful, inspectable, timed."""

    def __init__(self, label: str = "graph") -> None:
        self.label = label
        self.id = uuid.uuid4().hex[:8]
        self.projects: Dict[str, FakeProject] = {}
        self.revisions: Dict[str, FakeRevision] = {}
        self.calls: List[Dict[str, Any]] = []
        self.retrieve_latencies_ms: List[float] = []
        self.retrieve_error: Optional[str] = None
        self._clock = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._store_seq = 0
        self.sdk = self._make_sdk_module()

    # --- deterministic clock: every write is strictly newer than the last ---
    def tick(self) -> str:
        self._clock += timedelta(minutes=1)
        return self._clock.isoformat()

    # --- SDK module surface -------------------------------------------------
    def _make_sdk_module(self) -> types.ModuleType:
        fake = types.ModuleType("kumiho")
        fake.OUTGOING, fake.INCOMING, fake.BOTH = OUTGOING, INCOMING, BOTH
        fake.get_project = self.get_project
        fake.get_revision = self.get_revision
        fake.search = self.search
        mcp = types.ModuleType("kumiho.mcp_server")
        mcp.tool_memory_store = self.tool_memory_store
        mcp.tool_memory_retrieve = self.tool_memory_retrieve
        fake.mcp_server = mcp
        return fake

    def get_project(self, name: str):
        project = self.projects.get(name)
        if project is None:
            project = FakeProject(self, name)
            self.projects[name] = project
        return project

    def get_revision(self, uri: str):
        return self.revisions[uri]  # KeyError == "not found", as callers expect

    def search(self, query: str, context: str = "", kind: Optional[str] = None,
               include_revision_metadata: bool = False, **_: Any):
        self.calls.append({"op": "search", "context": context, "kind": kind})
        q = _tokens(query)
        parent = "/" + (context or "").strip("/")
        hits = []
        for project in self.projects.values():
            for item in project.items.values():
                if item.deprecated or (kind and item.kind != kind):
                    continue
                if context and item.parent_path != parent:
                    continue
                rev = item.get_latest_revision()
                if rev is None:
                    continue
                text = " ".join(str(rev.metadata.get(k, "")) for k in ("title", "summary", "claim", "decision"))
                score = _overlap(q, _tokens(text))
                if score > 0:
                    hits.append((score, item, rev))
        hits.sort(key=lambda h: (-h[0], h[2].created_at))
        return [SimpleNamespace(item=item, revision=rev, score=score) for score, item, rev in hits]

    # --- the memory store/retrieve pair (what kumiho.mcp_server exposes) ---
    def tool_memory_store(self, **kw: Any) -> Dict[str, Any]:
        self.calls.append({"op": "store", "project": kw.get("project"), "memory_type": kw.get("memory_type")})
        project = self.get_project(kw.get("project") or "CognitiveMemory")
        space = (kw.get("space_path") or kw.get("space_hint") or "").strip("/")
        parent = f"/{project.name}/{space}" if space else f"/{project.name}"
        kind = kw.get("memory_item_kind") or "conversation"
        title = kw.get("title") or kw.get("summary") or "memory"
        slug = slugify(title, hash_on_truncate=True) or f"memory-{self._store_seq}"
        item = None
        if kw.get("stack_revisions", True):
            item = project.items.get((parent, slug, kind))
        if item is None:
            self._store_seq += 1
            uniq = slug
            while (parent, uniq, kind) in project.items:
                uniq = f"{slug}-{self._store_seq}"
                self._store_seq += 1
            item = project.create_item(uniq, kind, parent_path=parent)
        metadata = {
            "title": kw.get("title", ""),
            "summary": kw.get("summary", ""),
            "type": kw.get("memory_type", "summary"),
            "memory_type": kw.get("memory_type", "summary"),
            "space": space,
        }
        for key in ("evidence_level", "source"):
            if kw.get(key):
                metadata[key] = kw[key]
        metadata.update({k: v for k, v in (kw.get("metadata") or {}).items() if v is not None})
        rev = item.create_revision(metadata=metadata, tags=kw.get("tags") or [])
        for src in kw.get("source_revision_krefs") or []:
            target = self.revisions.get(src)
            if target is not None:
                rev.create_edge(target, kw.get("edge_type") or "DERIVED_FROM")
        return {"success": True, "revision_kref": rev.kref.uri, "item_kref": item.kref.uri}

    def tool_memory_retrieve(self, project: str = "CognitiveMemory", query: str = "", limit: int = 5,
                             space_paths: Optional[List[str]] = None,
                             memory_types: Optional[List[str]] = None,
                             memory_item_kind: str = "conversation", **_: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            self.calls.append({"op": "retrieve", "project": project, "query": query})
            if self.retrieve_error:
                return {"error": self.retrieve_error}
            proj = self.projects.get(project)
            if proj is None:
                return {"revision_krefs": [], "scores": []}
            q = _tokens(query)
            wanted_spaces = [s.strip("/") for s in (space_paths or []) if s]
            scored = []
            for item in proj.items.values():
                if item.deprecated or item.kind != memory_item_kind:
                    continue
                rev = item.get_latest_revision()
                if rev is None:
                    continue
                if wanted_spaces and not any(
                    (rev.metadata.get("space") or "") == s or (rev.metadata.get("space") or "").startswith(s + "/")
                    for s in wanted_spaces
                ):
                    continue
                if memory_types and rev.metadata.get("type") not in memory_types:
                    continue
                text = f"{rev.metadata.get('title', '')} {rev.metadata.get('summary', '')}"
                score = _overlap(q, _tokens(text))
                if score > 0:
                    scored.append((score, rev))
            scored.sort(key=lambda s: (-s[0], s[1].created_at))
            top = scored[: max(int(limit or 5), 1)]
            return {
                "revision_krefs": [rev.kref.uri for _, rev in top],
                "scores": [round(score, 4) for score, _ in top],
            }
        finally:
            self.retrieve_latencies_ms.append((time.perf_counter() - started) * 1000.0)

    # --- SDK-side edge tool stand-in (kumiho_create_edge; SUPERSEDES withheld there) ---
    def create_edge(self, source_uri: str, target_uri: str, edge_type: str, metadata=None) -> None:
        if edge_type == "SUPERSEDES":
            raise ValueError("SUPERSEDES is not offered on the bare edge tool; use memory_decompose")
        self.revisions[source_uri].create_edge(self.revisions[target_uri], edge_type, metadata)

    # --- inspection helpers used by the checks ---
    def status_of(self, uri: str) -> str:
        rev = self.revisions.get(uri)
        return str((rev.metadata if rev else {}).get("status", "") or "")

    def conversation_count(self) -> int:
        return sum(1 for p in self.projects.values() for i in p.items.values() if i.kind == "conversation")

    def items_of_kind(self, kind: str) -> List[FakeItem]:
        return [i for p in self.projects.values() for i in p.items.values() if i.kind == kind]


@contextlib.contextmanager
def install_sdk(graph: FakeGraph):
    """Bind ``graph`` as the ``kumiho`` SDK for the duration of a block.

    Sets both ``sys.modules['kumiho']`` and ``sys.modules['kumiho.mcp_server']``
    (``tool_memory_reflect`` does ``from kumiho.mcp_server import ...``) and
    restores whatever was there afterwards — never pops, per repo convention.
    """
    saved = {name: sys.modules.get(name) for name in ("kumiho", "kumiho.mcp_server")}
    sys.modules["kumiho"] = graph.sdk
    sys.modules["kumiho.mcp_server"] = graph.sdk.mcp_server
    try:
        yield graph
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@contextlib.contextmanager
def continuity_env(tmp_root: Path):
    """The env the manager reads at construction, scoped to one run."""
    overrides = {
        "KUMIHO_MEMORY_ONTOLOGY": "1",
        "KUMIHO_MEMORY_AS_OF_RECALL": "1",
        "KUMIHO_MEMORY_ENTITY_PROMOTION": "0",
        "KUMIHO_MEMORY_DECISIONS": "0",
        "KUMIHO_MEMORY_ARTIFACT_ROOT": str(tmp_root / "artifacts"),
        "KUMIHO_RETRY_QUEUE_DIR": str(tmp_root / "retry"),
        "KUMIHO_FAILURE_LEDGER_DIR": str(tmp_root / "ledger"),
    }
    removed = ("KUMIHO_SESSION_ID", "CLAUDE_CODE_SESSION_ID", "KUMIHO_AUTH_TOKEN")
    saved = {k: os.environ.get(k) for k in list(overrides) + list(removed)}
    os.environ.update(overrides)
    for k in removed:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def load_fixture(path: Path = FIXTURE_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_sha256(path: Path = FIXTURE_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_ids(fixture: Optional[Dict[str, Any]] = None) -> List[str]:
    fixture = fixture or load_fixture()
    return [s["id"] for s in fixture["scenarios"]]


def scenario_by_id(scenario_id: str, fixture: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fixture = fixture or load_fixture()
    for s in fixture["scenarios"]:
        if s["id"] == scenario_id:
            return s
    raise KeyError(scenario_id)


# ---------------------------------------------------------------------------
# Scripted agent
# ---------------------------------------------------------------------------


def option_of(mem: Dict[str, Any], signals: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """Structured outcome of one recalled entry: an ``option:*`` tag, else the
    scenario's declared per-language signal mapping for typed facts."""
    for tag in mem.get("tags") or []:
        tag = str(tag)
        if tag.startswith("option:"):
            return tag[len("option:"):]
    if signals and ".fact" in (mem.get("kref") or ""):
        text = " ".join(str(mem.get(k, "") or "") for k in ("title", "summary", "content"))
        hits = [opt for opt, words in signals.items() if any(w in text for w in words)]
        if len(hits) == 1:
            return hits[0]
    return None


def scripted_decision(results: List[Dict[str, Any]], scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic stand-in for the answering agent.

    Rules (declared, not learned): ignore entries without a structured option;
    never use a superseded or as-of-excluded entry; if more than one option
    survives and any survivor is contested, the answer is ``contested``;
    otherwise take the best-scored survivor. ``reconsider`` is raised when the
    chosen entry is grounding-stale. No results → ``unknown`` (never a guess).
    """
    signals = scenario.get("option_signals")
    candidates = []
    for mem in results:
        opt = option_of(mem, signals)
        if opt is not None:
            candidates.append((mem, opt))
    usable = [(m, o) for m, o in candidates if not m.get("superseded") and not m.get("as_of_excluded")]
    if not usable:
        return {"outcome": "unknown", "kref": None, "reconsider": False,
                "considered": len(candidates), "ignored_superseded": len(candidates) - len(usable)}
    options = {o for _, o in usable}
    if len(options) > 1 and any(m.get("contested_by") for m, _ in usable):
        return {"outcome": "contested", "kref": None, "reconsider": False,
                "considered": len(candidates), "ignored_superseded": len(candidates) - len(usable),
                "options": sorted(options)}
    top_mem, top_opt = max(usable, key=lambda mo: (float(mo[0].get("score") or 0.0), str(mo[0].get("created_at") or "")))
    return {"outcome": top_opt, "kref": top_mem.get("kref"), "reconsider": bool(top_mem.get("grounding_stale")),
            "considered": len(candidates), "ignored_superseded": len(candidates) - len(usable)}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _Process:
    """One manager 'process' over a session's working memory and an arm's graph."""

    def __init__(self, project: str, redis: FakeRedis, graph: FakeGraph, artifact_root: Path,
                 graph_augmented: bool, fact_recall: bool) -> None:
        graph_cfg = None
        if graph_augmented:
            from kumiho_memory.graph_augmentation import GraphAugmentationConfig
            graph_cfg = GraphAugmentationConfig(fact_recall=fact_recall)
        self.manager = UniversalMemoryManager(
            project=project,
            redis_buffer=RedisMemoryBuffer(client=redis, redis_url="redis://continuity"),
            summarizer=StubSummarizer(),
            pii_redactor=PIIRedactor(),
            memory_store=graph.tool_memory_store,
            memory_retrieve=graph.tool_memory_retrieve,
            artifact_root=str(artifact_root),
            graph_augmentation=graph_cfg,
            entity_promotion=False,
            consolidation_threshold=10_000,
        )
        mcp_tools_module._manager = self.manager
        mcp_tools_module._recall_recent.clear()

    @staticmethod
    def teardown() -> None:
        mcp_tools_module._manager = None
        mcp_tools_module._recall_recent.clear()


def _subst(value: Any, binds: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return re.sub(r"\{(\w+)\}", lambda m: binds[m.group(1)], value)
    if isinstance(value, list):
        return [_subst(v, binds) for v in value]
    if isinstance(value, dict):
        return {k: _subst(v, binds) for k, v in value.items()}
    return value


def _messages(manager: UniversalMemoryManager, session_id: str) -> int:
    got = asyncio.run(manager.redis_buffer.get_messages(project=manager.project, session_id=session_id, limit=1000))
    return len(got.get("messages") or [])


def run_scenario(scenario: Dict[str, Any], arm: str, workdir: Optional[Path] = None,
                 shared_graph: Optional[FakeGraph] = None) -> Dict[str, Any]:
    """Run one scenario in one arm; return the structured result record."""
    if arm not in ARMS_RUN:
        raise ValueError(f"arm {arm!r} is not run by the deterministic tier ({ARMS_NOT_RUN})")
    workdir = Path(workdir or tempfile.mkdtemp(prefix="continuity-"))
    final = scenario["final"]
    graph_b = shared_graph or FakeGraph(label=f"{scenario['id']}:B")
    binds: Dict[str, str] = {}
    boundaries: List[Dict[str, Any]] = []
    restart_checks: List[bool] = []
    graphs_seen: List[FakeGraph] = []
    session_id = None
    redis: Optional[FakeRedis] = None
    graph: Optional[FakeGraph] = None
    process: Optional[_Process] = None
    project = scenario["project"]

    def _graph_for(boundary: str) -> FakeGraph:
        if arm == "B":
            return graph_b
        return FakeGraph(label=f"{scenario['id']}:A:{boundary}:{len(graphs_seen)}")

    with continuity_env(workdir):
        try:
            for sess in scenario["sessions"]:
                boundary = sess["boundary"]
                project = sess.get("project") or scenario["project"]
                if boundary == "new_host_session" or session_id is None:
                    session_id = f"{scenario['id']}:{arm}:{sess['name']}"
                    redis = FakeRedis()
                    graph = _graph_for(boundary)
                elif boundary == "process_restart":
                    graph = _graph_for(boundary)
                else:
                    raise ValueError(f"unknown boundary {boundary!r}")
                graphs_seen.append(graph)
                boundaries.append({"session": sess["name"], "boundary": boundary, "graph": graph.id})
                with install_sdk(graph):
                    process = _Process(project, redis, graph, workdir / arm / sess["name"],
                                       bool(final.get("graph_augmented")), bool(final.get("fact_recall")))
                    if boundary == "process_restart":
                        restart_checks.append(_messages(process.manager, session_id) > 0)
                    for step in sess["steps"]:
                        _run_step(step, session_id, binds, graph, process.manager)
                # leave the process installed; the final task runs in the last one

            assert process is not None and graph is not None and session_id is not None
            with install_sdk(graph):
                mcp_tools_module._manager = process.manager
                mcp_tools_module._recall_recent.clear()
                started = time.perf_counter()
                engage = tool_memory_engage({
                    "query": final["query"], "limit": final.get("limit", 5),
                    "graph_augmented": bool(final.get("graph_augmented")),
                })
                engage_ms = (time.perf_counter() - started) * 1000.0
                results = engage.get("results") or []
                decision = scripted_decision(results, scenario)
                as_of_decision = None
                if final.get("current_as_of"):
                    # The tool boundary has no as-of argument yet (#15/#28); the
                    # manager API is the contract this tier can exercise.
                    current = asyncio.run(process.manager.recall_memories(
                        final["query"], limit=final.get("limit", 5),
                        graph_augmented=bool(final.get("graph_augmented")),
                        query_time=datetime.fromisoformat(final["current_as_of"]),
                    ))
                    decision = scripted_decision(current, scenario)
                    results_current = current
                else:
                    results_current = results
                if final.get("as_of"):
                    historical = asyncio.run(process.manager.recall_memories(
                        final["query"], limit=final.get("limit", 5),
                        graph_augmented=bool(final.get("graph_augmented")),
                        query_time=datetime.fromisoformat(final["as_of"]),
                    ))
                    as_of_decision = scripted_decision(historical, scenario)
                other = None
                if final.get("also_in_other_project") and scenario.get("other_project"):
                    other_proc = _Process(scenario["other_project"], FakeRedis(), graph,
                                          workdir / arm / "other", False, False)
                    mcp_tools_module._recall_recent.clear()
                    other_engage = tool_memory_engage({"query": final["query"], "limit": final.get("limit", 5)})
                    other = {
                        "results": other_engage.get("results") or [],
                        "decision": scripted_decision(other_engage.get("results") or [], scenario),
                    }
                    mcp_tools_module._manager = process.manager
        finally:
            _Process.teardown()

    ctx = engage.get("context") or ""
    record: Dict[str, Any] = {
        "id": scenario["id"], "family": scenario["family"], "lang": scenario["lang"],
        "split": scenario["split"], "arm": arm, "project": scenario["project"],
        "boundaries": boundaries, "binds": dict(binds),
        "results": results_current, "engage_results": results, "context": ctx,
        "count": engage.get("count", 0), "backend_error": engage.get("backend_error"),
        "decision": decision, "as_of_decision": as_of_decision, "other_project": other,
        "restart_checks": restart_checks,
        "graph": graph, "graphs_seen": graphs_seen, "graph_b": graph_b,
        "metrics": {
            "engage_ms": round(engage_ms, 3),
            "retrieve_calls": sum(1 for c in graph.calls if c["op"] == "retrieve"),
            "retrieve_latencies_ms": [round(x, 3) for x in graph.retrieve_latencies_ms],
            "payload_bytes": len(json.dumps(engage, ensure_ascii=False, default=str).encode("utf-8")),
            "context_chars": len(ctx),
            "approx_tokens": engage.get("approx_tokens"),
        },
    }
    record["checks"] = evaluate_checks(record, scenario)
    record["passed"] = all(c["ok"] for c in record["checks"].values())
    return record


def _run_step(step: Dict[str, Any], session_id: str, binds: Dict[str, str], graph: FakeGraph,
              manager: UniversalMemoryManager) -> None:
    op = step["op"]
    if op == "ingest":
        tool_memory_ingest({"user_id": "bench-user", "context": "continuity",
                            "session_id": session_id, "message": step["message"]})
    elif op == "reflect":
        out = tool_memory_reflect({
            "session_id": session_id, "response": step["response"],
            "captures": step.get("captures") or [], "discover_edges": False,
        })
        bind = step.get("bind")
        krefs = out.get("stored_krefs") or []
        if isinstance(bind, list):
            if len(krefs) != len(bind):
                raise RuntimeError(f"reflect stored {len(krefs)} captures, bind expects {len(bind)}: {out}")
            binds.update(dict(zip(bind, krefs)))
        elif bind:
            if not krefs:
                raise RuntimeError(f"reflect stored nothing to bind {bind!r}: {out}")
            binds[bind] = krefs[0]
    elif op == "decompose":
        out = tool_memory_decompose(_subst({
            "kref": step["anchor"], "facts": step.get("facts") or [],
            "entities": step.get("entities") or [], "relations": step.get("relations") or [],
            "supersedes": step.get("supersedes") or [], "contradicts": step.get("contradicts") or [],
        }, binds))
        if out.get("errors"):
            raise RuntimeError(f"decompose failed: {out}")
    elif op == "link_depends_on":
        graph.create_edge(_subst(step["source"], binds), _subst(step["target"], binds),
                          "DEPENDS_ON", {"basis": "agent"})
    elif op == "consolidate":
        out = tool_memory_consolidate({"session_id": session_id, "summary": step["summary"]})
        if not out.get("success", True):
            raise RuntimeError(f"consolidate failed: {out}")
        if step.get("bind") and out.get("revision_kref"):
            binds[step["bind"]] = out["revision_kref"]
    elif op == "backend":
        graph.retrieve_error = step.get("retrieve_error")
    else:
        raise ValueError(f"unknown step op {op!r}")


# ---------------------------------------------------------------------------
# Checks — deterministic assertions on tool results + graph state
# ---------------------------------------------------------------------------


def _ck(ok: bool, detail: str = "") -> Dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def _na(reason: str) -> Dict[str, Any]:
    return {"ok": True, "detail": f"n/a: {reason}", "not_applicable": True}


def _by_kref(results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {m.get("kref"): m for m in results if m.get("kref")}


def evaluate_checks(rec: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    arm = rec["arm"]
    expect = scenario["expect"]
    want = expect.get(arm) or {}
    graph: FakeGraph = rec["graph"]
    binds = rec["binds"]
    results = rec["results"]
    by = _by_kref(results)
    decision = rec["decision"]
    out: Dict[str, Dict[str, Any]] = {}

    for name in expect["checks"]:
        if name == "outcome_matches":
            ok = decision["outcome"] == want.get("outcome")
            detail = f"outcome={decision['outcome']!r} expected={want.get('outcome')!r}"
            if ok and "reconsider" in want:
                ok = decision.get("reconsider") == want["reconsider"]
                detail += f" reconsider={decision.get('reconsider')}"
            if ok and want.get("outcome_as_of") is not None:
                got = (rec.get("as_of_decision") or {}).get("outcome")
                ok = got == want["outcome_as_of"]
                detail += f" as_of={got!r} expected={want['outcome_as_of']!r}"
            out[name] = _ck(ok, detail)

        elif name == "working_memory_survives_restart":
            checks = rec["restart_checks"]
            out[name] = _ck(bool(checks) and all(checks), f"restarts={checks}")

        elif name == "no_superseded_reuse":
            kref = decision.get("kref")
            if not kref:
                out[name] = _ck(True, "no memory was used")
            else:
                out[name] = _ck(graph.status_of(kref) != "superseded" and not by.get(kref, {}).get("superseded"),
                                f"used={kref} status={graph.status_of(kref)!r}")

        elif name == "no_fabrication_in_A":
            if arm != "A":
                out[name] = _na("memory arm")
            else:
                out[name] = _ck(decision["outcome"] == "unknown" and rec["count"] == 0,
                                f"outcome={decision['outcome']!r} count={rec['count']}")

        elif name == "no_fabrication":
            out[name] = _ck(decision["outcome"] == "unknown" and rec["count"] == 0,
                            f"outcome={decision['outcome']!r} count={rec['count']}")

        elif name == "backend_error_reported":
            out[name] = _ck(bool(rec.get("backend_error")), f"backend_error={rec.get('backend_error')!r}")

        elif name == "correction_retained":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                kref = binds.get(expect["correction"])
                mem = by.get(kref)
                ok = bool(mem) and option_of(mem) == want.get("outcome") and not mem.get("superseded")
                out[name] = _ck(ok, f"correction={kref} present={bool(mem)}")

        elif name == "old_decision_demoted":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                kref = binds.get(expect["replaced"])
                out[name] = _ck(graph.status_of(kref) == "superseded", f"{kref} status={graph.status_of(kref)!r}")

        elif name == "superseded_marked":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                unmarked = [k for k, m in by.items() if graph.status_of(k) == "superseded" and not m.get("superseded")]
                replaced = binds.get(expect.get("replaced", ""), "")
                present = replaced in by
                out[name] = _ck(not unmarked and present,
                                f"unmarked={unmarked} replaced_in_results={present}")

        elif name == "context_has_superseded_note":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                out[name] = _ck("[superseded:" in rec["context"], "note present" if "[superseded:" in rec["context"] else "note missing")

        elif name == "context_has_grounding_note":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                out[name] = _ck("[grounding stale" in rec["context"], "note present" if "[grounding stale" in rec["context"] else "note missing")

        elif name == "proposal_not_promoted":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                prop = scenario["proposal_option"]
                markers = scenario.get("proposal_markers") or []
                promoted_in_results = [m.get("kref") for m in results if option_of(m) == prop]
                typed = [i.kref.uri for i in graph.items_of_kind("decision")
                         if any(mk in " ".join(str(v) for v in (i.get_latest_revision().metadata.values() if i.get_latest_revision() else [])) for mk in markers)]
                captured = [i.kref.uri for i in graph.items_of_kind("conversation")
                            if i.get_latest_revision() is not None
                            and i.get_latest_revision().metadata.get("type") == "decision"
                            and any(mk in str(i.get_latest_revision().metadata.get("summary", "")) for mk in markers)]
                out[name] = _ck(not promoted_in_results and not typed and not captured,
                                f"in_results={promoted_in_results} typed_decisions={typed} decision_captures={captured}")

        elif name == "contradiction_unresolved":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                a, b = (binds.get(k) for k in expect["pair"])
                both_present = a in by and b in by
                neither_demoted = graph.status_of(a) != "superseded" and graph.status_of(b) != "superseded"
                marked = any(bool(by.get(k, {}).get("contested_by")) for k in (a, b))
                out[name] = _ck(both_present and neither_demoted and marked and decision["outcome"] == "contested",
                                f"present={both_present} neither_demoted={neither_demoted} marked={marked} outcome={decision['outcome']!r}")

        elif name == "newer_did_not_win":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                a, b = (binds.get(k) for k in expect["pair"])
                newer_opt = option_of(by.get(b, {})) if b in by else None
                out[name] = _ck(decision["outcome"] != newer_opt and graph.status_of(a) != "superseded",
                                f"outcome={decision['outcome']!r} newer={newer_opt!r} older_status={graph.status_of(a)!r}")

        elif name == "dependent_flagged":
            if arm != "B":
                out[name] = _na("no-memory arm")
            else:
                dep = binds.get(expect["dependent"])
                rev = graph.revisions.get(dep)
                in_graph = bool(rev) and str(rev.metadata.get("grounding_stale", "")).casefold() == "true"
                in_recall = bool(by.get(dep, {}).get("grounding_stale"))
                out[name] = _ck(in_graph and in_recall and decision.get("reconsider") is True,
                                f"graph={in_graph} recall={in_recall} reconsider={decision.get('reconsider')}")

        elif name == "as_of_differs":
            hist = (rec.get("as_of_decision") or {}).get("outcome")
            out[name] = _ck(arm != "B" or (hist not in (None, "unknown") and hist != decision["outcome"]),
                            f"as_of={hist!r} current={decision['outcome']!r}")

        elif name == "no_scope_leak":
            foreign = [k for k in by if _project_of(k) != scenario["project"]]
            out[name] = _ck(not foreign, f"foreign={foreign}")

        elif name == "other_project_isolated":
            other = rec.get("other_project")
            if arm != "B":
                out[name] = _na("no-memory arm")
            elif not other:
                out[name] = _ck(False, "other-project run missing")
            else:
                leaked = [m.get("kref") for m in other["results"] if _project_of(m.get("kref", "")) != scenario["other_project"]]
                want_other = scenario["final"]["also_in_other_project"]["outcome"]
                out[name] = _ck(not leaked and other["decision"]["outcome"] == want_other,
                                f"leaked={leaked} outcome={other['decision']['outcome']!r} expected={want_other!r}")

        else:
            raise ValueError(f"unknown check {name!r} in scenario {scenario['id']}")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pctl(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def _git_sha(repo_root: Path) -> Optional[str]:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def run_all(fixture: Optional[Dict[str, Any]] = None, arms: Tuple[str, ...] = ARMS_RUN,
            workdir: Optional[Path] = None, only: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    fixture = fixture or load_fixture()
    wanted = set(only) if only else None
    records = []
    for scenario in fixture["scenarios"]:
        if wanted and scenario["id"] not in wanted:
            continue
        for arm in arms:
            records.append(run_scenario(scenario, arm, workdir))
    return records


def build_report(records: List[Dict[str, Any]], fixture_path: Path = FIXTURE_PATH) -> Dict[str, Any]:
    """Machine-readable report: one run id linking fixture, product, results."""
    fixture = load_fixture(fixture_path)
    try:
        from importlib.metadata import version as _v
        product_version = _v("kumiho-memory")
    except Exception:  # noqa: BLE001
        product_version = None
    repo_root = Path(__file__).resolve().parents[1]
    run_id = uuid.uuid4().hex
    started = datetime.now(timezone.utc).isoformat()
    scenarios_out = []
    latencies: List[float] = []
    for r in records:
        latencies.extend(r["metrics"]["retrieve_latencies_ms"])
        scenarios_out.append({
            "id": r["id"], "family": r["family"], "lang": r["lang"], "split": r["split"], "arm": r["arm"],
            "passed": r["passed"],
            "checks": {k: {kk: vv for kk, vv in v.items()} for k, v in r["checks"].items()},
            "outcome": r["decision"]["outcome"], "as_of_outcome": (r.get("as_of_decision") or {}).get("outcome"),
            "reconsider": r["decision"].get("reconsider"),
            "boundaries": [b["boundary"] for b in r["boundaries"]],
            "metrics": r["metrics"],
        })

    def _rate(rows):
        n = len(rows)
        return {"passed": sum(1 for x in rows if x["passed"]), "total": n,
                "rate": (round(sum(1 for x in rows if x["passed"]) / n, 4) if n else None)}

    per_arm = {arm: _rate([s for s in scenarios_out if s["arm"] == arm]) for arm in ARMS_RUN}
    per_family = {}
    for fam in sorted({s["family"] for s in scenarios_out}):
        per_family[fam] = {arm: _rate([s for s in scenarios_out if s["family"] == fam and s["arm"] == arm]) for arm in ARMS_RUN}
    per_split = {sp: {arm: _rate([s for s in scenarios_out if s["split"] == sp and s["arm"] == arm]) for arm in ARMS_RUN}
                 for sp in sorted({s["split"] for s in scenarios_out})}
    per_lang = {lg: {arm: _rate([s for s in scenarios_out if s["lang"] == lg and s["arm"] == arm]) for arm in ARMS_RUN}
                for lg in sorted({s["lang"] for s in scenarios_out})}
    # Behavioral counters the issue asks for, over arm B (the memory arm).
    b_rows = [r for r in records if r["arm"] == "B"]
    counters = {
        "superseded_reused": sum(1 for r in b_rows if r["decision"].get("kref") and r["graph"].status_of(r["decision"]["kref"]) == "superseded"),
        "superseded_ignored": sum(int(r["decision"].get("ignored_superseded") or 0) for r in b_rows),
        "correction_retained": sum(1 for r in b_rows if r["checks"].get("correction_retained", {}).get("ok") and not r["checks"].get("correction_retained", {}).get("not_applicable")),
        "correction_scenarios": sum(1 for r in b_rows if "correction_retained" in r["checks"]),
        "unsupported_promotion": sum(1 for r in b_rows if "proposal_not_promoted" in r["checks"] and not r["checks"]["proposal_not_promoted"]["ok"]),
        "scope_leaks": sum(1 for r in b_rows if "no_scope_leak" in r["checks"] and not r["checks"]["no_scope_leak"]["ok"]),
        "fabricated_continuity_in_A": sum(1 for r in records if r["arm"] == "A" and r["decision"]["outcome"] != "unknown"),
    }
    return {
        "report_version": "1",
        "run_id": run_id,
        "started_at": started,
        "tier": fixture["meta"]["tier"],
        "not_evidence_of_model_quality": True,
        "fixture": {"path": str(fixture_path.name), "name": fixture["meta"]["name"],
                    "version": fixture["meta"]["version"], "sha256": fixture_sha256(fixture_path),
                    "scenarios": len(fixture["scenarios"]),
                    "items_by_split": {sp: sum(1 for s in fixture["scenarios"] if s["split"] == sp) for sp in ("dev", "heldout")}},
        "product": {"kumiho_memory_version": product_version, "git_sha": _git_sha(repo_root),
                    "python": sys.version.split()[0]},
        "arms_run": list(ARMS_RUN), "arms_not_run": ARMS_NOT_RUN,
        "trials_per_scenario": 1,
        "summary": {"per_arm": per_arm, "per_family": per_family, "per_split": per_split, "per_lang": per_lang,
                    "counters": counters,
                    "retrieve_latency_ms": {"n": len(latencies), "p50": _pctl(latencies, 0.5), "p95": _pctl(latencies, 0.95),
                                            "mean": (round(statistics.fmean(latencies), 3) if latencies else None),
                                            "note": "in-process fake graph; measures the memory layer's own overhead, not a server"},
                    "payload_bytes": {"max": max((s["metrics"]["payload_bytes"] for s in scenarios_out), default=None)},
                    "tokens_cost": {"provider_reported": None, "note": "keyless tier: no provider calls; unavailable, not zero"}},
        "scenarios": scenarios_out,
    }


def render_summary(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"# decision continuity — deterministic tier (run {report['run_id'][:8]})",
        "",
        f"fixture `{report['fixture']['name']}` v{report['fixture']['version']} "
        f"(sha256 {report['fixture']['sha256'][:12]}…), {report['fixture']['scenarios']} scenarios "
        f"({report['fixture']['items_by_split']}), 1 trial each; kumiho-memory {report['product']['kumiho_memory_version']} "
        f"@ {str(report['product']['git_sha'])[:8]}",
        "",
        "This tier proves protocol behaviour of the memory layer. It is NOT evidence that a model decides better.",
        "",
        "| arm | passed / total |", "|---|---|",
    ]
    for arm, r in s["per_arm"].items():
        lines.append(f"| {arm} | {r['passed']} / {r['total']} |")
    lines += ["", "| family | B | A |", "|---|---|---|"]
    for fam, r in s["per_family"].items():
        lines.append(f"| {fam} | {r['B']['passed']}/{r['B']['total']} | {r['A']['passed']}/{r['A']['total']} |")
    lines += ["", "| split | B | A |", "|---|---|---|"]
    for sp, r in s["per_split"].items():
        lines.append(f"| {sp} | {r['B']['passed']}/{r['B']['total']} | {r['A']['passed']}/{r['A']['total']} |")
    c = s["counters"]
    lines += ["", "counters (arm B unless stated): "
              f"superseded reused={c['superseded_reused']}, superseded ignored={c['superseded_ignored']}, "
              f"corrections retained={c['correction_retained']}/{c['correction_scenarios']}, "
              f"unsupported promotions={c['unsupported_promotion']}, scope leaks={c['scope_leaks']}, "
              f"fabricated continuity in A={c['fabricated_continuity_in_A']}",
              "", f"retrieve latency (in-process fake): n={s['retrieve_latency_ms']['n']} "
              f"p50={s['retrieve_latency_ms']['p50']} ms p95={s['retrieve_latency_ms']['p95']} ms",
              f"arms not run: {report['arms_not_run']}", ""]
    failed = [x for x in report["scenarios"] if not x["passed"]]
    if failed:
        lines += ["## failed", ""]
        for x in failed:
            bad = {k: v["detail"] for k, v in x["checks"].items() if not v["ok"]}
            lines.append(f"- `{x['id']}` [{x['arm']}]: {bad}")
    return "\n".join(lines) + "\n"
