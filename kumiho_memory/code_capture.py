"""Decision Memory capture adapter v1: mine git commits into decisions.

Pipeline (docs/DECISION_MEMORY_DESIGN.md §4)::

    [1] enumerate   git log (subprocess — zero new dependencies)
    [2] prefilter   deterministic noise cut (false-negative-asymmetric:
                    only CERTAIN noise is dropped; the LLM judges the rest)
    [3] packet      per-commit evidence packet (message-first, diff-as-
                    evidence, deterministic truncation)
    [4] structure   batched LLM call (json_mode strict schema) — a decision
                    is a CHOICE (alternative picked / default set / reversal /
                    measured trade-off), never a restatement of the change;
                    zero decisions is a valid answer
    [5] validate    anchors checked against the commit's real changed-file
                    list (hallucinated anchors dropped, stat fallback);
                    low-confidence evidence-free decisions dropped
    [6] write       get-or-create nodes + edges (edge existence checked
                    before create — server-side dedupe is not assumed)
    [7] supersede   anchor-scoped 3-signal SUPERSEDES pass
    [8] marker      the ``code_commit`` node is written LAST — its presence
                    is the idempotency marker, so a partial failure leaves
                    no marker and the commit is retried on the next run

Re-running the same range is free: marked commits are skipped without any
LLM call, every node write is get-or-create on sha-free slugs, and every
edge write checks for an existing edge first.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kumiho_memory._bounded import run_bounded_in_thread
from kumiho_memory.code_decisions import (
    CodeMemoryConfig,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTED_IN,
    EDGE_MOTIVATED_BY,
    EDGE_SUPERSEDES,
    EVIDENCE_KINDS,
    KIND_COMMIT,
    KIND_DECISION,
    KIND_EVIDENCE,
    SCHEMA_VERSION,
    commit_slug,
    ensure_space,
    evidence_slug,
    get_or_create_anchor,
    get_or_create_decision_item,
    get_or_create_item,
    normalize_path,
    write_revision,
)

logger = logging.getLogger(__name__)

_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"

#: Diff paths that are certainly generated/lock noise (prefilter denylist).
_DENYLIST_GLOBS = (
    "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock",
    "*.pb.go", "*_pb2.py", "*_pb2_grpc.py", "*.min.js", "*.min.css",
    "*__pycache__*", "*.snap", "dist/*", "build/*",
)

_BUMP_SUBJECT_MARKERS = ("bump", "version bump", "release v", "chore(release)")


# ---------------------------------------------------------------------------
# [1] git enumeration (subprocess helpers)
# ---------------------------------------------------------------------------

@dataclass
class CommitInfo:
    hash: str
    author: str
    author_date: str
    subject: str
    body: str
    parents: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    packet: str = ""


@dataclass
class IngestStats:
    commits_seen: int = 0
    skipped_marker: int = 0
    skipped_prefilter: int = 0
    llm_calls: int = 0
    decisions: int = 0
    evidence: int = 0
    anchors: int = 0
    edges: int = 0
    superseded: int = 0
    failed_commits: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _run_git(repo_path: str, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    return out.stdout


def derive_repo_id(repo_path: str) -> str:
    """Repo identifier: origin URL tail (sans ``.git``), else the dir name."""
    try:
        url = _run_git(repo_path, "remote", "get-url", "origin").strip()
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        if tail:
            return tail
    except Exception:  # noqa: BLE001 — local repos without origin are fine
        pass
    norm = normalize_path(repo_path)
    return norm.rsplit("/", 1)[-1] or "repo"


def enumerate_commits(
    repo_path: str, rev_range: Optional[str], max_commits: int,
) -> List[CommitInfo]:
    fmt = _FIELD_SEP.join(["%H", "%an", "%aI", "%P", "%s", "%b"]) + _RECORD_SEP
    args = ["log", f"--format={fmt}", f"--max-count={max_commits}"]
    if rev_range:
        args.append(rev_range)
    raw = _run_git(repo_path, *args)
    commits: List[CommitInfo] = []
    for record in raw.split(_RECORD_SEP):
        record = record.strip("\n\r ")
        if not record:
            continue
        parts = record.split(_FIELD_SEP)
        if len(parts) < 6:
            continue
        sha, author, adate, parents, subject = parts[0], parts[1], parts[2], parts[3], parts[4]
        body = _FIELD_SEP.join(parts[5:])  # bodies may contain the separator
        commits.append(CommitInfo(
            hash=sha.strip(),
            author=author.strip(),
            author_date=adate.strip(),
            subject=subject.strip(),
            body=body.strip(),
            parents=[p for p in parents.strip().split(" ") if p],
        ))
    return commits


def _changed_files(repo_path: str, sha: str) -> List[str]:
    raw = _run_git(repo_path, "show", "--name-only", "--format=", sha)
    return [normalize_path(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# [2] prefilter — cut only CERTAIN noise
# ---------------------------------------------------------------------------

def _is_denylisted(path: str) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in _DENYLIST_GLOBS)


def prefilter(commit: CommitInfo) -> Tuple[bool, str]:
    """Returns ``(keep, reason)``.  Ambiguity always passes to the LLM;
    conventional-commit *type* is never a criterion (``chore:`` commits can
    carry real decisions — measured on this repo)."""
    has_body = bool(commit.body.strip())
    if len(commit.parents) >= 2 and not has_body:
        return False, "merge without body"
    if commit.files and all(_is_denylisted(f) for f in commit.files):
        return False, "lockfile/generated-only diff"
    subject = commit.subject.lower()
    if not has_body and any(m in subject for m in _BUMP_SUBJECT_MARKERS):
        return False, "version bump without body"
    if not has_body and len(commit.subject.split()) <= 3 and len(commit.files) == 0:
        return False, "trivial subject, empty commit"
    return True, ""


# ---------------------------------------------------------------------------
# [3] evidence packet — message-first, diff-as-evidence
# ---------------------------------------------------------------------------

def _is_comment_change(line: str) -> bool:
    stripped = line[1:].lstrip()
    return stripped.startswith(("#", "//", "*", '"""', "'''", "--"))


def _truncate_file_diff(lines: List[str], budget: int) -> List[str]:
    """Deterministic truncation preserving rationale carriers: hunk headers
    and function-signature context always; then comment/docstring changes;
    then additions; then deletions."""
    if len(lines) <= budget:
        return lines
    always = [
        (i, l) for i, l in enumerate(lines)
        if l.startswith("@@") or l.startswith("+++") or l.startswith("---")
    ]
    tiers = [
        [(i, l) for i, l in enumerate(lines)
         if l[:1] in "+-" and not l.startswith(("+++", "---")) and _is_comment_change(l)],
        [(i, l) for i, l in enumerate(lines)
         if l.startswith("+") and not l.startswith("+++") and not _is_comment_change(l)],
        [(i, l) for i, l in enumerate(lines)
         if l.startswith("-") and not l.startswith("---") and not _is_comment_change(l)],
    ]
    chosen: Dict[int, str] = dict(always)
    for tier in tiers:
        for i, l in tier:
            if len(chosen) >= budget:
                break
            chosen.setdefault(i, l)
        if len(chosen) >= budget:
            break
    kept = [l for _, l in sorted(chosen.items())]
    kept.append(f"[...truncated {len(lines) - len(kept)} lines]")
    return kept


def build_packet(repo_path: str, commit: CommitInfo, config: CodeMemoryConfig) -> str:
    """Assemble the per-commit packet: full message, full changed-file list
    (anchor ground truth), and a budgeted diff excerpt."""
    parts = [
        f"commit {commit.hash}",
        f"author {commit.author}  date {commit.author_date}",
        f"subject: {commit.subject}",
    ]
    if commit.body:
        parts.append("body:\n" + commit.body)
    parts.append("changed files:\n" + "\n".join(f"- {f}" for f in commit.files))

    try:
        raw = _run_git(repo_path, "show", "--format=", "--unified=3", commit.hash)
    except Exception as exc:  # noqa: BLE001
        logger.debug("code capture: diff fetch failed for %s: %s", commit.hash, exc)
        raw = ""
    if raw:
        file_blocks: List[Tuple[str, List[str]]] = []
        cur_name, cur_lines = "", []
        for line in raw.splitlines():
            if line.startswith("diff --git"):
                if cur_name and not _is_denylisted(cur_name):
                    file_blocks.append((cur_name, cur_lines))
                cur_name = normalize_path(line.split(" b/")[-1]) if " b/" in line else line
                cur_lines = []
            else:
                cur_lines.append(line)
        if cur_name and not _is_denylisted(cur_name):
            file_blocks.append((cur_name, cur_lines))

        diff_parts: List[str] = []
        used = 0
        for name, lines in file_blocks[:6]:
            kept = _truncate_file_diff(lines, config.per_file_diff_lines)
            block = f"--- diff: {name} ---\n" + "\n".join(kept)
            if used + len(block) > config.per_commit_diff_chars:
                block = block[: max(0, config.per_commit_diff_chars - used)]
                if not block:
                    break
                block += "\n[...truncated at commit budget]"
            diff_parts.append(block)
            used += len(block)
            if used >= config.per_commit_diff_chars:
                break
        if diff_parts:
            parts.append("\n".join(diff_parts))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# [4] LLM structuring — batched, strict schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You extract engineering DECISIONS from git commits for a decision-memory "
    "graph. A decision is a choice: (a) an alternative picked over another "
    "('Y instead of X'), (b) a default/policy set ('default ON, opt-out'), "
    "(c) a reversal of previous behavior, or (d) a measured trade-off. "
    "A decision is NOT a restatement of what changed (git already knows), a "
    "mechanical rename, or the mere existence of a bug. Zero decisions is a "
    "valid answer — never invent one. Emit 0-3 decisions per commit. "
    "evidence.text MUST be quoted verbatim from the commit message or code "
    "comments in the diff. anchors.file MUST come from the commit's changed-"
    "file list. Write everything in English."
)


def _structuring_schema() -> Dict[str, Any]:
    def obj(props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": props,
            "required": list(props.keys()),
        }

    evidence = obj({
        "kind": {"type": "string", "enum": list(EVIDENCE_KINDS)},
        "text": {"type": "string"},
        "source_ref": {"type": "string"},
    })
    anchor = obj({
        "file": {"type": "string"},
        "line_start": {"type": "integer"},
        "line_end": {"type": "integer"},
        "role": {"type": "string", "enum": ["primary", "touched"]},
    })
    decision = obj({
        "title": {"type": "string"},
        "decision": {"type": "string"},
        "rationale": {"type": "string"},
        "why_question": {"type": "string"},
        "symbols": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": evidence},
        "anchors": {"type": "array", "items": anchor},
        "supersedes_hint": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    })
    commit = obj({
        "hash": {"type": "string"},
        "decisions": {"type": "array", "items": decision},
    })
    return obj({"commits": {"type": "array", "items": commit}})


async def _structure_batch(
    adapter: Any, model: str, packets: List[str],
) -> List[Dict[str, Any]]:
    prompt = (
        "Extract decisions from each commit below. Commits are delimited by "
        "'=== COMMIT n ==='. Return one entry per commit (matching its hash), "
        "in order.\n\n"
        + "\n\n".join(f"=== COMMIT {i + 1} ===\n{p}" for i, p in enumerate(packets))
    )
    raw = await adapter.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        system=_SYSTEM_PROMPT,
        max_tokens=4096,
        json_mode={"name": "code_decisions", "schema": _structuring_schema()},
    )
    data = json.loads(raw)
    return list(data.get("commits", []))


# ---------------------------------------------------------------------------
# [5] validation — hallucination defenses
# ---------------------------------------------------------------------------

def validate_decisions(
    commit: CommitInfo,
    decisions: List[Dict[str, Any]],
    config: CodeMemoryConfig,
) -> List[Dict[str, Any]]:
    changed = {normalize_path(f) for f in commit.files}
    out: List[Dict[str, Any]] = []
    for d in decisions[: config.max_decisions_per_commit]:
        evidence = [
            e for e in (d.get("evidence") or [])[: config.max_evidence_per_decision]
            if str(e.get("text", "")).strip()
        ]
        if (
            config.drop_low_confidence_without_evidence
            and d.get("confidence") == "low"
            and not evidence
        ):
            continue
        anchors = []
        for a in d.get("anchors") or []:
            norm = normalize_path(str(a.get("file", "")))
            if norm in changed:
                a = dict(a)
                a["file"] = norm
                anchors.append(a)
            else:
                logger.debug(
                    "code capture: dropping hallucinated anchor %r for %s",
                    a.get("file"), commit.hash[:12],
                )
        if not anchors:
            # stat fallback: file-level anchors from ground truth
            anchors = [
                {"file": f, "line_start": 0, "line_end": 0, "role": "touched"}
                for f in sorted(changed) if not _is_denylisted(f)
            ]
        d = dict(d)
        d["evidence"] = evidence
        d["anchors"] = anchors[: config.max_anchors_per_decision]
        if not str(d.get("title", "")).strip():
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# [6-8] write / supersede / marker (blocking worker, one commit at a time)
# ---------------------------------------------------------------------------

def _compose_embedding_text(meta: Dict[str, str], subject: str) -> str:
    """§1.6 composition: why-question-led, identifier-bearing, English."""
    basenames = ", ".join(
        f.rsplit("/", 1)[-1] for f in str(meta.get("files", "")).split(",") if f
    )
    parts = []
    if meta.get("why_question"):
        parts.append(meta["why_question"])
    parts.append(f"{meta.get('decision', '')}.")
    if meta.get("rationale"):
        parts.append(f"Rationale: {meta['rationale']}.")
    anchored = basenames
    if meta.get("symbols"):
        anchored = f"{basenames} ({meta['symbols']})" if basenames else meta["symbols"]
    if anchored:
        parts.append(f"Anchored: {anchored}.")
    if subject:
        parts.append(f'Commit: "{subject}".')
    return " ".join(p for p in parts if p)


def _edge_exists(src_rev: Any, edge_type: str, target_uri: str) -> bool:
    try:
        for e in src_rev.get_edges(edge_type_filter=edge_type, direction=0):
            if getattr(getattr(e, "target_kref", None), "uri", "") == target_uri:
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _create_edge_once(src_rev: Any, target_rev: Any, edge_type: str,
                      metadata: Dict[str, str], stats: IngestStats) -> None:
    target_uri = getattr(getattr(target_rev, "kref", None), "uri", "")
    if _edge_exists(src_rev, edge_type, target_uri):
        return
    src_rev.create_edge(target_rev, edge_type, metadata=metadata)
    stats.edges += 1


def _marker_complete(project: Any, config: CodeMemoryConfig, slug: str) -> bool:
    try:
        item = project.get_item(slug, KIND_COMMIT,
                                parent_path=f"/{project.name}/{config.commits_space}")
    except Exception:  # noqa: BLE001
        return False
    if item is None:
        return False
    try:
        return item.get_latest_revision() is not None
    except Exception:  # noqa: BLE001
        return False


def _supersede_pass(
    project: Any,
    config: CodeMemoryConfig,
    decision_rev: Any,
    decision_meta: Dict[str, str],
    anchor_revs: List[Any],
    hint: str,
    stats: IngestStats,
) -> None:
    """Anchor-scoped 3-signal SUPERSEDES (§2.1): shared anchor file AND
    token-Jaccard over title+decision AND strict author-date ordering."""
    import kumiho

    from kumiho_memory.relations import _jaccard, _tokens

    my_uri = getattr(getattr(decision_rev, "kref", None), "uri", "")
    my_text = f"{decision_meta.get('title', '')} {decision_meta.get('decision', '')}"
    my_date = decision_meta.get("decided_at", "")
    threshold = config.supersede_jaccard_hinted if hint else config.supersede_jaccard_blind

    seen: set = set()
    for anchor_rev in anchor_revs:
        try:
            edges = anchor_rev.get_edges(
                edge_type_filter=EDGE_IMPLEMENTED_IN, direction=kumiho.INCOMING,
            )
        except Exception:  # noqa: BLE001
            continue
        for edge in edges or []:
            src = getattr(getattr(edge, "source_kref", None), "uri", "")
            if not src or src == my_uri or src in seen:
                continue
            seen.add(src)
            try:
                old_rev = kumiho.get_revision(src)
            except Exception:
                continue
            old_meta = getattr(old_rev, "metadata", {}) or {}
            old_text = f"{old_meta.get('title', '')} {old_meta.get('decision', '')}"
            if hint and _jaccard(_tokens(hint), _tokens(old_text)) < 0.2:
                continue  # the hint narrows candidates
            overlap = _jaccard(_tokens(my_text), _tokens(old_text))
            if overlap < threshold:
                continue
            old_date = str(old_meta.get("decided_at", ""))
            if not old_date or not my_date or not old_date < my_date:
                continue  # strict time ordering — ingest order must not matter
            _create_edge_once(
                decision_rev, old_rev, EDGE_SUPERSEDES,
                {"reason": "belief update", "overlap": f"{overlap:.2f}"},
                stats,
            )
            # Demote the old decision so query-time ranking sinks it.
            if old_meta.get("status") != "superseded":
                new_meta = dict(old_meta)
                new_meta["status"] = "superseded"
                try:
                    target_item = old_rev.get_item()
                    write_revision(
                        target_item, new_meta,
                        _compose_embedding_text(new_meta, ""),
                    )
                    stats.superseded += 1
                except Exception as exc:  # noqa: BLE001
                    logger.debug("code capture: status demotion failed: %s", exc)


def _sync_write_commit(
    project_name: str,
    config: CodeMemoryConfig,
    repo: str,
    commit: CommitInfo,
    decisions: List[Dict[str, Any]],
    capture_version: str,
    stats: IngestStats,
) -> None:
    """Stages [6]-[8] for one commit.  Crash-safe: the marker revision is
    the very last write, so any partial failure leaves the commit unmarked
    and the next run retries it (all writes are get-or-create)."""
    import kumiho

    project = kumiho.get_project(project_name)
    if project is None:
        project = kumiho.create_project(project_name)
    for space in (config.decisions_space, config.anchors_space,
                  config.commits_space, config.evidence_space):
        ensure_space(project, space)

    decision_payloads: List[Tuple[Any, Dict[str, str], List[Any], str]] = []
    for d in decisions:
        files_csv = ",".join(a["file"] for a in d["anchors"])
        ranges = ";".join(
            f"{a['file']}:{a['line_start']}-{a['line_end']}"
            for a in d["anchors"] if a.get("line_start")
        )
        meta = {
            "title": str(d["title"])[:80],
            "summary": f"{d.get('decision', '')} — {d.get('rationale', '')}"[:400],
            "decision": str(d.get("decision", "")),
            "rationale": str(d.get("rationale", "")),
            "why_question": str(d.get("why_question", "")),
            "symbols": ",".join(d.get("symbols") or []),
            "repo": repo,
            "commit_hash": commit.hash,
            "files": files_csv,
            "line_ranges": ranges,
            "author": commit.author,
            "decided_at": commit.author_date,
            "confidence": str(d.get("confidence", "medium")),
            "status": "active",
            "schema_version": SCHEMA_VERSION,
        }

        # evidence nodes first (decision edges point at them)
        evidence_revs: List[Tuple[Any, Dict[str, Any]]] = []
        for ev in d["evidence"]:
            slug = evidence_slug(str(ev["text"]))
            if not slug:
                continue
            item = get_or_create_item(
                project, slug, KIND_EVIDENCE,
                f"/{project_name}/{config.evidence_space}",
            )
            rev = item.get_latest_revision()
            if rev is None:
                rev = write_revision(item, {
                    "statement": str(ev["text"]),
                    "evidence_kind": str(ev.get("kind", "constraint")),
                    "source_ref": str(ev.get("source_ref", f"commit:{commit.hash[:12]}")),
                    "schema_version": SCHEMA_VERSION,
                }, embedding_text=str(ev["text"]))
                stats.evidence += 1
            evidence_revs.append((rev, ev))

        # anchors
        anchor_revs: List[Tuple[Any, Dict[str, Any]]] = []
        for a in d["anchors"]:
            rev = get_or_create_anchor(project, config, repo, a["file"])
            if rev is not None:
                anchor_revs.append((rev, a))
                stats.anchors += 1

        # decision node (embedding_text injected — §1.6)
        item = get_or_create_decision_item(
            project, config, meta["title"], commit.author_date, meta["decision"],
        )
        decision_rev = item.get_latest_revision()
        if decision_rev is None:
            decision_rev = write_revision(
                item, meta, _compose_embedding_text(meta, commit.subject),
            )
            stats.decisions += 1

        # edges: IMPLEMENTED_IN + MOTIVATED_BY (existence-checked)
        for rev, a in anchor_revs:
            _create_edge_once(decision_rev, rev, EDGE_IMPLEMENTED_IN, {
                "commit_hash": commit.hash,
                "line_start": str(a.get("line_start", 0) or ""),
                "line_end": str(a.get("line_end", 0) or ""),
                "role": str(a.get("role", "touched")),
            }, stats)
        for rev, _ev in evidence_revs:
            _create_edge_once(decision_rev, rev, EDGE_MOTIVATED_BY,
                              {"commit_hash": commit.hash}, stats)

        # [7] supersede pass
        _supersede_pass(
            project, config, decision_rev, meta,
            [r for r, _ in anchor_revs], str(d.get("supersedes_hint", "")), stats,
        )
        decision_payloads.append(
            (decision_rev, meta, [r for r, _ in evidence_revs], commit.hash)
        )

    # [8] marker LAST: commit node + provenance edges
    slug = commit_slug(repo, commit.hash)
    marker_item = get_or_create_item(
        project, slug, KIND_COMMIT, f"/{project_name}/{config.commits_space}",
    )
    marker_rev = marker_item.get_latest_revision()
    if marker_rev is None:
        marker_rev = write_revision(marker_item, {
            "repo": repo,
            "hash": commit.hash,
            "subject": commit.subject,
            "author": commit.author,
            "committed_at": commit.author_date,
            "decisions_count": str(len(decision_payloads)),
            "capture_version": capture_version,
            "schema_version": SCHEMA_VERSION,
        }, embedding_text=commit.subject)
    for decision_rev, _meta, evidence_revs, _sha in decision_payloads:
        _create_edge_once(decision_rev, marker_rev, EDGE_DERIVED_FROM, {}, stats)
        for ev_rev in evidence_revs:
            _create_edge_once(ev_rev, marker_rev, EDGE_DERIVED_FROM, {}, stats)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_repo(
    repo_path: str = ".",
    rev_range: Optional[str] = None,
    *,
    project_name: str,
    config: Optional[CodeMemoryConfig] = None,
    adapter: Any = None,
    model: str = "",
    force: bool = False,
    max_commits: Optional[int] = None,
) -> IngestStats:
    """Mine *rev_range* (or the recent history, incrementally) into the graph.

    Foreground work invoked explicitly by the user: returns full stats and
    the list of failed commits instead of swallowing errors.  Omitting
    *rev_range* enables incremental mode — already-marked commits are skipped
    (the graph itself is the cursor; no ledger file).
    """
    config = config or CodeMemoryConfig()
    stats = IngestStats()
    if adapter is None or not model:
        stats.errors.append("no LLM adapter/model configured")
        return stats

    repo = (config.repo or derive_repo_id(repo_path)).strip()
    limit = max_commits or config.max_commits
    try:
        commits = enumerate_commits(repo_path, rev_range, limit)
    except Exception as exc:  # noqa: BLE001
        stats.errors.append(f"git enumeration failed: {exc}")
        return stats
    stats.commits_seen = len(commits)

    # marker skip (incremental) — checked before any LLM spend
    def _sync_marker_check(sha: str) -> bool:
        import kumiho

        project = kumiho.get_project(project_name)
        if project is None:
            return False
        return _marker_complete(project, config, commit_slug(repo, sha))

    pending: List[CommitInfo] = []
    for c in commits:
        if not force:
            marked = await run_bounded_in_thread(
                lambda sha=c.hash: _sync_marker_check(sha),
                timeout=config.write_timeout, label="code marker check",
                on_timeout=False, on_error=False,
            )
            if marked:
                stats.skipped_marker += 1
                continue
        keep, _reason = prefilter(c)
        if not keep:
            stats.skipped_prefilter += 1
            continue
        c.files = _changed_files(repo_path, c.hash)
        if c.files and all(_is_denylisted(f) for f in c.files):
            stats.skipped_prefilter += 1
            continue
        c.packet = build_packet(repo_path, c, config)
        pending.append(c)

    # batched structuring, concurrency 3
    sem = asyncio.Semaphore(3)
    batches = [
        pending[i : i + config.llm_batch_size]
        for i in range(0, len(pending), config.llm_batch_size)
    ]

    async def _run_batch(batch: List[CommitInfo]) -> List[Tuple[CommitInfo, List[Dict[str, Any]]]]:
        async with sem:
            try:
                entries = await _structure_batch(adapter, model, [c.packet for c in batch])
                stats.llm_calls += 1
            except Exception as exc:  # noqa: BLE001
                for c in batch:
                    stats.failed_commits.append(c.hash[:12])
                stats.errors.append(f"structuring failed: {exc}")
                return []
        by_hash = {str(e.get("hash", ""))[:12]: e for e in entries}
        out = []
        for i, c in enumerate(batch):
            entry = by_hash.get(c.hash[:12])
            if entry is None and i < len(entries):
                entry = entries[i]  # positional fallback (order is contractual)
            decisions = validate_decisions(c, list((entry or {}).get("decisions", [])), config)
            out.append((c, decisions))
        return out

    results = await asyncio.gather(*(_run_batch(b) for b in batches))

    capture_version = SCHEMA_VERSION
    for batch_result in results:
        for c, decisions in batch_result:
            ok = await run_bounded_in_thread(
                lambda c=c, d=decisions: _sync_write_commit(
                    project_name, config, repo, c, d, capture_version, stats,
                ) or True,
                timeout=config.write_timeout, label="code commit write",
                on_timeout=None, on_error=None,
            )
            if ok is not True:
                stats.failed_commits.append(c.hash[:12])
    return stats
