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

import re

from kumiho_memory._bounded import run_bounded_in_thread
from kumiho_memory.evidence import CORROBORATED, SINGLE_SOURCE, UNVERIFIED
from kumiho_memory.code_decisions import (
    CodeMemoryConfig,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTED_IN,
    EDGE_MOTIVATED_BY,
    EDGE_SUPERSEDES,
    EVIDENCE_KINDS,
    KIND_COMMIT,
    KIND_EVIDENCE,
    SCHEMA_VERSION,
    commit_slug,
    deprecate_marker_decisions,
    ensure_space,
    evidence_slug,
    get_or_create_anchor,
    get_or_create_decision_item,
    get_or_create_item,
    marker_provenance_complete,
    normalize_path,
    parse_decided_at,
    undeprecate_item,
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
    deprecated: int = 0
    failed_commits: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


#: Rev-range charset: hashes, refs, ``..``/``...``, ``~^@{}`` navigation.
#: The leading character must not be ``-`` — a range like ``--output=x``
#: would otherwise be parsed by git as an option (argv option injection:
#: ``--output`` overwrites arbitrary files).
_REV_RANGE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./~^@{}\-]*(\.\.\.?[A-Za-z0-9_][A-Za-z0-9_./~^@{}\-]*)?$")


def _validate_rev_range(rev_range: str) -> str:
    if not _REV_RANGE_RE.match(rev_range):
        raise ValueError(f"unsafe rev range: {rev_range!r}")
    return rev_range


def _run_git(repo_path: str, *args: str) -> str:
    if str(repo_path).startswith("-"):
        raise ValueError(f"unsafe repo path: {repo_path!r}")
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
        args.append(_validate_rev_range(rev_range))
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


def _changed_files(repo_path: str, sha: str, parents: Optional[List[str]] = None) -> List[str]:
    """Changed-file list — the anchor ground truth.

    Merge commits get the FIRST-PARENT diff: ``git show`` on a merge prints
    a combined diff that is empty for clean merges, which would leave the
    rationale-carrying squash-merge commits (§4.2 deliberately keeps them)
    with no anchors at all.
    """
    if parents and len(parents) >= 2:
        raw = _run_git(repo_path, "diff", "--name-only", f"{sha}^1", sha)
    else:
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
    "When one commit makes SEVERAL distinct choices (e.g. a mechanism AND a "
    "guard/limit on it AND a scope restriction), emit each as its own "
    "decision — do not collapse them into one shallow summary. "
    "Each decision's title and decision text must name the concrete choice "
    "(e.g. 'single-worker executor', not 'asynchronous processing'). "
    "anchors.file MUST come from the commit's changed-file list; mark "
    "role='primary' on the file where the decided behavior is DEFINED "
    "(the mechanism's home), 'touched' for call sites and tests. "
    "evidence.text MUST be quoted verbatim from the commit message or code "
    "comments in the diff, and MUST carry the WHY — prefer sentences with "
    "measurements, review findings, incidents, or 'because...' reasoning "
    "over sentences that merely restate the new behavior. "
    "Write everything in English."
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
        max_tokens=8192,  # rationale-rich batches overflow 4096 (measured)
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
        # UNION with the changed-file ground truth (deterministic coverage):
        # the LLM's anchor picks contribute line ranges and primary roles,
        # but coverage must never depend on them — a measured miss (the
        # extractor anchoring release notes + a call site while skipping the
        # file that DEFINES the mechanism) silently kills the deterministic
        # query leg for exactly the file an agent will ask about.
        picked = {a["file"] for a in anchors}
        for f in sorted(changed):
            if len(anchors) >= config.max_anchors_per_decision:
                break
            if f in picked or _is_denylisted(f):
                continue
            anchors.append({"file": f, "line_start": 0, "line_end": 0,
                            "role": "touched"})
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
    """Edge-idempotency precheck (server-side dedupe is NOT assumed).

    Deliberately lets read errors PROPAGATE: swallowing them here would
    turn a transient outage into duplicate edges on the retry pass — the
    bounded worker converts the raised error into a failed commit, which
    re-runs cleanly next time.
    """
    for e in src_rev.get_edges(edge_type_filter=edge_type, direction=0):
        if getattr(getattr(e, "target_kref", None), "uri", "") == target_uri:
            return True
    return False


def _create_edge_once(src_rev: Any, target_rev: Any, edge_type: str,
                      metadata: Dict[str, str], stats: Any) -> None:
    # stats is duck-typed on .edges (IngestStats | SessionMineStats) —
    # code_session reuses this exact write primitive.
    target_uri = getattr(getattr(target_rev, "kref", None), "uri", "")
    if _edge_exists(src_rev, edge_type, target_uri):
        return
    src_rev.create_edge(target_rev, edge_type, metadata=metadata)
    stats.edges += 1


def _force_deprecate_commit_decisions(
    project: Any, config: CodeMemoryConfig, slug: str, stats: IngestStats,
) -> None:
    """--force pre-pass (design §4.6): deprecate the commit's existing
    decision nodes before re-mining, so a prompt-upgraded extraction never
    leaves orphaned stale generations behind.

    Only DECISION items are deprecated — evidence atoms are shared across
    decisions (slugged on their verbatim statement) and may be referenced by
    other commits' decisions.  The revision also gets ``status=deprecated``
    in place so query-side ranking demotes it even before the item-level
    flag propagates to search filters.
    """
    try:
        item = project.get_item(slug, KIND_COMMIT,
                                parent_path=f"/{project.name}/{config.commits_space}")
        marker_rev = item.get_latest_revision()
    except Exception:  # noqa: BLE001 — no marker, nothing to deprecate
        return
    deprecate_marker_decisions(marker_rev, stats)


def _marker_complete(project: Any, config: CodeMemoryConfig, slug: str) -> bool:
    """A commit counts as captured only when its marker revision exists AND
    its promised provenance edges are present (``decisions_count`` —
    see :func:`code_decisions.marker_provenance_complete`)."""
    try:
        item = project.get_item(slug, KIND_COMMIT,
                                parent_path=f"/{project.name}/{config.commits_space}")
    except Exception:  # noqa: BLE001
        return False
    return marker_provenance_complete(item, ("decisions_count",))[0]


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
    my_date = parse_decided_at(decision_meta.get("decided_at", ""))
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
            # Strict time ordering via PARSED aware datetimes — git author
            # dates carry the author's local UTC offset, so raw string
            # comparison misorders cross-timezone histories.
            old_date = parse_decided_at(old_meta.get("decided_at", ""))
            if old_date is None or my_date is None or not old_date < my_date:
                continue  # ingest order must not matter
            _create_edge_once(
                decision_rev, old_rev, EDGE_SUPERSEDES,
                {"reason": "belief update", "overlap": f"{overlap:.2f}"},
                stats,
            )
            # Demote the old decision IN PLACE (set_attribute on the SAME
            # revision the edges are pinned to).  Writing a new revision
            # here was the reviewed-and-confirmed critical: edges are
            # revision-scoped, so a new revision splits the decision's
            # identity — the anchor leg keeps seeing 'active' on the old
            # revision while the semantic leg surfaces an edgeless copy
            # whose superseded_by can never resolve.
            if old_meta.get("status") != "superseded":
                try:
                    old_rev.set_attribute("status", "superseded")
                    stats.superseded += 1
                except Exception as exc:  # noqa: BLE001
                    logger.debug("code capture: status demotion failed: %s", exc)


#: Evidence atoms that empirically substantiate a decision (vs a bare stated
#: constraint or rejected option).  Their presence lifts the decision's
#: Level-of-Evidence grade so ``code_why`` ranks well-evidenced decisions
#: ahead of thin ones within the same factual tier.
_STRONG_EVIDENCE_KINDS = frozenset(
    {"measurement", "review_finding", "benchmark", "incident"}
)


def _evidence_grade(evidence_atoms: List[Dict[str, Any]]) -> str:
    """Deterministic Level-of-Evidence for a code decision from its atoms.

    Keyless (no LLM): a measurement / review_finding / benchmark / incident
    atom is empirical corroboration; a bare constraint / rejected_alternative
    is a single stated source; no atoms is unverified.  ``official`` is never
    auto-assigned — :mod:`kumiho_memory.evidence` reserves it for an explicit
    operator flag.
    """
    kinds = {str(ev.get("kind", "constraint")) for ev in (evidence_atoms or [])}
    if kinds & _STRONG_EVIDENCE_KINDS:
        return CORROBORATED
    if kinds:
        return SINGLE_SOURCE
    return UNVERIFIED


def _sync_write_commit(
    project_name: str,
    config: CodeMemoryConfig,
    repo: str,
    commit: CommitInfo,
    decisions: List[Dict[str, Any]],
    capture_version: str,
    stats: IngestStats,
    force: bool = False,
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
            # Level-of-Evidence grade, derived deterministically from the
            # decision's evidence atoms (§6) — code_why weights it into
            # ranking so well-substantiated decisions surface first.
            "evidence_level": _evidence_grade(d["evidence"]),
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
        if force:
            undeprecate_item(item)
        decision_rev = item.get_latest_revision()
        if decision_rev is None or force:
            # force always writes a NEW revision — a re-capture exists to
            # replace stale extraction content, not to converge silently.
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
    if marker_rev is None or force:
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

    def _sync_force_deprecate(sha: str) -> bool:
        import kumiho

        project = kumiho.get_project(project_name)
        if project is None:
            return False
        _force_deprecate_commit_decisions(
            project, config, commit_slug(repo, sha), stats,
        )
        return True

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
        else:
            # design §4.6 deprecate-then-rewrite: retire the commit's stale
            # decision generation before re-mining it.
            await run_bounded_in_thread(
                lambda sha=c.hash: _sync_force_deprecate(sha),
                timeout=config.write_timeout, label="code force deprecate",
                on_timeout=False, on_error=False,
            )
        # Changed files are loaded BEFORE the prefilter: its trivial-subject
        # rule reads commit.files, and evaluating it against a not-yet-loaded
        # empty list silently dropped every bodyless short-subject commit
        # regardless of its real diff (reviewed-and-confirmed defect).
        try:
            c.files = _changed_files(repo_path, c.hash, parents=c.parents)
        except Exception as exc:  # noqa: BLE001
            stats.failed_commits.append(c.hash[:12])
            stats.errors.append(f"changed-files failed for {c.hash[:12]}: {exc}")
            continue
        keep, _reason = prefilter(c)
        if not keep:
            stats.skipped_prefilter += 1
            continue
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
        def _match_entry(c: CommitInfo) -> Optional[Dict[str, Any]]:
            # Prefix-tolerant hash matching: models echo abbreviated hashes
            # (7 chars, measured) — require at least 7 matching prefix chars
            # in either direction.
            want = c.hash.lower()
            for e in entries:
                h = str(e.get("hash", "")).strip().lower()
                if len(h) >= 7 and (want.startswith(h) or h.startswith(want[:12])):
                    return e
            return None

        out = []
        for i, c in enumerate(batch):
            entry = _match_entry(c)
            if entry is None and i < len(entries):
                candidate = entries[i]  # positional fallback (order is contractual)
                # Only accept the positional entry when its hash slot is
                # empty/unknown — a mismatched hash means the model lost
                # alignment and this commit's verdict is unreliable.
                if not str(candidate.get("hash", "")).strip():
                    entry = candidate
            if entry is None:
                # A missing batch entry is a FAILURE, not a zero-decision
                # verdict: writing a marker here would permanently skip a
                # commit the model never actually judged.
                stats.failed_commits.append(c.hash[:12])
                stats.errors.append(f"LLM batch entry missing for {c.hash[:12]}")
                continue
            decisions = validate_decisions(c, list(entry.get("decisions", [])), config)
            out.append((c, decisions))
        return out

    results = await asyncio.gather(*(_run_batch(b) for b in batches))

    capture_version = SCHEMA_VERSION
    for batch_result in results:
        for c, decisions in batch_result:
            ok = await run_bounded_in_thread(
                lambda c=c, d=decisions: _sync_write_commit(
                    project_name, config, repo, c, d, capture_version, stats,
                    force=force,
                ) or True,
                timeout=config.write_timeout, label="code commit write",
                on_timeout=None, on_error=None,
            )
            if ok is not True:
                stats.failed_commits.append(c.hash[:12])
    return stats


def _commit_info_for_ref(repo_path: str, ref: str) -> Optional[CommitInfo]:
    """CommitInfo for a single git ref (default HEAD), with changed files
    loaded — the deterministic ground truth the agent-driven capture path
    unions its anchors against."""
    commits = enumerate_commits(repo_path, ref, 1)
    if not commits:
        return None
    c = commits[0]
    c.files = _changed_files(repo_path, c.hash, parents=c.parents)
    return c


async def capture_decisions(
    repo_path: str,
    decisions: List[Dict[str, Any]],
    *,
    commit_ref: str = "HEAD",
    project_name: str,
    config: Optional[CodeMemoryConfig] = None,
) -> IngestStats:
    """Write AGENT-STRUCTURED decisions into the graph — **no LLM, no key**.

    The keyless counterpart to :func:`ingest_repo`.  The agent (Claude) has
    already read the diff / conversation and extracted the decision, so the
    structuring LLM call is skipped entirely — this mirrors
    ``kumiho_memory_reflect`` (the agent's own model identifies what matters;
    the tool just stores it).  The same deterministic validation still runs:
    anchors are UNIONED with the commit's real changed files, so a missing or
    loose anchor still lands on the right file, and hallucinated files drop.
    """
    config = config or CodeMemoryConfig()
    stats = IngestStats()
    if not decisions:
        stats.errors.append("no decisions provided")
        return stats

    repo = (config.repo or derive_repo_id(repo_path)).strip()
    try:
        commit = _commit_info_for_ref(repo_path, commit_ref)
    except Exception as exc:  # noqa: BLE001
        stats.errors.append(f"git resolution failed for {commit_ref!r}: {exc}")
        return stats
    if commit is None:
        stats.errors.append(f"commit not found: {commit_ref!r}")
        return stats

    stats.commits_seen = 1
    validated = validate_decisions(commit, list(decisions), config)
    if not validated:
        stats.errors.append(
            "all decisions dropped by validation (missing title, or "
            "low-confidence with no evidence)"
        )
        return stats

    ok = await run_bounded_in_thread(
        lambda: _sync_write_commit(
            project_name, config, repo, commit, validated, SCHEMA_VERSION, stats,
        ) or True,
        timeout=config.write_timeout, label="code capture write",
        on_timeout=None, on_error=None,
    )
    if ok is not True:
        stats.failed_commits.append(commit.hash[:12])
        stats.errors.append("write failed or timed out")
    return stats
