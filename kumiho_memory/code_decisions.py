"""Decision Memory schema: code-domain decisions anchored to git.

The code domain captures the one thing git cannot hold — *why* the code is
the way it is.  Decisions, their rationale, and their evidence become typed
graph nodes; the code itself is never copied.  Every decision points at git
via anchors ({repo, path} hub nodes) and edge metadata ({commit_hash,
line_start, line_end}), so the memory never rots: git stays the source of
truth for *what*, the graph holds the *why*.

Design (docs/DECISION_MEMORY_DESIGN.md, issue #43):

* **Physical isolation.**  Code nodes live in a dedicated ``{project}-code``
  kumiho project.  Typed embeddings sharing the conversation project's vector
  index measurably crowded out conversation recall (the ON-gate blocker
  incident); a separate project makes "conversation paths untouched" true by
  construction, not by gating.
* **sha-free identity.**  Node slugs never contain commit hashes: decisions
  key on ``title + author-date`` (both survive rebase/squash), anchors key on
  ``repo :: path``.  Volatile coordinates (commit hash, line ranges) live on
  edge/revision *metadata*, so history rewrites never orphan node identity.
* **This module is the single source** for kinds, spaces, edge types, path
  normalization, slugs, and the write path.  ``code_capture`` (write side)
  and ``code_query`` (read side) both import from here — the two sides can
  never drift on identity rules.

Everything is opt-in behind ``KUMIHO_MEMORY_CODE=1``; nothing here is
imported by the conversation recall/consolidation paths.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from kumiho._text import slugify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "kumiho.code_memory.v1"

#: Node kinds.  ``code_``-prefixed so they can never collide with the
#: conversation ontology's kinds (``decision``, ``fact``, ...) even if the
#: projects were ever merged by an operator.
KIND_DECISION = "code_decision"
KIND_ANCHOR = "code_anchor"
KIND_COMMIT = "code_commit"
KIND_EVIDENCE = "code_evidence"

#: Edge types.  ``create_edge`` accepts free-form UPPERCASE strings
#: (``kumiho/edge.py::validate_edge_type``) — no server changes involved.
#: ``DERIVED_FROM`` / ``SUPERSEDES`` reuse the graph-wide semantics the
#: traversal defaults already know.
EDGE_IMPLEMENTED_IN = "IMPLEMENTED_IN"   # code_decision -> code_anchor
EDGE_MOTIVATED_BY = "MOTIVATED_BY"       # code_decision -> code_evidence
EDGE_DERIVED_FROM = "DERIVED_FROM"       # code_decision -> code_commit
EDGE_SUPERSEDES = "SUPERSEDES"           # newer decision -> older decision

#: Evidence taxonomy (``code_evidence.evidence_kind``).
EVIDENCE_KINDS = (
    "measurement",
    "review_finding",
    "incident",
    "benchmark",
    "constraint",
)


@dataclass
class CodeMemoryConfig:
    """Configuration for the code-decision domain.

    ``project`` defaults to ``{memory_project}-code`` — derived at wiring
    time by :func:`resolve_project_name`, kept empty here so the dataclass
    carries no hidden global state.
    """

    #: Dedicated kumiho project for code nodes (physical isolation — see
    #: module docstring).  Empty means "derive from the memory project".
    project: str = ""

    #: Repo identifier baked into anchor slugs and node metadata.  Empty
    #: means "derive from the origin URL, falling back to the directory
    #: name" (``code_capture`` owns that heuristic).
    repo: str = ""

    decisions_space: str = "decisions"
    anchors_space: str = "anchors"
    commits_space: str = "commits"
    evidence_space: str = "evidence"

    # --- capture budgets (consumed by code_capture) ---
    llm_batch_size: int = 6
    max_commits: int = 50
    per_commit_diff_chars: int = 4000
    per_file_diff_lines: int = 40
    max_decisions_per_commit: int = 4
    max_anchors_per_decision: int = 8
    max_evidence_per_decision: int = 6

    # --- query knobs (consumed by code_query) ---
    #: Slack (in lines) when intersecting a queried line with an edge's
    #: recorded ``[line_start, line_end]`` — lines are the fastest-rotting
    #: coordinate, so the match is generous and boost-only.
    line_slack: int = 20

    # --- supersede thresholds (§2.1: 3-signal confluence) ---
    supersede_jaccard_hinted: float = 0.35
    supersede_jaccard_blind: float = 0.5

    #: Drop low-confidence decisions that carry no evidence atoms — the
    #: honesty floor for thin commit messages.
    drop_low_confidence_without_evidence: bool = True

    #: Deadline for blocking SDK writes (run through the bounded-thread
    #: helper by the capture pipeline).
    write_timeout: float = 60.0

    schema_version: str = SCHEMA_VERSION

    #: Decision-slug collision guard threshold: an existing item whose
    #: ``decision`` metadata shares less than this token-Jaccard with the
    #: incoming decision text is treated as a *different* decision that
    #: happens to share title+date, and the new one gets a ``-2`` suffix.
    slug_collision_jaccard: float = 0.3


def config_from_env(base: Optional[CodeMemoryConfig] = None) -> CodeMemoryConfig:
    """Overlay ``KUMIHO_MEMORY_CODE_*`` env vars onto *base* (or defaults)."""
    cfg = base or CodeMemoryConfig()
    project = os.getenv("KUMIHO_MEMORY_CODE_PROJECT", "").strip()
    if project:
        cfg.project = project
    repo = os.getenv("KUMIHO_MEMORY_CODE_REPO", "").strip()
    if repo:
        cfg.repo = repo
    return cfg


def code_memory_enabled() -> bool:
    """The opt-in gate: ``KUMIHO_MEMORY_CODE=1`` (default off)."""
    return os.getenv("KUMIHO_MEMORY_CODE", "").strip() == "1"


def resolve_project_name(memory_project: str, config: CodeMemoryConfig) -> str:
    """Dedicated code project name: explicit config wins, else ``{project}-code``."""
    if config.project:
        return config.project
    return f"{memory_project}-code"


# ---------------------------------------------------------------------------
# Path normalization — the single-source contract shared by write & query
# ---------------------------------------------------------------------------

def normalize_path(path: str, repo_root: str = "") -> str:
    """Normalize *path* to the canonical repo-relative form used in slugs.

    Contract (write and query MUST run the same function): forward slashes,
    no leading ``./``, repo-root relativized when *repo_root* is given and
    *path* falls under it.  Case is deliberately left alone — ``slugify``
    already casefolds, so case differences converge at the slug level.
    """
    p = str(path or "").strip().replace("\\", "/")
    if not p:
        return ""
    if repo_root:
        root = str(repo_root).strip().replace("\\", "/").rstrip("/")
        if root:
            lowered, root_l = p.casefold(), root.casefold()
            if lowered == root_l:
                return ""
            if lowered.startswith(root_l + "/"):
                p = p[len(root) + 1 :]
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


# ---------------------------------------------------------------------------
# Slugs — identity rules (sha-free; see module docstring)
# ---------------------------------------------------------------------------

def anchor_slug(repo: str, path: str) -> str:
    """Anchor identity = ``(repo, normalized path)``.  Never the commit."""
    norm = normalize_path(path)
    if not norm:
        return ""
    return slugify(f"{repo}::{norm}", hash_on_truncate=True)


def decision_slug(title: str, author_date: Any, suffix: int = 0) -> str:
    """Decision identity = ``title + author-date`` (+ collision suffix).

    Author date survives rebase/squash (committer date does not), so
    re-mining after a history rewrite converges on the same slug — the
    non-rotting property.  The date suffix keeps same-titled decisions from
    different eras distinct; *suffix* (>= 2) disambiguates the residual
    same-day-same-title-different-decision case (collision guard in
    :func:`get_or_create_decision_item`).
    """
    day = _author_day(author_date)
    base = f"{title} {day}" if day else str(title)
    if suffix >= 2:
        base = f"{base} {suffix}"
    return slugify(base, hash_on_truncate=True)


def commit_slug(repo: str, commit_hash: str) -> str:
    """Commit marker identity — sha-keyed on purpose (it *is* the sha's
    processing marker; duplicates after a rewrite are harmless skips)."""
    return slugify(f"{repo}-{str(commit_hash)[:12]}", hash_on_truncate=True)


def evidence_slug(statement: str) -> str:
    """Evidence identity = the verbatim statement (re-mining converges)."""
    return slugify(statement, hash_on_truncate=True)


def _author_day(author_date: Any) -> str:
    """``YYYYMMDD`` from a datetime or ISO-8601 string; '' when unparseable."""
    if isinstance(author_date, datetime):
        return author_date.strftime("%Y%m%d")
    raw = str(author_date or "").strip()
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y%m%d")
    except ValueError:
        # Fall back to the date prefix of an almost-ISO string; identity
        # stability matters more than strict parsing here.
        digits = "".join(ch for ch in raw[:10] if ch.isdigit())
        return digits if len(digits) == 8 else ""


# ---------------------------------------------------------------------------
# Write path — get-or-create + embedding_text injection
# ---------------------------------------------------------------------------

_item_locks_guard = threading.Lock()
_item_locks: Dict[str, threading.Lock] = {}


def _slug_lock(slug: str) -> threading.Lock:
    """Per-slug lock so concurrent writers converge on one anchor revision
    (same pattern as entity_promotion, kept local — modular boundary)."""
    with _item_locks_guard:
        lock = _item_locks.get(slug)
        if lock is None:
            lock = threading.Lock()
            _item_locks[slug] = lock
        return lock


def get_or_create_item(project: Any, slug: str, kind: str, space_path: str) -> Any:
    """create-or-get an item, absorbing ALREADY_EXISTS races."""
    import grpc

    try:
        return project.create_item(slug, kind, parent_path=space_path)
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise
        return project.get_item(slug, kind, parent_path=space_path)


def ensure_space(project: Any, space_name: str) -> None:
    """Idempotent space creation."""
    import grpc

    try:
        project.create_space(space_name)
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.ALREADY_EXISTS:
            raise


def write_revision(item: Any, metadata: Dict[str, str], embedding_text: str = "") -> Any:
    """Create a revision with an explicit ``embedding_text``.

    ``item.create_revision`` has no ``embedding_text`` parameter, and when it
    is omitted the server auto-embeds *all* metadata concatenated — anchor
    paths, hashes, and bookkeeping keys would pollute the vector.  The
    client-level ``create_revision`` accepts the override; ``get_client()``
    is the public accessor the codebase already uses (dream_state,
    _graph_walk).  Every code-domain write goes through here.
    """
    import kumiho

    if not embedding_text:
        return item.create_revision(metadata=metadata)
    client = kumiho.get_client()
    return client.create_revision(item.kref, metadata=metadata, embedding_text=embedding_text)


def get_or_create_anchor(
    project: Any,
    config: CodeMemoryConfig,
    repo: str,
    path: str,
) -> Optional[Any]:
    """Get-or-create the ``(repo, path)`` anchor hub; returns its single
    anchor *revision* (edge target), or ``None`` for unusable paths."""
    norm = normalize_path(path)
    slug = anchor_slug(repo, path)
    if not slug:
        return None
    space_path = f"/{project.name}/{config.anchors_space}"
    item = get_or_create_item(project, slug, KIND_ANCHOR, space_path)
    with _slug_lock(slug):
        anchor = item.get_latest_revision()
        if anchor is None:
            anchor = item.create_revision(
                metadata={
                    "repo": repo,
                    "path": norm,
                    "display_name": norm.rsplit("/", 1)[-1],
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return anchor


def get_or_create_decision_item(
    project: Any,
    config: CodeMemoryConfig,
    title: str,
    author_date: Any,
    decision_text: str,
) -> Any:
    """Get-or-create a decision item with the slug collision guard.

    Same slug + genuinely different decision (token Jaccard of the stored
    ``decision`` metadata vs the incoming text below the threshold) means an
    unrelated same-day/same-title decision — retry with a numbered suffix
    instead of silently merging.  A wrong merge is unrecoverable; a spurious
    split is stitched later by the SUPERSEDES pass.
    """
    from kumiho_memory.relations import _jaccard, _tokens

    space_path = f"/{project.name}/{config.decisions_space}"
    suffix = 0
    while True:
        slug = decision_slug(title, author_date, suffix=suffix)
        item = get_or_create_item(project, slug, KIND_DECISION, space_path)
        latest = item.get_latest_revision()
        if latest is None:
            return item  # fresh item — ours to fill
        existing = str((getattr(latest, "metadata", None) or {}).get("decision", ""))
        if not existing or not decision_text:
            return item
        if _jaccard(_tokens(existing), _tokens(decision_text)) >= config.slug_collision_jaccard:
            return item  # same decision re-mined — converge
        suffix = 2 if suffix == 0 else suffix + 1
        if suffix > 9:  # pathological; give up on suffixing deterministically
            logger.warning(
                "decision slug collision guard exhausted for title=%r date=%s",
                title, author_date,
            )
            return item
