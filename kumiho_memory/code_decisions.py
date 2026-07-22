"""Decision Memory schema: code-domain decisions anchored to git.

The code domain captures the one thing git cannot hold — *why* the code is
the way it is.  Decisions, their rationale, and their evidence become typed
graph nodes; the code itself is never copied.  Every decision points at git
via anchors ({repo, path} hub nodes) and edge metadata ({commit_hash,
line_start, line_end}), so the memory never rots: git stays the source of
truth for *what*, the graph holds the *why*.

Design (docs/DECISION_MEMORY_DESIGN.md, issue #43):

* **Physical isolation.**  Code nodes live in a dedicated ``{project}-decisions``
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

Everything is opt-in behind ``KUMIHO_MEMORY_DECISIONS=1`` (the deprecated
``KUMIHO_MEMORY_CODE`` is still read as a fallback); nothing here is
imported by the conversation recall/consolidation paths.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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
KIND_SESSION = "code_session"

#: Edge types.  ``create_edge`` accepts free-form UPPERCASE strings
#: (``kumiho/edge.py::validate_edge_type``) — no server changes involved.
#: ``DERIVED_FROM`` / ``SUPERSEDES`` reuse the graph-wide semantics the
#: traversal defaults already know.
EDGE_IMPLEMENTED_IN = "IMPLEMENTED_IN"   # code_decision -> code_anchor
EDGE_MOTIVATED_BY = "MOTIVATED_BY"       # code_decision -> code_evidence
EDGE_DERIVED_FROM = "DERIVED_FROM"       # code_decision -> code_commit | code_session
EDGE_SUPERSEDES = "SUPERSEDES"           # newer decision -> older decision
EDGE_DISCUSSED_IN = "DISCUSSED_IN"       # code_decision -> conversation revision (kref)

#: Evidence taxonomy (``code_evidence.evidence_kind``).
#: ``rejected_alternative`` is session mining's unique cargo — the option
#: that was considered and turned down, with the verbatim rejection sentence.
EVIDENCE_KINDS = (
    "measurement",
    "review_finding",
    "incident",
    "benchmark",
    "constraint",
    "rejected_alternative",
)


@dataclass
class CodeMemoryConfig:
    """Configuration for the code-decision domain.

    ``project`` defaults to ``{memory_project}-decisions`` — derived at wiring
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
    sessions_space: str = "sessions"

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

    # --- session mining budgets (consumed by code_session, §4.5) ---
    session_salience_min: int = 2
    session_per_message_chars: int = 800
    session_chunk_chars: int = 18000
    session_max_chunks: int = 6
    session_max_decisions: int = 8
    session_max_evidence_per_decision: int = 4
    session_max_alternatives_per_decision: int = 4
    #: Near-duplicate cut: a session evidence atom whose statement shares
    #: this token-Jaccard with one already MOTIVATED_BY the target decision
    #: is a rephrasing, not new information — skip it.
    evidence_dup_jaccard: float = 0.8
    #: Relaxed verbatim match: token containment floor when the exact
    #: normalized-substring check fails (multi-message paraphrase slack).
    evidence_containment: float = 0.6
    # --- correlation thresholds (§3.3: conjunction, biased to split) ---
    #: Live-dogfood calibrated: honest same-decision session/commit pairs
    #: measured ~0.26 over the FULL prose (title+decision+rationale+why),
    #: misquoted-sha negatives ~0.1 — 0.20 splits them with margin while
    #: the sha remains the structural witness.
    correlate_jaccard_sha: float = 0.20
    #: Same measured basis as the sha floor: when lex moved to FULL prose
    #: (honest pairs ~0.26), the anchored floor had to move with it — at the
    #: draft's 0.35 an honest pair could never enrich via the anchor path
    #: (dead zone).  The structural signal here is the shared anchor file
    #: plus the symbol-overlap/blind conjunction below, not the floor.
    correlate_jaccard_anchored: float = 0.20
    correlate_jaccard_blind: float = 0.50
    correlate_window_days: int = 14
    #: Re-mine a marked session when this many new messages arrived since.
    session_remine_message_delta: int = 10


#: The Decision Memory env family was renamed ``KUMIHO_MEMORY_CODE*`` ->
#: ``KUMIHO_MEMORY_DECISIONS*`` (product-name alignment).  The legacy names are
#: still read as a deprecated fallback because the master gate is documented in
#: the kumiho-memory plugin's SKILL.md (a *different* repo) as
#: ``KUMIHO_MEMORY_CODE=1`` — hard-renaming would silently stop activating the
#: feature for anyone following that doc.  One warning per legacy name.
_warned_legacy_envs: set = set()
_warned_legacy_guard = threading.Lock()


def _warn_legacy_env(legacy_name: str, new_name: str) -> None:
    """Emit a one-time deprecation warning for a legacy env var name."""
    with _warned_legacy_guard:
        if legacy_name in _warned_legacy_envs:
            return
        _warned_legacy_envs.add(legacy_name)
    logger.warning(
        "%s is deprecated and will be removed; use %s instead. Reading the "
        "legacy value for now.",
        legacy_name, new_name,
    )


def _env_with_legacy(new_name: str, legacy_name: str) -> str:
    """Read *new_name*, falling back to the deprecated *legacy_name*.

    The new name always wins when set (even to an empty string); only when it
    is entirely unset is the legacy CODE name consulted, and using it triggers
    a one-time :func:`_warn_legacy_env`.  Returns the raw string ("" when
    neither is set), so callers keep their own ``.strip()``/casefold contract.
    """
    val = os.getenv(new_name)
    if val is not None:
        return val
    legacy = os.getenv(legacy_name)
    if legacy is not None:
        _warn_legacy_env(legacy_name, new_name)
        return legacy
    return ""


def config_from_env(base: Optional[CodeMemoryConfig] = None) -> CodeMemoryConfig:
    """Overlay ``KUMIHO_MEMORY_DECISIONS_*`` env vars onto *base* (or defaults).

    The legacy ``KUMIHO_MEMORY_CODE_*`` names are honored as a deprecated
    fallback (see :func:`_env_with_legacy`).
    """
    cfg = base or CodeMemoryConfig()
    project = _env_with_legacy(
        "KUMIHO_MEMORY_DECISIONS_PROJECT", "KUMIHO_MEMORY_CODE_PROJECT",
    ).strip()
    if project:
        cfg.project = project
    repo = _env_with_legacy(
        "KUMIHO_MEMORY_DECISIONS_REPO", "KUMIHO_MEMORY_CODE_REPO",
    ).strip()
    if repo:
        cfg.repo = repo
    return cfg


def code_memory_enabled() -> bool:
    """The opt-in gate: ``KUMIHO_MEMORY_DECISIONS=1|true|yes|on`` (default off).

    The deprecated ``KUMIHO_MEMORY_CODE`` is read as a fallback when the new
    name is unset (see :func:`_env_with_legacy`).  Read at call time by the
    manager delegation, but the MCP tool registry reads it once at import —
    long-lived MCP servers must restart to pick up a gate change (documented
    behavior, mirrors the other env-gated wiring).
    """
    return _env_with_legacy(
        "KUMIHO_MEMORY_DECISIONS", "KUMIHO_MEMORY_CODE",
    ).strip().casefold() in (
        "1", "true", "yes", "on",
    )


def code_automine_enabled() -> bool:
    """Double opt-in for the consolidation chain (§2.2c): the master gate
    AND ``KUMIHO_MEMORY_DECISIONS_AUTOMINE`` must both be on.  Chaining an LLM
    pass onto consolidation latency is an explicit consent matter.  The
    deprecated ``KUMIHO_MEMORY_CODE_AUTOMINE`` is read as a fallback."""
    return code_memory_enabled() and _env_with_legacy(
        "KUMIHO_MEMORY_DECISIONS_AUTOMINE", "KUMIHO_MEMORY_CODE_AUTOMINE",
    ).strip().casefold() in ("1", "true", "yes", "on")


def resolve_project_name(memory_project: str, config: CodeMemoryConfig) -> str:
    """Dedicated decisions project name: explicit config wins, else ``{project}-decisions``.

    Guard: the whole isolation story rests on code nodes living in a
    *different* project than conversation memory (the measured
    vector-crowding incident class).  An explicit override equal to the
    conversation project would silently defeat that, so it is corrected to
    ``{project}-decisions`` with a warning instead of being honored.
    """
    if config.project:
        if memory_project and config.project == memory_project:
            logger.warning(
                "KUMIHO_MEMORY_DECISIONS_PROJECT=%r equals the conversation "
                "memory project — physical isolation requires a separate "
                "project; using %r instead.",
                config.project, f"{memory_project}-decisions",
            )
            return f"{memory_project}-decisions"
        return config.project
    return f"{memory_project}-decisions"


def parse_decided_at(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 author date into an aware datetime, or ``None``.

    The single source for temporal comparisons: git emits author dates with
    the author's LOCAL UTC offset, so raw string comparison misorders
    cross-timezone histories — every ordering decision must go through here.
    """
    if isinstance(value, datetime):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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


def session_slug(repo: str, session_id: str) -> str:
    """Session marker identity — repo-qualified so the same session_id
    reused against a different repo can never false-skip the marker."""
    return slugify(f"{repo}-session-{session_id}", hash_on_truncate=True)


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
        if suffix > 9:
            # Pathological pile-up: NEVER wrong-merge (unrecoverable) —
            # fall back to a deterministic content-hash suffix so the same
            # decision text still converges on re-mining.
            import hashlib

            digest = hashlib.sha1(decision_text.encode("utf-8")).hexdigest()[:8]
            slug = slugify(f"{title} {_author_day(author_date)} {digest}",
                           hash_on_truncate=True)
            logger.warning(
                "decision slug collision guard exhausted for title=%r date=%s"
                " — using content-hash slug %r",
                title, author_date, slug,
            )
            return get_or_create_item(project, slug, KIND_DECISION, space_path)


# ---------------------------------------------------------------------------
# Marker & force machinery — shared by commit capture and session mining
# ---------------------------------------------------------------------------

def edge_source_uri(edge: Any) -> str:
    """kref URI of an edge's source revision; '' when absent."""
    return getattr(getattr(edge, "source_kref", None), "uri", "") or ""


def edge_target_uri(edge: Any) -> str:
    """kref URI of an edge's target revision; '' when absent."""
    return getattr(getattr(edge, "target_kref", None), "uri", "") or ""


def undeprecate_item(item: Any) -> None:
    """Restore an item a force re-capture converged on.

    ``--force`` deprecates the old generation up front; when the fresh
    extraction converges on the same slug, the item must come back active
    (the new revision carries the new content)."""
    if not getattr(item, "deprecated", False):
        return
    try:
        item.set_deprecated(False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("code memory: un-deprecate failed: %s", exc)


def marker_provenance_complete(
    item: Any, expected_fields: Iterable[str],
) -> Tuple[bool, Optional[Any]]:
    """Marker completeness = revision exists AND carries at least the
    promised number of incoming DERIVED_FROM edges.

    The marker revision is written before its provenance edges (edges need
    the revision as a target), so a crash in that window would leave a
    marker that silently loses provenance forever — verifying the edge
    count against the metadata promise (the summed *expected_fields*) turns
    that window into a retry.  Returns ``(complete, marker_rev)``.
    """
    import kumiho

    if item is None:
        return False, None
    try:
        rev = item.get_latest_revision()
        if rev is None:
            return False, None
        meta = getattr(rev, "metadata", {}) or {}
        expected = sum(int(meta.get(k, "0") or 0) for k in expected_fields)
        if expected <= 0:
            return True, rev
        edges = rev.get_edges(edge_type_filter=EDGE_DERIVED_FROM,
                              direction=kumiho.INCOMING)
        return len(edges or []) >= expected, rev
    except Exception:  # noqa: BLE001
        return False, None


def deprecate_marker_decisions(
    marker_rev: Any,
    stats: Any,
    skip: Optional[Callable[[Dict[str, str]], bool]] = None,
) -> None:
    """--force pre-pass core: deprecate the marker's decision sources.

    Walks the marker's INCOMING ``DERIVED_FROM`` sources; evidence atoms
    (no ``decision`` in metadata) are shared assets and never touched, and
    a caller-supplied *skip* predicate narrows which decisions are this
    force's to retire (the session flavor must not deprecate commit-origin
    decisions it merely enriched).  ``stats`` is duck-typed on
    ``.deprecated`` (IngestStats | SessionMineStats).
    """
    import kumiho

    if marker_rev is None:
        return
    try:
        edges = marker_rev.get_edges(edge_type_filter=EDGE_DERIVED_FROM,
                                     direction=kumiho.INCOMING)
    except Exception:  # noqa: BLE001
        return
    for edge in edges or []:
        src = edge_source_uri(edge)
        if not src:
            continue
        try:
            rev = kumiho.get_revision(src)
            meta = getattr(rev, "metadata", {}) or {}
            if "decision" not in meta:
                continue  # evidence atom — shared, never deprecated here
            if skip is not None and skip(meta):
                continue
            rev.set_attribute("status", "deprecated")
            rev.get_item().set_deprecated(True)
            stats.deprecated += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("code memory: force deprecation failed for %s: %s",
                         src, exc)
