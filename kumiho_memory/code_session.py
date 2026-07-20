"""Decision Memory session mining: the conversation is the richest why-source.

A commit records the winning choice; the *session* that produced it holds
what the commit loses — the rejected alternatives, the measurements in their
original form, the reviewer findings, the moment agreement was reached.
This module mines an agent session's transcript into the code-decision graph
(docs/SESSION_MINING_DESIGN.md, issue #43 Phase 2):

* **Enrichment** — a session decision that correlates with an existing
  commit-mined ``code_decision`` (deterministic sha/anchor discovery +
  signal-conjunction judgment, §3) attaches its conversation-only evidence
  to that decision.  Enrichment is *additive by invariant*: new evidence
  nodes and edges only — never a new revision, never a metadata rewrite on
  the target (edges are revision-scoped; the supersede demote-in-place
  lesson).
* **Standalone capture** — a session decision that never reached a commit
  becomes a first-class ``code_decision`` with ``origin="session"``,
  anchored via ``role="mentioned"`` edges on ls-files-verified paths.
* **Bridge** — decisions link back to the consolidated conversation
  revision via ``DISCUSSED_IN`` (cross-project, kref; write happens only in
  the code domain, the conversation project is never touched).

Quality gates commits could not have: every evidence atom / alternative
quote is verified **verbatim against the (redacted) transcript packet** the
LLM saw, mentioned files against ``git ls-files``, mentioned commits against
``git rev-parse``.  Credential screening is per-atom — a raise-style
rejector applied to the whole session would make one pasted key permanently
unmineable (privacy.py raises; verified).

Everything is opt-in behind ``KUMIHO_MEMORY_CODE=1``; nothing here is
imported by the conversation recall/consolidation paths unless the
additional ``KUMIHO_MEMORY_CODE_AUTOMINE=1`` chain gate is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from kumiho_memory._bounded import run_bounded_in_thread
from kumiho_memory.code_decisions import (
    EDGE_DERIVED_FROM,
    EDGE_DISCUSSED_IN,
    EDGE_IMPLEMENTED_IN,
    EDGE_MOTIVATED_BY,
    EVIDENCE_KINDS,
    KIND_ANCHOR,
    KIND_COMMIT,
    KIND_EVIDENCE,
    KIND_SESSION,
    SCHEMA_VERSION,
    CodeMemoryConfig,
    anchor_slug,
    commit_slug,
    deprecate_marker_decisions,
    edge_source_uri,
    edge_target_uri,
    ensure_space,
    evidence_slug,
    get_or_create_anchor,
    get_or_create_decision_item,
    get_or_create_item,
    marker_provenance_complete,
    normalize_path,
    parse_decided_at,
    session_slug,
    undeprecate_item,
    write_revision,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionMineStats:
    messages_seen: int = 0
    messages_kept: int = 0
    chunks: int = 0
    llm_calls: int = 0
    decisions_created: int = 0
    decisions_enriched: int = 0
    evidence_added: int = 0
    anchors: int = 0
    edges: int = 0
    deprecated: int = 0
    skipped_marker: bool = False
    bridged: int = 0
    source: str = ""
    evidence_dropped_verbatim: int = 0
    evidence_dropped_dup: int = 0
    credentials_dropped: int = 0
    #: Per-candidate correlation trace ("why standalone / why this target").
    correlation_trace: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# [1] load — three sources, one shape
# ---------------------------------------------------------------------------

def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Uniform shape: ``{index, role, content, timestamp}`` with the message
    index as the stable ``m<idx>`` coordinate (source_ref / decided_at)."""
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(messages or []):
        content = str(m.get("content", "") or "")
        if not content.strip():
            continue
        ts = str(
            m.get("timestamp", "")
            or (m.get("metadata") or {}).get("timestamp", "")
            or ""
        )
        out.append({
            "index": i,
            "role": str(m.get("role", "user") or "user").lower(),
            "content": content,
            "timestamp": ts,
        })
    return out


def parse_claude_transcript(path: str) -> List[Dict[str, Any]]:
    """Parse a Claude Code / Cowork transcript JSONL into mine_session
    messages (``[{role, content, timestamp}]``).

    The transcript is JSONL — one JSON object per line.  Only user/assistant
    turns are kept; content blocks are flattened (text kept, tool_use marked,
    tool_result dropped as noise), and system-reminder injections are
    skipped.  This is the plugin SessionEnd input surface: the hook hands
    the worker a transcript path, the worker hands it here.  Kept in the SDK
    (not the plugin) so it is unit-tested and the plugin worker stays thin.
    """
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        message = entry.get("message") or entry
        role = str(message.get("role", ""))
        if role not in ("user", "assistant"):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(str(block.get("text", "")))
                    elif btype == "tool_use":
                        parts.append(f"*[tool: {block.get('name', 'unknown')}]*")
                    # tool_result: dropped (verbose, low decision signal)
            content = "\n".join(parts)
        elif not isinstance(content, str):
            continue
        content = content.strip()
        if not content or content.startswith("<system-reminder>"):
            continue
        ts = str(entry.get("timestamp") or message.get("timestamp") or "")
        out.append({"role": role, "content": content, "timestamp": ts})
    return out


#: Role headers as ``_build_conversation_markdown`` emits them:
#: ``### {role.capitalize()}`` — a single capitalized word on its own line.
_ARTIFACT_HEADER_RE = re.compile(r"^### ([A-Z][A-Za-z0-9_]*)\s*$")
_ARTIFACT_TS_RE = re.compile(r"^<sub>(.*)</sub>\s*$")


def parse_conversation_markdown(text: str) -> List[Dict[str, Any]]:
    """Reverse the deterministic ``_build_conversation_markdown`` format.

    Fallback input surface for sessions whose Redis buffer was already
    cleared by consolidation: the full transcript survives as a markdown
    artifact with ``### {Role}`` headers and ``<sub>{ts}</sub>`` stamps.
    Role, timestamp, and order are recovered; message metadata is
    accepted-lost.  The golden round-trip unit test pins builder and parser
    to one contract, so a format drift breaks loudly instead of silently.
    """
    lines = (text or "").split("\n")
    # Skip the front matter: everything up to and including the first ---
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "---":
            start = i + 1
            break
    messages: List[Dict[str, Any]] = []
    role: Optional[str] = None
    ts = ""
    buf: List[str] = []

    def _flush() -> None:
        if role is None:
            return
        content = "\n".join(buf).strip("\n").strip()
        if content:
            messages.append({"role": role.lower(), "content": content,
                             "timestamp": ts})

    for ln in lines[start:]:
        m = _ARTIFACT_HEADER_RE.match(ln)
        if m:
            _flush()
            role, ts, buf = m.group(1), "", []
            continue
        if role is not None and not buf:
            t = _ARTIFACT_TS_RE.match(ln.strip())
            if t:
                ts = t.group(1)
                continue
            if not ln.strip():
                continue
        if role is not None:
            buf.append(ln)
    _flush()
    return messages


# ---------------------------------------------------------------------------
# [2] salience — deterministic scoring, false-negative asymmetric
# ---------------------------------------------------------------------------

_DECISION_WORDS_RE = re.compile(
    r"decided|let'?s go with|we'?ll use|going with|opt(?:ed)? for|instead of|"
    r"rather than|reject(?:ed)?|revert|기각|채택|결정|선택|가기로|롤백|대신",
    re.IGNORECASE,
)
_ALTERNATIVE_RE = re.compile(
    r"why not\b|considered\b.{0,80}\bbut\b|\S+ 대신|\S+ 말고", re.IGNORECASE,
)
_MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|x\b|ms\b|s\b|tokens?\b|MB\b|점)|measured|benchmark|"
    r"\bF1\b|\brecall\b|p9\d|실측",
    re.IGNORECASE,
)
_REVIEW_RE = re.compile(r"critical|confirmed|defect|blocker|리뷰|지적", re.IGNORECASE)
_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_FILEPATH_RE = re.compile(r"\b[\w./\\-]+\.(?:py|rs|ts|tsx|js|go|java|c|cpp|h|md|toml|json|yaml|yml)\b")
_BACKTICK_RE = re.compile(r"`[^`\s][^`]*`")
_ASSENT_RE = re.compile(r"^\s*(?:yes|do it|go ahead|approve[d]?|좋아|그래|승인|ㄱㄱ)\b", re.IGNORECASE)
_BASE64ISH_RE = re.compile(r"[A-Za-z0-9+/=]{120,}")


def _salience(msg: Dict[str, Any], prev_score: int) -> int:
    text = msg["content"]
    score = 0
    if _DECISION_WORDS_RE.search(text):
        score += 3
    if _ALTERNATIVE_RE.search(text):
        score += 3
    if _MEASUREMENT_RE.search(text):
        score += 2
    if _REVIEW_RE.search(text):
        score += 2
    if _SHA_RE.search(text):
        score += 2
    identifiers = 0
    if _FILEPATH_RE.search(text):
        identifiers += 1
    identifiers += min(2, len(_BACKTICK_RE.findall(text)))
    score += min(3, identifiers)
    # Assent witness: a short user "yes/go ahead" right after a high-signal
    # message is the moment agreement was reached — must survive selection.
    if (
        msg["role"] == "user" and len(text) <= 80
        and prev_score >= 4 and _ASSENT_RE.search(text)
    ):
        score += 1
    # Noise: stack traces / base64 blobs / long pure-code pastes.
    lines = text.split("\n")
    if len(lines) >= 5:
        nonalpha = sum(
            1 for ln in lines
            if ln.strip() and sum(c.isalpha() for c in ln) / max(1, len(ln)) < 0.2
        )
        if nonalpha / len(lines) > 0.8:
            score -= 2
    if _BASE64ISH_RE.search(text):
        score -= 2
    if len(text) > 500 and text.count("\n") > 8 and not _DECISION_WORDS_RE.search(text):
        stripped = [ln for ln in lines if ln.strip()]
        codeish = sum(
            1 for ln in stripped
            if ln.startswith((" ", "\t", "def ", "class ", "import ", "}", "{"))
        )
        if stripped and codeish / len(stripped) > 0.7:
            score -= 2
    return score


def _truncate_message(text: str, budget: int) -> str:
    """Head+tail truncation — decision sentences live at the edges of a
    message; the middle is dominated by supporting pastes."""
    if len(text) <= budget:
        return text
    head = int(budget * 0.625)
    tail = budget - head
    return text[:head] + "\n[...]\n" + text[-tail:]


def select_messages(
    messages: List[Dict[str, Any]], config: CodeMemoryConfig,
) -> List[Dict[str, Any]]:
    """Salience selection: scored messages + their ±1 neighbors + the
    session frame (first 2, last 3), chronological, with elision markers."""
    if not messages:
        return []
    scores: List[int] = []
    prev = 0
    for m in messages:
        s = _salience(m, prev)
        scores.append(s)
        prev = s
    keep = set()
    for i, s in enumerate(scores):
        if s >= config.session_salience_min:
            keep.add(i)
            if i > 0:
                keep.add(i - 1)
            if i + 1 < len(messages):
                keep.add(i + 1)
    # Frame: the goal statement and the closing agreement live here.
    keep.update(range(min(2, len(messages))))
    keep.update(range(max(0, len(messages) - 3), len(messages)))

    out: List[Dict[str, Any]] = []
    elided = 0
    for i, m in enumerate(messages):
        if i in keep:
            if elided:
                out.append({"index": -1, "role": "system", "timestamp": "",
                            "content": f"[... {elided} messages elided (low signal)]"})
                elided = 0
            mm = dict(m)
            mm["score"] = scores[i]
            mm["content"] = _truncate_message(
                m["content"], config.session_per_message_chars,
            )
            out.append(mm)
        else:
            elided += 1
    if elided:
        out.append({"index": -1, "role": "system", "timestamp": "",
                    "content": f"[... {elided} messages elided (low signal)]"})
    return out


def build_chunks(
    session_id: str,
    selected: List[Dict[str, Any]],
    config: CodeMemoryConfig,
) -> List[str]:
    """Chronological chunks, split only at message boundaries, each message
    framed as ``[m<idx> <role> <ts>]`` (the source_ref / decided_at origin).
    Over the chunk cap, keep the highest mean-salience chunks (still in
    chronological order)."""
    rendered: List[Tuple[str, float]] = []
    for m in selected:
        if m["index"] < 0:
            rendered.append((m["content"], 0.0))
            continue
        frame = f"[m{m['index']} {m['role']}"
        if m["timestamp"]:
            frame += f" {m['timestamp']}"
        frame += "]"
        rendered.append((f"{frame}\n{m['content']}", float(m.get("score", 0))))

    chunks: List[Tuple[List[str], List[float]]] = [([], [])]
    used = 0
    for text, score in rendered:
        if used and used + len(text) > config.session_chunk_chars:
            chunks.append(([], []))
            used = 0
        chunks[-1][0].append(text)
        chunks[-1][1].append(score)
        used += len(text) + 2
    chunks = [c for c in chunks if c[0]]

    if len(chunks) > config.session_max_chunks:
        # Frame preservation (§2.3): the first and last chunks carry the
        # session frame — the goal statement and the closing agreement —
        # and are pinned ("무조건 포함"); density eviction only competes
        # over the middle.  Without the pin, a low-density opening chunk
        # (the goal is often stated before any high-salience signal) was
        # dropped wholesale.
        pinned = {0, len(chunks) - 1}
        budget = config.session_max_chunks - len(pinned)
        middle = [
            (sum(sc) / max(1, len(sc)), i)
            for i, (_, sc) in enumerate(chunks) if i not in pinned
        ]
        keep_idx = set(pinned) | {
            i for _, i in sorted(middle, key=lambda t: t[0], reverse=True)
            [: max(0, budget)]
        }
        if config.session_max_chunks < len(pinned):
            # Pathological cap (<2): keep the denser of the frame chunks.
            keep_idx = {max(
                pinned,
                key=lambda i: sum(chunks[i][1]) / max(1, len(chunks[i][1])),
            )}
        chunks = [c for i, c in enumerate(chunks) if i in keep_idx]

    total = len(chunks)
    out: List[str] = []
    for k, (texts, _) in enumerate(chunks, start=1):
        header = f"session {session_id}, chunk {k}/{total}"
        out.append(header + "\n\n" + "\n\n".join(texts))
    return out


# ---------------------------------------------------------------------------
# [4] structure — LLM extraction (session grammar of decision-worthiness)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You extract engineering DECISIONS from an agent-session transcript for "
    "a decision-memory graph. In a session, a decision is a SETTLED choice: "
    "(a) a proposal plus explicit acceptance ('let's / yes, do it'), (b) an "
    "alternative rejected with a spoken reason, (c) a measurement that "
    "forced a choice, or (d) a reviewer finding accepted with a fix "
    "direction. A decision is NOT an exploratory hypothesis that trailed "
    "off, a restatement of what the code already says, a TODO wish, or a "
    "question. Zero decisions is a valid answer — never invent one. "
    "`alternatives` is the session's unique cargo: capture every option "
    "that was considered and explicitly rejected or deferred, with the "
    "verbatim sentence carrying the rejection reason in `quote`. "
    "`evidence.text` and `alternatives[].quote` MUST be verbatim quotes "
    "from the transcript — a validator drops anything it cannot find in "
    "the original text. Copy every code identifier the session mentions "
    "(function/env/class names) into `symbols`, every file path into "
    "`files` EXACTLY as written in the transcript (full repo-relative "
    "path, never shortened), and every commit hash into "
    "`mentioned_commits` — these are the correlation coordinates that tie "
    "the decision back to the code. The transcript is windowed ([... "
    "elided] markers): never infer decisions from elided spans. Messages "
    "are framed as [m<index> <role> <timestamp>]; report message indexes "
    "where asked. Write titles/decisions/rationale in English; quotes "
    "stay original."
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
        "kind": {"type": "string",
                 "enum": [k for k in EVIDENCE_KINDS if k != "rejected_alternative"]},
        "text": {"type": "string"},
        "text_en": {"type": "string"},
        "message_index": {"type": "integer"},
    })
    alternative = obj({
        "option": {"type": "string"},
        "verdict": {"type": "string", "enum": ["rejected", "deferred"]},
        "quote": {"type": "string"},
        "quote_en": {"type": "string"},
        "message_index": {"type": "integer"},
    })
    decision = obj({
        "title": {"type": "string"},
        "decision": {"type": "string"},
        "rationale": {"type": "string"},
        "why_question": {"type": "string"},
        "symbols": {"type": "array", "items": {"type": "string"}},
        "files": {"type": "array", "items": {"type": "string"}},
        "mentioned_commits": {"type": "array", "items": {"type": "string"}},
        "alternatives": {"type": "array", "items": alternative},
        "evidence": {"type": "array", "items": evidence},
        "settled_by_message": {"type": "integer"},
        "status_hint": {"type": "string",
                        "enum": ["committed", "uncommitted", "unknown"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    })
    return obj({"decisions": {"type": "array", "items": decision}})


async def _structure_chunk(
    adapter: Any, model: str, packet: str, config: CodeMemoryConfig,
) -> List[Dict[str, Any]]:
    raw = await adapter.chat(
        messages=[{"role": "user", "content":
                   "Extract settled decisions from this session transcript "
                   "chunk.\n\n" + packet}],
        model=model,
        system=_SYSTEM_PROMPT,
        max_tokens=8192,
        json_mode={"name": "session_decisions",
                   "schema": _structuring_schema()},
    )
    data = json.loads(raw)
    return list(data.get("decisions", []))


# ---------------------------------------------------------------------------
# [5] validate — verbatim / sha / ls-files / per-atom credentials
# ---------------------------------------------------------------------------

def _norm_for_match(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _containment(quote: str, corpus_tokens: set) -> float:
    from kumiho_memory.relations import _tokens

    q = _tokens(quote)
    if not q:
        return 0.0
    return len(q & corpus_tokens) / len(q)


def _verbatim_ok(quote: str, packet_norm: str, corpus_tokens: set,
                 config: CodeMemoryConfig) -> bool:
    qn = _norm_for_match(quote)
    if not qn:
        return False
    if qn in packet_norm:
        return True
    return _containment(quote, corpus_tokens) >= config.evidence_containment


_HEX_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _verify_shas(repo_path: str, tokens: List[str]) -> Dict[str, str]:
    """hex-regex pre-screen (argv injection guard) + ``git rev-parse``
    existence check.  Returns ``{mentioned_token: full_sha}``."""
    from kumiho_memory.code_capture import _run_git

    out: Dict[str, str] = {}
    for tok in tokens or []:
        t = str(tok or "").strip().lower()
        if not _HEX_RE.match(t):
            continue
        try:
            full = _run_git(repo_path, "rev-parse", "--verify", "--quiet",
                            f"{t}^{{commit}}").strip()
        except Exception:  # noqa: BLE001 — unknown sha, not an error
            continue
        if full:
            out[t] = full
    return out


def _ls_files(repo_path: str) -> set:
    from kumiho_memory.code_capture import _run_git

    try:
        raw = _run_git(repo_path, "ls-files")
    except Exception:  # noqa: BLE001
        return set()
    return {normalize_path(l) for l in raw.splitlines() if l.strip()}


def _resolve_tracked_path(norm: str, tracked: set) -> str:
    """Deterministic path resolution against the ls-files ground truth.

    Models abbreviate repo-relative paths (a live extraction emitted
    ``kumiho_memory/recall_rerank.py`` for
    ``python/kumiho-memory/kumiho_memory/recall_rerank.py``).  An exact
    match wins; otherwise a UNIQUE suffix match (at a path-segment boundary)
    resolves the abbreviation.  Ambiguity means drop — a guessed anchor is
    worse than none."""
    if not norm:
        return ""
    if norm in tracked:
        return norm
    suffix = "/" + norm
    matches = [t for t in tracked if t.endswith(suffix)]
    return matches[0] if len(matches) == 1 else ""


def _drop_if_credential(redactor: Any, text: str, stats: SessionMineStats) -> bool:
    """Per-atom credential screen.  ``reject_credentials`` RAISES — applied
    to the whole session it would make one pasted key permanently unmineable
    (every retry hits the same raise).  Per atom, the atom drops and the
    session survives."""
    if redactor is None:
        return False
    try:
        redactor.reject_credentials(str(text or ""))
        return False
    except Exception:  # noqa: BLE001 — CredentialDetectedError et al.
        stats.credentials_dropped += 1
        return True


def _redact_packet_for_llm(
    redactor: Any, packet: str, stats: SessionMineStats,
) -> str:
    """Pre-LLM screen over one chunk packet (step [3]).

    Step [3] used to be ``anonymize_summary`` alone — PII only, no credential
    screen — while every OTHER leg of this module (``_drop_if_credential`` over
    decisions, quotes and ``session_line``) does screen credentials.  The gap
    was live on the consolidation flow: ``consolidate_session`` calls
    ``code_mine_session`` with the buffered conversation, so an API key pasted
    into chat was chunked and handed to the structuring model intact.

    Credentials are excised span-wise on the RAW packet FIRST — before the PII
    pass, which rewrites the digit runs the credential regexes anchor on and
    would otherwise defeat them — then PII is redacted in place.  Excising the
    matched span rather than dropping the containing line keeps hunk headers,
    paths and surrounding diff structure intact (anonymize content, not
    structure), and is the only screening that can see a multi-line PEM block.

    Deliberately does NOT add a whole-packet ``reject_credentials`` verdict:
    a packet is a large unit, and dropping one wholesale would starve the
    structuring model of a whole chunk.  The per-atom screen over what comes
    BACK (:func:`_drop_if_credential` over decisions, quotes and
    ``session_line``) stays as the second layer.
    """
    if redactor is None:
        return packet
    text = packet
    redact_credentials = getattr(redactor, "redact_credentials", None)
    if callable(redact_credentials):
        try:
            text, dropped = redact_credentials(text)
            stats.credentials_dropped += dropped
        except Exception:  # noqa: BLE001 — never crash the mine on redaction
            stats.credentials_dropped += 1
            return "[redacted]"
    return redactor.anonymize_summary(text)


def validate_session_decisions(
    candidates: List[Dict[str, Any]],
    *,
    packets: List[str],
    repo_path: str,
    tracked_files: set,
    config: CodeMemoryConfig,
    redactor: Any,
    stats: SessionMineStats,
) -> List[Dict[str, Any]]:
    """Deterministic hallucination defenses over the LLM output.

    The verification baseline is the REDACTED packet text — the same text
    the LLM saw and the same text that gets stored (single text stream:
    what is verified is what is written).
    """
    packet_norm = _norm_for_match("\n".join(packets))
    from kumiho_memory.relations import _tokens

    corpus_tokens = _tokens("\n".join(packets))

    out: List[Dict[str, Any]] = []
    for d in candidates[: config.session_max_decisions]:
        d = dict(d)
        if not str(d.get("title", "")).strip():
            continue
        # credentials: the decision's own prose
        core_text = " ".join(str(d.get(k, "")) for k in
                             ("title", "decision", "rationale", "why_question"))
        if _drop_if_credential(redactor, core_text, stats):
            continue
        # symbols are code identifiers, not prose: the credential gate applies
        # per entry (a pasted key stored as a "symbol" would reach metadata AND
        # the embedding text), but PII redaction does not — rewriting
        # identifiers would corrupt the correlation coordinates for no privacy
        # gain (an identifier is not PII).
        d["symbols"] = [
            s for s in (d.get("symbols") or [])
            if not _drop_if_credential(redactor, str(s), stats)
        ]

        evidence = []
        for ev in (d.get("evidence") or [])[: config.session_max_evidence_per_decision]:
            text = str(ev.get("text", "")).strip()
            if not text:
                continue
            if not _verbatim_ok(text, packet_norm, corpus_tokens, config):
                stats.evidence_dropped_verbatim += 1
                continue
            if _drop_if_credential(redactor, text, stats):
                continue
            evidence.append(dict(ev))
        d["evidence"] = evidence

        alternatives = []
        for alt in (d.get("alternatives") or [])[: config.session_max_alternatives_per_decision]:
            quote = str(alt.get("quote", "")).strip()
            option = str(alt.get("option", "")).strip()
            if not quote or not option:
                continue
            if not _verbatim_ok(quote, packet_norm, corpus_tokens, config):
                stats.evidence_dropped_verbatim += 1
                continue
            if _drop_if_credential(redactor, quote, stats):
                continue
            alternatives.append(dict(alt))
        d["alternatives"] = alternatives

        # files: git ls-files is the ground truth (the session's analogue of
        # the commit path's changed-file check).  All-dropped decisions
        # survive anchor-less (semantic-only).
        files = []
        for f in d.get("files") or []:
            resolved = _resolve_tracked_path(normalize_path(str(f)), tracked_files)
            if resolved and resolved not in files:
                files.append(resolved)
            elif not resolved:
                logger.debug("code session: dropping unverified file %r", f)
        d["files"] = files[: config.max_anchors_per_decision]

        # commits: verified {token: full_sha}
        d["verified_commits"] = _verify_shas(
            repo_path, list(d.get("mentioned_commits") or []),
        )

        if (
            config.drop_low_confidence_without_evidence
            and d.get("confidence") == "low"
            and not evidence and not alternatives
        ):
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# [6] correlate — deterministic discovery + signal conjunction (§3)
# ---------------------------------------------------------------------------

def _decision_sources_of(rev: Any, edge_type: str) -> List[Any]:
    """INCOMING *edge_type* source revisions of *rev* that are decision
    nodes (``decision`` in metadata — evidence atoms share the same
    provenance edges and must not be treated as correlation targets)."""
    import kumiho

    out: List[Any] = []
    try:
        edges = rev.get_edges(edge_type_filter=edge_type, direction=kumiho.INCOMING)
    except Exception:  # noqa: BLE001
        return out
    for edge in edges or []:
        src = edge_source_uri(edge)
        if not src:
            continue
        try:
            src_rev = kumiho.get_revision(src)
        except Exception:
            continue
        meta = getattr(src_rev, "metadata", {}) or {}
        if "decision" in meta:
            out.append(src_rev)
    return out


def _discover_targets(
    project: Any,
    config: CodeMemoryConfig,
    repo: str,
    candidate: Dict[str, Any],
) -> Tuple[List[Any], List[Any]]:
    """T1 (verified sha -> commit marker -> its decisions) and
    T2 (verified file -> anchor hub -> its decisions).  Graph/git only —
    probabilistic search never picks a merge target."""
    t1: List[Any] = []
    for full_sha in (candidate.get("verified_commits") or {}).values():
        slug = commit_slug(repo, full_sha)
        try:
            item = project.get_item(
                slug, KIND_COMMIT,
                parent_path=f"/{project.name}/{config.commits_space}",
            )
            marker_rev = item.get_latest_revision() if item is not None else None
        except Exception:  # noqa: BLE001
            marker_rev = None
        if marker_rev is not None:
            t1.extend(_decision_sources_of(marker_rev, EDGE_DERIVED_FROM))

    t2: List[Any] = []
    for f in candidate.get("files") or []:
        slug = anchor_slug(repo, f)
        if not slug:
            continue
        try:
            item = project.get_item(
                slug, KIND_ANCHOR,
                parent_path=f"/{project.name}/{config.anchors_space}",
            )
            anchor_rev = item.get_latest_revision() if item is not None else None
        except Exception:  # noqa: BLE001
            anchor_rev = None
        if anchor_rev is not None:
            t2.extend(_decision_sources_of(anchor_rev, EDGE_IMPLEMENTED_IN))
    return t1, t2


def _lex_text(fields: Dict[str, Any]) -> str:
    """The FULL prose a decision carries.  Sessions and commits title the
    same decision differently, but they share the rationale vocabulary (the
    actual why) — title+decision alone measured 0.14 on a live same-decision
    pair, full prose 0.26 (dogfood-calibrated)."""
    return " ".join(
        str(fields.get(k, "") or "")
        for k in ("title", "decision", "rationale", "why_question")
    )


def correlate(
    project: Any,
    config: CodeMemoryConfig,
    repo: str,
    candidate: Dict[str, Any],
    session_ts: Any,
    trace: Optional[List[str]] = None,
    session_id: str = "",
) -> Optional[Dict[str, Any]]:
    """ENRICH target or ``None`` (-> standalone).  Wrong-merge is
    unrecoverable, wrong-split is stitchable — lexical similarity alone can
    NEVER merge; every path requires a structural signal in conjunction.

    *trace* (when given) records why each discovered target was accepted or
    rejected — the diagnosis surface for "why did this become standalone?".
    """
    from kumiho_memory.relations import _jaccard, _tokens

    def _t(msg: str) -> None:
        if trace is not None:
            trace.append(msg)

    t1, t2 = _discover_targets(project, config, repo, candidate)
    _t(f"targets: sha-pool={len(t1)} anchor-pool={len(t2)}")
    cand_tokens = _tokens(_lex_text(candidate))
    cand_symbols = {s.strip().casefold() for s in candidate.get("symbols") or [] if s.strip()}
    ts = parse_decided_at(session_ts)

    best: Optional[Tuple[float, Any, str]] = None  # (lex, rev, correlation)
    seen: set = set()
    for rev, correlation, pool in (
        [(r, "sha", "t1") for r in t1] + [(r, "anchored", "t2") for r in t2]
    ):
        uri = getattr(getattr(rev, "kref", None), "uri", "")
        if not uri or (uri, pool) in seen:
            continue
        seen.add((uri, pool))
        meta = getattr(rev, "metadata", {}) or {}
        # Never enrich onto a RETIRED decision, and never self-correlate onto
        # this session's OWN decisions.  Both matter on the --force path: the
        # pre-pass deprecates this session's origin=session decisions, and an
        # anchored one is then rediscovered via its own IMPLEMENTED_IN edge —
        # with lex ~1.0 (it is literally itself) it would win the ENRICH
        # branch, which never un-deprecates, stranding the decision retired
        # forever.  Filtering here forces it back to the standalone branch,
        # where undeprecate_item restores it.
        if str(meta.get("status", "")) == "deprecated":
            _t(f"skip {meta.get('title', '')[:40]!r}: target is deprecated")
            continue
        if session_id and str(meta.get("session_id", "")) == session_id:
            _t(f"skip {meta.get('title', '')[:40]!r}: same-session (no self-enrich)")
            continue
        lex = _jaccard(cand_tokens, _tokens(_lex_text(meta)))
        tag = f"{correlation} {meta.get('title', '')[:40]!r} lex={lex:.2f}"
        if correlation == "sha":
            if lex < config.correlate_jaccard_sha:
                _t(f"reject {tag}: below sha sanity floor "
                   f"{config.correlate_jaccard_sha}")
                continue  # sanity floor — a misquoted sha must not merge
        else:
            if lex < config.correlate_jaccard_anchored:
                _t(f"reject {tag}: below anchored floor "
                   f"{config.correlate_jaccard_anchored}")
                continue
            target_symbols = {
                s.strip().casefold()
                for s in str(meta.get("symbols", "")).split(",") if s.strip()
            }
            if not (cand_symbols & target_symbols) and lex < config.correlate_jaccard_blind:
                _t(f"reject {tag}: no symbol overlap and below blind "
                   f"{config.correlate_jaccard_blind}")
                continue  # conjunction: symbol overlap OR strong lexical
            target_ts = parse_decided_at(meta.get("decided_at", ""))
            if ts is None or target_ts is None or abs(ts - target_ts) > timedelta(
                days=config.correlate_window_days,
            ):
                _t(f"reject {tag}: outside {config.correlate_window_days}-day window")
                continue  # same file, different era — split, don't merge
        _t(f"accept {tag}")
        if best is None or lex > best[0] or (
            lex == best[0]
            and _newer(getattr(rev, "metadata", {}) or {}, best[1])
        ):
            best = (lex, rev, correlation)

    if best is None:
        return None
    return {"rev": best[1], "overlap": best[0], "correlation": best[2]}


def _newer(meta: Dict[str, str], other_rev: Any) -> bool:
    a = parse_decided_at(meta.get("decided_at", ""))
    b = parse_decided_at((getattr(other_rev, "metadata", {}) or {}).get("decided_at", ""))
    if a is None:
        return False
    return b is None or a > b


# ---------------------------------------------------------------------------
# [7]-[9] write / bridge / marker (blocking worker — one session at a time)
# ---------------------------------------------------------------------------

def _session_source_ref(session_id: str, message_index: Any) -> str:
    try:
        idx = int(message_index)
    except (TypeError, ValueError):
        idx = -1
    if idx >= 0:
        return f"session:{session_id}#m{idx}"
    return f"session:{session_id}"


def _existing_evidence_statements(decision_rev: Any) -> List[str]:
    """OUTGOING MOTIVATED_BY statements already on the decision — the
    near-duplicate cut baseline."""
    import kumiho

    out: List[str] = []
    try:
        edges = decision_rev.get_edges(
            edge_type_filter=EDGE_MOTIVATED_BY, direction=0,
        )
    except Exception:  # noqa: BLE001
        return out
    for edge in edges or []:
        dst = edge_target_uri(edge)
        if not dst:
            continue
        try:
            rev = kumiho.get_revision(dst)
        except Exception:
            continue
        stmt = str((getattr(rev, "metadata", {}) or {}).get("statement", ""))
        if stmt:
            out.append(stmt)
    return out


def _evidence_atoms(candidate: Dict[str, Any], session_id: str) -> List[Dict[str, str]]:
    """Uniform evidence-atom view: rejected alternatives FIRST (they are the
    session's unique cargo — the per-decision cap must never crowd them out
    behind plain evidence), then evidence.  Statements stay verbatim
    (original language); the optional ``*_en`` translation is embedding-only
    (PR#20 mixed-script fragmentation)."""
    atoms: List[Dict[str, str]] = []
    for alt in candidate.get("alternatives") or []:
        atoms.append({
            "statement": str(alt.get("quote", "")),
            "statement_en": str(alt.get("quote_en", "") or ""),
            "evidence_kind": "rejected_alternative",
            "alternative": str(alt.get("option", "")),
            "source_ref": _session_source_ref(session_id, alt.get("message_index")),
        })
    for ev in candidate.get("evidence") or []:
        atoms.append({
            "statement": str(ev.get("text", "")),
            "statement_en": str(ev.get("text_en", "") or ""),
            "evidence_kind": str(ev.get("kind", "constraint")),
            "alternative": "",
            "source_ref": _session_source_ref(session_id, ev.get("message_index")),
        })
    return atoms


def _write_evidence_atoms(
    project: Any,
    config: CodeMemoryConfig,
    decision_rev: Any,
    atoms: List[Dict[str, str]],
    edge_meta: Dict[str, str],
    stats: SessionMineStats,
) -> List[Any]:
    """Dedup 3 layers (§4.2), then get-or-create + MOTIVATED_BY.

    Layer 1 is free slug convergence: an identical sentence reuses the
    existing (possibly commit-sourced) node — its metadata is NEVER touched
    (the first source's honesty is preserved; the edge metadata's session_id
    witnesses the session mention).  Returns the NEWLY CREATED evidence
    revisions (marker provenance targets).
    """
    from kumiho_memory.relations import _jaccard, _tokens

    from kumiho_memory.code_capture import _create_edge_once

    existing = _existing_evidence_statements(decision_rev)
    existing_tokens = [_tokens(s) for s in existing]
    created: List[Any] = []
    added = 0
    for atom in atoms:
        if added >= config.session_max_evidence_per_decision:
            break
        stmt = atom["statement"].strip()
        slug = evidence_slug(stmt)
        if not slug:
            continue
        # Layer 2: near-duplicate cut against what the decision already has.
        tok = _tokens(stmt)
        if any(_jaccard(tok, et) >= config.evidence_dup_jaccard for et in existing_tokens):
            stats.evidence_dropped_dup += 1
            continue
        item = get_or_create_item(
            project, slug, KIND_EVIDENCE,
            f"/{project.name}/{config.evidence_space}",
        )
        rev = item.get_latest_revision()
        if rev is None:
            meta = {
                "statement": stmt,
                "evidence_kind": atom["evidence_kind"],
                "source_ref": atom["source_ref"],
                "schema_version": SCHEMA_VERSION,
            }
            if atom["alternative"]:
                meta["alternative"] = atom["alternative"]
            if atom["statement_en"]:
                meta["statement_en"] = atom["statement_en"]
            rev = write_revision(
                item, meta, embedding_text=atom["statement_en"] or stmt,
            )
            stats.evidence_added += 1
            created.append(rev)
        _create_edge_once(decision_rev, rev, EDGE_MOTIVATED_BY, dict(edge_meta), stats)
        existing_tokens.append(tok)
        added += 1
    return created


def _compose_session_embedding_text(
    meta: Dict[str, str],
    alternatives: List[Dict[str, str]],
    session_line: str,
) -> str:
    """Session variant of the §1.6 composition — rejected alternatives join
    the embedding because real queries often arrive under the REJECTED
    option's name ('why not asyncio.to_thread?'); doc2query in reverse."""
    basenames = ", ".join(
        f.rsplit("/", 1)[-1] for f in str(meta.get("files", "")).split(",") if f
    )
    parts = []
    if meta.get("why_question"):
        parts.append(meta["why_question"])
    parts.append(f"{meta.get('decision', '')}.")
    if meta.get("rationale"):
        parts.append(f"Rationale: {meta['rationale']}.")
    alts = "; ".join(
        f"{a.get('option', '')} ({a.get('quote_en') or a.get('quote', '')})"
        for a in alternatives if a.get("option")
    )
    if alts:
        parts.append(f"Rejected alternatives: {alts}.")
    anchored = basenames
    if meta.get("symbols"):
        anchored = f"{basenames} ({meta['symbols']})" if basenames else meta["symbols"]
    if anchored:
        parts.append(f"Anchored: {anchored}.")
    if session_line:
        parts.append(f'Session: "{session_line}".')
    return " ".join(p for p in parts if p)


def _bridge(
    decision_rev: Any,
    conversation_kref: str,
    session_id: str,
    stats: SessionMineStats,
) -> None:
    """DISCUSSED_IN, code -> conversation (cross-project by kref).  The
    write happens in the code domain only; the conversation project's nodes
    and metadata are untouched, and the edge type is outside the recall
    graph-walk defaults — the bridge cannot perturb conversation recall."""
    import kumiho

    from kumiho_memory.code_capture import _create_edge_once

    if not conversation_kref:
        return
    try:
        conv_rev = kumiho.get_revision(conversation_kref)
    except Exception as exc:  # noqa: BLE001
        stats.errors.append(f"bridge target unresolvable: {exc}")
        return
    before = stats.edges
    _create_edge_once(decision_rev, conv_rev, EDGE_DISCUSSED_IN,
                      {"session_id": session_id}, stats)
    if stats.edges > before:
        stats.bridged += 1


def _session_marker_complete(project: Any, config: CodeMemoryConfig, slug: str) -> Tuple[bool, Optional[Any]]:
    """Marker-completeness check (see
    :func:`code_decisions.marker_provenance_complete`): the promised edge
    count is the sum of the three per-session counters.  Returns
    (complete, marker_rev)."""
    try:
        item = project.get_item(slug, KIND_SESSION,
                                parent_path=f"/{project.name}/{config.sessions_space}")
    except Exception:  # noqa: BLE001
        return False, None
    return marker_provenance_complete(
        item, ("decisions_created", "decisions_enriched", "evidence_added"),
    )


def _force_deprecate_session_decisions(
    project: Any, config: CodeMemoryConfig, slug: str, session_id: str,
    stats: SessionMineStats,
) -> None:
    """--force pre-pass, session flavor of the commit pattern.

    The skip predicate is the essence: the marker's INCOMING DERIVED_FROM
    sources include commit-origin decisions this session merely ENRICHED
    (the audit hub, §3.4) — the bare commit pattern would deprecate someone
    else's commit decisions.  Only ``origin == "session"`` decisions of THIS
    session are retired; evidence atoms are shared assets and never touched.
    """
    try:
        item = project.get_item(slug, KIND_SESSION,
                                parent_path=f"/{project.name}/{config.sessions_space}")
        marker_rev = item.get_latest_revision() if item is not None else None
    except Exception:  # noqa: BLE001
        return
    deprecate_marker_decisions(
        marker_rev, stats,
        skip=lambda meta: (
            str(meta.get("origin", "")) != "session"
            or str(meta.get("session_id", "")) != session_id
        ),
    )


def _sync_write_session(
    project_name: str,
    config: CodeMemoryConfig,
    repo: str,
    session_id: str,
    candidates: List[Dict[str, Any]],
    *,
    conversation_kref: str,
    message_count: int,
    source: str,
    session_last_ts: str,
    session_line: str,
    stats: SessionMineStats,
    force: bool = False,
    mark_complete: bool = True,
) -> None:
    """Stages [6]-[9] for one session.  Crash-safe: the marker revision is
    the very last write; a partial failure leaves the session unmarked and
    the next run retries (all writes are get-or-create / existence-checked).

    ``mark_complete=False`` (an LLM chunk failed to structure) writes the
    decisions that DID extract but withholds the marker, so the session is
    not recorded as processed and the next run re-mines the failed chunk —
    a missing LLM result is a failure, never a zero-decision verdict (the
    commit-ingest principle, applied to chunks)."""
    import kumiho

    from kumiho_memory.code_capture import _create_edge_once

    project = kumiho.get_project(project_name)
    if project is None:
        project = kumiho.create_project(project_name)
    for space in (config.decisions_space, config.anchors_space,
                  config.commits_space, config.evidence_space,
                  config.sessions_space):
        ensure_space(project, space)

    # provenance targets for the marker (written LAST).  Deduped by kref:
    # the marker's completeness check counts UNIQUE incoming edges, so a
    # second candidate converging on the same decision must not inflate the
    # promised count past what the edges can ever satisfy.
    provenance_revs: List[Any] = []
    provenance_seen: set = set()

    def _add_provenance(rev: Any) -> bool:
        uri = getattr(getattr(rev, "kref", None), "uri", "")
        if uri and uri in provenance_seen:
            return False
        if uri:
            provenance_seen.add(uri)
        provenance_revs.append(rev)
        return True

    for cand in candidates:
        settled_ts = str(cand.get("_settled_ts", "") or session_last_ts)
        atoms = _evidence_atoms(cand, session_id)

        trace: List[str] = [f"candidate {str(cand.get('title', ''))[:50]!r}"]
        target = correlate(project, config, repo, cand, settled_ts,
                           trace=trace, session_id=session_id)
        stats.correlation_trace.extend(trace)
        if target is not None:
            # --- ENRICH: additive only.  No new revision on the target, no
            # metadata writes anywhere on it (§3.4 invariant).
            decision_rev = target["rev"]
            edge_meta = {
                "session_id": session_id,
                "correlation": target["correlation"],
                "overlap": f"{target['overlap']:.2f}",
            }
            created = _write_evidence_atoms(
                project, config, decision_rev, atoms, edge_meta, stats,
            )
            for rev in created:
                _add_provenance(rev)
            if _add_provenance(decision_rev):
                stats.decisions_enriched += 1
            _bridge(decision_rev, conversation_kref, session_id, stats)
            continue

        # --- STANDALONE: Phase-1 write path, session provenance.
        files_csv = ",".join(cand.get("files") or [])
        meta = {
            "title": str(cand["title"])[:80],
            "summary": f"{cand.get('decision', '')} — {cand.get('rationale', '')}"[:400],
            "decision": str(cand.get("decision", "")),
            "rationale": str(cand.get("rationale", "")),
            "why_question": str(cand.get("why_question", "")),
            "symbols": ",".join(cand.get("symbols") or []),
            "repo": repo,
            "commit_hash": "",
            "files": files_csv,
            "line_ranges": "",
            "author": "",
            "decided_at": settled_ts,
            "confidence": str(cand.get("confidence", "medium")),
            "status": "active",
            "origin": "session",
            "session_id": session_id,
            "source_ref": _session_source_ref(
                session_id, cand.get("settled_by_message"),
            ),
            "status_hint": str(cand.get("status_hint", "unknown")),
            "schema_version": SCHEMA_VERSION,
        }
        item = get_or_create_decision_item(
            project, config, meta["title"], settled_ts, meta["decision"],
        )
        if force:
            undeprecate_item(item)
        decision_rev = item.get_latest_revision()
        if decision_rev is None or force:
            decision_rev = write_revision(
                item, meta,
                _compose_session_embedding_text(
                    meta, cand.get("alternatives") or [], session_line,
                ),
            )

        # anchors: role=mentioned — "a not-yet-committed decision touches
        # this file" is exactly what an agent about to edit it must know.
        for f in cand.get("files") or []:
            anchor_rev = get_or_create_anchor(project, config, repo, f)
            if anchor_rev is None:
                continue
            stats.anchors += 1
            _create_edge_once(decision_rev, anchor_rev, EDGE_IMPLEMENTED_IN, {
                "commit_hash": "",
                "role": "mentioned",
                "session_id": session_id,
            }, stats)

        created = _write_evidence_atoms(
            project, config, decision_rev, atoms,
            {"session_id": session_id}, stats,
        )
        for rev in created:
            _add_provenance(rev)
        if _add_provenance(decision_rev):
            stats.decisions_created += 1
        _bridge(decision_rev, conversation_kref, session_id, stats)

    # [9] marker LAST: session node + provenance edges.  Withheld entirely
    # when a chunk failed to structure — no marker means the next run
    # re-mines (successful decisions already written converge idempotently).
    if not mark_complete:
        return
    slug = session_slug(repo, session_id)
    marker_item = get_or_create_item(
        project, slug, KIND_SESSION, f"/{project_name}/{config.sessions_space}",
    )
    marker_rev = marker_item.get_latest_revision()
    # Reaching this write means a real (re-)mining pass ran — the complete-
    # marker skip returns before here.  A delta-triggered re-mine (marker
    # present, not force) MUST refresh the marker, or its stale message_count
    # keeps tripping the delta threshold and every subsequent mine re-pays
    # full LLM cost forever.  Rewrite whenever the recorded count is stale.
    recorded_mc = (
        str((getattr(marker_rev, "metadata", {}) or {}).get("message_count", ""))
        if marker_rev is not None else ""
    )
    if marker_rev is None or force or recorded_mc != str(message_count):
        marker_rev = write_revision(marker_item, {
            "repo": repo,
            "session_id": session_id,
            "mined_at": session_last_ts,
            "message_count": str(message_count),
            "source": source,
            "decisions_created": str(stats.decisions_created),
            "decisions_enriched": str(stats.decisions_enriched),
            "evidence_added": str(stats.evidence_added),
            "conversation_kref": conversation_kref,
            "capture_version": SCHEMA_VERSION,
            "schema_version": SCHEMA_VERSION,
        }, embedding_text=session_line)
    for rev in provenance_revs:
        _create_edge_once(rev, marker_rev, EDGE_DERIVED_FROM, {}, stats)


def _sync_bridge_only_pass(
    marker_rev: Any,
    conversation_kref: str,
    session_id: str,
    stats: SessionMineStats,
) -> None:
    """LLM-free reconciliation: a kref arrived after mining ("manual mine,
    consolidate later").  Backfill DISCUSSED_IN on the marker's decision
    sources and update the marker's own metadata (the marker belongs to the
    session domain — updating it is allowed; foreign decisions are not)."""
    for rev in _decision_sources_of(marker_rev, EDGE_DERIVED_FROM):
        _bridge(rev, conversation_kref, session_id, stats)
    try:
        marker_rev.set_attribute("conversation_kref", conversation_kref)
    except Exception as exc:  # noqa: BLE001
        logger.debug("code session: marker kref update failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def mine_session(
    session_id: str,
    *,
    project_name: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    conversation_kref: str = "",
    repo_path: str = ".",
    config: Optional[CodeMemoryConfig] = None,
    adapter: Any = None,
    model: str = "",
    redactor: Any = None,
    redis_buffer: Any = None,
    memory_project: str = "",
    force: bool = False,
) -> SessionMineStats:
    """Mine one agent session into the code-decision graph.

    Input surface (§2.1): explicit *messages* (hook/chain in-band) win, then
    the live Redis buffer, then the consolidated conversation artifact
    (markdown re-parse).  The compressed summary is never an input — it
    loses exactly the verbatim this pass exists to keep.
    """
    config = config or CodeMemoryConfig()
    stats = SessionMineStats()
    if not session_id:
        stats.errors.append("session_id is required")
        return stats
    if adapter is None or not model:
        stats.errors.append("no LLM adapter/model configured")
        return stats

    from kumiho_memory.code_capture import derive_repo_id

    repo = (config.repo or derive_repo_id(repo_path)).strip()

    # --- [1] load
    source = ""
    if messages:
        source = "explicit"
    elif redis_buffer is not None:
        try:
            result = await redis_buffer.get_messages(
                project=memory_project, session_id=session_id, limit=1000,
            )
            messages = list((result or {}).get("messages") or [])
            if messages:
                source = "redis"
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"redis load failed: {exc}")
    if not messages and conversation_kref:
        markdown = await run_bounded_in_thread(
            lambda: _load_conversation_artifact(conversation_kref),
            timeout=config.write_timeout, label="code session artifact load",
            on_timeout=None, on_error=None,
        )
        if markdown:
            messages = parse_conversation_markdown(markdown)
            if messages:
                source = "artifact"
    if not messages:
        stats.errors.append(
            "no transcript available (Redis empty/expired and no artifact) — "
            "the session cannot be mined"
        )
        return stats
    stats.source = source

    normalized = _normalize_messages(messages)
    stats.messages_seen = len(normalized)
    if not normalized:
        stats.errors.append("transcript contains no non-empty messages")
        return stats

    # --- idempotency (marker) / force / bridge-only reconciliation
    slug = session_slug(repo, session_id)

    def _sync_marker_state() -> Tuple[bool, Optional[Any]]:
        import kumiho

        project = kumiho.get_project(project_name)
        if project is None:
            return False, None
        return _session_marker_complete(project, config, slug)

    if force:
        def _sync_force() -> bool:
            import kumiho

            project = kumiho.get_project(project_name)
            if project is None:
                return False
            _force_deprecate_session_decisions(
                project, config, slug, session_id, stats,
            )
            return True

        await run_bounded_in_thread(
            _sync_force, timeout=config.write_timeout,
            label="code session force deprecate", on_timeout=False, on_error=False,
        )
    else:
        state = await run_bounded_in_thread(
            _sync_marker_state, timeout=config.write_timeout,
            label="code session marker check", on_timeout=(False, None),
            on_error=(False, None),
        ) or (False, None)
        complete, marker_rev = state
        if complete and marker_rev is not None:
            marker_meta = getattr(marker_rev, "metadata", {}) or {}
            recorded = int(marker_meta.get("message_count", "0") or 0)
            if (
                stats.messages_seen
                <= recorded + config.session_remine_message_delta
            ):
                # Bridge-only reconciliation before skipping (§5.2 [3]).
                recorded_kref = str(marker_meta.get("conversation_kref", ""))
                if conversation_kref and conversation_kref != recorded_kref:
                    await run_bounded_in_thread(
                        lambda: _sync_bridge_only_pass(
                            marker_rev, conversation_kref, session_id, stats,
                        ) or True,
                        timeout=config.write_timeout,
                        label="code session bridge pass",
                        on_timeout=None, on_error=None,
                    )
                stats.skipped_marker = True
                return stats
            # message_count grew past the delta: full re-mine (slug
            # convergence + dup cuts + edge existence checks keep it safe).

    # --- [2] salience + chunks
    selected = select_messages(normalized, config)
    stats.messages_kept = sum(1 for m in selected if m["index"] >= 0)
    packets = build_chunks(session_id, selected, config)
    stats.chunks = len(packets)
    if not packets:
        stats.errors.append("no salient content selected")
        return stats

    # --- [3] redact BEFORE the LLM: what the model sees is what the
    # validator checks against is what gets stored (single text stream).
    # Credentials AND PII — see _redact_packet_for_llm for why the order
    # (credentials first, on the raw text) is load-bearing.
    if redactor is not None:
        packets = [_redact_packet_for_llm(redactor, p, stats) for p in packets]

    # --- [4] structure (concurrency 2)
    sem = asyncio.Semaphore(2)
    chunk_failed = False

    async def _run_chunk(packet: str) -> List[Dict[str, Any]]:
        nonlocal chunk_failed
        async with sem:
            try:
                out = await _structure_chunk(adapter, model, packet, config)
                stats.llm_calls += 1
                return out
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"structuring failed: {exc}")
                chunk_failed = True
                return []

    chunk_results = await asyncio.gather(*(_run_chunk(p) for p in packets))
    candidates = [d for chunk in chunk_results for d in chunk]

    # --- [5] validate
    tracked = await run_bounded_in_thread(
        lambda: _ls_files(repo_path),
        timeout=config.write_timeout, label="code session ls-files",
        on_timeout=set(), on_error=set(),
    ) or set()
    ts_by_index = {m["index"]: m["timestamp"] for m in normalized}
    last_ts = next(
        (m["timestamp"] for m in reversed(normalized) if m["timestamp"]), "",
    )
    validated = validate_session_decisions(
        candidates, packets=packets, repo_path=repo_path,
        tracked_files=tracked, config=config, redactor=redactor, stats=stats,
    )
    for d in validated:
        try:
            settled = int(d.get("settled_by_message", -1))
        except (TypeError, ValueError):
            settled = -1
        d["_settled_ts"] = ts_by_index.get(settled, "") or last_ts

    # session_line feeds STORED embedding_text (marker + standalone
    # decisions), so it must go through the same redaction/credential
    # discipline as everything else that leaves the machine (§5.1: the
    # stored text stream is redacted text, no exceptions).  Redact and
    # screen the FULL first user line before truncating — an 80-char cut
    # could split a credential right past the detector.
    session_line = ""
    for m in normalized:
        if m["role"] == "user":
            line = " ".join(m["content"].split())
            if redactor is not None:
                line = redactor.anonymize_summary(line)
                if _drop_if_credential(redactor, line, stats):
                    # NOT "" — session_line feeds write_revision's
                    # embedding_text on the marker (and standalone-decision
                    # composition); an empty string takes the embed-ALL-metadata
                    # fallback (hashes/author/bookkeeping-key vector pollution).
                    # Mirror the commit path's F4 placeholder (code_capture
                    # subject, PR #111).
                    line = "[redacted]"
            session_line = line[:80]
            break

    # --- [6]-[9] write (single sync worker, marker last)
    ok = await run_bounded_in_thread(
        lambda: _sync_write_session(
            project_name, config, repo, session_id, validated,
            conversation_kref=conversation_kref,
            message_count=stats.messages_seen,
            source=source, session_last_ts=last_ts,
            session_line=session_line, stats=stats, force=force,
            mark_complete=not chunk_failed,
        ) or True,
        timeout=config.write_timeout, label="code session write",
        on_timeout=None, on_error=None,
    )
    if ok is not True:
        stats.errors.append("session write failed or timed out (unmarked — "
                            "the next run retries)")
    return stats


def _load_conversation_artifact(conversation_kref: str) -> Optional[str]:
    """Local markdown transcript of the consolidated conversation, resolved
    through the revision's artifact list.  Raw conversations never leave the
    machine — the artifact location is a local path by design."""
    import kumiho

    try:
        rev = kumiho.get_revision(conversation_kref)
        artifacts = rev.get_artifacts() or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("code session: artifact lookup failed: %s", exc)
        return None
    for art in artifacts:
        location = str(getattr(art, "location", "") or "")
        if not location.endswith(".md"):
            continue
        path = location[7:] if location.startswith("file://") else location
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            continue
    return None
