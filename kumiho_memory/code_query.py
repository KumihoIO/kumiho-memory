"""Decision Memory query engine: ask *why* code is the way it is.

Three legs, fused lexicographically (docs/DECISION_MEMORY_DESIGN.md §5):

1. **anchor leg** (deterministic) — ``file`` resolves to its ``(repo, path)``
   anchor hub by slug; INCOMING ``IMPLEMENTED_IN`` edges are the decisions.
   No search infrastructure involved: an anchor miss means "no recorded
   decision for this file", definitively (no fuzzy fallback — the semantic
   leg exists separately, and polluting the deterministic leg would turn
   fact into probability).
2. **semantic leg** — the natural-language question against the decisions
   space via direct ``kumiho.search`` (typed nodes carry no published/latest
   tags, so the retrieve tool would silently drop them — same lesson as the
   conversation fact-recall leg).
3. **evidence-bridge leg** — the same question against the evidence space;
   hits promote to their deciding node via INCOMING ``MOTIVATED_BY``.  Catches
   questions phrased closer to the *measurement* than to the decision
   ("displacement measurement"-style).

Fusion never sums scores across legs: an anchor match is *factual* evidence
("a decision about this file"), a cross-encoder score is *probabilistic* —
mixing the tiers would let a confident-sounding wrong answer outrank a
recorded fact.  Sort key is lexicographic::

    (anchor_line_hit, anchor_hit, active, ce_score)

Superseded decisions sink to the bottom of their tier and always carry
``superseded_by`` — an agent must never receive a reversed decision as the
answer without seeing what replaced it.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Dict, List, Optional, Tuple

from kumiho_memory._bounded import run_bounded_in_thread
from kumiho_memory.code_decisions import (
    CodeMemoryConfig,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTED_IN,
    EDGE_MOTIVATED_BY,
    EDGE_SUPERSEDES,
    KIND_DECISION,
    KIND_EVIDENCE,
    anchor_slug,
    normalize_path,
)

logger = logging.getLogger(__name__)

#: Deadline for each blocking SDK leg (daemon-thread poll).
QUERY_TIMEOUT = 30.0

#: Fan-out guards for the evidence-chain expansion (§5.3).
MAX_EDGES_PER_DECISION = 32
CHAIN_FETCH_FACTOR = 3


# ---------------------------------------------------------------------------
# Cross-encoder primitive (deliberately NOT rerank_async — §5.2)
# ---------------------------------------------------------------------------

async def _ce_scores(
    question: str,
    texts: List[str],
    reranker: Optional[Any],
) -> Optional[List[float]]:
    """Score *texts* against *question* with the cross-encoder, off-loop.

    Reuses only ``recall_rerank``'s primitives: the reranker callable and the
    dedicated single-worker executor (the cfec845 decision itself).  The full
    ``rerank_async`` is deliberately not used — its memory-dict shaping and
    recency/MMR semantics belong to the conversation domain.
    """
    if reranker is None or not question or len(texts) < 2:
        return None
    from kumiho_memory.recall_rerank import _rerank_executor

    try:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            _rerank_executor(), functools.partial(reranker, question, list(texts)),
        )
        scores = [float(s) for s in raw]
        return scores if len(scores) == len(texts) else None
    except Exception as exc:  # noqa: BLE001 — CE failure must never kill a why()
        logger.debug("code why: cross-encoder scoring failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Internal candidate model
# ---------------------------------------------------------------------------

def _kref_uri(obj: Any) -> str:
    return getattr(getattr(obj, "kref", None), "uri", "") or ""


def _new_candidate(kref: str, meta: Dict[str, str]) -> Dict[str, Any]:
    return {
        "kref": kref,
        "meta": dict(meta or {}),
        "anchor_hit": False,
        "anchor_line_hit": False,
        "semantic_score": 0.0,
        "in_semantic": False,
        "evidence_bridge": False,
        "anchor_edge_meta": {},
    }


def _line_intersects(edge_meta: Dict[str, str], line: int, slack: int) -> bool:
    try:
        start = int(str(edge_meta.get("line_start", "")).strip() or -1)
        end = int(str(edge_meta.get("line_end", "")).strip() or -1)
    except ValueError:
        return False
    if start < 0 or end < 0:
        return False
    return (start - slack) <= line <= (end + slack)


# ---------------------------------------------------------------------------
# Legs (blocking workers — each runs inside run_bounded_in_thread)
# ---------------------------------------------------------------------------

def _sync_anchor_leg(
    project_name: str,
    config: CodeMemoryConfig,
    repo: str,
    file: str,
    line: Optional[int],
    commit: Optional[str],
) -> List[Dict[str, Any]]:
    """Deterministic leg.  An anchor MISS (NOT_FOUND) is a definitive
    "no recorded decision"; any OTHER failure propagates — a transient
    outage must degrade the leg visibly (why() records a warning), never
    masquerade as a confident empty answer."""
    import grpc
    import kumiho

    project = kumiho.get_project(project_name)
    if project is None:
        return []
    slug = anchor_slug(repo, file)
    if not slug:
        return []
    space_path = f"/{project_name}/{config.anchors_space}"
    try:
        item = project.get_item(slug, "code_anchor", parent_path=space_path)
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.NOT_FOUND:
            return []  # definitive miss
        raise  # transient/server error → degraded leg, not an empty verdict
    if item is None:
        return []
    anchor_rev = item.get_latest_revision()
    if anchor_rev is None:
        return []

    out: List[Dict[str, Any]] = []
    try:
        edges = anchor_rev.get_edges(
            edge_type_filter=EDGE_IMPLEMENTED_IN, direction=kumiho.INCOMING,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("code why: anchor edge walk failed: %s", exc)
        return []
    for edge in edges or []:
        src = getattr(getattr(edge, "source_kref", None), "uri", "")
        if not src:
            continue
        try:
            rev = kumiho.get_revision(src)
        except Exception:
            continue
        meta = getattr(rev, "metadata", {}) or {}
        cand = _new_candidate(src, meta)
        cand["anchor_hit"] = True
        cand["anchor_edge_meta"] = dict(getattr(edge, "metadata", {}) or {})
        if line is not None and _line_intersects(
            cand["anchor_edge_meta"], int(line), config.line_slack,
        ):
            cand["anchor_line_hit"] = True
        if commit:
            edge_hash = str(cand["anchor_edge_meta"].get("commit_hash", ""))
            meta_hash = str(meta.get("commit_hash", ""))
            prefix = str(commit).strip()
            if prefix and (edge_hash.startswith(prefix) or meta_hash.startswith(prefix)):
                cand["commit_match"] = True
        out.append(cand)
    return out


def _sync_semantic_leg(
    project_name: str,
    config: CodeMemoryConfig,
    question: str,
    scan_limit: int,
) -> List[Dict[str, Any]]:
    import kumiho

    try:
        hits = kumiho.search(
            question,
            context=f"{project_name}/{config.decisions_space}",
            kind=KIND_DECISION,
            include_revision_metadata=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("code why: semantic search failed: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for r in (hits or [])[:scan_limit]:
        item = getattr(r, "item", None)
        if item is None:
            continue
        try:
            rev = item.get_latest_revision()
        except Exception:
            continue
        if rev is None:
            continue
        kref = _kref_uri(rev)
        if not kref:
            continue
        cand = _new_candidate(kref, getattr(rev, "metadata", {}) or {})
        cand["in_semantic"] = True
        cand["semantic_score"] = float(getattr(r, "score", 0.0) or 0.0)
        out.append(cand)
    return out


def _sync_evidence_bridge_leg(
    project_name: str,
    config: CodeMemoryConfig,
    question: str,
    scan_limit: int,
) -> List[Dict[str, Any]]:
    import kumiho

    try:
        hits = kumiho.search(
            question,
            context=f"{project_name}/{config.evidence_space}",
            kind=KIND_EVIDENCE,
            include_revision_metadata=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("code why: evidence search failed: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for r in (hits or [])[:scan_limit]:
        item = getattr(r, "item", None)
        if item is None:
            continue
        try:
            ev_rev = item.get_latest_revision()
        except Exception:
            continue
        if ev_rev is None:
            continue
        try:
            edges = ev_rev.get_edges(
                edge_type_filter=EDGE_MOTIVATED_BY, direction=kumiho.INCOMING,
            )
        except Exception:
            continue
        for edge in (edges or [])[:MAX_EDGES_PER_DECISION]:
            src = getattr(getattr(edge, "source_kref", None), "uri", "")
            if not src:
                continue
            try:
                rev = kumiho.get_revision(src)
            except Exception:
                continue
            cand = _new_candidate(src, getattr(rev, "metadata", {}) or {})
            cand["evidence_bridge"] = True
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Evidence-chain expansion (§5.3 — after the limit cut)
# ---------------------------------------------------------------------------

def _sync_expand_chain(kref: str, max_fetch: int) -> Dict[str, Any]:
    import kumiho

    chain: Dict[str, Any] = {
        "evidence": [], "commits": [], "supersedes": [], "superseded_by": None,
    }
    try:
        rev = kumiho.get_revision(kref)
        edges = rev.get_edges(direction=kumiho.BOTH)
    except Exception as exc:  # noqa: BLE001
        logger.debug("code why: chain expansion failed for %s: %s", kref, exc)
        return chain

    # SUPERSEDES edges are processed FIRST: superseded_by must never be
    # dropped by the fetch budget — a superseded decision without its
    # replacement is the exact state hard requirement 5 forbids.
    def _prio(e: Any) -> int:
        return 0 if getattr(e, "edge_type", "") == EDGE_SUPERSEDES else 1

    scan = sorted((edges or [])[:MAX_EDGES_PER_DECISION], key=_prio)
    fetched = 0
    me = kref
    for edge in scan:
        if fetched >= max_fetch:
            break
        etype = getattr(edge, "edge_type", "")
        src = getattr(getattr(edge, "source_kref", None), "uri", "")
        dst = getattr(getattr(edge, "target_kref", None), "uri", "")
        try:
            if etype == EDGE_MOTIVATED_BY and src == me:
                ev = kumiho.get_revision(dst); fetched += 1
                m = getattr(ev, "metadata", {}) or {}
                chain["evidence"].append({
                    "statement": m.get("statement", ""),
                    "kind": m.get("evidence_kind", ""),
                    "source_ref": m.get("source_ref", ""),
                })
            elif etype == EDGE_DERIVED_FROM and src == me:
                cm = kumiho.get_revision(dst); fetched += 1
                m = getattr(cm, "metadata", {}) or {}
                chain["commits"].append({
                    "sha": m.get("hash", ""),
                    "subject": m.get("subject", ""),
                    "date": m.get("committed_at", ""),
                })
            elif etype == EDGE_SUPERSEDES and src == me:
                old = kumiho.get_revision(dst); fetched += 1
                m = getattr(old, "metadata", {}) or {}
                chain["supersedes"].append({"kref": dst, "title": m.get("title", "")})
            elif etype == EDGE_SUPERSEDES and dst == me:
                new = kumiho.get_revision(src); fetched += 1
                m = getattr(new, "metadata", {}) or {}
                chain["superseded_by"] = {"kref": src, "title": m.get("title", "")}
        except Exception:  # noqa: BLE001 — partial chains are fine
            continue
    return chain


# ---------------------------------------------------------------------------
# Fusion + public API
# ---------------------------------------------------------------------------

def _merge_candidates(*legs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for leg in legs:
        for cand in leg:
            cur = merged.get(cand["kref"])
            if cur is None:
                merged[cand["kref"]] = cand
                continue
            cur["anchor_hit"] = cur["anchor_hit"] or cand["anchor_hit"]
            cur["anchor_line_hit"] = cur["anchor_line_hit"] or cand["anchor_line_hit"]
            cur["in_semantic"] = cur["in_semantic"] or cand["in_semantic"]
            cur["evidence_bridge"] = cur["evidence_bridge"] or cand["evidence_bridge"]
            cur["semantic_score"] = max(cur["semantic_score"], cand["semantic_score"])
            if cand.get("commit_match"):
                cur["commit_match"] = True
            if cand["anchor_edge_meta"] and not cur["anchor_edge_meta"]:
                cur["anchor_edge_meta"] = cand["anchor_edge_meta"]
            if cand["meta"] and not cur["meta"]:
                cur["meta"] = cand["meta"]
    return merged


def _sort_candidates(
    cands: List[Dict[str, Any]],
    ce_by_kref: Optional[Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Lexicographic fusion (§5.2).

    ``(anchor_line_hit, anchor_hit, active, ce_or_fallback, both_legs,
    commit_match, decided_at)`` — anchor facts dominate probability scores;
    superseded decisions sink within their factual tier; the probabilistic
    slot is the CE score when available, else the semantic-leg server score
    (anchor-only candidates tie there and fall through to recency).
    """
    def key(c: Dict[str, Any]) -> Tuple:
        meta = c["meta"]
        active = 0 if str(meta.get("status", "active")) == "superseded" else 1
        if ce_by_kref is not None and c["kref"] in ce_by_kref:
            prob = ce_by_kref[c["kref"]]
        else:
            prob = c["semantic_score"]
        both = 1 if (c["anchor_hit"] and c["in_semantic"]) else 0
        commit_match = 1 if c.get("commit_match") else 0
        # Recency tiebreak through PARSED timestamps — author dates carry
        # mixed local UTC offsets, so raw string comparison misorders them.
        from kumiho_memory.code_decisions import parse_decided_at

        dt = parse_decided_at(meta.get("decided_at", ""))
        decided_ts = dt.timestamp() if dt is not None else 0.0
        return (
            1 if c["anchor_line_hit"] else 0,
            1 if c["anchor_hit"] else 0,
            active,
            prob,
            both,
            commit_match,
            decided_ts,
        )

    return sorted(cands, key=key, reverse=True)


def _answer_from(cand: Dict[str, Any], chain: Dict[str, Any]) -> Dict[str, Any]:
    meta = cand["meta"]
    match = (
        "anchor+line" if cand["anchor_line_hit"]
        else "anchor" if cand["anchor_hit"]
        else "semantic"
    )
    anchors = []
    files = [f for f in str(meta.get("files", "")).split(",") if f.strip()]
    ranges = {
        part.split(":", 1)[0]: part.split(":", 1)[1]
        for part in str(meta.get("line_ranges", "")).split(";")
        if ":" in part
    }
    for f in files:
        f = f.strip()
        anchors.append({
            "file": f,
            "lines": ranges.get(f, ""),
            "commit": str(meta.get("commit_hash", ""))[:12],
        })
    return {
        "kref": cand["kref"],
        "title": meta.get("title", ""),
        "decision": meta.get("decision", ""),
        "rationale": meta.get("rationale", ""),
        "why_question": meta.get("why_question", ""),
        "confidence": meta.get("confidence", ""),
        "decided_at": meta.get("decided_at", ""),
        "status": meta.get("status", "active"),
        "anchors": anchors,
        "evidence": chain["evidence"],
        "commits": chain["commits"],
        "supersedes": chain["supersedes"],
        "superseded_by": chain["superseded_by"],
        "match": match,
        "score": cand.get("_final_score", 0.0),
    }


async def why(
    question: Optional[str] = None,
    *,
    file: Optional[str] = None,
    line: Optional[int] = None,
    commit: Optional[str] = None,
    repo: Optional[str] = None,
    limit: int = 5,
    project_name: str,
    config: Optional[CodeMemoryConfig] = None,
    reranker: Optional[Any] = None,
) -> Dict[str, Any]:
    """Answer "why is this code the way it is?" from captured decisions.

    At least one of *question* / *file* is required.  Returns
    ``{"decisions": [DecisionAnswer...], "context": str}`` where ``context``
    is a ready-to-inject markdown rendering (:func:`compose_why_context`).
    """
    config = config or CodeMemoryConfig()
    if not question and not file:
        return {"decisions": [], "context": ""}
    repo_id = (repo or config.repo or "").strip()
    if not repo_id and file:
        # Anchor slugs were written with a repo id derived at capture time;
        # querying with an empty repo would slug a name that was never
        # written and silently kill the deterministic leg (reviewed-and-
        # confirmed defect).  Mirror the capture-side derivation from the
        # current working directory.
        try:
            from kumiho_memory.code_capture import derive_repo_id

            repo_id = derive_repo_id(".")
        except Exception as exc:  # noqa: BLE001
            logger.debug("code why: repo id derivation failed: %s", exc)
    limit = max(1, int(limit))
    scan_limit = max(limit * 2, 10)
    warnings: List[str] = []

    legs: List[List[Dict[str, Any]]] = []
    if file:
        anchor_cands = await run_bounded_in_thread(
            functools.partial(
                _sync_anchor_leg, project_name, config, repo_id,
                normalize_path(file), line, commit,
            ),
            timeout=QUERY_TIMEOUT, label="code why anchor leg",
            on_timeout=None, on_error=None,
        )
        if anchor_cands is None:
            warnings.append(
                "anchor leg degraded (backend error) — results are "
                "semantic-only and may miss file-anchored decisions"
            )
            anchor_cands = []
        legs.append(anchor_cands)
    if question:
        semantic_cands = await run_bounded_in_thread(
            functools.partial(
                _sync_semantic_leg, project_name, config, question, scan_limit,
            ),
            timeout=QUERY_TIMEOUT, label="code why semantic leg",
            on_timeout=[], on_error=[],
        ) or []
        legs.append(semantic_cands)
        evidence_cands = await run_bounded_in_thread(
            functools.partial(
                _sync_evidence_bridge_leg, project_name, config, question, scan_limit,
            ),
            timeout=QUERY_TIMEOUT, label="code why evidence leg",
            on_timeout=[], on_error=[],
        ) or []
        legs.append(evidence_cands)

    merged = list(_merge_candidates(*legs).values())
    if not merged:
        out: Dict[str, Any] = {"decisions": [], "context": ""}
        if warnings:
            out["warnings"] = warnings
        return out

    ce_by_kref: Optional[Dict[str, float]] = None
    if question and len(merged) >= 2:
        texts = [
            f"{c['meta'].get('title', '')}. {c['meta'].get('summary', '')}"
            for c in merged
        ]
        scores = await _ce_scores(question, texts, reranker)
        if scores is not None:
            ce_by_kref = {c["kref"]: s for c, s in zip(merged, scores)}

    ranked = _sort_candidates(merged, ce_by_kref)[:limit]
    for i, cand in enumerate(ranked):
        if ce_by_kref is not None and cand["kref"] in ce_by_kref:
            cand["_final_score"] = round(float(ce_by_kref[cand["kref"]]), 6)
        else:
            cand["_final_score"] = round(float(cand["semantic_score"]), 6)

    max_fetch = limit * CHAIN_FETCH_FACTOR
    answers: List[Dict[str, Any]] = []
    for cand in ranked:
        chain = await run_bounded_in_thread(
            functools.partial(_sync_expand_chain, cand["kref"], max_fetch),
            timeout=QUERY_TIMEOUT, label="code why chain",
            on_timeout=None, on_error=None,
        ) or {"evidence": [], "commits": [], "supersedes": [], "superseded_by": None}
        answers.append(_answer_from(cand, chain))

    result: Dict[str, Any] = {
        "decisions": answers,
        "context": compose_why_context(answers),
    }
    if warnings:
        result["warnings"] = warnings
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _flat(text: Any) -> str:
    """Flatten newlines in captured text before context injection.

    Decision/evidence text originates from commit messages — attacker-
    influenceable input.  Flattening keeps a hostile multi-line commit
    message from fabricating its own markdown blocks (fake [D2] entries,
    fake role headers) inside the agent-visible context.  This is
    containment, not a full defense: consumers should treat rendered
    evidence as untrusted data, never as instructions.
    """
    return " ".join(str(text or "").split())


def compose_why_context(decisions: List[Dict[str, Any]], char_limit: int = 4000) -> str:
    """Render answers as an inject-ready markdown block (pure function).

    Additive discipline: over-budget decisions are truncated from the tail —
    evidence lives inside its own decision's block and never displaces
    another decision.  All captured text is newline-flattened (see
    :func:`_flat`) because it descends from commit messages.
    """
    blocks: List[str] = []
    for i, d in enumerate(decisions, start=1):
        sha = d["commits"][0]["sha"][:7] if d["commits"] else ""
        date = str(d.get("decided_at", ""))[:10]
        if not date and d["commits"]:
            date = str(d["commits"][0].get("date", ""))[:10]
        head = f"### [D{i}] {_flat(d['title'])}"
        if sha:
            head += f"  ({sha}{', ' + date if date else ''})"
        lines = [head]
        if d["anchors"]:
            files = ", ".join(
                _flat(a["file"]) + (f":{_flat(a['lines'])}" if a["lines"] else "")
                for a in d["anchors"][:4]
            )
            lines.append(f"files: {files}")
        if d["decision"]:
            lines.append(f"decision: {_flat(d['decision'])}")
        if d["rationale"]:
            lines.append(f"why: {_flat(d['rationale'])}")
        if d["evidence"]:
            lines.append("evidence:")
            for ev in d["evidence"]:
                ref = f"  [{_flat(ev['source_ref'])}]" if ev["source_ref"] else ""
                lines.append(f"- ({_flat(ev['kind'])}) \"{_flat(ev['statement'])}\"{ref}")
        sup = _flat(", ".join(s["title"] or s["kref"] for s in d["supersedes"])) or "none"
        sup_by = _flat(
            (d["superseded_by"] or {}).get("title")
            or (d["superseded_by"] or {}).get("kref")
            or ""
        ) or "none"
        if d["status"] == "superseded":
            lines.append(f"⚠ SUPERSEDED by: {sup_by}")
        lines.append(f"supersedes: {sup} / superseded_by: {sup_by}")
        blocks.append("\n".join(lines))

    out: List[str] = []
    used = 0
    for block in blocks:
        if used + len(block) + 2 > char_limit and out:
            break
        out.append(block)
        used += len(block) + 2
    return "\n\n".join(out)
