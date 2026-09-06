"""Applicability axes for recalled memory (ontology; kumiho-memory#28).

A recalled memory should be an *inspectable basis* for the current task, not a
bare string. Beyond the strength/provenance axes already defined
(:mod:`kumiho_memory.trust_vocab`, :mod:`kumiho_memory.evidence`) and the
belief-state markers already surfaced (``grounding_stale``, ``contested_by``,
``superseded``, valid-time), two axes were missing and are defined here:

* **Claim origin / actor** — WHO asserted or accepted the claim. Deliberately
  distinct from provenance grade: an agent SAYING something is not the same as
  that something being independently verified. ``agent`` origin is
  *self-asserted*, never authenticated; host-attested data (``user``,
  ``imported``, ``observed``) stays distinguishable from it.
* **Decision acceptance state** — whether a decision is a ``proposal`` the agent
  floated, an ``accepted`` decision the user approved, ``contested``,
  ``superseded``, or ``unknown``. A proposal must stay a proposal through
  reflect / consolidate / decompose and later recall until it is actually
  accepted or independently supported; repetition or self-citation never
  promotes it (nothing in this module upgrades a state — the state only changes
  when a writer sets it).

Conventions match the other markers: metadata is canonical, values are strings,
reads are additive and never raise, and an absent value reads as ``unknown``
(a legacy record is not silently trusted, nor rewritten).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Metadata keys (as they appear on a revision in the graph).
CLAIM_ORIGIN_META = "origin"
DECISION_STATE_META = "decision_state"

# --- claim origin / actor ---------------------------------------------------
ORIGIN_USER = "user"          # the person asserted or approved it (authoritative for their scope)
ORIGIN_AGENT = "agent"        # the agent asserted it — self-reported, NOT authenticated provenance
ORIGIN_IMPORTED = "imported"  # ingested from an external source
ORIGIN_OBSERVED = "observed"  # an observed tool/command result
ORIGIN_UNKNOWN = "unknown"
CLAIM_ORIGINS = (ORIGIN_USER, ORIGIN_AGENT, ORIGIN_IMPORTED, ORIGIN_OBSERVED, ORIGIN_UNKNOWN)

#: Origins that are self-asserted rather than host-attested: an answering model
#: must not read them as independently verified.
_UNAUTHENTICATED_ORIGINS = frozenset({ORIGIN_AGENT, ORIGIN_UNKNOWN})

# --- decision acceptance state ----------------------------------------------
STATE_PROPOSAL = "proposal"      # floated, not yet accepted
STATE_ACCEPTED = "accepted"      # explicitly accepted by an authorised actor
STATE_CONTESTED = "contested"    # an unresolved disagreement stands
STATE_SUPERSEDED = "superseded"  # replaced by a later belief
STATE_UNKNOWN = "unknown"
DECISION_STATES = (STATE_PROPOSAL, STATE_ACCEPTED, STATE_CONTESTED, STATE_SUPERSEDED, STATE_UNKNOWN)


def normalize_origin(value: Any) -> str:
    """A stored origin value normalised to the vocabulary, else ``unknown``."""
    v = str(value or "").strip().lower()
    return v if v in CLAIM_ORIGINS else ORIGIN_UNKNOWN


def normalize_decision_state(value: Any) -> str:
    """A stored decision state normalised to the vocabulary, else ``unknown``."""
    v = str(value or "").strip().lower()
    return v if v in DECISION_STATES else STATE_UNKNOWN


def is_unauthenticated_origin(origin: str) -> bool:
    """True when the origin is self-asserted (agent) or unknown."""
    return normalize_origin(origin) in _UNAUTHENTICATED_ORIGINS


def apply_applicability_marker(entry: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    """Surface the origin / decision-state axes onto a recall *entry* from *meta*.

    Additive and lossless (mirrors ``grounding.apply_grounding_marker`` and the
    ``evidence_level`` / valid-time reads): sets ``origin`` only when the
    metadata carries a recognised, non-unknown value, and ``decision_state``
    likewise. An absent value leaves the key unset — the reader treats its
    absence as ``unknown`` rather than seeing a stamped ``unknown`` on every
    legacy revision. Never removes or reorders anything.
    """
    if not meta:
        return
    origin = str(meta.get(CLAIM_ORIGIN_META, "") or "").strip().lower()
    if origin and origin in CLAIM_ORIGINS and origin != ORIGIN_UNKNOWN:
        entry["origin"] = origin
    state = str(meta.get(DECISION_STATE_META, "") or "").strip().lower()
    if state and state in DECISION_STATES and state != STATE_UNKNOWN:
        entry["decision_state"] = state


def applicability_notes(mem: Dict[str, Any]) -> str:
    """Terse qualifier notes for the applicability axes of one rendered block.

    Rendered by the shared ``context_compose.qualifier_notes`` so every context
    path an answering model reads carries them. High-signal only: a decision
    still a *proposal* is flagged so it is not read as settled, and a claim of
    *agent* origin is flagged as self-asserted rather than verified. Accepted /
    user / imported / observed and plain unknown add no note (they are the
    unremarkable cases, and noting every unknown-origin legacy memory would be
    noise). Supersession and contested state are noted by their own markers.
    """
    notes = ""
    if normalize_decision_state(mem.get("decision_state")) == STATE_PROPOSAL:
        notes += "\n[proposal: floated, not an accepted decision]"
    if normalize_origin(mem.get("origin")) == ORIGIN_AGENT:
        notes += "\n[origin: agent-asserted, not independently verified]"
    return notes
