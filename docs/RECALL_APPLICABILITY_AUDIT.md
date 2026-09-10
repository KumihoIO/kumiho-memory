# Recall applicability, origin, and uncertainty — audit + contract

Path matrix and contract for kumiho-memory#28. The goal: a recalled memory is an
*inspectable basis* for the current task — what the claim is, where/when it
applies, who asserted or accepted it, what supports it, and whether it is
superseded, contested, or grounded in changed facts. This starts from an audit of
what each write/read/render path already preserves, not an assumption that the
axes are absent.

## Axes

| axis | field(s) | source of truth | status before #28 |
|---|---|---|---|
| claim text | `title` / `summary` / `content` | revision metadata + artifact | preserved |
| provenance grade | `evidence_level` (`official`…`unverified`) | `evidence.py`, graded | preserved (recall + rerank + context) |
| self-reported strength | `certainty` / `confidence` | writer-asserted | preserved; **never lifts evidence_level** (`trust_vocab.py`) |
| valid-time applicability | `valid_from` / `valid_to`, `event_date` | `valid_time.py` | preserved; opt-in as-of demotion |
| grounding freshness | `grounding_stale`, `superseded_by` | `grounding.py` ripple | preserved (recall marker + context note) |
| contested / unresolved | `contested_by` | CONTRADICTS edges (`graph_augmentation.py`) | preserved (marker + "disputed" note) |
| superseded belief state | `superseded` | `status=superseded` at recall | added in #26 (marker + note; unified `qualifier_notes`) |
| **claim origin / actor** | `origin` (`user`/`agent`/`imported`/`observed`/`unknown`) | writer-declared metadata | **absent → added in #28** |
| **decision acceptance state** | `decision_state` (`proposal`/`accepted`/`contested`/`superseded`/`unknown`) | writer-declared metadata | **absent → added in #28** |

The two axes #28 adds are deliberately separate from provenance grade: an agent
*saying* something (`origin=agent`) is not the same as it being independently
verified (`evidence_level`), and a *floated* option (`decision_state=proposal`)
is not an accepted decision. High self-reported certainty still does not lift
either — the pre-existing `trust_vocab` limit is unchanged.

## Path matrix (origin / decision_state)

`P` preserved, `A` added by #28, `—` not applicable, `∘` intentionally left for a
documented follow-up.

| path | reads/writes | origin | decision_state |
|---|---|---|---|
| `reflect` (`tool_memory_reflect`) | write capture metadata | A (`cap.origin`, normalised) | A (`cap.decision_state`, normalised) |
| `decompose` (`memory_decompose`) | write typed facts | ∘ (agent-declared facts left unstamped rather than blanket `origin=agent`; a decompose-time default is a separate decision) | — (facts, not decisions) |
| `consolidate` (agent summary) | write typed decisions | ∘ (same follow-up) | ∘ (same follow-up) |
| direct `recall` (`_fetch_revision_metadata`) | read → entry | A (`apply_applicability_marker`) | A |
| `engage` (`build_recalled_context`) | render | A (`qualifier_notes` → `applicability_notes`) | A |
| composed context (`compose_context`) | render, incl. siblings + truncation | A (carried onto flattened revs; note appended after truncation) | A |
| graph-augmented fact-recall leg | read typed fact → entry | A (`apply_applicability_marker`) | A |
| rerank / scoring | ordering | — (additive, never alters score/order) | — |

Confirmed gaps closed by #28: the two axes were absent on every write and read
path. The `∘` rows are deliberate: stamping every agent-decomposed fact as
`origin=agent` would flag effectively all typed knowledge and drown the signal,
so a decompose/consolidate-time origin policy is left as a reviewed follow-up
rather than defaulted silently. A fixture demonstrating the reflect→recall→render
round-trip and the truncation-survival of the qualifier is in
`tests/test_recall_applicability.py`.

## Contract

1. **Origin is not provenance.** `origin=agent` and `origin=unknown` are
   *self-asserted / unattested*; recall renders `[origin: agent-asserted, not
   independently verified]` for the agent case. `user` / `imported` / `observed`
   are host-attested and add no note. `evidence_level` remains the separate,
   graded provenance axis; an agent origin never lifts it.
2. **A proposal stays a proposal.** `decision_state=proposal` renders
   `[proposal: floated, not an accepted decision]` and is never rendered as
   accepted. Nothing in the read/consolidate/decompose path *promotes* a state —
   repetition or self-citation cannot turn a proposal into an accepted decision,
   because the state only changes when a writer explicitly sets it.
3. **Absence reads as unknown, never as trusted.** A legacy revision without the
   fields is treated as unknown origin / unknown state and is neither stamped nor
   rewritten. No blanket back-fill, no newest-wins default.
4. **Current vs historical.** `created_at` is never used as validity or
   authority; valid-time (`valid_from`/`valid_to`) and the opt-in as-of read path
   own current-vs-historical, unchanged by #28.
5. **Qualifiers survive the envelope.** Every qualifier note (contested,
   grounding-stale, superseded, proposal, agent-origin) is appended to a block
   *after* its content is truncated, and rides on stacked/sibling revisions, so a
   tight per-section budget or `limit=1` cannot silently drop a material
   qualification. The one shared renderer is `context_compose.qualifier_notes`,
   used by both context assemblers.

## Compatibility

All additive. New metadata keys (`origin`, `decision_state`) and recall-entry
keys of the same names; new module `applicability.py`; no field renamed or
removed; capture-level inputs are optional and normalised (a bad value is
dropped, never raised). Provenance-grade separation and the certainty/confidence
limit are untouched.
