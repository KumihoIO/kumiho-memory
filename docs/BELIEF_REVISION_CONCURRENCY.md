# Belief revision under concurrency and recovery

Design decision for kumiho-memory#27. Extends the 1.4.1 shared belief-replacement
protocol (`supersession.supersede_revision`, `grounding.ripple_grounding_stale`)
from sequential retry-safety to an explicit, testable concurrency and recovery
contract. It documents what is guaranteed, what is deliberately not, the conflict
policy, the backend primitives that a stronger guarantee would require, and the
compatibility strategy.

This is a design-and-hardening change, not a claim of distributed atomicity.
Where a guarantee needs a server primitive the memory layer does not have, that
is stated as required upstream work rather than implied.

## The externally visible invariant

For one logical replacement `source SUPERSEDES target`:

1. **At most one live SUPERSEDES edge** `source → target`, however many times the
   operation is replayed by however many processes.
2. **The target is demoted at most once** (`status = superseded`) and only after
   its edge is confirmed. A demotion never happens without a confirmed edge.
3. **Every dependent grounded in the target** (`decision --DEPENDS_ON--> target`)
   is eventually flagged `grounding_stale`, bounded per call, with the remainder
   durably recorded for resume.
4. **A conflict is preserved, not resolved by arrival order.** Two independent
   proposals do not become a settled winner because one request arrived last.

The linearization point is the SUPERSEDES edge write. Everything after it
(demotion, ripple) is convergent repair that any later replay completes.

### The four cases, distinguished

| case | behaviour |
|---|---|
| sequential replay of the same op | edge existence + status pre-checks make it a no-op that still repairs an interrupted demotion/ripple |
| two distinct sources replace the same target | both edges recorded, target demoted once — convergent, not a conflict |
| competing reverse / cyclic replacement | rejected before any write (bounded cycle walk), or, if the reverse edge appears only after our write, detected on a post-write re-check and the demotion withheld (`reverse_conflict`) |
| unresolved CONTRADICTS | untouched here — CONTRADICTS is a separate first-class edge and already suppresses the lexical supersede fallback; disagreement never authorises a winner |

## What the implementation provides now (server-independent)

* **Convergent replay** — existence pre-check on the edge, status pre-check on
  the demotion, idempotent ripple (an already-flagged dependent is skipped).
* **Stable operation identity** — `SupersessionResult.op_id` is a deterministic
  hash of `(project scope, base source kref, base target kref)`, so a replay is
  recognisably the same operation across processes without a server sequence.
* **Payload / scope validation** — a replacement whose endpoints resolve to two
  different project scopes is rejected before any write; a self-edge is rejected.
* **Bounded cycle rejection** — before creating the edge the target's outgoing
  SUPERSEDES chain is walked to `SUPERSEDE_CYCLE_MAX_DEPTH` (default 8); if it
  reaches the source the write is refused (`cycle_rejected`). Depth 1 is the
  direct reverse edge. A read outage mid-walk fails the operation closed
  (retryable) rather than guessing.
* **Post-write reverse quarantine** — after our edge lands, the target is
  re-read for a reverse edge to the source; if a concurrent writer created one,
  the result is `reverse_conflict` and the demotion is withheld, so neither side
  is silently demoted.
* **Bounded, resumable invalidation** — the ripple processes at most `cap`
  (`RIPPLE_FANOUT_CAP`, 20) dependents per call. On truncation it persists a
  durable pending marker (`grounding_ripple_pending`) plus a cursor
  (`grounding_ripple_cursor`) on the superseded fact, so the remainder is
  discoverable and `resume_grounding_ripple` — invoked by the Dream State
  maintenance sweep (`GraphMaintainer._resume_pending_ripples`) or by a replay —
  continues from the cursor and clears the marker only when the fan-in is fully
  drained. A truncated ripple is therefore never a swallowed failure reported as
  zero affected dependents.
* **Machine-readable progress** — `SupersessionResult` reports `created`,
  `linked`, `demoted`, `stale`, `reverse_conflict`, `cycle_rejected`,
  `ripple_truncated`, `ripple_pending`, `op_id`, and an `events` trail; the
  `complete` property is true only when the foreground work finished with no
  conflict, no error, and nothing pending. `MaintenanceStats` adds
  `ripple_dependents_resumed` and `ripples_still_pending` so a backlog stays
  visible across runs.

## What it does NOT provide, and the upstream primitives required

* **Cross-process atomicity / exactly-once.** The edge is observed and created in
  two separate calls; two processes can both observe "no edge" and both create
  one. For the *same-direction* case this is benign (idempotent demotion leaves
  one live edge). The harmful case is two writers creating *mutually reverse*
  edges simultaneously: each passes its cycle check before the other's edge
  exists. The post-write re-check detects and quarantines this best-effort, but
  cannot prevent it. Preventing it requires one of:
  * a server **conditional edge write** (compare-and-set: "create only if no
    reverse edge exists"), or
  * a server **unique operation-identity** constraint keyed on `op_id`, or
  * a durable **reconciliation record** the server serialises.

  These belong to kumiho-SDKs / the graph server and are tracked as required
  upstream work. A process-local lock is explicitly **not** a distributed
  guarantee and is not used.
* **Automatic conflict resolution.** A detected reverse conflict is left for a
  human or an authorised revision policy; the layer does not pick a winner.
* **Unbounded foreground fan-out.** A pathological hub drains over successive
  maintenance runs by design, not in one unbounded pass.

## Conflict policy

A conflict is any state where accepting the write would leave two mutually
invalid "resolved" beliefs: a direct reverse edge, a SUPERSEDES cycle, or a
concurrently created reverse edge. The policy is **preserve, do not resolve**:
reject the write (cycle) or withhold the demotion (post-write reverse), report it
in the result, and leave both beliefs live for an authorised decision. This
mirrors the existing CONTRADICTS contract, where an explicit contradiction
protects both endpoints and a newer timestamp never wins on recency alone.

## Compatibility strategy

* `SupersessionResult` gains fields only; the original `created` / `linked` /
  `demoted` / `stale` / `error` are unchanged, so existing callers and their
  tests are unaffected.
* `ripple_grounding_stale` keeps its integer return (newly stamped count) and
  its signature; the pending marker/cursor are additive side effects and a new
  optional `get_revision` argument, defaulted to the ambient SDK.
* The new metadata keys (`grounding_ripple_pending`, `grounding_ripple_cursor`)
  are additive and, like every grounding marker, canonical in metadata; a reader
  that does not know them is unaffected. Legacy facts without them read as "no
  pending work".
* No public API is removed or renamed. Every belief-replacement producer already
  routes through `supersede_revision`, so the hardening reaches all of them
  (ontology decompose, code capture, graph maintenance, lexical fallback);
  profile-history SUPERSEDES edges remain outside this protocol.

## Test coverage

`tests/test_supersession_concurrency.py` exercises the adversarial matrix with
deterministic fault injection (no sleeps): two-writer convergence, distinct
sources on one target, direct-reverse and three-node-cycle rejection, a
long-chain non-cycle, concurrent post-write reverse quarantine, cross-project
rejection, read-outage fail-closed, truncated-then-resumed ripple across a
simulated process death, and the result contract. `tests/test_graph_maintenance.py`
covers the maintenance resume pass (drain + marker clear, dry-run accounting, and
the no-pending no-op).
