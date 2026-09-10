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

## The replacement contract

For one logical replacement `source SUPERSEDES target`, sequential replay
checks the existing edge and status, then retries unfinished demotion and
invalidation. Demotion requires a confirmed edge and a successful post-write
reverse check. Dependents are processed in bounded batches; acknowledged
pending metadata makes unfinished work discoverable for a later replay or
maintenance pass. Completion depends on successful backend operations and
continued retries.

These are repair steps, not a distributed transaction or a linearization
point. Without server uniqueness or compare-and-set, concurrent requests can
race between reads and writes. Conflict checks preserve detected disagreement;
they cannot guarantee that every concurrent conflict is observed in time.

### The four cases, distinguished

| case | behaviour |
|---|---|
| sequential replay of the same op | edge existence + status pre-checks make it a no-op that still repairs an interrupted demotion/ripple |
| two distinct sources replace the same target | both edges recorded, target marked superseded; neither source is thereby selected as the accepted winner |
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
  is silently selected as the winner. A failed re-check returns a retryable
  error and also withholds demotion; another concurrent writer may already
  have demoted an endpoint before the conflict became visible.
* **Bounded, resumable invalidation** — the ripple examines at most `cap`
  (`RIPPLE_FANOUT_CAP`, 20) dependent revisions per call. Before modifying
  dependents it persists pending work on the exact superseded revision.
  `grounding_ripple_cursor` counts a successful prefix of sorted references;
  `grounding_ripple_snapshot` binds that cursor to dependency membership and
  the superseding reference. Reordering cannot skip work, membership changes
  restart the prefix, and a failed dependent read/write stops advancement.
  Failed progress acknowledgements reach the caller as an error. This adds
  progress metadata writes on the replacement path, not on ordinary recall.
  Edge enumeration itself uses the SDK's unpaginated API.
* **Maintenance discovery** — pending fact, decision, and code-decision
  revisions, including historical revisions, are eligible for resume. The
  sweep caps inspected items/revisions and reports an incomplete scan when
  capped or when enumeration fails. A capped scan does not guarantee eventual
  discovery of every item; explicit replay can resume a known revision.
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
  one. Idempotent demotion does not establish edge uniqueness. Another race
  is two writers creating *mutually reverse*
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
in the result, and retain the conflict for an authorised decision. A concurrent
writer may already have changed a status before the conflict was observed. This
mirrors the existing CONTRADICTS contract, where an explicit contradiction
protects both endpoints and a newer timestamp never wins on recency alone.

## Compatibility strategy

* `SupersessionResult` gains fields only; the original `created` / `linked` /
  `demoted` / `stale` / `error` are unchanged, so existing callers and their
  tests are unaffected.
* `ripple_grounding_stale` keeps its integer return (newly stamped count) and
  its signature, including the optional `get_revision` SDK override. Progress
  persistence failures now raise; the shared supersession caller converts them
  into `SupersessionResult.error` so incomplete work is not reported complete.
* The new metadata keys (`grounding_ripple_pending`, `grounding_ripple_cursor`,
  `grounding_ripple_snapshot`)
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

Integration regressions cover post-write read outages, dependent read/write
failures across resume, reordered/changed dependencies, failed progress writes,
and pending historical decision revisions. These are deterministic protocol
tests; they do not establish a distributed transaction guarantee.
