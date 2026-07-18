# Release Notes — kumiho-memory

## v0.20.0

**Release Date:** 2026-07-19

**Ontology Phase 3 — identity and time** (epic #86) plus a reliability
workstream driven by Hugh Kim's independent review, all additive and
flag-gated default-OFF.

- **Write-time entity alias resolution (G5)** — before minting a new entity
  hub, decompose resolves against existing graph hubs by normalized name
  and stored aliases; on a match the existing hub is reused and the new
  alias appended instead of creating a duplicate node. Bounded to one
  cached lookup per new entity; any lookup failure falls back to today's
  behavior rather than blocking the write. Opt in with
  `KUMIHO_MEMORY_ALIAS_RESOLUTION=1`.
- **Embedding-assisted fact dedup in Dream State (G6)** — near-duplicate
  fact candidates are nominated via the server's existing hybrid
  `score_revisions` (no client-side embeddings, no LLM key, `kind=fact`
  filtered, capped fan-out), then routed through the unchanged
  `_confirm_and_collapse_fact` merge path — the same Jaccard threshold,
  per-kind budget, published protection, and soft deprecation guard every
  other Dream State merge already passes through. Opt in with
  `KUMIHO_DREAM_EMBED_FACT_DEDUP=1` (also requires `maintain_graph`).
- **Valid-time intervals + as-of recall (G8)** — additive `valid_from` /
  `valid_to` metadata alongside `event_date` (never touches `event_date`
  itself or its new confidence marker below). An opt-in as-of filter
  (`KUMIHO_MEMORY_AS_OF_RECALL=1`) deprioritizes facts whose interval
  excludes the requested date; with the flag off, recall is byte-identical
  to today.
- **`event_date` hallucination guard** — the LLM-extracted valid-time field
  is now cross-checked against the source text at write time: absolute
  dates (including Korean formats) are matched by normalized comparison,
  relative expressions ("yesterday", "지난주", "3개월 뒤") are verified by
  date arithmetic against the known session timestamp, and anything that
  doesn't corroborate is marked `unverified` and excluded from the
  event-proximity ranking boost. Existing rows without the new
  `event_date_confidence` key are treated as verified — the boost path for
  all previously-stored dates is unchanged.
- **LLM failure classification + parking ledger** — `retry.py` now
  classifies failures as transient / deterministic / unknown instead of a
  binary retry decision. A local, atomically-written, corruption-safe
  ledger (keyed by content identity) tracks cross-run attempts; content
  that fails deterministically twice is parked out of Dream State and
  consolidation selection, and un-parks automatically after 14 days
  (`KUMIHO_FAILURE_PARK_THRESHOLD`, TTL both env-tunable) so a fixed model
  or prompt gets another chance instead of a permanent poison loop.
  Wired on by default at the MCP/CLI entrypoints; opt out with
  `KUMIHO_FAILURE_LEDGER_DISABLED=1`.

Every item above is additive — no renamed keys, no data migration, no
default-recall reordering when its flag is off. The failure-parking ledger
is the one production default-path change (new local ledger file under
`~/.kumiho/failure_ledger`), documented for the opt-out above. The three
Phase 3 flags default OFF pending the same paired-benchmark gate that has
governed every prior default flip in this project; the confidence guard on
`event_date` ships unconditionally since it only ever *removes* a
ranking boost from a date that could not be corroborated.

## v0.19.0

**Release Date:** 2026-07-18

**Ontology Phase 2 — coherence becomes executable** (epic #86, PR #96) plus
the security, reliability, and performance workstream driven by Hugh Kim's
independent v0.18.0 review.

- **Explicit belief change** — decompose accepts agent-declared
  `supersedes` / `contradicts` (basis-labeled edges: `agent`,
  `lexical-overlap`, `evidence-assessor`); the lexical heuristic yields to
  explicit declarations.
- **CONTRADICTS is first-class** — assessor conflict verdicts bridge into
  graph edges; the reader traverses them and stamps bounded `contested_by`
  markers; composed context renders a disputed note. Dispute-basis scoping
  keeps entity-relation domain claims (from the predicate registry) out of
  the belief lane.
- **Grounding-staleness ripple** — superseding a fact flags its DEPENDS_ON
  dependents (`grounding_stale` + mirrored tag), recall surfaces the flag,
  and a keyless Dream State pass clears re-confirmed grounding. Closes the
  paper's deferred §15.6 loop.
- **One privacy boundary everywhere** — commit mining, session mining,
  skill ingestion, and every LLM-bound packet now pass the same per-atom
  PII/credential gate (modernized patterns incl. `sk-proj-`, JWT, AIza,
  Slack, DB URLs); external skills are statically scanned and quarantined
  until cleared. *Every write path passes the same privacy boundary.*
- **Reliability** — background auto-assess survives MCP per-call loop
  teardown (it silently died before); recall failures surface a
  `backend_error` field instead of masquerading as empty; session IDs
  retry before forking; `code_capture` runs under a 45s deadline with
  partial-success results and proven-idempotent retry.
- **Performance** — event-driven recall waits (the 0.5s poll floor is
  gone); sibling LLM reranking is capped and gathered; a deterministic
  belief-safety-first traversal contract (spec v3) makes identical graphs
  traverse identically; one context budget with token accounting.
- **Dream State destructive proposals** now pass keyless guards (min-age,
  evidence severity read across metadata AND tags — the LLM can no longer
  write the trust axis at all) with an optional second-model refutation
  layer.

Gate note: every change above passed its per-change gate (byte-diff recall
invariance, paired conv-26 answer-only runs, permutation tests), each run
inside a write-quiet window on the benchmark CE. The full-10 re-verification
was invalidated mid-campaign by a subtler mechanism, isolated via a
three-way controlled comparison: large writes to *other* projects on the
same server (a second LoCoMo corpus ingest — a semantic near-clone of the
benchmark corpus, doubling document frequency for its most distinctive
terms) shifted this project's recall rankings through corpus-global BM25
statistics
(kumiho-server#28) — same code, same corpus, 1,142 recall differences
across time, with same-code stability controls clean at both endpoints.
Client code was exonerated; the 0.5435 baseline remains internally valid
for its own write-quiet window. A fresh baseline will be established on an
isolated (write-quiet or #28-fixed) server state as the new reference.

## v0.18.0

**Release Date:** 2026-07-17

**Ontology Phase 1 — the typed graph gains an explicit semantic contract**
(epic #86, PR #91). Grounded in Gruber's ontology design principles; full
analysis in `docs/ONTOLOGY_PRINCIPLES_GAP.md`.

- **Canonical relation predicate registry** — agent-supplied relation
  predicates fold onto 10 canonical edge types (`DEPENDS_ON`, `USES`,
  `IMPLEMENTS`, `PART_OF`, `SUPERSEDES`, `SUPPORTS`, `CAUSES`,
  `CONTRADICTS`, `CONTAINS`, `RELATES_TO`); synonyms converge
  (`utilizes → USES`) and unregistered or unnormalizable predicates
  (incl. CJK) become `RELATES_TO` with the verbatim predicate preserved
  in edge metadata. Relations are never silently dropped.
- **Fetchable ontology spec** — node-kind definitions, edge semantics,
  the predicate registry, and the trust mapping persist as a versioned
  policy Item (`ontology/spec.policy`, tag `ontology.spec`), seeded at
  skill ingestion. Any agent can fetch it via `kumiho_get_revision_by_tag`
  and commit to the same semantics.
- **Trust-vocabulary mapping** — one documented correspondence across
  `certainty` / `confidence` (self-reported strength) and
  `evidence_level` (provenance grade) in `trust_vocab.py`; bands are
  tie-breakers and never lift provenance.
- **Relation traversal at recall (opt-in)** — with
  `KUMIHO_MEMORY_RELATION_TRAVERSAL=1`, the entity-mediated reader
  follows registered relation edges to neighbor entities' memories with
  full hop provenance and hard fan-out caps. Measured on the dedicated
  A/B bench (kumiho-benchmarks `relation_ab/`): relation-linked golds
  0/8 → 8/8 at latency parity. Ships **off** by default pending the
  paired LoCoMo F1 + LoCoMo-Plus gate.

## v0.17.4

**Release Date:** 2026-07-16

Follow-up hardening on the v0.17.3 `_run_git` fix (adversarial-review LOW
findings, #82):

- The re-raised `TimeoutExpired` now carries the post-kill drained
  output/stderr when the bounded grace drain succeeds — parity with
  `subprocess.run`, so callers can see *why* git stalled.
- The abandon path closes our pipe ends before abandoning the daemon
  reader threads, so they exit promptly instead of accumulating in
  long-lived MCP server processes.
- Test fakes now model pipes end to end, including the grace-success
  drain path.

## v0.17.3

**Release Date:** 2026-07-16

**`_run_git` waits are now bounded on every path — fixes a 30-minute
`kumiho_code_capture` hang on Windows (#79).**

`subprocess.run(timeout=...)` is not actually bounded on Windows: its
`TimeoutExpired` path calls `kill()` followed by a timeout-less
`communicate()`, and `TerminateProcess` is asynchronous — a git child stuck
in uninterruptible kernel I/O (Defender scan, disk stall) or a descendant
holding the inherited pipe handles keeps the pipes open, so the drain blocks
indefinitely. Observed in production as a 30-minute `kumiho_code_capture`
MCP hang that only the client idle timeout ended; the capture was lost.

`_run_git` now drives the subprocess explicitly: `communicate(timeout=20)`,
on timeout `kill()` plus a bounded 5 s grace drain, then abandon the daemon
reader threads instead of joining unboundedly. Any other escape (e.g.
`KeyboardInterrupt`) kills the child before propagating — parity with
`subprocess.run`'s bare-except. Exception surface is unchanged
(`TimeoutExpired` / `CalledProcessError`), so callers keep their existing
"git resolution failed" / repo-id fallback behavior. Everything that shells
out through this helper is covered: capture, `code_why`, ingest, session
mining. (#79, #80)

## v0.17.2

**Release Date:** 2026-07-15

Corrects the package `__version__` string — 0.17.1 shipped reporting `"0.17.0"`
(the `pyproject.toml` version was bumped but `__init__.py` was not). No functional
change from 0.17.1 (same loop-aware Redis client).

## v0.17.1

**Release Date:** 2026-07-15

**Loop-aware Redis client — repeated `reflect()` in one process no longer crashes.**
`RedisMemoryBuffer` created its `redis.asyncio` client once and cached it. Because
a `redis.asyncio` connection binds to the event loop of first use, a caller that
runs several `asyncio.run()` calls in one process — History Backfill replays one
`reflect()` per session, each under its own fresh loop — reused a connection tied
to an already-closed loop and crashed on the 2nd session with *"Event loop is
closed"* / *"'NoneType' object has no attribute 'send'"* (so a single ingest
process could only store one session).

`RedisMemoryBuffer.client` is now a **loop-aware property**: a client we created
ourselves (from `redis_url`) is cheaply recreated on loop change (`from_url` is
lazy — no connection until first command); a caller-injected client (e.g. a test
double) is returned untouched. The long-lived MCP server (single persistent loop)
is unaffected. Regression tests cover both paths.

## v0.17.0

**Release Date:** 2026-07-14

**`kumiho_memory_reflect` bulk-replay contract for resumable ingest.**
Adds the two pieces history backfill needs to drive the batched write path
(0.16.3 shipped the batch write itself but not the caller contract):
- **`idempotency_prefix`** parameter — writing the captures through one
  `BatchCreateRevisions` transaction keyed on `{prefix}:{index}`, so
  re-submitting the same captures replays committed rows as a no-op (resumable
  after an interrupted ingest). Passing it also forces the batched path for a
  single capture, so the result shape is consistent.
- **`capture_results`** in the result — a list positionally aligned with the
  input captures, each `{revision_kref, …}` on success or `{error}` on failure,
  so a bulk caller can map and mark each capture exactly (reflect's flat
  `stored_krefs` alone can't attribute a mid-batch failure).

Both are additive: a reflect without `idempotency_prefix` behaves exactly as in
0.16.3 (single capture → per-capture path; ≥2 → batch, no `capture_results`
unless the field is requested by prefix). This is the contract
`kumiho-plugins` History Backfill feature-detects (`"idempotency_prefix" in`
the reflect schema) to switch from per-capture replay to one batched reflect
per session.

## v0.16.3

**Release Date:** 2026-07-14

**`kumiho_memory_reflect` batches multi-capture writes through `BatchCreateRevisions`.**
A reflect with **≥2 captures** (history backfill and any bulk write) now lands in a
**single server transaction** via the new `tool_memory_store_batch` (kumiho core
≥0.10.7 / `BatchCreateRevisions`, kumiho-server ≥1.6.3) instead of N per-capture
writes. Measured against a local CE, naive per-capture concurrency intermittently
**deadlocked** neo4j on relationship-group locks and capped at ~1.67× even at zero
RTT; the batch write removes the deadlock (one transaction, no cross-item lock
contention) and collapses the heaviest `create_item` + `create_revision` + artifact
RPCs into one. A **single-capture** reflect — the common live case — keeps the
byte-identical per-capture path, and the batch path preserves every per-capture
semantic (credential screen, space resolution, fuzzy-stack, `event_date`, tags,
`topic` bundle, `DERIVED_FROM` edges); only the create/revision writes are batched
(the server has no batch RPC for tag/bundle/edge, so those stay per-item). Requires
`kumiho>=0.10.7`; degrades gracefully to the per-capture loop on older cores.
Additive; no API break.

## v0.16.2

**Release Date:** 2026-07-14

**`kumiho_memory_reflect` captures can now carry `event_date` (valid-time) (#68).**
Reflect is the keyless write path, but its capture schema had no `event_date` and
the handler dropped metadata — so agent-written memories could record *what*
happened but not *when*, and temporal recall had nothing to anchor on except prose
in the title. Captures now accept an optional `event_date` (`YYYY` / `YYYY-MM` /
`YYYY-MM-DD`), validated against the same `_ISO_EVENT_DATE_RE` the summarizer uses
and written into the revision metadata, so recall surfacing and the valid-time
rerank boost pick it up. A malformed or relative date is dropped and reported in
`result["dropped_event_dates"]` — reflect never fails the loop over a bad date —
and captures without `event_date` are byte-identical to before. Unblocks
deterministic history backfill from timestamped transcripts. Additive; no API break.

## v0.16.1

**Release Date:** 2026-07-14

**`kumiho_code_capture` can no longer hang on git (#64).** The keyless
Decision-Memory capture resolves git (`derive_repo_id` + `_commit_info_for_ref`)
*outside* the write bound — the graph write is already bounded by
`write_timeout` (60 s), but `_run_git` shelled out with **no** `subprocess`
timeout, so a hung git (a stuck credential/fsmonitor helper, a locked or slow
repo, a network mount, an odd ref) could hang the whole tool indefinitely
(observed as a multi-minute no-op with no output). A 20 s ceiling on every git
subprocess turns that into a fast, reported failure the callers already handle
(`"git resolution failed"` / repo-id fallback). Bug-fix only; no API change.

## v0.16.0

**Release Date:** 2026-07-13

**Dream State maintains the typed ontology & Decision Memory graphs (#59).**
Dream State consolidated only flat conversation revisions; the typed ontology
(`entity`/`fact`) and Decision Memory (`{repo}-code`) graphs shipped in 0.14.x /
0.15.0 were never maintained — duplicate entities accumulated, facts went stale,
and a code decision's `evidence_level` was stamped once at capture and never
rose. This extends the same consolidation cycle to both graphs, split by the
keyless constraint.

### Added

- **Keyless deterministic maintenance** (`graph_maintenance.GraphMaintainer`,
  no LLM key): merge duplicate entities (alias/slug) with edge repointing,
  dedup near-duplicate facts about the same entity, prune orphan entities,
  **re-grade a `code_decision`'s `evidence_level` from its current
  `MOTIVATED_BY` atoms** via the deterministic `_evidence_grade` — lifts
  `unverified`→`corroborated` when evidence accrues after capture (closes the
  §6 Level-of-Evidence auto-upgrade gap) — dedup decisions via `SUPERSEDES` +
  query-time status demotion, and **bridge `code_decision`→conversation
  `entity`** with `ABOUT` edges ("one brain" at the graph level).
- **Optional LLM entity-merge** (`maintenance_llm`): semantic merge suggestions
  the deterministic alias rule can't see are applied through the same keyless,
  budgeted write path — the model only suggests, never writes.
- **`DreamState`** gains `maintain_graph` / `maintenance_llm` / `code_project`
  (runs even with no new revisions — evidence accrues independently); the
  `kumiho_memory_dream_state` MCP tool exposes them; `GraphMaintainer` and
  `MaintenanceStats` are exported.

### Safety

Reuses the flat pass's guarantees on every typed-node change: `published` /
`evidence:official` protection, the `max_deprecation_ratio` cap, `dry_run`,
idempotency (edge-dedup precheck), and cursor-incremental scanning; a
maintenance failure never aborts the flat consolidation run. Default off
(tri-state `maintain_graph`; `KUMIHO_DREAM_MAINTAIN_GRAPH`).

Proven end-to-end by `scripts/dogfood_dream_maintenance.py` (keyless, live CE):
a seeded duplicate set collapses (entities 4→3, facts 3→2), a decision whose
evidence grew re-grades `unverified`→`corroborated`, ≥1 `code_decision` bridges
to its `entity`, `dry_run` mutates nothing, and the re-run is idempotent.
Hardened against a 3-agent adversarial review (published protection across all
typed-node deprecations, tag-carried grade reads, decision-dedup SUPERSEDES-cycle
guard, one-pass transitive alias chains, prune-after-bridge ordering, run-abort
isolation).

## v0.15.0

**Release Date:** 2026-07-13

**Level-of-Evidence ranking in Decision Memory (`code_why`).** Code decisions
carried no evidence grade, so `code_why` ranked a thin commit-message decision
the same as an empirically-measured one. Two surgical, keyless changes close
that gap:

### Added

- **`code_capture` grades each decision** — a deterministic `evidence_level` is
  stamped on every code decision from its evidence atoms: a
  `measurement` / `review_finding` / `benchmark` / `incident` atom is empirical
  corroboration (`corroborated`); a bare `constraint` / `rejected_alternative`
  is a single stated source (`single_source`); no atoms is `unverified`. No LLM;
  `official` is never auto-assigned (reserved for an explicit operator flag).
- **`code_why` weights evidence into ranking** — `_sort_candidates` folds the
  evidence delta (reusing `evidence_rank.DEFAULT_EVIDENCE_WEIGHTS`) into the
  probabilistic slot, so a well-substantiated decision outranks a thin one
  **within the same factual tier**. Anchor facts still dominate the sort;
  ungraded (pre-evidence) decisions resolve to `None` and are a **strict no-op**,
  preserving legacy ordering.

Proven end-to-end by `scripts/dogfood_loe_code.py` (keyless, live CE): the §6
decision, captured with a measurement atom, self-graded `corroborated` and
materialized as a `code_decision` node with 4 `IMPLEMENTED_IN` anchors and 3
`MOTIVATED_BY` evidence atoms.

## v0.14.1

**Release Date:** 2026-07-13

**Fix: `kumiho_memory_decompose` no longer reports a successful write as an empty
`{}`.** Ontology decomposition is a run of blocking gRPC writes (one round-trip
per node and per edge); against a cloud backend a decomposition takes ~25s, which
tripped the old 25s bound. On timeout the wrapper returned an empty `{}` while the
un-cancellable daemon worker kept writing in the background — so a fully
successful decomposition looked like a no-op (a live audit saw the tool return
`{}` even though 5 entities, 3 facts, and all ABOUT / typed-relation edges landed
in the graph). The write bound is widened to the codebase's `write_timeout`
convention (60s) so the common case returns real per-kind counts, and on the rare
genuine overflow the wrapper returns `{"status": "in_progress"}` instead of a bare
`{}`. Applies to both the keyless `decompose_and_link_agent` and the
consolidation-time `decompose_and_link`.

## v0.14.0

**Release Date:** 2026-07-13

**Keyless, agent-driven ontology decomposition.** The plugin runs inside Claude
with no external LLM key — but the ontology decomposition that turns a
conversation into a typed entity/fact graph ran only via the summarizer LLM at
consolidation, so under the keyless plugin the typed graph was never built
(a live audit found 0 entity nodes). This release brings the same keyless
pattern (the agent extracts, the tool validates + writes) to the ontology, so
the graph-native typed graph is built with no key.

### Added

- `kumiho_memory_decompose` MCP tool + `UniversalMemoryManager.memory_decompose`
  (gated on `KUMIHO_MEMORY_ONTOLOGY`): the in-loop agent passes
  `{entities, facts, relations}` distilled from a memory's summary; the tool
  validates structure and writes typed `entity`/`fact` nodes + `ABOUT` /
  `DERIVED_FROM` / typed relation edges, reusing the exact `_Materializer`
  writers and `slugify` identity as the LLM path — so recall / entity-resolution
  use the nodes (no ghost nodes).
- `scripts/dogfood_ontology_agent.py` — live keyless gate (entity 0→N,
  idempotent, no API key).

### Changed

- `ontology._Materializer.edge` gains an idempotent, best-effort edge-dedup
  precheck, so re-running a decomposition on the same memory adds no duplicate
  edges.

## v0.13.0

**Release Date:** 2026-07-12

**Keyless, agent-driven Decision Memory capture.** The plugin runs inside
Claude; requiring a separate OpenAI/Anthropic key to extract the *why*
betrays the point. `kumiho_memory_reflect` already proved the pattern for
conversation memory (the agent's own model identifies what matters; the tool
just stores it, no key) — this release brings it to code.

### Added

- **`kumiho_code_capture`** (MCP tool) + `manager.code_capture()` +
  `code_capture.capture_decisions()` — the keyless counterpart to
  `code-ingest`. The agent passes decisions it already extracted from the
  diff/conversation (`title`, `decision`, `rationale`, `why_question`,
  `files`, `evidence`, `confidence`), so the structuring LLM call is skipped
  entirely (no `adapter`). The same deterministic validation still runs —
  anchors UNION with the commit's real changed files (list files, not line
  ranges; hallucinated files drop) — and the same sha-anchored write path.
  Defaults to `HEAD` (the commit just made).

### Notes

- `code-ingest` / `code-mine-session` remain for the detached-hook / batch
  backfill that has no agent in the loop and therefore does need a model.
  When Claude is present, `kumiho_code_capture` is the primary path.
- Verified live **keyless** (`OPENAI_API_KEY` unset): capture → `why()`
  recall round-trips the decision + a verbatim rejected-alternative.

## v0.12.1

**Release Date:** 2026-07-11

Dependency fix — requires **`kumiho>=0.10.5`**. Decision Memory imports
`kumiho._text.slugify`, which the published core 0.10.4 predates (it landed
on the core repo after that tag without a re-publish). Against PyPI 0.10.4,
`from kumiho._text import slugify` failed with `ModuleNotFoundError` — so
Decision Memory could not actually run from a clean install. Pinning
`kumiho>=0.10.5` (published alongside this release) resolves it. No behavior
change to session mining itself.

## v0.12.0

**Release Date:** 2026-07-11

**Decision Memory Phase 2 — session mining.** A commit records *what*
landed; the session that produced it holds what git loses — the rejected
alternative, the measurement in its original form, the decision that never
reached a commit. This release mines the agent-session transcript into the
same git-anchored decision graph and closes the capture loop
(commit hook → SessionEnd worker). All opt-in behind `KUMIHO_MEMORY_CODE=1`;
conversation paths stay byte-identical when gated off.

### Added

- **Session mining** (`code_session.py`): salience selection → budget-capped
  chunks → LLM structuring → verbatim/sha/`git ls-files`/per-atom-credential
  validation → **enrich-or-standalone correlation** → additive write → a
  repo-qualified session marker (written last, completeness-checked).
  - **Enrichment is additive by invariant** — a session decision that
    correlates with a commit-mined decision attaches its conversation-only
    evidence via new nodes + `MOTIVATED_BY`/`DERIVED_FROM`/`DISCUSSED_IN`
    edges only, never a new revision or a metadata rewrite on the target.
    Correlation is deterministic (verified-sha or verified-anchor discovery)
    with signal conjunction; lexical similarity alone can never merge.
  - **Standalone** — commit-less decisions become `origin="session"` nodes
    with `role="mentioned"` anchors; rejected alternatives join the
    embedding text (queries arrive under the *rejected* option's name).
  - **Bridge** — `DISCUSSED_IN` links a decision to the consolidated
    conversation revision (cross-project by kref; the conversation project
    is never written to).
- **`code-mine-session` CLI** + `parse_claude_transcript()` — the loop-closer
  surface the plugin SessionEnd worker calls.
- **MCP** `kumiho_code_mine_session`; **manager** `code_mine_session` + an
  opt-in consolidation chain (`KUMIHO_MEMORY_CODE_AUTOMINE=1`).

### Changed

- **`--force` is now true deprecate-then-rewrite** (commit and session): the
  stale generation is retired via `Item.set_deprecated` before re-mining.
- **Query**: `DERIVED_FROM` routes session markers to `chain.sessions` (no
  more `{sha:""}` ghost commits); the SUPERSEDES priority sort now runs
  before the per-decision edge cap so `superseded_by` survives session
  enrichment; `origin`/`status_hint` passthrough.

### Verification

Unit: 68 code-domain tests (full suite 523 passed). Live dogfood gate 8/8 on
a CE. Real-session proof: mining a 7,959-turn transcript surfaced actual
decisions with verbatim measurements and rejected alternatives via `why()`.

## v0.11.0

**Release Date:** 2026-07-11

**Decision Memory** — a second domain profile on the same graph engine:
the *why-layer* for a codebase. git stays the lossless source of
what/when/who; this release adds the graph of **why** — decisions,
rationale, and verbatim evidence, anchored to git and queryable by coding
agents mid-session. Opt-in via `KUMIHO_MEMORY_CODE=1` (default OFF;
conversation paths are byte-identical when gated off — proven by
isolation tests).

### Added

- **Capture** (`code_capture.py`): an 8-stage git commit-mining pipeline —
  deterministic prefilter (only *certain* noise dropped; `chore:` can carry
  decisions), message-first evidence packets (comment/docstring diff lines
  survive truncation — rationale lives in comments), batched LLM
  structuring with a strict decision definition ("zero decisions is a
  valid answer"), hallucination defenses (anchors unioned with the
  changed-file ground truth), anchor-scoped 3-signal `SUPERSEDES` with
  in-place status demotion, and marker-last idempotency: re-running a
  range costs **zero LLM calls**, and partial failures retry themselves.
- **Query** (`code_query.py`): `why(question, file=, line=, commit=)` —
  a deterministic anchor leg (file → decisions, zero search infra), a
  semantic leg, and an evidence-bridge leg, fused **lexicographically**
  (anchor facts always outrank cross-encoder probabilities). Superseded
  decisions are demoted and always carry `superseded_by`. Returns
  structured answers plus an inject-ready markdown context block.
- **Schema** (`code_decisions.py`): sha-free identity (decisions key on
  `title + author-date`, anchors on `repo::path` hubs; volatile
  coordinates live on edge metadata) so history rewrites converge instead
  of duplicating — the non-rotting property.
- Surface: `manager.code_why` / `manager.code_ingest`,
  `kumiho_code_why` / `kumiho_code_ingest` MCP tools (registered only when
  gated on), and a `kumiho-memory code-ingest` CLI subcommand.
- Design doc: `docs/DECISION_MEMORY_DESIGN.md` (3-design judge-panel
  synthesis; all codebase constraints verified against real code).

### Notes

- Code nodes live in a dedicated **`{project}-code` kumiho project** —
  physical isolation from conversation recall (the measured
  vector-crowding incident class). Zero new dependencies, zero server
  changes.
- Live-verified on this repo's own history against a kumiho-server CE:
  three agent-style why-queries (single-worker executor / ontology
  default / unconditional partition) answer with the actual deciding
  commits and their verbatim measurements as evidence (3/3, machine-
  judged; `scripts/dogfood_code_memory.py`).
- Hardened by a 4-lens adversarial review (26 confirmed findings fixed,
  including a critical revision-pinning identity split in supersede
  demotion and a git argv option-injection guard).

## v0.10.1

**Release Date:** 2026-07-10

### Fixed

- **The fastembed cross-encoder rerank no longer blocks the event loop.** The
  opt-in cross-encoder stage (`KUMIHO_RERANK_CROSS_ENCODER=1`) is CPU-bound
  ONNX inference; invoked inline from the async recall paths it froze the loop
  and serialized every concurrent recall (measured on the 2026-07-10
  full-LoCoMo run: a concurrency-4 harness degraded to ~1 effective). Recall
  paths now await the new `rerank_async`, which runs the unchanged sync
  `rerank` on a dedicated single-worker executor — inference stays serialized
  (identical results and CPU profile), but concurrent recalls overlap the rest
  of their pipeline again. `rerank` itself is untouched and remains the sync
  API.

### Added

- `rerank_async` — public async wrapper around `rerank`. Offloads to the
  worker thread only for rerankers tagged `_kumiho_offload_safe` (set by
  `try_fastembed_reranker`); everything else runs inline: dormant configs
  (the deterministic priors are microseconds — no thread-hop overhead) and
  the LLM reranker / user callables (`KUMIHO_RERANK_LLM=1` drives the
  manager's shared async client, which must not be driven from a second
  event loop — it keeps its pre-0.10.1 inline behavior).

### Known limitations

- Sibling embedding filtering (`_filter_siblings_by_embedding`) still calls a
  blocking `embed()` on the event loop when an embedding adapter is
  configured — pre-existing, tracked as the next offload candidate.

## v0.10.0

**Release Date:** 2026-07-10

The ontology release: every conversation is decomposed into a typed knowledge
graph (entities, facts, decisions, events, actions, questions) at write time,
and recall consumes that structure — **on by default**. Decided on paired
same-corpus evidence: the ontology read stack contributes **+0.042 overall**
and the typed-fact recall leg **+0.054** with all five LoCoMo categories up
(23W/172T/4L, sign test p≈2e-4), while the base summary stays byte-identical.

### Added

- **Write-time ontology** (`ontology.py`, `relations.py`): schema-driven
  decomposition of each consolidation into typed nodes in dedicated spaces
  (`/facts`, `/decisions`, `/events`, `/entities`, `/actions`, `/questions`)
  with deterministic edges — `DERIVED_FROM` (provenance), `ABOUT`/`INVOLVES`
  (token-boundary mention matching, Hangul-aware), `DEPENDS_ON` (same-batch
  token overlap ≥ 0.4), `SUPERSEDES` (belief update, overlap ≥ 0.6). Zero
  extra LLM calls; the summarizer schema is byte-identical in both modes.
- **Entity-bridge join** (multi-hop recall): an entity reached via `ABOUT`
  from two or more reformulated angles is a bridge; its fact/event nodes
  surface with a real inherited score (0.9 × the weaker angle). Hub anchors
  (degree > 12) are deferred, not dropped.
- **Fact-recall leg**: typed fact nodes retrieved as first-class semantic
  candidates with the original query, scored relative to the weakest base
  hit (axis-invariant) and composed additively — they can never displace or
  outrank conversation evidence.
- Additive-slot discipline end to end: recall cap, manager trim, and context
  composition all reserve on-top budgets for structural evidence
  (`fact_budget` passthrough in `compose_context`).
- `KUMIHO_MEMORY_REFORMULATE_DRAWS` env knob for multi-draw query
  reformulation (default 1; higher values were measured to dilute recall on
  LoCoMo-Plus — leave at 1 unless you have paired evidence for your corpus).

### Changed

- **BREAKING (behavioral): the ontology is now opt-OUT.**
  `KUMIHO_MEMORY_ONTOLOGY` defaults to on; set `0` for the legacy pipeline
  (byte-identical output, asserted by tests). Scripts that relied on
  "unset means off" must now export `KUMIHO_MEMORY_ONTOLOGY=0`.
  `KUMIHO_MEMORY_ENTITY_PROMOTION` / `KUMIHO_MEMORY_FACT_RECALL` still force
  their components independently.

### Requires

- kumiho-server with the derived-kind search hygiene chain (fulltext
  exclusion + kind-filtered vector pool widening + in-arm fusion filter,
  server PR#35). Older servers work but leave typed-node pollution in the
  lexical index and starve the vector leg on large ontology corpora.


## v0.9.0

**Release Date:** 2026-07-08

Consolidates the full cognitive-recall pipeline into the SDK and recovers the
LoCoMo regression that the shipped v0.8.1 LLM-only sibling reranker had
introduced. The benchmark harness is now a thin shim that delegates to the SDK,
so the recall behavior that is measured is the recall behavior that ships.

### Added

- Cognitive recall now lives entirely in the SDK: `recall_memories(graph_augmented=True)`,
  `compose_context`, and `two_pass_rerank` are first-class APIs on
  `UniversalMemoryManager` (previously duplicated inside the benchmark harness).
- Graph-augmented recall: multi-query reformulation → edge traversal → sibling
  enrichment, followed by the rerank stack
  (cross-encoder → evidence → recency → event-proximity → MMR).
- Opt-in cross-encoder reranking (bge-reranker via fastembed), gated on
  `KUMIHO_RERANK_CROSS_ENCODER=1`.

### Fixed

- **Multi-hop recall regression.** The cross-encoder and widen-then-trim step are
  now applied **per sub-query** (per reformulated angle) rather than post-merge,
  so each angle keeps its best evidence instead of being averaged away. On the
  LoCoMo `conv-26` sample this moved multi-hop from **0.19 → 0.40** F1.
- **Sibling reranking now keeps a cosine-embedding fallback** instead of the
  v0.8.1 LLM-only replacement, which had regressed single- and multi-hop
  retrieval. The LLM signal refines ranking; the embedding signal guarantees the
  right sibling is never dropped.
- Reformulation fallback no longer demotes already-recalled items or loses query
  angles when a sub-query returns nothing.

### Measured

Full 10-conversation LoCoMo (token-F1, gpt-4o answer, clean backend):

| category | F1 | vs Mem0 |
|---|---|---|
| single-hop | 0.449 | +0.062 |
| multi-hop | 0.393 | **+0.107** (#1) |
| temporal | 0.530 | **+0.041** (#1) |
| open-domain | 0.313 | −0.164 |
| **5-cat** | **0.564** | restores the 0.565 record |

LoCoMo-Plus cognitive judge accuracy holds at **93.3%** parity (no crown-jewel
regression from the standard-LoCoMo recovery).

## v0.8.2

**Release Date:** 2026-07-06

Bug fix: recall now surfaces the LLM-extracted atomic facts and ranks
stacked-revision siblings on them — restoring direct single-hop retrieval that a
prior sibling-reranker change had regressed on the LoCoMo benchmark.

### Fixed

- `UniversalMemoryManager.build_recalled_context` appends a concise `Facts:`
  block from the revision's extracted facts, so the answering LLM reads the
  precise attribute→value claim directly (e.g. *"Melanie has been married for
  five years"*) instead of having to infer it from the narrative summary.
- `_filter_siblings_by_embedding` folds the extracted facts into the text that
  is scored against the query. A revision whose title/summary is off-topic but
  whose facts hold the answer (e.g. a *"Sweden"* fact under a *"counseling"*
  summary) now ranks into context instead of being dropped by the sibling
  reranker.

Measured on the official LoCoMo benchmark (summarized mode): recovers the
direct single-hop retrieval regressed by the LLM sibling reranker and improves
open-domain recall, with no change to multi-hop.

## v0.8.1

**Release Date:** 2026-07-05

Bug fix for the recall/engage deduplication guard.

### Fixed

- `kumiho_memory_recall` / `kumiho_memory_engage` keyed their 5-second dedup
  guard on a single global timestamp, so **any** recall within the window was
  suppressed regardless of query — a session-wide singleton lock. Under
  concurrency (e.g. parallel agents) this starved legitimate **distinct**
  recalls, returning `count=0` "Duplicate recall within dedup window" on a
  first, unrelated call. The guard now keys on a signature of the query +
  scope (`space_paths`, `memory_types`, `recall_mode`, `graph_augmented`), so
  only a **true duplicate** (same query + scope) within the window is
  suppressed; distinct queries — including concurrent ones — always execute.

## v0.8.0

**Release Date:** 2026-07-04

`v0.8.0` adds **`event_date`** — a semantic *valid-time* for each memory (when the
remembered event actually occurred), distinct from `created_at` (when it was
stored). The summarizing LLM already reads the raw conversation, so it tags the
temporal referent at write time ("prospective indexing").

### New — event_date (valid-time)

- **Extraction** — the summarizer emits a normalized ISO `event_date`
  (`YYYY-MM-DD`, degrading to `YYYY-MM` / `YYYY`) per event, resolving relative
  references ("last Tuesday", "two weeks ago") against an anchor in the
  conversation. Empty when no date is inferable — never fabricated.
- **Storage** — the earliest concrete event date is stored as clean revision
  metadata, kept strictly separate from the server-authoritative `created_at`.
- **Surfacing** — recall returns `event_date` in **both** summarized and full
  mode. In summarized mode (where raw content is not loaded) the recalled
  context is prefixed with `[event_date]`, giving temporal questions a date
  anchor they otherwise lack.
- **Ranking (opt-in, default-off)** — a temporal event-proximity prior in
  `recall_rerank`. It fires **only** for temporal queries: enable
  `RerankConfig.event_proximity_enabled` **and** pass `rerank(..., query_time=...)`.
  With no `query_time` it is a strict no-op, so general recall is never
  reweighted. Recency (storage age) and event-proximity (valid-time) are capped
  jointly so two correlated temporal priors can't outweigh relevance.

Backwards compatible: memories without an `event_date` (legacy or non-temporal)
carry no key and are unaffected at every stage. Requires kumiho-server to reserve
`event_date` from the fulltext blob ([kumiho-server#25](https://github.com/KumihoIO/kumiho-server/pull/25)); persistence itself needs no server change.

### Fixed

- Corrected the default cross-encoder model id (`Xenova/bge-reranker-base` →
  `BAAI/bge-reranker-base`), which had silently disabled the bundled
  `fastembed` reranker. A guard test now pins it to a supported id.

## v0.7.0

**Release Date:** 2026-07-04

`v0.7.0` adds a **post-recall reranking pipeline** — the reranking stage peer
memory systems (Zep, mem0) ship and Kumiho was missing — layered on top of the
existing evidence-grade weighting.

### New — `recall_rerank`

Recall now runs `cross-encoder (optional) → +evidence → +recency → sort → MMR`
in a single, deterministic pass:

- **Recency decay** — recent memories get a small exponential boost
  (half-life 45 days, max +0.12), so fresh knowledge breaks ties over stale
  memories. No-ops when a memory has no timestamp.
- **MMR diversity** — greedy maximal-marginal-relevance reorder (λ=0.72,
  relevance-dominant) suppresses near-duplicate revisions crowding the top-k,
  complementing Dream State's write-time dedup.
- **Relevance reranker (opt-in)** — a pluggable `Reranker` stage. Two backends:
  - `KUMIHO_RERANK_CROSS_ENCODER=1` — local `fastembed` multilingual
    `bge-reranker` (ONNX, no torch, no API).
  - `KUMIHO_RERANK_LLM=1` — the **host LLM itself** reranks, reusing the
    manager's already-configured adapter (`summarizer.adapter` + `light_model`)
    — no separate reranker model, download, or API key. This is the
    "the LLM running Kumiho reranks" design, wired as a first-class option via
    `make_llm_reranker`. One `chat` call per recall; any failure is a safe
    no-op. (Cross-encoder wins if both are set.)

Recency + MMR are **default on and conservative**; the server's relevance order
is still preserved when no signal (evidence, recency, cross-encoder) actually
reweights the set, so ungraded recall stays backward-compatible.
`KUMIHO_RECALL_RERANK=0` is a kill switch.

Part of the retrieval-optimization roadmap alongside kumiho-server's normalized
hybrid fusion (#23), configurable embeddings endpoint (#24), and Korean
tokenizer identifier fix (#22).

## v0.6.1

**Release Date:** 2026-07-03

`v0.6.1` is a patch release fixing the mirrored evidence-tag carrier
introduced in `v0.6.0`, discovered via live verification against a
self-hosted CE server.

### Bug fix

The kumiho server freezes a revision as immutable the instant its
`published` tag is applied — any tag applied afterward is silently
rejected (`PERMISSION_DENIED`). `consolidate_session` and
`skill_ingest`'s evidence-tag mirroring (issue #9, `v0.6.0`) tagged
`published` *before* the mirrored `evidence:<level>` tag, so the tag
carrier never actually landed on the server for any consolidated or
skill-ingested memory — only the `evidence_level` **metadata** carrier
worked correctly. Recall reranking and badges were unaffected (they
read `evidence_level` from metadata), but tag-based server-side
time-range auditing of evidence grades was silently broken.

- `consolidate_session`: evidence tag now applied before `published`.
- `skill_ingest.ingest_skill` / `ingest_file`: same ordering fix.
- Added order-regression assertions to the evidence test suite.

This fix pairs with a companion fix in the core `kumiho` SDK
(`kumiho>=0.10.1`): `tool_memory_store` — the default store backend
used by `consolidate_session` — called a nonexistent module-level
function for tag application, so **no tag was ever actually applied**
via that path, including the base `published` tag itself. Upgrading
`kumiho-memory` alone fixes the ordering; upgrading `kumiho` too is
required for any tag (including `published`) to land at all via the
default store path. See the `kumiho` package's own release notes.

### Upgrade

```bash
pip install -U "kumiho-memory[all]" "kumiho>=0.10.1"
```

No API changes; safe to upgrade from `v0.6.0` with no code changes.

---

## v0.6.0

**Release Date:** 2026-07-02

`v0.6.0` is a minor release introducing **Level-of-Evidence belief
revision**: memories carry a trust grade, and revision/consolidation/
recall decisions weigh that grade — official statements are pinned,
claims promote to facts only when independently corroborated, and
low-trust content is down-ranked instead of competing on relevance
alone. Entirely client-side; no kumiho-server changes.

### Highlights

#### Evidence-level schema (`evidence` module)

- New `evidence_level` revision-metadata convention (`official` /
  `corroborated` / `single_source` / `unverified`) mirrored into a
  `evidence:<level>` graph tag for server-side time-range history.
- `evidence_tag()` / `parse_evidence()` helpers; metadata wins when the
  two carriers diverge.
- `ingest_message` / `consolidate_session` accept `evidence_level` +
  `source`; grades are stashed at ingest and applied at consolidation.
  Grades are **only stamped when provided** — ungraded flows are
  byte-identical to previous behavior.
- `skill_ingest` and the `ingest-skill` CLI gain `--evidence-level`.

#### Corroboration-aware evidence assessor (`assessors` module)

- `create_evidence_assessor(adapter, policy=EvidencePolicy())` — a
  drop-in `AutoAssessFn` that grades incoming claims via a screened
  three-stage pipeline (heuristic → graph novelty → LLM judgment +
  policy):
  - **Official pinning** — claims contradicting an `evidence:official`
    memory are stored `unverified` with the conflict recorded; the
    pinned belief is never revised.
  - **Corroboration** — ≥ N agreeing memories with distinct sources and
    no contradiction promote to `corroborated`, `memory_type=fact`,
    with optional `SUPPORTS` edges to corroborators.
  - The assessor can never emit `official` — that grade is operator-only.
- New MCP env wiring: `KUMIHO_EVIDENCE_ASSESSOR=1`,
  `KUMIHO_EVIDENCE_MIN_CORROBORATION`, `KUMIHO_EVIDENCE_SUPPORTS_EDGES=1`.
- `EdgeType.SUPPORTS` added to `GraphAugmentationConfig`'s default
  traversal edge types.

#### Dream State deployment policy (`dream_state` module)

- `DreamState(extra_instructions=...)` appends deployment-specific
  policy (e.g. "never deprecate `evidence:official` memories") under a
  `## DEPLOYMENT POLICY` section of the assessment prompt.
- Three routes with documented precedence: explicit arg >
  `KUMIHO_DREAM_EXTRA_INSTRUCTIONS` env var; `""` disables the env
  policy. New CLI flag `kumiho-memory dream --policy`.
- The assessment payload now includes each memory's `evidence_level`
  and policy-relevant graph tags. Hard guardrails (deprecation cap,
  published protection, conservative-KEEP) remain enforced in code
  after the LLM's suggestions and are not overridable by policy.

#### Evidence-weighted recall + context badges (`evidence_rank` module)

- Deterministic score adjustment per grade (`official` +0.15,
  `corroborated` +0.08, `single_source` 0.0, `unverified` −0.10) — zero
  extra LLM calls, applied in both plain and graph-augmented recall
  before result caps.
- **Default ON**, with a strict no-op guarantee: recall results are
  byte-identical when no retrieved memory carries a grade. Kill switch:
  `KUMIHO_EVIDENCE_RERANK=0`.
- `kumiho_memory_engage` context is prefixed with grade badges
  (`[official]`, `[unverified]`); `kumiho_memory_recall` exposes the
  grade as the `evidence_level` field.

#### Space profiles (`space_profiler` module, new)

- `SpaceProfiler` aggregates per-Space churn/evidence/stability signals
  from existing SDK queries (pure aggregation, no LLM) and classifies
  each Space as `canonical` / `working` / `correspondence`.
- Profiles persist as versioned `kind="space-profile"` items with
  `SUPERSEDES` edges linking runs, so profile drift is itself a
  queryable chain. A `space_class` Space attribute pins the label; the
  profiler then reports pin/observation disagreement as drift instead
  of relabeling.
- New CLI subcommand `kumiho-memory profile` and MCP tool
  `kumiho_memory_space_profile`. Read-side API: `get_space_profile()`.

### MCP Tools

13 tool wrappers, up from 10:

| Tool | Description |
| ------ | ------------- |
| `kumiho_memory_engage` | Recall + build context in one call |
| `kumiho_memory_reflect` | Buffer response + store captures |
| `kumiho_memory_space_profile` | Profile each Space's knowledge dynamics |

(The other 10 are unchanged from `v0.5.3` — see the full table in
`README.md`.)

### Modules

New: `evidence`, `evidence_rank`, `assessors` (evidence-aware additions),
`space_profiler`.

### Test Coverage

281 tests total (up from 84 in `v0.3.1`), including dedicated suites for
`evidence`, `assessors` (evidence path), `evidence_rank`, `dream_state`
(policy injection), and `space_profiler`.

### Requirements

Unchanged from `v0.5.3` — no new external dependencies.

### Upgrade

```bash
pip install -U kumiho-memory[all]
```

No breaking API changes — every new parameter is additive with a
back-compatible default, and evidence-aware features are either
explicitly opt-in (assessor, Dream State policy) or strict no-ops on
ungraded data (recall reranking).

---

## v0.5.3

**Release Date:** 2026-05-13

`v0.5.3` is a patch release adding relevance-threshold filtering to
memory recall tools.

- Added `min_score` to `kumiho_memory_recall` and
  `kumiho_memory_engage`.
- Supports `CONSTRUCT_MEMORY_MIN_RELEVANCE_SCORE` and
  `KUMIHO_MEMORY_MIN_RELEVANCE_SCORE` as default thresholds when
  `min_score` is not passed by the caller.
- Filters low-scoring memories before `count`, `source_krefs`, and
  engage context are built.

---

## v0.3.1

**Release Date:** 2026-02-24

`v0.3.1` is a patch release introducing graph-augmented recall,
sibling revision filtering, recall deduplication, tool execution memory,
edge discovery, and expanded MCP tool coverage effeiciency.

---

### Highlights

#### Graph-Augmented Recall (New)

New `graph_augmentation` module with `GraphAugmentedRecall` engine that
enhances standard vector recall with graph traversal:

- **Multi-query reformulation** — LLM rewrites the user query into
  multiple search vectors for broader coverage (optional, skipped when no
  LLM adapter is configured — e.g. in Claude Code where the host agent
  IS the LLM).
- **Edge traversal** — follows `DERIVED_FROM`, `REFERENCED`, and other
  typed edges to discover connected memories that vector search alone
  misses.
- **Semantic fallback** — secondary vector search on traversal results
  for relevance scoring.
- Enabled via `KUMIHO_GRAPH_AUGMENTED_RECALL=1` env var.

#### Sibling Revision Filtering

Stacked items (multiple revisions on a single item) now return filtered
sibling context instead of raw revision dumps:

- **BM25-light keyword overlap** — default mode, scores siblings by
  term overlap with the query and returns the strongest matches within a
  character budget.
- **Embedding-based cosine filtering** — opt-in via
  `KUMIHO_SIBLING_SIMILARITY_THRESHOLD` env var.  Uses the configured
  `EmbeddingAdapter` (e.g. `text-embedding-3-small`) to rank siblings by
  semantic similarity and return top-k above the threshold.
- Configurable via `sibling_strong_score`, `sibling_char_budget`,
  `sibling_similarity_threshold`, and `sibling_top_k` on
  `UniversalMemoryManager`.

#### Recall Deduplication

Server-side guard against duplicate `kumiho_memory_recall` calls within
the same model response:

- `threading.Lock` serializes parallel recall calls.
- Any call within a 5-second window of the previous recall returns an
  empty result with a `deduplicated: true` flag and a warning note.
- Eliminates duplicate "Retrieved..." output lines when models generate
  parallel tool calls despite instructions.

#### Edge Discovery (`kumiho_memory_discover_edges`)

New MCP tool that creates relationship edges from a newly stored memory
to related existing memories:

- Generates implication queries (future scenarios where the memory would
  be relevant) using the LLM.
- Searches for matching memories and creates `REFERENCED` edges above a
  configurable similarity threshold.
- Designed to run after `kumiho_memory_store` or
  `kumiho_memory_consolidate`.

#### Tool Execution Memory (`kumiho_memory_store_execution`)

New MCP tool to store build/deploy/test outcomes as structured memories:

- Successful executions stored as `action` type; failures as `error`.
- Captures stdout, stderr, exit code, duration, tool names, and topics.
- Artifacts stored alongside the memory entry.

#### Recall Modes

`kumiho_memory_recall` now supports a `recall_mode` parameter:

- `full` (default) — includes artifact content (raw conversation text)
  in results.
- `summarized` — returns only title + summary for lighter context.

### MCP Tools

10 MCP tool wrappers, up from 9:

| Tool | Description |
| ------ | ------------- |
| `kumiho_chat_add` | Add message to Redis working memory |
| `kumiho_chat_get` | Retrieve session messages |
| `kumiho_chat_clear` | Clear session working memory |
| `kumiho_memory_ingest` | Buffer message + recall context |
| `kumiho_memory_add_response` | Add assistant response to buffer |
| `kumiho_memory_consolidate` | Summarize, redact, store to graph |
| `kumiho_memory_recall` | Semantic search with dedup guard |
| `kumiho_memory_discover_edges` | Link new memory to related memories |
| `kumiho_memory_store_execution` | Store tool/command results |
| `kumiho_memory_dream_state` | Run Dream State consolidation cycle |

### Other Changes

- `MemorySummarizer` adapter is now lazy-initialized — no API key
  required at import time.  Enables MCP server startup in Claude Code
  without an external LLM key.
- `GraphAugmentedRecall` works without an LLM adapter — edge traversal
  and semantic fallback run without an external API key.  Only
  multi-query reformulation is skipped.
- `OpenAICompatEmbeddingAdapter` added to `summarization` module for
  embedding-based sibling filtering.
- `CredentialDetectedError` added to `privacy` module for explicit
  secret rejection.
- `DreamState` patched to use `MemorySummarizer` consistently.

---

## Modules

| Module | Public API |
| -------- | ------------ |
| `memory_manager` | `UniversalMemoryManager`, `get_memory_space` |
| `redis_memory` | `RedisMemoryBuffer` |
| `summarization` | `MemorySummarizer`, `LLMAdapter`, `EmbeddingAdapter`, `OpenAICompatAdapter`, `OpenAICompatEmbeddingAdapter`, `AnthropicAdapter` |
| `privacy` | `PIIRedactor`, `CredentialDetectedError` |
| `retry` | `RetryQueue` |
| `dream_state` | `DreamState`, `MemoryAssessment`, `DreamStateStats` |
| `graph_augmentation` | `GraphAugmentedRecall`, `GraphAugmentationConfig` |
| `mcp_tools` | `MEMORY_TOOLS`, `MEMORY_TOOL_HANDLERS` |

---

## Test Coverage

84 tests total:

- 18 MCP tool tests (+2 dedup tests)
- 15 Dream State tests
- 28 memory manager tests
- 10 retry tests
- 9 Redis buffer tests
- 3 summarization tests
- 1 privacy test

---

## Requirements

- Python >= 3.10
- `kumiho` >= 0.9.0
- `redis[hiredis]` >= 5.0.0
- `requests` >= 2.31.0

Optional extras:

- `kumiho-memory[openai]`
- `kumiho-memory[anthropic]`
- `kumiho-memory[all]`

---

## Upgrade

```bash
pip install -U kumiho-memory[all]
```

### Breaking Changes

- `kumiho_memory_recall` duplicate calls now return empty results
  (`count: 0`, `deduplicated: true`) instead of executing.  Callers
  relying on rapid sequential recalls within 5 seconds will see empty
  responses.

---

## v0.1.2

**Release Date:** 2026-02-09

`v0.1.2` is a documentation-focused patch release.

- Updated `README.md` status block with latest patch metadata
- Corrected README heading formatting
- Synced package version metadata across `pyproject.toml` and `kumiho_memory.__version__`
- Corrected project changelog URL to point to `RELEASE_NOTES.md`

No breaking API changes are introduced in this release.

---

## v0.1.1

**Release Date:** 2026-02-08

`v0.1.1` is a patch release focused on MCP integration, Redis proxy/auth
hardening, and test expansion.

### MCP Tool Integration (New)

Added `kumiho_memory.mcp_tools` with **9 MCP tool wrappers** that are
auto-discovered by the core `kumiho` MCP server when `kumiho-memory` is
installed.

### Redis Proxy + Auth Resilience

- Better handling of Firebase token vs control-plane token flows
- Automatic token refresh and retry on proxy auth failures (401/403)
- Cleaner fallback path between discovery, direct URL, and proxy mode

### Documentation Updates

- Expanded README with onboarding/initialization guidance
- Added MCP integration setup and tool reference
- Refreshed usage examples for working memory and Dream State

No breaking API changes from `v0.1.0` are introduced in this release.
