# Belief-grounded insight: development plan

Status: phases 1-4 implemented as explicit host-driven workflows in PR #32;
available through explicit API flags in 1.5.0. Automated provider execution and broad rollout remain out of scope.
Date: 2026-09-09.

## Product goal

Use experiences, decisions and revisable beliefs to help an answering agent
notice a relevant connection, a changed premise, or an unresolved conflict in a
user's question. A useful answer states a hypothesis, names its evidence and
conditions, and suggests a way to check it. A direct factual answer remains a
valid outcome; the system must not invent an insight for every question.

The long-term loop is experience -> belief -> situated prediction -> observed
outcome -> belief revision. This PR implements bounded question-time preparation and validation, structured
experience/outcome capture, and explicit Dream State pattern prepare/store/check.
The host performs synthesis. Outcome reports support review; they do not establish
causality or automatically train a model or promote a belief.

## Existing foundation and gaps

- `graph_augmentation.py` prioritizes belief-change edges under traversal caps,
  distinguishes fact-level disputes from domain relations, and supplies
  `contested_by` markers. The absence of that marker is not a complete conflict
  search, especially when graph augmentation is disabled.
- `grounding.py` marks a decision whose factual support was superseded.
  `superseded_by` on such an entry identifies the replacement **fact**, not a
  replacement decision; the original decision might still be appropriate.
- `evidence.py` resolves provenance grades. `trust_vocab.py` keeps them separate
  from self-reported confidence. Repetition and multiple graph references do
  not constitute independent corroboration.
- `memory_manager.py` recalls summaries, decisions, sources and valid intervals.
  Decisions can contain their rationale in prose. The new experience records
  add explicit pinned outcome observations; legacy prose is not silently converted
  into an observed outcome.
- Existing PRs #29/#31 own continuity evaluation and claim-origin/acceptance
  markers. This PR starts from main, does not stack their commits, and consumes
  optional state fields only when present. It does not duplicate their writers.

## Phase 1: opt-in insight brief (this PR)

Add `include_insights: true` to `kumiho_memory_engage` and expose the pure Python
`kumiho_memory.insight.build_insight_brief` helper. The normal engage response
remains unchanged when omitted or false. When enabled, a separate
`insight_brief` uses the same filtered, scoped recall results, before summarized
mode removes sibling prose. Existing `context`, ranking, retrieval options and
source list retain their meaning. The payload token estimate includes the brief.

The brief supplies bounded review candidates for three cases:

1. **Changed premise**: an explicit grounding-stale marker warrants checking
   whether the old rationale still applies.
2. **Unresolved conflict**: an explicit dispute marker warrants comparing both
   claims, their sources and conditions before choosing either.
3. **Decision review**: a recalled decision may help evaluate a new choice,
   after comparing constraints and checking outcomes. This is an opportunity
   to reason, not proof of a repeated pattern or a successful past decision.

The pure helper also accepts `max_insights` (default 3, hard maximum 5) and
`retrieval_complete` (default true, false for partial/failed retrieval). The
builder scans at most 50 memories, 100 revisions and 10 siblings per item;
each candidate follows at most three already-supplied related references.
It returns at most 12,000 characters under ordinary `json.dumps`, removing
whole candidates when necessary. Truncation is reported; references are never
cut into apparently valid citations. This is a bound on the added brief, not
on the existing engage context or the whole response.

Every candidate carries bounded recalled snippets, exact revision references,
provenance/state information when available, missing related references, and a
verification question. Sibling grades are not inherited from a different
revision. Item-level warnings are labelled as such instead of pretending that
one exact sibling has been proven invalid. Stored prose is data, not an
instruction to the host.

The new builder performs no I/O, model invocation, writes, or graph expansion.
Existing recall can still use configured model/graph services; the flag adds
none. Evidence outside the recall scope or budget is not fetched. Unknown
applicability stays unknown, and no generated conclusion is persisted or
promoted. Scope authorization remains with the existing recall backend; the
standalone helper must receive only caller-authorized results.

`ready` means review candidates exist, not that their hypotheses are true.
`insufficient_evidence` means the selected recall supplies no suitable candidate.
`retrieval_incomplete` on engage means the backend reported an error, possibly
with partial results. Duplicate recalls retain the existing dedup response and
return no new brief: set the flag on the first engage call or use the pure
helper on already-held results. No backend outage is presented as an absence
of stored knowledge.

### Host answer contract

1. Read the current question and explicit goals; distinguish inferred goals.
2. Compare candidate evidence with current conditions and inspect missing
   evidence only when it matters and within authorized scope.
3. If useful, offer one concise hypothesis with source support, a competing
   explanation or uncertainty where material, and a practical check.
4. Do not turn the hypothesis into a fact through reflect/decompose merely
   because it appeared in an answer or the user agreed it sounded plausible.
5. If there is no useful supported connection, answer directly or ask a focused
   question. Never claim personal experience outside the provided memories.

### Verification gates

Offline unit/integration tests cover marker semantics, missing evidence,
provenance independence, revision attribution, malformed data, deterministic
bounds, unchanged defaults, shared dedup behavior, score filters, backend errors,
and tenant scoping. Existing non-live regressions run in CI on Python 3.10-3.12.
These gates establish the contract; they do **not** establish improved answer
quality, causal accuracy, or reduced overall latency.

## Phase 2: question-time synthesis and paired evaluation (implemented)

`engage(include_insights=true)` adds `synthesis_request` beside `insight_brief`.
Optional `current_context` and `goals` inform synthesis without changing retrieval.
The host follows the returned instructions and JSON `output_contract`, choosing
`hypothesis`, `direct`, or `clarify`. Hypotheses include exact source references,
conditions, an alternative explanation, a verification step and caveats. A useful
direct answer is valid even when the rule-based brief has no candidates.

`kumiho_memory_validate_insight_response` checks snapshot integrity, format,
lengths and citation membership. It explicitly returns
`semantic_support_verified: false`: a permitted citation need not support a claim.
The snapshot hash detects accidental changes; it is not an authentication token.

Saved experiences and pattern proposals use separate item kinds, so ordinary
conversation recall does not discover them. Also set `include_learned_sources:
true` to include those records in synthesis. This requires `include_insights:
true` and adds two kind-specific discovery calls, at most three canonical records,
and up to six distinct referenced pattern-health records (each reads revision and
item; at most 18 explicit SDK state reads). Existing SDK discovery may perform its
own fallback RPCs; returned counters describe this layer, not transport calls.
Caller `memory_types` and `min_score` also constrain learned results; a positive
score threshold excludes sources with unavailable scores and reports partial
retrieval. Canonical event/content IDs survive synthesis so duplicate reports do
not become apparently independent experiences.
Scope stays within the configured project and requested spaces; incompatible
custom retrieval signatures fail without broadening scope. No provider is called.

Fresh canonical learned rows precede ordinary rows within the same bounded
synthesis source budget, so old same-reference snippets cannot hide current health
markers. Sources can be omitted when that budget is full. `learned_source_status` reports discovery
and canonical-state checks; only `synthesis_request.source_krefs` identifies text
actually included for synthesis. A partial learned lookup marks synthesis as
retrieval-incomplete while preserving the ordinary context/results. Source-health
warnings are retained in summary text. Default engage, and `include_insights`
without this additional option, perform no extra retrieval for learned kinds.


Only title/summary and whitelisted provenance/state fields enter the new packet.
Credential-bearing atoms are screened before truncation and supported PII is
redacted. Pattern-based screening has the existing privacy module's documented
false positives/negatives; it is not a complete secret detector. Source JSON is
bounded to 12,000 characters by default (32,000 hard maximum), plus a separate
12,000-character brief, query (2,000), current context (4,000), and eight goals
(500 each). Omission/redaction/truncation receipts are included. Existing engage
`context` and `results` retain their established behavior. The complete response
estimate includes the added packet and brief; consumers should budget that total.

`prepare_baseline_request` preserves identical query/context/goals/source packets
and source budget while removing the brief and extra synthesis instruction.
`scripts/evaluate_insight.py` compares captured host outputs with externally
supplied blinded independent rubric labels. It does not call a provider or grade
meaning. Missing labels remain unscored. Invalid answers remain in comparisons;
unequal source packets disqualify quality comparisons. Guidance overhead is
reported separately: this is not an equal-total-token treatment comparison.

See [INSIGHT_EVALUATION.md](INSIGHT_EVALUATION.md) for the real-memory pilot and
its limitations, and `tests/fixtures/insight_scenarios.json` for synthetic cases.
Do not enable by default or claim general improvement from functional tests or
a small host demonstration.

## Phase 3: experiences and observed outcomes (implemented)

`kumiho_memory_record_experience` stores an explicit extraction with a stable
caller-owned `experience_id`, situation, goal, alternatives, decision, rationale,
applicability conditions and expected outcome. Observed outcome and `observed_at`
are either both present or both unknown. Origin, decision acceptance, outcome
status and user acceptance are separate fields. `recorded_at` is recording time.

`kumiho_memory_record_outcome` appends a separate observation referencing the
exact original experience revision. It does not rewrite the original decision.
Both tools validate source access within the configured project/current tenant
before writing, reject detected credentials, and redact supported PII. Source
identifiers are kept exact. Canonical `experience_record` JSON and source links
preserve lineage without requiring a provider API.

Records remain unverified and are not automatically published. The tools append
snapshots; this is not a claim that the backend freezes unpublished metadata.
`record_id` fingerprints content and `experience_id` identifies the actual event.
Retries can append duplicates (`idempotent: false`); neither duplicate revisions
nor multiple observations of one event count as independent experiences/evidence.
Source reads and writes are not transactional. SDK graph links are best effort;
source references are also retained in the canonical JSON.

## Phase 4: explicit Dream State patterns (implemented)

1. Call `kumiho_memory_prepare_patterns` with 1-12 pinned experience/outcome
   references and 1-8 explicit absolute project spaces. This calls
   `DreamState.prepare_patterns` without constructing the provider-backed cycle.
2. The host reviews the bounded packet (24,000 JSON characters) and either
   proposes no pattern or supplies a `conditional_lesson` / `recurring_pattern`
   with conditions, explicit counterexamples (empty means unknown) and sources.
3. Call `kumiho_memory_store_pattern` with the original request and candidate.
   Sources are re-read and sanitized snapshots/state are compared before writing.
   Recurring patterns require at least two distinct experience IDs and items.
   Repetition/shared lineage does not establish corroboration.
4. Before using a stored proposal, call `kumiho_memory_check_pattern` with its
   pinned reference and permitted spaces. Current revision and item markers are
   read: changed, contested or retired sources yield `stale`; missing/inaccessible
   sources yield `unknown`; healthy explicit sources yield `reviewable`.
   Applicability to the current situation always remains `unknown` for the host.

Patterns are stored as `inferred`, `proposal`, `unverified`, with canonical source
lineage and no automatic publishing/promotion. The check observes state and does
not mutate or retract graph nodes. Item warnings are labelled item-scoped because
they may describe another revision. No automatic graph-wide invalidation, full
contradiction search, causal proof, scheduler or provider call is added. Prepare
and store read at most 12 source revisions and their 12 items; check can also read
the pattern revision/item (at most 26 SDK reads). Shared source items may repeat.
Existing ordinary Dream State behavior and default recall stay unchanged.

## Research follow-up

[INSIGHT_RESEARCH_OUTLINE.md](INSIGHT_RESEARCH_OUTLINE.md) proposes a second paper
on revision-aware insight. The intended contribution is a testable integration of
belief state, experience/outcome lineage, conditional synthesis and re-review.
Formal AGM correspondence does not prove natural-language insight correctness.
Priority claims and broad performance claims require further evidence.
