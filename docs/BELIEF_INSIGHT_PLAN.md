# Belief-grounded insight: development plan

Status: first implementation in this PR; later phases are proposed, not shipped.
Date: 2026-09-09.

## Product goal

Use experiences, decisions and revisable beliefs to help an answering agent
notice a relevant connection, a changed premise, or an unresolved conflict in a
user's question. A useful answer states a hypothesis, names its evidence and
conditions, and suggests a way to check it. A direct factual answer remains a
valid outcome; the system must not invent an insight for every question.

The long-term loop is experience -> belief -> situated prediction -> observed
outcome -> belief revision. This PR implements the evidence preparation step,
not automatic causal discovery or outcome learning.

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
  Decisions can contain their rationale in prose; explicit decision-to-outcome
  links are not yet available. A summary is not an observed outcome.
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

## Phase 2: measured question-time synthesis

Build a consented or synthetic fixture set pairing questions with prior
experiences, decisions, counterevidence and current constraints. Include changed
user preferences, failed decisions, old but still valid beliefs, duplicated
sources, agent proposals, missing outcomes, and questions needing no insight.
Compare ordinary engage with the same recall plus the brief using a fixed
answer model and settings, with blinded assessment of:

- source and attribution accuracy;
- correct use of changed premises and conflicting evidence;
- applicability to the question and user-stated goals;
- usefulness of the proposed next check;
- unsupported causal/pattern claims and unnecessary insight generation;
- additional context, answer latency and cost.

Predeclare a release gate after establishing baseline variance: no regression
in grounding/abstention cases, improved useful-connection judgments, and an
explicit context/latency budget. Do not enable by default based only on passing
unit tests. Provider execution remains separately configured and authorized.

## Phase 3: experience and outcome records

Define structured records for situation, goal, alternatives, decision,
rationale, applicability conditions, expected outcome, observed outcome and
provenance. Link outcomes to the decision they evaluate, with observation time
separate from recording time. Record pending/unknown outcomes explicitly.

Distinguish user acceptance, observed success, and independently supported
belief. An outcome must not automatically prove the decision caused it.
Use the existing revision/decomposition machinery for actual belief changes;
retain source lineage and propagate staleness when premises change.

## Phase 4: Dream State pattern candidates

Use bounded offline passes to propose reusable conditional beliefs across
experiences. Persist them as clearly inferred candidates with source lineage,
conditions, counterexamples and unresolved questions. Deduplicate dependent
sources before considering corroboration. On recall, re-check current premises
and invalidate candidates when their supporting revisions change.

Start in preview mode, evaluate false pattern discovery and usefulness against
Phase 2 fixtures, then consider an explicit opt-in rollout. Do not add a
background scheduler, new graph predicates, automatic promotion, or broad
memory scans as part of Phase 1.
