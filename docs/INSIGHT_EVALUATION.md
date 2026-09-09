# Belief insight: implementation assessment and live-memory pilot

Date: 2026-09-09. PR #32. Opt-in, unreleased. This is a feasibility assessment,
not a paper result establishing general insight accuracy or superiority.

## Result

The host workflow produces useful grounded answers from actual recalled memories,
but this small pilot does not isolate a causal gain from the new guidance. In an
arm-masked same-host-family review, the enriched answer was judged more useful in
2 of 8 pairs, tied in 6, and worse in none. Strong baseline answers produced a
ceiling effect. The improvement was in applicability caveats and more discriminating
verification plans, not discovery of a new verified fact or causal law.

| Measure | Ordinary source packet | Sources + insight guidance |
| --- | ---: | ---: |
| Structurally valid responses | 8/8 | 8/8 |
| Out-of-packet references | 0 | 0 |
| Mean usefulness, subjective 0-4 rubric | 3.75 | 4.00 |
| Mean grounding, subjective 0-4 rubric | 3.875 | 4.00 |
| Clearly unsupported material claims found by reviewer | 0 | 0 |
| Arithmetic answered directly without memory citations | 1/1 | 1/1 |

Zero detected unsupported claims is a reviewer observation, not proof of zero
hallucinations. The mean differences (+0.25 usefulness, +0.125 grounding) are
purely descriptive; no significance, population accuracy, or reliability claim
is made. The JSON harness also reports mode-label agreement, a narrow formatting
proxy that must not be called abstention accuracy: a direct answer can contain
excellent insight.

## Protocol and reproducibility

Eight questions were purposively selected before examining their recalled results:
operational diagnosis, test-contract tradeoffs, evaluation design, evidence trust,
user design corrections, conjecture reuse, quality claims, and elementary arithmetic.
All eight attempts are retained, including unrelated retrievals. Each used the live
configured Kumiho recall tool with limit=4, summarized mode, graph_augmented=false.
Sibling expansion produced 4,4,4,4,4,5,4,10 source snapshots respectively. The
arithmetic question is an intentional negative control, not a use case needing memory.

`prepare_insight_request` made bounded source packets from the same captured result
for both conditions. `prepare_baseline_request` removed only the review brief and
extra synthesis instruction. Sources, source budget, question, context, goals and
output contract were identical. Source parity checks passed for all pairs. This
isolates available source text, but does NOT hold total prompt length constant.

The baseline generator was the root host; the enriched generator was a separate
host agent. Both already knew the implementation context. Each produced one answer
per question and all outputs passed structural validation without quality retries.
The exact model version, decoding settings and context equivalence were unavailable.
This is therefore a host demonstration, not a controlled identical-model experiment.
No new provider API was invoked by the Python code.

A third agent, who generated neither set, read only the common source text and
responses labeled A/B, randomized separately per case. The treatment mapping was
withheld until labels were saved. This reviewer knew the implementation and shared
the host-model family; it was not an independent human reviewer. Its ordinal rubric
scored usefulness and grounding from 0 to 4, counted clearly unsupported claims,
and recorded case-specific reasons. No keyword overlap metric or model confidence
was used as an insight score. Label independence in the evaluation script is a
caller assertion, not a property the script can certify.

Raw source packets, outputs, mapping and labels remain in ignored local
`.pytest_cache/insight-eval-real/`. They are not committed because they contain
private memory. [INSIGHT_PILOT_METRICS.json](INSIGHT_PILOT_METRICS.json) contains
aggregate/ID-only results, protocol limitations and preparation timing. Reproduce
the calculations locally with:

```powershell
python scripts/evaluate_insight.py --input .pytest_cache/insight-eval-real/captures.json --output .pytest_cache/insight-eval-real/report.json
```

Public users can exercise the workflow with
`tests/fixtures/insight_scenarios.json`; these seven synthetic inputs have no
quality labels and are not a replacement for the private live sample.

## Case analysis

| Case | Retrieval and observed behavior | Paired usefulness |
| --- | --- | --- |
| R1: disappearing working memory | Relevant historical diagnosis and correction supported checking current TTL/version before assuming restart damage. Both separated historical evidence from the present cause. | Tie |
| R2: failing edge-read test | Recall missed the specific error and returned related contract/test lessons. Both admitted missing diagnostic evidence; enrichment more clearly distinguished failed reads from empty results and made the contract transfer conditional. | Enriched +1 |
| R3: truncation notices | Recall missed the direct notice experiment. Both refused to infer accuracy gains from unrelated benchmarks; enrichment added truncated/intact-context controls and unnecessary-abstention checks. | Enriched +1 |
| R4: repeated confident claims | Both used evidence independence rather than repetition/confidence, ignoring an unrelated trust-status result. | Tie |
| R5: remembered design correction | Both recovered the explicit trust/belief-change design intent and avoided inventing automatic implementation details. This is strong correction recall, not a newly discovered pattern. | Tie |
| R6: reusing conjectures | Both separated hypotheses from facts and decisions from observations. Neither established that the proposed design improves personalization. | Tie |
| R7: tests versus insight quality | Both separated implementation tests from answer-quality evidence and proposed matched comparisons plus regressions. | Tie |
| R8: elementary arithmetic | Retrieval was entirely unrelated, yet both answered directly without forced memory connections. | Tie |

R2 and R3 expose a retrieval boundary: the host can propose a better check with
partial evidence, but cannot recover a missing diagnostic fact through guidance
alone. Several packets contained distractors, including an apparent test-fixture
preference. Its presence is corpus noise in this sample, not evidence of a
cross-tenant authorization failure. Existing PII screening also masked some long
numeric benchmark values as credit-card-like strings; the answers appropriately
did not reconstruct those obscured values. These issues should be tracked apart
from synthesis quality.

## Cost and implementation evidence

The enriched request added a mean 2,938.25 serialized characters of guidance per
case (roughly 735 tokens using chars/4, not tokenizer-measured). This is the
request-only difference. The complete engage envelope also includes its legacy
context/results and a separate brief, so consumers must use the returned total
payload estimate rather than assuming that 735 is the full transport overhead.

Local deterministic preparation took median 4.086 ms and p95 8.896 ms across
20 passes over the eight fixed packets (160 preparations, one Windows/Python3.12
run). Shared live recall averaged 247.8 ms across the eight original captures.
Neither number measures host synthesis, end-to-end response latency or provider
cost; those were not measured. Reusable patterns might save future reasoning,
but this pilot does not demonstrate that hypothesis.

Offline tests separately exercise the functional lifecycle: structured experience
and distinct observation, canonical source checks, distinct-event recurrence,
proposal-only persistence, changed-source submission rejection, current item and
revision retirement markers, missing evidence, tenant/project scopes and read-only
checks. They verify behavior against SDK-shaped fakes, not a live storage trial.
Production memory was read for this evaluation; no synthetic evaluation patterns
were inserted into the user's real graph.

The live pilot tests ordinary recalled memories plus question-time guidance.
It does NOT evaluate long-term outcome learning, discovery of stored new-kind
experiences/patterns, semantic pattern retirement after an actual live belief edit,
or comparisons against Generative Agents/Reflexion/A-MEM/HippoRAG. Additional opt-in
learned-source retrieval is covered by offline integration tests only. The Python
validator checks citation membership, not entailment. Source-health checks read
available revision/item markers and do not implement complete dependency traversal
or automatic graph retraction. Unpublished snapshots are append-only through these
tools, not guaranteed immutable by the server.

## Next research gate

Keep the feature opt-in. Use this pilot to refine a preregistered controlled set
with fixed model/version/decoding, matched total token budgets, independent human
ratings and related-scenario holdouts. Include true premise changes and irrelevant
changes, repeated dependent evidence, failed/unknown outcomes and direct-answer
controls. Measure grounded usefulness together with unsupported inference and
semantic retirement precision/recall. Report retrieval coverage separately from
synthesis success. The [second-paper outline](INSIGHT_RESEARCH_OUTLINE.md) specifies
the proposed ablations and explicitly leaves novelty and superiority claims open.
