# Revision-Aware Insight from Belief Memory

Research outline for a possible second Kumiho paper. Prepared 2026-09-09. This is a proposal, not a manuscript or a report of demonstrated answer-quality improvements. The implementation tests and any small live pilot must be reported separately from the planned research evaluation.

## Research question and hypotheses

Can a memory system help an unchanged answer model make useful connections across prior experiences while withdrawing those connections when their supporting premises change?

The main hypothesis is that linking decisions, explicit observed outcomes, conditional pattern proposals, and revision state will improve **grounded usefulness after a premise change**, compared with equally budgeted retrieval or reflection alone. A secondary hypothesis is that preparing reusable patterns can reduce repeated reasoning work without increasing unsupported claims. Both hypotheses remain to be tested. Correctly abstaining from a weak pattern can be a better outcome than producing a novel-sounding explanation.

The proposed contribution is an evaluated lifecycle: experience and outcome records → scoped evidence snapshot → host-generated conditional hypothesis → explicit unverified proposal → current-source checks → continued review or withdrawal. The research question concerns the quality and revisability of the resulting advice, rather than merely whether the system can produce reflection text.

## Positioning and novelty boundaries

Prior work already establishes important parts of this space:

| Work | Relevant established direction | Proposed comparison question |
| --- | --- | --- |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | Stores natural-language experiences and synthesizes higher-level reflections for behavior and planning. | Does explicit premise-state checking improve the usefulness and withdrawal of reflections? |
| [Reflexion](https://arxiv.org/abs/2303.11366) | Uses linguistic task feedback and episodic reflective memory to improve subsequent trials without weight updates. | Does distinguishing expected outcomes from observed results improve transfer beyond repeating a task? |
| [A-MEM](https://arxiv.org/abs/2502.12110) | Builds linked structured notes and evolves memory representations as new memories arrive. | What benefit comes specifically from pinned provenance, explicit proposal status, and controlled source revision? |
| [HippoRAG 2](https://arxiv.org/abs/2502.14802) | Studies factual, associative, and sense-making memory through graph-based retrieval and passage integration. | Can revision-aware synthesis add value after retrieval quality is held constant? |
| [Kumiho's first paper](https://arxiv.org/abs/2603.17244) | Presents versioned graph memory and a correspondence between belief-revision operations and graph semantics. | Does that infrastructure produce measurable improvements in useful, retractable cross-experience advice? |

This is a positioning hypothesis, not a verified novelty claim. The five papers above are a starting set, not an exhaustive related-work search. Before submission, examine belief-base maintenance, truth-maintenance systems, temporal knowledge graphs, reflective agent memory, and other systems that track evidence and retract derived claims. Do not describe Kumiho as the first or only AGM-based memory system, or the first system to infer reflections from experience, on the strength of this outline.

A potentially defensible contribution is narrower: a reproducible evaluation of **conditional cross-experience inference with explicit outcome evidence and revision-triggered withdrawal**, using controlled retrieval and inference budgets. Whether the combination or its evaluation is novel requires further review.

## Formal guarantees and semantic claims

The first paper frames its formal results over a deliberately restricted propositional representation of ground triples and belief-base operations, with stated limits on the postulates covered. Those results must not be extended silently to arbitrary natural-language implications. [Kumiho, Sections 7.1–7.7](https://arxiv.org/html/2603.17244v1)

For this study, separate three layers:

1. **Operational invariants:** exact revision identity, scope isolation, pinned append-only source snapshots and change detection, bounded output, proposal/evidence separation, and explicit handling of missing or stale sources. These are suitable for unit tests, state-machine tests, and narrowly specified proofs.
2. **Semantic judgments:** whether a source supports a hypothesis, whether an apparent contradiction is real, and whether a changed premise invalidates a recommendation. These require labeled examples and independent assessment.
3. **User value:** whether the advice supplies a helpful connection, improves a decision, or asks a useful clarifying question. These require human evaluation and, eventually, longitudinal observation.

A valid graph operation does not prove a natural-language hypothesis true. Two experience records do not prove independent corroboration. A user's acceptance of a suggestion does not prove its causal explanation. The current proposal store forces inferred/unverified status; no automatic belief promotion is part of the experiment.

## Experimental conditions

Use a preregistered within-question comparison with the same answer-model version, decoding settings, user question, eligible source pool, and total online input/output token budget. Freeze extraction and judge prompts before the held-out evaluation. Record exact corpus revisions and software commit hashes.

| Condition | Available mechanism |
| --- | --- |
| Recall-only | The same scoped source experiences in ordinary retrieved form; no prepared review prompts or stored reflections. |
| Reflection-only | Host-generated reflection text under the same total budget; no revision-state retirement mechanism. |
| AGM-state-aware | Recall plus pinned source lineage and explicit current/stale/contested state; no structured outcome-pattern lifecycle. |
| Outcomes + patterns | State-aware recall plus separate expected/observed outcomes, conditional pattern proposals, and their source-health review. |

The mechanism labels describe these experimental variants, not claims that the baseline papers implement none of the excluded features. Faithful external baseline implementations should be evaluated separately if feasible.

Run two evaluation tracks. **Fixed-retrieval** gives each condition the same eligible retrieved records to isolate synthesis and revision handling. **End-to-end** includes each system's retrieval and reports retrieval coverage separately, so better source access cannot be mistaken for better inference. Never improve one condition by silently supplying more facts, context tokens, or answer-model calls.

Useful additional ablations remove one component at a time: observed outcomes, applicability conditions, provenance identity checks, state propagation, or prepared patterns. A shuffled-state negative control can test whether the benefit follows the correct evidence state rather than merely an instruction to be cautious. Keep these distinct from the four primary comparisons to limit multiple-testing ambiguity.

## Controlled premise-revision tasks

Construct versioned scenarios with a pre-change query, a controlled update, and a post-change query. Label both the correct supporting evidence and the warranted response before evaluating models. Split related episodes and paraphrases by scenario family so one experience does not leak across train/pilot/test partitions.

Include:

- Genuine repeated outcomes across distinct events, with limited applicability conditions.
- Different revisions or repeated reports about one event, which must not count as additional experiences.
- Distinct events derived from a shared report, where provenance is correlated.
- A premise that changes and invalidates a prior recommendation, and an equally visible change that does not affect it.
- An outcome that contradicts the expected result, mixed outcomes, and outcomes that remain unknown.
- An explicit disagreement that should remain unresolved, a missing source, and a retrieval failure.
- A proposal that was never accepted, an old accepted decision, and a later reversal.
- A simple factual question for which adding an insight would be distracting.

For semantic retraction labels, source supersession alone is not enough. Annotators must state why the changed proposition is material to the hypothesis. Missing evidence should usually yield uncertainty, not a claim that the hypothesis has been refuted.

## Measures and analysis plan

Primary outcomes should jointly measure useful inference and harmful overreach:

| Measure | Proposed operationalization |
| --- | --- |
| Grounded usefulness | Blind human rubric for a correct, relevant connection beyond restating retrieved text; report a distribution and paired preference, not only an average. |
| Unsupported-claim rate | Fraction of substantive claims unsupported by cited evidence or explicitly stated assumptions, with uncertainty and independence mistakes counted separately. |
| Retirement accuracy | Precision/recall for withdrawing or qualifying a hypothesis when its material premise changes, plus false-retirement rate on irrelevant updates. |
| Provenance fidelity | Exact citation validity, source-to-claim support, and whether duplicated/shared sources are incorrectly presented as independent evidence. |
| Appropriate abstention | Rate of direct answers or uncertainty where evidence does not justify a useful pattern, alongside missed-useful-insight rate. |
| Efficiency | Median/tail latency, graph reads, host tokens, output tokens, and offline preparation cost amortized over repeated questions. |

Use at least two blinded evaluators for the human sample, randomize answer order, measure agreement, and adjudicate disagreements without exposing the condition labels. Model judges may assist triage but should not replace an independent human subset. Do not use the proposing model's self-rated confidence as an outcome.

Estimate uncertainty with paired, scenario-clustered resampling; preserve related pre/post-change examples in the same cluster. Choose the held-out sample size from pilot variance and a stated practically meaningful effect, then freeze it before the main run. Report negative findings, all primary endpoints, and failures by scenario family. A gain in interestingness accompanied by more unsupported claims is not an unqualified success.

## What current evidence can establish

Offline tests can show that malformed candidates are rejected, multiple revisions do not manufacture recurrence, source changes block stale submissions, and generated hypotheses remain proposals. These are implementation properties. They do not measure answer quality, natural-language entailment, causal learning, or improvement over the research baselines.

A small purposively selected live-memory pilot can check feasibility, source availability, failure modes, and whether a rubric captures the intended user experience. It cannot estimate population-level insight accuracy or substantiate broad superiority. Report selection criteria, all attempted cases, missing-data cases, the exact intervention, and evaluator involvement. Never turn a few compelling examples into a benchmark percentage.

Keyless implementation means no additional provider call is made inside the memory lifecycle. It does not mean reasoning is free: host synthesis still uses computation and tokens. Also disclose source-read/store race windows, best-effort graph-edge persistence, duplicate submissions, and the difference between available state markers and complete dependency maintenance.

## Proposed paper structure and completion gates

A future manuscript could contain: motivation and failure cases; related work; experience/outcome and proposal model; revision-aware lifecycle; narrowly scoped operational properties; experimental protocol; results and error analysis; limitations and reproducibility artifacts.

Before drafting a results section, complete the related-work audit, preregister the controlled comparison, freeze a held-out scenario set, run matched-budget baselines, obtain independent judgments, and publish reproducible aggregate evidence with privacy-safe fixtures. Until then, retain this file as a research plan and leave performance and novelty claims open.
