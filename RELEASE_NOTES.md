# Release Notes — kumiho-memory

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
