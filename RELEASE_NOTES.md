# Release Notes — kumiho-memory

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
