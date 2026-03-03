# Release Notes — kumiho-memory

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
