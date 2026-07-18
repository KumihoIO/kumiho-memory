# Kumiho Memory

---

## Client-side memory provider for AI agents — working memory, consolidation, and belief revision on the Kumiho Cognitive Memory graph

---

### Status


**Privacy invariant:** every write path — session mining, commit mining,
skill ingestion — and every LLM-bound packet passes the same per-atom
PII/credential boundary. Credential-bearing atoms are dropped, never stored.
> **Beta (0.19.x)**
> Actively developed; the public API is stabilizing but may still change
> between minor versions.
> Latest release: `0.19.0` (2026-07-18) — **Ontology Phase 2**: first-class
> CONTRADICTS with contested markers, grounding-staleness ripple,
> basis-labeled belief edges, one privacy boundary on every write path, and
> the deterministic traversal contract.
> Previous: `0.18.0` (2026-07-17) — **Ontology Phase 1**: canonical
> relation predicate registry, a fetchable ontology spec policy Item, the
> trust-vocabulary mapping, and opt-in relation traversal at recall
> (`KUMIHO_MEMORY_RELATION_TRAVERSAL=1`).
> See `RELEASE_NOTES.md` for the full version-by-version history — this
> status block names only the latest release.
> Earlier highlights: `0.16` Dream State graph maintenance · `0.11-0.13`
> **Decision Memory** (git-anchored why-layer, `KUMIHO_MEMORY_CODE=1`) ·
> `0.10.1` moved the cross-encoder rerank off the event
> loop (pure perf; recall byte-identical) · `0.10.0` made the **write-time
> ontology** the default — typed decomposition into facts/entities/
> decisions/events plus entity-bridge and fact-recall legs (+0.042 overall /
> +0.054 fact leg on paired LoCoMo evidence) · `0.9.0` consolidated the
> full cognitive-recall pipeline into the SDK (graph-augmented recall,
> hybrid sibling ranking, `compose_context`).
> See [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for the full history.

---

### What this package is

`kumiho-memory` is a **client-side memory provider**: it buffers working
memory in Redis, consolidates conversations into the Kumiho Cognitive
Memory graph, and layers belief-revision policy (evidence grading,
corroboration, deployment-controlled deprecation) on top — all without
requiring changes to the Kumiho server.

It is designed to be:

* Lightweight
* Model-agnostic
* Framework-agnostic
* Safe to use in local or sandboxed environments

### What this package is NOT

`kumiho-memory` is a client of the Kumiho graph, not the graph itself:

* ❌ It does not implement the graph store, hybrid search, or edge
  storage — that's the Kumiho server (`kumiho-server`), reached through
  the core `kumiho` SDK.
* ❌ It does not run its own consolidation server — `DreamState` and
  `SpaceProfiler` are library/CLI/MCP-invoked passes over the graph, not
  a standing service.

Within those bounds, this package **does** implement working-memory
buffering, LLM-based consolidation, offline consolidation (Dream State),
graph-augmented recall, and evidence-aware belief revision — all
client-side, no server changes required.

---

### Features

* **Working memory** — Redis-backed session buffering
  (`RedisMemoryBuffer`), proxy/auth-resilient.
* **Consolidation** — LLM summarization + PII redaction into versioned
  graph revisions (`UniversalMemoryManager.consolidate_session`).
* **Dream State** — offline consolidation pass: relevance assessment,
  deprecation (capped, published-protected), tag/metadata enrichment,
  relationship discovery. Accepts deployment policy via
  `extra_instructions`.
* **Write-time ontology** *(0.10, on by default)* — every consolidation
  is decomposed into a typed knowledge graph (entities / facts /
  decisions / events / actions / questions) with deterministic edges;
  recall consumes the structure via an entity-bridge join and a
  fact-recall leg, both strictly **additive** (structural evidence never
  displaces conversation evidence). Opt out with
  `KUMIHO_MEMORY_ONTOLOGY=0`.
* **Decision Memory** *(0.11, opt-in)* — a second, code-domain profile:
  mine git commits into decision nodes with rationale, verbatim evidence
  atoms, and `{repo, commit, file, line}` anchors, then ask
  ``why("why is this file like this?", file=...)`` mid-session. See the
  section below.
* **Graph-augmented recall** — multi-query reformulation + edge
  traversal + semantic fallback (`GraphAugmentedRecall`).
* **Sibling revision filtering** — BM25-light or embedding-based
  filtering of stacked-item history.
* **Auto-assessment** — background write-time screening
  (`create_llm_assessor`) with a heuristic pre-filter and graph novelty
  check before any LLM call.
* **Level-of-Evidence belief revision** — memories carry an evidence
  grade (`official` / `corroborated` / `single_source` / `unverified`);
  a corroboration-aware assessor grades claims automatically, recall
  reranks and badges by grade, and Dream State respects grade-aware
  deployment policy. See below for details.
* **Space profiles** — per-Space churn/evidence/stability signals
  classify each Space (`canonical` / `working` / `correspondence`) so
  extraction strategy can adapt per collection.
* **Skill ingest** — parse and version `SKILL.md` files and reference
  docs into the graph (`kumiho-memory ingest-skill`).
* **MCP tools** — 13 tool wrappers (15 with Decision Memory enabled),
  auto-discovered by the core `kumiho` MCP server (see table below).

---

### Typical use cases

* Production memory backend for AI agents and MCP-compatible runtimes
* Multi-agent and collaborative AI systems that need shared, versioned
  long-term memory
* Applications that must weigh conflicting information by source
  credibility (news, claims, multi-source corroboration)

---

### Installation

```bash
pip install kumiho-memory
```

---

### Minimal example

```python
from kumiho_memory import RedisMemoryBuffer

memory = RedisMemoryBuffer()

memory.add_message(
    project="example",
    session_id="demo-session",
    role="user",
    content="Hello!"
)
```

> This example demonstrates **temporary, short-term buffering only**.
> It does not represent long-term memory persistence.

---

### Architectural note

`kumiho-memory` is one component within a larger, model-agnostic memory
architecture. Persistent storage, hybrid search, and edge/relationship
storage live in the Kumiho server (`kumiho-server`), reached through the
core `kumiho` SDK — this package never talks to the server directly.

Working-memory buffering, consolidation, offline lifecycle management
(Dream State, SpaceProfiler), and belief-revision policy are all
implemented **client-side, in this package**, calling the server only
through the standard SDK operations (create/read revisions, tags,
metadata, edges). No server changes are required for any feature in this
package, including the Level-of-Evidence subsystem below.

---

### Evidence levels

Memories can carry an **evidence grade** describing how trustworthy they
are. The grade is stored in two mirrored places:

| carrier | key / format | why |
|---|---|---|
| revision metadata | `evidence_level` (+ optional `source`, `confidence`) | canonical value, machine-readable |
| graph tag | `evidence:<level>` | tags get server-side time-range history → point-in-time audits |

When the carriers diverge (tag application is best-effort per-tag), the
**metadata value wins** — `parse_evidence(meta, tags)` implements this.

**Levels** (most → least trustworthy):

| level | meaning |
|---|---|
| `official` | explicit operator/ingest flag — never LLM-inferred; SHOULD be paired with the `published` tag so Dream State's deprecation protection applies |
| `corroborated` | ≥ N independent agreeing sources, none contradicting |
| `single_source` | identified source, no corroboration |
| `unverified` | everything else |

**Promotion / demotion state machine:**

- `unverified → single_source` — first stored occurrence with an identified `source`
- `single_source → corroborated` — an assessor finds ≥ N independent agreeing memories, none contradicting
- `* → official` — only via explicit flag (`evidence_level="official"` on ingest/consolidate/CLI), never LLM-inferred
- demotion — only via Dream State policy or explicit API, never silently at write time

**Usage:**

```python
from kumiho_memory import UniversalMemoryManager, evidence_tag, parse_evidence

manager = UniversalMemoryManager()

# Grade at ingest time — stashed in session metadata, applied at consolidation
await manager.ingest_message(
    user_id="u1",
    message="Acme announced record earnings.",
    evidence_level="official",
    source="press-release:acme",
)

# ...or explicitly at consolidation (overrides the ingest-time grade)
await manager.consolidate_session(
    session_id=session_id,
    evidence_level="corroborated",
    source="news:reuters",
)

# Recall results expose the grade when present
results = await manager.recall_memories("acme earnings")
results[0].get("evidence_level")  # "official"
```

CLI: `kumiho-memory ingest-skill doc.md --evidence-level official`

Grades are **only stamped when provided** — memories stored without an
evidence level keep their existing metadata and tag set unchanged, and
`parse_evidence` returns `None` for them (callers may treat that as
`unverified` via `DEFAULT_EVIDENCE_LEVEL`).

#### Evidence assessor (automatic grading)

`create_evidence_assessor` plugs into the write-time screening seat
(`UniversalMemoryManager(auto_assess_fn=...)`) and grades incoming
claims automatically:

| rule | condition | outcome |
|---|---|---|
| official pinning | claim contradicts a memory tagged `evidence:official` | stored `unverified`, conflict recorded in `conflicts_with`; the pinned belief is never revised |
| corroboration | ≥ N agreeing memories with **distinct** `source`s, none contradicting | `corroborated`, `memory_type` forced to `fact`, optional `SUPPORTS` edges to corroborators |
| single source | claim has an identified source, no corroboration | `single_source` |
| default | — | `unverified` |

The assessor **never emits `official`** — that grade stays operator-only.
Corroboration counting needs `source` metadata on the recalled memories,
so it only fires once sources are being written (see the schema section).

The bare `published` tag deliberately does **not** trigger pinning by
default — this codebase stamps `published` on virtually every stored
revision as its currency tag. Deployments that use `published` as a
curated marker can opt in:
`EvidencePolicy(official_tags=frozenset({"evidence:official", "published"}))`.

```python
from kumiho_memory import EvidencePolicy, create_evidence_assessor

assessor = create_evidence_assessor(
    adapter,
    policy=EvidencePolicy(min_corroboration=2, create_supports_edges=True),
)
manager = UniversalMemoryManager(auto_assess_fn=assessor)
```

MCP env wiring: `KUMIHO_EVIDENCE_ASSESSOR=1` (takes precedence over
`KUMIHO_AUTO_ASSESS` when both are set), `KUMIHO_EVIDENCE_MIN_CORROBORATION`
(default 2), `KUMIHO_EVIDENCE_SUPPORTS_EDGES=1` for evidence-chain edges.
`SUPPORTS` edges are followed by graph-augmented recall (included in the
default `GraphAugmentationConfig.edge_types`).

#### Dream State deployment policy

Dream State's assessment prompt accepts deployment-specific policy via
`extra_instructions` — appended under a fenced `## DEPLOYMENT POLICY`
section. Three injection routes (precedence: explicit arg > env var;
pass `""` to explicitly disable the env policy):

```python
DreamState(extra_instructions="Never propose deprecation for memories "
                              "tagged evidence:official. Prefer deprecating "
                              "unverified duplicates over corroborated ones.")
```

```bash
kumiho-memory dream --policy "Never propose deprecation for memories tagged evidence:official."
export KUMIHO_DREAM_EXTRA_INSTRUCTIONS="..."   # fallback when no arg given
```

The MCP tool `kumiho_memory_dream_state` accepts the same text via its
`extra_instructions` argument. Each memory in the assessment payload
carries its `evidence_level` and policy-relevant `revision_tags`
(`published`, `evidence:*`) so the policy has data to act on.

**Hard guardrails are not overridable by policy** — they apply in code
*after* the LLM's suggestions: the `max_deprecation_ratio` cap,
published-revision protection (`allow_published_deprecation=False`), and
the conservative-KEEP rule (the core prompt states it takes precedence
over deployment policy). Run results and the Markdown report record the
active policy text for auditability.

#### Evidence-weighted recall (reranking + badges)

Server-side hybrid search ranks by relevance only — a rumor can outrank
an official statement. `kumiho-memory` adjusts scores client-side with a
deterministic delta per grade (no extra LLM calls, O(k)):

| grade | default delta |
|---|---|
| `official` | **+0.15** |
| `corroborated` | +0.08 |
| `single_source` | 0.0 |
| `unverified` | **−0.10** |

**Before/after example** — query returns `rumor (0.60, unverified)` and
`statement (0.50, official)`: unweighted order is `rumor, statement`;
weighted order is `statement (0.65), rumor (0.50)`.

Applied in both plain recall and graph-augmented recall (before each
result cap, idempotently — the original score is kept in the documented `base_score` result field).
**Default ON**, with a strict no-op guarantee: when no retrieved memory
carries a grade, results are byte-identical to previous behavior.
Kill switch: `KUMIHO_EVIDENCE_RERANK=0`. Library use:
`UniversalMemoryManager(evidence_rank=EvidenceRankConfig(...))`.

**Context badges** — `build_recalled_context` (used by
`kumiho_memory_engage`) prefixes graded memories so the answering model
can weigh sources: `[official] Acme Q2: record earnings...`,
`[unverified] Forum post: ...`. `single_source`/ungraded memories get no
badge. `kumiho_memory_recall` returns raw dicts — there the grade
surfaces as the `evidence_level` field instead of a text badge.

Note: `min_score` filtering (`KUMIHO_MEMORY_MIN_RELEVANCE_SCORE`)
applies to the **adjusted** score — an `unverified` memory sitting just
above the threshold can drop below it. That is the intended screening
behavior; use the `base_score` result field if you need the raw retrieval score.

#### Space profiles (per-collection extraction strategy)

A collection's observed dynamics are a signal about what kind of
knowledge lives in it. `SpaceProfiler` aggregates per-Space statistics
from existing SDK queries (pure aggregation, **no LLM**):

| signal | source |
|---|---|
| churn | revisions per item, revision rate in the window (`latest` tag-move proxy), SUPERSEDES chain depth |
| evidence histogram | `evidence_level` metadata per revision |
| deprecation ratio | `deprecated` flags on items/revisions |
| stability | published share, median revision age |

…and classifies each Space:

| label | meaning | thresholds |
|---|---|---|
| `canonical` | established concepts | stability ≥ 0.6, churn ≤ 0.4, and evidence ≥ 0.3 when any revision carries a grade (ungraded corpora are not penalized) |
| `correspondence` | claims / requests / responses | churn ≥ 0.6 and stability ≤ 0.4 |
| `working` | active projects/notes | everything else |

Stability and evidence describe the **live** (non-deprecated) revisions
only; churn counts historical stacking. Empty spaces are not classified
or persisted — "no data" is not a label.

The profile persists as a `kind="space-profile"` Item — one per Space,
one revision per run, with `SUPERSEDES` edges linking runs so profile
drift is itself a versioned chain. A Space owner pins the label with the
`space_class` Space attribute; the profiler then never relabels, and
instead reports pin/observation disagreement as drift (the observed
label is persisted alongside the pin as `observed_label`).

```bash
kumiho-memory profile --dry-run          # classify without persisting
kumiho-memory profile --window-days 14
```

MCP tool: `kumiho_memory_space_profile`. Read side for strategy
consumers (assessor / Dream State policy / recall):

```python
from kumiho_memory import get_space_profile

profile = get_space_profile("CognitiveMemory", "/CognitiveMemory/news")
if profile and profile.label == "correspondence":
    # e.g. store claims as events, never promote to fact
    ...
```

The extraction rule this enables: **in `correspondence` spaces, claims
are events, not facts** — store them attributed ("X claimed Y on DATE")
and raise corroboration thresholds, instead of promoting them into the
belief set. (Consumption hooks land with the assessor/Dream State/recall
pieces of the epic; the profiler + `get_space_profile` are the
foundation.)

Note: true `latest` tag-move counting is not possible client-side (the
SDK exposes point-in-time tag resolution, not tag-move events) —
revision-creation frequency is the documented proxy, valid because
`latest` moves on every `create_revision`.

---

### Decision Memory — the why-layer for a codebase (0.11, opt-in)

git is a lossless graph of *what/when/who*; it cannot hold the ***why***.
Decision Memory mines commits into typed **decision** nodes (title,
decision, rationale, the why-question they answer), **verbatim evidence
atoms** (measurements, review findings — quoted, never paraphrased), and
git anchors. Code is **never copied**: anchors are
`{repo, commit_hash, file, line_range}` pointers, and node identity is
sha-free (`title + author-date`), so rebases and squashes converge instead
of duplicating — the memory does not rot as history rewrites.

```bash
export KUMIHO_MEMORY_CODE=1            # opt-in gate (default off)
kumiho-memory code-ingest . --range HEAD~30..HEAD   # idempotent; re-runs cost zero LLM calls
```

```python
result = await manager.code_why(
    "why is rerank_async a single-worker executor?",
    file="kumiho_memory/recall_rerank.py", line=420,
)
# → decisions with rationale + evidence chains + superseded_by status,
#   plus an inject-ready markdown context block
```

Key properties:

* **Three query legs, lexicographic fusion** — a deterministic anchor leg
  (file → decisions, zero search), a semantic leg, and an evidence-bridge
  leg. Anchor *facts* always outrank cross-encoder *probabilities*.
* **Belief revision** — a reversed decision is linked with `SUPERSEDES`,
  demoted in ranking, and always carries `superseded_by`; an agent never
  receives a reversed decision as the answer without seeing its
  replacement.
* **Physical isolation** — code nodes live in a dedicated
  `{project}-code` kumiho project; conversation recall is untouched by
  construction (and by test).
* Design doc: [`docs/DECISION_MEMORY_DESIGN.md`](docs/DECISION_MEMORY_DESIGN.md).
  Live-verified on this repo's own history: *"why is the executor
  single-worker?"* answers with the actual offload commit and its
  concurrency measurement as evidence.

---

### MCP Tools

13 tool wrappers (15 with `KUMIHO_MEMORY_CODE=1`), auto-discovered by the
core `kumiho` MCP server:

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
| `kumiho_memory_engage` | Recall + build context in one call |
| `kumiho_memory_reflect` | Buffer response + store captures |
| `kumiho_memory_dream_state` | Run Dream State consolidation cycle |
| `kumiho_memory_space_profile` | Profile each Space's knowledge dynamics |
| `kumiho_code_why` | *(opt-in)* Why is this code the way it is? — anchored decisions + evidence |
| `kumiho_code_ingest` | *(opt-in)* Mine a git commit range into decision nodes (idempotent) |

---

### Modules

| Module | Public API |
| -------- | ------------ |
| `memory_manager` | `UniversalMemoryManager`, `AutoAssessFn`, `MemoryAssessResult`, `get_memory_space` |
| `redis_memory` | `RedisMemoryBuffer` |
| `summarization` | `MemorySummarizer`, `LLMAdapter`, `EmbeddingAdapter`, `OpenAICompatAdapter`, `OpenAICompatEmbeddingAdapter`, `AnthropicAdapter` |
| `privacy` | `PIIRedactor`, `CredentialDetectedError` |
| `retry` | `RetryQueue` |
| `dream_state` | `DreamState`, `MemoryAssessment`, `DreamStateStats` |
| `graph_augmentation` | `GraphAugmentedRecall`, `GraphAugmentationConfig` |
| `assessors` | `create_llm_assessor`, `create_evidence_assessor`, `EvidencePolicy`, `grade_evidence`, `heuristic_prefilter`, `DEFAULT_STORAGE_POLICY` |
| `evidence` | `evidence_tag`, `parse_evidence`, `OFFICIAL`, `CORROBORATED`, `SINGLE_SOURCE`, `UNVERIFIED`, `EVIDENCE_LEVELS`, `DEFAULT_EVIDENCE_LEVEL` |
| `evidence_rank` | `apply_evidence_weights`, `evidence_badge`, `EvidenceRankConfig` |
| `space_profiler` | `SpaceProfiler`, `SpaceProfile`, `SpaceSignals`, `get_space_profile`, `SPACE_CLASSES` |
| `skill_ingest` | `ingest_skill`, `ingest_file`, `ingest_batch`, `parse_skill` |
| `ontology` | write-time typed decomposition (facts/entities/decisions/events) |
| `relations` | deterministic edge derivation (`ABOUT`, `DEPENDS_ON`, `SUPERSEDES`) |
| `entity_promotion` | `EntityPromotionConfig` — entity anchor hubs |
| `context_compose` | `compose_context`, `collect_top_revisions`, `DEFAULT_CONTEXT_TOP_K` |
| `recall_rerank` | `RerankConfig`, `rerank`, `rerank_async`, `two_pass_rerank` |
| `code_decisions` | Decision Memory schema: `CodeMemoryConfig`, slugs, anchors |
| `code_capture` | `ingest_repo`, `IngestStats` — git commit mining |
| `code_query` | `why`, `compose_why_context` — the 3-leg why engine |
| `mcp_tools` | `MEMORY_TOOLS`, `MEMORY_TOOL_HANDLERS` |

---

### Roadmap

* `0.5.x` — Graph-augmented recall, sibling filtering, recall dedup
* `0.6.x` — Level-of-Evidence belief revision
* `0.9.x` — Cognitive-recall pipeline consolidated into the SDK
* `0.10.x` — Write-time ontology on by default (typed knowledge graph)
* `0.11.x` — Decision Memory: git-anchored code-decision domain (current)
* next — Decision Memory Phase 2: agent-session mining, conversation↔code
  bridges, editor-hook auto-capture (see issue #43 / kumiho-plugins#10)
* `1.0.0` — Stabilized public API, production-ready client SDK

The scope of this package will remain limited to **client-side concerns**
— no kumiho-server changes are required by anything on this roadmap.

---

### Acknowledgments

* **Hugh Kim** (author of *memory-bank*) — for the independent full-module
  deep-dive review of v0.18.0 (2026-07-17): it cross-validated this package's
  measurement discipline from the outside, sharpened the read-path roadmap,
  and drove the security & reliability workstream (#99–#109). The best kind
  of peer review — adversarial in method, generous in intent.

---

### License

MIT
