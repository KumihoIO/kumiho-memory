# Kumiho Memory

---

## Experimental client-side utilities for AI agent memory integration

---

### ⚠️ Status

> **Experimental / Preview (0.1.x)**
> This package is provided for early experimentation and reference usage.
> APIs and behavior may change without notice.
> Latest patch: `0.1.2` (2026-02-09) - README refresh and version metadata sync.

---

### What this package is

`kumiho-memory` provides **client-side utilities** that help AI agents
temporarily buffer interaction context and interface with the broader
Kumiho Cognitive Memory architecture.

It is designed to be:

* Lightweight
* Model-agnostic
* Framework-agnostic
* Safe to use in local or sandboxed environments

---

### What this package is NOT

To avoid confusion, this package **does NOT** implement:

* ❌ A full cognitive memory system
* ❌ Long-term memory graphs or lineage tracking
* ❌ Memory consolidation or offline processing
* ❌ Automated belief revision or pruning
* ❌ The "Dream State" consolidation pipeline

Those capabilities exist at the **system level** and are intentionally
decoupled from this client-side library.

---

### Design intent

This separation is intentional.

By keeping advanced memory logic outside the client library:

* Memory remains independent of any specific LLM
* Client environments stay fast and lightweight
* Sensitive or irreversible memory operations are centrally controlled
* The architecture remains portable across platforms and models

---

### Typical use cases

* Experimenting with memory-aware AI agents
* Prototyping agent workflows that require short-term context buffering
* Reference integration for platforms such as:

  * Multi-agent systems
  * Collaborative AI environments
  * MCP-compatible agent runtimes

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
architecture.

The full system includes:

* Client-side buffers (this package)
* Persistent memory storage
* Structured relationships between memories
* Offline consolidation and lifecycle management

This package intentionally exposes **only the client-side surface**.

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
result cap, idempotently — the original score is kept in `_base_score`).
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
behavior; use `_base_score` if you need the raw retrieval score.

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
| `canonical` | established concepts | stability ≥ 0.6 and churn ≤ 0.4 |
| `correspondence` | claims / requests / responses | churn ≥ 0.6 and stability ≤ 0.4 |
| `working` | active projects/notes | everything else |

The profile persists as a `kind="space-profile"` Item — one per Space,
one revision per run, with `SUPERSEDES` edges linking runs so profile
drift is itself a versioned chain. A Space owner pins the label with the
`space_class` Space attribute; the profiler then reports drift only.

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

### Roadmap

* `0.1.x` — Experimental preview (current)
* `0.2.x` — Stabilized client APIs
* `1.0.0` — Production-ready client SDK

The scope of this package will remain limited to **client-side concerns**.

---

### License

MIT
