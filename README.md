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

---

### Roadmap

* `0.1.x` — Experimental preview (current)
* `0.2.x` — Stabilized client APIs
* `1.0.0` — Production-ready client SDK

The scope of this package will remain limited to **client-side concerns**.

---

### License

MIT
