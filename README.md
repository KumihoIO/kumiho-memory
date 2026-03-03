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

### Roadmap

* `0.1.x` — Experimental preview (current)
* `0.2.x` — Stabilized client APIs
* `1.0.0` — Production-ready client SDK

The scope of this package will remain limited to **client-side concerns**.

---

### License

MIT
