# Ontology Design Principles — Gap Analysis & Plan

Kumiho's typed memory graph, audited against Gruber's ontology design
principles ("Toward Principles for the Design of Ontologies Used for
Knowledge Sharing", 1993): **clarity, coherence, extendibility, minimal
encoding bias, minimal ontological commitment** — plus the framing that an
ontology is a *shared semantic contract* agents commit to, not a database
schema.

Every gap below was verified against the code (file:line) and survived an
adversarial refutation pass on 2026-07-17. One original claim ("no
contradiction detection anywhere") was refuted and is restated accurately
in G3.

## Where Kumiho already aligns

- **Concept vs. representation split** — the Item/Revision model is exactly
  Gruber's `Document` vs. `Reference` distinction: an item is the identity,
  revisions are dated representations. This is a structural strength the
  typed layer under-uses (see G6).
- **Minimal commitment** — six node kinds (`entity, fact, decision, event,
  action, question`), a small structural edge set. No speculative kinds.
- **Precision-aware valid time** — `event_date` accepts `YYYY`, `YYYY-MM`,
  or `YYYY-MM-DD` (`memory_manager.py:109`), and recall-time comparison
  pads rather than coercing (`recall_rerank.py:161`) — the "1993 vs. March
  1993" example done right, at point (not interval) granularity.
- **Evidence levels with documented criteria** — `evidence.py` defines
  `official / corroborated / single_source / unverified` with explicit,
  non-circular criteria. This is the clarity bar the rest of the
  vocabulary should meet.
- **Conservative ABOUT linking** — token-run mention matching
  (`ontology.py:62`) prefers a missed edge over a wrong edge.

## Verified gaps

### G1. Open write vocabulary, closed read vocabulary (Clarity, Commitment) — CRITICAL

Agent-supplied relation predicates normalize to *any* ALL-CAPS token
(`ontology.py:446-454`) — `uses` and `utilizes` become distinct edge types;
the server gate is purely syntactic (`rust/src/edge.rs:17`). Meanwhile the
recall reader traverses a fixed whitelist (`graph_augmentation.py:54-57`:
DERIVED_FROM, DEPENDS_ON, REFERENCED, CONTAINS, CREATED_FROM, SUPERSEDES,
SUPPORTS, plus special-cased ABOUT/INVOLVES). **Entity→entity relation
edges are written but never traversed by any recall path** (only the
dashboard renders them). This is Gruber's failure mode verbatim: agents
share syntax, not meaning — and here the writer and reader of the *same
system* don't share a vocabulary.

### G2. No explicit, fetchable specification (Explicit specification, Commitment)

`OntologySchema` is a code-level dataclass (`ontology.py:81-107`); its own
docstring promises persistence as a policy Item "later". There is no MCP
surface from which another agent (OpenClaw, a future MCP client) can fetch
node-kind definitions and edge semantics to commit to. Ontological
commitment is currently "read the Python source".

### G3. Coherence is metadata-deep, not graph-deep (Coherence)

Contradiction detection *does* exist in the Level-of-Evidence path:
`assessors.py:311-407` consumes LLM `contradicts` verdicts, stores
`conflicts_with` metadata, and demotes conflicted beliefs. But in the
typed fact graph there is **no CONTRADICTS edge type anywhere**, and
SUPERSEDES fires on token-Jaccard ≥ 0.6 with newest-wins direction and no
semantic conflict check (`relations.py:26,151-155`). Consequences: a
paraphrased contradiction (low lexical overlap) coexists silently; a
*complementary* high-overlap fact can be wrongly marked superseded; the
conflict knowledge the assessor already extracts never becomes graph
structure the reader can traverse.

### G4. No downstream reassessment on belief change (Coherence; paper §15.6)

`DEPENDS_ON` edges ground decisions in facts, but superseding a fact
triggers nothing: the only re-grade pass (`graph_maintenance.py:541-588`)
recomputes a decision's grade from its own evidence atoms and never
inspects `DEPENDS_ON`. The Kumiho paper deferred this (§15.6); Atlas — the
Graphiti-fork that implements Kumiho's AGM spec — shipped it as "Ripple".
Our own spec's most-cited deferred feature is now a competitor's headline.

### G5. Write-time alias blindness (Encoding bias: concept ≠ name)

Alias resolution lives in a per-call dict (`ontology.py:486,505-510`);
nothing consults existing hubs' `aliases` metadata at write time. A new
session mentioning "PostgreSQL" duplicates an existing "Postgres" hub;
only the later Dream State merge (deterministic alias rule or LLM pass)
reconciles. Entity identity degrades to "surface string used this
session".

### G6. Typed-node identity = text hash; revision machinery unused (Encoding bias)

A fact's identity is `slugify(statement)` (`ontology.py:269`) — the concept
is conflated with one surface representation, Gruber's `(3, meter)` vs.
`(300, centimetre)` problem. Typed nodes are anchored at one revision and
never revised (`ontology.py:162-164`; no `create_revision` on hubs in
maintenance passes): belief updates always mint a new item + SUPERSEDES.
The Item/Revision machinery that solves exactly this sits unused one layer
below.

### G7. Three parallel trust vocabularies, no mapping (Clarity)

`certainty` (low/med/high, on facts — written at `summarization.py:1155`,
never read), `evidence_level` (evidence.py), and `confidence`
(code_capture.py) coexist with no defined correspondence in code or docs.

### G8. Valid time is a point, not an interval (Encoding bias)

Facts carry no `valid_from/valid_until`; `event_date` is a single date.
The bitemporal claim-graph blueprint (atomic claims + structured valid
intervals + date-aware timeline) already designs this and stays gated
behind the full-LoCoMo bar.

## Plan

Ordering follows the additive principle: every phase is opt-in or
lossless, pair-measured on LoCoMo F1 + LoCoMo-Plus before default-ON, and
checked against the typed-node vector-crowding constraint (typed
embeddings must not crowd vector k=10).

### Phase 1 — Specification & vocabulary (low risk, high leverage)

1. **Relation predicate registry** (G1 write side). Canonical predicate
   set with synonym folding (`uses/utilizes/relies_on → USES/DEPENDS_ON`…);
   unregistered predicates map to `RELATES_TO` with the verbatim predicate
   preserved in edge metadata — lossless and monotonic. Registry ships as
   data (see item 2), not code.
2. **Ontology spec as a versioned policy Item** (G2). Persist the schema —
   node kinds with natural-language definitions + constraints, edge types
   with semantics and direction, the predicate registry, and the trust-
   vocabulary mapping (item 3) — as a tagged revision (e.g.
   `ontology.spec`), seeded at onboarding like skill ingestion. Any agent
   can fetch it (`kumiho_get_revision_by_tag`) and commit to it; DreamState
   can diff observed usage against it. This discharges the docstring's
   promise and gives multi-agent deployments a real commitment artifact.
3. **Trust vocabulary mapping** (G7). One documented mapping
   (`certainty ↔ evidence_level ↔ confidence`) in the spec item + a parse
   helper. No stored-data migration.
4. **Reader traversal of registered relation edges** (G1 read side).
   Behind a flag; measure recall latency and context dilution before
   default-ON.

### Phase 2 — Coherence & reassessment

5. **Explicit belief-change input** (G3). `kumiho_memory_decompose` accepts
   agent-supplied `supersedes` and `contradicts` (same in-loop trust model
   as reflect/code_capture) instead of relying solely on Jaccard;
   heuristic SUPERSEDES keeps working as fallback but records its basis in
   edge metadata (`basis: lexical-overlap` vs `basis: agent`).
6. **CONTRADICTS as a first-class edge** (G3). New registered edge type,
   added to the reader whitelist; bridge the assessor's existing
   `conflicts_with` metadata into edges so recall can surface "this fact
   is contested" instead of silently returning one side.
7. **Grounding-staleness ripple** (G4). On SUPERSEDES landing on fact F,
   flag `DEPENDS_ON` dependents (`grounding_stale=true` + tag);
   recall surfaces the flag; DreamState re-grades or clears it. Keyless,
   deterministic core; closes paper §15.6 before Atlas's Ripple becomes
   the reference implementation of our own spec.

### Phase 3 — Identity & time (measured, gated)

8. **Write-time alias resolution** (G5). Bounded lookup of existing entity
   hubs (slug + aliases index) before creating a new hub; falls back to
   current behavior on miss. Shrinks DreamState's LLM-merge dependency.
9. **Embedding-assisted fact dedup** (G6). DreamState fact-dedup gains an
   embedding-similarity candidate stage (existing embedding infra) so
   paraphrase duplicates converge without loosening the unrecoverable-merge
   safety threshold. Semantic identity approached at maintenance time,
   not by changing write-time identity.
10. **Valid-time intervals + as-of recall** (G8). `valid_from/valid_until`
    on facts per the bitemporal claim-graph blueprint; stays behind the
    full-LoCoMo gate as decided.

### Non-goals

- No new node kinds (minimal commitment holds; the checklist question "does
  this distinction have an operational reason?" gates any addition).
- No migration/rewrite of stored data — every change is additive on write
  and tolerant on read.
- No renaming of existing edge types (extendibility: monotonic only).
