"""Canonical relation-predicate registry (ontology G1, write side).

Agent-supplied relation predicates are an OPEN vocabulary: ``uses``,
``utilizes`` and ``leverages`` each normalize to a distinct edge type, so the
writer and the recall reader never share meaning (Gruber: shared syntax is not
shared semantics). This registry folds synonyms onto a small CLOSED set of
canonical edge types, and routes everything unregistered — including predicates
that normalize to nothing (CJK, punctuation) — to ``RELATES_TO`` instead of
dropping the relation.

Lossless and monotonic: the caller preserves the verbatim predicate (and the
normalized token, when folded/fallen back) in edge metadata, and folding only
ADDS structure — no canonical is ever renamed.

Folding is defined on the SAME normalization the writer already applies
(``ontology._predicate_edge_type``), so ``relies on``, ``Relies-On`` and
``RELIES_ON`` fold identically.

The registry is plain data (see :func:`registry_as_dict`) so the ontology spec
Item can serialize it without importing edge-write code. Immutable by
convention: ``_REGISTRY`` is a module constant; accessors return copies.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

from .ontology import _predicate_edge_type

#: Universal fallback. Every predicate that is neither a canonical type nor a
#: registered synonym folds here — the resolve never returns None, so a relation
#: is never dropped for an unrecognized predicate.
RELATES_TO = "RELATES_TO"

#: canonical edge type -> synonym predicates that fold onto it. Synonyms are
#: written in readable lower form and normalized on load. Each must be
#: unambiguous (one synonym -> one canonical) and must not shadow a canonical;
#: both invariants are enforced at import by :func:`_build_synonym_index`.
_REGISTRY: Dict[str, Tuple[str, ...]] = {
    "DEPENDS_ON": ("relies_on", "requires", "needs", "based_on",
                   "depends_upon", "contingent_on", "predicated_on"),
    "USES": ("utilizes", "employs", "leverages", "consumes", "calls", "invokes"),
    "IMPLEMENTS": ("realizes", "fulfills", "satisfies", "conforms_to"),
    "PART_OF": ("belongs_to", "member_of", "component_of", "subset_of",
                "contained_in"),
    "SUPERSEDES": ("replaces", "obsoletes", "deprecates", "supplants", "overrides"),
    "SUPPORTS": ("corroborates", "confirms", "substantiates", "validates",
                 "reinforces"),
    "CAUSES": ("leads_to", "results_in", "triggers", "induces", "brings_about"),
    "CONTRADICTS": ("conflicts_with", "refutes", "contravenes", "disproves",
                    "negates", "opposes"),
    "CONTAINS": ("includes", "comprises", "has_part", "has_component",
                 "encompasses"),
    RELATES_TO: ("related_to", "associated_with", "linked_to", "connected_to",
                 "pertains_to", "concerns"),
}


def _normalize(predicate: str) -> str:
    """The writer's ALL-CAPS/underscore normalization; ``""`` when nothing
    normalizes (CJK, punctuation, digit-leading). Folding is defined on this
    token so case/spacing/separator variants collapse before lookup."""
    return _predicate_edge_type(predicate) or ""


_CANONICAL_TYPES: Tuple[str, ...] = tuple(_REGISTRY)
_CANONICAL_SET = frozenset(_CANONICAL_TYPES)


def _build_synonym_index() -> Dict[str, str]:
    """normalized synonym token -> canonical. Raises on a synonym that
    normalizes to nothing, shadows a canonical, or folds onto two canonicals —
    the registry must be self-consistent (checked here + in the test suite)."""
    index: Dict[str, str] = {}
    for canonical, synonyms in _REGISTRY.items():
        for synonym in synonyms:
            tok = _normalize(synonym)
            if not tok:
                raise ValueError(f"synonym {synonym!r} normalizes to nothing")
            if tok in _CANONICAL_SET:
                raise ValueError(f"synonym {synonym!r} shadows canonical {tok}")
            existing = index.get(tok)
            if existing is not None and existing != canonical:
                raise ValueError(
                    f"synonym {tok} maps to both {existing} and {canonical}")
            index[tok] = canonical
    return index


_SYNONYM_TO_CANONICAL: Dict[str, str] = _build_synonym_index()


class Resolution(NamedTuple):
    """Fold outcome carried alongside the canonical edge type.

    ``normalized`` is the ALL-CAPS token (``""`` when the predicate normalizes
    to nothing). ``folded`` marks a registered synonym rewritten onto a
    different canonical; ``fallback`` marks an unregistered/unnormalizable
    predicate routed to ``RELATES_TO``. Both false => the predicate was already
    a canonical type (passthrough).
    """

    normalized: str
    folded: bool
    fallback: bool


def resolve_predicate(predicate: str) -> Tuple[str, Resolution]:
    """Resolve *predicate* to ``(canonical_edge_type, Resolution)``.

    A predicate that normalizes to a canonical type or a registered synonym
    returns the canonical; anything else returns ``RELATES_TO``. Never None — a
    relation is never dropped for an unrecognized predicate.
    """
    tok = _normalize(predicate)
    if tok in _CANONICAL_SET:
        return tok, Resolution(tok, False, False)
    canonical = _SYNONYM_TO_CANONICAL.get(tok)
    if canonical is not None:
        return canonical, Resolution(tok, True, False)
    return RELATES_TO, Resolution(tok, False, True)


def canonical_types() -> Tuple[str, ...]:
    """The canonical edge types, in declaration order."""
    return _CANONICAL_TYPES


def registry_as_dict() -> Dict[str, Tuple[str, ...]]:
    """Deep copy of the registry (canonical -> synonyms) as plain data for the
    ontology spec Item to serialize. Callers must not mutate ``_REGISTRY``."""
    return {canonical: tuple(synonyms) for canonical, synonyms in _REGISTRY.items()}
