"""Static scan + quarantine gate for external skill content (issue #100).

Skills are an *agent-instruction supply chain*: text parsed out of an
external ``SKILL.md`` is stored in the shared graph and consumed by every
agent that recalls it.  Unscanned, a skill section can smuggle a
hidden-unicode payload, an injection directive ("ignore all previous
instructions"), or an embedded secret to every downstream agent.

This module is the gate.  It is **pure, deterministic, and keyless** — no
LLM, no network, no state.  :func:`scan_content` returns a structured
:class:`ScanVerdict` (``clean`` | ``flagged(reasons)``) that the ingest
path uses to decide whether the content earns the agent-consumable
markers (see :mod:`kumiho_memory.skill_ingest`).

Three detector families, all conservative and high-precision — a false
positive quarantines a *useful* skill, so every pattern is chosen to fire
on the attack shape, not on ordinary prose:

1. **Hidden / bidi unicode** — zero-width chars, bidi controls, and the
   Unicode Tags block (an ASCII-smuggling vector that never appears in
   legitimate prose).  Precision tradeoff: U+200C ZWNJ and U+200D ZWJ are
   deliberately NOT flagged — they are load-bearing in compound emoji
   (ZWJ sequences) and in Persian/Indic scripts (ZWNJ), so flagging them
   quarantines legitimate multilingual skills.  A steganographic payload
   built purely from ZWNJ/ZWJ therefore passes this detector; the
   remaining zero-widths (U+200B, U+2060..U+2064, mid-text U+FEFF) and
   the Tags block still catch the common smuggling shapes.  A single
   *leading* U+FEFF (a Windows-editor BOM) is tolerated; U+FEFF anywhere
   else is flagged.
2. **Injection heuristics** — a small set of instruction-hijack phrases.
   This is a *heuristic*, not a guarantee: it raises the cost of the
   obvious attacks, it does not prove content safe.
3. **Embedded secrets** — reuses
   :data:`kumiho_memory.privacy.PIIRedactor.CREDENTIAL_PATTERNS`.

The quarantine metadata/tag conventions used by the ingest path live here
too so producer and consumer agree on the canonical keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

from kumiho_memory.privacy import PIIRedactor

# ---------------------------------------------------------------------------
# Quarantine conventions (canonical metadata keys + mirrored tag)
# ---------------------------------------------------------------------------

#: Revision metadata key set to the string ``"true"`` on flagged content.
#: Metadata is the canonical carrier — the mirrored tag is best-effort.
QUARANTINE_META_KEY = "quarantined"

#: Revision metadata key holding the JSON-encoded list of scan reasons.
QUARANTINE_REASONS_KEY = "quarantine_reasons"

#: Mirrored graph tag applied to a quarantined revision (best-effort;
#: metadata wins on divergence, per the evidence.py convention).
QUARANTINE_TAG = "quarantine:flagged"


def is_quarantined(metadata: Optional[Mapping[str, object]]) -> bool:
    """Cheap, already-fetched-metadata check: is this revision quarantined?

    Consumers (skill recall/discovery) call this to skip flagged
    revisions.  Pure string comparison — no server call, no cost on the
    non-skill recall path.
    """
    return str((metadata or {}).get(QUARANTINE_META_KEY, "")).strip().lower() == "true"


# ---------------------------------------------------------------------------
# [1] Hidden / bidi unicode
# ---------------------------------------------------------------------------

# Codepoint ranges that are invisible or reorder visible text, as the issue
# specifies plus the Unicode Tags block (hidden-ASCII smuggling).  Expressed
# as integer (first, last) pairs and compiled with chr() so the source
# carries NO literal invisible characters — a security scanner must not
# itself hide bytes from human review.
_HIDDEN_UNICODE_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x200B, 0x200B),        # zero-width space
    # U+200C ZWNJ / U+200D ZWJ intentionally excluded — legitimate in
    # compound emoji and Persian/Indic scripts (see module docstring).
    (0x200E, 0x200F),        # LRM/RLM bidi marks
    (0x2060, 0x2064),        # word joiner + invisible math operators
    (0xFEFF, 0xFEFF),        # BOM / zero-width no-break space (mid-text only)
    (0x202A, 0x202E),        # bidi embeddings + overrides (LRE/RLE/PDF/LRO/RLO)
    (0x2066, 0x2069),        # bidi isolates (LRI/RLI/FSI/PDI)
    (0xE0000, 0xE007F),      # Unicode Tags block — ASCII smuggling vector
)

_HIDDEN_UNICODE_RE = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _HIDDEN_UNICODE_RANGES) + "]"
)

#: Cap on distinct hidden-codepoint reasons emitted (keeps metadata bounded).
_MAX_HIDDEN_REASONS = 8

_BOM = chr(0xFEFF)


def _scan_hidden_unicode(text: str) -> List[str]:
    # A single LEADING BOM is Windows-editor noise, not a payload — tolerate
    # it.  Any other U+FEFF occurrence (mid-text, or a second leading one)
    # still flags.
    if text.startswith(_BOM):
        text = text[1:]
    codepoints = sorted({ord(m.group()) for m in _HIDDEN_UNICODE_RE.finditer(text)})
    reasons = [f"hidden_unicode:U+{cp:04X}" for cp in codepoints[:_MAX_HIDDEN_REASONS]]
    if len(codepoints) > _MAX_HIDDEN_REASONS:
        reasons.append(f"hidden_unicode:+{len(codepoints) - _MAX_HIDDEN_REASONS}_more")
    return reasons


# ---------------------------------------------------------------------------
# [2] Injection heuristics (small, high-precision, order-stable)
# ---------------------------------------------------------------------------

# Each entry fires on the *attack shape*.  Documented as heuristic: passing
# this scan is not proof of safety, only that the obvious hijacks are absent.
_INJECTION_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "ignore_previous",
        re.compile(
            r"ignore\s+(?:all\s+)?(?:the\s+)?"
            r"(?:previous|prior|above|earlier|preceding)\s+"
            r"(?:instruction|rule|prompt|direction|context|message)s?",
            re.IGNORECASE,
        ),
    ),
    (
        "disregard_system",
        re.compile(
            r"disregard\s+(?:all\s+)?(?:your|the|any|previous|prior|above)\s+"
            r"(?:system|previous|prior|instruction|rule|prompt|guideline)s?",
            re.IGNORECASE,
        ),
    ),
    (
        # Narrow (issue #100 review F1): skills legitimately address agents in
        # second person ("you are now the reviewer", "you are now in the
        # planning phase"), so a bare article/role tail over-quarantines.
        # Require an explicit jailbreak/override tail.
        "persona_override",
        re.compile(
            r"you\s+are\s+now\s+(?:a\s+|an\s+|the\s+|in\s+)?"
            r"(?:dan\b|jailbroken|unrestricted|uncensored|"
            r"developer\s+mode|free\s+of|no\s+longer\s+bound|"
            r"acting\s+outside)",
            re.IGNORECASE,
        ),
    ),
    (
        "suppress_disclosure",
        re.compile(
            r"(?:do\s*not|don't|never)\s+"
            r"(?:tell|reveal|mention|disclose|inform|show|report)\s+"
            r"(?:this\s+)?(?:to\s+)?(?:the\s+)?"
            r"(?:user|operator|human|anyone|them)",
            re.IGNORECASE,
        ),
    ),
    (
        "hide_directive",
        re.compile(
            r"hide\s+this\s+(?:from|message|instruction|section|prompt|text)",
            re.IGNORECASE,
        ),
    ),
    (
        "exfil_http",
        re.compile(
            r"(?:send|forward|post|upload|exfiltrate|leak|email|transmit|deliver)\b"
            r"[^\n]{0,80}?\b(?:to|at|via)\s+https?://",
            re.IGNORECASE,
        ),
    ),
)

# base64 needs BOTH a long blob AND a decode directive — a lone base64 run
# (e.g. an embedded data URI or a hash) must not trip the gate on its own.
_B64_RUN_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_B64_DECODE_HINT_RE = re.compile(
    r"(?:base64\s*(?:-d|--?decode|_?decode)|b64decode|from_?base64|atob\s*\(|"
    r"decode\s+(?:this|the\s+following|it\b|and\s+(?:run|exec|execute)))",
    re.IGNORECASE,
)


def _scan_injection(text: str) -> List[str]:
    reasons = [name for name, rx in _INJECTION_PATTERNS if rx.search(text)]
    if _B64_RUN_RE.search(text) and _B64_DECODE_HINT_RE.search(text):
        reasons.append("base64_decode_directive")
    return [f"injection:{name}" for name in reasons]


# ---------------------------------------------------------------------------
# [3] Embedded secrets — reuse the privacy credential patterns
# ---------------------------------------------------------------------------


def _scan_credentials(text: str) -> List[str]:
    return [
        f"credential:{cred_type}"
        for cred_type, pattern in PIIRedactor.CREDENTIAL_PATTERNS.items()
        if re.search(pattern, text)
    ]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanVerdict:
    """Structured result of :func:`scan_content`.

    ``clean`` is ``True`` when no detector fired; otherwise ``reasons``
    holds order-stable, de-duplicated tags describing what matched
    (``hidden_unicode:U+200B``, ``injection:ignore_previous``,
    ``credential:api_key_generic``).
    """

    clean: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return not self.clean


def scan_content(text: str) -> ScanVerdict:
    """Scan skill *text* for hidden unicode, injection, and secrets.

    Pure and deterministic: the same input always yields the same
    verdict, reasons in a fixed order (hidden-unicode, then injection,
    then credential) with duplicates removed.
    """
    reasons: List[str] = []
    reasons.extend(_scan_hidden_unicode(text))
    reasons.extend(_scan_injection(text))
    reasons.extend(_scan_credentials(text))

    # De-dup while preserving first-seen order.
    seen: set = set()
    ordered: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            ordered.append(r)

    return ScanVerdict(clean=not ordered, reasons=ordered)


def summarize_reasons(reasons: Sequence[str]) -> str:
    """Compact one-line summary of scan reasons for CLI/log output."""
    return ", ".join(reasons) if reasons else "clean"
