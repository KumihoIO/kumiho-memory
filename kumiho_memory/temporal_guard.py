"""Corroborate LLM-extracted ``event_date`` values against their source text.

``event_date`` is a valid-time signal (prospective indexing): the summarizer
reads a conversation and emits a normalized ISO date for each event. Until now
the only gate was a precision-aware *format* regex (``YYYY`` / ``YYYY-MM`` /
``YYYY-MM-DD``) — a well-formed but **hallucinated** date passes it cleanly and
then pollutes the event-proximity ranking boost at recall (issue #119, from
Hugh Kim's review §2 caveat 3).

This module adds a *content* gate. :func:`classify_event_date` cross-checks the
extracted date against the actual transcript and returns one of three
confidence levels:

* ``"verified"`` — the date is literally present in the source text, in any of
  the common absolute formats (ISO, ``2026.7.18``, ``July 18 2026``, ``Jul 18``,
  ``2026년 7월 18일``, month-only, year-only, …), matched at the date's own
  precision.
* ``"derived"`` — the source contains a *relative* reference (``yesterday`` /
  ``어제`` / ``last Tuesday`` / ``지난주`` / ``two weeks ago`` …) and the extracted
  date is arithmetically consistent with the known session/message timestamp.
  This is the LLM doing legitimate date arithmetic, not fabricating.
* ``"unverified"`` — neither check passes. The date is kept as metadata but the
  recall layer excludes it from the event-proximity boost.

Design rules (mirrors the additive, back-compatible ethos of the rest of the
package):

* **Never over-verify.** When in doubt the answer is ``"unverified"`` — that
  only costs a ranking *boost*, which is opt-in and temporal-query-only, and
  degrades gracefully to today's undated behavior. Over-verifying, by contrast,
  would defeat the guard.
* **No LLM calls.** Pure deterministic string/date arithmetic.
* **Legacy rows are untouched.** This module only classifies *new* writes; rows
  stored before #119 carry no confidence key and the recall layer treats an
  absent key as trusted (see :mod:`kumiho_memory.recall_rerank`).
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Set, Tuple

__all__ = [
    "classify_event_date",
    "parse_timestamp",
    "VERIFIED",
    "DERIVED",
    "UNVERIFIED",
]

VERIFIED = "verified"
DERIVED = "derived"
UNVERIFIED = "unverified"

# Precision codes for a parsed event date.
_YEAR, _MONTH, _DAY = 1, 2, 3

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# A month name alternation for the English-date extractors. Longest-first so
# ``september`` wins over ``sep``.
_MONTH_ALT = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)"
)

_EVENT_DATE_RE = re.compile(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_timestamp(value: object) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (with or without ``Z``) into aware UTC.

    Used to turn a stored message ``timestamp`` into the reference instant for
    relative-date derivation. Returns ``None`` for anything unparseable so the
    caller simply skips the relative path.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_event_date(event_date: object) -> Optional[Tuple[int, int, int, int]]:
    """Parse ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD`` → ``(precision, y, m, d)``.

    Unused components are ``0``. Returns ``None`` for a malformed or
    calendar-invalid date.
    """
    if not isinstance(event_date, str):
        return None
    m = _EVENT_DATE_RE.match(event_date.strip())
    if not m:
        return None
    year = int(m.group(1))
    if m.group(3) is not None:
        month, day = int(m.group(2)), int(m.group(3))
        try:
            date(year, month, day)
        except ValueError:
            return None
        return (_DAY, year, month, day)
    if m.group(2) is not None:
        month = int(m.group(2))
        if not 1 <= month <= 12:
            return None
        return (_MONTH, year, month, 0)
    return (_YEAR, year, 0, 0)


def _iso_range(parsed: Tuple[int, int, int, int]) -> Tuple[date, date]:
    """The inclusive ``(start, end)`` calendar span a parsed date covers."""
    precision, year, month, day = parsed
    if precision == _DAY:
        d = date(year, month, day)
        return d, d
    if precision == _MONTH:
        return date(year, month, 1), date(year, month, monthrange(year, month)[1])
    return date(year, 1, 1), date(year, 12, 31)


# ---------------------------------------------------------------------------
# Absolute-date matching — is the date literally present in the source?
# ---------------------------------------------------------------------------

def _extract_source_dates(
    text: str,
) -> Tuple[Set[int], Set[Tuple[int, int]], Set[Tuple[int, int, int]], Set[Tuple[int, int]]]:
    """Pull every absolute date mention out of ``text``.

    Returns four sets, each normalized to integer components:

    * ``years``    — ``{year}`` (year-precision mentions and every fuller one)
    * ``months``   — ``{(year, month)}``
    * ``days``     — ``{(year, month, day)}``
    * ``md``       — ``{(month, day)}`` for year-less mentions (``Jul 18``,
      ``7월 18일``). The year is uncorroborated, so these only ever satisfy a
      day-precision event whose month+day match.

    Bare day-only mentions (``18일``, ``18th``) are deliberately NOT extracted:
    a lone day number is far too weak to verify a full date and would defeat
    the guard.
    """
    years: Set[int] = set()
    months: Set[Tuple[int, int]] = set()
    days: Set[Tuple[int, int, int]] = set()
    md: Set[Tuple[int, int]] = set()
    if not isinstance(text, str) or not text:
        return years, months, days, md
    low = text.lower()

    def add_day(y: int, m: int, d: int) -> None:
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return
        try:
            date(y, m, d)
        except ValueError:
            return
        days.add((y, m, d))
        months.add((y, m))
        years.add(y)
        # NB: a full (year-bearing) mention does NOT populate ``md`` — the
        # year-less set is reserved for genuinely year-less mentions so a
        # differently-dated full date can never masquerade as year-less
        # corroboration (see the conflict guard in _absolute_match).

    def add_month(y: int, m: int) -> None:
        if 1 <= m <= 12:
            months.add((y, m))
            years.add(y)

    def add_md(m: int, d: int) -> None:
        if 1 <= m <= 12 and 1 <= d <= 31:
            md.add((m, d))

    def add_year(y: int) -> None:
        if 1900 <= y <= 2099:
            years.add(y)

    # Numeric year-first: 2026-07-18, 2026.7.18, 2026/07/18.
    # Trailing guard is ``(?!\d)`` (not ``\b``): a ``\b`` fails when a Korean
    # particle is glued to the date (``2026-07-18에``), because Hangul is a word
    # char, so the day silently drops out and the date is wrongly unverified.
    # ``(?!\d)`` still forbids extending the number but tolerates any non-digit
    # follower (particle, punctuation, EOL).
    for mt in re.finditer(r"\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})(?!\d)", low):
        add_day(int(mt[1]), int(mt[2]), int(mt[3]))
    # Numeric trailing-year (ambiguous order): 07/18/2026 or 18/07/2026 — try
    # both interpretations; only the calendar-valid ones survive add_day.
    for mt in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)", low):
        a, b, y = int(mt[1]), int(mt[2]), int(mt[3])
        add_day(y, a, b)
        add_day(y, b, a)
    # Numeric year-month: 2026-07, 2026.7, 2026/07
    for mt in re.finditer(r"\b(\d{4})[./-](\d{1,2})(?!\d)", low):
        add_month(int(mt[1]), int(mt[2]))

    # English month-day-year: July 18 2026 / July 18th, 2026
    for mt in re.finditer(
        _MONTH_ALT + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", low
    ):
        add_day(int(mt[3]), _MONTHS[mt[1]], int(mt[2]))
    # English day-month-year: 18 July 2026 / 18th of July, 2026
    for mt in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_ALT + r",?\s+(\d{4})\b",
        low,
    ):
        add_day(int(mt[3]), _MONTHS[mt[2]], int(mt[1]))
    # English month-year: July 2026 / Jul, 2026
    for mt in re.finditer(_MONTH_ALT + r"\.?,?\s+(\d{4})\b", low):
        add_month(int(mt[2]), _MONTHS[mt[1]])
    # English month-day (no year): July 18 / Jul 18th
    for mt in re.finditer(_MONTH_ALT + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", low):
        add_md(_MONTHS[mt[1]], int(mt[2]))
    # English day-month (no year): 18 July / 18th of Jul
    for mt in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_ALT + r"\b", low
    ):
        add_md(_MONTHS[mt[2]], int(mt[1]))

    # Korean year-month-day: 2026년 7월 18일
    for mt in re.finditer(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", low):
        add_day(int(mt[1]), int(mt[2]), int(mt[3]))
    # Korean year-month: 2026년 7월
    for mt in re.finditer(r"(\d{4})\s*년\s*(\d{1,2})\s*월", low):
        add_month(int(mt[1]), int(mt[2]))
    # Korean month-day (no year): 7월 18일
    for mt in re.finditer(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", low):
        add_md(int(mt[1]), int(mt[2]))
    # Korean year: 2026년
    for mt in re.finditer(r"(\d{4})\s*년", low):
        add_year(int(mt[1]))

    # Bare 4-digit year in a plausible range — only ever verifies a
    # year-precision event (or corroborates the year of fuller mentions).
    for mt in re.finditer(r"\b(\d{4})\b", low):
        add_year(int(mt[1]))

    return years, months, days, md


def _absolute_match(parsed: Tuple[int, int, int, int], source_text: str) -> bool:
    """True when the parsed date appears literally in the source, at its precision."""
    precision, year, month, day = parsed
    years, months, days, md = _extract_source_dates(source_text)
    if precision == _DAY:
        if (year, month, day) in days:
            return True
        # Year-less month+day mention (Jul 18 / 7월 18일): month+day corroborated,
        # only the year is inferred — accepted per issue #119's format list, BUT
        # rejected if the source explicitly dates this month+day to a DIFFERENT
        # year (an explicit conflicting year outranks a year-less inference).
        if (month, day) in md and not any(
            mm == month and dd == day and yy != year for (yy, mm, dd) in days
        ):
            return True
        return False
    if precision == _MONTH:
        if (year, month) in months:
            return True
        return any(yy == year and mm == month for (yy, mm, _dd) in days)
    # Year precision.
    if year in years:
        return True
    if any(yy == year for (yy, _mm) in months):
        return True
    return any(yy == year for (yy, _mm, _dd) in days)


# ---------------------------------------------------------------------------
# Relative-date derivation — is the date consistent with a relative token?
# ---------------------------------------------------------------------------

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_KO_WEEKDAYS = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}

# Small spelled-out cardinals so "two weeks ago" / "a month ago" parse like
# their digit forms. ``a``/``an`` read as 1.
_WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}
_NUM = (
    r"(\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)"
)


def _num(token: str) -> int:
    """Digit or small spelled-out cardinal → int (0 if unrecognized)."""
    return int(token) if token.isdigit() else _WORD_NUM.get(token, 0)


def _shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    """Calendar month arithmetic. ``delta`` negative = earlier."""
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def _relative_hits(
    parsed: Tuple[int, int, int, int], low: str, ref: date
) -> bool:
    """True when a relative token in ``low`` resolves (arithmetically) to the date.

    ``low`` is the already-lowercased source; ``ref`` is the single reference
    anchor date the relative arithmetic is measured from.

    Windows are kept deliberately tight/bounded: a legitimate LLM resolution of
    ``yesterday`` / ``two weeks ago`` lands inside them, while a hallucinated
    far-off date does not. Vague tokens (``last week``) use a bounded multi-day
    window; calendar tokens (``last month``/``작년``) use exact calendar spans.
    """
    ev_start, ev_end = _iso_range(parsed)

    hits: List[bool] = []

    def day_window(lo: int, hi: int) -> None:
        """Event overlaps ``[ref - hi, ref - lo]`` (positive days = past)."""
        w_lo = ref - timedelta(days=hi)
        w_hi = ref - timedelta(days=lo)
        hits.append(not (ev_end < w_lo or ev_start > w_hi))

    def month_offset(delta: int) -> None:
        yy, mm = _shift_month(ref.year, ref.month, delta)
        m_start = date(yy, mm, 1)
        m_end = date(yy, mm, monthrange(yy, mm)[1])
        hits.append(not (ev_end < m_start or ev_start > m_end))

    def year_offset(delta: int) -> None:
        yy = ref.year + delta
        hits.append(not (ev_end < date(yy, 1, 1) or ev_start > date(yy, 12, 31)))

    # --- single-day anchors ---
    if re.search(r"\byesterday\b", low) or "어제" in low:
        day_window(1, 1)
    if re.search(r"\btoday\b", low) or "오늘" in low:
        day_window(0, 0)
    if re.search(r"\btomorrow\b", low) or "내일" in low:
        day_window(-1, -1)
    if re.search(r"\bday before yesterday\b", low) or "그저께" in low or "그제" in low:
        day_window(2, 2)
    if re.search(r"\bday after tomorrow\b", low) or "모레" in low:
        day_window(-2, -2)

    # --- N days ago / later ---
    for mt in re.finditer(r"\b" + _NUM + r"\s+days?\s+ago\b", low):
        n = _num(mt[1]); day_window(n, n)
    for mt in re.finditer(r"(\d{1,3})\s*일\s*전", low):
        n = int(mt[1]); day_window(n, n)
    for mt in re.finditer(r"\b" + _NUM + r"\s+days?\s+(?:later|from now|from today)\b", low):
        n = _num(mt[1]); day_window(-n, -n)
    for mt in re.finditer(r"(\d{1,3})\s*일\s*(?:후|뒤)", low):
        n = int(mt[1]); day_window(-n, -n)

    # --- N weeks ago / later (±3 day slack; the exact weekday is ambiguous) ---
    for mt in re.finditer(r"\b" + _NUM + r"\s+weeks?\s+ago\b", low):
        n = _num(mt[1]); day_window(7 * n - 3, 7 * n + 3)
    for mt in re.finditer(r"(\d{1,2})\s*주\s*전", low):
        n = int(mt[1]); day_window(7 * n - 3, 7 * n + 3)
    for mt in re.finditer(r"\b" + _NUM + r"\s+weeks?\s+(?:later|from now)\b", low):
        n = _num(mt[1]); day_window(-7 * n - 3, -7 * n + 3)
    for mt in re.finditer(r"(\d{1,2})\s*주\s*(?:후|뒤)", low):
        n = int(mt[1]); day_window(-7 * n - 3, -7 * n + 3)

    # --- N months / years ago / later ---
    for mt in re.finditer(r"\b" + _NUM + r"\s+months?\s+ago\b", low):
        month_offset(-_num(mt[1]))
    for mt in re.finditer(r"(\d{1,2})\s*(?:개월|달)\s*전", low):
        month_offset(-int(mt[1]))
    for mt in re.finditer(r"\b" + _NUM + r"\s+months?\s+(?:later|from now)\b", low):
        month_offset(_num(mt[1]))
    for mt in re.finditer(r"(\d{1,2})\s*(?:개월|달)\s*(?:후|뒤)", low):
        month_offset(int(mt[1]))
    for mt in re.finditer(r"\b" + _NUM + r"\s+years?\s+ago\b", low):
        year_offset(-_num(mt[1]))
    for mt in re.finditer(r"(\d{1,2})\s*년\s*전", low):
        year_offset(-int(mt[1]))
    for mt in re.finditer(r"\b" + _NUM + r"\s+years?\s+(?:later|from now)\b", low):
        year_offset(_num(mt[1]))
    for mt in re.finditer(r"(\d{1,2})\s*년\s*(?:후|뒤)", low):
        year_offset(int(mt[1]))

    # --- vague week/month/year references ---
    if re.search(r"\blast week\b", low) or "지난주" in low or "지난 주" in low \
            or "저번주" in low or "저번 주" in low:
        day_window(1, 14)
    if re.search(r"\bnext week\b", low) or "다음주" in low or "다음 주" in low:
        day_window(-14, -1)
    if re.search(r"\bthis week\b", low) or "이번주" in low or "이번 주" in low:
        day_window(-7, 7)
    if re.search(r"\blast month\b", low) or "지난달" in low or "저번달" in low:
        month_offset(-1)
    if re.search(r"\bnext month\b", low) or "다음달" in low:
        month_offset(1)
    if re.search(r"\bthis month\b", low) or "이번달" in low:
        month_offset(0)
    if re.search(r"\blast year\b", low) or "작년" in low or "지난해" in low \
            or "지난 해" in low:
        year_offset(-1)
    if re.search(r"\bnext year\b", low) or "내년" in low:
        year_offset(1)
    if re.search(r"\bthis year\b", low) or "올해" in low or "금년" in low:
        year_offset(0)

    # --- "last <weekday>" — only meaningful at day precision ---
    if parsed[0] == _DAY:
        wd = ev_start.weekday()
        for name, target in _WEEKDAYS.items():
            if target == wd and re.search(r"\blast " + name + r"\b", low):
                day_window(1, 13)
        for kname, target in _KO_WEEKDAYS.items():
            if target == wd and (
                ("지난 " + kname + "요일") in low or ("지난" + kname + "요일") in low
            ):
                day_window(1, 13)

    return any(hits)


def _relative_derivable(
    parsed: Tuple[int, int, int, int],
    source_text: str,
    reference_ts: Optional[datetime],
) -> bool:
    """True when a relative token resolves to the date against *any* known anchor.

    Relative arithmetic needs a reference instant. Two independent anchors are
    honored, and a hit against either yields ``derived``:

    * the session/message ``reference_ts`` (the live-conversation anchor), and
    * every absolute day-precision date literally present in ``source_text``.

    The second is essential for backfilled / historical / LoCoMo-style corpora:
    ``redis_memory.add_message`` always stamps the *wall-clock ingest* time, so
    ``reference_ts`` is the time the row was written, not the conversation time.
    The summarizer, however, resolves relative tokens against an *in-content*
    anchor (e.g. a ``[7 May 2023]`` message prefix — see the summarizer prompt),
    so a legitimately derived historical date (``yesterday`` → ``2023-05-06``) is
    consistent with the in-text anchor, not the ingest clock. Anchoring on the
    literal source dates recovers those dates instead of misclassifying them
    ``unverified`` and dropping their event-proximity boost (issue #119 (1)(b)).

    The relative *token* is still required, so a hallucinated date with no
    relative reference stays ``unverified``; the per-anchor windows stay tight.
    """
    if not isinstance(source_text, str) or not source_text:
        return False
    low = source_text.lower()

    refs: List[date] = []
    if reference_ts is not None:
        refs.append(reference_ts.date())
    # In-content absolute day anchors the summarizer may have resolved against.
    _years, _months, days, _md = _extract_source_dates(source_text)
    refs.extend(date(y, m, d) for (y, m, d) in sorted(days))
    if not refs:
        return False

    return any(_relative_hits(parsed, low, ref) for ref in refs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_event_date(
    event_date: str,
    source_text: str,
    reference_ts: Optional[datetime] = None,
) -> str:
    """Classify an extracted ``event_date`` as verified / derived / unverified.

    * ``verified``   — literally present in ``source_text`` at the date's precision.
    * ``derived``    — a relative token in ``source_text`` resolves to the date
      against a known anchor: the session/message ``reference_ts`` and/or any
      absolute date literally present in the source (the summarizer resolves
      relative tokens against in-content anchors, so those must be honored even
      when ``reference_ts`` is the wall-clock ingest time of a backfilled row).
      Yields ``unverified`` when neither ``reference_ts`` nor an in-content
      anchor is available.
    * ``unverified`` — neither; keep as metadata but exclude from ranking boost.

    Absolute matching wins over relative derivation (a literal date is the
    stronger signal). A malformed ``event_date`` (should not occur — callers
    pass a format-validated value) classifies as ``unverified``.
    """
    parsed = _parse_event_date(event_date)
    if parsed is None:
        return UNVERIFIED
    if _absolute_match(parsed, source_text):
        return VERIFIED
    if _relative_derivable(parsed, source_text, reference_ts):
        return DERIVED
    return UNVERIFIED
