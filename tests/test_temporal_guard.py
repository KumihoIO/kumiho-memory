"""Tests for event_date corroboration (issue #119).

``classify_event_date`` gates the LLM-extracted valid-time date against the
source transcript: literal presence -> ``verified``; a relative token that
resolves against the session timestamp -> ``derived``; otherwise ``unverified``
(kept as metadata, excluded from the ranking boost).
"""

from datetime import date, datetime, timezone

import pytest

from kumiho_memory.temporal_guard import (
    DERIVED,
    UNVERIFIED,
    VERIFIED,
    classify_event_date,
    parse_timestamp,
    _extract_source_dates,
    _parse_event_date,
    _shift_month,
    _iso_range,
)

# A fixed reference instant for the relative-derivation tests. 2026-07-18.
REF = datetime(2026, 7, 18, 9, 30, tzinfo=timezone.utc)


def c(event_date, source, ref=REF):
    return classify_event_date(event_date, source, ref)


# --------------------------------------------------------------------------
# parse_timestamp
# --------------------------------------------------------------------------

def test_parse_timestamp_passthrough_datetime_naive_becomes_utc():
    naive = datetime(2026, 7, 18, 1, 2, 3)
    out = parse_timestamp(naive)
    assert out.tzinfo is not None and out.utcoffset().total_seconds() == 0


def test_parse_timestamp_string_with_z_and_offset():
    assert parse_timestamp("2026-07-18T00:00:00Z").year == 2026
    assert parse_timestamp("2026-07-18T00:00:00+00:00").hour == 0
    # A non-UTC offset is normalized to UTC.
    assert parse_timestamp("2026-07-18T09:00:00+09:00").hour == 0


@pytest.mark.parametrize("bad", [None, "", "   ", "not-a-date", 12345, [], {}])
def test_parse_timestamp_rejects_garbage(bad):
    assert parse_timestamp(bad) is None


# --------------------------------------------------------------------------
# _parse_event_date / _iso_range / _shift_month
# --------------------------------------------------------------------------

def test_parse_event_date_precisions():
    assert _parse_event_date("2026") == (1, 2026, 0, 0)
    assert _parse_event_date("2026-07") == (2, 2026, 7, 0)
    assert _parse_event_date("2026-07-18") == (3, 2026, 7, 18)


@pytest.mark.parametrize("bad", ["", "last week", "2026-13", "2026-02-30", "26-07", 5, None])
def test_parse_event_date_rejects_bad(bad):
    assert _parse_event_date(bad) is None


def test_iso_range_spans():
    assert _iso_range((3, 2026, 7, 18)) == (date(2026, 7, 18), date(2026, 7, 18))
    assert _iso_range((2, 2026, 7, 0)) == (date(2026, 7, 1), date(2026, 7, 31))
    assert _iso_range((2, 2026, 2, 0)) == (date(2026, 2, 1), date(2026, 2, 28))
    assert _iso_range((1, 2026, 0, 0)) == (date(2026, 1, 1), date(2026, 12, 31))


def test_shift_month_wraps_years():
    assert _shift_month(2026, 7, -1) == (2026, 6)
    assert _shift_month(2026, 1, -1) == (2025, 12)
    assert _shift_month(2026, 12, 1) == (2027, 1)
    assert _shift_month(2026, 7, -7) == (2025, 12)


# --------------------------------------------------------------------------
# Absolute matching -> verified
# --------------------------------------------------------------------------

def test_iso_numeric_separators_all_verify():
    for sep in ("-", ".", "/"):
        src = f"the incident on 2026{sep}07{sep}18 was logged"
        assert c("2026-07-18", src) == VERIFIED
    # Single-digit month/day numeric form.
    assert c("2026-07-18", "happened 2026.7.18 sharp") == VERIFIED


def test_trailing_year_numeric_both_orderings():
    assert c("2026-07-18", "dated 07/18/2026 on the form") == VERIFIED  # m/d/y
    assert c("2026-07-18", "dated 18/07/2026 on the form") == VERIFIED  # d/m/y


def test_english_absolute_forms_verify():
    assert c("2026-07-18", "met on July 18, 2026 downtown") == VERIFIED
    assert c("2026-07-18", "met on July 18th 2026 downtown") == VERIFIED
    assert c("2026-07-18", "met on 18 July 2026") == VERIFIED
    assert c("2026-07-18", "met on the 18th of July, 2026") == VERIFIED
    assert c("2026-07-18", "met on Jul 18 for lunch") == VERIFIED  # year-less md
    assert c("2026-07-18", "the 18th of July still stands") == VERIFIED


def test_korean_absolute_forms_verify():
    assert c("2026-07-18", "우리는 2026년 7월 18일에 만났다") == VERIFIED
    assert c("2026-07-18", "우리는 7월 18일에 만났다") == VERIFIED  # year-less md


def test_precision_month_matches_month_and_day_mentions():
    # A month-precision row verifies against a bare month mention...
    assert c("2026-07", "back in July 2026 we began") == VERIFIED
    assert c("2026-07", "2026-07 kickoff") == VERIFIED
    # ...and against a fuller day mention of the same month (day-truncated).
    assert c("2026-07", "logged 2026-07-18 in the tracker") == VERIFIED
    assert c("2026-07", "우리는 2026년 7월 18일에 만났다") == VERIFIED


def test_precision_year_matches_bare_year_and_fuller_mentions():
    assert c("2026", "sometime in 2026 it started") == VERIFIED
    assert c("2026", "back in July 2026") == VERIFIED
    assert c("2026", "on 2026-07-18") == VERIFIED
    assert c("2026", "2026년에 시작했다") == VERIFIED


def test_day_precision_not_verified_by_month_only_mention():
    # Only the month is present; the specific day is not corroborated.
    assert c("2026-07-18", "back in July 2026 nothing specific") == UNVERIFIED


def test_bare_day_only_does_not_verify():
    # A lone day number ("18th" / "18일") is too weak to verify a full date.
    assert c("2026-07-18", "the 18th was a blur", ref=None) == UNVERIFIED
    assert c("2026-07-18", "18일은 정신없었다", ref=None) == UNVERIFIED


def test_hallucinated_wellformed_date_is_unverified():
    # The core bug: a plausible but absent date must NOT be trusted.
    src = "We talked about the vacation and the new job. No dates were mentioned."
    assert c("2026-03-15", src) == UNVERIFIED


def test_wrong_year_same_month_day_is_unverified_when_year_present():
    # Source pins 2024 explicitly; a 2026 day-precision row must not verify —
    # the (year, month, day) triple is absent and there is no year-less md form.
    assert c("2026-07-18", "the meeting was on 2024-07-18") == UNVERIFIED


def test_empty_source_is_unverified():
    assert c("2026-07-18", "") == UNVERIFIED
    assert c("2026-07-18", "   ") == UNVERIFIED


def test_malformed_event_date_is_unverified():
    assert classify_event_date("last week", "we met on 2026-07-18", REF) == UNVERIFIED
    assert classify_event_date("", "2026-07-18", REF) == UNVERIFIED


# --------------------------------------------------------------------------
# Relative derivation -> derived
# --------------------------------------------------------------------------

def test_yesterday_today_tomorrow_english_and_korean():
    assert c("2026-07-17", "that happened yesterday") == DERIVED
    assert c("2026-07-18", "we discussed it today") == DERIVED
    assert c("2026-07-19", "the launch is tomorrow") == DERIVED
    assert c("2026-07-17", "그건 어제 있었던 일이야") == DERIVED
    assert c("2026-07-18", "오늘 결정했어") == DERIVED
    assert c("2026-07-19", "내일 출시야") == DERIVED


def test_day_before_after_variants():
    assert c("2026-07-16", "the day before yesterday it broke") == DERIVED
    assert c("2026-07-16", "그저께 고장났어") == DERIVED
    assert c("2026-07-20", "the day after tomorrow we ship") == DERIVED
    assert c("2026-07-20", "모레 배포한다") == DERIVED


def test_n_days_ago_digit_and_word_and_korean():
    assert c("2026-07-15", "it was 3 days ago") == DERIVED
    assert c("2026-07-15", "it was three days ago") == DERIVED
    assert c("2026-07-15", "3일 전에 있었어") == DERIVED
    # future
    assert c("2026-07-23", "5 days from now we present") == DERIVED
    assert c("2026-07-23", "5일 후에 발표") == DERIVED


def test_n_weeks_ago_with_slack_and_korean():
    assert c("2026-07-04", "two weeks ago we launched") == DERIVED
    assert c("2026-07-04", "2주 전에 출시했어") == DERIVED
    assert c("2026-07-11", "a week ago we launched") == DERIVED


def test_n_months_and_years_ago():
    assert c("2026-04", "three months ago we moved") == DERIVED
    assert c("2026-04-10", "3개월 전에 이사했어") == DERIVED
    assert c("2024", "two years ago it started") == DERIVED
    assert c("2024", "2년 전에 시작했어") == DERIVED


def test_vague_week_month_year_windows():
    assert c("2026-07-13", "last week we started") == DERIVED
    assert c("2026-07-13", "지난주에 시작했어") == DERIVED
    assert c("2026-07-25", "next week we ship") == DERIVED
    assert c("2026-06", "last month was busy") == DERIVED
    assert c("2026-06", "지난달은 바빴어") == DERIVED
    assert c("2026-08", "next month we launch") == DERIVED
    assert c("2025", "last year we moved") == DERIVED
    assert c("2025", "작년에 이사했어") == DERIVED
    assert c("2027", "next year we expand") == DERIVED
    assert c("2026", "this year has been good") == DERIVED


def test_last_weekday_english_and_korean():
    # Pick an event 4 days before REF and name its weekday.
    ev = date(2026, 7, 14)  # a Tuesday, 4 days before REF (2026-07-18, Saturday)
    assert ev.weekday() == 1
    assert c("2026-07-14", "we met last Tuesday") == DERIVED
    assert c("2026-07-14", "지난 화요일에 만났어") == DERIVED
    # Wrong weekday name -> the token is present but the arithmetic fails.
    assert c("2026-07-14", "we met last Monday") == UNVERIFIED


def test_relative_inconsistent_date_is_unverified():
    # "yesterday" present but the date is nowhere near ref-1.
    assert c("2026-01-01", "that happened yesterday") == UNVERIFIED
    # "two weeks ago" but the date is months off.
    assert c("2026-01-01", "two weeks ago we launched") == UNVERIFIED


def test_relative_requires_reference_ts():
    # Without a reference timestamp the relative path is dormant.
    assert classify_event_date("2026-07-17", "that happened yesterday", None) == UNVERIFIED


def test_absolute_wins_over_relative():
    # Date is both literally present AND has a relative token -> verified (stronger).
    src = "yesterday, i.e. 2026-07-17, we shipped"
    assert c("2026-07-17", src) == VERIFIED


# --------------------------------------------------------------------------
# _extract_source_dates direct coverage
# --------------------------------------------------------------------------

def test_extract_source_dates_populates_all_sets():
    years, months, days, md = _extract_source_dates(
        "on 2026-07-18 and July 2025 and Jul 4 in 1999"
    )
    assert (2026, 7, 18) in days
    assert (2026, 7) in months and (2025, 7) in months
    assert 1999 in years and 2026 in years
    assert (7, 4) in md


def test_extract_source_dates_skips_invalid_calendar():
    _, _, days, _ = _extract_source_dates("bogus 2026-02-30 and 2026-13-01")
    assert not any(d[0] == 2026 for d in days)


def test_extract_source_dates_empty():
    assert _extract_source_dates("") == (set(), set(), set(), set())
    assert _extract_source_dates(None) == (set(), set(), set(), set())
