# -*- coding: utf-8 -*-
"""Valid-time intervals + as-of recall (ontology G8, kumiho_memory.valid_time).

Covers the interval grammar (partial-precision padding), the exclusion
semantics (open-ended, absent, boundary), the opt-in as-of demotion, and — the
load-bearing guarantee — that with the flag OFF the recall list is byte-
identical (a strict no-op, no marker, same object).
"""
import types
from datetime import date, datetime, timezone

import pytest

from kumiho_memory import valid_time as vt
from kumiho_memory.memory_manager import UniversalMemoryManager


# --------------------------------------------------------------------------- #
# normalize_valid_date                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,expected", [
    ("2020", "2020"),
    ("2020-03", "2020-03"),
    ("2020-03-15", "2020-03-15"),
    ("  2020-03-15  ", "2020-03-15"),   # trimmed
    ("last week", ""),                  # prose rejected
    ("2020-03-15T00:00:00Z", ""),       # timestamp rejected
    ("2020/03/15", ""),                 # wrong separator rejected
    ("", ""),
    (None, ""),
    (2020, ""),                          # non-str rejected
])
def test_normalize_valid_date(value, expected):
    assert vt.normalize_valid_date(value) == expected


# --------------------------------------------------------------------------- #
# interval_of — partial precision pads to whole periods                       #
# --------------------------------------------------------------------------- #

def test_interval_of_year_pads_to_whole_year():
    lower, upper = vt.interval_of({"valid_from": "2020", "valid_to": "2020"})
    assert lower == date(2020, 1, 1)
    assert upper == date(2020, 12, 31)


def test_interval_of_month_pads_to_whole_month_including_leap():
    lower, upper = vt.interval_of({"valid_from": "2020-02", "valid_to": "2020-02"})
    assert lower == date(2020, 2, 1)
    assert upper == date(2020, 2, 29)   # 2020 is a leap year


def test_interval_of_absent_is_open_on_both_ends():
    assert vt.interval_of({}) == (None, None)
    assert vt.interval_of(None) == (None, None)
    assert vt.interval_of({"valid_from": "garbage"}) == (None, None)


# --------------------------------------------------------------------------- #
# interval_excludes — the core semantics                                      #
# --------------------------------------------------------------------------- #

def test_within_closed_interval_is_included():
    meta = {"valid_from": "2019-01-01", "valid_to": "2022-12-31"}
    assert vt.interval_excludes(meta, date(2020, 6, 1)) is False


def test_before_valid_from_is_excluded():
    meta = {"valid_from": "2021-01-01"}
    assert vt.interval_excludes(meta, date(2020, 6, 1)) is True


def test_after_valid_to_is_excluded():
    meta = {"valid_to": "2019-12-31"}
    assert vt.interval_excludes(meta, date(2020, 6, 1)) is True


def test_open_ended_upper_is_still_valid_after_start():
    # only valid_from → valid forever after it
    meta = {"valid_from": "2018-01-01"}
    assert vt.interval_excludes(meta, date(2030, 1, 1)) is False


def test_open_ended_lower_is_valid_before_end():
    meta = {"valid_to": "2030-01-01"}
    assert vt.interval_excludes(meta, date(2000, 1, 1)) is False


def test_absent_interval_is_never_excluded():
    assert vt.interval_excludes({}, date(2020, 6, 1)) is False
    assert vt.interval_excludes({"title": "x"}, date(2020, 6, 1)) is False


def test_boundaries_are_inclusive():
    meta = {"valid_from": "2020-01-01", "valid_to": "2020-12-31"}
    assert vt.interval_excludes(meta, date(2020, 1, 1)) is False   # == lower
    assert vt.interval_excludes(meta, date(2020, 12, 31)) is False  # == upper


def test_year_upper_bound_covers_whole_year():
    # valid_to "2020" must not exclude a mid-2020 query (end-of-period padding)
    meta = {"valid_from": "2020", "valid_to": "2020"}
    assert vt.interval_excludes(meta, date(2020, 6, 1)) is False
    assert vt.interval_excludes(meta, date(2021, 1, 1)) is True
    assert vt.interval_excludes(meta, date(2019, 12, 31)) is True


# --------------------------------------------------------------------------- #
# as_of_recall_enabled                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("nope", False),
])
def test_as_of_recall_enabled(raw, expected):
    assert vt.as_of_recall_enabled({vt.AS_OF_RECALL_ENV: raw}) is expected


def test_as_of_recall_enabled_missing_key():
    assert vt.as_of_recall_enabled({}) is False


# --------------------------------------------------------------------------- #
# apply_valid_interval_marker                                                 #
# --------------------------------------------------------------------------- #

def test_marker_surfaces_valid_bounds():
    entry = {}
    vt.apply_valid_interval_marker(entry, {"valid_from": "2020", "valid_to": "2021-06"})
    assert entry == {"valid_from": "2020", "valid_to": "2021-06"}


def test_marker_drops_invalid_and_absent():
    entry = {}
    vt.apply_valid_interval_marker(entry, {"valid_from": "last year"})
    assert entry == {}
    vt.apply_valid_interval_marker(entry, {})
    assert entry == {}
    vt.apply_valid_interval_marker(entry, None)
    assert entry == {}


# --------------------------------------------------------------------------- #
# apply_as_of_recall — demotion + the byte-identical OFF guarantee            #
# --------------------------------------------------------------------------- #

def _mems():
    return [
        {"kref": "a", "valid_from": "2019", "valid_to": "2019"},   # lapsed by 2021
        {"kref": "b", "valid_from": "2020"},                        # open, valid
        {"kref": "c"},                                              # no interval
        {"kref": "d", "valid_from": "2025"},                        # not yet valid
    ]


def test_flag_off_is_byte_identical_noop():
    mems = _mems()
    snapshot = [dict(m) for m in mems]
    out = vt.apply_as_of_recall(mems, datetime(2021, 1, 1, tzinfo=timezone.utc),
                                enabled=False)
    assert out is mems                       # same object
    assert mems == snapshot                  # no key mutated, no reorder


def test_enabled_but_no_as_of_is_noop():
    mems = _mems()
    snapshot = [dict(m) for m in mems]
    out = vt.apply_as_of_recall(mems, None, enabled=True)
    assert out is mems
    assert mems == snapshot


def test_enabled_no_exclusions_leaves_list_untouched():
    # A query date inside every interval excludes nothing → no marker, no reorder.
    mems = [
        {"kref": "b", "valid_from": "2020"},
        {"kref": "c"},
    ]
    snapshot = [dict(m) for m in mems]
    out = vt.apply_as_of_recall(mems, date(2021, 1, 1), enabled=True)
    assert out is mems
    assert mems == snapshot
    assert all("as_of_excluded" not in m for m in mems)


def test_excluded_facts_are_stably_demoted_and_marked():
    mems = _mems()
    vt.apply_as_of_recall(mems, date(2021, 1, 1), enabled=True)
    order = [m["kref"] for m in mems]
    # included (b, c) keep order at the front; excluded (a lapsed, d pending)
    # keep order at the back.
    assert order == ["b", "c", "a", "d"]
    assert mems[0].get("as_of_excluded") is None and mems[1].get("as_of_excluded") is None
    assert mems[2]["as_of_excluded"] is True     # a
    assert mems[3]["as_of_excluded"] is True     # d


def test_datetime_as_of_uses_its_date():
    mems = [{"kref": "d", "valid_from": "2025"}]
    vt.apply_as_of_recall(mems, datetime(2021, 5, 1, 12, 30, tzinfo=timezone.utc),
                          enabled=True)
    assert mems[0]["as_of_excluded"] is True


def test_empty_list_is_safe():
    assert vt.apply_as_of_recall([], date(2021, 1, 1), enabled=True) == []


# --------------------------------------------------------------------------- #
# Manager wiring: _apply_as_of_recall honors the per-instance flag            #
# (unbound-method call keeps this cheap — no full MemoryManager construction) #
# --------------------------------------------------------------------------- #

def test_manager_apply_as_of_off_is_noop():
    fake = types.SimpleNamespace(as_of_recall_enabled=False)
    mems = _mems()
    snapshot = [dict(m) for m in mems]
    out = UniversalMemoryManager._apply_as_of_recall(
        fake, mems, datetime(2021, 1, 1, tzinfo=timezone.utc)
    )
    assert out is mems
    assert mems == snapshot


def test_manager_apply_as_of_on_demotes():
    fake = types.SimpleNamespace(as_of_recall_enabled=True)
    mems = _mems()
    UniversalMemoryManager._apply_as_of_recall(
        fake, mems, datetime(2021, 1, 1, tzinfo=timezone.utc)
    )
    assert [m["kref"] for m in mems] == ["b", "c", "a", "d"]
