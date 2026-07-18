"""Tests for kumiho_memory.failure_ledger — cross-run parking ledger (#118)."""

import json
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone

import pytest

from kumiho_memory.failure_ledger import (
    FailureLedger,
    content_key,
    default_failure_ledger,
)


def _now() -> datetime:
    return datetime(2026, 7, 18, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# content_key
# ---------------------------------------------------------------------------


def test_content_key_is_stable_and_order_sensitive():
    assert content_key("a", "b") == content_key("a", "b")
    assert content_key("a", "b") != content_key("b", "a")
    # Empty / falsy parts are dropped, not hashed as blanks.
    assert content_key("a", "", "b") == content_key("a", "b")


def test_content_key_hex_digest():
    key = content_key("hello")
    assert isinstance(key, str)
    assert len(key) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# Absent file → clean start
# ---------------------------------------------------------------------------


def test_absent_file_is_clean_start():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        # No file exists yet; every read is empty and side-effect-free.
        assert len(ledger) == 0
        assert ledger.is_parked("anything") is False
        assert ledger.get("anything") is None
        # Constructing + reading must NOT create the file.
        assert not (ledger._file).exists()


def test_construction_does_not_touch_filesystem():
    with tempfile.TemporaryDirectory() as tmp:
        sub = os.path.join(tmp, "does-not-exist-yet")
        FailureLedger(sub)
        # Directory is created lazily on first write only.
        assert not os.path.exists(sub)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_record_failure_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        entry = ledger.record_failure("k1", "transient", now=_now())
        assert entry["attempts"] == 1
        assert entry["last_error_class"] == "transient"
        assert entry["first_seen"] == _now().isoformat()
        assert entry["parked_at"] is None

        # A fresh ledger instance sees the persisted state.
        reopened = FailureLedger(tmp)
        got = reopened.get("k1")
        assert got is not None
        assert got["attempts"] == 1
        assert len(reopened) == 1


def test_attempts_accumulate_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        FailureLedger(tmp).record_failure("k", "unknown", now=_now())
        FailureLedger(tmp).record_failure("k", "unknown", now=_now())
        assert FailureLedger(tmp).get("k")["attempts"] == 2


def test_first_seen_preserved_last_seen_advances():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        t0 = _now()
        t1 = t0 + timedelta(hours=3)
        ledger.record_failure("k", "transient", now=t0)
        ledger.record_failure("k", "deterministic", now=t1)
        entry = ledger.get("k")
        assert entry["first_seen"] == t0.isoformat()
        assert entry["last_seen"] == t1.isoformat()
        assert entry["last_error_class"] == "deterministic"


# ---------------------------------------------------------------------------
# Parking rules
# ---------------------------------------------------------------------------


def test_deterministic_parks_at_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2)
        ledger.record_failure("k", "deterministic", now=_now())
        # One deterministic failure: not yet parked (below threshold).
        assert ledger.is_parked("k", now=_now()) is False
        ledger.record_failure("k", "deterministic", now=_now())
        # Second deterministic failure reaches the threshold → parked.
        assert ledger.is_parked("k", now=_now()) is True


def test_transient_never_parks():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2)
        for _ in range(5):
            ledger.record_failure("k", "transient", now=_now())
        assert ledger.is_parked("k", now=_now()) is False


def test_unknown_never_parks():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2)
        for _ in range(5):
            ledger.record_failure("k", "unknown", now=_now())
        assert ledger.is_parked("k", now=_now()) is False


def test_park_threshold_one_parks_immediately():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1)
        ledger.record_failure("k", "deterministic", now=_now())
        assert ledger.is_parked("k", now=_now()) is True


def test_mixed_then_deterministic_parks_on_total_attempts():
    """A transient attempt then a deterministic one reaches threshold=2."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2)
        ledger.record_failure("k", "transient", now=_now())
        ledger.record_failure("k", "deterministic", now=_now())
        assert ledger.is_parked("k", now=_now()) is True


# ---------------------------------------------------------------------------
# TTL un-park
# ---------------------------------------------------------------------------


def test_ttl_unpark():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1, park_ttl_days=14)
        t_park = _now()
        ledger.record_failure("k", "deterministic", now=t_park)
        assert ledger.is_parked("k", now=t_park) is True
        # Still parked one day before TTL.
        assert ledger.is_parked("k", now=t_park + timedelta(days=13)) is True
        # After TTL elapses, the item un-parks (selectable again).
        assert ledger.is_parked("k", now=t_park + timedelta(days=14, hours=1)) is False


def test_is_parked_is_pure_read_no_mutation():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1, park_ttl_days=1)
        t_park = _now()
        ledger.record_failure("k", "deterministic", now=t_park)
        # A TTL-expired is_parked returns False but leaves parked_at in place.
        assert ledger.is_parked("k", now=t_park + timedelta(days=5)) is False
        assert ledger.get("k")["parked_at"] == t_park.isoformat()


def test_deterministic_failure_after_ttl_reparks():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=2, park_ttl_days=14)
        t0 = _now()
        ledger.record_failure("k", "deterministic", now=t0)
        ledger.record_failure("k", "deterministic", now=t0)
        assert ledger.is_parked("k", now=t0) is True
        # After TTL, selectable again.
        later = t0 + timedelta(days=15)
        assert ledger.is_parked("k", now=later) is False
        # It re-fails deterministically → parked_at refreshes → re-parked.
        ledger.record_failure("k", "deterministic", now=later)
        assert ledger.is_parked("k", now=later) is True


def test_sweep_expired_unparks_and_resets():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1, park_ttl_days=14)
        t0 = _now()
        ledger.record_failure("stale", "deterministic", now=t0)
        ledger.record_failure("fresh", "deterministic", now=t0 + timedelta(days=20))
        # Sweep 30 days later: "stale" is expired, "fresh" (20d) still parked.
        swept = ledger.sweep_expired(now=t0 + timedelta(days=30))
        assert swept == 1
        assert ledger.get("stale")["parked_at"] is None
        assert ledger.get("stale")["attempts"] == 0
        assert ledger.get("fresh")["parked_at"] is not None


def test_sweep_expired_noop_when_nothing_expired():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1, park_ttl_days=14)
        ledger.record_failure("k", "deterministic", now=_now())
        assert ledger.sweep_expired(now=_now()) == 0


# ---------------------------------------------------------------------------
# record_success clears history
# ---------------------------------------------------------------------------


def test_record_success_clears_entry():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1)
        ledger.record_failure("k", "deterministic", now=_now())
        assert ledger.is_parked("k", now=_now()) is True
        ledger.record_success("k")
        assert ledger.get("k") is None
        assert ledger.is_parked("k", now=_now()) is False


def test_record_success_absent_key_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        ledger.record_success("never-seen")  # must not raise or create the file
        assert not ledger._file.exists()


# ---------------------------------------------------------------------------
# Corruption-safe load
# ---------------------------------------------------------------------------


def test_corrupt_garbage_file_starts_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        ledger._file.parent.mkdir(parents=True, exist_ok=True)
        ledger._file.write_text("{ this is not valid json ::::", encoding="utf-8")
        # Load must not raise; treats corruption as empty.
        assert len(ledger) == 0
        assert ledger.is_parked("k") is False
        # And a subsequent write recovers to a valid file.
        ledger.record_failure("k", "transient", now=_now())
        assert json.loads(ledger._file.read_text(encoding="utf-8"))["entries"]


def test_truncated_file_starts_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        ledger._file.parent.mkdir(parents=True, exist_ok=True)
        # A half-written JSON object (simulating a crash mid-write).
        ledger._file.write_text('{"version": 1, "entries": {"k": {"att', encoding="utf-8")
        assert len(ledger) == 0


def test_wrong_shape_starts_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        ledger._file.parent.mkdir(parents=True, exist_ok=True)
        # Valid JSON but not our schema (a list instead of an object).
        ledger._file.write_text("[1, 2, 3]", encoding="utf-8")
        assert len(ledger) == 0


def test_wrong_version_starts_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        ledger._file.parent.mkdir(parents=True, exist_ok=True)
        ledger._file.write_text(
            json.dumps({"version": 999, "entries": {"k": {"attempts": 3}}}),
            encoding="utf-8",
        )
        assert len(ledger) == 0


def test_entries_not_dict_starts_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        ledger._file.parent.mkdir(parents=True, exist_ok=True)
        ledger._file.write_text(
            json.dumps({"version": 1, "entries": ["nope"]}), encoding="utf-8"
        )
        assert len(ledger) == 0


# ---------------------------------------------------------------------------
# Atomic write / concurrency
# ---------------------------------------------------------------------------


def test_write_is_atomic_no_temp_left_behind():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        ledger.record_failure("k", "transient", now=_now())
        leftovers = [p for p in os.listdir(tmp) if p.endswith(".tmp")]
        assert leftovers == []
        # Exactly one ledger file.
        assert os.path.isfile(ledger._file)


def test_concurrent_writers_do_not_corrupt():
    """Many threads recording distinct keys must never corrupt the file."""
    with tempfile.TemporaryDirectory() as tmp:
        n_threads = 8
        per_thread = 20
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker(tid: int) -> None:
            barrier.wait()
            try:
                led = FailureLedger(tmp)
                for i in range(per_thread):
                    led.record_failure(f"t{tid}-k{i}", "deterministic", now=_now())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # The file must always be valid JSON with our shape (atomic replace).
        raw = FailureLedger(tmp)._file.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed["version"] == 1
        assert isinstance(parsed["entries"], dict)
        # The per-file lock serializes in-process writers, so every distinct
        # key survives (no lost updates, no corruption).
        assert len(parsed["entries"]) == n_threads * per_thread
        # No stray temp files.
        assert [p for p in os.listdir(tmp) if p.endswith(".tmp")] == []


# ---------------------------------------------------------------------------
# Bounded size / pruning
# ---------------------------------------------------------------------------


def test_pruning_caps_entries_and_keeps_parked():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1, max_entries=3)
        base = _now()
        # One parked (deterministic) entry, seen earliest.
        ledger.record_failure("parked-old", "deterministic", now=base)
        # Several non-parked entries seen later.
        for i in range(5):
            ledger.record_failure(
                f"transient-{i}", "transient", now=base + timedelta(minutes=i + 1)
            )
        # Capped to max_entries.
        assert len(ledger) == 3
        # The parked entry is retained despite being the oldest.
        assert ledger.is_parked("parked-old", now=base) is True
        # The most-recent transient entries survive; the oldest were pruned.
        assert ledger.get("transient-4") is not None
        assert ledger.get("transient-0") is None


def test_pruning_drops_oldest_parked_when_all_parked():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=1, max_entries=2)
        base = _now()
        for i in range(4):
            ledger.record_failure(
                f"p-{i}", "deterministic", now=base + timedelta(minutes=i)
            )
        assert len(ledger) == 2
        # Oldest parked dropped, newest retained.
        assert ledger.get("p-0") is None
        assert ledger.get("p-3") is not None


# ---------------------------------------------------------------------------
# Env configuration + default factory
# ---------------------------------------------------------------------------


def test_env_overrides(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("KUMIHO_FAILURE_LEDGER_DIR", tmp)
        monkeypatch.setenv("KUMIHO_FAILURE_PARK_THRESHOLD", "3")
        monkeypatch.setenv("KUMIHO_FAILURE_PARK_TTL_DAYS", "7")
        monkeypatch.setenv("KUMIHO_FAILURE_LEDGER_MAX_ENTRIES", "42")
        ledger = FailureLedger()
        assert str(ledger.ledger_dir) == tmp
        assert ledger.park_threshold == 3
        assert ledger.park_ttl == timedelta(days=7)
        assert ledger.max_entries == 42


def test_env_bad_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("KUMIHO_FAILURE_PARK_THRESHOLD", "not-an-int")
    monkeypatch.setenv("KUMIHO_FAILURE_PARK_TTL_DAYS", "")
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp)
        assert ledger.park_threshold == 2
        assert ledger.park_ttl == timedelta(days=14)


def test_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("KUMIHO_FAILURE_PARK_THRESHOLD", "9")
    with tempfile.TemporaryDirectory() as tmp:
        ledger = FailureLedger(tmp, park_threshold=4)
        assert ledger.park_threshold == 4


def test_default_failure_ledger_enabled(monkeypatch):
    monkeypatch.delenv("KUMIHO_FAILURE_LEDGER_DISABLED", raising=False)
    ledger = default_failure_ledger()
    assert isinstance(ledger, FailureLedger)


def test_default_failure_ledger_disabled(monkeypatch):
    monkeypatch.setenv("KUMIHO_FAILURE_LEDGER_DISABLED", "1")
    assert default_failure_ledger() is None
