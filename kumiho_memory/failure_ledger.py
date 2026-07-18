"""Cross-run failure ledger for parking deterministically-poisoned content.

Some content fails the same way every time it is processed — a validation
error, a content-filter refusal, a 4xx-class semantic rejection.  When such an
item keeps getting re-selected by Dream State or consolidation, it re-fails run
after run: a *poison loop*.

This module records failed attempts *across process runs* keyed by a stable
identity (an item kref, or a content hash).  Once an item has failed
deterministically at least ``park_threshold`` times it is *parked*: selection
sites (Dream State collection, the consolidation store seam) skip parked items
so the poison loop stops.  Parked items un-park after ``park_ttl_days`` so a
fixed model or prompt gets a fresh retry chance.

The ledger is a single small JSON file written atomically (temp file +
``os.replace``) so a crashed or concurrent writer can never leave a partial,
corrupt file.  A truncated/garbage file loads as an empty ledger — it never
raises into the caller.

The ledger lives under the same local-state root as the retry queue and Dream
State artifacts (``~/.kumiho/failure_ledger`` by default), overridable with
``KUMIHO_FAILURE_LEDGER_DIR``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

from kumiho_memory.retry import FailureClass

logger = logging.getLogger(__name__)

# Per-file locks so writers in the same process serialize their
# read-modify-write cycles — this prevents lost updates and (on Windows)
# self-inflicted ``os.replace`` sharing-violation contention.  Cross-process
# safety still rests on atomic replace + the read/replace retries below.
_PATH_LOCKS: "Dict[str, threading.RLock]" = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> "threading.RLock":
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[path] = lock
        return lock

# On Windows ``os.replace`` transiently fails with a sharing violation
# (PermissionError) when another thread/process holds the target open for
# reading.  The file is never corrupt — the replace is atomic — so we simply
# retry the replace a few times to absorb the tiny contention window.
_REPLACE_RETRIES = 20
_REPLACE_BACKOFF = 0.01

# Schema version stamped into the file so a future format change can be
# detected (a mismatched version is treated like corruption: start fresh).
_LEDGER_VERSION = 1

_DEFAULT_PARK_THRESHOLD = 2
_DEFAULT_PARK_TTL_DAYS = 14
_DEFAULT_MAX_ENTRIES = 2000


def content_key(*parts: str) -> str:
    """Return a stable hash key for the given content *parts*.

    Used to key the ledger by content when no kref exists yet (the
    consolidation store seam runs before a revision is created).
    """
    joined = "\x1f".join(p for p in parts if p)
    return sha256(joined.encode("utf-8", "replace")).hexdigest()


def default_failure_ledger() -> "Optional[FailureLedger]":
    """Construct the default runtime ledger, or ``None`` if disabled.

    Set ``KUMIHO_FAILURE_LEDGER_DISABLED=1`` to opt out (parking off).
    Construction is side-effect-free — the storage directory is only created
    on the first write — so this is safe to call from entrypoints.
    """
    disabled = os.getenv("KUMIHO_FAILURE_LEDGER_DISABLED", "").strip().casefold() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if disabled:
        return None
    try:
        return FailureLedger()
    except Exception as exc:  # noqa: BLE001 — never let ledger setup break a run
        logger.warning("Failed to construct failure ledger (parking disabled): %s", exc)
        return None


class FailureLedger:
    """File-backed, corruption-safe record of cross-run failed attempts.

    Parameters
    ----------
    ledger_dir:
        Directory holding ``ledger.json``.  Defaults to
        ``$KUMIHO_FAILURE_LEDGER_DIR`` or ``~/.kumiho/failure_ledger``.
    park_threshold:
        Number of *deterministic* failures after which the item is parked
        (transient/unknown failures never count toward it).  Defaults to
        ``$KUMIHO_FAILURE_PARK_THRESHOLD`` or ``2``.
    park_ttl_days:
        Days a parked item stays parked before it un-parks.  Defaults to
        ``$KUMIHO_FAILURE_PARK_TTL_DAYS`` or ``14``.
    max_entries:
        Soft cap on stored entries; the least-important entries are pruned
        when exceeded.  Defaults to ``$KUMIHO_FAILURE_LEDGER_MAX_ENTRIES`` or
        ``2000``.

    The directory is created lazily (on first write), so constructing a ledger
    that is never written to touches no filesystem state.
    """

    def __init__(
        self,
        ledger_dir: Optional[str] = None,
        *,
        park_threshold: Optional[int] = None,
        park_ttl_days: Optional[float] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        self.ledger_dir = Path(
            ledger_dir
            or os.getenv("KUMIHO_FAILURE_LEDGER_DIR")
            or os.path.join(os.path.expanduser("~"), ".kumiho", "failure_ledger")
        )
        self._file = self.ledger_dir / "ledger.json"
        self.park_threshold = max(
            1,
            _int_env(park_threshold, "KUMIHO_FAILURE_PARK_THRESHOLD", _DEFAULT_PARK_THRESHOLD),
        )
        self.park_ttl = timedelta(
            days=max(
                0.0,
                _float_env(park_ttl_days, "KUMIHO_FAILURE_PARK_TTL_DAYS", _DEFAULT_PARK_TTL_DAYS),
            )
        )
        self.max_entries = max(
            1,
            _int_env(max_entries, "KUMIHO_FAILURE_LEDGER_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES),
        )
        self._lock = _lock_for(os.path.abspath(str(self._file)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_failure(
        self,
        key: str,
        error_class: str,
        *,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Record one failed attempt for *key* classified as *error_class*.

        Increments the attempt counters, refreshes ``last_seen`` /
        ``last_error_class``, and parks the item once it has failed
        *deterministically* at least ``park_threshold`` times.  Returns a copy
        of the resulting entry.
        """
        now = now or _utcnow()
        now_iso = now.isoformat()
        with self._lock:
            data = self._load()
            entries = data["entries"]

            entry = entries.get(key)
            if not isinstance(entry, dict):
                entry = {"attempts": 0, "first_seen": now_iso, "parked_at": None}

            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["last_seen"] = now_iso
            entry["last_error_class"] = error_class
            entry.setdefault("first_seen", now_iso)

            det_attempts = int(entry.get("deterministic_attempts", 0) or 0)
            if error_class == FailureClass.DETERMINISTIC:
                det_attempts += 1
            entry["deterministic_attempts"] = det_attempts

            # Park only after ``park_threshold`` *deterministic* failures (issue
            # #118 asks for "classified deterministic >= 2 attempts").  Counting
            # deterministic attempts specifically — not total attempts of any
            # class — avoids pulling storable content out of rotation for the
            # whole TTL after a single deterministic failure that merely
            # happened to follow an unrelated transient blip.
            if (
                error_class == FailureClass.DETERMINISTIC
                and det_attempts >= self.park_threshold
            ):
                # Refresh the park timestamp so the TTL clock restarts on every
                # deterministic failure past the threshold.
                entry["parked_at"] = now_iso

            entries[key] = entry
            self._prune(entries, protect=key)
            self._save(data)
            return dict(entry)

    def record_success(self, key: str) -> None:
        """Clear any failure history for *key* (it succeeded)."""
        with self._lock:
            data = self._load()
            if key in data["entries"]:
                del data["entries"][key]
                self._save(data)

    def is_parked(self, key: str, *, now: Optional[datetime] = None) -> bool:
        """Return ``True`` if *key* is currently parked (TTL not elapsed).

        Pure read — never mutates the file.  A parked entry whose TTL has
        elapsed reads as *not* parked (it becomes selectable again); its stale
        ``parked_at`` is refreshed on the next failure or cleared on the next
        success/sweep.
        """
        now = now or _utcnow()
        with self._lock:
            entry = self._load()["entries"].get(key)
        if not isinstance(entry, dict):
            return False
        parked_at = _parse_iso(entry.get("parked_at"))
        if parked_at is None:
            return False
        return (now - parked_at) < self.park_ttl

    def sweep_expired(self, *, now: Optional[datetime] = None) -> int:
        """Un-park every entry whose TTL has elapsed; return how many.

        Un-parking clears ``parked_at`` and resets the attempt counter so a
        recovered item gets a full fresh window before it can re-park.
        Optional housekeeping — :meth:`is_parked` already treats expired
        entries as selectable without a sweep.
        """
        now = now or _utcnow()
        with self._lock:
            data = self._load()
            changed = 0
            for entry in data["entries"].values():
                if not isinstance(entry, dict):
                    continue
                parked_at = _parse_iso(entry.get("parked_at"))
                if parked_at is not None and (now - parked_at) >= self.park_ttl:
                    entry["parked_at"] = None
                    entry["attempts"] = 0
                    entry["deterministic_attempts"] = 0
                    changed += 1
            if changed:
                self._save(data)
            return changed

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return a copy of *key*'s entry, or ``None`` if absent."""
        with self._lock:
            entry = self._load()["entries"].get(key)
        return dict(entry) if isinstance(entry, dict) else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._load()["entries"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        """Load the ledger, returning a fresh empty one on genuine corruption.

        A missing, truncated, or garbage file — including one whose bytes are
        not valid UTF-8 — or one with an unexpected shape/version yields an
        empty ledger rather than raising.  A *transient* OS lock (a concurrent
        writer mid-replace on Windows) is retried rather than mistaken for
        corruption, so concurrent writers do not wipe each other's entries.
        """
        if not self._file.exists():
            return _empty()

        raw_bytes: Optional[bytes] = None
        for attempt in range(_REPLACE_RETRIES):
            try:
                # Read as bytes and decode below so invalid UTF-8 is handled as
                # corruption (start fresh) rather than raising a
                # UnicodeDecodeError past this method.  Text-mode ``read()``
                # would decode eagerly and escape both ``except`` arms here
                # (UnicodeDecodeError is a ValueError, not an OSError),
                # permanently disabling parking on a corrupt file.
                with open(self._file, "rb") as f:
                    raw_bytes = f.read()
                break
            except FileNotFoundError:
                # Replaced away between exists() and open() — treat as empty.
                return _empty()
            except OSError:
                # Transient sharing violation — retry; the file is valid.
                if attempt + 1 >= _REPLACE_RETRIES:
                    logger.warning("Failure ledger read kept failing — starting fresh")
                    return _empty()
                time.sleep(_REPLACE_BACKOFF)

        try:
            # ``UnicodeDecodeError`` (invalid UTF-8) is a ``ValueError`` subclass;
            # both a decode failure and a JSON parse failure land here and are
            # treated as corruption.
            raw = (raw_bytes or b"").decode("utf-8")
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("Failure ledger corrupt (%s) — starting fresh", exc)
            return _empty()
        if (
            not isinstance(data, dict)
            or data.get("version") != _LEDGER_VERSION
            or not isinstance(data.get("entries"), dict)
        ):
            logger.warning("Failure ledger has unexpected shape — starting fresh")
            return _empty()
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        """Atomically write the ledger (temp file + ``os.replace``).

        The directory is created here (not at construction) so an unused
        ledger touches no filesystem state.
        """
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="ledger-", suffix=".tmp", dir=str(self.ledger_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
                f.flush()
                os.fsync(f.fileno())
            self._replace_with_retry(tmp_path)
        except BaseException:
            # Best-effort cleanup of the temp file; never leave it behind.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _replace_with_retry(self, tmp_path: str) -> None:
        """``os.replace`` the temp file onto the ledger, retrying on Windows
        sharing violations (a concurrent reader briefly locks the target)."""
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp_path, self._file)
                return
            except OSError:
                # PermissionError / sharing violation from a concurrent
                # reader — retry; the replace itself is atomic.
                if attempt + 1 >= _REPLACE_RETRIES:
                    raise
                time.sleep(_REPLACE_BACKOFF)

    def _prune(self, entries: Dict[str, Any], *, protect: Optional[str] = None) -> None:
        """Drop the least-important entries when over ``max_entries``.

        Non-parked, least-recently-seen entries are dropped first so parked
        (poison) items are retained as long as possible — dropping a parked
        entry would let its poison content be re-selected.

        ``protect`` names an entry that must never be dropped — the caller
        passes the key it just recorded.  Without this, a poison item building
        toward the park threshold (still non-parked, so ranked first for
        dropping) would be evicted every run when the ledger is saturated with
        parked entries, and could never accumulate enough deterministic
        failures to park.
        """
        if len(entries) <= self.max_entries:
            return

        def sort_key(item: Any) -> Any:
            _, entry = item
            parked = 1 if isinstance(entry, dict) and entry.get("parked_at") else 0
            last_seen = entry.get("last_seen", "") if isinstance(entry, dict) else ""
            return (parked, last_seen)

        ordered = sorted(entries.items(), key=sort_key)
        drop = len(entries) - self.max_entries
        dropped = 0
        for key, _ in ordered:
            if dropped >= drop:
                break
            if key == protect:
                continue
            del entries[key]
            dropped += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty() -> Dict[str, Any]:
    return {"version": _LEDGER_VERSION, "entries": {}}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _int_env(explicit: Optional[int], env: str, default: int) -> int:
    if explicit is not None:
        return int(explicit)
    raw = os.getenv(env)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float_env(explicit: Optional[float], env: str, default: float) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.getenv(env)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
