"""One-shot session reset markers: a consolidate must not look like a new session.

Role: the part of the Redis working-memory buffer that empties a session's buffer
on purpose (``consolidate_session``, ``kumiho_chat_clear``) while keeping the
session itself, and remembers that it did so for exactly one next write.
:class:`SessionResetMixin` is mixed into
:class:`kumiho_memory.redis_memory.RedisMemoryBuffer`; it holds no state of its
own and uses only the buffer's existing primitives (``client``, ``default_ttl``,
``_proxy_request``, ``set_session_metadata`` / ``get_session_metadata`` /
``clear_session``, the session key builders).

Contract (``created_bucket`` semantics): a write that creates a buffer reports
``created_bucket: true`` unless a reset marker is pending for that session id, in
which case it reports ``false`` plus ``buffer_reset_by: <reason>`` and consumes the
marker.  So ``true`` means the id never had a buffer or lost it to idle-TTL expiry --
with the residual cases listed in :meth:`SessionResetMixin.reset_session`.
No schema or action change on the hosted proxy: markers are session metadata of the
reserved sibling id ``"<session_id>:~reset"``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Suffix of the reserved sibling session id that holds a session's one-shot
#: reset marker (see :meth:`RedisMemoryBuffer.reset_session`).  The marker is
#: kept as the *session metadata* of ``"{session_id}:~reset"`` rather than under
#: a purpose-built key so the identical bytes work against a direct Redis and
#: against the control-plane proxy, whose action list is closed and whose
#: request schema is strict (an unknown action or field is a 400).
RESET_MARKER_SUFFIX = ":~reset"

#: A reset marker lives at least this long (seconds), whatever the buffer TTL.
#: The marker is only ever pending while NOTHING has been written since the
#: reset, so a long life normally hides no loss of turns -- there are none to
#: lose (the residual cases are listed in ``reset_session``).
#: What it prevents is the false alarm after "consolidate, then a pause longer
#: than the buffer TTL".  Same day-long horizon as the generation counter; a
#: memory-only hosted tier clamps it to its own ceiling (6 h) server-side.
RESET_MARKER_MIN_TTL = 86400

#: A memory-only hosted tenant's server trims the buffer to its last 250 turns on
#: every append (kumiho-control ``MEMORY_APPEND_SCRIPT``, ``LTRIM -250``).  At the
#: cap a turn appended during a consolidate leaves the count unchanged, so it
#: cannot be told apart from "nothing arrived".
HOSTED_BUFFER_CAP = 250


class SessionResetMixin:
    """``reset_session`` and its marker bookkeeping (see the module docstring)."""

    # Provided by RedisMemoryBuffer; declared for readers and type checkers.
    client: Any
    default_ttl: int

    async def reset_session(
        self,
        project: str,
        session_id: str,
        *,
        reason: str,
        expected_count: Optional[int] = None,
        read_complete: bool = True,
        carry_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Empty a session's buffer while the session itself carries on.

        What ``consolidate_session`` (``reason="consolidate"``) and
        ``kumiho_chat_clear`` (``reason="clear"``) do.  The result is
        :meth:`clear_session`'s; the difference is a one-shot *reset marker*
        that makes the next write into this same session id report
        ``created_bucket: false`` (plus ``buffer_reset_by: <reason>``) instead of
        a fresh bucket.

        ``created_bucket: true`` therefore means one thing only: this session id
        has no buffer and no pending deliberate reset -- it never existed, or its
        buffer was lost to idle-TTL expiry.  Without the marker a consolidate
        emptied the buffer, the next RPUSH returned 1, and the unchanged session
        looked recreated mid-conversation.

        ``expected_count`` is how many turns the caller actually read (and, for a
        consolidate, summarized).  Turns appended after that read were never
        consolidated: on a direct Redis they are KEPT (the buffer is trimmed by
        ``expected_count`` instead of deleted) and no marker is written, since the
        buffer is not empty.  Through the proxy, which can only delete, they are
        discarded as before -- but not excused: no marker and a WARNING, so the
        next write normally reports ``created_bucket: true``.

        The reset also refuses to excuse what it cannot vouch for, and says so in
        a WARNING: ``read_complete=False`` (the read was truncated, so older turns
        were never summarized), fewer turns present than were read (a concurrent
        reset, or the buffer expired meanwhile), and -- through the proxy -- a
        buffer at the server's 250-turn cap (:data:`HOSTED_BUFFER_CAP`).

        ``carry_to`` names a second id that will carry the session on (the
        generation-rotated ``"<env id>:c<n>"`` of a host-env session); it gets
        the same marker.  The reset id keeps its own marker too, because a
        caller that passes the id explicitly keeps writing to it.

        Properties the callers rely on:

        * **Only when something was discarded.**  The marker is written only if
          the buffer held messages; resetting an empty buffer leaves any
          existing marker (or the absence of one) as it was.
        * **Written before the buffer is deleted.**  A write racing the reset
          either lands in the still-live buffer (count > 1) or finds the marker;
          none can see "empty buffer, no marker".
        * **Single use, long lived.**  The first write after the reset consumes
          it (a ``reset_id`` nonce, so a newer reset is never overwritten by an
          older consume).  It lives ``max(TTL, 24 h)``: it is only pending while
          nothing was written since, so normally it hides no loss of turns, and a
          pause after a consolidate does not read as a lost session.  A later
          loss of the post-reset buffer reads as ``true`` (but see the residual
          below).
        * **Best effort.**  If the marker cannot be written, the buffer is still
          cleared and the old behaviour (``true`` on the next write) applies.
        * **No schema or action change.**  It uses the existing
          ``set_session_metadata`` / ``get_session_metadata`` / ``clear_session``
          / ``get_messages`` actions on the reserved sibling id
          ``"{session_id}:~reset"`` (:data:`RESET_MARKER_SUFFIX`; ids ending in
          it are refused by :meth:`add_message`).

        **Known residual, not closed.**  A marker stays pending over a live buffer
        when the first write after a reset is made by a client that predates the
        marker, or -- hosted, non-memory tier, whose server-side rPush/lLen is not
        atomic -- when two first writes both receive count 2.  Until it expires
        (up to 24 h) such a marker can excuse a REAL idle-TTL loss of that buffer
        as "reset by consolidate".  A write that gets count 2 cannot safely retire
        it: it cannot tell a stale marker from one the in-flight count-1 writer is
        about to consume, and consuming it first makes that writer report a fresh
        bucket (the two-appends race test).  Two concurrent consolidates of one
        session that read the same turns are caught by the "fewer turns than were
        read" check except when exactly as many turns arrived as were read.

        :meth:`clear_session` itself is unchanged: it is the full wipe.
        """
        count = await self._message_count(project, session_id)
        late = (
            expected_count is not None and count is not None
            and count > expected_count
        )
        owners = [session_id] + ([carry_to] if carry_to and carry_to != session_id else [])
        marked: List[str] = []
        unvouched = self._cannot_vouch(
            session_id, count, expected_count, read_complete,
        )
        if not late and not unvouched and (count is None or count > 0):
            reset_id = uuid.uuid4().hex
            for owner in owners:
                if await self._write_reset_marker(project, owner, reason, reset_id):
                    marked.append(owner)

        result, extra, kept = await self._discard_buffer(project, session_id, expected_count)

        if extra:
            # Turns the caller never read: the buffer is not (or must not read
            # as) emptied, so there is nothing to excuse.
            for owner in marked:
                await self._drop_reset_marker(project, owner)
            if kept:
                logger.info(
                    "reset of session %s kept %d turn(s) appended after the "
                    "consolidation read.", session_id, extra,
                )
            else:
                logger.warning(
                    "reset of session %s discarded %d turn(s) appended after "
                    "the consolidation read (never summarized); no reset marker "
                    "is left for them, so the next write normally reports "
                    "created_bucket=true (a write racing this reset may not).",
                    session_id, extra,
                )
        return result

    def _cannot_vouch(
        self,
        session_id: str,
        count: Optional[int],
        expected_count: Optional[int],
        read_complete: bool,
    ) -> bool:
        """True when the reset cannot say that it discards only what was read.

        Then no marker is written (the loss is not excused) and a WARNING says
        why.  ``expected_count`` is None for a plain clear, which read nothing and
        so has nothing to compare.
        """
        why: Optional[str] = None
        if not read_complete:
            why = "turns older than the consolidation read limit were never summarized"
        elif expected_count is not None and count is not None and count < expected_count:
            why = (
                "the buffer holds fewer turns than the consolidation read "
                "(a concurrent reset, or it expired meanwhile)"
            )
        elif (
            self.client is None and expected_count is not None
            and count is not None and count >= HOSTED_BUFFER_CAP
        ):
            why = (
                f"the buffer is at the hosted cap of {HOSTED_BUFFER_CAP} turns, "
                "where a turn appended during the consolidation cannot be seen"
            )
        if why is None:
            return False
        logger.warning(
            "reset of session %s is not excused: %s; the next write may report "
            "created_bucket=true.", session_id, why,
        )
        return True

    async def _message_count(self, project: str, session_id: str) -> Optional[int]:
        """How many messages the session's buffer holds right now (None: unknown).

        A reset only needs to excuse the next write when it actually discarded
        buffered turns.  Resetting an empty buffer -- consolidating a session
        whose buffer already idled out, a second consolidate before anything new
        was written, a clear of an id nobody used -- discards nothing, so it must
        neither turn a genuine TTL loss into "reset by consolidate" nor overwrite
        a marker that is still pending.  Unknown (the read failed) is treated by
        the caller as "yes, something to excuse": it is resetting a session it
        has been using, and a false alarm is the worse error.
        """
        try:
            if self.client is None:
                got = await self._proxy_request(
                    action="get_messages",
                    payload={
                        "project": project, "session_id": session_id,
                        "limit": 1, "offset": 0,
                    },
                )
                return int(got.get("message_count", 0))
            return int(await self.client.llen(
                self._session_messages_key(project, session_id)
            ))
        except Exception:  # noqa: BLE001 - unknown, see the docstring
            return None

    async def _discard_buffer(
        self, project: str, session_id: str, expected_count: Optional[int],
    ) -> Tuple[Dict[str, Any], int, bool]:
        """Empty the buffer; returns ``(result, extra, kept)``.

        ``extra`` counts turns beyond ``expected_count`` found at delete time and
        ``kept`` says whether they survived.  A direct Redis trims the first
        ``expected_count`` entries (LTRIM is atomic, so a concurrent append can
        never be deleted unseen); the proxy has no such action and deletes.
        Without ``expected_count`` this is exactly :meth:`clear_session`.
        """
        if expected_count is None:
            return await self.clear_session(project, session_id), 0, False
        if self.client is None:
            result = await self.clear_session(project, session_id)
            try:
                cleared = int(result.get("cleared_count") or 0)
            except (TypeError, ValueError, AttributeError):
                cleared = 0
            return result, max(0, cleared - expected_count), False
        key = self._session_messages_key(project, session_id)
        before = int(await self.client.llen(key))
        await self.client.ltrim(key, expected_count, -1)
        remaining = int(await self.client.llen(key))
        await self.client.delete(self._session_metadata_key(project, session_id))
        cleared = max(0, before - remaining)
        return {"success": True, "cleared_count": cleared}, remaining, True

    @staticmethod
    def _reset_marker_session_id(session_id: str) -> str:
        return f"{session_id}{RESET_MARKER_SUFFIX}"

    def _reset_marker_ttl(self) -> int:
        return max(self.default_ttl, RESET_MARKER_MIN_TTL)

    async def _write_reset_marker(
        self, project: str, session_id: str, reason: str, reset_id: str,
    ) -> bool:
        try:
            await self.set_session_metadata(
                project,
                self._reset_marker_session_id(session_id),
                {"reset_id": reset_id, "reason": reason},
                ttl_seconds=self._reset_marker_ttl(),
            )
        except Exception as exc:  # noqa: BLE001 - never fail the reset over it
            logger.warning(
                "Could not write the reset marker for session %s (%s); the "
                "next write into it will report created_bucket=true.",
                session_id, exc,
            )
            return False
        return True

    async def _drop_reset_marker(self, project: str, session_id: str) -> None:
        """Delete a marker (existing ``clear_session`` action, a removal that
        read-only tenants may still call)."""
        try:
            await self.clear_session(project, self._reset_marker_session_id(session_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not remove the reset marker of session %s (%s); it "
                "stays pending until its TTL.", session_id, exc,
            )

    async def _excuse_deliberate_reset(
        self, project: str, session_id: str, result: Dict[str, Any],
    ) -> None:
        """Turn a write's ``created_bucket: true`` into ``false`` after a reset.

        Only a write that minted the bucket (count == 1) ever looks at the
        marker, and RPUSH hands "1" to exactly one concurrent writer, so the
        marker is consumed once.  Consuming records ``consumed = <reset_id>``
        (HSET merges), so a newer reset that wrote its own ``reset_id`` in the
        meantime is still pending.  Adds ``buffer_reset_by`` (additive) when it
        applies; leaves ``result`` untouched otherwise.
        """
        if result.get("created_bucket") is not True:
            return
        marker_id = self._reset_marker_session_id(session_id)
        try:
            marker = await self.get_session_metadata(project, marker_id)
        except Exception as exc:  # noqa: BLE001 - unreadable == absent
            logger.warning(
                "Could not read the reset marker for session %s (%s); "
                "reporting created_bucket=true.",
                session_id, exc,
            )
            return
        reset_id = marker.get("reset_id")
        if not reset_id or marker.get("consumed") == reset_id:
            return
        try:
            await self.set_session_metadata(
                project, marker_id, {"consumed": reset_id},
                ttl_seconds=self._reset_marker_ttl(),
            )
        except Exception as exc:  # noqa: BLE001 - the reset is real regardless
            logger.warning(
                "Could not mark the reset marker of session %s consumed (%s); "
                "removing it instead.", session_id, exc,
            )
            # A pending marker over a live buffer would excuse a LATER loss.
            await self._drop_reset_marker(project, session_id)
        result["created_bucket"] = False
        result["buffer_reset_by"] = str(marker.get("reason") or "reset")
