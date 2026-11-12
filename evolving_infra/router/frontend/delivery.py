# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Bounded, in-memory delivery of idempotent Backend metadata, never generation."""

import asyncio
import logging
import time
from collections import deque

import aiohttp

from .store import ProtocolError

LOG = logging.getLogger(__name__)

# A queued event that has not been accepted this long after it entered the
# outbox is given up after its next failed attempt.
RETRY_SECONDS = 600.0


def permanently_rejected(status):
    """Tell whether a callback HTTP status means the Backend will never accept the event.

    Args:
        status: HTTP status code of the callback response.

    Returns:
        True for a 4xx other than 408 and 429; resending the same event cannot succeed.
    """
    return 400 <= status < 500 and status not in (408, 429)


class CallbackOutbox:
    """Reserve room for every live plan's events before either engine can start.

    A slot remains occupied after its HTTP request ends if metadata still needs
    delivery. This is deliberately not crash-durable storage or engine recovery.
    """

    def __init__(self, service):
        """Create bounded in-memory ownership for live plans' callback delivery.

        Args:
            service: Tokenizer service providing HTTP client, limits, and counters.

        Returns:
            None; initializes an empty open outbox.
        """
        self.service = service
        self.capacity = service.config.callback_outbox_max_plans
        self.plans = {}
        self.closed = False

    def check_capacity(self):
        """Reject new plans when callback ownership is closed or exhausted.

        Args:
            None.

        Returns:
            None when a slot is available; does not reserve one.

        Raises:
            ProtocolError: The outbox is closed or at its configured plan capacity.
        """
        if self.closed or len(self.plans) >= self.capacity:
            raise ProtocolError(503, "CALLBACK_BACKLOG", "Callback outbox is at capacity.")

    def reserve(self, plan):
        """Reserve one callback slot before a plan can submit engine work.

        Args:
            plan: Validated P/D plan with callback endpoint and credential.

        Returns:
            A CallbackDelivery registered under the plan's reservation ID.

        Raises:
            ProtocolError: Capacity is exhausted or the reservation already exists.
        """
        self.check_capacity()
        if plan.reservation_id in self.plans:
            raise ProtocolError(409, "PLAN_ALREADY_FORWARDED", "Plan is already being forwarded.")
        delivery = CallbackDelivery(self, plan)
        self.plans[plan.reservation_id] = delivery
        return delivery

    def summary(self):
        """Summarize active plans and queued metadata events.

        Args:
            None.

        Returns:
            A dictionary of capacity, reserved/active plans, and pending counts.
        """
        return {
            "capacity_plans": self.capacity,
            "reserved_plans": len(self.plans),
            "active_plans": sum(not p.finished for p in self.plans.values()),
            "pending_plans": sum(bool(p.pending) for p in self.plans.values()),
            "pending_events": sum(len(p.pending) for p in self.plans.values()),
        }

    async def close(self):
        """Stop admission and cancel outstanding metadata retry tasks.

        Logs unresolved plans; this is not durable delivery or engine recovery.

        Args:
            None.

        Returns:
            None after all retry tasks finish cancellation.
        """
        self.closed = True
        if self.plans:
            LOG.error("callback_outbox_shutdown_unresolved %s; Backend must reconcile after process loss",
                      self.summary())
        tasks = [p.retry for p in self.plans.values() if p.retry is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class CallbackDelivery:
    """One ordered queue retained by the service after a forward request closes."""

    def __init__(self, outbox, plan):
        """Create an ordered metadata queue for one reserved execution plan.

        Args:
            outbox: Owner retaining this delivery until completion and an empty queue.
            plan: P/D plan supplying reservation ID and callback credentials.

        Returns:
            None; records callback destination and initializes delivery state.
        """
        self.outbox = outbox
        self.reservation = plan.reservation_id
        self.endpoint, self.token = plan.callback.endpoint, plan.callback.token
        self.pending = deque()
        self.finished = False
        self.retry = None

    async def attempt(self, event):
        """POST one callback event once.

        Args:
            event: Idempotent Backend metadata envelope to send.

        Returns:
            The response's HTTP status code.

        Raises:
            aiohttp.ClientError, TimeoutError: The request failed or timed out.
        """
        service = self.outbox.service
        async with service.client.post(
            self.endpoint, json=event,
            headers={"Authorization": "Bearer " + self.token},
            timeout=aiohttp.ClientTimeout(total=service.config.callback_timeout_seconds),
            allow_redirects=False,
        ) as response:
            return response.status

    async def send(self, event, attempts, final_on_reject):
        """Attempt bounded callback delivery without queuing the event.

        Only metadata is retried; generation never is. A final failure increments
        the service counter without logging credentials.

        Args:
            event: Idempotent Backend metadata envelope to send.
            attempts: Maximum HTTP attempts; failures between attempts wait briefly.
            final_on_reject: Stop at a permanent rejection instead of using the
                remaining attempts.

        Returns:
            "accepted" after a 2xx, "rejected" after a permanent rejection
            (only with final_on_reject), "failed" after all attempts fail.
        """
        service = self.outbox.service
        for attempt in range(attempts):
            try:
                status = await self.attempt(event)
                if 200 <= status < 300:
                    return "accepted"
                if final_on_reject and permanently_rejected(status):
                    service.callback_failures += 1
                    return "rejected"
                raise RuntimeError("callback status " + str(status))
            except (aiohttp.ClientError, TimeoutError, RuntimeError) as exc:
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.05)
                else:
                    service.callback_failures += 1
                    LOG.warning("pd_callback_failed rid=%r event=%r error=%s",
                                event["request_id"], event["event_id"], type(exc).__name__)
        return "failed"

    async def send_immediate(self, event, attempts=2):
        """Attempt bounded callback delivery of a claim without queuing it.

        Claims use this path directly; every non-2xx answer uses an attempt.

        Args:
            event: Idempotent Backend metadata envelope to send.
            attempts: Maximum HTTP attempts; failures between attempts wait briefly.

        Returns:
            True after an HTTP success; False after all attempts fail.
        """
        return await self.send(event, attempts, final_on_reject=False) == "accepted"

    def drop(self, reason):
        """Give up the oldest queued event: log it once and count it.

        Args:
            reason: "rejected" (permanent 4xx) or "expired" (retry age bound reached).

        Returns:
            None; the next queued event becomes the oldest.
        """
        event, _queued = self.pending.popleft()
        self.outbox.service.callback_dropped += 1
        LOG.error("pd_callback_dropped rid=%r event=%r reason=%s",
                  event["request_id"], event["event_id"], reason)

    async def deliver(self, event):
        """Deliver or queue a non-claim event while preserving delivery order.

        Once an older event is pending, newer events cannot bypass it. Foreground
        cancellation also leaves an unacknowledged event queued. An event the
        Backend rejects permanently (a 4xx other than 408 and 429) is dropped.

        Args:
            event: Idempotent execution-state envelope; must not be a claim.

        Returns:
            None; the event is accepted remotely or retained for retry.

        Raises:
            ValueError: The event is a claim, which must not be deferred.
        """
        if event["state"] == "claimed":
            raise ValueError("A claim must never be deferred")
        # The request has one sequential reporter. Once delivery is deferred,
        # subsequent events append without bypassing an earlier unacknowledged one.
        queued = (event, time.monotonic())
        if self.pending:
            self.pending.append(queued)
            return
        self.pending.append(queued)
        outcome = None
        try:
            outcome = await self.send(event, 2, final_on_reject=True)
        finally:
            if outcome == "accepted":
                self.pending.popleft()
            elif outcome == "rejected":
                self.drop("rejected")
                self.release_if_done()
            else:
                # Also retains the event if the foreground reporter was cancelled.
                self.retry = asyncio.create_task(self.retry_pending())

    async def retry_pending(self):
        """Retry the oldest queued metadata until the queue drains or cancellation.

        Uses bounded exponential backoff. The oldest event is dropped when the
        Backend rejects it permanently, or when an attempt fails RETRY_SECONDS or
        more after it entered the outbox; the next event is then tried. Releases a
        finished plan's slot when no event remains.

        Args:
            None.

        Returns:
            None when no pending events remain.
        """
        delay = 0.5
        try:
            while self.pending:
                await asyncio.sleep(delay)
                event, queued = self.pending[0]
                outcome = await self.send(event, 1, final_on_reject=True)
                if outcome == "accepted":
                    self.pending.popleft()
                    delay = 0.5
                elif outcome == "rejected":
                    self.drop("rejected")
                    delay = 0.5
                elif time.monotonic() - queued >= RETRY_SECONDS:
                    self.drop("expired")
                    delay = 0.5
                else:
                    delay = min(delay * 2, 30)
        finally:
            self.release_if_done()

    def finish(self):
        """Mark request-side reporting finished and release the slot if drained.

        Args:
            None.

        Returns:
            None; pending events keep their outbox ownership until accepted or dropped.
        """
        self.finished = True
        self.release_if_done()

    def release_if_done(self):
        """Remove outbox ownership only after reporting finishes and the queue empties.

        Args:
            None.

        Returns:
            None; an unfinished or pending delivery remains registered.
        """
        if self.finished and not self.pending:
            self.outbox.plans.pop(self.reservation, None)
