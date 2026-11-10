"""Upload prepared inputs once and relay single-engine or split P/D responses.

Own local upload buffers, validate terminal engine responses, and report
execution metadata to the Rust Backend. Local HTTP cleanup does not establish
that engine work or KV allocations have been released.
"""

import asyncio
import contextlib
import logging
import time
from contextlib import AsyncExitStack
import aiohttp
import anyio
import orjson
from fastapi.responses import Response, StreamingResponse

from .store import ProtocolError

LOG = logging.getLogger(__name__)

class Upload:
    """Own outgoing bytes until the HTTP sender finishes using them.

    Input: body bytes, the service's counters and request_id.
    Output: an async iterable Upload instance that releases its body on close.
    """

    def __init__(self, data: bytes, service, request_id: str):
        """Own one serialized input body and account for its upload memory.

        Args:
            data: Serialized request-body bytes retained until upload cleanup.
            service: Tokenizer service owning upload counters.
            request_id: Request identity used in release diagnostics.

        Returns:
            None; increments current and peak upload-byte counters.
        """
        self.data, self.service, self.request_id = data, service, request_id
        self.size, self.started = len(data), time.perf_counter()
        service.upload_bytes += self.size
        service.peak_upload_bytes = max(service.peak_upload_bytes, service.upload_bytes)

    async def __aiter__(self):
        """Feed retained body chunks to the HTTP sender and release on iterator exit.

        The finally block invokes aclose even on cancellation or early exit.

        Args:
            None.

        Yields:
            Body chunks no larger than 64 KiB.
        """
        try:
            for offset in range(0, len(self.data), 65536):
                yield self.data[offset : offset + 65536]
        finally:
            await self.aclose()

    async def aclose(self):
        """Clear retained input bytes and update upload accounting once.

        Args:
            None.

        Returns:
            None; repeated calls after release have no effect.
        """
        if self.data:
            self.data = b""
            self.service.upload_bytes -= self.size
            LOG.info(
                "input_released request_id=%r upload_seconds=%.6f body_bytes=%d",
                self.request_id,
                time.perf_counter() - self.started,
                self.size,
            )


class ForwardResponse(StreamingResponse):
    """Relay an upstream stream and close its resources when delivery ends.

    Input: an upstream response, downstream headers and the input Upload.
    Output: a StreamingResponse-compatible instance with disconnect cleanup.
    """

    def __init__(self, upstream: aiohttp.ClientResponse, headers: dict, upload: Upload):
        """Bind a single upstream stream and its upload to downstream delivery.

        Args:
            upstream: Open aiohttp response to relay.
            headers: Headers passed to the downstream HTTP response.
            upload: Owned input upload to close when delivery ends.

        Returns:
            None; configures status, headers, and response-body iteration.
        """
        self.upstream, self.upload = upstream, upload

        async def chunks():
            """Read bytes from the captured upstream response without reinterpretation.

            Uses the upstream response captured by the enclosing constructor.

            Args:
                None.

            Yields:
                Available upstream body byte chunks.
            """
            async for part in upstream.content.iter_any():
                yield part

        super().__init__(chunks(), status_code=upstream.status, headers=headers)

    async def __call__(self, scope, receive, send):
        """Serve a stream and close its iterator, upstream response, and upload.

        Cleanup runs on normal completion, failure, or downstream disconnect.

        Args:
            scope: ASGI connection/request scope.
            receive: ASGI callable receiving client messages.
            send: ASGI callable sending response messages.

        Returns:
            None; emits response messages through send.
        """
        async with AsyncExitStack() as cleanup:
            cleanup.push_async_callback(self.upload.aclose)
            cleanup.push_async_callback(self.upstream.wait_for_close)
            cleanup.callback(self.upstream.close)
            cleanup.push_async_callback(self.body_iterator.aclose)
            await super().__call__(scope, receive, send)


class SharedBody:
    """Keep one serialized input until its HTTP uploaders have released it."""

    def __init__(self, data, service, *, owners=2):
        """Retain one serialized input prefix for one or both P/D uploads.

        Args:
            data: Serialized common JSON prefix, excluding its final closing brace.
            service: Tokenizer service owning upload-byte counters.
            owners: Number of uploads retaining this prefix.

        Returns:
            None; accounts for the common bytes once.
        """
        self.data, self.service, self.remaining = data, service, owners
        service.upload_bytes += len(data)
        service.peak_upload_bytes = max(service.peak_upload_bytes, service.upload_bytes)

    def release(self):
        """Drop one uploader's ownership and clear bytes after the last release.

        Each uploader must call this exactly once.

        Args:
            None.

        Returns:
            None; decrements upload accounting when all owners have released.
        """
        self.remaining -= 1
        if self.remaining == 0:
            self.service.upload_bytes -= len(self.data)
            self.data = b""


class SharedUpload:
    def __init__(self, common, suffix, uploaded):
        """Combine a shared input prefix with one leg-specific JSON suffix.

        Args:
            common: SharedBody jointly retained by prefill and decode uploaders.
            suffix: Serialized leg-specific fields that finish the JSON body.
            uploaded: Zero-argument callback invoked after the iterator yields all data.

        Returns:
            None; accounts for suffix bytes and records the full HTTP body length.
        """
        self.common, self.suffix, self.uploaded = common, suffix, uploaded
        self.size = len(common.data) + len(suffix)
        self.closed = False
        common.service.upload_bytes += len(suffix)
        common.service.peak_upload_bytes = max(
            common.service.peak_upload_bytes, common.service.upload_bytes
        )

    async def __aiter__(self):
        """Yield common input and the leg suffix, then signal upload completion.

        The callback marks iterator completion, not engine admission. Finalization
        always releases this uploader's memory ownership.

        Args:
            None.

        Yields:
            Common-body chunks followed by this leg's JSON suffix.
        """
        try:
            for offset in range(0, len(self.common.data), 65536):
                yield self.common.data[offset:offset + 65536]
            yield self.suffix
            self.uploaded()
        finally:
            await self.aclose()

    async def aclose(self):
        """Release this leg's suffix and shared-body ownership at most once.

        Args:
            None.

        Returns:
            None; updates upload counters and clears the suffix.
        """
        if not self.closed:
            self.closed = True
            self.common.service.upload_bytes -= len(self.suffix)
            self.suffix = b""
            self.common.release()


USAGE_FIELDS = {
    "prefill": ("prompt_tokens", "cached_tokens", "vision_matmul_flops", "image_attention_pairs"),
    "decode": ("completion_tokens", "spec_verify_ct"),
}


def leg_usage(value, leg):
    """Token counts from an engine response's meta_info for the Backend's P/D ratio window."""
    meta = value.get("meta_info") if isinstance(value, dict) else None
    if not isinstance(meta, dict):
        return None
    usage = {
        key: meta[key]
        for key in USAGE_FIELDS[leg]
        if isinstance(meta.get(key), int) and not isinstance(meta[key], bool) and meta[key] >= 0
    }
    return usage or None


def response_usage(data, leg):
    """leg_usage of a JSON result or SSE data payload; None when it does not parse."""
    try:
        return leg_usage(orjson.loads(data), leg)
    except orjson.JSONDecodeError:
        return None


# A resumed request reports these as the original did; completion_tokens continues.
STITCHED_FIELDS = ("prompt_tokens", "cached_tokens", "cached_tokens_details", "id")


# The engine's finish_reason err_type of a late-D candidate the prefill released
# (another candidate won the request): HTTP 409 on /v1/completions, an abort
# event on /generate.
LATE_D_RELEASED_ERR = "late_d_released"
# A loser candidate still open this long after the bind is closed (as the router).
LATE_D_LOSER_DRAIN_SECONDS = 30


def first_data(buf):
    """Payload of the first complete, non-empty SSE `data:` line in `buf`, else None."""
    lines = bytes(buf).split(b"\n")
    for line in lines[:-1]:
        line = line.strip()
        if line.startswith(b"data:") and line[5:].strip():
            return line[5:].strip()
    return None


def late_d_outcome(data):
    """Classify a late-D candidate's first event (stream) or body (non-stream).

    Args:
        data: The first SSE data payload, or the whole JSON body, as bytes.

    Returns:
        "released" for the engine's released abort, "failed" for any other
        error, abort or unparsable payload, None for generation output (the
        candidate the prefill admitted).
    """
    if data is None or data == b"[DONE]":
        return "failed"
    try:
        value = orjson.loads(data)
    except orjson.JSONDecodeError:
        return "failed"
    if not isinstance(value, dict) or "error" in value:
        return "failed"
    meta = value.get("meta_info")
    reason = meta.get("finish_reason") if isinstance(meta, dict) else None
    if isinstance(reason, dict) and reason.get("type") in ("abort", "error"):
        return "released" if reason.get("err_type") == LATE_D_RELEASED_ERR else "failed"
    return None


# Bytes of an HTTP 503 answer read for its abort message.
ENGINE_ABORT_BODY_BYTES = 65536
ENGINE_ABORT_MESSAGES = ("aborted by client", "compute-before-allocation aborted")


def engine_abort_message(body):
    """The abort message of an engine error answer, else None.

    Args:
        body: The (bounded) response body bytes.

    Returns:
        "finish_reason abort: <message>" for a JSON body whose meta_info finish_reason
        type is abort; the known engine abort message found in the body; else None.
        At most 200 characters.
    """
    text = bytes(body).decode("utf-8", "replace")
    try:
        value = orjson.loads(body)
    except orjson.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        meta = value.get("meta_info")
        reason = meta.get("finish_reason") if isinstance(meta, dict) else None
        if isinstance(reason, dict) and reason.get("type") == "abort":
            return ("finish_reason abort: " + str(reason.get("message", "")))[:200]
    low = text.lower()
    for message in ENGINE_ABORT_MESSAGES:
        if message in low:
            return message
    return None


def backend_url(plan, name):
    """URL of the Backend endpoint `name` next to the plan's callback endpoint.

    The callback endpoint, checked against callback_targets, is the Backend's
    /symmetric_pd/events; the same origin serves /symmetric_pd/migration and /symmetric_pd/prepared.

    Args:
        plan: A validated ForwardPD plan.
        name: Endpoint name under /symmetric_pd, e.g. "migration".

    Returns:
        The endpoint URL.
    """
    return plan.callback.endpoint.rsplit("/", 1)[0] + "/" + name


class PDForward:
    """Own both uploads, completion callbacks and the D output connection.

    The Backend retains the reservation and owns engine-side abort/reconciliation.
    Closing HTTP here is only local cleanup, not proof an engine stopped computing.
    One exception: after start() gave up on a prefill failure, the D request this
    exchange submitted is aborted on its D engine again once that engine can have
    it (see _reabort_decode), because the Backend's aborts can arrive first.
    """

    # A finite grace for the *original* responses, never a generation retry.
    # Callback delivery retains its existing independent bounded HTTP attempts.
    CANCEL_GRACE_SECONDS = 30
    # Re-abort of the D request after a prefill failure in start(): the first
    # wait between attempts, its cap (doubling), and the whole window.
    REABORT_FIRST_DELAY_SECONDS = 0.05
    REABORT_MAX_DELAY_SECONDS = 1.0
    REABORT_WINDOW_SECONDS = 10

    def __init__(self, service, plan):
        """Register ownership of a split P/D exchange and its callback-delivery slot.

        Args:
            service: Tokenizer service owning HTTP client and live exchanges.
            plan: Validated P/D execution plan with callback credentials.

        Returns:
            None; initializes uploads, response state, events, and cleanup ownership.

        Raises:
            ProtocolError: The callback outbox cannot reserve this plan.
        """
        self.service, self.plan = service, plan
        self.delivery = service.callback_outbox.reserve(plan)
        service.pd_exchanges.add(self)
        self.uploads, self.tasks, self.responses = {}, {}, {}
        self.terminal = set()
        self.reported = set()
        self.started = set()
        self.events = asyncio.Queue()
        self.reporter = None
        self.closed = False
        self.cleanup_task = None
        # Survive handoff from the downstream iterator to the sole cleanup reader.
        self.pending = bytearray()
        self.last_data = b""
        self.saw_done = False
        self.decode_terminal = False
        # The D engine's own abort/error event was read: terminal engine evidence
        # for the D-leg, recorded before any await so a cancellation that lands
        # while the event is handled (e.g. inside migrated()) cannot lose it.
        self.decode_aborted = False
        # start() gave up because the prefill failed while the D submit could
        # still be in flight: the cleanup re-aborts the D request on its engine.
        self.reabort_decode = False
        self.decode_body = bytearray()
        self.decode_reader = asyncio.Lock()
        # A streamed request keeps its engine input (minus the closing brace) and
        # the output tokens sent downstream, so the rest can be resent when the
        # Backend migrates its D-leg. Images also remain available for a worker
        # recomputing failed imports, even when the stream cannot be resumed.
        self.input_body = None
        self.resumable = False
        self.generated = []
        self.stitched = {}
        # Late D choose: one D-leg per candidate. The admitted one becomes the
        # decode response; the bytes read to recognize it are replayed first.
        self.candidate_tasks = []
        self.candidate_responses = {}
        self.candidate_reported = set()
        self.decode_winner = None
        self.decode_prefix = bytearray()
        self.loser_task = None

    async def prepare(self, entry):
        """Convert one taken input entry into two uploads sharing common serialized bytes.

        JSON conversion runs in a worker thread. On success the entry body is
        cleared and shared upload ownership retains the necessary bytes.

        Args:
            entry: Store Entry whose serialized input ownership has been transferred here.

        Returns:
            None; initializes stream mode and both upload objects.
        """
        def build():
            """Build the shared input prefix and P/D-specific JSON suffixes off the event loop.

            Reads the enclosing entry and plan. Keeps sampling parameters intact and
            adds only the selected leg's stream and KV-transfer fields.

            Args:
                None.

            Returns:
                A tuple of common-prefix, prefill-suffix, and decode-suffix bytes.
            """
            payload = orjson.loads(entry.data)
            payload.pop("stream", None)
            # Retention identity is controlled by the router, never staged input.
            payload.pop("kv_session", None)
            if "image_data" in payload:
                if not self.service.config.public_url:
                    raise ProtocolError(422, "UNSUPPORTED_INPUT", "Image PD requires a worker-reachable public_url.")
                payload["prepared_image_source"] = {
                    "endpoint": self.service.config.public_url.rstrip("/") + "/image",
                    "protocol_version": 1, "config_id": self.plan.config_id,
                    "request_id": self.plan.request_id, "ref": self.plan.ref,
                }
            # Common includes all sampling params, including max_new_tokens. SGLang's
            # symmetric export mode stops after the first token and needs its metadata.
            common = orjson.dumps(payload)[:-1]
            # Keep original images with the resumable input, but omit their
            # bytes from D's upload. Both workers receive the same layout.
            if "image_data" in payload:
                payload.pop("image_data")
                decode_common = orjson.dumps(payload)[:-1]
            else:
                decode_common = common
            p = {"stream": False, "kv_export": self.plan.prefill.kv_export}
            if self.plan.prefill.pull_hint is not None:
                p["kv_pull_hint"] = self.plan.prefill.pull_hint
            if self.plan.prefill.late_d is not None:
                p["late_d"] = self.plan.prefill.late_d
            if self.plan.prefill.kv_session is not None:
                p["kv_session"] = self.plan.prefill.kv_session.model_dump()
            if self.plan.decode_candidates is not None:
                # Late D: no D-leg carries the session tag; the prefill sends
                # it to the admitted candidate with ADMIT.
                ds = [{"stream": entry.stream, "kv_import": c.kv_import, "late_d": c.late_d}
                      for c in self.plan.decode_candidates]
            else:
                d = {"stream": entry.stream, "kv_import": self.plan.decode.kv_import}
                if self.plan.decode.kv_session is not None:
                    d["kv_session"] = self.plan.decode.kv_session.model_dump()
                ds = [d]
            return (common, decode_common, b"," + orjson.dumps(p)[1:],
                    [b"," + orjson.dumps(d)[1:] for d in ds])

        common, decode_common, p_suffix, d_suffixes = await asyncio.to_thread(build)
        entry.data = b""
        self.stream = entry.stream
        self.resumable = self.stream
        if self.stream or decode_common is not common:
            self.input_body = common
        shared = decode_common is common
        k = len(d_suffixes)
        common = SharedBody(common, self.service, owners=1 + k if shared else 1)
        decode_common = common if shared else SharedBody(decode_common, self.service, owners=k)
        self.uploads = {
            "prefill": SharedUpload(common, p_suffix, lambda: self.report("prefill", "uploaded")),
        }
        if self.plan.decode_candidates is None:
            self.uploads["decode"] = SharedUpload(
                decode_common, d_suffixes[0], lambda: self.report("decode", "uploaded"))
        else:
            # A candidate's upload is not the D-leg's: "uploaded" is reported
            # for the admitted candidate when it is recognized.
            for i, suffix in enumerate(d_suffixes):
                self.uploads[f"decode#{i}"] = SharedUpload(decode_common, suffix, lambda: None)

    def event(self, leg, state, *, http_status=None, error=None, terminal_response=False,
              not_submitted=False, usage=None, engine_abort=None):
        """Build an idempotently identified execution-state callback envelope.

        Args:
            leg: Execution leg name: prefill or decode.
            state: Backend execution state to report.
            http_status: Optional observed engine HTTP status.
            error: Optional non-sensitive error label.
            terminal_response: Whether engine terminal response evidence was observed.
            not_submitted: Whether this leg is proven never to have submitted engine work.
            usage: Optional engine token counts of a completed leg (leg_usage).
            engine_abort: Optional engine abort message of a failed answer (the router
                decides whether it is terminal, e.g. HTTP 503 "aborted by client").

        Returns:
            A new metadata dictionary without performing network I/O.
        """
        event = {
            "protocol_version": 1,
            "request_id": self.plan.request_id,
            "reservation_id": self.plan.reservation_id,
            "transfer_id": self.plan.transfer_id,
            "leg": leg, "state": state,
            "event_id": f"{self.plan.reservation_id}:{leg}:{state}",
        }
        if http_status is not None:
            event["http_status"] = http_status
        if error is not None:
            event["error"] = error
        if terminal_response:
            event["terminal_response"] = True
        if not_submitted:
            event["not_submitted"] = True
        if usage:
            event["usage"] = usage
        if engine_abort:
            event["engine_abort"] = engine_abort
        return event

    def report_candidate(self, idx, state, **fields):
        """Queue a late-D candidate event (bound or failed) once per candidate and state.

        Args:
            idx: The candidate's index (`late_d.cand_idx`).
            state: "bound" (its D-leg is the request's) or "failed".
            **fields: Optional evidence fields accepted by event.

        Returns:
            None; candidate events never change the legs' terminal sets.
        """
        if (idx, state) in self.candidate_reported:
            return
        self.candidate_reported.add((idx, state))
        event = self.event("decode", state, **fields)
        event["candidate"] = idx
        event["event_id"] = f"{self.plan.reservation_id}:decode#{idx}:{state}"
        self.events.put_nowait(event)

    async def candidate(self, idx):
        """POST one late-D candidate's D-leg and read up to its first event.

        Args:
            idx: The candidate's index.

        Returns:
            (idx, outcome, response, head): outcome is "winner" (generation
            output: the prefill admitted it), "released", "failed" (an HTTP
            error, an abort or a truncated answer) or "transport" (no answer;
            engine admission unknown); head holds the bytes read.

        A non-stream request is classified only from its whole body, so its
        bind is reported when the decode has finished; late D under external
        role control is meant for streamed requests.
        """
        upload = self.uploads[f"decode#{idx}"]
        target = self.plan.decode_candidates[idx].target
        try:
            response = await self.service.client.post(
                target.endpoint, data=upload,
                headers={"Content-Type": "application/json", "Content-Length": str(upload.size)},
                allow_redirects=False,
            )
        except (aiohttp.ClientError, TimeoutError) as exc:
            LOG.warning("pd_submit_error rid=%r leg=decode#%d exception=%s",
                        self.plan.request_id, idx, type(exc).__name__)
            return idx, "transport", None, b""
        self.candidate_responses[idx] = response
        if response.status == 409:
            # /v1/completions-style release: HTTP 409 with X-Late-D-Reason, or
            # an error body naming late_d_released; any other 409 is a failure.
            if response.headers.get("X-Late-D-Reason", "").lower() == "released":
                return idx, "released", response, b""
            try:
                body = await response.content.read(self.service.config.max_response_bytes)
            except (aiohttp.ClientError, TimeoutError):
                body = b""
            released = LATE_D_RELEASED_ERR.encode() in body
            return idx, "released" if released else "failed", response, b""
        if not 200 <= response.status < 300:
            return idx, "failed", response, b""
        head = bytearray()
        data = None
        try:
            while True:
                part = await response.content.readany()
                if not part:
                    if not self.stream:
                        data = bytes(head)
                    break
                head.extend(part)
                if len(head) > self.service.config.max_response_bytes:
                    return idx, "failed", response, b""
                if self.stream and (data := first_data(head)) is not None:
                    break
        except (aiohttp.ClientError, TimeoutError):
            return idx, "transport", response, b""
        return idx, late_d_outcome(data) or "winner", response, bytes(head)

    def close_candidate(self, idx):
        """Close a loser candidate's response, if open."""
        response = self.candidate_responses.pop(idx, None)
        if response is not None:
            response.close()

    async def submit_candidates(self):
        """Send every late-D candidate's D-leg and return the admitted one's response.

        The first candidate whose first event is generation output is the
        request's D-leg ("bound" for it, then "uploaded" for the decode leg).
        Released candidates are closed silently, failed ones reported so the
        router leaves them out of a later decision. Candidates still open after
        the bind finish in the background for up to LATE_D_LOSER_DRAIN_SECONDS.

        Returns:
            The admitted candidate's open response.

        Raises:
            ProtocolError: No candidate was admitted (terminal decode evidence
                when every candidate answered).
        """
        self.started.add("decode")
        tasks = [asyncio.create_task(self.candidate(i))
                 for i in range(len(self.plan.decode_candidates))]
        self.candidate_tasks = tasks
        pending = set(tasks)
        answered = True
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    idx, outcome, response, head = task.result()
                    if outcome == "winner" and self.decode_winner is None:
                        self.decode_winner = idx
                        self.candidate_responses.pop(idx, None)
                        self.responses["decode"] = response
                        self.decode_prefix.extend(head)
                        self.report_candidate(idx, "bound")
                        self.report("decode", "uploaded")
                        continue
                    answered &= outcome != "transport"
                    self.close_candidate(idx)
                    if outcome != "released" and self.decode_winner is None:
                        self.report_candidate(
                            idx, "failed",
                            http_status=None if response is None else response.status)
                if self.decode_winner is not None:
                    if pending:
                        self.loser_task = asyncio.create_task(self.finish_candidates(pending))
                    return self.responses["decode"]
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        self.report("decode", "failed", error="NO_CANDIDATE_ADMITTED", terminal_response=answered)
        error = ProtocolError(502, "PD_SUBMIT_FAILED", "No decode candidate was admitted")
        if answered:
            error.terminal_response = True
            error.terminal_leg = "decode"
        raise error

    async def finish_candidates(self, pending):
        """Close the loser candidates after the bind, within the loser drain."""
        try:
            async with asyncio.timeout(LATE_D_LOSER_DRAIN_SECONDS):
                for task in asyncio.as_completed(pending):
                    with contextlib.suppress(Exception):
                        idx, *_ = await task
                        self.close_candidate(idx)
        except TimeoutError:
            LOG.warning("late_d_loser_drain_timeout rid=%r open=%d",
                        self.plan.request_id, sum(not t.done() for t in pending))
        finally:
            for task in pending:
                task.cancel()
            for idx in list(self.candidate_responses):
                self.close_candidate(idx)

    def report(self, leg, state, **fields):
        """Queue a distinct execution-state update and record confirmed terminal evidence.

        A failed/cancelled transport alone is not confirmed engine termination;
        terminal legs ignore later reports.

        Args:
            leg: Execution leg name: prefill or decode.
            state: Backend execution state to report.
            **fields: Optional evidence fields accepted by event.

        Returns:
            None; updates local deduplication/terminal sets and the callback queue.
        """
        if leg in self.terminal:
            return
        confirmed = state == "completed" or (state == "failed" and
                    (fields.get("terminal_response") or fields.get("not_submitted")))
        key = leg, state, bool(confirmed)
        if key in self.reported:
            return
        self.reported.add(key)
        if confirmed:
            self.terminal.add(leg)
        self.events.put_nowait(self.event(leg, state, **fields))

    async def send_event(self, event):
        """Send a claim immediately and require its acknowledgement before engine work.

        Does not defer claims into the retry outbox or authorize delayed work.

        Args:
            event: Claim metadata envelope built for this plan.

        Returns:
            True if the callback was accepted; otherwise False.
        """
        return await self.delivery.send_immediate(event)

    async def callbacks(self):
        """Sequentially deliver queued execution metadata until the shutdown sentinel.

        The delivery outbox retries idempotent state notifications only.

        Args:
            None.

        Returns:
            None after the queue supplies None.
        """
        while (event := await self.events.get()) is not None:
            await self.delivery.deliver(event)

    async def submit(self, leg):
        """POST one prepared leg to its selected engine exactly once.

        Marks the leg started before I/O. A transport failure can leave engine
        admission unknown; this method never retries generation.

        Args:
            leg: prefill or decode, selecting upload body and target.

        Returns:
            The open successful aiohttp response, retained for subsequent reading.

        Raises:
            ProtocolError: The engine returns a non-success HTTP status.
        """
        self.started.add(leg)
        upload = self.uploads[leg]
        target = getattr(self.plan, leg).target
        try:
            response = await self.service.client.post(
                target.endpoint, data=upload,
                headers={"Content-Type": "application/json", "Content-Length": str(upload.size)},
                allow_redirects=False,
            )
        except (aiohttp.ClientError, TimeoutError) as exc:
            LOG.warning("pd_submit_error rid=%r leg=%s exception=%s",
                        self.plan.request_id, leg, type(exc).__name__)
            raise
        self.responses[leg] = response
        if not 200 <= response.status < 300:
            abort = None
            if response.status == 503:
                # A symmetric prefill abort (late D) answers HTTP 503; pass its
                # message so the router can retire the P reservation.
                with contextlib.suppress(aiohttp.ClientError, TimeoutError):
                    abort = engine_abort_message(await response.content.read(ENGINE_ABORT_BODY_BYTES))
            self.report(leg, "failed", http_status=response.status, engine_abort=abort)
            raise ProtocolError(502, "PD_SUBMIT_FAILED", f"{leg} returned HTTP {response.status}")
        return response

    async def prefill(self):
        """Submit prefill, consume its bounded JSON result, and report completion evidence.

        Reports cancellation/failure metadata and always closes its HTTP response.

        Args:
            None.

        Returns:
            None after a valid terminal prefill response.

        Raises:
            ProtocolError: Submission, response size, or terminal-response validation fails.
            asyncio.CancelledError: The task is cancelled; cancellation is reported then
                propagated.
        """
        try:
            response = await self.submit("prefill")
            data = bytearray()
            async for part in response.content.iter_any():
                if len(data) + len(part) > self.service.config.max_response_bytes:
                    raise ProtocolError(502, "FORWARD_IO_ERROR", "Prefill response exceeds limit")
                data.extend(part)
            # An HTTP-200 engine error must not falsely release a completed reservation.
            self.check_result(data, "prefill")
            self.report("prefill", "completed", http_status=response.status,
                        usage=response_usage(data, "prefill"))
        except asyncio.CancelledError:
            self.report("prefill", "cancelled")
            raise
        except BaseException as exc:
            self.report("prefill", "failed", error=type(exc).__name__,
                        terminal_response=getattr(exc, "terminal_leg", None) == "prefill")
            raise
        finally:
            if "prefill" in self.responses:
                self.responses["prefill"].close()

    async def start(self):
        # Backend atomically claims both reserved legs. An unclaimed/expired plan
        # must never reach an engine, even if its delayed response reaches us later.
        """Claim both reservations and start P/D concurrently.

        Neither engine is submitted before claim acknowledgement.

        Args:
            None.

        Returns:
            The open successful decode response.

        Raises:
            ProtocolError: Claim or submission fails.
        """
        try:
            claimed = await self.send_event(self.event("prefill", "claimed"))
        except BaseException:
            await self.release_unsubmitted()
            raise
        if not claimed:
            await self.release_unsubmitted()
            raise ProtocolError(502, "PD_CLAIM_FAILED", "Backend did not accept the P/D plan claim")
        self.reporter = asyncio.create_task(self.callbacks())
        self.tasks["prefill"] = asyncio.create_task(self.prefill())
        self.tasks["decode"] = asyncio.create_task(
            self.submit("decode") if self.plan.decode_candidates is None else self.submit_candidates())
        try:
            done, _ = await asyncio.wait(self.tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            if self.tasks["prefill"] in done:
                try:
                    self.tasks["prefill"].result()
                except BaseException:
                    # The D submit continues (or has just finished); the Backend's
                    # aborts for the failure reports can reach the D engine before
                    # this D request does. The cleanup aborts it again.
                    self.reabort_decode = self.plan.decode_candidates is None
                    raise
            return await asyncio.shield(self.tasks["decode"])
        except BaseException as exc:
            self.report("decode", "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                        error=type(exc).__name__)
            raise

    async def response(self):
        """Start the exchange and construct the D response.

        Nonstreaming completion also waits for validated prefill completion.

        Args:
            None.

        Returns:
            A streaming PDForwardResponse or buffered HTTP Response from decode.

        Raises:
            ProtocolError: Claim, submission, response-size, or terminal validation fails.
        """
        upstream = await self.start()
        headers = {
            "X-Tokenizer-Protocol-Version": "1",
            "X-Tokenizer-Response-Origin": "decode",
        }
        if "content-type" in upstream.headers:
            headers["content-type"] = upstream.headers["content-type"]
        if self.stream:
            return PDForwardResponse(self, upstream.status, headers)
        async for part in self.chunks():
            if len(self.decode_body) + len(part) > self.service.config.max_response_bytes:
                self.report("decode", "failed", error="RESPONSE_TOO_LARGE")
                raise ProtocolError(502, "FORWARD_IO_ERROR", "Decode response exceeds buffer limit")
            self.decode_body.extend(part)
        try:
            self.check_result(self.decode_body, "decode")
        except ProtocolError as exc:
            self.report("decode", "failed", error=exc.code,
                        terminal_response=getattr(exc, "terminal_leg", None) == "decode")
            raise
        self.report("decode", "completed", http_status=upstream.status,
                    usage=response_usage(self.decode_body, "decode"))
        await asyncio.shield(self.tasks["prefill"])
        return Response(bytes(self.decode_body), status_code=upstream.status, headers=headers)

    async def release_unsubmitted(self):
        """Report that both legs were never submitted after a missing claim acknowledgement.

        Called only before engine POST tasks are created. It can resolve a claim
        whose acknowledgement was lost without asserting engine completion.

        Args:
            None.

        Returns:
            None after metadata is delivered or retained by the outbox.
        """
        # Called only before creating either engine POST task. Failed cleanup
        # metadata remains in the service outbox; no later claim is authorized.
        with anyio.CancelScope(shield=True):
            for leg in ("prefill", "decode"):
                await self.delivery.deliver(self.event(leg, "failed", not_submitted=True))

    @staticmethod
    def check_result(data, leg, *, require_terminal=True):
        """Validate an engine JSON result and recognize explicit terminal evidence.

        Args:
            data: JSON response or SSE data payload as bytes.
            leg: prefill or decode, used in errors and terminal evidence.
            require_terminal: Require a successful terminal finish reason when True.

        Returns:
            True for a length/stop finish reason; False for allowed intermediate output.

        Raises:
            ProtocolError: JSON is invalid, reports an engine error, or lacks required
                termination.
        """
        try:
            value = orjson.loads(data)
        except orjson.JSONDecodeError as exc:
            raise ProtocolError(502, "PD_INVALID_RESPONSE", f"{leg} returned invalid JSON") from exc
        meta = value.get("meta_info") if isinstance(value, dict) else None
        reason = meta.get("finish_reason") if isinstance(meta, dict) else None
        kind = reason.get("type") if isinstance(reason, dict) else None
        if kind in ("abort", "error") or (isinstance(value, dict) and "error" in value):
            error = ProtocolError(502, "PD_ENGINE_ERROR", f"{leg} returned an engine error")
            if kind in ("abort", "error"):
                error.terminal_response = True
                error.terminal_leg = leg
            raise error
        if require_terminal and kind not in ("length", "stop"):
            raise ProtocolError(502, "PD_INVALID_RESPONSE", f"{leg} has no terminal finish_reason")
        return kind in ("length", "stop")

    async def chunks(self, *, draining=False):
        # Acquire once per response, not per token. A cancelled downstream must
        # relinquish its iterator before cleanup can consume the same socket.
        """Read D output with one socket reader and validate streamed terminal evidence.

        A response-wide lock prevents concurrent downstream and cleanup readers.
        SSE parsing retains incomplete lines across ownership handoff; complete
        lines a cancelled reader left unparsed are parsed at EOF before the stream
        counts as truncated. Errors report execution metadata before propagating.
        An engine abort/error event is terminal D evidence from the moment it is
        read, even when the reader is cancelled while handling it.

        Args:
            draining: Read during cleanup without waiting on the prefill task in this iterator.

        Yields:
            Decode-response byte chunks: unmodified when buffered; whole lines when
            streamed, continued by the resumed request's events after a migration.

        Raises:
            ProtocolError: An event is too large, invalid, truncated, or contains an engine
                error.
        """
        await self.decode_reader.acquire()
        try:
            while True:
                if not draining and self.tasks["prefill"].done():
                    self.tasks["prefill"].result()
                if self.decode_prefix:
                    # Late D: the admitted candidate's bytes read to recognize it.
                    part = bytes(self.decode_prefix)
                    self.decode_prefix.clear()
                else:
                    part = await self.responses["decode"].content.readany()
                # At EOF, complete lines an interrupted reader left in `pending`
                # (e.g. `[DONE]` read together with an abort event) are parsed
                # before the stream is judged truncated.
                if not part and not (self.stream and b"\n" in self.pending):
                    break
                if not self.stream:
                    yield part
                    continue
                self.pending.extend(part)
                out = bytearray()
                while (end := self.pending.find(b"\n")) >= 0:
                    line = bytes(self.pending[:end]).strip()
                    del self.pending[:end + 1]
                    if len(line) > self.service.config.max_response_bytes:
                        raise ProtocolError(502, "FORWARD_IO_ERROR", "Decode event exceeds buffer limit")
                    data = line[5:].strip() if line.startswith(b"data:") else None
                    if data == b"[DONE]":
                        self.saw_done = True
                    elif data is not None:
                        self.last_data = data
                    # Intermediate finish_reason=null does not need JSON parsing.
                    reason = line.partition(b'"finish_reason"')[2].lstrip()
                    has_reason = reason.startswith(b":") and reason[1:].lstrip().startswith(b"{")
                    if data is not None and (has_reason or b'"error"' in line or b'"abort"' in line):
                        try:
                            self.decode_terminal |= self.check_result(data, "decode", require_terminal=False)
                        except ProtocolError as exc:
                            # Record the engine's terminal evidence before awaiting
                            # the Backend: a cancellation inside migrated() then still
                            # reports a terminal D-leg instead of an unknown one.
                            if getattr(exc, "terminal_leg", None) == "decode":
                                self.decode_aborted = True
                            if draining or not await self.migrated(exc):
                                raise
                            # The abort event is replaced by the resumed request's events.
                            if out:
                                yield bytes(out)
                            async with contextlib.aclosing(self.resume()) as parts:
                                async for piece in parts:
                                    yield piece
                            return
                    if data is not None and data != b"[DONE]":
                        self.note_output(data)
                    out += line + b"\n"
                if max(len(self.pending), len(self.last_data)) > self.service.config.max_response_bytes:
                    raise ProtocolError(502, "FORWARD_IO_ERROR", "Decode event exceeds buffer limit")
                if out:
                    yield bytes(out)
                if not part:
                    break  # EOF after the left-over complete lines
            if self.stream:
                if self.pending or not self.saw_done:
                    raise ProtocolError(502, "PD_INVALID_RESPONSE", "Truncated decode event stream")
                self.check_result(self.last_data, "decode")
                # The last event carries the request's final counts.
                self.report("decode", "completed", http_status=self.responses["decode"].status,
                            usage=response_usage(self.last_data, "decode"))
                if not draining:
                    await asyncio.shield(self.tasks["prefill"])
        except (asyncio.CancelledError, GeneratorExit):
            if self.decode_aborted:
                # The engine already ended the D-leg (abort/error event read).
                self.report("decode", "failed", error="PD_ENGINE_ERROR", terminal_response=True)
            elif self.decode_terminal:
                self.report("decode", "failed", error="CLIENT_DISCONNECTED", terminal_response=True)
            else:
                self.report("decode", "cancelled")
            raise
        except BaseException as exc:
            self.report("decode", "failed", error=type(exc).__name__,
                        terminal_response=self.decode_terminal or self.decode_aborted
                        or getattr(exc, "terminal_leg", None) == "decode")
            raise
        finally:
            self.decode_reader.release()

    def note_output(self, data):
        """Record the output tokens of one decode event sent downstream.

        Only incremental streams (each event carries its new tokens, whose running
        total equals completion_tokens) can be resumed; anything else turns
        resumption off for this exchange.

        Args:
            data: SSE data payload bytes of a non-terminal or successful event.

        Returns:
            None.
        """
        if self.input_body is None or not self.resumable:
            return
        try:
            value = orjson.loads(data)
            ids, meta = value["output_ids"], value["meta_info"]
            if not all(type(token) is int and 0 <= token < 2**32 for token in ids):
                raise ValueError("invalid output token")
            self.generated.extend(ids)
            if meta["completion_tokens"] != len(self.generated):
                raise ValueError("not an incremental stream")
            if not self.stitched:
                self.stitched = {key: meta[key] for key in STITCHED_FIELDS if key in meta}
        except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
            self.resumable, self.generated = False, []

    async def migrated(self, exc):
        """Whether a decode abort is the Backend moving this D-leg to another worker.

        Args:
            exc: ProtocolError raised for a terminal decode event.

        Returns:
            True when the request can be resumed and the Backend confirms the
            migration; False otherwise, including when the Backend is unreachable.
        """
        if (getattr(exc, "terminal_leg", None) != "decode"
                or self.input_body is None or not self.resumable):
            return False
        url = backend_url(self.plan, "migration")
        try:
            async with asyncio.timeout(self.service.config.callback_timeout_seconds):
                async with self.service.client.post(
                    url, json={"reservation_id": self.plan.reservation_id},
                    headers={"Authorization": "Bearer " + self.plan.callback.token},
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        return False
                    value = await response.json()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            return False
        return isinstance(value, dict) and value.get("migrating") is True

    async def resume(self):
        """Run the rest of a migrated request on a new P/D pair and relay it as ours.

        The new request's input is ours plus the output tokens already sent
        downstream, with the remaining output budget (and forced tokens). Its
        events are rewritten so the client sees one stream: completion_tokens
        continues from ours, prompt and cached counts and the engine id stay ours,
        and the prompt token IDs are not repeated.

        Args:
            None.

        Yields:
            The resumed request's decode events, rewritten.

        Raises:
            ProtocolError: Preparing, placing or running the resumed request fails.
        """
        self.report("decode", "failed", error="MIGRATED", terminal_response=True)
        body, self.input_body = self.input_body, None
        sent, stitched = self.generated, self.stitched
        self.generated = []
        payload = await asyncio.to_thread(orjson.loads, body + b"}")
        del body
        follow = await self.service.continue_pd(self.plan, payload, sent)
        del payload
        LOG.info("pd_resumed rid=%r resumed_rid=%r sent_tokens=%d", self.plan.request_id,
                 follow.plan.request_id, len(sent))
        try:
            async with contextlib.aclosing(follow.chunks()) as parts:
                async for part in parts:
                    yield restitch(part, len(sent), stitched)
        finally:
            with anyio.CancelScope(shield=True):
                await follow.aclose()

    async def aclose(self):
        """Close exchange ownership or start bounded cleanup for unresolved engine responses.

        Repeated close awaits an existing cleanup task. Unknown engine outcomes
        remain unknown; local HTTP cleanup does not prove KV release.

        Args:
            None.

        Returns:
            None; cleanup may continue in a service-owned task after an early close.
        """
        if self.closed:
            if self.cleanup_task is not None:
                await asyncio.shield(self.cleanup_task)
            return
        self.closed = True
        if self.reporter and len(self.terminal) != 2:
            for leg, task in self.tasks.items():
                if leg not in self.started:
                    # This task has never entered submit(), so it cannot have admitted work.
                    task.cancel()
                    self.report(leg, "failed", not_submitted=True)
                else:
                    self.report(leg, "cancelled")
            # Not a child of Starlette's cancelled task group. The service owns it
            # and keeps the shared session/outbox alive through bounded shutdown.
            self.cleanup_deadline = asyncio.get_running_loop().time() + self.CANCEL_GRACE_SECONDS
            self.cleanup_task = asyncio.create_task(self._cleanup())
            return
        # Normal response completion can also be cancelled during callback delivery.
        # Keep one service-owned task so repeated close/shutdown can finish it.
        self.cleanup_task = asyncio.create_task(self._dispose())
        await asyncio.shield(self.cleanup_task)

    async def _drain_decode(self):
        """Consume the original decode response during cancellation cleanup.

        Does not resubmit generation. Already terminal decode legs need no read.

        Args:
            None.

        Returns:
            None; reports terminal completion or failure evidence when available.
        """
        if "decode" in self.terminal:
            return
        try:
            await asyncio.shield(self.tasks["decode"])
            async for part in self.chunks(draining=True):
                if not self.stream:
                    if len(self.decode_body) + len(part) > self.service.config.max_response_bytes:
                        raise ProtocolError(502, "FORWARD_IO_ERROR", "Decode response exceeds limit")
                    self.decode_body.extend(part)
            if not self.stream:
                self.check_result(self.decode_body, "decode")
                self.report("decode", "completed", http_status=self.responses["decode"].status,
                            usage=response_usage(self.decode_body, "decode"))
        except Exception as exc:
            self.report("decode", "failed", error=type(exc).__name__,
                        terminal_response=getattr(exc, "terminal_leg", None) == "decode")

    async def _reabort_decode(self):
        """Abort this exchange's D request on its D engine until the D-leg has ended.

        Runs beside the cleanup drain only after start() gave up on a prefill
        failure. The Backend aborts both engines when it receives the failure
        reports (and once more for a D upload reported after them), but an engine
        ignores an abort for a request it has not admitted yet. The D body can
        still be in flight or being parsed at that time, so the D engine could keep
        the request (waiting for KV that never comes) and no terminal D evidence
        would reach the Backend. This sends the same best-effort engine abort
        (POST <engine>/abort_request {"rid"}) at once, again as soon as the D
        submit answers, and then with doubling waits, until the drain has read
        terminal D evidence (the engine's abort event, reported with
        terminal_response by chunks()/_drain_decode), the D submit failed, or
        REABORT_WINDOW_SECONDS passed. An abort acknowledgement is never terminal
        evidence. The cleanup cancels this task when its drain ends.

        Args:
            None.

        Returns:
            None; logs one pd_decode_reabort line.
        """
        endpoint = self.plan.decode.target.endpoint
        if not endpoint.endswith("/generate"):
            return  # not an SGLang /generate target: no known abort endpoint
        url = endpoint[:-len("/generate")] + "/abort_request"
        submit = self.tasks["decode"]
        loop = asyncio.get_running_loop()
        end = loop.time() + self.REABORT_WINDOW_SECONDS
        delay = self.REABORT_FIRST_DELAY_SECONDS
        sent = acked = 0
        try:
            while "decode" not in self.terminal and loop.time() < end:
                if submit.done() and (submit.cancelled() or submit.exception() is not None):
                    break  # no D answer to end; submit() reported its own failure
                sent += 1
                acked += await self._abort_decode_engine(url)
                wait = min(delay, end - loop.time())
                if "decode" in self.terminal or wait <= 0:
                    break
                if submit.done():
                    await asyncio.sleep(wait)
                else:
                    # A D answer means the engine has the request: abort again at once.
                    await asyncio.wait({submit}, timeout=wait)
                delay = min(delay * 2, self.REABORT_MAX_DELAY_SECONDS)
        finally:
            LOG.warning("pd_decode_reabort rid=%r aborts=%d acked=%d answered=%s decode_terminal=%s",
                        self.plan.request_id, sent, acked,
                        submit.done() and not submit.cancelled() and submit.exception() is None,
                        "decode" in self.terminal)

    async def _abort_decode_engine(self, url):
        """POST one best-effort engine abort for this exchange's request ID.

        Args:
            url: The D engine's /abort_request URL.

        Returns:
            1 when the engine answered 2xx (an acknowledgement, not terminal
            evidence), else 0.
        """
        try:
            async with asyncio.timeout(self.service.config.callback_timeout_seconds):
                async with self.service.client.post(
                    url, json={"rid": self.plan.request_id}, allow_redirects=False,
                ) as response:
                    return int(200 <= response.status < 300)
        except (aiohttp.ClientError, TimeoutError):
            return 0

    async def _cleanup(self):
        """Wait within the cancellation grace period for original P/D responses.

        Deadline expiry is not engine termination evidence. Unresolved legs are
        logged and remain subject to Backend reconciliation. After a prefill
        failure in start(), the D request is re-aborted beside the drain.

        Args:
            None.

        Returns:
            None after local disposal finishes.
        """
        reabort = None
        if self.reabort_decode and "decode" not in self.terminal:
            reabort = asyncio.create_task(self._reabort_decode())
        try:
            async with asyncio.timeout_at(self.cleanup_deadline):
                await asyncio.gather(asyncio.shield(self.tasks["prefill"]),
                                     self._drain_decode(), return_exceptions=True)
        except TimeoutError:
            pass  # Deadline is local resource cleanup, never engine terminal evidence.
        finally:
            if reabort is not None:
                reabort.cancel()
                await asyncio.gather(reabort, return_exceptions=True)
            if len(self.terminal) != 2:
                LOG.warning("pd_cleanup_unknown rid=%r terminal_legs=%s",
                            self.plan.request_id, sorted(self.terminal))
            await self._dispose()

    async def _dispose(self):
        """Cancel local tasks, close responses/uploads, and finish callback ownership.

        Unacknowledged callback metadata can remain owned by the outbox after this
        exchange is removed. Local disposal does not certify engine-side cleanup.

        Args:
            None.

        Returns:
            None; releases exchange buffers and removes the live service registration.
        """
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        late = [*self.candidate_tasks, *([self.loser_task] if self.loser_task else [])]
        for task in late:
            if not task.done():
                task.cancel()
        if late:
            await asyncio.gather(*late, return_exceptions=True)
        for idx in list(self.candidate_responses):
            self.close_candidate(idx)
        for response in self.responses.values():
            response.close()
            await response.wait_for_close()
        for upload in self.uploads.values():
            await upload.aclose()
        if self.reporter:
            for leg in ("prefill", "decode"):
                self.report(leg, "cancelled")
            self.events.put_nowait(None)
            await self.reporter
        self.delivery.finish()
        self.service.pd_exchanges.discard(self)
        self.pending.clear()
        self.last_data = b""
        self.decode_body.clear()
        self.input_body = None
        self.generated.clear()


def restitch(part, sent, stitched):
    """Rewrite a resumed request's whole SSE lines as events of the original request.

    Args:
        part: Whole lines (each ending in a newline) from the resumed decode stream.
        sent: Output tokens the original request had already sent downstream.
        stitched: The original's STITCHED_FIELDS metadata values.

    Returns:
        The lines with completion_tokens offset by sent, the stitched fields
        restored and, after sent tokens, prompt_token_ids removed; [DONE] and
        non-data lines unchanged.
    """
    out = bytearray()
    for line in part.split(b"\n")[:-1]:
        data = line[5:].strip() if line.startswith(b"data:") else None
        if data is not None and data != b"[DONE]":
            value = orjson.loads(data)
            if isinstance(value, dict):
                if sent:
                    value.pop("prompt_token_ids", None)
                meta = value.get("meta_info")
                if isinstance(meta, dict):
                    if type(meta.get("completion_tokens")) is int:
                        meta["completion_tokens"] += sent
                    meta.update(stitched)
            line = b"data: " + orjson.dumps(value)
        out += line + b"\n"
    return bytes(out)


class PDForwardResponse(StreamingResponse):
    def __init__(self, exchange, status, headers):
        """Wrap the exchange's decode iterator as an HTTP streaming response.

        Args:
            exchange: PDForward owning decode data and cleanup.
            status: HTTP status received from decode.
            headers: Downstream response headers.

        Returns:
            None; records the exchange and configures response delivery.
        """
        self.exchange = exchange
        super().__init__(exchange.chunks(), status_code=status, headers=headers)

    async def __call__(self, scope, receive, send):
        """Serve decode output and shield exchange cleanup from disconnect cancellation.

        Closes the response iterator before handing its socket to bounded cleanup.

        Args:
            scope: ASGI connection/request scope.
            receive: ASGI callable receiving client messages.
            send: ASGI callable sending response messages.

        Returns:
            None; emits ASGI response messages.
        """
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette's disconnect task cancels its task group. Finish local and
            # bounded callback cleanup even when cancellation is level-triggered.
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
                await self.exchange.aclose()
