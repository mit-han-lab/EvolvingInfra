"""Async client for the Rust runtime; engine adaptation and state stay in Rust."""
from collections.abc import AsyncIterator
import asyncio
import json
import logging
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

import aiohttp

from evolving_infra.router.client.types import Request
from evolving_infra.router.client.agent.identity import RequestIdentity

LOG = logging.getLogger(__name__)


class PreparedHTTPError(RuntimeError):
    """A prepared placement or forwarding endpoint answered with a non-success status.

    Attributes:
        status: The HTTP status the endpoint returned.
    """

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


async def _prepared_forward_error(response):
    """Preserve bounded protocol diagnostics without echoing arbitrary bodies."""
    message = f"prepared forwarding returned HTTP {response.status}"
    data = bytearray()

    async def read_envelope():
        while len(data) <= 4096:
            part = await response.content.read(4097 - len(data))
            if not part:
                break
            data.extend(part)

    try:
        await asyncio.wait_for(read_envelope(), 1)
        if len(data) > 4096:
            return PreparedHTTPError(message + " (error envelope exceeds limit)", response.status)
        body = json.loads(data)
        error = body.get('error') if isinstance(body, dict) else None
        if isinstance(error, dict):
            code, detail = error.get('code'), error.get('message')
            if isinstance(code, str) and re.fullmatch(r'[A-Z_]{1,64}', code):
                message += ' [' + code + ']'
            if isinstance(detail, str) and re.fullmatch(
                    r'[A-Za-z][A-Za-z0-9_]{0,79}(?:Error|Exception)|(?:prefill|decode) returned HTTP [1-5][0-9]{2}', detail):
                message += ' ' + detail
    except (ValueError, RecursionError, aiohttp.ClientError, asyncio.TimeoutError):
        pass  # Keep the original status even when its body is not usable.
    return PreparedHTTPError(message, response.status)


def _identifier(value: Any) -> bool:
    """Check whether a wire identifier is a nonempty bounded string.

    Args:
        value: Untrusted identifier value.

    Returns:
        True for strings of 1 through 512 characters; otherwise False.
    """
    return isinstance(value, str) and 0 < len(value) <= 512


def _prepared_description(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy the compact tokenizer descriptor for runtime placement.

    Rejects raw token arrays and unexpected fields; it does not contact either
    the tokenizer or runtime.

    Args:
        prepared: Untrusted descriptor returned by the input-preparation service.

    Returns:
        A shallow copy of the validated descriptor.

    Raises:
        ValueError: The descriptor shape, identity, counts, or hashes are invalid.
    """
    fields = {
        "protocol_version", "node_id", "request_id", "ref", "config_id",
        "prompt_tokens", "max_new_tokens", "content", "expires_in_ms",
    }
    if not isinstance(prepared, Mapping) or set(prepared) != fields:
        raise ValueError("expected a complete tokenizer preparation descriptor")
    body = dict(prepared)
    if type(body["protocol_version"]) is not int or body["protocol_version"] != 1:
        raise ValueError("unsupported preparation protocol")
    if any(not _identifier(body[key]) for key in ("node_id", "request_id", "ref", "config_id")):
        raise ValueError("invalid preparation identity")
    if (any(type(body[key]) is not int or body[key] < 0
            for key in ("prompt_tokens", "max_new_tokens", "expires_in_ms"))
            or not body["prompt_tokens"] or not body["expires_in_ms"]):
        raise ValueError("invalid preparation counts")
    content = body["content"]
    if not isinstance(content, dict) or set(content) != {"namespace", "hash_spec", "blocks"}:
        raise ValueError("invalid preparation content metadata")
    if not _identifier(content["namespace"]):
        raise ValueError("invalid preparation namespace")
    blocks, spec = content["blocks"], content["hash_spec"]
    spec_fields = {
        "mode", "page_size", "model", "cache_namespace", "implementation", "encoding",
        "is_eagle", "partial_blocks", "window_tokens", "digest", "chain",
        "image_config",
    }
    if (not isinstance(spec, dict) or not isinstance(blocks, list)
            or set(spec) - spec_fields
            or any(value is not None and type(value) not in (str, int, bool)
                   for key, value in spec.items() if key != "image_config")
            or any(not isinstance(block, dict) or set(block) != {"hash", "valid_tokens"}
                   or type(block["hash"]) not in (str, int)
                   or type(block["valid_tokens"]) is not int or block["valid_tokens"] < 1
                   for block in blocks)):
        raise ValueError("expected block hashes, not raw input tokens")
    if "image_config" in spec:
        image = spec["image_config"]
        if isinstance(image, dict) and image.get("layout_type") == "kimi_k25":
            # Validate the same checkpoint contract that T and the engine use.
            # The shared CPU module imports neither SGLang nor Torch.
            from evolving_infra.router.frontend._engine import image_layout

            image_config = image_layout().image_config

            try:
                normalized = image_config(image, image["image_token_id"])
                if normalized != image:
                    raise ValueError("Kimi image contract requires canonical fields")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid image preprocessing contract") from error
        else:
            fields = {"image_token_id", "vision_patch_size", "vision_downsample_ratio",
                      "vision_max_n_token", "vision_min_pixels", "vision_max_wh_ratio"}
            if (not isinstance(image, dict) or set(image) != fields
                    or any(type(value) is not int or value <= 0 for key, value in image.items()
                           if not (key == "vision_max_wh_ratio" and value is None))):
                raise ValueError("invalid image preprocessing contract")
    return body


_INVALID_PLAN = "invalid prepared execution plan"
# The router's late-D link fields, equal on the P-leg and on every candidate
# (router/core/src/late_bind.rs prefill_field / decode_field).
_LATE_D_LINK = ("v", "mode", "router_epoch", "k")
# Candidates per request: router late_bind::MAX_CANDIDATES, tokenizer ForwardPD.
_LATE_D_MAX_CANDIDATES = 64


def _leg_endpoint(leg: dict, mark_name: str, config_id: Any, transfer: int) -> str:
    """Check one leg's SGLang target and KV mark against the plan.

    Args:
        leg: Untrusted leg object that has the "target" and mark_name keys.
        mark_name: "kv_export" (P-leg) or "kv_import" (D-leg or candidate).
        config_id: The descriptor's configuration identifier.
        transfer: The plan's validated transfer ID.

    Returns:
        The leg's normalized endpoint.

    Raises:
        ValueError, TypeError, AttributeError: The target or the mark is
            malformed or names another configuration or transfer.
    """
    target, mark = leg["target"], leg[mark_name]
    if (not isinstance(target, dict) or set(target) != {"transport", "endpoint", "config_id"}
            or target["transport"] != "sglang_generate"
            or target["config_id"] != config_id
            or not isinstance(mark, dict) or type(mark.get("transfer_id")) is not int
            or mark["transfer_id"] != transfer):
        raise ValueError(_INVALID_PLAN)
    return endpoint_url(target["endpoint"])


def _late_d_candidate_endpoints(plan: dict, config_id: Any, transfer: int) -> list[str]:
    """Check the late-D fields of a plan whose common fields and two legs passed.

    With ``--pd-late-decode choose`` under external role control the router
    (router/core/src/backend/execution.rs ``Executions::install``; session tag in
    router/core/src/server/routes/prepared.rs ``plan``) adds to the classic plan:
    ``prefill.late_d`` {v, mode, router_epoch, k, default_idx, local_ok: false};
    ``prefill.kv_export.candidates``, one entry per candidate; ``decode_candidates``,
    one D-leg {target, kv_import, late_d} per candidate in ``cand_idx`` order,
    its ``late_d`` {v, mode, router_epoch, k, cand_idx}; and ``decode`` = the
    arrival pick, the target and kv_import of candidate ``default_idx``. The
    session tag rides on the P-leg only. As the engine's late_d check, this
    requires the late_d fields it reads and ignores others, with local_ok false
    when absent; the export entries may come in any order but name each
    candidate once (symmetric marks). The engine owns the values of v and mode
    and the KV mark contents. Otherwise this checks the structure the tokenizer
    relies on (protocol.ForwardPD.matching_transfer) and the router's
    invariants between the legs.

    Args:
        plan: Plan with validated identity, transfer ID, callback and legs.
        config_id: The descriptor's configuration identifier.
        transfer: The plan's validated transfer ID.

    Returns:
        The candidates' normalized endpoints in ``cand_idx`` order.

    Raises:
        ValueError, TypeError, AttributeError, LookupError: A late-D field is
            malformed or inconsistent with the other legs.
    """
    candidates, late = plan["decode_candidates"], plan["prefill"]["late_d"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= _LATE_D_MAX_CANDIDATES:
        raise ValueError(_INVALID_PLAN)
    k = len(candidates)
    if (not isinstance(late, dict) or not {*_LATE_D_LINK, "default_idx"} <= set(late)
            or type(late["v"]) is not int or late["v"] < 0
            or not _identifier(late["mode"])
            or type(late["router_epoch"]) is not int or not 0 <= late["router_epoch"] < 2**32
            or type(late["k"]) is not int or late["k"] != k
            or type(late["default_idx"]) is not int or not 0 <= late["default_idx"] < k
            # Prepared plans never offer LOCAL: the tokenizer streams only an
            # admitted candidate's D-leg.
            or late.get("local_ok", False) is not False):
        raise ValueError(_INVALID_PLAN)
    exports = plan["prefill"]["kv_export"].get("candidates")
    if (not isinstance(exports, list) or len(exports) != k
            or any(not isinstance(export, dict) or type(export.get("cand_idx")) is not int
                   for export in exports)
            or {export["cand_idx"] for export in exports} != set(range(k))):
        raise ValueError(_INVALID_PLAN)
    endpoints = []
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != {"target", "kv_import", "late_d"}:
            raise ValueError(_INVALID_PLAN)
        endpoints.append(_leg_endpoint(candidate, "kv_import", config_id, transfer))
        field = candidate["late_d"]
        if (not isinstance(field, dict) or not {*_LATE_D_LINK, "cand_idx"} <= set(field)
                or any(type(field[key]) is not type(late[key]) or field[key] != late[key]
                       for key in _LATE_D_LINK)
                or type(field["cand_idx"]) is not int or field["cand_idx"] != i):
            raise ValueError(_INVALID_PLAN)
    # The arrival pick: the tokenizer uses its target and transfer ID only.
    arrival, decode = candidates[late["default_idx"]], plan["decode"]
    if (decode["target"] != arrival["target"]
            or decode["kv_import"]["transfer_id"] != arrival["kv_import"]["transfer_id"]):
        raise ValueError(_INVALID_PLAN)
    return endpoints


def _validate_prepared_plan(plan: Any, prepared: Mapping[str, Any]) -> None:
    """Validate a split P/D execution envelope against its input descriptor.

    Checks protocol fields without importing the tokenizer service. Errors do
    not include callback credentials or untrusted plan text. A plan with a
    top-level ``decode_candidates`` key is a late-D plan (router
    ``--pd-late-decode choose`` with external role control): its P-leg also
    carries ``late_d``, its decode leg (the arrival pick) carries no session
    tag, and its candidates are checked by _late_d_candidate_endpoints. Any
    other plan must have the classic shape, with equal P and D session tags.

    Args:
        plan: Untrusted execution plan returned by the runtime.
        prepared: Original compact descriptor used to obtain the plan.

    Returns:
        None if the plan matches and names distinct valid endpoints.

    Raises:
        ValueError: The plan is malformed, mismatched, or uses one endpoint twice.
    """
    fields = {
        "protocol_version", "request_id", "config_id", "ref", "prefill", "decode",
        "callback", "reservation_id", "transfer_id",
    }
    invalid = ValueError(_INVALID_PLAN)
    late = isinstance(plan, dict) and "decode_candidates" in plan
    if not isinstance(plan, dict) or set(plan) != ((fields | {"decode_candidates"}) if late else fields):
        raise invalid
    if (type(plan["protocol_version"]) is not int
            or any(plan[key] != prepared[key] for key in ("protocol_version", "request_id", "config_id", "ref"))):
        raise invalid
    transfer = plan["transfer_id"]
    if type(transfer) is not int or not 0 <= transfer < 2**64 or not _identifier(plan["reservation_id"]):
        raise invalid
    callback = plan["callback"]
    if (not isinstance(callback, dict) or set(callback) != {"endpoint", "token"}
            or not _identifier(callback["token"])):
        raise invalid
    try:
        endpoint_url(callback["endpoint"])
        endpoints = []
        for name, mark_name in (("prefill", "kv_export"), ("decode", "kv_import")):
            leg = plan[name]
            required = {"target", mark_name}
            allowed = required | {"kv_session"} | (
                {"pull_hint"} if name == "prefill" else set()
            )
            if late and name == "prefill":
                required, allowed = required | {"late_d"}, allowed | {"late_d"}
            elif late:
                # The prefill forwards its session tag to the admitted
                # candidate; the arrival pick carries none (ForwardPD).
                allowed = required
            if not isinstance(leg, dict) or not required <= set(leg) <= allowed:
                raise invalid
            session = leg.get("kv_session")
            if session is not None and (
                not isinstance(session, dict)
                or set(session) != {"session_id", "version"}
                or not _identifier(session["session_id"])
                or type(session["version"]) is not int
                or not 0 < session["version"] < 2**63
            ):
                raise invalid
            endpoints.append(_leg_endpoint(leg, mark_name, prepared["config_id"], transfer))
        if late:
            # P, then every candidate; the arrival pick is one of them.
            endpoints[1:] = _late_d_candidate_endpoints(plan, prepared["config_id"], transfer)
        elif plan["prefill"].get("kv_session") != plan["decode"].get("kv_session"):
            raise invalid
        if len(set(endpoints)) != len(endpoints):
            raise invalid
    except (TypeError, AttributeError, LookupError, ValueError):
        # Never interpolate an untrusted plan or its callback credential.
        raise invalid from None


def endpoint_url(endpoint: str) -> str:
    """Validate and normalize an HTTP(S) service base URL.

    Args:
        endpoint: Service base URL without query or fragment.

    Returns:
        The URL with trailing slashes removed.

    Raises:
        ValueError: The URL has no HTTP(S) scheme/authority or has query/fragment.
    """
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("endpoint must be an HTTP(S) base URL without query or fragment")
    return endpoint.rstrip("/")


class Backend:
    """SGLang-compatible HTTP backend, using one async client session.

    Each instance belongs to one asyncio event loop. A scheduler passes the
    external Rust router URL to generate/stream; get_progress targets workers.
    This class performs no placement or local load accounting.
    Owned sessions use a fresh connection per request without retrying POST;
    caller-provided sessions retain their connection policy and ownership.
    """

    def __init__(self, *, timeout_s: float = 3600, session: aiohttp.ClientSession | None = None):
        """Configure one event-loop-owned asynchronous transport client.

        Args:
            timeout_s: Positive total HTTP timeout in seconds for an owned session.
            session: Optional caller-owned aiohttp session; retains its own policy.

        Returns:
            None; session creation is deferred unless a session is supplied.

        Raises:
            ValueError: timeout_s is not positive.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self._session = session
        self._owns_session = session is None
        self._closed = False

    def _client(self) -> aiohttp.ClientSession:
        """Return the reusable client session, creating an owned one if needed.

        Owned sessions create a fresh connection per request and do not retry POST.
        This client belongs to one event loop.

        Args:
            None.

        Returns:
            The open aiohttp ClientSession.

        Raises:
            RuntimeError: The Backend has been closed.
        """
        if self._closed:
            raise RuntimeError("Backend is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                # A busy client can receive a response after the server's idle
                # timeout, putting its closed socket into the pool with a new
                # timestamp before processing FIN. Do not reuse idle sockets or
                # retry non-idempotent POSTs. Active SSE streams remain open.
                connector=aiohttp.TCPConnector(limit=0, force_close=True),
                trust_env=False,
            )
        return self._session

    async def generate(self, endpoint: str, request: Request, *, chat: bool = False) -> dict[str, Any]:
        """Submit one nonstreaming request and decode the HTTP JSON response.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            request: Model request with a stable identity.
            chat: Select chat completions instead of text completions.

        Returns:
            The decoded generation-response object.

        Raises:
            aiohttp.ClientError: HTTP transport fails or the endpoint returns an error status.
            ValueError: The endpoint or request identity is invalid.
        """
        path = "/v1/chat/completions" if chat else "/v1/completions"
        async with self._client().post(endpoint_url(endpoint) + path,
                                       json=request.wire_payload(stream=False),
                                       headers=request.identity.headers()) as response:
            response.raise_for_status()
            return await response.json()

    async def stream(self, endpoint: str, request: Request, *, chat: bool = False) -> AsyncIterator[bytes]:
        """Stream a model response without interpreting or re-framing its SSE bytes.

        Closing/cancelling the iterator closes its upstream HTTP response. Engine
        cancellation follows the router's policy; disconnect alone does not confirm
        KV release.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            request: Model request with a stable identity.
            chat: Select chat completions instead of text completions.

        Yields:
            Raw byte chunks, which may split SSE records.

        Raises:
            aiohttp.ClientError: HTTP transport fails or the endpoint returns an error status.
        """
        path = "/v1/chat/completions" if chat else "/v1/completions"
        async with self._client().post(endpoint_url(endpoint) + path,
                                       json=request.wire_payload(stream=True),
                                       headers=request.identity.headers()) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_any():
                yield chunk

    async def get_progress(self, worker_url: str) -> dict[str, Any]:
        """Read raw progress directly from one engine without router reservations.

        Args:
            worker_url: Inference engine HTTP(S) base URL.

        Returns:
            The engine's JSON core, queue, and prefill-work observations.

        Raises:
            aiohttp.ClientError: The engine query fails.
        """
        async with self._client().get(endpoint_url(worker_url) + "/v1/loads",
                                      params={"include": "core,queues,prefill_work"}) as response:
            response.raise_for_status()
            return await response.json()

    async def get_backend_loads(self, endpoint: str) -> dict[str, Any]:
        """Read the runtime's merged reservations and latest engine observations.

        Does not trigger engine polling, create a reservation, or advance decode
        round-robin.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.

        Returns:
            The JSON load view owned by the Rust Backend.

        Raises:
            RuntimeError: The runtime returns a non-success HTTP status.
        """
        async with self._client().get(
            endpoint_url(endpoint) + "/symmetric_pd/backend/loads", allow_redirects=False
        ) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Backend load query returned HTTP {response.status}")
            return await response.json()

    async def end_trajectory(
        self, endpoint: str, agent_id: str, subagent_id: int
    ) -> dict[str, Any]:
        """Explicitly end one trajectory; transport close never calls this.

        Args:
            endpoint: Runtime endpoint owning the trajectory.
            agent_id: Unique problem lifecycle identifier.
            subagent_id: Branch to end; siblings remain active.

        Returns:
            Async acceptance only; engine reclamation is not yet confirmed.
        """
        RequestIdentity(agent_id, subagent_id, 0)
        async with self._client().post(
            endpoint_url(endpoint) + "/kv_sessions/end",
            json={"agent_id": agent_id, "subagent_id": subagent_id},
            allow_redirects=False,
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _prepared_plan(
        self,
        endpoint: str,
        prepared: Mapping[str, Any],
        identity: RequestIdentity | None = None,
    ) -> dict[str, Any]:
        """Ask the runtime to place a prepared descriptor and reserve its P/D work.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            prepared: Compact input descriptor; raw tokens are rejected.
            identity: Optional original request identity; RID must match the descriptor.

        Returns:
            The decoded execution-plan envelope; validated by the caller.

        Raises:
            ValueError: The descriptor or endpoint is invalid.
            RuntimeError: The placement request returns a non-success HTTP status.
        """
        if identity is not None and prepared.get("request_id") != identity.engine_rid:
            raise ValueError("prepared request_id conflicts with trajectory identity")
        async with self._client().post(
            endpoint_url(endpoint) + "/symmetric_pd/prepared",
            json={"prepared": _prepared_description(prepared)},
            headers=identity.headers() if identity is not None else None,
            allow_redirects=False,
        ) as response:
            if not 200 <= response.status < 300:
                raise PreparedHTTPError(f"prepared placement returned HTTP {response.status}", response.status)
            return await response.json()

    async def _cancel_prepared(self, endpoint: str, plan: Any) -> None:
        """Best-effort notify the Rust owner that a prepared request needs cancellation.

        Acknowledgement does not establish engine cleanup. This only sends metadata
        and never retries generation.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            plan: Plan carrying the reservation ID and callback credential.

        Returns:
            None; malformed plans are ignored and notification failures are logged.
        """
        if not isinstance(plan, dict):
            return
        reservation, callback = plan.get("reservation_id"), plan.get("callback")
        token = callback.get("token") if isinstance(callback, dict) else None
        if not _identifier(reservation) or not _identifier(token):
            return
        try:
            async with self._client().post(
                endpoint_url(endpoint) + "/symmetric_pd/cancel",
                json={"reservation_id": reservation},
                headers={"Authorization": "Bearer " + token},
                timeout=aiohttp.ClientTimeout(total=min(2, self.timeout_s)),
                allow_redirects=False,
            ) as response:
                if not 200 <= response.status < 300:
                    LOG.warning("Prepared cancellation notification failed (HTTP %d); Backend must reconcile",
                                response.status)
        except (aiohttp.ClientError, TimeoutError, RuntimeError, ValueError) as exc:
            # The exception's repr/request headers may contain the credential.
            LOG.warning("Prepared cancellation notification failed (%s); Backend must reconcile",
                        type(exc).__name__)

    async def _finish_prepared(self, endpoint: str, plan: Any) -> None:
        """Run cancellation notification in a shielded task before propagating exit.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            plan: Execution plan to pass to cancellation notification.

        Returns:
            None after notification cleanup completes.

        Raises:
            asyncio.CancelledError: A repeated cancellation is re-raised after cleanup is
                awaited.
        """
        cleanup = asyncio.create_task(self._cancel_prepared(endpoint, plan))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # A second cancellation must not strand an unobserved cleanup task.
            await cleanup
            raise

    async def generate_prepared(
        self,
        endpoint: str,
        tokenizer_endpoint: str,
        prepared: Mapping[str, Any],
        *,
        identity: RequestIdentity | None = None,
    ) -> dict[str, Any]:
        """Place an existing descriptor, forward its input once, and decode D output.

        The input must have been prepared for nonstreaming generation. Failures
        notify the Rust owner; no tokenization or generation retry occurs.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            tokenizer_endpoint: Service holding the prepared body.
            prepared: Compact descriptor returned by that service's /prepare endpoint.
            identity: Optional trajectory identity for versioned CPU KV retention.

        Returns:
            The decoded D JSON response.

        Raises:
            ValueError: The descriptor or execution plan is invalid.
            RuntimeError: Placement or forwarding returns a non-success status.
        """
        target = endpoint_url(tokenizer_endpoint) + "/forward_pd"
        plan = await self._prepared_plan(endpoint, prepared, identity)
        completed = False
        try:
            _validate_prepared_plan(plan, prepared)
            async with self._client().post(target, json=plan, allow_redirects=False) as response:
                if not 200 <= response.status < 300:
                    raise await _prepared_forward_error(response)
                result = await response.json()
            completed = True
            return result
        finally:
            if not completed:
                await self._finish_prepared(endpoint, plan)

    async def stream_prepared(
        self,
        endpoint: str,
        tokenizer_endpoint: str,
        prepared: Mapping[str, Any],
        *,
        identity: RequestIdentity | None = None,
    ) -> AsyncIterator[bytes]:
        """Place an existing descriptor and relay the prepared D stream verbatim.

        The input must have been prepared for streaming. Errors, cancellation, and
        early iterator close notify the Rust Backend without retrying generation.
        The caller must close the iterator when stopping early.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            tokenizer_endpoint: Service holding the prepared body.
            prepared: Compact descriptor returned by that service's /prepare endpoint.
            identity: Optional trajectory identity for versioned CPU KV retention.

        Yields:
            Raw decode-response byte chunks.

        Raises:
            ValueError: The descriptor or execution plan is invalid.
            RuntimeError: Placement or forwarding returns a non-success status.
        """
        target = endpoint_url(tokenizer_endpoint) + "/forward_pd"
        plan = await self._prepared_plan(endpoint, prepared, identity)
        completed = False
        try:
            _validate_prepared_plan(plan, prepared)
            async with self._client().post(target, json=plan, allow_redirects=False) as response:
                if not 200 <= response.status < 300:
                    raise await _prepared_forward_error(response)
                async for chunk in response.content.iter_any():
                    yield chunk
            completed = True
        finally:
            if not completed:
                await self._finish_prepared(endpoint, plan)

    async def close(self):
        """Mark the client closed and release its session when locally owned.

        Args:
            None.

        Returns:
            None; a supplied session remains caller-owned.
        """
        if not self._closed and self._owns_session and self._session is not None:
            await self._session.close()
        self._closed = True

    async def __aenter__(self):
        """Ensure a session exists and enter the asynchronous client context.

        Args:
            None.

        Returns:
            This Backend instance.

        Raises:
            RuntimeError: The client is already closed.
        """
        self._client()
        return self

    async def __aexit__(self, *exc):
        """Close owned transport resources on asynchronous context exit.

        Args:
            *exc: Context exit exception type, value, and traceback; unused.

        Returns:
            None; context exceptions are not suppressed.
        """
        await self.close()
