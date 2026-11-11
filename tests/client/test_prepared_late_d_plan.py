"""Late D choose through the client SDK: the router's prepared plan with decode candidates.

With ``--pd-late-decode choose`` under external role control, the router's
/symmetric_pd/prepared plan names every decode candidate. The SDK's plan check
(evolving_infra/router/client/backend/backend.py _validate_prepared_plan) must accept such a
plan before /forward_pd. CPU protocol tests only: no GPU, no
engine; fake routers and engines run on local TCP ports.

The plan builders mirror the router:
- evolving_infra/router/core/src/backend/execution.rs Executions::install: the classic
  plan; for late D the P-leg kv_export and late_d, ``decode`` = the arrival
  pick (target and kv_import of candidate default_idx) and ``decode_candidates``;
- evolving_infra/router/core/src/server/routes/prepared.rs plan: the session tag on the
  P-leg only for a late plan, on both legs otherwise;
- evolving_infra/router/core/src/backend/sglang/kv.rs marks / late_marks: the KV marks;
- evolving_infra/router/core/src/late_bind.rs sym_prefill_field / decode_field;
- evolving_infra/router/core/src/backend/sglang/request.rs prepared_target.
"""
import asyncio
from contextlib import asynccontextmanager
import copy
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

from aiohttp import web
from aiohttp.test_utils import TestServer
import httpx
import pytest

from evolving_infra.router.client.agent.identity import RequestIdentity
from evolving_infra.router.client.backend.backend import Backend, _identifier, _validate_prepared_plan, endpoint_url
from evolving_infra.router.frontend.protocol import ForwardPD, NodeConfig
from evolving_infra.router.frontend.server import TokenizerService

SECRET = "callback-credential-0123456789"
INVALID = "invalid prepared execution plan"
# A late plan returned by the real router binary (see its "source" field).
CAPTURED = Path(__file__).with_name("fixtures") / "router_late_d_plan.json"
P = "http://10.0.0.1:30000"
DS = ["http://10.0.0.2:30000", "http://10.0.0.3:30000", "http://10.0.0.4:30000"]
TAG = {"session_id": "agent-7:0", "version": 3}


def descriptor(request_id="rid-1", config_id="glm-test"):
    """A compact descriptor with the fields of the tokenizer's /prepare answer."""
    return {
        "protocol_version": 1, "node_id": "T0", "request_id": request_id, "ref": "ref-1",
        "config_id": config_id, "prompt_tokens": 70, "max_new_tokens": 8, "expires_in_ms": 300000,
        "content": {"namespace": config_id, "hash_spec": {"mode": "dynamo", "page_size": 64},
                    "blocks": [{"hash": 5, "valid_tokens": 64}]},
    }


def target(worker, config_id):
    """request.rs prepared_target."""
    return {"transport": "sglang_generate", "endpoint": worker.rstrip("/") + "/generate",
            "config_id": config_id}


def addr(worker, port):
    return "tcp://" + worker.split("//")[1].split(":")[0] + ":" + str(port)


def classic_plan(prepared, p=P, d=DS[0], *, tag=None, pull_hint=None, transfer=4242,
                 callback="http://router:30000/symmetric_pd/events"):
    """execution.rs install without late D, kv.rs marks, prepared.rs tag on both legs."""
    config_id = prepared["config_id"]
    plan = {
        "protocol_version": 1, "request_id": prepared["request_id"], "config_id": config_id,
        "ref": prepared["ref"], "reservation_id": "6f9619ff-8b86-4011-b42d-00c04fc964ff",
        "transfer_id": transfer,
        "prefill": {"target": target(p, config_id),
                    "kv_export": {"transfer_id": transfer, "attempt_id": 9001, "reader_worker": d,
                                  "reader_addr": addr(d, 18999)},
                    "pull_hint": copy.deepcopy(pull_hint)},
        "decode": {"target": target(d, config_id),
                   "kv_import": {"transfer_id": transfer, "attempt_id": 9001, "owner_worker": p,
                                 "owner_addr": addr(p, 18998),
                                 "expected_tokens": prepared["prompt_tokens"], "cache_policy": "retain"}},
        "callback": {"endpoint": callback, "token": SECRET},
    }
    if tag is not None:
        plan["prefill"]["kv_session"] = dict(tag)
        plan["decode"]["kv_session"] = dict(tag)
    return plan


def late_plan(prepared, p=P, candidates=DS, default_idx=1, *, tag=None, pull_hint=None, epoch=7,
              transfer=4242, callback="http://router:30000/symmetric_pd/events"):
    """execution.rs install with late D, kv.rs late_marks, late_bind.rs fields, prepared.rs tag on P only."""
    config_id, k = prepared["config_id"], len(candidates)
    plan = classic_plan(prepared, p, candidates[default_idx], pull_hint=pull_hint, transfer=transfer,
                        callback=callback)
    attempts = [7000 + i for i in range(k)]
    plan["prefill"]["kv_export"] = {"transfer_id": transfer, "candidates": [
        {"cand_idx": i, "reader_worker": w, "reader_addr": addr(w, 18999), "attempt_id": attempts[i]}
        for i, w in enumerate(candidates)]}
    plan["prefill"]["late_d"] = {"v": 1, "mode": "choose", "router_epoch": epoch, "k": k,
                                 "default_idx": default_idx, "local_ok": False}

    def imports(i):
        return {"transfer_id": transfer, "attempt_id": attempts[i], "owner_worker": p,
                "owner_addr": addr(p, 18998), "expected_tokens": prepared["prompt_tokens"],
                "cache_policy": "gpu_only"}

    plan["decode"] = {"target": target(candidates[default_idx], config_id), "kv_import": imports(default_idx)}
    plan["decode_candidates"] = [
        {"target": target(w, config_id), "kv_import": imports(i),
         "late_d": {"v": 1, "mode": "choose", "router_epoch": epoch, "k": k, "cand_idx": i}}
        for i, w in enumerate(candidates)]
    if tag is not None:
        plan["prefill"]["kv_session"] = dict(tag)
    return plan


def outcome(validate, plan, prepared):
    """'ok', or the exception type and text, of one validation."""
    try:
        validate(plan, prepared)
    except Exception as error:  # noqa: BLE001 - the outcome itself is compared
        return type(error).__name__, str(error)
    return "ok"


def assert_rejected(plan, prepared):
    with pytest.raises(ValueError) as caught:
        _validate_prepared_plan(plan, prepared)
    assert str(caught.value) == INVALID
    assert SECRET not in repr(caught.value) and caught.value.__cause__ is None


# ---------------------------------------------------------------------------------------------------------------
# (a) Classic plans: accepted and rejected exactly as before the late-D change.
# ---------------------------------------------------------------------------------------------------------------
def _old_validate_prepared_plan(plan: Any, prepared: Mapping[str, Any]) -> None:
    """Verbatim body of _validate_prepared_plan before late-D support (classic plans only), for the differential test."""
    fields = {
        "protocol_version", "request_id", "config_id", "ref", "prefill", "decode",
        "callback", "reservation_id", "transfer_id",
    }
    invalid = ValueError("invalid prepared execution plan")
    if not isinstance(plan, dict) or set(plan) != fields:
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
            allowed = {"target", mark_name, "kv_session"} | (
                {"pull_hint"} if name == "prefill" else set()
            )
            if not isinstance(leg, dict) or not {"target", mark_name} <= set(leg) <= allowed:
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
            target, mark = leg["target"], leg[mark_name]
            if (not isinstance(target, dict) or set(target) != {"transport", "endpoint", "config_id"}
                    or target["transport"] != "sglang_generate"
                    or target["config_id"] != prepared["config_id"]
                    or not isinstance(mark, dict) or type(mark.get("transfer_id")) is not int
                    or mark["transfer_id"] != transfer):
                raise invalid
            endpoints.append(endpoint_url(target["endpoint"]))
        if plan["prefill"].get("kv_session") != plan["decode"].get("kv_session"):
            raise invalid
        if endpoints[0] == endpoints[1]:
            raise invalid
    except (TypeError, AttributeError, ValueError):
        # Never interpolate an untrusted plan or its callback credential.
        raise invalid from None


REPLACEMENTS = (None, "", "x", "http://10.0.0.9:1", 0, -1, 1, 4242, 2**64, True, False, 1.5, [], [1], {},
                {"transfer_id": 4242}, {"session_id": "s", "version": 1})


def _paths(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield prefix + (key,)
            yield from _paths(item, prefix + (key,))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield prefix + (i,)
            yield from _paths(item, prefix + (i,))


def _get(plan, path):
    node = plan
    for key in path:
        node = node[key]
    return node


def _parent(plan, path):
    return _get(plan, path[:-1])


ADDED = {"extra": 1, "late_d": {"k": 1}, "kv_session": None, "pull_hint": None, "decode_candidates": [],
         "candidates": []}


def mutations(plan):
    """Every single-field change of a plan: each value replaced, each key deleted, a new key in every object."""
    for path in list(_paths(plan)):
        for value in REPLACEMENTS:
            changed = copy.deepcopy(plan)
            _parent(changed, path)[path[-1]] = copy.deepcopy(value)
            yield "set %s=%r" % ("/".join(map(str, path)), value), changed
        if isinstance(path[-1], str):
            changed = copy.deepcopy(plan)
            del _parent(changed, path)[path[-1]]
            yield "del %s" % "/".join(map(str, path)), changed
    for path in [()] + [p for p in _paths(plan) if isinstance(_get(plan, p), dict)]:
        for key, value in ADDED.items():
            if key in _get(plan, path):
                continue
            changed = copy.deepcopy(plan)
            _get(changed, path)[key] = copy.deepcopy(value)
            yield "add %s/%s" % ("/".join(map(str, path)), key), changed


def classic_corpus():
    prepared = descriptor()
    bases = {
        "untagged": classic_plan(prepared),
        "tagged": classic_plan(prepared, tag=TAG),
        "hint": classic_plan(prepared, tag=TAG, pull_hint={"source": "http://10.0.0.5:30000", "tokens": 512}),
    }
    for name, base in bases.items():
        yield name, base
        for label, changed in mutations(base):
            yield name + ": " + label, changed
    plan = classic_plan(prepared)
    plan["decode"]["target"]["endpoint"] = plan["prefill"]["target"]["endpoint"] + "/"
    yield "P = D after normalization", plan
    plan = classic_plan(prepared, tag=TAG)
    plan["decode"]["kv_session"]["version"] = 4
    yield "P and D tags differ", plan
    plan = classic_plan(prepared, tag=TAG)
    del plan["decode"]["kv_session"]
    yield "P tag only (classic)", plan
    for value in (None, [], {}, "x"):
        plan = classic_plan(prepared)
        plan["decode_candidates"] = value
        yield "classic plan + decode_candidates=%r" % (value,), plan


def test_classic_plans_are_validated_exactly_as_before():
    """Differential: every classic plan and single-field mutation gets the old outcome."""
    prepared = descriptor()
    accepted = total = 0
    for label, plan in classic_corpus():
        old, new = outcome(_old_validate_prepared_plan, plan, prepared), outcome(_validate_prepared_plan, plan, prepared)
        assert new == old, label
        total += 1
        accepted += old == "ok"
    # The corpus exercises both outcomes (the base plans pass, most mutations fail).
    assert total > 500 and 3 <= accepted < total - 400


@pytest.mark.parametrize("tag, hint", [(None, None), (TAG, None), (TAG, {"source": "http://10.0.0.5:30000"})])
def test_classic_router_plans_are_accepted(tag, hint):
    prepared = descriptor()
    plan = classic_plan(prepared, tag=tag, pull_hint=hint)
    _validate_prepared_plan(plan, prepared)
    ForwardPD.model_validate(plan)


def test_classic_plan_rejections_are_unchanged():
    prepared = descriptor()
    plan = classic_plan(prepared, tag=TAG)
    del plan["decode"]["kv_session"]  # the P-only tag belongs to late plans
    assert_rejected(plan, prepared)
    plan = classic_plan(prepared)
    plan["prefill"]["late_d"] = late_plan(prepared)["prefill"]["late_d"]  # late_d needs candidates
    assert_rejected(plan, prepared)
    plan = classic_plan(prepared, d=P)
    assert_rejected(plan, prepared)


# ---------------------------------------------------------------------------------------------------------------
# (b) Late-D plans as the router builds them: accepted, and accepted by the tokenizer protocol too.
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("candidates, default_idx, tag, hint", [
    (DS[:1], 0, None, None),
    (DS, 1, TAG, None),
    (DS, 2, TAG, {"source": "http://10.0.0.5:30000", "tokens": 1024}),
    (["http://10.0.1.%d:30000" % i for i in range(64)], 63, TAG, None),
])
def test_router_late_plans_are_accepted(candidates, default_idx, tag, hint):
    prepared = descriptor()
    plan = late_plan(prepared, candidates=candidates, default_idx=default_idx, tag=tag, pull_hint=hint)
    original = copy.deepcopy(plan)
    _validate_prepared_plan(plan, prepared)
    assert plan == original  # validation never edits what is forwarded
    parsed = ForwardPD.model_validate(plan)
    assert len(parsed.decode_candidates) == len(candidates)
    assert parsed.decode.kv_session is None
    assert (parsed.prefill.kv_session is None) == (tag is None)


def shape(value):
    """Nested keys and value types of a JSON value; every item of a list must have one shape."""
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, list):
        shapes = [shape(item) for item in value]
        assert all(item == shapes[0] for item in shapes)
        return shapes[:1]
    return type(value).__name__


def test_captured_router_late_plan_is_accepted_and_the_builder_matches_it():
    """A plan of the real router binary: accepted now, rejected by the old check, same shape as late_plan()."""
    captured = json.loads(CAPTURED.read_text())
    prepared, plan = captured["descriptor"], captured["plan"]
    _validate_prepared_plan(plan, prepared)
    ForwardPD.model_validate(plan)
    assert outcome(_old_validate_prepared_plan, plan, prepared) == ("ValueError", INVALID)  # the pre-late-D rejection
    workers = [c["target"]["endpoint"].removesuffix("/generate") for c in plan["decode_candidates"]]
    mirror = late_plan(prepared, p=plan["prefill"]["target"]["endpoint"].removesuffix("/generate"),
                       candidates=workers, default_idx=plan["prefill"]["late_d"]["default_idx"],
                       tag=plan["prefill"]["kv_session"])
    assert shape(mirror) == shape(plan)
    _validate_prepared_plan(mirror, prepared)


# ---------------------------------------------------------------------------------------------------------------
# (c) Malformed late-D plans: rejected with the generic error, never the plan text.
# ---------------------------------------------------------------------------------------------------------------
def _set(path, value):
    def change(plan):
        _parent(plan, path)[path[-1]] = value
    return change


def _del(path):
    def change(plan):
        del _parent(plan, path)[path[-1]]
    return change


def _swap_candidates(plan):
    cands = plan["decode_candidates"]
    cands[0], cands[1] = cands[1], cands[0]


def _same_endpoint(index, endpoint):
    """Give candidate `index` (and the decode leg when it is the arrival pick) this endpoint."""
    def change(plan):
        plan["decode_candidates"][index]["target"]["endpoint"] = endpoint
        if index == plan["prefill"]["late_d"]["default_idx"]:
            plan["decode"]["target"]["endpoint"] = endpoint
    return change


MALFORMED = {
    "decode_candidates missing (late_d alone)": _del(("decode_candidates",)),
    "decode_candidates null": _set(("decode_candidates",), None),
    "decode_candidates empty": _set(("decode_candidates",), []),
    "decode_candidates an object": _set(("decode_candidates",), {"0": {}}),
    "candidate not an object": _set(("decode_candidates", 0), "http://10.0.0.2:30000/generate"),
    "candidate without late_d": _del(("decode_candidates", 0, "late_d")),
    "candidate without kv_import": _del(("decode_candidates", 2, "kv_import")),
    "candidate without target": _del(("decode_candidates", 1, "target")),
    "candidate with a session tag": _set(("decode_candidates", 1, "kv_session"), TAG),
    "candidate with an unknown key": _set(("decode_candidates", 0, "extra"), 1),
    "candidate transport": _set(("decode_candidates", 0, "target", "transport"), "http"),
    "candidate config": _set(("decode_candidates", 0, "target", "config_id"), "other"),
    "candidate target extra key": _set(("decode_candidates", 0, "target", "extra"), 1),
    "candidate endpoint with query": _set(("decode_candidates", 0, "target", "endpoint"), DS[0] + "/generate?x=1"),
    "candidate endpoint not a URL": _set(("decode_candidates", 0, "target", "endpoint"), 5),
    "candidate transfer mismatch": _set(("decode_candidates", 2, "kv_import", "transfer_id"), 4243),
    "candidate transfer as text": _set(("decode_candidates", 2, "kv_import", "transfer_id"), "4242"),
    "candidate kv_import not an object": _set(("decode_candidates", 2, "kv_import"), [4242]),
    "candidate order": _swap_candidates,
    "candidate cand_idx": _set(("decode_candidates", 2, "late_d", "cand_idx"), 1),
    "candidate cand_idx bool": _set(("decode_candidates", 1, "late_d", "cand_idx"), True),
    "candidate k": _set(("decode_candidates", 0, "late_d", "k"), 2),
    "candidate epoch": _set(("decode_candidates", 0, "late_d", "router_epoch"), 8),
    "candidate mode": _set(("decode_candidates", 0, "late_d", "mode"), "steer"),
    "candidate v type": _set(("decode_candidates", 0, "late_d", "v"), 1.0),
    "candidate late_d missing key": _del(("decode_candidates", 0, "late_d", "router_epoch")),
    "P late_d missing": _del(("prefill", "late_d")),
    "P late_d null": _set(("prefill", "late_d"), None),
    "P late_d k": _set(("prefill", "late_d", "k"), 2),
    "P late_d k bool": _set(("prefill", "late_d", "k"), True),
    "P late_d default_idx = k": _set(("prefill", "late_d", "default_idx"), 3),
    "P late_d default_idx negative": _set(("prefill", "late_d", "default_idx"), -1),
    "P late_d local_ok true": _set(("prefill", "late_d", "local_ok"), True),
    "P late_d local_ok 0": _set(("prefill", "late_d", "local_ok"), 0),
    "P late_d epoch negative": _set(("prefill", "late_d", "router_epoch"), -1),
    "P late_d epoch over u32": _set(("prefill", "late_d", "router_epoch"), 2**32),
    "P late_d mode empty": _set(("prefill", "late_d", "mode"), ""),
    "P late_d v negative": _set(("prefill", "late_d", "v"), -1),
    "decode leg with the P tag": _set(("decode", "kv_session"), TAG),
    "decode leg with a null tag": _set(("decode", "kv_session"), None),
    "decode leg with late_d": _set(("decode", "late_d"), {"k": 3}),
    "decode leg is not the arrival pick (target)": _set(("decode", "target", "endpoint"), DS[0] + "/generate"),
    "decode transfer mismatch": _set(("decode", "kv_import", "transfer_id"), 1),
    "P tag version 0": _set(("prefill", "kv_session", "version"), 0),
    "P tag extra key": _set(("prefill", "kv_session", "extra"), 1),
    "P export transfer mismatch": _set(("prefill", "kv_export", "transfer_id"), 1),
    "P export without candidates": _del(("prefill", "kv_export", "candidates")),
    "P export candidates short": _del(("prefill", "kv_export", "candidates", 2)),
    "P export candidate repeated": _set(("prefill", "kv_export", "candidates", 0, "cand_idx"), 1),
    "P export cand_idx out of range": _set(("prefill", "kv_export", "candidates", 2, "cand_idx"), 3),
    "P export cand_idx bool": _set(("prefill", "kv_export", "candidates", 1, "cand_idx"), True),
    "P export candidate not an object": _set(("prefill", "kv_export", "candidates", 0), 0),
    "P leg unknown key": _set(("prefill", "extra"), 1),
    "top-level unknown key": _set(("late_d",), {"k": 3}),
    "request mismatch": _set(("request_id",), "rid-2"),
    "reservation id empty": _set(("reservation_id",), ""),
    "plan transfer as text": _set(("transfer_id",), "4242"),
    "callback with extra key": _set(("callback", "extra"), 1),
    "candidate repeats another": _same_endpoint(0, DS[2] + "/generate"),
    "candidate repeats another after normalization": _same_endpoint(0, DS[2] + "/generate/"),
    "candidate is the P": _same_endpoint(2, P + "/generate"),
    "arrival pick is the P": _same_endpoint(1, P + "/generate"),
}


@pytest.mark.parametrize("label", sorted(MALFORMED))
def test_malformed_late_plans_are_rejected(label):
    prepared = descriptor()
    plan = late_plan(prepared, tag=TAG)
    MALFORMED[label](plan)
    assert_rejected(plan, prepared)


# Fields the engine or the tokenizer does not read: accepted.
RELAXED = {
    # The engine's late_d check reads the fields it needs and ignores others; local_ok defaults to false.
    "P late_d extra key": _set(("prefill", "late_d", "deadline_ms"), 50),
    "P late_d without local_ok": _del(("prefill", "late_d", "local_ok")),
    "candidate late_d extra key": _set(("decode_candidates", 0, "late_d", "deadline_ms"), 50),
    # The tokenizer uses only the arrival pick's target and transfer ID.
    "decode kv_import differs from the arrival pick's beyond the transfer ID":
        _set(("decode", "kv_import", "attempt_id"), 7000),
    # The engine needs each candidate exported once, in any order (symmetric marks).
    "P export candidates in another order": lambda plan: plan["prefill"]["kv_export"]["candidates"].reverse(),
}


@pytest.mark.parametrize("label", sorted(RELAXED))
def test_late_plan_fields_that_nobody_reads_are_accepted(label):
    prepared = descriptor()
    plan = late_plan(prepared, tag=TAG)
    RELAXED[label](plan)
    _validate_prepared_plan(plan, prepared)
    ForwardPD.model_validate(plan)


def test_too_many_candidates_are_rejected():
    prepared = descriptor()
    plan = late_plan(prepared, candidates=["http://10.0.1.%d:30000" % i for i in range(65)], default_idx=0)
    assert_rejected(plan, prepared)


def test_late_plan_mutations_the_sdk_accepts_are_accepted_by_the_tokenizer():
    """The SDK is never looser than the tokenizer's ForwardPD on late plans (pull_hint excepted, as classic)."""
    prepared = descriptor()
    accepted = 0
    for label, plan in mutations(late_plan(prepared, tag=TAG, pull_hint={"tokens": 512})):
        if label.startswith(("set prefill/pull_hint", "del prefill/pull_hint")):
            continue  # The classic check never looked at the hint either.
        if outcome(_validate_prepared_plan, plan, prepared) == "ok":
            accepted += 1
            ForwardPD.model_validate(plan)
    assert accepted  # e.g. other attempt IDs or cache policies of a non-arrival candidate


# ---------------------------------------------------------------------------------------------------------------
# SDK smokes: the real Backend against a fake router and a recording tokenizer.
# ---------------------------------------------------------------------------------------------------------------
WINNER = (b'data: {"text":"a","output_ids":[5],"meta_info":{"completion_tokens":1}}\n\n',
          b'data: {"text":"ab","output_ids":[6],"meta_info":{"completion_tokens":2,"spec_verify_ct":1,'
          b'"finish_reason":{"type":"length"}}}\n\n',
          b'data: [DONE]\n\n')
RELEASED = (b'data: {"text":"","meta_info":{"completion_tokens":0,"finish_reason":'
            b'{"type":"abort","status_code":409,"err_type":"late_d_released","message":"released"}}}\n\n'
            b'data: [DONE]\n\n')


async def serve(app):
    server = TestServer(app)
    await server.start_server()
    return server, str(server.make_url("")).rstrip("/")


@asynccontextmanager
async def fake_router(plan_for, calls):
    """POST /symmetric_pd/prepared answers plan_for(descriptor); /symmetric_pd/cancel is recorded."""
    async def placement(request):
        body = await request.json()
        calls.append(("place", body, request.headers.get("x-agent-context")))
        return web.json_response(plan_for(body["prepared"]))

    async def cancel(request):
        calls.append(("cancel", await request.json(), request.headers.get("Authorization")))
        return web.json_response({"cancellation_requested": True, "engine_termination_confirmed": False})

    app = web.Application()
    app.router.add_post("/symmetric_pd/prepared", placement)
    app.router.add_post("/symmetric_pd/cancel", cancel)
    server, url = await serve(app)
    try:
        yield url
    finally:
        await server.close()


@asynccontextmanager
async def recording_tokenizer(calls, relay=None):
    """POST /forward_pd records the body, then answers it (fake) or relays it to a real tokenizer app."""
    async def forward_pd(request):
        body = await request.json()
        calls.append(("forward", body))
        if relay is not None:
            response = await relay.post("/forward_pd", json=body)
            kind = response.headers.get("content-type", "application/json").split(";")[0]
            return web.Response(status=response.status_code, body=response.content, content_type=kind)
        if body["prefill"]["target"]["endpoint"].endswith("/stream/generate"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for part in WINNER:
                await response.write(part)
            await response.write_eof()
            return response
        return web.json_response({"text": "answer", "meta_info": {"id": body["request_id"]}})

    app = web.Application()
    app.router.add_post("/forward_pd", forward_pd)
    server, url = await serve(app)
    try:
        yield url
    finally:
        await server.close()


@pytest.mark.parametrize("mode", ["json", "stream"])
def test_sdk_forwards_the_late_plan_with_its_candidates(mode):
    async def scenario():
        identity = RequestIdentity("agent-7", 0, 3)
        prepared = descriptor(request_id=identity.engine_rid)
        p = "http://10.0.0.1:30000/stream" if mode == "stream" else P
        plans, calls = [], []

        def plan_for(body):
            plans.append(late_plan(body, p=p, default_idx=2, tag=TAG))
            return plans[-1]

        async with fake_router(plan_for, calls) as router, recording_tokenizer(calls) as tokenizer:
            async with Backend() as sdk:
                if mode == "stream":
                    out = b"".join([c async for c in sdk.stream_prepared(router, tokenizer, prepared,
                                                                         identity=identity)])
                    assert out == b"".join(WINNER)
                else:
                    result = await sdk.generate_prepared(router, tokenizer, prepared, identity=identity)
                    assert result["text"] == "answer"
        assert [c[0] for c in calls] == ["place", "forward"]  # forwarded once, never cancelled
        assert calls[0][1] == {"prepared": prepared} and calls[0][2] is not None
        forwarded = calls[1][1]
        assert forwarded == plans[0]  # the router's plan, byte for byte in JSON terms
        assert [c["late_d"]["cand_idx"] for c in forwarded["decode_candidates"]] == [0, 1, 2]
        assert forwarded["prefill"]["late_d"]["k"] == 3 and forwarded["prefill"]["kv_session"] == TAG
        assert "kv_session" not in forwarded["decode"]
        ForwardPD.model_validate(forwarded)
    asyncio.run(scenario())


def test_sdk_rejects_a_malformed_late_plan_before_forwarding_and_cancels_it():
    async def scenario():
        prepared, calls = descriptor(), []

        def plan_for(body):
            plan = late_plan(body, tag=TAG)
            plan["decode"]["kv_session"] = dict(TAG)  # not the router's late shape
            return plan

        async with fake_router(plan_for, calls) as router, recording_tokenizer(calls) as tokenizer:
            async with Backend() as sdk:
                with pytest.raises(ValueError, match=INVALID) as caught:
                    await sdk.generate_prepared(router, tokenizer, prepared)
        assert SECRET not in str(caught.value)
        assert [c[0] for c in calls] == ["place", "cancel"]
        assert calls[1][1:] == ({"reservation_id": "6f9619ff-8b86-4011-b42d-00c04fc964ff"}, "Bearer " + SECRET)
    asyncio.run(scenario())


# ---------------------------------------------------------------------------------------------------------------
# SDK -> real tokenizer service -> fake engines (only the router is fake).
# ---------------------------------------------------------------------------------------------------------------
def page_hashes(ids, page_size=4):
    seed, out = b"", []
    for offset in range(0, len(ids), page_size):
        page = ids[offset:offset + page_size]
        seed = hashlib.sha256(seed + struct.pack(f"<{len(page)}I", *page)).digest()
        out.append(seed.hex())
    return out


def test_sdk_late_plan_reaches_the_real_tokenizer_and_every_candidate():
    async def scenario():
        seen, events = [], []

        async def tokenize(request):
            return web.json_response({"tokens": list(range(19)), "count": 19})

        async def prefill(request):
            seen.append((request.path, await request.json()))
            return web.json_response({"text": "", "meta_info": {"finish_reason": {"type": "length"}}})

        def decode(idx):
            async def handler(request):
                seen.append((request.path, await request.json()))
                if idx == 1:  # /v1/completions-style release
                    return web.json_response({"error": {"type": "late_d_released"}}, status=409,
                                             headers={"X-Late-D-Reason": "released"})
                response = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await response.prepare(request)
                await asyncio.sleep(.05 if idx == 2 else .01)
                for part in (WINNER if idx == 2 else (RELEASED,)):
                    await response.write(part)
                return response
            return handler

        async def report(request):
            events.append(await request.json())
            return web.json_response({"accepted": True})

        engines = web.Application()
        engines.router.add_post("/tokenize", tokenize)
        engines.router.add_post("/p/generate", prefill)
        for i in range(3):
            engines.router.add_post(f"/d{i}/generate", decode(i))
        engines.router.add_post("/events", report)
        upstream, base = await serve(engines)
        workers = [base + f"/d{i}" for i in range(3)]
        settings = NodeConfig(
            node_id="cpu-a", config_id="test", model_path="/model", tokenize_endpoint=base + "/tokenize",
            targets={base + "/p/generate": "test", **{w + "/generate": "test" for w in workers}},
            callback_targets=[base + "/events"], context_length=4096, page_size=4)
        service = TokenizerService(settings, page_hashes)
        identity = RequestIdentity("agent-e2e", 0, 1)
        plans, calls = [], []
        try:
            async with service.lifespan(service.app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                                             base_url="http://tokenizer") as api:
                    prepared = (await api.post("/prepare", json={
                        "protocol_version": 1, "request_id": identity.engine_rid, "config_id": "test",
                        "input": {"prompt": "test", "add_special_tokens": False},
                        "generation": {"sampling_params": {"max_new_tokens": 4, "temperature": 0},
                                       "stream": True},
                    })).json()

                    def plan_for(body):
                        plans.append(late_plan(body, p=base + "/p", candidates=workers, default_idx=0,
                                               tag=TAG, callback=base + "/events"))
                        return plans[-1]

                    async with fake_router(plan_for, calls) as router, \
                            recording_tokenizer(calls, relay=api) as front:
                        async with Backend() as sdk:
                            out = b"".join([c async for c in sdk.stream_prepared(router, front, prepared,
                                                                                 identity=identity)])
                    assert out == b"".join(WINNER)
                    async with asyncio.timeout(3):
                        while not any(e["leg"] == "decode" and e["state"] == "completed" for e in events):
                            await asyncio.sleep(.005)
                        while service.pd_exchanges:
                            await asyncio.sleep(.005)
        finally:
            await upstream.close()
        assert [c[0] for c in calls] == ["place", "forward"]
        assert calls[1][1] == plans[0]
        p_body = next(body for path, body in seen if path == "/p/generate")
        assert p_body["late_d"] == plans[0]["prefill"]["late_d"]
        assert p_body["kv_export"] == plans[0]["prefill"]["kv_export"] and p_body["kv_session"] == TAG
        for i, cand in enumerate(plans[0]["decode_candidates"]):
            d_body = next(body for path, body in seen if path == f"/d{i}/generate")
            assert d_body["late_d"] == cand["late_d"] and d_body["kv_import"] == cand["kv_import"]
            assert "kv_session" not in d_body and d_body["input_ids"] == p_body["input_ids"]
        assert sorted((e["candidate"], e["state"]) for e in events if "candidate" in e) == [(2, "bound")]
    asyncio.run(scenario())
