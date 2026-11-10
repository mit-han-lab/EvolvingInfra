"""Strict-replay conformance for one worker or regular router endpoint.

Collects one completion per prompt, then verifies on the same endpoint chain:

1. Extended-field probe — ``forced_output_token_ids`` and ``return_token_ids``
   must round-trip losslessly through every hop. The first collected output ID
   is deterministically changed before forcing, so a gateway that drops the
   extension cannot pass merely because natural decoding is reproducible.
   The probe alone does not detect every stock gateway: a raw-passthrough
   completions route can pass it. The strict-replay gate below enforces the
   full request-admission contract and is the decisive stock-chain detector;
   probe and gate must always run together.
2. DSPARK boundary probe — native ``/generate`` must reproduce forced token
   sequences of lengths ``1, 2, W-1, W, W+1, 2W+1``, where ``W`` is the
   configured verify width.
3. Mixed-batch probe — the same native endpoint receives those forced requests
   concurrently with a longer normal request. Every forced request must finish
   while the normal request remains active. The normal request's exact content
   is NOT compared: batched kernels are not batch-invariant (greedy near-ties
   flip with batch composition; live V4/H100 runs produced three distinct
   benign variants, with the forced-neighbor variant also reachable under
   purely unforced neighbors). Instead the normal output must have the exact
   requested length and must not contain any token from the forced-pattern
   alphabet. Ordinary reference, control and mixed requests exclude that
   alphabet using a large finite negative logit bias; forced requests retain
   their exact sentinel sequences. Token IDs 0/1 are ordinary punctuation on
   GLM, so seeing either without this exclusion is not evidence of leakage.
   All ordinary controls must obey the exclusion before the mixed request is
   judged. Full native responses are retained next to --output for diagnosis.
   The two deterministic replays after this probe also check forced-state cleanup.
4. Two independent strict replays — per request, the blake2b hashes of the
   prompt/output token IDs and the server-reported acceptance-sequence digest
   (blake2b over the ordered per-verify-round correct-draft counts) must be
   identical across runs.
5. Workload integrity — the canonical workload content hash is recomputed
   after the runs; any tampering with the recorded traces fails the check.

The JSON report embeds the workload hash, per-request hashes/digests, and the
ambiguous-reconstruction count so run manifests can pin them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import re
import threading
import time
from typing import Any

import httpx
from openai import OpenAI

_NATIVE_CAPTURE_PATH: pathlib.Path | None = None
_NATIVE_CAPTURE_LOCK = threading.Lock()
_FORCED_ALPHABET = (0, 1)

from trie import Client, Generate, ReplayTrace


def _extra_field(value: Any, name: str) -> Any:
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    extra = getattr(value, "model_extra", None) or {}
    return extra.get(name)


def _blake2b_of_ids(token_ids: list[int]) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for token in token_ids:
        hasher.update(int(token).to_bytes(4, "little", signed=False))
    return hasher.hexdigest()


def _workload_hash(traces: list[dict[str, Any]]) -> str:
    canonical = json.dumps(traces, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()


def _native_base(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint[: -len("/v1")] if endpoint.endswith("/v1") else endpoint


def _verify_width_arg(value: str) -> int:
    width = int(value)
    # W=3 duplicates boundary length 2 (colliding forced-2 labels break
    # validate_report's by-label digest checks); W>=8 makes the mixed-batch
    # normal request longer than the 64 tokens validate_report hardcodes.
    if not 4 <= width <= 7:
        raise argparse.ArgumentTypeError(
            "verify width must be between 4 and 7 (validate_report contract)"
        )
    return width


def _boundary_lengths(verify_width: int) -> list[int]:
    return [
        1,
        2,
        verify_width - 1,
        verify_width,
        verify_width + 1,
        2 * verify_width + 1,
    ]


def _sentinel_ids_arg(value: str) -> tuple[int, int]:
    try:
        ids = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sentinels must be two distinct nonnegative token IDs") from exc
    if len(ids) != 2 or min(ids) < 0 or ids[0] == ids[1]:
        raise argparse.ArgumentTypeError("sentinels must be two distinct nonnegative token IDs")
    return ids


def _forced_pattern(length: int, variant: int = 0, alphabet=None) -> list[int]:
    """Alternate two caller-validated IDs, including single-token leakage cases."""
    alphabet = _FORCED_ALPHABET if alphabet is None else alphabet
    return [alphabet[(position + variant) % 2] for position in range(length)]


def _native_generate(
    client: httpx.Client,
    native_base: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    label: str,
    forced: list[int] | None = None,
    forbidden_tokens: set[int] | None = None,
) -> dict[str, Any]:
    sampling_params: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "n": 1,
        "ignore_eos": True,
        "max_new_tokens": max_new_tokens,
    }
    if forced is not None:
        if len(forced) != max_new_tokens:
            raise ValueError(f"{label}: forced length does not match max_new_tokens")
        sampling_params["forced_output_token_ids"] = list(forced)

    if forbidden_tokens:
        if forced is not None:
            raise ValueError("Forced probes must not also suppress their sentinel IDs")
        # Native SamplingParams permits finite logit biases without the OpenAI
        # [-100,100] restriction. Reserve the sentinel alphabet for forced lanes.
        sampling_params["logit_bias"] = {str(token): -1e9 for token in sorted(forbidden_tokens)}

    response = client.post(
        f"{native_base}/generate",
        json={
            "input_ids": list(prompt_ids),
            "return_prompt_token_ids": True,
            "sampling_params": sampling_params,
        },
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"{label}: native /generate did not return a JSON object")
    if _NATIVE_CAPTURE_PATH is not None:
        record = dict(at_ns=time.time_ns(), label=label, input_ids=list(prompt_ids),
                      sampling_params=sampling_params, response=body)
        with _NATIVE_CAPTURE_LOCK:
            with _NATIVE_CAPTURE_PATH.open("a") as capture:
                capture.write(json.dumps(record) + "\n")
    return body


def _native_output_ids(body: dict[str, Any], label: str) -> list[int]:
    output_ids = body.get("output_ids")
    if not isinstance(output_ids, list) or any(
        type(token) is not int for token in output_ids
    ):
        raise RuntimeError(f"{label}: native /generate returned invalid output_ids")
    return output_ids


def _native_prompt_ids(body: dict[str, Any]) -> list[int] | None:
    meta_info = body.get("meta_info") or {}
    return body.get("prompt_token_ids") or meta_info.get("prompt_token_ids")


def _forced_result(
    body: dict[str, Any],
    forced: list[int],
    prompt_ids: list[int],
    require_accept_digest: bool,
    label: str,
) -> dict[str, Any]:
    output_ids = _native_output_ids(body, label)
    if output_ids != forced:
        raise RuntimeError(
            f"{label}: returned token IDs diverge from the forced sequence: "
            f"forced[:8]={forced[:8]}, output[:8]={output_ids[:8]}"
        )
    echoed_prompt_ids = _native_prompt_ids(body)
    if echoed_prompt_ids != prompt_ids:
        raise RuntimeError(f"{label}: prompt_token_ids did not round-trip")

    meta_info = body.get("meta_info") or {}
    accept_sequence_digest = meta_info.get("spec_accept_sequence_digest")
    digest_required = require_accept_digest and len(forced) > 1
    if digest_required and not accept_sequence_digest:
        raise RuntimeError(
            f"{label}: meta_info lacks spec_accept_sequence_digest; expected on "
            "a speculative deployment after at least one verify round (use "
            "--no-require-accept-digest for no-spec control arms)"
        )
    return {
        "prompt_ids_blake2b": _blake2b_of_ids(echoed_prompt_ids),
        "output_ids_blake2b": _blake2b_of_ids(output_ids),
        "accept_sequence_digest": accept_sequence_digest,
        "accept_digest_required": digest_required,
        "spec_verify_ct": meta_info.get("spec_verify_ct"),
        "spec_num_correct_drafts": meta_info.get("spec_num_correct_drafts"),
        "spec_num_proposed_drafts": meta_info.get("spec_num_proposed_drafts"),
    }


def _collect(api: OpenAI, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    response = api.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        extra_body={"return_token_ids": True, "top_k": 1, "ignore_eos": True},
    )
    choice = response.choices[0]
    prompt_ids = _extra_field(choice, "prompt_token_ids")
    output_ids = _extra_field(choice, "token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise RuntimeError("SGLang did not return prompt_token_ids")
    if not isinstance(output_ids, list) or not output_ids:
        raise RuntimeError("SGLang did not return token_ids")
    return {
        "prompt": prompt,
        "text": choice.text,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": output_ids,
    }


def _probe_extended_fields(
    api: OpenAI, model: str, trace: dict[str, Any], index: int
) -> None:
    """Fail unless the forced sequence round-trips exactly through the chain."""
    forced = list(trace["output_token_ids"])
    # Token IDs 0 and 1 exist in every supported model vocabulary. Choosing the
    # one different from the collected token guarantees this probe cannot pass
    # when an intermediary drops forced_output_token_ids and decodes naturally.
    forced[0] = 1 if forced[0] == 0 else 0
    response = api.completions.create(
        model=model,
        prompt=trace["prompt"],
        max_tokens=len(forced),
        temperature=0,
        top_p=1,
        extra_body={
            "return_token_ids": True,
            "top_k": 1,
            "ignore_eos": True,
            "forced_output_token_ids": forced,
        },
    )
    choice = response.choices[0]
    echoed = _extra_field(choice, "token_ids")
    if echoed != forced:
        raise RuntimeError(
            f"extended-field probe failed for trace {index}: echoed token_ids "
            "diverge from forced_output_token_ids (a stock gateway dropping "
            f"extension fields causes exactly this): forced[:8]={forced[:8]}, "
            f"echoed[:8]={(echoed or [])[:8]}"
        )
    probe_prompt_ids = _extra_field(choice, "prompt_token_ids")
    if probe_prompt_ids != trace["prompt_token_ids"]:
        raise RuntimeError(
            f"extended-field probe failed for trace {index}: prompt_token_ids "
            "did not round-trip"
        )


def _forced_generate(
    client: httpx.Client,
    native_base: str,
    trace: dict[str, Any],
    require_accept_digest: bool,
    index: int,
) -> dict[str, Any]:
    """One forced strict-replay request via the native generate route."""
    forced = list(trace["output_token_ids"])
    label = f"strict replay trace {index}"
    body = _native_generate(
        client,
        native_base,
        trace["prompt_token_ids"],
        len(forced),
        label=label,
        forced=forced,
    )
    return _forced_result(
        body,
        forced,
        trace["prompt_token_ids"],
        require_accept_digest,
        label,
    )


def _boundary_probe(
    client: httpx.Client,
    native_base: str,
    trace: dict[str, Any],
    verify_width: int,
    require_accept_digest: bool,
) -> dict[str, Any]:
    """Exercise forced sequence endings around DSPARK verify boundaries."""
    requests = []
    for variant, length in enumerate(_boundary_lengths(verify_width)):
        label = f"DSPARK boundary length {length}"
        forced = _forced_pattern(length, variant)
        body = _native_generate(
            client,
            native_base,
            trace["prompt_token_ids"],
            length,
            label=label,
            forced=forced,
        )
        result = _forced_result(
            body,
            forced,
            trace["prompt_token_ids"],
            require_accept_digest,
            label,
        )
        requests.append({"forced_tokens": length, **result})
    return {
        "verify_width": verify_width,
        "lengths": _boundary_lengths(verify_width),
        "requests": requests,
        "status": "pass",
    }


def _mixed_batch_probe(
    client: httpx.Client,
    native_base: str,
    trace: dict[str, Any],
    verify_width: int,
    require_accept_digest: bool,
    lag_mode: str = "strict",
) -> dict[str, Any]:
    """Mix forced and normal requests, then verify filtering and state cleanup.

    ``lag_mode='strict'`` (collocated/single-engine chains) additionally
    requires the normal request to outlive every forced request, proving batch
    co-residency. On PD-disaggregated chains that ordering is not implied by
    the engine batch: prefill and decode placement differ per request and e2e
    latency includes per-pair bootstrap/transfer, so a 13-token forced request
    may legitimately finish after the 64-token normal one. ``lag_mode='overlap'``
    records the (possibly negative) lag without gating on its sign; the barrier
    start-spread gate still holds and the two deterministic replays remain the
    decisive gate for forced-state cleanup.
    """
    prompt_ids = trace["prompt_token_ids"]
    lengths = _boundary_lengths(verify_width)
    normal_tokens = max(8 * verify_width + 1, 64)
    forced_alphabet = {
        token
        for variant, length in enumerate(lengths)
        for token in _forced_pattern(length, variant)
    }
    reference_body = _native_generate(
        client,
        native_base,
        prompt_ids,
        normal_tokens,
        label="mixed-batch normal reference",
    )
    reference_ids = _native_output_ids(reference_body, "mixed-batch normal reference")
    if len(reference_ids) != normal_tokens:
        raise RuntimeError(
            "mixed-batch normal reference ended early despite ignore_eos=true: "
            f"expected {normal_tokens}, got {len(reference_ids)}"
        )
    if set(reference_ids) & forced_alphabet:
        raise RuntimeError("mixed-batch reference naturally emitted a sentinel; probe is not discriminating")
    analysis_window = normal_tokens
    reference_sig = reference_ids[:analysis_window]

    def _control_normal_ids(attempt: int) -> list[int]:
        """Sample the benign batch effect: identical neighbors, no forcing."""
        control_barrier = threading.Barrier(len(lengths) + 1)
        outputs: dict[str, list[int]] = {}

        def run_side(length: int) -> None:
            control_barrier.wait(timeout=30)
            _native_generate(
                client,
                native_base,
                prompt_ids,
                length,
                label=f"mixed-batch control-{attempt} side-{length}",
            )

        def run_normal() -> None:
            control_barrier.wait(timeout=30)
            body = _native_generate(
                client,
                native_base,
                prompt_ids,
                normal_tokens,
                label=f"mixed-batch control-{attempt} normal",
            )
            outputs["normal"] = _native_output_ids(
                body, f"mixed-batch control-{attempt} normal"
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(lengths) + 1
        ) as pool:
            futures = [pool.submit(run_side, length) for length in lengths]
            futures.append(pool.submit(run_normal))
            for future in concurrent.futures.as_completed(futures):
                future.result()
        ids = outputs["normal"]
        if len(ids) != normal_tokens:
            raise RuntimeError(
                f"mixed-batch control-{attempt} normal ended early despite "
                f"ignore_eos=true: expected {normal_tokens}, got {len(ids)}"
            )
        if set(ids) & forced_alphabet:
            raise RuntimeError(f"mixed-batch control-{attempt} naturally emitted a sentinel; probe is not discriminating")
        return ids

    control_ids = [_control_normal_ids(attempt) for attempt in range(3)]
    control_sigs = [ids[:analysis_window] for ids in control_ids]

    jobs: list[tuple[str, list[int] | None]] = [
        (f"forced-{length}", _forced_pattern(length, variant))
        for variant, length in enumerate(lengths)
    ]
    jobs.append(("normal", None))
    barrier = threading.Barrier(len(jobs))

    def run_one(label: str, forced: list[int] | None) -> dict[str, Any]:
        barrier.wait(timeout=30)
        started = time.monotonic()
        max_new_tokens = len(forced) if forced is not None else normal_tokens
        body = _native_generate(
            client,
            native_base,
            prompt_ids,
            max_new_tokens,
            label=f"mixed-batch {label}",
            forced=forced,
        )
        finished = time.monotonic()
        if forced is None:
            output_ids = _native_output_ids(body, "mixed-batch normal")
            if len(output_ids) != normal_tokens:
                raise RuntimeError(
                    "mixed-batch normal ended early despite ignore_eos=true: "
                    f"expected {normal_tokens}, got {len(output_ids)}"
                )
            window_ids = output_ids[:analysis_window]
            bleed = sorted(set(window_ids) & forced_alphabet)
            if bleed:
                raise RuntimeError(
                    "mixed-batch normal output contains forced-pattern tokens "
                    f"{bleed}; forced request state "
                    "leaked"
                )
            if window_ids == reference_sig:
                matched_via = "solo_reference"
            else:
                matched_via = next(
                    (
                        f"control-{idx}"
                        for idx, control in enumerate(control_sigs)
                        if window_ids == control
                    ),
                    "batch_variant",
                )
            result = {
                "kind": "normal",
                "output_tokens": len(output_ids),
                "output_ids_blake2b": _blake2b_of_ids(window_ids),
                "matched_via": matched_via,
                "forced_alphabet_bleed": False,
            }
        else:
            result = {
                "kind": "forced",
                "forced_tokens": len(forced),
                **_forced_result(
                    body,
                    forced,
                    prompt_ids,
                    require_accept_digest,
                    f"mixed-batch {label}",
                ),
            }
        return {"label": label, "started": started, "finished": finished, **result}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(run_one, label, forced) for label, forced in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    start_zero = min(result["started"] for result in results)
    start_spread_ms = (
        max(result["started"] for result in results) - start_zero
    ) * 1000
    if start_spread_ms > 1000:
        raise RuntimeError(
            "mixed-batch requests did not start together: "
            f"start spread was {start_spread_ms:.3f} ms"
        )

    normal_result = next(result for result in results if result["kind"] == "normal")
    forced_results = [
        result for result in results if result["kind"] == "forced"
    ]
    latest_forced_result = max(
        forced_results, key=lambda result: result["finished"]
    )
    completion_lag_ms = (
        normal_result["finished"] - latest_forced_result["finished"]
    ) * 1000
    if completion_lag_ms <= 0 and lag_mode == "strict":
        raise RuntimeError(
            "mixed-batch probe did not keep the normal request active after all "
            "forced requests finished"
        )

    rendered_results = []
    for result in sorted(results, key=lambda item: item["label"]):
        rendered_results.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"started", "finished"}
            }
            | {
                "start_offset_ms": round((result["started"] - start_zero) * 1000, 3),
                "latency_ms": round(
                    (result["finished"] - result["started"]) * 1000, 3
                ),
            }
        )
    return {
        "verify_width": verify_width,
        "forced_lengths": lengths,
        "normal_tokens": normal_tokens,
        "analysis_window": analysis_window,
        "normal_reference_blake2b": _blake2b_of_ids(reference_sig),
        "control_normal_blake2b": [_blake2b_of_ids(sig) for sig in control_sigs],
        "start_spread_ms": round(start_spread_ms, 3),
        "completion_lag_mode": lag_mode,
        "normal_completion_lag_after_all_forced_ms": round(completion_lag_ms, 3),
        "requests": rendered_results,
        "status": "pass",
    }


def _trie_replay_gate(
    endpoint: str, model: str, api_key: str, traces: list[dict[str, Any]]
) -> dict[str, Any]:
    """End-to-end strict replay of every trace through the Trie client."""
    replay_traces = [
        ReplayTrace(
            trace_id=f"conformance/sglang-strict-replay-{index}",
            metadata={"routing_key": f"conformance-session-{index}"},
            events=[
                Generate(
                    max_tokens=len(trace["output_token_ids"]),
                    recorded_assistant={
                        "role": "assistant",
                        "content": trace["text"],
                    },
                    recorded_prompt_token_ids=trace["prompt_token_ids"],
                    recorded_output_token_ids=trace["output_token_ids"],
                    source_model=model,
                )
            ],
        )
        for index, trace in enumerate(traces)
    ]
    result = Client(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        max_retries=0,
        timeout=600,
    ).sync_run(
        replay_traces,
        concurrency=1,
        duration=600,
        replay_once=True,
        drain_timeout=600,
        stream=False,
        context_source="recorded",
        strict_replay=True,
    )
    if result.failed_requests or result.completed_requests != len(traces):
        raise RuntimeError(f"trie strict replay gate failed: {result}")
    return {
        "completed_requests": result.completed_requests,
        "failed_requests": result.failed_requests,
    }


def _load_report(path: str) -> dict[str, Any]:
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    with open(path, encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=no_duplicate_keys)


def validate_report(
    report: dict[str, Any],
    *,
    expected_endpoint: str,
    expected_model: str,
    expected_width: int,
    expected_workload_blake2b: str,
    expected_sentinel_token_ids: tuple[int, int] = (0, 1),
) -> None:
    """Validate every field consumed as strict-replay evidence."""

    def fail(message: str) -> None:
        raise ValueError(f"invalid conformance report: {message}")

    def exact_keys(value: Any, expected: set[str], path: str) -> None:
        if type(value) is not dict:
            fail(f"{path} is not an object")
        actual = set(value)
        if actual != expected:
            fail(
                f"{path} keys differ: missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )

    def integer(value: Any, path: str, minimum: int | None = None) -> int:
        if type(value) is not int or (minimum is not None and value < minimum):
            fail(f"{path} is invalid: {value!r}")
        return value

    def number(value: Any, path: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(value):
            fail(f"{path} is not finite: {value!r}")
        return float(value)

    def hex32(value: Any, path: str) -> str:
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{32}", value) is None:
            fail(f"{path} is not lowercase BLAKE2b-128")
        return value

    def ids_hash(values: list[int]) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for token in values:
            digest.update(token.to_bytes(4, "little", signed=False))
        return digest.hexdigest()

    top_keys = {
        "endpoint",
        "max_tokens",
        "model",
        "prompt_count",
        "require_accept_digest",
        "verify_width",
        "workload_blake2b",
        "ambiguous_reconstruction_events",
        "traces",
        "extended_field_probe",
        "boundary_probe",
        "mixed_batch_probe",
        "runs",
        "determinism",
    }
    forced_keys = {
        "prompt_ids_blake2b",
        "output_ids_blake2b",
        "accept_sequence_digest",
        "accept_digest_required",
        "spec_verify_ct",
        "spec_num_correct_drafts",
        "spec_num_proposed_drafts",
    }
    if "sentinel_token_ids" in report:
        top_keys.add("sentinel_token_ids")
    exact_keys(report, top_keys, "report")
    if report.get("sentinel_token_ids", [0, 1]) != list(expected_sentinel_token_ids):
        fail("sentinel token IDs mismatch")

    width = integer(report["verify_width"], "verify_width", 3)
    if width != expected_width:
        fail("verify_width mismatch")
    if report["endpoint"] != expected_endpoint.rstrip("/"):
        fail(f"endpoint mismatch: {report['endpoint']!r}")
    if report["model"] != expected_model:
        fail(f"model mismatch: {report['model']!r}")
    if integer(report["max_tokens"], "max_tokens") != 32:
        fail("max_tokens must be 32")
    if integer(report["prompt_count"], "prompt_count") != 2:
        fail("prompt_count must be 2")
    if report["require_accept_digest"] is not True:
        fail("require_accept_digest must be true")
    if integer(
        report["ambiguous_reconstruction_events"], "ambiguous count"
    ) != 0:
        fail("ambiguous reconstruction count must be zero")
    if report["extended_field_probe"] != "pass" or report["determinism"] != "pass":
        fail("top-level probes did not pass")

    workload = hex32(report["workload_blake2b"], "workload_blake2b")
    if workload != hex32(
        expected_workload_blake2b, "expected workload_blake2b"
    ):
        fail(f"workload_blake2b mismatch: {workload}")

    traces = report["traces"]
    if type(traces) is not list or len(traces) != 2:
        fail("expected exactly two traces")
    for index, trace in enumerate(traces):
        exact_keys(trace, {"prompt_tokens", "output_tokens"}, f"traces[{index}]")
        integer(trace["prompt_tokens"], f"traces[{index}].prompt_tokens", 1)
        if integer(trace["output_tokens"], f"traces[{index}].output_tokens", 1) != 32:
            fail(f"traces[{index}].output_tokens must be 32")

    def validate_forced(
        item: dict[str, Any],
        tokens: int,
        path: str,
        expected_output: str | None = None,
    ) -> str:
        prompt_digest = hex32(item["prompt_ids_blake2b"], f"{path}.prompt hash")
        output_digest = hex32(item["output_ids_blake2b"], f"{path}.output hash")
        required = tokens > 1
        if item["accept_digest_required"] is not required:
            fail(f"{path}.accept_digest_required mismatch")

        digest = item["accept_sequence_digest"]
        stats = [
            item["spec_verify_ct"],
            item["spec_num_correct_drafts"],
            item["spec_num_proposed_drafts"],
        ]
        if required or digest is not None or any(value is not None for value in stats):
            hex32(digest, f"{path}.accept_sequence_digest")
            verify = integer(stats[0], f"{path}.spec_verify_ct", 1)
            correct = integer(stats[1], f"{path}.spec_num_correct_drafts", 0)
            proposed = integer(stats[2], f"{path}.spec_num_proposed_drafts", 1)
            if proposed != verify * (width - 1):
                fail(f"{path} speculative width evidence mismatch")
            if correct > proposed:
                fail(f"{path} correct drafts exceed proposed drafts")
        elif stats != [None, None, None]:
            fail(f"{path} has partial speculative statistics")

        if expected_output is not None and output_digest != expected_output:
            fail(f"{path} forced output digest mismatch")
        return prompt_digest

    lengths = _boundary_lengths(width)
    boundary = report["boundary_probe"]
    exact_keys(
        boundary, {"verify_width", "lengths", "requests", "status"}, "boundary_probe"
    )
    if boundary["status"] != "pass" or boundary["verify_width"] != width:
        fail("boundary probe configuration mismatch")
    if boundary["lengths"] != lengths:
        fail(f"boundary lengths mismatch: {boundary['lengths']!r}")
    if type(boundary["requests"]) is not list or len(boundary["requests"]) != 6:
        fail("boundary probe must contain six requests")

    trace0_digest = None
    for variant, (request, length) in enumerate(
        zip(boundary["requests"], lengths)
    ):
        path = f"boundary_probe.requests[{variant}]"
        exact_keys(request, forced_keys | {"forced_tokens"}, path)
        if integer(request["forced_tokens"], f"{path}.forced_tokens") != length:
            fail(f"{path} forced length mismatch")
        prompt_digest = validate_forced(
            request,
            length,
            path,
            ids_hash(_forced_pattern(length, variant, expected_sentinel_token_ids)),
        )
        trace0_digest = trace0_digest or prompt_digest
        if prompt_digest != trace0_digest:
            fail("boundary requests do not share trace-zero prompt")

    mixed = report["mixed_batch_probe"]
    mixed_keys = {
        "verify_width",
        "forced_lengths",
        "normal_tokens",
        "analysis_window",
        "normal_reference_blake2b",
        "control_normal_blake2b",
        "start_spread_ms",
        "completion_lag_mode",
        "normal_completion_lag_after_all_forced_ms",
        "requests",
        "status",
    }
    exact_keys(mixed, mixed_keys, "mixed_batch_probe")
    if mixed["status"] != "pass" or mixed["verify_width"] != width:
        fail("mixed-batch configuration mismatch")
    if mixed["forced_lengths"] != lengths:
        fail("mixed-batch forced lengths mismatch")
    if integer(mixed["normal_tokens"], "mixed normal_tokens") != 64:
        fail("mixed-batch normal_tokens must be 64")
    window = integer(mixed["analysis_window"], "mixed analysis_window")
    if not 4 * width + 2 <= window <= 64:
        fail("mixed-batch analysis window out of range")
    spread = number(mixed["start_spread_ms"], "mixed start spread")
    lag = number(
        mixed["normal_completion_lag_after_all_forced_ms"],
        "mixed completion lag",
    )
    lag_mode = mixed["completion_lag_mode"]
    if lag_mode not in ("strict", "overlap"):
        fail("mixed-batch completion_lag_mode is invalid")
    if not 0 <= spread <= 1000 or (lag_mode == "strict" and lag <= 0):
        fail("mixed-batch timing gate failed")

    requests = mixed["requests"]
    expected_labels = sorted([f"forced-{length}" for length in lengths] + ["normal"])
    if type(requests) is not list or len(requests) != 7:
        fail("mixed-batch must contain seven requests")
    if [request.get("label") for request in requests if type(request) is dict] != expected_labels:
        fail("mixed-batch request labels or order mismatch")
    by_label = {request["label"]: request for request in requests}

    def validate_timing(request: dict[str, Any], path: str) -> float:
        offset = number(request["start_offset_ms"], f"{path}.start_offset_ms")
        latency = number(request["latency_ms"], f"{path}.latency_ms")
        if offset < 0 or offset > spread + 0.002 or latency <= 0:
            fail(f"{path} timing is invalid")
        return offset + latency

    forced_ends = []
    for variant, length in enumerate(lengths):
        label = f"forced-{length}"
        request = by_label[label]
        path = f"mixed_batch_probe.{label}"
        exact_keys(
            request,
            forced_keys
            | {"label", "kind", "forced_tokens", "start_offset_ms", "latency_ms"},
            path,
        )
        if request["kind"] != "forced" or request["forced_tokens"] != length:
            fail(f"{path} identity mismatch")
        prompt_digest = validate_forced(
            request,
            length,
            path,
            ids_hash(_forced_pattern(length, variant, expected_sentinel_token_ids)),
        )
        if prompt_digest != trace0_digest:
            fail(f"{path} prompt digest differs from trace zero")
        forced_ends.append(validate_timing(request, path))

    normal = by_label["normal"]
    normal_keys = {
        "label",
        "kind",
        "output_tokens",
        "output_ids_blake2b",
        "matched_via",
        "forced_alphabet_bleed",
        "start_offset_ms",
        "latency_ms",
    }
    exact_keys(normal, normal_keys, "mixed_batch_probe.normal")
    if normal["kind"] != "normal" or normal["output_tokens"] != 64:
        fail("mixed-batch normal request mismatch")
    if normal["forced_alphabet_bleed"] is not False:
        fail("mixed normal request must record forced_alphabet_bleed=false")
    controls = mixed["control_normal_blake2b"]
    if type(controls) is not list or len(controls) != 3:
        fail("mixed-batch must record exactly three unforced-neighbor controls")
    control_digests = [
        hex32(item, f"mixed control digest {index}")
        for index, item in enumerate(controls)
    ]
    normal_digest = hex32(normal["output_ids_blake2b"], "mixed normal output hash")
    reference_digest = hex32(
        mixed["normal_reference_blake2b"], "mixed normal reference hash"
    )
    matched_via = normal["matched_via"]
    if matched_via == "solo_reference":
        if normal_digest != reference_digest:
            fail("mixed normal output does not match its claimed benign reference")
    elif type(matched_via) is str and re.fullmatch(r"control-[0-2]", matched_via):
        if normal_digest != control_digests[int(matched_via.rsplit("-", 1)[1])]:
            fail("mixed normal output does not match its claimed benign reference")
    elif matched_via == "batch_variant":
        if normal_digest == reference_digest or normal_digest in control_digests:
            fail("mixed normal batch_variant claim contradicts recorded digests")
    else:
        fail("mixed normal matched_via is invalid")
    normal_end = validate_timing(normal, "mixed_batch_probe.normal")
    if abs((normal_end - max(forced_ends)) - lag) > 0.01:
        fail("mixed-batch completion lag is internally inconsistent")

    runs = report["runs"]
    if type(runs) is not list or len(runs) != 2:
        fail("expected exactly two replay runs")
    run_requests = []
    for repeat, run in enumerate(runs):
        path = f"runs[{repeat}]"
        exact_keys(run, {"repeat", "trie_gate", "requests"}, path)
        if integer(run["repeat"], f"{path}.repeat") != repeat:
            fail(f"{path} repeat number mismatch")
        gate = run["trie_gate"]
        exact_keys(gate, {"completed_requests", "failed_requests"}, f"{path}.trie_gate")
        if integer(gate["completed_requests"], f"{path}.completed") != 2:
            fail(f"{path} did not complete two Trie requests")
        if integer(gate["failed_requests"], f"{path}.failed") != 0:
            fail(f"{path} has failed Trie requests")
        if type(run["requests"]) is not list or len(run["requests"]) != 2:
            fail(f"{path} must contain two request records")
        for index, request in enumerate(run["requests"]):
            request_path = f"{path}.requests[{index}]"
            exact_keys(request, forced_keys, request_path)
            prompt_digest = validate_forced(request, 32, request_path)
            if index == 0 and prompt_digest != trace0_digest:
                fail(f"{request_path} prompt digest differs from boundary probe")
        if (
            run["requests"][0]["prompt_ids_blake2b"]
            == run["requests"][1]["prompt_ids_blake2b"]
        ):
            fail(f"{path} unexpectedly contains duplicate prompts")
        run_requests.append(run["requests"])

    if run_requests[0] != run_requests[1]:
        fail("two replay runs are not fully deterministic")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="repeatable; defaults to two built-in prompts",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--mixed-batch-lag",
        choices=("strict", "overlap"),
        default="strict",
        help=(
            "strict (default): the mixed-batch normal request must outlive all "
            "forced requests (single-engine chains); overlap: record the lag "
            "without gating its sign (PD-disaggregated chains, where per-pair "
            "placement breaks the ordering guarantee)"
        ),
    )
    parser.add_argument(
        "--verify-width",
        type=_verify_width_arg,
        default=6,
        help=(
            "DSPARK verify width W used by boundary and mixed-batch probes "
            "(default: 6)"
        ),
    )
    parser.add_argument(
        "--no-require-accept-digest",
        action="store_false",
        dest="require_accept_digest",
        help="tolerate missing acceptance digests (no-spec control arms)",
    )
    parser.add_argument(
        "--no-accept-digest-determinism",
        action="store_false",
        dest="accept_digest_determinism",
        help=(
            "leave the acceptance digest out of the two-run determinism check, for a natural "
            "(non-oracle) draft whose accept sequence does not repeat exactly; prompt and output "
            "IDs are still compared"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="also write the report JSON to this path (stdout carries client logs)",
    )
    parser.add_argument(
        "--validate-report",
        help=(
            "validate an existing evidence report instead of sending requests; "
            "only accepts default-configuration reports (max-tokens 32, the two "
            "built-in prompts, accept digest required)"
        ),
    )
    parser.add_argument(
        "--expected-workload-blake2b",
        help="required workload anchor for --validate-report",
    )
    parser.add_argument(
        "--sentinel-token-ids", type=_sentinel_ids_arg, default=(0, 1),
        help="Two valid model-vocabulary IDs excluded from normal probe output; recorded in evidence. GLM callers use unused output-head rows, not punctuation IDs 0/1.")
    args = parser.parse_args()
    global _FORCED_ALPHABET
    _FORCED_ALPHABET = args.sentinel_token_ids
    global _NATIVE_CAPTURE_PATH
    if args.output:
        _NATIVE_CAPTURE_PATH = pathlib.Path(str(args.output) + ".native.jsonl")
        _NATIVE_CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.validate_report:
        if not args.endpoint or not args.model or not args.expected_workload_blake2b:
            parser.error(
                "--validate-report requires --endpoint, --model, and "
                "--expected-workload-blake2b"
            )
        try:
            report = _load_report(args.validate_report)
            validate_report(
                report,
                expected_endpoint=args.endpoint,
                expected_model=args.model,
                expected_width=args.verify_width,
                expected_workload_blake2b=args.expected_workload_blake2b,
                expected_sentinel_token_ids=args.sentinel_token_ids,
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(f"CONFORMANCE_REPORT_VALID {args.validate_report}")
        return 0
    if not args.endpoint or not args.model:
        parser.error("request mode requires --endpoint and --model")

    prompts = args.prompts or [
        "Return one short sentence about the weather.",
        "List three prime numbers separated by commas.",
    ]

    api = OpenAI(base_url=args.endpoint, api_key=args.api_key, timeout=600)
    native_base = _native_base(args.endpoint)

    traces = [_collect(api, args.model, p, args.max_tokens) for p in prompts]
    workload_content = [
        {
            "prompt": t["prompt"],
            "prompt_token_ids": t["prompt_token_ids"],
            "output_token_ids": t["output_token_ids"],
        }
        for t in traces
    ]
    workload_blake2b = _workload_hash(workload_content)

    for index, trace in enumerate(traces):
        _probe_extended_fields(api, args.model, trace, index)

    with httpx.Client(timeout=600) as native_client:
        boundary_probe = _boundary_probe(
            native_client,
            native_base,
            traces[0],
            args.verify_width,
            args.require_accept_digest,
        )
        mixed_batch_probe = _mixed_batch_probe(
            native_client,
            native_base,
            traces[0],
            args.verify_width,
            args.require_accept_digest,
            lag_mode=args.mixed_batch_lag,
        )

        runs = []
        for repeat in range(2):
            gate = _trie_replay_gate(args.endpoint, args.model, args.api_key, traces)
            per_request = [
                _forced_generate(
                    native_client,
                    native_base,
                    trace,
                    args.require_accept_digest,
                    index,
                )
                for index, trace in enumerate(traces)
            ]
            runs.append(
                {"repeat": repeat, "trie_gate": gate, "requests": per_request}
            )

    mismatches = []
    for index in range(len(traces)):
        first, second = runs[0]["requests"][index], runs[1]["requests"][index]
        keys = ["prompt_ids_blake2b", "output_ids_blake2b"]
        if args.accept_digest_determinism:
            keys.append("accept_sequence_digest")
        for key in keys:
            if first[key] != second[key]:
                mismatches.append(
                    {
                        "trace": index,
                        "field": key,
                        "run0": first[key],
                        "run1": second[key],
                    }
                )
    if mismatches:
        raise RuntimeError(f"two-run determinism failed: {mismatches}")

    if _workload_hash(workload_content) != workload_blake2b:
        raise RuntimeError("workload content hash changed during the run")

    report = {
        "sentinel_token_ids": list(args.sentinel_token_ids),
        "endpoint": args.endpoint.rstrip("/"),
        "max_tokens": args.max_tokens,
        "model": args.model,
        "prompt_count": len(prompts),
        "require_accept_digest": args.require_accept_digest,
        "verify_width": args.verify_width,
        "workload_blake2b": workload_blake2b,
        # Self-collected traces carry authoritative token IDs, so the
        # text-normalization fallback never runs here. Driver-based replays of
        # text traces must pin the client-reported ambiguous count into their
        # run manifest instead.
        "ambiguous_reconstruction_events": 0,
        "traces": [
            {
                "prompt_tokens": len(t["prompt_token_ids"]),
                "output_tokens": len(t["output_token_ids"]),
            }
            for t in traces
        ],
        "extended_field_probe": "pass",
        "boundary_probe": boundary_probe,
        "mixed_batch_probe": mixed_batch_probe,
        "runs": runs,
        "determinism": "pass",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        path = pathlib.Path(args.output)
        temporary = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        try:
            with temporary.open("x") as output:
                output.write(rendered)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
