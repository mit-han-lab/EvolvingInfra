"""Measured P/D step models and a one-time integer capacity decision.

This is a capacity model, not a queue-time simulator or a tail-latency proof.
All times are seconds. No H100/DeepSeek coefficients are embedded here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import nnls


def prefill_feature_schema(count):
    """The existing restore-aware model, with actual vision work when present."""
    fields = ['steps', 'new_tokens', 'attention_pairs', 'completed_restore_tokens']
    if count == 5:
        fields.append('vision_matmul_flops')
    elif count != 4:
        raise ValueError('Unknown prefill feature schema')
    return fields


def expert_fraction(tokens, experts: int, top_k: int):
    if not 0 < top_k <= experts:
        raise ValueError("Require 0 < top_k <= experts")
    tokens = np.asarray(tokens, dtype=float)
    if not np.isfinite(tokens).all() or (tokens < 0).any():
        raise ValueError("Token counts must be finite and nonnegative")
    if top_k == experts:
        return (tokens > 0).astype(float)
    return -np.expm1(tokens * math.log1p(-top_k / experts))


def fit_steps(features, seconds, *, minimum_samples=60, train_samples=None):
    """Chronological 70/30 holdout; scale columns before nonnegative fitting.

    Condition/rank are reported because accurate predictions do not imply that
    individual a,b,c coefficients are separately identifiable. A rank-deficient
    design is rejected instead of inventing a per-token coefficient.
    """
    x, y = np.asarray(features, dtype=float), np.asarray(seconds, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(x) < minimum_samples:
        raise ValueError("Insufficient step samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Non-finite features or nonpositive durations")
    split = int(len(y) * 0.7) if train_samples is None else train_samples
    if not isinstance(split, int) or not x.shape[1] <= split < len(y):
        raise ValueError("Invalid chronological training split")
    scale = np.maximum(np.linalg.norm(x[:split], axis=0), 1e-30)
    scaled = x[:split] / scale
    rank = int(np.linalg.matrix_rank(scaled))
    if rank != x.shape[1]:
        raise ValueError(
            f"Step coefficients not identifiable: rank {rank}/{x.shape[1]}"
        )
    coefficients = nnls(scaled, y[:split])[0] / scale
    prediction = x[split:] @ coefficients
    truth = y[split:]
    relative = np.abs(prediction - truth) / truth
    errors = dict(
        mean_relative=float(relative.mean()),
        p95_relative=float(np.quantile(relative, 0.95)),
        total_relative=float(abs(prediction.sum() / truth.sum() - 1)),
    )
    # Refit on all samples only AFTER evaluating the untouched holdout.
    full_scale = np.maximum(np.linalg.norm(x, axis=0), 1e-30)
    final = nnls(x / full_scale, y)[0] / full_scale
    return dict(
        coefficients=final.tolist(),
        holdout_coefficients=coefficients.tolist(),
        samples=len(y),
        train_samples=split,
        errors=errors,
        condition_number=float(np.linalg.cond(scaled)),
        rank=rank,
        feature_min=x.min(axis=0).tolist(),
        feature_max=x.max(axis=0).tolist(),
        passed=(
            errors["mean_relative"] <= 0.15
            and errors["p95_relative"] <= 0.30
            and errors["total_relative"] <= 0.10
        ),
    )


def prefill_request_seconds(
    n: int, hit: int, chunk: int, coefficients, *, host_hit=None, device_hit=None
) -> float:
    """Amortized service demand, NOT the isolated request's finish time.

    Sum causal attention pairs exactly. Attribute fixed overhead c in proportion
    to n/chunk, where chunk is the measured mean tokens per busy step. This
    reproduces aggregate fixed overhead at the observed packing efficiency.
    """
    if n < 0 or hit < 0 or chunk <= 0:
        raise ValueError("Invalid token count or chunk")
    if len(coefficients) not in (3, 5):
        raise ValueError("Unknown prefill feature schema")
    c, a, b = coefficients[:3]
    seconds = c * n / chunk + a * n + b * (n * hit + n * (n + 1) / 2)
    if len(coefficients) == 5:
        if host_hit is None or device_hit is None or min(host_hit, device_hit) < 0:
            raise ValueError("Cache-tier capacity requires measured host/device hits")
        if host_hit + device_hit != hit:
            raise ValueError("Cache tiers must account for the complete cached prefix")
        # Empirical preparation cost charged once per request, not each chunk.
        seconds += coefficients[3] * host_hit + coefficients[4] * device_hit
    return seconds


def select_ratio(
    *,
    nodes,
    concurrency,
    chunk,
    prefill_fit,
    decode_fit,
    prompt_samples,
    mean_decode_context,
    kv_capacity_tokens,
    max_running_requests,
    accepted_tokens_per_step,
    observed_decode_batches,
    experts,
    top_k,
    draft_tokens,
    mean_wait_s,
    packing_efficiency,
    safety_fraction=0.95,
    min_user_tps=100,
):
    """Evaluate integers within MEASURED batch support, then freeze externally.

    D's maximum batch obeys TPS, mean KV occupancy and observed support. Match
    saturated P/D capacities as in the paper. A population ceiling estimate is
    reported separately; an estimated average batch must not become a smaller
    hard admission cap. Remaining memory/tail risk is measured in the formal run.
    """
    if not prefill_fit["passed"] or not decode_fit["passed"]:
        raise ValueError("Held-out step-model validation failed")
    if not (nodes >= 2 and concurrency > 0 and min_user_tps > 0 and chunk > 0):
        raise ValueError("Invalid cluster or SLO")
    if not 0 < packing_efficiency <= 1.000001:
        raise ValueError("Invalid measured prefill packing efficiency")
    effective_chunk = chunk * packing_efficiency
    if not prompt_samples or mean_decode_context <= 0 or kv_capacity_tokens <= 0:
        raise ValueError("Missing workload or KV capacity measurements")
    if not 0 < safety_fraction <= 1 or not observed_decode_batches:
        raise ValueError("Missing measured decode support")
    if accepted_tokens_per_step <= 0 or mean_wait_s < 0:
        raise ValueError("Invalid acceptance/wait measurement")
    residual = decode_fit["errors"]["p95_relative"]
    if not math.isfinite(residual) or not 0 <= residual < 1:
        raise ValueError("Decode relative-error bound must be finite and below one")
    aggregate = prefill_fit.get("aggregate_request_features")
    if aggregate is not None:
        if (prefill_fit.get("feature_schema") != prefill_feature_schema(len(aggregate))
                or len(prefill_fit["coefficients"]) != len(aggregate)):
            raise ValueError("Unknown aggregate capacity feature schema")
        if not np.isfinite(aggregate).all() or min(aggregate) < 0:
            raise ValueError("Invalid aggregate service demand")
        demand = float(np.dot(aggregate, prefill_fit["coefficients"]))
    else:
        demand = float(np.mean(
            [
                prefill_request_seconds(
                    p["uncached"],
                    p["hit"],
                    effective_chunk,
                    prefill_fit["coefficients"],
                    host_hit=p.get("host_hit"),
                    device_hit=p.get("device_hit"),
                )
                for p in prompt_samples
            ]
        ))
    output = float(np.mean([p["output"] for p in prompt_samples]))
    if demand <= 0 or output <= 0:
        raise ValueError("Nonpositive service demand")
    # Reserve part of the KV capacity; a bound based on mean context is not
    # an assurance that every long-context tail fits simultaneously.
    memory_batch = math.floor(
        safety_fraction * kv_capacity_tokens / mean_decode_context
    )
    observed = sorted(set(int(b) for b in observed_decode_batches if b > 0))
    operational = []
    for batch in observed:
        if batch > min(memory_batch, max_running_requests, concurrency):
            continue
        features = [
            1.0,
            batch,
            float(expert_fraction(draft_tokens * batch, experts, top_k)),
        ]
        if len(decode_fit["coefficients"]) == 4:
            features.append(batch * mean_decode_context)
        step = float(np.dot(features, decode_fit["coefficients"]))
        # fit_steps normalizes residuals by OBSERVED time: |p-t|/t <= e
        # implies t <= p/(1-e), not p*(1+e). This is an empirical
        # held-out quantile correction, not a future per-request TPS guarantee.
        conservative_step = step / (1 - residual)
        user_tps = accepted_tokens_per_step / conservative_step
        if user_tps >= min_user_tps:
            operational.append(
                dict(
                    batch=batch,
                    step_s=step,
                    conservative_step_s=conservative_step,
                    conservative_user_tps=user_tps,
                    capacity_rps=batch * user_tps / output,
                )
            )
    if not operational:
        raise ValueError(
            f"No observed decode batch satisfies the requested {min_user_tps}-TPS/memory bound"
        )
    # Paper section 5.2: admission cap = min(speed, memory, configured bounds).
    # B/t(B) is monotone for the nonnegative linear + saturating-expert model.
    # A smaller actual batch in a closed loop does not require a smaller cap.
    op = max(operational, key=lambda r: r["batch"])
    fastest_decode_s = output / max(r["conservative_user_tps"] for r in operational)
    population_cap = concurrency / (demand + fastest_decode_s + mean_wait_s)
    rows = []
    for p in range(1, nodes):
        d = nodes - p
        predicted = min(p / demand, d * op["capacity_rps"])
        rows.append(
            dict(
                prefill_nodes=p,
                decode_nodes=d,
                prefill_capacity_rps=p / demand,
                decode_capacity_rps=d * op["capacity_rps"],
                population_ceiling_estimate_rps=population_cap,
                predicted_capacity_rps=predicted,
                **op,
            )
        )
    best_value = max(r["predicted_capacity_rps"] for r in rows)
    # Numerical ties prefer more D (as in the paper); no arbitrary post-run tuning.
    tied = [
        r
        for r in rows
        if math.isclose(
            r["predicted_capacity_rps"], best_value, rel_tol=1e-9, abs_tol=1e-12
        )
    ]
    chosen = max(tied, key=lambda r: (r["decode_nodes"], -r["batch"]))
    sensitivity = None
    if aggregate is not None:
        error = prefill_fit["errors"]["p95_relative"]
        if not 0 <= error < 1:
            raise ValueError("Invalid prefill capacity error envelope")
        interval = [demand/(1+error), demand/(1-error)]
        choices = []
        for value in interval:
            # Same objective and tie handling, at both empirical error bounds.
            scores = [(min(p/value, (nodes-p)*op["capacity_rps"]), nodes-p, p)
                      for p in range(1, nodes)]
            choices.append(max(scores)[2])
        if any(p != chosen["prefill_nodes"] for p in choices):
            raise ValueError("P/D choice changes within the empirical capacity error envelope")
        sensitivity = dict(service_demand_interval_s=interval,
                           prefill_nodes_at_bounds=choices,
                           note="Empirical sensitivity check, not a confidence interval or future guarantee")
    return dict(
        selected=chosen,
        candidates=rows,
        decode_batch_candidates=operational,
        prefill_service_demand_s=demand,
        mean_output_tokens=output,
        mean_decode_context=mean_decode_context,
        memory_batch_bound=memory_batch,
        min_user_tps=min_user_tps,
        packing_efficiency=packing_efficiency,
        mean_tokens_per_step=effective_chunk,
        prefill_error_sensitivity=sensitivity,
        limitations=[
            "capacity estimate, not p95/p99 guarantee",
            "mean KV occupancy cannot bound every context-length tail",
            "cache hit, step packing and transfer cost may change when roles change",
            "decode operating batch requires matching admission cap",
            "closed-loop traffic may not saturate the selected capacities",
        ],
    )


def fingerprint(configuration: dict) -> str:
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze(path: Path, report: dict, configuration: dict) -> dict:
    """Atomically create a decision ONCE. Never overwrite an existing ratio."""
    required = configuration.get("learning_duration_s", 900)
    if not isinstance(required, (int, float)) or not math.isfinite(required) or required <= 0:
        raise ValueError("Invalid learning window")
    if report.get("elapsed_s", 0) < required:
        raise ValueError(f"Cannot freeze before the full {required}-second learning window")
    if not (report.get("decision") or {}).get("selected"):
        raise ValueError("No validated ratio to freeze")
    if configuration.get("require_restore_collection_audit"):
        audit = report.get("prefill_measured_restore", {})
        if not audit.get("full_collection_interval_included") or not audit.get("passed"):
            raise ValueError("Cannot freeze without the required restore collection audit")
    if configuration.get('require_all_worker_traces'):
        coverage = report.get('worker_trace_coverage') or {}
        for role in ('prefill','decode'):
            counts = coverage.get(role) or {}
            if (len(counts) != configuration.get('initial_'+role+'_nodes')
                    or not counts or any(type(n) is not int or n <= 0 for n in counts.values())):
                raise ValueError('Cannot freeze without usable traces from every '+role+' worker')
    payload = dict(
        schema_version=1,
        state="frozen",
        configuration=configuration,
        configuration_sha256=fingerprint(configuration),
        report=report,
    )
    payload["payload_sha256"] = fingerprint(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)  # atomic no-replace, including concurrent writers
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def load_frozen(path: Path, configuration: dict) -> dict:
    payload = json.loads(Path(path).read_text())
    checksum = payload.pop("payload_sha256")
    if checksum != fingerprint(payload):
        raise ValueError("Frozen decision checksum mismatch")
    payload["payload_sha256"] = checksum
    if payload["state"] != "frozen" or payload["configuration_sha256"] != fingerprint(
        configuration
    ):
        raise ValueError("Frozen decision belongs to another experiment configuration")
    return payload
