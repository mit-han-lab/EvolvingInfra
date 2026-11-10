"""Explicitly invoked P/D calibration and capacity recommendations.

The scheduler supplies normalized measurements; this module neither polls an
engine nor starts a timer, places requests, or changes worker roles. It reuses
``evolving_infra.planner.calibration.model`` so offline experiments and the public Python API share
one numerical implementation. Install ``evolving-infra[planner]`` to
enable the optional numerical dependencies.

Times are seconds and capacities are completed requests per second. Calibration
durations must represent busy service time, including the overhead being modeled,
not queueing or idle time. GPU kernel time alone is not end-to-end service demand.
Measurements must come from one homogeneous engine configuration, in chronological
order per role. Mixed P/D steps require a separate model and are not supported.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["calibrate", "recommend"]


def _copy(value: Any) -> Any:
    """Copy JSON input without aliasing caller state; reject nonfinite values.

    Args: value: A JSON-compatible configuration, observation, or model.
    Returns: An independent JSON value. Raises ValueError for invalid input.
    """
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Expected finite JSON-compatible data") from error


def _fingerprint(value: Any) -> str:
    """Hash a JSON value deterministically for configuration/model validation.

    Args: value: Finite JSON-compatible data.
    Returns: Its SHA-256 hexadecimal digest, independent of dictionary key order.
    """
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    """Validate a physical measurement without accepting booleans or strings.

    Args: value: Numeric measurement; name: Error label; positive: Require > 0.
    Returns: A finite float. Raises ValueError for negative/invalid measurements.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    """Validate an integer configuration/count field.

    Args: value: Integer; name: Error label; minimum: Inclusive lower bound.
    Returns: The integer. Raises ValueError for bool, fractions or small values.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _configuration(engine_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and snapshot the homogeneous engine configuration.

    Args: engine_config: Model/GPU identities, precisions, TP/chunk and MoE/spec
        settings. Additional JSON fields (e.g. engine/model revisions, host cache
        settings) are preserved and also participate in compatibility checks.
    Returns: An independent configuration. Raises ValueError for missing fields.
    """
    config = _copy(dict(engine_config))
    for name in ("model_id", "gpu_type", "weight_dtype", "kv_cache_dtype"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"engine_config requires nonempty {name}")
    for name in ("tp_size", "chunk_size", "experts", "top_k", "draft_tokens"):
        _integer(config.get(name), name)
    if config["top_k"] > config["experts"]:
        raise ValueError("top_k must not exceed experts")
    return config


def _core():
    """Load the shared solver only when the scheduler requests a calculation.

    Args: None.
    Returns: The legacy numerical module; raises ImportError with install guidance
        when optional NumPy/SciPy dependencies have not been installed.
    """
    try:
        from evolving_infra.planner.calibration import model
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in {"numpy", "scipy"}:
            raise ImportError("Install evolving-infra[planner] for P/D planning") from error
        raise
    return model


def _fit(rows: Sequence[Mapping[str, Any]], role: str, config: dict, minimum: int) -> dict:
    """Construct named measurement features and fit one role with holdout checks.

    Args: rows: Chronologically ordered busy-step samples; role: prefill/decode;
        config: Validated engine configuration; minimum: Minimum sample count.
    Returns: Shared solver fit and feature schema. Raises ValueError for malformed,
        insufficient, mixed-schema, or rank-deficient samples.
    """
    if not rows:
        raise ValueError(f"No {role} step samples")
    extra = "restored_tokens" if role == "prefill" else "context_tokens"
    with_extra = [extra in row for row in rows]
    if any(with_extra) != all(with_extra):
        raise ValueError(f"{extra} must be present in every {role} sample or none")
    features, seconds = [], []
    schema = (["steps", "new_tokens", "attention_pairs"] if role == "prefill"
              else ["intercept", "batch_size", "expert_fraction"])
    if all(with_extra):
        schema.append("completed_restore_tokens" if role == "prefill" else "context_tokens")
    core = _core()
    for row in rows:
        seconds.append(_number(row["duration_s"], "duration_s", positive=True))
        if role == "prefill":
            values = [1.0, _number(row["uncached_tokens"], "uncached_tokens"),
                      _number(row["attention_pairs"], "attention_pairs")]
        else:
            batch = _integer(row["batch_size"], "batch_size")
            # For dense models (experts == top_k), the expert term is constant
            # and indistinguishable from the intercept. Fit the identifiable two
            # columns, then expand to the shared solver's three-column schema.
            values = [1.0, batch]
            if config["experts"] != config["top_k"]:
                values.append(float(core.expert_fraction(
                    batch * config["draft_tokens"], config["experts"], config["top_k"])))
        if all(with_extra):
            values.append(_number(row[extra], extra))
        features.append(values)
    fitted = core.fit_steps(features, seconds, minimum_samples=minimum)
    if role == "decode" and config["experts"] == config["top_k"]:
        for key in ("coefficients", "holdout_coefficients"):
            fitted[key].insert(2, 0.0)
        for key in ("feature_min", "feature_max"):
            fitted[key].insert(2, 1.0)
        fitted["constant_expert_term_folded_into_intercept"] = True
    fitted["feature_schema"] = schema
    return fitted


def calibrate(step_samples: Mapping[str, Sequence[Mapping[str, Any]]],
              engine_config: Mapping[str, Any], *, minimum_samples: int = 60) -> dict[str, Any]:
    """Learn P/D service-time models without interacting with the serving system.

    Args:
        step_samples: ``prefill`` and ``decode`` lists in time order. Every sample
            has ``duration_s``. P samples have ``uncached_tokens`` and
            ``attention_pairs`` (sum over participating requests); optionally all
            have ``restored_tokens`` from completed transfers. D samples have
            ``batch_size``; optionally all have ``context_tokens`` (batch sum).
        engine_config: See _configuration; include revisions and cache/transport
            settings needed to identify the measured execution environment.
        minimum_samples: Minimum number of samples per role, default 60.

    Returns:
        A JSON-compatible model with configuration/digest, both fits, measured D
        batch support, ``passed`` and ``reason``. Insufficient/unidentifiable or
        inaccurate data produces passed=False; callers keep their current ratio.
        Malformed configuration raises ValueError; missing dependencies raise
        ImportError. No input is modified, and nothing is automatically saved.
    """
    config = _configuration(engine_config)
    minimum = _integer(minimum_samples, "minimum_samples", minimum=2)
    _core()
    result = {"schema_version": 1, "configuration": config,
              "configuration_sha256": _fingerprint(config)}
    reasons = []
    for role in ("prefill", "decode"):
        try:
            rows = step_samples.get(role, [])
            fitted = _fit(rows, role, config, minimum)
            if not fitted["passed"]:
                reasons.append(f"{role}: chronological holdout error exceeds limits")
        except (ValueError, KeyError, TypeError) as error:
            fitted = {"passed": False, "error": str(error)}
            reasons.append(f"{role}: {error}")
        result[f"{role}_fit"] = fitted
    result["observed_decode_batches"] = (sorted({r["batch_size"] for r in step_samples["decode"]})
                                           if result["decode_fit"]["passed"] else [])
    contexts = ([r["context_tokens"] / r["batch_size"] for r in step_samples["decode"]]
                if result["decode_fit"]["passed"]
                and "context_tokens" in result["decode_fit"]["feature_schema"] else [])
    result["decode_context_per_request_range"] = [min(contexts), max(contexts)] if contexts else None
    result["passed"] = not reasons
    result["reason"] = "; ".join(reasons) if reasons else "Both chronological holdout checks passed"
    result["model_sha256"] = _fingerprint(result)
    return result


def _verified_model(model, engine_config):
    """Copy a frozen model and verify its content and execution configuration."""
    fitted = _copy(dict(model))
    digest = fitted.pop("model_sha256")
    if fitted.get("schema_version") != 1 or digest != _fingerprint(fitted):
        raise ValueError("Calibration model checksum/version mismatch")
    config = _configuration(engine_config)
    if (fitted["configuration"] != config
            or fitted["configuration_sha256"] != _fingerprint(config)):
        raise ValueError("Engine configuration differs from calibration; recalibrate")
    if not fitted["passed"]:
        raise ValueError(f"Calibration unavailable: {fitted['reason']}")
    return fitted, digest, config


def recommend(model: Mapping[str, Any], workload: Mapping[str, Any], total_nodes: int,
              constraints: Mapping[str, Any], *, engine_config: Mapping[str, Any]) -> dict[str, Any]:
    """Recommend an integer P/D allocation; do not change admission or worker roles.

    Args:
        model: JSON model returned by calibrate (or its JSON round trip).
        workload: ``requests`` with actual completed ``uncached``, ``hit``,
            ``output`` token counts; ``mean_decode_context``,
            ``accepted_tokens_per_step`` (sum of output tokens divided by sum of
            participating request counts across steps), ``packing_efficiency`` and optional
            ``mean_wait_s`` (default 0). A restore-aware P model additionally
            requires measured ``prefill_features_per_request`` in its feature
            schema order. Do not use future replay outputs or max_new_tokens as
            observed output. Keep unfinished requests out of completed averages
            and account for their sampling bias in the caller's observation window.
        total_nodes: Fixed homogeneous model-node budget, at least two.
        constraints: ``concurrency`` in requests (not multiagent tasks), per-node
            ``kv_capacity_tokens`` and ``max_running_requests``; optional
            ``min_user_tps`` (100) and ``safety_fraction`` (0.95).
        engine_config: Current settings, checked against the calibrated settings.

    Returns:
        JSON report with passed/reason, selected P/D counts and scored candidates.
        Unusable data, incompatible models or infeasible constraints yield
        passed=False and selected=None. The scheduler retains its current plan.
        A selected decode batch is a capacity-model assumption, not a command to
        change the engine's admission cap. No timers, networking or file writes.
    """
    unavailable = {"passed": False, "selected": None, "candidates": []}
    core = _core()
    try:
        fitted, digest, config = _verified_model(model, engine_config)
        nodes = _integer(total_nodes, "total_nodes", minimum=2)
        concurrency = _integer(constraints["concurrency"], "concurrency")
        max_running = _integer(constraints["max_running_requests"], "max_running_requests")
        kv_capacity = _integer(constraints["kv_capacity_tokens"], "kv_capacity_tokens")
        requests = _copy(workload["requests"])
        if not requests:
            raise ValueError("No completed request samples")
        for request in requests:
            for name in ("uncached", "hit", "output"):
                _integer(request[name], name, minimum=1 if name == "output" else 0)
        context = _number(workload["mean_decode_context"], "mean_decode_context", positive=True)
        acceptance = _number(workload["accepted_tokens_per_step"], "accepted_tokens_per_step", positive=True)
        # Permit a verifier's bonus token, but reject accidentally passing the
        # entire batch's output count as the per-request acceptance length.
        if acceptance > config["draft_tokens"] + 1:
            raise ValueError("accepted_tokens_per_step exceeds draft budget plus one bonus token")
        packing = _number(workload["packing_efficiency"], "packing_efficiency", positive=True)
        wait = _number(workload.get("mean_wait_s", 0), "mean_wait_s")
        safety = _number(constraints.get("safety_fraction", .95), "safety_fraction", positive=True)
        tps = _number(constraints.get("min_user_tps", 100), "min_user_tps", positive=True)
        if packing > 1 or safety > 1:
            raise ValueError("packing_efficiency and safety_fraction must be <= 1")
        p_fit, d_fit = fitted["prefill_fit"], fitted["decode_fit"]
        if len(p_fit["coefficients"]) == 4:
            aggregate = workload["prefill_features_per_request"]
            if len(aggregate) != 4:
                raise ValueError("Expected four measured prefill_features_per_request")
            p_fit["aggregate_request_features"] = [
                _number(value, "prefill_features_per_request") for value in aggregate]
        batches = fitted["observed_decode_batches"]
        if len(d_fit["coefficients"]) == 4:
            context_min, context_max = fitted["decode_context_per_request_range"]
            if not context_min <= context <= context_max:
                raise ValueError("mean_decode_context is outside calibrated per-request context range")
            # Both the per-request mean and batch total must be supported. These
            # range checks do not prove support for every joint input distribution.
            lower, upper = d_fit["feature_min"][3], d_fit["feature_max"][3]
            batches = [b for b in batches if lower <= b * context <= upper]
        report = core.select_ratio(
            nodes=nodes, concurrency=concurrency, chunk=config["chunk_size"],
            prefill_fit=p_fit, decode_fit=d_fit, prompt_samples=requests,
            mean_decode_context=context, kv_capacity_tokens=kv_capacity,
            max_running_requests=max_running, accepted_tokens_per_step=acceptance,
            observed_decode_batches=batches, experts=config["experts"],
            top_k=config["top_k"], draft_tokens=config["draft_tokens"],
            mean_wait_s=wait, packing_efficiency=packing, safety_fraction=safety,
            min_user_tps=tps,
        )
        report.update(passed=True, reason="Highest modeled bottleneck capacity among feasible integer ratios",
                      model_sha256=digest, configuration_sha256=fitted["configuration_sha256"])
        report["limitations"].append("A recommendation does not apply roles or account for drain/migration delay")
        return _copy(report)
    except (ValueError, KeyError, TypeError) as error:
        return {**unavailable, "reason": str(error)}
