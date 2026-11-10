"""Read SGLang scheduler trace data without importing the engine or token IDs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import expert_fraction, fit_steps, select_ratio


def json_lines(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            # A live writer may not have finished its final record yet.
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def load_steps(paths, start_unix, end_unix):
    rows = []
    for path in paths:
        previous = None
        for row in json_lines(path):
            if not start_unix <= row["ts"] <= end_unix:
                previous = None
                continue
            row = dict(row, source=str(path), busy_interval_s=None)
            if (
                previous is not None
                and not row["after_idle"]
                and row["forward_iter"] == previous["forward_iter"] + 1
                and row["mode"] == previous["mode"]
            ):
                interval = row["launch_mono"] - previous["launch_mono"]
                if interval > 0:
                    # Interval i -> i+1 belongs to step i, not step i+1.
                    previous["busy_interval_s"] = interval
            rows.append(row)
            previous = row
    return sorted(rows, key=lambda r: (r["ts"], r["source"], r["forward_iter"]))


def _features(row, role, experts, top_k):
    if role == "prefill":
        return [1.0, row["new_prefill_tokens"], row["prefill_attention_pairs"]]
    b = row["launch_bs"]
    q = row["spec_num_draft_tokens"] or 1
    return [1.0, b, float(expert_fraction(b * q, experts, top_k))]


def worker_coverage(paths, rows, role):
    """Require usable GPU-timed steps from every declared worker stream."""
    expected = [str(path) for path in paths]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError(f'Missing or duplicate {role} trace streams')
    modes = {'extend'} if role == 'prefill' else {'decode', 'target_verify'}
    counts = {path: 0 for path in expected}
    for row in rows:
        if (row.get('mode') in modes and row.get('launch_bs', 0)
                and row.get('forward_gpu_ms') is not None and row['forward_gpu_ms'] > 0
                and (role != 'prefill' or row.get('prefill_attention_pairs') is not None)):
            counts[row['source']] += 1
    missing = [path for path, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f'No usable GPU-timed {role} steps from: {missing}')
    return counts


def _capacity_windows(sampled, features, size=8):
    """Sum nonoverlapping, consecutive steps to average overlap launch jitter.

    Linearity preserves the original step coefficients: sum(X @ beta) equals
    sum(X) @ beta. Never bridge an idle gap, missing step, worker, or mode.
    """
    pending = {}
    windows = []
    for row, feature in zip(sampled, features):
        group = pending.setdefault(row["source"], [])
        if group and (
            row["forward_iter"] != group[-1][0]["forward_iter"] + 1
            or row["mode"] != group[-1][0]["mode"]
        ):
            group.clear()
        group.append((row, feature))
        if len(group) == size:
            windows.append(
                dict(
                    ts=row["ts"],
                    interval_start_ts=(group[0][0]["ts"] - group[0][0]["dur_ms"] / 1000
                                       if "dur_ms" in group[0][0] else None),
                    interval_end_ts=(row["ts"] - row["dur_ms"] / 1000 + row["busy_interval_s"]
                                     if "dur_ms" in row else None),
                    source=row["source"],
                    busy_interval_s=sum(r["busy_interval_s"] for r, _ in group),
                    features=np.sum([x for _, x in group], axis=0).tolist(),
                )
            )
            group.clear()
    return sorted(windows, key=lambda r: (r["ts"], r["source"]))


def _busy_episodes(sampled, features):
    """Measure complete P busy periods, including their final forward.

    An idle-marked launch opens a period and the next idle-marked launch proves
    it has closed. Its end is the last result-processing timestamp. Neither
    idle waiting nor a partial period at the capture boundary is included.
    Summing features preserves the original per-step coefficients.
    """
    pending = {}
    episodes = []
    for row, feature in zip(sampled, features):
        source = row["source"]
        group = pending.get(source)
        if group and (
            row["forward_iter"] != group[-1][0]["forward_iter"] + 1
            or row["mode"] != group[-1][0]["mode"]
        ):
            group = None  # A missing forward makes the entire period unusable.
        if row.get("after_idle"):
            if group:
                last = group[-1][0]
                end = last["launch_mono"] + last["dur_ms"] / 1000
                start = group[0][0]["launch_mono"]
                if start < end <= row["launch_mono"]:
                    episodes.append(
                        dict(
                            ts=last["ts"],
                            source=source,
                            busy_interval_s=end - start,
                            features=np.sum([x for _, x in group], axis=0).tolist(),
                            step_count=len(group),
                        )
                    )
            group = []
        if group is not None:
            group.append((row, feature))
        pending[source] = group
    return sorted(episodes, key=lambda r: (r["ts"], r["source"]))


def fit_role(rows, role, experts, top_k, *, context_term=False):
    modes = {"extend"} if role == "prefill" else {"decode", "target_verify"}
    usable = [
        r
        for r in rows
        if r["mode"] in modes
        and r.get("launch_bs")
        and (role != "prefill" or r.get("prefill_attention_pairs") is not None)
    ]
    report = {
        "observed_steps": len(usable),
        "steps_after_idle": sum(bool(r.get("after_idle")) for r in usable),
        "time_models": {},
    }
    clocks = [
        ("gpu", "forward_gpu_ms", 0.001),
        ("busy_wall", "busy_interval_s", 1.0),
        ("busy_capacity", "busy_interval_s", 1.0),
    ]
    if role == "prefill":
        clocks.append(("busy_episodes", "dur_ms", 1.0))
    for clock, field, factor in clocks:
        sampled = [r for r in usable if r.get(field) is not None and r[field] > 0]
        features = [_features(r, role, experts, top_k) for r in sampled]
        if context_term and role == "decode":
            for x, r in zip(features, sampled):
                x.append(r["launch_seq_lens_sum"])
        support = sampled
        coverage = 1.0
        if clock == "busy_capacity":
            sampled = _capacity_windows(sampled, features)
            coverage = len(sampled) * 8 / len(support) if support else 0
            features = [r["features"] for r in sampled]
        elif clock == "busy_episodes":
            sampled = _busy_episodes(sampled, features)
            coverage = (
                sum(r["step_count"] for r in sampled) / len(support) if support else 0
            )
            features = [r["features"] for r in sampled]
            field = "busy_interval_s"
        try:
            if coverage < 0.5:
                raise ValueError(
                    "Fewer than half of eligible steps form complete capacity samples"
                )
            fitted = fit_steps(features, [r[field] * factor for r in sampled])
            fitted["validation_unit"] = {
                "busy_capacity": "8-step total",
                "busy_episodes": "complete busy period, first launch to final result",
            }.get(clock, "step")
            fitted["step_window"] = (
                None
                if clock == "busy_episodes"
                else 8 if clock == "busy_capacity" else 1
            )
            fitted["step_coverage"] = coverage
            # Per-worker aggregate error catches one slow node hidden by pooling.
            split = fitted["train_samples"]
            coefficients = np.asarray(fitted["holdout_coefficients"])
            node_errors = {}
            for source in sorted({r["source"] for r in sampled[split:]}):
                indexes = [
                    i
                    for i in range(split, len(sampled))
                    if sampled[i]["source"] == source
                ]
                predicted = sum(
                    float(np.dot(features[i], coefficients)) for i in indexes
                )
                measured = sum(sampled[i][field] * factor for i in indexes)
                node_errors[source] = abs(predicted / measured - 1)
            fitted["worker_total_errors"] = node_errors
            fitted["passed"] &= all(e <= 0.15 for e in node_errors.values())
            # Observed support is retained, including q; no silent MTP change.
            fitted["batch_support"] = sorted({r["launch_bs"] for r in support})
            fitted["draft_token_support"] = sorted(
                {r["spec_num_draft_tokens"] or 1 for r in support}
            )
            report["time_models"][clock] = fitted
        except ValueError as error:
            report["time_models"][clock] = {
                "passed": False,
                "reason": str(error),
                "samples": len(sampled),
            }
    return report


def calibrate(
    *,
    prefill_paths,
    decode_paths,
    requests,
    start_unix,
    end_unix,
    configuration,
    kv_capacity_tokens,
    prefill_request_paths=None,
    telemetry_path=None,
):
    p = load_steps(prefill_paths, start_unix, end_unix)
    d = load_steps(decode_paths, start_unix, end_unix)
    k, e = configuration["top_k"], configuration["experts"]
    p_fit = fit_role(p, "prefill", e, k)
    d_fit = fit_role(d, "decode", e, k)
    context_fit = fit_role(d, "decode", e, k, context_term=True)
    # Use the paper's three terms unless a measured context term materially
    # improves chronological holdout error. Report both fits either way.
    plain = d_fit["time_models"]["busy_capacity"]
    with_context = context_fit["time_models"]["busy_capacity"]
    use_context = with_context.get("passed", False) and (
        not plain.get("passed", False)
        or with_context["errors"]["mean_relative"]
        < 0.8 * plain["errors"]["mean_relative"]
    )
    selected_d = with_context if use_context else plain
    real_p = [r for r in p if r["mode"] == "extend" and r["new_prefill_tokens"] > 0]
    real_d = [
        r
        for r in d
        if r["mode"] in ("decode", "target_verify") and r.get("launch_bs", 0)
    ]
    report = dict(
        elapsed_s=end_unix - start_unix,
        start_unix=start_unix,
        end_unix=end_unix,
        prefill=p_fit,
        decode=d_fit,
        decode_with_context=context_fit,
        decode_context_term_selected=use_context,
        capacity_validation_unit="P: complete busy periods; D: 8 consecutive busy steps. Raw GPU/launch-gap/8-step fits retained.",
        decision=None,
        decision_error=None,
    )
    try:
        if configuration.get('require_all_worker_traces'):
            report['worker_trace_coverage'] = dict(
                prefill=worker_coverage(prefill_paths, p, 'prefill'),
                decode=worker_coverage(decode_paths, d, 'decode'))
        method = configuration.get("prefill_capacity_model", "original")
        if method not in ("original", "cache_tier_totals", "measured_restore_totals"):
            raise ValueError("Unknown prefill capacity model")
        selected_p = p_fit["time_models"]["busy_episodes"]
        if method == "measured_restore_totals":
            from .restore import fit_measured_restore
            if not prefill_request_paths or not telemetry_path:
                raise ValueError("Measured restores require request traces and telemetry")
            selected_p = fit_measured_restore(p, prefill_request_paths, telemetry_path,
                configuration.get("restore_counter_replicas"), start_unix, end_unix,
                collection_clock_pad_s=configuration.get("restore_collection_clock_pad_s"))
            report["prefill_measured_restore"] = selected_p
            report["capacity_validation_unit"] = "P: 60s wall windows, exact busy overlap and bracketed completed restore counters; D: 8 busy steps"
            report["validation_scope"] = (
                "Posthoc aggregate capacity model; not independent model-selection validation. "
                "Original step-model failures remain valid. Uses completed H2D counters, not "
                "inconsistent request cache tiers. Formal runs must test frozen predictions."
            )
        if method == "cache_tier_totals":
            from .capacity import fit_cache_capacity
            if not prefill_request_paths:
                raise ValueError("Cache capacity model requires request traces")
            selected_p = fit_cache_capacity(p, prefill_request_paths, start_unix, end_unix)
            report["prefill_capacity_totals"] = selected_p
            report["capacity_validation_unit"] = "P: 60s bins of complete busy episodes with cache tiers; D: 8 busy steps"
            report["validation_scope"] = (
                "Aggregate service capacity only. Original GPU/episode failures remain valid. "
                "This model was developed after inspecting the Roadmap learning data; its "
                "chronological check is exploratory, not independent model-selection evidence. "
                "Frozen predictions require prospective checks in formal runs."
            )
        if (
            (method == "original" and not p_fit["time_models"]["gpu"]["passed"])
            or not (context_fit if use_context else d_fit)["time_models"]["gpu"][
                "passed"
            ]
        ):
            raise ValueError("Held-out GPU step-model validation failed")
        if not real_d or not real_p:
            raise ValueError("No measured prefill/decode steps")
        drafts = sorted({r["spec_num_draft_tokens"] or 1 for r in real_d})
        if len(drafts) != 1:
            raise ValueError("MTP draft budget changed within learning window")
        acceptance = sum(r["decode_tokens"] for r in real_d) / sum(
            r["launch_bs"] for r in real_d
        )
        mean_context = sum(r["launch_seq_lens_sum"] for r in real_d) / sum(
            r["launch_bs"] for r in real_d
        )
        packing = (
            np.mean([r["new_prefill_tokens"] for r in real_p]) / configuration["chunk"]
        )
        report["measurements"] = dict(
            accepted_tokens_per_step=acceptance,
            mean_decode_context=mean_context,
            packing_efficiency=float(packing),
            request_samples=len(requests),
        )
        report["decision"] = select_ratio(
            nodes=configuration["nodes"],
            concurrency=configuration["concurrency"],
            chunk=configuration["chunk"],
            prefill_fit=selected_p,
            decode_fit=selected_d,
            prompt_samples=requests,
            mean_decode_context=mean_context,
            kv_capacity_tokens=kv_capacity_tokens,
            max_running_requests=configuration["max_running_requests"],
            accepted_tokens_per_step=acceptance,
            observed_decode_batches=[r["launch_bs"] for r in real_d],
            experts=e,
            top_k=k,
            draft_tokens=drafts[0],
            mean_wait_s=configuration["mean_wait_s"],
            packing_efficiency=float(packing),
            min_user_tps=configuration["min_user_tps"],
        )
        if configuration.get("concurrency_unit") == "complete_multiagent_tasks":
            # A task contains parallel agents: task population is not request
            # population. This diagnostic does not participate in ratio selection.
            for candidate in report["decision"]["candidates"]:
                candidate["population_ceiling_estimate_rps"] = None
            report["decision"]["selected"]["population_ceiling_estimate_rps"] = None
            report["population_ceiling_note"] = (
                "Not estimated: concurrency counts complete multiagent tasks, "
                "not sequential request-producing sessions. Capacity-based "
                "P/D selection is unchanged; parallel tool waits are replayed."
            )
    except (ValueError, KeyError) as error:
        report["decision_error"] = str(error)
    return report
