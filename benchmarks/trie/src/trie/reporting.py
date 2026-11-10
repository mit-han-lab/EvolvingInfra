from typing import Any

import numpy as np
import structlog
from rich.console import Console
from rich.table import Table

from trie.results import BenchmarkResult

logger = structlog.get_logger(__name__)
console = Console()

PER_TRACE_ROW_NAMES = ("mean", "min", "p50", "p90", "p95", "p99", "max")


def _format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    )


def _safe_rate(total: int, elapsed: float) -> float:
    return total / elapsed if elapsed > 0 else 0.0


def _overall_progress_rates(
    result: BenchmarkResult, elapsed: float
) -> tuple[tuple[str, float], ...]:
    return (
        ("total_prompt_t/s", _safe_rate(result.completed_prompt_tokens, elapsed)),
        ("cached_prompt_t/s", _safe_rate(result.cached_prompt_tokens, elapsed)),
        ("uncached_prompt_t/s", _safe_rate(result.new_prompt_tokens, elapsed)),
        ("compl_t/s", _safe_rate(result.completed_completion_tokens, elapsed)),
    )


def _last_30s_progress_rates(result: BenchmarkResult) -> tuple[tuple[str, float], ...]:
    last_30s = result.last_30s_rates()
    return (
        ("total_prompt_t/s", last_30s.prompt_tok_s),
        ("cached_prompt_t/s", last_30s.cached_tok_s),
        ("uncached_prompt_t/s", last_30s.new_prompt_tok_s),
        ("compl_t/s", last_30s.completion_tok_s),
    )


def _format_progress_rate_line(
    prefix: str, rates: tuple[tuple[str, float], ...]
) -> str:
    return prefix + "  ".join(f"{label}={value:.0f}" for label, value in rates)


def build_progress_lines(
    result: BenchmarkResult, elapsed: float, duration: float
) -> str:
    trace_rate = _safe_rate(result.completed_requests, elapsed)
    return "\n".join(
        [
            (
                "Running benchmark  "
                f"elapsed={_format_elapsed(elapsed)}/{_format_elapsed(duration)}  "
                f"completed={result.completed_requests}  "
                f"model_requests={result.completed_model_requests}  "
                f"trace/s={trace_rate:.2f}"
            ),
            _format_progress_rate_line(
                "overall:  ", _overall_progress_rates(result, elapsed)
            ),
            _format_progress_rate_line("last30:   ", _last_30s_progress_rates(result)),
        ]
    )


def _percentile_stats(values: list[float]) -> dict[str, float]:
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(np.max(values)),
    }


def build_json_summary(
    result: BenchmarkResult, num_gpus: int | None = None
) -> dict[str, Any]:
    """Build a stable, machine-readable benchmark result.

    Keep the raw samples alongside aggregates so experiment analysis can be
    changed without rerunning an expensive replay.
    """
    wall_time = result.wall_time
    last_30s = result.last_30s_rates()
    steady = result.steady_state_rates()

    def rates(rates: Any) -> dict[str, float]:
        return {
            "prompt_tok_s": rates.prompt_tok_s,
            "cached_prompt_tok_s": rates.cached_tok_s,
            "new_prompt_tok_s": rates.new_prompt_tok_s,
            "completion_tok_s": rates.completion_tok_s,
        }

    overall = {
        "prompt_tok_s": _safe_rate(result.completed_prompt_tokens, wall_time),
        "cached_prompt_tok_s": _safe_rate(result.cached_prompt_tokens, wall_time),
        "new_prompt_tok_s": _safe_rate(result.new_prompt_tokens, wall_time),
        "completion_tok_s": _safe_rate(result.completed_completion_tokens, wall_time),
    }
    distributions = {
        "latency_s": result.latencies,
        "ttft_s": result.ttfts,
        "ttfat_s": result.ttfats,
        "decode_tpot_ms": [1000.0 / value for value in result.tps_values if value > 0],
        "inter_token_latency_ms": result.inter_token_latencies_ms,
        "cache_hit_rate": [m.cache_hit_rate for m in result.server_metrics],
        "eligible_cache_hit_rate": [
            m.eligible_cache_hit_rate for m in result.server_metrics
        ],
        "client_prefix_tokens": result.client_prefix_tokens,
        "block_aligned_prefix_tokens": result.block_aligned_prefix_tokens,
        "server_cached_tokens": result.server_cached_tokens,
    }
    return {
        "schema_version": 1,
        "wall_time_s": wall_time,
        "expected_requests": result.expected_requests,
        "completed_requests": result.completed_requests,
        "completed_model_requests": result.completed_model_requests,
        "failed_requests": result.failed_requests,
        "migrated_requests": result.migrated_requests,
        "completed_prompt_tokens": result.completed_prompt_tokens,
        "cached_prompt_tokens": result.cached_prompt_tokens,
        "new_prompt_tokens": result.new_prompt_tokens,
        "completed_completion_tokens": result.completed_completion_tokens,
        "trace_per_s": _safe_rate(result.completed_requests, wall_time),
        "inflight_requests": result.inflight_requests,
        "throughput": {
            "overall": overall,
            "last_30s": rates(last_30s),
            "steady_state": rates(steady),
            "steady_state_per_gpu": (
                {key: value / num_gpus for key, value in rates(steady).items()}
                if num_gpus is not None
                else None
            ),
        },
        "distribution_summaries": {
            key: _percentile_stats(values)
            for key, values in distributions.items()
            if values
        },
        "samples": distributions,
    }


def _build_distribution_table(
    columns: dict[str, tuple[dict[str, float], str]],
) -> Table:
    table = Table(title="Per-trace metrics")
    table.add_column("Metric")
    for label in columns:
        table.add_column(label, justify="right")
    for row_name in PER_TRACE_ROW_NAMES:
        table.add_row(
            row_name, *(f"{stats[row_name]:{fmt}}" for stats, fmt in columns.values())
        )
    return table


def _build_workload_metrics_table(
    rows: list[tuple[str, float, float, float, float | None]],
    completed_requests: int,
    total_requests: int,
    trace_s: float,
    num_gpus: int | None,
) -> Table:
    include_gpu_column = num_gpus is not None
    table = Table(
        title=f"Workload metrics\ncompleted={completed_requests}/{total_requests}  trace/s={trace_s:.2f}"
    )
    for header in ("Metric", "Overall", "Last 30s Window", "Steady State"):
        table.add_column(header, justify="right" if header != "Metric" else "left")
    if include_gpu_column:
        table.add_column("Steady State / GPU", justify="right")
    for label, overall, last_30s, steady_state, steady_state_per_gpu in rows:
        row = [label, f"{overall:.2f}", f"{last_30s:.2f}", f"{steady_state:.2f}"]
        if include_gpu_column:
            row.append(f"{steady_state_per_gpu:.2f}")
        table.add_row(*row)
    return table


def log_summary(result: BenchmarkResult, num_gpus: int | None = None) -> None:
    wall_time = result.wall_time
    total_requests = result.expected_requests or (
        result.completed_requests + result.failed_requests
    )
    trace_s = result.completed_requests / wall_time if wall_time > 0 else 0.0

    rows: list[tuple[str, float, float, float, float | None]] = []
    workload_metrics: dict[str, float] = {"trace_per_s": trace_s}
    if result.completed_completion_tokens > 0 and wall_time > 0:
        last_30s = result.last_30s_rates()
        steady = result.steady_state_rates()
        for label, log_key, total, last_30s_rate, steady_rate in (
            (
                "total prompt tok/s",
                "prompt_tok_per_s",
                result.completed_prompt_tokens,
                last_30s.prompt_tok_s,
                steady.prompt_tok_s,
            ),
            (
                "cached prompt tok/s",
                "cached_prompt_tok_per_s",
                result.cached_prompt_tokens,
                last_30s.cached_tok_s,
                steady.cached_tok_s,
            ),
            (
                "uncached prompt tok/s",
                "new_prompt_tok_per_s",
                result.new_prompt_tokens,
                last_30s.new_prompt_tok_s,
                steady.new_prompt_tok_s,
            ),
            (
                "completion tok/s",
                "completion_tok_per_s",
                result.completed_completion_tokens,
                last_30s.completion_tok_s,
                steady.completion_tok_s,
            ),
        ):
            overall = total / wall_time
            rows.append(
                (
                    label,
                    overall,
                    last_30s_rate,
                    steady_rate,
                    steady_rate / num_gpus if num_gpus is not None else None,
                )
            )
            workload_metrics[log_key] = overall
            workload_metrics[f"steady_state_{log_key}"] = steady_rate
            workload_metrics[f"last_30s_window_{log_key}"] = last_30s_rate

    columns: dict[str, tuple[dict[str, float], str]] = {}
    dist_log_metrics: dict[str, float] = {}

    for label, prefix, fmt, values in (
        ("Latency (s)", "latency_s", ".3f", result.latencies),
        ("TTFT (s)", "ttft_s", ".3f", result.ttfts),
        ("TTFAT (s)", "ttfat_s", ".3f", result.ttfats),
        (
            "Decode TPOT (ms/tok)",
            "mean_turn_decode_tpot_ms",
            ".2f",
            [1000.0 / value for value in result.tps_values if value > 0],
        ),
        (
            "ITL (ms)",
            "inter_token_latency_ms",
            ".2f",
            result.inter_token_latencies_ms,
        ),
        (
            "Cache hit rate (%)",
            "cache_hit_rate_pct",
            ".1f",
            [m.cache_hit_rate * 100 for m in result.server_metrics],
        ),
        (
            "Eligible cache hit rate (%)",
            "eligible_cache_hit_rate_pct",
            ".1f",
            [m.eligible_cache_hit_rate * 100 for m in result.server_metrics],
        ),
        (
            "Client prefix overlap (tok)",
            "client_prefix_tokens",
            ".0f",
            result.client_prefix_tokens,
        ),
        (
            "Block-aligned prefix (tok)",
            "block_aligned_prefix_tokens",
            ".0f",
            result.block_aligned_prefix_tokens,
        ),
        (
            "Server cached prompt (tok)",
            "server_cached_tokens",
            ".0f",
            result.server_cached_tokens,
        ),
    ):
        if not values:
            continue
        stats = _percentile_stats(values)
        columns[label] = (stats, fmt)
        dist_log_metrics.update(
            {f"{prefix}_{name}": value for name, value in stats.items()}
        )

    logger.info(
        "benchmark complete",
        wall_time_s=wall_time,
        completed_requests=result.completed_requests,
        expected_requests=result.expected_requests,
        completed_model_requests=result.completed_model_requests,
        failed_requests=result.failed_requests,
        **workload_metrics,
        **dist_log_metrics,
    )

    tables = []
    if columns:
        tables.append(_build_distribution_table(columns))
    if rows:
        tables.append(
            _build_workload_metrics_table(
                rows, result.completed_requests, total_requests, trace_s, num_gpus
            )
        )
    for table in tables:
        console.print()
        console.print(table)
