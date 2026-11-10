"""Unit scheduling: sequential or multiprocessing conversion over corpus units.

A unit is one picklable work item (a JSONL line locator or a source file
path). Each worker converts its unit's traces, writes finished rows to a
private part file, and returns statistics; the parent concatenates the parts
in corpus order so output ordering is deterministic regardless of worker
count. Rows are written by the worker that built them — giant traces never
travel through pickle.

Failure/memory behavior: workers run under a ProcessPoolExecutor (spawn
context), which raises BrokenProcessPool if a worker dies (e.g. OOM-killed)
instead of hanging the parent the way Pool.imap does. Unit byte sizes drive
a worker-count clamp: converting a unit peaks around 12x its source bytes of
RSS (measured on edgebench, whose worst line is 1.22 GB), so the parent
refuses to oversubscribe physical memory. Largest units are scheduled first
so stragglers surface early.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any

from strict_convert import adapters
from strict_convert.builder import build_trace, merge_stats, new_stats
from strict_convert.rendering import TargetRenderer

_WORKER: dict[str, Any] = {}

# Measured peak RSS multiple over source unit bytes (json.loads + canonical
# copies + token id lists + serialized output), rounded up for safety.
RSS_MULTIPLE = 12


def _init_worker(adapter_name: str, options: dict[str, Any]) -> None:
    adapter = adapters.get(adapter_name)
    renderer = TargetRenderer(options["target_tokenizer"], options.get("chat_encoding", "auto"),
                              options.get("thinking_mode", "thinking"),
                              options.get("history_reasoning", "native"),
                              options.get("template_kwargs"))
    _WORKER["adapter_name"] = adapter_name
    _WORKER["adapter"] = adapter
    _WORKER["renderer"] = renderer
    _WORKER["options"] = options
    _WORKER["context"] = adapter.make_context(options, renderer)


def _convert_unit(job: tuple[int, dict[str, Any]]) -> tuple[int, str, dict[str, Any]]:
    index, unit = job
    adapter = _WORKER["adapter"]
    renderer: TargetRenderer = _WORKER["renderer"]
    options = _WORKER["options"]
    stats = new_stats()
    stats.update({"traces_in": 0, "generates_in": 0, "unit_warnings": []})
    part_path = f"{options['output']}.part-{index:05d}"
    with open(part_path, "w") as sink:
        rows_direct = getattr(adapter, "ROWS", False)
        for trace in adapter.convert_unit(unit, _WORKER["context"], options, stats):
            row = trace if rows_direct else build_trace(trace, renderer, _WORKER["adapter_name"], stats)
            if row is not None:
                sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                sink.write("\n")
    return index, part_path, stats


def _heartbeat(
    done: int, total: int, unit: dict[str, Any], stats: dict[str, Any]
) -> None:
    print(
        f"strict_convert: unit {done}/{total} done: {unit.get('name', '?')} "
        f"(traces={stats.get('traces_out', 0)}, "
        f"generates={stats.get('generates_out', 0)})",
        file=sys.stderr,
        flush=True,
    )


def total_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None


def clamp_workers(
    units: list[dict[str, Any]], workers: int
) -> tuple[int, list[str]]:
    """Cap the worker count so unit peak RSS cannot exceed physical RAM."""
    warnings: list[str] = []
    if workers <= 1:
        return workers, warnings
    largest = max((unit.get("bytes") or 0) for unit in units)
    ram = total_ram_bytes()
    if not largest or ram is None:
        return workers, warnings
    budget = int(ram * 0.8)
    per_worker = largest * RSS_MULTIPLE
    if per_worker * workers <= budget:
        return workers, warnings
    allowed = max(1, budget // per_worker)
    warnings.append(
        f"clamped --workers {workers} -> {allowed}: largest unit is "
        f"{largest / 1e9:.2f} GB and conversion peaks near {RSS_MULTIPLE}x "
        f"unit bytes per worker ({per_worker / 1e9:.1f} GB), RAM budget "
        f"{budget / 1e9:.1f} GB"
    )
    if allowed == 1 and per_worker > budget:
        warnings.append(
            "even one worker may exceed the RAM budget on the largest unit; "
            "expect heavy swapping or an OOM kill (BrokenProcessPool)"
        )
    return int(allowed), warnings


def run_units(
    adapter_name: str,
    options: dict[str, Any],
    units: list[dict[str, Any]],
    workers: int,
) -> tuple[dict[str, Any], list[str]]:
    """Convert all units; returns (merged stats, warnings)."""
    total = new_stats()
    total.update({"traces_in": 0, "generates_in": 0, "unit_warnings": []})
    workers, warnings = clamp_workers(units, workers)
    for warning in warnings:
        print(f"strict_convert: {warning}", file=sys.stderr, flush=True)
    jobs = list(enumerate(units))
    results: list[tuple[int, str, dict[str, Any]]] = []
    if workers <= 1:
        _init_worker(adapter_name, options)
        for done, job in enumerate(jobs, 1):
            result = _convert_unit(job)
            _heartbeat(done, len(jobs), job[1], result[2])
            results.append(result)
    else:
        # Largest first: stragglers start early, tail latency shrinks.
        ordered = sorted(jobs, key=lambda job: -(job[1].get("bytes") or 0))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(adapter_name, options),
        ) as pool:
            pending = {pool.submit(_convert_unit, job): job for job in ordered}
            done_count = 0
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = pending.pop(future)
                    result = future.result()  # BrokenProcessPool if worker died
                    done_count += 1
                    _heartbeat(done_count, len(jobs), job[1], result[2])
                    results.append(result)
    results.sort(key=lambda item: item[0])

    with open(options["output"], "w") as sink:
        for _, part_path, stats in results:
            warnings.extend(stats.pop("unit_warnings", []))
            merge_stats(total, stats)
            with open(part_path) as part:
                for line in part:
                    sink.write(line)
            os.remove(part_path)
    total.pop("unit_warnings", None)
    return total, warnings


def unit_matches(unit: dict[str, Any], include: str) -> bool:
    haystack = " ".join(
        str(unit.get(key, "")) for key in ("name", "trace_id", "path")
    )
    return include in haystack


def select_units(
    units: list[dict[str, Any]],
    includes: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Keep corpus order; force-include matches, then fill up to the limit."""
    if not includes and limit is None:
        return units

    def matches(unit: dict[str, Any]) -> bool:
        return any(unit_matches(unit, include) for include in includes)

    included = [unit for unit in units if matches(unit)]
    if limit is None:
        return units
    fill = max(limit - len(included), 0)
    others = [unit for unit in units if not matches(unit)][:fill]
    chosen = {id(unit) for unit in included + others}
    return [unit for unit in units if id(unit) in chosen]
