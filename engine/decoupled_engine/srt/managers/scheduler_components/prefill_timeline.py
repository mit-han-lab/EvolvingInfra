"""Completed CPU boundary samples for a causal prefill warm-up calibration.

No CUDA events, tensor reads, cache mutation, or forecast. The collector fits
only samples whose start/end were observed before its warm-up cutoff.
"""

import math
import time
from array import array
from collections import deque


def scalar(value):
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
        else None
    )


class PrefillTimeline:
    def __init__(self, capacity=64):
        self.rows = deque(maxlen=max(1, min(128, capacity)))
        self.pending = {}
        self.previous = None
        self.sequence = 0
        self.unknown = 0

    def append(self, kind, start, end, features, iteration, rid=None):
        if (
            scalar(start) is None
            or scalar(end) is None
            or end < start
            or features is None
        ):
            self.unknown += 1
            return
        self.sequence += 1
        self.rows.append(
            dict(
                sequence=self.sequence,
                kind=kind,
                forward_iter=iteration,
                rid=rid,
                duration_s=end - start,
                completed_mono=end,
                features=dict(features),
            )
        )

    def on_launch(self, batch):
        if not batch.forward_mode.is_extend_without_speculative():
            self.previous = None
            return
        reqs, prefixes, extends = batch.reqs, batch.prefix_lens, batch.extend_lens
        from decoupled_engine.srt.managers.utils import is_health_check_generate_req

        if reqs and all(is_health_check_generate_req(req) for req in reqs):
            self.previous = None
            return
        load = getattr(batch, "_prefill_routing_load_features", None)
        if (
            not all(isinstance(v, (list, tuple)) for v in (reqs, prefixes, extends))
            or len(reqs) != len(prefixes)
            or len(reqs) != len(extends)
            or len(reqs) > 128
            or any(type(v) is not int or v < 0 for v in (*prefixes, *extends))
            or not isinstance(load, tuple)
            or len(load) != 5
            or load[1] is None
            or load[2] is None
        ):
            self.previous = None
            self.unknown += 1
            return
        inputs = [getattr(r, "origin_input_ids", None) for r in reqs]
        known_inputs = all(isinstance(v, (list, tuple, array)) for v in inputs)
        features = dict(
            new_tokens=sum(extends),
            attention_pairs=sum(
                n * h + n * (n + 1) // 2 for h, n in zip(prefixes, extends)
            ),
            batch_size=len(reqs),
            host_restore_tokens=load[1],
            storage_restore_tokens=load[2],
            max_input_tokens=(
                max((len(r.origin_input_ids) for r in reqs), default=0)
                if all(
                    isinstance(
                        getattr(r, "origin_input_ids", None), (list, tuple, array)
                    )
                    for r in reqs
                )
                else None
            ),
            # Requests whose last prefill chunk is in this pass: their results
            # are processed after this forward (KV send, tree insert).
            finishing_requests=(
                sum(1 for v, h, n in zip(inputs, prefixes, extends) if h + n >= len(v))
                if known_inputs
                else None
            ),
        )
        launch, iteration = batch.launch_ts, batch.forward_iter
        previous = None if batch.after_idle_gap else self.previous
        if previous is not None and previous[0] + 1 == iteration:
            # The interval belongs to the previous pass; the admission work of
            # this pass (host and storage restore) also falls inside it.
            interval = dict(
                previous[2],
                next_new_tokens=features["new_tokens"],
                next_batch_size=features["batch_size"],
                next_host_restore_tokens=load[1],
                next_storage_restore_tokens=load[2],
            )
            self.append("launch_interval", previous[1], launch, interval, previous[0])
        self.previous = (iteration, launch, features)
        self.pending[iteration] = (launch, features)
        if len(self.pending) > 8:
            self.pending.pop(min(self.pending))
            self.unknown += 1
        for req in reqs:
            if getattr(req, "extend_batch_idx", None) == 1:
                stamp = getattr(
                    getattr(req, "time_stats", None), "forward_entry_time", None
                )
                self.append(
                    "entry_to_launch",
                    stamp,
                    launch,
                    features,
                    iteration,
                    getattr(req, "rid", None),
                )

    def on_result(self, batch):
        previous = self.pending.pop(getattr(batch, "forward_iter", None), None)
        if previous is None:
            return
        launch, features = previous
        for req in batch.reqs:
            times = getattr(req, "time_stats", None)
            done = getattr(times, "prefill_finished_time", None)
            if scalar(done) is not None and done >= launch:
                self.append(
                    "last_launch_to_done",
                    launch,
                    done,
                    features,
                    batch.forward_iter,
                    getattr(req, "rid", None),
                )
                # A completed-label stream, separate from the step-cost
                # training labels. This exposes same-P waiting/service and
                # cache provenance so route->receive can be clock-bounded
                # later. It is never a future target's routing input.
                received = scalar(getattr(times, "scheduler_recv_time", None))
                entry = scalar(getattr(times, "forward_entry_time", None))
                tokens = getattr(req, "origin_input_ids", None)
                if (
                    received is not None
                    and entry is not None
                    and received <= entry <= done
                ):
                    self.append(
                        "request_complete",
                        received,
                        done,
                        dict(
                            input_tokens=(
                                len(tokens)
                                if isinstance(tokens, (list, tuple, array))
                                else None
                            ),
                            receive_to_entry_s=entry - received,
                            entry_to_completion_s=done - entry,
                            last_launch_to_completion_s=done - launch,
                            cached_tokens_device=scalar(
                                getattr(req, "cached_tokens_device", None)
                            ),
                            cached_tokens_host=scalar(
                                getattr(req, "cached_tokens_host", None)
                            ),
                            cached_tokens_storage=scalar(
                                getattr(req, "cached_tokens_storage", None)
                            ),
                            cache_breakdown_computed=getattr(
                                req, "_cache_breakdown_computed", False
                            ),
                        ),
                        batch.forward_iter,
                        getattr(req, "rid", None),
                    )

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        return dict(
            schema="prefill-cpu-timeline-samples.v1",
            last_sequence=self.sequence,
            unknown=self.unknown,
            samples=[
                dict((k, v) for k, v in row.items() if k != "completed_mono")
                | {"completed_age_s": max(0.0, now - row["completed_mono"])}
                for row in self.rows
            ],
        )
