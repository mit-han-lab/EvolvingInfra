import json
import logging
import os
from pathlib import Path
from typing import Literal

import chz
import structlog

from trie.client import Client
from trie.client_lifecycle import ClientLifecycleWriter
from trie.reporting import build_json_summary


@chz.chz
class RunArgs:
    workload_path: str = chz.field(doc="Path to the JSONL workload file.")
    workload_schedule: str | None = chz.field(
        default=None, doc="Optional JSON time/slot schedule with validated workload source paths.",
    )
    endpoint: str = chz.field(doc="OpenAI-compatible endpoint base URL.")
    model: str = chz.field(doc="Model name to send to the completion API.")
    concurrency: int | None = chz.field(
        default=None,
        doc="Maximum active workloads: one v2 agent or one complete v3 multi-agent task.",
    )
    arrival_rate: float | None = chz.field(
        default=None, doc="Workload starts per second."
    )
    request_concurrency: int | None = chz.field(
        default=None,
        doc=(
            "Maximum concurrent model requests across replay traces. With "
            "replay_once and no trace admission policy, all traces share one "
            "benchmark-relative timeline."
        ),
    )
    tokenizer_model: str | None = chz.field(
        default=None,
        doc=(
            "Optional tokenizer override for synthetic or text-fallback workloads; "
            "defaults to model. Token-authoritative strict replay does not load it."
        ),
    )
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = (
        chz.field(
            default=None,
            doc="Optional reasoning effort forwarded to the local chat template.",
        )
    )
    duration: float = chz.field(
        default=300.0,
        doc="Duration in seconds to run the benchmark.",
    )
    duration_update_interval: float | None = chz.field(
        default=None,
        doc="Optional interval in seconds for logging remaining benchmark duration.",
    )
    seed: int | None = chz.field(
        default=None,
        doc="Optional random seed for reproducible workload resolution and prompt generation.",
    )
    api_key: str | None = chz.field(
        default=None,
        doc="API key for the endpoint. Falls back to OPENAI_API_KEY env var if not set.",
    )
    max_retries: int = chz.field(
        default=2,
        doc="Number of retries for the endpoint, openai default is 2.",
    )
    strict_replay_connect_retries: int = chz.field(
        default=0,
        doc=(
            "Retries for strict replay connection-establishment failures only; "
            "read/write/protocol failures are never retried."
        ),
    )
    strict_replay_connect_retry_delay_s: float = chz.field(
        default=1.0,
        doc="Delay in seconds between strict replay connection retries.",
    )
    migration_limit: int = chz.field(
        default=0,
        doc=(
            "Times one request may be re-sent after the server aborts it or its "
            "stream fails, as Dynamo's frontend request migration does; 0 disables."
        ),
    )
    timeout: float = chz.field(
        default=600.0,
        doc="Timeout for the endpoint, openai default is 600 seconds.",
    )
    num_gpus: int | None = chz.field(
        default=None,
        doc="Optional GPU count used for the steady-state per-GPU throughput column.",
    )
    stream: bool = chz.field(
        default=False,
        doc="Use streaming completions and report TTFT, TTFAT, and TPS per trace.",
    )
    delay_scale: float = chz.field(
        default=1.0,
        doc=(
            "Scale global trace start offsets and recorded user/tool waits; "
            "0 removes delays and 1 replays real time."
        ),
    )
    max_user_delay: float | None = chz.field(
        default=None,
        doc="Optional cap in seconds applied to each recorded user wait.",
    )
    max_tool_delay: float | None = chz.field(
        default=None,
        doc="Optional cap in seconds applied to each recorded tool wait.",
    )
    context_source: Literal["generated", "recorded"] = chz.field(
        default="generated",
        doc="Use live model output or recorded assistant output in the next prompt.",
    )
    prefix_block_size: int = chz.field(
        default=16,
        doc="Token block size used for client-side prefix-cache overlap metrics.",
    )
    cache_salt_mode: Literal["global", "session"] = chz.field(
        default="session",
        doc="Share vLLM prefix cache globally or isolate it by replay session.",
    )
    trajectory_prefix_tokens: int = chz.field(
        default=0,
        doc=(
            "Prepend this many random ordinary tokens per trajectory instance; "
            "use 10 for isolated Roadmap/EdgeBench replay. Shared by all turns/agents. "
            "Requires strict_replay and loads the model tokenizer; 0 preserves the input."
        ),
    )
    workload_order: Literal["natural", "shuffle", "prefix"] = chz.field(
        default="natural",
        doc="Trace ordering; prefix is a cache-locality upper-bound experiment.",
    )
    prefix_sort_tokens: int = chz.field(
        default=256,
        doc="Initial prompt tokens used when workload_order=prefix.",
    )
    replay_once: bool = chz.field(
        default=False,
        doc="Admit each trace once instead of cycling the workload until duration.",
    )
    drain_timeout: float = chz.field(
        default=0.0,
        doc="Seconds to drain in-flight traces after admission stops; 0 cancels immediately.",
    )
    strict_replay: bool = chz.field(
        default=False,
        doc=(
            "Replay schema-v2 agents or schema-v3 multi-agent tasks with recorded context and greedy sampling. "
            "This preserves the real P/D and MTP execution path."
        ),
    )
    replay_output_mode: Literal["forced", "model"] = chz.field(
        default="forced",
        doc="Force recorded output IDs, or generate real greedy output of the recorded length.",
    )
    kv_session_context: bool = chz.field(
        default=False,
        doc="Send x-kv-session-context per agent request and POST /kv_sessions/end when an agent finishes.",
    )
    replay_context_mode: Literal["recorded", "actual_output"] = chz.field(
        default="recorded",
        doc="Keep recorded assistant tokens in later prompts, or substitute the tokens actually generated (requires replay_output_mode=model).",
    )
    strict_replay_openai_fast_path: bool = chz.field(
        default=False,
        doc=(
            "Skip the OpenAI generated parameter transform for streamed, "
            "token-authoritative strict replay. Requires openai==2.30.0 and "
            "preserves the SDK HTTP, retry, status, and SSE paths."
        ),
    )
    warm_start: Literal["off", "stationary"] = chz.field(
        default="off",
        doc=(
            "stationary starts the first admission of every slot at a uniform cut "
            "through the workload's recorded span (needs source_t_s on generates)."
        ),
    )
    lazy_workload: bool = chz.field(
        default=False,
        doc="Parse each workload row when it is admitted, not at startup.",
    )
    shard_index: int = chz.field(
        default=0,
        doc="This process's shard when several client processes split one load (0-based).",
    )
    shard_count: int = chz.field(
        default=1,
        doc="Number of client processes splitting the load; each replays every shard_count-th workload.",
    )
    output_json: str | None = chz.field(
        default=None,
        doc="Optional path for a machine-readable benchmark result.",
    )
    lifecycle_output_jsonl: str | None = chz.field(
        default=None,
        doc="Optional path for privacy-safe per-request lifecycle JSONL.",
    )
    lifecycle_run_id: str | None = chz.field(
        default=None,
        doc="Experiment run ID recorded in lifecycle events.",
    )
    lifecycle_cell_id: str | None = chz.field(
        default=None,
        doc="Experiment cell ID recorded in lifecycle events.",
    )
    lifecycle_arm: Literal["off", "on"] | None = chz.field(
        default=None,
        doc="Optimistic-prefill arm recorded in lifecycle events.",
    )
    lifecycle_concurrency: int | None = chz.field(
        default=None,
        doc="Agent-session concurrency recorded in lifecycle events.",
    )

    @chz.validate
    def _validate_fields(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")
        if (
            self.duration_update_interval is not None
            and self.duration_update_interval <= 0
        ):
            raise ValueError("duration_update_interval must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if self.strict_replay and self.max_retries != 0:
            raise ValueError("strict_replay requires max_retries=0")
        if self.migration_limit < 0:
            raise ValueError("migration_limit must be greater than or equal to 0")
        if self.strict_replay_openai_fast_path and not self.strict_replay:
            raise ValueError(
                "strict_replay_openai_fast_path requires strict_replay=true"
            )
        if self.strict_replay_openai_fast_path and not self.stream:
            raise ValueError(
                "strict_replay_openai_fast_path requires stream=true"
            )
        if self.strict_replay_connect_retries < 0:
            raise ValueError(
                "strict_replay_connect_retries must be greater than or equal to 0"
            )
        if self.strict_replay_connect_retries and not self.strict_replay:
            raise ValueError(
                "strict_replay_connect_retries requires strict_replay=true"
            )
        if self.strict_replay_connect_retry_delay_s < 0:
            raise ValueError(
                "strict_replay_connect_retry_delay_s must be nonnegative"
            )
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.num_gpus is not None and self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        if self.delay_scale < 0:
            raise ValueError("delay_scale must be non-negative")
        if self.max_user_delay is not None and self.max_user_delay < 0:
            raise ValueError("max_user_delay must be non-negative")
        if self.max_tool_delay is not None and self.max_tool_delay < 0:
            raise ValueError("max_tool_delay must be non-negative")
        if self.prefix_block_size <= 0:
            raise ValueError("prefix_block_size must be greater than 0")
        if self.prefix_sort_tokens <= 0:
            raise ValueError("prefix_sort_tokens must be greater than 0")
        if self.drain_timeout < 0:
            raise ValueError("drain_timeout must be non-negative")
        lifecycle_context = (
            self.lifecycle_run_id,
            self.lifecycle_cell_id,
            self.lifecycle_arm,
            self.lifecycle_concurrency,
        )
        if self.lifecycle_output_jsonl is None:
            if any(value is not None for value in lifecycle_context):
                raise ValueError(
                    "lifecycle context fields require lifecycle_output_jsonl"
                )
        elif any(value is None for value in lifecycle_context):
            raise ValueError(
                "lifecycle_output_jsonl requires lifecycle_run_id, "
                "lifecycle_cell_id, lifecycle_arm, and lifecycle_concurrency"
            )
        if self.lifecycle_concurrency is not None:
            if self.lifecycle_concurrency < 1:
                raise ValueError("lifecycle_concurrency must be at least 1")
            if self.lifecycle_concurrency != self.concurrency:
                raise ValueError(
                    "lifecycle_concurrency must equal benchmark concurrency"
                )


def main() -> None:
    args: RunArgs = chz.entrypoint(RunArgs)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    lifecycle_writer = None
    if args.lifecycle_output_jsonl is not None:
        assert args.lifecycle_run_id is not None
        assert args.lifecycle_cell_id is not None
        assert args.lifecycle_arm is not None
        assert args.lifecycle_concurrency is not None
        lifecycle_writer = ClientLifecycleWriter(
            output_path=Path(args.lifecycle_output_jsonl),
            run_id=args.lifecycle_run_id,
            cell_id=args.lifecycle_cell_id,
            arm=args.lifecycle_arm,
            concurrency=args.lifecycle_concurrency,
        )

    if args.kv_session_context:
        from trie import kv_session

        kv_session.install(Client)
    result = Client(
        endpoint=args.endpoint,
        model=args.model,
        tokenizer_model=args.tokenizer_model,
        api_key=args.api_key,
        seed=args.seed,
        max_retries=args.max_retries,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
        strict_replay_connect_retries=args.strict_replay_connect_retries,
        strict_replay_connect_retry_delay_s=(
            args.strict_replay_connect_retry_delay_s
        ),
        strict_replay_openai_fast_path=args.strict_replay_openai_fast_path,
        replay_output_mode=args.replay_output_mode,
        replay_context_mode=args.replay_context_mode,
        lifecycle_writer=lifecycle_writer,
        migration_limit=args.migration_limit,
    ).sync_run(
        args.workload_path,
        concurrency=args.concurrency,
        arrival_rate=args.arrival_rate,
        request_concurrency=args.request_concurrency,
        duration=args.duration,
        duration_update_interval=args.duration_update_interval,
        num_gpus=args.num_gpus,
        stream=args.stream,
        delay_scale=args.delay_scale,
        max_user_delay=args.max_user_delay,
        max_tool_delay=args.max_tool_delay,
        context_source=args.context_source,
        prefix_block_size=args.prefix_block_size,
        cache_salt_mode=args.cache_salt_mode,
        trajectory_prefix_tokens=args.trajectory_prefix_tokens,
        workload_order=args.workload_order,
        prefix_sort_tokens=args.prefix_sort_tokens,
        replay_once=args.replay_once,
        drain_timeout=args.drain_timeout,
        strict_replay=args.strict_replay,
        warm_start=args.warm_start,
        lazy_workload=args.lazy_workload,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        workload_schedule=args.workload_schedule,
    )
    if args.kv_session_context:
        kv_session.save_stats()
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        temporary_path.write_text(
            json.dumps(
                build_json_summary(result, num_gpus=args.num_gpus),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)


if __name__ == "__main__":
    main()
