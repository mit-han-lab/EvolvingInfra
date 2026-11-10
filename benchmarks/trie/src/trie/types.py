import hashlib
import json
import re
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from trie.blocks import run_blocks, validate_block_runs, validate_block_spec
from trie.dependencies import validate_dependencies
from trie.validation import validate_replay_task_row, validate_replay_trace_row

_MAX_TOKEN_ID = 2**32 - 1


def token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class Turn:
    assistant_response_length: int
    tool_call_output_length: int
    tool_call_latency: float

    def __post_init__(self) -> None:
        if self.assistant_response_length < 0:
            raise ValueError("assistant_response_length must be non-negative")
        if self.tool_call_output_length < 0:
            raise ValueError("tool_call_output_length must be non-negative")
        if self.tool_call_latency < 0:
            raise ValueError("tool_call_latency must be non-negative")


@dataclass
class Workload:
    input_prompt_length: int
    turns: list[Turn]
    final_assistant_response_length: int

    def cumulative_prompt_tokens(self, through_turn: int) -> int:
        return self.input_prompt_length + sum(
            self.turns[i].assistant_response_length
            + self.turns[i].tool_call_output_length
            for i in range(through_turn)
        )

    @classmethod
    def from_jsonl_row(cls, d: dict[str, Any]) -> "Workload":
        num_turns = d["num_turns"]
        resp = d["assistant_response_length"]
        out = d["tool_call_output_length"]
        lat = d["tool_call_latency"]
        for name, values in (
            ("assistant_response_length", resp),
            ("tool_call_output_length", out),
            ("tool_call_latency", lat),
        ):
            if not isinstance(values, list):
                raise TypeError(f"{name} must be a list of length {num_turns}")
            if len(values) != num_turns:
                raise ValueError(
                    f"{name} length {len(values)} != num_turns {num_turns}"
                )
        turns = [Turn(resp[i], out[i], lat[i]) for i in range(num_turns)]
        return cls(
            input_prompt_length=d["input_prompt_length"],
            turns=turns,
            final_assistant_response_length=d["final_assistant_response_length"],
        )


@dataclass
class AppendMessage:
    message: dict[str, Any]

    def __post_init__(self) -> None:
        role = self.message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("append_message.message.role must be a non-empty string")


@dataclass
class Wait:
    seconds: float
    actor: Literal["user", "tool", "other"] = "other"

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("wait seconds must be non-negative")
        if self.actor not in ("user", "tool", "other"):
            raise ValueError("wait actor must be user, tool, or other")


def _token_ids_invalid(token_ids: list[Any]) -> bool:
    """Per-id integer/bounds check via C primitives (see validation.py)."""
    try:
        packed = array("q", token_ids)
    except (TypeError, OverflowError):
        return True
    if any(type(token_id) is bool for token_id in token_ids):
        return True
    return min(packed) < 0 or max(packed) > _MAX_TOKEN_ID


@dataclass
class Generate:
    max_tokens: int
    recorded_assistant: dict[str, Any] | None = None
    recorded_prompt_token_ids: list[int] | None = None
    recorded_prompt_token_count: int | None = None
    recorded_prompt_token_ids_sha256: str | None = None
    recorded_output_token_ids: list[int] | None = None
    source_request_id: str | None = None
    source_model: str | None = None
    source_timestamp: str | None = None
    # Block-run form of the prompt / output (see trie.blocks); expanded at send time.
    recorded_prompt_block_runs: list[list[int]] | None = None
    recorded_output_block_runs: list[list[int]] | None = None
    source_t_s: float | None = None
    source_api_time_s: float | None = None
    # Uploaded data URIs or client-side files, ordered like the prompt's image
    # placeholders. Files are resolved against trace.metadata.image_root.
    recorded_image_data: list[str] | None = None

    def __post_init__(self) -> None:
        if self.recorded_image_data is not None:
            if (not isinstance(self.recorded_image_data, list)
                    or not self.recorded_image_data
                    or any(not isinstance(image, str) or not image
                           for image in self.recorded_image_data)):
                raise ValueError("recorded_image_data must be a non-empty list of image data URIs or files")
            if (self.recorded_prompt_token_ids is None
                    and self.recorded_prompt_block_runs is None):
                raise ValueError("recorded images require authoritative prompt token IDs")
        if self.recorded_prompt_block_runs is not None:
            validate_block_runs(
                self.recorded_prompt_block_runs, "recorded_prompt_block_runs"
            )
            if self.recorded_prompt_token_ids is not None:
                raise ValueError(
                    "recorded_prompt_block_runs and recorded_prompt_token_ids are "
                    "mutually exclusive"
                )
        if self.recorded_output_block_runs is not None:
            validate_block_runs(
                self.recorded_output_block_runs, "recorded_output_block_runs"
            )
            if self.recorded_output_token_ids is not None:
                raise ValueError(
                    "recorded_output_block_runs and recorded_output_token_ids are "
                    "mutually exclusive"
                )
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("generate max_tokens must be a positive integer")
        if self.recorded_assistant is not None:
            if not isinstance(self.recorded_assistant, dict):
                raise TypeError("recorded_assistant must be an object")
            role = self.recorded_assistant.get("role")
            if role != "assistant":
                raise ValueError("recorded_assistant role must be assistant")
        if self.recorded_prompt_token_ids is not None:
            if not isinstance(self.recorded_prompt_token_ids, list) or (
                not self.recorded_prompt_token_ids
                or _token_ids_invalid(self.recorded_prompt_token_ids)
            ):
                raise ValueError(
                    "recorded_prompt_token_ids must be a non-empty list of "
                    f"non-negative integers no greater than {_MAX_TOKEN_ID}"
                )
        prompt_contract_fields = (
            self.recorded_prompt_token_count,
            self.recorded_prompt_token_ids_sha256,
        )
        if any(value is not None for value in prompt_contract_fields):
            if (
                type(self.recorded_prompt_token_count) is not int
                or self.recorded_prompt_token_count <= 0
                or not isinstance(self.recorded_prompt_token_ids_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.recorded_prompt_token_ids_sha256)
                is None
            ):
                raise ValueError(
                    "recorded prompt token count and SHA256 must be a positive "
                    "integer and 64 lowercase hex characters"
                )
            if self.recorded_prompt_token_ids is not None and (
                len(self.recorded_prompt_token_ids) != self.recorded_prompt_token_count
                or token_ids_sha256(self.recorded_prompt_token_ids)
                != self.recorded_prompt_token_ids_sha256
            ):
                raise ValueError(
                    "recorded prompt token IDs differ from their count/SHA256 contract"
                )
        if self.recorded_output_token_ids is not None:
            if not isinstance(self.recorded_output_token_ids, list):
                raise ValueError(
                    "recorded_output_token_ids must be a list of non-negative integers"
                )
            if len(self.recorded_output_token_ids) != self.max_tokens:
                raise ValueError(
                    "recorded_output_token_ids length must equal max_tokens"
                )
            if _token_ids_invalid(self.recorded_output_token_ids):
                raise ValueError(
                    "recorded_output_token_ids must contain non-negative integers "
                    f"no greater than {_MAX_TOKEN_ID}"
                )
        if self.source_request_id is not None and not self.source_request_id:
            raise ValueError("source_request_id must be non-empty")


@dataclass
class ReplaceContext:
    messages: list[dict[str, Any]]
    reason: Literal["compaction", "reset", "branch", "other"] = "other"

    def __post_init__(self) -> None:
        for message in self.messages:
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("replace_context messages must have non-empty roles")


ReplayEvent: TypeAlias = AppendMessage | Wait | Generate | ReplaceContext


_GENERATE_FIELDS = (
    "recorded_assistant",
    "recorded_prompt_token_ids",
    "recorded_prompt_token_count",
    "recorded_prompt_token_ids_sha256",
    "recorded_output_token_ids",
    "source_request_id",
    "source_model",
    "source_timestamp",
    "recorded_prompt_block_runs",
    "recorded_output_block_runs",
    "source_t_s",
    "source_api_time_s",
    "recorded_image_data",
)


@dataclass
class ReplayTrace:
    trace_id: str
    events: list[ReplayEvent]
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset_s: float = 0.0
    # A task's metadata.block_tokens serves agents without their own; it is
    # serialized with the task, not with each agent.
    shared_block_tokens: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )
    schema_version: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")

        if self.start_offset_s < 0:
            raise ValueError("start_offset_s must be non-negative")
        if self.metadata.get("control_only") is True:
            if not self.events or any(not isinstance(event, Wait) for event in self.events):
                raise ValueError("control-only traces must contain only wait events")
        elif not any(isinstance(event, Generate) for event in self.events):
            raise ValueError("replay trace must contain at least one generate event")
        block_events = [
            event
            for event in self.events
            if isinstance(event, Generate)
            and (
                event.recorded_prompt_block_runs is not None
                or event.recorded_output_block_runs is not None
            )
        ]
        if block_events:
            spec = self.block_spec
            validate_block_spec(spec)
            block_size = spec["block_size"]
            for event in block_events:
                runs = event.recorded_output_block_runs
                if runs is not None and run_blocks(runs) * block_size < event.max_tokens:
                    raise ValueError(
                        "recorded_output_block_runs hold fewer tokens than max_tokens: "
                        f"trace={self.trace_id!r}"
                    )
        for message in self.initial_messages:
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("initial messages must have non-empty roles")

    @property
    def block_spec(self) -> dict[str, Any] | None:
        """This trace's metadata.block_tokens, else the enclosing task's."""
        return self.metadata.get("block_tokens", self.shared_block_tokens)

    @classmethod
    def from_jsonl_row(
        cls,
        d: dict[str, Any],
        *,
        shared_block_tokens: dict[str, Any] | None = None,
    ) -> "ReplayTrace":
        if type(d.get("schema_version")) is not int or d["schema_version"] != 2:
            raise ValueError("replay trace schema_version must be 2")
        validate_replay_trace_row(d)

        events: list[ReplayEvent] = []
        for index, raw_event in enumerate(d.get("events", [])):
            event_type = raw_event.get("type")
            if event_type == "append_message":
                events.append(AppendMessage(message=raw_event["message"]))
            elif event_type == "wait":
                events.append(
                    Wait(
                        seconds=raw_event["seconds"],
                        actor=raw_event.get("actor", "other"),
                    )
                )
            elif event_type == "generate":
                events.append(
                    Generate(
                        max_tokens=raw_event["max_tokens"],
                        **{name: raw_event.get(name) for name in _GENERATE_FIELDS},
                    )
                )
            elif event_type == "replace_context":
                events.append(
                    ReplaceContext(
                        messages=raw_event["messages"],
                        reason=raw_event.get("reason", "other"),
                    )
                )
            else:
                raise ValueError(
                    f"unknown replay event type at index {index}: {event_type!r}"
                )

        return cls(
            trace_id=d["trace_id"],
            initial_messages=d.get("initial_messages", []),
            events=events,
            tools=d.get("tools"),
            metadata=d.get("metadata", {}),
            start_offset_s=d.get("start_offset_s", 0.0),
            shared_block_tokens=shared_block_tokens,
        )

    def to_jsonl_row(self) -> dict[str, Any]:
        raw_events: list[dict[str, Any]] = []
        for event in self.events:
            if isinstance(event, AppendMessage):
                raw_events.append({"type": "append_message", "message": event.message})
            elif isinstance(event, Wait):
                raw_events.append(
                    {"type": "wait", "seconds": event.seconds, "actor": event.actor}
                )
            elif isinstance(event, Generate):
                raw_event: dict[str, Any] = {
                    "type": "generate",
                    "max_tokens": event.max_tokens,
                }
                for name in _GENERATE_FIELDS:
                    value = getattr(event, name)
                    if value is not None:
                        raw_event[name] = value
                raw_events.append(raw_event)
            else:
                raw_events.append(
                    {
                        "type": "replace_context",
                        "messages": event.messages,
                        "reason": event.reason,
                    }
                )
        return {
            "schema_version": 2,
            "trace_id": self.trace_id,
            "start_offset_s": self.start_offset_s,
            "initial_messages": self.initial_messages,
            "tools": self.tools,
            "events": raw_events,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EventRef:
    trace_id: str
    event_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise ValueError("dependency trace_id must be a non-empty string")
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("dependency event_index must be a non-negative integer")

    @property
    def key(self) -> tuple[str, int]:
        return self.trace_id, self.event_index


@dataclass(frozen=True)
class Dependency:
    before: EventRef
    after: EventRef
    when: Literal["complete", "start"] = "complete"

    def __post_init__(self) -> None:
        if self.when not in ("complete", "start"):
            raise ValueError("dependency when must be complete or start")


@dataclass
class ReplayTask:
    """One admission slot containing concurrent agents and completion edges."""

    trace_id: str
    traces: list[ReplayTrace]
    dependencies: list[Dependency]
    start_offset_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # "task": every agent of one task invocation shares a cache salt, so the
    # agents' real shared prefixes can hit; "agent" isolates each agent.
    cache_salt_scope: Literal["agent", "task"] = "agent"
    schema_version: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id or not self.traces:
            raise ValueError("multi-agent task requires a trace_id and agent traces")
        if self.cache_salt_scope not in ("agent", "task"):
            raise ValueError("cache_salt_scope must be agent or task")
        if self.start_offset_s < 0:
            raise ValueError("start_offset_s must be non-negative")
        if not any(isinstance(event, Generate) for trace in self.traces for event in trace.events):
            raise ValueError("multi-agent task must contain a generate event")
        validate_dependencies(self)

    @classmethod
    def from_jsonl_row(cls, row: dict[str, Any]) -> "ReplayTask":
        validate_replay_task_row(row)
        shared = (row.get("metadata") or {}).get("block_tokens")
        return cls(
            trace_id=row["trace_id"],
            traces=[
                ReplayTrace.from_jsonl_row(trace, shared_block_tokens=shared)
                for trace in row["traces"]
            ],
            dependencies=[
                Dependency(EventRef(**edge["before"]), EventRef(**edge["after"]),
                           when=edge.get("when", "complete"))
                for edge in row["dependencies"]
            ],
            start_offset_s=row.get("start_offset_s", 0.0),
            metadata=row.get("metadata", {}),
            cache_salt_scope=row.get("cache_salt_scope", "agent"),
        )

    def to_jsonl_row(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.cache_salt_scope != "agent":
            extra["cache_salt_scope"] = self.cache_salt_scope
        return {
            "schema_version": 3,
            "trace_id": self.trace_id,
            "traces": [trace.to_jsonl_row() for trace in self.traces],
            "dependencies": [
                {
                    **{name: {"trace_id": ref.trace_id, "event_index": ref.event_index}
                       for name, ref in (("before", edge.before), ("after", edge.after))},
                    **({"when": edge.when} if edge.when != "complete" else {}),
                }
                for edge in self.dependencies
            ],
            "start_offset_s": self.start_offset_s,
            "metadata": self.metadata,
            **extra,
        }


TraceWorkload: TypeAlias = Workload | ReplayTrace | ReplayTask


def agent_traces(workload: ReplayTrace | ReplayTask) -> list[ReplayTrace]:
    return workload.traces if isinstance(workload, ReplayTask) else [workload]


def iter_generates(workload: ReplayTrace | ReplayTask) -> Iterator[Generate]:
    for trace in agent_traces(workload):
        for event in trace.events:
            if isinstance(event, Generate):
                yield event
