import json
import os
from array import array
from functools import cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator, validators


_MAX_TOKEN_ID = 2**32 - 1
_TOKEN_ARRAY_KEYS = ("recorded_prompt_token_ids", "recorded_output_token_ids")


def _fast_valid_token_array(value: object) -> bool:
    """C-primitive equivalent of the schema's per-id walk for big id arrays.

    Matches items {type: integer (bool excluded), minimum 0, maximum 2^32-1}
    over a non-empty list. Anything this cannot vouch for is left to the
    jsonschema pass unmodified.
    """
    if type(value) is not list or not value:
        return False
    try:
        packed = array("q", value)
    except (TypeError, OverflowError):
        return False
    if any(type(item) is bool for item in value):
        return False
    return min(packed) >= 0 and max(packed) <= _MAX_TOKEN_ID


def _is_integer(checker: object, value: object) -> bool:
    del checker
    return type(value) is int


_StrictDraft202012Validator = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        "integer",
        _is_integer,
    ),
)


@cache
def _replay_trace_validator(version: int = 2) -> Draft202012Validator:
    schema_path = files("trie").joinpath(
        "schema",
        f"replay-{'trace' if version == 2 else 'task'}-v{version}.schema.json",
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if version == 3:
        # ReplayTask parses each agent through the v2 validator separately.
        # Do not traverse large token arrays twice; preserve the existing v2
        # attested-corpus bypass without ever skipping the task/graph checks.
        schema["properties"]["traces"]["items"] = {"type": "object"}
    _StrictDraft202012Validator.check_schema(schema)
    return _StrictDraft202012Validator(schema)


def validate_replay_trace_row(row: dict[str, Any]) -> None:
    # dpacp-cc1 annotation: rows are sha-attested by the corpus receipt before
    # load, and jsonschema Draft2020-12 validation of ~85MB rows costs hours
    # per load at replay scale. Skip is env-gated and default-off so pinned
    # behavior is preserved everywhere else.
    if os.environ.get("TRIE_SKIP_ROW_VALIDATION") == "1":
        return
    # Pre-validate the (potentially multi-million element) token id arrays
    # with C primitives and swap in a placeholder for the jsonschema walk —
    # the per-id python visit costs ~7 us/id and dominated load time on
    # tier-A workloads. Arrays that fail the fast check are left in place so
    # jsonschema reports them with its usual message.
    swapped: list[tuple[dict[str, Any], str, list[int]]] = []
    events = row.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or event.get("type") != "generate":
                continue
            for key in _TOKEN_ARRAY_KEYS:
                value = event.get(key)
                if _fast_valid_token_array(value):
                    swapped.append((event, key, value))
                    event[key] = [0]
    try:
        _validate_row(row, 2)
    finally:
        for event, key, value in swapped:
            event[key] = value


def validate_replay_task_row(row: dict[str, Any]) -> None:
    # Dependencies and their enclosing task are always validated before dispatch.
    _validate_row(row, 3)


def _validate_row(row: dict[str, Any], version: int) -> None:
    errors = sorted(
        _replay_trace_validator(version).iter_errors(row),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return

    error = errors[0]
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )
    raise ValueError(
        f"replay trace schema validation failed at {path}: {error.message}"
    )
