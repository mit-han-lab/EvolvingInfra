"""Adapter: shape-spec rows -> tier-A synthetic strict-replay traces.

Accepts, as-is, the row format emitted by trie/scripts/generate_workload.py
(``input_prompt_length`` / ``assistant_response_length[]`` / ``num_turns`` /
``tool_call_latency[]`` / ``tool_call_output_length[]`` /
``final_assistant_response_length``; validated through the vendored
``trie.types.Workload.from_jsonl_row``) and manufactures fully
token-authoritative traces with exact token counts, so controlled shapes can
be strict-replayed with forced output ids and DSpark acceptance controls.

Optional per-row extensions (all documented in the README):

* ``trace_id`` — base trace id (default ``synth-<row index>``); ``--repeat``
  replicates a row with ``-r<k>`` suffixes and distinct content.
* ``reasoning_length`` — per-turn think-token budget carved out of each
  turn's response (list of ``num_turns`` values, ``num_turns + 1`` to cover
  the final response, or a scalar broadcast to all). Default 0 = all
  content. The converter measures the target template rather than assuming a
  model-specific think-block layout.
* ``shared_prefix_length`` — first N prompt tokens identical across every
  trace built from the row (system-prompt sharing knob; 0 = traces diverge
  right after the template's constant header).
* ``start_offset_s`` — trace start offset (default 0).

Synthesis discipline:

* All text comes from a word pool derived from the target tokenizer at
  startup: common English words verified to encode to exactly one token
  with a leading space and to be boundary-stable in concatenation (the same
  trick the trie client uses because full-vocab random sampling breaks
  prefix hash chains).
* Token counts are exact, calibrated by measurement: trailing pool words
  add exactly one token each, so the calibration loop converges in a couple
  of iterations; infeasible specs (target below fixed template overhead)
  fail the row with a reason in the report — never silently clamped.
* Content sampling is seeded from blake2b(seed_salt | trace_id | turn |
  role); no wall clock, no global random state; reruns are byte-identical.
* Canonical turns feed the shared core builder, so chaining, invariants,
  in-band tokenizer identity, reports and validate.py work unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from strict_convert._vendor import types_module
from strict_convert.adapters import _trie_common
from strict_convert.builder import _common_prefix_length
from strict_convert.canonical import CanonicalTrace, CanonicalTurn, assistant_message

_CANDIDATE_WORDS = (
    "time year people way day man thing woman life child world school state "
    "family student group country problem hand part place case week company "
    "system program question work government number night point home water "
    "room mother area money story fact month lot right study book eye job "
    "word business issue side kind head house service friend father power "
    "hour game line end member law car city community name team minute idea "
    "body information back parent face level office door health person art "
    "war history party result change morning reason research girl guy moment "
    "air teacher force education foot boy age policy process music market "
    "sense nation plan college interest death experience effect use class "
    "control care field development role effort rate heart drug show leader "
    "light voice wife whole police mind price report decision son view town "
    "road arm difference value building action model season society tax "
    "director position player record paper space ground form event official "
    "matter center couple site project activity star table need court oil "
    "situation cost industry figure street image phone data picture practice "
    "piece land product doctor wall patient worker news test movie north "
    "love support technology south house plant garden river mountain stone "
    "bridge window glass paper letter answer summer winter spring autumn "
    "morning evening dinner coffee bread butter cheese apple orange lemon "
    "chicken garden flower forest island ocean beach storm cloud thunder "
    "silver golden purple yellow orange simple little large small quick slow "
    "early late happy quiet loud clean dirty empty full heavy light strong "
    "weak young old new fresh sweet bitter warm cool cold hot deep high low "
    "long short wide narrow thick thin soft hard smooth rough bright dark "
    "clear plain fancy cheap rich poor safe wild calm brave smart wise fair "
    "true false real fake open closed near far left right north south east "
    "west first second third last next front back top bottom inside outside "
    "walk run jump climb swim fly drive ride read write draw paint sing "
    "dance play watch listen speak talk hear touch smell taste think know "
    "learn teach study remember forget understand believe hope wish want "
    "need like love hate enjoy prefer choose decide agree refuse accept "
    "offer give take bring carry hold drop throw catch push pull lift move "
    "stay leave arrive return travel visit meet greet help serve build make "
    "create design repair break destroy grow plant harvest cook bake boil "
    "fry mix stir pour fill empty wash clean dry fold cut slice chop peel"
).split()

_TOOL_NAME = "lookup"
_MIN_POOL = 128
_MAX_CALIBRATION_ROUNDS = 12


class ShapeError(Exception):
    """A spec value is infeasible for the target template."""


def list_units(options: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    return _trie_common.list_jsonl_units(options["inputs"])


def prepare_options(options: dict[str, Any], units: list[dict[str, Any]]) -> None:
    del options, units


class _Context:
    """Word pool, template probes and calibration helpers (per worker)."""

    def __init__(self, options: dict[str, Any], renderer: Any) -> None:
        self.renderer = renderer
        self.salt = options.get("seed_salt") or "strict-synth-v1"
        self.repeat = max(int(options.get("repeat") or 1), 1)
        self.pool = self._build_pool()
        self.pool_sha256 = hashlib.sha256(
            json.dumps(self.pool, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": "Synthetic retrieval tool for strict-replay "
                    "shape workloads.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        self.template_head_tokens = self._probe_template_head()
        self._shared_words_cache: dict[int, int] = {}

    def _build_pool(self) -> list[str]:
        renderer = self.renderer
        accepted = []
        seen = set()
        for word in _CANDIDATE_WORDS:
            if word in seen:
                continue
            seen.add(word)
            if len(renderer.encode(" " + word)) == 1:
                accepted.append(word)
        # Boundary stability across concatenation: a trailing " word" must
        # contribute exactly its own single token, chunk-verified.
        stable = []
        for start in range(0, len(accepted), 64):
            chunk = accepted[start : start + 64]
            text = " " + " ".join(chunk)
            ids = renderer.encode(text)
            if len(ids) == len(chunk) and renderer.decode(ids) == text:
                stable.extend(chunk)
            else:
                for word in chunk:
                    ids = renderer.encode(" " + word + " " + word)
                    if len(ids) == 2:
                        stable.append(word)
        if len(stable) < _MIN_POOL:
            raise SystemExit(
                f"single-token word pool too small for this tokenizer: "
                f"{len(stable)} < {_MIN_POOL}"
            )
        return stable

    def _probe_template_head(self) -> int:
        """Prompt-token count shared by traces before any system content."""
        first = self._prompt_ids("alpha probe", "user probe")
        second = self._prompt_ids("beta probe", "user probe")
        return _common_prefix_ids(first, second)

    def _prompt_ids(self, system_content: str, user_content: str) -> list[int]:
        rendered = self.renderer.render(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            tools=self.tools,
            add_generation_prompt=True,
        )
        return self.renderer.encode(rendered)

    def words(self, seed: str, count: int) -> str:
        """Deterministic, index-stable word sequence (blake2b per index)."""
        picks = []
        for index in range(count):
            digest = hashlib.blake2b(
                f"{self.salt}|{seed}|{index}".encode("utf-8"), digest_size=4
            ).digest()
            picks.append(self.pool[int.from_bytes(digest, "big") % len(self.pool)])
        return " ".join(picks)

    def ordinal_words(self, ordinal: int) -> str:
        """Two little-endian base-N digit words: traces diverge on the first
        token after any shared prefix (exact for batches up to len(pool);
        up to one extra shared token for colliding pairs beyond that)."""
        base = len(self.pool)
        return f"{self.pool[ordinal % base]} {self.pool[(ordinal // base) % base]}"

    def shared_word_count(self, target_tokens: int) -> int:
        """Words needed so the cross-trace shared prompt prefix == target."""
        if target_tokens in self._shared_words_cache:
            return self._shared_words_cache[target_tokens]
        if target_tokens <= self.template_head_tokens:
            raise ShapeError(
                f"shared_prefix_length={target_tokens} is not above the "
                f"template's constant head ({self.template_head_tokens} tokens)"
            )
        count = target_tokens - self.template_head_tokens
        for _ in range(_MAX_CALIBRATION_ROUNDS):
            shared = self.words("shared-prefix", count)
            first = self._prompt_ids(shared + " " + self.ordinal_words(0), "u")
            second = self._prompt_ids(shared + " " + self.ordinal_words(1), "u")
            measured = _common_prefix_ids(first, second)
            if measured == target_tokens:
                self._shared_words_cache[target_tokens] = count
                return count
            count += target_tokens - measured
            if count < 1:
                raise ShapeError(
                    f"shared_prefix_length={target_tokens} below the reachable "
                    "minimum for this template"
                )
        raise ShapeError(
            f"shared_prefix_length={target_tokens} did not calibrate"
        )


def make_context(options: dict[str, Any], renderer: Any) -> _Context:
    return _Context(options, renderer)


def _common_prefix_ids(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    shared = 0
    while shared < limit and a[shared] == b[shared]:
        shared += 1
    return shared


def _reasoning_per_generate(row: dict[str, Any], num_turns: int) -> list[int]:
    raw = row.get("reasoning_length", 0)
    generates = num_turns + 1
    if isinstance(raw, int):
        return [raw] * generates
    if isinstance(raw, list):
        if len(raw) == generates:
            return [int(v) for v in raw]
        if len(raw) == num_turns:
            return [int(v) for v in raw] + [0]
    raise ShapeError(
        "reasoning_length must be a scalar or a list of num_turns or "
        f"num_turns+1 values; got {raw!r}"
    )


def _measure_output_tokens(
    context: _Context,
    messages: list[dict[str, Any]],
    assistant: dict[str, Any],
    prompt_text: str,
) -> int:
    """Token count the core builder will assign to this generate."""
    renderer = context.renderer
    completed = renderer.render(
        messages + [assistant], tools=context.tools, add_generation_prompt=False
    )
    if completed.startswith(prompt_text):
        return len(renderer.encode(completed[len(prompt_text) :]))
    common = _common_prefix_length(prompt_text, completed)
    structural_limit = len(prompt_text) - renderer.forced_suffix_len
    if common > structural_limit:
        common = structural_limit
    return len(renderer.encode(completed[common:]))


def _calibrated_assistant(
    context: _Context,
    messages: list[dict[str, Any]],
    prompt_text: str,
    *,
    trace_id: str,
    turn_index: int,
    target_tokens: int,
    reasoning_tokens: int,
    tool_call: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assistant message whose builder-derived output length == target."""

    def build(n_reason: int, n_content: int) -> dict[str, Any]:
        reasoning = (
            context.words(f"{trace_id}|{turn_index}|reasoning", n_reason)
            if n_reason > 0
            else None
        )
        content = (
            context.words(f"{trace_id}|{turn_index}|content", n_content)
            if n_content > 0
            else ""
        )
        return assistant_message(
            reasoning, content, [tool_call] if tool_call else None
        )

    n_reason = reasoning_tokens
    n_content = max(target_tokens - reasoning_tokens, 0)
    adjust_content = True
    if n_content == 0 and n_reason > 0:
        adjust_content = False  # all budget is think: trim reasoning words
    for _ in range(_MAX_CALIBRATION_ROUNDS):
        assistant = build(n_reason, max(n_content, 0) if adjust_content else 0)
        measured = _measure_output_tokens(context, messages, assistant, prompt_text)
        diff = target_tokens - measured
        if diff == 0:
            return assistant
        if adjust_content:
            n_content += diff
            if n_content < 0:
                if n_reason > 0:
                    adjust_content = False
                    n_reason += n_content
                    n_content = 0
                    if n_reason < 1:
                        break
                else:
                    break
        else:
            n_reason += diff
            if n_reason < 1:
                break
    overhead = _measure_output_tokens(
        context, messages, build(1 if n_reason > 0 else 0, 0), prompt_text
    )
    raise ShapeError(
        f"turn {turn_index}: assistant_response_length={target_tokens} "
        f"(reasoning_length={reasoning_tokens}) is infeasible; fixed template "
        f"overhead for this turn shape is ~{overhead} tokens"
    )


def _calibrated_user_content(
    context: _Context,
    system_content: str,
    trace_id: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    """User content sized so the full rendered prompt encodes to target."""
    count = max(target_tokens // 2, 1)
    probe = context._prompt_ids(system_content, context.words(f"{trace_id}|user", 1))
    minimum = len(probe)
    if target_tokens < minimum:
        raise ShapeError(
            f"input_prompt_length={target_tokens} below fixed prompt overhead "
            f"({minimum} tokens: template + system + tools + one content word)"
        )
    for _ in range(_MAX_CALIBRATION_ROUNDS):
        user_content = context.words(f"{trace_id}|user", count)
        ids = context._prompt_ids(system_content, user_content)
        diff = target_tokens - len(ids)
        if diff == 0:
            return user_content, ids
        count += diff
        if count < 1:
            raise ShapeError(
                f"input_prompt_length={target_tokens} below fixed prompt "
                f"overhead ({minimum} tokens)"
            )
    raise ShapeError(f"input_prompt_length={target_tokens} did not calibrate")


def _build_trace(
    context: _Context,
    row: dict[str, Any],
    trace_id: str,
    ordinal: int,
) -> CanonicalTrace:
    num_turns = row["num_turns"]
    reasoning = _reasoning_per_generate(row, num_turns)
    shared_prefix = int(row.get("shared_prefix_length") or 0)

    if shared_prefix > 0:
        shared_count = context.shared_word_count(shared_prefix)
        system_content = (
            context.words("shared-prefix", shared_count)
            + " "
            + context.ordinal_words(ordinal)
            + " "
            + context.words(f"{trace_id}|system", 6)
        )
    else:
        system_content = (
            context.ordinal_words(ordinal)
            + " "
            + context.words(f"{trace_id}|system", 6)
        )

    user_content, prompt_ids = _calibrated_user_content(
        context, system_content, trace_id, row["input_prompt_length"]
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    turns: list[CanonicalTurn] = []
    pending_ops: list[tuple] = []
    pending_waits: list[tuple[float, str]] = []
    renderer = context.renderer
    for index in range(num_turns + 1):
        final = index == num_turns
        prompt_text = renderer.render(
            messages, tools=context.tools, add_generation_prompt=True
        )
        target = (
            row["final_assistant_response_length"]
            if final
            else row["assistant_response_length"][index]
        )
        tool_call = None
        if not final:
            tool_call = {
                "id": f"call_{index + 1}",
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "arguments": json.dumps(
                        {"query": context.words(f"{trace_id}|{index}|query", 3)},
                        ensure_ascii=False,
                    ),
                },
            }
        assistant = _calibrated_assistant(
            context,
            messages,
            prompt_text,
            trace_id=trace_id,
            turn_index=index,
            target_tokens=target,
            reasoning_tokens=reasoning[index],
            tool_call=tool_call,
        )
        original_counts: dict[str, Any] = {
            "spec_output_tokens": target,
            "spec_reasoning_tokens": reasoning[index],
        }
        if index == 0:
            original_counts["spec_prompt_tokens"] = row["input_prompt_length"]
            original_counts["measured_prompt_tokens"] = len(prompt_ids)
        turns.append(
            CanonicalTurn(
                assistant=assistant,
                context_ops=pending_ops,
                waits_before=pending_waits,
                original_counts=original_counts,
            )
        )
        messages = messages + [assistant]
        pending_ops, pending_waits = [], []
        if not final:
            tool_words = int(row["tool_call_output_length"][index])
            tool_message = {
                "role": "tool",
                "tool_call_id": f"call_{index + 1}",
                "content": _sized_tool_content(context, trace_id, index, tool_words),
            }
            pending_ops = [("append", [tool_message])]
            pending_waits = [(float(row["tool_call_latency"][index]), "tool")]
            messages = messages + [tool_message]

    metadata = {
        "source": "synthetic-shape-spec",
        "routing_key": trace_id,
        "strict_conversion_extras": {
            "source": "synthetic-shape-spec",
            "shape_spec": {
                key: value for key, value in row.items() if key != "trace_id"
            },
            "seed_salt": context.salt,
            "word_pool_sha256": context.pool_sha256,
            "template_head_tokens": context.template_head_tokens,
        },
    }
    return CanonicalTrace(
        trace_id=trace_id,
        turns=turns,
        initial_messages=[messages[0], messages[1]],
        tools=context.tools,
        start_offset_s=float(row.get("start_offset_s") or 0.0),
        metadata=metadata,
        source_model="synthetic",
    )


def _sized_tool_content(
    context: _Context, trace_id: str, turn_index: int, target_tokens: int
) -> str:
    """Tool payload whose standalone encode is exactly target_tokens."""
    if target_tokens <= 0:
        return ""
    count = target_tokens
    for _ in range(_MAX_CALIBRATION_ROUNDS):
        content = context.words(f"{trace_id}|{turn_index}|tool", count)
        measured = len(context.renderer.encode(content))
        diff = target_tokens - measured
        if diff == 0:
            return content
        count += diff
        if count < 1:
            raise ShapeError(
                f"turn {turn_index}: tool_call_output_length={target_tokens} "
                "not reachable with the word pool"
            )
    raise ShapeError(
        f"turn {turn_index}: tool_call_output_length={target_tokens} did not "
        "calibrate"
    )


def convert_unit(
    unit: dict[str, Any],
    context: _Context,
    options: dict[str, Any],
    stats: dict[str, Any],
):
    del options
    row = _trie_common.load_row(unit)
    row_index = int(unit["name"].rsplit(":", 1)[-1])
    try:
        types_module().Workload.from_jsonl_row(row)  # vendored shape validation
    except (KeyError, TypeError, ValueError) as exc:
        stats["traces_in"] += context.repeat
        stats["dropped_traces"]["invalid_shape_spec"] = (
            stats["dropped_traces"].get("invalid_shape_spec", 0) + context.repeat
        )
        stats["unit_warnings"].append(f"{unit['name']}: invalid shape spec: {exc}")
        return
    base_id = row.get("trace_id") or f"synth-{row_index:04d}"
    for replica in range(context.repeat):
        trace_id = base_id if context.repeat == 1 else f"{base_id}-r{replica}"
        stats["traces_in"] += 1
        stats["generates_in"] += row["num_turns"] + 1
        try:
            yield _build_trace(context, row, trace_id, replica)
        except ShapeError as exc:
            stats["dropped_traces"]["shape_infeasible"] = (
                stats["dropped_traces"].get("shape_infeasible", 0) + 1
            )
            stats["unit_warnings"].append(f"{trace_id}: {exc}")
            # A shape failure is deterministic; skip remaining replicas but
            # keep the input accounting exact.
            remaining = context.repeat - replica - 1
            if remaining:
                stats["traces_in"] += remaining
                stats["generates_in"] += (row["num_turns"] + 1) * remaining
                stats["dropped_traces"]["shape_infeasible"] += remaining
            return
