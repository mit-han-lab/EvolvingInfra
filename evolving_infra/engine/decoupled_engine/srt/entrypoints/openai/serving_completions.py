# Adapted from SGLang python/sglang/srt/entrypoints/openai/serving_completions.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging
import re
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from decoupled_engine.srt.disaggregation.late_d import (
    LATE_D_OUTCOME_ATTR,
    LATE_D_STATUS_ROUTE_ATTR,
    LateDLegEnded,
    late_d_leg_headers,
)
from decoupled_engine.srt.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    SglExt,
)
from decoupled_engine.srt.entrypoints.openai.serving_base import OpenAIServingBase
from decoupled_engine.srt.entrypoints.openai.usage_processor import UsageProcessor
from decoupled_engine.srt.entrypoints.openai.utils import (
    cached_tokens_details_from_dict,
    process_cached_tokens_details_from_ret,
    process_hidden_states_for_response,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    process_spec_tokens_details_from_ret,
    should_include_usage,
    spec_tokens_details_from_meta_info,
    to_openai_style_logprobs,
)
from decoupled_engine.srt.managers.io_struct import GenerateReqInput
from decoupled_engine.srt.parser.code_completion_parser import (
    generate_completion_prompt_from_request,
)
from decoupled_engine.srt.utils.weight_versions import build_endpoint_weight_version_metadata
from decoupled_engine.utils import convert_json_schema_to_str

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.tokenizer_manager import TokenizerManager
    from decoupled_engine.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)

class OpenAIServingCompletion(OpenAIServingBase):
    """Handler for /v1/completion requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager

    def _request_id_prefix(self) -> str:
        return "cmpl-"

    def _validate_request(self, request: CompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        prompt = request.prompt
        packed = request.packed_prompt_ids
        if packed is not None:
            # A packed prompt (late D) passed its checks in
            # CompletionRequest; prompt holds a placeholder.
            if self.template_manager.completion_template_name is not None:
                return "prompt_ids_b64 cannot be used with a completion template"
        elif not prompt or (isinstance(prompt, list) and all(not p for p in prompt)):
            return "Prompt cannot be empty"

        prompt_is_token_authoritative = (
            packed is None
            and isinstance(request.prompt, list)
            and bool(request.prompt)
            and all(type(token) is int for token in request.prompt)
        )
        forced_ids_are_authoritative = (
            isinstance(request.forced_output_token_ids, list)
            and bool(request.forced_output_token_ids)
            and all(type(token) is int for token in request.forced_output_token_ids)
            and request.max_tokens == len(request.forced_output_token_ids)
        )
        if request.strict_replay_deadline_audit and (
            request.stream is not True
            or not prompt_is_token_authoritative
            or not forced_ids_are_authoritative
            or request.return_token_ids is not True
            or request.ignore_eos is not True
            or request.temperature != 0.0
            or request.top_p != 1.0
            or request.top_k != 1
            or request.n != 1
            or request.best_of not in (None, 1)
            or request.stop is not None
            or request.stop_token_ids is not None
            or self.template_manager.completion_template_name is not None
            or not isinstance(request.rid, str)
            or re.fullmatch(r"rid_[0-9a-f]{32}", request.rid) is None
            or request.disconnect_abort_delay_seconds != 0.0
        ):
            return (
                "strict_replay_deadline_audit requires one streamed deterministic "
                "token-authoritative strict-replay request, a rid_<32 lowercase "
                "hex> request ID, and zero disconnect abort delay"
            )
        if (
            request.disconnect_abort_delay_seconds is not None
            and not request.strict_replay_deadline_audit
        ):
            return (
                "disconnect_abort_delay_seconds is only valid with "
                "strict_replay_deadline_audit"
            )

        return None

    def _convert_to_internal_request(
        self,
        request: CompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, CompletionRequest]:
        """Convert OpenAI completion request to internal format"""
        if request.strict_replay_deadline_audit:
            audit_error = self._validate_request(request)
            if audit_error is not None:
                raise ValueError(audit_error)
        # NOTE: with openai API, the prompt's logprobs are always not computed
        if request.echo and request.logprobs:
            logger.warning(
                "Echo is not compatible with logprobs. "
                "To compute logprobs of input prompt, please use the native /generate API."
            )
        # Process prompt
        prompt = request.prompt
        if self.template_manager.completion_template_name is not None:
            prompt = generate_completion_prompt_from_request(request)

        # Set logprob start length based on echo and logprobs
        if request.echo and request.logprobs:
            logprob_start_len = 0
        else:
            logprob_start_len = -1

        # Build sampling parameters
        sampling_params = self._build_sampling_params(request)

        # Determine prompt format
        if request.packed_prompt_ids is not None:
            prompt_kwargs = {"input_ids": request.packed_prompt_ids}
        elif isinstance(prompt, str) or (
            isinstance(prompt, list) and isinstance(prompt[0], str)
        ):
            prompt_kwargs = {"text": prompt}
        else:
            prompt_kwargs = {"input_ids": prompt}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=request.image_data,
            prepared_image_layout=request.prepared_image_layout,
            prepared_image_source=request.prepared_image_source,
            sampling_params=sampling_params,
            return_logprob=request.logprobs is not None,
            top_logprobs_num=request.logprobs if request.logprobs is not None else 0,
            logprob_start_len=logprob_start_len,
            return_text_in_logprobs=True,
            stream=request.stream,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            kv_pull_hint=request.kv_pull_hint,
            kv_export=request.kv_export,
            kv_import=request.kv_import,
            late_d=request.late_d,
            kv_session=request.kv_session,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            routed_experts_start_len=request.routed_experts_start_len,
            # return_prompt_token_ids=false drops only the prompt echo: the
            # tokenizer manager then keeps no prompt copy, and no response
            # carries prompt_token_ids. The generated token_ids stay.
            return_prompt_token_ids=(
                request.return_token_ids
                and request.return_prompt_token_ids is not False
            ),
            rid=request.rid,
            strict_replay_deadline_audit=request.strict_replay_deadline_audit,
            disconnect_abort_delay_seconds=request.disconnect_abort_delay_seconds,
            session_id=request.session_id,
            extra_key=request.extra_key,
            cache_salt=request.cache_salt,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
        )
        # This process-local marker is intentionally not a GenerateReqInput
        # wire field. It proves the benchmark-only disconnect behavior passed
        # the strict OpenAI completion contract above; native /generate and
        # other entrypoints cannot opt into it by supplying JSON fields.
        adapted_request._strict_replay_deadline_audit_validated = (
            request.strict_replay_deadline_audit
        )
        # Late D (choose): this route answers a leg that ends before its first
        # chunk with the leg's HTTP status (LateDLegEnded).
        setattr(adapted_request, LATE_D_STATUS_ROUTE_ATTR, True)

        return adapted_request, request

    def _build_sampling_params(self, request: CompletionRequest) -> Dict[str, Any]:
        """Build sampling parameters for the request"""
        # Start with common parameters
        sampling_params = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "min_new_tokens": request.min_tokens,
            "stop": request.stop,
            "stop_token_ids": request.stop_token_ids,
            "stop_regex": request.stop_regex,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repetition_penalty": request.repetition_penalty,
            "regex": request.regex,
            "json_schema": request.json_schema,
            "ebnf": request.ebnf,
            "n": request.n,
            "no_stop_trim": request.no_stop_trim,
            "ignore_eos": request.ignore_eos,
            "skip_special_tokens": request.skip_special_tokens,
            "logit_bias": request.logit_bias,
            "custom_params": request.custom_params,
            "sampling_seed": request.seed,
            "forced_output_token_ids": request.forced_output_token_ids,
            "stream_interval": request.stream_interval,
        }

        # Handle response_format constraints
        if request.response_format and request.response_format.type == "json_schema":
            json_schema = request.response_format.json_schema
            schema = getattr(json_schema, "schema_", None)
            if schema is None:
                raise ValueError(
                    "schema_ is required for json_schema response format request."
                )
            sampling_params["json_schema"] = convert_json_schema_to_str(schema)
        elif request.response_format and request.response_format.type == "json_object":
            sampling_params["json_schema"] = '{"type": "object"}'
        elif (
            request.response_format and request.response_format.type == "structural_tag"
        ):
            sampling_params["structural_tag"] = convert_json_schema_to_str(
                request.response_format.model_dump(by_alias=True)
            )

        return sampling_params

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming completion request"""
        generator = self._generate_completion_stream(
            adapted_request, request, raw_request
        )

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))
        except LateDLegEnded as ended:
            return self.late_d_error_response(ended)

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
            headers=late_d_leg_headers(
                adapted_request, getattr(adapted_request, LATE_D_OUTCOME_ATTR, None)
            ),
        )

    async def _generate_completion_stream(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion response"""
        created = int(time.time())

        # State tracking for streaming
        stream_offsets = {}
        n_prev_tokens = {}
        n_prev_token_ids = {}

        # Usage tracking
        prompt_tokens = {}
        completion_tokens = {}
        reasoning_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}
        cached_tokens_details = {}
        spec_tokens_details = {}

        stream_started = False
        try:
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                self.tokenizer_manager.server_args.stream_response_default_include_usage,
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)
                if not stream_started:
                    # The first chunk says whether a symmetric choose
                    # P-leg runs Local (read by _handle_streaming_request).
                    setattr(
                        adapted_request,
                        LATE_D_OUTCOME_ATTR,
                        content["meta_info"].get("late_d_outcome"),
                    )

                text = content["text"]
                prompt_tokens[index] = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                reasoning_tokens[index] = content["meta_info"].get(
                    "reasoning_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)
                cached_tokens_details[index] = content["meta_info"].get(
                    "cached_tokens_details", None
                )
                if request.return_spec_tokens_details:
                    spec_tokens_details[index] = spec_tokens_details_from_meta_info(
                        content["meta_info"]
                    )

                is_first_chunk = index not in stream_offsets
                offset = stream_offsets.get(index, 0)
                # Handle echo for first chunk
                if is_first_chunk:  # The first chunk
                    if request.echo:
                        echo_text = self._get_echo_text(request, index)
                        text = echo_text + text

                # Handle logprobs
                logprobs = None
                if request.logprobs is not None:
                    # The first chunk and echo is enabled.
                    if is_first_chunk and request.echo:
                        input_token_logprobs = content["meta_info"][
                            "input_token_logprobs"
                        ]
                        input_top_logprobs = content["meta_info"]["input_top_logprobs"]
                    else:
                        input_token_logprobs = None
                        input_top_logprobs = None

                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if (
                        n_prev_token < total_output_logprobs
                        or input_token_logprobs is not None
                    ):
                        output_token_logprobs = content["meta_info"][
                            "output_token_logprobs"
                        ]
                        output_top_logprobs = content["meta_info"].get(
                            "output_top_logprobs", []
                        )
                        if (
                            not self.tokenizer_manager.server_args.incremental_streaming_output
                        ):
                            output_token_logprobs = output_token_logprobs[
                                n_prev_token:total_output_logprobs
                            ]
                            output_top_logprobs = output_top_logprobs[
                                n_prev_token:total_output_logprobs
                            ]
                        logprobs = to_openai_style_logprobs(
                            input_token_logprobs=input_token_logprobs,
                            input_top_logprobs=input_top_logprobs,
                            output_token_logprobs=output_token_logprobs,
                            output_top_logprobs=output_top_logprobs,
                        )
                    n_prev_tokens[index] = total_output_logprobs

                chunk_token_ids = None
                chunk_prompt_token_ids = None
                if request.return_token_ids:
                    output_ids = content["output_ids"]
                    if (
                        not self.tokenizer_manager.server_args.incremental_streaming_output
                    ):
                        n_prev_token_id = n_prev_token_ids.get(index, 0)
                        chunk_token_ids = output_ids[n_prev_token_id:]
                        n_prev_token_ids[index] = len(output_ids)
                    else:
                        chunk_token_ids = output_ids
                    if is_first_chunk:
                        chunk_prompt_token_ids = content.get("prompt_token_ids")

                # Generate delta
                if self.tokenizer_manager.server_args.incremental_streaming_output:
                    delta = text
                else:
                    delta = text[offset:]
                stream_offsets[index] = len(content["text"])
                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Abort with an explicit error status_code is a system error
                # (timeout, OOM, validation): emit a streaming error chunk.
                # A graceful abort (no status_code, e.g. user-initiated via
                # /abort_request or session lifecycle cleanup) falls through
                # to the normal chunk path, matching the non-stream behavior
                # in tokenizer_manager._handle_abort_finish_reason.
                if finish_reason_type == "abort" and isinstance(
                    finish_reason.get("status_code"), HTTPStatus
                ):
                    code = finish_reason["status_code"]
                    error = self.create_streaming_error_response(
                        finish_reason.get("message", "Generation aborted."),
                        code.name,
                        code.value,
                    )
                    yield f"data: {error}\n\n"
                    break

                choice_data = CompletionResponseStreamChoice(
                    index=index,
                    text=delta,
                    logprobs=logprobs,
                    finish_reason=finish_reason_type,
                    matched_stop=(
                        finish_reason["matched"]
                        if finish_reason and "matched" in finish_reason
                        else None
                    ),
                    token_ids=chunk_token_ids,
                    prompt_token_ids=chunk_prompt_token_ids,
                )
                chunk = CompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=created,
                    object="text_completion",
                    choices=[choice_data],
                    model=request.model,
                )

                # Add usage stats if continuous_usage_stats is enabled
                if continuous_usage_stats:
                    chunk.usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                    )

                yield f"data: {chunk.model_dump_json()}\n\n"
                stream_started = True

            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        response_hidden_states = process_hidden_states_for_response(
                            choice_hidden_states, request.return_hidden_states
                        )
                        hidden_states_chunk = CompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=created,
                            object="text_completion",
                            choices=[
                                CompletionResponseStreamChoice(
                                    index=index,
                                    text="",
                                    hidden_states=response_hidden_states,
                                    finish_reason=None,
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            sglext_routed = None
            if request.return_routed_experts and routed_experts:
                sglext_routed = next(
                    (v for v in routed_experts.values() if v is not None), None
                )

            sglext_cached_tokens_details = None
            if request.return_cached_tokens_details and cached_tokens_details:
                first_details = next(
                    (v for v in cached_tokens_details.values() if v is not None), None
                )
                if first_details is not None:
                    sglext_cached_tokens_details = cached_tokens_details_from_dict(
                        first_details
                    )

            sglext_spec_tokens_details = None
            if request.return_spec_tokens_details and spec_tokens_details:
                spec_details = [
                    spec_tokens_details[index]
                    for index in sorted(spec_tokens_details)
                    if spec_tokens_details[index] is not None
                ]
                if spec_details:
                    sglext_spec_tokens_details = (
                        spec_details if request.n > 1 else spec_details[0]
                    )

            if any(
                obj is not None
                for obj in [
                    sglext_routed,
                    sglext_cached_tokens_details,
                    sglext_spec_tokens_details,
                ]
            ):
                sglext_chunk = CompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=created,
                    object="text_completion",
                    choices=[],  # sglext is at response level
                    model=request.model,
                    sglext=SglExt(
                        routed_experts=sglext_routed,
                        cached_tokens_details=sglext_cached_tokens_details,
                        spec_tokens_details=sglext_spec_tokens_details,
                    ),
                )
                yield f"data: {sglext_chunk.model_dump_json()}\n\n"

            # Handle final usage chunk
            if include_usage:
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                )
                final_usage_chunk = CompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=created,
                    choices=[],
                    model=request.model,
                    usage=usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(exclude_none=True)
                yield f"data: {final_usage_data}\n\n"

        except Exception as e:
            if not stream_started:
                raise
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Union[CompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming completion request"""
        try:
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )
            ret = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))
        except LateDLegEnded as ended:
            return self.late_d_error_response(ended)

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_completion_response(
            request,
            ret,
            int(time.time()),
        )
        headers = late_d_leg_headers(
            adapted_request, ret[0]["meta_info"].get("late_d_outcome")
        )
        if headers is not None:
            return ORJSONResponse(content=response.model_dump(), headers=headers)
        return response

    def _build_completion_response(
        self,
        request: CompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> CompletionResponse:
        """Build completion response from generation results"""
        choices = []
        echo = False

        # Prepare echo prompts if needed
        echo_prompts = []
        if request.echo:
            echo_prompts = self._prepare_echo_prompts(request)
            echo = True

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        spec_details = [
            detail
            for detail in (
                process_spec_tokens_details_from_ret(item, request) for item in ret
            )
            if detail is not None
        ]
        spec_tokens_details = (
            spec_details
            if request.n > 1
            else (spec_details[0] if spec_details else None)
        )
        response_sglext = None
        if routed_experts or cached_tokens_details or spec_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
                spec_tokens_details=spec_tokens_details,
            )

        for idx, ret_item in enumerate(ret):
            text = ret_item["text"]

            # Handle echo
            if echo:
                prompt_index = idx // request.n
                text = echo_prompts[prompt_index] + text

            # Handle logprobs
            logprobs = None
            if request.logprobs is not None:
                if echo:
                    input_token_logprobs = ret_item["meta_info"]["input_token_logprobs"]
                    input_top_logprobs = ret_item["meta_info"]["input_top_logprobs"]
                else:
                    input_token_logprobs = None
                    input_top_logprobs = None

                logprobs = to_openai_style_logprobs(
                    input_token_logprobs=input_token_logprobs,
                    input_top_logprobs=input_top_logprobs,
                    output_token_logprobs=ret_item["meta_info"].get(
                        "output_token_logprobs", []
                    ),
                    output_top_logprobs=ret_item["meta_info"].get(
                        "output_top_logprobs", []
                    ),
                )

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]

            choice_data = CompletionResponseChoice(
                index=idx,
                text=text,
                logprobs=logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
                token_ids=(
                    ret_item["output_ids"] if request.return_token_ids else None
                ),
                prompt_token_ids=(
                    ret_item.get("prompt_token_ids")
                    if request.return_token_ids
                    else None
                ),
            )
            choices.append(choice_data)

        # Calculate usage
        cache_report = self.tokenizer_manager.server_args.enable_cache_report
        usage = UsageProcessor.calculate_response_usage(
            ret, n_choices=request.n, enable_cache_report=cache_report
        )

        return CompletionResponse(
            id=ret[0]["meta_info"]["id"],
            model=request.model,
            created=created,
            choices=choices,
            usage=usage,
            metadata=build_endpoint_weight_version_metadata(ret[0]["meta_info"]),
            sglext=response_sglext,
        )

    def _get_echo_text(self, request: CompletionRequest, index: int) -> str:
        """Get echo text for streaming response"""
        if isinstance(request.prompt, str):
            # for the case of single str prompts
            return request.prompt
        elif isinstance(request.prompt, list):
            if isinstance(request.prompt[0], str):
                # for the case of multiple str prompts
                return request.prompt[index // request.n]
            elif isinstance(request.prompt[0], int):
                # for the case of single token ids prompt
                return self.tokenizer_manager.tokenizer.decode(
                    request.prompt, skip_special_tokens=True
                )
            elif isinstance(request.prompt[0], list) and isinstance(
                request.prompt[0][0], int
            ):
                # for the case of multiple token ids prompts
                return self.tokenizer_manager.tokenizer.decode(
                    request.prompt[index // request.n],
                    skip_special_tokens=True,
                )
        return ""

    def _prepare_echo_prompts(self, request: CompletionRequest) -> List[str]:
        """Prepare echo prompts for non-streaming response"""
        # TODO: handle the case prompt is token ids
        if isinstance(request.prompt, list) and isinstance(request.prompt[0], str):
            # for the case of multiple str prompts
            return request.prompt
        elif isinstance(request.prompt, list) and isinstance(request.prompt[0], list):
            # for the case of multiple token ids prompts
            return [
                self.tokenizer_manager.tokenizer.decode(
                    prompt, skip_special_tokens=True
                )
                for prompt in request.prompt
            ]
        elif isinstance(request.prompt, list) and isinstance(request.prompt[0], int):
            # for the case of single token ids prompt
            return [
                self.tokenizer_manager.tokenizer.decode(
                    request.prompt, skip_special_tokens=True
                )
            ]
        else:
            # for the case of single str prompt
            return [request.prompt]
