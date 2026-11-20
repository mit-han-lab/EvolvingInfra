"""Config-time override declarations for falcon_h1."""

from typing import Any

from decoupled_engine.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from decoupled_engine.srt.runtime_context import get_platform


@_register_for(
    "FalconH1ForCausalLM", "JetNemotronForCausalLM", "JetVLMForConditionalGeneration"
)
def _falcon_h1_jet_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if get_platform().is_sm100 and cfg.attention_backend is None:
        return {"attention_backend": "triton"}
    return {}
