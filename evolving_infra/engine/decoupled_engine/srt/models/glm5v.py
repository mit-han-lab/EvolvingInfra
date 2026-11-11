# SPDX-License-Identifier: Apache-2.0
"""glm5v on SGLang: MoonViT (Kimi-K2.5) vision tower + trained PatchMerger
projector + GLM-5.2 (GlmMoeDsa) text decoder.

It is SGLang's in-tree ``KimiK25ForConditionalGeneration`` with three swaps:

  1. text backbone ``DeepseekV3ForCausalLM`` -> ``GlmMoeDsaForCausalLM``
     (GLM-5.2; SGLang ships it natively in ``glm4_moe.py``, DSA path included).
  2. projector ``linear_2`` output dim -> GLM hidden 6144, via
     ``vision_config.text_hidden_size`` (pure config, no code).
  3. image placeholder -> GLM's ``<|image|>`` (154854), via
     ``config.media_placeholder_token_id`` + the Glm5v processor (see
     ``processor.py``; the model itself is placeholder-agnostic).

Only swap (1) needs code, so ``__init__`` is a copy of the parent's with the
language-model class replaced. Everything else (MoonViT tower, projector,
``get_image_feature``, ``forward``, ``load_weights`` incl. the
``mm_projector.proj.0 -> linear_1`` key remap) is inherited verbatim.

Vision tower + projector stay bf16: the parent passes ``quant_config=None`` to
the tower (unless ModelSlim) and builds the projector unquantized; only the GLM
text backbone sees the FP8 quant config.
"""

import logging

from torch import nn

from decoupled_engine.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from decoupled_engine.srt.layers.quantization.quark.quark import QuarkConfig
from decoupled_engine.srt.models.glm4_moe import GlmMoeDsaForCausalLM
from decoupled_engine.srt.models.kimi_k25 import (
    K2VLMultiModalProjector,
    KimiK25ForConditionalGeneration,
    MoonViT3dPretrainedModel,
)
from decoupled_engine.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class Glm5vForConditionalGeneration(KimiK25ForConditionalGeneration):
    """glm5v: MoonViT vision + trained projector + GLM-5.2 text."""

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
        **kwargs,  # match parent (absorbs init_tts etc.)
    ) -> None:
        # Skip KimiK25ForConditionalGeneration.__init__ (it hardcodes the
        # DeepseekV3 text class); replicate its body with GlmMoeDsa instead.
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # Vision tower (bf16; quant only under ModelSlim, same as parent).
        self.vision_tower = MoonViT3dPretrainedModel(
            config.vision_config,
            use_data_parallel=self.use_data_parallel,
            quant_config=(
                quant_config if isinstance(quant_config, ModelSlimConfig) else None
            ),
            prefix="vision_tower",
        )
        # MM projector (unquantized; linear_2 out = vision_config.text_hidden_size).
        self.mm_projector = K2VLMultiModalProjector(config.vision_config)

        self.language_model = None
        if not config.encoder_only:
            # THE swap: GLM-5.2 (MLA + MoE + DSA sparse attention) text backbone.
            self.language_model = GlmMoeDsaForCausalLM(
                config.text_config,
                quant_config,
                prefix=(
                    "language_model"
                    if isinstance(quant_config, (ModelSlimConfig, QuarkConfig))
                    else ""
                ),
            )

        # Match vision/projector dtype to the language model (parent behavior).
        if self.language_model is not None and hasattr(self.language_model, "dtype"):
            target_dtype = self.language_model.dtype
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)


EntryClass = [Glm5vForConditionalGeneration]
