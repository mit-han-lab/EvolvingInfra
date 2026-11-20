# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

This package contains the various stages that can be composed to create
complete diffusion pipelines.
"""

from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation import (
    ComfyUILatentPreparationStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.condition_encoding import (
    AuxiliaryConditionEncodingStage,
    ConditionEncodingStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.denoising_dmd import (
    DmdDenoisingStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    DMDTimestepPreparationStage,
    TimestepPreparationStage,
)
from decoupled_engine.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
    VLAActionPostprocessStage,
    VLAObservationPreprocessStage,
    VLAPrefixEncodingStage,
)

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "DMDTimestepPreparationStage",
    "LatentPreparationStage",
    "ComfyUILatentPreparationStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "CausalDMDDenoisingStage",
    "ConditionEncodingStage",
    "AuxiliaryConditionEncodingStage",
    "DecodingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "TextEncodingStage",
    "VLAObservationPreprocessStage",
    "VLAPrefixEncodingStage",
    "VLAActionDenoisingStage",
    "VLAActionPostprocessStage",
]
