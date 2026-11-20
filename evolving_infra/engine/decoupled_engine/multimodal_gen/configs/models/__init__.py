# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from decoupled_engine.multimodal_gen.configs.models.base import ModelConfig
from decoupled_engine.multimodal_gen.configs.models.dits.base import DiTConfig
from decoupled_engine.multimodal_gen.configs.models.encoders.base import EncoderConfig
from decoupled_engine.multimodal_gen.configs.models.vaes.base import VAEConfig

__all__ = ["ModelConfig", "VAEConfig", "DiTConfig", "EncoderConfig"]
