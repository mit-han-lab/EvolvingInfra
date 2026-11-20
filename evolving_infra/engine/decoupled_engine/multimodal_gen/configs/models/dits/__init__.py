# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from decoupled_engine.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from decoupled_engine.multimodal_gen.configs.models.dits.helios import HeliosConfig
from decoupled_engine.multimodal_gen.configs.models.dits.hunyuan3d import Hunyuan3DDiTConfig
from decoupled_engine.multimodal_gen.configs.models.dits.hunyuanvideo import HunyuanVideoConfig
from decoupled_engine.multimodal_gen.configs.models.dits.ideogram import (
    Ideogram4DistilledDiTConfig,
    Ideogram4DiTConfig,
)
from decoupled_engine.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEConfig,
)
from decoupled_engine.multimodal_gen.configs.models.dits.lingbot_world import (
    LingBotWorldVideoConfig,
)
from decoupled_engine.multimodal_gen.configs.models.dits.llada_image import LLaDAImageDitConfig
from decoupled_engine.multimodal_gen.configs.models.dits.longlive2 import LongLive2VideoConfig
from decoupled_engine.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from decoupled_engine.multimodal_gen.configs.models.dits.mova_audio import MOVAAudioConfig
from decoupled_engine.multimodal_gen.configs.models.dits.mova_video import MOVAVideoConfig
from decoupled_engine.multimodal_gen.configs.models.dits.sana_video import SanaVideoConfig
from decoupled_engine.multimodal_gen.configs.models.dits.stablediffusion3 import (
    StableDiffusion3TransformerConfig,
)
from decoupled_engine.multimodal_gen.configs.models.dits.wan_animate_2 import (
    WanAnimate2Config,
    WanAnimate2DistilledConfig,
)
from decoupled_engine.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig

__all__ = [
    "Cosmos3VideoConfig",
    "HeliosConfig",
    "HunyuanVideoConfig",
    "Ideogram4DiTConfig",
    "Ideogram4DistilledDiTConfig",
    "LingBotWorldVideoConfig",
    "LingBotVideoMoEConfig",
    "LLaDAImageDitConfig",
    "LongLive2VideoConfig",
    "MiniMaxH3DiTConfig",
    "WanAnimate2Config",
    "WanAnimate2DistilledConfig",
    "WanVideoConfig",
    "Hunyuan3DDiTConfig",
    "MOVAAudioConfig",
    "MOVAVideoConfig",
    "SanaVideoConfig",
    "StableDiffusion3TransformerConfig",
]
