# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from decoupled_engine.utils import LazyImport

if TYPE_CHECKING:
    from decoupled_engine.multimodal_gen.configs.pipeline_configs import PipelineConfig
    from decoupled_engine.multimodal_gen.configs.sample import SamplingParams

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]

DiffGenerator = LazyImport(
    "decoupled_engine.multimodal_gen.runtime.entrypoints.diffusion_generator",
    "DiffGenerator",
)


def __getattr__(name: str) -> Any:
    if name == "PipelineConfig":
        from decoupled_engine.multimodal_gen.configs.pipeline_configs import PipelineConfig

        value = PipelineConfig
    elif name == "SamplingParams":
        from decoupled_engine.multimodal_gen.configs.sample import SamplingParams

        value = SamplingParams
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


# Trigger multimodal CI tests
