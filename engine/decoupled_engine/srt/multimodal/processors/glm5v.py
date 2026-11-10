# SPDX-License-Identifier: Apache-2.0
"""SGLang multimodal processor for glm5v.

Subclass of the in-tree Kimi-K2.5 GPU image processor with GLM's image
placeholder: ``<|image|>`` (id from ``config.media_placeholder_token_id``,
154854) instead of Kimi's ``<|media_pad|>``. The GLM chat template
(``chat_template.jinja`` in the assembled checkpoint) wraps each image as
``<|begin_of_image|><|image|><|end_of_image|>``; only the inner ``<|image|>``
run is expanded to per-patch tokens.

All preprocessing (NaViT resize math, GPU patchify, grid_thw handling) is
inherited unchanged — it is the same MoonViT pipeline the projector was
trained against.
"""

import os
import re

import numpy as np
import torch
from PIL import Image

from decoupled_engine.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from decoupled_engine.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    KimiK2_5VLImageProcessor,
    _ensure_chw_rgb,
    _get_image_dimensions,
    _gpu_preprocess_images,
    image_config,
    navit_resize_config,
)

from decoupled_engine.srt.models.glm5v import Glm5vForConditionalGeneration


def _cpu_chw(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
    return _ensure_chw_rgb(image).cpu()


# GPU memory the HTTP process may hold for screenshot preprocessing.
_GPU_BUDGET_BYTES = int(os.environ.get("SGLANG_GLM5V_MM_GPU_BUDGET_MB", "4096")) << 20


class Glm5vProcessorWrapper(KimiGPUProcessorWrapper):
    """The Kimi GPU wrapper, kept within a small GPU budget, else run on the CPU.

    The HTTP process shares GPU 0 with the scheduler, and GLM's scheduler uses
    nearly all of it (DSA buffers on top of the static pool). Concurrent CUA
    requests carry dozens of screenshots each; preprocessing them on the GPU left
    ~14 GiB in this process's allocator cache and the scheduler then hit CUDA out
    of memory. Features travel to the scheduler through CPU memory anyway
    (--mm-feature-transport=cpu), so each result moves to the CPU at once and the
    cache is released. A batch that would exceed the budget, or a GPU attempt
    that runs out of memory, runs the same resize/normalize/patchify on the CPU.
    """

    def _gpu_call(self, text, images, original_input_ids=None):
        need = 0
        for image in images:
            w, h = _get_image_dimensions(image)
            need += w * h * 3 * 4 * 3  # uint8 source, float resize and patch copies
        if torch.cuda.memory_reserved() + need <= _GPU_BUDGET_BYTES:
            try:
                out = super()._gpu_call(text, images, original_input_ids)
                out["pixel_values"] = out["pixel_values"].cpu()
                return out
            except RuntimeError as error:
                if "out of memory" not in str(error):
                    raise
            finally:
                torch.cuda.empty_cache()
        input_text = text[0] if isinstance(text, list) else text
        resize_configs = []
        for image in images:
            w, h = _get_image_dimensions(image)
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )
        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids
        )
        image_scale, image_bias = self._get_gpu_norm_tensors(device="cpu")
        pixel_values, grid_thws = _gpu_preprocess_images(
            images,
            resize_configs,
            image_scale,
            image_bias,
            self._patch_size,
            to_chw=_cpu_chw,
        )
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thws,
        }

    def _get_gpu_norm_tensors(self, device="cuda"):
        if device == "cuda":
            return super()._get_gpu_norm_tensors(device)
        image_scale = torch.tensor(
            [1.0 / (255.0 * std) for std in self._image_std], dtype=torch.float32
        ).view(1, 3, 1, 1)
        image_bias = torch.tensor(
            [-mean / std for mean, std in zip(self._image_mean, self._image_std)],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        return image_scale, image_bias


class Glm5vImageProcessor(KimiK2_5VLImageProcessor):
    models = [Glm5vForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # The parent __init__ hardcodes Kimi's <|media_pad|>; this is its body with
        # GLM's <|image|> and the same GPU wrapper and image layout.
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|image\|>)+"),
        ).build(_processor)

        media_proc_cfg = _processor.media_processor.media_proc_cfg
        processor = Glm5vProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=media_proc_cfg["image_mean"],
            image_std=media_proc_cfg["image_std"],
        )
        super(KimiK2_5VLImageProcessor, self).__init__(
            hf_config, server_args, processor, *args, **kwargs
        )
        self.mm_tokens = mm_tokens
        self.image_config = image_config(
            hf_config, mm_tokens.image_token_id, media_proc_cfg
        )
