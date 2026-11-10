# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DeepSeek-V4 image preprocessing, matching inference/image_processor.py in
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp. Block lengths must match the wrapper's
embeddings, including compression padding and sentinels.
"""

import asyncio
import hashlib
import io
import re
from typing import List, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from decoupled_engine.image_layout import (
    COMPRESS_PAD_TO,
    build_image_block,
    expand_image_layout,
    image_config,
    image_grid,
    prepare_image_layout,
)

from decoupled_engine.srt.managers.mm_utils import hash_feature
from decoupled_engine.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from decoupled_engine.srt.models.deepseek_v4_vl import DeepseekV4ForCausalLM
from decoupled_engine.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def load_image_patches(image, config):
    """Reference RGB preprocessing; the layout functions themselves need no pixels."""
    p = config["vision_patch_size"]
    if not isinstance(image, Image.Image):
        arr = torch.as_tensor(image)
        if arr.dim() == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.permute(1, 2, 0)
        image = Image.fromarray(arr.cpu().numpy().astype(np.uint8))
    image = image.convert("RGB")
    n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_grid(*image.size, config)
    size = (n_vit_w * p, n_vit_h * p)
    max_ratio = config["vision_max_wh_ratio"]
    if max_ratio is not None and image.width >= max_ratio * image.height:
        image = image.resize(size)
    else:
        image = ImageOps.pad(image, size, color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def materialize_prepared_images(mm_items):
    """Recover original pixels before a metadata-only D request becomes a prefill.

    Called on CPU before GPU execution. Normal successful KV imports never call
    this function; each request keeps its own opaque image-owner reference.
    """
    import requests

    for item in mm_items:
        if item.feature is not None or not item.is_image():
            continue
        data = item.model_specific_data
        source = data.get("prepared_image_source")
        if source is None:
            raise ValueError("Metadata-only images have no live image owner")
        body = {key: value for key, value in source.items() if key != "endpoint"}
        response = requests.post(source["endpoint"], json=body, timeout=60)
        response.raise_for_status()
        raw = response.content
        if hashlib.sha256(raw).hexdigest() != data["sha256"]:
            raise ValueError("Recovered image differs from its prepared identity")
        with Image.open(io.BytesIO(raw)) as image:
            patches, h, w, _, _ = load_image_patches(image, data["image_config"])
        if (h, w) != (data["n_vit_h"], data["n_vit_w"]):
            raise ValueError("Recovered image differs from its prepared grid")
        item.feature = patches


class DeepseekV4VLImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    prefer_tokenized_input = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        image_token_id = self._tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if image_token_id is None or image_token_id == self._tokenizer.unk_token_id:
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        self.image_token_id = image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=IMAGE_PLACEHOLDER,
            image_token_id=image_token_id,
        ).build(_processor)

        # vision preprocessing hyperparameters from the HF config
        self.patch_size = hf_config.vision_patch_size
        self.downsample_ratio = hf_config.vision_downsample_ratio
        self.max_n_token = hf_config.vision_max_n_token
        self.min_pixels = hf_config.vision_min_pixels
        self.max_wh_ratio = hf_config.vision_max_wh_ratio
        self.image_config = image_config(hf_config, image_token_id)

    def _load_image(self, image):
        return load_image_patches(image, self.image_config)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        layout = getattr(request_obj, "prepared_image_layout", None)
        if layout is not None:
            if not isinstance(input_text, list):
                raise ValueError("Prepared image layout requires original input_ids")
            expanded, cached, metadata = expand_image_layout(
                input_text, layout, self.image_config
            )
            if image_data:
                actual, _, _, _ = prepare_image_layout(
                    input_text, image_data, self.image_config
                )
                if actual != layout:
                    raise ValueError(
                        "Uploaded images differ from their prepared layout"
                    )
                base = await self.load_mm_data(
                    input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
                )
                images = base.images
            else:
                images = []
            source = getattr(request_obj, "prepared_image_source", None)
            if not images and source is None:
                raise ValueError("Metadata-only images require an original image owner")
        else:
            base_output = await self.load_mm_data(
                input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
            )
            images = base_output.images
            if isinstance(input_text, list):
                # Preserve exact text IDs instead of decoding and re-encoding them.
                original = input_text
            else:
                parts = re.split(re.escape(IMAGE_PLACEHOLDER), base_output.input_text)
                original = []
                for i, part in enumerate(parts):
                    if part:
                        original.extend(self._tokenizer.encode(part))
                    if i < len(parts) - 1:
                        original.append(self.image_token_id)
            if original.count(self.image_token_id) != len(images):
                raise ValueError("Image placeholders and image inputs differ in count")

        def prepare_items():
            # Resize, normalize and feature hashing are CPU work. Keep them off
            # the request event loop so independent requests can make progress.
            mm_items = []
            if layout is not None:
                for i, data in enumerate(metadata):
                    patches = self._load_image(images[i])[0] if images else None
                    specific = {
                        key: data[key]
                        for key in ("types", "perm", "n_vit_h", "n_vit_w")
                    }
                    specific.update(
                        image_config=self.image_config,
                        sha256=layout["images"][i]["sha256"],
                    )
                    if source is not None:
                        specific["prepared_image_source"] = {**source, "image_index": i}
                    if i == 0:
                        specific.update(
                            raw_input_ids=list(input_text), prepared_image_layout=layout
                        )
                        if source is None:
                            specific["original_image_data"] = image_data
                    mm_items.append(
                        MultimodalDataItem(
                            modality=Modality.IMAGE,
                            feature=patches,
                            hash=int(data["hash"], 16),
                            pad_value=data["pad_value"],
                            offsets=data["offsets"],
                            model_specific_data=specific,
                        )
                    )
                if not images and getattr(request_obj, "kv_import", None) is None:
                    # Only a symmetric KV import can defer fetching pixels.
                    materialize_prepared_images(mm_items)
                return MultimodalProcessorOutput(
                    mm_items=mm_items,
                    input_ids=expanded,
                    padded_input_ids=cached,
                    im_token_id=self.image_token_id,
                )

            input_ids = []
            image_index = 0
            for token in original:
                if token != self.image_token_id:
                    input_ids.append(token)
                    continue
                patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self._load_image(
                    images[image_index]
                )
                image_index += 1
                types, perm = build_image_block(n_llm_h, n_llm_w, len(input_ids))
                start = len(input_ids)
                input_ids.extend([self.image_token_id] * len(types))
                mm_items.append(
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        feature=patches,
                        hash=hash_feature(
                            [
                                patches,
                                torch.tensor(
                                    [n_vit_h, n_vit_w, start % COMPRESS_PAD_TO],
                                    dtype=torch.int64,
                                ),
                            ]
                        ),
                        offsets=[(start, len(input_ids) - 1)],
                        model_specific_data={
                            "types": types,
                            "perm": perm,
                            "n_vit_h": n_vit_h,
                            "n_vit_w": n_vit_w,
                        },
                    )
                )
            return MultimodalProcessorOutput(
                mm_items=mm_items, input_ids=input_ids, im_token_id=self.image_token_id
            )

        if (
            layout is not None
            and not images
            and getattr(request_obj, "kv_import", None) is not None
        ):
            return prepare_items()  # Metadata-only decode has no pixel computation.
        return await asyncio.to_thread(prepare_items)
