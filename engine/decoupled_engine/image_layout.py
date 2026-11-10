"""CPU-only checkpoint image layouts shared by preparation and engine workers.

The module itself uses only the standard library (Pillow lazily) and loads no
model; importing it as ``decoupled_engine.image_layout`` runs the package
``__init__``.

The input is the original token sequence (one placeholder per image), image dimensions,
and a SHA256 of each uploaded file. It produces the expanded prompt and the
pad IDs used by the worker's prefix cache. Image pixels remain at the worker.
"""

import base64
import hashlib
import io
import json
import math
import re

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
MM_PAD_SHIFT_VALUE = 1_000_000
_CONFIG_FIELDS = (
    "image_token_id",
    "vision_patch_size",
    "vision_downsample_ratio",
    "vision_max_n_token",
    "vision_min_pixels",
    "vision_max_wh_ratio",
)
_KIMI_CONFIG_FIELDS = (
    "layout_type",
    "image_token_id",
    "patch_size",
    "merge_kernel_size",
    "in_patch_limit",
    "patch_limit_on_one_side",
    "fixed_output_tokens",
    "image_mean",
    "image_std",
)


def image_config(config, image_token_id, processor_config=None):
    """Freeze only the checkpoint fields that determine image preprocessing."""
    get = config.get if isinstance(config, dict) else lambda key: getattr(config, key)
    kind = (
        config.get("layout_type", config.get("model_type"))
        if isinstance(config, dict)
        else getattr(config, "model_type", None)
    )
    # GLM-5.2-Vision runs the same MoonViT tower and merger, with GLM's placeholder.
    if kind in ("kimi_k25", "glm5v"):
        if processor_config is None:
            if not isinstance(config, dict) or set(config) != set(_KIMI_CONFIG_FIELDS):
                raise ValueError(
                    "Kimi image layout requires the checkpoint processor configuration"
                )
            result = dict(config)
        else:
            settings = processor_config.get("media_proc_cfg", processor_config)
            result = {
                key: settings.get(key)
                for key in _KIMI_CONFIG_FIELDS
                if key not in ("layout_type", "image_token_id")
            }
            result.update(layout_type="kimi_k25", image_token_id=image_token_id)
        if result["image_token_id"] != image_token_id:
            raise ValueError("Kimi image placeholder differs from the checkpoint")
        for key in (
            "image_token_id",
            "patch_size",
            "merge_kernel_size",
            "in_patch_limit",
            "patch_limit_on_one_side",
        ):
            if type(result[key]) is not int or result[key] <= 0:
                raise ValueError(f"Invalid Kimi image configuration: {key}")
        # MoonViT's patch merger determines the number of output vectors. The
        # K2.6 checkpoint does not use an independent fixed-token resampler.
        if result["fixed_output_tokens"] is not None:
            raise ValueError(
                "Prepared Kimi images require variable MoonViT output tokens"
            )
        for key in ("image_mean", "image_std"):
            values = result[key]
            if (
                not isinstance(values, (tuple, list))
                or len(values) != 3
                or any(
                    type(v) not in (float, int) or not math.isfinite(v) for v in values
                )
                or (key == "image_std" and min(values) <= 0)
            ):
                raise ValueError(f"Invalid Kimi image normalization: {key}")
            result[key] = list(values)
        return result
    result = {key: get(key) for key in _CONFIG_FIELDS if key != "image_token_id"}
    result["image_token_id"] = image_token_id
    for key, value in result.items():
        if key == "vision_max_wh_ratio" and value is None:
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid DeepSeek-V4 image configuration: {key}")
    if result["vision_max_n_token"] < 8:
        raise ValueError("Image token budget is too small")
    return result


def kimi_resize_config(width, height, config):
    """MoonViT/NaViT resize and padding, without Torch or decoded pixels."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    p, merge = config["patch_size"], config["merge_kernel_size"]
    limit = config["patch_limit_on_one_side"]
    scale = min(
        1.0,
        math.sqrt(
            config["in_patch_limit"] / (max(1.0, width // p) * max(1.0, height // p))
        ),
        limit * p / width,
        limit * p / height,
    )
    w, h = min(max(1, int(width * scale)), limit * p), min(
        max(1, int(height * scale)), limit * p
    )
    factor = p * merge
    pad_w, pad_h = (factor - w % factor) % factor, (factor - h % factor) % factor
    return {
        "num_tokens": ((h + pad_h) // factor) * ((w + pad_w) // factor),
        "new_width": w,
        "new_height": h,
        "pad_width": pad_w,
        "pad_height": pad_h,
    }


def kimi_vision_matmul_flops(config, n_vit_h, n_vit_w):
    """Actual MoonViT image encoder and K2 patch-merger matrix work.

    GELU's two MLP projections differ from DeepSeek's gated MLP. K2's aligner
    projects merged_dim -> merged_dim -> text_hidden_size. Count multiply/add
    as two operations; omit normalization, activation, resize and softmax.
    """
    get = config.get if isinstance(config, dict) else lambda key: getattr(config, key)
    vision = get("vision_config")
    vg = vision.get if isinstance(vision, dict) else lambda key: getattr(vision, key)
    dim, intermediate = vg("vt_hidden_size"), vg("vt_intermediate_size")
    merge_h, merge_w = vg("merge_kernel_size")
    patches = n_vit_h * n_vit_w
    vectors = math.ceil(n_vit_h / merge_h) * math.ceil(n_vit_w / merge_w)
    merged = dim * merge_h * merge_w
    patch_projection = 2 * patches * 3 * vg("patch_size") ** 2 * dim
    transformer = vg("vt_num_hidden_layers") * (
        patches * (8 * dim**2 + 4 * dim * intermediate) + 4 * patches**2 * dim
    )
    projector = 2 * vectors * (merged**2 + merged * vg("text_hidden_size"))
    return patch_projection + transformer + projector


def image_grid(width, height, config):
    """Reference resize grid without decoding pixels; returns ViT/LLM grid sizes."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    p, r = config["vision_patch_size"], config["vision_downsample_ratio"]
    max_ratio = config["vision_max_wh_ratio"]
    if max_ratio is not None and width > height * max_ratio:
        width = height * max_ratio
    if width * height < config["vision_min_pixels"]:
        ratio = math.sqrt(config["vision_min_pixels"] / (width * height))
        width, height = int(width * ratio), int(height * ratio)
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p

    def grid_tokens(h, w):
        llm_h, llm_w = math.ceil((h // p) / r), math.ceil((w // p) / r)
        count = llm_h * (llm_w + 1) + 2
        if llm_h % 2:
            count += llm_w + 1
        count += ((llm_h + 1) // 2 * (llm_w + 1) % 2) * 2
        return llm_h, llm_w, count

    max_tokens = config["vision_max_n_token"] - (COMPRESS_PAD_TO - 1)
    budget = max_tokens
    llm_h, llm_w, count = grid_tokens(best_h, best_w)
    while count > max_tokens:
        aspect = height / width
        max_w_float = math.sqrt((budget - 2) / aspect + 0.25) - 0.5
        max_h_float = max_w_float * aspect
        if max_w_float < 1:
            max_w = 1
            max_h = (budget - 2) // 2
            max_h -= max_h % 2
            best_w, best_h = max_w * p * r, max_h * p * r
        elif max_h_float < 2:
            max_h, max_w = 2, (budget - 2) // 2 - 1
            best_w, best_h = max_w * p * r, max_h * p * r
        else:
            max_w, max_h = math.floor(max_w_float), math.floor(max_h_float)
            max_h -= max_h % 2
            beta = min(max_w * p * r / width, max_h * p * r / height)
            best_w = math.floor(width * beta / p) * p
            best_h = math.floor(height * beta / p) * p
        if min(best_w, best_h) <= 0:
            raise ValueError("Image cannot fit its token budget")
        llm_h, llm_w, count = grid_tokens(best_h, best_w)
        budget -= 1
    return best_h // p, best_w // p, llm_h, llm_w


def vision_matmul_flops(config, n_vit_h, n_vit_w):
    """Matrix multiply FLOPs for one ViT + aligner invocation (not GPU time).

    Count multiply/add as two operations; exclude normalization, activation,
    and softmax. The capacity fit learns seconds per unit of this work.
    """
    get = config.get if isinstance(config, dict) else lambda key: getattr(config, key)
    patches = n_vit_h * n_vit_w
    dim, intermediate = get("vision_dim"), get("vision_inter_dim")
    ratio, hidden = get("vision_downsample_ratio"), get("hidden_size")
    vectors = math.ceil(n_vit_h / ratio) * math.ceil(n_vit_w / ratio)
    patch_projection = 2 * patches * 3 * get("vision_patch_size") ** 2 * dim
    transformer = get("vision_n_layers") * (
        patches * (8 * dim**2 + 6 * dim * intermediate) + 4 * patches**2 * dim
    )
    aligner = 2 * vectors * (dim * ratio**2 * hidden + hidden**2)
    return patch_projection + transformer + aligner


def build_image_block(n_llm_h, n_llm_w, start_pos):
    """Reference N-layout: sentinel types and row-major aligner permutation."""
    rows, row_len = n_llm_h + n_llm_h % 2, n_llm_w + 1
    types, perm = [], []
    for base in range(0, rows, 2):
        for col in range(row_len):
            for row in (base, base + 1):
                if row >= n_llm_h:
                    types.append(IMAGE_PAD)
                elif col == n_llm_w:
                    types.append(IMAGE_NEW_LINE)
                else:
                    types.append(IMAGE)
                    perm.append(row * n_llm_w + col)
    types = (
        [IMAGE_PAD] * (COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO)
        + [IMAGE_START]
        + types
        + [IMAGE_PAD] * ((rows // 2 * row_len % 2) * 2)
        + [IMAGE_END]
    )
    return types, perm


def prepare_image_layout(input_ids, image_data, config):
    """Read uploaded PNG/JPEG headers and hash file bytes, without pixel decoding."""
    from PIL import Image

    if not isinstance(image_data, list) or not image_data:
        raise ValueError("image_data must contain uploaded images")
    images = []
    for encoded in image_data:
        if not isinstance(encoded, str):
            raise ValueError("Prepared images must be Base64 strings or data URIs")
        if encoded.startswith("data:"):
            header, encoded = encoded.split(",", 1)
            if not header.startswith("data:image/") or not header.endswith(";base64"):
                raise ValueError("Expected a Base64 image data URI")
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
        images.append(
            {
                "width": width,
                "height": height,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    layout = {"version": 1, "config": dict(config), "images": images}
    expanded, cached, items = expand_image_layout(input_ids, layout, config)
    return layout, expanded, cached, items


def expand_image_layout(input_ids, layout, config):
    """Validate metadata and expand exact IDs for both pixel-owning P and metadata-only D."""
    config = image_config(config, config["image_token_id"])
    if (
        not isinstance(layout, dict)
        or set(layout) != {"version", "config", "images"}
        or type(layout["version"]) is not int
        or layout["version"] != 1
        or layout["config"] != config
    ):
        raise ValueError("Prepared image layout differs from the worker configuration")
    images = layout["images"]
    if not isinstance(images, list) or not images:
        raise ValueError("Image layout requires nonempty image metadata")
    if input_ids.count(config["image_token_id"]) != len(images):
        raise ValueError("Image placeholders and uploaded images differ in count")
    expanded, cached, items = [], [], []
    image_index = 0
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    for token in input_ids:
        if token != config["image_token_id"]:
            expanded.append(token)
            cached.append(token)
            continue
        info = images[image_index]
        image_index += 1
        if (
            not isinstance(info, dict)
            or set(info) != {"width", "height", "sha256"}
            or not isinstance(info["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None
        ):
            raise ValueError("Invalid uploaded image identity")
        start = len(expanded)
        resize = None
        if config.get("layout_type") == "kimi_k25":
            resize = kimi_resize_config(info["width"], info["height"], config)
            p, merge = config["patch_size"], config["merge_kernel_size"]
            vit_h = (resize["new_height"] + resize["pad_height"]) // p
            vit_w = (resize["new_width"] + resize["pad_width"]) // p
            llm_h, llm_w = vit_h // merge, vit_w // merge
            types, perm = [IMAGE] * resize["num_tokens"], list(
                range(resize["num_tokens"])
            )
            identity_input = (
                b"sglang-kimi-image-layout-v1\0"
                + config_bytes
                + bytes.fromhex(info["sha256"])
                + f":{vit_h}:{vit_w}".encode()
            )
        else:
            vit_h, vit_w, llm_h, llm_w = image_grid(
                info["width"], info["height"], config
            )
            types, perm = build_image_block(llm_h, llm_w, start)
            identity_input = (
                b"sglang-dsv4-image-layout-v1\0"
                + config_bytes
                + bytes.fromhex(info["sha256"])
                + f":{vit_h}:{vit_w}:{start % COMPRESS_PAD_TO}".encode()
            )
        identity = hashlib.sha256(identity_input).hexdigest()[-16:]
        pad_value = MM_PAD_SHIFT_VALUE + int(identity, 16) % (1 << 30)
        expanded.extend([config["image_token_id"]] * len(types))
        cached.extend([pad_value] * len(types))
        items.append(
            {
                "hash": identity,
                "pad_value": pad_value,
                "offsets": [(start, len(expanded) - 1)],
                "n_vit_h": vit_h,
                "n_vit_w": vit_w,
                "n_llm_h": llm_h,
                "n_llm_w": llm_w,
                "types": types,
                "perm": perm,
            }
        )
        if resize is not None:
            items[-1].update(resize_config=resize, image_grid_thw=[1, vit_h, vit_w])
    return expanded, cached, items
