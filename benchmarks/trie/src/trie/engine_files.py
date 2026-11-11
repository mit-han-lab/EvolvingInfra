"""Standalone engine helper files that Trie loads by path, without importing the engine.

Image replay and the DSV4 converters need the exact serving-side image layout and DSV4 chat
encoder. Both files depend only on the standard library, but importing them through the
``decoupled_engine`` package would pull in torch. They are loaded from the engine source tree:
``DECOUPLED_ENGINE_DIR`` (the ``decoupled_engine`` package directory) when set, else the checkout
this Trie copy lives in (``<repo>/engine/decoupled_engine``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

# benchmarks/trie/src/trie/engine_files.py -> repository root is parents[4].
_CHECKOUT_ENGINE = Path(__file__).resolve().parents[4] / "evolving_infra" / "engine" / "decoupled_engine"
# Module name the engine used for the image layout before the rename; a module registered under it
# (tests, or an environment that provides it) takes precedence over the engine tree.
_LEGACY_IMAGE_LAYOUT = "sglang_image_layout"


def engine_dir() -> Path:
    return Path(os.environ.get("DECOUPLED_ENGINE_DIR") or _CHECKOUT_ENGINE)


def dsv4_encoder_path() -> Path:
    return engine_dir() / "srt" / "entrypoints" / "openai" / "encoding_dsv4.py"


def load_file(path: Path, name: str) -> ModuleType:
    """Execute one engine source file as module ``name`` (cached in sys.modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None or not Path(path).is_file():
        raise ModuleNotFoundError(
            f"engine file {path} not found; set DECOUPLED_ENGINE_DIR to the decoupled_engine package directory"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def image_layout_path() -> Path:
    """The engine's copy of SGLang's top-level ``sglang_image_layout`` module."""
    return engine_dir() / "image_layout.py"


def image_layout() -> ModuleType:
    if _LEGACY_IMAGE_LAYOUT in sys.modules:
        return sys.modules[_LEGACY_IMAGE_LAYOUT]
    return load_file(image_layout_path(), "decoupled_engine_image_layout")


def dsv4_encoder() -> ModuleType:
    return load_file(dsv4_encoder_path(), "decoupled_engine_encoding_dsv4")
