"""Lazy access to the two engine modules the frontend shares with DecoupledEngine.

Both are optional: a text-only deployment with ``hash_mode="dynamo"`` needs
neither. Importing this module imports no engine code.

- ``image_layout()``: the CPU-only checkpoint image layout
  (``decoupled_engine/image_layout.py``, standard library only). The frontend
  and the engine must use the same file so prompt expansion and image cache IDs
  agree. It is loaded by file path, because importing it as
  ``decoupled_engine.image_layout`` runs the package ``__init__`` (environment
  setup, transformers patches, Torch).
- ``native_hash()``: the engine's native prefix-page hash, used only by
  ``hash_mode="sglang_native"``. This one imports the engine normally.
"""

import importlib
import importlib.util
import os
from pathlib import Path
import sys

IMAGE_LAYOUT_MODULE = "decoupled_engine_image_layout"
NATIVE_HASH_MODULE = "decoupled_engine.srt.mem_cache.cpp_utils.native_hash"
# router/frontend/_engine.py -> repository root is parents[2].
_CHECKOUT_ENGINE = Path(__file__).resolve().parents[2] / "engine" / "decoupled_engine"


def image_layout_candidates():
    """Return the paths searched for ``image_layout.py``, in preference order.

    ``DECOUPLED_ENGINE_DIR`` (the ``decoupled_engine`` package directory) when
    set, then the installed ``decoupled_engine`` package (located without
    importing it), then the engine in this source checkout.
    """
    dirs = []
    if os.environ.get("DECOUPLED_ENGINE_DIR"):
        dirs.append(Path(os.environ["DECOUPLED_ENGINE_DIR"]))
    try:
        spec = importlib.util.find_spec("decoupled_engine")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        dirs.extend(Path(p) for p in spec.submodule_search_locations)
    dirs.append(_CHECKOUT_ENGINE)
    return [d / "image_layout.py" for d in dirs]


def image_layout():
    """Return the shared checkpoint image layout module.

    Raises:
        ModuleNotFoundError: No candidate file exists.
    """
    if IMAGE_LAYOUT_MODULE in sys.modules:
        return sys.modules[IMAGE_LAYOUT_MODULE]
    candidates = image_layout_candidates()
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(IMAGE_LAYOUT_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[IMAGE_LAYOUT_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[IMAGE_LAYOUT_MODULE]
            raise
        return module
    raise ModuleNotFoundError(
        "Image inputs need the engine's image layout file; install decoupled-engine or set "
        f"DECOUPLED_ENGINE_DIR (tried {', '.join(map(str, candidates))})",
        name=IMAGE_LAYOUT_MODULE)


def native_hash():
    """Return the engine's ``get_native_hash`` function."""
    return importlib.import_module(NATIVE_HASH_MODULE).get_native_hash
