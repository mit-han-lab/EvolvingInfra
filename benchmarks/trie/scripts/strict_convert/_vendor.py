"""Load the vendored trie package without executing its heavy __init__.

``trie/__init__.py`` imports the async client (httpx, openai, rich), none of
which the conversion needs. Registering the package in ``sys.modules``
without running its ``__init__`` lets us import the exact vendored
``trie.tokenizer_manager`` / ``trie.validation`` / ``trie.types`` modules the
replay client uses, so conversion-time rendering and validation share one
code path with replay. A minimal ``structlog`` shim is installed when the
real library is absent (the trace-conversion venv does not ship it).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

TRIE_ROOT = Path(__file__).resolve().parents[2]
TRIE_SRC = TRIE_ROOT / "src" / "trie"
SCHEMA_PATH = TRIE_ROOT / "schema" / "replay-trace-v2.schema.json"


def _install_structlog_shim() -> None:
    if importlib.util.find_spec("structlog") is not None:
        return

    class _ShimLogger:
        def __getattr__(self, name: str):
            def _log(*args: object, **kwargs: object) -> None:
                del args, kwargs

            return _log

    shim = types.ModuleType("structlog")
    shim.get_logger = lambda *args, **kwargs: _ShimLogger()  # type: ignore[attr-defined]
    sys.modules["structlog"] = shim


def bootstrap() -> None:
    """Make ``trie.<submodule>`` importable; safe to call repeatedly."""
    if "trie" in sys.modules:
        return
    _install_structlog_shim()
    init_path = TRIE_SRC / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"vendored trie package not found at {TRIE_SRC}")
    spec = importlib.util.spec_from_file_location(
        "trie", init_path, submodule_search_locations=[str(TRIE_SRC)]
    )
    if spec is None:
        raise ImportError("could not build a module spec for the vendored trie")
    package = importlib.util.module_from_spec(spec)
    # Registering without exec_module skips trie/__init__ (and its client
    # dependency chain) while keeping submodule imports and package
    # resources (schema JSON) fully functional.
    sys.modules["trie"] = package


def tokenizer_manager_module() -> types.ModuleType:
    bootstrap()
    return importlib.import_module("trie.tokenizer_manager")


def types_module() -> types.ModuleType:
    bootstrap()
    return importlib.import_module("trie.types")


def validation_module() -> types.ModuleType:
    bootstrap()
    return importlib.import_module("trie.validation")


def engine_files_module() -> types.ModuleType:
    bootstrap()
    return importlib.import_module("trie.engine_files")
