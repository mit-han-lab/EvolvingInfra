# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Skip a test module, with its reason, when the extra it exercises is not installed.

Install everything with ``pip install -e ".[client,frontend,indexer,planner,test]"``.
Tests that need the engine (``decoupled-engine``) skip themselves.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
FRONTEND = ("aiohttp", "fastapi", "pydantic", "orjson", "uvicorn", "httpx")

# (path prefix relative to tests/, minimum Python, modules, extra to install)
REQUIREMENTS = (
    ("frontend/", (3, 11), FRONTEND, "frontend,test"),
    ("client/test_prepared_late_d_plan.py", (3, 11), FRONTEND, "client,frontend,test"),
    ("client/", (3, 10), ("aiohttp",), "client"),
    ("indexer/test_query.py", (3, 10), ("aiohttp",), "indexer"),
    ("planner/", (3, 10), ("numpy", "scipy"), "planner"),
)


def missing_reason(path: Path):
    """Return why the module at ``path`` cannot run here, or None."""
    rel = path.resolve().relative_to(HERE).as_posix()
    for prefix, python, modules, extra in REQUIREMENTS:
        if not rel.startswith(prefix):
            continue
        if sys.version_info < python:
            return f"needs Python {'.'.join(map(str, python))}+"
        absent = [m for m in modules if importlib.util.find_spec(m) is None]
        if absent:
            return f"needs {', '.join(absent)} (pip install -e '.[{extra}]')"
        return None
    return None


class SkippedModule(pytest.Module):
    """A test module reported as skipped instead of failing to import."""

    reason = ""

    def collect(self):
        pytest.skip(self.reason)


def pytest_pycollect_makemodule(module_path, parent):
    reason = missing_reason(module_path)
    if reason is None:
        return None
    module = SkippedModule.from_parent(parent, path=module_path)
    module.reason = reason
    return module
