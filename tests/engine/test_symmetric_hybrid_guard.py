# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Symmetric P/D refuses hybrid linear-attention pools at startup.

The guard is compiled from its source file (no CUDA imports), like
test_symmetric_registration.py does for the registration code.
"""
import ast
from pathlib import Path

import pytest

SRT = Path(__file__).resolve().parents[2] / "evolving_infra/engine/decoupled_engine/srt"
MIXIN = SRT / "disaggregation/symmetric/scheduler_mixin.py"


@pytest.fixture(scope="module")
def guard():
    tree = ast.parse(MIXIN.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "reject_unsupported_symmetric_pool")
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MIXIN), "exec"), ns)
    return ns["reject_unsupported_symmetric_pool"]


def test_init_calls_guard_first():
    tree = ast.parse(MIXIN.read_text())
    init = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "init_symmetric_disaggregation")
    first = [s for s in init.body if not isinstance(s, (ast.ImportFrom, ast.Import))][0]
    assert isinstance(first, ast.Expr) and first.value.func.id == "reject_unsupported_symmetric_pool"


def test_rejects_hybrid_linear_and_subclasses(guard):
    class HybridLinearKVPool:
        pass

    class UnifiedHybridLinearKVPool(HybridLinearKVPool):
        pass

    for cls in (HybridLinearKVPool, UnifiedHybridLinearKVPool):
        with pytest.raises(ValueError, match="hybrid linear-attention"):
            guard(cls())


@pytest.mark.parametrize("name", ["MHATokenToKVPool", "MLATokenToKVPool", "DeepSeekV4TokenToKVPool"])
def test_accepts_other_pools(guard, name):
    guard(type(name, (), {})())
