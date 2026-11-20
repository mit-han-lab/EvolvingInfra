# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""SGLANG_V41_EPLB_EARLY_FIRST_REBALANCE: the first EPLB rebalance comes after
interval // 4 forward passes (never fewer than the statistics window); every later
one keeps the full interval. Needs the engine and torch."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("decoupled_engine")

from decoupled_engine.srt.eplb import eplb_manager as eplb_manager_mod  # noqa: E402
from decoupled_engine.srt.eplb.eplb_manager import (  # noqa: E402
    EPLBManager,
    first_rebalance_num_iterations,
)


def test_deepseek_settings():
    # Prefill: interval 1,000, window 100 -> 250. Decode: 10,000 / 1,000 -> 2,500.
    assert first_rebalance_num_iterations(1000, 100) == 250
    assert first_rebalance_num_iterations(10000, 1000) == 2500


def test_never_below_the_window_nor_above_the_interval():
    assert first_rebalance_num_iterations(1000, 1000) == 1000
    assert first_rebalance_num_iterations(1000, 400) == 400
    assert first_rebalance_num_iterations(3, None) == 1
    assert first_rebalance_num_iterations(1000, -1) == 250


def _manager(interval, early):
    manager = object.__new__(EPLBManager)
    manager._rebalance_num_iterations = interval
    manager._early_first_rebalance_pending = early
    manager.rebalances = []
    manager.passes = 0

    def rebalance():
        manager.rebalances.append(manager.passes)
        yield

    manager.rebalance = rebalance
    return manager


def _drive(manager, num_passes):
    manager.reset_generator()
    for _ in range(num_passes):
        manager.passes += 1
        manager.on_forward_pass_end()


@pytest.fixture
def window_100():
    exec_cfg = SimpleNamespace(moe=SimpleNamespace(expert_distribution_recorder_buffer_size=100))
    with patch.object(eplb_manager_mod, "get_exec", return_value=exec_cfg):
        yield


def test_default_waits_a_full_interval(window_100):
    manager = _manager(interval=1000, early=False)
    _drive(manager, 2200)
    # A rebalance runs on the pass after `interval` counted passes.
    assert manager.rebalances == [1001, 2002]


def test_early_first_then_the_full_interval(window_100):
    manager = _manager(interval=1000, early=True)
    _drive(manager, 2200)
    assert manager.rebalances == [251, 1252]
    assert not manager._early_first_rebalance_pending


def test_a_reset_after_the_first_keeps_the_interval(window_100):
    manager = _manager(interval=1000, early=True)
    _drive(manager, 300)
    assert manager.rebalances == [251]
    manager.passes, manager.rebalances = 0, []
    _drive(manager, 1100)  # reset_generator: no second early trigger
    assert manager.rebalances == [1001]
