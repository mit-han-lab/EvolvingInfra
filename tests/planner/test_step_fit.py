# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Step traces and the step-time fit shared by the P and D models."""
import json

import numpy as np
import pytest

from evolving_infra.planner.perf_model.step_fit import fit_steps
from evolving_infra.planner.perf_model.step_trace import consecutive_step_windows, load_steps


def test_fit_recovers_coefficients_and_rejects_changed_regime():
    rng = np.random.default_rng(8)
    n = rng.integers(1000, 8193, 400)
    h = rng.integers(0, 130000, 400)
    x = np.column_stack([np.ones(400), n, n * h + n * (n + 1) / 2])
    expected = [0.03, 0.00002, 1e-10]
    y = x @ expected
    good = fit_steps(x, y)
    assert good["passed"]
    assert good["coefficients"] == pytest.approx(expected, rel=1e-6)
    y[280:] *= 2
    assert not fit_steps(x, y)["passed"]
    with pytest.raises(ValueError, match="not identifiable"):
        fit_steps(np.ones((100, 3)), np.ones(100))



def test_interlaunch_interval_belongs_to_previous_step_and_excludes_idle(tmp_path):
    rows = [
        dict(ts=100 + i, forward_iter=i, launch_mono=t, mode="extend", after_idle=idle)
        for i, t, idle in [
            (1, 10, False),
            (2, 10.5, False),
            (3, 40, True),
            (5, 40.2, False),
        ]
    ]
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"partial":')
    loaded = load_steps([path], 100, 110)
    assert [r["busy_interval_s"] for r in loaded] == [0.5, None, None, None]



def test_window_split_uses_launch_intervals_not_emission_time():
    rows = [dict(source='a', ts=102+i, dur_ms=2000, busy_interval_s=1,
                 forward_iter=i, mode='decode') for i in range(8)]
    w = consecutive_step_windows(rows, [[1] for _ in rows])[0]
    assert w['interval_start_ts'] == 100
    assert w['interval_end_ts'] == 108
    assert w['ts'] == 109
