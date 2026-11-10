"""Holds before a decision while an earlier role change is still visible."""
from evolving_infra.planner.control.transition import HoldGuard, adjusted_loads, residual_drains


def test_transition_guard_bounds_holds_but_never_overrides_drains():
    now = [0.]
    guard, rows = HoldGuard(max_wait_s=120, clock=lambda: now[0]), []
    assert guard.hold("x", True, {}, rows.append)
    now[0] = 121
    assert not guard.hold("x", True, {}, rows.append) and rows[-1]["event"] == "transition_guard_max_wait"
    assert guard.hold("d", True, {}, rows.append, overridable=False)
    now[0] = 1000
    assert guard.hold("d", True, {}, rows.append, overridable=False)
    loads = adjusted_loads(dict(workers={"u": dict(active_prefill_reqs=3, unknown_prefill_reqs=2)}))
    assert loads["workers"]["u"]["active_prefill_reqs"] == 1
    assert residual_drains({"u": dict(draining_age_s=1)})
