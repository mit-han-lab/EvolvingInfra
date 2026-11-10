"""When the cold start is over: the per-minute share of input tokens served from cache stops changing."""
import pytest

from evolving_infra.planner.control.cold_start import MAX_WAIT_S, ColdStart

# Cached share and completed requests of minutes 1-12 of a GLM-5.2 Claude Code run on nine GB300 nodes (90 agents).
CLUSTER_MINUTES = [(0.0, 59), (0.0, 20), (0.0941, 18), (0.6985, 113), (0.9296, 189), (0.9126, 126), (0.9723, 120),
                   (0.9582, 176), (0.9656, 137), (0.9580, 155), (0.9468, 129), (0.9449, 128)]


def feed(cold, minutes, start=1000., per_minute_tokens=1_000_000):
    """Observe once at `start` (the first completed request) and at every minute end; returns the closed slices."""
    completed = dict(requests=1, input_tokens=0, uncached_tokens=0)
    cold.observe(start, dict(completed))
    slices = []
    for i, (share, requests) in enumerate(minutes, 1):
        completed = dict(requests=completed["requests"] + requests,
                         input_tokens=completed["input_tokens"] + per_minute_tokens,
                         uncached_tokens=completed["uncached_tokens"] + round((1 - share) * per_minute_tokens))
        slices.append(cold.observe(start + 60 * i, dict(completed)))
        if cold.done:
            break
    return slices


def test_cluster_run_settles_at_the_end_of_minute_12():
    cold = ColdStart()
    slices = feed(cold, CLUSTER_MINUTES)
    assert cold.done["reason"] == "cache_share_stable" and cold.done["waited_s"] == 12 * 60
    assert cold.done["cached_share"] == pytest.approx(0.9449) and abs(cold.done["change"]) <= 0.003
    # Minute 9 (0.73 points from minute 8) and minute 10 (0.76) are not stable at 0.3 points.
    assert [round(s["change"] * 100, 2) for s in slices[8:11]] == [0.74, -0.76, -1.12]
    # Minutes 1 and 2 served nothing from cache: an unchanged 0 % is the cold start, not its end.
    assert [s["cached_share"] for s in slices[:2]] == [None, None]
    assert len(slices) == 12


def test_minute_without_completions_breaks_the_comparison():
    cold = ColdStart()
    feed(cold, [(0.90, 50), (0.0, 0), (0.905, 50)])
    assert cold.done is None  # 0.905 has no comparable previous minute
    cold = ColdStart()
    feed(cold, [(0.90, 50), (0.0, 0), (0.905, 50), (0.906, 50)])
    assert cold.done["reason"] == "cache_share_stable" and cold.done["waited_s"] == 4 * 60


def test_unsettled_share_starts_learning_after_the_maximum_wait():
    cold = ColdStart()
    feed(cold, [(0.80 if i % 2 else 0.85, 30) for i in range(40)])
    assert cold.done["reason"] == "max_wait" and cold.done["waited_s"] == MAX_WAIT_S


def test_nothing_happens_before_the_first_completed_request_and_a_router_restart_starts_again():
    cold = ColdStart()
    assert cold.observe(1000., dict(requests=0, input_tokens=0, uncached_tokens=0)) is None
    assert cold.observe(1060., None) is None and cold.first_at is None
    feed(cold, [(0.5, 10), (0.7, 10)], start=1100.)
    assert cold.first_at == 1100.
    cold.observe(1300., dict(requests=3, input_tokens=10, uncached_tokens=5))  # counters went back: a new router
    assert cold.first_at == 1300. and cold.previous is None and cold.done is None
