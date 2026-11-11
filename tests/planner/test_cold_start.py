"""When the cold start is over: the per-minute share of input tokens served from cache stops rising."""
import pytest

from evolving_infra.planner.managers.cold_start import ColdStart

# Cached share and completed requests of minutes 1-12 of a GLM-5.2 Claude Code run on nine GB300 nodes (90 agents).
CLUSTER_MINUTES = [(0.0, 59), (0.0, 20), (0.0941, 18), (0.6985, 113), (0.9296, 189), (0.9126, 126), (0.9723, 120),
                   (0.9582, 176), (0.9656, 137), (0.9580, 155), (0.9468, 129), (0.9449, 128)]
# Cached share of minutes 1-23 of the same experiment with 126 agents. Once warm (minute 7) the share moves by 0.3 to
# 4.4 points a minute; two consecutive minutes are first within 0.3 points of each other in minute 23.
CLUSTER_126_SHARES = [0.0, 0.0, 0.0, 0.0654, 0.7590, 0.8872, 0.9600, 0.9457, 0.9726, 0.9286, 0.9245, 0.9507, 0.9410,
                      0.9538, 0.9618, 0.9447, 0.9591, 0.9621, 0.9747, 0.9637, 0.9548, 0.9501, 0.9513]


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


def test_cluster_runs_end_the_cold_start_at_the_first_minute_whose_share_does_not_rise():
    cold = ColdStart()
    slices = feed(cold, CLUSTER_MINUTES)
    assert cold.done["reason"] == "cache_share_stable" and cold.done["waited_s"] == 6 * 60
    assert cold.done["cached_share"] == pytest.approx(0.9126) and cold.done["change"] == pytest.approx(-0.0170)
    # The share rises in minutes 4 and 5, while the cache fills.
    assert [round(s["change"] * 100, 2) for s in slices[3:5]] == [60.44, 23.11]
    # Minutes 1 and 2 served nothing from cache: an unchanged 0 % is the cold start, not its end.
    assert [s["cached_share"] for s in slices[:2]] == [None, None]
    assert len(slices) == 6
    cold = ColdStart()
    feed(cold, [(share, 100) for share in CLUSTER_126_SHARES])
    assert cold.done["waited_s"] == 8 * 60 and cold.done["cached_share"] == pytest.approx(0.9457)


def test_minute_without_completions_breaks_the_comparison():
    cold = ColdStart()
    feed(cold, [(0.90, 50), (0.0, 0), (0.905, 50)])
    assert cold.done is None  # 0.905 has no comparable previous minute
    cold = ColdStart()
    feed(cold, [(0.90, 50), (0.0, 0), (0.905, 50), (0.906, 50)])
    assert cold.done["reason"] == "cache_share_stable" and cold.done["waited_s"] == 4 * 60


def test_share_that_keeps_rising_starts_learning_after_the_maximum_wait():
    cold = ColdStart()
    feed(cold, [(0.50 + 0.01 * i, 30) for i in range(40)])
    assert cold.done["reason"] == "max_wait" and cold.done["waited_s"] == cold.max_wait_secs == 1800


def test_nothing_happens_before_the_first_completed_request_and_a_router_restart_starts_again():
    cold = ColdStart()
    assert cold.observe(1000., dict(requests=0, input_tokens=0, uncached_tokens=0)) is None
    assert cold.observe(1060., None) is None and cold.first_at is None
    feed(cold, [(0.5, 10), (0.7, 10)], start=1100.)
    assert cold.first_at == 1100.
    cold.observe(1300., dict(requests=3, input_tokens=10, uncached_tokens=5))  # counters went back: a new router
    assert cold.first_at == 1300. and cold.previous is None and cold.done is None
