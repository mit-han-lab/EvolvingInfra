"""When the cold start is over: the share of input tokens served from cache stops changing.

From the first observation with completed requests on, the router's completed-usage counters are cut into one-minute
slices at the controller's observations. A slice's cached share is 1 - uncached / input tokens of the requests that
completed in it. The cold start is over at the end of the first slice whose share differs from the previous slice's by
at most STABLE_CHANGE (0.3 percentage points). A slice without a completed request, or without any cached token, does
not count and breaks the comparison: a share that stays at 0 % is the cold start itself (in a GLM-5.2 Claude Code run
on nine GB300 nodes the first two minutes both served 0 % from cache, and the share settled at the end of minute 12,
0.19 points from minute 11). Learning starts anyway MAX_WAIT_S after the first completed request, so a workload whose
share never settles is still planned for. A router restart (the counters start again from zero) restarts the detection.
"""

SLICE_S = 60.
STABLE_CHANGE = .003
MAX_WAIT_S = 1800.


class ColdStart:
    """Feed every router observation to `observe`. `done` is None while the cold start lasts, then a dict with the
    `reason` ('cache_share_stable' or 'max_wait') and its evidence."""

    def __init__(self):
        self.done = None
        self.restart()

    def restart(self):
        self.first_at = self.boundary_at = self.boundary = self.previous = None

    def observe(self, at, completed):
        """`at`: monotonic time of the observation; `completed`: its completed-usage counters (or None).

        Returns the one-minute slice this observation closes ({start, end, requests, cached_share, change}), for the
        journal, else None. Sets `done` when the cold start is over; later observations change nothing.
        """
        if self.done is not None or not completed:
            return None
        if self.boundary is not None and completed["requests"] < self.boundary["requests"]:
            self.restart()  # the router restarted: its counters start again
        if self.first_at is None:
            if completed["requests"] > 0:
                self.first_at = self.boundary_at = at
                self.boundary = completed
            return None
        if at - self.first_at >= MAX_WAIT_S:
            self.done = dict(reason="max_wait", waited_s=at - self.first_at, max_wait_s=MAX_WAIT_S)
            return None
        if at - self.boundary_at < SLICE_S:
            return None
        requests = completed["requests"] - self.boundary["requests"]
        tokens = completed["input_tokens"] - self.boundary["input_tokens"]
        uncached = completed["uncached_tokens"] - self.boundary["uncached_tokens"]
        share = 1 - uncached / tokens if requests > 0 and tokens > 0 and 0 <= uncached < tokens else None
        change = None if share is None or self.previous is None else share - self.previous
        record = dict(start=self.boundary_at, end=at, requests=requests, cached_share=share, change=change)
        self.boundary_at, self.boundary, self.previous = at, completed, share
        if change is not None and abs(change) <= STABLE_CHANGE:
            self.done = dict(reason="cache_share_stable", waited_s=at - self.first_at, cached_share=share,
                             change=change, stable_change=STABLE_CHANGE)
        return record
