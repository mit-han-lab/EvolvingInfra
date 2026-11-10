"""Planner: adapts the prefill/decode split of a fixed set of symmetric workers while the service runs.

The modules follow the data from the engines to a role change:

- ``collect``: the per-host trace server and the controller's copies of its traces, the host clock audit, the
  P workers' restored-token counters, and the frozen snapshot of a learning interval;
- ``fit``: per-worker step-time models fitted once at the end of learning, and the P workers' restore ratio and
  packing re-observed before every decision;
- ``decide``: the completed workload of an interval, the capacity of a role map, and the decision rule;
- ``control``: the controller state, the transition holds, and the live control loop;
- ``api`` (router and engine endpoints), ``config``, ``priors`` and ``common`` (shared helpers).

Run ``python -m evolving_infra.planner --help``. Importing this package loads no numerical library.
"""
