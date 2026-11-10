"""Planner: adapts the prefill/decode split of a fixed set of symmetric workers while the service runs.

The packages follow the data from the engines to a role change:

- ``observability``: the per-host trace server and the planner's copies of its traces, the host clock audit, the
  P workers' restored-token counters, and the frozen snapshot of the learning interval;
- ``perf_model``: per-worker iteration-time models fitted once at the end of learning, the predeclared prefill
  shapes, and the P workers' restore ratio and packing re-observed before every decision;
- ``policy``: the completed workload of an interval, the capacity of a role map, and the switch rule;
- ``managers``: the main loop, the decision state, the cold-start detector and the transition holds.

``planner_args`` holds every launch parameter with its default, ``envs`` the environment variables for internals,
``router_client`` and ``engine_client`` the HTTP endpoints, and ``utils`` the shared helpers.

Run ``python -m evolving_infra.planner --help``. Importing this package loads no numerical library.
"""
