# Planner

`evolving_infra.planner` adapts the prefill/decode split of a fixed set of
symmetric workers while the service runs. It never restarts an engine: a role
change is one Router role assignment.

```bash
pip install -e ".[planner]"          # numpy, scipy
python -m evolving_infra.planner control --help
```

How to run it, and every launch parameter: [docs/dynamic-pd.md](../docs/dynamic-pd.md).

## Code layout

The packages follow the data from the engines to a role change.

| Module | Purpose |
|---|---|
| `planner_args.py` | Every launch parameter with its default (`PlannerArgs`); flags, `--config` file, checks |
| `envs.py` | Environment variables for internals: HTTP timeouts, clock probe round trips |
| `router_client.py`, `engine_client.py` | Router role API; engine `/server_info` and `/model_info` |
| `observability/` | Per-host trace server (`trace_server.py`, command `serve-traces`) and the planner's copies of its traces (`trace_mirror.py`), host clock audit (`clock_sync.py`), prefill workers' restored-token counters (`restore_metrics.py`), frozen snapshot of the learning interval (`trace_snapshot.py`) |
| `perf_model/` | Iteration loading (`step_trace.py`) and the least-squares fit (`step_fit.py`), prefill windows (`prefill_windows.py`), per-worker prefill and decode models (`worker_model.py`), the learning-end fit (`calibration.py`, command `fit`), the prefill workers' packing and restore ratio before each decision (`prefill_observation.py`), predeclared prefill shapes (`priors.py`, command `prior`) |
| `policy/` | Completed workload, decode population and slices (`workload.py`), throughput of a role map (`capacity.py`), the switch rule (`split_policy.py`) |
| `managers/` | Main loop (`planner_manager.py`, command `control`), decision state: propose, apply, adopt (`split_controller.py`), end of the cold start (`cold_start.py`), holds while an earlier switch is still visible (`transition_guard.py`) |
| `utils.py` | JSON files, HTTP without proxies, number checks |
