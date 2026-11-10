# Planner

`evolving_infra.planner` adapts the prefill/decode split of a fixed set of
symmetric workers while the service runs. It never restarts an engine: a role
change is one Router role assignment.

```bash
pip install -e ".[planner]"          # numpy, scipy
python -m evolving_infra.planner --help
```

The modules follow the data from the engines to a role change:

| Module | Purpose |
|---|---|
| `collect/` | Per-host trace server and the controller's copies of its traces (`trace_server.py`), host clock audit (`clock.py`), P workers' restored-token counters (`restore_counters.py`), frozen snapshot of an interval (`snapshot.py`) |
| `fit/` | Step loading and the least-squares step fit (`steps.py`), closed 60-s P windows (`windows.py`), per-worker P and D models (`models.py`), the learning-end fit (`learning.py`, command `fit`), the current P workers' restore ratio and packing before each decision (`current_p.py`) |
| `decide/` | Completed workload, decode population and one-minute slices (`workload.py`), capacity of a role map (`capacity.py`), the decision rule (`rule.py`) |
| `control/` | When the cold start is over (`cold_start.py`); controller state: propose, apply, adopt (`controller.py`); holds while an earlier switch is still visible (`transition.py`); the live control loop (`loop.py`, command `control`) |
| `api.py` | Router role API and engine `/server_info`, `/model_info` |
| `config.py` | Defaults and config checks |
| `priors.py` | Predeclared prefill step-model shapes, used only when a worker's own fit is not identifiable (command `prior`) |
| `common.py` | JSON files, HTTP without proxies, number checks |

How the planner decides, how to run it, and its configuration: [docs/dynamic-pd.md](../docs/dynamic-pd.md).
