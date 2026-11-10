# Planner

`evolving_infra.planner` adapts the prefill/decode split of a fixed set of
symmetric workers while the service runs. It never restarts an engine: a role
change is one Router role assignment.

```bash
pip install -e ".[planner]"          # numpy, scipy
python -m evolving_infra.planner --help
```

| Module | Purpose |
|---|---|
| `calibration/` | Step-time fits per worker (P: steps, new tokens, causal attention pairs, restored tokens; D: step overhead, batch size, active-expert fraction), capacity models and the P/D decision rule |
| `loop.py` | Live controller (`control`), learning-end fit (`fit`), per-host trace server (`serve-traces`) |
| `priors.py` | Predeclared prefill step-model shapes, used only when a worker's own four-term fit is not identifiable |
| `pd_ratio.py` | Explicit calibrate/recommend API for one homogeneous configuration (no I/O) |
| `node_plan.py` | Fixed per-node P/D plan from a completed learning capture (`node-plan`) |

How the planner decides, how to run it, and its configuration: [docs/dynamic-pd.md](../docs/dynamic-pd.md).
