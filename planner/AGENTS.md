# planner

The online P/D planner. [`README.md`](README.md) has the models and the decision rule;
[`docs/dynamic-pd.md`](../docs/dynamic-pd.md) has every launch parameter.

- **Every launch parameter is a field of `PlannerArgs`** (`planner_args.py`) with a default
  and a help text; the flag name comes from the field name. Add the row to the tables in
  `docs/dynamic-pd.md`: every flag is documented there with its default. A renamed or removed
  key goes in `RENAMED` or `REMOVED`, so an old config fails with the new name.
- **The planner changes roles only through the router's role API**, as one compare-and-set
  update, and never restarts an engine. It does not retry an update whose answer was lost;
  the next poll adopts whatever the router holds.
- **The fit runs once, as a subprocess** (`fit --config <run_dir>/planner_args.json`), at the
  end of learning. The control loop reads only its `calibration/report.json`.
- **It stops only when it cannot continue**: the router's workers differ from the arguments,
  the roles changed before the fit, or the fit failed. Every other error is journaled in
  `role-decisions.jsonl` and skips at most one decision. Keep that split when adding a check.
- **Timing is part of the behavior.** Learning starts when the cold start ends
  (`managers/cold_start.py`), so the first decision comes at cold start plus
  `--learning-secs`. When a result is compared with another run, compare the switch times in
  `role-decisions.jsonl` first.
- **Layout follows the data**: `observability/` (traces, clocks, counters) to `perf_model/`
  (fits) to `policy/` (throughput of a split, the switch rule) to `managers/` (the loop).
  A module imports only from stages before it.

Tests: `python -m pytest -q tests/planner` (a fake router and a virtual clock; no GPU). A
change to the rule or the fit is also run once with
`examples/gb300/submit.py`, which starts the planner.
