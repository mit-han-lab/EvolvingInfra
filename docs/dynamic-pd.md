# Dynamic P/D

Agent workloads change between prefill-heavy phases (long new context after a tool call) and
decode-heavy ones. Evolving Infra changes how many workers serve prefill (P) and how many serve
decode (D) while the service runs. A switch is one call to the router; no engine restarts, and no
request is dropped.

## How a switch works

```text
 engines ── step traces ──▶ planner ── GET /symmetric_pd/roles ──▶ router
                              │      ◀── roles, epoch, revision, completed workload ──
                              └──── PUT /symmetric_pd/roles (new P/D weights, compare-and-set) ──▶ router
```

1. Every worker has a prefill weight and a decode weight in the router (`1,0` is P, `0,1` is D).
2. The planner sends new weights with `PUT /symmetric_pd/roles`, along with the epoch and revision
   it read. The router rejects the update if the roles changed in between.
3. Later requests are placed with the new weights. A worker moving from D to P stops taking new
   decode legs, finishes its running ones for up to `--symmetric-pd-role-drain-timeout-secs`, then
   aborts the rest; the frontend resumes those on other D workers.
4. A worker moving from P to D keeps its cache. With `--symmetric-pd-pull-from-any-holder`, new P
   workers can still pull the prefixes it holds.

## The planner's decision rule

The planner (`python -m evolving_infra.planner control`) runs in two phases:

- **Learning** (default 30 minutes). It reads each engine's scheduler step traces and, at the
  end, fits one step-time model per worker: prefill time from new tokens, attention work and
  restored tokens; decode time from batch size and active experts.
- **Control** (every 10 minutes by default). It scores every other P count on the recent
  workload, one 60-second slice at a time, and switches only when the predicted gain is larger than two
  standard errors. After a switch it waits before scoring again, so it measures the new split.

A rejected or failed update is logged and not retried.

## Run it

1. Start each engine with `--sched-trace-dir <root>/<worker name>`.
2. On every host, serve the traces and the host clock (port 18500):

   ```bash
   python -m evolving_infra.planner serve-traces --root <root>
   ```

3. Start the router with role control and a token file:

   ```bash
   head -c 32 /dev/urandom | base64 > /secure/role-token
   EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE=/secure/role-token EVOLVING_INFRA_PREPARED_CONFIG=prepared.json \
     sgl-router ... --symmetric-pd-external-role-control --symmetric-pd-role-drain-timeout-secs 30
   ```

   Role control needs the prepared path: the router counts completed work from the frontend's
   reports.

4. Start the planner with a [configuration](#configuration):

   ```bash
   pip install -e ".[planner]"
   python -m evolving_infra.planner control controller.json
   ```

   Learning starts when the caller writes `<run_dir>/replay-timing.json` with `start_wall`
   (epoch seconds) and `duration_s`.

The planner writes its decisions to `<run_dir>/role-decisions.jsonl` and, on failure,
`<run_dir>/PLANNER_FAILED`.

## Configuration

```json
{
  "router_url": "http://127.0.0.1:30900",
  "token_file": "/secure/role-token",
  "run_dir": "/path/to/run",
  "scratch": "/path/to/scratch",
  "hosts": [{"index": 0, "ip": "192.0.2.1"}],
  "workers": [
    {"node": "node0", "idx": 0, "name": "w0", "host": 0, "ip": "192.0.2.1",
     "url": "http://192.0.2.1:30000", "trace_url": "http://192.0.2.1:18500", "initial_role": "prefill"},
    {"node": "node1", "idx": 1, "name": "w1", "host": 0, "ip": "192.0.2.1",
     "url": "http://192.0.2.1:31000", "trace_url": "http://192.0.2.1:18500", "initial_role": "decode"}
  ]
}
```

| Key | Description |
|---|---|
| `workers[].node` | Must be `node<idx>`. |
| `workers[].name` | The worker's trace subdirectory under `--root`. |
| `learning_s`, `period_s` | Learning window and decision period (defaults 1800 and 600 seconds). |
| `restore_counter_replicas` | The engine's TP size. |
| `prefill_shape_prior` | Fallback prefill model shape when a worker's own fit is not identifiable. The built-in `dsv4-flash-h200-tp4` was measured on DeepSeek-V4-Flash TP4 on H200; build one for another setup with `python -m evolving_infra.planner prior <report.json>`. |
| `role_script` | Testing only: timed role changes instead of decisions, e.g. `[[2400, {"node0": "decode"}]]`. |

Every other tunable has a default in `evolving_infra.planner.loop.DEFAULTS`; any key in the
file overrides it. Keep the token file outside `run_dir`.

## Limits

- Workers must be identical and symmetric.
- The models are fitted once, at the end of learning.
- The decision rule is an approximate capacity model, not a full scheduler simulation.
- Tested: scripted switches (2P2D → 1P3D → 2P2D on two H200 nodes). A switch triggered by the
  planner's own decision rule has not been run end to end yet.
