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

The planner (`python -m evolving_infra.planner control`) runs in three phases:

- **Cold start.** From the first completed request it computes, every minute, the share of input
  tokens served from cache. When two consecutive minutes differ by at most 0.3 percentage points, the
  cold start is over. A minute with no cache hit at all does not count, and learning starts anyway
  30 minutes after the first completed request.
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

4. Start the planner with its [arguments](#configuration):

   ```bash
   pip install -e ".[planner]"
   python -m evolving_infra.planner control --config planner.json
   ```

   The planner can start before traffic: the cold start begins with the first completed request.
   It runs until it is stopped.

The planner writes its decisions to `<run_dir>/role-decisions.jsonl` and the learning start to
`<run_dir>/learning-start.json`. It stops, with `<run_dir>/PLANNER_FAILED`, only when the router's
workers differ from `--workers`, the roles change before the learning-end fit, or that fit fails.
Any other error, such as an unreachable router or engine, a failed clock probe or a failed role update,
skips at most one decision and is logged; roles changed by another client after the fit are adopted.

## Configuration

Every launch parameter is a flag with a default, as in SGLang's server arguments.
`--config FILE` reads a JSON object of the same names (with underscores), and a flag given on the
command line overrides the file. `python -m evolving_infra.planner control --help` lists every flag.
The planner logs the resolved arguments at startup and saves them to `<run_dir>/planner_args.json`,
which the learning-end fit reads too.

```bash
python -m evolving_infra.planner control --config planner.json --decision-period-secs 300
```

```json
{
  "router_url": "http://127.0.0.1:30900",
  "role_control_token_file": "/secure/role-token",
  "run_dir": "/path/to/run",
  "scratch_dir": "/path/to/scratch",
  "prefill_shape_prior": "glm52-gb300-tp4",
  "hosts": [{"index": 0, "ip": "192.0.2.1"}],
  "workers": [
    {"node": "node0", "idx": 0, "name": "w0", "host": 0, "ip": "192.0.2.1",
     "url": "http://192.0.2.1:30000", "trace_url": "http://192.0.2.1:18500", "initial_role": "prefill"},
    {"node": "node1", "idx": 1, "name": "w1", "host": 0, "ip": "192.0.2.1",
     "url": "http://192.0.2.1:31000", "trace_url": "http://192.0.2.1:18500", "initial_role": "decode"}
  ]
}
```

**Deployment** (all required, except the prior's checksum):

| Flag | Description |
|---|---|
| `--router-url` | The router, started with `--symmetric-pd-external-role-control`. |
| `--role-control-token-file` | The router's role-control token. Keep it outside the run directory. |
| `--run-dir` | Journals, the fit report and the resolved arguments. |
| `--scratch-dir` | Trace copies and snapshots (large; node-local is fine). |
| `--hosts` | Worker hosts, a JSON list of `{index, ip}`. |
| `--workers` | Workers, a JSON list as above. `node` must be `node<idx>`; `name` is the worker's trace subdirectory under `--root`; the router starts with the `initial_role`s. |
| `--prefill-shape-prior` | Prefill model shape for a worker whose own fit cannot separate the terms. Built in: `dsv4-flash-h200-tp4` (DeepSeek-V4-Flash, TP4, H200), `glm52-gb300-tp4` (GLM-5.2, TP4, GB300) and `kimi-k26-gb300-tp4` (Kimi-K2.6, TP4, GB300). Build one for another setup with `python -m evolving_infra.planner prior <report.json>`, or give `{"coefficients": [...]}`. |
| `--prefill-shape-prior-sha256` | Optional checksum that pins the prior. |

**Method**:

| Flag | Default | Description |
|---|---|---|
| `--learning-secs` | 1800 | Learning interval, from the end of the cold start to the fit. |
| `--decision-period-secs` | 600 | Time between two decisions. |
| `--slice-secs` | 60 | Slice length: decisions compare the splits slice by slice, the cold start compares consecutive slices, and the prefill model is fitted and the prefill workers re-measured on windows of this length. |
| `--switch-standard-errors` | 2 | A split is chosen only when its mean gain per slice exceeds this many standard errors. |
| `--post-switch-exclusion-secs` | 120 | Samples skipped after a switch has drained, so the next decision measures the new split. |
| `--transition-hold-max-secs` | 5 | Longest wait of a decision while requests of a worker's old role still run; a drain is always waited for. |
| `--observation-gap-max-secs` | 15 | A decision is skipped when two router reads within its interval are further apart. |
| `--disable-prefill-observation` | off | Keep the learning-interval packing and restore ratio instead of measuring them before each decision. |
| `--cold-start-stable-change` | 0.003 | The cold start is over when the cached share of two consecutive slices differs by at most this (0.3 percentage points). |
| `--cold-start-max-wait-secs` | 1800 | Learning starts at the latest this long after the first completed request. |
| `--router-poll-interval-secs` | 3 | Interval between two reads of the router. |
| `--restore-scrape-interval-secs` | 3 | Interval between two scrapes of the prefill workers' restored-token counters. |
| `--clock-probe-interval-secs` | 300 | Interval between two clock probes of every host. |
| `--clock-tolerance-secs` | 0.5 | Largest allowed drift of a host clock from its offset at start. |
| `--trace-port` | 18500 | Port of `serve-traces` on every host. |
| `--role-script` | none | Testing only: role changes instead of decisions, at the given seconds after the learning start (after the fit), e.g. `[[2400, {"node0": "decode"}]]`. |

The engines' TP size, which scales the restore counters, is read from each engine's
`/server_info`. The thresholds of the fit (70% of the learning interval trains, the rest measures
the error; the minimum number of windows), the 8-iteration decode groups, and the decode batch
bound of 95% of the KV capacity are fixed in the code.

**Environment variables** for internals that are not part of the method:

| Variable | Default |
|---|---|
| `EVOLVING_INFRA_PLANNER_ROLE_API_TIMEOUT_SECS` | 5 |
| `EVOLVING_INFRA_PLANNER_BACKEND_LOADS_TIMEOUT_SECS` | 5 |
| `EVOLVING_INFRA_PLANNER_SERVER_INFO_TIMEOUT_SECS` | 30 |
| `EVOLVING_INFRA_PLANNER_MODEL_INFO_TIMEOUT_SECS` | 5 |
| `EVOLVING_INFRA_PLANNER_METRICS_TIMEOUT_SECS` | 4 |
| `EVOLVING_INFRA_PLANNER_TRACE_LIST_TIMEOUT_SECS` | 30 |
| `EVOLVING_INFRA_PLANNER_TRACE_READ_TIMEOUT_SECS` | 120 |
| `EVOLVING_INFRA_PLANNER_CLOCK_PROBE_TIMEOUT_SECS` | 15 |
| `EVOLVING_INFRA_PLANNER_CLOCK_PROBE_EXCHANGES` | 8 |

## Limits

- Workers must be identical and symmetric.
- The models are fitted once, at the end of learning.
- The decision rule is an approximate capacity model, not a full scheduler simulation.
- Tested: scripted switches (2P2D → 1P3D → 2P2D on two H200 nodes). A switch triggered by the
  planner's own decision rule has not been run end to end yet.
