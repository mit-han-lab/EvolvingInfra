---
name: debug-run
description: Find why an Evolving Infra run failed to start, failed a request, or scored below a reference. Use when `evolving-infra serve` exits or never prints ready, a replay or request errors, or a benchmark is slower than expected.
---

# Debug a run

**Scope**: repository, read-only until the cause is named. Reads the run's logs and the code
of every component they point to.
**Inputs**: the work directory of `evolving-infra serve` (or `runs/<job id>/` of
`submit.py`), and what was expected.
**Returns**: the root cause with the log lines or numbers that show it, and the fix or the
decision that is the owner's. Do not change code to make the symptom go away before the
cause is named.

## 1. It did not start

`serve` stops when a process it started exits, and names every one that did:
`RuntimeError: engine-w1 exited with -9; see <dir>/logs/engine-w1.log`. Under `submit.py`
that line is in `<run>/serve.log`. Find the first error in each named log (engine logs are
long: weight loading and graph capture print progress lines):

```bash
grep -n -E "Error|Address already in use|Cannot assign" <dir>/logs/<name>.log | grep -v 'File "' | head -5
```

| Message | Cause | Fix |
|---|---|---|
| `Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=X. Raise ... above Y` | the weights fill the worker's GPUs; typical when a config written for one worker per host runs on one host, where each worker gets half the GPUs | the one-node config, or `--mem-fraction-static` above Y, or more GPUs per worker |
| `ports already in use on this host: ...` (from `serve`) | another process holds a port the layout needs | stop that process, or change the port in `launch/layout.py` |
| `Address already in use (addr='tcp://<ip>:<port>')`, `OwnerPlane failed to bind` | the port is in another engine's range, or a host service holds it | each engine takes two ports per GPU from its `--disaggregation-bootstrap-port` (in its launch line at the top of the log): if the port is in another engine's range the ranges overlap, else `ss -ltnp \| grep :<port>` names the service; fix in `launch/layout.py` |
| `Cannot assign requested address (addr='tcp://<ip>:...')` | the engine was given another host's address | compare `<ip>` with this host's `host N: <ip>` line in `serve.log`; they differ when a host's rank does not match its place in `--hosts`: let `serve` find the host by name, do not pass a scheduler's node index |
| router log: a worker is refused, or `engine must enable ...` | the engine and the router run different presets, or engine code read a raw argument | the same preset on both sides; in engine code `resolution_result` (`engine/AGENTS.md`) |

Nothing exited and no `ready` after 40 minutes: read the tail of each engine log; an engine
still loading weights or capturing graphs is slow, not stuck.

## 2. A request or the replay fails

| Message | Cause | Fix |
|---|---|---|
| `dynamic role switching with KV session retention requires /symmetric_pd/prepared lifecycle callbacks` | a plain OpenAI request while the planner is on | send through the prepared path (the replay does), or serve without `--planner` |
| replay: `The api_key client option must be set` | the client needs any key | `api_key=EMPTY` |
| `score.py`: `no lifecycle events` | the replay never ran | read `bench.log` |

For a failing request in a running stack, follow it through the logs in this order: `router`
(which P and D it chose), the P's engine log, the D's engine log.

## 3. Slower than the reference

A run can differ from the reference in more than one way: go through steps 1 to 4 for both
runs before concluding. `<run>/bench` is the directory with `trie-lifecycle-*.jsonl`; scripts
are in `skills/debug-run/scripts/` and `skills/run-benchmark/scripts/`.

1. **When the planner switched.** `switch_times.py <planner dir> <run>/bench` for both runs
   prints the switches on the replay clock. Pick, for each P/D split, a window in which both
   runs held that split (start 120 s after a switch) and compare `benchmarks/replay/score.py`
   on it. A gap only over the whole run is a gap in when the planner switched.
2. **Same work.** `request_breakdown.py <run>/bench --skip-s S --window-s W` for both. If
   `replay_context_modes`, `uncached_tokens_mean` or `output_tokens_mean` differ, the replay
   settings or the inputs differ. The benchmark replays with
   `replay_context_mode=actual_output` (an argument of the replay command; `bench.log` prints
   the arguments at its start).
3. **Which stage.** The same output gives `ttft_s_*` (router, prefill, transfer, decode
   admission) and `decode_ms_per_token_*` (decode).
4. **Decode.** `engine_batches.py decode <run>/bench --skip-s S --window-s W <logs>` for both,
   with the logs of the workers that were D in the window: worker `w<i>` logs to
   `engine-w<i>.log`, workers are numbered in the order of `--hosts`, the first
   `--prefill-workers` start as P, and `switch_times.py` shows later changes. Check the
   "window in log time" line it prints against the log's own timestamps. Compare rows with
   the same running requests: equal `ms/step` with a lower `accept len` means speculative
   decoding accepts less (the outputs differ: step 2's replay settings, the draft
   configuration); a higher `ms/step` means the decode step itself is slower (kernels, GPU).
5. **Prefill**, when time to first token differs and decode does not:
   `engine_batches.py prefill ...` on the P workers gives new tokens per second and queue
   lengths.
6. **Noise.** Repeat runs of one configuration differ by about 4 % in requests per second.

Report each step's numbers for both runs on the same window, and what differs.
