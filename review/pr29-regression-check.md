We ran a regression check of this PR on 9 GB300 nodes. The workload is the Claude Code replay at concurrency 126 on GLM-5.2 (`examples/gb300/glm52.yaml`), scored over [1800 s, 5400 s).

**Result: no regression, and no measurable gain on this workload.**

### Setup

Base and PR use the same image, router binary and examples. The PR runs mount the four changed files (`pull/api.py`, `pull/importer.py`, `symmetric/import_queue.py`, `phase_timers.py`) over the image's copies. Those copies are byte-identical to the PR base, so the only difference is this diff. Each pair ran concurrently.

### Fixed roles: 3P6D, planner off, `SGLANG_SCHED_PHASE_TIMERS=1`

| | base | PR |
|---|---:|---:|
| requests/s | 5.41 | 5.35 |
| TTFT p50 | 7.06 s | 6.78 s |
| TTFT p95 | 28.0 s | 28.4 s |
| cache hit | 0.962 | 0.962 |
| dst_pages lookup per D node | ~86/min, mean ~10 ms, max 0.34–0.66 s | ~85/min, mean 10–14 ms, max 0.50–0.59 s |
| READ call | mean 111–123 ms | mean 106–133 ms |
| P forwards/min | 97–103 | 102–105 |

Both runs had no import failures or recompute fallbacks.

### Planner on

| | base | PR |
|---|---:|---:|
| requests/s | 5.50 | 5.96 |

Earlier single planner-on runs of the PR scored 5.30 and 4.63. Those drops track the planner's 3P→4P path, which sits near its 2-standard-error threshold. They also track clusters of 700k–950k-token uncached prefills landing while only 3 P workers serve. The same pattern appears in base runs, so it is benchmark variance, not this change.

### What this says about the change

On the old code, the socket thread spent about 1.4% of its time in `dst_pages` on each decode worker, with single waits of up to about 0.7 s. So the blocking this PR removes does happen on a real workload. Its effect is below what end-to-end throughput and TTFT can resolve here.

On this evidence we would hold the PR until a workload shows a measurable cost from the blocked socket thread. Examples are long CUDA waits during imports, such as P-side fetches during long prefill forwards, or cancellation-heavy traffic. The change adds lasting constraints: thread-safe `dst_pages`, and cancellation that drains queued lookups. If the blocking does become a problem, a CPU-side page lookup (the `cpu_rows` path) would remove the CUDA wait instead of moving it to another thread.
