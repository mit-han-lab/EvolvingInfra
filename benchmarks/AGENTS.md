# benchmarks

Trie (the replay client), the adapters that run it through the router's prepared path, the
run checks and the scoring. Each workload has its own guide; read it before running or
changing that workload: [`claude-code/`](claude-code/AGENTS.md), [`putnam/`](putnam/AGENTS.md),
[`osworld/`](osworld/AGENTS.md).

- **Trie has its own environment** (`/opt/trie-venv` in the image; it pins `transformers`).
  Run its tests from `benchmarks/trie` with `PYTHONPATH=$PWD/src`, so the tests import this
  checkout and not an installed copy.
- **The replay's flags are its interface.** `replay/run_trie.py` takes `stack`,
  `workload_path`, `concurrency`, `duration` and `output_dir`; every other setting defaults to
  the paper's (`PAPER` in that file). Pass another setting only when the experiment is about
  it, and report `replay_differs_from_preset` of `score.py` with the result.
- **A new setting that changes what is measured gets a default in `PAPER`** and a row in
  `replay/README.md`, so leaving it out still runs the paper's replay.
- **A result is a score on a stated window**: `replay/score.py <dir> --skip-s S --window-s W`
  over the lifecycle files. Report the window with every number. Two runs are compared on
  the same window and the same P/D split; the planner's switch times are in the first
  host's `planner/role-decisions.jsonl`.
- **Before blaming the system for a gap, compare the work and the stages**: uncached tokens
  and output tokens per request, time to first token, decode time per token, then per
  worker the decode step time and accepted draft length at equal batch size
  (the `debug-run` skill has the scripts).
- **A workload row is never edited by hand.** Converters and `trie/scripts/portable_blocks.py`
  write workloads; a file's sha256 identifies the run's input.
- **The same seed gives the same replay.** `warm_start=stationary` cuts each first trace at a
  point drawn from `seed`; compare systems with the same seed and the same number of client
  processes.

Tests: `PYTHONPATH=$PWD/src python -m pytest -q tests` in `benchmarks/trie`; the replay
adapters are exercised by a GPU run.
