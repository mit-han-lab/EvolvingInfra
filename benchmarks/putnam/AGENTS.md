# benchmarks/putnam

Read [`README.md`](README.md) first. Rules for running and changing this workload:

- **16 client processes, one shard each.** `shards.json` fixes which tasks a process reads;
  each process gets the total concurrency divided by 16. One process at the full concurrency
  is a different experiment.
- **`examples/gb300/submit.py` runs one client process**, which is the Claude Code setup. For
  Putnam, start the stack with the `serve` skill and run the 16 replay processes on the host
  that runs the router.
- **Two runs of the same setting differ by up to 20 %** (7.75 and 9.15 requests/s at 256
  agents), and the planner settles on different splits. One run does not show a gain or
  a regression; repeat it and report every run.
- **Prompts are rendered at replay time.** A change to the chat template or tokenizer changes
  the prompt tokens and the prefix sharing; say so when comparing models.
