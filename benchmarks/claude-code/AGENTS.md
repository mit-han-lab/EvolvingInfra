# benchmarks/claude-code

Read [`README.md`](README.md) first. Rules for running and changing this workload:

- **Prompts are block ids, not text.** Nothing in a row can be read or re-tokenized. Do not
  write code that needs the prompt text.
- **`prefix_block_size` equals the engine's `page_size`** (64 for GLM-5.2). Another value
  changes which prefixes are shared.
- **One replay client process**, `cache_salt_mode=global`, `replay_context_mode=actual_output`,
  `replay_output_mode=model`, `seed=0`. `examples/gb300/submit.py` is the reference command;
  change a setting only when the comparison is about that setting.
- **The portable file needs `tokenizer.json`** in the model directory. A model without one
  (Kimi-K2.6) cannot replay it.
- **Compare per P/D split.** The planner moves 6P3D to 3P6D and 4P5D in the paper's run;
  the whole-window number depends on when it switched
  ([`examples/gb300`](../../examples/gb300/README.md#results)).
