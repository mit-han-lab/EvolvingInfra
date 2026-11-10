# Cleaned Claude Code fixed-start replay

`python scripts/convert_cc_fixed_start.py --input <cleaned-weka-directory> --tokenizer <GLM-tokenizer> --output <new-jsonl> --cut-seconds 3600`

This optional conversion retains the existing CC chain inference and completion dependencies. Its clock is the event DAG evaluated with the cleaned `think_time` (required <=120 seconds) and recorded `api_time`. Original timestamps remain provenance; their gaps do not restore waits removed by cleaning. Input requests must already have positive output lengths.

Each source event keeps its original index and dependencies. Requests starting at or after the cut retain their entire prompt and output budget. Earlier requests become zero-duration or residual control waits; a request spanning the cut still delays its dependent agents until its recorded remaining interval ends. Other waits retain only their remaining portion. Every event's new virtual phase equals `max(0, original phase - cut)`; conversion checks this invariant. Agents with only control events remain in the task. A root with no retained model request is excluded, and an empty eligible corpus is rejected.

The standalone converter audits every source root and publishes a validation receipt alongside the output. This is a separate conversion mode; the original full-corpus validator is not its validator. Real Trie parsing checks every retained root, retained payloads and dependencies remain exact, and eligibility is computed independently from a timing-only source graph. CC062126's complete-source independent audit additionally checks every retained event index and payload against the raw cleaned data.

Run this output with `warm_start=off`, `delay_scale=1`, no additional delay caps, `replay_once=False`, and `workload_order=natural`. Every admission, including later refills, uses the same cut suffix. Existing `replay_cycle_id` token synthesis gives each repeated instance different content, while all agents within that instance retain shared prefix blocks. No extra one-token prewarming is performed: the first retained request sends its full prompt, and its cold prefill cost is real.

This is a synthetic source-time suffix, not restoration of a target engine at its real one-hour mark. Source API durations locate the cut and remaining control waits; future requests execute with actual target service times. Timings, source inference, and task-selection bias must be disclosed in experiment reports.
