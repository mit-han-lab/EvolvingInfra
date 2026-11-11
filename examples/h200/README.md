# DeepSeek-V4-Flash on H200

DeepSeek-V4-Flash on 8-GPU H200 nodes: two TP4 workers per node, DSPARK speculative decoding,
host KV tier with DSV4 host dedup. Config: [`dsv4-flash.yaml`](dsv4-flash.yaml).

| | |
|---|---|
| Workers | 2 × TP4 per node; one node is 1P1D |
| Block size | 256 tokens |
| KV session retention | off: DSV4's cache tree is FULL+SWA (`retention: false`) |
| Workload | Claude Code traces for this tokenizer, `claude-code/cc062126-dsv4-flash.jsonl` |

## Run

On a host with the image and the model at `/models/DeepSeek-V4-Flash`:

```bash
evolving-infra serve --config examples/h200/dsv4-flash.yaml
```

On several nodes, the same command on each with `--hosts <node0>,<node1>,...` and
`--prefill-workers <n>`; add `--planner` to let the split change while serving.

With Slurm, serve, replay and score as one job:

```bash
python examples/slurm/submit.py --config examples/h200/dsv4-flash.yaml \
  --account <account> --partition <partition> --image <image>
```

[`examples/modal`](../modal/README.md) runs this setup on Modal H200 nodes and gives the
ranges a correct build lands in.
