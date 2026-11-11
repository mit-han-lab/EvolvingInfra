# DecoupledEngine

The serving engine of evolving-infra: SGLang's runtime (`srt`) with symmetric
prefill/decode workers, late decode binding, pull-based and peer KV transfer, a
unified host KV cache with session retention, and DeepSeek-V4 attention changes.

## Install and run

```bash
pip install -e evolving_infra/engine               # Python >= 3.10, CUDA 13 GPU host
decoupled-engine --model-path <model> [options]   # = python -m decoupled_engine.launch_server
```

Dependency pins are those of the SGLang revision the engine derives from. The
Rust extension crates (native gRPC server, Rust tokenizer server) are not part
of this package; the default HTTP server does not use them.

## Relation to SGLang

The engine is SGLang's `python/sglang` at upstream commit
[`9a489f8d2f`](https://github.com/sgl-project/sglang/commit/9a489f8d2fc376fbe5594393af367fd99ada7aac),
renamed to `decoupled_engine` and substantially modified. Each modified upstream file says
so in its header ("Adapted from SGLang ..."); files without that line are new or differ
only by the rename. [CHANGES.md](CHANGES.md) lists the 85 new files and the 145 modified
ones. The main changes:

- **Symmetric P/D workers** (`srt/disaggregation/symmetric/`): any worker can run the
  prefill or the decode leg, and KV moves to the decode worker chunk by chunk during prefill.
- **Late decode choice** (`srt/disaggregation/late_d*.py`): the decode worker is bound at
  the first prefill chunk instead of at admission.
- **Pull-based and peer KV transfer** (`srt/disaggregation/`): the decode side pulls KV,
  and a prefill worker can fetch a host-resident prefix from a peer instead of recomputing it.
- **Host KV tier** (`srt/mem_cache/`): write-through host cache with per-architecture
  dedup and load-back, and branch-level session retention for FULL-only caches.
- **Router integration**: prefill-state push, load reports and KV events for the router
  and its KV directory.
- **Model support**: DeepSeek-V4 attention and memory layout, plus speculative decoding and
  multimodal fixes for the tested models.

## Naming

The Python package is `decoupled_engine` (SGLang's `sglang`, with `srt/` and
`kernels/` kept at the same level). Environment variables keep the
`SGLANG_` prefix and Prometheus metrics keep the `sglang:` prefix, so router,
planner and dashboards read the same names as before.
