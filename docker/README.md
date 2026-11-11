# Runtime image

`docker/Dockerfile` builds one image with every component from this repository's source:
DecoupledEngine, the Rust router, the frontend / indexer / planner, and the Trie replay client.

```bash
docker build -f docker/Dockerfile -t evolving-infra .    # from the repository root
```

The same file builds on x86_64 (Modal H200 and B300, `examples/modal/app.py`) and on aarch64
(GB300). On clusters without a Docker daemon, build it as an enroot or Singularity image from the
same steps.

## Prebuilt image

`docker.io/weili0234/evolving-infra:v0.1.0` is one tag for both architectures
(index `sha256:e0038c04e3fc8f21042ac1eaa05d911b4d24c6b529d5747b2aaff39d424395bf`); `docker pull`
picks the one of the host.

| Architecture | Image | Source |
|---|---|---|
| x86_64 (H200, B300) | `docker.io/weili0234/evolving-infra@sha256:55a40438be30bb3ccf3d5b9211cd9d110c8b355c56f8f8d59ae105a55563fe0a` | `67b4761` |
| aarch64 (GB300) | `docker.io/weili0234/evolving-infra@sha256:f29e1a3f4189fee9ca612332cb969c43e9e8a3c8e5ecf8a8807d4304dd3384bb` | `ce30732` |

`docker/build_and_push.sh` builds and pushes one architecture from a committed revision
(`git archive`, so local files never enter the image) and prints the digest; build each
architecture on a host of that architecture. To publish both under one tag:

```bash
docker buildx imagetools create -t docker.io/weili0234/evolving-infra:<tag> \
  docker.io/weili0234/evolving-infra:<sha12>-amd64 docker.io/weili0234/evolving-infra:<sha12>-arm64
```

## Base image

The image starts from `docker.io/lmsysorg/sglang:v0.5.20-cu130`, the base the benchmarks run on. That base
already has the hard parts at matching versions: CUDA 13 and its toolkit, PyTorch, `sglang-kernel`,
FlashInfer, DeepGEMM, TileLang, the Mooncake wheel, NCCL / RDMA user space and a Rust toolchain.
Starting from it keeps the build identical to what was measured. Override the base with
`--build-arg BASE_IMAGE=...`.

## Layers

| Step | What it does | Why |
|---|---|---|
| apt | RDMA tools, protoc, Mooncake build dependencies | router build (protobuf) and the Mooncake rebuild |
| engine | removes the base's `sglang` / `sgl-router`, then `pip install -e evolving_infra/engine/` | pip keeps every dependency the base already satisfies and adds only what the engine's pins need beyond it |
| FlashInfer | `flashinfer-cubin` (and `flashinfer-jit-cache` when present) at the resolved FlashInfer version | prebuilt kernels instead of JIT on first use |
| Mooncake | rebuilds Mooncake `v0.3.13` with `patches/mooncake-rdma-polling-only.patch` and swaps its `engine.so` into the wheel | see below |
| router | `cargo build --release --locked --bin sgl-router` in `evolving_infra/router/core`, installed as `/usr/local/bin/sgl-router` | toolchain from `evolving_infra/router/core/rust-toolchain.toml`, dependencies from its `Cargo.lock` |
| `/opt/indexer-venv` | `ai-dynamo==1.4.0`, then the root package with `[frontend,indexer,planner]` and `fastapi<0.142` | the indexer supervisor starts the pinned `python -m dynamo.indexer` from this venv; FastAPI 0.142 enables OpenTelemetry export at startup and the frontend exits |
| `/opt/trie-venv` | `benchmarks/trie`, `openai==2.30.0`, `httpx==0.28.1`, and the root package (router client SDK) | Trie pins `transformers==4.57.6`, which must not meet the engine's; the client SDK serves `benchmarks/replay` |

The engine runs from the system Python; the frontend, indexer supervisor and planner from
`/opt/indexer-venv/bin/python`; the replay from `/opt/trie-venv/bin/python`.

## Mooncake patch

The patch adds `MC_RDMA_POLLING_ONLY=1`: completion queues are busy-polled, no completion
channels are created, and a failed epoll registration of the device's async fd is a warning
instead of an error. Sandboxes such as gVisor (Modal) hand back unusable completion-channel fds and
refuse epoll on RDMA device fds. Without the variable the patched engine behaves like upstream.
The rebuilt engine has no Mooncake store, so the wheel's `mooncake.store` import is disabled.

## Environment

The image sets only the values every deployment needs (`SGLANG_DISAGGREGATION_*_TIMEOUT=3600`,
`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`, `MOONCAKE_PROTOCOL=rdma`, and
`DECOUPLED_ENGINE_DIR` for Trie's engine-file lookups). On gVisor also set
`MC_RDMA_POLLING_ONLY=1` and `WITH_NVIDIA_PEERMEM=0`.

The engine keeps SGLang's `SGLANG_*` environment variables and `sglang:` Prometheus metric names
so the router, planner and replay checks read the upstream names.
