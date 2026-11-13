# Runtime image

`docker/Dockerfile` builds one image with every component from this repository's source:
DecoupledEngine, the Rust router, the frontend / indexer / planner, and the Trie replay client.

```bash
docker build -f docker/Dockerfile -t evolving-infra .    # from the repository root
```

The same file builds on x86_64 (H200 and B300; `examples/modal/app.py` runs it on Modal H200
nodes) and on aarch64 (GB300). On clusters without a Docker daemon, build it as an enroot or
Singularity image from the same steps.

## Prebuilt image

`docker.io/weili0234/evolving-infra:v0.1.0` is one tag for both architectures
(index `sha256:9d7d128f9e258ef2fcff01988e5c4e41b6a1d5e5c0045b1c8fa55b2bd08809d3`); `docker pull`
picks the one of the host. A tag can be moved and a digest cannot, so a deployment that must
run the same image every time pins the digest of its architecture:

| Architecture | Image | Source |
|---|---|---|
| x86_64 (H200, B300) | `docker.io/weili0234/evolving-infra@sha256:55a40438be30bb3ccf3d5b9211cd9d110c8b355c56f8f8d59ae105a55563fe0a` | `67b4761` |
| aarch64 (GB300) | `docker.io/weili0234/evolving-infra@sha256:ef0ecc7b1aed1937a32ce4d968131418a8330540b74bc7b0868f22ee12f6241e` | `486dcd1` |

The x86_64 image predates the router flag names of the release (for example `--prefill-wait`,
`--global-queue`, `--copy-price`); until it is rebuilt, build it from this repository on x86_64 hosts
(`docker build` or `docker/build_and_push.sh`).

`docker/build_and_push.sh` builds and pushes one architecture from a committed revision
(`git archive`, so local files never enter the image) and prints the digest; build each
architecture on a host of that architecture. To publish both under one tag:

```bash
docker buildx imagetools create -t docker.io/weili0234/evolving-infra:<tag> \
  docker.io/weili0234/evolving-infra:<sha12>-amd64 docker.io/weili0234/evolving-infra:<sha12>-arm64
```

## Without the image

Install the same pieces on a host that has what the [base image](#base-image) provides (CUDA 13,
PyTorch, the kernels, NCCL / RDMA user space), plus `protoc` (`apt-get install protobuf-compiler`)
and `rustup` for the router. These are the steps of
`docker/Dockerfile`, run from the repository root:

```bash
# Engine, into the Python that `python3` runs. Where upstream SGLang is installed (as in the base
# image), remove it first: pip uninstall -y sglang sgl-router sglang-router
PIP_CONSTRAINT= pip install -e evolving_infra/engine
pip install mooncake-transfer-engine-cuda13==0.3.13      # KV transfer; the base image's wheel and version
pip install --no-deps "flashinfer-cubin==$(pip show flashinfer_python | awk '/^Version/{print $2}')" \
  --index-url https://flashinfer.ai/whl                  # FlashInfer's prebuilt kernels
# Where flashinfer-jit-cache is installed, its version must match FlashInfer's
pip show flashinfer-jit-cache >/dev/null 2>&1 && pip install --no-deps \
  "flashinfer-jit-cache==$(pip show flashinfer_python | awk '/^Version/{print $2}')" --index-url https://flashinfer.ai/whl/cu130

# Router; rustup takes the Rust version from rust-toolchain.toml
(cd evolving_infra/router/core && cargo build --release --locked --bin sgl-router)
install -m 0755 evolving_infra/router/core/target/release/sgl-router /usr/local/bin/   # or any directory on PATH

# Launcher, frontend, KV directory and planner, in a venv of their own (Python >= 3.11)
python3 -m venv .venv && .venv/bin/pip install -e ".[frontend,indexer,planner]"

# Replay client, in another venv: Trie pins transformers 4.57.6
python3 -m venv .venv-trie && .venv-trie/bin/pip install -e benchmarks/trie 'openai==2.30.0' 'httpx==0.28.1' tiktoken pillow -e ".[client]"
```

Start the stack with `.venv/bin/evolving-infra` and leave the venv inactive: the launcher runs the
frontend, KV directory and planner with its own interpreter and the engines with `python3`. Set
the variables the image sets ([Environment](#environment)) first:

```bash
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false \
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
```

The image also rebuilds Mooncake with a patch that only gVisor needs ([Mooncake patch](#mooncake-patch)).

## Base image

The image starts from `docker.io/lmsysorg/sglang:v0.5.20-cu130`, the base the benchmarks run on. That base
already has the hard parts at matching versions: CUDA 13 and its toolkit, PyTorch, `sglang-kernel`,
FlashInfer, DeepGEMM, TileLang, the Mooncake wheel (`mooncake-transfer-engine-cuda13==0.3.13`),
NCCL / RDMA user space and a Rust toolchain.
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
| `/opt/trie-venv` | `benchmarks/trie`, `openai==2.30.0`, `httpx==0.28.1`, `tiktoken`, `pillow`, and the root package (router client SDK) | Trie pins `transformers==4.57.6`, which must not meet the engine's; the client SDK serves `benchmarks/replay` |

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
