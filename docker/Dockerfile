# syntax=docker/dockerfile:1.7
# evolving-infra runtime image, built from this repository's source.
# Build from the repository root:  docker build -f docker/Dockerfile -t evolving-infra .
# Why this base and what each layer does: docker/README.md.

ARG BASE_IMAGE=docker.io/lmsysorg/sglang:v0.5.20-cu130
FROM ${BASE_IMAGE}

ARG MOONCAKE_VERSION=v0.3.13
ARG DYNAMO_VERSION=1.4.0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Runtime tools for RDMA and the router's protobuf build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      protobuf-compiler rdma-core ibverbs-utils perftest iproute2 pciutils \
      cmake ninja-build git curl ca-certificates pkg-config patchelf \
      libibverbs-dev libgoogle-glog-dev libgflags-dev libjsoncpp-dev libunwind-dev libnuma-dev \
      libboost-all-dev libssl-dev libyaml-cpp-dev libcurl4-openssl-dev liburing-dev libzstd-dev \
      libasio-dev libxxhash-dev libmsgpack-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- engine
# Replace the base image's SGLang and router with DecoupledEngine. pip keeps every dependency the base
# already satisfies and adds only what the engine's pins require beyond it.
# The base image's working directory is /sgl-workspace/sglang, which this step deletes; build from
# /opt/evolving-infra so pip never runs in a removed directory.
COPY engine /opt/evolving-infra/engine
WORKDIR /opt/evolving-infra
RUN pip uninstall -y sglang sgl-router sglang-router 2>/dev/null || true \
 && rm -rf /sgl-workspace/sglang \
 && PIP_CONSTRAINT= pip install -e /opt/evolving-infra/engine \
 && python3 -c 'import decoupled_engine' \
 && pip list 2>/dev/null | grep -i -E '^(decoupled-engine|sglang-kernel|flashinfer-python|mooncake-transfer-engine|torch) '

# FlashInfer's prebuilt kernels at the exact version the engine resolved.
RUN FI=$(pip show flashinfer_python | awk '/^Version/{print $2}') \
 && pip install --no-deps "flashinfer-cubin==${FI}" --index-url https://flashinfer.ai/whl \
 && (! pip show flashinfer-jit-cache >/dev/null 2>&1 \
     || pip install --no-deps "flashinfer-jit-cache==${FI}" --index-url https://flashinfer.ai/whl/cu130)

# Mooncake transfer engine rebuilt with the polling-only RDMA patch (opt-in via MC_RDMA_POLLING_ONLY=1).
# Its engine.so replaces the pip wheel's; the wheel's store module is disabled because the rebuilt
# engine is built without the store.
COPY docker/patches/mooncake-rdma-polling-only.patch /opt/mooncake/rdma-polling-only.patch
RUN git clone --depth 1 --branch "${MOONCAKE_VERSION}" https://github.com/kvcache-ai/Mooncake /opt/mooncake/src \
 && cd /opt/mooncake/src \
 && git submodule update --init --depth 1 extern/pybind11 \
 && git apply /opt/mooncake/rdma-polling-only.patch \
 && mkdir build && cd build \
 && cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DWITH_STORE=OFF \
      -DWITH_P2P_STORE=OFF -DWITH_EP=OFF -DWITH_STORE_RUST=OFF -DUSE_ETCD=OFF -DUSE_HTTP=OFF \
      -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF \
      -DPython3_EXECUTABLE="$(which python3)" \
      -DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs \
      -DCMAKE_SHARED_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs \
 && ninja engine \
 && MC=$(python3 -c 'import os, importlib.util as u; print(os.path.dirname(u.find_spec("mooncake").origin))') \
 && cp "$MC/engine.so" "$MC/engine.so.pip" \
 && cp "$(find /opt/mooncake/src/build -name 'engine*.so' | head -1)" "$MC/engine.so" \
 && sed -i 's/^    from mooncake.store import BufferPool$/    raise ImportError("mooncake.store disabled for the patched engine")/' "$MC/buffer_pool.py" \
 && grep -q 'disabled for the patched engine' "$MC/buffer_pool.py" \
 && ! (ldd "$MC/engine.so" | grep 'not found' | grep -v libcuda.so.1) \
 && cd / \
 && rm -rf /opt/mooncake/src/build

# ---------------------------------------------------------------- router
# The toolchain comes from router/core/rust-toolchain.toml; dependencies from its Cargo.lock.
COPY router/core /opt/evolving-infra/router/core
RUN if ! command -v rustup >/dev/null && [ ! -x /root/.cargo/bin/rustup ]; then \
      curl --proto '=https' --tlsv1.2 --retry 3 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain none; \
    fi \
 && export PATH=/root/.cargo/bin:$PATH \
 && cd /opt/evolving-infra/router/core \
 && rustup show active-toolchain \
 && cargo build --release --locked --bin sgl-router \
 && install -D -m 0755 target/release/sgl-router /usr/local/bin/sgl-router \
 && rm -rf target /root/.cargo/registry /root/.cargo/git \
 && sgl-router --version

# ---------------------------------------------------------------- frontend, indexer, planner
# Their own venv with the pinned Dynamo indexer (python -m dynamo.indexer). fastapi < 0.142: 0.142
# turns on OpenTelemetry export at startup and the frontend exits.
COPY pyproject.toml README.md LICENSE NOTICE /opt/evolving-infra/
COPY router/frontend /opt/evolving-infra/router/frontend
COPY router/indexer /opt/evolving-infra/router/indexer
COPY router/client /opt/evolving-infra/router/client
COPY planner /opt/evolving-infra/planner
COPY launch /opt/evolving-infra/launch
RUN python3 -m venv /opt/indexer-venv \
 && /opt/indexer-venv/bin/pip install -q "ai-dynamo==${DYNAMO_VERSION}" msgpack pyzmq \
 && /opt/indexer-venv/bin/pip install -q "/opt/evolving-infra[frontend,indexer,planner]" 'fastapi<0.142' \
 && /opt/indexer-venv/bin/python -c 'import dynamo.llm, evolving_infra.router.frontend, evolving_infra.router.indexer, evolving_infra.planner, evolving_infra.launch' \
 && ln -s /opt/indexer-venv/bin/evolving-infra /usr/local/bin/evolving-infra

# ---------------------------------------------------------------- replay client
# Trie pins transformers 4.57.6, so it gets a venv of its own; the router client SDK beside it serves
# the prepared-path adapter in benchmarks/replay.
COPY benchmarks /opt/evolving-infra/benchmarks
RUN python3 -m venv /opt/trie-venv \
 && /opt/trie-venv/bin/pip install -q /opt/evolving-infra/benchmarks/trie 'openai==2.30.0' 'httpx==0.28.1' \
 && /opt/trie-venv/bin/pip install -q "/opt/evolving-infra[client]" \
 && /opt/trie-venv/bin/python -c 'import trie.cli, evolving_infra.router.client'

ENV PYTHONUNBUFFERED=1 \
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 \
    MOONCAKE_PROTOCOL=rdma \
    DECOUPLED_ENGINE_DIR=/opt/evolving-infra/engine/decoupled_engine

WORKDIR /opt/evolving-infra
