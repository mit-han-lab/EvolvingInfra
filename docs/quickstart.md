# Quickstart

Run Evolving Infra on one 8-GPU node: two engine workers, the KV directory, a frontend and the
router. Worker `w0` (GPUs 0–3) starts as prefill (P), `w1` (GPUs 4–7) as decode (D).

## Prerequisites

- One 8-GPU H200-class node with RDMA NICs, CUDA 13 drivers, and a model the engine supports
  (here DeepSeek-V4-Flash; see the [models table](README.md#models-and-hardware)).
- The runtime image, which has every component installed:

  ```bash
  docker pull docker.io/weili0234/evolving-infra:v0.1.0
  docker tag docker.io/weili0234/evolving-infra:v0.1.0 evolving-infra
  git clone https://github.com/mit-han-lab/EvolvingInfra.git && cd EvolvingInfra
  docker run --gpus all --network host --ipc host --device /dev/infiniband \
    -v /models:/models -it evolving-infra bash
  ```

  To build the image yourself, run `docker build -f docker/Dockerfile -t evolving-infra .` in the
  repository instead of the pull; to install from source, see [Installation](#installation-from-source).

Inside the container, set the model path and the RDMA device of every GPU. Each TP rank looks up
its own GPU, so the map must list all eight:

```bash
export MODEL=/models/DeepSeek-V4-Flash-0731
export NICS='{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3","4":"mlx5_4","5":"mlx5_5","6":"mlx5_6","7":"mlx5_7"}'
```

## 1. Start the engines

Start `w0`:

```bash
SGLANG_HOST_IP=127.0.0.1 python3 -m decoupled_engine.launch_server \
  --model-path "$MODEL" --served-model-name deepseek-v4-flash --trust-remote-code \
  --tp-size 4 --base-gpu-id 0 --host 0.0.0.0 --port 30000 \
  --moe-runner-backend flashinfer_mxfp4 --enable-hierarchical-cache \
  --disaggregation-mode symmetric --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$NICS" --disaggregation-bootstrap-port 8998 \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557", "replay_endpoint": "tcp://*:5558"}' \
  --enable-metrics --enable-cache-report --incremental-streaming-output &
```

Start `w1` with the same command, changing `--base-gpu-id 4 --port 31000
--disaggregation-bootstrap-port 9998 --peer-kv-servicer-port 19100` and the KV event ports to
`6557` / `6558`. In symmetric mode each engine runs the paper's configuration, including a peer KV
servicer, so the two engines on one node need their own servicer ports.

Wait until both answer `curl -sf http://127.0.0.1:30000/health` (and `:31000`). The first start
compiles kernels and takes about 14 minutes; later starts take a few minutes.

Check that Mooncake found the RDMA devices: the engine log should say `Using RDMA transport`.
If it says `Using TCP transport (no HCA detected)`, fix `NICS` before going on.

## 2. Start the KV directory

The indexer runs Dynamo's KV indexer on the engines' KV events and tells the router which worker
holds which prefix blocks.

```bash
cat > indexer.json <<'EOF'
{"model_name": "deepseek-v4-flash", "block_size": 256, "eagle": false,
 "indexer_port": 8090, "query_port": 8091, "threads": 4,
 "workers": [
  {"instance_id": 20001, "events_endpoint": "tcp://127.0.0.1:5557", "replay_endpoint": "tcp://127.0.0.1:5558",
   "bridge_pub_port": 26100, "bridge_replay_port": 26101},
  {"instance_id": 20002, "events_endpoint": "tcp://127.0.0.1:6557", "replay_endpoint": "tcp://127.0.0.1:6558",
   "bridge_pub_port": 26102, "bridge_replay_port": 26103}]}
EOF
/opt/indexer-venv/bin/python -m evolving_infra.router.indexer --config indexer.json --output indexer &
```

It is ready when `indexer/ready.json` exists.

## 3. Start the frontend

The frontend tokenizes each request once and uploads it to the P and D the router picks.

```bash
cat > frontend.json <<EOF
{"node_id": "t0", "config_id": "dsv4", "model_path": "$MODEL",
 "tokenize_endpoint": "http://127.0.0.1:30000/tokenize",
 "targets": {"http://127.0.0.1:30000/generate": "dsv4", "http://127.0.0.1:31000/generate": "dsv4"},
 "page_size": 256, "context_length": 1048576, "hash_mode": "dynamo", "is_eagle": false, "cache_salt": null,
 "callback_targets": ["http://127.0.0.1:30900/symmetric_pd/events"]}
EOF
/opt/indexer-venv/bin/python -m evolving_infra.router.frontend --config frontend.json --host 0.0.0.0 --port 28100 &
```

## 4. Start the router

```bash
cat > prepared.json <<EOF
{"callback_url": "http://127.0.0.1:30900/symmetric_pd/events", "max_inflight": 4096,
 "contracts": {"dsv4": {"page_size": 256, "is_eagle": false, "model_path": "$MODEL", "cache_namespace": null}}}
EOF
EVOLVING_INFRA_PREPARED_CONFIG=prepared.json sgl-router --host 0.0.0.0 --port 30900 \
  --model-id deepseek-v4-flash --tokenizer-path "$MODEL/tokenizer.json" \
  --policy symmetric_pd \
  --symmetric-pd-worker http://127.0.0.1:30000,8998,1,0 \
  --symmetric-pd-worker http://127.0.0.1:31000,9998,0,1 \
  --kv-indexer-url "$(python3 -c 'import json; print(json.load(open("indexer/ready.json"))["query_url"])')" \
  --kv-worker-map '[{"id": 20001, "url": "http://127.0.0.1:30000", "servicer": "127.0.0.1:18100"}, {"id": 20002, "url": "http://127.0.0.1:31000", "servicer": "127.0.0.1:19100"}]' &
until curl -sf http://127.0.0.1:30900/readyz; do sleep 2; done
```

`--symmetric-pd-worker URL,BOOTSTRAP_PORT,P_WEIGHT,D_WEIGHT` sets each worker's starting role:
`1,0` is prefill only, `0,1` decode only.

## 5. Send a request

```bash
curl -s http://127.0.0.1:30900/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}'
```

The router runs the prefill on `w0`, moves the KV to `w1` over RDMA, and returns `w1`'s output.
To tag requests with the agent they belong to, see [Connect an agent](#connect-an-agent).

## Connect an agent

Send OpenAI-compatible requests to the router and add the `x-agent-context` header with the
agent, the branch (sub-agent) and the step:

```python
import json
import openai

client = openai.OpenAI(base_url="http://127.0.0.1:30900/v1", api_key="unused")
identity = {"agent_id": "task-42", "subagent_id": 0, "request_id": 7}
client.completions.create(
    model="deepseek-v4-flash",
    prompt="The capital of France is",
    max_tokens=16,
    extra_headers={"x-agent-context": json.dumps(identity)},
)
```

The router turns the identity into the engine request ID and uses it for
[KV retention](kv-retention.md) when that is enabled. For the prepared path through the frontend, use the
[client SDK](api-reference.md#client-sdk).

## Next steps

- Let the planner change the P/D split while the service runs: [Dynamic P/D](dynamic-pd.md).
- The flags the benchmarks use, and what they do: [Configuration](configuration.md).
- How the router picks P and D: [Placement](placement.md).

## Installation from source

The engine needs a CUDA 13 GPU host; the other components run anywhere. The steps are in
[Without the image](../docker/README.md#without-the-image).

Run the CPU tests:

```bash
pip install -e ".[client,frontend,indexer,planner,test]" && pytest -q
(cd evolving_infra/router/core && cargo test --locked)
pip install -e 'benchmarks/trie[test]' && pytest -q benchmarks/trie/tests   # Trie has its own pins: use a separate venv
```

## Benchmarks on Modal

[`examples/modal`](../examples/modal/README.md) builds the image on Modal and runs the reference
benchmarks: a static 1P1D stack on one node, and a two-node stack whose P/D split changes during
the run. You need a Modal account with H200 access, the model weights and a replay workload in
two Modal Volumes.

```bash
m() { uv run --no-project --with modal==1.5.5 modal "$@"; }
m run examples/modal/app.py::check_image_main
m run --detach examples/modal/app.py::static     # 1 node, 1P1D, 1200 s replay
m run --detach examples/modal/app.py::dynamic    # 2 nodes, 2P2D -> 1P3D -> 2P2D, 1800 s replay
```

A run counts only when `valid` is true in `<run_id>/node0/summary.json`. Expected results are in
[benchmarks/reproduce](../benchmarks/reproduce/README.md); replaying your own workload is
described in [benchmarks/replay](../benchmarks/replay/README.md).
