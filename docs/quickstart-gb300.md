# Quickstart on GB300

Run Evolving Infra with the paper's configuration on two GB300 nodes (4 GPUs each), serving
GLM-5.2: node A runs a prefill worker (P), the KV directory, the frontend and the router; node B
runs a decode worker (D). The router picks the decode worker when the prefill starts, places
prefills by predicted time to first token, pulls KV between workers' host memory, and keeps the
context of live agent branches.

## Prerequisites

- Two GB300 nodes on one RDMA network, with GLM-5.2 (NVFP4) at the same path on both.
- The runtime image on both nodes, built from this repository:

  ```bash
  git clone https://github.com/mit-han-lab/EvolvingInfra.git && cd EvolvingInfra
  docker build -f docker/Dockerfile -t evolving-infra .
  docker run --gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 \
    -v /models:/models -it evolving-infra bash
  ```

  Under Slurm with Pyxis, run the same image with `srun --container-image=...` on each node.

On both nodes, inside the container:

```bash
export MODEL=/models/GLM-5.2-NVFP4
export A_IP=10.0.0.1 B_IP=10.0.0.2          # the two nodes' addresses
export ME_IP=$A_IP                          # $B_IP on node B
export WITH_NVIDIA_PEERMEM=0                # GB300: register GPU memory through dma-buf
NICS='{"0":"mlx5_0,mlx5_1","1":"mlx5_0,mlx5_1","2":"mlx5_2,mlx5_3","3":"mlx5_2,mlx5_3"}'
```

## 1. Start the engines

Run the same command on both nodes:

```bash
SGLANG_HOST_IP=$ME_IP python3 -m decoupled_engine.launch_server \
  --model-path "$MODEL" --served-model-name glm-5.2 --trust-remote-code \
  --tp-size 4 --host 0.0.0.0 --port 30000 --page-size 64 --chunked-prefill-size 8192 \
  --attention-backend dsa --moe-runner-backend flashinfer_trtllm --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --disaggregation-mode symmetric --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$NICS" --disaggregation-bootstrap-port 8998 \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557", "replay_endpoint": "tcp://*:5558"}' \
  --enable-metrics --enable-cache-report --incremental-streaming-output &
```

In symmetric mode the engine runs the paper's configuration: host KV tier with dedup, peer KV
pulls, late decode choice, KV session retention and the prefill work snapshot. It logs them in a
`Symmetric features:` line ([Configuration](configuration.md#on-by-default-in-symmetric-mode)). Wait until `curl -sf http://127.0.0.1:30000/health`
answers on both nodes; the first start takes about 15 minutes.

## 2. Start the KV directory, the frontend and the router (node A)

```bash
cat > indexer.json <<EOF
{"model_name": "glm-5.2", "block_size": 64, "eagle": false, "indexer_port": 8090, "query_port": 8091, "threads": 4,
 "workers": [
  {"instance_id": 20001, "events_endpoint": "tcp://$A_IP:5557", "replay_endpoint": "tcp://$A_IP:5558",
   "bridge_pub_port": 26100, "bridge_replay_port": 26101},
  {"instance_id": 20002, "events_endpoint": "tcp://$B_IP:5557", "replay_endpoint": "tcp://$B_IP:5558",
   "bridge_pub_port": 26102, "bridge_replay_port": 26103}]}
EOF
/opt/indexer-venv/bin/python -m evolving_infra.router.indexer --config indexer.json --output indexer &
until [ -e indexer/ready.json ]; do sleep 2; done

cat > frontend.json <<EOF
{"node_id": "t0", "config_id": "glm52", "model_path": "$MODEL",
 "tokenize_endpoint": "http://$A_IP:30000/tokenize",
 "targets": {"http://$A_IP:30000/generate": "glm52", "http://$B_IP:30000/generate": "glm52"},
 "page_size": 64, "context_length": 1048576, "hash_mode": "dynamo", "is_eagle": false, "cache_salt": null,
 "callback_targets": ["http://$A_IP:30900/symmetric_pd/events"]}
EOF
/opt/indexer-venv/bin/python -m evolving_infra.router.frontend --config frontend.json --host 0.0.0.0 --port 28100 &

cat > prepared.json <<EOF
{"callback_url": "http://$A_IP:30900/symmetric_pd/events", "max_inflight": 4096,
 "contracts": {"glm52": {"page_size": 64, "is_eagle": false, "model_path": "$MODEL", "cache_namespace": null}}}
EOF
EVOLVING_INFRA_PREPARED_CONFIG=prepared.json sgl-router --host 0.0.0.0 --port 30900 \
  --model-id glm-5.2 --tokenizer-path "$MODEL/tokenizer.json" --policy symmetric_pd \
  --request-timeout-secs 3600 --stale-request-timeout-secs 3600 \
  --symmetric-pd-worker http://$A_IP:30000,8998,1,0 --symmetric-pd-worker http://$B_IP:30000,8998,0,1 \
  --kv-indexer-url "$(python3 -c 'import json; print(json.load(open("indexer/ready.json"))["query_url"])')" \
  --kv-worker-map "[{\"id\": 20001, \"url\": \"http://$A_IP:30000\", \"servicer\": \"$A_IP:18100\"},
                    {\"id\": 20002, \"url\": \"http://$B_IP:30000\", \"servicer\": \"$B_IP:18100\"}]" &
until curl -sf http://127.0.0.1:30900/readyz; do sleep 2; done
```

With `--policy symmetric_pd` the router runs the paper's configuration: placement by predicted
time to first token with peer pulls, late decode choice and KV session retention
([Configuration](configuration.md#on-by-default-with---policy-symmetric_pd)).
`--symmetric-pd-worker URL,BOOTSTRAP_PORT,P_WEIGHT,D_WEIGHT` sets each worker's starting role:
node A starts as prefill, node B as decode. With more nodes, add one engine, one `workers` entry
in `indexer.json`, one `targets` entry in `frontend.json`, and one `--symmetric-pd-worker` and
`--kv-worker-map` entry per node.

## 3. Send requests

```bash
curl -s http://127.0.0.1:30900/v1/completions -H 'Content-Type: application/json' \
  -H 'x-agent-context: {"agent_id": "task-42", "subagent_id": 0, "request_id": 1}' \
  -d '{"model": "glm-5.2", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

The `x-agent-context` header tags the request with its agent and branch, which KV retention
uses ([Connect an agent](quickstart.md#connect-an-agent)). To replay a recorded agent trace
against this stack, see [benchmarks/replay](../benchmarks/replay/README.md); to change the P/D
split while it runs, see [Dynamic P/D](dynamic-pd.md).
