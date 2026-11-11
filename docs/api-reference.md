# API Reference

HTTP endpoints of the router (`sgl-router`). The engines expose the SGLang-style endpoints
(`/generate`, `/v1/completions`, `/v1/loads`, `/kv_session/control`, and others) that the router
and the frontend call. The frontend's endpoints (`/prepare`, `/forward_pd`) are in
[evolving_infra/router/frontend](../evolving_infra/router/frontend/README.md).

## Health and metrics

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | Liveness. |
| GET | `/readyz` | Ready once the workers are registered and healthy. |
| GET | `/metrics` | Prometheus metrics, readable before readiness. |
| GET | `/v1/models` | The model the router serves. |

## Generation

These routes pick the workers and proxy the request to the same path on them. Under
`--policy symmetric_pd` they choose a P and a D per request.

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions. |
| POST | `/v1/completions` | OpenAI-compatible completions. `prompt` may be a token-ID array. |
| POST | `/generate` | SGLang-native generation with `input_ids`. Conformance checks use it. |
| POST | `/v1/tokenize`, `/v1/detokenize` | Tokenize or detokenize with the router's tokenizer. |

Agent identity: an optional `x-agent-context` header,
`{"agent_id": "<1..512 bytes>", "subagent_id": <u64>, "request_id": <u64>}`. The router derives
the engine request ID (`rid`) from it and rejects a body `rid` that conflicts with it.

## Prepared path

The frontend tokenizes once and sends only a compact descriptor to the router. The router never
sees the full token arrays. Use it through the [client SDK](#client-sdk).

| Method | Path | Description |
|---|---|---|
| POST | `/symmetric_pd/prepared` | Plan a request: query the KV directory, pick P and D, reserve the work, and return the plan. |
| POST | `/symmetric_pd/events` | Lifecycle events from the frontend (claim, upload, completion, failure), authenticated with the plan's callback token. |
| POST | `/symmetric_pd/cancel` | Cancel a prepared execution and abort its engine legs. |
| POST | `/symmetric_pd/migration` | Tell the frontend whether an aborted D leg was moved off a draining worker. |
| GET | `/symmetric_pd/executions` | Counts and states of the registered executions. |
| GET | `/symmetric_pd/backend/loads` | The router's merged view of worker loads. |
| POST | `/symmetric_pd/place` | Placement only, for an external ingress ([below](#post-symmetric_pdplace)). |

## Roles (dynamic P/D)

Enabled by `--symmetric-pd-external-role-control`.

| Method | Path | Description |
|---|---|---|
| GET | `/symmetric_pd/roles` | Current P/D weights per worker, the epoch and revision, and completed workload totals. |
| PUT | `/symmetric_pd/roles` | Replace the weights, compare-and-set on epoch and revision. Authenticated with the token file named by `EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE`. |

The planner is the usual caller ([planner](../evolving_infra/planner/README.md)).

## Cache and sessions

| Method | Path | Description |
|---|---|---|
| POST | `/kv_sessions/end` | End an agent branch: the router queues `end` to every engine that held the session. Refused when retention is off. |
| POST | `/flush_cache` | Flush every worker's cache; `502` with a per-worker breakdown if any worker fails. |

`/kv_sessions/end` takes `{"agent_id": "...", "subagent_id": N}`. See [KV retention](kv-retention.md).

## `POST /symmetric_pd/place`

Returns a placement without dispatching it, for a gateway that sends the two legs itself.

```json
{"input_ids": [1, 2, 3, 4], "prompt_len": 4096, "cache_salt": null, "account": true, "route": "/generate"}
```

`input_ids` (or `hashes`, the page hashes from the frontend's `/prepare`) is the KV directory
key; `prompt_len` is the prompt length in tokens. `account: false` is a dry run that adds no
load. The response:

```json
{
  "p_leg": {"worker": "http://10.0.0.1:30000", "endpoint": "http://10.0.0.1:30000/generate"},
  "d_leg": {"worker": "http://10.0.0.5:30000", "endpoint": "http://10.0.0.5:30000/generate"},
  "local": false,
  "transfer_id": 4611686018427387905,
  "targets": [
    {"leg": "p", "endpoint": "http://10.0.0.1:30000/generate", "mark": {"kv_export": {"...": "..."}}},
    {"leg": "d", "endpoint": "http://10.0.0.5:30000/generate", "mark": {"kv_import": {"...": "..."}}}
  ]
}
```

Merge each target's `mark` into the request body, send both targets at the same time, and stream
the D leg's response to the client. Errors: `404 symmetric_pd_disabled` (router not under
`--policy symmetric_pd`), `503 symmetric_pd_no_placement` (no usable P/D pair).

## Client SDK

`evolving_infra.router.client` (`pip install -e ".[client]"`) gives each request an agent
identity and runs the prepared path.

```python
import aiohttp
from evolving_infra.router.client.agent import Agent
from evolving_infra.router.client.scheduler import Scheduler

async def generate():
    frontend = "http://127.0.0.1:28100"
    async with Scheduler("http://127.0.0.1:30900") as scheduler:
        agent = Agent(scheduler, agent_id="task-42")   # one agent run; branch 0
        request = agent.request({})                   # the next step of branch 0
        body = {
            "protocol_version": 1,
            "request_id": request.identity.engine_rid,
            "config_id": "dsv4",
            "input": {"prompt": "Explain prefix caching.", "add_special_tokens": False},
            "generation": {"sampling_params": {"max_new_tokens": 64}, "stream": False},
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(frontend + "/prepare", json=body) as response:
                response.raise_for_status()
                prepared = await response.json()
        return await scheduler.submit_prepared(frontend, prepared)
```

- `agent.fork()` starts a new branch (sub-agent) with its own request counter.
- For streaming, prepare with `"stream": true` and iterate `scheduler.stream_prepared(...)`.
- Nothing is retried automatically: not preparation, placement or generation.
- A closed stream does not prove the engine stopped; cancellation goes to the router, which
  aborts both legs.
