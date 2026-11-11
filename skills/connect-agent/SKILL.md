---
name: connect-agent
description: Connect an agent or agent framework to a running Evolving Infra stack, so that every model request carries its agent identity and every branch is ended when it finishes. Use when asked to point an agent, a sub-agent framework or OpenAI-client code at Evolving Infra, or when its requests are refused or its prefix hits stay low. Not for starting the stack; use `deploy-stack`.
---

# Connect an agent

Make an agent's model requests carry the identity that Evolving Infra uses to keep the KV of
live branches, and end each branch when it finishes. Needs a running stack
([`deploy-stack`](../deploy-stack/SKILL.md)).

## Scope

- **Reads**: `docs/api-reference.md`, `docs/kv-retention.md`, `evolving_infra/router/client/`:
  the identity header, the prepared path, ending a branch, and the Python client.
- **Writes**: nothing in this repository.
- **Never changes**: `evolving_infra/`: connecting a client needs no change to the server. If
  it seems to, stop and say why.
- **Outside the repository**: edits the agent's code, which belongs to the person. Confirm the
  change with the owner before making it.

## Inputs

- The stack's `stack.json`, in the work directory of `evolving-infra serve` on the first host:
  `router_url`, `tokenizer_url` (the frontend), `config_id` and `retention`.
- Whether the stack serves with `--planner`.
- The agent's code. Ask where it is; do not guess which repository or file sends the requests.

## Steps

### 1. Find the requests and the branches

In the agent's code, find:

- every call that sends a model request;
- where a task starts: one task run is one `agent_id`;
- where a sub-agent starts and where it returns: each sub-agent is one branch, one
  `subagent_id`;
- where the task ends.

**Check**: every model request of the agent goes through the calls found; no other HTTP
client sends model requests.

### 2. Give every request its identity

Each request carries `x-agent-context: {"agent_id": ..., "subagent_id": ..., "request_id": ...}`
([API reference](../../docs/api-reference.md#generation)):

| Field | Value |
|---|---|
| `agent_id` | one per task run, 1 to 512 bytes; two runs never share one (a new UUID per run) |
| `subagent_id` | `0` for the main branch; a new number, unique within the task, for each sub-agent |
| `request_id` | the request's position in its branch: `0`, `1`, `2`, ... |

In Python, the client SDK allocates the three numbers: `Agent(scheduler)` for the task (it draws
a new `agent_id`), `agent.fork()` for a sub-agent, `agent.request(payload)` for the identity of
the next request ([client SDK](../../docs/api-reference.md#client-sdk)). `agent.submit(payload)`
also sends it, on the plain path (`/v1/completions` or `/v1/chat/completions`). With another
client, send the header on each request:

```python
client = openai.OpenAI(base_url="<router_url>/v1", api_key="EMPTY")
identity = {"agent_id": run_id, "subagent_id": branch, "request_id": step}
client.chat.completions.create(model="<served_model_name>", messages=messages, max_tokens=1024,
                               extra_headers={"x-agent-context": json.dumps(identity)})
```

On the plain path, send chat messages, or token IDs as `prompt` on `/v1/completions`: the router
tokenizes chat messages and looks the prefix up in the KV directory, but it does not tokenize a
text `prompt`, so such a request is placed without knowing which worker holds its prefix. Do not
set `rid` in the body: the router derives it from the header and refuses a different one.

**Check**: the reply's `id` is `agent_` followed by 64 hexadecimal digits: the router used the
identity. In Python it equals `RequestIdentity(agent_id, subagent_id, request_id).engine_rid`
(`evolving_infra/router/client/agent/identity.py`).

### 3. With the planner on, send requests through the prepared path

The planner learns from requests that go through the prepared path only. When it runs with KV
retention on in the router (the default; `retention: false` in the config turns it off), the
router refuses plain `/v1/completions`, `/v1/chat/completions` and `/generate` requests with HTTP
400 `dynamic role switching with KV session retention requires /symmetric_pd/prepared lifecycle
callbacks`. So whenever the stack serves with `--planner`, send each request through the
frontend, as in the [client SDK](../../docs/api-reference.md#client-sdk) example:
`POST <tokenizer_url>/prepare` with `"request_id": request.identity.engine_rid` and the stack's
`config_id`, then `scheduler.submit_prepared(tokenizer_url, prepared, identity=request.identity)`,
or `stream_prepared` with `"stream": true`.

**Check**: a request returns generated text, and its `meta_info.id` starts with `agent_` (the
prepared path returns the engine's native reply). A refused request gets HTTP 400 with the
message above; the router does not log it. Without `--planner`, this step does not apply: say so.

### 4. End each branch

When a branch finishes (a sub-agent returns, or the task ends), end it:
`await agent.end_trajectory()` in Python (each `fork()` separately, before the `Scheduler` closes), or

```bash
curl -X POST <router_url>/kv_sessions/end -H 'Content-Type: application/json' \
  -d '{"agent_id": "<agent_id>", "subagent_id": <subagent_id>}'
```

Ending is idempotent: retry it when the call fails; sibling branches are not affected. End only a
branch that sent at least one request with its identity: the router refuses any other
(`unknown KV session`). After the end, that identity cannot send again. A router without KV
retention (`"retention": false` in `stack.json`) refuses every end (`KV session retention is
disabled`): then leave it out ([KV retention](../../docs/kv-retention.md)).

**Check**: the reply holds `"accepted": true`.

### 5. Run a task with two turns and a sub-agent

Run one short task in which the main branch sends at least two requests and starts one
sub-agent. Make the first prompt longer than a few KV pages: `#cached-token` counts whole pages
(`page_size` tokens: 64 in the GB300 configs, 256 for DeepSeek-V4-Flash).

**Check**: every reply carries an `agent_` ID (step 2 or 3); each branch's end is accepted; and
the log of the P worker that ran the second request of the main branch,
`<work dir>/logs/engine-w<i>.log` on the host of worker `w<i>`, shows its prefill batch with
`#cached-token` greater than 0.

## Returns

- The change to the agent's code: where the identity is set, where sub-agents fork, where
  branches end, and whether requests use the prepared path.
- The evidence of step 5: one reply ID, the end replies, and the `#cached-token` line.
- What the change does not cover (streaming, retries, a second client), as decisions for the
  owner.

## Pitfalls

- A prepared request sent without `identity=` runs, but its branch is not registered for KV
  retention: pass the request's identity to `submit_prepared` and `stream_prepared`.
- Two task runs that share an `agent_id` collide. On the prepared path, or with KV retention on,
  the second run's requests are refused (`request already planned; do not repeat generation`,
  `stale trajectory request`, or `trajectory has ended; use a new agent/subagent identity`); on
  the plain path without retention they run under the first run's request IDs. Make a new one
  per run.
- A closed stream does not prove that the engine stopped. With the client SDK, cancel the task
  that runs `submit_prepared`, or close the `stream_prepared` iterator: either sends
  `POST /symmetric_pd/cancel`, and the router asks both legs' engines to abort.
