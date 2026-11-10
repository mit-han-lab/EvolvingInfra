# KV Retention

An agent pauses between turns for tool calls or user input, then comes back with the same context
plus a little more. Plain LRU eviction treats that context like any other cache entry and often
evicts it during the pause. KV retention lets the router tell the engines which context agents
still need, so the engines evict unreferenced KV first.

Retention runs on models with a FULL-only cache tree, such as GLM-5.2 and Kimi-K2.6.

## Enable it

- Engines: `--enable-kv-session-retention`, with the host tier in write-through mode
  (`--enable-hierarchical-cache --hicache-write-policy write_through`). Under late D the
  engine also turns on `SGLANG_KV_SESSION_ADMIT_TAG`, so the chosen D counts the branch.
- Router: `--symmetric-pd-kv-session-retention`.
- Clients: tag every request with `x-agent-context` and end each branch with
  `POST /kv_sessions/end` (below).

## How it works

**Identity.** Each request carries `x-agent-context: {"agent_id", "subagent_id",
"request_id"}`. A sub-agent is one branch of an agent's trajectory; one branch is one session.
The router derives a session key from the identity. Cache salt and prefix hashes do not change.

**Reference.** The router adds a `kv_session` tag `{session_id, version}` to the P leg (and to
the D leg when late D is off). The engine records one reference per session on the committed
context. Host pages that some session references are evicted after unreferenced ones. Among
referenced pages, the session predicted to return last goes first: the engine estimates the
remaining idle time from how long the session has been idle, and treats a session idle for 150 s
or longer as finished. `SGLANG_KV_SESSION_RETENTION_ORDER=lru` orders them by last access
instead. The version moves forward on every new request of the branch and whenever its P changes.

**Moving a session.** When a branch's next request goes to a different P, the old worker keeps
its copy until the new P reports the full context ready. Then the router releases the old
reference. A missing reply keeps the extra copy; KV is never released early.

**Ending a branch.** When a branch finishes, the client calls:

```bash
curl -X POST http://router:30900/kv_sessions/end -H 'Content-Type: application/json' \
  -d '{"agent_id": "task-42", "subagent_id": 0}'
```

The router sends `end` to every engine that held the session. A router without retention
refuses this call, so leave it off in the client too.

**Control messages.** The router batches `status`, `release` and `end` operations to each engine
(`POST /kv_session/control`) in the background. Each is fenced by the session version and the
engine's boot ID, so a late or repeated message cannot drop a newer reference or touch a
restarted engine. Role changes release a drained worker's references the same way.
