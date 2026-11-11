# launch

`evolving-infra serve`: one command per host starts that host's engines and, on the first
host, the KV directory, the frontend, the router and the planner.
[`examples/gb300/README.md`](../examples/gb300/README.md) shows its use.

- **Three interpreters.** The launcher runs in the environment that has the frontend, the KV
  directory and the planner (`sys.executable`). Engines start with the interpreter that has
  `decoupled_engine` (`engine_python()`), the router is the `sgl-router` binary on `PATH`.
- **Ports are assigned in `layout.py`** and nowhere else. `Worker.ports()` lists every port a
  worker binds; `serve.py` checks them before starting. A new listening port is added to both.
- **What the launcher reads from the node is in `hardware.py`**: GPU count, the RDMA NICs
  closest to each GPU (InfiniBand ports when there are any; `nvidia-smi topo -m`), whether
  `nvidia_peermem` is loaded. No host, NIC or cluster name appears in code.
- **A host finds itself in `--hosts`** by name (`hardware.own_index`). Do not take the rank
  from a scheduler variable: a scheduler's node index is not the position in a sorted list.
- **The planner is on only with `--planner`.** The config's `planner:` section holds its
  arguments. With the planner and KV session retention on, the router accepts only
  prepared-path requests, so a plain OpenAI request works only without `--planner`.
- **A config can start from another** (`base: other.yaml`); an engine flag given in the
  child replaces the same flag of the base. Model settings go in `examples/`, not in code.
- **`examples/gb300/submit.py` only writes a job script.** Inspect it with `--dry-run`. What
  runs on the nodes is `evolving-infra serve` and the replay command, the same as by hand.

Tests: `python -m pytest -q tests/launch` (hardware is faked with `monkeypatch`). Then
`evolving-infra serve --config examples/gb300/glm52-1node.yaml` on a node: it prints `ready`
and writes `READY` in the work directory; each process logs to `<work dir>/logs/<name>.log`.
