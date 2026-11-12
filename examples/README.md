# Examples

A config says what to serve on one kind of hardware. It does not say where the machines come
from: the same config runs on a host you log in to or under Slurm. The Modal harness keeps its
own settings in [`modal/stack.py`](modal/stack.py).

| Hardware and model | Config | Status |
|---|---|---|
| [GB300, GLM-5.2](gb300/README.md) | [`gb300/glm52.yaml`](gb300/glm52.yaml), [`gb300/glm52-1node.yaml`](gb300/glm52-1node.yaml) | the paper's main experiment |
| GB300, Kimi-K2.6 | [`gb300/kimi-k26.yaml`](gb300/kimi-k26.yaml) | the paper's OSWorld experiment ([`benchmarks/osworld`](../benchmarks/osworld/README.md)) |
| [H200, DeepSeek-V4-Flash](h200/README.md) | [`h200/dsv4-flash.yaml`](h200/dsv4-flash.yaml) | the paper's H200 setup |

| Where it runs | How |
|---|---|
| A host with a shell | `evolving-infra serve --config <config>` on every host ([`deploy-stack`](../skills/deploy-stack/SKILL.md) skill) |
| Slurm with Pyxis | [`slurm/submit.py`](slurm/submit.py)` --config <config>`: serve, replay and score as one job |
| Modal | [`modal/`](modal/README.md): builds the image and runs the H200 benchmarks on one, two and eight nodes |

[`agents/`](agents/README.md) runs Pi, Claude Code and Codex on a running stack: configs with tool-call
parsers and one launch script per harness.

For other hardware or another model, the [`deploy-stack`](../skills/deploy-stack/SKILL.md) skill derives a
config; add it here as `<hardware>/<model>.yaml`.
