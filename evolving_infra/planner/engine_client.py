"""The engine endpoints the planner reads: capacities and TP size at start, the served model's expert counts."""
import json
from pathlib import Path

from evolving_infra.planner import envs
from evolving_infra.planner.utils import http_json, open_url


def served_experts(urls, timeout=None):
    """Routed experts and experts per token of the one model all engines serve.

    Each engine's `/model_info` names its model path; the counts come from that
    model's config.json (its `text_config` for a multimodal model), which must be readable at
    the same path on this host.
    """
    counts = set()
    timeout = envs.EVOLVING_INFRA_PLANNER_MODEL_INFO_TIMEOUT_SECS if timeout is None else timeout
    for url in urls:
        with open_url(url.rstrip('/') + '/model_info', timeout) as response:
            path = Path(json.load(response)['model_path'])
        config = json.loads((path / 'config.json').read_text())
        if 'n_routed_experts' not in config:  # a multimodal model keeps the language model's settings apart
            config = config.get('text_config') or config
        counts.add((config['n_routed_experts'], config['num_experts_per_tok']))
    if len(counts) != 1:
        raise ValueError('Engines do not serve one expert configuration: ' + repr(counts))
    return counts.pop()


def worker_capacities(workers):
    """{node: kv_capacity_tokens, max_running_requests, tp_size} from every engine's ``/server_info``.

    The request cap is the launch flag when the engine was given one; otherwise the engine computed it at
    startup, and it is the sum of the value each data-parallel scheduler reports.
    """
    capacities = {}
    for w in workers:
        info = http_json(w["url"] + "/server_info", timeout=envs.EVOLVING_INFRA_PLANNER_SERVER_INFO_TIMEOUT_SECS)
        cap = info.get("max_running_requests")
        if cap is None:
            cap = sum(int(s["effective_max_running_requests_per_dp"]) for s in info["internal_states"])
        capacities[w["node"]] = dict(kv_capacity_tokens=int(info["max_total_num_tokens"]),
                                     max_running_requests=int(cap), tp_size=int(info["tp_size"]))
    return capacities
