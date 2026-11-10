"""Apply a previously validated, frozen decision to an engine launch template.

This prepares a new launch configuration; it does not mutate running engines.
The caller must stop the learning deployment before starting this one.
"""

from copy import deepcopy
from pathlib import Path

from .model import load_frozen


def apply_frozen_ratio(
    template: dict, decision_path: Path, learning_config: dict,
    *, duration_s: int = 3600, warmup_excluded_s: int = 900,
) -> dict:
    if not 0 <= warmup_excluded_s < duration_s:
        raise ValueError("Require 0 <= warmup_excluded_s < duration_s")
    frozen = load_frozen(decision_path, learning_config)
    selected = frozen["report"]["decision"]["selected"]
    p, d = selected["prefill_nodes"], selected["decode_nodes"]
    if p + d != len(template["nodes"]) or p < 1 or d < 1:
        raise ValueError("Frozen ratio does not match the deployment's node count")
    expected = {
        "chunked_prefill_size": learning_config["chunk"],
        "kv_cache_dtype": learning_config["kv_dtype"],
        "hicache_size_gb": learning_config["hicache_size_gb"],
        "host_copies": learning_config["host_copies"],
        "model_config_sha256": learning_config["model_config_sha256"],
        "model_index_sha256": learning_config["model_index_sha256"],
    }
    if any(template.get(k) != v for k, v in expected.items()):
        raise ValueError("Deployment model/cache settings differ from calibration")
    result = deepcopy(template)
    result.update(
        phase="formal",
        prefill_nodes=p,
        decode_nodes=d,
        decode_max_running_requests=selected["batch"],
        frozen_ratio_path=str(Path(decision_path).resolve()),
        frozen_ratio_sha256=frozen["payload_sha256"],
        duration_s=duration_s,
        warmup_excluded_s=warmup_excluded_s,
        steady_s=duration_s - warmup_excluded_s,
        drain_timeout_s=0,
    )
    for index, node in enumerate(result["nodes"]):
        node["role"] = "prefill" if index < p else "decode"
    return result
