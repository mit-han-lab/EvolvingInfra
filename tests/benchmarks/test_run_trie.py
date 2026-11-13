# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The replay adapter's arguments: the paper preset, stack.json, and the record of differences."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks/replay" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_trie = load("run_trie")
STACK = {"router_url": "http://10.0.0.1:30900", "tokenizer_url": "http://10.0.0.1:28100", "model": "glm-5.2",
         "model_path": "/models/GLM", "page_size": 64, "config_id": "default", "retention": True, "gpus": 36}


def resolve(tmp_path, *argv, env=None):
    (tmp_path / "stack.json").write_text(json.dumps(STACK))
    args, settings, record = run_trie.resolve([f"stack={tmp_path}", *argv], env or {})
    return dict(a.split("=", 1) for a in args), settings, record


def test_five_arguments_run_the_papers_replay(tmp_path):
    args, settings, record = resolve(tmp_path, "workload_path=w.jsonl", "concurrency=126", "duration=5400",
                                     f"output_dir={tmp_path}/bench")
    assert {k: args[k] for k in run_trie.PAPER} == run_trie.PAPER
    assert args["endpoint"] == "http://10.0.0.1:30900/v1" and args["model"] == "glm-5.2"
    assert args["tokenizer_model"] == "/models/GLM" and args["prefix_block_size"] == "64" and args["num_gpus"] == "36"
    assert args["output_json"].endswith("bench/trie-results-0.json") and args["lifecycle_concurrency"] == "126"
    assert settings == {"router_url": "http://10.0.0.1:30900", "tokenizer_urls": ["http://10.0.0.1:28100"],
                        "config_id": "default", "model_name": "glm-5.2", "model_path": "/models/GLM",
                        "incremental_streaming_output": True, "retention_enabled": True}
    assert record["preset"] == "paper" and record["differs_from_preset"] == {}
    assert not set(args) & set(run_trie.OWN)


def test_an_explicit_argument_wins_and_is_recorded(tmp_path):
    args, _, record = resolve(tmp_path, "concurrency=8", "replay_context_mode=recorded", "seed=0", "stream=True")
    assert args["replay_context_mode"] == "recorded"
    assert record["differs_from_preset"] == {"replay_context_mode": {"value": "recorded", "preset": "actual_output"}}


def test_processes_split_the_concurrency(tmp_path):
    args, _, record = resolve(tmp_path, "concurrency=224", "processes=16", "process=3", f"output_dir={tmp_path}")
    assert args["concurrency"] == "14" and args["output_json"].endswith("trie-results-3.json")
    assert record["process"] == 3 and record["processes"] == 16
    with pytest.raises(SystemExit):
        resolve(tmp_path, "concurrency=100", "processes=16")


def test_preset_none_adds_nothing(tmp_path):
    env = {"EVOLVING_INFRA_REPLAY": json.dumps({"router_url": "http://r:1", "tokenizer_url": "http://t:1"})}
    args, settings, record = run_trie.resolve(["preset=none", "workload_path=w", "cache_salt_mode=global"], env)
    assert args == ["workload_path=w", "cache_salt_mode=global"] and settings["router_url"] == "http://r:1"
    assert record["differs_from_preset"] is None
    with pytest.raises(SystemExit):
        run_trie.resolve(["preset=none"], {})


def test_score_reports_the_differences(tmp_path):
    score = load("score")
    event = {"ts_ns": 0, "event": "TASK_START", "request": {"experiment_request_id": "a"}, "attrs": {}}
    (tmp_path / "trie-lifecycle-0.jsonl").write_text(json.dumps(event) + "\n")
    differs = {"seed": {"value": "1", "preset": "0"}}
    (tmp_path / "replay-settings-0.json").write_text(json.dumps({"preset": "paper", "differs_from_preset": differs}))
    result = score.window_score(str(tmp_path), 0, 10)
    assert result["replay_preset"] == ["paper"] and result["replay_differs_from_preset"] == differs


def test_processes_give_the_same_trace_different_agents():
    identity = load("replay_agent_identity")
    one = identity._agent_id("trace", "putnam_1990_a2", 0)
    identity.PROCESS = "process-3:"
    assert identity._agent_id("trace", "putnam_1990_a2", 0) != one
    identity.PROCESS = ""
    assert identity._agent_id("trace", "putnam_1990_a2", 0) == one
