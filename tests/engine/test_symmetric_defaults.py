# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Environment defaults that follow from the symmetric P/D server arguments."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "evolving_infra/engine/decoupled_engine/srt/arg_groups/symmetric_defaults.py"
)
spec = importlib.util.spec_from_file_location("symmetric_defaults", PATH)
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)

LATE_D_CHOOSE = {
    "SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE": "1",
    "SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION": "1.0",
    "SGLANG_SYMMETRIC_IMPORT_L2_RESTORE": "1",
    "SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION": "1",
    "SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT": "1",
    "SGLANG_SYMMETRIC_BUDGET_STANDARD_PD": "1",
    "SGLANG_SYMMETRIC_LAUNCH_THREAD": "1",
    "SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH": "1",
    "SGLANG_HICACHE_DECODE_CLEAR_CONSUMER": "1",
    "SGLANG_KV_SESSION_ADMIT_TAG": "1",
    "SGLANG_DECODE_SELECTION_METRICS": "1",
}
HOST = {
    "SGLANG_PREBUILT_NONBLOCKING_H2D": "1",
    "SGLANG_PREFILL_UNSERIALIZE_LOOP": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1500",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1500",
    "SGLANG_RETRACT_BACKUP_MAX_GB": "20",
    "SGLANG_HICACHE_PIPELINED_SYNC": "1",
    "SGLANG_HICACHE_SKIP_IDLE_SYNC": "1",
}
SNAPSHOT = {
    "SGLANG_PREFILL_ROUTING_SNAPSHOT": "1",
    "SGLANG_SYMMETRIC_ROUTING_SNAPSHOT": "1",
    "SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_RECORDS": "256",
    "SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_BYTES": "98304",
    "SGLANG_LOAD_SNAPSHOT_SLOT_BYTES": "131072",
}


def cfg(**kw):
    base = dict(
        disaggregation_mode="symmetric",
        disaggregation_late_d="choose",
        disable_overlap_schedule=False,
        enable_hierarchical_cache=True,
        disaggregation_bootstrap_prefetch="off",
        enable_kv_session_retention=True,
        enable_prefill_work_snapshot=True,
        enable_dsa_hicache_host_dedup=True,
        enable_mla_hicache_host_dedup=False,
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_symmetric_late_d_choose_turns_on_everything():
    env, fields = sd.plan(cfg(), {})
    assert env == {**LATE_D_CHOOSE, **HOST, **SNAPSHOT, "SGLANG_DSA_HOST_DEDUP_FANOUT": "ce_peer",
                   "SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY": "0"}
    assert fields == {"hicache_io_backend": "direct", "hicache_mem_layout": "page_first_direct"}


def test_kv_events_publish_lazily_in_batches():
    env, _ = sd.plan(cfg(kv_events_config='{"publisher": "zmq"}'), {})
    for name in ("SGLANG_KV_EVENTS_LAZY_LOAD_BACK", "SGLANG_KV_EVENTS_LAZY_FAST_BUILD",
                 "SGLANG_KV_EVENTS_BATCH_PUBLISH"):
        assert env[name] == "1"
    env, _ = sd.plan(cfg(), {})
    assert "SGLANG_KV_EVENTS_BATCH_PUBLISH" not in env


def test_user_settings_win():
    environ = {
        "SGLANG_SYMMETRIC_LAUNCH_THREAD": "0",
        "SGLANG_DSA_HOST_DEDUP_FANOUT": "nccl",
    }
    env, fields = sd.plan(cfg(), environ)
    assert "SGLANG_SYMMETRIC_LAUNCH_THREAD" not in env
    assert "SGLANG_DSA_HOST_DEDUP_FANOUT" not in env
    assert fields == {}


def test_values_resolution_set_are_reapplied():
    environ = {
        "SGLANG_DSA_HOST_DEDUP_FANOUT": "ce_peer",
        sd.APPLIED_ENV: "SGLANG_DSA_HOST_DEDUP_FANOUT",
    }
    env, fields = sd.plan(cfg(), environ)
    assert env["SGLANG_DSA_HOST_DEDUP_FANOUT"] == "ce_peer"
    assert fields["hicache_mem_layout"] == "page_first_direct"


def test_not_symmetric_only_snapshot():
    env, fields = sd.plan(cfg(disaggregation_mode="prefill"), {})
    assert env == {k: v for k, v in SNAPSHOT.items() if k != "SGLANG_SYMMETRIC_ROUTING_SNAPSHOT"}
    assert fields == {}


def test_late_d_alloc_only_admission_rules():
    env, _ = sd.plan(
        cfg(disaggregation_late_d="alloc", enable_prefill_work_snapshot=False,
            enable_dsa_hicache_host_dedup=False),
        {},
    )
    assert env == {
        "SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE": "1",
        "SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION": "1.0",
        **HOST,
    }


def test_conflicts_are_left_off():
    env, _ = sd.plan(
        cfg(disable_overlap_schedule=True, disaggregation_bootstrap_prefetch="local",
            enable_hierarchical_cache=False, enable_kv_session_retention=False),
        {"SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY": "1"},
    )
    for name in (
        "SGLANG_SYMMETRIC_LAUNCH_THREAD",
        "SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH",
        "SGLANG_SYMMETRIC_ROUTING_SNAPSHOT",
        "SGLANG_SYMMETRIC_IMPORT_L2_RESTORE",
        "SGLANG_SYMMETRIC_BUDGET_STANDARD_PD",
        "SGLANG_KV_SESSION_ADMIT_TAG",
    ):
        assert name not in env


def test_bootstrap_prefetch_keeps_walk_before_launch():
    env, _ = sd.plan(cfg(disaggregation_bootstrap_prefetch="both"), {})
    assert env["SGLANG_SYMMETRIC_LAUNCH_THREAD"] == "1"
    assert "SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH" not in env


def test_mla_dedup_and_other_layouts():
    env, fields = sd.plan(
        cfg(enable_dsa_hicache_host_dedup=False, enable_mla_hicache_host_dedup=True), {}
    )
    assert env["SGLANG_MLA_HOST_DEDUP_FANOUT"] == "ce_peer"
    assert "SGLANG_DSA_HOST_DEDUP_FANOUT" not in env
    env, fields = sd.plan(cfg(hicache_mem_layout="layer_first"), {})
    assert "SGLANG_DSA_HOST_DEDUP_FANOUT" not in env and fields == {}


def test_refuse(monkeypatch):
    name = "SGLANG_SYMMETRIC_LAUNCH_THREAD"
    monkeypatch.setenv(name, "1")
    monkeypatch.delenv(sd.APPLIED_ENV, raising=False)
    with pytest.raises(ValueError, match="no overlap"):
        sd.refuse(name, "no overlap")
    monkeypatch.setenv(sd.APPLIED_ENV, f"OTHER,{name}")
    sd.refuse(name, "no overlap")
    assert sd.os.environ[name] == "0"


def test_host_switches_follow_the_servicer_and_plain_tp():
    env, _ = sd.plan(cfg(enable_peer_kv_servicer=True), {"SGLANG_HICACHE_TP_DIGEST_MODE": "log"})
    assert env["SGLANG_PEER_KV_PUSH_FAST"] == "1"
    assert env["SGLANG_HICACHE_PIPELINED_SYNC"] == "1"
    assert "SGLANG_HICACHE_SKIP_IDLE_SYNC" not in env
    env, _ = sd.plan(cfg(pp_size=2), {})
    assert "SGLANG_HICACHE_PIPELINED_SYNC" not in env
    assert "SGLANG_PEER_KV_PUSH_FAST" not in env
    env, _ = sd.plan(cfg(disaggregation_mode="prefill"), {})
    assert "SGLANG_PREBUILT_NONBLOCKING_H2D" not in env


NICS = '{"0": "mlx5_0,mlx5_1", "1": "mlx5_0,mlx5_1", "2": "mlx5_2,mlx5_3", "3": "mlx5_2,mlx5_3"}'
DSA = {"dedup": "enable_dsa_hicache_host_dedup", "full_only": True}


def bare(**kw):
    """A symmetric engine with only the required flags."""
    base = dict(
        disaggregation_mode="symmetric",
        symmetric_preset="paper",
        enable_hierarchical_cache=False,
        enable_prefill_work_snapshot=False,
        disaggregation_late_d="off",
        disaggregation_late_d_control_port=None,
        disaggregation_late_d_lead_mode="count",
        disaggregation_late_d_allocation_trigger="lead",
        disaggregation_ib_device=NICS,
        enable_peer_kv_servicer=False,
        hicache_storage_backend=None,
        peer_kv_servicer_max_leased_gb=16.0,
        enable_kv_session_retention=False,
        enable_dsa_hicache_host_dedup=False,
        enable_mla_hicache_host_dedup=False,
        enable_dsv4_hicache_host_dedup=False,
        pp_size=1, dp_size=1, attn_cp_size=1, nnodes=1,
        enable_dp_attention=False, disable_overlap_schedule=False,
        disable_radix_cache=False, radix_eviction_policy="lru",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_paper_features_turn_on_for_a_bare_symmetric_engine():
    f = sd.plan_features(bare(), DSA, free_port=lambda: 21776, ratio_unset=True)
    extra = f.pop("hicache_storage_backend_extra_config")
    assert f == {
        "enable_hierarchical_cache": True,
        "enable_prefill_work_snapshot": True,
        "disaggregation_late_d": "choose",
        "disaggregation_late_d_control_port": 21776,
        "disaggregation_late_d_lead_mode": "need",
        "disaggregation_late_d_allocation_trigger": "first_chunk",
        "enable_peer_kv_servicer": True,
        "hicache_storage_backend": "dynamic",
        "peer_kv_servicer_max_leased_gb": 64.0,
        "enable_dsa_hicache_host_dedup": True,
        "hicache_ratio": 4.445,
        "enable_kv_session_retention": True,
    }
    assert sd.json.loads(extra)["ib_device"] == "mlx5_0,mlx5_1,mlx5_2,mlx5_3"
    # The environment plan then sees the features as set.
    env, fields = sd.plan(sd._Overlay(bare(hicache_io_backend="kernel", hicache_mem_layout="page_first"), f), {})
    assert env["SGLANG_DSA_HOST_DEDUP_FANOUT"] == "ce_peer"
    assert env["SGLANG_KV_SESSION_ADMIT_TAG"] == "1"
    assert env["SGLANG_PEER_KV_PUSH_FAST"] == "1"
    assert fields["hicache_mem_layout"] == "page_first_direct"


def test_paper_features_keep_what_the_user_set_and_skip_what_cannot_run():
    assert sd.plan_features(bare(symmetric_preset="none"), DSA) == {}
    assert sd.plan_features(bare(disaggregation_mode="prefill"), DSA) == {}
    f = sd.plan_features(
        bare(disaggregation_late_d="alloc", enable_mla_hicache_host_dedup=True,
             hicache_storage_backend="mooncake", disaggregation_ib_device=None),
        DSA, free_port=lambda: 1, ratio_unset=False,
    )
    assert "disaggregation_late_d" not in f
    assert "enable_dsa_hicache_host_dedup" not in f and "hicache_ratio" not in f
    assert "enable_peer_kv_servicer" not in f
    # An explicit --disaggregation-late-d=off stays off (chat completions need it).
    f = sd.plan_features(bare(_disaggregation_late_d_explicit=True), DSA, free_port=lambda: 1)
    assert "disaggregation_late_d" not in f and "disaggregation_late_d_control_port" not in f
    f = sd.plan_features(bare(enable_dp_attention=True), DSA, free_port=lambda: 1)
    assert "disaggregation_late_d" not in f and "enable_dsa_hicache_host_dedup" not in f
    f = sd.plan_features(bare(), {"dedup": "enable_dsv4_hicache_host_dedup", "full_only": False},
                         free_port=lambda: 1, ratio_unset=True)
    assert f["enable_dsv4_hicache_host_dedup"] and "enable_kv_session_retention" not in f
    f = sd.plan_features(bare(), None, free_port=lambda: 1)
    assert "enable_kv_session_retention" not in f and "hicache_ratio" not in f


def test_ib_devices():
    assert sd.ib_devices(None) == ""
    assert sd.ib_devices("mlx5_0") == "mlx5_0"
    assert sd.ib_devices("mlx5_0, mlx5_1,mlx5_0") == "mlx5_0,mlx5_1"
    assert sd.ib_devices(NICS) == "mlx5_0,mlx5_1,mlx5_2,mlx5_3"
