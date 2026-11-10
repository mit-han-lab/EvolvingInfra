"""Environment defaults that follow from the symmetric P/D server arguments."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "engine/decoupled_engine/srt/arg_groups/symmetric_defaults.py"
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
}
HOST = {
    "SGLANG_PREBUILT_NONBLOCKING_H2D": "1",
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
    assert env == {**LATE_D_CHOOSE, **HOST, **SNAPSHOT, "SGLANG_DSA_HOST_DEDUP_FANOUT": "ce_peer"}
    assert fields == {"hicache_io_backend": "direct", "hicache_mem_layout": "page_first_direct"}


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
