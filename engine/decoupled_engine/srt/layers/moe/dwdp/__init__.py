"""DWDP (Distributed Weight Data Parallelism): MoE prefill with tokens kept on-rank and peer expert weights prefetched via NVLink into a composite VMM address space."""

from decoupled_engine.srt.layers.moe.dwdp.dwdp_manager import DwdpManager
from decoupled_engine.srt.runtime_context import (
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)

__all__ = [
    "DwdpManager",
    "get_global_dwdp_manager",
    "set_global_dwdp_manager",
]
