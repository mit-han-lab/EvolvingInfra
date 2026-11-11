# Temp workaround, make layer utils more fine-grained later
from decoupled_engine.srt.layers.utils.common import *

# Deprecated re-export kept for external users; in-repo code subclasses
# decoupled_engine.kernels.fused_op.BaseFusedOp directly (RFC #29630).
from decoupled_engine.srt.layers.utils.multi_platform import MultiPlatformOp
