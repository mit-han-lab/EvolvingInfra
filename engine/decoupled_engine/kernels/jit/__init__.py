"""Internal JIT home under ``decoupled_engine.kernels`` (RFC #29630).

Mirrors the legacy ``decoupled_engine.kernels.jit`` tree; shared build/runtime
infrastructure lives in :mod:`decoupled_engine.kernels.jit.utils`. csrc / include /
operators migrate here in later phases.
"""
