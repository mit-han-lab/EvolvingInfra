"""Embedding kernels."""

from decoupled_engine.kernels.registry import register_kernel
from decoupled_engine.kernels.spec import KernelBackend, KernelSpec

register_kernel(
    KernelSpec(
        op="embeddings.vocab_parallel_embedding",
        backend=KernelBackend.TRITON,
        target=(
            "decoupled_engine.kernels.ops.embeddings.vocab_parallel_embedding:"
            "vocab_parallel_embedding"
        ),
    )
)

__all__ = []
