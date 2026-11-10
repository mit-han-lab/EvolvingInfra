from __future__ import annotations

from typing import Optional

from decoupled_engine.srt.kv_canary.token_oracle.oracle import HashOracle
from decoupled_engine.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
from decoupled_engine.srt.kv_canary.token_oracle.sampler import install_oracle_sampler
from decoupled_engine.srt.runtime_context import get_exec


def install_token_oracle_from_env(*, vocab_size: int) -> Optional[TokenOracleManager]:
    # Must be called before create_sampler() so the factory is present when the
    # Sampler is first constructed.
    if get_exec().kernel.sampling_backend != "token_oracle":
        return None

    oracle = HashOracle(vocab_size=vocab_size)
    return install_oracle_sampler(oracle=oracle)
