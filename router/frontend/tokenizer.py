"""Reuse engine/indexer hashing; never substitute one hash contract for another."""

import hashlib
from .protocol import NodeConfig

class NativeHasher:
    """Wrap the native hash function used by SGLang's get_hash_str wrapper.

    Input: NodeConfig at construction and token IDs when called.
    Output: a callable instance producing ordered prefix-page hash strings.
    """

    def __init__(self, config: NodeConfig):
        """Initialize and warm SGLang's native prefix-page hash implementation.

        The warmup call loads the native extension before serving input.

        Args:
            config: Fixed node configuration including page size and cache salt.

        Returns:
            None; records the hash function, page size, and optional salt seed.
        """
        from ._engine import native_hash

        get_native_hash = native_hash()

        self.function, self.page_size = get_native_hash, config.page_size
        self.seed = (
            hashlib.sha256(b"sglang-cache-salt-v1\0" + config.cache_salt.encode()).digest()
            if config.cache_salt
            else None
        )
        self([1])  # Initialize the native extension before serving requests.

    def __call__(self, ids):
        """Hash ordered token pages using the configured native prefix seed.

        Args:
            ids: Sequence of model token IDs.

        Returns:
            Ordered hexadecimal prefix-page hashes.
        """
        return self.function(ids, self.seed, page_size=self.page_size)


class DynamoHasher:
    """Use the installed indexer's exact block hash implementation and namespace."""

    def __init__(self, config: NodeConfig):
        """Load and validate the installed Dynamo query-hash implementation.

        Runs a startup probe so missing EAGLE or namespace support fails early.

        Args:
            config: Fixed page size, EAGLE mode, namespace, and model configuration.

        Returns:
            None; records the implementation and immutable configuration.

        Raises:
            ValueError: extra_key is configured, which this hash mode does not support.
        """
        from dynamo.llm import compute_block_hash_for_seq

        if config.extra_key:
            raise ValueError("Dynamo hash mode does not support extra_key")
        self.function, self.config = compute_block_hash_for_seq, config
        # Fail at startup if this Dynamo installation lacks EAGLE/namespace support.
        self(list(range(config.page_size + 1)))

    def __call__(self, ids):
        """Compute Dynamo block hashes and express them as signed 64-bit integers.

        Signed conversion changes representation only, not the hash algorithm.

        Args:
            ids: Sequence of model token IDs.

        Returns:
            Ordered signed hash values accepted by query_by_hash.
        """
        values = self.function(
            ids, self.config.page_size,
            is_eagle=self.config.is_eagle,
            cache_namespace=self.config.cache_salt or None,
        )
        # query_by_hash consumes signed i64; this changes representation, not hashing.
        return [value if value < 2**63 else value - 2**64 for value in values]


def hash_spec(config: NodeConfig) -> dict:
    """Describe the configured cache-query hash contract.

    The query contract is separate from physical KV page hashing.

    Args:
        config: Fixed tokenizer-node configuration.

    Returns:
        A dictionary identifying namespace, algorithm, encoding, and block rules.
    """
    common = {
        "mode": config.hash_mode,
        "page_size": config.page_size,
        "model": config.model_path,
        "cache_namespace": config.cache_salt or None,
    }
    if config.image_config is not None:
        common["image_config"] = config.image_config
    if config.hash_mode == "dynamo":
        return {
            **common, "implementation": "dynamo.llm.compute_block_hash_for_seq",
            "encoding": "signed_i64", "is_eagle": config.is_eagle,
            "partial_blocks": False, "window_tokens": config.page_size + int(config.is_eagle),
        }
    return {
        **common, "implementation": "decoupled_engine.get_native_hash", "encoding": "hex",
        "digest": "sha256", "chain": "prefix", "partial_blocks": True,
    }


def hash_count(config: NodeConfig, token_count: int) -> int:
    """Calculate how many query hashes the configured algorithm should produce.

    Args:
        config: Hash mode, page size, and EAGLE settings.
        token_count: Number of prompt tokens to hash.

    Returns:
        Expected nonnegative number of hash blocks.
    """
    if config.hash_mode == "dynamo":
        return max(0, token_count - int(config.is_eagle)) // config.page_size
    return (token_count + config.page_size - 1) // config.page_size
