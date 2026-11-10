import pytest

from trie.prefix import (
    GlobalPrefixRegistry,
    block_aligned_prefix_length,
    common_prefix_length,
)


def test_common_prefix_length() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 4, 5]) == 2
    assert common_prefix_length([], [1]) == 0
    assert common_prefix_length([1, 2], [1, 2]) == 2


def test_block_aligned_prefix_length() -> None:
    assert block_aligned_prefix_length(35, 16) == 32
    assert block_aligned_prefix_length(15, 16) == 0
    with pytest.raises(ValueError, match="block_size"):
        block_aligned_prefix_length(10, 0)


def test_global_prefix_registry_empty_and_validation() -> None:
    with pytest.raises(ValueError, match="block_size"):
        GlobalPrefixRegistry(0)
    registry = GlobalPrefixRegistry(4)
    assert registry.match_tokens([1, 2, 3, 4, 5]) == 0


def test_global_prefix_registry_block_aligned_match() -> None:
    registry = GlobalPrefixRegistry(4)
    registry.insert(list(range(10)))  # blocks [0..3], [4..7]; tail 8,9 dropped
    # Full first block reusable; 5th token is a miss for the second block.
    assert registry.match_tokens([0, 1, 2, 3, 99, 98, 97, 96, 95]) == 4
    # Both stored blocks match; the trailing partial block never counts.
    assert registry.match_tokens(list(range(9))) == 8


def test_global_prefix_registry_final_token_clamp() -> None:
    registry = GlobalPrefixRegistry(4)
    registry.insert(list(range(8)))
    # An 8-token prompt has only 7 reusable tokens -> one aligned block.
    assert registry.match_tokens(list(range(8))) == 4
    assert registry.match_tokens(list(range(9))) == 8


def test_global_prefix_registry_contiguity_required() -> None:
    registry = GlobalPrefixRegistry(4)
    registry.insert(list(range(4)))
    registry.insert([100, 101, 102, 103] + list(range(4, 8)))
    # [4..7] exists in the registry but only behind the [100..103] chain;
    # after matching [0..3] the walk must stop, not skip ahead.
    assert registry.match_tokens(list(range(12))) == 4


def test_global_prefix_registry_cross_session_reuse() -> None:
    registry = GlobalPrefixRegistry(4)
    shared_context = list(range(16))
    registry.insert(shared_context)
    # A different "session" sharing the same rendered prefix hits globally,
    # which is exactly what the one-engine/infinite-cache bound should count.
    assert registry.match_tokens(shared_context + [42, 43]) == 16
