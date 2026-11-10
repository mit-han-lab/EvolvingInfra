"""Synthetic-oracle controlled draft-accept rate for DSpark strict replay.

With ``--synthetic-oracle-dspark-accept-rate R`` set, each forced-output
(strict replay) row controls the DSpark linear-block verify path: it
samples this round's correct-draft count from a seeded truncated geometric,
injects exactly that many replay tokens into the drafted block, and corrupts
the next draft position so the trace-authoritative verify
cannot accept past the sampled count even when the natural draft happens to
match the trace.

Ordering contract (see ``DSparkWorkerV2._forward_decode``): injection runs
AFTER the draft forward, the confidence computation, and the ragged-verify
planner (so the confidence head and the verify-length schedule keep acting on
the REAL draft), and BEFORE the target-verify forward (so the committed
positions' KV is written from the injected replay tokens, exactly as a natural
full-hit would).

Determinism contract: the uniform at (request, cursor, depth) depends only on
the request's trace seed (blake2b over forced_output_token_ids) and the replay
cursor (derived from seq_lens), so reruns and cross-arm comparisons see
identical acceptance sequences regardless of batch composition.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

# Two's-complement int64 views of the splitmix64 constants.
_SPLITMIX64_GAMMA = -7046029254386353131  # 0x9E3779B97F4A7C15
_SPLITMIX64_MULT1 = -4658895280553007687  # 0xBF58476D1CE4E5B9
_SPLITMIX64_MULT2 = -7723592293110705685  # 0x94D049BB133111EB
_ORACLE_CURSOR_STRIDE = -5476376624701213403  # 0xB40BC1BE45E92EA5

_UINT64_MASK = (1 << 64) - 1


def solve_chain_accept_prob(accept_rate: float, num_steps: int) -> float:
    """Solve the per-position chain probability for a target draft accept rate.

    With linear-chain drafting, a per-position continue probability ``p``
    yields an expected number of correct drafts of ``sum_{i=1..n} p**i``. This
    inverts that sum so the realized aggregate accept rate
    (``correct_drafts / proposed_drafts``) converges to ``accept_rate``.
    """
    if not 0.0 < accept_rate <= 1.0:
        raise ValueError(f"accept_rate must be in (0, 1]; got {accept_rate}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive; got {num_steps}")
    target = accept_rate * num_steps
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2
        total = 0.0
        power = 1.0
        for _ in range(num_steps):
            power *= mid
            total += power
        if total < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _logical_rshift64(values: torch.Tensor, bits: int) -> torch.Tensor:
    return (values >> bits) & ((1 << (64 - bits)) - 1)


def _splitmix64(values: torch.Tensor) -> torch.Tensor:
    z = values + _SPLITMIX64_GAMMA
    z = (z ^ _logical_rshift64(z, 30)) * _SPLITMIX64_MULT1
    z = (z ^ _logical_rshift64(z, 27)) * _SPLITMIX64_MULT2
    return z ^ _logical_rshift64(z, 31)


def sample_synthetic_oracle_num_correct_drafts(
    seeds: torch.Tensor,
    cursors: torch.Tensor,
    chain_accept_prob: float,
    num_steps: int,
) -> torch.Tensor:
    """Sample per-request correct-draft counts for one verify round.

    Stateless and replay-deterministic: the uniform at (request, cursor, depth)
    depends only on the request's trace seed and the replay cursor, so reruns
    and cross-arm comparisons see identical acceptance sequences. The chain
    stops at the first rejected position (truncated geometric), matching the
    prefix-acceptance shape of natural linear-chain drafting.
    """
    depths = torch.arange(1, num_steps + 1, dtype=torch.int64, device=seeds.device)
    mixed = (
        seeds[:, None]
        + cursors[:, None] * _ORACLE_CURSOR_STRIDE
        + depths[None, :] * _SPLITMIX64_GAMMA
    )
    uniforms = _logical_rshift64(_splitmix64(mixed), 11).to(torch.float64) * (
        1.0 / (1 << 53)
    )
    accept_flags = (uniforms < chain_accept_prob).to(torch.int64)
    return torch.cumprod(accept_flags, dim=1).sum(dim=1)


def _to_int64(value: int) -> int:
    value &= _UINT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


def _splitmix64_cpu(value: int) -> int:
    z = (value + _SPLITMIX64_GAMMA) & _UINT64_MASK
    z = ((z ^ (z >> 30)) * (_SPLITMIX64_MULT1 & _UINT64_MASK)) & _UINT64_MASK
    z = ((z ^ (z >> 27)) * (_SPLITMIX64_MULT2 & _UINT64_MASK)) & _UINT64_MASK
    return z ^ (z >> 31)


def sample_synthetic_oracle_num_correct_drafts_cpu(
    seed: int,
    cursor: int,
    chain_accept_prob: float,
    num_steps: int,
) -> int:
    """Pure-python mirror of the tensor sampler for scheduler-side accounting.

    Must stay bit-identical to ``sample_synthetic_oracle_num_correct_drafts``
    (covered by test_dspark_synthetic_oracle.py::test_cpu_sampler_matches_tensor).
    """
    count = 0
    for depth in range(1, num_steps + 1):
        mixed = _to_int64(
            seed + cursor * _ORACLE_CURSOR_STRIDE + depth * _SPLITMIX64_GAMMA
        )
        uniform = (_splitmix64_cpu(mixed) >> 11) * (1.0 / (1 << 53))
        if uniform >= chain_accept_prob:
            break
        count += 1
    return count


class SyntheticOracleMarkers(msgspec.Struct):
    """Per-round [bs] int64 device markers; -1 on rows without forced output.

    ``eligible_drafts`` is the pre-corruption trace-eligible proposal width
    (including zero at the trace tail) — the accept-rate denominator.
    ``inject_drafts`` is the number of replay tokens actually injected this
    round, which the verifier must accept exactly before the ragged-verify cap
    (``inject_drafts <= eligible_drafts``).
    """

    eligible_drafts: torch.Tensor
    inject_drafts: torch.Tensor


def inject_synthetic_oracle_into_draft_tokens(
    *,
    draft_tokens: torch.Tensor,
    root_tokens: torch.Tensor,
    sampling_info,
    chain_accept_prob: float,
    gamma: int,
) -> Optional[SyntheticOracleMarkers]:
    """Overwrite forced rows of the drafted block with sampled replay prefixes.

    ``draft_tokens`` is the (bs, gamma) drafted block WITHOUT the root anchor
    column; it is mutated in place so both the target-verify forward and
    ``BuildOutTokens`` see the injected tokens. The root anchor is the
    previously committed trace token and is left untouched. Replay tokens are
    injected only while at least one trace token remains for the verifier's
    bonus slot, so trace-tail bookkeeping stays identical to natural replay.

    Returns ``None`` when the batch carries no forced-output state.
    """
    active = sampling_info.forced_output_active_mask()
    if active is None:
        return None
    # Greedy is enforced per active row (device assert), not via the
    # batch-global is_all_greedy flag: a non-greedy NON-forced co-request is a
    # supported batch shape (mixed accept) and must not crash the scheduler.
    # Non-greedy requests are additionally rejected at admission on
    # synthetic-oracle servers (scheduler handle path).
    torch._assert_async(
        torch.all((~active) | (sampling_info.top_ks <= 1)),
        "synthetic oracle DSpark requires greedy (top_k<=1) forced rows",
    )
    if sampling_info.forced_output_seeds is None:
        raise ValueError("synthetic oracle DSpark requires forced_output_seeds")
    if draft_tokens.shape[1] != gamma:
        raise ValueError(
            f"drafted block width {draft_tokens.shape[1]} != gamma {gamma}"
        )

    remaining = (
        sampling_info.forced_output_lengths - sampling_info.forced_output_offsets
    )
    eligible = torch.clamp(remaining - 1, min=0, max=gamma)
    sampled_num_correct_drafts = sample_synthetic_oracle_num_correct_drafts(
        sampling_info.forced_output_seeds,
        sampling_info.forced_output_offsets,
        chain_accept_prob,
        gamma,
    )
    injected = torch.minimum(eligible, sampled_num_correct_drafts)

    markers = SyntheticOracleMarkers(
        eligible_drafts=torch.where(
            sampling_info.forced_output_mask,
            eligible,
            torch.full_like(eligible, -1),
        ),
        inject_drafts=torch.where(
            sampling_info.forced_output_mask,
            injected,
            torch.full_like(injected, -1),
        ),
    )
    if gamma == 0 or sampling_info.forced_output_ids.shape[1] == 0:
        return markers

    torch._assert_async(
        torch.all((~active) | (sampling_info.forced_output_offsets > 0)),
        "synthetic oracle DSpark requires a committed forced-output root token",
    )
    root_indices = torch.clamp(
        sampling_info.forced_output_offsets - 1,
        min=0,
        max=sampling_info.forced_output_ids.shape[1] - 1,
    )
    expected_roots = sampling_info.forced_output_ids.gather(
        1, root_indices[:, None]
    ).squeeze(1)
    torch._assert_async(
        torch.all((~active) | (root_tokens == expected_roots)),
        "synthetic oracle DSpark root candidate does not match the committed "
        "forced-output token",
    )

    depths = torch.arange(
        gamma,
        dtype=sampling_info.forced_output_offsets.dtype,
        device=draft_tokens.device,
    )
    trace_indices = sampling_info.forced_output_offsets[:, None] + depths
    safe_indices = torch.minimum(
        trace_indices,
        torch.clamp_min(sampling_info.forced_output_lengths[:, None] - 1, 0),
    )
    oracle_tokens = sampling_info.forced_output_ids.gather(1, safe_indices)
    inject_mask = active[:, None] & (depths[None, :] < injected[:, None])
    draft_tokens[:] = torch.where(
        inject_mask,
        oracle_tokens.to(draft_tokens.dtype),
        draft_tokens,
    )
    # Guarantee rejection at the first non-injected position: a natural draft
    # there may coincidentally match the trace token, which would drift the
    # realized accept rate above the target and violate the verify-side
    # correct_drafts == inject_drafts invariant. Corrupting with a valid but
    # different vocab id keeps the verify embedding lookup in range; the
    # candidate is rejected so it is never committed.
    corrupt_mask = (
        active[:, None]
        & (depths[None, :] == injected[:, None])
        & (depths[None, :] < eligible[:, None])
    )
    corrupted = torch.where(
        oracle_tokens >= 1, oracle_tokens - 1, oracle_tokens + 1
    )
    draft_tokens[:] = torch.where(
        corrupt_mask,
        corrupted.to(draft_tokens.dtype),
        draft_tokens,
    )
    return markers
