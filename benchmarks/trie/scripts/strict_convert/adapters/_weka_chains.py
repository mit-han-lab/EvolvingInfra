"""Flattened-agent chain detection for Weka traces from hash_id LCP evidence.

Ported from aiperf ``src/aiperf/dataset/loader/weka_agent_chains.py``
(commit a8e3dc7, Apache-2.0, Copyright NVIDIA CORPORATION & AFFILIATES) and
kept algorithmically identical: phase 1 builds chains greedily (a request that
fully extends a same-model chain tail ending no later than its start is that
chain's next turn; anything else forks a new chain from the deepest LCP tail),
phase 2 splices a fork back onto its source tail as a join seam when the
longer pre-shrink state is never used again. The worker-group / aux
classification of aiperf is not ported; the replay only needs the partition.

A request is a dict with ``t``, ``api_time``, ``hash_ids`` and ``model``.
Every chain's requests are time-ordered and non-overlapping by construction.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EPSILON_SECONDS = 1e-6

Request = dict[str, Any]
IndexedRequest = tuple[int, Request]


def request_end(req: Request) -> float:
    duration = req.get("api_time")
    if duration is None or not math.isfinite(duration):
        duration = 0.0
    return req["t"] + max(duration, 0.0)


def _np_lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    neq = a[:n] != b[:n]
    i = int(neq.argmax())
    return i if neq[i] else n


@dataclass
class ChainFork:
    parent_chain: int | None
    fork_outer_idx: int | None
    depth: int
    fork_time: float


@dataclass
class AgentChain:
    requests: list[IndexedRequest] = field(default_factory=list)
    fork: ChainFork | None = None
    spliced_into: int | None = None
    tail_outer_idx: int = -1
    tail_hash: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    tail_end: float = 0.0
    tail_model: str = ""


@dataclass
class ChainDetectionResult:
    chains: list[AgentChain]
    main_index: int
    worker_indices: list[int]
    seams_merged: int


class _Phase1:
    def __init__(self) -> None:
        self.chains: list[AgentChain] = []
        self.chain_of_request: dict[int, int] = {}
        self.forks_by_tail: dict[int, list[int]] = {}
        self.req_by_outer: dict[int, Request] = {}

    def _append(self, chain_idx: int, outer_idx: int, req: Request) -> None:
        chain = self.chains[chain_idx]
        chain.requests.append((outer_idx, req))
        self.chain_of_request[outer_idx] = chain_idx
        chain.tail_outer_idx = outer_idx
        chain.tail_hash = np.asarray(req["hash_ids"], dtype=np.int64)
        chain.tail_end = request_end(req)
        chain.tail_model = req["model"]

    def classify(self, outer_idx: int, req: Request) -> None:
        if not req["hash_ids"]:
            raise ValueError(f"request {outer_idx} has no hash_ids")
        self.req_by_outer[outer_idx] = req
        h = np.asarray(req["hash_ids"], dtype=np.int64)
        if not self.chains:
            self.chains.append(AgentChain())
            self._append(0, outer_idx, req)
            return
        target = _find_extension_target(self.chains, h, req["t"], req["model"])
        if target is not None:
            self._append(target, outer_idx, req)
            return
        parent, depth = _max_lcp_chain(self.chains, h)
        fork = ChainFork(
            parent_chain=parent,
            fork_outer_idx=self.chains[parent].tail_outer_idx if parent is not None else None,
            depth=depth,
            fork_time=req["t"],
        )
        new_idx = len(self.chains)
        self.chains.append(AgentChain(fork=fork))
        self._append(new_idx, outer_idx, req)
        if fork.fork_outer_idx is not None and depth > 0:
            self.forks_by_tail.setdefault(fork.fork_outer_idx, []).append(new_idx)


def detect_agent_chains(
    requests: list[IndexedRequest],
    *,
    seam_max_gap_seconds: float = 3600.0,
    seam_min_overlap_ratio: float = 0.5,
) -> ChainDetectionResult:
    """Partition requests into per-agent chains (aiperf defaults for the seam guard)."""
    ordered = sorted(requests, key=lambda item: (item[1]["t"], item[0]))
    state = _Phase1()
    for outer_idx, req in ordered:
        state.classify(outer_idx, req)
    chains = state.chains
    seams = _resolve_seams(
        chains,
        state.forks_by_tail,
        state.chain_of_request,
        state.req_by_outer,
        max_gap_seconds=seam_max_gap_seconds,
        min_overlap_ratio=seam_min_overlap_ratio,
    )
    alias = {i: c.spliced_into for i, c in enumerate(chains) if c.spliced_into is not None}

    def resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    for chain in chains:
        if chain.spliced_into is None and chain.fork is not None and chain.fork.parent_chain is not None:
            chain.fork.parent_chain = resolve(chain.fork.parent_chain)
    main_index = resolve(state.chain_of_request[ordered[0][0]])
    workers = [i for i, c in enumerate(chains) if c.spliced_into is None and i != main_index]
    workers.sort(key=lambda i: (chains[i].requests[0][1]["t"], chains[i].requests[0][0]))
    return ChainDetectionResult(chains=chains, main_index=main_index, worker_indices=workers, seams_merged=seams)


def _find_extension_target(chains: list[AgentChain], h: np.ndarray, t: float, model: str) -> int | None:
    best: int | None = None
    best_len = -1
    hn = h.shape[0]
    for idx, chain in enumerate(chains):
        tl = chain.tail_hash.shape[0]
        if tl == 0 or tl > hn or tl <= best_len:
            continue
        if chain.tail_model != model:
            continue
        if chain.tail_end > t + _EPSILON_SECONDS:
            continue
        if chain.tail_hash[tl - 1] != h[tl - 1]:
            continue
        if bool((h[:tl] == chain.tail_hash).all()):
            best, best_len = idx, tl
    return best


def _max_lcp_chain(chains: list[AgentChain], h: np.ndarray) -> tuple[int | None, int]:
    best_idx: int | None = None
    best_key = (0, 0)
    for idx, chain in enumerate(chains):
        if chain.tail_hash.shape[0] == 0:
            continue
        depth = _np_lcp(chain.tail_hash, h)
        if depth == 0:
            continue
        key = (depth, chain.tail_hash.shape[0])
        if key > best_key:
            best_idx, best_key = idx, key
    return best_idx, best_key[0]


def _elect_continuation(
    chains: list[AgentChain],
    registered: list[int],
    tail_req: Request,
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int | None:
    tail_end = request_end(tail_req)
    tail_blocks = len(tail_req["hash_ids"])

    def seam_blocked(ci: int) -> bool:
        gap = chains[ci].requests[0][1]["t"] - tail_end
        overlap = chains[ci].fork.depth / tail_blocks
        return gap > max_gap_seconds and overlap < min_overlap_ratio

    candidates = [
        ci
        for ci in registered
        if chains[ci].fork is not None
        and chains[ci].fork.depth > 0
        and tail_end <= chains[ci].requests[0][1]["t"] + _EPSILON_SECONDS
        and chains[ci].requests[0][1]["model"] == tail_req["model"]
        and not seam_blocked(ci)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda ci: (chains[ci].fork.depth, -chains[ci].fork.fork_time, -ci))


def _rekey_leftover_forks(
    *,
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    req_by_outer: dict[int, Request],
    registered: list[int],
    elected: int,
    owner: int,
    new_tail_outer: int,
) -> bool:
    new_tail_hash = np.asarray(req_by_outer[new_tail_outer]["hash_ids"], np.int64)
    rekeyed = False
    for ci in registered:
        chain = chains[ci]
        if ci == elected or chain.spliced_into is not None or chain.fork is None:
            continue
        depth = _np_lcp(new_tail_hash, np.asarray(chain.requests[0][1]["hash_ids"], dtype=np.int64))
        if depth <= 0:
            continue
        chain.fork.fork_outer_idx = new_tail_outer
        chain.fork.depth = depth
        chain.fork.parent_chain = owner
        forks_by_tail.setdefault(new_tail_outer, []).append(ci)
        rekeyed = True
    return rekeyed


def _resolve_seams(
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    chain_of_request: dict[int, int],
    req_by_outer: dict[int, Request],
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int:
    alias: dict[int, int] = {}

    def resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    seams = 0
    keys = sorted(forks_by_tail)
    heapq.heapify(keys)
    processed: set[int] = set()
    while keys:
        fork_outer_idx = heapq.heappop(keys)
        if fork_outer_idx in processed:
            continue
        processed.add(fork_outer_idx)
        owner = resolve(chain_of_request[fork_outer_idx])
        owner_chain = chains[owner]
        if owner_chain.requests[-1][0] != fork_outer_idx:
            continue  # the longer state was extended, so every fork is a spawn
        registered = [ci for ci in forks_by_tail[fork_outer_idx] if chains[ci].spliced_into is None]
        elected = _elect_continuation(
            chains,
            registered,
            req_by_outer[fork_outer_idx],
            max_gap_seconds=max_gap_seconds,
            min_overlap_ratio=min_overlap_ratio,
        )
        if elected is None:
            continue
        target = chains[elected]
        owner_chain.requests.extend(target.requests)
        for oi, _ in target.requests:
            chain_of_request[oi] = owner
        target.spliced_into = owner
        alias[elected] = owner
        seams += 1
        new_tail_outer = owner_chain.requests[-1][0]
        if _rekey_leftover_forks(
            chains=chains,
            forks_by_tail=forks_by_tail,
            req_by_outer=req_by_outer,
            registered=registered,
            elected=elected,
            owner=owner,
            new_tail_outer=new_tail_outer,
        ):
            processed.discard(new_tail_outer)
            heapq.heappush(keys, new_tail_outer)
    return seams
