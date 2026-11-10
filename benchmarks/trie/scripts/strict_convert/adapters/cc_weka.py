"""Adapter: AgentX Claude Code Weka traces -> schema-v3 tasks for strict replay.

A source row is one session: top-level requests (``type`` ``n``/``s``) and
``subagent`` entries with their inner requests, each carrying only timing,
``in``/``out`` counts and ``hash_ids`` (one per 64 input tokens, session-local).
Each session becomes one v3 task with one agent per chain. The design is
described in benchmarks/trie/docs/cc-trace-replay.md.

* Chains: the LCP chain detection (``_weka_chains``) runs once on the
  top-level stream and once inside every sub-agent, so no chain holds two
  requests that overlap in recorded time. The chain owning a stream's first
  request is ``main`` (or ``sa:<agent_id>``); the others are derived chains.
* Parent: a derived chain's parent is the chain it forked from. A sub-agent,
  or a derived chain that shares no context, hangs on the chain of the last
  request of the enclosing stream that completed before it started (the
  agent that issued it), else on the stream's main chain. A session's first
  request is often a small one-shot, so its chain is not the working agent.
* Spawn: a child starts after the last parent request that had completed when
  it started (an edge from that generate to the child's first event), or with
  its parent if none had.
* Join: if the parent's next request after the child started began after the
  child's subtree ended, that request waits for the child (blocking);
  otherwise the child is background and joins before the first parent
  request that began after it ended, or not at all (the task still waits for
  every agent). A join is an edge from the last event of the child, and of
  every descendant joined at the child's end, to the first event of the
  parent's turn.
* Waits: only the request's recorded ``think_time`` precedes it; the source
  measures it from the end of the previous event, sub-agents included.
* Tokens: prompts are runs of the recorded hash IDs and every output gets
  fresh block IDs; the replay client expands a block into 64 tokens drawn from
  the task's ``block_tokens`` (a per-session seed and the target's regular
  vocabulary) by ``trie.blocks``, so equal IDs give equal tokens across the
  session and the recorded prefix relations hold exactly at 64-token
  granularity. Every agent of a task shares one cache salt.
* Sequential agents: a source row with ``"sequential_agents": true`` (Codex
  sessions converted by ``scripts/codex_to_ccweka.py``; ``"single_agent"`` is
  the older name) skips chain detection: the top-level requests are one
  ``main`` chain and every sub-agent one ``sa:<agent_id>`` chain, each in
  recorded order and without overlaps. A sub-agent hangs on the agent named
  by its ``parent_agent_id`` (default: main); spawns and joins, the one cache
  salt and the session routing key are as above.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from strict_convert.adapters._weka_chains import detect_agent_chains, request_end

BLOCK_TOKENS = 64
DEFAULT_SEED = "cc-weka-v1"
_EPSILON_SECONDS = 1e-6


@dataclass
class Chain:
    chain_id: str
    kind: str  # main | derived | subagent
    parent: str | None
    requests: list[dict[str, Any]]
    children: list[str] = field(default_factory=list)
    anchor: int | None = None  # spawned after parent.requests[anchor]; None = at parent start
    join_before: int | None = None  # joined before parent.requests[join_before]; None = at parent end
    join_mode: str | None = None  # blocking | background | end
    subtree_end: float = 0.0

    @property
    def start(self) -> float:
        return self.requests[0]["t"]


@dataclass
class Session:
    session_id: str
    chains: dict[str, Chain]
    source_requests: int
    seams: int


def _source_request(req: dict[str, Any], request_id: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "t": float(req["t"]),
        "api_time": float(req.get("api_time") or 0.0),
        "think_time": float(req.get("think_time") or 0.0),
        "hash_ids": list(req["hash_ids"]),
        "in": int(req["in"]),
        "out": int(req["out"]),
        "model": req["model"],
    }


def _detect(stream: list[dict[str, Any]], main_id: str, main_kind: str,
            derived_prefix: str) -> tuple[list[Chain], int]:
    """Chains of one stream; ``parent`` is set only for forks with shared context."""
    result = detect_agent_chains(list(enumerate(stream)))
    live = [result.main_index, *result.worker_indices]
    ids = {result.main_index: main_id}
    for number, index in enumerate(result.worker_indices):
        ids[index] = f"{derived_prefix}fa:{number:03d}"
    chains = []
    for index in live:
        source = result.chains[index]
        parent = None
        if index != result.main_index:
            fork = source.fork
            if fork is not None and fork.depth > 0 and fork.parent_chain in ids:
                parent = ids[fork.parent_chain]
        ordered = sorted(source.requests, key=lambda item: (item[1]["t"], item[0]))
        chains.append(Chain(chain_id=ids[index], kind=main_kind if index == result.main_index else "derived",
                            parent=parent, requests=[req for _, req in ordered]))
    return chains, result.seams_merged


def _issuer(stream_chains: list[Chain], start: float, default: str, exclude: str) -> str:
    """Chain of the stream's latest request that completed by ``start``."""
    best, best_end = default, None
    for chain in stream_chains:
        if chain.chain_id == exclude:
            continue
        for req in chain.requests:
            end = request_end(req)
            if end <= start + _EPSILON_SECONDS and (best_end is None or end > best_end):
                best, best_end = chain.chain_id, end
    return best


def build_session(row: dict[str, Any]) -> Session:
    """Split one source session into chains and place every spawn and join."""
    session_id = row["id"]
    top: list[dict[str, Any]] = []
    subagents: list[tuple[str, list[dict[str, Any]]]] = []
    parents: dict[str, str] = {}
    count = 0
    for outer, entry in enumerate(row["requests"]):
        if entry["type"] == "subagent":
            inner = [_source_request(req, f"r{outer}.{i}") for i, req in enumerate(entry["requests"])]
            if inner:
                subagents.append((entry["agent_id"], inner))
                if entry.get("parent_agent_id"):
                    parents[entry["agent_id"]] = entry["parent_agent_id"]
            count += len(inner)
        else:
            top.append(_source_request(entry, f"r{outer}"))
            count += 1
    if row.get("sequential_agents") or row.get("single_agent"):
        return _sequential_session(session_id, top, subagents, parents, count)
    top_chains, seams = _detect(top, "main", "main", "")
    for chain in top_chains[1:]:
        if chain.parent is None:
            chain.parent = _issuer(top_chains, chain.start, "main", chain.chain_id)
    chains = list(top_chains)
    for agent_id, inner in subagents:
        sa_id = f"sa:{agent_id}"
        sub_chains, sub_seams = _detect(inner, sa_id, "subagent", f"{sa_id}:")
        sub_chains[0].parent = _issuer(top_chains, sub_chains[0].start, "main", "")
        for chain in sub_chains[1:]:
            if chain.parent is None:
                chain.parent = _issuer(sub_chains, chain.start, sa_id, chain.chain_id)
        chains.extend(sub_chains)
        seams += sub_seams
    by_id = {chain.chain_id: chain for chain in chains}
    if len(by_id) != len(chains):
        raise ValueError(f"{session_id}: duplicate chain ids")
    for chain in chains:
        if chain.parent is not None:
            by_id[chain.parent].children.append(chain.chain_id)
    _subtree_end(by_id, "main")
    for chain in chains:
        if chain.parent is not None:
            _place(chain, by_id[chain.parent])
    return Session(session_id=session_id, chains=by_id, source_requests=count, seams=seams)


def _sequential_session(session_id: str, top: list[dict[str, Any]],
                        subagents: list[tuple[str, list[dict[str, Any]]]], parents: dict[str, str],
                        count: int) -> Session:
    """A source row marked ``sequential_agents`` (e.g. a converted Codex session): every
    agent is one chain in recorded order, with no LCP chain detection. A prompt that drops
    part of the previous one (compaction, dropped reasoning) is the same agent's next turn,
    not a fork."""
    streams = [("main", "main", top)] + [(f"sa:{agent_id}", "subagent", inner) for agent_id, inner in subagents]
    chains: dict[str, Chain] = {}
    for chain_id, kind, requests in streams:
        if chain_id in chains:
            raise ValueError(f"{session_id}: duplicate chain id {chain_id}")
        if not requests:
            raise ValueError(f"{session_id}: {chain_id} has no requests")
        for prev, cur in zip(requests, requests[1:]):
            if cur["t"] < request_end(prev) - _EPSILON_SECONDS:
                raise ValueError(f"{session_id}: {chain_id} request {cur['id']} starts before {prev['id']} ends")
        chains[chain_id] = Chain(chain_id=chain_id, kind=kind, parent=None, requests=requests)
    for agent_id, _ in subagents:
        chain = chains[f"sa:{agent_id}"]
        parent = parents.get(agent_id)
        chain.parent = f"sa:{parent}" if parent and f"sa:{parent}" in chains else "main"
    # A parent chain that loops back to its child would never start.
    for chain in chains.values():
        seen, cur = set(), chain
        while cur.parent is not None:
            if cur.chain_id in seen:
                raise ValueError(f"{session_id}: sub-agent parents form a cycle at {cur.chain_id}")
            seen.add(cur.chain_id)
            cur = chains[cur.parent]
    for chain in chains.values():
        if chain.parent is not None:
            chains[chain.parent].children.append(chain.chain_id)
    _subtree_end(chains, "main")
    for chain in chains.values():
        if chain.parent is not None:
            _place(chain, chains[chain.parent])
    return Session(session_id=session_id, chains=chains, source_requests=count, seams=0)


def _subtree_end(chains: dict[str, Chain], chain_id: str) -> float:
    chain = chains[chain_id]
    end = max(request_end(req) for req in chain.requests)
    for child in chain.children:
        end = max(end, _subtree_end(chains, child))
    chain.subtree_end = end
    return end


def _place(child: Chain, parent: Chain) -> None:
    start = child.start
    anchor = None
    for index, req in enumerate(parent.requests):
        if request_end(req) <= start + _EPSILON_SECONDS:
            anchor = index
    first = -1 if anchor is None else anchor
    following = [i for i in range(first + 1, len(parent.requests)) if parent.requests[i]["t"] >= start - _EPSILON_SECONDS]
    child.anchor = anchor
    if not following:
        child.join_before, child.join_mode = None, "end"
        return
    after_end = [i for i in following if parent.requests[i]["t"] >= child.subtree_end - _EPSILON_SECONDS]
    if after_end and after_end[0] == following[0]:
        child.join_before, child.join_mode = following[0], "blocking"
    else:
        child.join_before, child.join_mode = (after_end[0] if after_end else None), "background"


# ---------------------------------------------------------------- tokens


def allowed_token_ids(tokenizer_dir: str) -> np.ndarray:
    """Regular vocabulary IDs: everything except added and special tokens."""
    data = json.loads((Path(tokenizer_dir) / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    ids = set(vocab.values() if isinstance(vocab, dict) else range(len(vocab)))
    ids -= {token["id"] for token in data.get("added_tokens", [])}
    # Some models declare PAD/BOS/EOS only in config.json, not added_tokens.
    config_path = Path(tokenizer_dir) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = config.get(name)
            ids.difference_update(value if isinstance(value, list) else [value])
        if "vocab_size" in config:
            ids = {token for token in ids if 0 <= token < config["vocab_size"]}
    return np.array(sorted(ids), dtype=np.int64)


def block_seed(seed: str, session_id: str) -> int:
    """64-bit seed of a session's block content; the replay mixes in each cycle."""
    return int.from_bytes(hashlib.blake2b(f"{seed}|{session_id}".encode("utf-8"), digest_size=8).digest(), "little")


# ---------------------------------------------------------------- agents


@dataclass
class AgentLayout:
    """One chain as a schema-v2 agent: events plus where each turn starts."""

    events: list[dict[str, Any]]
    turn_first_event: list[int]  # first event of turn i (its wait, else its generate)
    generate_event: list[int]  # event index of generate i


def agent_layouts(session: Session, generate: Any) -> dict[str, AgentLayout]:
    """Per chain: only ``think_time`` precedes a turn; ``generate(chain, i)`` builds generate i."""
    layouts = {}
    for chain_id, chain in session.chains.items():
        events: list[dict[str, Any]] = []
        firsts, generates = [], []
        for index, req in enumerate(chain.requests):
            firsts.append(len(events))
            if req["think_time"] > 0:
                events.append({"type": "wait", "seconds": req["think_time"], "actor": "other"})
            generates.append(len(events))
            events.append(generate(chain, index))
        layouts[chain_id] = AgentLayout(events, firsts, generates)
    return layouts


def _spawn_source(session: Session, layouts: dict[str, AgentLayout], chain: Chain) -> tuple[str, int] | None:
    """Event that must complete before the chain starts; None starts it with the task."""
    while chain.parent is not None:
        parent = session.chains[chain.parent]
        if chain.anchor is not None:
            return parent.chain_id, layouts[parent.chain_id].generate_event[chain.anchor]
        chain = parent  # spawned at the parent's start: inherit the parent's gate
    return None


def completion_refs(session: Session, layouts: dict[str, AgentLayout], chain: Chain) -> list[tuple[str, int]]:
    """Events whose completion means the chain and every child joined at its end are done."""
    refs = [(chain.chain_id, len(layouts[chain.chain_id].events) - 1)]
    for child_id in chain.children:
        child = session.chains[child_id]
        if child.join_before is None:
            refs.extend(completion_refs(session, layouts, child))
    return refs


def dependencies(session: Session, layouts: dict[str, AgentLayout]) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Completion edges: every spawn gate and every join; the task itself waits for all agents."""
    edges = set()
    for chain in session.chains.values():
        if chain.parent is None:
            continue
        source = _spawn_source(session, layouts, chain)
        if source is not None:
            edges.add((source, (chain.chain_id, 0)))
        if chain.join_before is not None:
            target = (chain.parent, layouts[chain.parent].turn_first_event[chain.join_before])
            for ref in completion_refs(session, layouts, chain):
                edges.add((ref, target))
    return sorted(edges)


def _agent_order(session: Session) -> list[str]:
    return sorted(session.chains, key=lambda c: (session.chains[c].parent is not None, session.chains[c].start, c))


# ---------------------------------------------------------------- checks


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def check_session(session: Session, row: dict[str, Any], blocks: dict[str, list[int]]) -> dict[str, Any]:
    """Offline checks of one converted session (a v3 task row) against its source timing and hashes.

    ``blocks`` maps each source request id to the block IDs its prompt runs
    expand to. With distinct blocks giving distinct tokens (checked by the
    caller), block prefixes are token prefixes.
    """
    errors: list[str] = []
    chains = session.chains
    ids = [req["id"] for chain in chains.values() for req in chain.requests]
    if len(ids) != len(set(ids)) or len(ids) != session.source_requests:
        errors.append(f"requests covered {len(set(ids))}/{session.source_requests} (duplicates {len(ids) - len(set(ids))})")
    overlaps = 0
    for chain in chains.values():
        for prev, cur in zip(chain.requests, chain.requests[1:]):
            if cur["t"] < request_end(prev) - _EPSILON_SECONDS:
                overlaps += 1
                errors.append(f"{chain.chain_id}: {cur['id']} starts before {prev['id']} ends")
    for chain in chains.values():
        if chain.parent is None:
            continue
        parent = chains[chain.parent]
        if chain.anchor is not None and request_end(parent.requests[chain.anchor]) > chain.start + _EPSILON_SECONDS:
            errors.append(f"{chain.chain_id}: spawned after {parent.requests[chain.anchor]['id']}, which ends after the child starts")
        later = range(0 if chain.anchor is None else chain.anchor + 1, len(parent.requests))
        if any(request_end(parent.requests[i]) <= chain.start + _EPSILON_SECONDS for i in later):
            errors.append(f"{chain.chain_id}: a later parent request had already completed when the child started")
        stop = len(parent.requests) if chain.join_before is None else chain.join_before
        for i in later:
            req = parent.requests[i]
            if i < stop and req["t"] >= chain.start - _EPSILON_SECONDS and req["t"] >= chain.subtree_end - _EPSILON_SECONDS:
                errors.append(f"{chain.chain_id}: parent {req['id']} began after the child ended but does not wait for it")
        if chain.join_before is not None and parent.requests[chain.join_before]["t"] < chain.subtree_end - _EPSILON_SECONDS:
            errors.append(f"{chain.chain_id}: joined before {parent.requests[chain.join_before]['id']}, which began before the child ended")

    # The row's agents and edges must be exactly the split's, and no edge may
    # order two events against the recorded times.
    agents = {trace["trace_id"]: trace for trace in row["traces"]}
    if set(agents) != set(chains):
        errors.append(f"task agents {sorted(set(agents) ^ set(chains))} differ from the split")
    by_request = {}
    for chain_id, trace in agents.items():
        chain = chains.get(chain_id)
        generates = [(i, e["source_request_id"]) for i, e in enumerate(trace["events"]) if e["type"] == "generate"]
        if chain is not None and [rid for _, rid in generates] != [req["id"] for req in chain.requests]:
            errors.append(f"{chain_id}: agent generate order differs from the chain")
        for index, rid in generates:
            by_request[(chain_id, index)] = rid
    layouts = {chain_id: AgentLayout(agents[chain_id]["events"], [], []) for chain_id in agents}
    for chain_id, layout in layouts.items():
        turn = 0
        for index, event in enumerate(layout.events):
            if event["type"] == "generate":
                layout.generate_event.append(index)
                turn += 1
        layout.turn_first_event.extend(
            [0] + [layout.generate_event[i] + 1 for i in range(len(layout.generate_event) - 1)]
        )
    requests_by_id = {req["id"]: req for chain in chains.values() for req in chain.requests}
    got = {((e["before"]["trace_id"], e["before"]["event_index"]), (e["after"]["trace_id"], e["after"]["event_index"]))
           for e in row["dependencies"]}
    if set(chains) == set(agents):
        expected = set(dependencies(session, layouts))
        if got != expected:
            errors.append(f"dependencies differ from the split: {len(got - expected)} extra, {len(expected - got)} missing")
    for (a_id, a_index), (b_id, b_index) in sorted(got):
        before = requests_by_id.get(by_request.get((a_id, a_index), ""))
        after_index = next((i for i in layouts.get(b_id, AgentLayout([], [], [])).generate_event if i >= b_index), None)
        after = requests_by_id.get(by_request.get((b_id, after_index), "")) if after_index is not None else None
        if before is None or after is None:
            errors.append(f"dependency ({a_id},{a_index})->({b_id},{b_index}) does not join two generates")
        elif after["t"] < request_end(before) - _EPSILON_SECONDS:
            errors.append(f"dependency {before['id']}->{after['id']} contradicts the recorded order")

    # Prefix hits: each prompt must be exactly its recorded hash IDs as block IDs,
    # so every block LCP (consecutive chain requests, session-wide seen set)
    # equals the hash LCP; the caller checks that distinct IDs give distinct tokens.
    by_id = {req["id"]: req for chain in chains.values() for req in chain.requests}
    for req in by_id.values():
        ids = blocks.get(req["id"])
        if ids is None or len(ids) * BLOCK_TOKENS != req["in"]:
            errors.append(f"{req['id']}: prompt has {len(ids or []) * BLOCK_TOKENS} tokens, source in={req['in']}")
        elif ids != req["hash_ids"]:
            errors.append(f"{req['id']}: prompt block IDs differ from the recorded hash IDs")
    chained = mismatched = 0
    for chain in chains.values():
        for prev, cur in zip(chain.requests, chain.requests[1:]):
            chained += 1
            if _lcp(blocks.get(prev["id"], []), blocks.get(cur["id"], [])) != _lcp(prev["hash_ids"], cur["hash_ids"]):
                mismatched += 1
    ideal_hash = ideal_block = 0
    seen_hash: set[bytes] = set()
    seen_block: set[bytes] = set()
    for req in sorted(by_id.values(), key=lambda r: (r["t"], r["id"])):
        ideal_hash += _prefix_hit(_hash_digests(req["hash_ids"]), seen_hash)
        ideal_block += _prefix_hit(_hash_digests(blocks.get(req["id"], [])), seen_block)
    if mismatched:
        errors.append(f"{mismatched}/{chained} consecutive chain prompts differ from the hash LCP")
    if ideal_hash != ideal_block:
        errors.append(f"session prefix hit blocks: prompts {ideal_block} vs hashes {ideal_hash}")
    return {
        "session": session.session_id,
        "requests": session.source_requests,
        "chains": {kind: sum(1 for c in chains.values() if c.kind == kind) for kind in ("main", "subagent", "derived")},
        "join_modes": _join_modes(session),
        "seams_merged": session.seams,
        "chain_overlaps": overlaps,
        "chained_pairs_checked": chained,
        "session_prefix_hit_blocks": {"hash": ideal_hash, "prompt": ideal_block,
                                      "of": sum(len(req["hash_ids"]) for req in by_id.values())},
        "errors": errors,
    }


def _hash_digests(hash_ids: list[int]) -> Iterator[bytes]:
    digest = b""
    for hash_id in hash_ids:
        digest = hashlib.blake2b(digest + int(hash_id).to_bytes(8, "little", signed=True), digest_size=16).digest()
        yield digest


def _prefix_hit(digests: Iterator[Any], seen: set) -> int:
    """Blocks of this prompt already in ``seen`` as a contiguous prefix; then add them all."""
    hit, contiguous = 0, True
    for digest in digests:
        if contiguous and digest in seen:
            hit += 1
        else:
            contiguous = False
        seen.add(digest)
    return hit


# ---------------------------------------------------------------- rows

# convert_unit yields finished schema-v2 rows, not canonical traces.
ROWS = True
_ID_RE = re.compile(rb'"id"\s*:\s*"([^"]+)"')


def list_units(options: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    units: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in options["inputs"]:
        offset = 0
        with open(path, "rb") as source:
            for line_number, line in enumerate(source):
                if line.strip():
                    match = _ID_RE.search(line[:4096])
                    units.append({"name": f"{path}:{line_number}", "path": path, "offset": offset,
                                  "trace_id": match.group(1).decode() if match else "", "bytes": len(line)})
                offset += len(line)
        if not units:
            warnings.append(f"no sessions found in {path}")
    return units, warnings


def prepare_options(options: dict[str, Any], units: list[dict[str, Any]]) -> None:
    del options, units


def make_context(options: dict[str, Any], renderer: Any) -> dict[str, Any]:
    return {"pool": allowed_token_ids(options["target_tokenizer"]).tolist(),
            "seed": options.get("seed_salt") or DEFAULT_SEED,
            "identity": renderer.identity_summary()}


def load_source(unit: dict[str, Any]) -> dict[str, Any]:
    with open(unit["path"], "rb") as source:
        source.seek(unit["offset"])
        return json.loads(source.readline())


def _runs(block_ids: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for block in block_ids:
        if runs and runs[-1][0] + runs[-1][1] == block:
            runs[-1][1] += 1
        else:
            runs.append([block, 1])
    return runs


def build_row(session: Session, pool: list[int], seed: str, identity: dict[str, Any],
              counters: dict[str, int]) -> dict[str, Any]:
    """One schema-v3 task: a chain per agent, their completion edges, and hash-block runs.

    Prompts are runs of the recorded hash IDs; each output gets fresh block IDs above the
    session's largest hash ID. Every agent inherits the task's ``block_tokens``, so equal
    IDs give equal tokens across the whole session and each replay cycle gets new content.
    """
    requests = sorted((req for chain in session.chains.values() for req in chain.requests),
                      key=lambda req: (req["t"], req["id"]))
    next_block = 1 + max(h for req in requests for h in req["hash_ids"])
    output_runs: dict[str, list[list[int]]] = {}
    for req in requests:
        count = -(-max(req["out"], 1) // BLOCK_TOKENS)
        output_runs[req["id"]] = [[next_block, count]]
        next_block += count

    def generate(chain: Chain, index: int) -> dict[str, Any]:
        req = chain.requests[index]
        max_tokens = req["out"]
        if max_tokens < 1:
            counters["zero_output_raised_to_one"] = counters.get("zero_output_raised_to_one", 0) + 1
            max_tokens = 1
        return {
            "type": "generate",
            "max_tokens": max_tokens,
            "recorded_prompt_block_runs": _runs(req["hash_ids"]),
            "recorded_output_block_runs": output_runs[req["id"]],
            "source_request_id": req["id"],
            "source_model": req["model"],
            "source_t_s": req["t"],
            "source_api_time_s": req["api_time"],
        }

    layouts = agent_layouts(session, generate)
    edges = dependencies(session, layouts)
    chains = session.chains.values()
    summary = {
        "adapter": "cc_weka",
        "seed": seed,
        "target_tokenizer": identity,
        "requests": session.source_requests,
        "prompt_blocks": len({h for req in requests for h in req["hash_ids"]}),
        "chains": {kind: sum(1 for c in chains if c.kind == kind) for kind in ("main", "subagent", "derived")},
        "join_modes": _join_modes(session),
        "seams_merged": session.seams,
    }
    agents = [
        {
            "schema_version": 2,
            "trace_id": chain_id,
            "events": layouts[chain_id].events,
            # One routing key per session keeps sticky routing at session scope.
            "metadata": {"routing_key": session.session_id, "kind": session.chains[chain_id].kind,
                         "parent": session.chains[chain_id].parent},
        }
        for chain_id in _agent_order(session)
    ]
    return {
        "schema_version": 3,
        "trace_id": session.session_id,
        # Sub-agents reuse each other's and the parent's prefixes; one salt per session keeps those hits.
        "cache_salt_scope": "task",
        "traces": agents,
        "dependencies": [
            {"before": {"trace_id": a[0], "event_index": a[1]}, "after": {"trace_id": b[0], "event_index": b[1]}}
            for a, b in edges
        ],
        "start_offset_s": 0.0,
        "metadata": {
            "block_tokens": {"block_size": BLOCK_TOKENS, "seed": block_seed(seed, session.session_id), "pool": pool},
            "strict_conversion": summary,
        },
    }


def _join_modes(session: Session) -> dict[str, int]:
    modes: dict[str, int] = {}
    for chain in session.chains.values():
        if chain.parent is not None:
            key = f"{chain.kind}:{chain.join_mode}"
            modes[key] = modes.get(key, 0) + 1
    return dict(sorted(modes.items()))


def convert_unit(unit: dict[str, Any], context: dict[str, Any], options: dict[str, Any],
                 stats: dict[str, Any]) -> Iterator[dict[str, Any]]:
    session = build_session(load_source(unit))
    counters = stats.setdefault("adapter_counters", {})
    row = build_row(session, context["pool"], context["seed"], context["identity"], counters)
    if options.get("cc_timing") == "causal-gaps":
        from strict_convert.adapters.cc_timing import apply_policy
        row = apply_policy(session, row, zero_output=options.get("cc_zero_output", "raise-to-one"))
        removed = len(row["metadata"]["cc_replay_policy"]["removed_requests"])
        counters["zero_output_raised_to_one"] = counters.get("zero_output_raised_to_one", 0) - removed
        counters["zero_output_removed"] = counters.get("zero_output_removed", 0) + removed
    elif options.get("cc_zero_output") == "drop-advance":
        raise ValueError("drop-advance requires causal-gaps timing")
    generates = session.source_requests
    retained = [e for t in row["traces"] for e in t["events"] if e["type"] == "generate"]
    stats["traces_in"] += 1
    stats["generates_in"] += generates
    stats["traces_out"] += 1
    stats["generates_out"] += len(retained)
    stats["prompt_tokens"] += sum(count * BLOCK_TOKENS for e in retained for _, count in e["recorded_prompt_block_runs"])
    stats["output_tokens"] += sum(e["max_tokens"] for e in retained)
    for key, count in row["metadata"]["strict_conversion"]["join_modes"].items():
        counters[f"join.{key}"] = counters.get(f"join.{key}", 0) + count
    for key, count in row["metadata"]["strict_conversion"]["chains"].items():
        counters[f"chains.{key}"] = counters.get(f"chains.{key}", 0) + count
    counters["seams_merged"] = counters.get("seams_merged", 0) + session.seams
    counters["prompt_blocks"] = counters.get("prompt_blocks", 0) + row["metadata"]["strict_conversion"]["prompt_blocks"]
    yield row


def row_blocks(row: dict[str, Any], key: str = "recorded_prompt_block_runs") -> dict[str, list[int]]:
    """Every generate's prompt (or output) as the block IDs its runs expand to."""
    return {
        event["source_request_id"]: [
            start + offset for start, count in event[key] for offset in range(count)
        ]
        for trace in row["traces"]
        for event in trace["events"]
        if event["type"] == "generate"
    }
