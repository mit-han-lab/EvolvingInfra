#!/usr/bin/env python3
"""Convert the cleaned Codex request trace into cc-weka session rows that Trie's `cc_weka` adapter reads.

Input: one JSON row per model request (`codex/requests.jsonl.gz`: session_id, round_index, user, input_tokens_total,
prefix_tokens, output_tokens, timing_events, tools, ...). Output: one JSON row per root session, in the semianalysis
cc-weka format (`{"id", "models", "block_size": 64, "hash_id_scope": "local", "requests": [...]}`), with the session's
sub-agents as `{"type": "subagent", "agent_id", "parent_agent_id"?, "requests"}` entries, so
`benchmark/trie/scripts/strict_convert` (adapter `cc_weka`) turns it into a multi-agent Trie task.

Rules:
- Sub-agents: Codex spawns a sub-agent with the `spawn_agent` tool, and the sub-agent is recorded as its own session.
  The released trace has no parent link (`parent_thread_id` is not exported), so a spawn is linked to the earliest
  not-yet-linked session of the same user whose first request starts within [spawn emitted - 1 s, spawn result +
  60 s]. Spawns are taken in time order; a link that would make a cycle is skipped. A linked session becomes a
  sub-agent of its parent's root session; a sub-agent spawned by a sub-agent names it in `parent_agent_id`.
- Agents: every agent (the root and each sub-agent) is one sequential chain (`"sequential_agents": true`): a request
  that starts before the agent's previous request ends is started at that end (`overlap_serialized`).
- Prefix: within an agent, a request shares floor(prefix_tokens / 64) blocks with the agent's previous request (the
  provider's reported cached tokens, a lower bound of the true common prefix); every other block is new. Agents share
  no blocks with each other: a sub-agent starts from a fresh context, not a fork of its parent's conversation. As for
  CC, the whole session (all its agents) shares one cache salt and one routing key in the replay.
- `in` is input_tokens_total rounded to the nearest multiple of 64 (at least 64); `out` is output_tokens (it includes
  reasoning tokens).
- Timing: a request starts at its latest input event (tool_result / user_message), else at the agent's previous end,
  and ends at its usage_report. Per session family (root plus sub-agents), every stretch in which no request of the
  family is in flight is clipped to max_think (120 s, as in the CC trace) and the clipped time is removed from the
  family's timeline. think_time = start - the later of the agent's previous end and the end of any of its direct
  sub-agents that ended before the start (the cc-weka convention: measured from the previous event, sub-agents
  included), and is capped at max_think per request as in the cleaned CC trace (`think_capped`: an agent idle
  while another agent of its session was busy).
- Hash ids are session-local, so prompts shared across sessions (system prompt) are not shared in the replay.

Usage: python3 codex_to_ccweka.py IN.jsonl.gz OUT.jsonl.gz [--max-think 120] [--report report.json]
Stdlib only.
"""
import argparse, bisect, collections, gzip, json
from datetime import datetime

BLOCK = 64
INPUT_EVENTS = {"tool_result", "user_message"}
LINK_BEFORE_S, LINK_AFTER_S = 1.0, 60.0


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def agent_requests(rows, stats):
    """(start, end, row) per request of one recorded session, serialized so none overlaps the previous one."""
    out, prev_end = [], None
    for r in sorted(rows, key=lambda r: r["round_index"]):
        ev = r.get("timing_events") or []
        ends = [ts(e["timestamp"]) for e in ev if e.get("event_type") == "usage_report"]
        if not ends:
            stats["drop_no_usage_report"] += 1
            continue
        ins = [ts(e["timestamp"]) for e in ev if e.get("event_type") in INPUT_EVENTS]
        end = max(ends)
        start = max(ins) if ins else (prev_end if prev_end is not None else min(ts(e["timestamp"]) for e in ev))
        if start > end:
            stats["start_after_end"] += 1
            start = end
        if prev_end is not None and start < prev_end:
            stats["overlap_serialized"] += 1
            start, end = prev_end, max(end, prev_end)
        out.append((start, end, r))
        prev_end = end
    return out


def link_subagents(sessions, reqs, stats):
    """child session -> parent session, from spawn_agent calls (see the module docstring)."""
    by_user = collections.defaultdict(list)
    for sid, rs in reqs.items():
        by_user[rs[0][2].get("user")].append((rs[0][0], sid))
    for lst in by_user.values():
        lst.sort()
    spawns = []
    for sid, rows in sessions.items():
        if sid not in reqs:
            continue
        for r in rows:
            for tool in r.get("tools") or []:
                if tool.get("tool_name") == "spawn_agent" and tool.get("emitted_at"):
                    emitted = ts(tool["emitted_at"])
                    result = ts(tool["result_at"]) if tool.get("result_at") else emitted
                    spawns.append((emitted, result, sid, r.get("user")))
    spawns.sort()
    parent = {}

    def lineage(sid):
        out = {sid}
        while sid in parent:
            sid = parent[sid]
            out.add(sid)
        return out

    for emitted, result, sid, user in spawns:
        stats["spawns"] += 1
        lst = by_user.get(user, [])
        i = bisect.bisect_left(lst, (emitted - LINK_BEFORE_S, ""))
        candidates = []
        while i < len(lst) and lst[i][0] <= result + LINK_AFTER_S:
            child = lst[i][1]
            # child must be unlinked and not the spawner or one of its ancestors (no cycles)
            if child not in parent and child not in lineage(sid):
                candidates.append(child)
            i += 1
        if not candidates:
            stats["spawns_unlinked"] += 1
            continue
        stats["spawns_linked_ambiguous" if len(candidates) > 1 else "spawns_linked_unique"] += 1
        parent[candidates[0]] = sid
    return parent


def family_clock(intervals, max_think):
    """Map recorded times of one family onto a timeline whose idle stretches are clipped to max_think."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    starts, shifts, shift, clipped = [], [], 0.0, 0
    for k, (start, end) in enumerate(merged):
        if k:
            gap = start - merged[k - 1][1]
            if gap > max_think:
                shift += gap - max_think
                clipped += 1
        starts.append(start)
        shifts.append(shift)

    def clock(t):
        return t - shifts[max(0, bisect.bisect_right(starts, t) - 1)]
    return clock, clipped


def agent_id(sid):
    return sid.replace("codex:", "codex-")


def convert_family(root, members, parent, reqs, max_think, stats):
    intervals = [(s, e) for m in members for s, e, _ in reqs[m]]
    clock, clipped = family_clock(intervals, max_think)
    stats["idle_clipped"] += clipped
    base = clock(min(s for s, _ in intervals))
    children = collections.defaultdict(list)
    for m in members:
        if m != root:
            children[parent[m]].append(m)
    child_ends = {m: sorted(round(clock(reqs[c][-1][1]) - base, 3) for c in children[m]) for m in members}
    next_id, out_agents, models = 0, {}, set()
    for m in members:
        agent_reqs, prev_ids, prev_end, last_end = [], [], None, 0.0
        ends = child_ends[m]
        for start, end, r in reqs[m]:
            t = max(round(clock(start) - base, 3), last_end)  # rounding must not create an overlap
            api = round(end - start, 3)
            k = bisect.bisect_right(ends, t)
            refs = [x for x in (prev_end, ends[k - 1] if k else None) if x is not None]
            think = max(0.0, t - max(refs)) if refs else 0.0
            if think > max_think:
                stats["think_capped"] += 1
                think = max_think
            n_in = max(BLOCK, int(round(r["input_tokens_total"] / BLOCK)) * BLOCK)
            n_blocks = n_in // BLOCK
            shared = min((r.get("prefix_tokens") or 0) // BLOCK, len(prev_ids), n_blocks)
            ids = prev_ids[:shared] + list(range(next_id, next_id + n_blocks - shared))
            next_id += n_blocks - shared
            stats["blocks_total"] += n_blocks
            stats["blocks_shared"] += shared
            models.add(r["model"])
            agent_reqs.append({"t": t, "model": r["model"], "in": n_in,
                               "out": max(1, int(r.get("output_tokens") or 0)), "hash_ids": ids,
                               "api_time": api, "think_time": round(think, 3)})
            prev_ids = ids
            prev_end = last_end = round(t + api, 3)
        out_agents[m] = agent_reqs
    entries = [dict(q, type="n") for q in out_agents[root]]
    for m in members:
        if m == root:
            continue
        entry = {"type": "subagent", "agent_id": agent_id(m), "requests": out_agents[m]}
        if parent[m] != root:
            entry["parent_agent_id"] = agent_id(parent[m])
        entries.append(entry)
    entries.sort(key=lambda e: e["requests"][0]["t"] if e["type"] == "subagent" else e["t"])
    stats["requests"] += sum(len(v) for v in out_agents.values())
    stats["subagents"] += len(members) - 1
    return {"id": agent_id(root), "models": sorted(models), "block_size": BLOCK, "hash_id_scope": "local",
            "sequential_agents": True, "requests": entries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--max-think", type=float, default=120.0)
    ap.add_argument("--report")
    a = ap.parse_args()
    sessions = collections.defaultdict(list)
    n_rows = 0
    with gzip.open(a.inp, "rt") as f:
        for line in f:
            r = json.loads(line)
            sessions[r["session_id"]].append(r)
            n_rows += 1
    stats = collections.Counter()
    reqs = {}
    for sid, rows in sessions.items():
        rs = agent_requests(rows, stats)
        if rs:
            reqs[sid] = rs
        else:
            stats["drop_empty_session"] += 1
    parent = link_subagents(sessions, reqs, stats)

    def root_of(sid):
        while sid in parent:
            sid = parent[sid]
        return sid

    families = collections.defaultdict(list)
    for sid in sorted(reqs, key=lambda s: (reqs[s][0][0], s)):
        families[root_of(sid)].append(sid)
    stats["nested_subagents"] = sum(1 for p in parent.values() if p in parent)
    with gzip.open(a.out, "wt") as f:
        for root in sorted(families):
            members = [root] + [m for m in families[root] if m != root]
            f.write(json.dumps(convert_family(root, members, parent, reqs, a.max_think, stats),
                               separators=(",", ":")) + "\n")
            stats["sessions"] += 1
            stats["sessions_with_subagents"] += len(members) > 1
    report = {"input_rows": n_rows, "input_sessions": len(sessions), **stats,
              "shared_block_share": stats["blocks_shared"] / max(1, stats["blocks_total"]),
              "max_think_s": a.max_think, "block": BLOCK, "link_window_s": [-LINK_BEFORE_S, LINK_AFTER_S]}
    print(json.dumps(report, indent=1))
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
