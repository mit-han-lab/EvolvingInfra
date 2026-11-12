"""Agent identities for clients that send none.

A coding-agent harness resends the whole conversation on every turn and never names its agent
or branch. The tracker infers both from the messages: a request whose messages start with the
messages of an earlier request continues that request's branch; any other request starts a new
agent. Two requests that continue the same branch at once (a harness's side request next to the
main turn) cannot share it, because a branch's request numbers must arrive in order: the second
one gets a new branch of the same agent, and so does a request that starts with an earlier turn of
a branch but not its last one (a side request forked from the conversation).

Messages are compared by role, text and tool calls only, so markers that move between turns
(Anthropic `cache_control`, content split into parts or not) do not break a match.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
import uuid

HISTORY = 16  # earlier requests per branch that a fork can start from


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", "output_text"):
                    out.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    out.append("<image:" + hashlib.sha256(str(url).encode()).hexdigest()[:16] + ">")
            else:
                out.append(str(part))
        return "".join(out)
    return json.dumps(content, sort_keys=True)


def message_key(message: dict) -> str:
    """A digest of what a message says: its role, text, tool calls and the call it answers."""
    calls = [(c.get("function") or {}).get("name", "") + "\0" + str((c.get("function") or {}).get("arguments", ""))
             for c in message.get("tool_calls") or [] if isinstance(c, dict)]
    body = json.dumps([message.get("role"), _text(message.get("content")), calls,
                       message.get("tool_call_id") or ""], ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class Branch:
    agent_id: str
    subagent_id: int
    prefix: tuple = ()          # message keys of the branch's last request
    history: list = field(default_factory=list)  # message keys of its earlier requests, newest last
    next_request_id: int = 0
    inflight: int = 0
    sent: bool = False          # at least one request reached the router with this identity
    last_used: float = field(default_factory=time.monotonic)
    ended: bool = False


@dataclass(frozen=True)
class Assignment:
    branch: Branch
    request_id: int

    @property
    def header(self) -> str:
        return json.dumps({"agent_id": self.branch.agent_id, "subagent_id": self.branch.subagent_id,
                           "request_id": self.request_id}, separators=(",", ":"))


class Tracker:
    """Assigns (agent_id, subagent_id, request_id) to requests from their messages."""

    def __init__(self):
        self.branches: list[Branch] = []
        self.next_subagent: dict[str, int] = {}

    def assign(self, messages: list, now: float | None = None) -> Assignment:
        keys = tuple(message_key(m) for m in messages if isinstance(m, dict))
        now = time.monotonic() if now is None else now
        live = [b for b in self.branches if not b.ended]
        best = max((b for b in live if b.prefix and len(b.prefix) <= len(keys) and keys[:len(b.prefix)] == b.prefix),
                   key=lambda b: len(b.prefix), default=None)
        if best is not None and not best.inflight:
            branch = best
        else:
            # A busy branch, or a fork from an earlier turn: a new branch of the same agent.
            def depth(b):
                hit = [len(p) for p in b.history + [b.prefix] if p and len(p) <= len(keys) and keys[:len(p)] == p]
                return max(hit, default=0)
            parent = max(live, key=depth, default=None) if best is None else best
            if parent is None or depth(parent) == 0:
                agent = uuid.uuid4().hex
                self.next_subagent[agent] = 1
                branch = Branch(agent, 0)
            else:
                agent = parent.agent_id
                branch = Branch(agent, self.next_subagent[agent])
                self.next_subagent[agent] += 1
            self.branches.append(branch)
        request_id = branch.next_request_id
        branch.next_request_id += 1
        if branch.prefix:
            branch.history = (branch.history + [branch.prefix])[-HISTORY:]
        branch.prefix = keys
        branch.inflight += 1
        branch.last_used = now
        return Assignment(branch, request_id)

    def finish(self, assignment: Assignment, *, reached_router: bool, now: float | None = None) -> None:
        branch = assignment.branch
        branch.inflight -= 1
        branch.sent = branch.sent or reached_router
        branch.last_used = time.monotonic() if now is None else now

    def idle(self, idle_secs: float, now: float | None = None) -> list[Branch]:
        """Branches idle for idle_secs; they are marked ended and forgotten."""
        now = time.monotonic() if now is None else now
        done = [b for b in self.branches if not b.inflight and now - b.last_used >= idle_secs]
        for b in done:
            b.ended = True
        self.branches = [b for b in self.branches if not b.ended]
        live = {b.agent_id for b in self.branches}
        self.next_subagent = {a: n for a, n in self.next_subagent.items() if a in live}
        return done
