#!/usr/bin/env python3
"""Turn an agent session into a Markdown transcript that is safe to attach to a pull request.

    python tools/scrub_transcript.py SESSION [-o transcript.md] [--redact NAME ...]

SESSION is a Claude Code session file (~/.claude/projects/<project>/<session>.jsonl) or any text
file. Secrets, e-mail addresses, home directories, IP addresses and the login name are replaced
with placeholders, and so is every --redact NAME (account, host, organization names the script
cannot know). It prints the counts. Exit code 1: a string still looks like a secret; the
output is written but is not ready to publish. Remove the value from the output, or pass
--allow with the characters printed for it once it is confirmed harmless. Standard library only.
"""
import argparse
import getpass
import json
import math
from pathlib import Path
import re
import sys

LIMIT = 2000  # characters kept of one tool input or output

SECRETS = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("docker token", re.compile(r"\bdckr_(?:pat|oat)_[A-Za-z0-9_-]{16,}")),
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("github token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("api key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("aws key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("bearer token", re.compile(r"(?i)\b(bearer|authorization:)\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("url password", re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")),
    ("signed url", re.compile(
        r"(?i)\b(X-(?:Amz|Goog)-(?:Signature|Credential|Security-Token)|Signature|sig)=[A-Za-z0-9%._~+/-]{16,}")),
    # NAME_TOKEN=value: the value has a letter and a digit and at least 16 characters.
    ("assigned secret", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY))(\s*[=:]\s*)(['\"]?)"
        r"(?=[^\s'\"]*[A-Za-z])(?=[^\s'\"]*\d)[^\s'\"]{16,}\3")),
]
BLOB = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")  # embedded base64 data, e.g. an image
PRIVATE = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<email>"),
    ("home directory", re.compile(r"(/Users|/home)/[A-Za-z0-9._-]+"), r"\1/<user>"),
    ("ip address", re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
]
CANDIDATE = re.compile(r"[A-Za-z0-9+/_=-]{32,}")
HEX = re.compile(r"[0-9a-fA-F]+")


def entropy(text: str) -> float:
    counts = {c: text.count(c) for c in set(text)}
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def looks_secret(text: str) -> bool:
    """A long part with lower case, upper case and digits and high entropy. Hex digests, paths and
    names made of words joined by - or _ are not flagged."""
    if "/" in text:
        return False
    for part in re.split(r"[-_=]", text):
        if len(part) < 24 or HEX.fullmatch(part):
            continue
        kinds = sum(bool(re.search(p, part)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]"))
        if kinds == 3 and entropy(part) >= 4.3:
            return True
    return False


def scrub(text: str, allow=(), redact=()) -> tuple:
    """(scrubbed text, {kind: count}, [(line number, suspected secret left)])."""
    counts = {}

    def count(kind, replacement):
        def sub(match):
            counts[kind] = counts.get(kind, 0) + 1
            return match.expand(replacement) if "\\" in replacement else replacement
        return sub

    keeps = {"assigned secret": r"\1\2[REDACTED]", "bearer token": r"\1 [REDACTED]", "signed url": r"\1=[REDACTED]"}
    for kind, pattern in SECRETS:
        text = pattern.sub(count(kind, keeps.get(kind, "[REDACTED]")), text)
    text = BLOB.sub(count("data blob", "[DATA]"), text)
    for kind, pattern, replacement in PRIVATE:
        text = pattern.sub(count(kind, replacement), text)
    for name in sorted(set(redact), key=len, reverse=True):
        if len(name) >= 3:
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", count("name", "<redacted>"), text)
    left, seen = [], set()
    for number, line in enumerate(text.splitlines(), 1):
        for m in CANDIDATE.findall(line):
            if m not in seen and looks_secret(m) and not any(m.startswith(a) for a in allow):
                seen.add(m)
                left.append((number, m))
    return text, counts, left


def clip(text: str) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return text if len(text) <= LIMIT else text[:LIMIT] + f"\n... [{len(text) - LIMIT} more characters]"


def blocks(content):
    """Markdown pieces of one message: text, tool calls and tool results; no hidden reasoning."""
    if isinstance(content, str):
        yield content
        return
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            yield block.get("text", "")
        elif kind == "tool_use":
            given = block.get("input") or {}
            body = given.get("command") or given.get("file_path") or json.dumps(given, ensure_ascii=False)
            yield f"**Tool: {block.get('name')}**\n\n```\n{clip(body)}\n```"
        elif kind == "tool_result":
            result = block.get("content")
            if isinstance(result, list):
                result = "\n".join(b.get("text", "") for b in result if isinstance(b, dict))
            yield f"**Result**\n\n```\n{clip(result or '')}\n```"


def render(path: Path) -> str:
    """A Claude Code session as Markdown; any other file as it is."""
    raw = path.read_text(errors="replace")
    if path.suffix != ".jsonl":
        return raw
    out = [f"# Session transcript ({path.stem})\n"]
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") not in ("user", "assistant") or row.get("isSidechain"):
            continue
        parts = [p for p in blocks((row.get("message") or {}).get("content")) if p.strip()]
        if parts:
            out.append(f"## {row['type'].capitalize()}\n\n" + "\n\n".join(parts) + "\n")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("transcript.md"))
    parser.add_argument("--redact", action="append", default=[], metavar="NAME",
                        help="also replace this name (account, host, organization); repeatable")
    parser.add_argument("--allow", action="append", default=[], metavar="PREFIX",
                        help="a flagged string, by the characters printed for it, that is not a secret")
    parser.add_argument("--keep-login", action="store_true", help="do not replace the login name")
    args = parser.parse_args(argv)
    names = list(args.redact) + ([] if args.keep_login else [getpass.getuser()])
    text, counts, left = scrub(render(args.session), tuple(args.allow), names)
    args.output.write_text(text)
    summary = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())) or "nothing"
    print(f"wrote {args.output} ({len(text)} characters); replaced: {summary}", flush=True)
    if left:
        print(f"NOT READY TO PUBLISH: {len(left)} string(s) in {args.output} may be secrets:")
        for number, item in left[:20]:
            print(f"  line {number}: {item[:16]}... ({len(item)} characters)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
