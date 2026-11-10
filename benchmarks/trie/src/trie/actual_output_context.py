"""Replay recorded context changes while retaining actual assistant token IDs.

Recorded snapshots still decide which history is retained, compacted or removed.
This does not execute model-selected tools or add history absent from the trace.
All generations retain their recorded output-token budget.
"""

from array import array
from dataclasses import dataclass
import hashlib


def token_digest(tokens):
    return hashlib.sha256(array("q", tokens).tobytes()).hexdigest()


@dataclass
class _Patch:
    source: bytes
    actual: bytes
    anchor: bytes
    position: int
    rid: str | None


class ActualOutputContext:
    def __init__(self):
        self.previous_source = b""
        self.previous_output = b""
        self.previous_recorded_output = b""
        self.previous_rid = None
        self.patches = []

    @staticmethod
    def _unique_find(haystack, needle):
        if not needle:
            return -1
        at = haystack.find(needle)
        while at >= 0 and at % 8:
            at = haystack.find(needle, at + 1)
        if at < 0:
            return -1
        other = haystack.find(needle, at + 8)
        while other >= 0 and other % 8:
            other = haystack.find(needle, other + 1)
        return at if other < 0 else -1

    def prepare(self, source_tokens):
        source = array("q", source_tokens).tobytes()
        fresh = None
        if self.previous_output:
            start = len(self.previous_source)
            n = len(self.previous_output)
            # Token-authoritative traces can retain an assistant turn after a
            # snapshot rewrite, including a changed image or context compaction.
            recorded = self.previous_recorded_output
            previous_anchor = self.previous_source[-32 * 8:]
            if recorded and source[max(0, start - len(previous_anchor)):start + n] == previous_anchor + recorded:
                position = start
            elif recorded:
                found = self._unique_find(source, previous_anchor + recorded)
                position = found + len(previous_anchor) if found >= 0 else -1
            else:
                position = -1
            # Schema-v3 block traces deliberately synthesize output and input
            # blocks independently. Their retained append slot is identified by
            # the unchanged preceding prompt and recorded output-token budget.
            if position < 0 and source.startswith(self.previous_source) and len(source) >= start + n:
                position = start
            if position >= 0:
                fresh = _Patch(source[position:position + n], self.previous_output,
                               source[max(0, position - 32 * 8):position],
                               position, self.previous_rid)
                self.patches.append(fresh)

        result = bytearray(source)
        retained = []
        occupied = []
        applied_tokens = 0
        parent = None
        for patch in self.patches:
            start = patch.position
            needle = patch.anchor + patch.source
            anchor_start = start - len(patch.anchor)
            if anchor_start < 0 or source[anchor_start:start + len(patch.source)] != needle:
                found = self._unique_find(source, needle)
                if found < 0:
                    # A compacted/removed turn must not be reintroduced.
                    continue
                start = found + len(patch.anchor)
            end = start + len(patch.source)
            if any(start < hi and end > lo for lo, hi in occupied):
                raise ValueError("Ambiguous overlapping actual-output context spans")
            occupied.append((start, end))
            result[start:end] = patch.actual
            patch.position = start
            retained.append(patch)
            applied_tokens += len(patch.actual) // 8
            if patch is fresh:
                parent = {"rid": patch.rid, "start_token": start // 8,
                          "tokens": len(patch.actual) // 8,
                          "output_sha256": hashlib.sha256(patch.actual).hexdigest()}
        self.patches = retained
        tokens = array("q")
        tokens.frombytes(result)
        attrs = {"replay_context_mode": "actual_output",
                 "actual_output_history_spans": len(retained),
                 "actual_output_history_tokens": applied_tokens,
                 "actual_output_parent": parent,
                 "actual_output_wire_sha256": hashlib.sha256(result).hexdigest()}
        return tokens.tolist(), attrs

    def commit(self, source_tokens, generated_tokens, *, recorded_output=None, rid=None):
        self.previous_source = array("q", source_tokens).tobytes()
        self.previous_output = array("q", generated_tokens).tobytes()
        self.previous_recorded_output = array("q", recorded_output or []).tobytes()
        self.previous_rid = rid
