#!/usr/bin/env python3
"""KV-event wire compatibility: the engine's KV-event format -> Dynamo 1.4.0.

One process per worker. This changes representation only: existing salted STORE
metadata becomes the official tagged-map cache_salt field, and replay gains the
topic frame expected by Dynamo. See README_KV_EVENT_BRIDGE.md for limitations.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import msgpack
import zmq
import zmq.asyncio


class WireError(ValueError):
    pass


def transform_payload(payload: bytes) -> tuple[bytes, int, int]:
    """Return payload, event count, salted STORE count; never alter identifiers.

    Unsalted-only batches remain byte-for-byte identical. Unknown producer
    schemas are rejected instead of silently discarding additional metadata.

    Args:
        payload: MessagePack SGLang batch containing timestamp, events and DP rank.

    Returns:
        Tuple of wire bytes, total event count and converted salted STORE count.
        Unsalted-only payloads retain their original bytes; unsupported schemas raise
        WireError.
    """
    batch = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    if not isinstance(batch, list) or len(batch) != 3:
        raise WireError("expected SGLang [timestamp, events, dp_rank] batch")
    events = batch[1]
    if not isinstance(events, list):
        raise WireError("batch events must be an array")
    output = []
    converted = 0
    for event in events:
        if not isinstance(event, list) or not event:
            raise WireError("expected tagged-array SGLang event")
        tag = event[0]
        if tag == "BlockStored":
            if len(event) == 7:
                output.append(event)
                continue
            if len(event) != 8 or not isinstance(event[7], dict):
                raise WireError("unsupported BlockStored extension")
            metadata = event[7]
            if set(metadata) != {"cache_salt"} or not isinstance(metadata["cache_salt"], str):
                raise WireError("expected only string metadata.cache_salt")
            if event[5] is not None:
                raise WireError("salted LoRA events have not been validated")
            output.append({
                "type": "BlockStored",
                "block_hashes": event[1],
                "parent_block_hash": event[2],
                "token_ids": event[3],
                "block_size": event[4],
                "lora_id": event[5],
                "medium": event[6],
                "cache_salt": metadata["cache_salt"],
            })
            converted += 1
        elif tag == "BlockRemoved" and len(event) == 3:
            output.append(event)
        elif tag == "AllBlocksCleared" and len(event) == 1:
            output.append(event)
        else:
            raise WireError("unsupported SGLang event tag or array length")
    if not converted:
        return payload, len(events), 0
    batch[1] = output
    return msgpack.packb(batch, use_bin_type=True), len(events), converted


def log(event: str, **fields) -> None:
    """Emit a timestamped bridge diagnostic.

    Args:
        event: Short diagnostic event name.
        fields: JSON-serializable metadata; callers must exclude tokens, salts and
            client identities.

    Returns:
        None. Writes and flushes one JSON line to stdout.
    """
    print(json.dumps({"time_unix": time.time(), "event": event, **fields}, sort_keys=True), flush=True)


class Bridge:
    def __init__(self, args):
        """Initialize one worker event bridge and its lifecycle state.

        Args:
            args: Validated CLI namespace with endpoints, topic, queue limits, retry
                settings and optional stats path.

        Returns:
            None. Creates bridge state and ZMQ context but does not connect or bind
            sockets; requires a running asyncio loop.
        """
        self.args = args
        self.topic = args.topic.encode("utf-8")
        self.context = zmq.asyncio.Context()
        self.sockets = []
        self.children: set[asyncio.Task] = set()
        self.identities: set[bytes] = set()
        self.stop = asyncio.Event()
        self.failure = asyncio.get_running_loop().create_future()
        self.stats = {
            "started_unix": time.time(), "pid": os.getpid(), "state": "starting",
            "live_batches": 0, "replay_batches": 0, "live_events": 0,
            "replay_events": 0, "salted_stores_converted": 0,
            "replay_requests": 0, "replay_completed": 0, "errors": 0,
            "replay_retries": 0, "replay_gaps": 0, "replay_timeouts": 0,
            "last_live_seq": None, "last_replay_seq": None,
            "upstream_pub": args.upstream_pub, "upstream_replay": args.upstream_replay,
            "downstream_pub": args.downstream_pub, "downstream_replay": args.downstream_replay,
            "topic": args.topic,
        }

    def socket(self, kind):
        """Allocate an owned socket with the bridge queue limits.

        Args:
            kind: ZMQ socket type such as SUB, PUB, ROUTER or DEALER.

        Returns:
            New socket with zero linger and configured send/receive high-water marks,
            tracked for cleanup.
        """
        sock = self.context.socket(kind)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, self.args.hwm)
        sock.setsockopt(zmq.RCVHWM, self.args.hwm)
        self.sockets.append(sock)
        return sock

    def write_stats(self):
        """Persist the current counters and update timestamp.

        Args:
            None. Uses the current instance state.

        Returns:
            None. Atomically replaces the configured statistics file when enabled;
            filesystem errors propagate.
        """
        if self.args.stats_file:
            path = Path(self.args.stats_file)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({**self.stats, "updated_unix": time.time()}, indent=2) + "\n")
            temporary.replace(path)

    def convert(self, payload, seq, replay=False):
        """Transform one event batch and account for its source.

        Args:
            payload: Encoded SGLang event batch.
            seq: Exactly eight bytes encoding the unsigned sequence number in big-endian
                order.
            replay: Whether to update replay counters instead of live counters.

        Returns:
            Converted batch bytes. Updates event/sequence statistics; raises WireError for
            an invalid sequence or schema.
        """
        if len(seq) != 8:
            raise WireError("sequence must be exactly 8 bytes")
        wire, count, converted = transform_payload(payload)
        name = "replay" if replay else "live"
        self.stats[name + "_batches"] += 1
        self.stats[name + "_events"] += count
        self.stats["salted_stores_converted"] += converted
        self.stats["last_" + name + "_seq"] = int.from_bytes(seq, "big")
        return wire

    async def live_loop(self, upstream, downstream):
        """Forward live events while preserving topic and sequence.

        Args:
            upstream: Connected SUB socket receiving topic/sequence/payload frames.
            downstream: Bound PUB socket for converted batches.

        Returns:
            No normal return. Publishes converted frames until cancelled; malformed frames
            or I/O errors terminate the loop.
        """
        while True:
            frames = await upstream.recv_multipart()
            if len(frames) != 3 or frames[0] != self.topic:
                raise WireError("live requires [configured topic, sequence, payload]")
            topic, seq, payload = frames
            await downstream.send_multipart([topic, seq, self.convert(payload, seq)])

    async def one_replay(self, identity, start_seq, downstream):
        """Forward a contiguous replay; repair upstream ROUTER drops.

        SGLang's replay ROUTER can silently drop messages at its send HWM,
        including the end marker. Each retry uses a fresh DEALER and starts at
        the first unforwarded sequence. Never skip or publish a missing batch.

        Args:
            identity: Downstream client identity already registered as active.
            start_seq: Big-endian eight-byte first sequence requested.
            downstream: ROUTER socket on which to return replay frames.

        Returns:
            None after forwarding contiguous batches and the end marker. Raises on
            unrepaired gaps; always removes the active identity and closes each temporary
            upstream socket.
        """
        self.stats["replay_requests"] += 1
        start = int.from_bytes(start_seq, "big")
        expected = start
        # The publisher retained these live events before servicing this replay.
        # A received end marker alone cannot prove a dropped tail was delivered.
        required_through = self.stats["last_live_seq"]
        log("replay_start", start_seq=start)
        try:
            for attempt in range(self.args.replay_max_retries + 1):
                upstream = self.socket(zmq.DEALER)
                upstream.connect(self.args.upstream_replay)
                try:
                    await upstream.send_multipart([b"", expected.to_bytes(8, "big")])
                    while True:
                        try:
                            frames = await asyncio.wait_for(
                                upstream.recv_multipart(), self.args.replay_timeout_s)
                        except asyncio.TimeoutError:
                            self.stats["replay_timeouts"] += 1
                            break
                        if len(frames) != 3 or frames[0] != b"" or len(frames[1]) != 8:
                            raise WireError("upstream replay requires [empty, sequence, payload]")
                        _, seq, payload = frames
                        if not payload:
                            if seq != b"\xff" * 8:
                                raise WireError("unexpected replay end marker")
                            if required_through is not None and expected <= required_through:
                                self.stats["replay_gaps"] += 1
                                break
                            await downstream.send_multipart([identity, b"", self.topic, seq, payload])
                            self.stats["replay_completed"] += 1
                            log("replay_complete", start_seq=start, next_seq=expected, attempts=attempt + 1)
                            return
                        value = int.from_bytes(seq, "big")
                        if value < expected:
                            continue
                        if value > expected:
                            self.stats["replay_gaps"] += 1
                            break
                        payload = self.convert(payload, seq, replay=True)
                        await downstream.send_multipart([identity, b"", self.topic, seq, payload])
                        expected += 1
                finally:
                    upstream.close(linger=0)
                    self.sockets.remove(upstream)
                if attempt < self.args.replay_max_retries:
                    self.stats["replay_retries"] += 1
                    log("replay_retry", next_seq=expected, attempt=attempt + 1)
            raise WireError(f"replay incomplete after retries at sequence {expected}")
        finally:
            self.identities.remove(identity)

    def replay_finished(self, task):
        """Observe replay task completion without losing exceptions.

        Args:
            task: Completed replay task registered in children.

        Returns:
            None. Removes the task and forwards its first non-cancellation failure to the
            bridge failure future.
        """
        self.children.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None and not self.failure.done():
                self.failure.set_exception(exc)

    async def replay_loop(self, downstream):
        """Accept replay requests and dispatch independent replay tasks.

        Args:
            downstream: Bound ROUTER socket receiving identity/empty/start-sequence frames.

        Returns:
            No normal return. Starts tracked replay tasks; rejects duplicate client
            identities or excess concurrent replays.
        """
        while True:
            frames = await downstream.recv_multipart()
            if len(frames) != 3 or frames[1] != b"" or len(frames[2]) != 8:
                raise WireError("downstream request requires [identity, empty, start sequence]")
            identity, _, start_seq = frames
            if identity in self.identities:
                raise WireError("overlapping replay from one downstream identity")
            if len(self.children) >= self.args.max_replays:
                raise WireError("too many concurrent replay requests")
            self.identities.add(identity)
            task = asyncio.create_task(self.one_replay(identity, start_seq, downstream))
            self.children.add(task)
            task.add_done_callback(self.replay_finished)

    async def stats_loop(self):
        """Periodically persist bridge statistics.

        Args:
            None. Uses the current instance state.

        Returns:
            No normal return. Writes statistics once per second until cancelled or an I/O
            error occurs.
        """
        while True:
            self.write_stats()
            await asyncio.sleep(1)

    async def run(self):
        """Bind bridge endpoints and supervise live, replay and statistics loops.

        Args:
            None. Uses the current instance state.

        Returns:
            None on a stop signal. Propagates loop failures after recording them; always
            cancels tasks and closes sockets/context.
        """
        tasks = []
        try:
            sub = self.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE, self.topic)
            sub.connect(self.args.upstream_pub)
            pub = self.socket(zmq.PUB)
            pub.bind(self.args.downstream_pub)
            replay = self.socket(zmq.ROUTER)
            # Do not reproduce upstream's silent HWM drops on the repaired link.
            replay.setsockopt(zmq.ROUTER_MANDATORY, 1)
            replay.bind(self.args.downstream_replay)
            self.stats["state"] = "ready"
            self.write_stats()
            log("ready", **self.stats)
            tasks = [asyncio.create_task(self.live_loop(sub, pub)),
                     asyncio.create_task(self.replay_loop(replay)),
                     asyncio.create_task(self.stats_loop()),
                     asyncio.create_task(self.stop.wait())]
            done, _ = await asyncio.wait([*tasks, self.failure], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            self.stats["state"] = "stopped"
        except Exception as exc:
            self.stats["state"] = "failed"
            self.stats["errors"] += 1
            # Never log payloads, tokens, salts, or ZMQ client identities.
            log("fatal", error_type=type(exc).__name__, reason=str(exc) if isinstance(exc, WireError) else "bridge I/O or decode failure")
            raise
        finally:
            for task in [*tasks, *self.children]:
                task.cancel()
            await asyncio.gather(*tasks, *self.children, return_exceptions=True)
            if not self.failure.done():
                self.failure.cancel()
            for sock in self.sockets:
                sock.close(linger=0)
            self.context.term()
            self.write_stats()


async def async_main(args):
    """Run a bridge with cooperative signal-driven shutdown.

    Args:
        args: Validated CLI namespace for one bridge.

    Returns:
        None after bridge shutdown; bridge failures propagate. Installs SIGINT/SIGTERM
        handlers on the current event loop.
    """
    bridge = Bridge(args)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bridge.stop.set)
    await bridge.run()


def main():
    """Parse and validate bridge CLI options, then run the asyncio service.

    Args:
        None. Uses the enclosing service state or process arguments.

    Returns:
        Process exit code 0 for normal shutdown or 1 for runtime failure; CLI
        validation may raise SystemExit.
    """
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream-pub", "upstream-replay", "downstream-pub", "downstream-replay"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--topic", default="")
    p.add_argument("--replay-timeout-s", type=float, default=30)
    p.add_argument("--replay-max-retries", type=int, default=16)
    p.add_argument("--hwm", type=int, default=10000)
    p.add_argument("--max-replays", type=int, default=16)
    p.add_argument("--stats-file", type=Path)
    args = p.parse_args()
    if args.replay_timeout_s <= 0 or args.hwm <= 0 or args.max_replays <= 0 or args.replay_max_retries < 0:
        p.error("timeout, HWM and maximum replay count must be positive")
    if args.stats_file and not args.stats_file.parent.is_dir():
        p.error("stats-file parent directory must exist")
    try:
        asyncio.run(async_main(args))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
