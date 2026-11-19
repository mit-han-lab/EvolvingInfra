# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Pull-reader progress and cancellation over real ZMQ channels, without CUDA."""
import importlib
from pathlib import Path
import sys
import threading
import types
import uuid

import pytest
import zmq


@pytest.fixture
def pull_modules(monkeypatch):
    # Import the CPU-only data plane without the engine package's CUDA startup.
    root = Path(__file__).resolve().parents[2] / "evolving_infra/engine/decoupled_engine"
    previous = {name: module for name, module in sys.modules.items()
                if name == "decoupled_engine" or name.startswith("decoupled_engine.")}
    for name in previous:
        monkeypatch.delitem(sys.modules, name)
    package = types.ModuleType("decoupled_engine")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "decoupled_engine", package)
    prefix = "decoupled_engine.srt.disaggregation.pull."
    try:
        yield types.SimpleNamespace(**{
            name: importlib.import_module(prefix + name)
            for name in ("api", "protocol", "importer", "read_engine")})
    finally:
        for name in list(sys.modules):
            if name == "decoupled_engine" or name.startswith("decoupled_engine."):
                sys.modules.pop(name)
        sys.modules.update(previous)


@pytest.fixture
def channel(pull_modules):
    modules = pull_modules
    wire, api = modules.protocol, modules.api
    ctx = zmq.Context()
    identity = uuid.uuid4().hex
    owner_addr, reader_addr = f"inproc://owner-{identity}", f"inproc://reader-{identity}"
    owner = ctx.socket(zmq.PULL)
    owner.setsockopt(zmq.RCVTIMEO, 3000)
    owner.bind(owner_addr)
    publisher = ctx.socket(zmq.PUSH)
    publisher.setsockopt(zmq.SNDTIMEO, 3000)
    publisher.connect(reader_addr)
    remote = modules.read_engine.FakeReadEngine(session_id=f"owner-{identity}")
    local = modules.read_engine.FakeReadEngine(session_id=f"reader-{identity}")
    src = remote.memory.alloc(16, fill=17)
    dst = local.memory.alloc(16)
    src_aux = remote.memory.alloc(4, fill=23)
    dst_aux = local.memory.alloc(4)

    def registration(engine, kv, aux):
        return wire.Registration(
            session_id=engine.session_id, tp_rank=0, attn_tp_size=1, page_size=4,
            pools={"full": wire.PoolRegion(ptrs=[kv], page_nbytes=[4])},
            aux=wire.ItemRegion(ptrs=[aux], item_lens=[4]))

    reader = modules.importer.ReaderPlane(
        registration=registration(local, dst, dst_aux), read_engine=local,
        bind_addr=reader_addr, zmq_ctx=ctx, waiting_timeout_s=10, send_begin=False)
    release = threading.Event()

    def send(message):
        publisher.send(wire.encode(message))

    send(wire.make_register(reg=registration(remote, src, src_aux), owner_addr=owner_addr))

    def begin(transfer, lookup):
        reader.begin_import(api.ImportMark(
            transfer_id=transfer, owner_worker="owner", owner_addr=owner_addr,
            local_aux_index=0, attempt_id=transfer), f"request-{transfer}", lookup)

    def chunk(transfer, *, seq=0, start=0, end=4, page=0, last=True):
        send(wire.make_chunk_ready(chunk=api.ChunkReady(
            transfer_id=transfer, chunk_seq=seq, token_start=start, token_end=end,
            pools=[api.PoolPages(pool="full", page_indices=[page], generations=[1])],
            is_last=last), attempt_id=transfer))

    def final(transfer, *, tokens=4, generations=None):
        send(wire.make_final(final=api.FinalMeta(
            transfer_id=transfer, total_tokens=tokens, aux_index=0),
            attempt_id=transfer, generations=generations or {"full": {0: 1}}))

    def receive(kind, transfer):
        while True:
            try:
                message = wire.decode(owner.recv())
            except zmq.Again:
                pytest.fail(f"reader did not send {kind} for transfer {transfer}")
            if message.type == kind and message.transfer_id == transfer:
                return message

    try:
        yield types.SimpleNamespace(
            reader=reader, wire=wire, api=api, local=local, remote=remote,
            dst=dst, dst_aux=dst_aux, begin=begin, chunk=chunk, final=final,
            send=send, receive=receive, release=release)
    finally:
        release.set()
        reader.close()
        publisher.close(linger=0)
        owner.close(linger=0)
        ctx.term()


def test_blocked_page_lookup_does_not_stall_another_import(channel):
    c = channel
    entered = threading.Event()

    def blocked(chunk):
        entered.set()
        assert c.release.wait(10), "page lookup was not released"
        return {"full": [0]}

    c.begin(1, blocked)
    c.chunk(1)
    assert entered.wait(3)
    c.final(1)
    c.begin(2, lambda chunk: {"full": [1]})
    c.chunk(2)
    c.final(2)
    c.receive(c.wire.MSG_DONE, 2)
    assert c.reader.poll(2) == c.api.TransferState.READY
    assert bytes(c.local.memory.view(c.dst + 4, 4)) == bytes([17]) * 4
    assert c.reader.poll(1) == c.api.TransferState.TRANSFERRING
    c.release.set()
    c.receive(c.wire.MSG_DONE, 1)
    assert c.reader.poll(1) == c.api.TransferState.READY


def test_owner_abort_waits_for_page_lookup_and_skips_the_read(channel):
    c = channel
    entered = threading.Event()

    def blocked(chunk):
        entered.set()
        assert c.release.wait(10), "page lookup was not released"
        return {"full": [0]}

    c.begin(1, blocked)
    c.chunk(1)
    assert entered.wait(3)
    c.send(c.wire.make_abort(transfer_id=1, attempt_id=1, reason="cancelled"))
    # A later completed transfer is a channel barrier for the preceding ABORT.
    c.begin(2, lambda chunk: {"full": [1]})
    c.chunk(2)
    c.final(2)
    c.receive(c.wire.MSG_DONE, 2)
    assert c.reader.poll(1) == c.api.TransferState.TRANSFERRING
    c.release.set()
    c.receive(c.wire.MSG_ABORT, 1)
    assert c.reader.poll(1) == c.api.TransferState.ABORTED
    assert bytes(c.local.memory.view(c.dst, 4)) == bytes(4)
    assert c.reader.snapshot()["history"]["1"]["reads"] == []


def test_page_lookup_failure_finishes_without_copying(channel):
    c = channel

    def broken(chunk):
        raise ValueError("missing destination page")

    c.begin(1, broken)
    c.chunk(1)
    c.final(1)
    c.receive(c.wire.MSG_ABORT, 1)
    assert c.reader.poll(1) == c.api.TransferState.FAILED
    assert c.reader.failure_reason(1) == "dst_pages:missing destination page"
    assert bytes(c.local.memory.view(c.dst, 4)) == bytes(4)


def test_multiple_chunks_verify_and_copy_aux_after_all_reads(channel):
    c = channel
    c.begin(1, lambda chunk: {"full": [chunk.chunk_seq]})
    c.chunk(1, last=False)
    c.chunk(1, seq=1, start=4, end=8, page=1)
    c.final(1, tokens=8, generations={"full": {0: 1, 1: 1}})
    c.receive(c.wire.MSG_DONE, 1)
    assert c.reader.poll(1) == c.api.TransferState.READY
    assert c.reader.final_meta(1).total_tokens == 8
    assert bytes(c.local.memory.view(c.dst, 8)) == bytes([17]) * 8
    assert bytes(c.local.memory.view(c.dst_aux, 4)) == bytes([23]) * 4
