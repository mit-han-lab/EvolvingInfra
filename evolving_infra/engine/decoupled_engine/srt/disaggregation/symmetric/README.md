# Symmetric P/D mode — status notes

`--disaggregation-mode symmetric`: every worker can run the prefill leg and the decode leg of a request; the
router picks the legs per request and the decode leg (reader) pulls KV from the prefill leg (owner) with
Mooncake one-sided RDMA READ, chunk by chunk under the running prefill (`export_table.py`, `import_queue.py`,
`planes.py`, `scheduler_mixin.py`). The router side is in `docs/placement.md` at the repository root.

## Request-held transfer lifetime

Symmetric exports retain their source KV and metadata until every TP reader
rank sends DONE or acknowledges ABORT after its reads have stopped. The
request-held GPU lease has no independent TTL: D can wait for capacity longer
than the old 30-second TTL plus cooling period. Export timeout requests
revocation; it does not authorize page reuse. A cancelled prefill chunk also
waits for its outstanding GPU result before returning its pages.

Reader cancellation drains queued/running chunk and aux reads before sending
ABORT or allowing destination pages to be freed. This wait is on the failure
path; successful reads retain the existing chunk/FINAL/DONE protocol. Pending
imports that have not allocated pages send an explicit cancellation without
issuing READs. Explicit `/abort_request` calls in symmetric mode reach the
scheduler even after the P response has left tokenizer state, so the still-live
export can be revoked. An HTTP response to that call is **not** a quiescence
acknowledgement.

An unreachable reader leaves source pages held. Likewise, a Mooncake error
does not expose a cancel/drain fence through the current wrapper: the affected
import is quarantined, with no page reuse or ABORT acknowledgement. Recovery
requires coordinated teardown of the affected workers/transport; a longer
timer is not a substitute. Monitor retained exports and the reader timeline's
`quarantined_at` field. CPU tests cover delayed import, chunk/aux cancellation,
failed reads concurrent with other reads, and completed-P cancellation. GPU
stress and cancellation validation are required before performance acceptance.
Deploy the updated owner and reader together on every participating TP rank;
an older reader's ABORT does not provide the drain guarantee.

For cancellation before the first owner publication, put the same nonzero
uint64 `attempt_id` in both `kv_export` and `kv_import`. The reader can then
retire that exact attempt immediately, without leaving late messages in its
early-publication buffer. Every retry needs a fresh attempt ID, even when it
reuses a `transfer_id`; explicit attempts reject wildcard ABORT(0). Omitting
the field retains the legacy owner-generated identity and its cancellation
limitations. Deploy both updated peers together; mixing explicit and legacy
marks does not provide this guarantee. Cancellation before the owner has
registered its export remains a separate limitation. CPU protocol coverage
does not replace real-load cleanup validation.

## Known gaps (bookkept TODOs)

- **Reader-side host-pool READ for Split is not implemented.** The intended path is a per-page-range fallback
  ladder GPU READ → host-pool READ → recompute. Today the importer has only the GPU READ rung: when the owner
  demoted exported pages to its host pool between CHUNK_READY and FINAL (generation mismatch), the reader falls
  back to local recompute and counts it (`import_fallback_recompute`). CPU→CPU reuse therefore goes
  through Local placement + hicache load-back and the existing peer-kv servicer path, not through a Split
  reader reading the owner's host pool. Deliberately deferred.
- GPU-direct RDMA registration on hosts without the `nvidia_peermem` kernel module (and in sandboxed runtimes
  such as Modal) requires `WITH_NVIDIA_PEERMEM=0` (Mooncake DMA-BUF path); the registration method does not
  change the data path.
- Decode-role workers keep imported KV in their radix tree and back it up to host under `write_through`; the
  cost of that under a pure decode weight is not removed here.

## Deferred decode-to-prefill host return

The router can select the original P as the destination of a **CPU-only** return
by adding this optional directive to the existing `kv_import` mark:

```json
{
  "transfer_id": 42,
  "attempt_id": 7,
  "owner_worker": "http://10.0.0.1:30000",
  "owner_addr": "tcp://10.0.0.1:8998",
  "expected_tokens": 8192,
  "cache_policy": "retain",
  "host_return": {
    "destination_servicer_addr": "10.0.0.1:18100",
    "source_servicer_addr": "10.0.0.2:18100",
    "timeout_ms": 10000,
    "max_tokens": 65536
  }
}
```

These are rank-0 **PeerKVServicer** endpoints, not HTTP or symmetric P/D
bootstrap endpoints. All participating workers need the matching model,
cache geometry, peer transport and HiCache host tier. D requires
`write_through`, host memory mode `cache`, and `retain`; P needs the existing
dynamic `peer_kv` storage backend. Plain TP with PP=CP=1 is supported,
including the existing single-copy host pool and TP GPU broadcast path.
IPv6 endpoints are rejected because this peer transport binds IPv4 today.
Omitting the directive keeps the ordinary import protocol. The engine's
existing default import policy is `retain`; the router uses that default
instead of implicitly selecting `gpu_only` for a zero-prefill-weight D.
Explicit `gpu_only`/`drop` remain available when return is disabled.

The lifecycle reuses the existing data plane:

1. After successful decode, the normal finished-request insertion supplies the
   committed, page-aligned cache key, including EAGLE's boundary handling.
   The return uses the **same** write-through operation; it does not launch a
   second GPU-to-CPU copy. A bounded CPU hold survives the request response.
2. Once its normal write ACK arrives, D's existing peer IO thread sends a
   notification to P. The scheduler only enqueues it. P matches its CPU prefix
   and pulls the missing suffix through `prefetch_from_storage` / PeerKVBackend.
   Neither worker launches H2D for this background return.
3. P publishes the CPU cache insertion only after the existing prefetch commit
   verifies all required pools. A later request uses the unchanged asynchronous
   host-load/next-prefill path. P acknowledges completion so D can release its
   temporary hold; deadlines release abandoned holds without freeing an active
   RDMA lease's pinned source pages.

The original P's cache lifetime is carried in optional `FINAL.owner_cache_epoch`.
P rejects a return from before its flush, detach or restart. An older P without
this field continues serving requests, but its D return is skipped. Source and
destination identities and budgets come from the router; the engine does not
choose destinations, reschedule requests or retry failed returns.

Returns are best effort: aborted/retracted requests, unavailable host backing,
an evicted/pending prompt anchor, exhausted budgets, or a suffix below the
existing prefetch threshold are skipped. `max_tokens` bounds the missing
generated suffix, not the full prompt. Full token metadata is limited to 1M
tokens per command and 4M retained tokens across at most 64 source commands;
IO queues have independent aggregate/count bounds. The deadline starts at
finished insertion and includes waiting for D2H. A full/failed notification
queue cannot block the inference step. Metadata serialization and socket waits
run on the existing peer IO thread, with no additional transfer thread.

Return commands, completions and timeouts are applied after a TP-consistent
exchange. Allocator-only TP ranks apply the same changes even though they have
no network servicer. The existing pipelined HiCache agreement carries one
readiness bit; event batches exchange only when there is work.

CPU tests cover protocol compatibility, real ZMQ notifications, source pin
and ACK ordering, timeout and reset cleanup, EAGLE page boundaries, real
host-prefetch commit, and TP follower agreement. This does **not** establish
GPU correctness under load or a throughput improvement.
Actual-output replay must also preserve generated tokens in the next turn for
that turn to reuse these pages. Changing a previous GPU-only D to `retain`
adds its normal host-backup traffic; CPU2CPU does not add another GPU copy,
but CPU, network and host-memory contention still require measurement.

### D2P token normalization

The peer event decodes token IDs into a Python list. Resident radix keys use
`array("q")`. Normalize the event at `_begin_host_return_import` before matching;
do not relax `RadixKey.match`'s representation check. The regression uses real
ordinary and EAGLE key matching, checks the suffix and namespace delivered to
storage prefetch, and asserts that the return itself does not invoke H2D.

A correct return shows three stages: D's peer reads of the KV bytes, P's CPU
commit of the returned tokens, and a later request that reports those tokens
as host hits. A notification alone is not commit proof. Decode host return has
no GPU validation in this repository.
