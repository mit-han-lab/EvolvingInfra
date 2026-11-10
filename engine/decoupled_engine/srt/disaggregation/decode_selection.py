"""Read-only D selection state. Counts named commitments, never DEFERRED legs."""
from __future__ import annotations
import math


def committed_requests(scheduler, prealloc):
    """Deduplicate phase overlap by bootstrap room, falling back to RID."""
    found = {}
    def add(value, pending=False):
        req = getattr(value, "req", value)
        if getattr(req, "finished_reason", None) is not None:
            return
        room = getattr(req, "bootstrap_room", None)
        key = ("room", room) if isinstance(room, int) else ("rid", req.rid)
        old = found.get(key)
        if old is None or (old[1] and not pending):
            found[key] = (req, pending)
    for req in scheduler.running_batch.reqs:
        add(req)
    for req in scheduler.waiting_queue:
        add(req)
    for entry in prealloc.transfer_queue.queue:
        add(entry)
    last = getattr(scheduler, "last_batch", None)
    if last is not None and last.forward_mode.is_prebuilt():
        for req in last.reqs:
            add(req)
    for entry in prealloc.queue:
        # A normal early-allocation leg is already committed to this worker.
        # A late candidate is not committed until ADMIT; RELEASED/FAILED entries
        # are excluded by their abort/finish marker or state.
        state = getattr(entry, "late_d_state", None)
        if state not in (None, "admitted", "held", "allocated"):
            continue
        add(entry, pending=getattr(entry.req, "req_pool_idx", None) is None)
    return list(found.values())


def selection_load(prealloc):
    """Dense full-pool capacity, same admission-budget helper as D's allocator.

    This does not match/evict a radix node or inspect GPU tensors. Pending
    selected legs are reserved once here; router reservations must be added
    only for rooms absent from allocated_rooms.
    """
    scheduler = prealloc.scheduler
    pairs = committed_requests(scheduler, prealloc)
    rooms = sorted({req.bootstrap_room for req, _ in pairs
                    if isinstance(getattr(req, "bootstrap_room", None), int)})
    truncated = len(rooms) > 256
    page = prealloc.token_to_kv_pool_allocator.page_size
    reserve = prealloc.num_reserved_decode_tokens
    pending = [(req, is_pending) for req, is_pending in pairs if is_pending]
    context = sum(len(r.origin_input_ids) + len(r.output_ids) for r, _ in pairs)
    result = dict(capacity_available=False, capacity_unsupported_reason=None,
        allocated_requests=len(pairs), allocated_context_tokens=context,
        allocated_rooms=rooms[:256], allocated_rooms_truncated=truncated,
        allocated_room_ids_complete=len(rooms)==len(pairs) and not truncated,
        output_reserve_tokens=reserve, page_size=page,
        committed_unallocated_requests=len(pending), physically_allocated_requests=len(pairs)-len(pending),
        allocatable_tokens=None, free_gpu_tokens=None, committed_reservation_tokens=None,
        decode_step_ewma_s=getattr(scheduler, "decode_step_ewma_s", None),
        decode_step_samples=getattr(scheduler, "decode_step_ewma_samples", 0),
        decode_step_batch_ewma=getattr(scheduler, "decode_step_batch_ewma", None))
    if scheduler.enable_hisparse or prealloc._uses_swa_tail_prealloc() or prealloc._growth_ratio_applies() > 0:
        result["capacity_unsupported_reason"] = "requires_dense_fixed_reserve"
        return result
    if not result["allocated_room_ids_complete"]:
        result["capacity_unsupported_reason"] = "missing_or_truncated_room_ids"
        return result
    unsafe = prealloc._cap_unsafe_running_rids()
    retractable = sum(len(r.origin_input_ids)+len(r.output_ids)
                      for r in scheduler.running_batch.reqs if r.rid not in unsafe)
    budget = prealloc._allocatable_token_budgets(
        retractable_tokens=retractable, count_retracted=True,
        hicache_reserved_tokens=prealloc._hicache_pending_restore_tokens(), cap_unsafe_rids=unsafe)
    committed = sum(((prealloc._pre_alloc_fill_len(r)+page-1)//page)*page + reserve for r, _ in pending)
    result.update(capacity_available=True, free_gpu_tokens=prealloc.token_to_kv_pool_allocator.available_size(),
        allocatable_tokens=max(0, budget-committed), committed_reservation_tokens=committed)
    return result


def update_step_ewma(scheduler, step_us, batch_size, alpha=.1):
    """Busy CPU launch-to-launch interval, not GPU kernel elapsed time."""
    if step_us <= 0 or batch_size <= 0:
        return
    value = step_us/1e6
    if not math.isfinite(value):
        return
    old = getattr(scheduler, "decode_step_ewma_s", None)
    scheduler.decode_step_ewma_s = value if old is None else old+alpha*(value-old)
    old_batch = getattr(scheduler, "decode_step_batch_ewma", None)
    scheduler.decode_step_batch_ewma = float(batch_size) if old_batch is None else old_batch+alpha*(batch_size-old_batch)
    scheduler.decode_step_ewma_samples = getattr(scheduler, "decode_step_ewma_samples", 0)+1


# ---------------------------------------------------------------------------
# Symmetric workers (symmetric P/D): the same /v1/loads ``late_d`` capacity
# fields for a symmetric worker in its D role
# (SGLANG_DECODE_SELECTION_METRICS).
#
# The router's native fit (router late_fit.rs, sgl-model-gateway
# decode_capacity.rs) reads allocated_requests / allocated_rooms /
# allocated_context_tokens / allocatable_tokens / free_gpu_tokens /
# output_reserve_tokens / page_size. A symmetric worker reports them from its
# own pool through its ImportQueue, with the admission rule that its late-D
# walk applies (ImportQueue._late_need_and_budgets), so "fits" in the router
# means "the walk admits it on this tick" for the same state.

# Late-D states of an ImportQueue.pending entry that are committed to this
# worker but hold no pages yet. DEFERRED (candidate, not bound) and RELEASED
# entries are not commitments.
_SYM_COMMITTED_PENDING = (None, "admitted", "held")

# The ``pending`` value of a symmetric_committed_requests pair: False when the
# request holds its pages, else why it holds none yet (both values are true).
# SYM_PENDING_IMPORT: an ImportQueue.pending import bound to this worker; the
# walk charges it _required_tokens (page-rounded prompt + decode headroom).
# SYM_PENDING_QUEUED: a waiting-queue request without a request slot (a Local
# request, or a retracted D leg that recomputes here); the walk budget already
# holds its decode headroom (num_active_reqs counts the waiting queue).
SYM_PENDING_IMPORT = "import"
SYM_PENDING_QUEUED = "queued"


def _sym_room(req):
    room = getattr(req, "bootstrap_room", None)
    return room if isinstance(room, int) else None


def _page_ceil(tokens, page):
    return ((tokens + page - 1) // page) * page


def symmetric_committed_requests(queue, running_reqs, waiting_queue, chunked_req=None):
    """D-role commitments of a symmetric worker, deduplicated by transfer room.

    Returns (room_pairs, local_pairs): lists of (req, pending), where pending
    is False for a request that holds its pages, else SYM_PENDING_IMPORT or
    SYM_PENDING_QUEUED. room_pairs are D legs (kv_import), keyed by
    bootstrap_room = transfer_id: decoding, active imports (pages allocated,
    waiting for READY) and pending imports bound to this worker (ADMITTED /
    HELD, or not late). A retracted D leg keeps its room and recomputes from
    the waiting queue (SYM_PENDING_QUEUED). local_pairs are Local requests
    (no export and no import mark) that will decode here: they have no room,
    so the router cannot match them to a reservation; they are reported apart
    (local_requests) and their tokens still count in context and reserve.
    Export P legs (kv_export) are never D commitments.
    """
    rooms = {}
    locals_ = {}

    def add(req, pending):
        if req is None or getattr(req, "finished_reason", None) is not None:
            return
        if getattr(req, "export_mark", None) is not None:
            return
        room = _sym_room(req)
        if room is None:
            if getattr(req, "import_mark", None) is None:
                key = req.rid
                old = locals_.get(key)
                if old is None or (old[1] and not pending):
                    locals_[key] = (req, pending)
            return
        old = rooms.get(room)
        if old is None or (old[1] and not pending):
            rooms[room] = (req, pending)

    for req in running_reqs:
        add(req, False)
    for entry in getattr(queue, "active", {}).values():
        add(entry.req, False)
    for entry in getattr(queue, "pending", []):
        if getattr(entry, "late_d_state", None) not in _SYM_COMMITTED_PENDING:
            continue
        add(entry.req, SYM_PENDING_IMPORT)
    for req in waiting_queue:
        if getattr(req, "import_mark", None) is None:
            no_slot = getattr(req, "req_pool_idx", None) is None
            add(req, SYM_PENDING_QUEUED if no_slot else False)
    if chunked_req is not None and getattr(chunked_req, "import_mark", None) is None:
        add(chunked_req, False)
    return list(rooms.values()), list(locals_.values())


def symmetric_export_view(export_table):
    """Compute-first export legs on this worker (statistics only):
    computed and retained for their reader, and still computing. Their pages
    are held by the request (not free, not evictable), so free_gpu_tokens and
    allocatable_tokens already exclude them; these fields show how much of the
    pool they hold."""
    retained_reqs = retained_tokens = computing_reqs = computing_tokens = 0
    if export_table is None:
        return dict(export_retained_reqs=0, export_retained_tokens=0,
                    export_computing_reqs=0, export_computing_tokens=0)
    for entry in getattr(export_table, "entries", {}).values():
        if not getattr(entry, "compute_before_allocation", False):
            continue
        if getattr(entry, "abort_reason", None) is not None:
            continue
        if getattr(entry, "prefill_finished_at", None) is not None:
            retained_reqs += 1
            retained_tokens += int(getattr(entry, "computed_end", 0) or 0)
        else:
            computing_reqs += 1
            computing_tokens += int(getattr(entry, "computed_end", 0) or 0)
    return dict(export_retained_reqs=retained_reqs, export_retained_tokens=retained_tokens,
                export_computing_reqs=computing_reqs, export_computing_tokens=computing_tokens)


def symmetric_selection_load(queue, *, running_reqs, waiting_queue, chunked_req=None,
                             export_table=None, scheduler=None):
    """The decode-side selection_load() fields for a symmetric worker.

    allocatable_tokens = the walk's full-token budget
    (ImportQueue._allocatable_tokens: free + evictable - decode headroom of
    every active request - the F-A local reserve), less the
    _required_tokens (page-rounded prompt + output reserve) of every committed
    import that holds no pages yet, less the sequence pages (prompt + output
    so far) of every queued request without a slot (its decode headroom is in
    the walk budget already), and capped by the F-B import budget and by the
    M4 clean-page budget when those are on. Real-time engine state only: no
    static capacity, no GPU read, no radix match.
    """
    room_pairs, local_pairs = symmetric_committed_requests(
        queue, running_reqs, waiting_queue, chunked_req)
    rooms = sorted(_sym_room(req) for req, _ in room_pairs)
    truncated = len(rooms) > 256
    page = int(queue.page_size)
    reserve = int(queue.num_reserved_decode_tokens)
    pending = [req for req, kind in room_pairs if kind]
    pending_local = [req for req, kind in local_pairs if kind]
    imports = [req for req, kind in room_pairs if kind == SYM_PENDING_IMPORT]
    queued = [req for req, kind in room_pairs + local_pairs
              if kind == SYM_PENDING_QUEUED]
    context = sum(len(r.origin_input_ids) + len(r.output_ids)
                  for r, _ in room_pairs + local_pairs)
    result = dict(capacity_available=False, capacity_unsupported_reason=None,
        allocated_requests=len(room_pairs), allocated_context_tokens=context,
        allocated_rooms=rooms[:256], allocated_rooms_truncated=truncated,
        allocated_room_ids_complete=not truncated,
        output_reserve_tokens=reserve, page_size=page,
        committed_unallocated_requests=len(pending) + len(pending_local),
        physically_allocated_requests=len(room_pairs) - len(pending),
        allocatable_tokens=None, free_gpu_tokens=None, committed_reservation_tokens=None,
        decode_step_ewma_s=getattr(scheduler, "decode_step_ewma_s", None),
        decode_step_samples=getattr(scheduler, "decode_step_ewma_samples", 0),
        decode_step_batch_ewma=getattr(scheduler, "decode_step_batch_ewma", None),
        local_requests=len(local_pairs),
        **symmetric_export_view(export_table))
    if getattr(queue, "swa_tail_import", False):
        # The router checks full tokens only; a SWA-tail import also needs SWA.
        result["capacity_unsupported_reason"] = "requires_dense_fixed_reserve"
        return result
    if truncated:
        result["capacity_unsupported_reason"] = "missing_or_truncated_room_ids"
        return result
    required = queue._required_tokens
    import_tokens = sum(required(r) for r in imports)
    # A queued request (a Local request, or a retracted D leg that recomputes
    # here) has its decode headroom in the walk budget already
    # (num_active_reqs counts the waiting queue): add its sequence pages only,
    # prompt + output so far, since a retracted request recomputes both.
    committed = import_tokens + sum(
        _page_ceil(len(r.origin_input_ids) + len(r.output_ids), page) for r in queued)
    allocatable = queue._allocatable_tokens() - committed
    budget = queue._budget_tokens()
    if budget is not None and (queue.active or imports):
        # F-B caps imports only (ImportQueue._within_budget). The walk
        # allocates the bound imports before a new ADMIT (claim, then ADMIT
        # order), so a new import must also fit under the cap after them, even
        # while no import is active yet.
        allocatable = min(
            allocatable, budget - queue.active_import_tokens() - import_tokens)
    clean = queue._allocatable_clean_tokens()
    if clean is not None:
        clean -= sum(_page_ceil(len(r.origin_input_ids), page) for r in imports)
        allocatable = min(allocatable, clean + reserve)
    result.update(capacity_available=True,
        free_gpu_tokens=int(queue.token_to_kv_pool_allocator.available_size()),
        allocatable_tokens=max(0, int(allocatable)), committed_reservation_tokens=int(committed))
    return result
