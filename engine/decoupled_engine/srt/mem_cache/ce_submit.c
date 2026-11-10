/* Paced copy-engine submission for shared host-dedup loads (ce_2d_load.py).
 *
 * One call issues a whole load op from C: its 2D copies (one per page and
 * layer run) and its 1D copies (height-1 runs such as the MTP draft layer,
 * sent as cudaMemcpyBatchAsync batches). The Python caller holds no GIL
 * while it runs (ctypes releases it) and never pays a GIL hand-off per copy.
 * Copies go in chunks; after each chunk an event is recorded, and at most
 * `window` chunks are in flight: before reusing a chunk event the helper
 * waits for it (cudaEventSynchronize on a BlockingSync event, or polling
 * with a short sleep when no sync function is given), so no copy call ever
 * blocks or spins on a full stream queue.
 *
 * C2g (TP0 load + follower peer pulls): the
 * source publishes its progress in a shared-memory word as each chunk is
 * known complete (word[1] = seq<<24 | total chunks at op start, word[0] =
 * seq<<24 | chunks done); a follower polls that word (short sleep, CPU only)
 * before issuing the same chunk as peer copies. No GPU waits on another
 * process's work.
 *
 * Rate cap: `rate_bps` points at a live word (bytes/s; NULL or 0 = uncapped)
 * read before every chunk and while sleeping (1 ms slices), so the owner can
 * lift the cap of a running op (a bootstrap-window prefetch capped to spare
 * the forward, once an admission load queues behind it). Pacing is a token
 * bucket that restarts whenever the effective rate changes: the bytes issued
 * since the last change must fit the rate since that change, so a cap that
 * switches on mid-op starts with one chunk of credit instead of stalling the
 * op until its average since op start falls to the cap. (For the rate
 * sequences used before the adaptive cap, constant or constant-then-lifted,
 * this is the same schedule as pacing from op start.)
 *
 * Adaptive cap (bootstrap-window prefetch): with
 * `busy` and `deadline_ns` given, the cap applies only while *busy != 0 (the
 * owner's "a forward is in flight on this rank" word) and the op's remaining
 * bytes fit the cap before *deadline_ns (CLOCK_MONOTONIC ns; 0 = no known
 * slack); otherwise the chunk goes uncapped. These words change timing only.
 *
 * CUDA entry points are passed in as function pointers (resolved by the
 * caller from the libcudart it already uses), so this file links only libc.
 */
#include <stddef.h>
#include <stdint.h>
#include <time.h>

typedef int (*memcpy2d_fn)(void *dst, size_t dpitch, const void *src, size_t spitch,
                           size_t width, size_t height, int kind, void *stream);
typedef int (*event_record_fn)(void *event, void *stream);
typedef int (*event_query_fn)(void *event);
typedef int (*event_sync_fn)(void *event);
/* CUDA >= 13.0 */
typedef int (*batch13_fn)(void **dsts, void **srcs, size_t *sizes, size_t count, void *attrs,
                          size_t *attrs_idxs, size_t num_attrs, void *stream);
/* CUDA 12.8 - 12.x */
typedef int (*batch12_fn)(void **dsts, void **srcs, size_t *sizes, size_t count, void *attrs,
                          size_t *attrs_idxs, size_t num_attrs, size_t *fail_idx, void *stream);

#define CE_NOT_READY 600 /* cudaErrorNotReady */
#define CE_GATE_ERROR 9001   /* the source published an error */
#define CE_GATE_TIMEOUT 9002 /* the source made no progress in time */
#define SEQ_SHIFT 24
#define LOW_MASK ((1ull << SEQ_SHIFT) - 1)
#define RATE_SLICE_NS 1000000ull /* re-read the rate word at least every 1 ms */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

typedef struct {
  volatile uint64_t *pub; /* source: [0] progress, [1] op start, [2] error */
  uint64_t seq;
  uint64_t done;          /* chunks of this op known complete */
  event_record_fn rec;
  event_query_fn query;
  event_sync_fn sync;
  void **events;
  int window;
  void *stream;
  struct timespec nap;
  int slot, inflight;
  uint64_t waited, chunks;
} pacer_t;

/* Wait for the oldest in-flight chunk; returns 0 or a CUDA error. */
static int pacer_wait_oldest(pacer_t *p) {
  if (p->inflight <= 0) return 0;
  uint64_t t0 = now_ns();
  void *ev = p->events[(p->slot - p->inflight + p->window) % p->window];
  if (p->sync) {
    int r = p->sync(ev);
    if (r) return r;
  } else {
    for (;;) {
      int r = p->query(ev);
      if (r == 0) break;
      if (r != CE_NOT_READY) return r;
      nanosleep(&p->nap, NULL);
    }
  }
  p->waited += now_ns() - t0;
  p->inflight--;
  if (p->pub) {
    p->done++;
    __atomic_store_n(&p->pub[0], (p->seq << SEQ_SHIFT) | p->done, __ATOMIC_RELEASE);
  }
  return 0;
}

/* Non-blocking: retire (and publish) every in-flight chunk that already
 * completed, oldest first, so a relay forwards chunk k as soon as it lands
 * rather than when the window wait reaches it. */
static int pacer_pump(pacer_t *p) {
  while (p->inflight > 0) {
    void *ev = p->events[(p->slot - p->inflight + p->window) % p->window];
    int r = p->query(ev);
    if (r == CE_NOT_READY) return 0;
    if (r) return r;
    p->inflight--;
    if (p->pub) {
      p->done++;
      __atomic_store_n(&p->pub[0], (p->seq << SEQ_SHIFT) | p->done, __ATOMIC_RELEASE);
    }
  }
  return 0;
}

/* Make room for one more chunk (window full: wait for the oldest). */
static int pacer_wait(pacer_t *p) {
  if (p->inflight < p->window) return 0;
  return pacer_wait_oldest(p);
}

/* Follower: wait until the source has completed `need` chunks of op `seq`
 * (or moved past the op). Returns 0, CE_GATE_ERROR or CE_GATE_TIMEOUT. */
static int gate_wait(volatile uint64_t *gate, uint64_t seq, uint64_t need, uint64_t timeout_ns,
                     struct timespec *nap, uint64_t *waited, pacer_t *pump) {
  uint64_t target = (seq << SEQ_SHIFT) | need;
  uint64_t t0 = now_ns();
  for (;;) {
    if (__atomic_load_n(&gate[2], __ATOMIC_ACQUIRE)) return CE_GATE_ERROR;
    if (__atomic_load_n(&gate[0], __ATOMIC_ACQUIRE) >= target) break;
    if (timeout_ns && now_ns() - t0 > timeout_ns) return CE_GATE_TIMEOUT;
    if (pump && pump->pub) {
      int r = pacer_pump(pump); /* relay: forward what already landed */
      if (r) return r;
    }
    nanosleep(nap, NULL);
  }
  *waited += now_ns() - t0;
  return 0;
}

/* Follower: the source's chunk count for op `seq` (0 if it already moved past
 * the op, i.e. the op is complete). */
static int gate_total(volatile uint64_t *gate, uint64_t seq, uint64_t timeout_ns,
                      struct timespec *nap, uint64_t *waited, uint64_t *total) {
  uint64_t t0 = now_ns();
  for (;;) {
    if (__atomic_load_n(&gate[2], __ATOMIC_ACQUIRE)) return CE_GATE_ERROR;
    uint64_t v = __atomic_load_n(&gate[1], __ATOMIC_ACQUIRE);
    if ((v >> SEQ_SHIFT) == seq) { *total = v & LOW_MASK; break; }
    if ((v >> SEQ_SHIFT) > seq) { *total = 0; break; }
    if (timeout_ns && now_ns() - t0 > timeout_ns) return CE_GATE_TIMEOUT;
    nanosleep(nap, NULL);
  }
  *waited += now_ns() - t0;
  return 0;
}

static int pacer_mark(pacer_t *p) {
  int r = p->rec(p->events[p->slot], p->stream);
  if (r) return r;
  p->slot = (p->slot + 1) % p->window;
  p->inflight++;
  p->chunks++;
  return p->pub ? pacer_pump(p) : 0;
}

/* Returns 0 or a CUDA error code. *fail_idx: failing copy index, 2D copies
 * first then 1D copies (n2d + j); n2d + n1d for a pacing/event error.
 * stats[0] = ns waiting for chunk events, stats[1] = chunks submitted,
 * stats[2] = ns gated on the source, stats[3] = ns sleeping for the rate;
 * stats[4] = effective-rate changes (written only with busy or deadline_ns,
 * so callers of the older layout may pass a 4-word array).
 * state[0] / state[1] = next event slot / chunks possibly in flight, carried
 * across calls on the same stream so the window also bounds the previous
 * op's tail. gate_whole != 0: a follower waits for the source's whole op
 * before its first chunk even when the chunk counts match (the source copies
 * a different shape, so chunk k of each side covers different rows).
 * tail_bytes: bytes of the job still to come after this call (its later
 * layer parts); the deadline check counts them as remaining (0: whole op). */
int ce_submit_op(memcpy2d_fn cp, event_record_fn rec, event_query_fn query, event_sync_fn sync,
                 const uint64_t *dst, const uint64_t *dpitch, const uint64_t *src,
                 const uint64_t *spitch, const uint64_t *width, const uint64_t *height,
                 size_t n2d, size_t chunk2d, void *batch, int batch_has_fail_idx,
                 uint64_t *d1_dst, uint64_t *d1_src, uint64_t *d1_size, size_t n1d,
                 size_t chunk1d, void *attr, void **events, int window, void *stream,
                 uint64_t sleep_ns, size_t *fail_idx, uint64_t *stats, int *state,
                 int kind2d, volatile uint64_t *pub, volatile uint64_t *gate, uint64_t seq,
                 uint64_t timeout_ns, const volatile uint64_t *rate_bps, int gate_whole,
                 const volatile uint64_t *busy, const volatile uint64_t *deadline_ns,
                 uint64_t tail_bytes) {
  pacer_t p;
  int r = 0;
  uint64_t gate_waited = 0, total = 0, src_total = 0, chunk_no = 0;
  /* Optional pacing of the issue rate (bytes/s, live word): before each
   * chunk, sleep until the bytes issued since the last effective-rate change
   * fit that rate since the change (token bucket; (t_ref, b_ref) restart on
   * every change); a rate of 0 (cap lifted, or adaptive: idle / no slack)
   * ends the sleep within one slice. */
  uint64_t t_op = now_ns(), bytes = 0, rate_slept = 0;
  uint64_t t_ref = t_op, b_ref = 0, cur_rate = UINT64_MAX, flips = 0, op_bytes = 0;
  if (deadline_ns) {
    for (size_t j = 0; j < n2d; j++) op_bytes += width[j] * height[j];
    for (size_t j = 0; j < n1d; j++) op_bytes += d1_size[j];
  }
  p.rec = rec;
  p.query = query;
  p.sync = sync;
  p.events = events;
  p.window = window < 1 ? 1 : window;
  p.stream = stream;
  p.nap.tv_sec = 0;
  p.nap.tv_nsec = (long)(sleep_ns ? sleep_ns : 20000);
  p.slot = state[0];
  p.inflight = state[1];
  if (p.slot < 0 || p.slot >= p.window) p.slot = 0;
  if (p.inflight < 0 || p.inflight > p.window) p.inflight = p.window;
  p.waited = 0;
  p.chunks = 0;
  p.pub = NULL;
  p.seq = seq;
  p.done = 0;
  if (chunk2d == 0) chunk2d = 256;
  if (chunk1d == 0) chunk1d = 256;
  if (kind2d == 0) kind2d = 1; /* cudaMemcpyHostToDevice */
  *fail_idx = 0;
  total = (n2d + chunk2d - 1) / chunk2d + (n1d + chunk1d - 1) / chunk1d;
  if (pub) {
    /* Drain the previous op's tail (its chunks are not this op's progress). */
    while (p.inflight > 0) {
      if ((r = pacer_wait_oldest(&p))) { *fail_idx = n2d + n1d; goto out; }
    }
    p.pub = pub;
    __atomic_store_n(&pub[1], (seq << SEQ_SHIFT) | total, __ATOMIC_RELEASE);
  }
  if (gate) {
    r = gate_total(gate, seq, timeout_ns, &p.nap, &gate_waited, &src_total);
    if (r) { *fail_idx = n2d + n1d; goto out; }
  }

#define RATE_CHUNK()                                                                     \
  do {                                                                                   \
    for (;;) {                                                                           \
      uint64_t now = now_ns();                                                           \
      uint64_t eff = rate_bps ? __atomic_load_n(rate_bps, __ATOMIC_ACQUIRE) : 0;         \
      if (eff && busy && !__atomic_load_n(busy, __ATOMIC_ACQUIRE)) eff = 0;              \
      if (eff && deadline_ns) {                                                          \
        uint64_t dl = __atomic_load_n(deadline_ns, __ATOMIC_ACQUIRE);                    \
        double rem = (op_bytes > bytes ? (double)(op_bytes - bytes) : 0.0) +             \
                     (double)tail_bytes;                                                 \
        if (!dl || (double)now + rem * 1e9 / (double)eff > (double)dl) eff = 0;          \
      }                                                                                  \
      if (eff != cur_rate) {                                                             \
        if (cur_rate != UINT64_MAX) { /* restart the bucket at the change */            \
          flips++;                                                                       \
          t_ref = now;                                                                   \
          b_ref = bytes;                                                                 \
        }                                                                                \
        cur_rate = eff;                                                                  \
      }                                                                                  \
      if (!eff || bytes == b_ref) break;                                                 \
      uint64_t due = t_ref + (uint64_t)((double)(bytes - b_ref) * 1e9 / (double)eff);   \
      if (now >= due) break;                                                             \
      struct timespec ts;                                                                \
      ts.tv_sec = 0;                                                                     \
      ts.tv_nsec = (long)(due - now < RATE_SLICE_NS ? due - now : RATE_SLICE_NS);        \
      nanosleep(&ts, NULL);                                                              \
      rate_slept += now_ns() - now;                                                      \
    }                                                                                    \
  } while (0)

#define GATE_CHUNK()                                                                     \
  do {                                                                                   \
    if (gate && src_total) {                                                             \
      /* Same chunking and shape on both sides: wait for this chunk; else for */         \
      /* the whole op. */                                                                \
      uint64_t need = (!gate_whole && src_total == total) ? chunk_no + 1 : src_total;    \
      if ((r = gate_wait(gate, seq, need, timeout_ns, &p.nap, &gate_waited, &p))) {      \
        *fail_idx = n2d + n1d;                                                           \
        goto out;                                                                        \
      }                                                                                  \
    }                                                                                    \
    chunk_no++;                                                                          \
  } while (0)

  for (size_t i = 0; i < n2d; i += chunk2d) {
    if ((r = pacer_wait(&p))) { *fail_idx = n2d + n1d; goto out; }
    GATE_CHUNK();
    RATE_CHUNK();
    size_t end = i + chunk2d < n2d ? i + chunk2d : n2d;
    for (size_t j = i; j < end; j++) {
      bytes += width[j] * height[j];
      r = cp((void *)(uintptr_t)dst[j], (size_t)dpitch[j], (const void *)(uintptr_t)src[j],
             (size_t)spitch[j], (size_t)width[j], (size_t)height[j], kind2d, stream);
      if (r) { *fail_idx = j; goto out; }
    }
    if ((r = pacer_mark(&p))) { *fail_idx = n2d + n1d; goto out; }
  }
  for (size_t i = 0; i < n1d; i += chunk1d) {
    if ((r = pacer_wait(&p))) { *fail_idx = n2d + n1d; goto out; }
    GATE_CHUNK();
    RATE_CHUNK();
    size_t cnt = i + chunk1d < n1d ? chunk1d : n1d - i;
    size_t idx0 = 0, fidx = 0;
    if (batch_has_fail_idx)
      r = ((batch12_fn)batch)((void **)(d1_dst + i), (void **)(d1_src + i), (size_t *)(d1_size + i),
                              cnt, attr, &idx0, 1, &fidx, stream);
    else
      r = ((batch13_fn)batch)((void **)(d1_dst + i), (void **)(d1_src + i), (size_t *)(d1_size + i),
                              cnt, attr, &idx0, 1, stream);
    if (r) { *fail_idx = n2d + i + fidx; goto out; }
    for (size_t j = i; j < i + cnt; j++) bytes += d1_size[j];
    if ((r = pacer_mark(&p))) { *fail_idx = n2d + n1d; goto out; }
  }
  if (pub) {
    /* Publish the op's tail: every chunk complete before returning. */
    while (p.inflight > 0) {
      if ((r = pacer_wait_oldest(&p))) { *fail_idx = n2d + n1d; goto out; }
    }
    if (total == 0) __atomic_store_n(&pub[0], seq << SEQ_SHIFT, __ATOMIC_RELEASE);
  }
out:
  stats[0] = p.waited;
  stats[1] = p.chunks;
  stats[2] = gate_waited;
  stats[3] = rate_slept;
  if (busy || deadline_ns) stats[4] = flips;
  state[0] = p.slot;
  state[1] = p.inflight;
  return r;
}
