"""Opt-in guards for the P/D role planner. Every guard is off by default.

With all guards off, `decide` runs exactly as before and records nothing new.

G1, restore floor (`restore_floor`). Before any option is estimated, each P
node's observed restore ratio is raised to the cached-token-weighted mean ratio
of all nodes in the current onboarding observations. Those observations cover
the current P nodes over the last decision interval. A node measured while it
shared the P role can restore far fewer host tokens than it would once other
P nodes leave; the weighted mean is the ratio of the whole P set. Nodes without
an observation keep their calibration ratio, as before.

G3, minimum relative gain (`min_relative_gain`). A candidate must also gain at
least this fraction of the current option's mean minute capacity, in addition
to the two-standard-error rule.

Q', queue-aware add-P (`queue_p95_s`). The +1 P candidate is selected when all
of these hold:
  - the live trailing-window P_queue p95 is above the threshold;
  - at least `queue_min_since_switch_s` have passed since the last confirmed
    role switch;
  - the standard rule (two standard errors and G3) does not already add a P;
  - the candidate's mean minute gain is positive. The standard-error and G3
    thresholds do not apply to it.
Q' requires G1, so its gain uses floored ratios. P_queue is
forward_entry_time - wait_queue_entry_time of P legs, keyed on forward entry.
It is read from the workers' scheduler request traces, the same files the
onboarding observation reads. No engine flag or endpoint is added.

Q-hold, queue removal hold (`queue_removal_floor_s`). After a role switch
selected by Q' has been confirmed earlier in the run, no candidate with fewer
P than the current roles is selected while the live P_queue p95 (the
observation Q' uses) is at or above the floor. When the standard rule (two
standard errors and G3) would remove P, the planner re-selects among the
remaining selectable candidates in the standard order (largest mean minute
gain, then fewest changes); these all add P. If none is left, it holds. A
missing p95 does not block. Q' reads the standard rule's choice, not
Q-hold's: it fires exactly as with Q-hold off, and its +1 P overrides Q-hold's
choice in the same decision, so Q' is never blocked. Before the first Q'
switch, removals are allowed: the first
switch of a run often removes P while P_queue is high. Q-hold requires Q'.
The caller keeps the run state (`queue_added`); `decide` stays pure.

Queue signal (`queue_signal`, 'all' or 'ungated'). With 'ungated', the
P_queue observation that Q' (decisions and probes) and Q-hold read uses only
P legs that never hit the late-D park gate (req_trace
late_d_need.late_d_gate_skips == 0). The engine gate skips a leg while unread
computed KV is above its share of the P pool or free P memory is below the
reserve, so a gated leg mostly waits for decode back-pressure, not for P
compute. A leg without late_d_need counts as ungated and is counted. With P
legs in the window but none ungated there is no p95: Q' does not fire and
Q-hold does not block. The observation records both views. Requires Q'.

D batch (`decode_batch_source`, 'formula' or 'measured'). The formula caps a
D node's batch at min(max_running_requests, floor(0.95 * kv_capacity_tokens /
context)). With 'measured', the decode cap of every role map uses an
effective KV fraction f learned from the measured running batch:
cap = min(max_running_requests, f * kv_capacity_tokens / context), a fluid
operating point like the population bound (which stays as before). f is
learned only in a memory-bound window, where the requests whose P leg has
completed (the router's p_completed, time-weighted over the window) exceed
the measured running decode requests by at least max(2, 0.2 * p_completed);
then f = median over D nodes of running * context / kv_capacity_tokens.
Otherwise the last learned f is kept; before the first learned f the
formula's floored cap stays in use. A window in which some, but not all, D
nodes have no decode pass counts as unmeasured (f is kept). The running
batch of a D node is sum(bs * dur) / sum(dur) over its iter_trace decode
passes in the decision window (DecodeBatchMonitor). The caller keeps f.

E-hold, eviction-pressure removal hold (`evict_pressure_hold`). No
candidate with fewer P than the current roles is selected while the P nodes
evicted more than `evict_pressure_tokens` class-1 host-only tokens (KV of a
live session dropped from the host pool, `evicted_c1_host_only`) in the last
`evict_pressure_window_s`, summed over the nodes that held the P role. The
counter is the engine's cumulative TP0 KV_SESSION_STATS line in the worker's
engine.log (one line every 60 s with KV session retention); no engine flag
or endpoint is added (HostEvictionMonitor). A 60-s counter interval counts
for a node when the interval ends inside the window and the node held the
P role at that end; a removed P node keeps its P intervals, a new P node
counts from its confirmed switch on. Like Q-hold, a blocked removal is
replaced by the best remaining standard-selectable candidate that does not
remove P, else a hold. E-hold never blocks an addition, reads the standard
choice, so Q' fires exactly as with E-hold off. With Q-hold also on, a
removal is blocked when either blocks. No signal (an engine without the
line, no fresh sample) does not block; with some P nodes unobserved, only a
visible total above the threshold blocks. The engine log prefix has no time
zone: each read detects the log clock's offset from the controller's clock
(15-min steps), so the controller's zone and DST do not matter.
`evict_pressure_hold='shadow'` computes and records the same would-block
result without changing any decision. Release bound
(`evict_pressure_release`, default 'queue_p95'): a removal that the
pressure would block is still allowed when the all-leg P_queue p95 of the
Q' observation is below EVICT_RELEASE_P95_S ('queue_p95', needs Q'), when
its predicted mean relative gain is at least EVICT_RELEASE_GAIN ('gain'),
or at the run's first decision ('first_decision'); 'none' never releases.
"""
import calendar
import json
import math
import os
import re
import threading
import time

# configuration.json key -> PlannerGuards argument. Keys are read like the other
# controller_* options; an unknown controller_guard_* key is an error.
CONFIGURATION_KEYS = dict(
    controller_guard_restore_floor='restore_floor',
    controller_guard_min_relative_gain='min_relative_gain',
    controller_guard_queue_p95_s='queue_p95_s',
    controller_guard_queue_window_s='queue_window_s',
    controller_guard_queue_min_since_switch_s='queue_min_since_switch_s',
    controller_guard_queue_check_s='queue_check_s',
    controller_guard_queue_removal_floor_s='queue_removal_floor_s',
    controller_guard_queue_signal='queue_signal',
    controller_guard_decode_batch_source='decode_batch_source',
    controller_guard_evict_pressure_hold='evict_pressure_hold',
    controller_guard_evict_pressure_tokens='evict_pressure_tokens',
    controller_guard_evict_pressure_window_s='evict_pressure_window_s',
    controller_guard_evict_pressure_release='evict_pressure_release',
)
QUEUE_SIGNALS = ('all', 'ungated')
DECODE_BATCH_SOURCES = ('formula', 'measured')
# E-hold defaults. Evidence (KV_SESSION_STATS of four recorded guard runs): P-node class-1 host-only evictions in
# the 600 s before a removal were 0 before all four harmless removals and 16.41 M tokens (9 P nodes) before the
# harmful 9P -> 7P one; whenever P nodes evicted class-1 KV during traffic, the 600-s total was >= 2.17 M. The
# 1 M floor (about five 200k-token sessions) ignores a few isolated evictions and is far below every sustained level.
EVICT_PRESSURE_TOKENS = 1.e6
EVICT_PRESSURE_WINDOW_S = 600.
EVICT_PRESSURE_MODES = (False, True, 'shadow')
# Release bound (replay of the recorded decisions): 'queue_p95' keeps the harmful +6,316.5 s block (all-leg p95 20.6 s)
# and frees most later 9-tray decision points (p95 5.6-9.3 s); the ungated p95 (3.5 s there) would release the harmful
# one.
EVICT_RELEASES = ('queue_p95', 'gain', 'first_decision', 'none')
EVICT_RELEASE_P95_S = 10.
EVICT_RELEASE_GAIN = .20
# Log clock: offset = round((reference - naive time of the chunk's last time-stamped line) / step) * step, accepted
# when the residual lies in [-EVICT_CLOCK_AHEAD_S, EVICT_CLOCK_LAG_S]. The reference is the file's modification time
# (its last write, from the engine node's clock like the prefix; a stale file resolves too), or the controller's clock
# if the mtime is later. The previous offset is kept while it fits.
EVICT_CLOCK_STEP_S = 900.
EVICT_CLOCK_AHEAD_S = 30.
EVICT_CLOCK_LAG_S = 300.
# A P node whose newest visible KV_SESSION_STATS sample is older than this has no signal. The engine logs every
# 60 s and the driver reads every queue_check_s.
EVICT_PRESSURE_MAX_AGE_S = 300.


def _choice(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError('Invalid planner guard option ' + name)
    return value


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Invalid planner guard option ' + name)
    if value < 0 or (positive and value <= 0):
        raise ValueError('Invalid planner guard option ' + name)
    return float(value)


class PlannerGuards:
    """Guard options. The default instance enables nothing."""

    def __init__(self, *, restore_floor=False, min_relative_gain=0., queue_p95_s=0.,
                 queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60.,
                 queue_removal_floor_s=0., queue_signal='all', decode_batch_source='formula',
                 evict_pressure_hold=False, evict_pressure_tokens=EVICT_PRESSURE_TOKENS,
                 evict_pressure_window_s=EVICT_PRESSURE_WINDOW_S, evict_pressure_release='queue_p95'):
        if type(restore_floor) is not bool:
            raise ValueError('Invalid planner guard option restore_floor')
        self.restore_floor = restore_floor
        self.min_relative_gain = _number(min_relative_gain, 'min_relative_gain')
        if self.min_relative_gain >= 1:
            raise ValueError('Invalid planner guard option min_relative_gain')
        self.queue_p95_s = _number(queue_p95_s, 'queue_p95_s')
        self.queue_window_s = _number(queue_window_s, 'queue_window_s', positive=True)
        self.queue_min_since_switch_s = _number(queue_min_since_switch_s, 'queue_min_since_switch_s')
        self.queue_check_s = _number(queue_check_s, 'queue_check_s', positive=True)
        self.queue_removal_floor_s = _number(queue_removal_floor_s, 'queue_removal_floor_s')
        self.queue_signal = _choice(queue_signal, 'queue_signal', QUEUE_SIGNALS)
        self.decode_batch_source = _choice(decode_batch_source, 'decode_batch_source', DECODE_BATCH_SOURCES)
        if self.queue_p95_s and not self.restore_floor:
            raise ValueError('The queue add-P guard requires the restore floor')
        if self.queue_removal_floor_s and not self.queue_p95_s:
            raise ValueError('The queue removal hold requires the queue add-P guard')
        if self.queue_signal != 'all' and not self.queue_p95_s:
            raise ValueError('The ungated queue signal requires the queue add-P guard')
        if not (type(evict_pressure_hold) is bool or evict_pressure_hold == 'shadow'):
            raise ValueError('Invalid planner guard option evict_pressure_hold')
        self.evict_pressure_hold = evict_pressure_hold
        self.evict_pressure_tokens = _number(evict_pressure_tokens, 'evict_pressure_tokens')
        self.evict_pressure_window_s = _number(evict_pressure_window_s, 'evict_pressure_window_s', positive=True)
        self.evict_pressure_release = _choice(evict_pressure_release, 'evict_pressure_release', EVICT_RELEASES)
        if not evict_pressure_hold and (self.evict_pressure_tokens != EVICT_PRESSURE_TOKENS
                                        or self.evict_pressure_window_s != EVICT_PRESSURE_WINDOW_S
                                        or self.evict_pressure_release != 'queue_p95'):
            raise ValueError('The eviction-pressure options require the eviction-pressure hold')
        if evict_pressure_hold and self.evict_pressure_release == 'queue_p95' and not self.queue_p95_s:
            raise ValueError("The eviction-pressure release 'queue_p95' requires the queue add-P guard")

    @property
    def enabled(self):
        return bool(self.restore_floor or self.min_relative_gain or self.queue_p95_s
                    or self.queue_removal_floor_s or self.decode_batch_source != 'formula'
                    or self.evict_pressure_hold)

    def as_dict(self):
        options = dict(restore_floor=self.restore_floor, min_relative_gain=self.min_relative_gain,
                       queue_p95_s=self.queue_p95_s, queue_window_s=self.queue_window_s,
                       queue_min_since_switch_s=self.queue_min_since_switch_s,
                       queue_check_s=self.queue_check_s)
        # Listed only when on, so records and entry checks without them stay byte-identical.
        if self.queue_removal_floor_s:
            options['queue_removal_floor_s'] = self.queue_removal_floor_s
        if self.queue_signal != 'all':
            options['queue_signal'] = self.queue_signal
        if self.decode_batch_source != 'formula':
            options['decode_batch_source'] = self.decode_batch_source
        if self.evict_pressure_hold:
            options.update(evict_pressure_hold=self.evict_pressure_hold, evict_pressure_tokens=self.evict_pressure_tokens,
                           evict_pressure_window_s=self.evict_pressure_window_s,
                           evict_pressure_release=self.evict_pressure_release)
        return options

    @classmethod
    def from_configuration(cls, cfg):
        unknown = sorted(k for k in cfg if k.startswith('controller_guard_') and k not in CONFIGURATION_KEYS)
        if unknown:
            raise ValueError('Unknown planner guard options: ' + ', '.join(unknown))
        return cls(**{name: cfg[key] for key, name in CONFIGURATION_KEYS.items() if key in cfg})


def floor_restore_ratios(onboarding):
    """G1: return (floored observation map, record of the floor).

    Every observation whose ratio is below the cached-token-weighted mean of
    all observations is raised to that mean; the others are returned as is.
    """
    record = dict(applied=False, reference_ratio=None, cached_tokens=None, nodes={})
    if not onboarding:
        record['reason'] = 'no current onboarding observations'
        return onboarding, record
    cached = sum(o['cached_tokens'] for o in onboarding.values())
    if not (math.isfinite(cached) and cached > 0):
        record['reason'] = 'no cached tokens observed'
        return onboarding, record
    reference = sum(o['restored_per_cached_token']*o['cached_tokens'] for o in onboarding.values())/cached
    floored = {}
    for node, observation in onboarding.items():
        own = observation['restored_per_cached_token']
        # Equal ratios are not raised by a rounding difference in the mean.
        raised = own < reference and not math.isclose(own, reference, rel_tol=1e-12, abs_tol=0.)
        floored[node] = (dict(observation, restored_per_cached_token=reference,
                              observed_restored_per_cached_token=own) if raised else observation)
        record['nodes'][node] = dict(observed=own, used=floored[node]['restored_per_cached_token'],
                                     cached_tokens=observation['cached_tokens'], raised=raised)
    record.update(applied=any(n['raised'] for n in record['nodes'].values()),
                  reference_ratio=reference, cached_tokens=cached)
    return floored, record


# A record whose prefill_finished_time is zero is not a P leg (decode legs, aborts); skip it unparsed.
_NOT_PREFILLED = re.compile(rb'"prefill_finished_time":\s*0(?:\.0*)?\s*[,}]')


def percentile(values, fraction):
    """Linear-interpolation percentile, the definition used by the analysis."""
    if not values:
        return None
    values = sorted(values)
    k = (len(values)-1)*fraction
    a, b = math.floor(k), math.ceil(k)
    return values[a]+(values[b]-values[a])*(k-a)


_STREAM = re.compile(r'\.dp(\d+)_pp(\d+)\.')


def _trace_stream(path):
    """(dp, pp) rank of a scheduler trace file such as iter_trace.dp0_pp0.pid12.jsonl; (0, 0) if not named."""
    match = _STREAM.search(path.rsplit('/', 1)[-1])
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


class _TraceMonitor:
    """Role history and incremental complete-line reads of the workers' scheduler trace files.

    read() may run off the control thread: it only adds parsed items under the lock. Roles and windows are applied
    later, on the thread that records roles. Subclasses define _parse(node, line, stream) and _add(items); the
    stream of a file is its (dp, pp) rank from the trace file name.
    """

    def __init__(self, roles):
        self.history = {node: [(-math.inf, role)] for node, role in roles.items()}
        self.offsets = {}
        self.unreadable_lines = 0
        self.disabled = None
        self._lock = threading.Lock()

    def disable(self, reason):
        """Stop observing (e.g. the role history could not be kept); every later observation fails."""
        self.disabled = str(reason)

    def set_roles(self, roles, at):
        """Record the roles that hold from wall-clock time `at` (a confirmed switch)."""
        for node, role in roles.items():
            history = self.history.setdefault(node, [(-math.inf, None)])
            if at < history[-1][0]:
                raise ValueError('Role history must be recorded in time order')
            if history[-1][1] != role:
                history.append((at, role))

    def role_at(self, node, at):
        role = None
        for start, value in self.history.get(node, ()):
            if start > at:
                break
            role = value
        return role

    def read(self, traces):
        """Consume new complete lines of {node: [trace paths]}."""
        for node, paths in traces.items():
            for path in paths:
                path = str(path)
                stream = _trace_stream(path)
                offset = self.offsets.get(path, 0)
                with open(path, 'rb') as f:
                    f.seek(offset)
                    data = f.read()
                end = data.rfind(b'\n')+1
                items = [item for item in (self._parse(node, line, stream) for line in data[:end].splitlines()) if item]
                with self._lock:
                    self._add(items)
                self.offsets[path] = offset+end


class PrefillQueueMonitor(_TraceMonitor):
    """Trailing-window P_queue p95 from the workers' scheduler request traces.

    A P leg is a trace record with positive wait_queue_entry_time,
    forward_entry_time and prefill_finished_time on a node that held the
    prefill role at the leg's wait-queue entry. Engine monotonic times are
    converted with the record's mono_to_unix. P_queue = forward entry - wait
    entry; the window is keyed on forward entry. A record is written when its
    request completes on that worker, so the newest legs appear a few seconds
    late. Files are read incrementally, complete lines only.

    read() may run off the control thread: it only appends parsed legs. Roles
    and the window are applied by observe(), on the thread that records roles.

    With signal 'ungated' the observation (n, p95_s, max_s) uses only legs with
    late_d_need.late_d_gate_skips == 0 and also records the all-leg view; with
    'all' (the default) late_d_need is not parsed and nothing is added.
    """

    def __init__(self, roles, *, window_s=300., signal='all'):
        super().__init__(roles)
        self.window_s = _number(window_s, 'queue_window_s', positive=True)
        self.signal = _choice(signal, 'queue_signal', QUEUE_SIGNALS)
        self.legs = []

    def _add(self, legs):
        self.legs.extend(legs)

    def _parse(self, node, line, stream=None):
        if b'"prefill_finished_time"' not in line or _NOT_PREFILLED.search(line):
            return None
        try:
            row = json.loads(line)
            done, wait, forward = (float(row.get(k) or 0) for k in
                                   ('prefill_finished_time', 'wait_queue_entry_time', 'forward_entry_time'))
            shift = float(row.get('mono_to_unix') or 0) if done < 1e9 else 0.
            written = math.inf if row.get('ts') is None else float(row['ts'])
            gated = missing = None
            if self.signal == 'ungated':
                need = row.get('late_d_need')
                missing = not isinstance(need, dict)
                gated = not missing and float(need.get('late_d_gate_skips') or 0) > 0
        except (ValueError, TypeError, AttributeError):
            self.unreadable_lines += 1
            return None
        if not (done > 0 and wait > 0 and forward > 0):
            return None
        wait, forward = wait+shift, forward+shift
        return (forward, forward-wait, written, node, wait, gated, missing)

    def observe(self, now, *, visible_by=None):
        """P_queue p95 over legs with forward entry in (now - window, now].

        `visible_by` keeps only records written by then (offline replays of the
        live view). Legs far older than the window are dropped; callers use a
        nondecreasing `now`.
        """
        if self.disabled is not None:
            raise ValueError('P_queue monitor disabled: '+self.disabled)
        low = now-self.window_s
        with self._lock:
            self.legs = [leg for leg in self.legs if leg[0] > low-self.window_s]
            legs = list(self.legs)
        legs = [leg for leg in legs if low < leg[0] <= now and (visible_by is None or leg[2] <= visible_by)
                and self.role_at(leg[3], leg[4]) == 'prefill']
        waits = [leg[1] for leg in legs]
        if self.signal == 'all':
            return dict(at=now, window_s=self.window_s, n=len(waits), p95_s=percentile(waits, .95),
                        max_s=max(waits) if waits else None)
        ungated = [leg[1] for leg in legs if not leg[5]]
        p95 = percentile(ungated, .95)
        return dict(at=now, window_s=self.window_s, n=len(ungated), p95_s=p95,
                    max_s=max(ungated) if ungated else None, signal='ungated',
                    n_all=len(waits), p95_all=percentile(waits, .95), max_all=max(waits) if waits else None,
                    n_ungated=len(ungated), p95_ungated=p95,
                    gated_fraction=(len(waits)-len(ungated))/len(waits) if waits else None,
                    missing_late_d_need=sum(1 for leg in legs if leg[6]))



# Decode passes only; extend (prefill) passes and other records are skipped unparsed.
_DECODE_PASS = re.compile(rb'"mode":\s*"decode"')


class DecodeBatchMonitor(_TraceMonitor):
    """Time-weighted running decode batch per node from the workers' scheduler iteration traces.

    A decode pass is an iter_trace record with mode 'decode', ts (wall clock at the pass end), dur_ms and bs. It
    counts for a node only while the node held the decode role at the pass end. running = sum(bs * dur) / sum(dur)
    over the passes that end in the window: decode passes overlap (about x1.75), so neither a count-weighted nor a
    wall-clock mean is the running batch. A D node without a decode pass in the window runs 0. A node's DP ranks
    (one trace file each, dp<N> in the name) are summed; worker generations of one rank are pooled; pipeline ranks
    other than pp0 are not counted. Passes are kept as per-second sums; seconds older than `keep_s` before a
    window start are dropped.
    """

    def __init__(self, roles, *, keep_s=1800.):
        super().__init__(roles)
        self.keep_s = _number(keep_s, 'decode_keep_s', positive=True)
        self.seconds = {}

    def _add(self, passes):
        for node, dp, second, weighted, duration in passes:
            sums = self.seconds.setdefault(node, {}).setdefault(dp, {}).setdefault(second, [0., 0., 0])
            sums[0] += weighted
            sums[1] += duration
            sums[2] += 1

    def _parse(self, node, line, stream=(0, 0)):
        if stream[1] != 0 or not _DECODE_PASS.search(line):
            return None
        try:
            row = json.loads(line)
            if row.get('mode') != 'decode' or row.get('ts') is None or row.get('dur_ms') is None:
                return None
            end, duration, batch = float(row['ts']), float(row['dur_ms'])/1000., float(row.get('bs') or 0)
        except (ValueError, TypeError, AttributeError):
            self.unreadable_lines += 1
            return None
        if not (math.isfinite(end) and math.isfinite(duration) and duration > 0 and batch >= 0):
            return None
        return (node, stream[0], math.floor(end), batch*duration, duration)

    def running(self, start, end):
        """{node: {batch, passes, dur_s}} for the nodes holding the decode role at `end`, over passes in (start, end]."""
        if self.disabled is not None:
            raise ValueError('Decode batch monitor disabled: '+self.disabled)
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError('Invalid decode batch window')
        low, high = math.floor(start), math.floor(end)
        with self._lock:
            for ranks in self.seconds.values():
                for seconds in ranks.values():
                    for second in [k for k in seconds if k < low-self.keep_s]:
                        del seconds[second]
            per_node = {node: {dp: [(second, list(sums)) for second, sums in seconds.items() if low < second <= high]
                               for dp, seconds in ranks.items()} for node, ranks in self.seconds.items()}
        out = {}
        for node in sorted(self.history):
            if self.role_at(node, end) != 'decode':
                continue
            batch = duration = 0.
            passes = 0
            for dp, seconds in sorted(per_node.get(node, {}).items()):
                w = d = 0.
                for second, (weighted, dur, n) in seconds:
                    if self.role_at(node, second) == 'decode':
                        w, d, passes = w+weighted, d+dur, passes+n
                batch, duration = batch+(w/d if d > 0 else 0.), duration+d
            out[node] = dict(batch=batch, passes=passes, dur_s=duration)
        return out


# TP0 KV_SESSION_STATS line of an engine log: "[2026-10-23 16:44:01.123 TP0] KV_SESSION_STATS final=0 live=..".
_SESSION_STATS = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(\d+))? TP0\] KV_SESSION_STATS ')
_COUNTER = re.compile(r'(\w+)=(-?\d+)(?=\s|$)')
_COUNTERS = ('evicted_c1_host_only', 'evicted_c1_dup', 'evicted_active_tokens')
_PREFIX = re.compile(rb'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(\d+))?[ \]]')


def naive_log_time(text, fraction=None):
    """The engine log prefix read as if it were UTC (the prefix carries no zone); the read adds the log clock offset."""
    return calendar.timegm(time.strptime(text, '%Y-%m-%d %H:%M:%S')) + (float('0.'+fraction) if fraction else 0.)


def log_clock_offset(now, naive):
    """(offset, residual): the log clock offset in EVICT_CLOCK_STEP_S steps that puts `naive` just before `now`."""
    offset = round((now-naive)/EVICT_CLOCK_STEP_S)*EVICT_CLOCK_STEP_S
    return offset, now-(naive+offset)


class HostEvictionMonitor(_TraceMonitor):
    """P-node class-1 host-only evictions over a trailing window, from the engines' KV_SESSION_STATS lines.

    Each engine log file (worker generation; a new inode or a shorter file starts a new stream) is one stream of
    cumulative counters that start at 0 with the process; only TP0 lines are read. The log prefix has no zone: each
    read keeps the stream's (else the node's) previous offset while it puts the chunk's last time-stamped line
    within [-EVICT_CLOCK_AHEAD_S, EVICT_CLOCK_LAG_S] s before the file's last write (mtime; the controller's clock if
    the mtime is later); otherwise it takes the 15-min step that does, else the previous offset; a line that would
    then lie in
    the future or before the stream's previous sample uses the previous offset (a DST change inside one read), and a
    line that still does not fit is dropped (counted). An interval between consecutive samples of a stream counts
    when its end lies in (now - window, now] and the node held the prefill role at that end; a negative step
    (counter reset) counts 0. Samples later than `now` are not visible. A current P node without a visible sample
    in the last EVICT_PRESSURE_MAX_AGE_S has no signal. A file or node that cannot be read is skipped and recorded
    (`read_failures`), the other nodes are read.
    """

    def __init__(self, roles, *, window_s=EVICT_PRESSURE_WINDOW_S, max_age_s=EVICT_PRESSURE_MAX_AGE_S, clock=None):
        super().__init__(roles)
        self.window_s = _number(window_s, 'evict_pressure_window_s', positive=True)
        self.max_age_s = _number(max_age_s, 'evict_pressure_max_age_s', positive=True)
        self.clock = clock  # None: time.time() at each read
        self.samples = {}
        self.files = {}
        self.clock_offsets = {}
        self.clock_unresolved = {}
        self.read_failures = {}
        self.restarted_streams = 0

    def _parse(self, node, line, stream=None):
        """(naive time, counters) of a TP0 KV_SESSION_STATS line, else None."""
        if b' TP0] KV_SESSION_STATS ' not in line:
            return None
        try:
            text = line.decode(errors='replace')
            match = _SESSION_STATS.match(text)
            if not match:
                return None
            naive = naive_log_time(match.group(1), match.group(2))
            counters = {k: int(v) for k, v in _COUNTER.findall(text[match.end():])}
            values = tuple(counters[k] if k in counters else None for k in _COUNTERS)
        except (ValueError, OverflowError):
            self.unreadable_lines += 1
            return None
        if values[0] is None or not math.isfinite(naive):
            self.unreadable_lines += 1
            return None
        return naive, values

    def read(self, traces, failed=None):
        """Consume new complete lines of {node: [engine log paths]}; `failed` = {node: reason} the caller could not list."""
        failures = dict(failed or {})
        for node, paths in traces.items():
            for path in paths:
                path = str(path)
                try:
                    self._read_file(node, path)
                except (OSError, ValueError, OverflowError) as error:
                    failures[node] = path+': '+str(error)
        self.read_failures = failures

    def _read_file(self, node, path):
        state = self.files.get(path)
        with open(path, 'rb') as f:
            info = os.fstat(f.fileno())
            if state is None or state['inode'] != info.st_ino or info.st_size < state['offset']:
                if state is not None:
                    self.restarted_streams += 1
                state = dict(inode=info.st_ino, offset=0, epoch=0 if state is None else state['epoch']+1,
                             offset_s=None, last=None)
                self.files[path] = state
            f.seek(state['offset'])
            data = f.read()
            written = os.fstat(f.fileno()).st_mtime
        end = data.rfind(b'\n')+1
        lines = data[:end].splitlines()
        parsed = [item for item in (self._parse(node, line) for line in lines) if item]
        state['offset'] += end
        if not parsed:
            return
        now = time.time() if self.clock is None else self.clock()
        reference = written if written <= now+EVICT_CLOCK_AHEAD_S else now
        last = parsed[-1][0]
        for line in reversed(lines):  # the last time-stamped line of any kind was written just before the mtime
            match = _PREFIX.match(line)
            if match:
                try:
                    last = naive_log_time(match.group(1).decode(), match.group(2) and match.group(2).decode())
                except ValueError:
                    continue
                break
        fits = lambda candidate: -EVICT_CLOCK_AHEAD_S <= reference-(last+candidate) <= EVICT_CLOCK_LAG_S  # noqa: E731
        # The stream's previous offset (else the node's, for a new generation file) is kept while it fits.
        previous = state['offset_s'] if state['offset_s'] is not None else self.clock_offsets.get(node)
        if previous is not None and fits(previous):
            offset = previous
        else:
            offset = log_clock_offset(reference, last)[0]
            if not fits(offset):
                offset = previous  # unresolved this read: keep the previous offset, if any
        if offset is not None:
            state['offset_s'] = offset
        rows, dropped = [], 0
        for naive, values in parsed:
            at = None
            for candidate in (offset, previous):
                if candidate is None:
                    continue
                unix = naive+candidate
                if unix <= now+EVICT_CLOCK_AHEAD_S and (state['last'] is None or unix >= state['last']-EVICT_CLOCK_AHEAD_S):
                    at = unix
                    break
            if at is None:
                dropped += 1
                continue
            state['last'] = at
            rows.append((node, (path, state['inode'], state['epoch']), at, values))
        if offset is not None:
            self.clock_offsets[node] = offset
        if dropped:
            self.clock_unresolved[node] = self.clock_unresolved.get(node, 0)+dropped
            self.unreadable_lines += dropped
        with self._lock:
            self._add(rows)

    def _add(self, samples):
        for node, stream, at, values in samples:
            self.samples.setdefault((node, stream), []).append((at, values))

    def add_samples(self, samples):
        """Add (node, stream, unix time, (c1 host-only, c1 dup, active)) samples (offline replays)."""
        with self._lock:
            self._add(samples)

    def observe(self, now):
        """The P nodes' class-1 host-only evicted tokens in (now - window, now], per node and summed."""
        if self.disabled is not None:
            raise ValueError('Eviction monitor disabled: '+self.disabled)
        low = now-self.window_s
        with self._lock:
            streams = {key: sorted(rows, key=lambda row: row[0]) for key, rows in self.samples.items()}
        tokens, dup, active, newest, later = {}, 0, 0, {}, 0
        for (node, _), rows in sorted(streams.items(), key=lambda item: (item[0][0], str(item[0][1]))):
            previous = (0, 0, 0)  # the counters start at 0 with the engine process
            for at, values in rows:
                if at > now:
                    later += 1
                    continue
                newest[node] = max(newest.get(node, -math.inf), at)
                steps = [max(0, (v or 0)-(p or 0)) for v, p in zip(values, previous)]
                previous = values
                if at > low and self.role_at(node, at) == 'prefill':
                    tokens[node] = tokens.get(node, 0)+steps[0]
                    dup += steps[1]
                    active += steps[2]
        prefill = sorted(node for node in self.history if self.role_at(node, now) == 'prefill')
        missing = [node for node in prefill if not newest.get(node, -math.inf) >= now-self.max_age_s]
        total = sum(tokens.values())
        return dict(at=now, window_s=self.window_s, counter='evicted_c1_host_only', tokens=total,
                    tokens_per_p=total/len(prefill) if prefill else None,
                    nodes=dict(sorted(tokens.items())), prefill_nodes=prefill, nodes_without_signal=missing,
                    signal=len(missing) < len(prefill), complete=not missing, c1_dup_tokens=dup,
                    active_tokens=active, max_age_s=self.max_age_s, samples_after_now=later,
                    newest_age_s={node: now-newest[node] for node in prefill if node in newest},
                    read_failed_nodes=sorted(self.read_failures))
