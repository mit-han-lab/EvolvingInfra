"""Opt-in eviction-pressure removal hold (E-hold) and its engine-log monitor (HostEvictionMonitor)."""
import calendar
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from evolving_infra.planner.calibration.controller import Controller
from evolving_infra.planner.calibration.dynamic import _prefill_count, _removal_and_add_guards, decide
from evolving_infra.planner.calibration.guards import (EVICT_PRESSURE_TOKENS, HostEvictionMonitor, PlannerGuards, log_clock_offset,
                                   naive_log_time)

ALL = dict(restore_floor=True, min_relative_gain=.03, queue_p95_s=20.)
QHOLD = dict(ALL, queue_removal_floor_s=10.)
EHOLD = dict(evict_pressure_hold=True, evict_pressure_release='none')  # the core rule, no release bound
EHOLD_Q = dict(ALL, evict_pressure_hold=True)  # the default release bound 'queue_p95' (needs Q')
PDT = 7*3600  # log clock offset of a PDT engine log: unix = naive + offset
STEADY_2 = [.02, .021, .019, .02, .021, .019, .02, .02, .021, .019]
STEADY_3 = [.03, .031, .029, .03, .031, .029, .03, .03, .031, .029]
REMOVE_2P = {2: [g+.1 for g in STEADY_3]}  # 3P -> 2P: +13 %, consistent; every other P count -50 %
FIXTURES = Path(__file__).parent/'fixtures'
FLIP = dict(prefill='decode', decode='prefill')


def scripted(gains, current_p=3, nodes=9, guards=None, queue=None, queue_added=False, evict=None, **extra):
    """decide() with system rps 10 * (1 + gains[P count][minute]); the current P count is 10."""
    roles = {str(i): 'prefill' if i < current_p else 'decode' for i in range(nodes)}
    count = len(next(iter(gains.values()), [0.]*10))

    def estimate(workers, candidate, work, **kwargs):
        p = sum(r == 'prefill' for r in candidate.values())
        rps = 10. if p == current_p else 10.*(1+gains.get(p, [-.5]*count)[work['minute']])
        return dict(system_rps=rps, prefill_rps=rps, decode_rr_rps=rps)
    minutes = [dict(work=dict(minute=i), decode_population=None) for i in range(count)]
    with patch('evolving_infra.planner.calibration.dynamic.capacities', side_effect=estimate):
        return decide({n: {} for n in roles}, roles, dict(minute=0), minutes=minutes, experts=256, top_k=8,
                      guards=guards, queue=queue, queue_added=queue_added, evict_pressure=evict, **extra)


def p_count(option):
    return sum(r == 'prefill' for r in option['roles'].values())


def pressure(tokens, complete=True, signal=True):
    return dict(at=1.e9, window_s=600., counter='evicted_c1_host_only', tokens=tokens, nodes={'0': tokens},
                prefill_nodes=['0', '1', '2'], nodes_without_signal=[] if complete else ['2'], signal=signal,
                complete=complete)


def queue(p95=15., since=300.):
    return dict(p95_s=p95, n=1500, window_s=300., at=1.e9, since_switch_s=since)


def stats_line(at, c1, rank='TP0', dup=0, active=0, ms=True, offset=PDT):
    """A KV_SESSION_STATS line at unix `at` from an engine whose log clock is `offset` s behind UTC (PDT default)."""
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(at-offset)) + ('.%03d' % round((at % 1)*1000) if ms else '')
    return (f'[{stamp} {rank}] KV_SESSION_STATS final=0 live=3 released=1 evicted_c0_host_only=7 evicted_c0_dup=0 '
            f'evicted_c1_host_only={c1} evicted_c1_dup={dup} class0_host_tokens=0 evicted_active_tokens={active} '
            f'scan_ms=1.2\n')


class OptionTests(unittest.TestCase):
    def test_off_by_default_and_not_listed(self):
        guards = PlannerGuards()
        self.assertFalse(guards.evict_pressure_hold)
        self.assertFalse(guards.enabled)
        self.assertEqual((guards.evict_pressure_tokens, guards.evict_pressure_window_s), (1.e6, 600.))
        self.assertEqual(guards.as_dict(), dict(restore_floor=False, min_relative_gain=0., queue_p95_s=0.,
                         queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60.))
        self.assertEqual(PlannerGuards(**QHOLD, evict_pressure_hold=False).as_dict(), PlannerGuards(**QHOLD).as_dict())
        self.assertNotIn('evict_pressure_hold', PlannerGuards(**ALL).as_dict())

    def test_configuration_keys_and_independence_from_the_queue_guards(self):
        guards = PlannerGuards.from_configuration(dict(controller_guard_evict_pressure_hold=True,
                                                       controller_guard_evict_pressure_release='none'))
        self.assertTrue(guards.enabled)  # without the queue release it needs neither G1 nor Q'
        self.assertEqual(guards.as_dict(), dict(restore_floor=False, min_relative_gain=0., queue_p95_s=0.,
                         queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60.,
                         evict_pressure_hold=True, evict_pressure_tokens=1.e6, evict_pressure_window_s=600.,
                         evict_pressure_release='none'))
        # The default release bound reads the Q' observation, so it needs Q'.
        with self.assertRaisesRegex(ValueError, "release 'queue_p95' requires the queue add-P guard"):
            PlannerGuards.from_configuration(dict(controller_guard_evict_pressure_hold=True))
        with self.assertRaisesRegex(ValueError, "release 'queue_p95' requires the queue add-P guard"):
            PlannerGuards(evict_pressure_hold='shadow')
        cell = PlannerGuards.from_configuration(dict(
            controller_guard_restore_floor=True, controller_guard_min_relative_gain=.03, controller_guard_queue_p95_s=20,
            controller_guard_evict_pressure_hold=True))
        self.assertEqual(cell.as_dict(), dict(restore_floor=True, min_relative_gain=.03, queue_p95_s=20.,
                         queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60., evict_pressure_hold=True,
                         evict_pressure_tokens=1.e6, evict_pressure_window_s=600., evict_pressure_release='queue_p95'))
        shadow = PlannerGuards(**ALL, evict_pressure_hold='shadow')
        self.assertTrue(shadow.enabled)
        self.assertEqual(shadow.as_dict()['evict_pressure_hold'], 'shadow')
        guards = PlannerGuards.from_configuration(dict(
            controller_guard_restore_floor=True, controller_guard_min_relative_gain=.03, controller_guard_queue_p95_s=20,
            controller_guard_queue_removal_floor_s=10, controller_guard_evict_pressure_hold=True,
            controller_guard_evict_pressure_tokens=0, controller_guard_evict_pressure_window_s=300))
        self.assertEqual(guards.as_dict(), dict(restore_floor=True, min_relative_gain=.03, queue_p95_s=20.,
                         queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60., queue_removal_floor_s=10.,
                         evict_pressure_hold=True, evict_pressure_tokens=0., evict_pressure_window_s=300.,
                         evict_pressure_release='queue_p95'))

    def test_invalid_options_fail(self):
        for bad in (dict(evict_pressure_hold=1), dict(evict_pressure_hold='true'), dict(evict_pressure_hold=None),
                    dict(evict_pressure_hold='Shadow'), dict(evict_pressure_hold=0),
                    dict(EHOLD, evict_pressure_release='ungated'), dict(EHOLD, evict_pressure_release=None),
                    dict(EHOLD, evict_pressure_tokens=-1.), dict(EHOLD, evict_pressure_tokens=float('nan')),
                    dict(EHOLD, evict_pressure_tokens=float('inf')), dict(EHOLD, evict_pressure_tokens=True),
                    dict(EHOLD, evict_pressure_tokens='1e6'), dict(EHOLD, evict_pressure_window_s=0.),
                    dict(EHOLD, evict_pressure_window_s=-600.), dict(EHOLD, evict_pressure_window_s=None)):
            with self.assertRaises(ValueError, msg=bad):
                PlannerGuards(**bad)
        for orphan in (dict(evict_pressure_tokens=0.), dict(evict_pressure_window_s=300.),
                       dict(evict_pressure_release='gain'), dict(ALL, evict_pressure_release='none')):
            with self.assertRaisesRegex(ValueError, 'eviction-pressure options require the eviction-pressure hold'):
                PlannerGuards(**orphan)
        # Default values without the hold are accepted (nothing changes).
        self.assertFalse(PlannerGuards(evict_pressure_tokens=EVICT_PRESSURE_TOKENS, evict_pressure_window_s=600).enabled)


class OffIdentityTests(unittest.TestCase):
    def test_off_records_are_unchanged_whatever_the_observation(self):
        cases = [REMOVE_2P, {4: STEADY_2}, {**REMOVE_2P, 4: [g+.01 for g in STEADY_3]}, {}]
        for gains in cases:
            for options in (None, ALL, QHOLD, dict(decode_batch_source='formula')):
                for q, added in ((queue(25.), True), (queue(15.), False), (None, True)):
                    guards = None if options is None else PlannerGuards(**options)
                    base = scripted(gains, guards=guards, queue=q, queue_added=added)
                    for evict in (pressure(5.e7), pressure(0), None):
                        result = scripted(gains, guards=guards, queue=q, queue_added=added, evict=evict)
                        self.assertEqual(json.dumps(result), json.dumps(base), (gains, options))
                        self.assertNotIn('evict_pressure_hold', (result.get('guards') or {}))


class RuleTests(unittest.TestCase):
    def run_rule(self, evict, gains=REMOVE_2P, options=EHOLD, **kwargs):
        return scripted(gains, guards=PlannerGuards(**options), evict=evict, **kwargs)

    def test_blocks_a_removal_above_the_threshold_and_holds(self):
        result = self.run_rule(pressure(16.41e6))
        self.assertEqual((result['action'], result['selected']), ('hold', result['candidates'][0]))
        g = result['guards']
        e = g['evict_pressure_hold']
        self.assertEqual({k: e[k] for k in ('threshold_tokens', 'window_s', 'tokens', 'complete', 'blocked', 'selected',
                                            'reason')},
                         dict(threshold_tokens=1.e6, window_s=600., tokens=16.41e6, complete=True, blocked=True,
                              selected=None, reason='P removal blocked, no remaining candidate: hold'))
        self.assertEqual(e['candidate']['prefill_nodes'], 2)
        self.assertEqual(e['observation']['tokens'], 16.41e6)
        self.assertEqual((g['selected_by'], g['fired'], g['changed_decision']),
                         ('evict_pressure_hold', ['evict_pressure_hold'], True))
        self.assertEqual(g['selections']['standard']['prefill_nodes'], 2)  # the standard choice is still recorded
        self.assertIsNone(g['queue_add_prefill'])
        self.assertEqual(g['options']['evict_pressure_hold'], True)
        json.dumps(result)

    def test_threshold_edge(self):
        for tokens, threshold, blocked in ((1.e6, 1.e6, False), (1.e6+1, 1.e6, True), (0, 0., False), (1, 0., True),
                                           (2.17e6, 1.e6, True), (999999, 1.e6, False)):
            result = self.run_rule(pressure(tokens), options=dict(EHOLD, evict_pressure_tokens=threshold))
            e = result['guards']['evict_pressure_hold']
            self.assertEqual(e['blocked'], blocked, (tokens, threshold))
            self.assertEqual(p_count(result['selected']), 3 if blocked else 2)
            if not blocked:
                self.assertEqual(e['reason'], 'P class-1 host evictions not above the threshold')
                self.assertEqual((result['guards']['selected_by'], result['guards']['fired']), ('minute_standard_error', []))

    def test_missing_signal_does_not_block(self):
        for evict in (None, pressure(5.e7, signal=False), pressure(float('nan')), pressure(None), pressure(True),
                      pressure(float('inf'))):
            result = self.run_rule(evict)
            e = result['guards']['evict_pressure_hold']
            self.assertEqual((e['blocked'], e['reason']), (False, 'no eviction signal'), evict)
            self.assertEqual(p_count(result['selected']), 2)

    def test_incomplete_signal_blocks_only_on_a_visible_total_above_the_threshold(self):
        result = self.run_rule(pressure(5.e6, complete=False))
        self.assertTrue(result['guards']['evict_pressure_hold']['blocked'])
        self.assertFalse(result['guards']['evict_pressure_hold']['complete'])
        result = self.run_rule(pressure(5.e5, complete=False))
        e = result['guards']['evict_pressure_hold']
        self.assertEqual((e['blocked'], e['reason']),
                         (False, 'eviction signal incomplete, visible total not above the threshold'))
        self.assertEqual(p_count(result['selected']), 2)

    def test_additions_and_holds_are_never_blocked(self):
        add = self.run_rule(pressure(9.e9), gains={4: [g+.01 for g in STEADY_3]})
        hold = self.run_rule(pressure(9.e9), gains={})
        self.assertEqual((add['action'], p_count(add['selected'])), ('switch', 4))
        self.assertEqual(hold['action'], 'hold')
        for result in (add, hold):
            e = result['guards']['evict_pressure_hold']
            self.assertEqual((e['blocked'], e['candidate'], e['selected'], e['reason']),
                             (False, None, None, 'standard rule does not remove a P'))
            self.assertNotIn('evict_pressure_hold', result['guards']['fired'])

    def test_blocked_removal_selects_the_best_candidate_without_removal(self):
        gains = {**REMOVE_2P, 1: [g+.07 for g in STEADY_3], 4: [g+.01 for g in STEADY_3], 5: [g+.005 for g in STEADY_3]}
        result = self.run_rule(pressure(5.e7), gains=gains)
        self.assertEqual((result['action'], p_count(result['selected'])), ('switch', 4))
        e = result['guards']['evict_pressure_hold']
        self.assertEqual((e['candidate']['prefill_nodes'], e['selected']['prefill_nodes'], e['reason']),
                         (2, 4, 'P removal blocked, best remaining candidate selected'))
        # A remaining candidate must pass the standard rule (2 SE); a G3-failing +P is not selected with G3 on.
        self.assertEqual(self.run_rule(pressure(5.e7), gains={**REMOVE_2P, 4: STEADY_2},
                                       options=dict(ALL, **EHOLD), queue=queue(15.))['action'], 'hold')

    def test_queue_add_prefill_is_unchanged_and_overrides(self):
        # Q' reads the standard choice: wherever it fires with E-hold off it fires the same with E-hold on.
        cases = [REMOVE_2P, {**REMOVE_2P, 4: STEADY_2}, {**REMOVE_2P, 4: [g-.02 for g in STEADY_3], 5: STEADY_3},
                 {4: STEADY_2}, {2: STEADY_3}]
        fired = 0
        for gains in cases:
            for p95 in (15., 25., 60.):
                off = scripted(gains, guards=PlannerGuards(**ALL), queue=queue(p95))
                on = scripted(gains, guards=PlannerGuards(**ALL, **EHOLD), queue=queue(p95), evict=pressure(5.e7))
                self.assertEqual(on['guards']['queue_add_prefill'], off['guards']['queue_add_prefill'])
                if off['guards']['queue_add_prefill']['fired']:
                    fired += 1
                    self.assertEqual((on['action'], on['selected']['changed']), (off['action'], off['selected']['changed']))
                    self.assertEqual(on['guards']['selected_by'], 'queue_add_prefill')
        self.assertGreaterEqual(fired, 4)
        # The removal is blocked and the +1 P fails G3: Q' adds it in the same decision; both guards are listed.
        result = scripted({**REMOVE_2P, 4: STEADY_2}, guards=PlannerGuards(**ALL, **EHOLD), queue=queue(25.),
                          evict=pressure(5.e7))
        self.assertEqual((result['action'], p_count(result['selected'])), ('switch', 4))
        self.assertTrue(result['guards']['evict_pressure_hold']['blocked'])
        self.assertEqual(result['guards']['fired'], ['min_relative_gain', 'evict_pressure_hold', 'queue_add_prefill'])

    def test_with_queue_removal_hold_either_hold_blocks(self):
        both = dict(QHOLD, **EHOLD)
        cases = (  # (queue_added, p95, evicted tokens) -> Q-hold blocks, E-hold blocks
            (True, 15., 5.e7, True, True), (True, 15., 0, True, False), (False, 15., 5.e7, False, True),
            (True, 5., 5.e7, False, True), (False, 15., 0, False, False))
        for added, p95, tokens, qhold, ehold in cases:
            result = scripted(REMOVE_2P, guards=PlannerGuards(**both), queue=queue(p95), queue_added=added,
                              evict=pressure(tokens))
            g = result['guards']
            self.assertEqual((g['queue_removal_hold']['blocked'], g['evict_pressure_hold']['blocked']), (qhold, ehold))
            self.assertEqual(p_count(result['selected']), 3 if qhold or ehold else 2)
            self.assertEqual(g['selected_by'], 'queue_removal_hold' if qhold else 'evict_pressure_hold' if ehold
                             else 'minute_standard_error')
            self.assertEqual(g['fired'], [n for n, f in (('queue_removal_hold', qhold), ('evict_pressure_hold', ehold)) if f])
            # Q-hold's own record is the one it has without E-hold.
            alone = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD), queue=queue(p95), queue_added=added)
            self.assertEqual(g['queue_removal_hold'], alone['guards']['queue_removal_hold'])


# Verbatim engine.log lines (TP0 KV_SESSION_STATS, TP1 eviction-order line, TP0 prefill batch line).
REAL_TP0_A = ('[2026-10-19 00:23:28.226 TP0] KV_SESSION_STATS final=0 live=0 released=0 ended=0 tagged_arrivals=0 aborts=0 '
              'abort_reasons=- mark_untracked=4 mark_retired=0 known_zero_host_tokens=0 evicted_c0_host_only=0 '
              'evicted_c0_dup=0 evicted_c1_host_only=0 evicted_c1_dup=0 prefetch_extended=0 soft_pin=1 class0_host_tokens=0 '
              'active_nodes=0 evicted_active_tokens=0 scan_ms=0.8 order=lru order_active=0 table=- return_gaps=0 '
              'table_gaps=0 m0=- m30=- m60=- m90=- m110=- c1_evicted_idle_0_10=0 c1_evicted_idle_10_60=0 '
              'c1_evicted_idle_60_100=0 c1_evicted_idle_100p=0 clock_stamps=48 clock_s=57.8')
REAL_TP0_B = ('[2026-10-19 02:00:53.581 TP0] KV_SESSION_STATS final=0 live=56 released=17 ended=1506 tagged_arrivals=9110 '
              'aborts=0 abort_reasons=- mark_untracked=99 mark_retired=0 known_zero_host_tokens=0 '
              'evicted_c0_host_only=31278912 evicted_c0_dup=2582144 evicted_c1_host_only=22535744 evicted_c1_dup=0 '
              'prefetch_extended=0 soft_pin=1 class0_host_tokens=0 active_nodes=0 evicted_active_tokens=0 scan_ms=7.1 '
              'order=lru order_active=0 table=- return_gaps=7521 table_gaps=7521 m0=18.3 m30=58.5 m60=56.6 m90=38.1 '
              'm110=21.4 c1_evicted_idle_0_10=0 c1_evicted_idle_10_60=558592 c1_evicted_idle_60_100=5728384 '
              'c1_evicted_idle_100p=16248768 clock_stamps=8549 clock_s=5903.7')
REAL_TP1 = ('[2026-10-20 19:13:55.265 TP1] HICACHE_EVICT_ORDER rank=1 checks=60315 host_release=2098601451120 '
            'device_evict=3793770574544 flagged=0 next_node_id=653')
REAL_TP0_OTHER = ('[2026-10-18 17:26:40.178 TP0] Prefill batch, #new-seq: 2, #new-token: 6912, #cached-token: 344320, '
                  'token usage: 0.32, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, '
                  'input throughput (token/s): 13614.59, fwd occupancy: 67.40%')


def evict_stats_values(line):
    """(unix, counters) of a line as evict_stats.py 9239f67e computes them (its regexes; log clock PDT, +7 h)."""
    import re
    ts = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.(\d+) TP0\] (KV_SESSION_STATS|HICACHE_EVICT_ORDER) ')
    kv = re.compile(r'(\w+)=(-?\d+)(?=\s|$)')
    m = ts.match(line)
    if not m or m.group(3) != 'KV_SESSION_STATS':
        return None
    unix = calendar.timegm(time.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')) + float('0.' + m.group(2)) + 7 * 3600
    return unix, {k: int(v) for k, v in kv.findall(line[m.end():])}


def in_zone(zone, function):
    """Run `function` with the process time zone set to `zone` (TZ + tzset), then restore it."""
    old = os.environ.get('TZ')
    os.environ['TZ'] = zone
    time.tzset()
    try:
        return function()
    finally:
        if old is None:
            del os.environ['TZ']
        else:
            os.environ['TZ'] = old
        time.tzset()


def utc(text):
    return calendar.timegm(time.strptime(text, '%Y-%m-%d %H:%M:%S'))


def rows(monitor, node):
    """[(unix, values)] of every stream of `node`, in stream then time order."""
    return [row for (n, _), samples in sorted(monitor.samples.items(), key=lambda kv: str(kv[0])) if n == node
            for row in samples]


class MonitorTests(unittest.TestCase):
    T = 1.7927e9  # a 2026-10 unix time

    def test_clock_offset_steps(self):
        for offset in (0, PDT, 8*3600, -19800, 20700):  # UTC, PDT, PST, India (+5:30), Nepal (+5:45)
            for age in (0., 59., 290.):
                self.assertEqual(log_clock_offset(self.T+age, self.T-offset), (offset, age))
        self.assertEqual(log_clock_offset(self.T+500., self.T-PDT)[1], 500.-900.)  # 500 s old: not accepted

    def test_real_lines_match_evict_stats_in_any_controller_zone(self):
        expected = [evict_stats_values(line) for line in (REAL_TP0_A, REAL_TP0_B)]
        self.assertIsNone(evict_stats_values(REAL_TP1))
        self.assertIsNone(evict_stats_values(REAL_TP0_OTHER))

        def parse():
            with tempfile.TemporaryDirectory() as temp:
                log = Path(temp)/'engine.log'
                log.write_text('\n'.join((REAL_TP0_A, REAL_TP1, REAL_TP0_OTHER, REAL_TP0_B)) + '\n')
                os.utime(log, (expected[1][0]+1., expected[1][0]+1.))  # the file's last write: just after line B
                m = HostEvictionMonitor({'n0': 'prefill'}, clock=lambda: expected[1][0]+40.)
                m.read({'n0': [log]})
                return rows(m, 'n0'), m.clock_offsets, m.unreadable_lines
        for zone in ('UTC', 'America/Los_Angeles', 'Asia/Kolkata'):
            got, offsets, unreadable = in_zone(zone, parse)
            self.assertEqual((offsets, unreadable), ({'n0': PDT}, 0), zone)
            self.assertEqual(len(got), 2, zone)
            for (at, values), (unix, counters) in zip(got, expected):
                self.assertAlmostEqual(at, unix, places=6)
                self.assertEqual(values, tuple(counters[k] for k in ('evicted_c1_host_only', 'evicted_c1_dup',
                                                                      'evicted_active_tokens')))
        self.assertEqual(got[1][1][0], 22535744)

    def test_dst_end_and_start_keep_true_times(self):
        def run(lines_by_read, offsets_by_line):
            with tempfile.TemporaryDirectory() as temp:
                log = Path(temp)/'engine.log'
                now = [0.]
                m = HostEvictionMonitor({'n0': 'prefill'}, clock=lambda: now[0])
                for lines, read_at in lines_by_read:
                    with log.open('a') as f:
                        f.write(''.join(stats_line(at, 1, offset=offsets_by_line(at)) for at in lines))
                    os.utime(log, (lines[-1]+1., lines[-1]+1.))
                    now[0] = read_at
                    m.read({'n0': [log]})
                return [at for at, _ in rows(m, 'n0')], m.clock_unresolved
        # 2026-11-01: PDT (UTC-7) until 09:00Z, then PST (UTC-8): local 01:00-02:00 repeats.
        change = utc('2026-11-01 09:00:00')
        fall = lambda at: PDT if at < change else 8*3600  # noqa: E731
        minutes = [change+60*k for k in range(-3, 3)]
        two_reads = run([(minutes[:3], minutes[2]+30), (minutes[3:], minutes[5]+20)], fall)
        mixed = run([(minutes[:1], minutes[0]+20), (minutes[1:], minutes[5]+20)], fall)  # one read spans the change
        self.assertEqual(two_reads, (minutes, {}))
        self.assertEqual(mixed, (minutes, {}))
        # 2027-03-14: PST until 10:00Z, then PDT: local 02:00-03:00 is skipped.
        change = utc('2027-03-14 10:00:00')
        spring = lambda at: 8*3600 if at < change else PDT  # noqa: E731
        minutes = [change+60*k for k in range(-2, 2)]
        self.assertEqual(run([(minutes[:1], minutes[0]+20), (minutes[1:], minutes[3]+20)], spring), (minutes, {}))

    def test_stale_file_resolves_from_its_mtime_and_an_unmatched_write_is_unresolved(self):
        T = self.T
        with tempfile.TemporaryDirectory() as temp:
            # A dead engine's log (last line 2 h before the read): its mtime gives the offset, the node is stale.
            dead = Path(temp)/'dead.log'
            dead.write_text(stats_line(T, 10) + stats_line(T+60, 20))
            os.utime(dead, (T+61., T+61.))
            m = HostEvictionMonitor({'n0': 'prefill'}, clock=lambda: T+7200.)
            m.read({'n0': [dead]})
            self.assertEqual(([at for at, _ in rows(m, 'n0')], m.clock_offsets, m.clock_unresolved),
                             ([T, T+60], {'n0': PDT}, {}))
            obs = m.observe(T+7200)
            self.assertEqual((obs['signal'], obs['nodes_without_signal'], obs['tokens']), (False, ['n0'], 0))
            # Stats lines, then other time-stamped output: the last time-stamped line is the reference.
            busy = Path(temp)/'busy.log'
            stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(T+3000-PDT))
            busy.write_text(stats_line(T, 10) + stats_line(T+60, 20) + f'[{stamp}.500 TP0] Decode batch, #running-req: 9\n')
            os.utime(busy, (T+3001., T+3001.))
            m = HostEvictionMonitor({'n1': 'prefill'}, clock=lambda: T+3010.)
            m.read({'n1': [busy]})
            self.assertEqual(([at for at, _ in rows(m, 'n1')], m.clock_offsets), ([T, T+60], {'n1': PDT}))
            # The last write 450 s after the last time-stamped line (no offset fits) and no previous offset:
            # unresolved, dropped and counted; a later fresh line resolves the clock.
            late = Path(temp)/'late.log'
            late.write_text(stats_line(T, 10) + stats_line(T+60, 20) + 'Traceback (most recent call last):\n')
            os.utime(late, (T+510., T+510.))
            m = HostEvictionMonitor({'n2': 'prefill'}, clock=lambda: T+520.)
            m.read({'n2': [late]})
            self.assertEqual((rows(m, 'n2'), m.clock_unresolved, m.unreadable_lines), ([], {'n2': 2}, 2))
            with late.open('a') as f:
                f.write(stats_line(T+600, 90))
            os.utime(late, (T+601., T+601.))
            m.clock = lambda: T+610.
            m.read({'n2': [late]})
            self.assertEqual((rows(m, 'n2'), m.clock_offsets), ([(T+600, (90, 0, 0))], {'n2': PDT}))
            self.assertEqual(m.observe(T+610)['tokens'], 90)

    def test_parses_tp0_lines_only_and_counts_unreadable(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'engine.log'
            log.write_text(stats_line(self.T, 100) + stats_line(self.T, 999, rank='TP1')
                           + f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.T-PDT))} TP0] other line\n'
                           + stats_line(self.T+60, 300, ms=False) + stats_line(self.T+120, 400).replace(
                               'evicted_c1_host_only=400', 'evicted_c1_host_only=x')
                           + stats_line(self.T+180, 700, dup=4, active=9) + stats_line(self.T+240, 900)[:40])
            now = [self.T+200]
            m = HostEvictionMonitor({'n0': 'prefill'}, clock=lambda: now[0])
            m.read({'n0': [log]})
            self.assertEqual(m.unreadable_lines, 1)  # the line without a numeric evicted_c1_host_only
            got = rows(m, 'n0')
            self.assertEqual([round(t-self.T) for t, _ in got], [0, 60, 180])
            self.assertEqual([v for _, v in got], [(100, 0, 0), (300, 0, 0), (700, 4, 9)])
            obs = m.observe(self.T+200)
            self.assertEqual((obs['tokens'], obs['c1_dup_tokens'], obs['active_tokens'], obs['tokens_per_p']),
                             (700, 4, 9, 700.))
            with log.open('a') as f:  # the partial last line is read once complete
                f.write(stats_line(self.T+240, 900)[40:])
            now[0] = self.T+250
            m.read({'n0': [log]})
            self.assertEqual(m.observe(self.T+250)['tokens'], 900)

    def test_truncated_or_replaced_log_starts_a_new_stream(self):
        T = self.T
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'engine.log'
            log.write_text(''.join(stats_line(T+60*i, 100*i) for i in range(4)))
            now = [T+200]
            m = HostEvictionMonitor({'n0': 'prefill'}, clock=lambda: now[0])
            m.read({'n0': [log]})
            log.write_text(stats_line(T+240, 7))  # truncated: shorter than the read offset; a new process from 0
            now[0] = T+250
            m.read({'n0': [log]})
            fresh = Path(temp)/'new.log'
            fresh.write_text(stats_line(T+240, 7) + stats_line(T+300, 9) + stats_line(T+360, 15))
            fresh.replace(log)  # replaced: a new inode, longer than the old offset
            now[0] = T+370
            m.read({'n0': [log]})
            self.assertEqual(m.restarted_streams, 2)
            self.assertEqual(len({key for key in m.samples}), 3)
            # (T, T+370]: first stream 100 + 100 + 100, second 7, third 7 + 2 + 6
            self.assertEqual(m.observe(T+370)['tokens'], 300+7+15)

    def test_a_node_that_cannot_be_read_does_not_stop_the_others(self):
        T = self.T
        with tempfile.TemporaryDirectory() as temp:
            good = Path(temp)/'good.log'
            good.write_text(stats_line(T, 0) + stats_line(T+60, 50))
            m = HostEvictionMonitor({'n0': 'prefill', 'n1': 'prefill', 'n2': 'decode'}, clock=lambda: T+70)
            m.read({'n0': [Path(temp)/'missing.log'], 'n1': [good]}, failed={'n2': 'KeyError: output'})
            self.assertEqual(sorted(m.read_failures), ['n0', 'n2'])
            obs = m.observe(T+70)
            self.assertEqual((obs['tokens'], obs['signal'], obs['complete'], obs['nodes_without_signal'],
                              obs['read_failed_nodes']), (50, True, False, ['n0'], ['n0', 'n2']))
            m.read({'n0': [good], 'n1': [good]})
            self.assertEqual(m.read_failures, {})

    def test_window_roles_streams_and_freshness(self):
        T = self.T
        with tempfile.TemporaryDirectory() as temp:
            gen0, gen1, other = Path(temp)/'g0.log', Path(temp)/'g1.log', Path(temp)/'n1.log'
            # n0, generation 0: cumulative 0, 10, 30, 30, 70, 20 (reset: counts 0) at T+0..T+300.
            gen0.write_text(''.join(stats_line(T+60*i, c) for i, c in enumerate([0, 10, 30, 30, 70, 20])))
            # n0, generation 1 (a new process): its first sample counts from 0.
            gen1.write_text(stats_line(T+330, 5) + stats_line(T+390, 11))
            # n1: P until T+200 (then D), evicting 100 per minute; its intervals while D do not count.
            other.write_text(''.join(stats_line(T+60*i, 100*i) for i in range(8)))
            m = HostEvictionMonitor({'n0': 'prefill', 'n1': 'prefill', 'n2': 'decode'}, window_s=300.,
                                    clock=lambda: T+430)
            m.set_roles({'n0': 'prefill', 'n1': 'decode', 'n2': 'decode'}, T+200)
            m.read({'n0': [gen0, gen1], 'n1': [other], 'n2': []})
            obs = m.observe(T+400)  # window (T+100, T+400]
            # n0: (T+120: 20) + (T+180: 0) + (T+240: 40) + (T+300: reset 0) + (T+330: 5) + (T+390: 6) = 71
            # n1: interval ends T+120 and T+180 while P (100 each); T+240.. while D are not counted.
            self.assertEqual(obs['nodes'], {'n0': 71, 'n1': 200})
            self.assertEqual(obs['tokens'], 271)
            self.assertEqual((obs['prefill_nodes'], obs['signal'], obs['complete']), (['n0'], True, True))
            self.assertAlmostEqual(obs['newest_age_s']['n0'], 10.)
            # A node turning P counts from its confirmed switch: n2 has no samples, so the P set lacks a signal.
            m.set_roles({'n0': 'prefill', 'n1': 'decode', 'n2': 'prefill'}, T+400)
            obs = m.observe(T+410)
            self.assertEqual((obs['nodes_without_signal'], obs['signal'], obs['complete']), (['n2'], True, False))
            # Samples after `now` are not visible; a P node without a sample in the last 300 s has no signal.
            early = m.observe(T+90)  # n0: 10 (T+60); n1, still P: 100 (T+60); 4 + 2 + 6 samples later than T+90
            self.assertEqual((early['tokens'], early['samples_after_now']), (110, 12))
            stale = m.observe(T+400+300+1)
            self.assertEqual((stale['signal'], stale['nodes_without_signal']), (False, ['n0', 'n2']))

    def test_role_change_inside_the_window_counts_the_p_intervals(self):
        # n0 is P until T+300 and D afterwards; n1 is D until T+300 and P afterwards; both evict 60 per minute.
        T = self.T
        with tempfile.TemporaryDirectory() as temp:
            logs = {n: Path(temp)/f'{n}.log' for n in ('n0', 'n1')}
            for path in logs.values():
                path.write_text(''.join(stats_line(T+60*i, 60*i) for i in range(12)))
            m = HostEvictionMonitor({'n0': 'prefill', 'n1': 'decode'}, clock=lambda: T+670)
            m.set_roles({'n0': 'decode', 'n1': 'prefill'}, T+300)
            m.read({n: [p] for n, p in logs.items()})
            obs = m.observe(T+600)  # window (T+0, T+600]: ends T+60..T+600
            self.assertEqual(obs['nodes'], {'n0': 240, 'n1': 360})  # n0: ends T+60..T+240 (4); n1: T+300..T+600 (6)
            self.assertEqual(obs['prefill_nodes'], ['n1'])

    def test_no_lines_means_no_signal_and_a_disabled_monitor_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'engine.log'
            log.write_text(REAL_TP0_OTHER + '\n')
            m = HostEvictionMonitor({'n0': 'prefill', 'n1': 'decode'})
            m.read({'n0': [log], 'n1': []})
            obs = m.observe(self.T)
            self.assertEqual((obs['tokens'], obs['signal'], obs['complete'], obs['nodes_without_signal']),
                             (0, False, False, ['n0']))
            m.disable('role history: out of order')
            with self.assertRaisesRegex(ValueError, 'Eviction monitor disabled'):
                m.observe(self.T)
        with self.assertRaises(ValueError):
            HostEvictionMonitor({}, window_s=0.)


class ReleaseAndShadowTests(unittest.TestCase):
    def test_queue_release_uses_the_all_leg_p95(self):
        for p95, blocked in ((9.99, False), (10., True), (20.6, True), (None, True), (float('nan'), True)):
            result = scripted(REMOVE_2P, guards=PlannerGuards(**EHOLD_Q), queue=queue(p95), evict=pressure(5.e7))
            e = result['guards']['evict_pressure_hold']
            self.assertEqual((e['blocked'], e['release']['mode'], e['release']['released']), (blocked, 'queue_p95', not blocked))
            self.assertEqual(p_count(result['selected']), 3 if blocked else 2)
        # An ungated observation: the release reads p95_all, not the ungated p95_s (3.5 s at the first run's +6,316.5 s).
        ungated = dict(queue(3.5), signal='ungated', p95_all=20.6)
        result = scripted(REMOVE_2P, guards=PlannerGuards(**EHOLD_Q, queue_signal='ungated'), queue=ungated,
                          evict=pressure(5.e7))
        self.assertEqual((result['guards']['evict_pressure_hold']['blocked'],
                          result['guards']['evict_pressure_hold']['release']['value']), (True, 20.6))
        # No pressure: the release is not evaluated.
        low = scripted(REMOVE_2P, guards=PlannerGuards(**EHOLD_Q), queue=queue(25.), evict=pressure(0))
        self.assertIsNone(low['guards']['evict_pressure_hold']['release'])

    def test_gain_and_first_decision_releases(self):
        big = {2: [g+.2 for g in STEADY_3]}  # +23 %
        for gains, blocked in ((REMOVE_2P, True), (big, False)):
            result = scripted(gains, guards=PlannerGuards(evict_pressure_hold=True, evict_pressure_release='gain'),
                              evict=pressure(5.e7))
            self.assertEqual(result['guards']['evict_pressure_hold']['blocked'], blocked)
        guards = PlannerGuards(evict_pressure_hold=True, evict_pressure_release='first_decision')
        for first, blocked in ((True, False), (False, True)):
            result = scripted(REMOVE_2P, guards=guards, evict=pressure(5.e7), first_decision=first)
            e = result['guards']['evict_pressure_hold']
            self.assertEqual((e['blocked'], e['release']['value']), (blocked, first))

    def test_shadow_records_the_would_block_and_changes_nothing(self):
        cases = [REMOVE_2P, {**REMOVE_2P, 4: [g+.01 for g in STEADY_3]}, {4: STEADY_2}, {}]
        for gains in cases:
            for q in (queue(25.), queue(15.), queue(5.)):
                for evict in (pressure(5.e7), pressure(0), None):
                    off = scripted(gains, guards=PlannerGuards(**ALL), queue=q)
                    on = scripted(gains, guards=PlannerGuards(**EHOLD_Q), queue=q, evict=evict)
                    shadow = scripted(gains, guards=PlannerGuards(**ALL, evict_pressure_hold='shadow'), queue=q,
                                      evict=evict)
                    e, s = on['guards']['evict_pressure_hold'], shadow['guards']['evict_pressure_hold']
                    self.assertEqual((shadow['action'], shadow['selected']), (off['action'], off['selected']))
                    self.assertEqual({k: v for k, v in shadow['guards'].items() if k not in ('options', 'evict_pressure_hold')},
                                     {k: v for k, v in off['guards'].items() if k != 'options'})
                    self.assertEqual((s['mode'], s['blocked'], s['would_block'], s['selected'], s['release']),
                                     ('shadow', False, e['blocked'], e['selected'], e['release']))
                    self.assertEqual(e['would_block'], e['blocked'])
                    self.assertNotIn('evict_pressure_hold', shadow['guards']['fired'])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.urls = {'node0': 'http://a', 'node1': 'http://b', 'node2': 'http://c'}
        self.initial = dict(assignment=dict(epoch='e', revision=0,
                            roles={'http://a': 'prefill', 'http://b': 'prefill', 'http://c': 'decode'}),
                            completed_workload=dict(requests=0, input_tokens=0, uncached_tokens=0,
                                causal_attention_pairs=0, output_tokens=0, verified_output_tokens=0, verify_steps=0),
                            decode_population=dict(fields=['p_completed', 'p_pending', 'unknown'],
                                                   workers={'http://c': [3, 5, 0]}))
        p = dict(coefficients=[.01, .0001, 1e-9, .00001], tokens_per_step=4096., restored_per_cached_token=.5)
        d = dict(coefficients=[.02, .001, .01], draft_tokens=6, batch_support=list(range(1, 129)))
        self.workers = {n: dict(prefill=dict(origin='measured', models=[p]), decode=dict(origin='measured', models=[d]),
                                max_running_requests=128, kv_capacity_tokens=2000000) for n in self.urls}

    def samples(self):
        end = copy.deepcopy(self.initial)
        end['completed_workload'] = dict(requests=100, input_tokens=1200, uncached_tokens=500,
                                         causal_attention_pairs=5000, output_tokens=400,
                                         verified_output_tokens=400, verify_steps=100)
        a, b = self.initial['completed_workload'], end['completed_workload']
        out = []
        for i in range(201):
            s = copy.deepcopy(end)
            s['completed_workload'] = {k: a[k]+(b[k]-a[k])*i*3//600 for k in b}
            out.append(dict(at=100.+i*3, snapshot=s))
        return out

    def controller(self, guards):
        return Controller(dict(workers=self.workers), self.urls, self.initial, experts=256, top_k=8, guards=guards)

    def test_observation_is_passed_only_with_the_hold_on(self):
        seen = []

        def stub(workers, roles, work, **kwargs):
            seen.append(kwargs.get('evict_pressure', 'not passed'))
            current = dict(changed=[], roles=dict(roles))
            return dict(action='hold', selected=current, candidates=[current], guards={})
        for guards in (PlannerGuards(**EHOLD), PlannerGuards(**ALL), PlannerGuards(**QHOLD, **EHOLD)):
            with patch('evolving_infra.planner.calibration.controller.decide', side_effect=stub):
                self.controller(guards).propose(self.samples(), queue=queue(), evict_pressure=pressure(7.))
        self.assertEqual(seen, [pressure(7.), 'not passed', pressure(7.)])
        plain = self.controller(None).propose(self.samples())
        self.assertEqual(self.controller(None).propose(self.samples(), evict_pressure=pressure(9.e9)), plain)

    def test_first_decision_flag_is_passed_for_its_release_only(self):
        seen = []

        def stub(workers, roles, work, **kwargs):
            seen.append(kwargs.get('first_decision', 'not passed'))
            current = dict(changed=[], roles=dict(roles))
            return dict(action='hold', selected=current, candidates=[current], guards={})
        c = self.controller(PlannerGuards(evict_pressure_hold=True, evict_pressure_release='first_decision'))
        with patch('evolving_infra.planner.calibration.controller.decide', side_effect=stub):
            decision = c.propose(self.samples(), evict_pressure=pressure(0))
            c.apply(self.initial, decision, None, lambda row: None)  # a hold: no PUT
            c.propose(self.samples(), evict_pressure=pressure(0))
            self.controller(PlannerGuards(**EHOLD)).propose(self.samples(), evict_pressure=pressure(0))
        self.assertEqual(seen, [True, False, 'not passed'])

    def test_real_decide_records_the_hold(self):
        # 2P1D with the real decide(): the D side is the bottleneck and the standard rule removes a P (1P2D).
        for tokens, removes in ((0, True), (2.e6, False), (None, True)):
            c = self.controller(PlannerGuards(**EHOLD))
            decision = c.propose(self.samples(), evict_pressure=None if tokens is None else pressure(tokens))
            g = decision['guards']
            self.assertEqual(g['selections']['standard']['prefill_nodes'], 1)
            self.assertEqual(g['evict_pressure_hold']['blocked'], not removes)
            self.assertEqual((decision['action'], p_count(decision['selected'])),
                             ('switch', 1) if removes else ('hold', 2))
            json.dumps(decision)


def recorded():
    """(run options, decision row, rebuilt standard decision) of the qhold fixture and the per-job E-hold samples."""
    qhold = json.loads((FIXTURES/'qhold-recorded-decisions.json').read_text())
    evp = {r['job']: r for r in json.loads((FIXTURES/'evict-pressure-recorded.json').read_text())['runs']}
    rows = []
    for run in qhold['runs']:
        for row in run['decisions']:
            roles = row['roles']
            candidates = [dict(o, roles={n: FLIP[r] if n in o['changed'] else r for n, r in roles.items()})
                          for o in row['candidates']]
            standard = row['guards']['selections']['standard']
            selected = next(o for o in candidates if o['changed'] == standard['changed'])
            rows.append((run, row, dict(action=standard['action'], selected=selected, candidates=candidates)))
    return qhold, evp, rows


def recorded_monitor(evp_run, decisions, window_s=600.):
    """The monitor over the recorded samples; each recorded switch holds from just after its decision (the PUT is
    confirmed after the decision, so the decision itself sees the roles before it)."""
    m = HostEvictionMonitor(evp_run['initial_roles'], window_s=window_s)
    m.add_samples([(node, 'engine.log', t, tuple(v)) for node, rows in evp_run['samples'].items() for t, *v in rows])
    for row in decisions:
        if row['event'] == 'role_update_intent':
            m.set_roles({n: FLIP[r] if n in row['selected_changed'] else r for n, r in row['roles'].items()},
                        row['t']+1.e-3)
    return m


class RecordedDecisionTests(unittest.TestCase):
    """E-hold (1 M tokens, 600 s) on the 26 recorded decisions of the three guard runs (the first: 18 trays) [S].

    Decision level only: a changed decision would change every later input of that run.
    """

    def replay(self, tokens=1.e6, window_s=600., qhold=False, release='queue_p95'):
        qfix, evp, rows = recorded()
        out = []
        monitors = {run['job']: recorded_monitor(evp[run['job']], run['decisions'], window_s) for run in qfix['runs']}
        for run, row, decision in rows:
            options = dict(run['options'], **({'queue_removal_floor_s': 10.} if qhold else {}))
            guards = PlannerGuards(**options, evict_pressure_hold=True, evict_pressure_tokens=tokens,
                                   evict_pressure_window_s=window_s, evict_pressure_release=release)
            final = dict(decision)
            first = row is run['decisions'][0]
            hold, evict, add = _removal_and_add_guards(final, decision['candidates'][0]['roles'], guards,
                                                       row['guards']['queue_add_prefill']['observation'],
                                                       row['queue_added'], monitors[run['job']].observe(row['t']), first)
            out.append((run, row, decision, final, hold, evict, add))
        return out

    def test_only_the_harmful_removal_changes(self):
        changed, removals = [], []
        for run, row, decision, final, hold, evict, add in self.replay():
            current = _prefill_count(decision['candidates'][0]['roles'])
            if decision['action'] == 'switch' and _prefill_count(decision['selected']['roles']) < current:
                removals.append((run['job'], row['t'], evict['blocked']))
            self.assertTrue(evict['observation']['complete'], (run['job'], row['t']))
            self.assertEqual(add['fired'], row['guards']['queue_add_prefill']['fired'])  # Q' as recorded
            if sorted(final['selected']['changed']) != sorted(row['selected_changed']) or \
                    (final['action'] == 'switch') != (row['action'] == 'switch'):
                changed.append((run['job'], row['t'], current, _prefill_count(final['selected']['roles'])))
        self.assertEqual(changed, [(281395, 6316.5, 9, 9)])
        self.assertEqual(removals, [(281395, 1824., False), (281395, 6316.5, True), (281124, 1817.9, False),
                                    (275272, 1818.1, False)])

    def test_harmless_removals_see_no_class1_evictions_and_the_harmful_one_does(self):
        rows = {(run['job'], row['t']): evict for run, row, _, _, _, evict, _ in self.replay()}
        for key in ((281395, 1824.), (281124, 1817.9), (275272, 1818.1)):
            self.assertEqual(rows[key]['tokens'], 0, key)
            self.assertEqual(rows[key]['reason'], 'P class-1 host evictions not above the threshold')
        harmful = rows[(281395, 6316.5)]
        self.assertGreater(harmful['tokens'], 15.e6)  # 15.39 M over (5716.5, 6316.5]; Addendum 4d bins: 16.41 M
        self.assertEqual(len(harmful['observation']['prefill_nodes']), 9)  # the 9 P before the switch
        self.assertEqual(len(harmful['observation']['nodes']), 9)  # every P node evicted class-1 KV in the window
        self.assertEqual(harmful['reason'], 'P removal blocked, no remaining candidate: hold')

    def test_robust_to_threshold_window_and_together_with_qhold(self):
        for tokens, window_s in ((0., 600.), (5.e6, 600.), (1.e6, 300.), (1.e6, 900.)):
            blocked = [(run['job'], row['t']) for run, row, _, _, _, evict, _ in self.replay(tokens, window_s)
                       if evict['blocked']]
            self.assertEqual(blocked, [(281395, 6316.5)], (tokens, window_s))
        self.assertFalse(any(e['blocked'] for *_, e, _ in self.replay(16.e6)))  # above the recorded 15.39 M
        for release in ('queue_p95', 'gain', 'first_decision', 'none'):
            blocked = [(run['job'], row['t']) for run, row, _, _, _, evict, _ in self.replay(release=release)
                       if evict['blocked']]
            self.assertEqual(blocked, [(281395, 6316.5)], release)
        both = [(run['job'], row['t'], hold['blocked'], evict['blocked'])
                for run, row, _, _, hold, evict, _ in self.replay(qhold=True) if hold['blocked'] or evict['blocked']]
        self.assertEqual(both, [(281395, 6316.5, True, True)])

    def test_273659_first_removal_sees_no_class1_evictions(self):
        evp = {r['job']: r for r in json.loads((FIXTURES/'evict-pressure-recorded.json').read_text())['runs']}
        obs = recorded_monitor(evp[273659], []).observe(1818.)
        self.assertEqual((obs['tokens'], obs['complete'], len(obs['prefill_nodes'])), (0, True, 6))


if __name__ == '__main__':
    unittest.main()
