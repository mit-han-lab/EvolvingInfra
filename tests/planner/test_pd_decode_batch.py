"""Opt-in measured decode batch cap (decode_batch_source 'measured') and its trace monitor."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evolving_infra.planner.calibration.controller import Controller
from evolving_infra.planner.calibration.dynamic import _decode_kv_fraction, decide, decode_capacity
from evolving_infra.planner.calibration.guards import DecodeBatchMonitor, PlannerGuards

P = dict(coefficients=[.01, .0001, 1e-9, .00001], tokens_per_step=4096., restored_per_cached_token=.5)
D = dict(coefficients=[.02, .001, .01], draft_tokens=6, batch_support=list(range(1, 129)))
WORK = dict(uncached=8192., cached=40000., pairs=8192*44096., output=800., context=50000., acceptance=3.)
MINUTES = [dict(work=WORK, decode_population=30), dict(work=dict(WORK, output=900.), decode_population=30),
           dict(work=dict(WORK, output=850.), decode_population=28)]
ALL = dict(restore_floor=True, min_relative_gain=.03, queue_p95_s=20.)
MEASURED = PlannerGuards(decode_batch_source='measured')
P2D2 = {'0': 'prefill', '1': 'prefill', '2': 'decode', '3': 'decode'}
P3D1 = {'0': 'prefill', '1': 'prefill', '2': 'prefill', '3': 'decode'}


def workers(kv=2000000, max_running=128):
    return {str(i): dict(prefill=dict(origin='measured', models=[P]), decode=dict(origin='measured', models=[D]),
                         max_running_requests=max_running, kv_capacity_tokens=kv) for i in range(4)}


def running(batch, roles=P2D2):
    return {n: dict(batch=batch, passes=100, dur_s=50.) for n, r in roles.items() if r == 'decode'}


def measured(batch, p_completed=30., previous=None, roles=P2D2):
    return dict(running=running(batch, roles), p_completed=p_completed, population=30., kv_fraction=previous)


def run(roles=P2D2, guards=MEASURED, decode_batch=None, **kwargs):
    return decide(workers(), roles, WORK, minutes=MINUTES, experts=256, top_k=8, decode_population=30,
                  guards=guards, decode_batch=decode_batch, **kwargs)


def p_count(option):
    return sum(r == 'prefill' for r in option['roles'].values())


class OptionTests(unittest.TestCase):
    def test_decode_batch_source_option(self):
        guards = PlannerGuards.from_configuration(dict(controller_guard_decode_batch_source='measured'))
        self.assertEqual(guards.decode_batch_source, 'measured')
        self.assertTrue(guards.enabled)  # independent of the queue guards
        self.assertEqual(guards.as_dict()['decode_batch_source'], 'measured')
        self.assertEqual(PlannerGuards().decode_batch_source, 'formula')
        self.assertFalse(PlannerGuards(decode_batch_source='formula').enabled)
        self.assertNotIn('decode_batch_source', PlannerGuards(**ALL, decode_batch_source='formula').as_dict())
        self.assertEqual(PlannerGuards(**ALL, decode_batch_source='formula').as_dict(), PlannerGuards(**ALL).as_dict())
        for bad in ('Measured', 'kv', '', None, 0, True):
            with self.assertRaises(ValueError, msg=bad):
                PlannerGuards(decode_batch_source=bad)


class DecodeCapacityTests(unittest.TestCase):
    def test_kv_fraction_gives_a_fluid_cap_and_none_keeps_the_formula(self):
        worker = workers()['0']
        work = dict(WORK, context=60000.)
        formula = decode_capacity(D, worker, work, 256, 8)
        self.assertEqual(formula['batch'], 31)  # floor(0.95 * 2e6 / 6e4)
        self.assertEqual(decode_capacity(D, worker, work, 256, 8, None, None), formula)
        self.assertAlmostEqual(decode_capacity(D, worker, work, 256, 8, None, .2)['batch'], 2e6*.2/6e4)  # 6.67
        self.assertEqual(decode_capacity(D, worker, work, 256, 8, None, .95)['batch'], 2e6*.95/6e4)  # not floored
        self.assertEqual(decode_capacity(D, dict(worker, max_running_requests=4), work, 256, 8, None, .5)['batch'], 4.)
        self.assertEqual(decode_capacity(D, worker, work, 256, 8, None, 1e-6)['batch'], 1.)  # at least one request
        self.assertEqual(decode_capacity(D, worker, work, 256, 8, 3., .5)['batch'], 3.)  # population bound stays
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            decode_capacity(D, worker, dict(WORK, context=3e6), 256, 8, None, .5)


class KvFractionTests(unittest.TestCase):
    def fraction(self, batch, p_completed=30., previous=None, roles=P2D2, nodes=None, **kw):
        inputs = measured(batch, p_completed, previous, roles)
        if nodes is not None:
            inputs['running'] = nodes
        return _decode_kv_fraction(kw.get('workers', workers()), roles, WORK, inputs)

    def test_memory_bound_window_learns_the_median_fraction(self):
        record = self.fraction(8.)  # p_completed 30 vs running 16: gap 14 >= max(2, 6)
        self.assertEqual((record['memory_bound'], record['learned'], record['reason']), (True, True, 'memory-bound'))
        self.assertAlmostEqual(record['kv_fraction'], 8*50000/2e6)
        self.assertEqual(record['kv_fraction'], record['kv_fraction_measured'])
        self.assertEqual((record['margin'], record['running_total'], record['running_per_d']['median']), (6., 16., 8.))
        self.assertEqual(record['cap_formula'], dict(median=38., nodes={'2': 38, '3': 38}))
        self.assertEqual(record['cap_used']['nodes'], {'2': 8., '3': 8.})
        nodes = {'2': dict(batch=6., passes=1, dur_s=1.), '3': dict(batch=20., passes=1, dur_s=1.)}
        larger = dict(workers(), **{'3': dict(workers()['3'], kv_capacity_tokens=4e6)})
        mixed = self.fraction(None, p_completed=50., nodes=nodes, workers=larger)
        self.assertAlmostEqual(mixed['kv_fraction'], (6*5e4/2e6+20*5e4/4e6)/2)  # median of the per-node fractions (.2)
        self.assertEqual(self.fraction(128., p_completed=400.)['kv_fraction'], 1.)  # at most the whole pool

    def test_supply_limited_window_keeps_the_last_fraction(self):
        for previous, expected, applied in ((None, .95, False), (.4, .4, True)):
            record = self.fraction(14., previous=previous)  # gap 2 < 6
            self.assertEqual((record['memory_bound'], record['learned'], record['kv_fraction']), (False, False, expected))
            self.assertEqual((record['previous_kv_fraction'], record['applied']), (previous, applied))
        # Before the first learned f the record's cap is the floored formula, not the fluid 0.95 cap.
        early = self.fraction(14., previous=None)
        self.assertEqual(early['cap_used'], dict(median=38, nodes={'2': 38, '3': 38}))
        # Small populations: the margin is at least 2 requests (here 0.2 * 5 = 1).
        self.assertTrue(self.fraction(1.5, p_completed=5.)['memory_bound'])  # gap 2.0 >= 2
        self.assertFalse(self.fraction(1.6, p_completed=5.)['memory_bound'])  # gap 1.8 < 2
        self.assertFalse(self.fraction(.9, p_completed=3.5)['memory_bound'])  # gap 1.7 < 2

    def test_incomplete_inputs_never_learn(self):
        cases = (('no running measurement for every D node', dict(nodes={'2': dict(batch=1., passes=1, dur_s=1.)})),
                 ('no running measurement for every D node', dict(nodes={})),
                 ('no p_completed population', dict(p_completed=None)),
                 ('no running decode request', dict(batch=0.)))
        for reason, kw in cases:
            kw.setdefault('batch', 8.)
            record = self.fraction(previous=.5, **kw)
            self.assertEqual((record['memory_bound'], record['learned'], record['kv_fraction'], record['reason']),
                             (None, False, .5, reason), kw)
        record = _decode_kv_fraction(workers(), P2D2, WORK, None)
        self.assertEqual((record['kv_fraction'], record['learned']), (.95, False))
        idle = self.fraction(None, p_completed=40., nodes={'2': dict(batch=0., passes=0, dur_s=0.),
                                                           '3': dict(batch=0., passes=0, dur_s=0.)})
        self.assertEqual(idle['reason'], 'no running decode request')
        roles = {'0': 'prefill', '1': 'decode', '2': 'decode', '3': 'decode'}
        two_idle = self.fraction(None, roles=roles, nodes={'1': dict(batch=0., passes=0, dur_s=0.),
                                                           '2': dict(batch=0., passes=0, dur_s=0.),
                                                           '3': dict(batch=12., passes=9, dur_s=1.)})
        self.assertEqual((two_idle['memory_bound'], two_idle['learned'], two_idle['reason'], two_idle['idle_d_nodes']),
                         (None, False, 'no decode pass in the window on D node 1, 2', ['1', '2']))

    def test_a_partly_idle_window_does_not_learn(self):
        # One of three D nodes without a decode pass (idle, stalled or no trace). Counting it as 0 would
        # make this supply-limited window (gap 10 against margin 6) memory-bound and learn f = median(0, .25, .25).
        roles = {'0': 'prefill', '1': 'decode', '2': 'decode', '3': 'decode'}
        nodes = {'1': dict(batch=0., passes=0, dur_s=0.), '2': dict(batch=10., passes=50, dur_s=30.),
                 '3': dict(batch=10., passes=50, dur_s=30.)}
        for previous, kept, applied in ((.7, .7, True), (None, .95, False)):
            record = self.fraction(None, p_completed=30., previous=previous, roles=roles, nodes=nodes)
            self.assertEqual((record['memory_bound'], record['learned'], record['kv_fraction'], record['applied']),
                             (None, False, kept, applied))
            self.assertEqual(record['reason'], 'no decode pass in the window on D node 1')
            self.assertIsNone(record['running_total'])
        busy = dict(nodes, **{'1': dict(batch=10., passes=50, dur_s=30.)})  # all three measured: gap 0, not memory-bound
        self.assertEqual(self.fraction(None, p_completed=30., previous=.7, roles=roles, nodes=busy)['reason'],
                         'not memory-bound')


class DecideTests(unittest.TestCase):
    def test_measured_cap_changes_the_choice_and_is_recorded(self):
        formula = run(guards=None)
        self.assertEqual((formula['action'], p_count(formula['selected'])), ('switch', 3))
        result = run(decode_batch=measured(8.))
        self.assertEqual(result['action'], 'hold')
        g = result['guards']
        b = g['decode_batch']
        self.assertEqual((b['source'], b['learned'], b['kv_fraction']), ('measured', True, .2))
        self.assertEqual(g['selections']['decode_formula']['prefill_nodes'], 3)
        self.assertEqual(g['selections']['without_guards']['prefill_nodes'], 3)  # no guards: the formula
        self.assertEqual(g['fired'], ['decode_batch'])
        self.assertTrue(g['changed_decision'])
        self.assertLess(b['estimate']['decode_rr_rps'], b['formula_estimate']['decode_rr_rps'])
        self.assertEqual(b['estimate']['prefill_rps'], b['formula_estimate']['prefill_rps'])
        current = result['candidates'][0]['estimate']['workers']['2']['operating_points'][0]
        self.assertEqual(current['batch'], 8.)  # min(cap 8, population 15 per D)
        json.dumps(result)
        # Every candidate map uses the learned fraction: from 3P1D the planner moves to 2P2D.
        three = run(roles=P3D1, decode_batch=measured(8., roles=P3D1))
        self.assertEqual((three['action'], p_count(three['selected'])), ('switch', 2))
        self.assertEqual(run(roles=P3D1, guards=None)['action'], 'hold')

    def test_not_memory_bound_before_learning_keeps_the_formula(self):
        result = run(decode_batch=measured(14.5))
        b = result['guards']['decode_batch']
        self.assertEqual((b['memory_bound'], b['kv_fraction'], b['applied'], b['cap_used']['median']), (False, .95, False, 38))
        self.assertEqual((result['action'], p_count(result['selected'])), ('switch', 3))
        self.assertEqual(result['guards']['fired'], [])
        self.assertEqual(json.dumps(result['candidates']), json.dumps(run(guards=None)['candidates']))
        # Where the fluid 0.95 cap exceeds the floored formula (context 60k: 31.67 vs 31) the formula is used.
        work = dict(WORK, context=60000.)
        early = decide(workers(), P2D2, work, minutes=MINUTES, experts=256, top_k=8, decode_population=100,
                       guards=MEASURED, decode_batch=measured(14.5))
        point = early['candidates'][0]['estimate']['workers']['2']['operating_points'][0]
        self.assertEqual(point['batch'], 31)
        self.assertEqual(early['guards']['selections']['standard'], early['guards']['selections']['decode_formula'])
        # A previously learned fraction is kept in a supply-limited window.
        kept = run(decode_batch=measured(14.5, previous=.2))
        self.assertEqual((kept['action'], kept['guards']['decode_batch']['kv_fraction']), ('hold', .2))
        # Without any measurement the decision still runs, on the last fraction.
        none = run(decode_batch=None)
        self.assertEqual(none['guards']['decode_batch']['reason'], 'no running measurement for every D node')

    def test_off_ignores_decode_batch_inputs(self):
        for guards in (None, PlannerGuards(), PlannerGuards(**ALL), PlannerGuards(**ALL, decode_batch_source='formula')):
            base = decide(workers(), P2D2, WORK, minutes=MINUTES, experts=256, top_k=8, decode_population=30,
                          guards=guards, queue=dict(p95_s=5., since_switch_s=None))
            other = decide(workers(), P2D2, WORK, minutes=MINUTES, experts=256, top_k=8, decode_population=30,
                           guards=guards, queue=dict(p95_s=5., since_switch_s=None), decode_batch=measured(8.))
            self.assertEqual(json.dumps(other), json.dumps(base))
            self.assertNotIn('decode_batch', base.get('guards', {}))

    def test_measured_cap_with_the_queue_guards(self):
        result = run(guards=PlannerGuards(**ALL, decode_batch_source='measured'), decode_batch=measured(8.),
                     queue=dict(p95_s=5., n=10, window_s=300., at=1., since_switch_s=None))
        g = result['guards']
        self.assertEqual(list(g)[:5], ['options', 'restore_floor', 'min_relative_gain', 'selections', 'queue_add_prefill'])
        self.assertEqual(list(g['selections']), ['without_guards', 'restore_floor_only', 'min_relative_gain_only',
                                                 'standard', 'decode_formula'])
        self.assertIn('decode_batch', g)
        self.assertEqual(g['queue_add_prefill']['reason'], 'P_queue p95 not above threshold')


class ControllerDecodeBatchTests(unittest.TestCase):
    def setUp(self):
        self.urls = {f'node{i}': f'http://h{i}' for i in range(4)}
        self.roles = {'http://h0': 'prefill', 'http://h1': 'prefill', 'http://h2': 'decode', 'http://h3': 'decode'}
        self.initial = dict(assignment=dict(epoch='e', revision=0, roles=dict(self.roles)),
                            completed_workload=dict(requests=0, input_tokens=0, uncached_tokens=0,
                                causal_attention_pairs=0, output_tokens=0, verified_output_tokens=0, verify_steps=0),
                            decode_population=dict(fields=['p_completed', 'p_pending', 'unknown'],
                                                   workers={'http://h2': [10, 5, 0], 'http://h3': [10, 5, 0]}))

    def controller(self, guards):
        fit = dict(workers={node: w for node, w in zip(self.urls, workers().values())})
        c = Controller(fit, self.urls, self.initial, experts=256, top_k=8, guards=guards)
        c.clock = lambda: 1000.
        return c

    def samples(self):
        end = copy.deepcopy(self.initial)
        end['completed_workload'] = dict(requests=100, input_tokens=4_800_000, uncached_tokens=819_200,
                                         causal_attention_pairs=100*8192*44096, output_tokens=80_000,
                                         verified_output_tokens=80_000, verify_steps=26_667)
        a, b = self.initial['completed_workload'], end['completed_workload']
        out = []
        for i in range(201):
            s = copy.deepcopy(end)
            s['completed_workload'] = {k: a[k]+(b[k]-a[k])*i*3//600 for k in b}
            out.append(dict(at=100.+i*3, wall=1.e9+i*3, snapshot=s))
        return out

    def test_learned_fraction_is_kept_by_apply_and_passed_to_the_next_decision(self):
        c = self.controller(PlannerGuards(decode_batch_source='measured'))
        seen = []

        def stub(workers_, roles, work, **kwargs):
            seen.append(kwargs.get('decode_batch'))
            current = dict(changed=[], roles=dict(roles))
            record = dict(kv_fraction=.3, learned=len(seen) == 1)
            return dict(action='hold', selected=current, candidates=[current], guards=dict(decode_batch=record))
        run_one = dict(node2=dict(batch=7., passes=10, dur_s=5.), node3=dict(batch=9., passes=10, dur_s=5.))
        with patch('evolving_infra.planner.calibration.controller.decide', side_effect=stub):
            first = c.propose(self.samples(), decode_running=run_one)
            self.assertIsNone(c.decode_kv_fraction)  # propose alone (a probe hold) keeps nothing
            c.apply(self.initial, first, None, lambda row: None)
            self.assertEqual(c.decode_kv_fraction, .3)
            second = c.propose(self.samples())
            c.apply(self.initial, second, None, lambda row: None)
        self.assertEqual(c.decode_kv_fraction, .3)  # not learned in the second window: kept
        self.assertEqual(seen[0]['running'], run_one)
        self.assertAlmostEqual(seen[0]['p_completed'], 20.)
        self.assertAlmostEqual(seen[0]['population'], 30.)
        self.assertIsNone(seen[0]['kv_fraction'])
        self.assertEqual((seen[1]['running'], seen[1]['kv_fraction']), (None, .3))

    def test_real_decide_records_and_saturated_estimates_use_the_fraction(self):
        c = self.controller(PlannerGuards(decode_batch_source='measured'))
        run_one = dict(node2=dict(batch=4., passes=10, dur_s=5.), node3=dict(batch=4., passes=10, dur_s=5.))
        decision = c.propose(self.samples(), decode_running=run_one)
        b = decision['guards']['decode_batch']
        self.assertTrue(b['memory_bound'])  # p_completed 20 vs running 8
        self.assertTrue(b['applied'])
        self.assertAlmostEqual(b['kv_fraction'], 4*b['context']/2e6)
        point = decision['candidates'][0]['saturated_estimate']['workers']['node2']['operating_points'][0]
        self.assertAlmostEqual(point['batch'], b['cap_used']['nodes']['node2'])
        json.dumps(decision)
        # Not memory-bound before any learned f (p_completed 20 vs running 18): the saturated estimates keep the formula.
        early = self.controller(PlannerGuards(decode_batch_source='measured')).propose(
            self.samples(), decode_running=dict(node2=dict(batch=9., passes=10, dur_s=5.),
                                                node3=dict(batch=9., passes=10, dur_s=5.)))
        e = early['guards']['decode_batch']
        self.assertEqual((e['memory_bound'], e['applied']), (False, False))
        point = early['candidates'][0]['saturated_estimate']['workers']['node2']['operating_points'][0]
        self.assertEqual(point['batch'], e['cap_formula']['nodes']['node2'])

    def test_off_passes_no_decode_batch(self):
        seen = []

        def stub(workers_, roles, work, **kwargs):
            seen.append(sorted(kwargs))
            current = dict(changed=[], roles=dict(roles))
            out = dict(action='hold', selected=current, candidates=[current])
            if 'guards' in kwargs:
                out['guards'] = {}
            return out
        for guards in (None, PlannerGuards(**ALL)):
            with patch('evolving_infra.planner.calibration.controller.decide', side_effect=stub):
                self.controller(guards).propose(self.samples(), decode_running=dict(node2=dict(batch=1.)))
        self.assertTrue(all('decode_batch' not in kwargs for kwargs in seen))


def recorded_inputs():
    """The bd_replay inputs of the recorded decisions of the three guard runs (see the fixture's 'about')."""
    fixture = json.loads((Path(__file__).parent/'fixtures'/'bd-recorded-inputs.json').read_text())
    for run_ in fixture['runs']:
        nodes = {n: dict(v) for n, v in run_['kv'].items()}
        for row in run_['decisions']:
            roles = {n: 'decode' if n in row['d_nodes'] else 'prefill' for n in nodes}
            yield run_, nodes, roles, row


class RecordedInputTests(unittest.TestCase):
    """The memory-bound rule on the measured inputs of the recorded decisions [S, decision level]."""

    def classify(self):
        out, fraction = [], {}
        for run_, nodes, roles, row in recorded_inputs():
            pop = row['population']
            record = _decode_kv_fraction(nodes, roles, dict(context=row['context']),
                                         dict(running=row['running'], p_completed=pop['p_completed'],
                                              population=pop['potential'], kv_fraction=fraction.get(run_['job'])))
            if record['learned']:
                fraction[run_['job']] = record['kv_fraction']
            out.append((run_['job'], row['t'], record, row))
        return out

    def test_memory_bound_windows_and_learned_fractions(self):
        rows = self.classify()
        self.assertEqual(len(rows), 26)
        bound = [(job, t) for job, t, record, row in rows if record['memory_bound']]
        self.assertEqual(bound, [(281395, 1824.), (281395, 5709.1), (281395, 6316.5), (281124, 1817.9), (275272, 1818.1)])
        for job, t, record, row in rows:
            gap = (record['p_completed']-record['running_total'])/record['p_completed']
            # The bound windows have gap fractions >= 0.24, every other window <= 0.18: the 0.2 margin sits between.
            self.assertTrue(gap >= .24 if record['memory_bound'] else gap <= .18, (job, t, gap))
            self.assertTrue(.63 <= record['kv_fraction'] <= .72, (job, t, record['kv_fraction']))
            self.assertLess(record['cap_used']['median'], record['cap_formula']['median'])
        at_6317 = next(record for job, t, record, row in rows if (job, t) == (281395, 6316.5))
        self.assertAlmostEqual(at_6317['running_total']/9, 8.2, delta=.1)  # p_busy v3 (300-s bins): 8.21 per D
        self.assertEqual(at_6317['cap_formula']['median'], 11)

    def test_full_population_would_learn_in_the_prefill_bound_window(self):
        # The full population (p_completed + p_pending) also counts requests still at P. At 6,924 s (7P11D, P-bound,
        # ungated P_queue p95 117 s) it is 8x the running decode requests; a rule on it would learn f ~ 0.23 and cut the
        # D cap to about 2.3 requests. p_completed is about the running count there.
        rows = self.classify()
        record, row = next((record, row) for job, t, record, row in rows if (job, t) == (281395, 6924.1))
        potential, total = row['population']['potential'], record['running_total']
        self.assertGreaterEqual(potential-total, max(2., .1*potential))
        self.assertLess(record['p_completed']-total, 0)
        self.assertFalse(record['memory_bound'])
        self.assertGreater(row['queue_ungated']['p95_ungated'], 100.)
        would = sum(1 for job, t, rec, r in rows
                    if r['population']['potential']-rec['running_total'] >= max(2., .1*r['population']['potential']))
        self.assertGreaterEqual(would, 24)  # that rule calls almost every window memory-bound


class DecodeBatchMonitorTests(unittest.TestCase):
    def line(self, end, dur_ms, bs, mode='decode'):
        return json.dumps(dict(forward_iter=1, ts=end, dur_ms=dur_ms, mode=mode, bs=bs, num_waiting_reqs=0)) + '\n'

    def test_time_weighted_batch_with_overlapping_passes_and_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a.jsonl', Path(temp)/'b.jsonl'
            # node0: overlapping decode passes (0.2 s every 0.1 s) of batch 2 and 6 -> (2*.2+6*.2+6*.6)/(1.0) weighting.
            a.write_text(self.line(1001.1, 200., 2) + self.line(1001.2, 200., 6) + self.line(1002.5, 600., 6)
                         + self.line(1003., 5000., 9, mode='extend') + self.line(2000., 100., 7) + '{"partial": ')
            # node1 holds the P role until 1500, then D.
            b.write_text(self.line(1001.5, 100., 3) + self.line(1600.5, 100., 5))
            m = DecodeBatchMonitor({'node0': 'decode', 'node1': 'prefill'})
            m.set_roles({'node0': 'decode', 'node1': 'decode'}, at=1500.)
            m.read({'node0': [a], 'node1': [b]})
            out = m.running(1000., 1100.)
            self.assertAlmostEqual(out['node0']['batch'], (2*.2+6*.2+6*.6)/1.0)
            self.assertEqual((out['node0']['passes'], round(out['node0']['dur_s'], 6)), (3, 1.))
            self.assertNotIn('node1', out)  # P at the window end
            later = m.running(1550., 2000.)
            self.assertEqual((later['node0']['batch'], later['node1']['batch']), (7., 5.))
            self.assertEqual(m.unreadable_lines, 0)
            with a.open('a') as f:
                f.write('"x": 1}\n' + self.line(2001.5, 100., 4))
            m.read({'node0': [a], 'node1': [b]})
            self.assertEqual(m.unreadable_lines, 0)  # the completed partial line is not a decode pass
            self.assertEqual(m.running(2000., 2002.)['node0']['batch'], 4.)

    def test_only_decode_role_nodes_and_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp)/'a.jsonl'
            a.write_text(self.line(1001., 100., 2) + '{"mode": "decode", "ts": "x", "dur_ms": 1}\n'
                         + json.dumps(dict(mode='decode', ts=1002.)) + '\n')
            m = DecodeBatchMonitor({'node0': 'decode', 'node1': 'prefill'})
            m.read({'node0': [a]})
            self.assertEqual(sorted(m.running(1000., 1010.)), ['node0'])
            self.assertEqual(m.unreadable_lines, 1)
            for start, end in ((10., 10.), (float('nan'), 1.)):
                with self.assertRaisesRegex(ValueError, 'Invalid decode batch window'):
                    m.running(start, end)
            m.disable('role history: test')
            with self.assertRaisesRegex(ValueError, 'disabled'):
                m.running(1000., 1010.)

    def test_old_seconds_are_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp)/'a.jsonl'
            a.write_text(self.line(100., 100., 2) + self.line(5000., 100., 4))
            m = DecodeBatchMonitor({'node0': 'decode'}, keep_s=600.)
            m.read({'node0': [a]})
            self.assertEqual(m.running(4900., 5000.)['node0']['batch'], 4.)
            self.assertEqual(sorted(m.seconds['node0'][0]), [5000])

    def test_dp_ranks_are_summed_generations_pooled_and_other_pipeline_ranks_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            gen0, gen1 = Path(temp)/'generation0-prefill'/'scheduler', Path(temp)/'generation1-decode'/'scheduler'
            gen0.mkdir(parents=True)
            gen1.mkdir(parents=True)
            files = {gen0/'iter_trace.dp0_pp0.pid1.jsonl': self.line(1001., 100., 3),
                     gen0/'iter_trace.dp1_pp0.pid1.jsonl': self.line(1001., 100., 5) + self.line(1002., 300., 5),
                     gen0/'iter_trace.dp0_pp1.pid1.jsonl': self.line(1001., 100., 100),
                     gen1/'iter_trace.dp0_pp0.pid2.jsonl': self.line(1003., 300., 7)}
            for path, text in files.items():
                path.write_text(text)
            m = DecodeBatchMonitor({'node0': 'decode'})
            m.read({'node0': sorted(files)})
            out = m.running(1000., 1010.)['node0']
            # dp0: generations pooled, (3*.1+7*.3)/.4 = 6; dp1: 5; pp1 is not counted.
            self.assertAlmostEqual(out['batch'], 6.+5.)
            self.assertEqual(out['passes'], 4)


if __name__ == '__main__':
    unittest.main()
