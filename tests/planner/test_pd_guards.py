"""Opt-in planner guards: G1 restore floor, G3 minimum gain, Q' queue-aware add-P, Q-hold queue removal hold."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evolving_infra.planner.calibration.controller import Controller
from evolving_infra.planner.calibration.dynamic import _queue_add_prefill, _queue_guards, decide, prefill_seconds
from evolving_infra.planner.calibration.guards import (PlannerGuards, PrefillQueueMonitor, floor_restore_ratios,
                                   percentile)

ALL = dict(restore_floor=True, min_relative_gain=.03, queue_p95_s=20.)
QHOLD = dict(ALL, queue_removal_floor_s=10.)


def scripted(gains, current_p=3, nodes=9, guards=None, queue=None, onboarding=None, queue_added=False):
    """decide() with system rps 10 * (1 + gains[P count][minute]); the current P count is 10."""
    roles = {str(i): 'prefill' if i < current_p else 'decode' for i in range(nodes)}
    count = len(next(iter(gains.values()), [0.]*10))

    def estimate(workers, candidate, work, **kwargs):
        p = sum(r == 'prefill' for r in candidate.values())
        rps = 10. if p == current_p else 10.*(1+gains.get(p, [-.5]*count)[work['minute']])
        return dict(system_rps=rps, prefill_rps=rps, decode_rr_rps=rps)
    minutes = [dict(work=dict(minute=i), decode_population=None) for i in range(count)]
    with patch('evolving_infra.planner.calibration.dynamic.capacities', side_effect=estimate):
        return decide({n: {} for n in roles}, roles, dict(minute=0), minutes=minutes, experts=256,
                      top_k=8, onboarding=onboarding, guards=guards, queue=queue, queue_added=queue_added)


def p_count(option):
    return sum(r == 'prefill' for r in option['roles'].values())


STEADY_2 = [.02, .021, .019, .02, .021, .019, .02, .02, .021, .019]
STEADY_3 = [.03, .031, .029, .03, .031, .029, .03, .03, .031, .029]
NOISY_POS = [.2, -.15, .18, -.12, .1, .05, -.1, .12, -.05, .07]


class GuardOptionTests(unittest.TestCase):
    def test_defaults_enable_nothing(self):
        guards = PlannerGuards()
        self.assertFalse(guards.enabled)
        self.assertEqual(guards.as_dict(), dict(restore_floor=False, min_relative_gain=0., queue_p95_s=0.,
                         queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60.))
        self.assertFalse(PlannerGuards.from_configuration(dict(controller_period_s=600)).enabled)

    def test_configuration_keys(self):
        guards = PlannerGuards.from_configuration(dict(
            controller_guard_restore_floor=True, controller_guard_min_relative_gain=.03,
            controller_guard_queue_p95_s=20, controller_guard_queue_window_s=300,
            controller_guard_queue_min_since_switch_s=120, controller_guard_queue_check_s=60))
        self.assertEqual(guards.as_dict(), dict(restore_floor=True, min_relative_gain=.03, queue_p95_s=20.,
                         queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60.))
        with self.assertRaisesRegex(ValueError, 'Unknown planner guard options: controller_guard_typo'):
            PlannerGuards.from_configuration(dict(controller_guard_typo=1))

    def test_invalid_options_fail(self):
        for bad in (dict(restore_floor=1), dict(min_relative_gain=-.01), dict(min_relative_gain=1.),
                    dict(min_relative_gain=True), dict(queue_p95_s=float('nan')),
                    dict(restore_floor=True, queue_p95_s=20., queue_window_s=0.),
                    dict(restore_floor=True, queue_p95_s=20., queue_check_s=0.)):
            with self.assertRaises(ValueError, msg=bad):
                PlannerGuards(**bad)
        with self.assertRaisesRegex(ValueError, 'requires the restore floor'):
            PlannerGuards(queue_p95_s=20.)

    def test_queue_removal_floor_option(self):
        guards = PlannerGuards.from_configuration(dict(
            controller_guard_restore_floor=True, controller_guard_min_relative_gain=.03,
            controller_guard_queue_p95_s=20, controller_guard_queue_removal_floor_s=10))
        self.assertEqual(guards.queue_removal_floor_s, 10.)
        self.assertTrue(guards.enabled)
        self.assertEqual(guards.as_dict(), dict(restore_floor=True, min_relative_gain=.03, queue_p95_s=20.,
                         queue_window_s=300., queue_min_since_switch_s=120., queue_check_s=60.,
                         queue_removal_floor_s=10.))
        # Off (the default or 0): the option is not listed, so options records stay as before.
        self.assertEqual(PlannerGuards().queue_removal_floor_s, 0.)
        self.assertNotIn('queue_removal_floor_s', PlannerGuards(**ALL).as_dict())
        self.assertEqual(PlannerGuards(**ALL, queue_removal_floor_s=0).as_dict(), PlannerGuards(**ALL).as_dict())
        for bad in (-1., float('nan'), float('inf'), True, '10', None):
            with self.assertRaises(ValueError, msg=bad):
                PlannerGuards(**ALL, queue_removal_floor_s=bad)
        for without_queue_guard in (dict(queue_removal_floor_s=10.), dict(restore_floor=True, queue_removal_floor_s=10.),
                                    dict(restore_floor=True, min_relative_gain=.03, queue_removal_floor_s=10.)):
            with self.assertRaisesRegex(ValueError, 'removal hold requires the queue add-P guard'):
                PlannerGuards(**without_queue_guard)
        with self.assertRaisesRegex(ValueError, 'requires the restore floor'):
            PlannerGuards(queue_p95_s=20., queue_removal_floor_s=10.)
        with self.assertRaisesRegex(ValueError, 'removal hold requires the queue add-P guard'):
            PlannerGuards.from_configuration(dict(controller_guard_restore_floor=True,
                                                  controller_guard_queue_removal_floor_s=10))


    def test_queue_signal_option(self):
        guards = PlannerGuards.from_configuration(dict(controller_guard_restore_floor=True,
                                                       controller_guard_queue_p95_s=20,
                                                       controller_guard_queue_signal='ungated'))
        self.assertEqual(guards.queue_signal, 'ungated')
        self.assertEqual(guards.as_dict()['queue_signal'], 'ungated')
        self.assertEqual(PlannerGuards().queue_signal, 'all')
        self.assertNotIn('queue_signal', PlannerGuards(**ALL, queue_signal='all').as_dict())
        self.assertEqual(PlannerGuards(**ALL, queue_signal='all').as_dict(), PlannerGuards(**ALL).as_dict())
        for bad in ('ALL', 'gated', '', None, 1, True):
            with self.assertRaises(ValueError, msg=bad):
                PlannerGuards(**ALL, queue_signal=bad)
        with self.assertRaisesRegex(ValueError, 'ungated queue signal requires the queue add-P guard'):
            PlannerGuards(restore_floor=True, queue_signal='ungated')
        with self.assertRaisesRegex(ValueError, 'Invalid planner guard option queue_signal'):
            PrefillQueueMonitor({}, signal='gated')


class GuardsOffTests(unittest.TestCase):
    def test_off_is_identical(self):
        p = dict(coefficients=[.01, .0001, 1e-9, .00001], tokens_per_step=4096., restored_per_cached_token=.5)
        d = dict(coefficients=[.02, .001, .01], draft_tokens=6, batch_support=list(range(1, 129)))
        workers = {str(i): dict(prefill=dict(origin='measured', models=[p]),
                                decode=dict(origin='measured', models=[d]),
                                max_running_requests=128, kv_capacity_tokens=2000000) for i in range(4)}
        roles = {'0': 'prefill', '1': 'decode', '2': 'decode', '3': 'decode'}
        work = dict(uncached=8192., cached=40000., pairs=8192*44096., output=800., context=50000., acceptance=3.)
        minutes = [dict(work=work, decode_population=30), dict(work=dict(work, output=900.), decode_population=30)]
        onboarding = {'0': dict(restored_per_cached_token=.01, cached_tokens=10.)}
        args = dict(minutes=minutes, experts=256, top_k=8, decode_population=30, onboarding=onboarding)
        plain = decide(workers, roles, work, **args)
        self.assertNotIn('guards', plain)
        for guards in (None, PlannerGuards()):
            self.assertEqual(decide(workers, roles, work, **args, guards=guards,
                                    queue=dict(p95_s=99., since_switch_s=None)), plain)
        for name, gains in (('steady', STEADY_2), ('noisy', NOISY_POS)):
            self.assertEqual(scripted({4: gains}, guards=PlannerGuards()), scripted({4: gains}), name)

    def test_queue_removal_hold_off_records_are_unchanged(self):
        # G1 + G3 + Q' on, Q-hold off: queue_added changes nothing and the record has no new field.
        remove = {2: [g+.1 for g in STEADY_3]}
        for gains in (remove, {4: STEADY_2}, {**remove, 4: STEADY_2}, {}):
            for queue in (dict(p95_s=25., n=10, window_s=300., at=1.e9, since_switch_s=300.),
                          dict(p95_s=15., n=10, window_s=300., at=1.e9, since_switch_s=None), None):
                base = scripted(gains, guards=PlannerGuards(**ALL), queue=queue)
                self.assertNotIn('queue_removal_hold', base['guards'])
                self.assertNotIn('queue_removal_floor_s', base['guards']['options'])
                self.assertNotIn('queue_removal_hold', base['guards']['fired'])
                for added in (True, False):
                    self.assertEqual(json.dumps(scripted(gains, guards=PlannerGuards(**ALL), queue=queue,
                                                         queue_added=added)), json.dumps(base))
        # All guards off: queue_added is ignored.
        self.assertEqual(scripted(remove, queue_added=True), scripted(remove))


class RestoreFloorTests(unittest.TestCase):
    def test_floor_raises_only_observations_below_the_weighted_mean(self):
        onboarding = {'0': dict(restored_per_cached_token=0., cached_tokens=1000.),
                      '1': dict(restored_per_cached_token=.9, cached_tokens=9000.)}
        floored, record = floor_restore_ratios(onboarding)
        self.assertAlmostEqual(record['reference_ratio'], .81)
        self.assertEqual(record['cached_tokens'], 10000.)
        self.assertTrue(record['applied'])
        self.assertAlmostEqual(floored['0']['restored_per_cached_token'], .81)
        self.assertEqual(floored['0']['observed_restored_per_cached_token'], 0.)
        self.assertIs(floored['1'], onboarding['1'])
        self.assertEqual(onboarding['0']['restored_per_cached_token'], 0.)  # inputs are not mutated
        self.assertEqual({k: v['raised'] for k, v in record['nodes'].items()}, {'0': True, '1': False})

    def test_floor_needs_observations_with_cached_tokens(self):
        for onboarding, reason in ((None, 'no current onboarding'), ({}, 'no current onboarding'),
                                   ({'0': dict(restored_per_cached_token=0., cached_tokens=0.)}, 'no cached')):
            floored, record = floor_restore_ratios(onboarding)
            self.assertIs(floored, onboarding)
            self.assertFalse(record['applied'])
            self.assertIn(reason, record['reason'])
        same = {'0': dict(restored_per_cached_token=.4, cached_tokens=5.),
                '1': dict(restored_per_cached_token=.4, cached_tokens=7.)}
        floored, record = floor_restore_ratios(same)
        self.assertFalse(record['applied'])
        self.assertEqual(floored, same)

    def test_floor_uses_onboarding_nodes_only_and_new_p_nodes_keep_calibration(self):
        p = dict(coefficients=[.01, .0001, 1e-9, .00001], tokens_per_step=4096., restored_per_cached_token=.5)
        d = dict(coefficients=[.02, .001, .01], draft_tokens=6, batch_support=list(range(1, 129)))
        workers = {str(i): dict(prefill=dict(origin='measured', models=[p]),
                                decode=dict(origin='measured', models=[d]),
                                max_running_requests=128, kv_capacity_tokens=2000000) for i in range(4)}
        roles = {'0': 'prefill', '1': 'prefill', '2': 'decode', '3': 'decode'}
        work = dict(uncached=8192., cached=40000., pairs=8192*44096., output=800., context=50000., acceptance=3.)
        minutes = [dict(work=work, decode_population=30), dict(work=dict(work, output=900.), decode_population=30)]
        onboarding = {'0': dict(restored_per_cached_token=0., cached_tokens=1000.),
                      '1': dict(restored_per_cached_token=.9, cached_tokens=9000.)}
        result = decide(workers, roles, work, minutes=minutes, experts=256, top_k=8, decode_population=30,
                        onboarding=onboarding, guards=PlannerGuards(restore_floor=True))
        current = result['candidates'][0]['estimate']['workers']
        self.assertAlmostEqual(current['0']['onboarding']['restored_per_cached_token'], .81)
        self.assertEqual(current['1']['onboarding']['restored_per_cached_token'], .9)
        three = next(o for o in result['candidates'] if p_count(o) == 3)
        added = three['changed'][0]
        self.assertIsNone(three['estimate']['workers'][added]['onboarding'])
        self.assertEqual(three['estimate']['workers'][added]['seconds'], prefill_seconds(p, work))
        self.assertEqual(result['guards']['fired'], ['restore_floor'])
        off = decide(workers, roles, work, minutes=minutes, experts=256, top_k=8, decode_population=30,
                     onboarding=onboarding)
        self.assertEqual(off['candidates'][0]['estimate']['workers']['0']['onboarding']['restored_per_cached_token'], 0.)

    def test_floor_stops_a_single_p_overshoot(self):
        # P rate per node 1/(0.5 + restore ratio); D rate 0.6 per D (+-1 % by minute). With node0's own ratio 0.0,
        # 1P3D (node0 alone) looks 50 % better than 2P2D; floored at 0.9 it is worse, so the planner holds.
        factors = [1., 1.01, .99, 1., 1.01, .99]
        roles = {'0': 'prefill', '1': 'prefill', '2': 'decode', '3': 'decode'}
        onboarding = {'0': dict(restored_per_cached_token=0., cached_tokens=100.),
                      '1': dict(restored_per_cached_token=1., cached_tokens=900.)}

        def estimate(workers, candidate, work, *, onboarding=None, **kwargs):
            ratio = lambda n: (onboarding or {}).get(n, dict(restored_per_cached_token=.5))['restored_per_cached_token']
            p = sum(1/(.5+ratio(n)) for n, r in candidate.items() if r == 'prefill')
            dd = .6*work['f']*sum(r == 'decode' for r in candidate.values())
            return dict(system_rps=min(p, dd), prefill_rps=p, decode_rr_rps=dd)
        minutes = [dict(work=dict(f=f), decode_population=None) for f in factors]
        with patch('evolving_infra.planner.calibration.dynamic.capacities', side_effect=estimate):
            off = decide({n: {} for n in roles}, roles, dict(f=1.), minutes=minutes, experts=256, top_k=8,
                         onboarding=onboarding)
            on = decide({n: {} for n in roles}, roles, dict(f=1.), minutes=minutes, experts=256, top_k=8,
                        onboarding=onboarding, guards=PlannerGuards(restore_floor=True))
        self.assertEqual((off['action'], off['selected']['changed']), ('switch', ['1']))
        self.assertEqual(on['action'], 'hold')
        self.assertEqual(on['guards']['selections']['without_guards']['changed'], ['1'])
        self.assertTrue(on['guards']['changed_decision'])
        self.assertAlmostEqual(on['guards']['restore_floor']['reference_ratio'], .9)


class MinimumGainTests(unittest.TestCase):
    def test_consistent_gain_below_minimum_holds_and_is_recorded(self):
        off = scripted({4: STEADY_2})
        self.assertEqual((off['action'], p_count(off['selected'])), ('switch', 4))
        on = scripted({4: STEADY_2}, guards=PlannerGuards(min_relative_gain=.03))
        self.assertEqual(on['action'], 'hold')
        plus = next(o for o in on['candidates'] if p_count(o) == 4)
        self.assertTrue(plus['minute_gain']['eligible'])  # the two-standard-error rule alone still passes
        self.assertFalse(plus['minute_gain']['passes_min_relative_gain'])
        blocked = on['guards']['min_relative_gain']['blocked']
        self.assertEqual([b['prefill_nodes'] for b in blocked], [4])
        self.assertAlmostEqual(blocked[0]['mean_relative'], .02)
        self.assertEqual(on['guards']['fired'], ['min_relative_gain'])
        self.assertTrue(on['guards']['changed_decision'])

    def test_gain_at_or_above_minimum_switches(self):
        exact = scripted({4: [.0625]*10}, guards=PlannerGuards(min_relative_gain=.0625))  # exact binary fractions
        self.assertEqual((exact['action'], p_count(exact['selected'])), ('switch', 4))
        on = scripted({4: [g+.001 for g in STEADY_3]}, guards=PlannerGuards(min_relative_gain=.03))
        self.assertEqual((on['action'], p_count(on['selected'])), ('switch', 4))
        self.assertEqual(on['guards']['fired'], [])
        self.assertFalse(on['guards']['changed_decision'])
        both = scripted({4: STEADY_2, 5: [g+.02 for g in STEADY_2]}, guards=PlannerGuards(min_relative_gain=.03))
        self.assertEqual(p_count(both['selected']), 5)
        self.assertEqual([b['prefill_nodes'] for b in both['guards']['min_relative_gain']['blocked']], [4])


class QueueAddPrefillTests(unittest.TestCase):
    def queue(self, p95=25., since=300.):
        return dict(p95_s=p95, n=1500, window_s=300., at=1.e9, since_switch_s=since)

    def test_fires_for_positive_gain_below_the_standard_thresholds(self):
        for gains in (STEADY_2, NOISY_POS):
            result = scripted({4: gains}, guards=PlannerGuards(**ALL), queue=self.queue())
            self.assertEqual((result['action'], p_count(result['selected'])), ('switch', 4))
            q = result['guards']['queue_add_prefill']
            self.assertEqual((q['fired'], q['reason']), (True, 'fired'))
            self.assertEqual(q['candidate']['prefill_nodes'], 4)
            self.assertEqual(result['guards']['selected_by'], 'queue_add_prefill')
            self.assertIn('queue_add_prefill', result['guards']['fired'])
            self.assertEqual(result['guards']['selections']['standard']['action'], 'hold')

    def test_first_switch_has_no_minimum_wait(self):
        result = scripted({4: STEADY_2}, guards=PlannerGuards(**ALL), queue=self.queue(since=None))
        self.assertTrue(result['guards']['queue_add_prefill']['fired'])

    def test_not_within_minimum_time_after_a_switch(self):
        for since in (0., 60., 119.9):
            result = scripted({4: STEADY_2}, guards=PlannerGuards(**ALL), queue=self.queue(since=since))
            self.assertEqual(result['action'], 'hold')
            self.assertEqual(result['guards']['queue_add_prefill']['reason'], 'too soon after the last role switch')
        self.assertTrue(scripted({4: STEADY_2}, guards=PlannerGuards(**ALL),
                                 queue=self.queue(since=120.))['guards']['queue_add_prefill']['fired'])

    def test_not_when_the_plus_one_gain_is_not_positive(self):
        for gains in ([-.01]*10, [0.]*10, [-.2, .1, -.2, .1, -.2, .1, -.2, .1, -.2, .1]):
            result = scripted({4: gains}, guards=PlannerGuards(**ALL), queue=self.queue(p95=60.))
            self.assertEqual(result['action'], 'hold')
            self.assertEqual(result['guards']['queue_add_prefill']['reason'], '+1 P mean minute gain not positive')

    def test_not_at_or_below_threshold_or_without_observation(self):
        for queue, reason in ((self.queue(p95=20.), 'not above threshold'), (self.queue(p95=None), 'no live'),
                              (None, 'no live')):
            result = scripted({4: STEADY_2}, guards=PlannerGuards(**ALL), queue=queue)
            self.assertEqual(result['action'], 'hold')
            self.assertIn(reason, result['guards']['queue_add_prefill']['reason'])

    def test_ungated_signal_feeds_the_trigger(self):
        # The guard reads p95_s, which is the ungated p95 with signal 'ungated': all legs 25 s, ungated 8 s -> no add.
        ungated = dict(self.queue(p95=8.), signal='ungated', n_all=1500, p95_all=25., n_ungated=300, p95_ungated=8.,
                       gated_fraction=.8, missing_late_d_need=0)
        result = scripted({4: STEADY_2}, guards=PlannerGuards(**ALL, queue_signal='ungated'), queue=ungated)
        self.assertEqual(result['action'], 'hold')
        self.assertEqual(result['guards']['queue_add_prefill']['reason'], 'P_queue p95 not above threshold')
        self.assertEqual(result['guards']['queue_add_prefill']['observation'], ungated)
        # P legs in the window, none ungated: no p95, with its own reason (Q' and Q-hold).
        none = dict(ungated, n=0, p95_s=None, max_s=None, n_ungated=0, p95_ungated=None, gated_fraction=1.)
        result = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD, queue_signal='ungated'), queue=none, queue_added=True)
        self.assertEqual(result['guards']['queue_add_prefill']['reason'], 'no ungated P leg in the window')
        self.assertEqual((result['guards']['queue_removal_hold']['blocked'],
                          result['guards']['queue_removal_hold']['reason']), (False, 'no ungated P leg in the window'))
        self.assertEqual(p_count(result['selected']), 2)
        # No P leg at all, or the all-leg signal: the old reason.
        for queue in (dict(none, n_all=0, p95_all=None), dict(self.queue(p95=None))):
            self.assertEqual(scripted({4: STEADY_2}, guards=PlannerGuards(**ALL), queue=queue)
                             ['guards']['queue_add_prefill']['reason'], 'no live P_queue observation')

    def test_standard_rule_adding_p_is_kept(self):
        result = scripted({4: STEADY_2, 5: [g+.1 for g in STEADY_2]}, guards=PlannerGuards(**ALL), queue=self.queue())
        self.assertEqual(p_count(result['selected']), 5)
        self.assertEqual(result['guards']['queue_add_prefill']['reason'], 'standard rule already adds a P')
        self.assertEqual(result['guards']['selected_by'], 'minute_standard_error')

    def test_replaces_a_p_removing_switch(self):
        # Documented choice: with the queue above threshold and a positive +1 P gain, Q' replaces the standard choice.
        result = scripted({2: [g+.1 for g in STEADY_2], 4: STEADY_2}, guards=PlannerGuards(**ALL), queue=self.queue())
        self.assertEqual(result['guards']['selections']['standard']['prefill_nodes'], 2)
        self.assertEqual(p_count(result['selected']), 4)


REMOVE_2P = {2: [g+.1 for g in STEADY_3]}  # 3P -> 2P: +13 %, consistent; every other P count -50 %


class QueueRemovalHoldTests(unittest.TestCase):
    def queue(self, p95=15., since=300.):
        return dict(p95_s=p95, n=1500, window_s=300., at=1.e9, since_switch_s=since)

    def test_blocked_removal_without_another_candidate_holds(self):
        result = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD), queue=self.queue(), queue_added=True)
        self.assertEqual(result['action'], 'hold')
        self.assertEqual(result['selected'], result['candidates'][0])
        g = result['guards']
        hold = g['queue_removal_hold']
        self.assertEqual({k: hold[k] for k in ('floor_s', 'queue_added', 'p95_s', 'blocked', 'selected', 'reason')},
                         dict(floor_s=10., queue_added=True, p95_s=15., blocked=True, selected=None,
                              reason='P removal blocked, no remaining candidate: hold'))
        self.assertEqual(hold['candidate']['prefill_nodes'], 2)
        self.assertAlmostEqual(hold['candidate']['mean_relative'], .13)
        self.assertEqual(g['selections']['standard']['prefill_nodes'], 2)  # the standard choice is recorded as before
        self.assertEqual(g['selected_by'], 'queue_removal_hold')
        self.assertEqual(g['fired'], ['queue_removal_hold'])
        self.assertTrue(g['changed_decision'])
        self.assertEqual(g['queue_add_prefill']['reason'], 'P_queue p95 not above threshold')
        self.assertEqual(g['options']['queue_removal_floor_s'], 10.)
        json.dumps(result)  # journal-serialisable

    def test_blocked_removal_selects_the_best_candidate_without_removal(self):
        # 1P is also a removal and is skipped; 4P beats 5P (larger mean), as in the standard order.
        gains = {**REMOVE_2P, 1: [g+.07 for g in STEADY_3], 4: [g+.01 for g in STEADY_3],
                 5: [g+.005 for g in STEADY_3]}
        self.assertEqual(p_count(scripted(gains, guards=PlannerGuards(**ALL), queue=self.queue(),
                                          queue_added=True)['selected']), 2)
        result = scripted(gains, guards=PlannerGuards(**QHOLD), queue=self.queue(), queue_added=True)
        self.assertEqual((result['action'], p_count(result['selected'])), ('switch', 4))
        g = result['guards']
        self.assertEqual((g['queue_removal_hold']['candidate']['prefill_nodes'],
                          g['queue_removal_hold']['selected']['prefill_nodes']), (2, 4))
        self.assertEqual(g['queue_removal_hold']['reason'], 'P removal blocked, best remaining candidate selected')
        self.assertEqual((g['selected_by'], g['fired']), ('queue_removal_hold', ['queue_removal_hold']))
        self.assertEqual(g['queue_add_prefill']['reason'], 'P_queue p95 not above threshold')
        # Above the Q' threshold, Q' sees the re-selected +P as the standard choice and does not add another P.
        # Above the Q' threshold, Q' reads the standard choice (a removal) and adds the +1 P itself.
        high = scripted(gains, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=25.), queue_added=True)
        self.assertEqual(p_count(high['selected']), 4)
        self.assertEqual(high['guards']['queue_add_prefill']['reason'], 'fired')
        self.assertEqual(high['guards']['selected_by'], 'queue_add_prefill')
        self.assertEqual(high['guards']['queue_removal_hold']['selected']['prefill_nodes'], 4)
        self.assertEqual(high['guards']['fired'], ['queue_removal_hold', 'queue_add_prefill'])
        # Equal means: fewest changes first (4P over 5P), as in _decide.
        tie = scripted({**REMOVE_2P, 4: [g+.01 for g in STEADY_3], 5: [g+.01 for g in STEADY_3]},
                       guards=PlannerGuards(**QHOLD), queue=self.queue(), queue_added=True)
        self.assertEqual(p_count(tie['selected']), 4)
        # A remaining candidate must pass the standard rule: an ineligible or G3-failing +P is not selected.
        for weak in (NOISY_POS, STEADY_2):
            self.assertEqual(scripted({**REMOVE_2P, 4: weak}, guards=PlannerGuards(**QHOLD), queue=self.queue(),
                                      queue_added=True)['action'], 'hold')

    def test_queue_add_prefill_overrides_a_plus_two_reselection(self):
        # The standard rule removes a P (2P, +13 %); +2 P (5P, +6 %) passes the standard rule; +1 P (4P, +1 %)
        # fails G3. Q-hold re-selects 5P, but Q' reads the standard choice and adds the +1 P as with Q-hold off, so a
        # probe (which applies only +1 P) still adds a P.
        gains = {**REMOVE_2P, 4: [g-.02 for g in STEADY_3], 5: [g+.03 for g in STEADY_3]}
        off = scripted(gains, guards=PlannerGuards(**ALL), queue=self.queue(p95=25.), queue_added=True)
        on = scripted(gains, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=25.), queue_added=True)
        self.assertEqual((off['action'], p_count(off['selected'])), ('switch', 4))
        self.assertEqual((on['action'], p_count(on['selected'])), ('switch', 4))
        self.assertEqual(on['guards']['queue_removal_hold']['selected']['prefill_nodes'], 5)
        self.assertEqual(on['guards']['queue_add_prefill'], off['guards']['queue_add_prefill'])
        self.assertEqual(on['guards']['selected_by'], 'queue_add_prefill')
        # Between the floor and the Q' threshold, the re-selected +2 P stands.
        mid = scripted(gains, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=15.), queue_added=True)
        self.assertEqual((mid['action'], p_count(mid['selected']), mid['guards']['selected_by']),
                         ('switch', 5, 'queue_removal_hold'))

    def test_queue_add_prefill_is_never_changed_by_the_hold(self):
        # Wherever Q' fires with Q-hold off, the same decision with Q-hold armed has the same final roles and Q' record.
        cases = [REMOVE_2P, {**REMOVE_2P, 4: STEADY_2}, {**REMOVE_2P, 4: [g-.02 for g in STEADY_3], 5: STEADY_3},
                 {**REMOVE_2P, 1: STEADY_3, 4: NOISY_POS, 6: [g+.01 for g in STEADY_3]}, {4: STEADY_2}, {2: STEADY_3}]
        fired = 0
        for gains in cases:
            for p95 in (20.5, 25., 60.):
                off = scripted(gains, guards=PlannerGuards(**ALL), queue=self.queue(p95=p95), queue_added=True)
                on = scripted(gains, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=p95), queue_added=True)
                self.assertEqual(on['guards']['queue_add_prefill'], off['guards']['queue_add_prefill'])
                if off['guards']['queue_add_prefill']['fired']:
                    fired += 1
                    self.assertEqual((on['action'], on['selected']['changed']), (off['action'], off['selected']['changed']))
        self.assertGreaterEqual(fired, 6)

    def test_nan_p95_does_not_block_and_has_its_own_reason(self):
        result = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=float('nan')), queue_added=True)
        self.assertEqual(p_count(result['selected']), 2)
        self.assertEqual((result['guards']['queue_removal_hold']['blocked'],
                          result['guards']['queue_removal_hold']['reason']), (False, 'P_queue p95 is NaN'))

    def test_p95_below_the_floor_allows_the_removal(self):
        for p95, blocked in ((9.99, False), (10., True), (25., True)):
            result = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=p95), queue_added=True)
            hold = result['guards']['queue_removal_hold']
            self.assertEqual(hold['blocked'], blocked, p95)
            self.assertEqual(p_count(result['selected']), 3 if blocked else 2, p95)
            self.assertEqual(hold['candidate']['prefill_nodes'], 2)
        result = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=9.99), queue_added=True)
        g = result['guards']
        self.assertEqual(g['queue_removal_hold']['reason'], 'P_queue p95 below the floor')
        self.assertEqual((g['selected_by'], g['fired'], g['changed_decision']), ('minute_standard_error', [], False))

    def test_removal_before_a_confirmed_queue_add_is_allowed(self):
        # The first switch of a run often removes P while P_queue is high; Q-hold arms only after a Q' switch.
        result = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=25.), queue_added=False)
        self.assertEqual((result['action'], p_count(result['selected'])), ('switch', 2))
        g = result['guards']
        self.assertEqual((g['queue_removal_hold']['blocked'], g['queue_removal_hold']['queue_added']), (False, False))
        self.assertEqual(g['queue_removal_hold']['reason'], 'no queue add-P switch confirmed earlier in the run')
        self.assertEqual(g['selected_by'], 'minute_standard_error')
        self.assertEqual(g['queue_add_prefill']['reason'], '+1 P mean minute gain not positive')
        self.assertFalse(g['changed_decision'])

    def test_missing_p95_does_not_block(self):
        for queue in (None, self.queue(p95=None)):
            result = scripted(REMOVE_2P, guards=PlannerGuards(**QHOLD), queue=queue, queue_added=True)
            self.assertEqual(p_count(result['selected']), 2)
            hold = result['guards']['queue_removal_hold']
            self.assertEqual((hold['blocked'], hold['p95_s'], hold['reason']),
                             (False, None, 'no live P_queue observation'))

    def test_adding_or_holding_is_never_blocked(self):
        add = scripted({4: [g+.01 for g in STEADY_3]}, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=60.),
                       queue_added=True)
        hold = scripted({}, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=60.), queue_added=True)
        self.assertEqual((add['action'], p_count(add['selected'])), ('switch', 4))
        self.assertEqual(hold['action'], 'hold')
        for result in (add, hold):
            record = result['guards']['queue_removal_hold']
            self.assertEqual((record['blocked'], record['candidate'], record['selected'], record['reason']),
                             (False, None, None, 'standard rule does not remove a P'))
            self.assertNotIn('queue_removal_hold', result['guards']['fired'])

    def test_queue_add_prefill_still_fires_in_the_same_decision(self):
        # The removal is blocked; the +1 P fails G3 (+2 %), so the standard choice becomes a hold; Q' then adds that
        # +1 P (positive gain, p95 above its threshold) in the same decision.
        gains = {**REMOVE_2P, 4: STEADY_2}
        result = scripted(gains, guards=PlannerGuards(**QHOLD), queue=self.queue(p95=25.), queue_added=True)
        self.assertEqual((result['action'], p_count(result['selected'])), ('switch', 4))
        g = result['guards']
        self.assertTrue(g['queue_removal_hold']['blocked'])
        self.assertIsNone(g['queue_removal_hold']['selected'])
        self.assertEqual((g['queue_add_prefill']['fired'], g['queue_add_prefill']['reason']), (True, 'fired'))
        self.assertEqual(g['selected_by'], 'queue_add_prefill')
        self.assertEqual(g['fired'], ['min_relative_gain', 'queue_removal_hold', 'queue_add_prefill'])
        self.assertTrue(g['changed_decision'])
        # Q' itself is never blocked: the same decision without Q-hold also adds the P (F5), only the record differs.
        without = scripted(gains, guards=PlannerGuards(**ALL), queue=self.queue(p95=25.), queue_added=True)
        self.assertEqual(without['selected'], result['selected'])
        self.assertEqual(without['guards']['fired'], ['min_relative_gain', 'queue_add_prefill'])


def recorded_decisions():
    """The recorded decisions of the three guards runs (fixture), each with its rebuilt standard decision.

    A fixture candidate stores only `changed`; its roles are the current roles with those nodes flipped, which is how
    _role_maps builds every candidate (checked on every recorded candidate when the fixture was built).
    The standard decision (before Q') is the recorded `selections.standard`.
    """
    fixture = json.loads((Path(__file__).parent/'fixtures'/'qhold-recorded-decisions.json').read_text())
    flip = dict(prefill='decode', decode='prefill')
    for run in fixture['runs']:
        for row in run['decisions']:
            roles = row['roles']
            candidates = [dict(o, roles={n: flip[r] if n in o['changed'] else r for n, r in roles.items()})
                          for o in row['candidates']]
            standard = row['guards']['selections']['standard']
            selected = next(o for o in candidates if o['changed'] == standard['changed'])
            decision = dict(action=standard['action'], selected=selected, candidates=candidates)
            yield run, row, decision


def replay_queue_guards(run, row, decision, floor_s=10., queue_added=None):
    """Q-hold then Q' on one recorded standard decision, through the same helper as `decide`."""
    guards = PlannerGuards(**run['options'], queue_removal_floor_s=floor_s)
    decision = dict(decision)
    hold, add = _queue_guards(decision, decision['candidates'][0]['roles'], guards,
                              row['guards']['queue_add_prefill']['observation'],
                              row['queue_added'] if queue_added is None else queue_added)
    return decision, hold, add


class RecordedDecisionTests(unittest.TestCase):
    """Q-hold (floor 10 s) on the recorded decisions of the three recorded guard runs (the first: 18 trays, CC c252).

    Decision level only: a changed decision would change every later input of that run.
    """

    def at(self, job, t):
        return next((run, row, decision) for run, row, decision in recorded_decisions()
                    if run['job'] == job and row['t'] == t)

    def test_fixture_matches_the_standard_rule_and_the_recorded_decisions(self):
        rows = list(recorded_decisions())
        self.assertEqual(len(rows), 26)
        for run, row, decision in rows:
            current = p_count(decision['candidates'][0])
            # The standard order re-selects the recorded standard choice (2 SE + G3, mean then fewest changes).
            selectable = [o for o in decision['candidates'][1:] if 'minute_gain' in o and o['minute_gain']['eligible']
                          and o['minute_gain']['passes_min_relative_gain']]
            best = max(selectable, key=lambda o: (o['minute_gain']['mean_rps'], -len(o['changed'])), default=None)
            self.assertEqual(decision['selected'], best or decision['candidates'][0], (run['job'], row['t']))
            # Q' on the standard decision gives the recorded result (fixture check, Q-hold not applied).
            add, plus = _queue_add_prefill(decision, decision['candidates'][0]['roles'], PlannerGuards(**run['options']),
                                           row['guards']['queue_add_prefill']['observation'])
            self.assertEqual((add['fired'], add['reason']), (row['guards']['queue_add_prefill']['fired'],
                                                             row['guards']['queue_add_prefill']['reason']))
            final = plus if add['fired'] else decision['selected']
            self.assertEqual(sorted(final['changed']), sorted(row['selected_changed']), (run['job'], row['t']))
            self.assertEqual(current, p_count(dict(roles=row['roles'])))

    def test_only_the_p_removal_after_two_queue_adds_changes(self):
        changed = []
        for run, row, decision in recorded_decisions():
            final, hold, add = replay_queue_guards(run, row, decision)
            recorded = sorted(row['selected_changed'])
            if (final['action'] == 'switch', sorted(final['selected']['changed'])) != (row['action'] == 'switch', recorded):
                changed.append((run['job'], row['t'], p_count(decision['candidates'][0]), p_count(final['selected'])))
            # Q' fires exactly where it fired in the run.
            self.assertEqual(add['fired'], row['guards']['queue_add_prefill']['fired'], (run['job'], row['t']))
        self.assertEqual(changed, [(281395, 6316.5, 9, 9)])

    def test_281395_first_switch_is_not_blocked(self):
        run, row, decision = self.at(281395, 1824.)
        final, hold, add = replay_queue_guards(run, row, decision)
        self.assertEqual((p_count(decision['candidates'][0]), p_count(final['selected'])), (12, 6))
        self.assertEqual((hold['blocked'], hold['queue_added'], hold['reason']),
                         (False, False, 'no queue add-P switch confirmed earlier in the run'))
        self.assertGreaterEqual(hold['p95_s'], 20.)  # 24.4 s: a plain p95 rule would block the 12P -> 6P switch
        self.assertTrue(replay_queue_guards(run, row, decision, queue_added=True)[1]['blocked'])

    def test_281395_queue_add_at_3119_is_unchanged(self):
        run, row, decision = self.at(281395, 3118.7)
        final, hold, add = replay_queue_guards(run, row, decision)
        self.assertEqual((hold['blocked'], hold['reason']), (False, 'standard rule does not remove a P'))
        self.assertTrue(add['fired'])
        self.assertEqual((p_count(decision['candidates'][0]), p_count(final['selected'])), (7, 8))

    def test_281395_removal_at_6317_is_blocked(self):
        run, row, decision = self.at(281395, 6316.5)
        self.assertTrue(row['queue_added'])  # Q' switches confirmed at 3,119 s and 5,102 s
        final, hold, add = replay_queue_guards(run, row, decision)
        self.assertTrue(hold['blocked'])
        self.assertAlmostEqual(hold['p95_s'], 20.57, places=2)
        self.assertEqual(hold['candidate']['prefill_nodes'], 7)
        # 8P is also selectable but removes a P too; no +P candidate is selectable, so the planner holds 9P9D.
        self.assertIsNone(hold['selected'])
        self.assertEqual(add['reason'], '+1 P mean minute gain not positive')
        self.assertEqual((final['action'], p_count(final['selected'])), ('hold', 9))
        # Allowed below the floor (e.g. 25 s) and before any confirmed queue add.
        self.assertFalse(replay_queue_guards(run, row, decision, floor_s=25.)[1]['blocked'])
        self.assertFalse(replay_queue_guards(run, row, decision, queue_added=False)[1]['blocked'])

    def test_decisions_after_queue_adds_in_281124_and_275272_are_unchanged(self):
        for job, t, p in ((281124, 2943.8, 4), (281124, 5357.6, 4), (275272, 3784.6, 4), (275272, 5162.7, 5)):
            run, row, decision = self.at(job, t)
            self.assertTrue(row['queue_added'], (job, t))
            final, hold, add = replay_queue_guards(run, row, decision)
            self.assertEqual((hold['blocked'], hold['reason']), (False, 'standard rule does not remove a P'))
            self.assertEqual(p_count(final['selected']), p, (job, t))
            self.assertEqual(add['fired'], row['guards']['queue_add_prefill']['fired'])


def ungated_observations():
    """{(job, t): ungated P_queue observation} replayed from the recorded traces (bd-recorded-inputs fixture)."""
    fixture = json.loads((Path(__file__).parent/'fixtures'/'bd-recorded-inputs.json').read_text())
    return {(run['job'], row['t']): row['queue_ungated'] for run in fixture['runs'] for row in run['decisions']}


class RecordedUngatedSignalTests(unittest.TestCase):
    """Q' and Q-hold on the ungated P_queue signal at the recorded decisions [S, decision level]."""

    def test_no_recorded_probe_or_queue_add_survives_the_ungated_signal(self):
        observed = ungated_observations()
        first_change = {}
        for run, row, decision in recorded_decisions():
            queue = observed[(run['job'], row['t'])]
            guards = PlannerGuards(**run['options'], queue_removal_floor_s=10., queue_signal='ungated')
            final = dict(decision)
            hold, add = _queue_guards(final, final['candidates'][0]['roles'], guards, queue, row['queue_added'])
            self.assertFalse(add['fired'], (run['job'], row['t']))
            # The all-leg view of the replay is close to the recorded live observation.
            self.assertLess(abs(queue['p95_all']-row['guards']['queue_add_prefill']['observation']['p95_s']), 2.5)
            if row['trigger'] == 'queue_probe':
                self.assertLessEqual(queue['p95_s'], 20.)  # a probe needs p95 above 20 s: it would not run
                first_change.setdefault(run['job'], row['t'])
            elif row['t'] < 6400:
                self.assertLessEqual(queue['p95_s'], 20.)
        self.assertEqual(first_change, {281395: 2157.5, 281124: 2340.3, 275272: 3181.2})
        # At 6,317 s (9P9D -> 7P11D) the ungated p95 is 3.5 s: Q-hold would not block that removal.
        self.assertLess(observed[(281395, 6316.5)]['p95_s'], 10.)
        self.assertGreater(observed[(281395, 6316.5)]['gated_fraction'], .8)


class ControllerGuardTests(unittest.TestCase):
    def setUp(self):
        self.urls = {'node0': 'http://a', 'node1': 'http://b', 'node2': 'http://c'}
        self.initial = dict(assignment=dict(epoch='e', revision=0,
                            roles={'http://a': 'prefill', 'http://b': 'decode', 'http://c': 'decode'}),
                            completed_workload=dict(requests=0, input_tokens=0, uncached_tokens=0,
                                causal_attention_pairs=0, output_tokens=0, verified_output_tokens=0, verify_steps=0),
                            decode_population=dict(fields=['p_completed', 'p_pending', 'unknown'],
                                                   workers={'http://b': [3, 5, 0]}))
        p = dict(coefficients=[.01, .0001, 1e-9, .00001], tokens_per_step=4096., restored_per_cached_token=.5)
        d = dict(coefficients=[.02, .001, .01], draft_tokens=6, batch_support=list(range(1, 129)))
        self.workers = {n: dict(prefill=dict(origin='measured', models=[p]), decode=dict(origin='measured', models=[d]),
                                max_running_requests=128, kv_capacity_tokens=2000000) for n in self.urls}

    def controller(self, guards, **kwargs):
        c = Controller(dict(workers=self.workers), self.urls, self.initial, experts=256, top_k=8, guards=guards, **kwargs)
        self.now = 1000.
        c.clock = lambda: self.now
        return c

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

    def test_off_propose_is_unchanged(self):
        plain = self.controller(None).propose(self.samples())
        self.assertNotIn('guards', plain)
        self.assertEqual(self.controller(PlannerGuards()).propose(self.samples(), queue=dict(p95_s=99.)), plain)

    def test_probe_waits_for_the_minimum_time_after_a_switch(self):
        guards = PlannerGuards(**ALL)
        c = self.controller(guards, exclusion_s=10.)
        high, low = dict(p95_s=30.), dict(p95_s=20.)
        self.assertTrue(c.queue_probe_due(high))  # no switch yet
        self.assertFalse(c.queue_probe_due(low))
        self.assertFalse(c.queue_probe_due(None))
        self.assertFalse(self.controller(None).queue_probe_due(high))
        self.assertFalse(self.controller(PlannerGuards(restore_floor=True)).queue_probe_due(high))
        after = copy.deepcopy(self.initial['assignment'])
        after.update(revision=1)
        after['roles']['http://b'] = 'prefill'

        class Client:
            timeout = 5
            def request(self, body=None):
                return dict(assignment=after, draining=[])
        decision = dict(action='switch', before=self.initial['assignment'],
                        selected=dict(roles={'node0': 'prefill', 'node1': 'prefill', 'node2': 'decode'}))
        c.apply(self.initial, decision, Client(), lambda row: None)
        self.assertEqual(c.last_switch_at, 1000.)
        current = copy.deepcopy(self.initial)
        current['assignment'] = after
        self.now = 1005.
        self.assertFalse(c.queue_probe_due(high))  # excluded: prediction sampling not resumed
        self.assertEqual(c.prediction_observation(current, 1011., lambda row: None), 'reset')
        self.now = 1119.
        self.assertEqual(c.since_switch(), 119.)
        self.assertFalse(c.queue_probe_due(high))
        self.now = 1120.
        self.assertTrue(c.queue_probe_due(high))

    def test_propose_records_trigger_queue_and_time_since_switch(self):
        c = self.controller(PlannerGuards(**ALL))
        decision = c.propose(self.samples(), queue=dict(p95_s=31., n=900, window_s=300., at=5.), trigger='queue_probe')
        g = decision['guards']
        self.assertEqual(g['trigger'], 'queue_probe')
        self.assertEqual(g['queue_add_prefill']['observation'],
                         dict(p95_s=31., n=900, window_s=300., at=5., since_switch_s=None))
        self.assertEqual(g['options']['queue_p95_s'], 20.)
        json.dumps(decision)  # journal-serialisable

    def run_switches(self, guards, switches):
        """Propose before and after each role switch; return the queue_added passed to decide() at each proposal.

        `switches` holds (prefill nodes, guards.selected_by of the applied decision, confirmed).
        """
        c = self.controller(guards, exclusion_s=10.)
        seen = []

        def stub(workers, roles, work, **kwargs):
            seen.append(kwargs.get('queue_added', 'not passed'))
            current = dict(changed=[], roles=dict(roles))
            return dict(action='hold', selected=current, candidates=[current], guards={})

        def propose():
            samples = self.samples()
            for row in samples:
                row['snapshot']['assignment'] = c.assignment
            with patch('evolving_infra.planner.calibration.controller.decide', side_effect=stub):
                c.propose(samples, queue=dict(p95_s=25., n=10, window_s=300., at=5.))
        propose()
        for prefill, selected_by, confirmed in switches:
            before = c.assignment
            after = dict(before, revision=before['revision']+1,
                         roles={url: 'prefill' if node in prefill else 'decode' for node, url in self.urls.items()})

            class Client:
                timeout = 5
                def request(self, body=None):
                    return dict(assignment=after if confirmed else before, draining=[])
            snapshot = dict(copy.deepcopy(self.initial), assignment=before)
            decision = dict(action='switch', before=before, guards=dict(selected_by=selected_by),
                            selected=dict(roles={n: 'prefill' if n in prefill else 'decode' for n in self.urls}))
            if not confirmed:
                with self.assertRaisesRegex(RuntimeError, 'unconfirmed'):
                    c.apply(snapshot, decision, Client(), lambda row: None)
                self.assertEqual(c.assignment, before)
            else:
                c.apply(snapshot, decision, Client(), lambda row: None)
                self.now += 20.
                self.assertEqual(c.prediction_observation(dict(snapshot, assignment=after), self.now,
                                                          lambda row: None), 'reset')
            propose()
        return seen, c

    def test_queue_added_follows_the_confirmed_queue_add_switch(self):
        guards = PlannerGuards(**QHOLD)
        seen, c = self.run_switches(guards, [
            ({'node0', 'node1'}, 'queue_add_prefill', False),     # intent journaled, PUT unconfirmed: not armed
            ({'node0', 'node1'}, 'minute_standard_error', True),  # a standard +P switch does not arm
            ({'node0'}, 'queue_removal_hold', True),              # neither does a Q-hold re-selection
            ({'node0', 'node1'}, 'queue_add_prefill', True),      # the confirmed Q' switch arms
            ({'node0'}, 'minute_standard_error', True),           # and the state survives later switches
            ({'node0', 'node2'}, None, True)])
        self.assertEqual(seen, [False, False, False, False, True, True, True])
        self.assertTrue(c.queue_added)

    def test_queue_added_is_passed_only_with_the_removal_hold_on(self):
        seen, c = self.run_switches(PlannerGuards(**ALL), [({'node0', 'node1'}, 'queue_add_prefill', True)])
        self.assertEqual(seen, ['not passed', 'not passed'])
        seen, c = self.run_switches(None, [({'node0', 'node1'}, 'queue_add_prefill', True)])
        self.assertEqual(seen, ['not passed', 'not passed'])

    def test_propose_records_the_removal_hold(self):
        # 2P1D with the real decide(): the D side is the bottleneck, so the standard rule removes a P (1P2D). Armed and
        # with p95 at or above the floor, Q-hold blocks it; 3P0D is not a map, so the planner holds.
        self.initial['assignment']['roles'] = {'http://a': 'prefill', 'http://b': 'prefill', 'http://c': 'decode'}
        self.initial['decode_population']['workers'] = {'http://c': [3, 5, 0]}
        for added, p95, removes in ((False, 12., True), (True, 9., True), (True, 12., False)):
            c = self.controller(PlannerGuards(**QHOLD))
            c.queue_added = added
            decision = c.propose(self.samples(), queue=dict(p95_s=p95, n=900, window_s=300., at=5.))
            g = decision['guards']
            self.assertEqual(g['selections']['standard']['prefill_nodes'], 1)
            self.assertEqual((g['queue_removal_hold']['queue_added'], g['queue_removal_hold']['p95_s'],
                              g['queue_removal_hold']['blocked']), (added, p95, not removes))
            self.assertEqual(g['queue_removal_hold']['candidate']['prefill_nodes'], 1)
            self.assertEqual((decision['action'], p_count(decision['selected'])),
                             ('switch', 1) if removes else ('hold', 2))
            self.assertEqual(g['options']['queue_removal_floor_s'], 10.)
            json.dumps(decision)


class QueueMonitorTests(unittest.TestCase):
    def leg(self, wait, forward, done=None, ts=None, shift=1000.):
        done = forward+1. if done is None else done
        return json.dumps(dict(rid='r', ts=forward+shift+2. if ts is None else ts, mono_to_unix=shift,
                               wait_queue_entry_time=wait, forward_entry_time=forward,
                               prefill_finished_time=done))+'\n'

    def test_p95_window_roles_and_incremental_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a.jsonl', Path(temp)/'b.jsonl'
            # node0 is P throughout; node1 turns P at unix 1500 (monotonic 500 with mono_to_unix 1000).
            a.write_text(''.join(self.leg(100.+i, 100.+i+w) for i, w in enumerate([1., 2., 3., 4., 50.]))
                         + self.leg(300., 330., done=0.) + '{"partial": ')
            b.write_text(self.leg(400., 460.) + self.leg(510., 540.))
            m = PrefillQueueMonitor({'node0': 'prefill', 'node1': 'decode'}, window_s=300.)
            m.set_roles({'node0': 'prefill', 'node1': 'prefill'}, at=1500.)
            m.read({'node0': [a], 'node1': [b]})
            obs = m.observe(1400.)
            self.assertEqual(obs['n'], 5)  # node1's leg queued before 1500 and the unfinished leg are not P legs
            self.assertAlmostEqual(obs['p95_s'], percentile([1., 2., 3., 4., 50.], .95))
            self.assertEqual(obs['max_s'], 50.)
            self.assertEqual(m.observe(1560.)['n'], 1)  # window (1260, 1560]: only node1's leg forwarding at 1540
            with a.open('a') as f:
                f.write('"x"}\n' + self.leg(600., 625.))
            m.read({'node0': [a], 'node1': [b]})
            self.assertEqual(m.unreadable_lines, 0)
            obs = m.observe(1640.)
            self.assertEqual((obs['n'], obs['max_s']), (2, 30.))
            self.assertEqual(m.observe(1999.)['n'], 0)
            self.assertIsNone(m.observe(1999.)['p95_s'])

    def test_roles_at_wait_entry_across_repeated_switches(self):
        # node0: P, then D from 1500, then P again from 2000. Roles recorded after the read still apply.
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp)/'a.jsonl'
            a.write_text(self.leg(400., 490.) + self.leg(495., 560.) + self.leg(600., 650.) + self.leg(1100., 1130.)
                         + json.dumps(dict(ts=1., prefill_finished_time=0.0, wait_queue_entry_time=5.,
                                           forward_entry_time=6.)) + '\n')
            m = PrefillQueueMonitor({'node0': 'prefill'}, window_s=1000.)
            m.read({'node0': [a]})
            m.set_roles({'node0': 'decode'}, at=1500.)
            m.set_roles({'node0': 'prefill'}, at=2000.)
            obs = m.observe(2200.)
            # P legs queued at 1400 and 1495 (still P at wait entry, forwarded after the switch) and 2100; the leg
            # queued at 1600 is a D-period leg; the zero prefill_finished_time record is skipped unparsed.
            self.assertEqual(obs['n'], 3)
            self.assertEqual(obs['max_s'], 90.)
            self.assertEqual(m.unreadable_lines, 0)
            self.assertEqual(m.role_at('node0', 1499.), 'prefill')
            self.assertEqual(m.role_at('node0', 1500.), 'decode')
            with self.assertRaisesRegex(ValueError, 'time order'):
                m.set_roles({'node0': 'decode'}, at=1999.)
            m.disable('role history: test')
            with self.assertRaisesRegex(ValueError, 'disabled'):
                m.observe(2300.)

    def test_visible_by_keeps_only_records_written_in_time(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp)/'a.jsonl'
            a.write_text(self.leg(100., 130., ts=1200.) + self.leg(110., 135., ts=1140.))
            m = PrefillQueueMonitor({'node0': 'prefill'})
            m.read({'node0': [a]})
            self.assertEqual(m.observe(1150., visible_by=1150.)['n'], 1)
            self.assertEqual(m.observe(1150.)['n'], 2)

    def gated_leg(self, wait, forward, skips=None, need=True):
        row = json.loads(self.leg(wait, forward))
        if need:
            row['late_d_need'] = dict(late_d_gate_skips=skips or 0, late_d_park_started=0.0)
        return json.dumps(row)+'\n'

    def test_ungated_signal_uses_legs_without_gate_skips_and_records_both_views(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp)/'a.jsonl'
            # Waits 1-4 s ungated, 30 s / 40 s gated, 2 s and 50 s without late_d_need (counted as ungated).
            a.write_text(self.gated_leg(100., 101.) + self.gated_leg(110., 112.) + self.gated_leg(120., 123.)
                         + self.gated_leg(130., 134.) + self.gated_leg(140., 170., skips=3)
                         + self.gated_leg(150., 190., skips=1) + self.gated_leg(160., 162., need=False)
                         + self.gated_leg(165., 215., need=False))
            all_legs = PrefillQueueMonitor({'node0': 'prefill'})
            ungated = PrefillQueueMonitor({'node0': 'prefill'}, signal='ungated')
            for m in (all_legs, ungated):
                m.read({'node0': [a]})
            plain = all_legs.observe(1300.)
            self.assertEqual(sorted(plain), ['at', 'max_s', 'n', 'p95_s', 'window_s'])  # 'all': nothing added
            obs = ungated.observe(1300.)
            waits_all, waits_ungated = [1., 2., 3., 4., 30., 40., 2., 50.], [1., 2., 3., 4., 2., 50.]
            self.assertEqual((obs['n'], obs['n_all'], obs['n_ungated']), (6, 8, 6))
            self.assertAlmostEqual(obs['p95_s'], percentile(waits_ungated, .95))
            self.assertEqual(obs['p95_ungated'], obs['p95_s'])
            self.assertAlmostEqual(obs['p95_all'], percentile(waits_all, .95))
            self.assertEqual((plain['n'], plain['p95_s'], plain['max_s']), (8, obs['p95_all'], obs['max_all']))
            self.assertEqual((obs['max_s'], obs['signal'], obs['missing_late_d_need']), (50., 'ungated', 2))
            self.assertAlmostEqual(obs['gated_fraction'], .25)
            self.assertEqual(ungated.unreadable_lines, 0)

    def test_ungated_signal_without_ungated_legs_has_no_p95(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp)/'a.jsonl'
            a.write_text(self.gated_leg(100., 130., skips=2) + self.gated_leg(110., 135., skips=1))
            m = PrefillQueueMonitor({'node0': 'prefill'}, signal='ungated')
            m.read({'node0': [a]})
            obs = m.observe(1300.)
            self.assertEqual((obs['n'], obs['p95_s'], obs['max_s'], obs['n_all'], obs['gated_fraction']),
                             (0, None, None, 2, 1.))
            empty = m.observe(5000.)
            self.assertEqual((empty['n_all'], empty['gated_fraction'], empty['p95_all']), (0, None, None))

    def test_malformed_late_d_need_is_parsed_only_for_the_ungated_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp)/'a.jsonl'
            row = json.loads(self.leg(100., 105.))
            row['late_d_need'] = dict(late_d_gate_skips='many')
            a.write_text(json.dumps(row)+'\n')
            plain = PrefillQueueMonitor({'node0': 'prefill'})
            plain.read({'node0': [a]})
            self.assertEqual((plain.observe(1300.)['n'], plain.unreadable_lines), (1, 0))
            ungated = PrefillQueueMonitor({'node0': 'prefill'}, signal='ungated')
            ungated.read({'node0': [a]})
            self.assertEqual((ungated.observe(1300.)['n_all'], ungated.unreadable_lines), (0, 1))

    def test_percentile_is_linear(self):
        self.assertEqual(percentile([1., 2., 3., 4., 5.], .5), 3.)
        self.assertAlmostEqual(percentile(list(range(1, 101)), .95), 95.05)
        self.assertIsNone(percentile([], .95))


if __name__ == '__main__':
    unittest.main()
