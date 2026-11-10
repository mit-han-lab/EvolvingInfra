import copy
import math
from statistics import fmean, stdev
import unittest
from unittest.mock import patch
from evolving_infra.planner.policy.capacity import capacities, prefill_seconds
from evolving_infra.planner.policy.split_policy import decide
from evolving_infra.planner.policy.workload import workload_delta
from planner_support import DECODE_MODEL, MINUTES, PREFILL_MODEL, WORK, measured_workers

class DynamicCapacityTests(unittest.TestCase):
    def test_images_require_a_measured_vision_term_and_increase_only_prefill_demand(self):
        profile = copy.deepcopy(PREFILL_MODEL)
        text = dict(uncached=8192., cached=40000., pairs=8192*44096.)
        image = dict(text, vision_matmul_flops=1e12)
        with self.assertRaisesRegex(ValueError, 'measured vision coefficient'):
            prefill_seconds(profile, image)
        profile['coefficients'].append(2e-14)
        self.assertAlmostEqual(prefill_seconds(profile, image)-prefill_seconds(profile, text), .02)
    def setUp(self):
        self.workers=measured_workers()
        self.roles={'0':'prefill','1':'decode','2':'decode','3':'decode'}
        self.work,self.minutes=WORK,MINUTES
    def test_heterogeneous_rr_uses_slowest_not_mean(self):
        equal=capacities(self.workers,self.roles,self.work,experts=256,top_k=8)
        self.workers['1']=copy.deepcopy(self.workers['1'])
        self.workers['1']['decode']['models'][0]['coefficients']=[.04,.002,.02]
        slow=capacities(self.workers,self.roles,self.work,experts=256,top_k=8)
        self.assertAlmostEqual(slow['decode_rr_rps'],equal['decode_rr_rps']/2)
    def test_kv_and_population_bound_do_not_change_engine_limit(self):
        result=capacities(self.workers,self.roles,self.work,experts=256,top_k=8,decode_population=30)
        for i in ('1','2','3'):
            self.assertEqual(result['workers'][i]['operating_points'][0]['batch'],10)
            self.assertEqual(self.workers[i]['max_running_requests'],128)
    def test_low_population_does_not_invent_busy_decoders(self):
        empty=capacities(self.workers,self.roles,self.work,experts=256,top_k=8,decode_population=0)
        self.assertEqual(empty['decode_rr_rps'],0)
        low=capacities(self.workers,self.roles,self.work,experts=256,top_k=8,decode_population=.6)
        for i in ('1','2','3'):
            point=low['workers'][i]['operating_points'][0]
            self.assertAlmostEqual(point['batch']*point['duty_fraction'],.2)

    def test_each_p_count_is_one_candidate_with_fewest_changes(self):
        result=decide(self.workers,self.roles,self.work,minutes=self.minutes,experts=256,top_k=8)
        counts=[sum(r=='prefill' for r in o['roles'].values()) for o in result['candidates']]
        self.assertEqual(counts,[1,2,3])
        for count,option in zip(counts,result['candidates']):
            self.assertEqual(len(option['changed']),count-1)
            self.assertTrue(all(option['roles'][n]=='prefill' for n in option['changed']))
            self.assertEqual({n for n in self.roles if option['roles'][n]!=self.roles[n]},
                             set(option['changed']))
    def test_model_chooses_which_nodes_change(self):
        # P-bound at 2P: the node with the faster P model becomes the second P.
        self.workers['3']=copy.deepcopy(self.workers['3'])
        self.workers['3']['prefill']['models'][0]['coefficients']=[.005,.00005,5e-10,.000005]
        result=decide(self.workers,self.roles,self.work,minutes=self.minutes,experts=256,top_k=8)
        self.assertEqual([o['changed'] for o in result['candidates']],[[],['3'],['1','3']])
    def test_fractional_population_does_not_invent_scaling_gain(self):
        workers={str(i):copy.deepcopy(self.workers['0']) for i in range(9)}
        for worker in workers.values():
            worker['decode']['models'][0]['coefficients']=[.1,0.,0.]
        seven={str(i):'prefill' if i<2 else 'decode' for i in range(9)}
        eight={str(i):'prefill' if i<1 else 'decode' for i in range(9)}
        a=capacities(workers,seven,self.work,experts=256,top_k=8,decode_population=13)
        b=capacities(workers,eight,self.work,experts=256,top_k=8,decode_population=13)
        self.assertAlmostEqual(a['decode_rr_rps'],b['decode_rr_rps'])
    def nine_nodes(self, rates):
        """4P5D. Each P serves 1 rps; each D serves that minute's rate."""
        roles={str(i):'prefill' if i<4 else 'decode' for i in range(9)}
        def estimate(workers, candidate, work, **kwargs):
            p=sum(r=='prefill' for r in candidate.values())
            d=(len(candidate)-p)*work['d']
            return dict(system_rps=min(p*1.,d),prefill_rps=p*1.,decode_rr_rps=d)
        minutes=[dict(work=dict(d=d),decode_population=None) for d in rates]
        with patch('evolving_infra.planner.policy.split_policy.capacities',side_effect=estimate):
            return decide({n:{} for n in roles},roles,dict(d=fmean(rates)),minutes=minutes,
                          experts=256,top_k=8)

    def test_same_interval_gain_switches_only_when_minutes_agree(self):
        steady=self.nine_nodes([1.12,1.11,1.13,1.12,1.10,1.14,1.12,1.11,1.13,1.12])
        mixed=self.nine_nodes([1.25,.82,1.25,1.25,.81,1.25,1.25,.82,1.25,1.25])
        for result in (steady,mixed):
            # Whole interval: 4 rps now, 4.48 rps with one more P, a 12% gain.
            change=next(o for o in result['candidates'] if o['changed']==['4'])
            self.assertAlmostEqual(change['estimate']['system_rps'],4.48)
            self.assertAlmostEqual(change['minute_gain']['mean_relative'],.12)
        self.assertEqual(steady['action'],'switch')
        self.assertEqual(steady['selected']['changed'],['4'])
        self.assertEqual(steady['required_standard_errors'],2)
        self.assertAlmostEqual(steady['selected']['minute_gain']['standard_error_rps'],
                               4*stdev([.12,.11,.13,.12,.10,.14,.12,.11,.13,.12])/math.sqrt(10))
        self.assertEqual(mixed['action'],'hold')
        change=next(o for o in mixed['candidates'] if o['changed']==['4'])
        self.assertAlmostEqual(change['minute_gain']['standard_error_rps'],.2648,places=4)
        self.assertFalse(change['minute_gain']['eligible'])

    def test_consistent_small_gain_switches_and_equal_capacity_holds(self):
        # 6% in every minute; no fixed margin blocks a consistent gain.
        small=self.nine_nodes([1.05,1.06,1.07,1.06,1.05,1.07,1.06,1.06,1.05,1.07])
        self.assertEqual(small['action'],'switch')
        self.assertEqual(self.nine_nodes([1.]*10)['action'],'hold')

    def test_one_decision_may_change_several_nodes(self):
        # Each D serves 2 rps: 6P3D gives 6 rps, 5P4D gives 5, 4P5D now gives 4.
        result=self.nine_nodes([2.1,2.,2.2,2.1,2.,2.2,2.1,2.,2.2,2.1])
        self.assertEqual(result['action'],'switch')
        self.assertEqual(result['selected']['changed'],['4','5'])
        self.assertEqual(sum(r=='prefill' for r in result['selected']['roles'].values()),6)
        self.assertEqual(len(result['candidates']),8)

    def test_switch_selects_largest_mean_among_consistent_changes(self):
        # 2P has the largest mean, but its minutes disagree.
        gains={1:[0.]*4,2:[5.,-2.,5.,-2.],3:[1.,1.2,.8,1.]}
        def estimate(workers, roles, work, **kwargs):
            rps=10.+gains[sum(r=='prefill' for r in roles.values())][work['minute']]
            return dict(system_rps=rps,prefill_rps=rps,decode_rr_rps=rps)
        minutes=[dict(work=dict(minute=i),decode_population=None) for i in range(4)]
        with patch('evolving_infra.planner.policy.split_policy.capacities',side_effect=estimate):
            result=decide(self.workers,self.roles,dict(minute=0),minutes=minutes,experts=256,top_k=8)
        self.assertEqual(result['action'],'switch')
        self.assertEqual(result['selected']['changed'],['1','2'])
        self.assertFalse(result['candidates'][1]['minute_gain']['eligible'])

    def test_equal_gains_prefer_fewer_changes(self):
        # 2P3D now. 4P (two changes) is listed before 1P (one change); both gain 2 rps.
        roles={str(i):'prefill' if i<2 else 'decode' for i in range(5)}
        def estimate(workers, candidate, work, **kwargs):
            rps={1:12.,2:10.,3:11.,4:12.}[sum(r=='prefill' for r in candidate.values())]
            return dict(system_rps=rps,prefill_rps=rps,decode_rr_rps=rps)
        minutes=[dict(work=dict(minute=i),decode_population=None) for i in range(3)]
        with patch('evolving_infra.planner.policy.split_policy.capacities',side_effect=estimate):
            result=decide({n:{} for n in roles},roles,{},minutes=minutes,experts=256,top_k=8)
        self.assertEqual([o['changed'] for o in result['candidates']],
                         [[],['2'],['2','3'],['0']])
        self.assertEqual(result['selected']['changed'],['0'])

    def test_unestimable_node_is_passed_over(self):
        def estimate(workers, roles, work, **kwargs):
            if 'prefill' in (roles['1'],roles['3']):
                raise ValueError('Missing measured model or declared role prior')
            rps=10. if roles==self.roles else 12.
            return dict(system_rps=rps,prefill_rps=rps,decode_rr_rps=rps)
        minutes=[dict(work=dict(minute=i),decode_population=None) for i in range(3)]
        with patch('evolving_infra.planner.policy.split_policy.capacities',side_effect=estimate):
            result=decide(self.workers,self.roles,{},minutes=minutes,experts=256,top_k=8)
        self.assertEqual([o['changed'] for o in result['candidates']],[[],['2'],['2']])
        self.assertIn('error',result['candidates'][2])
        self.assertEqual(result['selected']['changed'],['2'])

    def test_failed_slice_excludes_a_change_but_not_the_current_roles(self):
        minutes=[dict(work=dict(minute=i),decode_population=None) for i in range(3)]
        for failing in ('2',None):
            def estimate(workers, roles, work, **kwargs):
                changed={n for n in roles if roles[n]!=self.roles[n]}
                if work.get('minute')==1 and (failing in changed if failing else not changed):
                    raise ValueError('Mean request cannot fit worker KV capacity')
                rps=12. if changed else 10.
                return dict(system_rps=rps,prefill_rps=rps,decode_rr_rps=rps)
            with self.subTest(failing=failing), patch('evolving_infra.planner.policy.split_policy.capacities',side_effect=estimate):
                if failing is None:
                    with self.assertRaises(ValueError):
                        decide(self.workers,self.roles,{},minutes=minutes,experts=256,top_k=8)
                    continue
                result=decide(self.workers,self.roles,{},minutes=minutes,experts=256,top_k=8)
            self.assertEqual(result['selected']['changed'],['1'])
            self.assertIn('error',result['candidates'][2])

    def test_one_slice_cannot_measure_its_error(self):
        with self.assertRaisesRegex(ValueError,'two workload slices'):
            decide(self.workers,self.roles,self.work,minutes=self.minutes[:1],experts=256,top_k=8)

    def test_decide_does_not_mutate_inputs(self):
        before=copy.deepcopy((self.workers,self.roles,self.work,self.minutes))
        decide(self.workers,self.roles,self.work,minutes=self.minutes,experts=256,top_k=8,
               decode_population=30)
        self.assertEqual((self.workers,self.roles,self.work,self.minutes),before)

    def test_work_uses_completed_causal_pairs_not_pair_of_averages(self):
        previous={k:0 for k in ('requests','input_tokens','uncached_tokens','causal_attention_pairs','output_tokens','verified_output_tokens','verify_steps')}
        current=dict(requests=2,input_tokens=20,uncached_tokens=8,causal_attention_pairs=76.,output_tokens=24,verified_output_tokens=24,verify_steps=8)
        result=workload_delta(previous,current)
        self.assertEqual(result['pairs'],38)
        self.assertEqual(result['output'],12)
        self.assertEqual(result['acceptance'],3)
        with self.assertRaises(ValueError):workload_delta(current,previous)
    def test_wrong_membership_and_unfit_memory_fail(self):
        with self.assertRaises(ValueError):capacities(self.workers,{'0':'prefill'},self.work,experts=256,top_k=8)
        self.workers['1']['kv_capacity_tokens']=1
        with self.assertRaises(ValueError):capacities(self.workers,self.roles,self.work,experts=256,top_k=8)

    def test_shared_worker_terms_equal_direct_estimates(self):
        # decide evaluates a worker once per slice, role, D count and P observation, for every role map; each recorded
        # estimate equals a direct capacities() call.
        fit = measured_workers()
        for i, worker in enumerate(fit.values()):
            worker['prefill'] = dict(origin='measured', models=[dict(PREFILL_MODEL, coefficients=[.01*(1+i/8), .0001, 1e-9, 1e-5])])
            worker['decode'] = dict(origin='measured', models=[dict(DECODE_MODEL, coefficients=[.02, .001*(1+i/4), .01])])
        onboarding = {'0': dict(restored_per_cached_token=.1, cached_tokens=100.),
                      '1': dict(restored_per_cached_token=.9, cached_tokens=900.)}
        roles = {'0': 'prefill', '1': 'prefill', '2': 'decode', '3': 'decode'}
        result = decide(fit, roles, WORK, minutes=MINUTES, experts=256, top_k=8, decode_population=30,
                        onboarding=onboarding)

        def direct(roles, work, population):
            return capacities(fit, roles, work, experts=256, top_k=8, decode_population=population, onboarding=onboarding)
        for option in result['candidates']:
            self.assertEqual(option['estimate'], direct(option['roles'], WORK, 30))
            self.assertEqual(option['minute_rps'],
                             [direct(option['roles'], m['work'], m['decode_population'])['system_rps'] for m in MINUTES])


if __name__=='__main__':unittest.main()
