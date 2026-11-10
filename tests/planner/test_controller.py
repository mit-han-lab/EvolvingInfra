import copy
import http.server
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from evolving_infra.planner.api import served_experts
from evolving_infra.planner.control.controller import Controller, node_assignment
from evolving_infra.planner.control.transition import transition_residual
from evolving_infra.planner.decide.workload import minute_slices, population_mean


def snapshot(revision=0):
    return dict(assignment=dict(epoch='one-router-process',revision=revision,
        roles={'http://a':'prefill','http://b':'decode','http://c':'decode'}),
        completed_workload=dict(requests=0,input_tokens=0,uncached_tokens=0,
            causal_attention_pairs=0,output_tokens=0,verified_output_tokens=0,verify_steps=0),
        decode_population=dict(fields=['p_completed','p_pending','unknown'],
            workers={'http://b':[3,5,0]}))


def rows(first, last, start=100., seconds=600, step=3):
    """Router observations every `step` seconds; completed counters grow linearly."""
    a,b=first['completed_workload'],last['completed_workload']
    out=[]
    for i in range(seconds//step+1):
        s=copy.deepcopy(last)
        s['completed_workload']={k:a[k]+(b[k]-a[k])*i*step//seconds for k in b}
        out.append(dict(at=start+i*step,snapshot=s))
    return out


class Client:
    def __init__(self, assignment, ambiguous=False, draining=None):
        self.assignment=assignment;self.calls=[];self.ambiguous=ambiguous
        self.draining = [] if draining is None else draining
    def request(self, body=None):
        self.calls.append(body)
        if body is not None and self.ambiguous:raise TimeoutError()
        return dict(assignment=self.assignment, draining=self.draining)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.initial=snapshot();self.urls={'node0':'http://a','node1':'http://b','node2':'http://c'}
        self.controller=Controller(dict(workers={n:{} for n in self.urls}),self.urls,
            self.initial,experts=256,top_k=8)
        self.decision=dict(action='switch',before=self.initial['assignment'],
            selected=dict(roles={'node0':'prefill','node1':'prefill','node2':'decode'}))
        self.after=copy.deepcopy(self.initial['assignment']);self.after['revision']=1
        self.after['roles']['http://b']='prefill'
    def test_live_controller_decides_from_minute_slices(self):
        p = dict(coefficients=[.01, .0001, 1e-9, .00001],
                 tokens_per_step=4096., restored_per_cached_token=.5)
        d = dict(coefficients=[.02, .001, .01], draft_tokens=6,
                 batch_support=list(range(1, 129)))
        workers = {node: dict(prefill=dict(origin='measured', models=[p]),
                              decode=dict(origin='measured', models=[d]),
                              max_running_requests=128, kv_capacity_tokens=2000000)
                   for node in self.urls}
        controller = Controller(dict(workers=workers), self.urls, self.initial,
                                experts=256, top_k=8)
        end = snapshot()
        end['completed_workload'] = dict(requests=100, input_tokens=1200,
            uncached_tokens=500, causal_attention_pairs=5000, output_tokens=400,
            verified_output_tokens=400, verify_steps=100)
        decision = controller.propose(rows(self.initial, end))
        self.assertEqual((decision['rule'], decision['required_standard_errors']),
                         ('minute_standard_error', 2))
        self.assertEqual(decision['slices'], 10)
        self.assertEqual([m['work']['requests'] for m in decision['minutes']], [10]*10)
        self.assertEqual(decision['workload']['requests'], 100)
        self.assertEqual({m['decode_population'] for m in decision['minutes']}, {8.})
        later = snapshot(); later['completed_workload']['requests'] = 1
        with self.assertRaisesRegex(ValueError, 'counter boundary'):
            controller.propose(rows(later, end))
        samples = rows(self.initial, end)
        for row in samples[21:41]:  # Nothing completes during the second minute.
            row['snapshot']['completed_workload'] = dict(samples[20]['snapshot']['completed_workload'])
        with self.assertRaisesRegex(ValueError, 'Workload slice 1'):
            controller.propose(samples)

    def test_slices_share_observed_boundaries(self):
        samples = [dict(at=10.+3*i) for i in range(201)]
        parts = minute_slices(samples)
        self.assertEqual([p[-1]['at']-p[0]['at'] for p in parts], [60.]*10)
        for a, b in zip(parts, parts[1:]):
            self.assertIs(a[-1], b[0])
        self.assertEqual(len(minute_slices(samples[:200])), 9)
        with self.assertRaisesRegex(ValueError, 'two workload slices'):
            minute_slices(samples[:40])

    def _confirmed_switch(self):
        self.now = 100.
        self.controller.clock = lambda: self.now
        self.log = []
        self.controller.apply(self.initial, self.decision, Client(self.after), self.log.append)
        current = snapshot()
        current['assignment'] = copy.deepcopy(self.after)
        return current

    def test_prediction_refuses_transition_data_even_without_observation_call(self):
        current = self._confirmed_switch()
        current['completed_workload'].update(requests=1, input_tokens=12,
            uncached_tokens=5, causal_attention_pairs=50, output_tokens=4)
        with patch('evolving_infra.planner.control.controller.decide', return_value=dict(candidates=[])):
            with self.assertRaisesRegex(ValueError, 'prediction'):
                self.controller.propose(rows(current, current))

    def test_transition_interval_resets_both_counters_and_population_boundary(self):
        current = self._confirmed_switch()
        current['completed_workload'].update(requests=90, input_tokens=900,
            uncached_tokens=450, causal_attention_pairs=9000, output_tokens=180)
        self.assertEqual(self.controller.prediction_observation(current, 219.999, self.log.append),
                         'excluded')
        self.assertEqual(self.controller.prediction_observation(current, 220., self.log.append),
                         'reset')
        end = copy.deepcopy(current)
        end['completed_workload'].update(requests=100, input_tokens=1100,
            uncached_tokens=500, causal_attention_pairs=10000, output_tokens=210)
        self.assertEqual(self.controller.prediction_observation(end, 223., self.log.append),
                         'included')
        with patch('evolving_infra.planner.control.controller.decide', return_value=dict(candidates=[])):
            decision = self.controller.propose(rows(current, end, start=220., seconds=120))
        self.assertEqual(decision['workload']['requests'], 10)
        self.assertEqual((decision['workload']['uncached']+decision['workload']['cached']), 20.)
        self.assertEqual(decision['workload']['output'], 3.)
        self.assertEqual(current['completed_workload']['requests'], 90)
        self.assertEqual(end['completed_workload']['requests'], 100)
        self.assertEqual(self.log[-1]['event'], 'prediction_sampling_resumed')

    def test_configured_exclusion_sets_the_resume_boundary(self):
        self.controller = Controller(dict(workers={n: {} for n in self.urls}), self.urls,
                                     self.initial, experts=256, top_k=8, exclusion_s=60)
        current = self._confirmed_switch()
        self.assertEqual(self.log[-1]['prediction_exclusion_s'], 60)
        self.assertEqual(self.controller.prediction_observation(current, 159.999, self.log.append),
                         'excluded')
        self.assertEqual(self.controller.prediction_observation(current, 160., self.log.append),
                         'reset')

    def test_prediction_waits_for_cleanup_after_requests_are_terminal(self):
        current = self._confirmed_switch()
        current['decode_population']['workers'] = {}
        current['draining'] = {'http://b': 200.}
        self.assertEqual(self.controller.prediction_observation(current, 300., self.log.append),
                         'excluded')
        current['draining'] = {}
        current['completed_workload']['requests'] = 80
        self.assertEqual(self.controller.prediction_observation(current, 303., self.log.append),
                         'excluded')
        self.assertEqual(self.controller.prediction_observation(current, 422.999, self.log.append),
                         'excluded')
        current['completed_workload']['requests'] = 100
        self.assertEqual(self.controller.prediction_observation(current, 423., self.log.append),
                         'reset')
        self.assertEqual(self.controller.previous['requests'], 100)

    def test_put_drain_signal_survives_until_first_completed_snapshot(self):
        self.controller.clock = lambda: 100.
        self.controller.apply(self.initial, self.decision,
            Client(self.after, draining=['http://b']), lambda _: None)
        current = snapshot(); current['assignment'] = self.after; current['draining'] = {}
        self.assertEqual(self.controller.prediction_observation(current, 300., lambda _: None),
                         'excluded')
        self.assertEqual(self.controller.prediction_resume_at, 420.)

    def test_unobserved_drain_cannot_enter_a_decision_window(self):
        samples = rows(self.initial, self.initial)
        samples[10]['snapshot']['draining'] = {'http://b': 20.}
        with self.assertRaisesRegex(ValueError, 'prediction'):
            self.controller.propose(samples)
        current = copy.deepcopy(self.initial); current['draining'] = {'http://b': 20.}
        client = Client(self.after)
        with self.assertRaisesRegex(ValueError, 'draining'):
            self.controller.apply(current, self.decision, client, lambda _: None)
        self.assertEqual(client.calls, [])

    def test_controller_started_during_drain_excludes_onboarding_and_sampling(self):
        current = copy.deepcopy(self.initial); current['draining'] = {'http://b': 20.}
        self.controller = Controller(dict(workers={n: {} for n in self.urls}), self.urls,
                                     current, experts=256, top_k=8)
        with self.assertRaisesRegex(ValueError, 'transitional'):
            self.controller.update_onboarding(current, dict(assignment=self.controller.assignment),
                                              lambda _: None)
        self.assertEqual(self.controller.prediction_observation(current, 100., lambda _: None),
                         'excluded')
        current['draining'] = {}
        self.assertEqual(self.controller.prediction_observation(current, 103., lambda _: None),
                         'excluded')
        self.assertEqual(self.controller.prediction_observation(current, 223., lambda _: None),
                         'reset')

    def test_exclusion_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'exclusion'):
            Controller(dict(workers={n: {} for n in self.urls}), self.urls, self.initial,
                       experts=256, top_k=8, exclusion_s=0)

    def test_late_first_observation_never_interpolates_a_boundary(self):
        current = self._confirmed_switch()
        current['completed_workload']['requests'] = 100
        self.assertEqual(self.controller.prediction_observation(current, 231., self.log.append),
                         'reset')
        self.assertEqual(self.controller.previous['requests'], 100)
        self.assertEqual(self.log[-1]['actual_resume_monotonic'], 231.)

    def test_hold_and_failed_put_do_not_begin_exclusion(self):
        hold = dict(action='hold', before=self.initial['assignment'])
        self.controller.apply(self.initial, hold, Client(self.after), lambda _: None)
        self.assertEqual(self.controller.prediction_observation(self.initial, 100., lambda _: None),
                         'included')
        with self.assertRaises(RuntimeError):
            self.controller.apply(self.initial, self.decision,
                Client(self.initial['assignment'], ambiguous=True), lambda _: None)
        self.assertEqual(self.controller.prediction_observation(self.initial, 101., lambda _: None),
                         'included')

    def test_clock_starts_after_confirmation_not_before_slow_put(self):
        self.now = 100.
        self.controller.clock = lambda: self.now
        outer = self
        class SlowClient(Client):
            def request(self, body=None):
                outer.now = 107.
                return super().request(body)
        self.controller.apply(self.initial, self.decision, SlowClient(self.after), lambda _: None)
        current = snapshot(); current['assignment'] = self.after
        self.assertEqual(self.controller.prediction_observation(current, 220., lambda _: None),
                         'excluded')
        self.assertEqual(self.controller.prediction_observation(current, 227., lambda _: None),
                         'reset')

    def test_revision_or_clock_error_still_fails_during_exclusion(self):
        current = self._confirmed_switch()
        for at in (float('nan'), float('inf'), 99.):
            with self.subTest(at=at), self.assertRaises(ValueError):
                self.controller.prediction_observation(current, at, self.log.append)
        current['assignment']['revision'] += 1
        with self.assertRaises(ValueError):
            self.controller.prediction_observation(current, 150., self.log.append)

    def test_ambiguous_put_is_confirmed_by_get_not_repeated(self):
        client=Client(self.after,ambiguous=True);log=[]
        self.assertEqual(self.controller.apply(self.initial,self.decision,client,log.append),self.after)
        self.assertEqual(sum(call is not None for call in client.calls),1)
        self.assertIsNone(client.calls[1]);self.assertEqual(log[0]['event'],'role_update_intent')
    def test_old_assignment_after_timeout_does_not_prove_no_future_update(self):
        client=Client(self.initial['assignment'],ambiguous=True)
        with self.assertRaises(RuntimeError):
            self.controller.apply(self.initial,self.decision,client,lambda _:None)
        self.assertEqual(len(client.calls),2)
        self.assertEqual(self.controller.assignment,self.initial['assignment'])
    def test_router_restart_or_membership_change_is_not_accepted(self):
        changed=snapshot();changed['assignment']['epoch']='other-process'
        with self.assertRaises(ValueError):self.controller.propose(rows(changed,changed))
        changed=snapshot();changed['assignment']['roles']['http://new']='decode'
        with self.assertRaises(ValueError):node_assignment(changed,self.urls)
    def test_population_includes_pending_and_weights_time(self):
        a=snapshot();b=snapshot();b['decode_population']['workers']['http://b']=[0,20,0]
        rows=[dict(at=0,snapshot=a),dict(at=2,snapshot=b),dict(at=5,snapshot=b)]
        result=population_mean(rows,0,5)
        self.assertAlmostEqual(result['potential'],15.2)
        self.assertAlmostEqual(result['p_completed'],1.2)
    def test_unknown_gap_or_incomplete_coverage_cannot_be_zero_load(self):
        a=snapshot();a['decode_population']['workers']['http://b'][2]=1
        with self.assertRaises(ValueError):population_mean([dict(at=0,snapshot=a),dict(at=3,snapshot=a)],0,3)
        with self.assertRaises(ValueError):population_mean([dict(at=0,snapshot=snapshot()),dict(at=3,snapshot=a)],0,3)
        a=snapshot()
        with self.assertRaises(ValueError):population_mean([dict(at=0,snapshot=a),dict(at=20,snapshot=a)],0,20)
        with self.assertRaises(ValueError):population_mean([dict(at=1,snapshot=a),dict(at=3,snapshot=a)],0,3)
    def test_role_change_does_not_move_existing_requests(self):
        s=snapshot();s['assignment']['roles']['http://b']='prefill'
        loads={'workers':{u:dict(active_prefill_reqs=0) for u in self.urls.values()}}
        loads['workers']['http://c']['active_prefill_reqs']=2
        residual=transition_residual(s,loads)
        self.assertEqual(residual['http://b']['old_role_requests'],8)
        self.assertEqual(residual['http://c']['old_role_requests'],2)
    def test_drain_residual_remains_when_only_session_cleanup_is_pending(self):
        s=snapshot();s['assignment']['roles']['http://b']='prefill'
        s['decode_population']['workers'] = {}; s['draining'] = {'http://b': 30.}
        loads={'workers':{u:dict(active_prefill_reqs=0) for u in self.urls.values()}}
        self.assertEqual(transition_residual(s, loads), {'http://b': dict(
            assigned_role='prefill', old_role_requests=0, draining_age_s=30.)})
    def test_skipped_window_counter_boundary_moves_with_population_boundary(self):
        s=snapshot();s['completed_workload']['requests']=3
        self.controller.skip(s,'unknown population',lambda _:None)
        self.assertEqual(self.controller.previous['requests'],3)


class ServedExpertsTests(unittest.TestCase):
    def test_counts_come_from_every_engine_model_config(self):
        with tempfile.TemporaryDirectory() as td:
            paths={}
            for name,experts,top_k in (('a',256,8),('b',256,8),('c',256,6)):
                model=Path(td)/name;model.mkdir()
                (model/'config.json').write_text(json.dumps(dict(n_routed_experts=experts,num_experts_per_tok=top_k)))
                paths[f'/{name}/model_info']=str(model)
            class Engines(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200);self.end_headers()
                    self.wfile.write(json.dumps(dict(model_path=paths[self.path])).encode())
                def log_message(self,*args):pass
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Engines)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            url=f'http://127.0.0.1:{server.server_port}/'
            try:
                self.assertEqual(served_experts([url+'a',url+'b']),(256,8))
                with self.assertRaisesRegex(ValueError,'one expert configuration'):
                    served_experts([url+'a',url+'c'])
            finally:server.shutdown()


if __name__=='__main__':unittest.main()
