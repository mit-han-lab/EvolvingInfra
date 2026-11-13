# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
import contextlib
import copy
import http.server
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from evolving_infra.planner.managers.split_controller import SplitController, node_assignment
from evolving_infra.planner.policy.workload import minute_slices, population_mean
from evolving_infra.planner.router_client import RoleClient, cas_update
from evolving_infra.planner.managers.transition_guard import transition_residual
from evolving_infra.planner.engine_client import served_experts
from planner_support import WINDOW_WORKLOAD, measured_workers, router_samples, router_snapshot


def snapshot(revision=0):
    return router_snapshot({'http://a':'prefill','http://b':'decode','http://c':'decode'},{'http://b':[3,5,0]},
                           epoch='one-router-process',revision=revision)


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
        self.controller=SplitController(dict(workers={n:{} for n in self.urls}),self.urls,
            self.initial,experts=256,top_k=8)
        self.decision=dict(action='switch',before=self.initial['assignment'],
            selected=dict(roles={'node0':'prefill','node1':'prefill','node2':'decode'}))
        self.after=copy.deepcopy(self.initial['assignment']);self.after['revision']=1
        self.after['roles']['http://b']='prefill'
    def test_live_controller_decides_from_minute_slices(self):
        controller = SplitController(dict(workers=measured_workers(self.urls)), self.urls, self.initial,
                                experts=256, top_k=8)
        end = dict(snapshot(), completed_workload=WINDOW_WORKLOAD)
        decision = controller.propose(router_samples(self.initial, end))
        self.assertEqual((decision['rule'], decision['required_standard_errors']),
                         ('minute_standard_error', 2))
        self.assertEqual(decision['slices'], 10)
        self.assertEqual([m['work']['requests'] for m in decision['minutes']], [10]*10)
        self.assertEqual(decision['workload']['requests'], 100)
        self.assertEqual({m['decode_population'] for m in decision['minutes']}, {8.})
        later = snapshot(); later['completed_workload']['requests'] = 1
        with self.assertRaisesRegex(ValueError, 'counter boundary'):
            controller.propose(router_samples(later, end))
        samples = router_samples(self.initial, end)
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
        with patch('evolving_infra.planner.managers.split_controller.decide', return_value=dict(candidates=[])):
            with self.assertRaisesRegex(ValueError, 'prediction'):
                self.controller.propose(router_samples(current, current))

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
        with patch('evolving_infra.planner.managers.split_controller.decide', return_value=dict(candidates=[])):
            decision = self.controller.propose(router_samples(current, end, start=220., seconds=120))
        self.assertEqual(decision['workload']['requests'], 10)
        self.assertEqual((decision['workload']['uncached']+decision['workload']['cached']), 20.)
        self.assertEqual(decision['workload']['output'], 3.)
        self.assertEqual(current['completed_workload']['requests'], 90)
        self.assertEqual(end['completed_workload']['requests'], 100)
        self.assertEqual(self.log[-1]['event'], 'prediction_sampling_resumed')

    def test_configured_exclusion_sets_the_resume_boundary(self):
        self.controller = SplitController(dict(workers={n: {} for n in self.urls}), self.urls,
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
        samples = router_samples(self.initial, self.initial)
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
        self.controller = SplitController(dict(workers={n: {} for n in self.urls}), self.urls,
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
            SplitController(dict(workers={n: {} for n in self.urls}), self.urls, self.initial,
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
        with self.assertRaises(ValueError):self.controller.propose(router_samples(changed,changed))
        changed=snapshot();changed['assignment']['roles']['http://new']='decode'
        with self.assertRaises(ValueError):node_assignment(changed,self.urls)
    def test_population_includes_pending_and_weights_time(self):
        a=snapshot();b=snapshot();b['decode_population']['workers']['http://b']=[0,20,0]
        rows=[dict(at=0,snapshot=a),dict(at=2,snapshot=b),dict(at=5,snapshot=b)]
        result=population_mean(rows)
        self.assertAlmostEqual(result['potential'],15.2)
        self.assertAlmostEqual(result['p_completed'],1.2)
    def test_unknown_or_gap_cannot_be_zero_load(self):
        a=snapshot();a['decode_population']['workers']['http://b'][2]=1
        with self.assertRaises(ValueError):population_mean([dict(at=0,snapshot=a),dict(at=3,snapshot=a)])
        with self.assertRaises(ValueError):population_mean([dict(at=0,snapshot=snapshot()),dict(at=3,snapshot=a)])
        a=snapshot()
        with self.assertRaises(ValueError):population_mean([dict(at=0,snapshot=a),dict(at=20,snapshot=a)])
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
            model=Path(td)/'vision';model.mkdir()  # a multimodal model: the counts are under text_config
            (model/'config.json').write_text(json.dumps(dict(text_config=dict(n_routed_experts=384,num_experts_per_tok=8))))
            paths['/vision/model_info']=str(model)
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
                self.assertEqual(served_experts([url+'vision']),(384,8))
                with self.assertRaisesRegex(ValueError,'one expert configuration'):
                    served_experts([url+'a',url+'c'])
            finally:server.shutdown()


class RoleClientTests(unittest.TestCase):
    def test_lost_put_reply_is_journaled_and_confirmed_by_one_get(self):
        state=dict(assignment=snapshot()['assignment'],puts=0)
        class Router(http.server.BaseHTTPRequestHandler):
            def do_GET(self):self.reply(200,json.dumps(dict(assignment=state['assignment'])))
            def do_PUT(self):  # applied, but the reply is an error
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                state['puts']+=1;state['assignment']=dict(epoch=body['epoch'],revision=1,roles=body['desired'])
                self.reply(500,'lost reply')
            def reply(self,code,text):
                self.send_response(code);self.send_header('Content-Length',str(len(text)));self.end_headers()
                self.wfile.write(text.encode())
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Router)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        log=[];desired={'http://a':'prefill','http://b':'prefill','http://c':'decode'}
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                response=cas_update(RoleClient(f'http://127.0.0.1:{server.server_port}','t',log.append),
                    snapshot()['assignment'],desired,log.append,lambda body:dict(body=body),'unconfirmed')
        finally:server.shutdown()
        self.assertEqual((response['assignment']['roles'],state['puts']),(desired,1))
        self.assertEqual([e['event'] for e in log],
            ['role_update_intent','role_request_http_error','role_update_response_unknown'])
        self.assertEqual({k:log[1][k] for k in ('method','status','response_body')},
            dict(method='PUT',status=500,response_body='lost reply'))


if __name__=='__main__':unittest.main()


def test_worker_capacities_use_the_launch_cap_else_the_engine_computed_one():
    from evolving_infra.planner import engine_client as api
    replies = {
        'http://flag': dict(max_total_num_tokens=1000, max_running_requests=256, tp_size=4,
                            internal_states=[dict(effective_max_running_requests_per_dp=200)]),
        'http://auto': dict(max_total_num_tokens=2000, max_running_requests=None, tp_size=4,
                            internal_states=[dict(effective_max_running_requests_per_dp=1024)] * 2),
    }
    original = api.http_json
    api.http_json = lambda url, timeout=10, headers=None: replies[url.removesuffix('/server_info')]
    try:
        caps = api.worker_capacities([dict(node='node0', url='http://flag'), dict(node='node1', url='http://auto')])
    finally:
        api.http_json = original
    assert caps == {'node0': dict(kv_capacity_tokens=1000, max_running_requests=256, tp_size=4),
                    'node1': dict(kv_capacity_tokens=2000, max_running_requests=2048, tp_size=4)}
