import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from evolving_infra.planner.calibration.live import prefill_windows

class LiveWindowsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.telemetry=Path(self.tmp.name)/'metrics.gz'
        self.rows=[];self.requests=[]
        self.source='/workers/node0/generation0-prefill/scheduler/trace'
        for i,t in enumerate(range(0,306,5)):
            self.rows.append(dict(source=self.source,mode='extend',launch_bs=1,
                ts=t+1.,dur_ms=1000.,forward_iter=i,after_idle=True,
                new_prefill_tokens=10,prefill_attention_pairs=55))
            self.requests.append(dict(rid=str(i),prefill_iters=[i],cached_tokens=0))
        with gzip.open(self.telemetry,'wt') as f:
            for t in range(-4,313,2):
                metrics='# HELP sglang:hicache_backup_bytes_total backup\n'+f'sglang:load_back_tokens_total{{pool="kv"}} {4*(t+4)}\n'
                f.write(json.dumps(dict(at=t+.01,nodes=[dict(node=0,role='prefill',at=t,metrics=metrics)]))+'\n')
    def tearDown(self):self.tmp.cleanup()
    def run_windows(self,rows=None,requests=None):
        return prefill_windows(self.rows if rows is None else rows,
            {'node0':self.requests if requests is None else requests}, {self.source:'node0'},
            self.telemetry,replicas=4,start=0,end=300,clock_pad_s=.01)
    def test_live_tail_and_generation_identity_are_explicit(self):
        windows,coverage=self.run_windows()
        self.assertEqual({w['node'] for w in windows},{'node0'})
        self.assertEqual(len(windows),4)
        self.assertEqual(coverage['node0']['open_steps'],1)
        self.assertTrue(all(w['restore_lo']<=w['features'][3]<=w['restore_hi'] for w in windows))
    def test_post_deadline_closure_cannot_complete_last_window(self):
        windows,_=self.run_windows()
        self.assertNotIn(240,[w['start'] for w in windows])
        future=self.rows+[dict(self.rows[-1],ts=601.,forward_iter=999)]
        other,_=self.run_windows(rows=future)
        self.assertEqual(windows,other)

    def test_unfinished_request_drops_entire_affected_window(self):
        requests=[r for r in self.requests if r['rid']!='14'] # 70s
        windows,coverage=self.run_windows(requests=requests)
        self.assertNotIn(60,[w['start'] for w in windows])
        self.assertGreater(coverage['node0']['invalid_episodes_or_gaps'],0)
    def test_missing_iteration_is_not_counted_as_idle(self):
        rows=[r for r in self.rows if r['forward_iter']!=26] # 130s
        windows,_=self.run_windows(rows=rows)
        self.assertNotIn(120,[w['start'] for w in windows])
    def test_missing_step_unknown_tail_extends_until_idle(self):
        rows=[dict(r) for r in self.rows if r['forward_iter']!=20]
        for row in rows:
            if 100 < row['ts'] < 211:
                row['after_idle']=False
        windows,_=self.run_windows(rows=rows)
        self.assertNotIn(180,[w['start'] for w in windows], 'unknown 180..210 must not be called idle')

    def test_duplicate_stream_generation_fails(self):
        with self.assertRaises(ValueError):self.run_windows(rows=self.rows+[copy.deepcopy(self.rows[0])])

    def delayed_counter_observations(self):
        with gzip.open(self.telemetry,'rt') as stream:
            records=[json.loads(line) for line in stream]
        # Point timestamps bracket 120s within 8s, but the complete HTTP
        # collection horizon is 11.12s. Both adjacent minutes are unusable.
        records=[r for r in records if r['nodes'][0]['at'] not in (116,118,120,124)]
        for record in records:
            if record['nodes'][0]['at']==114:record['at']=118.9
            if record['nodes'][0]['at']==122:record['at']=125.1
        with gzip.open(self.telemetry,'wt') as stream:
            for record in records:stream.write(json.dumps(record)+'\n')

    def test_full_collection_gap_excludes_only_affected_minutes(self):
        self.delayed_counter_observations()
        windows,coverage=self.run_windows()
        self.assertEqual([w['start'] for w in windows],[0,180])
        audit=coverage['node0']
        self.assertEqual(audit['restore_counter_excluded_windows'],2)
        self.assertEqual(audit['windows'],2)
        self.assertEqual(audit['eligible_steps'],sum(w['steps'] for w in windows))
        self.assertEqual([w['start'] for w in audit['restore_counter_exclusions']],[60,120])
        self.assertTrue(all(w['restore_lo']<=w['features'][3]<=w['restore_hi'] for w in windows))

    def test_strict_collection_audit_keeps_transactional_rejection(self):
        from evolving_infra.planner.calibration.restore_audit import include_scrape_collection_time
        self.delayed_counter_observations()
        windows=[dict(node='node0',start=0,end=60),dict(node='node0',start=60,end=120)]
        original=copy.deepcopy(windows)
        with self.assertRaisesRegex(ValueError,'bracket gap >=10s'):
            include_scrape_collection_time(windows,self.telemetry,4,.01)
        self.assertEqual(windows,original)

    def test_counter_reset_still_fails_instead_of_excluding_windows(self):
        with gzip.open(self.telemetry,'rt') as stream:records=[json.loads(line) for line in stream]
        records[30]['nodes'][0]['metrics']='# HELP sglang:hicache_backup_bytes_total backup\nsglang:load_back_tokens_total{pool="kv"} 0\n'
        with gzip.open(self.telemetry,'wt') as stream:
            for record in records:stream.write(json.dumps(record)+'\n')
        with self.assertRaisesRegex(ValueError,'counter reset'):
            self.run_windows()

    def test_saturated_worker_has_windows_without_a_later_idle_launch(self):
        rows=[dict(r,after_idle=i==0,vision_matmul_flops=1000)
              for i,r in enumerate(self.rows)]
        windows,coverage=self.run_windows(rows=rows)
        self.assertEqual([w['start'] for w in windows],[0,60,120,180])
        self.assertTrue(all(w['seconds']==60 for w in windows))
        self.assertTrue(all(w['features']==[12,120,660,60,12000] for w in windows))
        self.assertEqual(coverage['node0']['closed_episodes'],0)
        self.assertEqual(coverage['node0']['observed_busy_prefixes'],1)

    def test_onboarding_capture_can_begin_inside_an_existing_busy_interval(self):
        rows=[dict(r,after_idle=False) for r in self.rows]
        windows,_=self.run_windows(rows=rows)
        self.assertEqual([w['start'] for w in windows],[0,60,120,180])
        # A partial prefix cannot make the last minute fully observed.
        self.assertNotIn(240,[w['start'] for w in windows])

    def test_unjoined_forward_trims_the_continuing_busy_prefix(self):
        rows=[dict(r,after_idle=i==0) for i,r in enumerate(self.rows)]
        requests=[r for r in self.requests if r['rid']!='14']
        windows,coverage=self.run_windows(rows=rows,requests=requests)
        self.assertEqual([w['start'] for w in windows],[0])
        self.assertEqual(coverage['node0']['observed_busy_prefix_steps'],14)

    def test_missing_forward_does_not_restart_a_busy_prefix_without_idle(self):
        rows=[dict(r,after_idle=i==0) for i,r in enumerate(self.rows)
              if r['forward_iter']!=26]
        windows,_=self.run_windows(rows=rows)
        self.assertEqual(windows,[])

    def test_unused_overlap_decode_is_counted_only_with_completed_request_evidence(self):
        rows=[];requests=[]
        for original,request in zip(self.rows,self.requests):
            step=original['forward_iter']*2
            rows.append(dict(original,forward_iter=step))
            rows.append(dict(original,forward_iter=step+1,mode='decode',after_idle=False,
                ts=original['ts']+.1,dur_ms=100.,new_prefill_tokens=0,
                prefill_attention_pairs=None,launch_context_source='batch_sum',launch_seq_lens_sum=10))
            requests.append(dict(request,prefill_iters=[step],input_len=10,
                                 requested_output_len=1,output_len=1,decode_ct=0,spec_verify_ct=0))
        windows,_=self.run_windows(rows=rows,requests=requests)
        self.assertEqual(len(windows),4)
        self.assertTrue(all(w['features'][:3]==[24,120,660] for w in windows))
        self.assertTrue(all(abs(w['seconds']-13.2)<1e-9 for w in windows))
        for field,value in [('requested_output_len',128),('output_len',2),('decode_ct',1),('spec_verify_ct',1)]:
            with self.subTest(field=field):
                changed=[dict(request,**{field:value}) for request in requests]
                self.assertEqual(self.run_windows(rows=rows,requests=changed)[0],[])
        changed=[dict(row,launch_seq_lens_sum=11) if row['mode']=='decode' else row for row in rows]
        self.assertEqual(self.run_windows(rows=changed,requests=requests)[0],[])

if __name__=='__main__':unittest.main()
