import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evolving_infra.planner.observability.trace_snapshot import records_before,capture

class SnapshotTests(unittest.TestCase):
    def test_shift_before_cutoff_preserves_raw_time_duration_and_telemetry(self):
        import gzip,hashlib
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);step=root/'iter';req=root/'req';tele=root/'tele'
            raw='{"ts":8.5,"dur_ms":200}\n{"ts":9.5,"dur_ms":300}\n'
            step.write_text(raw);req.write_text(raw)
            tele.write_text('{"at":9,"nodes":[]}\n')
            args=[dict(node='node0',host='h',steps=str(step),requests=str(req))]
            # A slow worker event at9.5 is actually10.9: old cutoff leaked it.
            old,_=capture(root/'old',args,tele,start=0,end=10,clock_pad_s=.05)
            self.assertEqual(old['workers']['node0']['steps_count'],2)
            new,records=capture(root/'new',args,tele,start=0,end=10,clock_pad_s=.05,
                worker_clock_offsets_s={'node0':-1.4})
            w=new['workers']['node0'];self.assertEqual(w['steps_count'],1)
            row=json.loads(Path(w['steps']).read_text())
            self.assertAlmostEqual(row['ts'],9.9)
            self.assertEqual((row['worker_ts'],row['dur_ms']),(8.5,200))
            self.assertEqual(records[w['steps']],[row])
            self.assertEqual(w['steps_source']['sha256'],hashlib.sha256(raw.encode()).hexdigest())
            self.assertEqual(step.read_text(),raw)
            with gzip.open(new['telemetry'],'rt') as f:self.assertEqual(json.load(f)['at'],9)

    def test_fast_worker_is_not_excluded_before_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rows';p.write_text('{"ts":11}\n')
            rows,_=records_before(p,10,worker_offset_s=2)
            self.assertEqual(rows,[dict(ts=9,worker_ts=11)])

    def test_double_correction_and_incomplete_worker_mapping_fail(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rows';p.write_text('{"ts":1,"worker_ts":2}\n')
            with self.assertRaises(ValueError):records_before(p,10,worker_offset_s=1)
            with self.assertRaises(ValueError):
                capture(Path(td)/'out',[dict(node='node0')],p,start=0,end=10,
                    clock_pad_s=.05,worker_clock_offsets_s={})
    def test_decode_launch_interval_is_also_trimmed_by_clock_bound(self):
        from evolving_infra.planner.perf_model.calibration import fit_snapshot
        rows=[dict(ts=1.2,dur_ms=500,busy_interval_s=.5),
              dict(ts=2,dur_ms=500,busy_interval_s=.5),
              dict(ts=9,dur_ms=500,busy_interval_s=1)]
        m=dict(start=0,end=10,clock_pad_s=1,telemetry='/tmp/no-reading/restore.jsonl.gz',
            files={},workers={'node0':dict(role='decode',steps='d')})
        def inspect(**kw):
            self.assertEqual(kw['decode_rows']['node0'],[rows[1]])
            self.assertEqual((kw['start'],kw['end']),(0,10))
            return {}
        with patch('evolving_infra.planner.perf_model.calibration.load_steps',return_value=rows), \
             patch('evolving_infra.planner.perf_model.calibration.prefill_windows',return_value=([],{})), \
             patch('evolving_infra.planner.perf_model.calibration.fit_bundle',side_effect=inspect):
            fit_snapshot(m,{},capacities={'node0':{}},replicas=4,experts=256,top_k=8,
                predeclared_prefill_shape=[1,1,1,1])
    def test_late_complete_and_partial_rows_cannot_change_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rows'
            p.write_bytes(b'{"ts":1}\n{"ts":3}\n{"ts":')
            self.assertEqual(records_before(p,2)[0],[{'ts':1}])
    def test_prefix_is_fixed_when_the_read_starts_and_reports_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rows';p.write_text('{"ts":1}\n{"ts":1.5}\n')
            real=Path.open
            def appending_open(path,*args,**kwargs):  # a writer appends once the read has begun
                with real(path,'a') as f:f.write('{"ts":2}\n')
                return real(path,*args,**kwargs)
            with patch.object(Path,'open',appending_open):
                rows,proof=records_before(p,2)
            self.assertEqual(rows,[{'ts':1},{'ts':1.5}])
            self.assertEqual(proof['complete_bytes'],20)
            self.assertEqual(p.stat().st_size,29)
    def test_incremental_captures_equal_captures_from_scratch(self):
        # One cursor map across captures of growing files (control's onboarding) must yield the
        # proofs, files and rows of a fresh read each time: partial last lines, a window that
        # moves backwards, an unchanged repeat, a file rewritten in place and a replaced file.
        import gzip,os
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);step=root/'iter';req=root/'req';tele=root/'tele'
            for path in (step,req,tele):path.write_bytes(b'')
            def grow(path,lines):
                with path.open('ab') as f:f.write(lines.encode())
            def rows(key,a,b):return ''.join(json.dumps({key:t,'v':t*3})+'\n' for t in range(a,b))
            def replace(path):  # a new file: the same size and last line, another first line
                other=root/'other';other.write_bytes(path.read_bytes().replace(b'"v": 0}',b'"v": 7}',1))
                os.replace(other,path)
            def snapshot(name,start,end,cursors):
                manifest,records=capture(root/name,[dict(node='node0',steps=str(step),requests=str(req))],
                    tele,start=start,end=end,clock_pad_s=.5,worker_clock_offsets_s={'node0':.25},cursors=cursors)
                w=manifest['workers']['node0']
                with gzip.open(manifest['telemetry'],'rt') as f:telemetry=f.read()
                return ([w[k+'_source'] for k in ('steps','requests')],manifest['telemetry_source'],
                        manifest['files']['workers/node0/scheduler/steps.jsonl'],
                        [Path(w[k]).read_bytes() for k in ('steps','requests')],telemetry,
                        [records[w[k]] for k in ('steps','requests')])
            cursors={}
            plan=[(lambda:(grow(step,rows('ts',0,40)+'{"ts":40,'),grow(req,rows('ts',0,30)),
                           grow(tele,rows('at',0,50)+'{"at"')),0,20),
                  (lambda:(grow(step,'"v":0}\n'+rows('ts',41,80)),grow(req,rows('ts',30,70)+rows('ts',3,4)),
                           grow(tele,':50}\n'+rows('at',51,90))),25,60),
                  (lambda:None,10,45),
                  (lambda:None,30,70),
                  (lambda:(req.write_text(rows('ts',100,200)),grow(step,rows('ts',80,90)),replace(tele)),35,75)]
            for index,(change,start,end) in enumerate(plan):
                change()
                with self.subTest(capture=index):
                    self.assertEqual(snapshot(f'live{index}',start,end,cursors),
                                     snapshot(f'fresh{index}',start,end,None))
            self.assertGreater(cursors[str(step)]['skip'],0)
    def test_request_cutoff_and_explicit_generation_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);step=root/'iter';req=root/'req';tele=root/'telemetry'
            step.write_text('{"ts":5,"mode":"extend"}\n')
            req.write_text('{"ts":5,"rid":"early"}\n{"ts":15,"rid":"late"}\n')
            tele.write_text('{"at":5,"nodes":[]}\n{"at":15,"nodes":[]}\n')
            result,_=capture(root/'snapshot',[dict(node='node0',generation=2,host='physical',steps=str(step),requests=str(req))],tele,start=0,end=10,clock_pad_s=.01)
            got=result['workers']['node0']
            self.assertEqual(got['requests_count'],1)
            self.assertEqual(Path(got['requests']).parent.parent.name,'node0')
            self.assertEqual(json.loads(Path(got['requests']).read_text())['rid'],'early')
            with self.assertRaises(FileExistsError):capture(root/'snapshot',[],tele,start=0,end=10,clock_pad_s=.01)

if __name__=='__main__':unittest.main()
