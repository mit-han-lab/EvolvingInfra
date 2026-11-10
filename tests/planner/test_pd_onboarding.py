import copy
import unittest

from evolving_infra.planner.calibration.onboarding import estimate_onboarding, update_workers
from evolving_infra.planner.calibration.dynamic import prefill_seconds
from evolving_infra.planner.calibration.controller import Controller


def windows(node='node0', start=1200.):
    # Each minute: completed restores and the cached prefix of requests that started.
    return [dict(node=node, start=start+60*i, end=start+60*(i+1),
                 seconds=10., features=[10., 100., 1000., tokens],
                 restore_lo=tokens, restore_hi=tokens,
                 started_cached_tokens=2*tokens)
            for i,tokens in enumerate((1000., 3000., 2000., 4000.))]


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(coefficients=[.5,.04,.001,.01],
                      tokens_per_step=10., restored_per_cached_token=.9)

    def test_restore_ratio_and_packing_change_and_the_fit_stays_frozen(self):
        frozen=copy.deepcopy(self.p)
        report=estimate_onboarding(windows())
        self.assertEqual(report['restored_per_cached_token'],.5)
        self.assertEqual((report['restored_tokens'],report['cached_tokens']),(10000.,20000.))
        self.assertEqual((report['forward_steps'],report['new_tokens'],report['tokens_per_step']),
                         (40.,400.,10.))
        work=dict(uncached=100.,pairs=1000.,cached=2000.)
        # 0.5*10 steps + 0.04*100 tokens + 0.001*1000 pairs + 0.01 s * 2000*0.5 restored
        self.assertAlmostEqual(prefill_seconds(self.p,work,report),20.)
        self.assertAlmostEqual(prefill_seconds(self.p,work),28.)
        self.assertEqual(self.p,frozen)

    def test_observed_packing_sets_the_step_count(self):
        rows=windows()
        for r in rows:r['features'][0]=20.  # 100 new tokens over 20 forwards
        report=estimate_onboarding(rows)
        self.assertEqual(report['tokens_per_step'],5.)
        work=dict(uncached=100.,pairs=1000.,cached=2000.)
        # 0.5*20 steps + 0.04*100 tokens + 0.001*1000 pairs + 0.01 s * 2000*0.5 restored
        self.assertAlmostEqual(prefill_seconds(self.p,work,report),25.)

    def test_windows_without_new_tokens_keep_the_warmup_packing(self):
        rows=windows()
        for r in rows:r['features'][1]=0.
        report=estimate_onboarding(rows)
        self.assertNotIn('tokens_per_step',report)
        work=dict(uncached=100.,pairs=1000.,cached=2000.)
        self.assertAlmostEqual(prefill_seconds(self.p,work,report),20.)
        legacy={k:v for k,v in report.items() if k!='tokens_per_step'}
        self.assertAlmostEqual(prefill_seconds(self.p,work,legacy),20.)

    def test_zero_restores_is_an_observation(self):
        rows=windows()
        for r in rows:r['features'][3]=r['restore_lo']=r['restore_hi']=0.
        self.assertEqual(estimate_onboarding(rows)['restored_per_cached_token'],0.)

    def test_counter_time_uncertainty_is_retained(self):
        rows=windows()
        for r in rows:r['restore_lo']*=.8;r['restore_hi']*=1.2
        lo,hi=estimate_onboarding(rows)['restored_per_cached_token_bounds']
        self.assertAlmostEqual(lo,.4);self.assertAlmostEqual(hi,.6)

    def test_invalid_windows_are_rejected(self):
        cases=[windows()[:1]]
        rows=windows();rows[1]['start']=rows[0]['start'];cases.append(rows)
        rows=windows();rows[0]['node']='other';cases.append(rows)
        rows=windows();rows[0]['end']+=1.;cases.append(rows)
        rows=windows();rows[0]['features'][3]=-1.;cases.append(rows)
        rows=windows();rows[0]['features'][0]=-1.;cases.append(rows)
        rows=windows();rows[0]['features'][1]=-1.;cases.append(rows)
        rows=windows()
        for r in rows:r['features'][0]=0.
        cases.append(rows)
        rows=windows();rows[0]['restore_lo']=2000.;cases.append(rows)
        rows=windows()
        for r in rows:r['started_cached_tokens']=0.
        cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                estimate_onboarding(rows)

    def test_bad_worker_does_not_affect_good_workers_or_mutate_input(self):
        workers={'node0':dict(prefill=dict(models=[self.p])),
                 'node1':dict(prefill=dict(models=[self.p]))}
        before=copy.deepcopy(workers)
        result=update_workers(workers,windows()+windows(node='node1')[:1])
        self.assertEqual(set(result['workers']),{'node0'})
        self.assertEqual(set(result['failures']),{'node1'})
        self.assertEqual(workers,before)

    def test_controller_checks_revision_model_and_chronology(self):
        assignment=dict(epoch='router',revision=0,roles={'http://p':'prefill','http://d':'decode'})
        snapshot=dict(assignment=assignment,completed_workload={})
        workers={'node0':dict(prefill=dict(models=[self.p])),'node1':{}}
        c=Controller(dict(workers=workers),{'node0':'http://p','node1':'http://d'},snapshot,
                     experts=256,top_k=8,fit_digest='frozen')
        report=dict(assignment=copy.deepcopy(assignment),compute_model_sha256='frozen',
                    workers={'node0':estimate_onboarding(windows())})
        c.update_onboarding(snapshot,report,lambda _:None)
        self.assertEqual(workers['node0']['prefill']['models'][0],self.p)
        for mutation in ('repeat','epoch','digest','role','node'):
            invalid=copy.deepcopy(report)
            if mutation=='epoch':invalid['assignment']['epoch']='other'
            if mutation=='digest':invalid['compute_model_sha256']='other'
            if mutation=='role':invalid['workers']['node1']=invalid['workers'].pop('node0')
            if mutation=='node':invalid['workers']['node0']['node']='node1'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                c.update_onboarding(snapshot,invalid,lambda _:None)


if __name__=='__main__':unittest.main()
