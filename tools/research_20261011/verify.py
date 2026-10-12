"""Invariant tests for financially material implementation risks; synthetic data only."""
import unittest,copy
import numpy as np
import pandas as pd
from ledger import Market,metrics
from regimes import filtered
from factor_dsl import evaluate
from jev_adapter import Evidence,build_request,call_jev,validate_response,evaluate_probabilities
from common import RUN,dump

def fixture():
    dates=pd.bdate_range('2026-01-05',periods=12)
    p=pd.DataFrame([{'date':d,'stockid':s,'open':10.,'close':10.,'amount':1e7,'adv20':1e7,'observed':True,'vol20':.02,'mom20':.01}
                    for d in dates for s in ['BJ000001','BJ000002']])
    pred=p[['date','stockid']].copy();pred['score']=1.
    return dates,p,pred

class FinancialTests(unittest.TestCase):
    def test_t1_and_cash(self):
        dates,p,pred=fixture();d,m,o,t,_=Market(p).run(pred,horizon=1,save_details=True)
        self.assertTrue((pd.to_datetime(t.exit_date)>pd.to_datetime(t.entry_date)).all())
        self.assertEqual(pd.Timestamp(t.entry_date.min()),dates[1]);self.assertEqual(pd.Timestamp(t.exit_date.min()),dates[2])
        self.assertLess((d.equity-d.cash-d.market_value).abs().max(),1e-6)
        self.assertAlmostEqual(d.equity.iloc[-1],1e6-o.fee.sum(),places=5)
        self.assertGreater(m['open_lots_at_end'],0)
    def test_capacity_shared_and_missing_bar(self):
        _,p,pred=fixture();p['adv20']=1000000
        d,m,o,*_=Market(p).run(pred,horizon=2,save_details=True)
        o['notional']=o.shares*o.price
        self.assertLessEqual(o.groupby(['date','stockid']).notional.sum().max(),10000+1e-6)
        p.loc[p.date==p.date.unique()[3],'observed']=False
        _,_,o,*_=Market(p).run(pred,horizon=2,save_details=True)
        self.assertFalse((o.date==p.date.unique()[3]).any())
    def test_split_no_phantom_loss(self):
        dates,p,pred=fixture();p.loc[p.date>=dates[3],['open','close']]=5.
        a=pd.DataFrame([{'stockid':s,'date':dates[3],'dividend':0.,'split':2.} for s in p.stockid.unique()])
        d,m,*_=Market(p,a).run(pred,horizon=5,cost_bp=0)
        self.assertLess(abs(d.equity-1e6).max(),1e-6)
    def test_prefix_execution_causality(self):
        dates,p,pred=fixture();market=Market(p)
        a=market.run(pred,horizon=2,save_details=True)[0]
        changed=p.copy();changed.loc[changed.date>=dates[7],['open','close']]=20.
        b=Market(changed).run(pred,horizon=2)[0]
        np.testing.assert_allclose(a.equity.iloc[:7],b.equity.iloc[:7])
    def test_initial_capital_drawdown(self):
        d=pd.DataFrame({'date':pd.bdate_range('2026-01-01',periods=3),'equity':[900000,920000,910000],
                        'exposure':[1,1,1],'turnover':[0,0,0],'cost':[0,0,0]})
        self.assertAlmostEqual(metrics(d)['max_drawdown'],.1)
    def test_hmm_filter_prefix(self):
        class Model:
            startprob_=np.array([.5,.5]);transmat_=np.array([[.9,.1],[.1,.9]])
            def _compute_log_likelihood(self,x):return np.c_[-(x[:,0]+1)**2,-(x[:,0]-1)**2]
        x=np.arange(12).reshape(-1,1)/10
        a,_=filtered(Model(),x);b,_=filtered(Model(),np.r_[x[:6],x[6:]+100])
        np.testing.assert_allclose(a[:6],b[:6]);np.testing.assert_allclose(a.sum(1),1)
    def test_dsl_rejects_execution(self):
        f=pd.DataFrame({'a':[1.,2.],'b':[0.,2.]})
        self.assertTrue(np.isnan(evaluate('a / b',f).iloc[0]))
        for s in ["__import__('os').system('echo bad')",'a.shift(-1)','a[0]']:
            with self.assertRaises(ValueError):evaluate(s,f)
    def test_jev_time_gate_and_network_disabled(self):
        e=Evidence('TEST','1','local://synthetic','2026-01-01T18:00:00+08:00','2026-01-01T18:05:00+08:00','2026-01-01T18:05:00+08:00','Synthetic')
        with self.assertRaises(ValueError):build_request(e,'2026-01-01T15:00:00+08:00')
        with self.assertRaises(RuntimeError):call_jev(build_request(e,'2026-01-02T15:00:00+08:00'))
    def test_jev_contract_and_probability_evaluation(self):
        response={'model':'jev-1.13.0','answers':{
          'financial_report_classification':{'type':'choice','choice':'mixed','probabilities':{'improving':.1,'deteriorating':.1,'mixed':.7,'insufficient':.1},'confidence':.6},
          'news_event_score':{'type':'score','score':2.,'legend':{str(i):str(i) for i in range(5)},'probabilities':{'0':0.,'1':0.,'2':1.,'3':0.,'4':0.},'confidence':1.},
          'risk_signal_extraction':{'type':'noul','noul':.2}}}
        validate_response(response);bad=copy.deepcopy(response);bad['answers']['news_event_score']['score']=.8
        with self.assertRaises(ValueError):validate_response(bad)
        bad=copy.deepcopy(response);bad['model']='jev-different'
        with self.assertRaises(ValueError):validate_response(bad)
        bad=copy.deepcopy(response);bad['answers']['news_event_score']['probabilities']=[0,0,1,0,0]
        with self.assertRaises(ValueError):validate_response(bad)
        bad=copy.deepcopy(response);bad['answers']['financial_report_classification']['choice']='improving'
        with self.assertRaises(ValueError):validate_response(bad)
        with self.assertRaises(ValueError):evaluate_probabilities([[.8,.2]],[.9])
        with self.assertRaises(ValueError):evaluate_probabilities([],[])
        self.assertEqual(evaluate_probabilities([[1,0],[0,1]],[0,1])['brier_multiclass'],0.)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinancialTests))
    dump({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'data':'synthetic; not performance evidence'},RUN/'verification.json')
    raise SystemExit(not result.wasSuccessful())
