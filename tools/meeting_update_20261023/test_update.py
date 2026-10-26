import unittest,importlib.util,json
import numpy as np
import pandas as pd
from execution import CohortMarket,Rule
from run_experiments import selection
class Tests(unittest.TestCase):
    def fixture(self):
        dates=pd.bdate_range('2024-10-08',periods=160)
        p=pd.DataFrame([dict(date=d,stockid='A',open=10.,close=10.,adv20=1e9,observed=True,mom20=0.,vol20=.01,in_pool=True,adj_ret1=0.) for d in dates])
        s=p[['date','stockid']].copy();s['score']=1.
        return dates,p,s
    def run_toy(self,p,s,rule):return CohortMarket(p).run(s,rule,start=str(p.date.min().date()),end=str(p.date.max().date()),save_details=True,cost_bp=0)
    def test_stop_next_open_and_cooldown(self):
        dates,p,s=self.fixture();p.loc[2,'close']=9.;p.loc[3,'open']=8.8
        r=self.run_toy(p,s,Rule('stop',stop=.05,cooldown=10,allow_residual=True))
        t=r['trades'];stop=t[t.reason.eq('stop_previous_close')];self.assertEqual(stop.exit_date.iloc[0],dates[3]);self.assertEqual(stop.exit_price.iloc[0],8.8)
        buys=r['orders'].query("side=='buy' and filled_shares>0")
        self.assertFalse(buys.date.between(dates[3],dates[12]).any());self.assertTrue((buys.date==dates[16]).any())
    def test_failed_stop_remains_pending(self):
        dates,p,s=self.fixture();p.loc[2,'close']=9.;p.loc[3,'observed']=False
        r=self.run_toy(p,s,Rule(stop=.05,cooldown=10,allow_residual=True))
        self.assertEqual(r['trades'].exit_date.iloc[0],dates[4]);self.assertEqual(r['trades'].reason.iloc[0],'stop_previous_close')
    def test_profit_next_open(self):
        dates,p,s=self.fixture();p.loc[2,'close']=12.1;p.loc[3,'open']=12.2
        r=self.run_toy(p,s,Rule(take_profit=.20,allow_residual=True))
        t=r['trades'].query("reason=='take_profit_previous_close'");self.assertEqual(t.exit_date.iloc[0],dates[3])
    def test_split_does_not_invent_profit_signal(self):
        dates,p,s=self.fixture();p.loc[3:,['open','close']]=5.
        actions=pd.DataFrame([dict(date=dates[3],stockid='A',dividend=0.,split=2.)])
        r=CohortMarket(p,actions).run(s,Rule(take_profit=.2,allow_residual=True),start=str(dates[0].date()),end=str(dates[-1].date()),save_details=True,cost_bp=0)
        self.assertFalse(r['trades'].reason.eq('take_profit_previous_close').any())
        self.assertLess((r['daily'].equity-1e6).abs().max(),1e-6)
    def test_partial_basis_reconciles(self):
        dates,p,s=self.fixture();p.loc[:30,'adj_ret1']=.01;p.loc[32:,'adj_ret1']=-.05
        r=self.run_toy(p,s,Rule(topn=1,exposure='ma20',allow_residual=True))
        # Top1 is capped at10%, so use ten stocks to force a 50% portfolio trim.
        pp=pd.concat([p.assign(stockid=str(i)) for i in range(10)]);ss=pd.concat([s.assign(stockid=str(i)) for i in range(10)])
        r=self.run_toy(pp,ss,Rule(topn=10,exposure='ma20',allow_residual=True))
        t=r['trades'];self.assertTrue(t.reason.eq('market_ma20_reduce').any());self.assertLess(t.pnl.abs().max(),1e-6)
        self.assertLess((r['daily'].equity-1e6).abs().max(),1e-6)
    def test_future_quotes_do_not_change_past(self):
        dates,p,s=self.fixture();rule=Rule(stop=.05,exposure='ma20',allow_residual=True)
        a=self.run_toy(p,s,rule)['daily'];p.loc[70:,['open','close','adj_ret1']]=100
        b=self.run_toy(p,s,rule)['daily'];pd.testing.assert_frame_equal(a.iloc[:70],b.iloc[:70])
    def test_selection_future_invariance(self):
        cal=pd.bdate_range('2024-10-08','2026-10-19')
        a=pd.DataFrame(dict(date=cal,equity=1e6*np.cumprod(1+np.sin(np.arange(len(cal)))*.01+.001),exposure=.8))
        b=a.copy();b.equity=1e6*np.cumprod(1+np.cos(np.arange(len(cal)))*.02)
        x,log=selection({'a':a,'b':b},cal);cut=pd.Timestamp('2026-05-22');b.loc[b.date>cut,'equity']*=4
        y,_=selection({'a':a,'b':b},cal)
        self.assertEqual({k:v for k,v in x.items() if pd.Timestamp(k)<=cut},{k:v for k,v in y.items() if pd.Timestamp(k)<=cut})
        self.assertTrue((log.validation_end<log.decision_date).all())
    def test_no_rule_change_matches_previous_engine(self):
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('old_execution',Path(__file__).parents[1]/'audit_rebuild_20261021/risk_execution.py')
        import sys
        old=importlib.util.module_from_spec(spec);sys.modules['old_execution']=old;spec.loader.exec_module(old)
        dates,p,s=self.fixture();a=self.run_toy(p,s,Rule(allow_residual=True))['daily']
        b=old.CohortMarket(p).run(s,old.Rule(allow_residual=True),start=str(dates[0].date()),end=str(dates[-1].date()),save_details=True,cost_bp=0)['daily']
        np.testing.assert_allclose(a.equity,b.equity,rtol=0,atol=1e-7)

if __name__=='__main__':unittest.main(verbosity=2)
