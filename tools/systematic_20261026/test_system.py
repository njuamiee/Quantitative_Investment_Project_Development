"""Independent small scenarios for causal timestamps and cash accounting."""
import unittest
import numpy as np
from account import ledger,risk_grid,metrics
from config import HORIZONS

def market(n=7,ns=2):
    op=np.full((n,ns),10.);cl=op.copy();adv=np.full_like(op,1e9);obs=np.ones_like(op)
    div=np.zeros_like(op);split=np.ones_like(op);vol=np.full_like(op,.02);mom=np.full_like(op,.01)
    group=np.tile(np.arange(1,ns+1),(n,1)).astype(float);rank=group.copy()
    t=np.full(n,.1);ones=np.ones(n);breadth=np.full(n,.7);dd=np.zeros(n)
    return [op,cl,adv,obs,div,split,vol,mom.copy(),mom.copy(),mom.copy(),group,group,rank,
            t,t.copy(),t.copy(),breadth,dd,ones,ones.copy()]

def account(a,order=None,fee=0.,exit_rule=0,rebal=1):
    n,ns=a[0].shape
    if order is None:order=np.tile(np.arange(ns,dtype=np.int32),(n,1))
    return ledger(order,*a,1,n-1,20,0,0,0,exit_rule,0,0,100,rebal,fee,1e6)

class LedgerTests(unittest.TestCase):
    def test_cash_identity_flat(self):
        d,e=account(market());self.assertLess(e,1e-8);np.testing.assert_allclose(d[:,0],1e6)
        np.testing.assert_allclose(d[:,0],d[:,1]+d[:,2])
        self.assertEqual(d[0,5],0);self.assertGreater(d[1,5],0)
    def test_fees_reduce_equity(self):
        d,e=account(market(),fee=.001);self.assertLess(d[-1,0],1e6)
        self.assertAlmostEqual(d[-1,0],1e6-d[:,3].sum(),places=6)
    def test_tplusone_rotation(self):
        a=market();order=np.full((7,1),-1,np.int32);order[1,0]=0;order[2:,0]=1
        d,e=account(a,order);self.assertEqual(d[1,6],0);self.assertGreater(d[2,6],0)
    def test_missing_quote_retains_position(self):
        a=market();order=np.full((7,1),-1,np.int32);order[1,0]=0;order[2:,0]=1
        a[3][3,0]=0;d,e=account(a,order);self.assertEqual(d[2,6],0);self.assertGreater(d[3,6],0)
    def test_limit_down_defers_sell(self):
        a=market();order=np.full((7,1),-1,np.int32);order[1,0]=0;order[2:,0]=1
        a[0][3,0]=7.;a[1][3,0]=7.;d,e=account(a,order)
        self.assertEqual(d[2,6],0);self.assertGreater(d[2,8],0)
    def test_split_and_dividend_no_invented_return(self):
        a=market();a[4][3,0]=1.;a[5][3,0]=2.;a[0][3:,0]=4.5;a[1][3:,0]=4.5
        d,e=account(a,rebal=20);np.testing.assert_allclose(d[:,0],1e6,atol=1e-7)
    def test_future_price_does_not_change_past(self):
        a=market();b=[x.copy() for x in a];b[0][5:]*=2;b[1][5:]*=2
        d,_=account(a);f,_=account(b);np.testing.assert_array_equal(d[:4],f[:4])
    def test_shared_capacity(self):
        a=market();a[2][:]=1e5;d,e=account(a)
        self.assertTrue(np.all(d[:,5]+d[:,6]<=2*1e5*.01+1e-7))
    def test_peak_stop_uses_previous_close(self):
        a=market();a[1][3,0]=8.;a[0][4:,0]=8.;a[1][4:,0]=8.
        d,e=account(a,exit_rule=1,rebal=20)
        self.assertEqual(d[2,6],0);self.assertGreater(d[3,6],0)
    def test_metric_includes_initial_loss(self):
        m=metrics(np.array([900000.,950000.]),['2026-01-01','2026-01-02'])
        self.assertAlmostEqual(m['max_drawdown'],.1)
    def test_registered_horizons_and_dedup(self):
        self.assertEqual(HORIZONS,[1,2,3,4,5,10,15,20])
        self.assertEqual(len(risk_grid(1)),1728);self.assertEqual(len(risk_grid(2)),3456)
    def test_constant_schedule_equals_fixed_rule(self):
        a=market();n,ns=a[0].shape;order=np.tile(np.arange(ns,dtype=np.int32),(n,1))
        params=[20,0,0,0,0,0,0,100,1];sch=np.tile(np.array(params+[0.]),(n,1))
        d,_=account(a);b,_=ledger(order,*a,1,n-1,*params,0.,1e6,sch)
        np.testing.assert_array_equal(d,b)
    def test_bear_single_stock_does_not_mean_full_position(self):
        a=market();a[-1][:]=.1;n,ns=a[0].shape;order=np.tile(np.arange(ns,dtype=np.int32),(n,1))
        d,e=ledger(order,*a,1,n-1,20,0,0,2,0,0,0,100,1,0.,1e6)
        self.assertLessEqual(d[:,4].max(),.05001)
    def test_unknown_industry_is_not_neutral(self):
        a=market();a[10][:]=0;n,ns=a[0].shape;order=np.tile(np.arange(ns,dtype=np.int32),(n,1))
        d,e=ledger(order,*a,1,n-1,20,0,0,0,0,1,0,100,1,0.,1e6)
        self.assertEqual(d[:,5].sum(),0)
    def test_minute_cash_and_future_capacity(self):
        from minute_study import minute_ledger
        n,ns=6,2;a=market(n,ns);order=np.tile(np.arange(ns,dtype=np.int32),(n,1))
        price=np.full((n,2,ns),10.,np.float32);amount=np.full_like(price,1e6)
        d,e=minute_ledger(order,price,amount.copy(),a[1],a[4],a[5],1,5,1,20,.5,0.,.05)
        np.testing.assert_allclose(d[:,0],d[:,1]+d[:,2]);self.assertLess(e,1e-7)
        future=amount.copy();future[4:]*=100
        b,_=minute_ledger(order,price,future,a[1],a[4],a[5],1,5,1,20,.5,0.,.05)
        np.testing.assert_array_equal(d[:3],b[:3])
    def test_horizon_controls_match_identical_daily_rules(self):
        def rules(h):
            return [{k:v for k,v in r.items() if k not in ['risk_id','rebalance']} for r in risk_grid(h) if r['rebalance_days']==1]
        base=rules(5)
        for h in HORIZONS:self.assertEqual(rules(h),base)
    def test_risk_pairs_change_exactly_one_declared_coordinate(self):
        from controlled_comparisons import risk_pairs,RISK_BASELINES
        for horizon in [1,2,20]:
            rules=risk_grid(horizon)
            for (axis,baseline,alternative),pairs in risk_pairs(horizon).items():
                self.assertEqual(len(set(pairs[:,1])),len(pairs))
                for before,after in pairs:
                    a,b=rules[before],rules[after]
                    self.assertEqual(str(a[axis]),baseline);self.assertEqual(str(b[axis]),alternative)
                    for other in RISK_BASELINES:
                        if other!=axis:self.assertEqual(a[other],b[other])
            if horizon==1:self.assertFalse(any(k[0]=='rebalance' for k in risk_pairs(horizon)))
    def test_report_csv_survives_temporary_destination_lock(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        from config import write_csv
        original=Path.replace;attempts=[]
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'report.csv';target.write_text('old complete snapshot',encoding='utf-8')
            def replace(source,dest):
                attempts.append(1)
                if len(attempts)==1:
                    self.assertEqual(target.read_text(),'old complete snapshot')
                    raise PermissionError('reader holds destination')
                return original(source,dest)
            with patch.object(Path,'replace',replace),patch('config.time.sleep'):
                write_csv(pd.DataFrame({'completed':[8640]}),target)
            self.assertEqual(pd.read_csv(target).completed.iloc[0],8640)
            self.assertEqual(len(attempts),2)
    def test_compute_shards_cover_grid_once(self):
        from runner import configurations
        from config import key
        configs=configurations();core=set();extra=set();remote=set()
        for c in configs:
            if c['library']=='daily' and c['window']==252:core.add(key(c))
            elif c['model']=='xgboost':extra.add(key(c))
            else:remote.add(key(c))
        self.assertEqual((len(core),len(extra),len(remote)),(960,960,6720))
        self.assertEqual(len(core|extra|remote),8640)
        self.assertFalse(core&extra or core&remote or extra&remote)

if __name__=='__main__':unittest.main()
