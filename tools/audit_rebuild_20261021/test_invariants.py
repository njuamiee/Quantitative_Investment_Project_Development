"""Critical temporal/accounting regression checks, independent of performance."""
from pathlib import Path
import sys,json,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'training_runs/audit_rebuild_20261021'
from csi_rebuild import ledger,stats
from risk_execution import CohortMarket,Rule

class Invariants(unittest.TestCase):
    def toy(self):
        dates=pd.bdate_range('2025-07-02',periods=16)
        grid={'dates':dates.to_numpy(),'ids':np.array(['A','B']),'hfq_open':np.column_stack([np.arange(16)+10.,np.arange(16)+20.]),
              'hfq_close':np.column_stack([np.arange(16)+10.5,np.arange(16)+20.5])}
        p=pd.DataFrame([dict(date=d,stockid=s,si=i,score=2-i,label=100 if i==0 else -100) for d in dates for i,s in enumerate(['A','B'])])
        return p,grid

    def test_future_labels_never_change_holdings(self):
        p,g=self.toy();a,sa,_=ledger(p,g,'score',topn=1)
        p.label=np.nan;b,sb,_=ledger(p,g,'score',topn=1)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(sa,sb)

    def test_missing_entry_keeps_cash_without_hindsight_replacement(self):
        p,g=self.toy();g['hfq_open'][1,0]=np.nan
        d,s,f=ledger(p,g,'score',topn=1)
        self.assertEqual(d.positions.iloc[1],0);self.assertEqual(d.equity.iloc[1],1.)
        self.assertFalse((s.entry_date==pd.Timestamp(g['dates'][1])).any())

    def test_open_to_open_uses_exact_five_intervals(self):
        p,g=self.toy();d,s,f=ledger(p,g,'score',topn=1)
        # At day6 the first position has realized open6/open1 and the second starts.
        self.assertAlmostEqual(d.equity.iloc[5],15.5/11)
        self.assertAlmostEqual(d.equity.iloc[6],16/11*16.5/16)
        self.assertLess((d.equity-d.cash-d.market_value).abs().max(),1e-10)

    def test_unsellable_residual_does_not_freeze_other_cash(self):
        p,g=self.toy();g['hfq_open'][6:10,0]=np.nan
        d,s,f=ledger(p,g,'score',topn=2)
        self.assertTrue(((s.entry_date==pd.Timestamp(g['dates'][6]))&(s.stockid=='B')).any())
        self.assertGreater(d.market_value.iloc[6],0)
        self.assertGreaterEqual(d.cash.min(),-1e-10)

    def test_drawdown_includes_initial_capital(self):
        d=pd.DataFrame({'equity':[.8,.9],'exposure':[1.,1.]})
        self.assertAlmostEqual(stats(d)['mdd'],.2)

    def test_actual_corrected_labels_match_full_calendar(self):
        m=pd.read_parquet(RUN/'csi/meta.parquet');g=np.load(RUN/'csi/market_grid.npz')
        ix=np.random.default_rng(4).choice(np.flatnonzero(m.label.notna()),size=300,replace=False)
        a=m.iloc[ix];expected=g['hfq_open'][a.di.to_numpy()+6,a.si.to_numpy()]/g['hfq_open'][a.di.to_numpy()+1,a.si.to_numpy()]-1
        np.testing.assert_allclose(a.label,expected,rtol=0,atol=1e-12)
        expected_date=g['dates'][a.di.to_numpy()+6]
        np.testing.assert_array_equal(a.label_end.to_numpy(),expected_date)

    def test_every_new_fold_purges_actual_label_end(self):
        files=list((RUN/'csi').glob('rolling_fold_*.json'))+list((RUN/'expanded').glob('fold_*.json'))
        self.assertGreater(len(files),10)
        for p in files:
            records=json.loads(p.read_text(encoding='utf-8'));records=records if isinstance(records,list) else [records]
            for r in records:
                if 'actual_train_label_max' in r:
                    a,b,c,d=r['actual_train_label_max'],r['val_start'],r['actual_val_label_max'],r['predict_start']
                else:a,b,c,d=r['train_actual_last_exit'],r['val_start'],r['val_actual_last_exit'],r['test_start']
                self.assertLess(pd.Timestamp(a),pd.Timestamp(b),p.name)
                self.assertLess(pd.Timestamp(c),pd.Timestamp(d),p.name)

    def test_future_market_changes_do_not_change_past_risk_target(self):
        dates=pd.bdate_range('2025-01-02',periods=100)
        p=pd.DataFrame([dict(date=d,stockid='A',open=10,close=10,adv20=1e8,observed=True,mom20=.02,vol20=.02,in_pool=True,adj_ret1=.001+np.sin(i)*.01) for i,d in enumerate(dates)])
        scores=pd.DataFrame({'date':[dates[70]],'stockid':['A'],'score':[1.]})
        a=CohortMarket(p);rule=Rule(exposure='combined');v=a.target_exposure(70,rule,scores)
        p.loc[p.date>dates[70],['mom20','adj_ret1']]=100.
        b=CohortMarket(p);self.assertEqual(v,b.target_exposure(70,rule,scores));self.assertGreaterEqual(v,.25);self.assertLessEqual(v,.95)

    def test_original_history_reproduced(self):
        r=json.loads((RUN/'original/reproduction_check.json').read_text(encoding='utf-8'))
        self.assertLess(r['max_historical_equity_error'],.01)

    def test_factor_downloads_have_hashes_and_no_unresolved_transfer_errors(self):
        files=list((ROOT/'training_data/audit_rebuild_20261021/library').glob('*.json'));self.assertEqual(len(files),120)
        for p in files:
            r=json.loads(p.read_text(encoding='utf-8'));self.assertFalse(r['errors'],p.name)
            self.assertTrue(all(len(z['sha256'])==64 for z in r['records']))

if __name__=='__main__':unittest.main(verbosity=2)
