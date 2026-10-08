"""Meaningful synthetic accounting, execution and causal-data regression checks."""
import io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import prepare
from prepare import RUN
from accounts import simulate
from replay import fill_order

def fixture():
    dates=pd.bdate_range('2025-01-02',periods=8)
    prices=pd.DataFrame({'date':dates,'stockid':'BJ920001','open':10.,'close':10.,'high':10.1,'low':9.9,
        'amount':1e7,'observed':True,'adjustment_boundary':False})
    signals=pd.DataFrame({'date':[dates[0]],'stockid':['BJ920001'],'pred_selected':[1.],'adv20':[1e7]})
    return dates,prices,signals

class LedgerTests(unittest.TestCase):
    def test_flat_price_and_cash_conservation(self):
        dates,p,s=fixture();d,t,o,m=simulate(s,p,cost_bp=0)
        self.assertAlmostEqual(d.equity.iloc[-1],1e6)
        self.assertTrue(np.allclose(d.cash+d.market_value,d.equity))
        self.assertEqual(t.entry_date.iloc[0],dates[1]);self.assertEqual(t.exit_date.iloc[0],dates[5])
        self.assertTrue((d.cash>=0).all())
    def test_cost_is_real_cash_debit(self):
        _,p,s=fixture();d,t,o,m=simulate(s,p,cost_bp=15)
        self.assertAlmostEqual(1e6-d.equity.iloc[-1],o.cost.sum(),places=6)
    def test_zero_volume_does_not_trade(self):
        dates,p,s=fixture();p.loc[p.date==dates[1],'observed']=False
        _,t,o,m=simulate(s,p,cost_bp=0);self.assertEqual(m['unfilled_entries'],1);self.assertTrue(o.empty)
    def test_suspension_delays_exit(self):
        dates,p,s=fixture();p.loc[p.date==dates[5],'observed']=False
        _,t,_,_=simulate(s,p,cost_bp=0);self.assertEqual(t.exit_date.iloc[0],dates[6])
    def test_limit_proxy_blocks_entry(self):
        dates,p,s=fixture();p.loc[p.date==dates[1],'open']=13.
        _,_,o,m=simulate(s,p,cost_bp=0);self.assertTrue(o.empty)
    def test_t_plus_one_cap_sell(self):
        dates,p,s=fixture();caps=pd.Series([.7,0.],index=[dates[0],dates[1]])
        _,t,_,_=simulate(s,p,cost_bp=0,cap_series=caps)
        self.assertTrue((t.exit_date>t.entry_date).all());self.assertEqual(t.exit_date.iloc[0],dates[2])
    def test_cash_dividend_conservation(self):
        dates,p,s=fixture();p.loc[p.date>=dates[3],['open','high','low','close']]-=1
        actions=pd.DataFrame({'date':[dates[3]],'stockid':['BJ920001'],'dividend':[1.],'split':[1.]})
        d,_,_,m=simulate(s,p,cost_bp=0,actions=actions)
        self.assertAlmostEqual(d.equity.iloc[-1],1e6);self.assertGreater(m['corporate_action_cash'],0)
    def test_split_conservation(self):
        dates,p,s=fixture();p.loc[p.date>=dates[3],['open','high','low','close']]/=2
        actions=pd.DataFrame({'date':[dates[3]],'stockid':['BJ920001'],'dividend':[0.],'split':[2.]})
        d,_,_,_=simulate(s,p,cost_bp=0,actions=actions);self.assertAlmostEqual(d.equity.iloc[-1],1e6)

class ExecutionTests(unittest.TestCase):
    def bars(self):
        return pd.DataFrame({'datetime':pd.date_range('2025-01-02 09:31',periods=3,freq='min'),
            'open':[10,10,10],'high':[11,11,11],'low':[9,9,9],'close':[10,10,10],'volume':[1000,0,2000]})
    def test_partial_and_zero_volume(self):
        r=fill_order(self.bars(),100,'buy');self.assertEqual(r['filled'],30);self.assertEqual(r['remaining'],70);self.assertEqual(r['vwap'],11)
    def test_null_bar_and_limit_order(self):
        b=self.bars();b.loc[0,'high']=np.nan
        self.assertEqual(fill_order(b,100,'buy')['filled'],20)
        self.assertEqual(fill_order(b,100,'buy',limit_price=10.5)['filled'],0)

class CausalityTests(unittest.TestCase):
    def test_offline_signal_time_gate(self):
        from vnpy_adapter import available_signals
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'signals.csv'
            pd.DataFrame({'date':['2026-09-29'],'stockid':['BJ920001'],
                'signal_available_at':['2026-09-29 18:00'],'earliest_execution_date':['2026-09-30']}).to_csv(path,index=False)
            self.assertTrue(available_signals(path,'2026-09-29 19:00').empty)
            self.assertEqual(available_signals(path,'2026-09-30 09:30').vt_symbol.iloc[0],'920001.BSE')
            self.assertEqual(len(available_signals(path,'2026-09-30 01:30+00:00')),1)
            self.assertTrue(available_signals(path,'2026-10-01 09:30').empty)
    def test_future_price_mutation_preserves_past_pool_and_features(self):
        original_data,original_run=prepare.DATA,prepare.RUN
        try:
            with tempfile.TemporaryDirectory() as temp:
                prepare.DATA=Path(temp);prepare.RUN=Path(temp)/'results';prepare.RUN.mkdir()
                dates=pd.bdate_range('2022-01-03',periods=100)
                for adj in ['raw','qfq']:
                    folder=Path(temp)/'bj'/adj;folder.mkdir(parents=True)
                    for stock in range(3):
                        p=np.arange(100)*.01+10+stock
                        frame=pd.DataFrame({'日期':dates,'股票代码':f'92000{stock}','开盘':p,'收盘':p,'最高':p+.1,'最低':p-.1,'成交量':10000,'成交额':1e6*(stock+1)})
                        frame.to_csv(folder/f'92000{stock}-test.csv',index=False,encoding='utf-8-sig')
                first=prepare.prices('bj',dates)
                for p in (Path(temp)/'bj/raw').glob('*.csv'):
                    x=pd.read_csv(p);x.loc[80:,'成交额']=1e12;x.loc[80:,'收盘']=1000;x.to_csv(p,index=False,encoding='utf-8-sig')
                second=prepare.prices('bj',dates)
                cols=['date','stockid','in_pool','amount_rank']+prepare.PRICE_FEATURES
                pd.testing.assert_frame_equal(first[first.date<dates[80]][cols].reset_index(drop=True),second[second.date<dates[80]][cols].reset_index(drop=True))
        finally:prepare.DATA,prepare.RUN=original_data,original_run

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue())
    (RUN/'test_results.json').write_text(json.dumps({'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'details':stream.getvalue()},ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
