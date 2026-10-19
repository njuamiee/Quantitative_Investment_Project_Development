import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_20261017'))
from execution import CohortMarket as OriginalMarket,Rule as OriginalRule
from test_execution import DATES
from adaptive_execution import CohortMarket,Rule

def fixture():
    p=pd.DataFrame({'date':DATES,'stockid':'BJ920001','open':10.,'close':10.,'adv20':1e8,
                    'observed':True,'mom20':0.,'vol20':.2,'in_pool':True,'adj_ret1':0.})
    s=p[['date','stockid']].assign(score=1.)
    return p,s

class AdaptiveAccountTests(unittest.TestCase):
    def test_default_matches_original_account(self):
        p,scores=fixture();dates=DATES
        old=OriginalMarket(p).run(scores,OriginalRule(topn=1),start=dates[0],end=dates[-1],save_details=True)
        new=CohortMarket(p).run(scores,Rule(topn=1),start=dates[0],end=dates[-1],save_details=True)
        np.testing.assert_allclose(old['daily'][['equity','cash','market_value']],new['daily'][['equity','cash','market_value']])

    def test_market_exit_uses_previous_close_and_next_open(self):
        p,scores=fixture();dates=DATES
        market=CohortMarket(p);market.trend=pd.Series(1.,index=dates);market.trend.loc[dates[2]]=-.1
        result=market.run(scores,Rule(topn=1,market_gate='trend20'),start=dates[0],end=dates[-1],cost_bp=0,save_details=True)
        exits=result['trades'];first=exits.iloc[0]
        self.assertEqual(first.exit_date,dates[3])
        self.assertEqual(first.reason,'market_gate_previous_close')
        self.assertEqual(first.holding_session_intervals,2)
        self.assertAlmostEqual(first.exit_price,10.)

    def test_circuit_early_exit_and_cash_cooldown(self):
        p,scores=fixture();dates=DATES
        p.loc[p.date==dates[2],'close']=8.
        market=CohortMarket(p);market.trend=pd.Series(1.,index=dates);market.breadth=pd.Series(.8,index=dates)
        result=market.run(scores,Rule(topn=1,allocation=1.,circuit_drawdown=.01),start=dates[0],end=dates[-1],cost_bp=0,save_details=True)
        d=result['daily'].set_index('date')
        self.assertTrue(d.loc[dates[3],'circuit_fired'])
        self.assertTrue(d.loc[dates[3]:dates[7],'risk_paused'].all())
        self.assertTrue((d.loc[dates[3]:dates[7],'market_value']==0).all())
        self.assertEqual(result['trades'].iloc[0].reason,'portfolio_pause_previous_close')

    def test_changing_later_prices_does_not_change_past_actions(self):
        p,scores=fixture();dates=DATES
        p['adj_ret1']=0.;p['in_pool']=True
        changed=p.copy();changed.loc[changed.date>=dates[8],['open','close']]=1000.
        rule=Rule(topn=1,circuit_drawdown=.08)
        old=CohortMarket(p).run(scores,rule,start=dates[0],end=dates[-1],save_details=True)
        new=CohortMarket(changed).run(scores,rule,start=dates[0],end=dates[-1],save_details=True)
        pd.testing.assert_frame_equal(old['daily'].iloc[:8],new['daily'].iloc[:8])

if __name__=='__main__':unittest.main()
