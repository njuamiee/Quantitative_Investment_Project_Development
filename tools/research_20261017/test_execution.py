"""Independent synthetic tests for the five-session cash/share ledger."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution import CohortMarket, Rule


DATES = pd.bdate_range('2026-01-05', periods=13)


def fixture(ids=('BJ920001',), *, dates=DATES, opens=None, closes=None, observed=None, adv=None,
            adj_returns=None, scores=None, actions=None, metadata=None):
    rows=[]
    for i, day in enumerate(dates):
        for stock in ids:
            op=opens.get((i,stock),10.) if opens else 10.
            cl=closes.get((i,stock),10.) if closes else 10.
            obs=observed.get((i,stock),True) if observed else True
            value=adv.get((i,stock),1e8) if adv else 1e8
            rows.append(dict(date=day,stockid=stock,open=op,close=cl,adv20=value,
                             observed=obs,mom20=0.,vol20=.2,in_pool=True,
                             adj_ret1=adj_returns.get((i,stock),0.) if adj_returns else 0.))
    prices=pd.DataFrame(rows)
    if scores is None:
        scores=pd.DataFrame([dict(date=day,stockid=stock,score=len(ids)-j)
            for day in dates for j,stock in enumerate(ids)])
    else:
        scores=pd.DataFrame(scores)
    market=CohortMarket(prices,actions=actions,metadata=metadata)
    return market,scores


class FinancialLedgerTests(unittest.TestCase):
    def test_cash_identity_flow_and_no_borrow(self):
        m,s=fixture()
        result=m.run(s,rule=Rule(topn=1),start=DATES[0],end=DATES[-1],cost_bp=15,save_details=True)
        d=result['daily']
        self.assertTrue((d.cash>=-1e-8).all())
        self.assertTrue(np.allclose(d.cash+d.market_value,d.equity,atol=1e-7))
        self.assertTrue(np.allclose(d.cash,d.opening_cash+d.dividend_cash+d.sell_value-d.buy_value-d.cost,atol=1e-7))
        self.assertTrue((d.exposure<=1+1e-8).all())
        self.assertGreater(len(result['orders']),0)

    def test_one_cohort_enters_next_open_exits_fifth_session_close(self):
        m,s=fixture()
        r=m.run(s,rule=Rule(topn=1),start=DATES[0],end=DATES[6],cost_bp=0,save_details=True)
        fills=r['orders'].query('filled_shares > 0')
        first_buy=fills.query('side == "buy"').iloc[0]
        first_sell=fills.query('side == "sell"').iloc[0]
        self.assertEqual(first_buy.date,DATES[1])
        self.assertEqual(first_buy.when,'open')
        self.assertEqual(first_sell.date,DATES[5])
        self.assertEqual(first_sell.when,'close')
        self.assertEqual(first_sell.reason,'scheduled_fifth_session')
        self.assertEqual(r['trades'].iloc[0].holding_session_intervals,4)
        self.assertEqual(r['trades'].iloc[0].holding_sessions_inclusive,5)
        self.assertEqual(fills.query('side == "buy"').iloc[1].date,DATES[6])
        self.assertTrue((r['trades'].exit_date>r['trades'].entry_date).all())

    def test_stop_uses_prior_close_then_next_open_and_respects_t_plus_one(self):
        stock='BJ920001'
        cl={(1,stock):8.0}
        op={(2,stock):8.5}
        m,s=fixture(opens=op,closes=cl)
        r=m.run(s,rule=Rule(topn=1,stop=.10),start=DATES[0],end=DATES[3],cost_bp=0,save_details=True)
        sell=r['orders'].query('side == "sell" and filled_shares > 0').iloc[0]
        self.assertEqual(sell.date,DATES[2])
        self.assertEqual(sell.when,'open')
        self.assertEqual(sell.reason,'trailing_stop_previous_close')
        self.assertAlmostEqual(sell.price,8.5)
        self.assertEqual(r['trades'].iloc[0].holding_session_intervals,1)

    def test_halt_on_due_date_delays_exit_until_first_tradable_open(self):
        stock='BJ920001'
        obs={(5,stock):False,(6,stock):False}
        op={(5,stock):np.nan,(6,stock):np.nan}
        cl={(5,stock):np.nan,(6,stock):np.nan}
        m,s=fixture(opens=op,closes=cl,observed=obs)
        r=m.run(s,rule=Rule(topn=1),start=DATES[0],end=DATES[8],cost_bp=0,save_details=True)
        attempted=r['orders'].query('side == "sell"')
        self.assertEqual(attempted.iloc[0].date,DATES[5])
        self.assertEqual(attempted.iloc[0].filled_shares,0)
        self.assertEqual(attempted.iloc[0].reject_reason,'no_observed_bar')
        completed=attempted[attempted.filled_shares.gt(0)].iloc[0]
        self.assertEqual(completed.date,DATES[7])
        self.assertEqual(completed.when,'open')
        self.assertEqual(completed.reason,'overdue_exit')

    def test_cash_dividend_and_split_preserve_shares_and_pnl(self):
        stock='BJ920001'
        prices_open={(i,stock):5. for i in range(3,13)}
        prices_close={(i,stock):5. for i in range(3,13)}
        actions=pd.DataFrame([dict(date=DATES[3],stockid=stock,dividend=1.,split=2.)])
        m,s=fixture(opens=prices_open,closes=prices_close,actions=actions)
        r=m.run(s,rule=Rule(topn=1),start=DATES[0],end=DATES[5],cost_bp=0,save_details=True)
        buy=r['orders'].query('side == "buy" and filled_shares > 0').iloc[0]
        sell=r['orders'].query('side == "sell" and filled_shares > 0').iloc[0]
        self.assertAlmostEqual(sell.filled_shares,2*buy.filled_shares)
        self.assertAlmostEqual(r['daily'].dividend_cash.sum(),buy.filled_shares)
        self.assertAlmostEqual(r['trades'].iloc[0].pnl,2*buy.filled_shares*5+buy.filled_shares-buy.filled_shares*10)
        self.assertTrue(np.allclose(r['daily'].cash+r['daily'].market_value,r['daily'].equity))

    def test_ex_date_adjustment_does_not_trigger_false_stop(self):
        stock='BJ920001'
        actions=pd.DataFrame([dict(date=DATES[3],stockid=stock,dividend=1.,split=2.)])
        op={(i,stock):4.5 for i in range(3,13)}
        cl={(i,stock):4.5 for i in range(3,13)}
        m,s=fixture(opens=op,closes=cl,actions=actions)
        r=m.run(s,rule=Rule(topn=1,stop=.10),start=DATES[0],end=DATES[5],cost_bp=0,save_details=True)
        sales=r['orders'].query('side == "sell" and filled_shares > 0')
        self.assertEqual(len(sales),1)
        self.assertEqual(sales.iloc[0].reason,'scheduled_fifth_session')

    def test_cash_rule_does_not_generate_fake_buy_attempts(self):
        m,s=fixture()
        r=m.run(s,rule=Rule(name='cash',topn=1,exposure='cash'),start=DATES[0],end=DATES[6],save_details=True)
        self.assertTrue((r['daily'].exposure==0).all())
        self.assertEqual(len(r['orders']),0)
        self.assertEqual(r['daily'].orders_attempted.sum(),0)

    def test_old_halted_cohort_does_not_allow_new_cohort_to_breach_industry_limit(self):
        # Six 5% holdings in one industry remain unsold at expiry. A new cohort
        # from the same industry must account for those existing positions.
        ids=[f'BJ92{i:04d}' for i in range(12)]
        obs={(i,stock):False for i in (5,6) for stock in ids[:6]}
        op={(i,stock):np.nan for i in (5,6) for stock in ids[:6]}
        cl={(i,stock):np.nan for i in (5,6) for stock in ids[:6]}
        score=[]
        for j,stock in enumerate(ids):
            score.append(dict(date=DATES[0],stockid=stock,score=100-j if j<6 else -j))
            score.append(dict(date=DATES[5],stockid=stock,score=100-j if j>=6 else -j))
        meta=pd.DataFrame([dict(date=day,stockid=stock,industry='same_sector',
                                size_group='large',momentum_group='flat',volatility_group='low')
                            for day in DATES for stock in ids])
        m,s=fixture(ids,opens=op,closes=cl,observed=obs,scores=score,metadata=meta)
        r=m.run(s,rule=Rule(name='diversified',topn=20,diversify=True),start=DATES[0],end=DATES[6],cost_bp=0,save_details=True)
        x=r['risk_exposure']
        industry=x[(x.date==DATES[6])&(x.dimension=='industry')&(x.group=='same_sector')]
        self.assertEqual(len(industry),1)
        self.assertLessEqual(float(industry.weight.iloc[0]),.300001)
        self.assertTrue(bool(r['daily'].loc[r['daily'].date.eq(DATES[6]),'blocked_by_unsold'].iloc[0]))
        self.assertEqual(len(r['orders'].query('side == "buy" and date == @DATES[6]')),0)

    def test_real_metadata_tertile_column_names_reach_diversification(self):
        # The actual parquet uses *_tertile, not *_group. A missing alias would
        # reject every stock as having no historical size information.
        ids=['BJ920001','BJ920002']
        meta=pd.DataFrame([dict(date=day,stockid=stock,industry='documented_sector',
                                size_tertile='small',momentum_tertile='middle',volatility_tertile='low')
                            for day in DATES for stock in ids])
        m,s=fixture(ids,metadata=meta)
        r=m.run(s,rule=Rule(name='diversified',topn=10,diversify=True),start=DATES[0],end=DATES[2],cost_bp=0,save_details=True)
        buys=r['orders'].query('side == "buy" and filled_shares > 0')
        self.assertGreater(len(buys),0)

    def test_unknown_industry_is_rejected_in_diversified_rule(self):
        ids=['BJ920001','BJ920002']
        meta=pd.DataFrame([dict(date=day,stockid=stock,industry='unknown',
                                size_tertile='small',momentum_tertile='middle',volatility_tertile='low')
                            for day in DATES for stock in ids])
        m,s=fixture(ids,metadata=meta)
        r=m.run(s,rule=Rule(name='diversified',topn=10,diversify=True),
                start=DATES[0],end=DATES[2],cost_bp=0,save_details=True)
        self.assertTrue(r['orders'].empty)
        self.assertTrue((r['daily'].exposure==0).all())
        self.assertEqual(set(r['rejected_selection'].reason),{'missing_historical_industry'})

    def test_future_prices_and_scores_cannot_change_past_cash_or_orders(self):
        dates=pd.bdate_range('2026-01-05',periods=23)
        ids=['BJ920001','BJ920002']
        base_returns={(i,stock):.01+(.002 if i%2 else -.002) for i in range(len(dates)) for stock in ids}
        future_returns=base_returns.copy()
        for i in range(13,len(dates)):
            for stock in ids:future_returns[(i,stock)]=.2 if stock==ids[0] else -.2
        future_open={(i,stock):30. if stock==ids[0] else 3. for i in range(13,len(dates)) for stock in ids}
        future_close=future_open.copy()
        base_scores=[dict(date=day,stockid=stock,score=2-j) for day in dates for j,stock in enumerate(ids)]
        future_scores=[dict(date=r['date'],stockid=r['stockid'],
                            score=(-100-r['score']) if r['date']>=dates[13] else r['score']) for r in base_scores]
        m1,s1=fixture(ids,dates=dates,adj_returns=base_returns,scores=base_scores)
        m2,s2=fixture(ids,dates=dates,opens=future_open,closes=future_close,
                      adj_returns=future_returns,scores=future_scores)
        rule=Rule(topn=2,exposure='vol12')
        r1=m1.run(s1,rule=rule,start=dates[0],end=dates[-1],cost_bp=5,save_details=True)
        r2=m2.run(s2,rule=rule,start=dates[0],end=dates[-1],cost_bp=5,save_details=True)
        prefix1=r1['daily'].loc[r1['daily'].date.le(dates[12])].reset_index(drop=True)
        prefix2=r2['daily'].loc[r2['daily'].date.le(dates[12])].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix1,prefix2)
        orders1=r1['orders'].loc[r1['orders'].date.le(dates[12])].reset_index(drop=True)
        orders2=r2['orders'].loc[r2['orders'].date.le(dates[12])].reset_index(drop=True)
        pd.testing.assert_frame_equal(orders1,orders2)

    def test_quarter_schedule_is_reported_and_previous_signal_cap_is_applied(self):
        m,s=fixture()
        quarter=str(DATES[0].to_period('Q'))
        schedule={quarter:Rule(name='scheduled_vol',topn=1,exposure='vol12')}
        r=m.run(s,rule=Rule(name='fallback',topn=2),schedule=schedule,
                start=DATES[0],end=DATES[6],cost_bp=0,save_details=True)
        d=r['daily']
        entry=d.loc[d.date.eq(DATES[1])].iloc[0]
        self.assertEqual(entry.signal_used,DATES[0])
        self.assertEqual(entry.effective_rule,'scheduled_vol')
        self.assertEqual(entry.effective_topn,1)
        # With zero past volatility the predeclared fallback target is 25%.
        self.assertAlmostEqual(entry.target_exposure_on_entry,.25)
        self.assertEqual(r['metrics']['rule'],'quarterly_validation_selected')
        self.assertEqual(r['metrics']['topn'],[1])
        self.assertTrue(r['metrics']['schedule_applied'])


if __name__=='__main__':
    unittest.main()
