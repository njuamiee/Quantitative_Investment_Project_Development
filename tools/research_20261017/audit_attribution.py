"""Independent saved-file reconciliation of stock attribution and period anchors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261017'
FOLDER=RUN/'accounts/new_main_monthly'
OUT=RUN/'verification'
OUT.mkdir(parents=True,exist_ok=True)
daily=pd.read_csv(FOLDER/'daily.csv',parse_dates=['date']).sort_values('date').reset_index(drop=True)
orders=pd.read_csv(FOLDER/'orders.csv',parse_dates=['date'])
holdings=pd.read_csv(FOLDER/'holdings.csv',parse_dates=['date'])
actions=pd.read_csv(ROOT/'training_data/research_20261011/actions.csv',parse_dates=['date'])
period=pd.read_csv(RUN/'period_comparison.csv')
summary=pd.read_csv(RUN/'strategy_summary.csv')


def assert_close(label,a,b,tolerance=.01):
    if not np.isfinite(a) or not np.isfinite(b) or abs(a-b)>tolerance:
        raise AssertionError(f'{label}: {a} vs {b}')


dates=pd.DatetimeIndex(daily.date)
equity=daily.equity.to_numpy(float)
peak_before=np.maximum.accumulate(np.r_[1e6,equity])[1:]
drawdown=equity/peak_before-1
trough_i=int(np.argmin(drawdown))
peak_i=int(np.argmax(equity[:trough_i+1]))
peak_date=daily.date.iloc[peak_i]
trough_date=daily.date.iloc[trough_i]
max_drawdown=float(-drawdown[trough_i])
recorded_mdd=float(summary.set_index('strategy').loc['new_main_monthly','max_drawdown'])
assert_close('max drawdown magnitude',max_drawdown,recorded_mdd,1e-10)


def attribution(tag,baseline_date,end_date):
    baseline_date=pd.Timestamp(baseline_date);end_date=pd.Timestamp(end_date)
    start_i=int(dates.get_loc(baseline_date))+1
    start_date=dates[start_i]
    filled=orders.loc[orders.date.between(start_date,end_date)&orders.filled_shares.gt(0)].copy()
    filled['notional']=filled.filled_shares*filled.price
    h0=holdings.loc[holdings.date.eq(baseline_date)].groupby('stockid').value.sum()
    h1=holdings.loc[holdings.date.eq(end_date)].groupby('stockid').value.sum()
    buy=filled.loc[filled.side.eq('buy')].groupby('stockid').notional.sum()
    sell=filled.loc[filled.side.eq('sell')].groupby('stockid').notional.sum()
    fees=filled.groupby('stockid').fee.sum()
    # Independently trace each action to shares held at the preceding close.
    stock_div={}
    for act in actions.loc[actions.date.between(start_date,end_date)].itertuples(index=False):
        earlier=dates[dates<act.date]
        if len(earlier)==0:continue
        q=holdings.loc[holdings.date.eq(earlier[-1])&holdings.stockid.eq(act.stockid),'shares'].sum()
        if q:stock_div[act.stockid]=stock_div.get(act.stockid,0.)+q*act.dividend
    dividend=pd.Series(stock_div,dtype=float)
    fresh=pd.concat({'start_value':h0,'end_value':h1,'buys':buy,'sells':sell,'fees':fees,'dividends':dividend},axis=1).fillna(0.)
    fresh['price_pnl']=fresh.end_value-fresh.start_value+fresh.sells-fresh.buys
    fresh['net_pnl']=fresh.price_pnl+fresh.dividends-fresh.fees
    saved=pd.read_csv(RUN/(tag+'_stock_attribution.csv')).set_index('stockid')
    if set(fresh.index)!=set(saved.index):raise AssertionError(f'{tag} stock IDs differ')
    max_column_gap=0.
    for column in fresh.columns:
        gap=float((fresh[column].sort_index()-saved[column].sort_index()).abs().max())
        max_column_gap=max(max_column_gap,gap)
        assert_close(f'{tag}:{column}',gap,0,1e-6)
    baseline=daily.loc[daily.date.eq(baseline_date)].iloc[0]
    end=daily.loc[daily.date.eq(end_date)].iloc[0]
    actual_delta=float(end.equity-baseline.equity)
    source_cash_delta=float(end.cash-baseline.cash)
    reconstructed_cash_delta=float(sell.sum()-buy.sum()-fees.sum()+dividend.sum())
    asset_delta=float(h1.sum()-h0.sum()+reconstructed_cash_delta)
    assert_close(f'{tag} cash flows',source_cash_delta,reconstructed_cash_delta)
    assert_close(f'{tag} equity delta',actual_delta,asset_delta)
    assert_close(f'{tag} stock attribution',float(fresh.net_pnl.sum()),actual_delta)
    observed_div=float(daily.loc[daily.date.between(start_date,end_date),'dividend_cash'].sum())
    assert_close(f'{tag} dividends',observed_div,float(dividend.sum()))
    return {'tag':tag,'baseline_date':str(baseline_date.date()),'start_date':str(start_date.date()),
            'end_date':str(end_date.date()),'stock_rows':len(fresh),
            'start_equity':float(baseline.equity),'end_equity':float(end.equity),
            'equity_change':actual_delta,'attribution_sum':float(fresh.net_pnl.sum()),
            'cash_delta':source_cash_delta,'cash_flow_recomputed':reconstructed_cash_delta,
            'market_value_delta':float(h1.sum()-h0.sum()),
            'buy_notional':float(buy.sum()),'sell_notional':float(sell.sum()),
            'fees':float(fees.sum()),'dividends':float(dividend.sum()),
            'max_per_stock_column_difference':max_column_gap}


max_attr=attribution('max_drawdown',peak_date,trough_date)
recent_half_base=daily.loc[daily.date.lt('2026-04-01'),'date'].iloc[-1]
half_attr=attribution('recent_half',recent_half_base,'2026-09-30')

anchors=[]
for label,start in [('recent_year','2025-10-01'),('recent_half','2026-04-01')]:
    first=daily.loc[daily.date.ge(start),'date'].iloc[0]
    before=daily.loc[daily.date.lt(first)].iloc[-1]
    end=daily.iloc[-1]
    ret=float(end.equity/before.equity-1)
    recorded=float(period.loc[period.strategy.eq('new_main_monthly')&period.period.eq(label),'total_return'].iloc[0])
    assert_close(f'{label} prior-session base return',ret,recorded,1e-10)
    anchors.append({'period':label,'first_included_date':str(first.date()),
                    'baseline_date':str(before.date.date()),'baseline_equity':float(before.equity),
                    'end_equity':float(end.equity),'return_recomputed':ret,'return_table':recorded})

max_excess_error=0.
for r in period.itertuples(index=False):
    for baseline in ('equal_buy_hold','equal_pool_5day','bj50_price_index','old_main'):
        col='excess_vs_'+baseline
        base=float(period.loc[period.strategy.eq(baseline)&period.period.eq(r.period),'total_return'].iloc[0])
        observed=float(getattr(r,col))
        error=abs(observed-(r.total_return-base))
        max_excess_error=max(max_excess_error,error)
        assert_close(f'{r.strategy}/{r.period}/{col}',error,0,1e-10)

report={'pass':True,'max_drawdown':{'peak_date':str(peak_date.date()),'trough_date':str(trough_date.date()),
                                    'fraction':max_drawdown,**max_attr},
        'recent_half_attribution':half_attr,'recent_period_anchors':anchors,
        'maximum_excess_arithmetic_error':max_excess_error,
        'method':'Independent reads of account daily/holdings/orders and corporate actions; no call to report.stock_attribution().'}
(OUT/'attribution_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'attribution_audit.md').write_text(
    '# 回撤与最近半年股票归因独立核对\n\n'
    f"主账户最大回撤从 {peak_date:%Y-%m-%d} 的 {max_attr['start_equity']:,.2f} 元，到 {trough_date:%Y-%m-%d} 的 {max_attr['end_equity']:,.2f} 元，资产减少 {-max_attr['equity_change']:,.2f} 元，回撤率 {max_drawdown:.2%}。`max_drawdown_stock_attribution.csv` 的逐股净额合计 {max_attr['attribution_sum']:,.2f} 元，和现金、股票市值变化逐分对上；逐股各列最大差 {max_attr['max_per_stock_column_difference']:.3g} 元。\n\n"
    f"最近半年以 {recent_half_base:%Y-%m-%d} 收盘资产为起点，4月1日至9月30日资产增加 {half_attr['equity_change']:,.2f} 元，逐股归因合计 {half_attr['attribution_sum']:,.2f} 元；4月1日旧仓平掉后持有现金，不能把这段完整区间写成0收益。\n\n"
    '最近一年与最近半年的分段收益都以前一交易日收盘资产为分母；`period_comparison.csv` 全部策略对四个基准的超额逐项重算，最大算术误差 ' + f'{max_excess_error:.3g}。\n',
    encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='method'},ensure_ascii=False,indent=2))
