"""Independent arithmetic and timing checks on saved new account files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261017'
ACCOUNT=RUN/'accounts/new_main_monthly'
OUT=RUN/'verification'
OUT.mkdir(parents=True,exist_ok=True)


def close(a,b,tol=1e-6):
    return bool(np.isclose(float(a),float(b),atol=tol,rtol=1e-10))


def check(name,condition,detail=None):
    results.append({'name':name,'pass':bool(condition),'detail':detail})
    if not condition:
        raise AssertionError(f'{name}: {detail}')


results=[]
daily=pd.read_csv(ACCOUNT/'daily.csv',parse_dates=['date','signal_used'])
orders=pd.read_csv(ACCOUNT/'orders.csv',parse_dates=['date','signal_date'])
trades=pd.read_csv(ACCOUNT/'trades.csv',parse_dates=['signal_date','entry_date','exit_date'])
holdings=pd.read_csv(ACCOUNT/'holdings.csv',parse_dates=['date','entry_date'])
saved=json.loads((ACCOUNT/'metrics.json').read_text(encoding='utf-8'))
summary=pd.read_csv(RUN/'strategy_summary.csv')
period=pd.read_csv(RUN/'period_comparison.csv')
schedule=json.loads((RUN/'risk_schedule.json').read_text(encoding='utf-8'))

check('unique_sorted_dates',daily.date.is_unique and daily.date.is_monotonic_increasing)
check('no_negative_cash',daily.cash.min()>=-1e-6,float(daily.cash.min()))
identity=(daily.equity-daily.cash-daily.market_value).abs()
check('cash_plus_market_value_equals_equity',identity.max()<1e-6,float(identity.max()))
flow=(daily.cash-daily.opening_cash-daily.sell_value+daily.buy_value+daily.cost-daily.dividend_cash).abs()
check('each_day_cash_flow_reconciles',flow.max()<1e-6,float(flow.max()))
first_return=float(daily.equity.iloc[-1]/1e6-1)
check('saved_total_return_matches_daily',close(saved['total_return'],first_return),[saved['total_return'],first_return])
check('saved_mean_exposure_matches_daily',close(saved['mean_exposure'],daily.exposure.mean()),[saved['mean_exposure'],daily.exposure.mean()])
check('saved_total_cost_matches_daily',close(saved['total_cost'],daily.cost.sum()),[saved['total_cost'],daily.cost.sum()])

dayindex={d:i for i,d in enumerate(daily.date)}
check('all_fill_amounts_within_request',((orders.filled_shares>=0)&(orders.filled_shares<=orders.requested_shares+1e-8)).all())
filled=orders.loc[orders.filled_shares.gt(0)].copy()
check('buy_only_at_open',filled.loc[filled.side.eq('buy'),'when'].eq('open').all())
check('completed_sales_never_same_day_as_entry',(trades.exit_date>trades.entry_date).all())
check('buy_orders_follow_signal_by_one_session',all(dayindex[r.date]-dayindex[r.signal_date]==1
    for r in filled.loc[filled.side.eq('buy')].itertuples(index=False)))
check('bought_lot_ids_unique',filled.loc[filled.side.eq('buy'),'lot_id'].is_unique)
check('each_buy_lot_sold_once_by_period_end',trades.lot_id.is_unique and
      set(trades.lot_id)==set(filled.loc[filled.side.eq('buy'),'lot_id']) and daily.lots.iloc[-1]==0)
buy_notional=filled.loc[filled.side.eq('buy')].assign(value=lambda x:x.filled_shares*x.price).groupby('date').value.sum()
sell_notional=filled.loc[filled.side.eq('sell')].assign(value=lambda x:x.filled_shares*x.price).groupby('date').value.sum()
fee=filled.groupby('date').fee.sum()
audit=daily.set_index('date')
check('daily_buy_value_matches_fills',np.allclose(audit.buy_value,buy_notional.reindex(audit.index,fill_value=0).to_numpy(),atol=1e-6))
check('daily_sell_value_matches_fills',np.allclose(audit.sell_value,sell_notional.reindex(audit.index,fill_value=0).to_numpy(),atol=1e-6))
check('daily_fees_match_fills',np.allclose(audit.cost,fee.reindex(audit.index,fill_value=0).to_numpy(),atol=1e-6))
check('lot_holding_interval_matches_trade_dates',all(dayindex[r.exit_date]-dayindex[r.entry_date]==r.holding_session_intervals
    for r in trades.itertuples(index=False)))
check('holding_inclusive_equals_intervals_plus_one',(trades.holding_sessions_inclusive==trades.holding_session_intervals+1).all())
check('scheduled_exit_is_fifth_session_close',all(r.holding_sessions_inclusive>=5 and
    dayindex[r.exit_date]-dayindex[r.signal_date]>=5 for r in trades.loc[trades.reason.eq('scheduled_fifth_session')].itertuples(index=False)))
check('stop_exit_occurs_at_open',filled.loc[(filled.side.eq('sell'))&filled.reason.eq('trailing_stop_previous_close'),'when'].eq('open').all())
check('overdue_exit_occurs_at_open',filled.loc[(filled.side.eq('sell'))&filled.reason.eq('overdue_exit'),'when'].eq('open').all())

holdvalue=holdings.groupby('date').value.sum()
check('holdings_sum_to_market_value',np.allclose(audit.market_value,holdvalue.reindex(audit.index,fill_value=0).to_numpy(),atol=1e-6))
signal_days=daily.loc[daily.signal_used.notna()].copy()
check('signals_are_five_sessions_apart',all((dayindex[r.date]-1)%5==0 for r in signal_days.itertuples(index=False)))
check('effective_rule_matches_registered_schedule',all(r.effective_rule==schedule[str(r.signal_used.to_period('Q'))]['name']
    for r in signal_days.itertuples(index=False)))

row=summary.set_index('strategy').loc['new_main_monthly']
check('strategy_summary_return_matches_account',close(row.total_return,saved['total_return']))
check('strategy_summary_mdd_matches_account',close(row.max_drawdown,saved['max_drawdown']))
check('strategy_summary_sharpe_matches_account',close(row.sharpe_rf15,saved['sharpe_rf15']))
check('strategy_summary_calmar_matches_account',close(row.calmar,saved['calmar']))
full=period[(period.strategy=='new_main_monthly')&(period.period=='full')].iloc[0]
check('period_comparison_full_return_matches_account',close(full.total_return,saved['total_return']))

# Independent metric recomputation for all 13 aligned strategy curves.
def independent_performance(frame,capital=1e6,initial_date=None):
    e=frame.equity.to_numpy(float)
    ret=np.r_[e[0]/capital-1,e[1:]/e[:-1]-1]
    peak=np.maximum.accumulate(np.r_[capital,e])[1:]
    mdd=max(0.,float(-(e/peak-1).min()))
    first=pd.Timestamp(initial_date) if initial_date is not None else frame.date.iloc[0]
    years=max((frame.date.iloc[-1]-first).days/365.25,1/252)
    annual=(e[-1]/capital)**(1/years)-1
    std=ret.std(ddof=1) if len(ret)>1 else 0.
    rf=1.015**(1/252)-1
    return dict(total_return=float(e[-1]/capital-1),annual_return=float(annual),
                max_drawdown=mdd,sharpe_rf15=float((ret.mean()-rf)/std*np.sqrt(252)) if std>1e-12 else 0.,
                calmar=float(annual/mdd) if mdd>1e-12 else None,
                mean_exposure=float(frame.exposure.mean()))

all_daily=pd.read_parquet(RUN/'all_daily.parquet')
all_daily.date=pd.to_datetime(all_daily.date)
names=set(summary.strategy)
check('thirteen_registered_curves_present',len(names)==13 and set(all_daily.strategy)==names,len(names))
metrics_max_error=0.
for name,g in all_daily.groupby('strategy'):
    g=g.sort_values('date').reset_index(drop=True)
    check(f'{name}_dates_match_main',g.date.reset_index(drop=True).equals(daily.date.reset_index(drop=True)))
    calc=independent_performance(g)
    row=summary.set_index('strategy').loc[name]
    for key in ['total_return','annual_return','max_drawdown','sharpe_rf15','mean_exposure']:
        diff=abs(float(row[key])-calc[key]);metrics_max_error=max(metrics_max_error,diff)
        check(f'{name}_{key}_matches_curve',diff<1e-8,diff)
    if calc['calmar'] is not None:
        diff=abs(float(row.calmar)-calc['calmar']);metrics_max_error=max(metrics_max_error,diff)
        check(f'{name}_calmar_matches_curve',diff<1e-8,diff)

period_max_error=0.
for r in period.itertuples(index=False):
    g=all_daily[all_daily.strategy.eq(r.strategy)].sort_values('date').reset_index(drop=True)
    bounds={'full':('2024-10-08','2026-09-30'),'recent_year':('2025-10-01','2026-09-30'),
            'recent_half':('2026-04-01','2026-09-30')}
    left,right=bounds[r.period]
    part=g[g.date.between(left,right)]
    prev=g[g.date.lt(part.date.iloc[0])]
    cap=float(prev.equity.iloc[-1]) if len(prev) else 1e6
    initial=prev.date.iloc[-1] if len(prev) else None
    calc=independent_performance(part,cap,initial)
    for key in ['total_return','max_drawdown','sharpe_rf15']:
        diff=abs(float(getattr(r,key))-calc[key]);period_max_error=max(period_max_error,diff)
        check(f'{r.strategy}_{r.period}_{key}_matches_curve',diff<1e-8,diff)

# Every executable account independently obeys a single cash balance.
cash_accounts={}
for folder in sorted((RUN/'accounts').iterdir()):
    if not folder.is_dir():continue
    x=pd.read_csv(folder/'daily.csv',parse_dates=['date'])
    identity_error=float((x.equity-x.cash-x.market_value).abs().max())
    flow_error=float((x.cash-x.opening_cash-x.sell_value+x.buy_value+x.cost-x.dividend_cash).abs().max())
    minimum=float(x.cash.min())
    check(f'{folder.name}_cash_constraints',minimum>=-1e-6 and identity_error<1e-6 and flow_error<1e-6,
          [minimum,identity_error,flow_error])
    cash_accounts[folder.name]={'minimum_cash':minimum,'max_identity_error':identity_error,'max_flow_error':flow_error}

# Recompute the selected predeclared exposure from signal-date data.
prices=pd.read_parquet(ROOT/'training_data/research_20261011/prices.parquet',
                       columns=['date','stockid','in_pool','adj_ret1'])
pooled=prices.loc[prices.in_pool].groupby('date').adj_ret1.mean()
market_ret=pooled.reindex(daily.date).fillna(0.)
trend=(1+market_ret).rolling(20,min_periods=10).apply(np.prod,raw=True)-1
vol=market_ret.rolling(20,min_periods=10).std()*np.sqrt(252)
trend.index=daily.date;vol.index=daily.date
target_max_error=0.
for r in signal_days.itertuples(index=False):
    rule=schedule[str(r.signal_used.to_period('Q'))]
    if rule['exposure']=='full':expected=1.
    elif rule['exposure']=='cash':expected=0.
    elif rule['exposure']=='trend':expected=.95 if float(trend.loc[r.signal_used])>0 else .25
    elif rule['exposure']=='vol12':
        v=float(vol.loc[r.signal_used]);expected=float(np.clip(.12/v,.10,.95)) if np.isfinite(v) and v>0 else .25
    else:raise AssertionError('Unknown registered exposure rule')
    diff=abs(expected-r.target_exposure_on_entry)
    target_max_error=max(target_max_error,diff)
check('signal_date_exposure_cap_matches_past_returns',target_max_error<1e-8,target_max_error)

# Planned group weights are independently recomputed from actual historical descriptors.
meta=pd.read_parquet(RUN/'risk_metadata.parquet',columns=['date','stockid','industry','size_tertile',
                                                        'momentum_tertile','volatility_tertile'])
meta.date=pd.to_datetime(meta.date)
buys=orders.loc[orders.side.eq('buy')&orders.rule.astype(str).str.startswith('diversified')].copy()
check('diversified_buy_attempts_exist',len(buys)>0,len(buys))
buys=buys.merge(meta,left_on=['signal_date','stockid'],right_on=['date','stockid'],how='left',validate='many_to_one',suffixes=('','_meta'))
check('diversified_buys_have_known_industry',buys.industry.notna().all() and
      ~buys.industry.astype(str).str.lower().isin(['unknown','nan','none','missing','未知']).any())
check('diversified_buys_have_known_size',buys.size_tertile.notna().all())
group_weight_max={}
for column,limit in [('industry',.30),('size_tertile',.50),('momentum_tertile',.50),('volatility_tertile',.50)]:
    weights=buys.groupby(['signal_date',column],dropna=False).target_weight.sum()
    maximum=float(weights.max()) if len(weights) else 0.
    group_weight_max[column]=maximum
    check(f'{column}_planned_weight_within_entry_limit',maximum<=limit+1e-8,[maximum,limit])

observed_risk=pd.read_csv(ACCOUNT/'risk_exposure.csv',parse_dates=['date'])
observed_risk=observed_risk.merge(daily[['date','effective_rule']],on='date',how='left')
observed_risk=observed_risk[observed_risk.effective_rule.astype(str).str.startswith('diversified')]
max_actual_group_weight=observed_risk.groupby('dimension').weight.max().to_dict()

recent_half=daily[daily.date.ge('2026-04-01')]
check('main_flat_after_2026_04_01_close',recent_half.equity.iloc[1:].nunique()==1 and
      close(recent_half.equity.iloc[-1],recent_half.equity.iloc[0]))

report={
    'pass':all(r['pass'] for r in results), 'checks':results,
    'days':len(daily),'first':str(daily.date.iloc[0].date()),'last':str(daily.date.iloc[-1].date()),
    'account':str(ACCOUNT),
    'total_return':first_return,'mean_exposure':float(daily.exposure.mean()),
    'minimum_cash':float(daily.cash.min()),'maximum_cash_identity_error':float(identity.max()),
    'maximum_daily_cash_flow_error':float(flow.max()),
    'buy_fills':int(((orders.side=='buy')&(orders.filled_shares>0)).sum()),
    'sell_parts':len(trades),
    'mean_holding_session_intervals':float(trades.holding_session_intervals.mean()) if len(trades) else None,
    'mean_holding_sessions_inclusive':float(trades.holding_sessions_inclusive.mean()) if len(trades) else None,
    'scheduled_exit_parts':int(trades.reason.eq('scheduled_fifth_session').sum()),
    'stopped_exit_parts':int(trades.reason.eq('trailing_stop_previous_close').sum()),
    'overdue_exit_parts':int(trades.reason.eq('overdue_exit').sum()),
    'blocked_cohort_days':int(daily.blocked_by_unsold.sum()),
    'all_curve_metrics_max_error':metrics_max_error,
    'all_period_metrics_max_error':period_max_error,
    'all_account_cash_checks':cash_accounts,
    'signal_target_cap_max_error':target_max_error,
    'max_diversified_planned_group_weights':group_weight_max,
    'max_actual_group_weight_during_diversified_period':max_actual_group_weight,
    'recent_half_total_return':independent_performance(recent_half,float(daily.loc[daily.date.lt('2026-04-01'),'equity'].iloc[-1]),
                                                       daily.loc[daily.date.lt('2026-04-01'),'date'].iloc[-1])['total_return'],
}
(OUT/'ledger_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'ledger_audit.md').write_text(
    '# 新项目资金账本独立复核\n\n'
    f"13条策略资金曲线逐日日期完全一致，起止 {report['first']} 至 {report['last']}，共 {report['days']} 个交易日。全部13条曲线的收益、年化收益、最大回撤、夏普和平均仓位均由逐日总资产独立重算，与汇总表最大绝对差 {metrics_max_error:.2g}；各分段表最大差 {period_max_error:.2g}。\n\n"
    f"主账户收益 {report['total_return']:.2%}，平均股票仓位 {report['mean_exposure']:.2%}；现金最低 {report['minimum_cash']:.2f} 元，现金+股票市值与总资产最大误差 {report['maximum_cash_identity_error']:.2g} 元，逐日现金流水最大误差 {report['maximum_daily_cash_flow_error']:.2g} 元。11条真实现金账户均没有负现金，逐日现金流水全部对上。\n\n"
    f"主账户买入成交 {report['buy_fills']} 笔、卖出分笔 {report['sell_parts']} 笔；平均实际持有 {report['mean_holding_sessions_inclusive']:.3f} 个交易日（含买卖日），到期卖出分笔 {report['scheduled_exit_parts']}，止损分笔 {report['stopped_exit_parts']}，延期卖出分笔 {report['overdue_exit_parts']}，旧批未清而跳过的新批次数 {report['blocked_cohort_days']}。每笔成交核对了信号日、次日开盘买入、T+1 和到期卖出时点。\n\n"
    f"风险规则在信号日按当时已知的过去收益算目标仓位，重算最大误差 {target_max_error:.2g}；入场计划中行业、市值、动量、波动分组最大权重见 `ledger_audit.json`，均未超过预设额度。但持有期间重新按当天分组统计，行业最大 {max_actual_group_weight.get('industry',0):.2%}、市值最大 {max_actual_group_weight.get('size_group',0):.2%}、动量最大 {max_actual_group_weight.get('momentum_group',0):.2%}、波动最大 {max_actual_group_weight.get('volatility_group',0):.2%}；后三项会超过建仓时的50%额度。当前规则是信号日建仓约束和事后监测，不是任意时刻的硬减仓上限。\n\n"
    f"2026Q2 和 Q3 的规则是现金。2026-04-01 仍结清 Q1 持仓，较3月31日结算后收益 {report['recent_half_total_return']:.2%}；4月1日收盘后至9月30日总资产不变。\n",
    encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
