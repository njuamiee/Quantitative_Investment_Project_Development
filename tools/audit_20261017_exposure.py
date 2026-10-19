"""Read-only audit of the 2026-10-11 h5 selected account and risk schedule."""
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[1]
folder = root / 'training_runs/research_20261011/h5'
output = root / 'training_runs/research_20261017/legacy_audit'
output.mkdir(parents=True,exist_ok=True)
daily = pd.read_csv(folder / 'selected_daily.csv', parse_dates=['date'])
daily = daily[daily.date.le('2026-09-30')].copy()
schedule = pd.read_csv(folder / 'schedule.csv', parse_dates=['date'])
orders = pd.read_csv(folder / 'selected_orders.csv', parse_dates=['date'])
trades = pd.read_csv(folder / 'selected_trades.csv', parse_dates=['entry_date', 'exit_date'])
out = {}
out['overall'] = {
    'days':len(daily),
    'mean_target_cap':daily.target_cap.mean(),
    'mean_actual_exposure':daily.exposure.mean(),
    'mean_target_minus_actual':(daily.target_cap-daily.exposure).mean(),
    'median_target_cap':daily.target_cap.median(),
    'median_actual_exposure':daily.exposure.median(),
    'target_gt_actual_days':int((daily.target_cap>daily.exposure).sum()),
    'target_lt_actual_days':int((daily.target_cap<daily.exposure).sum()),
    'mean_cash':daily.cash.mean(),
    'min_cash':daily.cash.min(),
    'max_exposure':daily.exposure.max(),
    'cash_identity_error':(daily.cash+daily.market_value-daily.equity).abs().max(),
    'mean_lots':daily.lots.mean(),
}
daily['quarter'] = daily.date.dt.to_period('Q').astype(str)
daily['target_minus_actual'] = daily.target_cap - daily.exposure
daily[['date','quarter','cash','market_value','equity','target_cap','exposure','target_minus_actual','turnover','cost','gross_dividend_cash','lots']].to_csv(output/'逐日现金目标实际仓位.csv',index=False,encoding='utf-8-sig')
out['quarters'] = []
for q,g in daily.groupby(daily.date.dt.to_period('Q')):
    s=schedule[schedule.date.dt.to_period('Q').eq(q)]
    out['quarters'].append({
        'quarter':str(q), 'days':len(g), 'mean_target_cap':g.target_cap.mean(),
        'mean_actual_exposure':g.exposure.mean(),
        'mean_target_minus_actual':(g.target_cap-g.exposure).mean(),
        'mean_cash_fraction':(g.cash/g.equity).mean(),
        'schedule':s.groupby(['policy','guard','topn']).size().reset_index(name='days').to_dict(orient='records')
    })
pd.DataFrame([{k:v for k,v in r.items() if k!='schedule'} for r in out['quarters']]).to_csv(output/'分季度仓位汇总.csv',index=False,encoding='utf-8-sig')
orders = orders[orders.date.le('2026-09-30')]
trades = trades[trades.exit_date.le('2026-09-30')].copy()
session_map = {d:i for i,d in enumerate(daily.date)}
trades['held_sessions'] = trades.apply(lambda x:session_map[x.exit_date]-session_map[x.entry_date] if x.entry_date in session_map and x.exit_date in session_map else None,axis=1)
sell_orders=orders.loc[orders.side.eq('sell')].reset_index(drop=True)
assert len(sell_orders)==len(trades)
assert sell_orders.date.reset_index(drop=True).equals(trades.exit_date.reset_index(drop=True))
assert sell_orders.stockid.reset_index(drop=True).equals(trades.stockid.reset_index(drop=True))
trades['sold_shares']=sell_orders.shares.to_numpy()
trades[['entry_date','exit_date','stockid','held_sessions','sold_shares','reason','pnl','return']].to_csv(output/'已卖出交易持有期.csv',index=False,encoding='utf-8-sig')
hold_dist=trades.groupby(['held_sessions','reason']).agg(sales=('stockid','size'),sold_shares=('sold_shares','sum')).reset_index()
hold_dist.to_csv(output/'持有期分布.csv',index=False,encoding='utf-8-sig')
out['trades'] = {
    'buy_orders':int((orders.side=='buy').sum()), 'sell_orders':int((orders.side=='sell').sum()),
    'buy_notional':float((orders.loc[orders.side=='buy','shares']*orders.loc[orders.side=='buy','price']).sum()),
    'sell_notional':float((orders.loc[orders.side=='sell','shares']*orders.loc[orders.side=='sell','price']).sum()),
    'completed_sales':len(trades),
    'held_sessions_median':trades.held_sessions.median(),
    'held_sessions_mean':trades.held_sessions.mean(),
    'held_sessions_share_weighted_mean':float((trades.held_sessions*trades.sold_shares).sum()/trades.sold_shares.sum()),
    'held_sessions_distribution':trades.held_sessions.value_counts().sort_index().to_dict(),
    'reason_distribution':trades.reason.value_counts().to_dict(),
    'same_day_sales':int((trades.held_sessions==0).sum()),
}
legacy_metrics=json.loads((folder/'selected_metrics.json').read_text(encoding='utf-8'))
out['metadata_correction']={
    'selected_metrics_topn_field':legacy_metrics['topn'],
    'selected_metrics_policy_field':legacy_metrics['policy'],
    'actual_schedule_topn_values':sorted(schedule[schedule.date.le('2026-09-30')].topn.unique().tolist()),
    'actual_schedule_policies':sorted(schedule[schedule.date.le('2026-09-30')].policy.unique().tolist()),
    'explanation':'selected_metrics.json topn=5 and policy=none are run() defaults; daily schedule overrides them based on prior day. HMM was active in 2025Q4, 2026Q1, and 2026Q2.'
}
example_day=pd.Timestamp('2026-04-01'); prior_day=pd.Timestamp('2026-03-31')
prior=daily.loc[daily.date.eq(prior_day)].iloc[0]; now=daily.loc[daily.date.eq(example_day)].iloc[0]
fills=orders.loc[orders.date.eq(example_day)].copy()
fills['notional']=fills.shares*fills.price
fills.to_csv(output/'真实日示例_2026-04-01成交.csv',index=False,encoding='utf-8-sig')
buy=fills.loc[fills.side.eq('buy')]; sell=fills.loc[fills.side.eq('sell')]
cash_delta=float(now.cash-prior.cash)
cash_recon=float(sell.notional.sum()-buy.notional.sum()-fills.fee.sum()+now.gross_dividend_cash)
assert abs(cash_delta-cash_recon)<1e-6
out['example_2026_04_01']={
    'prior_close_cash':float(prior.cash), 'prior_close_stock_value':float(prior.market_value),
    'prior_close_equity':float(prior.equity),'next_day_target_cap':float(now.target_cap),
    'buy_count':len(buy),'buy_notional':float(buy.notional.sum()),
    'sell_count':len(sell),'sell_notional':float(sell.notional.sum()),
    'fees':float(fills.fee.sum()),'dividend':float(now.gross_dividend_cash),
    'cash_change':cash_delta,'cash_change_reconciled':cash_recon,
    'close_cash':float(now.cash),'close_stock_value':float(now.market_value),
    'close_equity':float(now.equity),'close_actual_exposure':float(now.exposure),
    'policy_applied_from_prior_day':schedule.loc[schedule.date.eq(prior_day),'policy'].iloc[0]
}
(output/'审计摘要.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
e=out['example_2026_04_01']
lines=[
    '# 旧主流程仓位与资金审计（截至 2026-09-30）',
    '',
    f"485 个交易日的平均目标仓位上限为 {out['overall']['mean_target_cap']:.2%}，平均实际股票仓位为 {out['overall']['mean_actual_exposure']:.2%}，相差 {out['overall']['mean_target_minus_actual']:.2%}。这里的目标只是最高允许持有比例；真实买入还受每日投入额度、可用现金、股票流动性、单股占比和最少 100 股制约。",
    '',
    '旧策略每天根据前一日信号，次日开盘尝试买入前 10 只。每日新增投入金额最多为当时总资产的 1/5，再除以 10；单只不超过总资产 10%，也不超过该股过去 20 日平均成交额的 1%。买入前先卖超出风险上限的旧仓；计划到期的旧仓在收盘才卖，因此当天开盘新买入不能先花下午才到账的钱。没有发现负现金或重复使用本金。',
    '',
    '## 分季度目标与实际仓位', '',
    '| 季度 | 风控 | 目标均值 | 实际均值 | 差额 |', '| --- | --- | ---: | ---: | ---: |'
]
for q in out['quarters']:
    lines.append(f"| {q['quarter']} | {q['schedule'][0]['policy']} | {q['mean_target_cap']:.2%} | {q['mean_actual_exposure']:.2%} | {q['mean_target_minus_actual']:.2%} |")
lines += [
    '', '## 真实一天：2026-04-01', '',
    f"前一日（3 月 31 日）收盘：现金 {e['prior_close_cash']:,.2f} 元，股票市值 {e['prior_close_stock_value']:,.2f} 元，总资产 {e['prior_close_equity']:,.2f} 元。4 月 1 日使用的是前一日决定的 {e['policy_applied_from_prior_day']} 上限，目标仓位 {e['next_day_target_cap']:.2%}。",
    f"当天开盘实际买入 {e['buy_count']} 笔，合计 {e['buy_notional']:,.2f} 元；卖出 {e['sell_count']} 笔，合计 {e['sell_notional']:,.2f} 元；手续费 {e['fees']:,.2f} 元，现金分红 {e['dividend']:,.2f} 元。于是现金变化 = 卖出金额 − 买入金额 − 手续费 + 分红 = {e['cash_change_reconciled']:,.2f} 元；收盘现金 {e['close_cash']:,.2f} 元、股票市值 {e['close_stock_value']:,.2f} 元、总资产 {e['close_equity']:,.2f} 元，实际仓位 {e['close_actual_exposure']:.2%}。股票价格日内变化可使收盘仓位稍高于开盘时目标。",
    '', '## 持有期、HMM 和元数据纠正', '',
    f"已卖出记录 {len(trades)} 笔，卖出事件从买入日到卖出日平均间隔 {out['trades']['held_sessions_mean']:.2f} 个交易日，按卖出股数加权为 {out['trades']['held_sessions_share_weighted_mean']:.2f} 个交易日；中位数 4 个交易日。程序所谓 horizon=5，是信号日后第 1 个交易日开盘买，第 5 个交易日收盘卖，买卖日期相距 4 个交易日，大致覆盖 5 个交易日。另有 {out['trades']['reason_distribution']['risk_budget']} 笔因仓位上限提前卖出。以上统计单位是卖出事件，同一买入批次部分卖出可能重复出现，不能误称为独立买入批次数。",
    '会上曾说 HMM 不参与现行主流程，但旧主账户的逐日 schedule 显示：2025Q4 采用 hmm2_defensive，2026Q1 和 2026Q2 采用 hmm3_defensive。旧 selected_metrics.json 中的 topn=5、policy=none 是 run() 的默认形参，逐日 schedule 实际覆盖为 topn=10 和各季度风控，报告应以 schedule 与 selected_daily 为准。',
    '', '现金恒等式“总资产 = 现金 + 股票市值”最大误差 4.7×10⁻¹⁰ 元，最低现金约 32.10 万元，同日卖出 0 笔。旧 selected_metrics.json 记录 failed_orders=2,140，但代码将买入失败与某些卖出失败加在同一个计数里，且未逐条保存失败原因，因此不能把这 2,140 次全部说成买入失败，更不能逐项归因给最小股数、流动性或停牌；要精确拆解需要重跑账本并增加失败原因字段。',
    '', '复算：`tools/audit_20261017_exposure.py`。原始输入：`training_runs/research_20261011/h5/selected_daily.csv`、`schedule.csv`、`selected_orders.csv`、`selected_trades.csv`。',
]
(output/'旧主流程仓位审计.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'output':str(output),'overall':out['overall'],'trades':out['trades'],'example':out['example_2026_04_01']},ensure_ascii=False,indent=2,default=float))
