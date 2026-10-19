from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(__file__).resolve().parents[1];B=Path(__file__).parent;H=R/'training_runs/research_20261011/h5'
daily=pd.read_csv(H/'selected_daily.csv',parse_dates=['date']).set_index('date')
hold=pd.read_csv(H/'selected_holdings.csv',parse_dates=['date'])
orders=pd.read_csv(H/'selected_orders.csv',parse_dates=['date'])
def attrib(start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    v0=hold[hold.date.eq(start)].groupby('stockid').value.sum()
    v1=hold[hold.date.eq(end)].groupby('stockid').value.sum()
    q=orders[(orders.date>start)&(orders.date<=end)].copy()
    q['flow']=q.shares*q.price*np.where(q.side.eq('sell'),1,-1)
    f=q.groupby('stockid').flow.sum();fees=q.groupby('stockid').fee.sum()
    frame=pd.concat({'start_value':v0,'end_value':v1,'net_trade_cash':f,'fees':fees},axis=1).fillna(0)
    frame['price_pnl']=frame.end_value-frame.start_value+frame.net_trade_cash
    frame=frame.sort_values('price_pnl')
    g=daily.loc[(daily.index>start)&(daily.index<=end)]
    change=float(daily.loc[end,'equity']-daily.loc[start,'equity'])
    recon=float(frame.price_pnl.sum()-g.cost.sum()+g.gross_dividend_cash.sum())
    assert abs(change-recon)<.001,(change,recon)
    frame.to_csv(B/f'attribution_{start.date()}_{end.date()}.csv')
    rets=g.equity/daily.equity.shift().reindex(g.index)-1
    return dict(start=str(start.date()),end=str(end.date()),start_equity=float(daily.loc[start,'equity']),end_equity=float(daily.loc[end,'equity']),net_change=change,
       return_pct=change/daily.loc[start,'equity'],price_pnl=float(frame.price_pnl.sum()),fees=float(g.cost.sum()),dividend=float(g.gross_dividend_cash.sum()),
       top_losses=frame.head(5).reset_index().to_dict('records'),other_price_pnl=float(frame.iloc[5:].price_pnl.sum()),
       worst_day=str(rets.idxmin().date()),worst_day_return=float(rets.min()),mean_exposure=float(g.exposure.mean()),days=len(g))
peak=daily.equity.cummax();dd=daily.equity/peak-1;trough=dd.idxmin();peakdate=daily.loc[:trough].equity.idxmax()
out={'max_drawdown':attrib(peakdate,trough),'recent_half':attrib('2026-03-31','2026-09-30')}
periods=[]
for period,g in daily.groupby(daily.index.to_period('Q')):
    prev=daily.loc[daily.index<g.index.min()].equity
    initial=float(prev.iloc[-1]) if len(prev) else 1e6
    periods.append({'quarter':str(period),'return':float(g.equity.iloc[-1]/initial-1),'pnl':float(g.equity.iloc[-1]-initial),'exposure':float(g.exposure.mean())})
out['quarterly_account']=periods
ic=pd.read_csv(R/'学习笔记/20261015_真实日期逐步解释/10_用于选因子的每一天IC.csv',parse_dates=['date'])
out['factor046_windows']=[{'n':n,'start':str(ic.tail(n).date.min().date()),'end':str(ic.tail(n).date.max().date()),'mean_ic':float(ic.tail(n).code_IC.mean())} for n in [21,63,126,499]]
ds=pd.read_csv(R/'.ppt_build_20261016/h5_lgbm_daily_ic_groups.csv',parse_dates=['date'])
out['diagnostic_columns']=list(ds.columns)
qics=ds.groupby(ds.date.dt.to_period('Q')).agg({'ic':['mean','std','count']})
out['model_ic_quarters']=[{'quarter':str(k),'ic':float(v['ic']['mean']),'days':int(v['ic']['count'])} for k,v in qics.iterrows()]
cards=[]
for q in ['2026Q1','2026Q2','2026Q3']:
    card=pd.read_csv(H/f'{q}_deployment_factors.csv');cards.append(set(card.loc[card.selected,'factor']))
out['feature_overlap']={'Q1_Q2':len(cards[0]&cards[1]),'Q2_Q3':len(cards[1]&cards[2])}
out['short_windows']=json.loads((R/'training_runs/research_20261011/short_windows/summary.json').read_text())
(B/'evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
