"""Cash/share ledger with T+1, liquidity caps, deferred exits and explicit action support.

Actual runs have no verified corporate-action ledger and are price-only research accounts.
"""
from __future__ import annotations
import json,math
import numpy as np
import pandas as pd
from prepare import DATA,RUN,read_csv,dump

def metrics(daily,capital=1e6):
    equity=daily.equity;ret=equity.pct_change().fillna(0);years=max((daily.date.iloc[-1]-daily.date.iloc[0]).days/365.25,1/252)
    return {'total_return':float(equity.iloc[-1]/capital-1),'annual_return':float((equity.iloc[-1]/capital)**(1/years)-1),
        'sharpe':float(ret.mean()/ret.std()*np.sqrt(252)) if ret.std()>0 else 0,
        'max_drawdown':float((equity/equity.cummax()-1).min()),'mean_exposure':float(daily.gross_exposure.mean()),
        'turnover_sum':float(daily.turnover.sum()),'total_cost':float(daily.cost.sum()),'last_equity':float(equity.iloc[-1])}

def simulate(scored,prices,score='pred_selected',topn=5,cost_bp=15,cap_series=None,actions=None,capital=1e6,end_date=None):
    calendar=pd.DatetimeIndex(sorted(prices.date.unique()));ci={d:i for i,d in enumerate(calendar)}
    last_signal=calendar[-6] if end_date is None else calendar[ci[pd.Timestamp(end_date)]-5]
    sig=scored[(scored.date<=last_signal)].copy()
    if score not in sig:sig[score]=0.
    signals={d:g.sort_values([score,'stockid'],ascending=[False,True],na_position='last').head(topn)
        for d,g in sig.dropna(subset=[score]).groupby('date')}
    if not signals:raise ValueError('No signals')
    start=min(signals);end=pd.Timestamp(end_date) if end_date is not None else calendar[-1]
    days=calendar[(calendar>=start)&(calendar<=end)]
    columns=['open','close','high','low','amount','observed','adjustment_boundary']
    lookup={(r.date,r.stockid):r for r in prices[['date','stockid']+columns].itertuples(index=False)}
    caps={} if cap_series is None else cap_series.to_dict()
    action_map={}
    if actions is not None:
        for r in actions.itertuples(index=False):action_map[(pd.Timestamp(r.date),r.stockid)]=(float(r.dividend),float(r.split))
    cash=float(capital);positions=[];last={};trades=[];orders=[];daily=[];unfilled=0;attempted=0;event_exposure=0;action_cash=0
    for date in days:
        turn=0.;cost=0.;action_today=0.;daytrades=0
        # Explicit per-old-share dividends and split ratios only; never infer events from qfq.
        for pos in positions:
            if (date,pos['stockid']) in action_map:
                dividend,split=action_map[(date,pos['stockid'])]
                action_today+=pos['shares']*dividend;pos['shares']*=split
            row=lookup.get((date,pos['stockid']))
            if row is not None and row.adjustment_boundary:event_exposure+=1
        cash+=action_today;action_cash+=action_today
        def mark(pos,when):
            row=lookup.get((date,pos['stockid']));value=getattr(row,when,np.nan)
            return float(value) if np.isfinite(value) and value>0 else last.get(pos['stockid'],pos['entry_price'])
        def can_fill(sid,when,side):
            row=lookup.get((date,sid))
            if row is None or not row.observed:return False
            value=float(getattr(row,when));previous=last.get(sid)
            # Conservative limit-price proxy; actual ex-right reference and ST status are unavailable.
            limit=.295 if sid.startswith('BJ') else .095
            if previous is not None and (value>=previous*(1+limit) if side=='buy' else value<=previous*(1-limit)):return False
            return np.isfinite(value) and value>0
        def sell(pos,qty,when,reason):
            nonlocal cash,turn,cost,daytrades
            if date<=pos['entry_date'] or not can_fill(pos['stockid'],when,'sell'):return False
            price=mark(pos,when);qty=min(qty,pos['shares'])
            if qty<pos['shares']:
                qty=min(pos['shares'],max(100,math.floor(qty)))
                if not pos['stockid'].startswith('BJ'):qty=max(100,(qty//100)*100)
            if qty<=0:return False
            notional=qty*price;fee=notional*cost_bp/10000
            allocation=pos['entry_cost']*qty/pos['shares'];cash+=notional-fee;turn+=notional;cost+=fee;daytrades+=1
            trades.append({**pos,'shares':qty,'entry_cost':allocation,'exit_date':date,'exit_price':price,
                'proceeds':notional-fee,'net_return':(notional-fee)/allocation-1,'exit_reason':reason})
            orders.append({'date':date,'stockid':pos['stockid'],'side':'sell','shares':qty,'reference_price':price,'when':when,'cost':fee,'reason':reason})
            pos['shares']-=qty;pos['entry_cost']-=allocation
            if pos['shares']<=1e-9:positions.remove(pos)
            return True
        previous_day=calendar[ci[date]-1] if ci[date]>0 else None
        cap=float(caps.get(previous_day,1.));open_value=sum(p['shares']*mark(p,'open') for p in positions)
        equity_open=cash+open_value;excess=max(0,open_value-cap*equity_open)
        for pos in list(positions):
            if excess<=1:break
            price=mark(pos,'open');qty=min(pos['shares'],math.ceil(excess/price))
            if sell(pos,qty,'open','exposure_cap'):excess-=qty*price
        open_value=sum(p['shares']*mark(p,'open') for p in positions);equity_open=cash+open_value
        candidates=signals.get(previous_day)
        if candidates is not None:
            budget=max(0,min(equity_open*.2,cash,cap*equity_open-open_value));per_stock=budget/len(candidates)
            for r in candidates.itertuples(index=False):
                attempted+=1
                if not can_fill(r.stockid,'open','buy'):unfilled+=1;continue
                row=lookup[(date,r.stockid)];price=float(row.open)
                existing=sum(p['shares']*price for p in positions if p['stockid']==r.stockid)
                target=min(per_stock,max(0,.1*equity_open-existing),r.adv20*.01,cash/(1+cost_bp/10000))
                qty=math.floor(target/(price*(1+cost_bp/10000)))
                if not r.stockid.startswith('BJ'):qty=(qty//100)*100
                if qty<100:unfilled+=1;continue
                notional=qty*price;fee=notional*cost_bp/10000;cash-=notional+fee;turn+=notional;cost+=fee
                positions.append({'stockid':r.stockid,'signal_date':previous_day,'entry_date':date,
                    'target_exit_date':calendar[ci[previous_day]+5],'entry_price':price,'shares':float(qty),'entry_cost':notional+fee})
                orders.append({'date':date,'stockid':r.stockid,'side':'buy','shares':qty,'reference_price':price,'when':'open','cost':fee,'reason':'signal'})
        for pos in list(positions):
            if date>=pos['target_exit_date']:sell(pos,pos['shares'],'close','scheduled_exit')
        # Update all available marks, not just holdings, for the next day's limit proxy.
        for r in prices[prices.date==date][['stockid','close']].itertuples(index=False):
            if np.isfinite(r.close) and r.close>0:last[r.stockid]=float(r.close)
        value=sum(p['shares']*last.get(p['stockid'],p['entry_price']) for p in positions)
        equity=cash+value
        if cash < -1e-6:raise AssertionError('negative cash')
        daily.append({'date':date,'cash':cash,'market_value':value,'equity':equity,'gross_exposure':value/equity,
            'target_cap':cap,'turnover':turn/equity,'cost':cost,'action_cash':action_today,'n_lots':len(positions)})
    d=pd.DataFrame(daily);m=metrics(d,capital)
    m.update({'attempted_entries':attempted,'unfilled_entries':unfilled,'unfilled_fraction':unfilled/attempted,
              'completed_sales':len(trades),'open_lots_at_end':len(positions),'candidate_adjustment_held_lot_events':event_exposure,
              'corporate_action_cash':action_cash,'corporate_action_ledger_present':actions is not None,'cost_bp':cost_bp})
    return d,pd.DataFrame(trades),pd.DataFrame(orders),m

def benchmark(date_index,code):
    p=next((DATA/'benchmark').glob(code+'*'));d=read_csv(p)
    values=pd.Series(d['收盘'].to_numpy(),index=pd.to_datetime(d['日期'])).reindex(date_index)
    if values.isna().any():raise ValueError('missing benchmark test dates')
    return values/values.iloc[0]

def run_market(market):
    scored=pd.read_parquet(RUN/(market+'_predictions.parquet'))
    prices=pd.read_parquet(DATA/(market+'_prices.parquet'))
    caps=pd.read_csv(RUN/'hmm_probabilities.csv',parse_dates=['date']).set_index('date') if market=='bj' else None
    variants={c:(c,5,None) for c in scored if c.startswith('pred_')}
    variants['pool_equal']=('constant',100,None)
    if market=='bj':
        variants.update({'hmm_cap':('pred_selected',5,caps.hmm_cap),'fixed85_cap':('pred_selected',5,caps.fixed_cap),'vol_cap':('pred_selected',5,caps.vol_cap)})
    results={};curves=[]
    for name,(score,n,cap) in variants.items():
        d,t,o,m=simulate(scored,prices,score,n,15,cap,end_date=prices.loc[prices.observed,'date'].max())
        code='899050.BJ' if market=='bj' else '000300.SH'
        b=benchmark(pd.DatetimeIndex(d.date),code).to_numpy();d['benchmark_nav']=b;d['relative_nav']=(d.equity/1e6)/b
        m['benchmark_price_return']=float(b[-1]-1);m['excess_percentage_points']=m['total_return']-m['benchmark_price_return'];m['relative_total_return']=float(d.relative_nav.iloc[-1]-1)
        d.to_csv(RUN/(market+'_'+name+'_account.csv'),index=False);t.to_csv(RUN/(market+'_'+name+'_trades.csv'),index=False)
        o.to_csv(RUN/(market+'_'+name+'_orders.csv'),index=False)
        results[name]=m;curves.append(d[['date','equity']].rename(columns={'equity':name}))
        print('account',market,name,round(m['total_return'],4),flush=True)
    if market=='bj':
        for cost in [5,30]:
            d,t,o,m=simulate(scored,prices,'pred_selected',5,cost)
            results['selected_cost_'+str(cost)]=m;d.to_csv(RUN/('bj_selected_cost_'+str(cost)+'_account.csv'),index=False)
        # Repeated random rankings with predeclared seeds; same account mechanics.
        random_results=[]
        for seed in range(10):
            rng=np.random.default_rng(20260930+seed);random_scored=scored.copy();random_scored['random']=rng.random(len(scored))
            _,_,_,m=simulate(random_scored,prices,'random',5,15);random_results.append({'seed':seed,**m})
        pd.DataFrame(random_results).to_csv(RUN/'bj_random_controls.csv',index=False)
    dump(results,RUN/(market+'_account_metrics.json'))

if __name__=='__main__':
    run_market('bj');run_market('shsz')
