"""A real cash/share minute stress ledger, including rejected and partially filled orders."""
from pathlib import Path
import sys,json,math
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from prepare import DATA,dump
from accounts import simulate,metrics
OLD=ROOT/'training_runs/teacher9_20260930';RUN=ROOT/'training_runs/teacher9_20261008'

def minute_simulate(scored,prices,participation=.01,cost_bp=15):
    cal=pd.DatetimeIndex(sorted(prices.date.unique()));index={d:i for i,d in enumerate(cal)}
    end=cal[-1];lastsignal=cal[-6];fee=cost_bp/10000
    signals={d:g.sort_values(['pred_selected','stockid'],ascending=[False,True]).head(5) for d,g in scored[scored.date<=lastsignal].groupby('date')}
    quote={(r.date,r.stockid):r for r in prices[['date','stockid','open','close','observed']].itertuples(index=False)}
    cash=1e6;lots=[];last={};daily=[];orders=[];fills=[];lotid=0;max_capacity_error=0
    for date in cal:
        equity_before=cash+sum(p['shares']*last.get(p['stockid'],p['entry_price']) for p in lots)
        dayturn=0.;daycost=0.;capacity={};used={}
        path=ROOT/'training_data/research_20261008/minute'/f'{date:%Y%m%d}.parquet'
        if not path.exists():path=DATA/'minute'/f'{date:%Y%m%d}.parquet'
        if path.exists():
            frame=pd.read_parquet(path);frame['datetime']=pd.to_datetime(frame.datetime);frame['stockid']='BJ'+frame.code.astype(str)
            assert not frame.duplicated(['datetime','stockid']).any()
            good=frame[['open','high','low','close','volume']].notna().all(axis=1)&(frame.volume>0)&(frame[['open','high','low','close']]>0).all(axis=1)
            frame=frame[good].copy();frame['clock']=frame.datetime.dt.strftime('%H:%M')
            bars={sid:g.sort_values('datetime') for sid,g in frame.groupby('stockid')}
        else:bars={}
        def execute(sid,quantity,side,window,lot,notional_budget=math.inf):
            nonlocal cash,dayturn,daycost,max_capacity_error
            requested=int(quantity);remaining=requested;filled=0;value=0;reason='no_usable_bar'
            group=bars.get(sid)
            if group is not None:
                for bar in group[(group.clock>=window[0])&(group.clock<=window[1])].itertuples(index=False):
                    if remaining<=0:break
                    price=float(bar.high if side=='buy' else bar.low)
                    # Completed-bar worst-price scenario; volume is a fill constraint, not advance information.
                    previous=last.get(sid)
                    if previous and (price>=previous*1.295 if side=='buy' else price<=previous*.705):continue
                    key=(sid,bar.datetime);cap=int(float(bar.volume)*participation);capacity[key]=cap
                    qty=min(remaining,max(0,cap-used.get(key,0)))
                    if side=='buy':qty=min(qty,int(cash/(price*(1+fee))),int(max(0,notional_budget-value)/(price*(1+fee))))
                    if qty<=0:continue
                    amount=qty*price;cost=amount*fee
                    cash+=(-amount-cost if side=='buy' else amount-cost)
                    lot['shares']+=(qty if side=='buy' else -qty)
                    if side=='buy':lot['entry_value']+=amount+cost
                    dayturn+=amount;daycost+=cost;remaining-=qty;filled+=qty;value+=amount+cost if side=='buy' else amount
                    used[key]=used.get(key,0)+qty;max_capacity_error=max(max_capacity_error,used[key]-cap)
                    fills.append({'date':date,'datetime':bar.datetime,'stockid':sid,'side':side,'shares':qty,'price':price,'cost':cost,'lot_id':lot['id'],'entry_date':lot['entry_date'],'bar_volume':bar.volume,'capacity':cap,'used_capacity':used[key]})
                    reason='full' if remaining==0 else 'partial'
            orders.append({'date':date,'stockid':sid,'side':side,'requested':requested,'filled':filled,'remaining':remaining,'status':reason,'lot_id':lot['id']})
            assert cash>=-1e-6
        prev=cal[index[date]-1] if index[date]>0 else None
        candidates=signals.get(prev)
        if candidates is not None:
            openvalues={p['id']:p['shares']*getattr(quote.get((date,p['stockid'])),'open',last.get(p['stockid'],p['entry_price'])) for p in lots}
            equity_open=cash+sum(v if np.isfinite(v) else 0 for v in openvalues.values())
            batch=min(equity_open*.2,cash);allocation=batch/len(candidates)
            for row in candidates.itertuples(index=False):
                q=quote.get((date,row.stockid));op=getattr(q,'open',np.nan)
                existing=sum(openvalues.get(p['id'],0) for p in lots if p['stockid']==row.stockid)
                budget=max(0,min(allocation,.1*equity_open-existing,row.adv20*.01,cash/(1+fee)))
                quantity=int(budget/(op*(1+fee))) if np.isfinite(op) and op>0 else 0
                lotid+=1;lot={'id':lotid,'stockid':row.stockid,'shares':0,'entry_date':date,'target_exit':cal[index[prev]+5],'entry_price':op,'entry_value':0.}
                if quantity>=100:
                    execute(row.stockid,quantity,'buy',('09:31','10:00'),lot,budget)
                    if lot['shares']>0:lots.append(lot)
                else:orders.append({'date':date,'stockid':row.stockid,'side':'buy','requested':quantity,'filled':0,'remaining':quantity,'status':'below_minimum_or_invalid_open','lot_id':lotid})
        # Partial exits remain in the account and retry on each later day; T+1 enforced per lot.
        for lot in list(lots):
            if lot['target_exit']<=date and lot['entry_date']<date:execute(lot['stockid'],lot['shares'],'sell',('14:50','15:00'),lot)
            if lot['shares']==0:lots.remove(lot)
        for (d,sid),q in quote.items():
            if d==date and np.isfinite(q.close) and q.close>0:last[sid]=float(q.close)
        mv=sum(p['shares']*last.get(p['stockid'],p['entry_price']) for p in lots)
        equity=cash+mv
        daily.append({'date':date,'cash':cash,'market_value':mv,'equity':equity,'gross_exposure':mv/equity,'turnover':dayturn/equity_before,'cost':daycost,'n_lots':len(lots)})
    d=pd.DataFrame(daily);f=pd.DataFrame(fills);o=pd.DataFrame(orders);m=metrics(d)
    m.update({'participation':participation,'cost_bp':cost_bp,'orders':len(o),'filled_orders':int((o.filled>0).sum()),'requested_shares':float(o.requested.sum()),'filled_shares':float(o.filled.sum()),'share_fill_fraction':float(o.filled.sum()/o.requested.sum()),'open_lots':len(lots),'cash_identity_max_error':float((d.equity-d.cash-d.market_value).abs().max()),'capacity_overrun':max_capacity_error,'minimum_cash':float(d.cash.min()),'same_day_sales':int(((f.side=='sell')&(f.date<=f.entry_date)).sum()),'label':'Continuous cash ledger, price-only, conservative minute execution scenario; not order-book queue or commercial platform validation'})
    return d,f,o,m

def main():
    scored=pd.read_parquet(OLD/'bj_predictions.parquet');prices=pd.read_parquet(DATA/'bj_prices.parquet')
    q3='--q3' in sys.argv
    start,end=('2026-06-30','2026-09-29') if q3 else ('2025-09-30','2025-12-31')
    prefix='q3_' if q3 else ''
    prices=prices[(prices.date>=start)&(prices.date<=end)];scored=scored[(scored.date>=start)&(scored.date<=end)]
    results=[]
    d,_,_,m=simulate(scored,prices);d.to_csv(RUN/(prefix+'minute_matched_daily_account.csv'),index=False);m['scenario']='matched_daily';results.append(m)
    for participation in [.005,.01,.02]:
        d,f,o,m=minute_simulate(scored,prices,participation);name=prefix+f'minute_{participation*100:g}pct'
        d.to_csv(RUN/(name+'_account.csv'),index=False);f.to_csv(RUN/(name+'_fills.csv'),index=False);o.to_csv(RUN/(name+'_orders.csv'),index=False)
        m['scenario']=name;results.append(m);print(name,m['total_return'],m['share_fill_fraction'],flush=True)
    dump(results,RUN/(prefix+'minute_cash_results.json'))

if __name__=='__main__':main()
