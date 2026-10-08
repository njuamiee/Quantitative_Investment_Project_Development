"""Order-level conservative minute replay and a platform-neutral signal handoff."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from prepare import DATA,RUN,dump

def fill_order(bars,shares,side,participation=.01,limit_price=None):
    remaining=float(shares);fills=[]
    for row in bars.sort_values('datetime').itertuples(index=False):
        if remaining<=0:break
        values=[row.open,row.high,row.low,row.close,row.volume]
        if any(pd.isna(v) for v in values) or row.volume<=0 or min(values[:4])<=0:continue
        # Order exists before this bar; use worst price in completed bar, not its open.
        price=float(row.high if side=='buy' else row.low)
        if limit_price is not None and (price>limit_price if side=='buy' else price<limit_price):continue
        capacity=int(row.volume*participation);qty=min(remaining,capacity)
        if qty<=0:continue
        fills.append({'datetime':row.datetime,'shares':qty,'price':price});remaining-=qty
    value=sum(r['shares']*r['price'] for r in fills);filled=shares-remaining
    return {'requested':shares,'filled':filled,'remaining':remaining,'vwap':value/filled if filled else np.nan,'fills':fills}

def main():
    orders=pd.read_csv(RUN/'bj_pred_selected_orders.csv',parse_dates=['date'])
    orders=orders[(orders.date>='2025-10-01')&(orders.date<='2025-12-31')].copy()
    # One report per executed daily-reference order; unfilled daily candidates are not reintroduced.
    rows=[];fills=[];quality=[];units=[]
    prices=pd.read_parquet(DATA/'bj_prices.parquet').set_index(['date','stockid'])
    for p in sorted((DATA/'minute').glob('*.parquet')):
        date=pd.to_datetime(p.stem);frame=pd.read_parquet(p)
        frame['code']=frame.code.astype(str);frame['datetime']=pd.to_datetime(frame.datetime)
        duplicate=int(frame.duplicated(['datetime','code']).sum())
        if duplicate:raise ValueError(f'duplicate minute keys {p.name}')
        active=frame[['open','high','low','close','volume']].notna().all(axis=1)&(frame.volume>0)
        quality.append({'date':date,'rows':len(frame),'securities':frame.code.nunique(),'usable_rows':int(active.sum()),
                        'unusable_fraction':float((~active).mean()),'duplicate_keys':duplicate})
        for code,g in frame[frame.code.str.startswith('920')].groupby('code'):
            key=(date,'BJ'+code)
            if key not in prices.index:continue
            daily=prices.loc[key]
            if daily.volume>0:
                units.append({'date':date,'stockid':key[1],'minute_volume_sum':g.volume.sum(),'daily_volume':daily.volume,
                    'minute_to_daily_volume_ratio':g.volume.sum()/daily.volume,
                    'amount_ratio':g.amount.sum()/daily.amount if daily.amount>0 else np.nan,
                    'close_difference':g.close.dropna().iloc[-1]-daily.close if g.close.notna().any() else np.nan})
        used_capacity={}
        for idx,order in orders[orders.date==date].iterrows():
            subset=frame[frame.code==order.stockid[2:]].copy()
            clock=subset.datetime.dt.strftime('%H:%M')
            # 30-minute morning execution window; end-of-day exits use the final ten minutes.
            if order.when=='open':subset=subset[(clock>='09:31')&(clock<='10:00')]
            else:subset=subset[(clock>='14:50')&(clock<='15:00')]
            # Shared per-bar capacity across all same-stock orders, including several held batches.
            for i,row in subset.iterrows():
                key=(order.stockid,row.datetime)
                if pd.notna(row.volume):subset.at[i,'volume']=max(0,float(row.volume)-used_capacity.get(key,0)/.01)
            result=fill_order(subset,float(order.shares),order.side)
            for fill in result.pop('fills'):
                key=(order.stockid,fill['datetime']);used_capacity[key]=used_capacity.get(key,0)+fill['shares']
                fills.append({'order_id':int(idx),'stockid':order.stockid,'side':order.side,**fill})
            penalty=(result['vwap']/order.reference_price-1)*(1 if order.side=='buy' else -1)
            rows.append({'order_id':int(idx),'date':date,'stockid':order.stockid,'side':order.side,
                'reference_price':order.reference_price,'adverse_price_difference':penalty,**result})
    report=pd.DataFrame(rows);report.to_csv(RUN/'minute_order_replay.csv',index=False)
    pd.DataFrame(fills).to_csv(RUN/'minute_fills.csv',index=False)
    pd.DataFrame(quality).to_csv(RUN/'minute_quality.csv',index=False)
    unit=pd.DataFrame(units);unit.to_csv(RUN/'daily_minute_unit_reconciliation.csv',index=False)
    summary={'minute_files_checked':len(quality),'quarter_orders':len(orders),'orders_with_minute_files':len(report),
        'requested_shares':float(report.requested.sum()),'filled_shares':float(report.filled.sum()),
        'share_fill_fraction':float(report.filled.sum()/report.requested.sum()),
        'full_fill_order_fraction':float((report.remaining==0).mean()),'unfilled_order_fraction':float((report.filled==0).mean()),
        'median_adverse_price_difference':float(report.adverse_price_difference.median()),
        'volume_ratio_median':float(unit.minute_to_daily_volume_ratio.median()),
        'volume_ratio_p05':float(unit.minute_to_daily_volume_ratio.quantile(.05)),
        'volume_ratio_p95':float(unit.minute_to_daily_volume_ratio.quantile(.95)),
        'amount_ratio_median':float(unit.amount_ratio.median()),
        'scope':'order-level stress replay, not a second cash-account return; worst bar price, 1% realized volume, no queue model; no commercial vn.py installation'}
    dump(summary,RUN/'minute_replay_summary.json')
    scored=pd.read_parquet(RUN/'bj_predictions.parquet')
    signal=scored.sort_values(['date','pred_selected'],ascending=[True,False]).groupby('date').head(5).copy()
    signal['signal_available_at']=signal.date+pd.Timedelta(hours=18)
    signal['earliest_execution_date']=signal.entry_date
    signal.loc[signal.date==pd.Timestamp('2026-09-29'),'earliest_execution_date']=pd.Timestamp('2026-09-30')
    signal['model_version']='teacher9_20260930/'+signal.date.dt.to_period('Q').astype(str)
    signal['target_new_batch_weight']=.04
    signal['horizon']='exit t+5 close; delayed if not tradable'
    signal['research_only']=True
    signal=signal[['date','stockid','pred_selected','signal_available_at','earliest_execution_date','model_version','target_new_batch_weight','horizon','research_only']]
    signal.to_csv(RUN/'execution_signals.csv',index=False,encoding='utf-8-sig')
    signal[signal.date==signal.date.max()].to_csv(RUN/'latest_research_signals.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
