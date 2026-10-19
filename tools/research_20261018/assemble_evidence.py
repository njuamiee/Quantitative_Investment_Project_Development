"""Common benchmarks, excess returns and reconciled drawdown attribution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from adaptive_execution import CohortMarket,Rule,period_metrics
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261018'
OUT=RUN/'evidence'

def drawdown_attribution(folder,outname,actions):
    daily=pd.read_csv(folder/'daily.csv',parse_dates=['date']).sort_values('date').reset_index(drop=True)
    orders=pd.read_csv(folder/'orders.csv',parse_dates=['date'])
    h=pd.read_csv(folder/'holdings.csv',parse_dates=['date'])
    e=daily.equity.to_numpy();with_initial=np.r_[1e6,e]
    dd=with_initial/np.maximum.accumulate(with_initial)-1
    end=int(dd.argmin());begin=int(with_initial[:end+1].argmax())
    baseline_date=daily.date.iloc[begin-1] if begin else daily.date.iloc[0]-pd.Timedelta(days=1)
    end_date=daily.date.iloc[end-1]
    chosen=orders[(orders.date>baseline_date)&(orders.date<=end_date)&(orders.filled_shares>0)].copy()
    chosen['notional']=chosen.filled_shares*chosen.price
    h0=h[h.date==baseline_date].groupby('stockid').value.sum()
    h1=h[h.date==end_date].groupby('stockid').value.sum()
    div={}
    for a in actions[(actions.date>baseline_date)&(actions.date<=end_date)].itertuples(index=False):
        previous=daily.loc[daily.date<a.date,'date']
        q=h.loc[(h.date==previous.iloc[-1])&(h.stockid==a.stockid),'shares'].sum() if len(previous) else 0
        div[a.stockid]=div.get(a.stockid,0.)+q*a.dividend
    table=pd.concat({'initial_value':h0,'final_value':h1,
          'buy_value':chosen[chosen.side=='buy'].groupby('stockid').notional.sum(),
          'sell_value':chosen[chosen.side=='sell'].groupby('stockid').notional.sum(),
          'fee':chosen.groupby('stockid').fee.sum(),'dividend':pd.Series(div,dtype=float)},axis=1).fillna(0)
    table['net_pnl']=table.final_value-table.initial_value+table.sell_value-table.buy_value-table.fee+table.dividend
    table=table[table.abs().sum(axis=1)>0].sort_values('net_pnl');table.index.name='stockid'
    delta=float(with_initial[end]-with_initial[begin])
    gap=float(table.net_pnl.sum()-delta)
    if abs(gap)>.02:raise AssertionError((outname,'attribution mismatch',gap))
    table.to_csv(OUT/(outname+'_drawdown_stocks.csv'),encoding='utf-8-sig')
    daily['day_pnl']=daily.equity.diff().fillna(daily.equity.iloc[0]-1e6)
    daily['day_return']=daily.equity.pct_change().fillna(daily.equity.iloc[0]/1e6-1)
    daily.nsmallest(10,'day_return').to_csv(OUT/(outname+'_worst_days.csv'),index=False,encoding='utf-8-sig')
    return {'strategy':outname,'peak_date':str(baseline_date.date()),'trough_date':str(end_date.date()),
            'peak_equity':float(with_initial[begin]),'trough_equity':float(with_initial[end]),
            'max_drawdown':float(-dd[end]),'equity_loss':delta,'attribution_sum':float(table.net_pnl.sum()),
            'identity_error':gap,'worst_stocks':table.head(5).reset_index().to_dict('records')}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=pd.read_parquet(ROOT/'training_data/research_20261011/prices.parquet',columns=['date','stockid','open','close','adv20','observed','mom20','vol20','in_pool','adj_ret1'])
    actions=pd.read_csv(ROOT/'training_data/research_20261011/actions.csv',parse_dates=['date'])
    market=CohortMarket(prices,actions)
    pool=prices.loc[prices.in_pool,['date','stockid']].assign(score=0.)
    benchmark_index=pd.read_csv(ROOT/'training_runs/research_20261011/bj50_benchmark.csv',parse_dates=['date'])
    experiments={'deep_risk':('all_daily.parquet','2024-10-08'),
                 'hmm_conditional':('account_daily.parquet','2024-10-08'),
                 'factor_loop':('account_daily.parquet','2026-01-05')}
    minute=RUN/'minute_incremental/account_daily.parquet'
    if minute.exists():experiments['minute_incremental']=('account_daily.parquet','2026-07-01')
    bases={};rows=[];quarterly=[]
    for start in sorted({v[1] for v in experiments.values()}):
        curves=[]
        for name,rule in [('equal_buy_hold',Rule('equal_buy_hold',topn=100,buy_hold=True)),('equal_pool_5day',Rule('equal_pool_5day',topn=100))]:
            result=market.run(pool,rule,start=start,end='2026-09-30',save_details=True)
            folder=OUT/'benchmarks'/start/name;folder.mkdir(parents=True,exist_ok=True)
            for key in ['daily','orders','trades','holdings']:result[key].to_csv(folder/(key+'.csv'),index=False,encoding='utf-8-sig')
            curves.append(result['daily'].assign(strategy=name))
        idx=benchmark_index[benchmark_index.date.between(start,'2026-09-30')].copy()
        idx['equity']=idx.equity/idx.equity.iloc[0]*1e6;idx['strategy']='bj50_price_index'
        curves.append(idx);bases[start]=pd.concat(curves,ignore_index=True)
        bases[start].to_parquet(OUT/('benchmarks_'+start+'.parquet'),index=False)
    for experiment,(filename,start) in experiments.items():
        all_daily=pd.read_parquet(RUN/experiment/filename)
        bench=bases[start]
        periods=[('full',start,'2026-09-30')]
        for key,a in [('recent_year','2025-10-01'),('recent_half','2026-04-01')]:
            if start<a:periods.append((key,a,'2026-09-30'))
        for name,d in all_daily.groupby('strategy'):
            for period,a,b in periods:
                m=period_metrics(d,a,b)
                record={'experiment':experiment,'strategy':name,'period':period,**m}
                for reference,g in bench.groupby('strategy'):
                    bm=period_metrics(g,a,b)
                    record['benchmark_return_'+reference]=bm['total_return']
                    record['excess_pp_'+reference]=m['total_return']-bm['total_return']
                    record['relative_growth_'+reference]=(1+m['total_return'])/(1+bm['total_return'])-1
                record['meets_targets']=bool(m['max_drawdown']<=.20 and m['sharpe_rf15']>1 and m['calmar'] is not None and m['calmar']>2)
                rows.append(record)
            for quarter,g in d.groupby(d.date.dt.to_period('Q')):
                m=period_metrics(d,g.date.min(),g.date.max())
                quarterly.append({'experiment':experiment,'strategy':name,'quarter':str(quarter),**m})
    pd.DataFrame(rows).to_csv(OUT/'all_periods_with_excess.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(quarterly).to_csv(OUT/'all_quarters.csv',index=False,encoding='utf-8-sig')
    attrs=[]
    targets=[('deep_risk','quarterly__past_validation_selected'),('deep_risk','simple__hmm3_gate'),
             ('deep_risk','monthly__breadth'),('factor_loop','augmented__equal'),('hmm_conditional','hmm3__hmm3_gate')]
    for experiment,strategy in targets:
        attrs.append(drawdown_attribution(RUN/experiment/'accounts'/strategy,experiment+'__'+strategy,actions))
    (OUT/'drawdown_attribution.json').write_text(json.dumps(attrs,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EVIDENCE_COMPLETE',len(rows),'period rows',len(attrs),'reconciled drawdowns',flush=True)

if __name__=='__main__':main()
