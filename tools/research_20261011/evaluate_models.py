"""All fixed comparisons, causal-selection metrics, benchmarks and uncertainty."""
import argparse,json
import numpy as np
import pandas as pd
from common import ROOT,DATA,RUN,HORIZONS,dump,SEED
from ledger import Market,metrics
from train_models import MODELS,ic_value

POLICIES=['none','fixed50','vol12','vol20','trend_vol','hmm2_defensive','hmm2_volatility','hmm3_defensive','hmm3_volatility','hmm3_growth']

def block_interval(daily,replicates=1500,block=20):
    equity=daily.equity.to_numpy();r=np.r_[equity[0]/1e6-1,equity[1:]/equity[:-1]-1]
    rng=np.random.default_rng(SEED);samples=[];n=len(r)
    for _ in range(replicates):
        starts=rng.integers(n,size=int(np.ceil(n/block)));idx=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        z=r[idx];e=np.cumprod(1+z);dd=1-e/np.maximum.accumulate(np.r_[1,e])[1:];annual=e[-1]**(252/n)-1
        sh=(z.mean()-(1.015**(1/252)-1))/z.std(ddof=1)*np.sqrt(252) if z.std()>0 else 0
        samples.append([sh,annual/dd.max() if dd.max()>0 else np.nan,annual])
    a=np.array(samples)
    return {'method':'circular moving-block bootstrap; conditional on fixed strategy, not research-selection corrected','block_sessions':block,'replicates':replicates,
            'sharpe_rf15_95':np.nanquantile(a[:,0],[.025,.975]).tolist(),'calmar_95':np.nanquantile(a[:,1],[.025,.975]).tolist(),
            'annual_return_95':np.nanquantile(a[:,2],[.025,.975]).tolist(),'bootstrap_joint_pass_fraction':float(((a[:,0]>1)&(a[:,1]>2)).mean())}

def period_metrics(d):
    rows=[]
    for period,g in d.groupby(d.date.dt.to_period('Q')):
        first=g.index[0];initial=float(d.loc[first-1,'equity']) if first>0 else 1e6
        m=metrics(g,initial);m['period']=str(period);m['short_period_annualization_not_gate']=len(g)<63;rows.append(m)
    return rows

def main(horizons):
    p=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','amount','adv20','observed','vol20','mom20','in_pool']);a=pd.read_csv(DATA/'actions.csv');reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date');market=Market(p,a,reg)
    for h in horizons:
        folder=RUN/f'h{h}';q=pd.read_parquet(folder/'predictions.parquet');schedule=pd.read_csv(folder/'schedule.csv',parse_dates=['date']).set_index('date')
        start=q.date.min();end=q.date.max();rows=[];navs=[]
        # Fixed methods are exploratory outer comparisons. They do not change validation selection.
        for model in MODELS:
            rank_ic=ic_value(q,q[model],h)
            for policy in POLICIES:
                d,m,*_=market.run(q,horizon=h,score=model,topn=20,policy=policy,start=start,end=end)
                m.update(model=model,selection='fixed exploratory comparison',rank_ic=rank_ic,cumulative_return_over_mdd=m['total_return']/m['max_drawdown'] if m['max_drawdown'] else None)
                rows.append(m);d['strategy']=model+'__'+policy;navs.append(d)
            print('compared',h,model,flush=True)
        pd.DataFrame(rows).to_csv(folder/'fixed_comparisons.csv',index=False);pd.concat(navs).to_parquet(folder/'fixed_daily.parquet',index=False)
        selected=pd.read_csv(folder/'selected_daily.csv',parse_dates=['date']);dump(period_metrics(selected),folder/'quarterly_metrics.json')
        mainperiod=selected[selected.date<='2026-09-30'].copy();mainmetrics=metrics(mainperiod)
        mainmetrics.update(horizon=h,model='validation_selected',label='Retrospective walk-forward through 2026Q3; previously inspected history',
                           cumulative_return_over_mdd=mainmetrics['total_return']/mainmetrics['max_drawdown'])
        dump(mainmetrics,folder/'main_period_metrics.json');dump(block_interval(mainperiod),folder/'bootstrap.json')
        sensitivity=[]
        for fee in [0,1,2,5,15]:
            _,m,*_=market.run(q,horizon=h,schedule=schedule,start=start,end=end,cost_bp=fee);m['scenario']='same signals and schedule';sensitivity.append(m)
        _,m,*_=market.run(q,horizon=h,schedule=schedule,start=start,end=end,cost_bp=5,actions=False);m['scenario']='raw prices without corporate actions';sensitivity.append(m)
        dump(sensitivity,folder/'sensitivity.json')
        # Same tradability/account rules, equal allocation across the full eligible pool.
        equal=p.loc[p.in_pool & p.date.between(start,end),['date','stockid']].copy();equal['score']=0.
        d,m,*_=market.run(equal,horizon=h,topn=100,start=start,end=end);m['benchmark']='rolling cohorts of full pool equal-weight';dump(m,folder/'equal_pool_metrics.json');d.to_csv(folder/'equal_pool_daily.csv',index=False)
        # Reproduce a concrete final fully labeled signal day with next-session orders.
        signal_date=q.loc[q[f'label_{h}'].notna(),'date'].max();chosen=schedule.loc[signal_date]
        picks=q[q.date==signal_date].sort_values(['score','stockid'],ascending=[False,True]).head(int(chosen.topn))
        picks.to_csv(folder/'example_selection.csv',index=False)
        # Realized mean-exposure matched constants: diagnostic fitted on evaluation, never a deployable selector.
        if h==5:
            matched=[]
            for policy in ['hmm2_defensive','hmm3_defensive','hmm3_volatility']:
                base=next(x for x in rows if x['model']=='ensemble' and x['policy']==policy);target=base['mean_exposure'];lo,hi=0.,1.
                for _ in range(10):
                    cap=(lo+hi)/2;reg['cap_matched']=cap;market.regimes=reg
                    d,m,*_=market.run(q,horizon=h,score='ensemble',topn=20,policy='matched',start=start,end=end)
                    if m['mean_exposure']<target:lo=cap
                    else:hi=cap
                m.update(model='ensemble',matched_policy=policy,constant_cap=cap,diagnostic_only=True,
                         delta_sharpe_hmm_minus_fixed=base['sharpe_rf15']-m['sharpe_rf15'],delta_calmar_hmm_minus_fixed=base['calmar']-m['calmar'],
                         delta_drawdown_hmm_minus_fixed=base['max_drawdown']-m['max_drawdown'])
                matched.append(m)
            dump(matched,folder/'hmm_matched_exposure.json')
        print('evaluation complete',h,'selected',mainmetrics['sharpe_rf15'],mainmetrics['calmar'],flush=True)
    # Price-only index benchmark; source index close, no tradability assumption.
    import sys
    sys.path.insert(0,str(ROOT/'tools/research_20260930'));from prepare import read_csv
    b=read_csv(next((DATA/'benchmark').glob('899050.BJ*')));dates=pd.to_datetime(b['日期']);close=pd.to_numeric(b['收盘'])
    series=pd.Series(close.to_numpy(),index=dates).sort_index();series=series[~series.index.duplicated()]
    idx=market.dates[(market.dates>='2024-10-08')&(market.dates<=pd.Timestamp('2026-10-09'))];prior=series[series.index<idx[0]].iloc[-1]
    d=pd.DataFrame({'date':idx,'equity':series.reindex(idx).to_numpy()/prior*1e6,'exposure':1.,'turnover':0.,'cost':0.})
    if d.equity.isna().any():raise ValueError('Benchmark missing dates')
    d.to_csv(RUN/'bj50_benchmark.csv',index=False);dump(metrics(d),RUN/'bj50_benchmark_metrics.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--horizons',default='1,2,5,10,20');args=parser.parse_args();main([int(x) for x in args.horizons.split(',')])
