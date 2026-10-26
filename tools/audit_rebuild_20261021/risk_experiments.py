"""Fixed risk ablations on identical scores; separate alpha/factor experiments."""
from pathlib import Path
import sys,json
from dataclasses import replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'training_data/audit_rebuild_20261021'
RUN=ROOT/'training_runs/audit_rebuild_20261021';REBALANCED='--rebalanced' in sys.argv
OUT=RUN/('risk_rebalanced' if REBALANCED else 'risk');OUT.mkdir(exist_ok=True)
from risk_execution import CohortMarket,Rule,period_metrics

RULES=[Rule('equal20'),Rule('div20',diversify=True),Rule('div40',topn=40,diversify=True),
       Rule('div40_trend',topn=40,diversify=True,exposure='soft_trend'),
       Rule('div40_vol15',topn=40,diversify=True,exposure='portfolio_vol'),
       Rule('div40_combined',topn=40,diversify=True,exposure='combined'),
       Rule('div40_fixed25',topn=40,diversify=True,exposure='fixed25'),
       Rule('div40_fixed50',topn=40,diversify=True,exposure='fixed50')]
if REBALANCED:RULES=[replace(r,allow_residual=True) for r in RULES]

def adjusted_period(d,a,b):
    result=period_metrics(d,a,b)
    part=d[d.date.between(a,b)]
    # Original engine gives Sharpe=0 for no variance. Expose undefined as null here.
    if len(part)>1 and part.equity.pct_change().dropna().std()<1e-12:
        result['sharpe_rf15']=None;result['sharpe_rf0']=None
    return result

def main():
    prices=pd.read_parquet(DATA/'original_inputs/prices.parquet')
    actions=pd.read_csv(DATA/'original_inputs/actions.csv');md=pd.read_parquet(RUN/'original/risk_metadata.parquet')
    # Only descriptor columns read by _groups; avoid retaining unrelated source strings in every dict.
    md=md[['date','stockid','industry','size_tertile','momentum_tertile','volatility_tertile']]
    market=CohortMarket(prices,actions,md)
    original=pd.read_parquet(RUN/'original/daily_predictions.parquet')
    expanded=pd.read_parquet(RUN/'expanded/predictions.parquet')
    jobs=[]
    for model,p,cols in [('original',original,['score_monthly']),('expanded',expanded,[c for c in expanded if c not in ['date','stockid','label_5']])]:
        for col in cols:
            name='original_monthly' if model=='original' else col
            rules=RULES if name in ['original_monthly','expanded_lgbm'] else [RULES[0],RULES[5]]
            for rule in rules:jobs.append((name,p[['date','stockid',col]].rename(columns={col:'score'}),rule,5.,0))
            if name in ['original_monthly','expanded_lgbm']:
                for fee,delay in [(0.,0),(15.,0),(5.,5)]:jobs.append((name,p[['date','stockid',col]].rename(columns={col:'score'}),RULES[5],fee,delay))
    summary=[];daily=[];diagnosis=[]
    for k,(name,p,rule,fee,delay) in enumerate(jobs):
        strategy=f'{name}__{rule.name}__cost{fee:g}__delay{delay}'
        path=OUT/(strategy+'_daily.csv')
        if path.exists():d=pd.read_csv(path,parse_dates=['date']);extra=json.loads((OUT/(strategy+'_diagnosis.json')).read_text(encoding='utf-8'))
        else:
            # Move the sparse forecast by one complete rebalance cycle, preserving grid alignment.
            if delay:
                cal=market.dates;idx=cal.get_indexer(p.date);ok=idx+delay<len(cal);p=p.loc[ok].copy();p.date=cal[idx[ok]+delay]
            r=market.run(p,rule,start='2024-10-08',end='2026-10-19',cost_bp=fee,save_details=True)
            d=r['daily'];d.to_csv(path,index=False,encoding='utf-8-sig')
            h=r['holdings'];rx=r['risk_exposure'];sel=r['orders']
            if len(h):
                h=h.merge(d[['date','equity']],on='date');h['weight']=h.value/h.equity
                ww=h.groupby(['date','stockid']).weight.sum();concentration=ww.groupby(level=0).apply(lambda w:(w*w).sum())
                invested=h.groupby('date').weight.sum();effective=invested**2/concentration
                max_stock=float(ww.max());median_n=float(effective.median())
            else:max_stock=0.;median_n=0.
            ind=rx[rx.dimension.eq('industry')] if len(rx) else rx
            extra=dict(strategy=strategy,max_stock_weight=max_stock,median_effective_holdings=median_n,
                       max_industry_weight=float(ind.weight.max()) if len(ind) else 0.,
                       cash_identity_error=float((d.equity-d.cash-d.market_value).abs().max()),
                       latency_sessions=delay,
                       warning='Delay5 moves forecasts by one 5-session rebalance cycle, not one session.')
            (OUT/(strategy+'_diagnosis.json')).write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
            if name in ['original_monthly','expanded_lgbm'] and rule.name in ['equal20','div40_combined'] and fee==5 and delay==0:
                r['orders'].to_csv(OUT/(strategy+'_orders.csv'),index=False,encoding='utf-8-sig')
                r['trades'].to_csv(OUT/(strategy+'_trades.csv'),index=False,encoding='utf-8-sig')
                r['risk_exposure'].to_csv(OUT/(strategy+'_group_weights.csv'),index=False,encoding='utf-8-sig')
            assert extra['cash_identity_error']<1e-5
        diagnosis.append(extra)
        for period,a,b in [('V','2025-07-02','2026-05-21'),('test','2026-05-22','2026-10-19'),('common','2025-07-02','2026-10-19'),('Q2','2026-04-01','2026-06-30'),('Q3','2026-07-01','2026-09-30')]:
            summary.append(dict(strategy=strategy,period=period,**adjusted_period(d,a,b)))
        if name=='original_monthly':summary.append(dict(strategy=strategy,period='old_full',**adjusted_period(d,'2024-10-08','2026-09-30')))
        d['strategy']=strategy;daily.append(d)
        print(k+1,len(jobs),strategy,flush=True)
    s=pd.DataFrame(summary);s.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnosis).to_csv(OUT/'concentration.csv',index=False,encoding='utf-8-sig')
    pd.concat(daily).to_parquet(OUT/'all_daily.parquet',index=False)
    # Paired 20-session block bootstrap over the observed test; uncertainty, not a fresh holdout.
    rng=np.random.default_rng(20261021);boot=[]
    strategies={d.strategy.iloc[0]:d for d in daily}
    base=strategies['alpha101_lgbm__equal20__cost5__delay0'];br=base.equity.pct_change().fillna(0).to_numpy()
    test=base.date.between('2026-05-22','2026-10-19').to_numpy()
    for name,d in strategies.items():
        if name.startswith('original') or '__cost5__delay0' not in name:continue
        excess=(d.equity.pct_change().fillna(0).to_numpy()-br)[test];n=len(excess)
        sims=[]
        for _ in range(1000):
            starts=rng.integers(0,n,size=int(np.ceil(n/20)));ii=np.concatenate([(np.arange(20)+j)%n for j in starts])[:n]
            sims.append(excess[ii].mean()*252)
        boot.append(dict(strategy=name,baseline='alpha101_lgbm equal20',annual_mean_return_difference=excess.mean()*252,
                         ci025=float(np.quantile(sims,.025)),ci975=float(np.quantile(sims,.975)),samples=1000,block_sessions=20))
    pd.DataFrame(boot).to_csv(OUT/'test_bootstrap.csv',index=False,encoding='utf-8-sig')
    print(s[s.period.eq('test')][['strategy','total_return','max_drawdown','sharpe_rf15','mean_exposure']].to_string(index=False),flush=True)

if __name__=='__main__':main()
