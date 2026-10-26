"""Fixed meeting-driven ablations and causal model/risk selection."""
from pathlib import Path
import sys,json,hashlib
from dataclasses import asdict
import numpy as np
import pandas as pd
from execution import CohortMarket,Rule,period_metrics
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'training_runs/audit_rebuild_20261021'
DATA=ROOT/'training_data/audit_rebuild_20261021'
OUT=ROOT/'training_runs/meeting_update_20261023';OUT.mkdir(exist_ok=True)
COMMON=pd.Timestamp('2025-07-02');START='2024-10-08';END='2026-10-19'
RULES=[Rule('equal20',allow_residual=True),Rule('equal40',topn=40,allow_residual=True)]
def risk(name,**kw):return Rule(name,topn=40,diversify=True,allow_residual=True,**kw)
RULES += [risk('industry_size40'),risk('fixed25',exposure='fixed25'),risk('fixed50',exposure='fixed50'),
          risk('ma20',exposure='ma20'),risk('stop5_ban10',stop=.05,cooldown=10),
          risk('stop10_take20_ban10',stop=.10,take_profit=.20,cooldown=10),
          risk('ma20_stop5_ban10',exposure='ma20',stop=.05,cooldown=10)]

def load():
    p=pd.read_parquet(DATA/'original_inputs/prices.parquet')
    md=pd.read_parquet(OLD/'original/risk_metadata.parquet')
    m=CohortMarket(p,pd.read_csv(DATA/'original_inputs/actions.csv'),md[['date','stockid','industry','size_tertile']])
    a=pd.read_parquet(OLD/'original/daily_predictions.parquet');b=pd.read_parquet(OLD/'expanded/predictions.parquet')
    frames={'original_monthly':a[['date','stockid','score_monthly']].rename(columns={'score_monthly':'score'})}
    for name in ['alpha101_lgbm','expanded_lgbm','expanded_ridge_all']:
        frames[name]=b[['date','stockid',name]].rename(columns={name:'score'})
    for k in frames:frames[k]=frames[k].loc[frames[k].date>=COMMON].copy()
    return m,frames,p,md

def selection(dailies,cal):
    schedules={};logs=[]
    bi=cal.get_loc(START);ci=cal.get_loc(COMMON)
    for i in range(bi,len(cal),20):
        # Ten complete sessions strictly before selection are excluded.
        end=i-10;begin=max(ci,end-60)
        if end-begin<40:continue
        candidates=[]
        for key,d in dailies.items():
            part=d[(d.date>=cal[begin])&(d.date<cal[end])]
            ret=d.set_index('date').equity.pct_change().reindex(part.date)
            sd=ret.std(ddof=1)
            objective=float(ret.mean()/sd*np.sqrt(252)) if sd>1e-12 else -1e9
            if not len(part) or part.exposure.mean()<.01:objective=-1e9
            candidates.append((key,objective))
            logs.append(dict(decision_date=cal[i],candidate=key,validation_start=cal[begin],validation_end=cal[end-1],
                             gap_sessions=10,n=len(part),objective_sharpe0=objective))
        best=max(candidates,key=lambda x:x[1])[0]
        schedules[str(cal[i].date())]=best
    log=pd.DataFrame(logs)
    if len(log):log['selected']=log.apply(lambda x:schedules[str(x.decision_date.date())]==x.candidate,axis=1)
    return schedules,log

def switched_scores(frames,schedule,default):
    dates=sorted(set().union(*(set(f.date) for f in frames.values())))
    maps={n:{d:g for d,g in f.groupby('date')} for n,f in frames.items()}
    out=[]
    for d in dates:
        eligible=[k for k in schedule if pd.Timestamp(k)<=d]
        key=schedule[max(eligible)] if eligible else default
        model=key.split('__')[0]
        if d in maps[model]:out.append(maps[model][d])
    return pd.concat(out,ignore_index=True)

def save(result,key,fee):
    stem=f'{key}__fee{fee}'
    for table in ['daily','orders','trades','holdings','risk_exposure','rejected_selection']:
        frame=result[table]
        if len(frame):frame.to_parquet(OUT/f'{stem}_{table}.parquet',index=False)
    return result['daily']

def summarize(d,key,fee):
    rows=[]
    for period,a,b in [('common','2025-07-02',END),('V','2025-07-02','2026-05-21'),('test','2026-05-22',END),('Q2','2026-04-01','2026-06-30'),('Q3','2026-07-01','2026-09-30')]:
        r=period_metrics(d,a,b)
        if d[d.date.between(a,b)].equity.pct_change().std()<1e-12:r['sharpe_rf0']=r['sharpe_rf15']=None
        rows.append(dict(strategy=key,fee_bp=fee,period=period,**r))
    return rows

def main():
    m,frames,prices,md=load();summary=[];dailies={};checks=[]
    registry=pd.read_csv(OLD/'expanded/factor_registry.csv')
    registry.to_csv(OUT/'factor_registry.csv',index=False,encoding='utf-8-sig')
    pool=md[md.in_pool & (md.date>=COMMON)]
    coverage=dict(industry_known=float(pool.industry.ne('unknown').mean()),size_known=float(pool.size_tertile.notna().mean()),
                  factor_fields=len(registry),primary_fields=int(registry.primary_eligible.sum()),
                  data_end=str(prices.date.max().date()),factor_prediction_start={k:str(v.date.min().date()) for k,v in frames.items()})
    (OUT/'input_coverage.json').write_text(json.dumps(coverage,indent=2,ensure_ascii=False),encoding='utf-8')
    hashes={}
    for p in [DATA/'original_inputs/prices.parquet',OLD/'original/risk_metadata.parquet',OLD/'original/daily_predictions.parquet',OLD/'expanded/predictions.parquet',Path(__file__).with_name('execution.py'),Path(__file__),ROOT/'实验更新_20261023/实验协议.md']:
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2,ensure_ascii=False),encoding='utf-8')
    for name,scores in frames.items():
        for rule in RULES:
            key=name+'__'+rule.name
            result=m.run(scores,rule,start=START,end=END,save_details=True)
            d=save(result,key,5);dailies[key]=d;summary+=summarize(d,key,5)
            checks.append(dict(strategy=key,cash_identity_error=float((d.equity-d.cash-d.market_value).abs().max()),min_cash=float(d.cash.min())))
            print(key,flush=True)
    sch,log=selection(dailies,m.dates);log.to_csv(OUT/'selection_all_candidates.csv',index=False,encoding='utf-8-sig')
    (OUT/'selection_schedule.json').write_text(json.dumps(sch,indent=2),encoding='utf-8')
    default='original_monthly__equal20';scores=switched_scores(frames,sch,default)
    rulemap={r.name:r for r in RULES};schedule={k:rulemap[v.split('__')[1]] for k,v in sch.items()}
    for fee in [5,15]:
        result=m.run(scores,RULES[0],schedule=schedule,start=START,end=END,cost_bp=fee,save_details=True)
        d=save(result,'rolling_selected',fee);summary+=summarize(d,'rolling_selected',fee)
        # Same fixed baseline specifications, not reselected at the higher cost.
        for name in ['original_monthly','expanded_ridge_all']:
            for rule in [RULES[0],RULES[4]]:
                if fee==5:continue
                key=name+'__'+rule.name
                result=m.run(frames[name],rule,start=START,end=END,cost_bp=fee,save_details=True)
                d=save(result,key,fee);summary+=summarize(d,key,fee)
        print('rolling/cost',fee,flush=True)
    # Pool membership is known at each signal close; buy-hold uses only first cohort.
    ps=prices.loc[prices.in_pool & (prices.date>=COMMON),['date','stockid']].copy();ps['score']=0.
    for rule in [Rule('pool_equal100',topn=100,allow_residual=True),Rule('initial_pool_buyhold',topn=100,buy_hold=True,allow_residual=True)]:
        result=m.run(ps,rule,start=START,end=END,save_details=True)
        d=save(result,rule.name,5);summary+=summarize(d,rule.name,5)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT/'account_checks.csv',index=False,encoding='utf-8-sig')
    print('DONE',len(summary),'rows',len(sch),'selection periods',flush=True)

if __name__=='__main__':main()
