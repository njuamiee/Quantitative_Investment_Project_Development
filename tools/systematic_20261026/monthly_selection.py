"""Choose with past OOS account returns, then replay one continuous cash account."""
import json
import numpy as np
import pandas as pd
from config import RUN,START,END,dump
from account import market_inputs,score_order,ledger,RISK_AXES,metrics

def param(risk):
    return [risk['topn']]+[RISK_AXES[a].index(risk[a]) for a in ['weighting','exposure','regime','exit','concentration','entry']]+[risk['pool'],risk['rebalance_days']]

def completed():
    records=[]
    for p in sorted((RUN/'models').glob('*/account_complete.json')):
        records.append(json.loads(p.read_text()))
    return records

def select_and_replay(records,save=True):
    cal,m,args=market_inputs();days=cal[(cal>=START)&(cal<=END)]
    decisions=np.flatnonzero(np.r_[True,days.to_period('M')[1:]!=days.to_period('M')[:-1]])
    best=[None]*len(decisions);bestscore=np.zeros(len(decisions));incumbent=None
    objectives={};registries={};configmap={r['job']:r['config'] for r in records}
    for record in records:
        folder=RUN/'models'/record['job'];a=np.load(folder/'risk_results.npz')
        score=a['objectives']
        for j in range(len(decisions)):
            k=int(np.argmax(score[:,j]));v=float(score[k,j])
            if v>bestscore[j]:bestscore[j]=v;best[j]=(record['job'],k)
    logs=[];chosen=[]
    for j,cut in enumerate(decisions):
        new=best[j];reason='past_63_126_day_maximin'
        if new is not None and incumbent is not None:
            if incumbent[0] not in objectives:
                objectives[incumbent[0]]=np.load(RUN/'models'/incumbent[0]/'risk_results.npz')['objectives']
            oldscore=float(objectives[incumbent[0]][incumbent[1],j])
            if oldscore>0 and bestscore[j]-oldscore<=.10:new=incumbent;reason='incumbent_within_0.10'
        if new is None:reason='no_positive_eligible_validation_candidate'
        if new is not None and new[0] not in objectives:
            objectives[new[0]]=np.load(RUN/'models'/new[0]/'risk_results.npz')['objectives']
        incumbent=new;chosen.append(new)
        logs.append(dict(signal_month=str(days[cut].to_period('M')),first_signal=str(days[cut].date()),
                         last_validation_close=str(days[cut-1].date()) if cut else None,
                         candidate=None if new is None else new[0],risk_id=None if new is None else new[1],reason=reason,
                         objective=None if new is None else float(objectives[new[0]][new[1],j]),
                         candidate_models_available=len(records),candidate_accounts_available=sum(r['completed_accounts'] for r in records)))
    predcache={};scoreids=np.full((len(cal),100),-1,np.int32)
    # A score of zero with a fixed ID tie break represents the dynamic pool baseline.
    meta=pd.read_parquet(RUN/'data/meta.parquet');base=meta[meta.date.between(START,END)].copy();base['score']=0.
    poolorder=score_order(base,cal,m['open'].shape[1])
    commonrisk=[100,0,0,0,0,0,0,100,5]
    schedule=np.tile(np.asarray(commonrisk+[0],float),(len(cal),1))
    cashschedule=schedule.copy()
    for j,cut in enumerate(decisions):
        begin=cal.get_loc(days[cut]);finish=cal.get_loc(days[decisions[j+1]]) if j+1<len(decisions) else len(cal)
        item=chosen[j]
        if item is None:
            scoreids[begin:finish]=poolorder[begin:finish]
            schedule[begin:finish,:9]=commonrisk;cashschedule[begin:finish,:9]=commonrisk
            cashschedule[begin:finish,2]=3
        else:
            job,riskid=item
            if job not in predcache:
                p=pd.read_parquet(RUN/'models'/job/'predictions.parquet');predcache[job]=score_order(p,cal,m['open'].shape[1])
                registries[job]=json.loads((RUN/'models'/job/'risk_registry.json').read_text())
            scoreids[begin:finish]=predcache[job][begin:finish]
            risk=registries[job][riskid];parameters=param(risk)
            schedule[begin:finish,:9]=parameters;cashschedule[begin:finish,:9]=parameters
            logs[j]['model_config']=configmap[job];logs[j]['risk_config']=risk
        schedule[begin:finish,9]=j+1;cashschedule[begin:finish,9]=j+1
    begin=cal.get_loc(START);finish=cal.get_loc(END);out={};summary=[]
    quarterschedule=np.tile(np.asarray(commonrisk+[0],float),(len(cal),1));quarterschedule[:,8]=100000
    quarterschedule[:,9]=np.asarray([d.year*4+d.quarter for d in cal])
    for name,order,sch in [('rolling_cash_fallback',scoreids,cashschedule),('rolling_equal_fallback',scoreids,schedule),
                            ('dynamic_pool_equal',poolorder,None),('quarter_pool_buyhold',poolorder,quarterschedule)]:
        for fee in [5.,15.]:
            d,error=ledger(order,*args,begin,finish,*commonrisk,fee/1e4,1e6,sch)
            key=name+f'_fee{int(fee)}';out[key]=d
            for label,a,b in [('common',0,len(days)),('recent',int(days.searchsorted('2026-05-22')),len(days))]:
                initial=1e6 if not a else d[a-1,0]
                mm=metrics(d[a:b,0],days[a:b],initial,days[a-1] if a else None);mm.update(strategy=key,period=label,mean_exposure=float(d[a:b,4].mean()),cash_flow_error=error)
                summary.append(mm)
    if save:
        folder=RUN/'selection';folder.mkdir(exist_ok=True)
        dump(logs,folder/'monthly_decisions.json');dump([r['job'] for r in records],folder/'candidate_snapshot.json')
        pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False,encoding='utf-8-sig')
        for name,d in out.items():
            f=pd.DataFrame(d,columns=['equity','cash','market_value','cost','exposure','buy_value','sell_value','attempts','limited','target']);f.insert(0,'date',days)
            f.to_parquet(folder/(name+'.parquet'),index=False)
    return days,out,summary,logs

if __name__=='__main__':
    r=completed();select_and_replay(r);print('Replayed continuous monthly selection from',len(r),'completed signal configurations')
