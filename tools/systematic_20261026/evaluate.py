"""Exhaustive risk grid for each trained signal; retain losers and real counts."""
import json,time
import numpy as np
import pandas as pd
from numba import njit
from config import RUN,START,END,dump,digest
from account import market_inputs,score_order,run_account,risk_grid,RISK_AXES,metrics

_MARKET=None
def get_market():
    global _MARKET
    if _MARKET is None:_MARKET=market_inputs()
    return _MARKET

@njit(cache=True)
def measure(eq,start,end,initial,years):
    n=end-start;r=np.zeros(n);peak=initial;dd=0.
    last=initial
    for j in range(n):
        v=eq[start+j];r[j]=v/last-1;last=v;peak=max(peak,v);dd=max(dd,1-v/peak)
    ret=last/initial-1;annual=(last/initial)**(1/max(years,1/252))-1
    mean=np.mean(r);sd=np.sqrt(np.sum((r-mean)**2)/max(n-1,1))
    sr=(mean-(1.015**(1/252)-1))/sd*np.sqrt(252) if sd>1e-12 else np.nan
    cm=annual/dd if dd>1e-12 else np.nan
    return np.array([ret,annual,dd,sr,cm])

def evaluate_job(result):
    folder=RUN/'models'/result['job'];complete=folder/'account_complete.json'
    if complete.exists():return json.loads(complete.read_text())
    tic=time.time();cal,m,args=get_market();h=result['config']['horizon'];risks=risk_grid(h)
    preds=pd.read_parquet(folder/'predictions.parquet');order=score_order(preds,cal,m['open'].shape[1])
    days=cal[(cal>=START)&(cal<=END)];months=np.unique(days.to_period('M'))
    decisions=[int(np.flatnonzero(days.to_period('M')==mon)[0]) for mon in months]
    n=len(risks);values=np.full((n,2,9),np.nan);objectives=np.full((n,len(decisions)),-1e30,np.float32)
    maxerror=0.;best_full=(-np.inf,None,None);best_recent=(-np.inf,None,None)
    windows=[(0,len(days),1e6),(int(days.searchsorted('2026-05-22')),len(days),None)]
    for k,risk in enumerate(risks):
        d,error=run_account(order,args,cal,risk);maxerror=max(maxerror,error)
        for pi,(begin,end,initial) in enumerate(windows):
            capital=initial if initial is not None else d[begin-1,0]
            first=days[begin-1] if begin else days[0];years=(days[end-1]-first).days/365.25
            values[k,pi,:5]=measure(d[:,0],begin,end,capital,years)
            values[k,pi,5:]=[d[begin:end,4].mean(),d[begin:end,3].sum(),d[begin:end,8].sum(),d[begin:end,7].sum()]
        for j,cut in enumerate(decisions):
            if cut<63:continue
            # All these daily account outcomes are known before the next month's first signal close.
            scores=[];eligible=True
            for window in [63,126]:
                begin=max(0,cut-window);capital=d[begin-1,0] if begin else 1e6
                years=(days[cut-1]-(days[begin-1] if begin else days[0])).days/365.25
                mm=measure(d[:,0],begin,cut,capital,years)
                if not np.isfinite(mm[3:]).all() or mm[0]<=0 or mm[2]>=.30 or d[begin:cut,5].sum()<=0:
                    eligible=False;break
                scores.append(min(mm[3],mm[4]/2))
            if eligible:objectives[k,j]=min(scores)
        for period in [0,1]:
            sr,cm=values[k,period,3:5]
            objective=min(sr,cm/2) if np.isfinite(sr) and np.isfinite(cm) else -np.inf
            record=best_full if period==0 else best_recent
            if objective>record[0]:
                record=(objective,k,d.copy())
                if period==0:best_full=record
                else:best_recent=record
    np.savez_compressed(folder/'risk_results.npz',metrics=values,objectives=objectives,
                        decision_indices=np.asarray(decisions),decision_dates=np.asarray([str(days[i].date()) for i in decisions]))
    for name,record in [('full_posthoc',best_full),('recent_posthoc',best_recent)]:
        if record[1] is not None:
            np.savez_compressed(folder/(name+'.npz'),daily=record[2],risk_id=record[1])
    dump(risks,folder/'risk_registry.json')
    stats={}
    for i,name in enumerate(['common','recent']):
        v=values[:,i];passed=(v[:,3]>1)&(v[:,4]>2)&(v[:,2]<.3)
        stats[name]=dict(pass_count=int(passed.sum()),max_return=float(np.nanmax(v[:,0])),
                         min_return=float(np.nanmin(v[:,0])),median_return=float(np.nanmedian(v[:,0])))
    summary=dict(job=result['job'],config=result['config'],actual_fits=result['actual_fits'],
                 completed_accounts=n,seconds=time.time()-tic,cash_flow_max_error=maxerror,
                 predictions_sha256=result['score_sha256'],results_sha256=digest(folder/'risk_results.npz'),
                 dates=[START,END],statistics=stats,status='complete',
                 simulation='daily-bar raw-share cash ledger; auction fill and limit rules are proxies',
                 posthoc_curves='labelled descriptive winners, never auto-selected from future results')
    if 'environment' in result:summary['environment']=result['environment']
    dump(summary,complete);return summary
