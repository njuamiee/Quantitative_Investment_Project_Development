"""Paired, one-coordinate changes with identical account rules and dates."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from config import RUN,key
from account import risk_grid

RISK_BASELINES=dict(topn=20,weighting='equal',exposure='full',regime='none',
    exit='scheduled',concentration='none',entry='rank',pool=100,rebalance='horizon')

@lru_cache(maxsize=8)
def risk_pairs(horizon):
    """Pair risk IDs by an exact match on every other declared coordinate."""
    rules=risk_grid(horizon);axes=list(RISK_BASELINES)
    lookup={tuple(r[a] for a in axes):r['risk_id'] for r in rules};groups={}
    for rule in rules:
        for axis,baseline in RISK_BASELINES.items():
            if rule[axis]==baseline:continue
            reference=dict(rule);reference[axis]=baseline
            k=tuple(reference[a] for a in axes)
            if k not in lookup:continue
            groups.setdefault((axis,str(baseline),str(rule[axis])),[]).append((lookup[k],rule['risk_id']))
    return {k:np.asarray(v,dtype=int) for k,v in groups.items()}

def risk_compare(record,values):
    rows=[]
    for (axis,baseline,alternative),pairs in risk_pairs(record['config']['horizon']).items():
        for pi,period in [(0,'common'),(1,'recent')]:
            a=values[pairs[:,1],pi];b=values[pairs[:,0],pi];delta=a-b
            finite=np.isfinite(delta[:,3])
            rows.append(dict(job=record['job'],axis=axis,baseline=baseline,alternative=alternative,
                period=period,matched_account_pairs=len(pairs),
                median_return_change=float(np.median(delta[:,0])),
                better_return_fraction=float((delta[:,0]>0).mean()),
                unchanged_return_fraction=float((np.abs(delta[:,0])<1e-12).mean()),
                median_drawdown_change=float(np.median(delta[:,2])),
                median_sharpe_change=float(np.median(delta[finite,3])) if finite.any() else np.nan,
                median_exposure_change=float(np.median(delta[:,5])),
                **record['config']))
    return rows

@lru_cache(maxsize=48)
def values(job):return np.load(RUN/'models'/job/'risk_results.npz')['metrics']

def compare(records):
    known={r['job']:r for r in records};rows=[]
    baselines={'frequency':'quarterly','count':20,'window':252,'library':'daily','model':'lightgbm','horizon':5}
    daily_indices={h:np.array([i for i,r in enumerate(risk_grid(h)) if r['rebalance_days']==1]) for h in [1,2,3,4,5,10,15,20]}
    for record in records:
        c=record['config'];v=values(record['job'])
        for axis,base in baselines.items():
            if c[axis]==base:continue
            ref=dict(c);ref[axis]=base;refkey=key(ref)
            if refkey not in known:continue
            b=values(refkey);a=v
            if axis=='horizon':
                # The account always rebalances daily on BOTH sides, separating
                # target horizon from actual rebalance frequency.
                a=a[daily_indices[c['horizon']]];b=b[daily_indices[base]]
            assert a.shape==b.shape
            for pi,period in [(0,'common'),(1,'recent')]:
                delta=a[:,pi,0]-b[:,pi,0];sh=a[:,pi,3]-b[:,pi,3];dd=a[:,pi,2]-b[:,pi,2]
                rows.append(dict(axis=axis,baseline=str(base),alternative=str(c[axis]),period=period,
                    reference_job=refkey,alternative_job=record['job'],matched_account_pairs=len(delta),
                    median_return_change=float(np.nanmedian(delta)),mean_return_change=float(np.nanmean(delta)),
                    better_return_fraction=float((delta>0).mean()),median_sharpe_change=float(np.nanmedian(sh)),
                    median_drawdown_change=float(np.nanmedian(dd)),**{k:v for k,v in c.items() if k!=axis}))
    return pd.DataFrame(rows)
