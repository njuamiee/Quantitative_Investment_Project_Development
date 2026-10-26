"""Actual rolling fits, with independently recorded factor/model fit cutoffs."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(v,'1')
import json,time,pickle,warnings,platform,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor,LGBMRanker
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from config import RUN,HORIZONS,START,END,SEED,key,dump,digest
from features import load_meta

_CACHE={}
def dataset(lib):
    if _CACHE.get('lib')!=lib:
        p=RUN/'data';meta,cal=load_meta();a=np.load(p/f'ic_{lib}.npz')
        _CACHE.clear();_CACHE.update(dict(lib=lib,meta=meta,cal=cal,
          x=np.load(p/f'X_{lib}.npy',mmap_mode='r'),y=np.load(p/'targets.npy',mmap_mode='r'),
          exits=np.load(p/'label_exit_index.npy',mmap_mode='r'),
          ic=a['ic'],coverage=a['coverage'],reg=pd.read_parquet(p/'regimes.parquet'),
          names=json.loads((p/f'names_{lib}.json').read_text()),
          screen={}))
    return _CACHE

def screen(d,h,window,t):
    cachekey=(h,window,t)
    if cachekey in d['screen']:return d['screen'][cachekey]
    hi=HORIZONS.index(h);end=t-h-1;begin=max(0,end-window)
    a=d['ic'][begin:end,hi];cov=np.nanmean(d['coverage'][begin:end],axis=0)
    n=np.isfinite(a).sum(axis=0);mu=np.nanmean(a,axis=0);sd=np.nanstd(a,axis=0)
    utility=np.abs(mu)/(sd+.05)*np.sqrt(n/max(1,end-begin))
    eligible=np.flatnonzero((cov>=.7)&(n>=40)&np.isfinite(utility))
    eligible=eligible[np.argsort(-utility[eligible],kind='stable')]
    train=np.flatnonzero((d['meta'].date_index.to_numpy()>=begin)&(d['meta'].date_index.to_numpy()<end))
    # Correlation rejection uses a deterministic subset of TRAINING rows only.
    train=train[::max(1,len(train)//3000)];chosen=[];vectors=[];source_counts={}
    for j in eligible:
        if len(chosen)>=100:break
        name=d['names'][j];source=name.split('__')[0] if name.startswith('vendor_') else 'daily'
        if source.startswith('vendor_') and source_counts.get(source,0)>=25:continue
        v=np.nan_to_num(d['x'][train,j]).astype(float);v-=v.mean();norm=np.linalg.norm(v)
        if norm<1e-10:continue
        v/=norm
        if vectors and np.max(np.abs(np.asarray(vectors)@v))>=.98:continue
        chosen.append(int(j));vectors.append(v);source_counts[source]=source_counts.get(source,0)+1
    if not chosen:raise ValueError('no causally eligible factors')
    result=(chosen,dict(screen_cutoff_index=t,screen_begin=begin,screen_end_exclusive=end,
                        last_outcome_index=end-1+h+1,available_features=len(chosen),
                        factors=[d['names'][j] for j in chosen],utility=[float(utility[j]) for j in chosen]))
    assert result[1]['last_outcome_index']<t
    d['screen'][cachekey]=result
    return result

def predict_fit(d,c,t,rows,cols):
    h=c['horizon'];hi=HORIZONS.index(h);end=t-h-1;begin=max(0,end-c['window'])
    dates=d['meta'].date_index.to_numpy();ex=d['exits'][:,hi];y=d['y'][:,hi]
    ix=np.flatnonzero((dates>=begin)&(dates<end)&(ex<t)&np.isfinite(y))
    if len(ix)<3000:raise ValueError('insufficient mature training rows')
    assert int(ex[ix].max())<t
    xx=np.nan_to_num(d['x'][np.ix_(ix,cols)]);zz=np.nan_to_num(d['x'][np.ix_(rows,cols)])
    yy=y[ix];name=c['model'];obj=None
    if name in ['ic','conditional_ic']:
        a=d['ic'][begin:end,hi][:,cols];mu=np.nanmean(a,axis=0)
        if name=='ic':
            w=np.nan_to_num(mu/(np.nanstd(a,axis=0)+.05));w/=np.abs(w).sum()+1e-12
            result=zz@w
        else:
            pp=d['reg'].iloc[begin:end][['hmm3_p0','hmm3_p1','hmm3_p2']].to_numpy()
            w=[]
            for state in range(3):
                mass=np.nansum(np.isfinite(a)*pp[:,state,None],axis=0)
                v=(np.nansum(a*pp[:,state,None],axis=0)+60*mu)/(mass+60)
                v=np.nan_to_num(v);v/=np.abs(v).sum()+1e-12;w.append(v)
            now=d['reg'].iloc[dates[rows]][['hmm3_p0','hmm3_p1','hmm3_p2']].to_numpy()
            result=(zz*(now@np.asarray(w))).sum(axis=1)
    else:
        common=dict(n_estimators=64,learning_rate=.04,num_leaves=15,max_depth=5,min_child_samples=100,
                    reg_lambda=12.,colsample_bytree=.8,verbosity=-1,n_jobs=1,random_state=SEED,deterministic=True)
        if name=='ridge':obj=Ridge(alpha=180.)
        elif name=='lightgbm':obj=LGBMRegressor(**common)
        elif name=='ranker':obj=LGBMRanker(**common,objective='lambdarank',label_gain=list(range(10)))
        elif name=='xgboost':obj=XGBRegressor(n_estimators=64,max_depth=3,learning_rate=.04,min_child_weight=100,
                        reg_lambda=15,subsample=.8,colsample_bytree=.8,n_jobs=1,random_state=SEED,tree_method='hist')
        elif name=='extra_trees':obj=ExtraTreesRegressor(n_estimators=48,max_depth=6,min_samples_leaf=100,max_features=.6,n_jobs=1,random_state=SEED)
        elif name=='catboost':obj=CatBoostRegressor(iterations=64,depth=4,learning_rate=.04,l2_leaf_reg=15,
                        verbose=False,thread_count=1,random_seed=SEED,allow_writing_files=False)
        else:raise ValueError(name)
        with threadpool_limits(1):
            if name=='ranker':obj.fit(xx,np.minimum(9,np.floor((yy+.5)*10)).astype(int),group=np.unique(dates[ix],return_counts=True)[1])
            else:obj.fit(xx,yy)
            result=obj.predict(zz)
    record=dict(fit_index=t,training_start_index=int(dates[ix].min()),training_end_index=int(dates[ix].max()),
                last_mature_outcome_index=int(ex[ix].max()),training_rows=len(ix),actual_factors=len(cols),prediction_rows=len(rows))
    return np.asarray(result,np.float32),record

def run_job(c):
    started=time.time();jobkey=key(c);folder=RUN/'models'/jobkey;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    d=dataset(c['library']);meta=d['meta'];cal=d['cal'];dates=meta.date_index.to_numpy()
    evalrows=np.flatnonzero(meta.date.between(START,END));evaldates=np.unique(dates[evalrows]);position={v:i for i,v in enumerate(evalrows)}
    scores=np.full(len(evalrows),np.nan,np.float32);logs=[];cards={}
    period={'quarterly':'Q','monthly':'M'}.get(c['frequency'])
    if period:
        groups=pd.Series(evaldates,index=cal[evaldates]).groupby(cal[evaldates].to_period(period))
        blocks=[v.to_numpy() for _,v in groups]
    else:blocks=[np.array([i]) for i in evaldates]
    checkpoint=folder/'checkpoint.npz';done=0
    if checkpoint.exists():
        a=np.load(checkpoint);scores=a['scores'];done=int(a['done'])
        logs=json.loads((folder/'fit_audit.json').read_text());cards=json.loads((folder/'factor_audit.json').read_text())
    try:
        for b,days in enumerate(blocks):
            if b<done:continue
            t=int(days[0]);quarter=cal[t].to_period('Q')
            qt=int(evaldates[np.flatnonzero(cal[evaldates].to_period('Q')==quarter)[0]])
            chosen,card=screen(d,c['horizon'],c['window'],qt);cols=chosen[:c['count']]
            rows=np.flatnonzero(np.isin(dates,days));pred,record=predict_fit(d,c,t,rows,cols)
            scores[[position[r] for r in rows]]=pred;record.update(factor_cutoff_index=qt,fold=b)
            logs.append(record);cards[str(quarter)]=card
            if (b+1)%10==0 or b+1==len(blocks):
                dump(logs,folder/'fit_audit.json');dump(cards,folder/'factor_audit.json')
                np.savez_compressed(folder/'checkpoint.tmp.npz',scores=scores,done=b+1)
                (folder/'checkpoint.tmp.npz').replace(checkpoint)
        assert np.isfinite(scores).all()
        result=meta.iloc[evalrows][['date','stockid','date_index','stock_index']].copy();result['score']=scores
        result.to_parquet(folder/'predictions.parquet',index=False)
        summary=dict(job=jobkey,config=c,actual_fits=len(logs),prediction_rows=len(scores),
                     seconds=time.time()-started,first_signal=START,last_signal=END,
                     score_sha256=digest(folder/'predictions.parquet'),status='trained')
        summary['environment']=dict(python=platform.python_version(),platform=platform.system(),
            packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','catboost','numba','pyarrow']})
        dump(summary,folder/'complete.json')
        return summary
    except Exception as e:
        dump(dict(config=c,error_type=type(e).__name__,error=str(e),completed_folds=len(logs)),folder/'error.json')
        raise

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);print(json.dumps(run_job(json.loads(a.config)),indent=2))
