"""Time-local factor statistics and forward-filtered market regimes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json, warnings, logging
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits
from config import RUN,HORIZONS,LIBRARIES,START,dump

def load_meta():
    p=RUN/'data'
    meta=pd.read_parquet(p/'meta.parquet')
    cal=pd.DatetimeIndex(pd.to_datetime(json.loads((p/'calendar.json').read_text())))
    return meta,cal

def statistics():
    p=RUN/'data';meta,cal=load_meta()
    y=np.load(p/'targets.npy',mmap_mode='r')
    groups=meta.groupby('date_index').indices
    for lib in LIBRARIES:
        dest=p/f'ic_{lib}.npz'
        if dest.exists():continue
        x=np.load(p/f'X_{lib}.npy',mmap_mode='r')
        ic=np.full((len(cal),len(HORIZONS),x.shape[1]),np.nan,np.float32)
        coverage=np.zeros((len(cal),x.shape[1]),np.float32)
        for di,ix in groups.items():
            a=x[ix].astype(float); cov=np.isfinite(a);coverage[di]=cov.mean(axis=0)
            for hi in range(len(HORIZONS)):
                b=y[ix,hi]; good=cov&np.isfinite(b[:,None]);n=good.sum(axis=0)
                aa=np.where(good,a,0);bb=np.where(good,b[:,None],0)
                num=n*(aa*bb).sum(axis=0)-aa.sum(axis=0)*bb.sum(axis=0)
                den=np.sqrt(np.maximum(n*(aa*aa).sum(axis=0)-aa.sum(axis=0)**2,0)*np.maximum(n*(bb*bb).sum(axis=0)-bb.sum(axis=0)**2,0))
                ic[di,hi]=np.divide(num,den,out=np.full(x.shape[1],np.nan),where=(den>1e-12)&(n>=20))
        np.savez_compressed(dest,ic=ic,coverage=coverage)
        print('IC statistics',lib,ic.shape,flush=True)

def filter_forward(model,x,alpha=None):
    result=[]
    for e in model._compute_log_likelihood(x):
        prior=model.startprob_ if alpha is None else alpha@model.transmat_
        z=np.log(np.maximum(prior,1e-300))+e
        alpha=np.exp(z-logsumexp(z));result.append(alpha.copy())
    return np.asarray(result),alpha

def regimes():
    p=RUN/'data';meta,cal=load_meta();m=np.load(p/'market.npz')
    pool=m['in_pool']>.5
    market=pd.Series(np.nanmean(np.where(pool,m['adj_ret1'],np.nan),axis=1),index=cal).fillna(0)
    level=(1+market).cumprod(); vol=market.rolling(20,min_periods=10).std()*np.sqrt(252)
    breadth=pd.Series(np.sum(pool&(m['mom20']>0),axis=1)/np.maximum(pool.sum(axis=1),1),index=cal)
    trend20=level/level.shift(20)-1;trend5=level/level.shift(5)-1
    drawdown=level/level.rolling(60,min_periods=20).max()-1
    frame=pd.DataFrame(dict(ret=market,logvol=np.log(vol.clip(lower=.01)),trend=trend20/(vol+.01),breadth=breadth)).replace([np.inf,-np.inf],np.nan)
    # Common state history for conditional factor weights and risk; never smoothed backwards.
    result=pd.DataFrame(dict(date=cal,market_return=market.to_numpy(),vol=vol.to_numpy(),
                             trend20=trend20.to_numpy(),trend5=trend5.to_numpy(),breadth=breadth.to_numpy(),
                             market_drawdown=drawdown.to_numpy(),level=level.to_numpy(),
                             ma20=level.rolling(20,min_periods=10).mean().to_numpy()))
    audit=[]
    logging.getLogger('hmmlearn.base').setLevel(logging.ERROR)
    for k in [2,3]:
        probabilities=np.full((len(cal),k),1/k); caps=np.full(len(cal),.5)
        for mon,idx in pd.Series(np.arange(len(cal)),index=cal).groupby(cal.to_period('M')):
            t=int(idx.iloc[0]);history=frame.iloc[max(0,t-504):t].dropna()
            if len(history)<126:continue
            scaler=StandardScaler().fit(history);xx=np.clip(scaler.transform(history),-8,8)
            best=None;attempt=[]
            for seed in [17,29]:
                model=GaussianHMM(k,covariance_type='diag',n_iter=120,tol=.02,min_covar=.05,random_state=seed)
                try:
                    with threadpool_limits(1):model.fit(xx)
                    pp,_=filter_forward(model,xx);mass=pp.sum(axis=0);lik=model.score(xx)
                    delta=list(model.monitor_.history)
                    valid=bool(np.isfinite(lik) and mass.min()>=8 and len(delta)>1 and delta[-1]-delta[-2]>=-.1)
                    attempt.append(dict(seed=seed,valid=valid,likelihood=lik,min_mass=float(mass.min())))
                    if valid and (best is None or lik>best[0]):best=(lik,model,pp,mass)
                except (ValueError,np.linalg.LinAlgError) as ex:attempt.append(dict(seed=seed,error=str(ex)))
            if best is None:
                audit.append(dict(month=str(mon),states=k,fallback=True,attempts=attempt));continue
            _,model,pp,mass=best
            mean=(pp*history.ret.to_numpy()[:,None]).sum(axis=0)/mass
            order=np.argsort(mean);shrunk=(mean*mass+60*history.ret.mean())/(mass+60)
            state_cap=np.where(shrunk>0,1.,.05)
            alpha=pp[-1]
            for j in idx.to_numpy():
                if frame.iloc[j].isna().any():continue
                obs=np.clip(scaler.transform(frame.iloc[[j]]),-8,8)
                posterior,alpha=filter_forward(model,obs,alpha)
                pred=posterior[0]@model.transmat_
                probabilities[j]=pred[order];caps[j]=pred@state_cap
            audit.append(dict(month=str(mon),states=k,training_start=str(history.index[0].date()),
                              training_end=str(history.index[-1].date()),first_signal=str(cal[t].date()),
                              state_mean_ordered=mean[order].tolist(),fallback=False,attempts=attempt))
        for j in range(k):result[f'hmm{k}_p{j}']=probabilities[:,j]
        result[f'hmm{k}_cap']=pd.Series(caps).ewm(alpha=.35,adjust=False).mean()
    result.to_parquet(p/'regimes.parquet',index=False);dump(audit,p/'hmm_audit.json')
    print('regimes',len(result),len(audit),flush=True)

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);statistics();regimes()
