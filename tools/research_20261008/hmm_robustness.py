"""Strict likelihood-fit HMM sensitivity, selected using training diagnostics only."""
import sys,json,warnings,logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from prepare import DATA,dump
from accounts import simulate
OLD=ROOT/'training_runs/teacher9_20260930';R=ROOT/'training_runs/teacher9_20261008'

def main():
    panel=pd.read_parquet(DATA/'bj_model_panel.parquet');prices=pd.read_parquet(DATA/'bj_prices.parquet');scored=pd.read_parquet(OLD/'bj_predictions.parquet')
    obs=panel.groupby('date').agg(market_return=('ret1','mean'),breadth=('ret1',lambda x:(x>0).mean()),dispersion=('ret1','std'))
    obs['log_vol']=np.log(obs.market_return.rolling(20,min_periods=16).std().clip(lower=1e-5));obs=obs.replace([np.inf,-np.inf],np.nan).dropna()
    columns=['market_return','log_vol','breadth','dispersion'];rows=[];fits=[];previous=1.
    for month,block in obs[obs.index>=scored.date.min()].groupby(lambda d:d.to_period('M')):
        past=obs[obs.index<block.index.min()].tail(504);scaler=StandardScaler().fit(past[columns]);tx=scaler.transform(past[columns]);candidates=[];diagnostics=[]
        for seed in [17,29,43]:
            # Remove the sticky transition prior so likelihood convergence is interpretable directly.
            model=GaussianHMM(n_components=2,covariance_type='diag',n_iter=400,tol=.001,min_covar=.01,transmat_prior=1.,random_state=seed)
            with warnings.catch_warnings():warnings.simplefilter('ignore');model.fit(tx)
            history=list(model.monitor_.history);delta=float(history[-1]-history[-2]);durations=1/np.clip(1-np.diag(model.transmat_),1e-9,None)
            accepted=0<=delta<model.tol and model.monitor_.iter<model.n_iter and durations.min()>=5
            diagnostics.append({'seed':seed,'delta':delta,'iterations':model.monitor_.iter,'duration_min':float(durations.min()),'accepted':bool(accepted),'train_log_likelihood':float(model.score(tx))})
            if accepted:candidates.append((model.score(tx),model))
        fits.append({'month':str(month),'train_end':str(past.index.max()),'first_prediction_date':str(block.index.min()),'fallback_fixed85':not bool(candidates),'candidates':diagnostics})
        if not candidates:
            for date in block.index:rows.append({'date':date,'hmm_strict_cap':.85,'p_high_next':np.nan})
            previous=.85;continue
        _,model=max(candidates,key=lambda x:x[0]);order=np.argsort(model.means_[:,1]);alpha=None
        def step(alpha,emission,first=False):
            prior=model.startprob_ if first else alpha@model.transmat_;logp=np.log(np.clip(prior,1e-300,None))+emission
            return np.exp(logp-logsumexp(logp))
        for i,emission in enumerate(model._compute_log_likelihood(tx)):alpha=step(alpha,emission,i==0)
        for date,row in block.iterrows():
            alpha=step(alpha,model._compute_log_likelihood(scaler.transform(row[columns].to_frame().T))[0]);prob=(alpha@model.transmat_)[order]
            previous=.8*previous+.2*float(prob@np.array([1,.7]));rows.append({'date':date,'hmm_strict_cap':previous,'p_high_next':prob[1]})
        print('HMM',str(month),'accepted',len(candidates),flush=True)
    caps=pd.DataFrame(rows);caps.to_csv(R/'hmm_strict_probabilities.csv',index=False);dump(fits,R/'hmm_strict_fits.json')
    d,_,_,m=simulate(scored,prices,cap_series=caps.set_index('date').hmm_strict_cap);d.to_csv(R/'hmm_strict_account.csv',index=False)
    m['fallback_months']=sum(x['fallback_fixed85'] for x in fits);m['months']=len(fits);m['policy']='Prior-free likelihood fit; train-only nonnegative <.001 final delta, no iteration limit hit, expected state duration >=5 days; otherwise fixed .85 cap. Old forecasts and fixed period.'
    dump(m,R/'hmm_strict_metrics.json')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
