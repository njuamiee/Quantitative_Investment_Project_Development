"""Monthly past-only HMM, forward filtering, simple risk controls, fit diagnostics."""
import logging,warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits
from common import DATA,RUN,dump

def filtered(model,x,alpha=None):
    out=[]
    for e in model._compute_log_likelihood(x):
        prior=model.startprob_ if alpha is None else alpha@model.transmat_
        z=np.log(np.maximum(prior,1e-300))+e;alpha=np.exp(z-logsumexp(z));out.append(alpha.copy())
    return np.array(out),alpha

def main(window=504,tag=''):
    p=pd.read_parquet(DATA/'prices.parquet');u=p[p.in_pool].copy()
    o=u.groupby('date').agg(market_return=('adj_ret1','mean'),dispersion=('adj_ret1','std'),
                             liquidity=('log_illiq','median'),trend20=('mom20','median'))
    o['market_vol']=o.market_return.ewm(span=20,min_periods=16,adjust=False).std()*np.sqrt(252)
    o['log_vol']=np.log(o.market_vol.clip(lower=.01))
    o['trend_strength']=o.trend20/(o.market_vol/np.sqrt(12)+.001)
    cols=['market_return','log_vol','trend_strength','liquidity','dispersion']
    o=o.replace([np.inf,-np.inf],np.nan).dropna(subset=cols)
    o['cap_none']=1.;o['cap_fixed50']=.5;o['cap_fixed75']=.75
    o['cap_vol12']=(.12/o.market_vol).clip(.05,1)
    o['cap_vol20']=(.20/o.market_vol).clip(.05,1)
    o['cap_trend']=np.where(o.trend20>0,1,.1)
    o['cap_trend_vol']=(o.cap_trend*o.cap_vol20).clip(.05,1)
    diagnostics=[]
    logging.getLogger('hmmlearn.base').setLevel(logging.ERROR)
    for k in [2,3]:
        output=[]
        for month,block in o[o.index>='2023-01-01'].groupby(lambda d:d.to_period('M')):
            past=o[o.index<block.index.min()].tail(window)
            if len(past)<min(160,int(window*.66)):continue
            scaler=StandardScaler().fit(past[cols]);x=np.clip(scaler.transform(past[cols]),-8,8)
            candidates=[];attempts=[]
            for seed in [17,29]:
                m=GaussianHMM(n_components=k,covariance_type='diag',n_iter=220,tol=.01,min_covar=.03,random_state=seed)
                try:
                    with warnings.catch_warnings():warnings.simplefilter('ignore');m.fit(x)
                    history=list(m.monitor_.history);delta=history[-1]-history[-2] if len(history)>1 else -np.inf
                    probs,_=filtered(m,x);occupancy=probs.sum(axis=0);durations=1/np.clip(1-np.diag(m.transmat_),1e-6,None)
                    accepted=bool(np.isfinite(m.score(x)) and delta>=-.05 and occupancy.min()>=10 and durations.min()>=2)
                    attempts.append({'seed':seed,'accepted':accepted,'likelihood':float(m.score(x)),'delta':float(delta),
                                     'iterations':m.monitor_.iter,'occupancy_min':float(occupancy.min()),'duration_min':float(durations.min())})
                    if accepted:candidates.append((m.score(x),m,probs,durations))
                except (ValueError,np.linalg.LinAlgError) as e:attempts.append({'seed':seed,'accepted':False,'error':type(e).__name__})
            record={'month':str(month),'states':k,'training_start':str(past.index.min()),'training_end':str(past.index.max()),
                    'prediction_start':str(block.index.min()),'attempts':attempts,'fallback':not bool(candidates)}
            if not candidates:
                for d,row in block.iterrows():output.append({'date':d,'defensive':row.cap_vol12,'volatility':row.cap_vol12,'growth':.5,'fallback':True})
                diagnostics.append(record);continue
            _,m,trainprobs,durations=max(candidates,key=lambda z:z[0])
            # State labels sorted using training-only conditional return volatility.
            r=past.market_return.to_numpy();mass=trainprobs.sum(axis=0)
            means=(trainprobs*r[:,None]).sum(axis=0)/mass
            std=np.sqrt((trainprobs*(r[:,None]-means)**2).sum(axis=0)/mass)
            order=np.argsort(std);state_sharpe=means/std*np.sqrt(252)
            # Shrink return means toward the global training mean for stability.
            shrunk=(mass*means+60*np.mean(r))/(mass+60)
            state_cap=np.where(shrunk>0,1.,.1)
            targets=np.where(shrunk>0,.20,.06)
            growthcap=np.clip(.5+state_sharpe/4,.05,1)
            probabilities,alpha=filtered(m,np.clip(scaler.transform(block[cols]),-8,8),alpha=trainprobs[-1])
            for (d,row),prob in zip(block.iterrows(),probabilities):
                pred=prob@m.transmat_
                recordrow={'date':d,'defensive':float(pred@state_cap),'volatility':float(np.clip((pred@targets)/row.market_vol,.03,1)),
                           'growth':float(pred@growthcap),'fallback':False}
                for j,idx in enumerate(order):recordrow['p_'+str(j)]=float(pred[idx])
                output.append(recordrow)
            record.update({'state_means':means[order].tolist(),'state_vol':std[order].tolist(),'state_sharpe':state_sharpe[order].tolist(),
                           'duration':durations[order].tolist(),'transition':m.transmat_[order][:,order].tolist(),'state_order':order.tolist()})
            diagnostics.append(record)
            print('hmm',k,str(month),'accepted',len(candidates),flush=True)
        q=pd.DataFrame(output).set_index('date')
        for policy in ['defensive','volatility','growth']:
            # Smooth only with current/past probabilities, no symmetric smoothing.
            o['cap_hmm'+str(k)+'_'+policy]=q[policy].ewm(alpha=.35,adjust=False).mean().reindex(o.index).fillna(.5)
        for c in [x for x in q if x.startswith('p_')]:o['hmm'+str(k)+'_'+c]=q[c].reindex(o.index)
    o.to_csv(RUN/(tag+'regime_probabilities.csv'));dump(diagnostics,RUN/(tag+'hmm_fits.json'))
    dump({'emission_features':cols,'max_states':3,'window':window,'refit':'monthly','inference':'forward filtering and one-step prediction only',
          'states_labeled_by':'training conditional market volatility','failures':'explicit fallback to volatility control',
          'missing_observations':['historical macro vintages','bid ask spread','complete financing series']},RUN/(tag+'hmm_protocol.json'))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--window',type=int,default=504);args=parser.parse_args()
    with threadpool_limits(limits=3):main(args.window,'' if args.window==504 else f'w{args.window}_')
