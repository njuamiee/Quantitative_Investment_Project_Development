"""Causal factor-return timing and HMM conditional shrinkage allocation diagnostics."""
import json,re
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
from common import DATA,RUN,dump
from ledger import Market

def main():
    panel=pd.read_parquet(DATA/'model_panel.parquet');reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date')
    cols=[c for c in panel if c.startswith('x_llm_') or c in ['x_mom2','x_mom5','x_mom20','x_vol20','x_drawdown60']]
    p=pd.read_parquet(DATA/'prices.parquet');g=p.groupby('stockid');fwd=g.adj_ret1.shift(-1)
    future=p[['date','stockid']].copy();future['next_return']=fwd;future['realized_date']=g.date.shift(-1)
    q=panel[['date','stockid']+cols].merge(future,on=['date','stockid']);factor_returns=[]
    for d,x in q.groupby('date'):
        y=x.next_return.to_numpy();z=x[cols].to_numpy();weights=np.nan_to_num(z-np.nanmean(z,axis=0))
        weights/=np.maximum(np.abs(weights).sum(axis=0),1e-12)
        ret=np.nansum(weights*y[:,None],axis=0)
        if x.realized_date.notna().any():factor_returns.append({'date':x.realized_date.iloc[0],**dict(zip(cols,ret))})
    fr=pd.DataFrame(factor_returns).set_index('date').sort_index();fr.to_csv(RUN/'factor_timing_realized_returns.csv')
    # At close t factor returns through t are known. No negative shift is used for timing weights.
    momentum=fr.rolling(63,min_periods=40).mean()/fr.rolling(63,min_periods=40).std().clip(lower=.001)
    rows=[];cards=[]
    for quarter,block in panel[panel.date>='2024-10-01'].groupby(panel.date.dt.to_period('Q')):
        past=fr[fr.index<block.date.min()].tail(504).dropna();x=past.to_numpy()
        # Return at day t is paired with state prediction available at t-1.
        probs=reg[['hmm3_p_'+str(k) for k in range(3)]].shift(1).reindex(past.index).fillna(1/3).to_numpy()
        cov=LedoitWolf().fit(x).covariance_;globalmean=x.mean(axis=0);weights=[]
        for k in range(3):
            mass=probs[:,k].sum();mu=(x*probs[:,k,None]).sum(axis=0)/(mass+1e-12)
            shrunk=(mass*mu+60*globalmean)/(mass+60)
            w=np.linalg.solve(cov+np.eye(len(cols))*1e-6,shrunk);w=np.clip(w,-10,10);w/=max(abs(w).sum(),1e-12);weights.append(w)
            variance=np.average((x-mu)**2,axis=0,weights=probs[:,k])
            for i,c in enumerate(cols):cards.append({'quarter':str(quarter),'state':k,'factor':c,'conditional_mean':mu[i],
                      'conditional_vol':np.sqrt(variance[i]),'conditional_sharpe':mu[i]/np.sqrt(variance[i])*np.sqrt(252),'allocation_weight':w[i],
                      'training_last_return_date':str(past.index.max())})
        weights=np.array(weights)
        for d,a in block.groupby('date'):
            prob=reg.reindex([d])[['hmm3_p_'+str(k) for k in range(3)]].fillna(1/3).to_numpy()[0]
            z=a[cols].fillna(0).to_numpy();w=prob@weights
            mw=momentum.reindex([d]).fillna(0).to_numpy()[0];mw=np.clip(mw,-1,1);mw/=max(abs(mw).sum(),1e-12)
            base=globalmean/max(abs(globalmean).sum(),1e-12)
            for i,s in enumerate(a.stockid):rows.append({'date':d,'stockid':s,'hmm_factor_mv':float(z[i]@w),
                                 'factor_momentum63':float(z[i]@mw),'static_factor_mean':float(z[i]@base)})
    scores=pd.DataFrame(rows);scores.to_parquet(RUN/'factor_timing_predictions.parquet',index=False);pd.DataFrame(cards).to_csv(RUN/'state_factor_matrix.csv',index=False)
    market=Market(p,pd.read_csv(DATA/'actions.csv'),reg);results=[]
    for h in [1,5,20]:
        for model in ['hmm_factor_mv','factor_momentum63','static_factor_mean']:
            for policy in ['none','vol12','hmm3_volatility']:
                d,m,*_=market.run(scores,horizon=h,score=model,topn=20,policy=policy,start=scores.date.min(),end=scores.date.max())
                m.update(model=model,status='Additional exploratory factor timing comparison; not exact paper replication');results.append(m)
                d.to_csv(RUN/f'factor_timing_h{h}_{model}_{policy}.csv',index=False)
    dump(results,RUN/'factor_timing_results.json');print('factor timing complete',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
