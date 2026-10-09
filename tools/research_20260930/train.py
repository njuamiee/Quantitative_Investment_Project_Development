"""Quarterly out-of-time models, training-only IC selection/deduplication and filtered HMM."""
from __future__ import annotations
import json, math, warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import lightgbm as lgb
from fast_correlation import daily_spearman
from prepare import DATA,RUN,PRICE_FEATURES,dump

SEED=20260930
CONFIG={'seed':SEED,'training_days':504,'validation_days':126,'first_bj_test_quarter':'2024Q4',
    'first_shsz_test_quarter':'2019Q1','max_selected_features':40,'correlation_threshold':.85,
    'minimum_feature_coverage':.7,'prediction_horizon_market_days':5,'feature_time':'after close t',
    'label':'open t+1 to close t+5, unadjusted price return; corporate actions not verified',
    'lgbm':{'num_leaves':15,'max_depth':5,'learning_rate':.03,'min_child_samples':200,'reg_lambda':5,
        'reg_alpha':.1,'n_estimators':400,'verbosity':-1,'n_jobs':4,'random_state':SEED,'deterministic':True,'force_col_wise':True},
    'ridge_alpha':10,'hmm_states':2,'hmm_train_days':504,'hmm_caps':[1,.7],'fixed_cap_control':.85,
    'initial_capital':1000000,'daily_new_batch_nav_fraction':.2,'single_stock_cap':.1,'adv20_participation':.01,
    'one_way_cost_bp_scenarios':[5,15,30],'test_policy':'fixed rules before viewing results; no optimization on outer test returns'}

def corr_columns(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    x=np.where(mask,x,np.nan);y=np.where(mask,y,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xc=x-np.nanmean(x,axis=0);yc=y-np.nanmean(y,axis=0)
        out=np.nansum(xc*yc,axis=0)/np.sqrt(np.nansum(xc*xc,axis=0)*np.nansum(yc*yc,axis=0))
    out[mask.sum(axis=0)<30]=np.nan
    return out

def daily_ic(panel,columns):
    rank_rows=[];pearson_rows=[]
    for date,g in panel.groupby('date',sort=True):
        x=g[columns].to_numpy(dtype=float,copy=True);y=np.broadcast_to(g.label_raw.to_numpy()[:,None],x.shape).copy()
        mask=np.isfinite(x)&np.isfinite(y);x[~mask]=np.nan;y[~mask]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rx=rankdata(x,axis=0,nan_policy='omit');ry=rankdata(y,axis=0,nan_policy='omit')
        rank_rows.append([date,*corr_columns(rx,ry)]);pearson_rows.append([date,*corr_columns(x,y)])
    return (pd.DataFrame(rank_rows,columns=['date']+columns).set_index('date'),
            pd.DataFrame(pearson_rows,columns=['date']+columns).set_index('date'))

def hac_summary(series,lag=5):
    v=series.dropna().to_numpy(dtype=float);n=len(v)
    if n<20:return np.nan,np.nan,np.nan
    e=v-v.mean();variance=float(e@e/n)
    for k in range(1,min(lag,n-1)+1):variance+=2*(1-k/(lag+1))*float(e[k:]@e[:-k]/n)
    se=math.sqrt(max(variance,0)/n);t=float(v.mean()/se) if se>0 else np.nan
    return se,t,float(2*norm.sf(abs(t)))

def factor_selection(train,ics,columns,fold):
    dates=train.date.unique(); stats=[]
    for col in columns:
        vals=ics.loc[ics.index.isin(dates),col]
        se,t,p=hac_summary(vals)
        stats.append({'feature':col,'coverage':train[col].notna().mean(),'mean_ic':vals.mean(),'icir':vals.mean()/vals.std() if vals.std()>0 else np.nan,
            'positive_fraction':(vals.dropna()>0).mean(),'n_ic_days':vals.notna().sum(),'hac_se':se,'hac_t':t,'p':p})
    card=pd.DataFrame(stats).set_index('feature')
    pv=card.p.fillna(1).sort_values();m=len(pv);harmonic=sum(1/i for i in range(1,m+1))
    adjusted=(pv.to_numpy()*m*harmonic/np.arange(1,m+1))
    adjusted=np.minimum.accumulate(adjusted[::-1])[::-1].clip(0,1)
    card['by_fdr_q']=pd.Series(adjusted,index=pv.index)
    card['abs_ic']=card.mean_ic.abs()
    candidates=card[(card.coverage>=.7)&(card.n_ic_days>=100)&card.mean_ic.notna()].sort_values(['abs_ic','coverage'],ascending=False).head(100).index.tolist()
    sums=np.zeros((len(candidates),len(candidates)));counts=np.zeros_like(sums)
    for _,g in train.groupby('date'):
        matrix=daily_spearman(g[candidates],min_periods=30).to_numpy()
        sums+=np.nan_to_num(matrix);counts+=np.isfinite(matrix)
    corr=pd.DataFrame(np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0),index=candidates,columns=candidates)
    selected=[]; reasons={}
    for col in card.index:
        if col not in candidates:reasons[col]='quality_or_candidate_budget'
    for col in candidates:
        overlap=[s for s in selected if abs(corr.loc[col,s])>=.85]
        if overlap:reasons[col]='redundant_with:'+overlap[0]
        elif len(selected)>=40:reasons[col]='feature_budget_40'
        else:selected.append(col);reasons[col]='selected'
    if not selected:raise ValueError('No acceptable training factors')
    card['decision']=pd.Series(reasons);card['direction']=np.sign(card.mean_ic)
    card.to_csv(fold/'factor_cards.csv',encoding='utf-8-sig');corr.to_csv(fold/'mean_daily_spearman.csv',encoding='utf-8-sig')
    return selected,card

def fit_lgb(train,valid,refit,test,features,directory,name):
    params=dict(CONFIG['lgbm'])
    model=lgb.LGBMRegressor(**params)
    model.fit(train[features],train.target,eval_set=[(valid[features],valid.target)],
              callbacks=[lgb.early_stopping(35,verbose=False)])
    best=max(1,model.best_iteration_);params['n_estimators']=best
    model=lgb.LGBMRegressor(**params).fit(refit[features],refit.target)
    (directory/(name+'.txt')).write_text(model.booster_.model_to_string(),encoding='utf-8')
    pd.Series(model.feature_importances_,index=features).sort_values(ascending=False).to_csv(directory/(name+'_importance.csv'))
    return model.predict(test[features]),best

def model_metrics(scored,score):
    rows=[]
    for date,g in scored.dropna(subset=['label_raw',score]).groupby('date'):
        if len(g)<30:continue
        ic=float(spearmanr(g[score],g.label_raw).statistic) if g[score].nunique()>1 else np.nan
        ranked=g.sort_values(score)
        rows.append({'date':date,'ic':ic,'hit':float(((g[score]>0)==(g.target>0)).mean()),
            'top5_label_mean':ranked.tail(5).label_raw.mean(),'pool_label_mean':g.label_raw.mean(),
            'top_bottom_spread':ranked.tail(20).label_raw.mean()-ranked.head(20).label_raw.mean()})
    daily=pd.DataFrame(rows)
    return {'dates':len(daily),'mean_ic':float(daily.ic.mean()),'icir':float(daily.ic.mean()/daily.ic.std()),
            'relative_direction_hit':float(daily.hit.mean()),'top5_5d_mean':float(daily.top5_label_mean.mean()),
            'pool_5d_mean':float(daily.pool_label_mean.mean()),'top_bottom_5d_spread':float(daily.top_bottom_spread.mean())},daily

def rolling(market):
    panel=pd.read_parquet(DATA/(market+'_model_panel.parquet')).sort_values(['date','stockid'])
    cols=[c for c in panel if c.startswith('x_')]; pricecols=['x_'+c for c in PRICE_FEATURES]
    ics,pearson=daily_ic(panel,cols)
    ics.to_csv(RUN/(market+'_daily_rank_ic.csv'),encoding='utf-8-sig')
    pearson.to_csv(RUN/(market+'_daily_rankfeature_pearson_ic.csv'),encoding='utf-8-sig')
    dates=pd.DatetimeIndex(sorted(panel.date.unique()));start=CONFIG['first_'+market+'_test_quarter']
    periods=pd.period_range(start,dates[-1].to_period('Q'),freq='Q');folds=[];predictions=[]
    for quarter in periods:
        test=panel[panel.date.dt.to_period('Q')==quarter].copy()
        if test.empty:continue
        begin=test.date.min();history=dates[dates<begin]
        if len(history)<630:
            folds.append({'quarter':str(quarter),'skipped':'less_than_630_previous_pool_dates','available_dates':len(history)});continue
        valid_start=history[-126];train_start=history[-630]
        train=panel[(panel.date>=train_start)&(panel.date<valid_start)&(panel.exit_date<valid_start)&panel.target.notna()]
        valid=panel[(panel.date>=valid_start)&(panel.date<begin)&(panel.exit_date<begin)&panel.target.notna()]
        refit=panel[(panel.date>=train_start)&(panel.date<begin)&(panel.exit_date<begin)&panel.target.notna()]
        directory=RUN/market/str(quarter);directory.mkdir(parents=True,exist_ok=True)
        assert train.exit_date.max()<valid_start and refit.exit_date.max()<begin
        selected,cards=factor_selection(train,ics,cols,directory)
        test['pred_price'],price_iter=fit_lgb(train,valid,refit,test,pricecols,directory,'price_lgbm')
        test['pred_selected'],best=fit_lgb(train,valid,refit,test,selected,directory,'selected_lgbm')
        ridge=Ridge(alpha=10).fit(refit[selected].fillna(0),refit.target)
        test['pred_ridge']=ridge.predict(test[selected].fillna(0))
        np.savez(directory/'ridge.npz',coef=ridge.coef_,intercept=ridge.intercept_,features=np.array(selected))
        top=selected[0];direction=float(cards.loc[top,'direction'])
        test['pred_simple']=test[top]*direction
        # Same training-selected feature identities, one extra day of Alpha latency, refitted model.
        if market=='bj':
            lagcols=[c.replace('x_c','xl_c',1) if c.startswith('x_c') else c for c in selected]
            test['pred_lag1'],lag_iter=fit_lgb(train,valid,refit,test,lagcols,directory,'selected_lag1_lgbm')
        else:lag_iter=None
        record={'quarter':str(quarter),'train_first':train.date.min(),'train_last':train.date.max(),
            'train_label_end':train.exit_date.max(),'validation_start':valid_start,'refit_label_end':refit.exit_date.max(),
            'test_first':begin,'test_last':test.date.max(),'train_rows':len(train),'valid_rows':len(valid),
            'refit_rows':len(refit),'test_rows':len(test),'selected_features':selected,'selected_count':len(selected),
            'best_iteration':best,'price_iteration':price_iter,'lag_iteration':lag_iter,'simple_factor':top,'simple_direction':direction}
        dump(record,directory/'fold.json');folds.append(record)
        keep=['date','stockid','entry_date','exit_date','label_raw','label_qfq_diagnostic','label_adjustment_boundary','target','adv20','close']
        keep += [c for c in test if c.startswith('pred_')]
        if 'factor_coverage' in test:keep.append('factor_coverage')
        predictions.append(test[keep]);print('trained',market,quarter,'features',len(selected),'iterations',best,flush=True)
    scored=pd.concat(predictions,ignore_index=True)
    scored.to_parquet(RUN/(market+'_predictions.parquet'),index=False)
    dump(folds,RUN/(market+'_folds.json'))
    metrics={}
    for score in [c for c in scored if c.startswith('pred_')]:
        metrics[score],daily=model_metrics(scored,score)
        daily.to_csv(RUN/(market+'_'+score+'_daily_metrics.csv'),index=False)
    dump(metrics,RUN/(market+'_prediction_metrics.json'))
    return scored

def hmm(scored):
    price=pd.read_parquet(DATA/'bj_model_panel.parquet',columns=['date','ret1','vol20'])
    price['breadth']=np.sign(price.ret1)
    obs=price.groupby('date').agg(market_return=('ret1','mean'),breadth=('breadth','mean'),dispersion=('ret1','std'))
    obs['log_vol']=np.log(obs.market_return.rolling(20).std().clip(lower=1e-6))
    obs=obs.replace([np.inf,-np.inf],np.nan).dropna()
    columns=['market_return','log_vol','breadth','dispersion'];rows=[];fits=[];previous=1.
    future=obs[obs.index>=scored.date.min()]
    for month,block in future.groupby(future.index.to_period('M')):
        past=obs[obs.index<block.index.min()].tail(504)
        scaler=StandardScaler().fit(past[columns]);tx=scaler.transform(past[columns]);candidates=[]
        for seed in [SEED,SEED+1,SEED+2]:
            model=GaussianHMM(n_components=2,covariance_type='diag',n_iter=200,tol=.001,min_covar=.01,
                transmat_prior=np.array([[10,1],[1,10]]),random_state=seed)
            with warnings.catch_warnings():warnings.simplefilter('ignore');model.fit(tx)
            candidates.append((model.score(tx),model))
        _,model=max(candidates,key=lambda x:x[0]);order=np.argsort(model.means_[:,1])
        def step(alpha,emission,first=False):
            prior=model.startprob_ if first else alpha@model.transmat_
            logp=np.log(np.clip(prior,1e-300,None))+emission
            return np.exp(logp-logsumexp(logp))
        alpha=None
        for i,emission in enumerate(model._compute_log_likelihood(tx)):alpha=step(alpha,emission,i==0)
        for date,row in block.iterrows():
            emission=model._compute_log_likelihood(scaler.transform(row[columns].to_frame().T))[0]
            alpha=step(alpha,emission);prob=(alpha@model.transmat_)[order]
            cap=float(prob@np.array([1,.7]));previous=.8*previous+.2*cap
            volcap=float(np.clip(np.exp(past.log_vol.median()-row.log_vol),.7,1))
            rows.append({'date':date,'p_low_next':prob[0],'p_high_next':prob[1],'hmm_cap':previous,'vol_cap':volcap,'fixed_cap':.85,'fit_month':str(month)})
        history=list(model.monitor_.history)
        final_delta=float(history[-1]-history[-2]) if len(history)>1 else None
        fits.append({'month':str(month),'train_end':past.index.max(),'train_rows':len(past),'converged':bool(model.monitor_.converged),
            'iterations':int(model.monitor_.iter),'final_log_likelihood_delta':final_delta,
            'strict_monotone_convergence':bool(final_delta is not None and 0<=final_delta<model.tol and model.monitor_.iter<model.n_iter),
            'expected_durations':(1/np.clip(1-np.diag(model.transmat_)[order],1e-9,None)).tolist(),
            'transition':model.transmat_[np.ix_(order,order)].tolist(),'means_z':model.means_[order].tolist(),
            'observation_columns':columns})
    pd.DataFrame(rows).to_csv(RUN/'hmm_probabilities.csv',index=False);dump(fits,RUN/'hmm_fits.json')
    print('hmm_complete',len(fits),'fits',flush=True)

def main():
    dump(CONFIG,RUN/'experiment_config.json')
    bj=rolling('bj');hmm(bj);rolling('shsz')

if __name__=='__main__':main()
