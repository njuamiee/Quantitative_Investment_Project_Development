"""Purged quarterly validation, finite registered models, continuous outer ledger."""
import argparse,json,warnings,joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor,LGBMRanker
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
from common import DATA,RUN,HORIZONS,dump,SEED
from ledger import Market

MODELS=['ridge60','lgbm100','ranker60','catboost60','xgboost60','extra60','llm_ridge','conditional_ic','ic','ensemble']
PROFILES=[('none','none'),('fixed50','none'),('fixed75','none'),('vol12','none'),('vol20','none'),
          ('trend_vol','none'),('hmm2_defensive','none'),('hmm2_volatility','none'),
          ('hmm3_defensive','none'),('hmm3_volatility','none'),('hmm3_growth','none'),
          ('vol20','drawdown'),('hmm3_volatility','drawdown'),('vol20','cppi'),('none','stop8')]

def screening(d,h,features,count=100):
    x=d[features];y=d[f'target_{h}'];groups=d.date
    cov=x.notna().mean();xc=x-x.groupby(groups).transform('mean');yc=y-y.groupby(groups).transform('mean')
    numerator=xc.mul(yc,axis=0).groupby(groups).sum()
    denom=np.sqrt(xc.pow(2).groupby(groups).sum().mul(yc.pow(2).groupby(groups).sum(),axis=0))
    ics=numerator/denom;avg=ics.mean();std=ics.std()
    rank=pd.DataFrame({'factor':features,'mean_ic':avg,'ic_std':std,'coverage':cov})
    rank['icir']=rank.mean_ic/rank.ic_std;rank['screen_score']=rank.mean_ic.abs()
    rank=rank[rank.coverage>=.7].sort_values(['screen_score','factor'],ascending=[False,True])
    corr=x[rank.factor.tolist()].fillna(0).corr().abs();chosen=[]
    for c in rank.factor:
        if not chosen or corr.loc[c,chosen].max()<.92:chosen.append(c)
        if len(chosen)>=count:break
    rank['selected']=rank.factor.isin(chosen)
    return chosen,rank

def ic_value(frame,pred,h):
    a=pd.DataFrame({'date':frame.date.to_numpy(),'p':pred,'y':frame[f'target_{h}'].to_numpy()})
    a=a.dropna();ranked=a[['p','y']].groupby(a.date).rank();centered=ranked-ranked.groupby(a.date).transform('mean')
    numerator=(centered.p*centered.y).groupby(a.date).sum()
    denominator=np.sqrt(centered.p.pow(2).groupby(a.date).sum()*centered.y.pow(2).groupby(a.date).sum())
    return float((numerator/denominator).mean())

def conditional_weights(train,h,cols):
    x=train[cols].fillna(0).to_numpy();y=train[f'target_{h}'].to_numpy()
    weights=[]
    for k in range(3):
        w=train['hmm3_p_'+str(k)].fillna(1/3).to_numpy()
        # Weighted correlation; all outcomes fully observed before fitting cutoff.
        xm=np.average(x,axis=0,weights=w);ym=np.average(y,weights=w)
        v=((x-xm)*(y-ym)[:,None]*w[:,None]).sum(axis=0)
        den=np.sqrt(((x-xm)**2*w[:,None]).sum(axis=0)*((y-ym)**2*w).sum())
        z=np.nan_to_num(v/np.maximum(den,1e-12));weights.append(z/(np.abs(z).sum()+1e-12))
    return np.array(weights)

def fit_predict(train,test,h,selected,registry):
    predictions={};objects={};y=train[f'target_{h}'].to_numpy()
    common={'n_estimators':180,'learning_rate':.035,'num_leaves':15,'max_depth':5,'min_child_samples':160,
            'reg_lambda':12.,'colsample_bytree':.8,'verbosity':-1,'n_jobs':3,'random_state':SEED}
    definitions={
      'ridge60':Ridge(alpha=180),
      'lgbm100':LGBMRegressor(**common),
      'ranker60':LGBMRanker(**common,objective='lambdarank',label_gain=list(range(10))),
      'catboost60':CatBoostRegressor(iterations=180,depth=5,learning_rate=.035,l2_leaf_reg=15,loss_function='RMSE',verbose=False,thread_count=3,random_seed=SEED,allow_writing_files=False),
      'xgboost60':XGBRegressor(n_estimators=180,max_depth=3,learning_rate=.035,min_child_weight=150,reg_lambda=15,subsample=.8,colsample_bytree=.8,n_jobs=3,random_state=SEED,tree_method='hist'),
      'extra60':ExtraTreesRegressor(n_estimators=120,max_depth=6,min_samples_leaf=150,max_features=.6,n_jobs=3,random_state=SEED),
      'llm_ridge':Ridge(alpha=180)}
    for name,m in definitions.items():
        cols=([c for c in train if c.startswith('x_llm_')] if name=='llm_ridge' else selected[:100 if name=='lgbm100' else 60])
        x=train[cols].fillna(0).to_numpy(dtype=np.float32);z=test[cols].fillna(0).to_numpy(dtype=np.float32)
        if name=='ranker60':m.fit(x,np.minimum(9,np.floor((y+.5)*10)).astype(int),group=train.groupby('date',sort=False).size().to_numpy())
        else:m.fit(x,y)
        predictions[name]=m.predict(z);objects[name]={'model':m,'features':cols}
    cols=selected[:60];weights=conditional_weights(train,h,cols)
    probabilities=test[['hmm3_p_'+str(k) for k in range(3)]].fillna(1/3).to_numpy()
    predictions['conditional_ic']=(test[cols].fillna(0).to_numpy()*(probabilities@weights)).sum(axis=1)
    objects['conditional_ic']={'weights':weights,'features':cols}
    w=registry.set_index('factor').loc[cols,'mean_ic'].to_numpy();w=w/(np.abs(w).sum()+1e-12)
    predictions['ic']=test[cols].fillna(0).to_numpy()@w;objects['ic']={'weights':w,'features':cols}
    ranked=pd.DataFrame({k:predictions[k] for k in ['ridge60','lgbm100','catboost60','xgboost60']},index=test.index)
    predictions['ensemble']=ranked.groupby(test.date).rank(pct=True).mean(axis=1).to_numpy()
    objects['ensemble']={'members':['ridge60','lgbm100','catboost60','xgboost60']}
    return predictions,objects

def choose(market,v,h,start,end):
    ics={name:ic_value(v,v[name],h) for name in MODELS}
    shortlist=sorted(ics,key=lambda k:(-np.nan_to_num(ics[k],nan=-1),k))[:3]
    rows=[]
    for model in shortlist:
        for top in [10,20]:
            for policy,guard in PROFILES:
                _,m,*_=market.run(v,horizon=h,score=model,topn=top,policy=policy,guard=guard,start=start,end=end)
                cal=m['calmar'] if m['calmar'] is not None else -10
                objective=min(m['sharpe_rf15'],cal/2)
                rows.append({'model':model,'topn':top,'policy':policy,'guard':guard,'objective':objective,'validation_ic':ics[model],**m})
    ranked=sorted(rows,key=lambda m:(-m['objective'],-m['annual_return'],m['model'],m['topn'],m['policy'],m['guard']))
    return ranked[0],rows,ics

def main(horizons):
    snapshot_hash=json.loads((RUN/'data_summary.json').read_text(encoding='utf-8'))['panel_sha256']
    dump({'models':MODELS,'profiles':PROFILES,'topn':[10,20],'shortlist':'top three validation mean daily rank IC',
          'selection':'maximize min(validation Sharpe_rf1.5, validation Calmar/2); annual return tie break',
          'training_sessions':504,'validation_sessions':126,'refit':'quarterly','max_features':100,
          'dedup_absolute_correlation':.92,'min_feature_coverage':.7,'seed':SEED},RUN/'model_registry.json')
    p=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','amount','adv20','observed','vol20','mom20']);a=pd.read_csv(DATA/'actions.csv');reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date')
    panel=pd.read_parquet(DATA/'model_panel.parquet').sort_values(['date','stockid']).reset_index(drop=True)
    panel=panel.merge(reg[['hmm3_p_'+str(k) for k in range(3)]],left_on='date',right_index=True,how='left')
    features=[c for c in panel if c.startswith('x_')];market=Market(p,a,reg);dates=market.dates
    quarters=sorted(panel.loc[panel.date>='2024-10-01','date'].dt.to_period('Q').unique())
    for h in horizons:
        outdir=RUN/('h'+str(h));outdir.mkdir(exist_ok=True)
        blocks=[];folds=[];settings=[]
        for quarter in quarters:
            checkpoint=outdir/(str(quarter)+'_predictions.parquet');choicefile=outdir/(str(quarter)+'_selection.json')
            cached_choice=json.loads(choicefile.read_text()) if choicefile.exists() else {}
            same_input=cached_choice.get('input_sha256')==snapshot_hash
            if checkpoint.exists() and same_input and cached_choice.get('ledger_version')==2:
                q=pd.read_parquet(checkpoint);choice=json.loads(choicefile.read_text());blocks.append(q);folds.append(choice)
                print('cached',h,str(quarter),flush=True);continue
            test=panel[panel.date.dt.to_period('Q')==quarter].copy()
            if test.empty:continue
            cutoff=test.date.min();past=dates[dates<cutoff];validation_dates=past[-126:];training_dates=past[-630:-126]
            train=panel[panel.date.isin(training_dates)&(panel[f'label_exit_{h}']<validation_dates[0])].dropna(subset=[f'target_{h}'])
            valid=panel[panel.date.isin(validation_dates)].copy()
            valid.loc[valid[f'label_exit_{h}']>=cutoff,f'target_{h}']=np.nan
            if train.date.nunique()<200 or valid.date.nunique()<70:continue
            cols,card=screening(train,h,features);card.to_csv(outdir/(str(quarter)+'_validation_factors.csv'),index=False)
            vp,_=fit_predict(train,valid,h,cols,card)
            for k,v in vp.items():valid[k]=v
            chosen,grid,ics=choose(market,valid,h,validation_dates[0],validation_dates[-1])
            pd.DataFrame(grid).to_csv(outdir/(str(quarter)+'_validation_grid.csv'),index=False)
            valid[['date','stockid',f'target_{h}']+MODELS].to_parquet(outdir/(str(quarter)+'_validation_predictions.parquet'),index=False)
            # At the quarter boundary refit using only fully matured labels in the latest 504 sessions.
            outer_train=panel[panel.date.isin(past[-504:])&(panel[f'label_exit_{h}']<cutoff)].dropna(subset=[f'target_{h}'])
            cols,card=screening(outer_train,h,features);card.to_csv(outdir/(str(quarter)+'_deployment_factors.csv'),index=False)
            if checkpoint.exists() and same_input:q=pd.read_parquet(checkpoint)
            else:
                predictions,objects=fit_predict(outer_train,test,h,cols,card)
                q=test[['date','stockid',f'target_{h}',f'label_{h}']].copy()
                for k,v in predictions.items():q[k]=v
                joblib.dump(objects,outdir/(str(quarter)+'_models.joblib'),compress=3)
            q['score']=q[chosen['model']];q['quarter']=str(quarter)
            q.to_parquet(checkpoint,index=False)
            choice={'ledger_version':2,'input_sha256':snapshot_hash,'quarter':str(quarter),'horizon':h,'validation_start':str(validation_dates[0]),'validation_end':str(validation_dates[-1]),
                    'training_start':str(outer_train.date.min()),'training_last_signal':str(outer_train.date.max()),
                    'training_last_label_exit':str(outer_train[f'label_exit_{h}'].max()),'test_start':str(cutoff),'test_end':str(test.date.max()),
                    'training_rows':len(outer_train),'validation_rows':len(valid),'selected':chosen,'validation_ics':ics}
            dump(choice,choicefile);blocks.append(q);folds.append(choice)
            print('fit',h,str(quarter),chosen['model'],chosen['policy'],chosen['guard'],'validation_objective',round(chosen['objective'],3),flush=True)
        allpred=pd.concat(blocks).sort_values(['date','stockid']);allpred.to_parquet(outdir/'predictions.parquet',index=False)
        for f in folds:
            days=allpred.loc[allpred.quarter==f['quarter'],'date'].unique()
            settings.extend({'date':d,'topn':f['selected']['topn'],'policy':f['selected']['policy'],'guard':f['selected']['guard']} for d in days)
        schedule=pd.DataFrame(settings).set_index('date').sort_index();schedule.to_csv(outdir/'schedule.csv');dump(folds,outdir/'folds.json')
        start=allpred.date.min();end=allpred.date.max()
        daily,m,orders,trades,holdings=market.run(allpred,horizon=h,start=start,end=end,schedule=schedule,save_details=True)
        for name,frame in [('daily',daily),('orders',orders),('trades',trades),('holdings',holdings)]:frame.to_csv(outdir/('selected_'+name+'.csv'),index=False)
        m['model']='validation_selected';dump(m,outdir/'selected_metrics.json')
        print('completed_horizon',h,'sharpe',m['sharpe_rf15'],'calmar',m['calmar'],'mdd',m['max_drawdown'],flush=True)

if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--horizons',default='1,2,5,10,20');opts=args.parse_args()
    with warnings.catch_warnings(),threadpool_limits(limits=3):
        warnings.simplefilter('ignore',FutureWarning);main([int(x) for x in opts.horizons.split(',')])
