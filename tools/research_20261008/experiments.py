"""Predeclared retrospective extensions; all outer-quarter choices use past validation only."""
from pathlib import Path
import sys,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from prepare import DATA,PRICE_FEATURES,dump
from train import CONFIG,model_metrics
from accounts import simulate
from fast_correlation import daily_spearman
OLD=ROOT/'training_runs/teacher9_20260930'
RUN=ROOT/'training_runs/teacher9_20261008';RUN.mkdir(exist_ok=True,parents=True)
GRID=[20,40,60,80,100]
PROTOCOL={'written_before_training':True,'counts':GRID,'correlation_threshold':.85,
 'candidate_budget':'all training-qualified original 304 features, not only top 100; nested selections',
 'period':'2024Q4-2026Q3, exact frozen September 30 panel and trading rules',
 'selection':'Each quarter choose by mean validation daily Rank IC, tie smaller count; refit only past labels',
 'count_models':['LightGBM fixed leaves=15','Ridge alpha=10'],
 'extra_models':['40 features leaves 7/31','Alpha101 only','Alpha191 only','40 plus lagged capital proxies','Alpha lag 2 trading days','40 leaves15 sampling seeds 17/29'],
 'limitations':'Retrospective extensions on a previously inspected period; no fresh holdout and no causal/live claim. Features ranked within current eligible pool. Missing values retained for trees, 0 imputation for Ridge.',
 'costs_bp':[5,15,30],'new_count_main':'validation-selected LightGBM count, not outer-return winner'}

def train_predict(train,valid,refit,test,features,directory,name,leaves=15,seed=None,ridge=False):
    if ridge:
        model=Ridge(alpha=10).fit(train[features].fillna(0),train.target)
        val=model.predict(valid[features].fillna(0))
        model=Ridge(alpha=10).fit(refit[features].fillna(0),refit.target)
        pred=model.predict(test[features].fillna(0));best=None
    else:
        params=dict(CONFIG['lgbm']);params['num_leaves']=leaves
        if seed is not None:params.update(random_state=seed,colsample_bytree=.8,subsample=.8,subsample_freq=1)
        model=lgb.LGBMRegressor(**params).fit(train[features],train.target,eval_set=[(valid[features],valid.target)],callbacks=[lgb.early_stopping(35,verbose=False)])
        best=max(1,model.best_iteration_);val=model.predict(valid[features]);params['n_estimators']=best
        model=lgb.LGBMRegressor(**params).fit(refit[features],refit.target);pred=model.predict(test[features])
        (directory/(name+'.txt')).write_text(model.booster_.model_to_string(),'utf-8')
    v=valid[['date','stockid','label_raw','target']].copy();v['score']=val
    metric,_=model_metrics(v,'score')
    return pred,float(metric['mean_ic']),best

def features_extra(panel):
    # Lag two on the full market calendar, before pool filtering; never lag by sparse pool membership.
    f=pd.read_parquet(DATA/'bj_factor_panel.parquet')
    fcols=[c for c in f if c not in ['date','stockid']]
    prices=pd.read_parquet(DATA/'bj_prices.parquet',columns=['date','stockid'])
    full=prices.merge(f,on=['date','stockid'],how='left',validate='one_to_one').sort_values(['stockid','date'])
    lag=full.groupby('stockid')[fcols].shift(2);lag.columns=['xll_'+c for c in fcols]
    lag=pd.concat([full[['date','stockid']].reset_index(drop=True),lag.reset_index(drop=True)],axis=1)
    panel=panel.merge(lag,on=['date','stockid'],how='left',validate='one_to_one')
    lc=list(lag.columns[2:]);panel[lc]=panel.groupby('date')[lc].rank(pct=True).sub(.5).astype('float32')
    c=pd.read_parquet(DATA/'bj_share_capital.parquet')
    c['cap_log_float']=np.log(c.circulating_shares.where(~c.capital_ordering_warning & (c.circulating_shares>0)))
    c['cap_turnover']=c.turnover_circulating_proxy
    c=c.sort_values(['stockid','date']);c['cap_turnover20']=c.groupby('stockid').cap_turnover.transform(lambda s:s.rolling(20,min_periods=16).mean())
    names=['cap_log_float','cap_turnover','cap_turnover20']
    # Use previous calendar day's snapshot, no hindsight report-date backfill.
    full=prices.merge(c[['date','stockid']+names],on=['date','stockid'],how='left',validate='one_to_one').sort_values(['stockid','date'])
    full[names]=full.groupby('stockid')[names].shift(1)
    panel=panel.merge(full[['date','stockid']+names],on=['date','stockid'],how='left',validate='one_to_one')
    panel[names]=panel.groupby('date')[names].rank(pct=True).sub(.5).astype('float32')
    return panel

def train_all():
    dump(PROTOCOL,RUN/'extension_protocol.json')
    panel=features_extra(pd.read_parquet(DATA/'bj_model_panel.parquet'))
    cols=[c for c in panel if c.startswith('x_')];dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    oldfolds=json.loads((OLD/'bj_folds.json').read_text('utf-8'));folds=[];frames=[];results=[]
    for oldfold in oldfolds:
        q=oldfold['quarter'];directory=RUN/'bj'/q;directory.mkdir(parents=True,exist_ok=True)
        checkpoint=directory/'predictions.parquet';meta=directory/'result.json'
        if checkpoint.exists() and meta.exists():
            saved=json.loads(meta.read_text('utf-8'));frames.append(pd.read_parquet(checkpoint));folds.append(saved['fold']);results.extend(saved['models']);print('reuse',q,flush=True);continue
        test=panel[panel.date.dt.to_period('Q')==pd.Period(q)].copy();begin=test.date.min();history=dates[dates<begin]
        valid_start=history[-126];train_start=history[-630]
        train=panel[(panel.date>=train_start)&(panel.date<valid_start)&(panel.exit_date<valid_start)&panel.target.notna()]
        valid=panel[(panel.date>=valid_start)&(panel.date<begin)&(panel.exit_date<begin)&panel.target.notna()]
        refit=panel[(panel.date>=train_start)&(panel.date<begin)&(panel.exit_date<begin)&panel.target.notna()]
        cards=pd.read_csv(OLD/'bj'/q/'factor_cards.csv',index_col=0)
        candidates=cards[(cards.coverage>=.7)&(cards.n_ic_days>=100)&cards.mean_ic.notna()].sort_values(['abs_ic','coverage'],ascending=False).index.tolist()
        corrfile=directory/'mean_daily_spearman.csv'
        if corrfile.exists():corr=pd.read_csv(corrfile,index_col=0)
        else:
            sums=np.zeros((len(candidates),len(candidates)));counts=np.zeros_like(sums)
            for _,g in train.groupby('date'):
                a=daily_spearman(g[candidates],min_periods=30).to_numpy();sums+=np.nan_to_num(a);counts+=np.isfinite(a)
            corr=pd.DataFrame(np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0),index=candidates,columns=candidates);corr.to_csv(corrfile)
        selected=[]
        for c in candidates:
            if not any(abs(corr.loc[c,s])>=.85 for s in selected):selected.append(c)
        dump({'candidate_count':len(candidates),'deduplicated_count':len(selected),'ordered_features':selected},directory/'selection.json')
        pred=test[['date','stockid','entry_date','exit_date','label_raw','target','adv20','close']].copy();models=[]
        def fit(name,features,**kw):
            values,vic,best=train_predict(train,valid,refit,test,features,directory,name,**kw)
            pred[name]=values;models.append({'quarter':q,'model':name,'features':len(features),'feature_ids':features,'validation_ic':vic,'iterations':best})
            print('trained',q,name,len(features),'validIC',round(vic,5),flush=True)
        for k in GRID:
            fit(f'count_{k}',selected[:k]);fit(f'ridge_{k}',selected[:k],ridge=True)
        for leaves in [7,31]:fit(f'leaves_{leaves}',selected[:40],leaves=leaves)
        for family in ['c01','c02']:fit(f'family_{family}',[c for c in selected if c.startswith('x_'+family)][:40])
        fit('capital_lag1',selected[:40]+['cap_log_float','cap_turnover','cap_turnover20'])
        fit('lag2',[c.replace('x_c','xll_c') if c.startswith('x_c') else c for c in selected[:40]])
        for seed in [17,29]:fit(f'sampling_seed_{seed}',selected[:40],seed=seed)
        count_choices=sorted([m for m in models if m['model'].startswith('count_')],key=lambda m:(-m['validation_ic'],m['features']))
        ridge_choices=sorted([m for m in models if m['model'].startswith('ridge_')],key=lambda m:(-m['validation_ic'],m['features']))
        parameter_choices=sorted([m for m in models if m['model'].startswith(('count_','leaves_'))],key=lambda m:(-m['validation_ic'],m['features'],m['model']))
        for name,choices in [('validation_count',count_choices),('validation_ridge',ridge_choices),('validation_count_leaves',parameter_choices)]:pred[name]=pred[choices[0]['model']]
        fold={'quarter':q,'train_start':str(train_start),'train_last':str(train.date.max()),'train_label_end':str(train.exit_date.max()),'validation_start':str(valid_start),'refit_label_end':str(refit.exit_date.max()),'test_start':str(begin),'test_end':str(test.date.max()),'selected_count_model':count_choices[0]['model'],'selected_ridge_model':ridge_choices[0]['model'],'selected_count_leaves_model':parameter_choices[0]['model'],'available_after_dedup':len(selected)}
        assert train.exit_date.max()<valid_start and refit.exit_date.max()<begin
        pred.to_parquet(checkpoint,index=False);dump({'fold':fold,'models':models},meta)
        folds.append(fold);frames.append(pred);results.extend(models)
    scored=pd.concat(frames,ignore_index=True);scored.to_parquet(RUN/'bj_predictions.parquet',index=False)
    dump(folds,RUN/'folds.json');dump(results,RUN/'validation_models.json')
    pd.DataFrame([{k:v for k,v in r.items() if k!='feature_ids'} for r in results]).to_csv(RUN/'validation_models.csv',index=False)

def account_all():
    scored=pd.read_parquet(RUN/'bj_predictions.parquet');prices=pd.read_parquet(DATA/'bj_prices.parquet')
    base=['date','stockid','entry_date','exit_date','label_raw','target','adv20','close'];scores=[c for c in scored if c not in base]
    out=[]
    for score in scores:
        dest=RUN/(score+'_account.csv');mp=RUN/(score+'_metrics.json')
        if dest.exists() and mp.exists():m=json.loads(mp.read_text('utf-8'));out.append(m);continue
        daily,trades,orders,m=simulate(scored,prices,score=score)
        pm,dm=model_metrics(scored,score);m.update(pm);m['model']=score
        daily.to_csv(dest,index=False);dm.to_csv(RUN/(score+'_daily_ic.csv'),index=False)
        if score in ['validation_count','count_40']:
            trades.to_csv(RUN/(score+'_trades.csv'),index=False);orders.to_csv(RUN/(score+'_orders.csv'),index=False)
        dump(m,mp);out.append(m);print('account',score,round(m['total_return'],5),flush=True)
    pd.DataFrame(out).to_csv(RUN/'extension_results.csv',index=False)
    for bp in [5,30]:
        daily,_,_,m=simulate(scored,prices,score='validation_count',cost_bp=bp)
        daily.to_csv(RUN/f'validation_count_cost_{bp}_account.csv',index=False);dump(m,RUN/f'validation_count_cost_{bp}_metrics.json')

if __name__=='__main__':
    with threadpool_limits(limits=4):
        if '--accounts' not in sys.argv:train_all()
        account_all()
