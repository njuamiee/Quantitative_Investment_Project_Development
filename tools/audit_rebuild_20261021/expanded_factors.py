"""All-table registry and predeclared rolling factor-library comparisons for BJ."""
from pathlib import Path
import os,sys,json,gc,warnings
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',category=RuntimeWarning)
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'training_data/audit_rebuild_20261021'
RUN=ROOT/'training_runs/audit_rebuild_20261021/expanded';RUN.mkdir(parents=True,exist_ok=True)
PUB={'001','017','018','051','053','054','055','057','058','101','102','103','104','105','106','107','108','201','701','071','601'}
REL={str(i).zfill(3) for i in range(72,101)}|{'120','121','122'}
MICRO={'005','006','007','008','009','021','022','027','028','032','033','035','041','042','043','045','046','047','048','049','059','061','062','063','064','065','066','067','068','069'}|{f'a{i:02d}' for i in range(1,9)}

def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def rankic(x,y):
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<10:return np.nan
    a=pd.Series(x[m]).rank().to_numpy();b=pd.Series(y[m]).rank().to_numpy()
    return np.corrcoef(a,b)[0,1] if a.std()>0 and b.std()>0 else np.nan

def prepare():
    prices=pd.read_parquet(DATA/'original_inputs/prices.parquet')
    days=pd.to_datetime(json.loads((DATA/'library_signal_dates.json').read_text(encoding='utf-8')))
    meta=prices[prices.in_pool&prices.date.isin(days)][['date','stockid','label_5','label_exit_5']].sort_values(['date','stockid']).reset_index(drop=True)
    # Original last-six-date labels were truncated at its old vintage. Repair only labels against the full calendar.
    raw=pd.read_parquet(DATA/'prices_hfq.parquet',columns=['date','stockid','open','close']);raw=raw[raw.stockid.str.startswith('BJ')]
    cal=pd.DatetimeIndex(sorted(prices.date.unique()));ids=sorted(meta.stockid.unique())
    op=raw.pivot(index='date',columns='stockid',values='open').reindex(index=cal,columns=ids).to_numpy()
    cl=raw.pivot(index='date',columns='stockid',values='close').reindex(index=cal,columns=ids).to_numpy()
    di=cal.get_indexer(meta.date);si=pd.Index(ids).get_indexer(meta.stockid);valid=di+5<len(cal)
    y=np.full(len(meta),np.nan);ex=np.full(len(meta),np.datetime64('NaT'),dtype='datetime64[ns]')
    y[valid]=cl[di[valid]+5,si[valid]]/op[di[valid]+1,si[valid]]-1;ex[valid]=cal.to_numpy()[di[valid]+5]
    meta['label_5']=y;meta['label_exit_5']=ex;meta['target']=meta.groupby('date').label_5.rank(pct=True)-.5
    keys=pd.MultiIndex.from_frame(meta[['date','stockid']]);chunks=[];catalog=[]
    dictionary=pd.read_csv(DATA/'dictionary_0.csv');meanings={(r['表名'].split('_')[-1],r['字段']):r['含义'] for _,r in dictionary.iterrows()}
    for path in sorted((DATA/'library').glob('*.parquet')):
        table=path.stem;f=pd.read_parquet(path)
        cols=[c for c in f if c not in ['date','stockid']]
        if not cols:continue
        if f.duplicated(['date','stockid']).any():raise ValueError('duplicate factor keys')
        f=f.set_index(['date','stockid']).reindex(keys)
        f=f[cols].replace([np.inf,-np.inf],np.nan).astype('float32');chunks.append(f)
        for c in cols:
            catalog.append(dict(factor=c,table=table,meaning=meanings.get((table,c.split('__',1)[1]),'dictionary_missing'),
                 publication_sensitive=table in PUB,relationship_sensitive=table in REL,microstructure=table in MICRO,
                 coverage_descriptive=float(f[c].notna().mean()),unique_descriptive=int(f[c].nunique()),
                 primary_eligible=table not in PUB|REL,reason='publication_or_relationship_history_unverified' if table in PUB|REL else 'candidate_only_train_screen_required'))
    raw=pd.concat(chunks,axis=1);del chunks,prices;gc.collect()
    cols=list(raw);x=np.full(raw.shape,np.nan,dtype=np.float32)
    for d,idx in meta.groupby('date').indices.items():
        v=raw.iloc[idx].to_numpy(float);v=np.clip(v,np.nanquantile(v,.01,axis=0),np.nanquantile(v,.99,axis=0))
        sd=np.nanstd(v,axis=0);sd[sd<1e-10]=np.nan;x[idx]=((v-np.nanmean(v,axis=0))/sd).astype(np.float32)
    np.save(RUN/'X.npy',x);meta.to_parquet(RUN/'meta.parquet',index=False)
    pd.DataFrame(catalog).to_csv(RUN/'factor_registry.csv',index=False,encoding='utf-8-sig')
    dump(cols,RUN/'columns.json');dump(dict(tables=120,fields=len(cols),sample_dates=meta.date.nunique(),rows=len(meta),
       primary_fields=sum(c['primary_eligible'] for c in catalog),publication_sensitive=sum(c['publication_sensitive'] for c in catalog),relationship_sensitive=sum(c['relationship_sensitive'] for c in catalog),
       warning='No whole-period quality statistic used for feature selection; quarterly training screens own all selection.'),RUN/'registry_summary.json')
    # ICs are computed for research once, then sliced only to matured training days by each fold.
    out=[]
    for d,idx in meta.groupby('date').indices.items():
        v=x[idx];y=meta.iloc[idx].target.to_numpy();m=np.isfinite(v)&np.isfinite(y[:,None]);n=m.sum(axis=0)
        xx=np.where(m,v,0);yy=np.where(m,y[:,None],0)
        den=np.sqrt(np.maximum(n*(xx*xx).sum(axis=0)-xx.sum(axis=0)**2,0)*np.maximum(n*(yy*yy).sum(axis=0)-yy.sum(axis=0)**2,0))
        ic=np.divide(n*(xx*yy).sum(axis=0)-xx.sum(axis=0)*yy.sum(axis=0),den,out=np.full(len(cols),np.nan),where=(den>1e-9)&(n>=20))
        out.append(ic)
    np.save(RUN/'factor_ic.npy',np.array(out));dump([str(pd.Timestamp(d).date()) for d in sorted(meta.date.unique())],RUN/'factor_ic_dates.json')
    print('expanded registry',len(cols),'fields',meta.shape,flush=True)

def valic(meta,p):
    vals=[]
    for idx in meta.groupby('date').indices.values():vals.append(rankic(p[idx],meta.iloc[idx].label_5.to_numpy()))
    return float(np.nanmean(vals))

def screen(x,ix,ics,icdays,trdates,candidates,cols,limit=100):
    cov=np.isfinite(x[np.ix_(ix,candidates)]).mean(axis=0)
    pos=icdays.get_indexer(trdates);pos=pos[pos>=0];a=ics[np.ix_(pos,candidates)]
    mean=np.nanmean(a,axis=0);std=np.nanstd(a,axis=0)
    # Conservative shrinkage and a source cap prevent hundreds of near copies dominating.
    utility=np.abs(mean)/(std+.05)*np.sqrt(np.isfinite(a).sum(axis=0)/max(1,len(a)))
    eligible=[int(candidates[j]) for j in np.argsort(-utility,kind='stable') if cov[j]>=.7 and np.isfinite(utility[j])]
    picked=[];cnt={};why={};cache=[]
    source_cap=max(20,int(np.ceil(100/max(1,len({cols[j].split('__')[0] for j in candidates})))))
    for j in eligible:
        table=cols[j].split('__')[0]
        if limit and len(picked)>=limit:why[j]='after_100_limit';continue
        if limit and cnt.get(table,0)>=source_cap:why[j]='source_cap_'+str(source_cap);continue
        v=np.nan_to_num(x[ix,j]);v=v-v.mean();v=v/(np.linalg.norm(v)+1e-12)
        if limit and cache and np.max(np.abs(np.asarray(cache)@v))>=.98:why[j]='train_abs_corr_ge_098';continue
        picked.append(j);cache.append(v) if limit else None;cnt[table]=cnt.get(table,0)+1;why[j]='selected'
    card=pd.DataFrame(dict(factor=[cols[j] for j in candidates],coverage=cov,mean_ic=mean,stability=utility,
                          decision=[why.get(j,'train_coverage_or_variance_failure') for j in candidates]))
    return picked,card

def train():
    meta=pd.read_parquet(RUN/'meta.parquet');x=np.load(RUN/'X.npy',mmap_mode='r');cols=json.loads((RUN/'columns.json').read_text(encoding='utf-8'))
    ics=np.load(RUN/'factor_ic.npy');icdays=pd.DatetimeIndex(pd.to_datetime(json.loads((RUN/'factor_ic_dates.json').read_text(encoding='utf-8'))))
    cal=pd.DatetimeIndex(sorted(pd.read_parquet(DATA/'original_inputs/prices.parquet',columns=['date']).date.unique()))
    start=cal.get_loc('2025-07-02');defs={
        'alpha101_lgbm':([i for i,c in enumerate(cols) if c.startswith('c01__')],'lgbm',False),
        'alpha292_lgbm':([i for i,c in enumerate(cols) if c[:3] in ['c01','c02']],'lgbm',False),
        'expanded_lgbm':([i for i,c in enumerate(cols) if c.split('__')[0] not in PUB|REL],'lgbm',False),
        'expanded_ridge_all':([i for i,c in enumerate(cols) if c.split('__')[0] not in PUB|REL],'ridge',True),
        'all_vendor_lgbm':(list(range(len(cols))),'lgbm',False),
        'all_vendor_ridge_all':(list(range(len(cols))),'ridge',True),
        'no_micro_lgbm':([i for i,c in enumerate(cols) if c.split('__')[0] not in PUB|REL|MICRO],'lgbm',False),
    }
    records=[];frames=[]
    for k,t in enumerate(range(start,len(cal),21)):
        if (RUN/f'pred_{k:02d}.parquet').exists():
            frames.append(pd.read_parquet(RUN/f'pred_{k:02d}.parquet'));records.extend(json.loads((RUN/f'fold_{k:02d}.json').read_text(encoding='utf-8')));continue
        trdates=cal[t-6-63-6-252:t-6-63-6];vdates=cal[t-6-63:t-6];tdates=cal[t:min(t+21,len(cal))]
        tr=np.flatnonzero(meta.date.isin(trdates)&meta.target.notna()&(meta.label_exit_5<vdates[0]))
        va=np.flatnonzero(meta.date.isin(vdates)&meta.target.notna()&(meta.label_exit_5<tdates[0]))
        te=np.flatnonzero(meta.date.isin(tdates));vm=meta.iloc[va].reset_index(drop=True)
        if not len(te):continue
        assert meta.iloc[tr].label_exit_5.max()<vdates[0] and meta.iloc[va].label_exit_5.max()<tdates[0]
        pf=meta.iloc[te][['date','stockid','label_5']].copy();logs=[]
        for name,(candidates,kind,allcols) in defs.items():
            picked,card=screen(x,tr,ics,icdays,pd.DatetimeIndex(meta.iloc[tr].date.unique()),candidates,cols,0 if allcols else 100)
            if not picked:raise ValueError('empty feature screen '+name)
            card.to_csv(RUN/f'fold{k:02d}_{name}_screen.csv',index=False,encoding='utf-8-sig')
            X=np.nan_to_num(x[np.ix_(tr,picked)]);V=np.nan_to_num(x[np.ix_(va,picked)]);T=np.nan_to_num(x[np.ix_(te,picked)])
            y=meta.iloc[tr].target.to_numpy();vy=vm.target.to_numpy()
            if kind=='ridge':
                best=(-np.inf,None,None)
                with threadpool_limits(3):
                    for a in [100.,1000.]:
                        m=Ridge(alpha=a).fit(X,y);value=valic(vm,m.predict(V))
                        if value>best[0]:best=(value,a,m)
                pred=best[2].predict(T);valid_ic=best[0];rounds=best[1]
            else:
                params=dict(objective='regression',verbosity=-1,learning_rate=.035,num_leaves=15,max_depth=5,
                   min_child_samples=80,lambda_l2=12.,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,
                   seed=20261021,num_threads=3,force_col_wise=True)
                m=lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=200,valid_sets=[lgb.Dataset(V,label=vy)],callbacks=[lgb.early_stopping(25,verbose=False)])
                pred=m.predict(T);valid_ic=valic(vm,m.predict(V));rounds=m.best_iteration
            pf[name]=pred
            logs.append(dict(fold=k,model=name,features=len(picked),val_ic=valid_ic,rounds_or_alpha=rounds,
                             train_start=trdates[0],train_end=trdates[-1],val_start=vdates[0],val_end=vdates[-1],test_start=tdates[0],test_end=tdates[-1],
                             train_actual_last_exit=meta.iloc[tr].label_exit_5.max(),val_actual_last_exit=meta.iloc[va].label_exit_5.max()))
        # ICIR reference uses only mature prior internal validation outcomes, same Alpha101 scope.
        aidx=np.array(defs['alpha101_lgbm'][0]);vp=icdays.get_indexer(pd.DatetimeIndex(meta.iloc[va].date.unique()));vv=ics[np.ix_(vp,aidx)]
        w=np.nan_to_num(np.nanmean(vv,axis=0)/(np.nanstd(vv,axis=0)+.05));pf['alpha101_icir']=np.nan_to_num(x[np.ix_(te,aidx)])@w
        pf.to_parquet(RUN/f'pred_{k:02d}.parquet',index=False);dump(logs,RUN/f'fold_{k:02d}.json')
        frames.append(pf);records.extend(logs);print('expanded fold',k,str(tdates[0])[:10],[(r['model'],r['features']) for r in logs],flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(RUN/'predictions.parquet',index=False)
    pd.DataFrame(records).to_csv(RUN/'folds.csv',index=False,encoding='utf-8-sig')
    out=[]
    for d,g in pd.concat(frames).groupby('date'):
        for c in list(defs)+['alpha101_icir']:out.append(dict(date=d,model=c,ic=rankic(g[c].to_numpy(),g.label_5.to_numpy())))
    pd.DataFrame(out).to_csv(RUN/'daily_ic.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    stage=sys.argv[1] if len(sys.argv)>1 else 'all'
    if stage in ['all','prepare']:prepare()
    if stage in ['all','train']:train()
