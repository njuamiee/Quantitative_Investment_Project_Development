"""Calendar-aligned CSI research, explicitly conditional on unverified membership.
No future outcome is used to define the prediction or portfolio universe.
"""
from pathlib import Path
import os,sys,json,time,gc,warnings
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',category=RuntimeWarning)
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'training_data/audit_rebuild_20261021'
RUN=ROOT/'training_runs/audit_rebuild_20261021/csi';RUN.mkdir(parents=True,exist_ok=True)
F=[f'oho_alpha_{i:03d}' for i in range(1,102)]
SEED=20261021

def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def ic(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<10:return np.nan
    a=pd.Series(a[m]).rank().to_numpy();b=pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else np.nan

def prepare():
    old=ROOT/'training_data/research_20261021/market/panel_member.parquet'
    p=pd.read_parquet(old,columns=['date','stockid']+F).sort_values(['date','stockid']).reset_index(drop=True)
    p.date=p.date.astype('datetime64[ns]')
    cal=pd.DatetimeIndex(sorted(p.date.unique()));ids=sorted(p.stockid.unique())
    grid={}
    for kind in ['raw','hfq']:
        pr=pd.read_parquet(DATA/f'prices_{kind}.parquet')
        pr=pr[pr.stockid.isin(ids)&pr.date.isin(cal)]
        for c in ['open','close','volume']:
            grid[kind+'_'+c]=pr.pivot(index='date',columns='stockid',values=c).reindex(index=cal,columns=ids).to_numpy(float)
        del pr;gc.collect()
    grid['dates']=cal.to_numpy();grid['ids']=np.array(ids)
    np.savez_compressed(RUN/'market_grid.npz',**grid)
    di=cal.get_indexer(p.date);si=pd.Index(ids).get_indexer(p.stockid)
    op=grid['hfq_open'];cl=grid['hfq_close'];ro=grid['raw_open'];rc=grid['raw_close']
    def label(a,b,h):
        v=np.full(len(p),np.nan);m=di+h<len(cal)
        entry=a[di[m]+1,si[m]];exit=b[di[m]+h,si[m]]
        v[m]=np.where((entry>0)&(exit>0),exit/entry-1,np.nan);return v
    meta=p[['date','stockid']].copy();meta['di']=di;meta['si']=si
    meta['label']=label(op,op,6);meta['label_oc']=label(op,cl,5);meta['label_raw_oc']=label(ro,rc,5)
    ex=np.full(len(p),np.datetime64('NaT'),dtype='datetime64[ns]');m=di+6<len(cal);ex[m]=cal.to_numpy()[di[m]+6]
    meta['label_end']=ex;meta['eligible_now']=np.isfinite(p[F].to_numpy(np.float32)).sum(axis=1)>=51
    x=np.lib.format.open_memmap(RUN/'X.npy',mode='w+',dtype=np.float32,shape=(len(p),101))
    rows=[]
    for d,idx in p.groupby('date',sort=True).indices.items():
        v=p.loc[idx,F].to_numpy(float);v[~np.isfinite(v)]=np.nan
        v=np.clip(v,np.nanquantile(v,.01,axis=0),np.nanquantile(v,.99,axis=0))
        sd=np.nanstd(v,axis=0);sd[sd<1e-12]=np.nan
        x[idx]=np.nan_to_num((v-np.nanmean(v,axis=0))/sd).astype(np.float32)
        lab=meta.loc[idx,'label'].to_numpy();fr=pd.DataFrame(v).rank().to_numpy();lr=pd.Series(lab).rank().to_numpy()
        vals=[]
        for j in range(101):
            valid=np.isfinite(fr[:,j])&np.isfinite(lr)
            vals.append(np.corrcoef(fr[valid,j],lr[valid])[0,1] if valid.sum()>30 and np.std(fr[valid,j])>0 else np.nan)
        rows.append([d]+vals)
    x.flush();del x,p
    meta['target']=meta.label-meta.groupby('date').label.transform('mean')
    meta.to_parquet(RUN/'meta.parquet',index=False)
    pd.DataFrame(rows,columns=['date']+F).to_parquet(RUN/'factor_ic.parquet',index=False)
    dump(dict(rows=len(meta),dates=len(cal),start=cal[0],end=cal[-1],eligible=int(meta.eligible_now.sum()),
        observed_label=float(meta.label.notna().mean()),prices='fresh SFTP hfq, full calendar before membership filtering',
        membership='user event reconstruction; announcement/effective time and final period unresolved'),RUN/'data_audit.json')
    print('prepared',len(meta),flush=True)

def meanic(meta,pred):
    return float(np.nanmean([ic(pred[idx],g.label.to_numpy()) for _,g in meta.assign(_i=np.arange(len(meta))).groupby('date') for idx in [g._i.to_numpy()]]))

def fit_one(meta,x,cal,icf,trdates,vdates,tdates,label,fold):
    cutoff=pd.Timestamp(tdates[0]);vstart=pd.Timestamp(vdates[0])
    tridx=np.flatnonzero(meta.date.isin(trdates)&(meta.label_end<vstart)&meta.target.notna()&meta.eligible_now)
    vidx=np.flatnonzero(meta.date.isin(vdates)&(meta.label_end<cutoff)&meta.target.notna()&meta.eligible_now)
    tidx=np.flatnonzero(meta.date.isin(tdates)&meta.eligible_now)
    assert meta.iloc[tridx].label_end.max()<vstart
    assert meta.iloc[vidx].label_end.max()<cutoff
    Xtr=np.array(x[tridx]);Xv=np.array(x[vidx]);Xt=np.array(x[tidx]);yt=meta.target.to_numpy()[tridx];yv=meta.target.to_numpy()[vidx]
    vm=meta.iloc[vidx].reset_index(drop=True)
    weights=icf.loc[vdates].mean()/icf.loc[vdates].std();weights=weights.replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
    result=meta.iloc[tidx][['date','stockid','di','si','label','label_end']].copy()
    result['icir']=Xt@weights
    scores=[];best=(-np.inf,None,None)
    with threadpool_limits(limits=3):
        for a in [1.,100.]:
            m=Ridge(alpha=a).fit(Xtr,yt);value=meanic(vm,m.predict(Xv));scores.append(dict(alpha=a,val_ic=value))
            if value>best[0]:best=(value,a,m)
    result['ridge']=best[2].predict(Xt)
    rounds={}
    for name,params in [('lgbm31',dict(num_leaves=31,min_child_samples=100,learning_rate=.05)),
                        ('lgbm15_transfer',dict(num_leaves=15,max_depth=5,min_child_samples=160,lambda_l2=12.,learning_rate=.035))]:
        base=dict(objective='regression',verbosity=-1,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,seed=SEED,num_threads=3,force_col_wise=True)
        base.update(params)
        # Transfer uses original rank target and regularization, not the original fitted strategy.
        ty=yt if name=='lgbm31' else meta.iloc[tridx].groupby('date').label.rank(pct=True).to_numpy()-.5
        vy=yv if name=='lgbm31' else vm.groupby('date').label.rank(pct=True).to_numpy()-.5
        booster=lgb.train(base,lgb.Dataset(Xtr,label=ty),num_boost_round=300,
                          valid_sets=[lgb.Dataset(Xv,label=vy)],callbacks=[lgb.early_stopping(30,verbose=False)])
        result[name]=booster.predict(Xt);rounds[name]=booster.best_iteration
        (RUN/f'{label}_{fold}_{name}.txt').write_text(booster.model_to_string(),encoding='utf-8')
    record=dict(fold=fold,mode=label,train_start=trdates[0],train_end=trdates[-1],val_start=vdates[0],val_end=vdates[-1],
       predict_start=tdates[0],predict_end=tdates[-1],actual_train_label_max=meta.iloc[tridx].label_end.max(),
       actual_val_label_max=meta.iloc[vidx].label_end.max(),train_rows=len(tridx),val_rows=len(vidx),ridge=scores,rounds=rounds)
    result.to_parquet(RUN/f'{label}_pred_{fold:02d}.parquet',index=False)
    dump(record,RUN/f'{label}_fold_{fold:02d}.json')
    print(label,fold,str(tdates[0])[:10],rounds,flush=True)
    return result,record

def train():
    meta=pd.read_parquet(RUN/'meta.parquet');x=np.load(RUN/'X.npy',mmap_mode='r')
    cal=pd.DatetimeIndex(sorted(meta.date.unique()));icf=pd.read_parquet(RUN/'factor_ic.parquet').set_index('date')
    frames=[];logs=[];start=252+63+12
    for k,t in enumerate(range(start,len(cal),21)):
        tr=cal[t-6-63-6-252:t-6-63-6];v=cal[t-6-63:t-6];test=cal[t:min(t+21,len(cal))]
        p=RUN/f'rolling_pred_{k:02d}.parquet'
        if p.exists():r=pd.read_parquet(p);log=json.loads((RUN/f'rolling_fold_{k:02d}.json').read_text(encoding='utf-8'))
        else:r,log=fit_one(meta,x,cal,icf,tr,v,test,'rolling',k)
        frames.append(r);logs.append(log)
    pd.concat(frames,ignore_index=True).to_parquet(RUN/'rolling_predictions.parquet',index=False)
    dump(logs,RUN/'rolling_folds.json')
    # All tuning/model fitting before the originally assigned train cutoff.
    # Fixed weights and trees are held unchanged throughout V and test.
    end=cal.searchsorted(pd.Timestamp('2025-06-04'))
    valend=end-6;v=cal[valend-63:valend];tr=cal[valend-63-6-252:valend-63-6]
    test=cal[cal>='2025-07-02']
    r,log=fit_one(meta,x,cal,icf,tr,v,test,'fixed',0)
    assert pd.Timestamp(log['actual_val_label_max'])<=pd.Timestamp('2025-06-03')
    r.to_parquet(RUN/'fixed_predictions.parquet',index=False)

def stats(d,initial=1.):
    e=d.equity.to_numpy(float);r=np.r_[e[0]/initial-1,e[1:]/e[:-1]-1]
    dd=1-np.r_[initial,e]/np.maximum.accumulate(np.r_[initial,e]);sd=r.std(ddof=1)
    ann=(e[-1]/initial)**(252/len(e))-1
    return dict(days=len(e),total=e[-1]/initial-1,annual=ann,mdd=dd.max(),sharpe=r.mean()/sd*np.sqrt(252) if sd>1e-12 else None,
                calmar=ann/dd.max() if dd.max()>1e-12 else None,exposure=d.exposure.mean(),cash_days=int((d.exposure<.01).sum()))

def ledger(pred,grid,col,cost=0.,delay=0,exclude_unknown=False,topn=20):
    dates=pd.DatetimeIndex(grid['dates']);op=grid['hfq_open'];cl=grid['hfq_close']
    begin=dates.get_loc('2025-07-02');end=len(dates)-1;fee=cost/10000
    bydate={d:g for d,g in pred.groupby('date')};cash=1.;held={};last={};due=None;rows=[];selection=[];failures=[]
    for i in range(begin,end+1):
        d=dates[i];costday=0.;traded=0.
        if held:
            for s,(q,entry) in list(held.items()):
                if i<entry+5:continue
                if np.isfinite(op[i,s]) and op[i,s]>0:
                    value=q*op[i,s];cash+=value*(1-fee);costday+=value*fee;traded+=value;del held[s]
                else:failures.append(dict(date=d,si=s,reason='missing_exit_open_carry_forward'))
            if not held:due=None
        signal=i-1-delay
        if signal>=begin and (signal-begin)%5==0:
            g=bydate.get(dates[signal])
            if g is not None:
                if exclude_unknown and dates[signal]>=pd.Timestamp('2026-09-15'):
                    g=g[~g.stockid.isin(['SZ001268','SZ002259','SH603680','SZ300277'])]
                if col=='equal_members':picks=g.sort_values('stockid')
                else:picks=g.sort_values([col,'stockid'],ascending=[False,True]).head(topn)
                n=len(picks) if col=='equal_members' else topn
                existing={s:q*(op[i,s] if np.isfinite(op[i,s]) and op[i,s]>0 else last[s]) for s,(q,entry) in held.items()}
                open_equity=cash+sum(existing.values())
                targets=[max(0.,open_equity/n-existing.get(int(r.si),0.)) if n else 0. for r in picks.itertuples(index=False)]
                scale=min(1.,cash/max(sum(targets),1e-12))
                for r,want in zip(picks.itertuples(index=False),targets):
                    s=int(r.si);price=op[i,s]
                    budget=want*scale
                    if budget<=1e-14:continue
                    if np.isfinite(price) and price>0:
                        q=budget/(price*(1+fee));cash-=budget;held[s]=(q,i);last[s]=price;costday+=budget-budget/(1+fee);traded+=budget/(1+fee)
                        selection.append(dict(signal_date=dates[signal],entry_date=d,stockid=r.stockid,si=s,weight=1/n))
                    else:failures.append(dict(date=d,si=s,reason='missing_entry_open_keep_cash'))
                due=i+5
        val=0.
        for s,(q,entry) in held.items():
            if np.isfinite(cl[i,s]) and cl[i,s]>0:last[s]=cl[i,s]
            else:failures.append(dict(date=d,si=s,reason='missing_close_last_known_mark'))
            val+=q*last[s]
        eq=cash+val
        assert cash>=-1e-8 and eq>0
        rows.append(dict(date=d,equity=eq,cash=cash,market_value=val,exposure=val/eq,cost=costday,turnover=traded/eq,positions=len(held)))
    return pd.DataFrame(rows),pd.DataFrame(selection),pd.DataFrame(failures)

def evaluate():
    grid=dict(np.load(RUN/'market_grid.npz'));allstats=[];daily=[];ics=[]
    for mode in ['rolling','fixed']:
        pred=pd.read_parquet(RUN/f'{mode}_predictions.parquet')
        for m in ['icir','ridge','lgbm31','lgbm15_transfer','equal_members']:
            scenarios=[(0.,0,False)]
            if mode=='rolling' and m!='equal_members':scenarios +=[(5.,0,False),(15.,0,False),(0.,1,False),(0.,0,True)]
            for cost,delay,exc in scenarios:
                name=f'{mode}__{m}__cost{cost:g}__delay{delay}__exclude{int(exc)}'
                d,s,f=ledger(pred,grid,m,cost,delay,exc)
                d.to_csv(RUN/(name+'_daily.csv'),index=False,encoding='utf-8-sig')
                if cost==0 and delay==0 and not exc:
                    s.to_csv(RUN/(name+'_selection.csv'),index=False,encoding='utf-8-sig')
                    f.to_csv(RUN/(name+'_missing_price.csv'),index=False,encoding='utf-8-sig')
                d['strategy']=name;daily.append(d)
                for period,a,b in [('V','2025-07-02','2026-05-21'),('test','2026-05-22','2026-10-19'),('test_old_start','2026-05-25','2026-10-19'),('full','2025-07-02','2026-10-19'),('Q2','2026-04-01','2026-06-30'),('Q3','2026-07-01','2026-09-30')]:
                    sub=d[d.date.between(a,b)];before=d[d.date<a]
                    if len(sub):allstats.append(dict(strategy=name,period=period,**stats(sub,float(before.equity.iloc[-1]) if len(before) else 1.),price_events=len(f)))
            if m!='equal_members':
                for date,g in pred.groupby('date'):
                    ics.append(dict(mode=mode,model=m,date=date,rank_ic=ic(g[m].to_numpy(),g.label.to_numpy())))
    pd.DataFrame(allstats).to_csv(RUN/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(ics).to_csv(RUN/'daily_ic.csv',index=False,encoding='utf-8-sig')
    pd.concat(daily).to_parquet(RUN/'all_daily.parquet',index=False)
    print(pd.DataFrame(allstats).query("period=='test'")[['strategy','total','mdd','sharpe','exposure']].to_string(index=False),flush=True)

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ['all','prepare']:prepare()
    if mode in ['all','train']:train()
    if mode in ['all','evaluate']:evaluate()
