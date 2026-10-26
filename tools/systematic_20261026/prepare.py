"""Causal matrices, full-calendar labels, finite AI-authored factor hypotheses."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from config import ROOT,RUN,DATA,HORIZONS,dump,digest,registry

def rank_day(a, groups):
    out=np.full(a.shape,np.nan,dtype=np.float32)
    for ix in groups.values():
        v=a[ix]; n=np.isfinite(v).sum(axis=0)
        out[ix]=rankdata(v,axis=0,nan_policy='omit')/np.maximum(n,1)-.5
    return out

def main():
    out=RUN/'data'; out.mkdir(parents=True,exist_ok=True)
    prices=pd.read_parquet(DATA/'prices.parquet').sort_values(['date','stockid'])
    panel=pd.read_parquet(DATA/'model_panel.parquet').sort_values(['date','stockid']).reset_index(drop=True)
    cal=pd.DatetimeIndex(sorted(prices.date.unique())); ids=pd.Index(sorted(prices.stockid.unique()))
    def pivot(c):return prices.pivot(index='date',columns='stockid',values=c).reindex(index=cal,columns=ids).to_numpy()
    di=cal.get_indexer(panel.date); si=ids.get_indexer(panel.stockid)
    groups=panel.groupby('date',sort=True).indices
    base=[c for c in panel if c.startswith('x_') and not c.startswith('x_llm_')]
    x=panel[base].to_numpy(np.float32); names=list(base)
    # The proposals below are authored by Codex before this batch's outcomes.
    # They are hypotheses; approval/selection happens inside every training fold.
    raw=prices.sort_values(['stockid','date']).reset_index(drop=True)
    generated={}; proposals=[]
    for w in [2,3,5,10,15,20,40,60,120]:
        g=raw.groupby('stockid',sort=False)
        mom=raw.adj_close/g.adj_close.shift(w)-1
        vol=g.adj_ret1.transform(lambda s:s.rolling(max(w,5),min_periods=max(4,w//2)).std())
        high=g.adj_close.transform(lambda s:s.rolling(max(w,10),min_periods=max(5,w//2)).max())
        low=g.adj_close.transform(lambda s:s.rolling(max(w,10),min_periods=max(5,w//2)).min())
        dd=raw.adj_close/high-1
        position=(raw.adj_close-low)/(high-low).replace(0,np.nan)
        variables={f'ai_momentum_risk_{w}':mom/(vol+.001),
                   f'ai_reversal_risk_{w}':-mom/(vol+.001),
                   f'ai_recovery_{w}':raw.mom2-dd,
                   f'ai_range_position_{w}':position,
                   f'ai_liquid_reversal_{w}':-mom*raw.amount_ratio.clip(0,5),
                   f'ai_acceleration_{w}':raw.mom2-mom*2/w}
        for n,v in variables.items():
            generated[n]=v
            proposals.append(dict(factor=n,window=w,author='Codex',uses='past/current close, past returns, past/current traded amount',
                                  acceptance='not accepted merely by generation; train-only coverage, IC and correlation screen',
                                  no_live_llm_call=True))
    gf=pd.DataFrame(generated); gf.index=pd.MultiIndex.from_frame(raw[['date','stockid']])
    gx=gf.reindex(pd.MultiIndex.from_frame(panel[['date','stockid']])).to_numpy(float)
    gx=rank_day(gx,groups); auto=np.concatenate([x,gx],axis=1)
    np.save(out/'X_daily.npy',x); np.save(out/'X_daily_auto.npy',auto)
    dump(names,out/'names_daily.json');dump(names+list(generated),out/'names_daily_auto.json')
    dump(proposals,out/'ai_factor_proposals.json')
    # Source snapshot values are carried FORWARD at most four market sessions.
    # This is a deliberately delayed/as-of library, not invented daily vendor observations.
    old=ROOT/'training_runs/audit_rebuild_20261021/expanded'
    vm=pd.read_parquet(old/'meta.parquet'); vx=np.load(old/'X.npy',mmap_mode='r')
    reg=pd.read_csv(old/'factor_registry.csv'); eligible=np.flatnonzero(reg.primary_eligible.to_numpy(bool))
    vcols=json.loads((old/'columns.json').read_text(encoding='utf-8'))
    ix=np.full(len(panel),-1,dtype=int); age=np.full(len(panel),-1,dtype=int)
    for sid, rows in panel.groupby('stockid').indices.items():
        vr=np.flatnonzero(vm.stockid.eq(sid)); vd=cal.get_indexer(vm.iloc[vr].date)
        sort=np.argsort(vd); vr=vr[sort];vd=vd[sort]
        if not len(vr): continue
        pos=np.searchsorted(vd,di[rows],side='right')-1
        ok=pos>=0; a=di[rows]-vd[np.maximum(pos,0)];ok&=a<=4
        ix[rows[ok]]=vr[pos[ok]];age[rows[ok]]=a[ok]
    added=np.full((len(panel),len(eligible)),np.nan,dtype=np.float32)
    ok=ix>=0;added[ok]=vx[np.ix_(ix[ok],eligible)]
    vendor=np.concatenate([auto,rank_day(added,groups)],axis=1)
    np.save(out/'X_vendor_asof.npy',vendor)
    dump(names+list(generated)+['vendor_'+vcols[j] for j in eligible],out/'names_vendor_asof.json')
    np.save(out/'vendor_age.npy',age)
    # Compute every outcome on the SAME full market calendar, never on sparse pool membership.
    op=pivot('adj_open'); labels=np.full((len(panel),len(HORIZONS)),np.nan,dtype=np.float32)
    targets=np.full_like(labels,np.nan); exits=np.full(labels.shape,-1,dtype=np.int32)
    for j,h in enumerate(HORIZONS):
        valid=di+h+1<len(cal); rows=np.flatnonzero(valid)
        labels[rows,j]=op[di[rows]+h+1,si[rows]]/op[di[rows]+1,si[rows]]-1
        exits[rows,j]=di[rows]+h+1
        targets[:,j]=rank_day(labels[:,j:j+1],groups)[:,0]
    np.save(out/'labels.npy',labels);np.save(out/'targets.npy',targets);np.save(out/'label_exit_index.npy',exits)
    meta=panel[['date','stockid']].copy();meta['date_index']=di;meta['stock_index']=si
    meta.to_parquet(out/'meta.parquet',index=False)
    dump([str(d.date()) for d in cal],out/'calendar.json');dump(list(ids),out/'stocks.json')
    # Full raw-price grid and corporate-action arrays for an actual share/cash account.
    market={c:pivot(c).astype(np.float64) for c in ['open','close','adv20','vol20','mom2','mom5','mom20','adj_ret1','observed','in_pool','amount_rank','adj_close']}
    market['dividend']=np.zeros(op.shape);market['split']=np.ones(op.shape)
    for r in pd.read_csv(DATA/'actions.csv').itertuples():
        d=cal.get_indexer([pd.Timestamp(r.date)])[0];s=ids.get_indexer([r.stockid])[0]
        if d>=0 and s>=0:market['dividend'][d,s]=r.dividend;market['split'][d,s]=r.split
    md=pd.read_parquet(ROOT/'training_runs/audit_rebuild_20261021/original/risk_metadata.parquet')
    industry=sorted(md.industry.dropna().unique()); mp={v:i+1 for i,v in enumerate(industry) if str(v) not in ['unknown','nan']}
    md['industry_id']=md.industry.map(mp).fillna(0)
    market['industry']=md.pivot(index='date',columns='stockid',values='industry_id').reindex(index=cal,columns=ids).fillna(0).to_numpy()
    sizes=sorted(str(v) for v in md.size_tertile.dropna().unique() if str(v) not in ['unknown','nan'])
    md['size_id']=md.size_tertile.map({v:i+1 for i,v in enumerate(sizes)}).fillna(0).astype(float)
    market['size']=md.pivot(index='date',columns='stockid',values='size_id').reindex(index=cal,columns=ids).fillna(0).to_numpy()
    np.savez_compressed(out/'market.npz',**market)
    minute=pd.read_parquet(ROOT/'training_data/research_20261011/minute_factors.parquet')
    mincols=[c for c in minute if c not in ['date','stockid','bars']]
    mx=minute.set_index(['date','stockid']).reindex(pd.MultiIndex.from_frame(meta[['date','stockid']]))[mincols].to_numpy()
    np.save(out/'minute.npy',rank_day(mx,groups));dump(mincols,out/'minute_names.json')
    summary=dict(rows=len(meta),days=meta.date.nunique(),stocks=meta.stockid.nunique(),
                 start=str(meta.date.min()),end=str(meta.date.max()),
                 daily_features=len(base),ai_proposals=len(generated),vendor_eligible_fields=len(eligible),
                 vendor_snapshot_days=vm.date.nunique(),vendor_asof_max_sessions=4,
                 vendor_covered_rows=int(ok.sum()),minute_days=minute.date.nunique(),
                 minute_start=str(minute.date.min()),minute_end=str(minute.date.max()),
                 source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [DATA/'prices.parquet',DATA/'model_panel.parquet',DATA/'actions.csv',old/'X.npy',old/'meta.parquet']},
                 limitations=['supplier historic release times/revisions not independently verified',
                              'industry history follows prior project metadata; unknown is rejected under concentration constraints',
                              'gross dividends credited at ex-date: research scenario, not broker settlement',
                              'minute observations incomplete; no fabricated fills on absent minute dates'])
    dump(summary,out/'manifest.json');dump(registry(),RUN/'registry.json')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);main()
