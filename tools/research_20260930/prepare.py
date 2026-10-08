"""Versioned research inputs, causal liquidity pools, factor and price quality audits."""
from __future__ import annotations
import collections, hashlib, io, json, re, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from audit_server_inventory import decode
DATA=ROOT/'training_data/research_20260930'
RUN=ROOT/'training_runs/teacher9_20260930'
RENAME={'日期':'date','股票代码':'code','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume','成交额':'amount'}
PRICE_FEATURES=['mom5','mom10','mom20','mom60','vol10','vol20','vol60','adv20','amount_ratio','range20','log_illiq','observed_days']

def dump(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def read_csv(path): return pd.read_csv(io.StringIO(decode(path.read_bytes())))

def read_zip(path,stock_prefix=None):
    with zipfile.ZipFile(path) as z:
        name=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        text=decode(z.read(name))
        if stock_prefix:
            lines=text.splitlines()
            key_index=lines[0].split(',').index('stockid')
            lines=[lines[0]]+[line for line in lines[1:] if line.split(',',key_index+1)[key_index].strip('"').startswith(stock_prefix)]
            text='\n'.join(lines)
        return pd.read_csv(io.StringIO(text),na_values=['<NIL>','NIL'])

def prices(market,calendar):
    frames=[]; audits=[]
    start=pd.Timestamp('2021-11-15' if market=='bj' else '2016-01-01')
    cal=calendar[calendar>=start]
    for p in sorted((DATA/market/'raw').glob('*.csv')):
        raw=read_csv(p).rename(columns=RENAME)
        raw['date']=pd.to_datetime(raw.date)
        dup=int(raw.duplicated('date').sum())
        if dup: raise ValueError(f'duplicate price dates: {p.name}')
        qpath=DATA/market/'qfq'/p.name
        q=read_csv(qpath).rename(columns=RENAME);q['date']=pd.to_datetime(q.date)
        if q.duplicated('date').any(): raise ValueError('duplicate qfq dates')
        numeric=['open','high','low','close','volume','amount']
        raw[numeric]=raw[numeric].apply(pd.to_numeric,errors='coerce')
        q[['open','high','low','close']]=q[['open','high','low','close']].apply(pd.to_numeric,errors='coerce')
        x=raw.set_index('date').reindex(cal)[numeric].copy()
        adjusted=q.set_index('date').reindex(cal)[['open','high','low','close']]
        ohlc=x[['open','high','low','close']]
        spread=ohlc.max(axis=1)-ohlc.min(axis=1)
        diff=ohlc-adjusted
        additive=(diff.max(axis=1)-diff.min(axis=1))<0.025
        xc=ohlc.sub(ohlc.mean(axis=1),axis=0);yc=adjusted.sub(adjusted.mean(axis=1),axis=0)
        slope=(xc*yc).sum(axis=1)/xc.pow(2).sum(axis=1).replace(0,np.nan)
        slope=slope.where(~additive,1).where(spread>0.01).ffill()
        intercept=adjusted.mean(axis=1)-slope*ohlc.mean(axis=1)
        breaks=(slope.diff().abs()>0.01)|(intercept.diff().abs()>0.03)
        # Candidate regime boundaries, not verified dividends/splits.
        x['adjustment_boundary']=breaks.fillna(False)
        x['qfq_close']=adjusted.close;x['qfq_open']=adjusted.open
        code=re.match(r'(\d{6})',p.name).group(1)
        sid=('BJ' if market=='bj' else 'SH' if code.startswith('6') else 'SZ')+code
        x['stockid']=sid;x.index.name='date'
        x['observed']=(x.amount>0)&(x.volume>0)&(x[['open','high','low','close']]>0).all(axis=1)
        x['observed_days']=x.observed.cumsum()
        x['active20']=x.observed.rolling(20,min_periods=20).sum()
        x['adv20']=x.amount.fillna(0).rolling(20,min_periods=20).mean()
        x['volume20']=x.volume.fillna(0).rolling(20,min_periods=20).mean()
        x['ret1']=x.close.pct_change(fill_method=None)
        for h in [5,10,20,60]:x[f'mom{h}']=x.close/x.close.shift(h)-1
        for h in [10,20,60]:x[f'vol{h}']=x.ret1.rolling(h,min_periods=int(h*.8)).std()
        x['amount_ratio']=x.amount.fillna(0).rolling(5).mean()/x.adv20.replace(0,np.nan)
        x['range20']=((x.high-x.low)/x.close.shift(1)).rolling(20,min_periods=16).mean()
        x['log_illiq']=np.log((x.ret1.abs()/x.amount.where(x.amount>0)).rolling(20,min_periods=16).mean().clip(lower=1e-16))
        x['eligible']=(x.observed_days>=60)&(x.active20>=16)&x.observed&(x.adv20>0)
        x['entry_date']=pd.Series(cal,index=cal).shift(-1)
        x['exit_date']=pd.Series(cal,index=cal).shift(-5)
        x['label_raw']=x.close.shift(-5)/x.open.shift(-1)-1
        x.loc[~x.observed.shift(-1,fill_value=False),'label_raw']=np.nan
        x['label_qfq_diagnostic']=x.qfq_close.shift(-5)/x.qfq_open.shift(-1)-1
        x['label_adjustment_boundary']=sum(x.adjustment_boundary.shift(-j,fill_value=False).astype(int) for j in range(2,6))>0
        audits.append({'stockid':sid,'file':p.name,'rows':len(raw),'first_observed':str(x.index[x.observed][0]) if x.observed.any() else None,
            'last_observed':str(x.index[x.observed][-1]) if x.observed.any() else None,'zero_amount_rows':int((x.amount==0).sum()),
            'negative_qfq_close_rows':int((x.qfq_close<=0).sum()),'additive_ohlc_fraction':float(additive[ohlc.close.notna()].mean()),
            'candidate_adjustment_boundaries':int(breaks.sum()),'duplicate_dates':dup})
        frames.append(x.reset_index())
    panel=pd.concat(frames,ignore_index=True).sort_values(['date','stockid']).reset_index(drop=True)
    for source,column in [('adv20','amount_rank'),('volume20','volume_rank')]:
        panel[column]=panel[source].where(panel.eligible).groupby(panel.date).rank(ascending=False,method='first')
    panel['in_pool']=panel.amount_rank<=100;panel['in_volume_pool']=panel.volume_rank<=100
    panel['exclude_reason']=np.select([panel.observed_days<60,panel.active20<16,~panel.observed,panel.amount_rank>100],
        ['observed_history_below_60','active_days_below_16_of_20','no_valid_current_trade','liquidity_rank_over_100'],default='included')
    panel.to_parquet(DATA/(market+'_prices.parquet'),index=False)
    pd.DataFrame(audits).to_csv(RUN/(market+'_price_quality.csv'),index=False,encoding='utf-8-sig')
    panel[['date','stockid','observed_days','active20','adv20','amount_rank','volume_rank','in_pool','in_volume_pool','exclude_reason']].to_parquet(RUN/(market+'_daily_universe.parquet'),index=False)
    daily=panel.groupby('date').agg(pool=('in_pool','sum'),volume_pool=('in_volume_pool','sum'),eligible=('eligible','sum'))
    daily['pool_overlap']=panel.assign(both=panel.in_pool&panel.in_volume_pool).groupby('date').both.sum()
    daily.to_csv(RUN/(market+'_universe_daily_summary.csv'),encoding='utf-8-sig')
    print('prices_prepared',market,len(panel),int(panel.in_pool.sum()),flush=True)
    return panel

def latest_quality():
    fields=[];tables=[]
    for p in sorted((DATA/'latest_factors').glob('*.zip')):
        df=read_zip(p);keys=df.stockid.astype(str)
        valid=keys.str.fullmatch(r'(BJ|SH|SZ)\d{6}')
        duplicate=int(df.duplicated(['stockid']).sum())
        columns=[c for c in df if c not in ['stockid','enddate','rdate']]
        tables.append({'table':p.stem,'rows':len(df),'fields':len(columns),'invalid_stockids':int((~valid).sum()),'duplicate_stockids':duplicate,
            'bj_rows':int(keys.str.startswith('BJ').sum()),'enddates':str(dict(collections.Counter(df.enddate)))})
        for c in columns:
            vals=pd.to_numeric(df.loc[valid,c],errors='coerce');finite=vals.replace([np.inf,-np.inf],np.nan)
            fields.append({'factor_id':p.stem+'.'+c,'table':p.stem,'field':c,'snapshot':'2026-09-29','rows':len(vals),
                'finite_coverage':float(finite.notna().mean()),'infinite':int(np.isinf(vals).sum()),'distinct':int(finite.nunique()),
                'constant':finite.nunique()<=1,'bj_coverage':float(finite[keys[valid].str.startswith('BJ')].notna().mean()),
                'meaning_status':'supplier_dictionary_not_individually_verified','availability_assumption':'filename date after close; financial publication history unverified'})
    pd.DataFrame(fields).to_csv(RUN/'all_factor_registry_quality.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tables).to_csv(RUN/'all_factor_table_quality.csv',index=False,encoding='utf-8-sig')
    return {'tables':len(tables),'fields':len(fields),'constant_snapshot_fields':sum(f['constant'] for f in fields),'low_coverage_snapshot_fields':sum(f['finite_coverage']<.7 for f in fields),'invalid_stockid_rows':sum(t['invalid_stockids'] for t in tables)}

def factor_panel():
    target=DATA/'bj_factor_panel.parquet'
    if target.exists():return pd.read_parquet(target)
    wide=[]
    for p in sorted((ROOT/'factor_demo_run/factor_lib').glob('Alpha*.parquet')):
        x=pd.read_parquet(p);x['date']=pd.to_datetime(x.date)
        if x.duplicated(['date','stockid']).any():raise ValueError('duplicate local factor keys')
        wide.append(x.set_index(['date','stockid']).value.rename('c01_'+p.stem))
    old=pd.concat(wide,axis=1);del wide
    samples=[]
    for suffix in ['c01','c02']:
        frames=[];paths=sorted((DATA/'factors'/suffix).glob('*/*.zip'))
        for i,p in enumerate(paths):
            x=read_zip(p,stock_prefix='BJ')
            x=x[x.stockid.astype(str).str.fullmatch(r'BJ\d{6}')].copy()
            if x.duplicated('stockid').any():raise ValueError(f'duplicate factor keys {p}')
            cols=[c for c in x if c not in ['stockid','enddate','rdate']]
            x=x[['stockid']+cols]
            ren={c:('c01_Alpha'+c.rsplit('_',1)[-1] if suffix=='c01' else suffix+'_'+c) for c in cols}
            x=x.rename(columns=ren);cols=list(ren.values())
            x[cols]=x[cols].apply(pd.to_numeric,errors='coerce').astype('float32')
            x=x.copy()
            x['date']=pd.to_datetime(p.stem)
            frames.append(x.set_index(['date','stockid']))
            if (i+1)%200==0: print('parsed',suffix,i+1,'/',len(paths),flush=True)
        part=pd.concat(frames).sort_index();del frames
        if suffix=='c01':old=pd.concat([old,part])
        else:old=old.join(part,how='outer')
    old=old.replace([np.inf,-np.inf],np.nan).astype('float32').reset_index()
    if old.duplicated(['date','stockid']).any():raise ValueError('duplicate combined factor keys')
    old.to_parquet(target,index=False)
    print('factors_prepared',old.shape,flush=True)
    return old

def build_panel(market,price,factors=None):
    factor_cols=[]
    if factors is not None:
        factor_cols=[c for c in factors if c not in ['date','stockid']]
        price=price.merge(factors,on=['date','stockid'],how='left',validate='one_to_one')
        price['factor_coverage']=price[factor_cols].notna().mean(axis=1)
        ordered=price.sort_values(['stockid','date'])
        lagged=ordered.groupby('stockid')[factor_cols].shift(1)
        lagged.columns=['lag_'+c for c in factor_cols]
        ordered=pd.concat([ordered,lagged],axis=1)
        price=ordered.sort_values(['date','stockid'])
    pool=price[price.in_pool].copy()
    features=PRICE_FEATURES+factor_cols
    rank=pool.groupby('date')[features].rank(pct=True).sub(.5).astype('float32');rank.columns=['x_'+c for c in features]
    pool=pd.concat([pool,rank],axis=1)
    if factor_cols:
        lag=pool.groupby('date')[['lag_'+c for c in factor_cols]].rank(pct=True).sub(.5).astype('float32')
        lag.columns=['xl_'+c for c in factor_cols];pool=pd.concat([pool,lag],axis=1)
        pool.drop(columns=['lag_'+c for c in factor_cols],inplace=True)
    pool['target']=pool.groupby('date').label_raw.rank(pct=True).sub(.5)
    pool.to_parquet(DATA/(market+'_model_panel.parquet'),index=False)
    return {'rows':len(pool),'stocks':pool.stockid.nunique(),'features':len(features),'signal_start':str(pool.date.min()),'signal_end':str(pool.date.max()),
            'factor_coverage_mean':float(pool.factor_coverage.mean()) if factor_cols else None,
            'factor_coverage_below_70pct_rows':int((pool.factor_coverage<.7).sum()) if factor_cols else None}

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    frozen=ROOT/'training_runs/lgbm_hmm_20260920'
    if not (RUN/'frozen_old_run_hashes.json').exists():dump({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir() if p.is_file()},RUN/'frozen_old_run_hashes.json')
    calfile=next((DATA/'benchmark').glob('000001.SH*'))
    calendar=pd.DatetimeIndex(pd.to_datetime(read_csv(calfile)['日期'])).sort_values().unique()
    calendar=calendar[calendar<=pd.Timestamp('2026-09-29')]
    bj=prices('bj',calendar);shsz=prices('shsz',calendar)
    quality=latest_quality();dump(quality,RUN/'latest_factor_quality_summary.json')
    factors=factor_panel()
    summary={'bj':build_panel('bj',bj,factors),'shsz':build_panel('shsz',shsz),'latest_factor_quality':quality,
        'price_returns':'unadjusted price-only research returns, not total returns; qfq ratios diagnostic only',
        'eligibility':'60 observed valid trading days, >=16/20 active, current positive OHLC/amount/volume; history proxy not official listing eligibility',
        'survivorship':'available-file universe; historical security master incomplete; SHSZ deterministic convenience sample'}
    dump(summary,RUN/'data_summary.json');print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
