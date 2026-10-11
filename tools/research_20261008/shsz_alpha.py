"""Matched SH/SZ Alpha101 extension using the newly available factor library."""
import sys,json,concurrent.futures,re
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from audit_server_inventory import connect
import fetch_data as fetch
import prepare,train
from accounts import simulate
D=ROOT/'training_data/research_20261008/shsz_alpha';R=ROOT/'training_runs/teacher9_20261008/shsz_alpha'
D.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)

def download():
    sample=pd.read_parquet(D.parent/'Alpha001.parquet');stocks=set(sample.loc[sample.stockid.str.startswith(('SH','SZ')),'stockid'])
    fetch.OUT=D;jobs=[];t,s=connect()
    try:
        for a in s.listdir_attr('/home/quant/factors/factor_lib'):
            if re.fullmatch('Alpha\d{3}.parquet',a.filename):jobs.append(('/home/quant/factors/factor_lib/'+a.filename,'factors/'+a.filename,a.st_size,a.st_mtime))
        for adj,folder in [('不复权','raw'),('前复权','qfq')]:
            remote='/home/quant/day-data/'+adj
            for a in s.listdir_attr(remote):
                code=a.filename[:6];sid=('SH' if code.startswith('6') else 'SZ')+code
                if sid in stocks and a.filename.endswith('.csv'):jobs.append((remote+'/'+a.filename,'shsz/'+folder+'/'+a.filename,a.st_size,a.st_mtime))
    finally:s.close();t.close()
    print('SHSZ jobs',len(jobs),'MB',sum(j[2] for j in jobs)/1e6,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=[]
        for i,r in enumerate(pool.map(fetch.download,jobs),1):
            records.append(r)
            if i%100==0:print('SHSZ download',i,flush=True)
    prepare.dump({'stocks':sorted(stocks),'universe':f'{len(stocks)} currently available SH/SZ symbols in new Alpha001 library; selection provenance unknown; survivor bias remains','records':records},D/'manifest.json')
    for t,s in fetch.CONNECTIONS:
        try:s.close();t.close()
        except Exception:pass

def main():
    if not (D/'manifest.json').exists():download()
    prepare.DATA=D;prepare.RUN=R
    if not (D/'shsz_model_panel.parquet').exists():
        calfile=next((D.parent/'benchmark').glob('000001.SH*'))
        cal=pd.DatetimeIndex(pd.to_datetime(prepare.read_csv(calfile)['日期'])).sort_values().unique();cal=cal[cal<=pd.Timestamp('2026-08-21')]
        price=prepare.prices('shsz',cal);pool=price[price.in_pool].copy();parts=[]
        for p in sorted((D/'factors').glob('Alpha*.parquet')):
            f=pd.read_parquet(p);f=f[f.stockid.str.startswith(('SH','SZ'))]
            assert not f.duplicated(['date','stockid']).any()
            parts.append(f.set_index(['date','stockid']).value.rename('library_'+p.stem))
        factors=pd.concat(parts,axis=1).replace([np.inf,-np.inf],np.nan).reset_index()
        pool=pool.merge(factors,on=['date','stockid'],how='left',validate='one_to_one')
        features=prepare.PRICE_FEATURES+[c for c in pool if c.startswith('library_')]
        ranked=pool.groupby('date')[features].rank(pct=True).sub(.5).astype('float32');ranked.columns=['x_'+c for c in features]
        pool=pd.concat([pool,ranked],axis=1);pool['target']=pool.groupby('date').label_raw.rank(pct=True).sub(.5)
        pool.to_parquet(D/'shsz_model_panel.parquet',index=False)
        prepare.dump({'pool_rows':len(pool),'pool_stocks':pool.stockid.nunique(),'price_files':len(list((D/'shsz/raw').glob('*.csv'))),'features':len(features),'alpha_coverage':float(pool[[c for c in features if c.startswith('library_')]].notna().mean().mean()),'first':str(pool.date.min()),'last':str(pool.date.max())},R/'data_summary.json')
    train.DATA=D;train.RUN=R
    if not (R/'shsz_predictions.parquet').exists():train.rolling('shsz')
    scored=pd.read_parquet(R/'shsz_predictions.parquet');prices=pd.read_parquet(D/'shsz_prices.parquet');results=[]
    for score in ['pred_price','pred_selected','pred_ridge']:
        daily,_,_,m=simulate(scored,prices,score=score);m['model']=score;results.append(m);daily.to_csv(R/(score+'_account.csv'),index=False)
    prepare.dump(results,R/'account_results.json')
    print('SHSZ complete',flush=True)

if __name__=='__main__':main()
