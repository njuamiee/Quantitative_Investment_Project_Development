"""Read all 120 SFTP factor tables on a predeclared five-session grid.

Raw ZIPs and hashes retained. The server is read-only. Numeric BSE fields are
extracted with source names; announcement-time validity is assessed separately.
"""
from pathlib import Path
import sys,json,stat,re,threading,io,zipfile,hashlib,time,warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'training_data/audit_rebuild_20261021'
sys.path.insert(0,str(ROOT/'tools'))
from audit_server_inventory import connect,decode
LOCAL=threading.local(); CONNECTIONS=[]; LOCK=threading.Lock()

def sf():
    if not hasattr(LOCAL,'s'):
        t,s=connect();s.get_channel().settimeout(90)
        LOCAL.s=s
        with LOCK:CONNECTIONS.append((t,s))
    return LOCAL.s

def process_table(item,dates):
    table=item['table']; suffix=table.split('_')[-1]
    dest=OUT/'library'/f'{suffix}.parquet'
    audit=OUT/'library'/f'{suffix}.json'
    if dest.exists() and audit.exists():
        previous=json.loads(audit.read_text(encoding='utf-8'))
        if not previous.get('errors'):return {k:v for k,v in previous.items() if k!='records'}
    frames=[]; records=[]; errors=[]
    for year in range(2023,2027):
        if str(year) not in item.get('years',[]):continue
        path=f'/home/quant/factors/kyzqsh11/{table}/{year}'
        try: listing=sf().listdir_attr(path)
        except Exception as e:
            errors.append({'year':year,'type':type(e).__name__});continue
        for a in sorted(listing,key=lambda a:a.filename):
            if a.filename[:8] not in dates or not a.filename.endswith('.zip'):continue
            remote=path+'/'+a.filename
            local=OUT/'library_raw'/suffix/a.filename
            local.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(3):
                try:
                    if not local.exists() or local.stat().st_size!=a.st_size:
                        part=local.with_suffix('.part');sf().get(remote,str(part))
                        if part.stat().st_size!=a.st_size:raise ValueError('size_changed')
                        part.replace(local)
                    raw=local.read_bytes()
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        name=next(n for n in z.namelist() if n.lower().endswith('.csv'))
                        f=pd.read_csv(io.StringIO(decode(z.read(name))),na_values=['<NIL>','NIL','--'],low_memory=False)
                    if not {'stockid','enddate'}.issubset(f.columns):raise ValueError('schema_keys_missing')
                    d=pd.to_datetime(pd.to_numeric(f.enddate,errors='raise'),unit='D',origin='1899-12-30')
                    snapshot=pd.Timestamp(a.filename[:8])
                    if (d>snapshot).any():raise ValueError('source_date_after_snapshot')
                    age=(snapshot-d).dt.days
                    old_rows=int(age.gt(7).sum())
                    # Financial/events have distinct timing semantics and remain diagnostic.
                    event_rejected=0
                    if 'actualstoragedate' in f:
                        storage=pd.to_datetime(pd.to_numeric(f.actualstoragedate,errors='coerce'),unit='D',origin='1899-12-30')
                        announcement=pd.to_datetime(f.announcementdate.astype(str),format='%Y%m%d',errors='coerce')
                        known=(storage<=snapshot)&(announcement<=snapshot)
                        event_rejected=int((~known).sum());f=f[known].copy()
                        f=f.sort_values('rdate').drop_duplicates('stockid',keep='last')
                    elif suffix not in ['001','051','053','054','055','057','058','101','102','103','104','105','106','107','108','201','701']:
                        numeric_cols=[c for c in f if c not in ['stockid','enddate','rdate','stockname','name']]
                        f.loc[age>7,numeric_cols]=np.nan
                    f=f[f.stockid.astype(str).str.startswith('BJ')].copy()
                    if f.duplicated('stockid').any():raise ValueError('duplicate_stockid')
                    result=pd.DataFrame({'date':pd.Timestamp(a.filename[:8]),'stockid':f.stockid.astype(str)})
                    for c in f.columns:
                        if c in ['stockid','enddate','rdate','name','stockname','announcementdate','actualstoragedate','datasources'] or c.startswith('Unnamed:'):continue
                        v=pd.to_numeric(f[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
                        # Keep numeric fields even if all missing in this single date.
                        result[suffix+'__'+c]=v.astype('float32')
                    frames.append(result)
                    records.append(dict(date=a.filename[:8],rows=len(f),bytes=len(raw),mtime=a.st_mtime,stale_gt7_rows=old_rows,late_storage_rows_removed=event_rejected,
                                        sha256=hashlib.sha256(raw).hexdigest(),remote=remote))
                    break
                except Exception as e:
                    if attempt==2:errors.append(dict(remote=remote,type=type(e).__name__,reason=str(e)[:120]))
                    else:time.sleep(.5)
    merged=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['date','stockid'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    merged.to_parquet(dest,index=False,compression='zstd')
    meta=dict(table=table,files=len(records),rows=len(merged),columns=list(merged),errors=errors,records=records)
    audit.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:v for k,v in meta.items() if k!='records'}

def main():
    cal=pd.read_csv(ROOT/'training_data/research_20261021/market/932000_official_daily_ext.csv',dtype={'tradeDate':str})
    days=pd.DatetimeIndex(sorted(pd.to_datetime(cal.tradeDate,format='%Y%m%d').unique()))
    anchor=days.get_loc('2024-10-08')
    chosen=[d for i,d in enumerate(days) if (i-anchor)%5==0 and d>=pd.Timestamp('2023-08-11')]
    dates={d.strftime('%Y%m%d') for d in chosen}
    (OUT/'library_signal_dates.json').write_text(json.dumps([str(d.date()) for d in chosen],indent=2),encoding='utf-8')
    catalog=json.loads((OUT/'remote_factor_catalog.json').read_text(encoding='utf-8'))
    print('PREDECLARED',len(dates),'dates',len(catalog),'tables',flush=True)
    t0=time.time();summary=[]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures={pool.submit(process_table,c,dates):c['table'] for c in catalog}
            for f in as_completed(futures):
                try:r=f.result()
                except Exception as e:r=dict(table=futures[f],error_type=type(e).__name__)
                summary.append(r)
                (OUT/'library_download_progress.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
                print(len(summary),'/',len(catalog),r.get('table'),r.get('files'),r.get('rows'),'errors',len(r.get('errors',[])),round(time.time()-t0),flush=True)
    finally:
        for t,s in CONNECTIONS:
            try:s.close();t.close()
            except Exception:pass

if __name__=='__main__':main()
