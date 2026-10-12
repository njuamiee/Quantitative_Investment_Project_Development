"""Read-only metadata audit and hashed SFTP snapshot. Never writes remotely."""
from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, io, json, re, stat, sys, time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from audit_server_inventory import connect, decode
AUD=ROOT/'training_data/audit_20261011'
DATA=ROOT/'training_data/research_20261011'
BASE='/home/quant/quantDataProcess/backend/data'
CUTOFF='2026-10-09'

def dump(x,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def listing(s,path):
    try:
        return [{'path':path+'/'+a.filename,'name':a.filename,'directory':stat.S_ISDIR(a.st_mode),
                 'bytes':a.st_size,'mtime':a.st_mtime} for a in s.listdir_attr(path)]
    except OSError as e:
        return [{'path':path,'error':type(e).__name__+': '+str(e)}]

def dated_scan(path):
    t,s=connect();s.get_channel().settimeout(60)
    try:
        root=listing(s,path);rows=[]
        for a in root:
            if a.get('directory') and re.fullmatch(r'20\d{2}',a['name']):rows.extend(listing(s,a['path']))
            elif not a.get('directory'):rows.append(a)
        return path,rows
    finally:s.close();t.close()

def inventory():
    AUD.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    out={'started_utc':datetime.now(timezone.utc).isoformat(),'cutoff':CUTOFF,
         'scope':'all listed date files and current file metadata; content sampled, not every value verified',
         'directories':{},'date_files':{},'errors':[]}
    t,s=connect();s.get_channel().settimeout(60)
    try:
        paths=['/home/quant',BASE,BASE+'/day-data',BASE+'/dividend',BASE+'/bar1m',BASE+'/tick',
               '/home/quant/day-data','/home/quant/factors','/home/quant/factors/kyzqsh11']
        for p in paths:out['directories'][p]=listing(s,p)
        for p in [BASE+'/day-data/tickflow/'+x for x in ['raw','qfq','hfq']]+[BASE+'/day-data/benchmark/raw']:
            out['directories'][p]=listing(s,p)
        factorpaths=[a['path'] for a in out['directories']['/home/quant/factors/kyzqsh11']
                     if a.get('directory') and a['name'].startswith('ts_stockfactordata_')]
    finally:s.close();t.close()
    jobs=factorpaths+[BASE+'/'+kind+'/'+m for kind in ['bar1m','tick'] for m in ['BJ','SH','SZ']]
    jobs+=['/home/quant/tick-All/'+m for m in ['BJ','SH','SZ']]
    dump(out,AUD/'inventory.partial.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,(p,rows) in enumerate(pool.map(dated_scan,jobs),1):
            out['date_files'][p]=rows
            if i%20==0:dump(out,AUD/'inventory.partial.json');print('audited',i,'/',len(jobs),flush=True)
    out['ended_utc']=datetime.now(timezone.utc).isoformat();dump(out,AUD/'inventory.json')
    old=json.loads((ROOT/'training_data/audit_20261008/inventory.json').read_text(encoding='utf-8'))
    old_tables={x['table']:x for x in old['factors']};summary=[];missing=[]
    calendar=old.get('reference_calendar',[])
    if isinstance(calendar,dict):calendar=calendar.get('dates',[])
    expected=set(calendar)|{'2026-10-08','2026-10-09'}
    expected={d for d in expected if d<=CUTOFF}
    for p,rows in out['date_files'].items():
        names=[a['name'] for a in rows if 'name' in a and re.match(r'^\d{8}',a['name'])]
        dates=sorted({n[:4]+'-'+n[4:6]+'-'+n[6:8] for n in names})
        isfactor='ts_stockfactordata_' in p;table=p.rsplit('/',1)[-1]
        prev=None
        if isfactor and table in old_tables:
            prev=max((v.get('last') or '' for v in old_tables[table]['years'].values()),default='')
        elif '/bar1m/BJ' in p:prev='2026-09-30'
        summary.append({'path':p,'files':len(names),'first':dates[0] if dates else None,'last':dates[-1] if dates else None,
                        'previous_last':prev,'new_dates_since_oct8':','.join(d for d in dates if d>'2026-10-08'),
                        'new_dates_since_previous_last':','.join(d for d in dates if prev and d>prev),
                        'list_error':';'.join(a.get('error','') for a in rows if 'error' in a)})
        if dates:
            start=max(dates[0],'2021-11-15') if ('/BJ' in p or isfactor) else dates[0]
            for d in sorted(expected-set(dates)):
                if d<start:continue
                ext='.zip' if isfactor else '_Tick1.tar.gz' if 'tick-All' in p else '.parquet'
                filename=d.replace('-','')+ext
                missing.append({'category':'factor_date' if isfactor else 'market_date','date':d,
                                'expected_path':p+'/'+d[:4]+'/'+filename,
                                'note':'date absent from full flat/year listing; expected-calendar cross-check'})
        else:missing.append({'category':'unavailable_directory','date':'','expected_path':p,'note':'empty or inaccessible; see inventory errors'})
    write_csv(summary,AUD/'update_summary.csv');write_csv(missing,AUD/'missing_dated_files.csv')
    print('inventory_complete',len(summary),'date_missing_entries',len(missing),flush=True)
    return out

def write_csv(rows,p):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def download_one(job):
    remote,relative,size,mtime=job;p=DATA/relative;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size==size:
        return {'remote':remote,'local':relative,'bytes':size,'mtime':mtime,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'cached'}
    for attempt in range(3):
        t=s=None
        try:
            t,s=connect();s.get_channel().settimeout(90)
            tmp=p.with_suffix(p.suffix+'.part');s.get(remote,str(tmp))
            if tmp.stat().st_size!=size:raise ValueError('size changed during download')
            tmp.replace(p)
            return {'remote':remote,'local':relative,'bytes':size,'mtime':mtime,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'downloaded'}
        except Exception as e:
            if attempt==2:return {'remote':remote,'local':relative,'bytes':size,'mtime':mtime,'sha256':'','status':'failed:'+type(e).__name__}
        finally:
            if s:s.close()
            if t:t.close()

def sync(out):
    jobs=[]
    for adj in ['raw','qfq','hfq']:
        p=BASE+'/day-data/tickflow/'+adj
        for a in out['directories'].get(p,[]):
            if a.get('name','').endswith('.csv') and not a.get('directory'):
                # Current backend originally contains BJ only; include BJ codes and record metadata for other markets.
                code=a['name'].split('-')[0].split('.')[0]
                if code.startswith(('4','8','92')):jobs.append((a['path'],'bj/'+adj+'/'+a['name'],a['bytes'],a['mtime']))
    for a in out['directories'].get(BASE+'/day-data/benchmark/raw',[]):
        if a.get('name','').endswith('.csv'):jobs.append((a['path'],'benchmark/'+a['name'],a['bytes'],a['mtime']))
    for a in out['directories'].get(BASE+'/dividend',[]):
        if a.get('name')=='BJ.parquet':jobs.append((a['path'],'dividend_BJ.parquet',a['bytes'],a['mtime']))
    for suffix in ['c01','c02','071','601']:
        p='/home/quant/factors/kyzqsh11/ts_stockfactordata_'+suffix
        for a in out['date_files'].get(p,[]):
            n=a.get('name','')
            if re.fullmatch(r'\d{8}\.zip',n) and '20261008'<=n[:8]<=CUTOFF.replace('-',''):
                jobs.append((a['path'],'factors/'+suffix+'/'+n[:4]+'/'+n,a['bytes'],a['mtime']))
    # Existing 2025Q4 and 2026Q3 minute snapshots are reused by manifest; new late dates and known gaps downloaded.
    for a in out['date_files'].get(BASE+'/bar1m/BJ',[]):
        n=a.get('name','')
        if re.fullmatch(r'\d{8}\.parquet',n) and (n[:8]>='20260930' or n[:8] in ['20250630','20260518']):
            jobs.append((a['path'],'minute/'+n,a['bytes'],a['mtime']))
    print('download_jobs',len(jobs),'MB',round(sum(j[2] for j in jobs)/1e6),flush=True)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(download_one,jobs),1):
            records.append(r)
            if i%100==0 or i==len(jobs):
                dump({'captured_utc':datetime.now(timezone.utc).isoformat(),'complete':i==len(jobs),'records':records},DATA/'manifest.json')
                print('downloaded',i,'/',len(jobs),flush=True)
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sync-only',action='store_true');p.add_argument('--inventory-only',action='store_true');args=p.parse_args()
    try:
        out=json.loads((AUD/'inventory.json').read_text(encoding='utf-8')) if args.sync_only else inventory()
        if not args.inventory_only:sync(out)
    except Exception as e:
        dump({'utc':datetime.now(timezone.utc).isoformat(),'status':'failed','error_type':type(e).__name__,'error':str(e)},AUD/'connection_status.json')
        raise
