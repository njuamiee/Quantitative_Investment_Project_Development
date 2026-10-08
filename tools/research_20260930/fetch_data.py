"""Freeze existing SFTP inputs locally. Only list/read/get operations are used."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, re, sys, threading, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from audit_server_inventory import connect
OUT = ROOT / 'training_data/research_20260930'
LOCAL = threading.local()
CONNECTIONS = []

def client():
    if not hasattr(LOCAL, 'sftp'):
        t, s = connect(); s.get_channel().settimeout(120)
        LOCAL.sftp = s; CONNECTIONS.append((t,s))
    return LOCAL.sftp

def download(job):
    remote, relative, size, mtime = job
    path = OUT / relative; path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            if not path.exists() or path.stat().st_size != size:
                temporary = path.with_suffix(path.suffix + '.part')
                client().get(remote, str(temporary))
                if temporary.stat().st_size != size: raise ValueError('remote size changed')
                temporary.replace(path)
            return {'remote':remote,'local':relative,'bytes':size,'remote_mtime':mtime,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception:
            if attempt == 2: raise
            if hasattr(LOCAL,'sftp'):
                try: LOCAL.sftp.close()
                except Exception: pass
                del LOCAL.sftp
    raise RuntimeError('unreachable')

def main(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    t,s=connect(); jobs=[]
    def add_folder(remote, relative, predicate):
        for a in s.listdir_attr(remote):
            if predicate(a.filename): jobs.append((remote+'/'+a.filename,relative+'/'+a.filename,a.st_size,a.st_mtime))
    try:
        if stage=='prices':
            base='/home/quant/quantDataProcess/backend/data/day-data'
            for adj in ['raw','qfq']:
                add_folder(base+'/tickflow/'+adj,'bj/'+adj,lambda n:n.endswith('.csv'))
            add_folder(base+'/benchmark/raw','benchmark',lambda n:n.endswith('.csv'))
            # A reproducible sample from current available files, NOT a historical constituent list.
            attrs=s.listdir_attr('/home/quant/day-data/不复权')
            names=[a.filename for a in attrs if re.match(r'^(60|00)\d{4}-.*\.csv$',a.filename)]
            selected=[]
            for prefix in ['60','00']:
                pool=[n for n in names if n.startswith(prefix)]
                selected.extend(sorted(pool,key=lambda n:hashlib.sha256(n.split('-')[0].encode()).hexdigest())[:150])
            (OUT/'shsz_sample_selection.json').write_text(json.dumps({'method':'SHA256(code) first 150 per SH60/SZ00 from currently available files; survivor-biased convenience sample','files':selected},ensure_ascii=False,indent=2),encoding='utf-8')
            for adj,folder in [('不复权','raw'),('前复权','qfq')]:
                add_folder('/home/quant/day-data/'+adj,'shsz/'+folder,lambda n:n in selected)
        elif stage in ('factors','capital'):
            base='/home/quant/factors/kyzqsh11'
            sources=[('c01',['2026']),('c02',[str(y) for y in range(2021,2027)])] if stage=='factors' else [('071',[str(y) for y in range(2021,2027)]),('601',[str(y) for y in range(2021,2027)])]
            for suffix,years in sources:
                for year in years:
                    add_folder(base+'/ts_stockfactordata_'+suffix+'/'+year,'factors/'+suffix+'/'+year,
                        lambda n:bool(re.fullmatch(r'\d{8}\.zip',n)) and n>='20211115.zip' and n<='20260929.zip')
        elif stage=='samples':
            inventory=json.loads((ROOT/'training_data/audit_20260930/inventory.json').read_text(encoding='utf-8'))
            for table in inventory['factors']:
                remote='/home/quant/factors/kyzqsh11/'+table['table']+'/2026/20260929.zip'
                a=s.stat(remote); jobs.append((remote,'latest_factors/'+table['table']+'.zip',a.st_size,a.st_mtime))
        elif stage=='minute':
            # Complete 2025 Q4 window for replay; 2026-09-24 for the new stock/volume cross-check.
            add_folder('/home/quant/quantDataProcess/backend/data/bar1m/BJ','minute',
                       lambda n:bool(re.fullmatch(r'\d{8}\.parquet',n)) and ('20251001'<=n[:8]<='20251231' or n=='20260924.parquet'))
    finally: s.close();t.close()
    print(stage,'jobs',len(jobs),'GB',round(sum(j[2] for j in jobs)/1e9,3),flush=True)
    records=[]; start=time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(download,j) for j in jobs]
            for i,f in enumerate(concurrent.futures.as_completed(futures),1):
                records.append(f.result())
                if i%50==0 or i==len(jobs):
                    print(stage,i,'/',len(jobs),'seconds',round(time.monotonic()-start),flush=True)
                    (OUT/(stage+'_manifest.json')).write_text(json.dumps({'stage':stage,'complete':i==len(jobs),
                        'captured_utc':datetime.now(timezone.utc).isoformat(),'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        for t,s in CONNECTIONS:
            try: s.close();t.close()
            except Exception: pass

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prices','factors','capital','samples','minute'])
    main(parser.parse_args().stage)
