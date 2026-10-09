"""Read-only October 8 snapshot. Supports migrated year-based minute directories."""
import sys,json,re,stat,hashlib,concurrent.futures
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from audit_server_inventory import connect,coverage
import fetch_data as fetch
OUT=ROOT/'training_data/research_20261008';OUT.mkdir(parents=True,exist_ok=True)
fetch.OUT=OUT
AUD=ROOT/'training_data/audit_20261008'

def main():
    inv=json.loads((AUD/'inventory.json').read_text('utf-8'))
    old=json.loads((ROOT/'training_data/audit_20261002/inventory.json').read_text('utf-8'))
    calendar=set(inv['reference_calendar']['dates']) if isinstance(inv['reference_calendar'],dict) and 'dates' in inv['reference_calendar'] else None
    t,s=connect();jobs=[];fixed={};metadata={}
    def add(remote,local):
        a=s.stat(remote);jobs.append((remote,local,a.st_size,a.st_mtime))
    try:
        # Actual reference dates saved by the audit are taken from the benchmark file.
        base='/home/quant/quantDataProcess/backend/data'
        for adj in ['raw','qfq','hfq']:
            d=base+'/day-data/tickflow/'+adj
            for a in s.listdir_attr(d):
                if a.filename.endswith('.csv'):add(d+'/'+a.filename,'bj/'+adj+'/'+a.filename)
        for a in s.listdir_attr(base+'/day-data/benchmark/raw'):
            if a.filename.endswith('.csv'):add(base+'/day-data/benchmark/raw/'+a.filename,'benchmark/'+a.filename)
        for key in ['bar1m/BJ','tick/BJ']:
            d=base+'/'+key;files=[]
            for a in s.listdir_attr(d):
                if stat.S_ISDIR(a.st_mode) and re.fullmatch(r'20\d{2}',a.filename):
                    files.extend((a.filename+'/'+b.filename,b.filename,b.st_size,b.st_mtime) for b in s.listdir_attr(d+'/'+a.filename) if re.fullmatch(r'\d{8}\.parquet',b.filename))
                elif re.fullmatch(r'\d{8}\.parquet',a.filename):files.append((a.filename,a.filename,a.st_size,a.st_mtime))
            metadata[key]=files
            # Keep original expected dates, replacing only the incorrect flat-folder listing.
            fixed[key]={}
            for y,original in inv['minute'][key].items():
                available={f'{name[:4]}-{name[4:6]}-{name[6:8]}' for _,name,_,_ in files if name.startswith(y)}
                expected=set(original['missing']) | (set() if original['files']==0 else set())
                # Reference calendar reconstructed below, not inferred from potentially corrupt filenames.
                fixed[key][y]={'dates':sorted(available),'files':len(available)}
            if key=='bar1m/BJ':
                for remote,name,size,mtime in files:
                    if '20260925'<=name[:8]<='20261008':jobs.append((d+'/'+remote,'minute/'+name,size,mtime))
        for suffix in ['c01','c02','071','601']:
            d='/home/quant/factors/kyzqsh11/ts_stockfactordata_'+suffix+'/2026'
            for a in s.listdir_attr(d):
                if re.fullmatch(r'\d{8}\.zip',a.filename) and '20260930'<=a.filename[:8]<='20261008':add(d+'/'+a.filename,'factors/'+suffix+'/2026/'+a.filename)
        add(base+'/dividend/BJ.parquet','dividend_BJ.parquet')
        # Verify formerly suspicious equal archives by actual content, not supplied MD5 alone.
        for market in ['BJ','SZ']:add('/home/quant/tick-All/'+market+'/2026/20260105_Tick1.tar.gz','tick_check/'+market+'_20260105.tar.gz')
    finally:s.close();t.close()
    (AUD/'recursive_minute_listing.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),'utf-8')
    print('snapshot_jobs',len(jobs),'MB',round(sum(j[2] for j in jobs)/1e6),flush=True)
    records=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i,record in enumerate(pool.map(fetch.download,jobs),1):
                records.append(record)
                if i%100==0 or i==len(jobs):print('snapshot',i,'/',len(jobs),flush=True)
    finally:
        for t,s in fetch.CONNECTIONS:
            try:s.close();t.close()
            except Exception:pass
    (OUT/'manifest.json').write_text(json.dumps({'captured_utc':datetime.now(timezone.utc).isoformat(),'records':records},ensure_ascii=False,indent=2),'utf-8')

if __name__=='__main__':main()
