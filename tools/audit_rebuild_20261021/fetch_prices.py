from pathlib import Path
import sys,re,threading,hashlib,json,io,time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'training_data/audit_rebuild_20261021'
sys.path.insert(0,str(ROOT/'tools'))
from audit_server_inventory import connect,decode
LOCAL=threading.local();CONNS=[];LOCK=threading.Lock()
BASE='/home/quant/quantDataProcess/backend/data'
RENAME={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume','成交额':'amount'}

def sf():
    if not hasattr(LOCAL,'s'):
        t,s=connect();s.get_channel().settimeout(90);LOCAL.s=s
        with LOCK:CONNS.append((t,s))
    return LOCAL.s

def fetch(j):
    dest=OUT/j['local'];dest.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            if not dest.exists() or dest.stat().st_size!=j['bytes']:
                part=dest.with_suffix('.part');sf().get(j['remote'],str(part))
                if part.stat().st_size!=j['bytes']:raise ValueError('size_changed')
                part.replace(dest)
            return dict(**j,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        except Exception as e:
            if attempt==2:return dict(**j,error_type=type(e).__name__)
            time.sleep(.5)

def main():
    mem=pd.read_parquet(ROOT/'training_data/research_20261021/membership/constituent_daily.parquet',columns=['stockid'])
    ids=set(mem.stockid);jobs=[]
    t,s=connect()
    try:
        for kind in ['raw','hfq','qfq']:
            path=BASE+'/day-data/tickflow/'+kind
            for a in s.listdir_attr(path):
                m=re.match(r'^(\d{6})\.(BJ|SH|SZ)-.*\.csv$',a.filename)
                if not m:continue
                code,ex=m.groups();sid=ex+code
                isbj=ex=='BJ' and code.startswith(('4','8','92')) and not code.startswith('899')
                if (sid in ids and kind!='qfq') or isbj:
                    jobs.append(dict(remote=path+'/'+a.filename,local=f'prices/{kind}/{sid}.csv',
                                     bytes=a.st_size,mtime=a.st_mtime,stockid=sid,kind=kind,name=a.filename))
        path=BASE+'/day-data/benchmark/raw'
        for a in s.listdir_attr(path):
            if a.filename.startswith(('000001.SH','899050.BJ')):
                jobs.append(dict(remote=path+'/'+a.filename,local='original_inputs/benchmark/'+a.filename,bytes=a.st_size,mtime=a.st_mtime,kind='benchmark'))
        path=BASE+'/dividend/BJ.parquet';a=s.stat(path)
        jobs.append(dict(remote=path,local='original_inputs/dividend_BJ.parquet',bytes=a.st_size,mtime=a.st_mtime,kind='dividend'))
    finally:s.close();t.close()
    print('price files',len(jobs),round(sum(j['bytes'] for j in jobs)/1e9,3),'GB',flush=True)
    records=[]
    try:
        with ThreadPoolExecutor(6) as ex:
            for i,r in enumerate(ex.map(fetch,jobs),1):
                records.append(r)
                if i%200==0:print(i,len(jobs),flush=True)
    finally:
        for t,s in CONNS:
            try:s.close();t.close()
            except Exception:pass
    (OUT/'price_download_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    for kind in ['raw','hfq']:
        frames=[]
        for j in records:
            if j['kind']!=kind or 'error_type' in j:continue
            f=pd.read_csv(io.StringIO(decode((OUT/j['local']).read_bytes()))).rename(columns=RENAME)
            f['date']=pd.to_datetime(f.date).astype('datetime64[ns]');f['stockid']=j['stockid']
            if f.date.duplicated().any():raise ValueError('duplicate price dates')
            f=f[f.date.between('2021-11-15','2026-10-19')]
            frames.append(f[['date','stockid','open','high','low','close','volume','amount']])
        f=pd.concat(frames,ignore_index=True)
        f.to_parquet(OUT/f'prices_{kind}.parquet',index=False,compression='zstd')
        print('parsed',kind,len(f),f.date.min(),f.date.max(),flush=True)

if __name__=='__main__':main()
