"""Read-only local audit and authorised SFTP catalog; never output credentials."""
from pathlib import Path
import sys, json, hashlib, io, zipfile, stat
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'training_data/audit_rebuild_20261021'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/'tools'))
from audit_server_inventory import connect, decode

def dump(obj, name):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def local():
    rows=[]
    for dirname in ['training_data/research_20261011','training_data/research_20261008',
                    'training_data/research_20261018/market','training_data/research_20261021/market']:
        for p in (ROOT/dirname).glob('*.parquet'):
            if p.stat().st_size==0:
                print('EMPTY_FILE',str(p.relative_to(ROOT)),flush=True)
                continue
            pf=pq.ParquetFile(p)
            rows.append(dict(path=str(p.relative_to(ROOT)), rows=pf.metadata.num_rows,
                             columns=pf.schema_arrow.names, bytes=p.stat().st_size))
    dump(rows,'local_parquet_inventory.json')
    for row in rows:
        print(row['path'],row['rows'],len(row['columns']),row['columns'][:12],flush=True)
    p=ROOT/'training_data/research_20261021/source/932000.CSI-成分进出记录-20261006.xlsx'
    x=pd.ExcelFile(p)
    print('MEMBERSHIP',x.sheet_names,flush=True)
    for s in x.sheet_names:
        f=pd.read_excel(x,sheet_name=s)
        print(s, f.shape, f.head(3).to_dict('records'),flush=True)
    p=ROOT/'training_data/snapshot_20260920/metadata/provider_data_dictionary.csv'
    if p.read_bytes()[:2]==b'PK':
        x=pd.ExcelFile(io.BytesIO(p.read_bytes()))
        print('DICTIONARY',x.sheet_names,flush=True)
        for s in x.sheet_names:
            f=pd.read_excel(x,sheet_name=s)
            f.to_csv(OUT/('dictionary_'+str(x.sheet_names.index(s))+'.csv'), index=False,encoding='utf-8-sig')
            print(s,f.shape,f.columns.tolist(),f.head(2).to_dict('records'),flush=True)

def remote():
    t,s=connect(); s.get_channel().settimeout(45)
    root='/home/quant/factors/kyzqsh11'
    catalog=[]
    try:
        dirs=sorted(a.filename for a in s.listdir_attr(root) if stat.S_ISDIR(a.st_mode))
        print('REMOTE factor directories',len(dirs),flush=True)
        for k,d in enumerate(dirs):
            row={'table':d}
            try:
                years=[a.filename for a in s.listdir_attr(root+'/'+d) if stat.S_ISDIR(a.st_mode)]
                row['years']=years
                listing=s.listdir_attr(root+'/'+d+'/2026') if '2026' in years else []
                files=sorted([a for a in listing if a.filename.endswith('.zip') and a.filename[:8]<='20261020'],key=lambda a:a.filename)
                row['files_2026']=len(files)
                row['bytes_2026']=sum(a.st_size for a in files)
                if files:
                    a=files[-1]; remote=root+'/'+d+'/2026/'+a.filename
                    raw=s.open(remote,'rb').read()
                    path=OUT/'catalog_samples'/d/a.filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
                        f=pd.read_csv(io.StringIO(decode(z.read(names[0]))),na_values=['<NIL>','NIL','--'])
                    row.update(last=a.filename[:8],remote_mtime=a.st_mtime,sha256=hashlib.sha256(raw).hexdigest(),
                               rows=len(f),columns=f.columns.tolist(),sample=f.head(1).to_dict('records'))
                    if 'stockid' in f:
                        row['bse_rows']=int(f.stockid.astype(str).str.startswith('BJ').sum())
                print(k+1,len(dirs),d,row.get('last'),len(row.get('columns',[])),flush=True)
            except Exception as e:
                row['error_type']=type(e).__name__; print('ERROR',d,type(e).__name__,flush=True)
            catalog.append(row);dump(catalog,'remote_factor_catalog.json')
        for remote in ['/home/quant/factors','/home/quant/quantDataProcess/backend/data/day-data/tickflow']:
            dump([dict(name=a.filename,size=a.st_size,directory=stat.S_ISDIR(a.st_mode)) for a in s.listdir_attr(remote)],
                 'listing_'+remote.split('/')[-1]+'.json')
    finally:s.close();t.close()

if __name__=='__main__':
    (remote if '--remote' in sys.argv else local)()
