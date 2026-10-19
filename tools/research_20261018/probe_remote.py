"""Read-only authorized data-source probe; never log authentication values."""
from pathlib import Path
import sys,json,stat,zipfile,io
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from audit_server_inventory import connect
OUT=ROOT/'training_data/research_20261018'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    transport,sftp=connect()
    sftp.get_channel().settimeout(45)
    results={}
    base='/home/quant/quantDataProcess/backend/data'
    try:
        for path in ['/home/quant/day-data',base+'/day-data',base+'/day-data/tickflow/raw',base+'/day-data/tickflow/hfq',
                     '/home/quant/factors','/home/quant/factors/kyzqsh11']:
            try:
                entries=sftp.listdir_attr(path)
                rows=[{'name':a.filename,'size':a.st_size,'directory':stat.S_ISDIR(a.st_mode)} for a in entries]
                results[path]={'count':len(rows),'entries':rows}
                print('LIST',path,'count',len(rows),'dirs',[r['name'] for r in rows if r['directory']],flush=True)
            except Exception as e:results[path]={'error_type':type(e).__name__}
        for suffix in ['c01','071','601']:
            remote=f'/home/quant/factors/kyzqsh11/ts_stockfactordata_{suffix}/2025/20250603.zip'
            try:
                data=sftp.open(remote,'rb').read()
                local=OUT/f'probe_{suffix}_20250603.zip';local.write_bytes(data)
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    name=z.namelist()[0];raw=z.read(name)
                    try:frame=pd.read_csv(io.BytesIO(raw),encoding='utf-8')
                    except UnicodeDecodeError:frame=pd.read_csv(io.BytesIO(raw),encoding='gb18030')
                results[remote]={'rows':len(frame),'columns':list(frame.columns),'inner_name':name,'bytes':len(data)}
                print('SAMPLE',suffix,results[remote],flush=True)
            except Exception as e:
                results[remote]={'error_type':type(e).__name__};print('SAMPLE_ERROR',suffix,type(e).__name__,flush=True)
    finally:sftp.close();transport.close()
    (OUT/'remote_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
