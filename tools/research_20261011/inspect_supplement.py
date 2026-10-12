"""Read-only additional data directories and representative content dates."""
import json,io,hashlib
from pathlib import Path
import pandas as pd
from audit_sync import connect,listing,ROOT,DATA,AUD,write_csv
from common import RUN,dump

def main():
    t,s=connect();s.get_channel().settimeout(60);extra={};checks=[]
    try:
        for path in ['/home/quant/factors/factor_lib','/home/quant/data-cvd','/home/quant/options','/home/quant/meeting']:
            extra[path]=listing(s,path)
        dirs=json.loads((AUD/'ashare_directory_inventory.json').read_text(encoding='utf-8'))
        # No full-market freshness claim from samples; every directory/file metadata is retained separately.
        for folder,files in dirs.items():
            samples=[a for code in ['000001','600000','920002'] for a in files if a.get('name','').startswith(code) and a['name'].endswith('.csv')]
            for a in samples:
                with s.open(a['path'],'rb') as f:raw=f.read()
                decoded=None
                for enc in ['utf-8-sig','gb18030']:
                    try:decoded=raw.decode(enc);break
                    except UnicodeDecodeError:pass
                frame=pd.read_csv(io.StringIO(decoded));datecol=next(c for c in frame if c in ['日期','date','datetime'])
                checks.append({'remote':a['path'],'rows':len(frame),'first_date':str(frame[datecol].min()),'last_date':str(frame[datecol].max()),'sha256':hashlib.sha256(raw).hexdigest(),'scope':'representative sample only'})
        for name in ['Alpha001.parquet','0_1.parquet']:
            a=next((x for x in extra['/home/quant/factors/factor_lib'] if x.get('name')==name),None)
            if not a:continue
            path=DATA/'audit_samples'/name;path.parent.mkdir(exist_ok=True);s.get(a['path'],str(path));f=pd.read_parquet(path)
            checks.append({'remote':a['path'],'rows':len(f),'first_date':str(f.date.min()) if 'date' in f else '',
                           'last_date':str(f.date.max()) if 'date' in f else '', 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'sampled factor library file'})
    finally:s.close();t.close()
    dump(extra,AUD/'supplementary_directories.json');write_csv(checks,RUN/'remote_content_date_samples.csv')
    print(pd.DataFrame(checks)[['remote','last_date']].to_string(index=False),flush=True)

if __name__=='__main__':main()
