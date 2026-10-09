"""Compare verified date sets after the server's year-directory migration."""
from pathlib import Path
import sys,json,hashlib,tarfile,collections
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from audit_server_inventory import coverage
A=ROOT/'training_data/audit_20261008';D=ROOT/'training_data/research_20261008';R=ROOT/'training_runs/teacher9_20261008';R.mkdir(exist_ok=True)
old=json.loads((ROOT/'training_data/audit_20261002/inventory.json').read_text('utf-8'))
new=json.loads((A/'inventory.json').read_text('utf-8'))
listing=json.loads((A/'recursive_minute_listing.json').read_text('utf-8'))
# Preserve the original audit as evidence of the layout detection; final inventory corrects the listing.
new['original_minute_scan_layout']='flat-only initial scan; corrected with recursive_minute_listing.json'
for key,files in listing.items():
    for year in new['minute'][key]:
        expected={d for d in new['reference_calendar'] if d[:4]==year and d>='2021-11-15' and d<=new['cutoff']}
        names=[name for _,name,_,_ in files if name[:4]==year]
        new['minute'][key][year]=coverage(names,expected,new['cutoff'])
(A/'inventory_corrected.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),'utf-8')
def gaps(inv,key):return sorted(d for y,v in inv['minute'][key].items() for d in v['missing'])
result={'snapshot_started_utc':new['started_utc'],'snapshot_ended_utc':new['ended_utc'],
 'comparison_baseline':'2026-10-02','latest_daily':new['daily']['raw']['last_counts'],
 'minute_layout':'bar1m/BJ/YYYY/YYYYMMDD.parquet; cleaned tick remains flat',
 'minute':{},'raw_tick':{},'factors':{},'notes':[
 'This snapshot was captured before the published 18:45 minute and 19:00 factor schedules on October 8; October 8 missing files are pending, not established failed jobs.',
 'Directory accessibility, HTTP health, supplied log excerpts and data coverage are separate evidence; none proves all scheduled tasks always succeed.']}
for key in listing:
    prior=gaps(old,key);now=gaps(new,key)
    result['minute'][key]={'files':sum(v['files'] for v in new['minute'][key].values()),'latest':max(v['last'] for v in new['minute'][key].values() if v['last']),
      'repaired_since_oct2':sorted(set(prior)-set(now)), 'historical_missing':[d for d in now if d<'2026-10-08'],'current_session_pending':[d for d in now if d=='2026-10-08']}
for market,years in new['tick'].items():
    now=sorted(d for v in years.values() for d in v['missing']);prior={d for v in old['tick'][market].values() for d in v['missing']}
    result['raw_tick'][market]={'repaired_since_oct2':sorted(prior-set(now)),'historical_missing':[d for d in now if d<'2026-10-08'],'current_session_pending':[d for d in now if d=='2026-10-08']}
result['factors']={'tables':len(new['factors']),'latest_2026_counts':dict(collections.Counter(x['years']['2026']['last'] for x in new['factors'])),
 'historical_2026_gaps':sum(sum(d<'2026-10-08' for d in x['years']['2026']['missing']) for x in new['factors']),
 'current_session_pending_tables':sum('2026-10-08' in x['years']['2026']['missing'] for x in new['factors'])}
result['tick_duplicate_verified']={}
for m in ['BJ','SZ']:
    path=D/'tick_check'/f'{m}_20260105.tar.gz'
    with tarfile.open(path) as tar:
        members=[x.name for x in tar.getmembers() if x.isfile()]
    result['tick_duplicate_verified'][m]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'members':len(members),'first_members':members[:8]}
result['tick_duplicate_verified']['same_bytes']=result['tick_duplicate_verified']['BJ']['sha256']==result['tick_duplicate_verified']['SZ']['sha256']
(R/'sftp_update_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
