"""Complete historical date gaps, archive contents, A-share file updates and local revisions."""
import json,re,zipfile,concurrent.futures
from pathlib import Path
import pandas as pd
from audit_sync import connect,listing,ROOT,DATA,AUD,BASE,write_csv
from common import RUN,dump,sha

def inspect_path(path):
    t,s=connect()
    try:return path,listing(s,path)
    finally:s.close();t.close()

def main():
    inv=json.loads((AUD/'inventory.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'training_data/audit_20261008/inventory.json').read_text(encoding='utf-8'))
    expected=set(old['reference_calendar'])|{'2026-10-08','2026-10-09'};missing=[];summary=[]
    for path,entries in inv['date_files'].items():
        isfactor='ts_stockfactordata_' in path
        files=[x for x in entries if re.match(r'^\d{8}(\.zip|\.parquet|_Tick.*\.tar\.gz)$',x.get('name',''))]
        dates=sorted({x['name'][:4]+'-'+x['name'][4:6]+'-'+x['name'][6:8] for x in files})
        gaps=[]
        if dates:
            start=max(dates[0],'2021-11-15') if '/BJ' in path else dates[0]
            gaps=sorted(d for d in expected-set(dates) if start<=d<='2026-10-09')
            for d in gaps:
                ext='.zip' if isfactor else '_Tick1.tar.gz' if 'tick-All' in path else '.parquet'
                sample=files[-1]['path'];yearfolder=bool(re.search(r'/20\d{2}/',sample))
                missing.append({'category':'factor_date' if isfactor else 'market_date','date':d,
                                'expected_path':path+('/'+d[:4] if yearfolder else '')+'/'+d.replace('-','')+ext,
                                'basis':'absent from listed dates after first available date; expected exchange calendar'})
        else:missing.append({'category':'directory','date':'','expected_path':path,'basis':'no data files in accessible listing'})
        summary.append({'path':path,'actual_data_files':len(files),'first':dates[0] if dates else '',
                        'last':dates[-1] if dates else '','missing_calendar_dates':len(gaps)})
    write_csv(missing,RUN/'missing_files.csv');write_csv(summary,RUN/'server_coverage.csv')
    tasks=[x['path'] for x in inv['directories']['/home/quant/day-data'] if x.get('directory')]
    extra={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for path,rows in pool.map(inspect_path,tasks):extra[path]=rows
    archive_path=BASE+'/bar1m/minute-all.zip';t,s=connect();archive={}
    try:
        with s.open(archive_path,'rb') as f:
            with zipfile.ZipFile(f) as z:
                members=[{'name':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size,'timestamp':x.date_time} for x in z.infolist()]
                archive={'path':archive_path,'members':members,'inspection':'ZIP central directory only, file bodies not validated'}
    except Exception as e:archive={'path':archive_path,'error':type(e).__name__+': '+str(e)}
    finally:s.close();t.close()
    dump(extra,AUD/'ashare_directory_inventory.json');dump(archive,AUD/'minute_archive_inventory.json')
    revisions=[]
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    for r in manifest['records']:
        local=Path(r['local']);previous=ROOT/'training_data/research_20261008'/local
        p=DATA/local
        if not p.exists():continue
        priorhash=sha(previous) if previous.exists() else None
        revisions.append({'local':r['local'],'sha256':r['sha256'],'previous_sha256':priorhash,
                          'change':'changed' if priorhash and priorhash!=r['sha256'] else 'same' if priorhash else 'new_to_snapshot','bytes':r['bytes']})
    write_csv(revisions,RUN/'file_revisions.csv')
    fieldgaps=[
      ('all markets','historical_security_master','Listing/delisting history, name/ST changes, code migration mapping and historical universe membership','Survivorship and code continuity'),
      ('all markets','daily_trade_status','Exact suspension and exchange limit-up/down/reference-price fields with release timestamp','Replace 30 percent execution proxy'),
      ('BJ','corporate_action_settlement','Cash payment date, dividend tax by holding duration, bonus-share tradable date','Ex-date immediate gross scenario is not verified real settlement'),
      ('factor supplier','historical_factor_vintages','Per-field computation/release timestamp, revisions and as-of archives','Current-vintage factors cannot establish point-in-time correctness'),
      ('factor supplier','alpha101_versioned_formulas','Exact implemented Alpha101 formulas, rank universe, adjustment mode and exception conventions for the supplied version','Existing extraction/API guides do not verify every supplied formula'),
      ('financial/news','text_document_archive','Issuer, publication/ingestion/version timestamps, source text and licensed usage','Jev classification/event/risk live evaluation'),
      ('financial/news','text_annotation_labels','Dated human labels and mature outcomes, chronological train/validation/test','Jev calibration and incremental benefit'),
      ('HMM','historical_macro_vintages','Macro releases, financing data and spread/liquidity fields','Enrich emissions without release-time leakage'),
      ('execution','orderbook_and_order_logs','Depth/queue snapshots, fills, impact/slippage and account fee schedule','Minute bar stress is not queue-level execution')]
    write_csv([dict(scope=a,required_file_or_field=b,content=c,affects=d) for a,b,c,d in fieldgaps],RUN/'missing_fields_and_documents.csv')
    dump({'date_gaps':len(missing),'factor_tables':sum('ts_stockfactordata_' in x['path'] for x in summary),
          'downloaded_manifest_files':len(manifest['records']),'failed_downloads':sum(str(x['status']).startswith('failed') for x in manifest['records']),
          'revision_counts':pd.Series([x['change'] for x in revisions]).value_counts().to_dict(),
          'ashare_directory_counts':{k:len(v) for k,v in extra.items()},'minute_archive_members':len(archive.get('members',[])),
          'minute_archive_example':archive.get('members',[])[:5]},RUN/'audit_summary.json')
    print('supplement audit complete',len(missing),'date/directory gaps; archive',len(archive.get('members',[])),flush=True)

if __name__=='__main__':main()
