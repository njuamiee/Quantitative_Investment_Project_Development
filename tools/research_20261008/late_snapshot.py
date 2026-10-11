"""Use the newly arrived October 8 factors for explicitly frozen-model monitoring."""
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from audit_server_inventory import connect
from prepare import read_zip,dump,PRICE_FEATURES
D=ROOT/'training_data/research_20261008';R=ROOT/'training_runs/teacher9_20261008'

def main():
    latest=json.loads((ROOT/'training_data/audit_20261008/final_check.json').read_text('utf-8'));parts=[];manifest=[];t,s=connect();tick_presence={}
    try:
        for market in ['BJ','SH','SZ']:
            path='/home/quant/tick-All/'+market+'/2026/20261008_Tick1.tar.gz'
            try:a=s.stat(path);tick_presence[market]={'present':True,'bytes':a.st_size}
            except FileNotFoundError:tick_presence[market]={'present':False}
        for suffix in ['c01','c02']:
            remote='/home/quant/factors/kyzqsh11/ts_stockfactordata_'+suffix+'/2026/20261008.zip';local=D/'late_factors'/suffix/'20261008.zip';local.parent.mkdir(parents=True,exist_ok=True);s.get(remote,str(local))
            manifest.append({'path':remote,'sha256':hashlib.sha256(local.read_bytes()).hexdigest(),'bytes':local.stat().st_size})
            x=read_zip(local,'BJ');x=x[x.stockid.str.fullmatch(r'BJ\d{6}')].copy();cols=[c for c in x if c not in ['stockid','enddate','rdate']]
            ren={c:('c01_Alpha'+c.rsplit('_',1)[-1] if suffix=='c01' else suffix+'_'+c) for c in cols};x=x[['stockid']+cols].rename(columns=ren)
            x[list(ren.values())]=x[list(ren.values())].apply(pd.to_numeric,errors='coerce');parts.append(x.set_index('stockid'))
    finally:s.close();t.close()
    factor=parts[0].join(parts[1],how='outer').reset_index().replace([np.inf,-np.inf],np.nan)
    prices=pd.read_parquet(D/'bj_prices.parquet');pool=prices[(prices.date=='2026-10-08')&prices.in_pool].merge(factor,on='stockid',how='left',validate='one_to_one')
    features=PRICE_FEATURES+[c for c in factor if c!='stockid'];rank=pool[features].rank(pct=True).sub(.5);rank.columns=['x_'+c for c in features];pool=pd.concat([pool,rank],axis=1)
    model=lgb.Booster(model_str=(R/'bj/2026Q3/count_100.txt').read_text('utf-8'))
    pool['score_frozen_q3_monitoring']=model.predict(pool[model.feature_name()]);pool['actual_capture_utc']=datetime.now(timezone.utc).isoformat();pool['model_version']='2026Q3_count100_frozen_monitoring_only';pool['not_retrained_for_2026Q4']=True;pool['future_label_status']='not_mature';pool['research_only']=True
    keep=['date','stockid','score_frozen_q3_monitoring','actual_capture_utc','model_version','not_retrained_for_2026Q4','future_label_status','research_only']
    pool[keep].sort_values('score_frozen_q3_monitoring',ascending=False).to_csv(R/'oct8_frozen_model_monitoring.csv',index=False,encoding='utf-8-sig')
    latest['oct8_raw_tick_presence']=tick_presence;latest['downloaded_factor_samples']=manifest;latest['monitoring_rows']=len(pool);latest['monitoring_factor_coverage']=float(pool[[c for c in factor if c!='stockid']].notna().mean().mean());latest['monitoring_status']='Saved current scores from explicitly frozen 2026Q3 model, not a newly trained Q4 strategy and not a realized outcome.'
    dump(latest,R/'late_server_check.json')
    server=json.loads((R/'sftp_update_summary.json').read_text('utf-8'));server['late_check']=latest;dump(server,R/'sftp_update_summary.json')
    summary=json.loads((R/'final_summary.json').read_text('utf-8'));summary['server']=server;dump(summary,R/'final_summary.json')
    print('late_snapshot',len(pool),latest['monitoring_factor_coverage'],tick_presence,flush=True)

if __name__=='__main__':main()
