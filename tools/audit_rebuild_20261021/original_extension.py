"""Extend the original 1017 BJ strategy without replacing its historical outputs.
Only Q4 selection/refit is new. Earlier predictions and account inputs are frozen.
"""
from pathlib import Path
import sys,os,json,hashlib,types,shutil,importlib.util,gc
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'training_data/audit_rebuild_20261021'
DATA=BASE/'original_inputs';RUN=ROOT/'training_runs/audit_rebuild_20261021/original'
DATA.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'tools/research_20261017'),str(ROOT/'tools/research_20261011'),str(ROOT/'tools')]
from audit_server_inventory import connect

def dump(x,p):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def link(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():os.link(src,dest)

def prepare():
    manifest=json.loads((BASE/'price_download_manifest.json').read_text(encoding='utf-8'))
    for r in manifest:
        if r.get('stockid','').startswith('BJ') and r['kind'] in ['raw','hfq','qfq'] and 'error_type' not in r:
            link(BASE/r['local'],DATA/'bj'/r['kind']/r['name'])
    old=ROOT/'training_data/research_20261011'
    for p in (old/'factors').rglob('*.zip'):
        link(p,DATA/'factors'/p.relative_to(old/'factors'))
    t,s=connect();s.get_channel().settimeout(60);records=[]
    try:
        for suffix in ['c01','c02']:
            remote=f'/home/quant/factors/kyzqsh11/ts_stockfactordata_{suffix}/2026'
            for a in s.listdir_attr(remote):
                if not ('20261010'<=a.filename[:8]<='20261019') or not a.filename.endswith('.zip'):continue
                dest=DATA/'factors'/suffix/'2026'/a.filename;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():s.get(remote+'/'+a.filename,str(dest))
                records.append(dict(remote=remote+'/'+a.filename,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    finally:s.close();t.close()
    dump(records,RUN/'factor_extension_sources.json')
    path=ROOT/'tools/research_20261011/prepare_data.py';source=path.read_text(encoding='utf-8-sig')
    # Same recipe with a shorter warm-up window for extension-only rows and new end date.
    source=source.replace("cal>='2021-11-15'","cal>='2025-08-01'").replace("cal<='2026-10-09'","cal<='2026-10-19'")
    needle="    cal=cal[(cal>='2025-08-01')&(cal<='2026-10-19')]"
    assert needle in source
    source=source.replace(needle,"    extra=pd.read_csv(ROOT/'training_data/research_20261021/market/932000_official_daily_ext.csv',dtype={'tradeDate':str})\n    cal=cal.union(pd.DatetimeIndex(pd.to_datetime(extra.tradeDate,format='%Y%m%d'))).sort_values()\n"+needle)
    module=types.ModuleType('original_prepare_extension');module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    module.DATA=DATA;module.RUN=RUN;module.main()
    del module;gc.collect()
    oldpanel=pd.read_parquet(old/'model_panel.parquet');newpanel=pd.read_parquet(DATA/'model_panel.parquet')
    last=oldpanel.date.max()
    assert last==pd.Timestamp('2026-10-09')
    panel=pd.concat([oldpanel,newpanel[newpanel.date>last]],ignore_index=True)
    assert not panel.duplicated(['date','stockid']).any()
    panel.to_parquet(DATA/'model_panel.parquet',index=False)
    oldpr=pd.read_parquet(old/'prices.parquet');newpr=pd.read_parquet(DATA/'prices.parquet')
    pr=pd.concat([oldpr,newpr[newpr.date>oldpr.date.max()]],ignore_index=True)
    pr.to_parquet(DATA/'prices.parquet',index=False)
    # Separate minimal descriptors for the unchanged historical risk engine.
    pr[['date','stockid','close','mom20','vol20','observed','in_pool']].to_parquet(DATA/'bj_prices.parquet',index=False)
    a=pd.read_csv(old/'actions.csv',parse_dates=['date']);b=pd.read_csv(DATA/'actions.csv',parse_dates=['date'])
    pd.concat([a[a.date<=oldpr.date.max()],b[b.date>oldpr.date.max()]],ignore_index=True).to_csv(DATA/'actions.csv',index=False)
    dump(dict(old_panel_end=last,new_panel_end=panel.date.max(),new_rows=int((panel.date>last).sum()),
              old_features=[c for c in panel if c.startswith('x_')],old_preparation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              changes='Only date extension; old historical price/panel rows and predictions preserved exactly.'),RUN/'extension_manifest.json')
    print('original extension prepared',panel.shape,panel.date.max(),flush=True)

def train():
    import models
    panel=pd.read_parquet(DATA/'model_panel.parquet');dates=pd.DatetimeIndex(sorted(panel.date.unique()));features=[c for c in panel if c.startswith('x_')]
    models.OUTER_END=pd.Timestamp('2026-10-19');out=RUN/'models';out.mkdir(exist_ok=True)
    fold,_=models.evaluate_quarter(panel,dates,pd.Period('2026Q4'),out,features)
    pred=models.deploy(panel,dates,fold,out,features)
    old=pd.read_parquet(ROOT/'training_runs/research_20261017/models/daily_predictions.parquet')
    pd.concat([old,pred],ignore_index=True).to_parquet(RUN/'daily_predictions.parquet',index=False)
    print('Q4 original model',fold['selected_name'],flush=True)

def evaluate():
    import risk_metadata
    from execution import CohortMarket,Rule,RULES,CASH,period_metrics
    risk_metadata.PRICE_FILE=DATA/'bj_prices.parquet';risk_metadata.OUTPUT=RUN
    md,meta=risk_metadata.build()
    oldmd=pd.read_parquet(ROOT/'training_runs/research_20261017/risk_metadata.parquet')
    md=pd.concat([oldmd,md[md.date>oldmd.date.max()]],ignore_index=True)
    md.to_parquet(RUN/'risk_metadata.parquet',index=False)
    prices=pd.read_parquet(DATA/'prices.parquet');actions=pd.read_csv(DATA/'actions.csv')
    market=CohortMarket(prices,actions,md)
    predictions=pd.read_parquet(RUN/'daily_predictions.parquet')
    oldschedule=json.loads((ROOT/'training_runs/research_20261017/risk_schedule.json').read_text(encoding='utf-8'))
    schedule={k:Rule(**v) for k,v in oldschedule.items()}
    fold=json.loads((RUN/'models/2026Q4/selection.json').read_text(encoding='utf-8'))
    vp=pd.read_parquet(RUN/'models/2026Q4/valid_predictions.parquet')[['date','stockid','score']]
    rows=[]
    for rule in RULES:
        r=market.run(vp,rule,start=fold['valid_start'],end=fold['valid_end'])
        m=r['metrics'];objective=min(m['sharpe_rf15'],m['calmar']/2) if m['calmar'] is not None else -1e9
        rows.append(dict(rule=rule.name,objective=objective,**{k:v for k,v in m.items() if k not in ['rule','schedule']}))
    viable=[r for r in rows if r['total_return']>0 and r['completed_sale_parts']>=5]
    best=max(viable,key=lambda r:(r['objective'],r['annual_return'],r['rule'])) if viable else None
    schedule['2026Q4']=next(r for r in RULES if r.name==best['rule']) if best else CASH
    pd.DataFrame(rows).to_csv(RUN/'Q4_risk_validation.csv',index=False,encoding='utf-8-sig')
    dump({k:v.__dict__ for k,v in schedule.items()},RUN/'risk_schedule.json')
    results=[]
    for name,col,selected in [('original_main','score_monthly',True),('original_monthly_no_cash','score_monthly',False),('original_quarterly_same_risk','score_quarterly',True),('original_simple_no_cash','score_simple',False)]:
        p=predictions[['date','stockid',col]].rename(columns={col:'score'})
        r=market.run(p,Rule(),start='2024-10-08',end='2026-10-19',schedule=schedule if selected else None,save_details=True)
        for key in ['daily','orders','trades','holdings']:r[key].to_csv(RUN/f'{name}_{key}.csv',index=False,encoding='utf-8-sig')
        for seg,a,b in [('test','2026-05-22','2026-10-19'),('V','2025-07-02','2026-05-21'),('full','2024-10-08','2026-10-19')]:
            metric=period_metrics(r['daily'],a,b)
            sub=r['daily'][r['daily'].date.between(a,b)]
            if sub.equity.pct_change().dropna().std()<1e-12:
                metric['sharpe_rf15']=None;metric['sharpe_rf0']=None;metric['calmar']=None
            results.append(dict(strategy=name,period=seg,**metric))
        if name=='original_main':
            old=pd.read_csv(ROOT/'training_runs/research_20261017/accounts/new_main_monthly/daily.csv',parse_dates=['date'])
            new=r['daily'][r['daily'].date<=old.date.max()]
            err=float(np.max(np.abs(new.equity.to_numpy()-old.equity.to_numpy())))
            assert err<.01,('historical reproduction mismatch',err)
            dump(dict(max_historical_equity_error=err,q4_rule=schedule['2026Q4'].name),RUN/'reproduction_check.json')
    pd.DataFrame(results).to_csv(RUN/'summary.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(results)[['strategy','period','total_return','sharpe_rf15','max_drawdown','mean_exposure']].to_string(index=False),flush=True)

if __name__=='__main__':
    stage=sys.argv[1] if len(sys.argv)>1 else 'all'
    if stage in ['all','prepare']:prepare()
    if stage in ['all','train']:train()
    if stage in ['all','evaluate']:evaluate()
