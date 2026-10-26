"""Independent numerical checks of WorkBuddy claims; no old files changed."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'training_runs/audit_rebuild_20261021';OUT.mkdir(parents=True,exist_ok=True)

def save(obj,name):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def metrics(d):
    e=d.equity.to_numpy();r=np.r_[e[0]/1e6-1,e[1:]/e[:-1]-1]
    mdd=-(np.r_[1e6,e]/np.maximum.accumulate(np.r_[1e6,e])-1).min()
    years=(d.date.iloc[-1]-d.date.iloc[0]).days/365.25
    ann=(e[-1]/1e6)**(1/years)-1
    rf=1.015**(1/252)-1
    return dict(total=e[-1]/1e6-1,sharpe=(r.mean()-rf)/r.std(ddof=1)*np.sqrt(252),
                calmar=ann/mdd,mdd=mdd,exposure=d.exposure.mean(),cash_fraction=(d.exposure<.01).mean())

def main():
    report={}
    src=ROOT/'training_data/research_20261021/source/932000.CSI-成分进出记录-20261006.xlsx'
    ev=pd.read_excel(src);ev['日期']=pd.to_datetime(ev['日期'])
    ev['stockid']=ev['代码'].str[-2:]+ev['代码'].str[:6]
    first=ev['日期'].min();state=set(ev.loc[(ev['日期']==first)&ev['纳入/剔除'].eq('纳入'),'stockid']);init=state.copy()
    problems=[]; snaps={first:state.copy()}
    for d,g in ev[ev['日期']>first].groupby('日期'):
        add=set(g.loc[g['纳入/剔除'].eq('纳入'),'stockid']);rm=set(g.loc[g['纳入/剔除'].eq('剔除'),'stockid'])
        if rm-state or add&state:problems.append(str(d))
        state=(state-rm)|add;snaps[d]=state.copy()
    terminal=state.copy()
    for d,g in sorted(ev[ev['日期']>first].groupby('日期'),key=lambda x:x[0],reverse=True):
        state=(state-set(g.loc[g['纳入/剔除'].eq('纳入'),'stockid']))|set(g.loc[g['纳入/剔除'].eq('剔除'),'stockid'])
    report['membership']=dict(rows=len(ev),initial_count=len(init),later_rows=len(ev)-len(init),
         duplicated=int(ev.duplicated(['日期','stockid','纳入/剔除']).sum()),illegal_events=problems,
         reverse_roundtrip=state==init,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
         last_event=str(ev['日期'].max().date()),comment='Internal consistency does not prove external completeness or effective/publication times.')
    mem=pd.read_parquet(ROOT/'training_data/research_20261021/membership/constituent_daily.parquet')
    cal=pd.DatetimeIndex(sorted(mem.date.unique())); pos=pd.Series(np.arange(len(cal)),index=cal)
    price=pd.read_parquet(ROOT/'training_data/research_20261018/market/prices_raw.parquet',columns=['date','stockid','open','close'])
    ext=pd.read_parquet(ROOT/'training_data/research_20261021/market/prices_ext.parquet',columns=['date','stockid','open','close'])
    ext.date=ext.date.astype('datetime64[ns]')
    report['extension_stockid_bug']=dict(six_digit_rows=int(ext.stockid.str.len().eq(6).sum()),
        six_digit_codes=sorted(ext.loc[ext.stockid.str.len().eq(6),'stockid'].unique().tolist()))
    ext.stockid=ext.stockid.str[6:]+ext.stockid.str[:6]
    price=pd.concat([price,ext],ignore_index=True).sort_values('date').drop_duplicates(['date','stockid'],keep='first')
    old=price.merge(mem,on=['date','stockid']).sort_values(['stockid','date'])
    gp=old.groupby('stockid',sort=False)
    old['actual_entry']=gp.date.shift(-1);old['actual_exit']=gp.date.shift(-5)
    old['entry_distance']=old.actual_entry.map(pos)-old.date.map(pos)
    old['exit_distance']=old.actual_exit.map(pos)-old.date.map(pos)
    bad=old[old.actual_exit.notna()&old.exit_distance.ne(5)]
    bad[['date','stockid','actual_entry','actual_exit','entry_distance','exit_distance']].to_csv(OUT/'old_label_calendar_errors.csv',index=False,encoding='utf-8-sig')
    report['labels']=dict(shift_rows=len(old),wrong_exit_calendar_rows=len(bad),
         wrong_entry_calendar_rows=int((old.actual_entry.notna()&old.entry_distance.ne(1)).sum()),
         max_exit_distance=float(bad.exit_distance.max()),examples=bad.nlargest(5,'exit_distance')[['date','stockid','actual_entry','actual_exit','exit_distance']].to_dict('records'))
    panel=pd.read_parquet(ROOT/'training_data/research_20261021/market/panel_member.parquet',columns=['date','stockid','label_5'])
    matched=panel.merge(bad[['date','stockid','actual_exit','exit_distance']],on=['date','stockid'])
    report['labels']['affected_saved_panel']=len(matched)
    folds=json.loads((ROOT/'training_runs/research_20261021/folds.json').read_text(encoding='utf-8'))
    foldrows=[]
    for f in folds:
        a,b=map(pd.Timestamp,f['train']);v,w=map(pd.Timestamp,f['val']);t,u=map(pd.Timestamp,f['test'])
        train=old[old.date.between(a,b)&old.actual_exit.notna()]
        valid=old[old.date.between(v,w)&old.actual_exit.notna()]
        foldrows.append(dict(fold=f['fold'],train_end=b,val_start=v,test_start=t,
            train_labels_reach_val=int((train.actual_exit>=v).sum()),
            train_labels_reach_test=int((train.actual_exit>=t).sum()),
            val_labels_reach_test=int((valid.actual_exit>=t).sum()),
            quarantine_train_days=len(cal[(cal>=a)&(cal<=b)&(cal>='2025-06-04')&(cal<='2025-07-01')]),
            quarantine_val_days=len(cal[(cal>=v)&(cal<=w)&(cal>='2025-06-04')&(cal<='2025-07-01')]),
            last_actual_train_exit=train.actual_exit.max(),last_actual_val_exit=valid.actual_exit.max()))
    fd=pd.DataFrame(foldrows);fd.to_csv(OUT/'old_fold_leakage.csv',index=False,encoding='utf-8-sig')
    report['folds']=dict(total=len(fd),train_val_no_gap_folds=int((fd.train_labels_reach_val>0).sum()),
        train_labels_reach_prediction_folds=int((fd.train_labels_reach_test>0).sum()),
        validation_labels_reach_prediction_folds=int((fd.val_labels_reach_test>0).sum()),
        isolation_used_folds=int(((fd.quarantine_train_days+fd.quarantine_val_days)>0).sum()))
    preds=pd.read_parquet(ROOT/'training_runs/research_20261021/predictions.parquet')
    batches=pd.read_csv(ROOT/'training_runs/research_20261021/portfolio_batches.csv',parse_dates=['signal_date','exit_date'])
    before=[]
    for row in batches.itertuples():
        g=preds[preds.date.eq(row.signal_date)]
        for m in ['icir','ridge','lgbm']:
            unfiltered=g.sort_values(['score_'+m,'stockid'],ascending=[False,True]).head(20)
            filtered=g[g.label_5.notna()].sort_values(['score_'+m,'stockid'],ascending=[False,True]).head(20)
            before.append(dict(date=row.signal_date,model=m,removed_top20=int(unfiltered.label_5.isna().sum()),
                               replaced=len(set(unfiltered.stockid)-set(filtered.stockid))))
    pd.DataFrame(before).to_csv(OUT/'old_future_label_selection.csv',index=False,encoding='utf-8-sig')
    report['portfolio']=dict(last_batch=batches.tail(1).to_dict('records'),
       counts={s:{c:int(g[c].notna().sum()) for c in ['ret_icir','ret_ridge','ret_lgbm','bench_equal_members','bench_index_tr']}
               for s,g in [('V',batches[batches.signal_date<='2026-05-22']),('test',batches[batches.signal_date>='2026-05-25'])]},
       changed_top20_batches=sum(x['removed_top20']>0 for x in before),
       cross_V_test_batches=batches[(batches.signal_date<='2026-05-22')&(batches.exit_date>'2026-05-22')].to_dict('records'))
    account=ROOT/'training_runs/research_20261017/accounts'
    report['accounts']={}
    for name in ['new_main_monthly','new_monthly_equal','new_quarterly_same_risk','new_monthly_diversified']:
        d=pd.read_csv(account/name/'daily.csv',parse_dates=['date'])
        report['accounts'][name]=metrics(d)
        report['accounts'][name]['cash_identity_max_error']=float((d.equity-d.cash-d.market_value).abs().max())
        for q in ['2026Q2','2026Q3']:
            g=d[d.date.dt.to_period('Q').astype(str).eq(q)]
            prev=d[d.date<g.date.iloc[0]].equity.iloc[-1]
            report['accounts'][name][q]=dict(return_=g.equity.iloc[-1]/prev-1,mean_exposure=g.exposure.mean(),days=len(g),cash_days=int((g.exposure<.01).sum()))
    risk=pd.read_csv(ROOT/'training_runs/research_20261017/risk_validation_grid.csv')
    report['cash_choice']=risk[risk.quarter.isin(['2026Q2','2026Q3'])][['quarter','rule','total_return','completed_sale_parts']].to_dict('records')
    bse=pd.read_parquet(ROOT/'training_data/research_20261011/prices.parquet',columns=['date','stockid','close','adj_close','in_pool','observed'])
    opportunity=[]
    for q,a,b in [('2026Q2','2026-04-01','2026-06-30'),('2026Q3','2026-07-01','2026-09-30')]:
        start=bse[bse.date.eq(bse.loc[bse.date<a,'date'].max())&bse.observed]
        start=start[start.in_pool].set_index('stockid')
        end=bse[(bse.date<=b)&(bse.date>=a)&bse.observed].sort_values('date').groupby('stockid').tail(1).set_index('stockid')
        ret=end.adj_close/start.adj_close-1;ret=ret.dropna()
        opportunity.append(dict(quarter=q,eligible_at_start=len(start),observed_return_stocks=len(ret),positive_stocks=int((ret>0).sum()),
                                positive_fraction=float((ret>0).mean()),median=float(ret.median()),best=float(ret.max()),
                                note='ex-post opportunity diagnostic, not an investable hindsight selection'))
    report['rising_stocks']=opportunity
    save(report,'old_audit.json');print(json.dumps(report,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__':main()
