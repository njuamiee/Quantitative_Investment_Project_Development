"""Use newly received prices, Alpha snapshots, dividends and stable-base hfq diagnostics."""
import sys,json
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/research_20260930'))
import prepare
from accounts import simulate
OLD_DATA=prepare.DATA;OLD_RUN=prepare.RUN
D=ROOT/'training_data/research_20261008';R=ROOT/'training_runs/teacher9_20261008'

def prepare_new():
    prepare.DATA=D;prepare.RUN=R
    calfile=next((D/'benchmark').glob('000001.SH*'));cal=pd.DatetimeIndex(pd.to_datetime(prepare.read_csv(calfile)['日期'])).sort_values().unique();cal=cal[cal<=pd.Timestamp('2026-10-08')]
    if (D/'bj_prices.parquet').exists():prices=pd.read_parquet(D/'bj_prices.parquet')
    else:prices=prepare.prices('bj',cal)
    if (D/'bj_model_panel.parquet').exists():return prices,pd.read_parquet(D/'bj_model_panel.parquet')
    factors=pd.read_parquet(OLD_DATA/'bj_factor_panel.parquet');parts=[]
    for suffix in ['c01','c02']:
        chunks=[]
        for p in (D/'factors'/suffix/'2026').glob('*.zip'):
            x=prepare.read_zip(p,'BJ');x=x[x.stockid.str.fullmatch(r'BJ\d{6}')].copy();x['date']=pd.to_datetime(p.stem)
            cols=[c for c in x if c not in ['stockid','date','enddate','rdate']]
            ren={c:('c01_Alpha'+c.rsplit('_',1)[-1] if suffix=='c01' else suffix+'_'+c) for c in cols}
            x=x[['date','stockid']+cols].rename(columns=ren)
            x[list(ren.values())]=x[list(ren.values())].apply(pd.to_numeric,errors='coerce');chunks.append(x.set_index(['date','stockid']))
        parts.append(pd.concat(chunks))
    appended=parts[0].join(parts[1],how='outer').reset_index();factors=pd.concat([factors,appended],ignore_index=True)
    assert not factors.duplicated(['date','stockid']).any()
    factors=factors.replace([np.inf,-np.inf],np.nan);factors.to_parquet(D/'bj_factor_panel.parquet',index=False)
    prepare.build_panel('bj',prices,factors)
    return prices,pd.read_parquet(D/'bj_model_panel.parquet')

def actions_audit(prices):
    x=pd.read_parquet(D/'dividend_BJ.parquet');x['stockid']='BJ'+x.symbol.str[:6]
    for c in ['equityRecordDate','exDividendDate','noticeDate']:x[c]=pd.to_datetime(x[c],errors='coerce')
    x['eligible']=(x.progress=='实施分配')&x.exDividendDate.notna()&x.equityRecordDate.notna()&(x.exDividendDate<=pd.Timestamp('2026-10-08'))
    x['dividend']=x.cashDividendPretaxPer10.fillna(0)/10
    x['bonus']=x.bonusTotalPer10.fillna(x.bonusPer10.fillna(0)+x.transferPer10.fillna(0))/10
    x['split']=1+x.bonus
    x['record_before_ex']=x.equityRecordDate<x.exDividendDate
    x['notice_no_later_record']=x.noticeDate.isna()|(x.noticeDate<=x.equityRecordDate)
    x=x[x.eligible&x.record_before_ex&x.notice_no_later_record].copy()
    # Only unique implemented events; never include announced future events.
    assert not x.duplicated(['stockid','exDividendDate']).any(), 'Review duplicate implemented events before crediting'
    cal=pd.DatetimeIndex(sorted(prices.date.unique()));prev=pd.Series(cal,index=cal).shift(1)
    x['record_is_previous_trading_day']=x.exDividendDate.map(prev)==x.equityRecordDate
    inwindow=x[(x.exDividendDate>=cal[0])&(x.exDividendDate<=cal[-1])]
    q=prices.sort_values(['stockid','date']).copy();q['previous_close']=q.groupby('stockid').close.shift(1)
    x=x.merge(q[['date','stockid','previous_close','open','close','qfq_close']],left_on=['exDividendDate','stockid'],right_on=['date','stockid'],how='left',validate='one_to_one')
    # An ex-right theoretical reference is a diagnostic, not the official limit reference.
    x['theoretical_ex_reference']=(x.previous_close-x.dividend)/x.split
    x['raw_open_gap']=x.open/x.previous_close-1;x['economic_open_gap']=x.open/x.theoretical_ex_reference-1
    x.to_csv(R/'dividend_event_audit.csv',index=False,encoding='utf-8-sig')
    summary={'source_rows':len(pd.read_parquet(D/'dividend_BJ.parquet')),'source_stocks':int(pd.read_parquet(D/'dividend_BJ.parquet').symbol.nunique()),
      'implemented_unique_nonfuture_events':len(x),'in_price_history_events':len(inwindow),'record_not_previous_trade_day':int((~inwindow.record_is_previous_trading_day).sum()),
      'events_in_original_test':int(((x.exDividendDate>='2024-10-08')&(x.exDividendDate<='2026-09-29')).sum()),
      'stock_bonus_events_in_original_test':int(((x.exDividendDate>='2024-10-08')&(x.exDividendDate<='2026-09-29')&(x.split>1)).sum()),
      'missing_fields':['cash payment date','bonus-share listing/tradability date','investor-specific withholding/tax timing','historical revision versions'],
      'interpretation':'Useful implemented-event source, not a complete verified settlement ledger; no price-only performance relabeled as total return.'}
    prepare.dump(summary,R/'dividend_summary.json')
    # Ex-date immediate settlement is a clearly labelled sensitivity assumption.
    actions=x[x.record_is_previous_trading_day][['exDividendDate','stockid','dividend','split']].rename(columns={'exDividendDate':'date'})
    return actions

def main():
    prices,panel=prepare_new();oldprices=pd.read_parquet(OLD_DATA/'bj_prices.parquet');oldpred=pd.read_parquet(OLD_RUN/'bj_predictions.parquet')
    overlap=oldprices[['date','stockid','open','close']].merge(prices[['date','stockid','open','close']],on=['date','stockid'],suffixes=('_old','_new'))
    revisions=((overlap.open_old-overlap.open_new).abs()>1e-8)|((overlap.close_old-overlap.close_new).abs()>1e-8)
    actions=actions_audit(prices)
    # Same frozen predictions and price data: isolate corporate-event accounting assumptions.
    d,t,o,m=simulate(oldpred,oldprices,actions=actions)
    m['scenario']='gross cash and bonus shares immediately usable on ex-date; unverified settlement timing; current final event snapshot; price forecasts unchanged'
    d.to_csv(R/'dividend_immediate_assumption_account.csv',index=False);prepare.dump(m,R/'dividend_immediate_assumption_metrics.json')
    # Cash dividends alone, marked as receivables and never reinvested; no claim to fix share-transfer events.
    oldorders=pd.read_csv(OLD_RUN/'bj_pred_selected_orders.csv',parse_dates=['date']);ent=[]
    for event in actions.itertuples(index=False):
        if event.dividend<=0:continue
        prior=oldorders[(oldorders.stockid==event.stockid)&(oldorders.date<event.date)]
        shares=float((prior.shares*np.where(prior.side=='buy',1,-1)).sum())
        if shares>0:ent.append({'date':event.date,'stockid':event.stockid,'record_shares':shares,'gross_receivable':shares*event.dividend})
    ent=pd.DataFrame(ent);ent.to_csv(R/'cash_dividend_receivables.csv',index=False)
    legacy=pd.read_csv(OLD_RUN/'bj_pred_selected_account.csv',parse_dates=['date'])
    bydate=ent.groupby('date').gross_receivable.sum().reindex(legacy.date,fill_value=0).to_numpy();legacy['cash_dividend_receivable']=bydate.cumsum();legacy['equity_plus_cash_receivable_only']=legacy.equity+legacy.cash_dividend_receivable
    legacy.to_csv(R/'cash_receivable_only_account.csv',index=False)
    # Extend scoring by the already trained Q3 model, not by retraining on October outcomes.
    model=lgb.Booster(model_str=(OLD_RUN/'bj/2026Q3/selected_lgbm.txt').read_text('utf-8'));features=model.feature_name()
    added=panel[(panel.date>oldpred.date.max())&(panel.factor_coverage>=.7)].copy()
    added['pred_selected']=model.predict(added[features])
    cols=[c for c in oldpred if c in added];scored=pd.concat([oldpred[cols],added[cols]],ignore_index=True)
    scored.to_parquet(R/'updated_frozen_model_predictions.parquet',index=False)
    daily,_,_,m=simulate(scored,prices,score='pred_selected');daily.to_csv(R/'updated_price_only_account.csv',index=False)
    m.update({'raw_price_history_revised_rows':int(revisions.sum()),'new_signal_dates':[str(v.date()) for v in sorted(added.date.unique())],
      'prices_through':str(prices.date.max().date()),'latest_full_factor_date':str(added.date.max().date()),'latest_signal_label_mature':bool(added.loc[added.date==added.date.max(),'label_raw'].notna().any()),
      'comparison_note':'Updated input vintage and two additional trading sessions; separate from the fixed-window factor-count comparison.',
      'cash_only_receivable_at_old_end':float(legacy.cash_dividend_receivable.iloc[-1])})
    prepare.dump(m,R/'new_data_extension_summary.json')
    added.sort_values(['date','pred_selected'],ascending=[True,False]).groupby('date').head(5).to_csv(R/'new_research_signals.csv',index=False)
    print(json.dumps(m,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
