"""Meaningful accounting, causality, numerical equivalence and frozen-result checks."""
import sys,json,hashlib,subprocess
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from fast_correlation import daily_spearman
from prepare import dump
R=ROOT/'training_runs/teacher9_20261008';OLD=ROOT/'training_runs/teacher9_20260930'

def main():
    checks={};errors=[]
    def verify(name,condition,details=None):
        checks[name]={'passed':bool(condition),'details':details}
        if not condition:errors.append(name)
    # Frozen September 20 artifacts remain byte-for-byte untouched.
    hashes=json.loads((OLD/'frozen_old_run_hashes.json').read_text('utf-8'))
    verify('september20_frozen_files',all(hashlib.sha256((ROOT/'training_runs/lgbm_hmm_20260920'/p).read_bytes()).hexdigest()==v for p,v in hashes.items()),len(hashes))
    original_sha=subprocess.check_output(['git','rev-parse','3407076:training_runs/teacher9_20260930/bj_predictions.parquet'],cwd=ROOT,text=True).strip()
    verify('frozen_bj_predictions_unchanged',subprocess.check_output(['git','hash-object',str(OLD/'bj_predictions.parquet')],cwd=ROOT,text=True).strip()==original_sha)
    # Exact pairwise Spearman equivalence, including heterogeneous missingness and ties.
    rng=np.random.default_rng(7);values=np.round(rng.normal(size=(90,18)),1);values[rng.random(values.shape)<.15]=np.nan;values[:,0]=1;values[:70,1]=np.nan
    frame=pd.DataFrame(values);a=frame.corr(method='spearman',min_periods=30).to_numpy();b=daily_spearman(frame).to_numpy()
    verify('fast_spearman_exact',np.array_equal(np.isnan(a),np.isnan(b)) and np.nanmax(abs(a-b))<1e-12,float(np.nanmax(abs(a-b))))
    folds=json.loads((R/'folds.json').read_text('utf-8'));selection=json.loads((R/'validation_models.json').read_text('utf-8'))
    verify('quarter_boundaries',all(pd.Timestamp(f['train_label_end'])<pd.Timestamp(f['validation_start']) and pd.Timestamp(f['refit_label_end'])<pd.Timestamp(f['test_start']) for f in folds),len(folds))
    for f in folds:
        options=[m for m in selection if m['quarter']==f['quarter'] and m['model'].startswith('count_')]
        winner=sorted(options,key=lambda m:(-m['validation_ic'],m['features']))[0]['model']
        verify('past_validation_choice_'+f['quarter'],winner==f['selected_count_model'])
        corr=pd.read_csv(R/'bj'/f['quarter']/'mean_daily_spearman.csv',index_col=0)
        for k in [20,40,60,80,100]:
            m=next(x for x in options if x['model']==f'count_{k}');features=m['feature_ids'];a=corr.loc[features,features].to_numpy();maxcorr=float(np.abs(a[np.triu_indices(k,1)]).max())
            verify(f"actual_count_and_dedup_{f['quarter']}_{k}",len(features)==k and maxcorr<.85,maxcorr)
    identity=[];cash=[]
    for p in list(R.glob('*_account.csv'))+list((R/'shsz_alpha').glob('*_account.csv')):
        d=pd.read_csv(p)
        if {'equity','cash','market_value'}<=set(d):identity.append(float((d.equity-d.cash-d.market_value).abs().max()));cash.append(float(d.cash.min()))
    verify('all_account_cash_identities',max(identity)<1e-5,{'accounts':len(identity),'maximum_error':max(identity)})
    verify('all_accounts_no_negative_cash',min(cash)>=-1e-5,min(cash))
    minute=[]
    for p in R.glob('*pct_fills.csv'):
        d=pd.read_csv(p,parse_dates=['date','datetime','entry_date'])
        minute.append({'file':p.name,'fills':len(d),'same_day_sales':int(((d.side=='sell')&(d.date<=d.entry_date)).sum()),'capacity_overrun':float((d.used_capacity-d.capacity).max()),'min_shares':float(d.shares.min())})
    verify('minute_tplus1_capacity_positive_fills',all(x['same_day_sales']==0 and x['capacity_overrun']<=0 and x['min_shares']>0 for x in minute),minute)
    # Lot inventories reconstructed independently from minute fills never go short.
    invmin=[]
    for p in R.glob('*pct_fills.csv'):
        d=pd.read_csv(p);d['signed']=d.shares*np.where(d.side=='buy',1,-1);invmin.append(float(d.groupby('lot_id').signed.cumsum().min()))
    verify('minute_no_negative_lot_inventory',min(invmin)>=0,min(invmin))
    oldpred=pd.read_parquet(OLD/'bj_predictions.parquet',columns=['date','stockid','pred_selected']);newpred=pd.read_parquet(R/'bj_predictions.parquet',columns=['date','stockid','count_40']);z=oldpred.merge(newpred,on=['date','stockid'],validate='one_to_one')
    verify('count40_reproduces_frozen_predictions',np.allclose(z.pred_selected,z.count_40,atol=1e-14,rtol=0),float(abs(z.pred_selected-z.count_40).max()))
    events=pd.read_csv(R/'dividend_event_audit.csv',parse_dates=['exDividendDate','equityRecordDate'])
    verify('events_not_future_or_duplicate',not events.duplicated(['stockid','exDividendDate']).any() and (events.exDividendDate<=pd.Timestamp('2026-10-08')).all() and (events.equityRecordDate<events.exDividendDate).all(),len(events))
    fitted=json.loads((R/'hmm_strict_fits.json').read_text('utf-8'))
    verify('hmm_uses_only_past',all(pd.Timestamp(x['train_end'])<pd.Timestamp(x['first_prediction_date']) for x in fitted),len(fitted))
    output={'passed':not errors,'checks':checks,'failed':errors,'count':len(checks)};dump(output,R/'verification.json');print(json.dumps({'passed':not errors,'checks':len(checks),'failed':errors},ensure_ascii=False))
    if errors:raise AssertionError(errors)

if __name__=='__main__':main()
