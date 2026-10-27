"""Independent output/timestamp audit of the frozen set of completed jobs."""
import json,warnings
import numpy as np
import pandas as pd
from config import RUN,REPORT,HORIZONS,dump,digest,key
from monthly_selection import completed
from features import load_meta

def main():
    records=completed();folds=0;preds=0;max_cash=0.;fits=0;accounts=0
    from runner import configurations
    expected_jobs={key(c) for c in configurations()};actual={r['job'] for r in records}
    assert len(actual)==len(records) and actual<=expected_jobs
    result_hashes=0
    for r in records:
        assert r['job']==key(r['config'])
        folder=RUN/'models'/r['job'];logs=json.loads((folder/'fit_audit.json').read_text());cards=json.loads((folder/'factor_audit.json').read_text())
        assert len(logs)==r['actual_fits'];fits+=len(logs);accounts+=r['completed_accounts'];max_cash=max(max_cash,r['cash_flow_max_error'])
        for log in logs:
            assert log['last_mature_outcome_index']<log['fit_index']
            assert log['training_end_index']<log['fit_index']
            assert log['factor_cutoff_index']<=log['fit_index']
            assert log['actual_factors']<=r['config']['count']
            folds+=1
        for card in cards.values():assert card['last_outcome_index']<card['screen_cutoff_index']
        result=np.load(folder/'risk_results.npz');v=result['metrics'];o=result['objectives']
        assert digest(folder/'risk_results.npz')==r['results_sha256'];result_hashes+=1
        assert len(v)==r['completed_accounts'] and v.shape[1:]==(2,9)
        assert np.isfinite(v[:,:,:3]).all() and (v[:,:,2]>=0).all() and (v[:,:,2]<=1).all()
        assert (o[:,:3]<-1e20).all(), 'short-history selection illegally admitted'
        # Validate a deterministic sample of score files rather than reading a billion rows on every monitor tick.
        if int(r['job'][:4],16)%31==0 or len(records)<10:
            p=pd.read_parquet(folder/'predictions.parquet');assert not p.duplicated(['date','stockid']).any();assert np.isfinite(p.score).all();preds+=len(p)
            assert digest(folder/'predictions.parquet')==r['predictions_sha256']
    meta,cal=load_meta();labels=np.load(RUN/'data/labels.npy');ex=np.load(RUN/'data/label_exit_index.npy')
    from config import START,END
    with np.load(RUN/'data/market.npz') as market:
        actions=(market['dividend']!=0)|(market['split']!=1)
        unquoted=actions&(~np.isfinite(market['open'])|~np.isfinite(market['close']))
        evaluation=(cal>=START)&(cal<=END)
        unquoted_evaluation_actions=int(unquoted[evaluation].sum())
        unquoted_outside_evaluation=int(unquoted[~evaluation].sum())
        assert unquoted_evaluation_actions==0, 'suspended corporate-action valuation requires a separate ledger revision'
    op=pd.read_parquet(RUN.parent.parent/'training_data/audit_rebuild_20261021/original_inputs/prices.parquet',columns=['date','stockid','adj_open'])
    lookup=op.set_index(['date','stockid']).adj_open
    label_checks=0
    for row in np.linspace(0,len(meta)-100,41).astype(int):
        d=int(meta.iloc[row].date_index);sid=meta.iloc[row].stockid
        for hi,h in enumerate(HORIZONS):
            if d+h+1>=len(cal):continue
            a=lookup.get((cal[d+1],sid),np.nan);b=lookup.get((cal[d+h+1],sid),np.nan)
            expected_return=b/a-1
            if np.isfinite(expected_return):assert np.isclose(labels[row,hi],expected_return,atol=1e-6)
            assert ex[row,hi]==d+h+1;label_checks+=1
    selected=RUN/'selection/monthly_decisions.json'
    if selected.exists():
        for j,row in enumerate(json.loads(selected.read_text())):
            if row['candidate'] is not None:
                assert pd.Timestamp(row['last_validation_close'])<pd.Timestamp(row['first_signal'])
                assert row['candidate'] in actual
                with np.load(RUN/'models'/row['candidate']/'risk_results.npz') as a:
                    assert np.isclose(row['objective'],a['objectives'][row['risk_id'],j])
    if actual==expected_jobs:
        assert fits==967680 and accounts==27993600
        assert set(json.loads((RUN/'selection/candidate_snapshot.json').read_text()))==expected_jobs
    result=dict(completed_models=len(records),actual_fits=fits,completed_accounts=accounts,fold_boundaries_checked=folds,
                registered_models=len(expected_jobs),missing_registered_models=len(expected_jobs-actual),result_hashes_checked=result_hashes,
                unquoted_actions_in_evaluation=unquoted_evaluation_actions,
                unquoted_actions_outside_evaluation=unquoted_outside_evaluation,
                sampled_prediction_rows_checked=preds,independent_label_checks=label_checks,max_cash_flow_error=max_cash,
                scope='timestamp, numeric and sampled prediction-file audit; not proof of future profitability',
                source_code={p.name:digest(p) for p in __import__('pathlib').Path(__file__).parent.glob('*.py')})
    assert max_cash<1e-4;dump(result,REPORT/'审计快照.json');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
