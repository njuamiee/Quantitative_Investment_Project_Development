"""Independent checks against saved selection dates, source scores and ledgers."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from run_experiments import load,RULES,START,COMMON
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'training_runs/meeting_update_20261023'
m,frames,p,md=load()
log=pd.read_csv(OUT/'selection_all_candidates.csv',parse_dates=['decision_date','validation_start','validation_end'])
schedule=json.loads((OUT/'selection_schedule.json').read_text())
for date,g in log.groupby('decision_date'):
    assert len(g)==36 and g.selected.sum()==1
    assert (g.n>=40).all() and (g.n<=60).all()
    assert (m.dates.get_loc(date)-m.dates.get_loc(g.validation_end.iloc[0])-1)==10
    expected=g.iloc[g.objective_sharpe0.argmax()].candidate
    assert schedule[str(date.date())]==expected
    for row in g.itertuples():
        d=pd.read_parquet(OUT/f'{row.candidate}__fee5_daily.parquet').set_index('date')
        ret=d.equity.pct_change().loc[row.validation_start:row.validation_end]
        score=float(ret.mean()/ret.std(ddof=1)*np.sqrt(252))
        assert abs(score-row.objective_sharpe0)<1e-10
orders=pd.read_parquet(OUT/'rolling_selected__fee5_orders.parquet')
maps={name:f.set_index(['date','stockid']).score for name,f in frames.items()}
for row in orders[orders.side.eq('buy')].itertuples():
    eligible=[k for k in schedule if pd.Timestamp(k)<=row.signal_date]
    model=(schedule[max(eligible)] if eligible else 'original_monthly__equal20').split('__')[0]
    assert row.date>row.signal_date
    assert abs(float(maps[model].loc[(row.signal_date,row.stockid)])-row.score)<1e-12
folds=pd.read_csv(ROOT/'training_runs/audit_rebuild_20261021/expanded/folds.csv')
assert (pd.to_datetime(folds.train_actual_last_exit)<pd.to_datetime(folds.val_start)).all()
assert (pd.to_datetime(folds.val_actual_last_exit)<pd.to_datetime(folds.test_start)).all()
saved=json.loads((OUT/'input_hashes.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in saved.items())
checks=pd.read_csv(ROOT/'实验更新_20261023/账户核验.csv')
assert len(checks)==44 and checks.identity_error.max()<1e-5 and checks.cash_flow_error.max()<1e-5
manifest={'accounts':44,'fixed_candidates':36,'selection_windows':len(schedule),'expanded_model_fold_rows_checked':len(folds),
          'rolling_buy_order_scores_verified':int(orders.side.eq('buy').sum()),'checks':'passed','data_end':'2026-10-19',
          'candidate_specs':[r.__dict__ for r in RULES],
          'selection':{'lookback':60,'gap':10,'step':20,'minimum_days':40,'objective':'historical_daily_sharpe_rf0','cash_candidate':False},
          'warning':'Historical exploratory protocol; no independent unseen test or live order integration.'}
(OUT/'final_audit.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
reportdir=ROOT/'实验更新_20261023'
receipt={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in reportdir.rglob('*') if f.is_file() and f.name!='交付校验.json'}
(reportdir/'交付校验.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['candidate_specs','selection']},ensure_ascii=False))
