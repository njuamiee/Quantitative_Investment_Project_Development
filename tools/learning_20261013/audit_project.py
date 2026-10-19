"""Read-only project audit and concrete classroom examples; no model fitting."""
from pathlib import Path
import ast, hashlib, json, sys
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'training_runs/learning_20261013'
OUT.mkdir(parents=True, exist_ok=True)
RUN = ROOT / 'training_runs/research_20261011'
DATA = ROOT / 'training_data/research_20261011'

def dump(obj, name):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), 'utf-8')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

files = sorted((ROOT/'tools/research_20261011').glob('*'))
files += [ROOT/'tools/research_20260930'/x for x in ['prepare.py','train.py','accounts.py']]
files += [ROOT/'tools/research_20261008/minute_account.py', ROOT/'tools/teacher_answers_20261012/build_answer.py']
index = []
for path in files:
    if path.suffix not in ['.py','.md','.txt','.ps1']: continue
    source = path.read_text('utf-8-sig')
    nodes = ast.parse(source).body if path.suffix == '.py' else []
    functions = []
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            functions.append({'name':node.name,'line':node.lineno,'end':node.end_lineno})
            if isinstance(node, ast.ClassDef):
                functions.extend({'name':node.name+'.'+n.name,'line':n.lineno,'end':n.end_lineno}
                                 for n in node.body if isinstance(n,ast.FunctionDef))
    imports = [n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.module] if nodes else []
    index.append({'path':path.relative_to(ROOT).as_posix(),'lines':len(source.splitlines()),'sha256':sha(path),'functions':functions,'from_imports':imports})
dump(index, 'source_index.json')

panel = pd.read_parquet(DATA/'model_panel.parquet').sort_values(['date','stockid']).reset_index(drop=True)
prices = pd.read_parquet(DATA/'prices.parquet', columns=['date','stockid','adj_open','adj_close'])
dates = pd.DatetimeIndex(sorted(prices.date.unique()))
examples=[];checks=[]
for quarter in ['2026Q3','2026Q4']:
    record=json.loads((RUN/'h5'/f'{quarter}_selection.json').read_text('utf-8'))
    objects=joblib.load(RUN/'h5'/f'{quarter}_models.joblib')
    saved=pd.read_parquet(RUN/'h5'/f'{quarter}_predictions.parquet').sort_values(['date','stockid']).reset_index(drop=True)
    test=panel[panel.date.dt.to_period('Q').astype(str)==quarter].copy().reset_index(drop=True)
    assert test[['date','stockid']].equals(saved[['date','stockid']])
    fitted=objects['lgbm100'];features=fitted['features'];model=fitted['model']
    whole=model.predict(test[features].fillna(0).to_numpy(dtype=np.float32))
    daily=np.concatenate([model.predict(g[features].fillna(0).to_numpy(dtype=np.float32)) for _,g in test.groupby('date',sort=True)])
    saved_error=float(np.max(np.abs(whole-saved.lgbm100.to_numpy())))
    daily_error=float(np.max(np.abs(whole-daily)))
    assert saved_error<1e-10 and daily_error<1e-10
    assert 'date' not in features and 'stockid' not in features and not any('target_' in c or 'label_' in c for c in features)
    cutoff=test.date.min();past=dates[dates<cutoff];vd=past[-126:];td=past[-630:-126]
    train=panel[panel.date.isin(td)&(panel.label_exit_5<vd[0])].dropna(subset=['target_5'])
    refit=panel[panel.date.isin(past[-504:])&(panel.label_exit_5<cutoff)].dropna(subset=['target_5'])
    assert len(refit)==record['training_rows']
    probe=saved[saved.date.isin(saved.date.unique()[-2:])].groupby('date').head(3).copy()
    probe=probe[['date','stockid','lgbm100','score','label_5','target_5']]
    probe.to_csv(OUT/f'{quarter}_daily_predictions.csv',index=False,encoding='utf-8-sig')
    ex={'quarter':quarter,'feature_count':len(features),'feature_first_two':features[:2],
        'screening_train_first':str(train.date.min().date()),'screening_train_last':str(train.date.max().date()),
        'screening_train_rows':len(train),'screening_train_days':train.date.nunique(),
        'screening_last_label':str(train.label_exit_5.max().date()),
        'refit_first':str(refit.date.min().date()),'refit_last':str(refit.date.max().date()),
        'refit_last_label':str(refit.label_exit_5.max().date()),'refit_rows':len(refit),'refit_days':refit.date.nunique(),
        'validation_first':str(vd[0].date()),'validation_last':str(vd[-1].date()),
        'prediction_first':str(test.date.min().date()),'prediction_last':str(test.date.max().date()),
        'prediction_days':test.date.nunique(),'prediction_rows':len(test),
        'selected_score_model':record['selected']['model'],'model_sha256':sha(RUN/'h5'/f'{quarter}_models.joblib'),
        'whole_vs_saved_max_error':saved_error,'daily_vs_whole_max_error':daily_error,
        'last_two_days_missing_labels':int(probe.label_5.isna().sum()),'last_two_days_rows':len(probe)}
    examples.append(ex)
    checks.extend([{'name':quarter+'_saved_predictions_reproduced','passed':saved_error<1e-10},
                   {'name':quarter+'_daily_equals_batch_inference','passed':daily_error<1e-10},
                   {'name':quarter+'_dates_identifiers_and_labels_not_features','passed':True}])
dump(examples,'model_examples.json');dump(checks,'inference_checks.json')

old=json.loads((ROOT/'training_runs/teacher_answers_20261012/original_folds.json').read_text('utf-8'))[-1]
ridge=np.load(ROOT/'training_runs/teacher9_20260930/bj/2026Q3/ridge.npz')
old_panel=pd.read_parquet(ROOT/'training_data/research_20260930/bj_model_panel.parquet')
refit=old_panel[(old_panel.date>=old['train_first'])&(old_panel.date<old['test_first'])&(old_panel.exit_date<old['test_first'])&old_panel.target.notna()]
assert len(refit)==62466 and refit.date.max()==pd.Timestamp('2026-06-23')
dump({'table':'09.2 / rows 21-40','last_column':'Ridge beta','quarter':'2026Q3','rows':len(refit),
      'signal_start':str(refit.date.min().date()),'signal_end':str(refit.date.max().date()),
      'label_last_date':str(refit.exit_date.max().date()),'signal_days':refit.date.nunique(),
      'alpha':10,'intercept':float(ridge['intercept']),
      'coefficients_21_40':dict(zip(ridge['features'][20:].tolist(),ridge['coef'][20:].tolist()))},'old_table_answer.json')

results={}
for h in [1,2,5,10,20]:
    results[str(h)]={name:json.loads((RUN/f'h{h}'/(name+'.json')).read_text('utf-8'))
                    for name in ['selected_metrics','main_period_metrics','candidate_metrics','candidate_subperiods']}
results['short_windows']=json.loads((RUN/'short_windows/summary.json').read_text('utf-8'))
results['verification']=json.loads((RUN/'verification.json').read_text('utf-8'))
results['artifact_checks']=json.loads((RUN/'artifact_checks.json').read_text('utf-8'))
dump(results,'results_checked.json')

artifacts = list(RUN.glob('*.json')) + [RUN/'model_registry.json',RUN/'h5/folds.json',DATA/'model_panel.parquet']
artifacts += [RUN/f'h{h}'/name for h in [1,2,5,10,20] for name in ['selected_metrics.json','main_period_metrics.json','candidate_metrics.json','candidate_subperiods.json']]
dump({p.relative_to(ROOT).as_posix():sha(p) for p in artifacts},'artifact_hashes.json')
print(json.dumps({'source_files':len(index),'latest_python_files':sum(x['path'].startswith('tools/research_20261011/') and x['path'].endswith('.py') for x in index),'checks':checks,'models':examples},ensure_ascii=False,indent=2))
