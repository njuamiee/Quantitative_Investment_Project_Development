import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'training_runs/research_20261011'
OUT=Path(__file__).parent
def read(name): return json.loads((RUN/name).read_text(encoding='utf-8-sig'))
def records(frame): return json.loads(frame.to_json(orient='records',date_format='iso'))
d={}
for name in ['candidate_summary','validation_selected_summary']:
    d[name]=records(pd.read_csv(RUN/f'{name}.csv'))
for name in ['short_windows/summary','h5/candidate_subperiods','h5/candidate_sensitivity','h5/hmm_matched_exposure','h5/candidate_hmm_controls','minute_factor_results','kronos_results','jev_status','bj50_benchmark_metrics','h5/equal_pool_metrics','h5/selected_metrics']:
    d[name]=read(name+'.json')
d['audit']=json.loads((ROOT/'学习笔记/20261015_真实日期逐步解释/核实结果.json').read_text(encoding='utf-8-sig'))
d['h5_folds']=read('h5/folds.json')
d['h5_selection']=read('h5/2026Q2_selection.json')
p=pd.read_parquet(RUN/'h5/predictions.parquet')
p=p[p.date<='2026-09-30'].dropna(subset=['target_5','label_5','lgbm100'])
rows=[]; gr=[]
for date,g in p.groupby('date'):
    ic=g.lgbm100.corr(g.target_5,method='spearman')
    bins=pd.qcut(g.lgbm100.rank(method='first'),5,labels=False)
    rr={'date':str(date.date()),'quarter':str(date.to_period('Q')),'ic':ic}
    for k in range(5): rr['g'+str(k+1)]=float(g.loc[bins==k,'label_5'].mean())
    rows.append(rr)
ic=pd.DataFrame(rows)
q=ic.groupby('quarter').ic.agg(['count','mean','std','min']).reset_index()
q['icir']=q['mean']/q['std']
d['quarterly_lgbm5_ic']=records(q)
d['lgbm5_ic_summary']={'days':len(ic),'mean':ic.ic.mean(),'std':ic.ic.std(),'icir':ic.ic.mean()/ic.ic.std(),'worst_quarter':q.loc[q['mean'].idxmin(),'quarter'],'quarter_mean_sd':q['mean'].std()}
d['lgbm5_groups']=[{'group':i,'return':ic['g'+str(i)].mean()} for i in range(1,6)]
ic.to_csv(OUT/'h5_lgbm_daily_ic_groups.csv',index=False,encoding='utf-8-sig')
for name in ['selected','candidate','equal_pool']:
    df=pd.read_csv(RUN/f'h5/{name}_daily.csv')
    d['chart_'+name]={'dates':df.date.str[:10].tolist(),'equity':df.equity.tolist()}
b=pd.read_csv(RUN/'bj50_benchmark.csv')
d['benchmark_columns']=b.columns.tolist()
timing=pd.DataFrame(read('factor_timing_results.json'))
d['factor_timing_summary']={'tested':len(timing),'passed':int(timing.pass_target.sum())}
mm=pd.DataFrame(read('minute_factor_results.json'))
d['minute_table']=records(mm[mm.policy=='none'][['horizon','total_return','sharpe_rf15','max_drawdown','calmar','days']])
sens=pd.DataFrame(d['h5/candidate_sensitivity'])
d['sensitivity_columns']=sens.columns.tolist()
d['sensitivity_table']=records(sens[[x for x in ['cost_bp','delay','label','scenario','total_return','annual_return','sharpe_rf15','max_drawdown','calmar'] if x in sens]])
fs=pd.read_csv(RUN/'h5/fixed_comparisons.csv')
d['fixed_hmm']=records(fs[(fs.model=='ensemble')&(fs.topn==20)&(fs.guard=='none')&fs.policy.isin(['none','fixed50','vol12','hmm2_defensive','hmm3_defensive','hmm3_volatility'])])
OUT.joinpath('deck_data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:d[k] for k in ['quarterly_lgbm5_ic','lgbm5_ic_summary','lgbm5_groups','factor_timing_summary','minute_table','sensitivity_table','benchmark_columns']},ensure_ascii=False,indent=2))
