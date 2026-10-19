"""Read-only reconstruction of published selections, folds and drawdown evidence."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'training_runs/teacher9_20260930'
EXT=ROOT/'training_runs/teacher9_20261008'
NEW=ROOT/'training_runs/research_20261011'
OUT=ROOT/'training_runs/teacher_answers_20261012'
OUT.mkdir(parents=True,exist_ok=True)

def load(path): return json.loads(path.read_text(encoding='utf-8'))
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def csv(name,frame): frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')

folds=load(OLD/'bj_folds.json'); records=[]; cards_all=[]; checks=[]
panel=pd.read_parquet(ROOT/'training_data/research_20260930/bj_model_panel.parquet',columns=['date','stockid','exit_date','target'])
for f in folds:
    q=f['quarter']; card=pd.read_csv(OLD/'bj'/q/'factor_cards.csv').set_index('feature')
    corr=pd.read_csv(OLD/'bj'/q/'mean_daily_spearman.csv',index_col=0)
    qualified=card[(card.coverage>=.7)&(card.n_ic_days>=100)&card.mean_ic.notna()].sort_values(['abs_ic','coverage'],ascending=False)
    candidates=qualified.head(100).index.tolist(); selected=[]; trace=[]
    for rank,name in enumerate(candidates,1):
        overlaps=[s for s in selected if abs(corr.loc[name,s])>=.85]
        reason='redundant_with:'+overlaps[0] if overlaps else ('feature_budget_40' if len(selected)>=40 else 'selected')
        maximum=max((abs(corr.loc[name,s]) for s in selected),default=0)
        trace.append({'quarter':q,'candidate_rank':rank,'feature':name,'mean_ic':card.loc[name,'mean_ic'],
                      'coverage':card.loc[name,'coverage'],'decision':reason,'max_abs_corr_to_previously_selected':maximum,
                      'blocking_corr':float(corr.loc[name,overlaps[0]]) if overlaps else None})
        if reason=='selected': selected.append(name)
    checks.append({'check':q+'_reconstruct_original40','passed':selected==f['selected_features']})
    ext=load(EXT/'bj'/q/'selection.json')
    checks.append({'check':q+'_extension_same40','passed':selected==ext['ordered_features'][:40]})
    valid=panel[(panel.date>=f['validation_start'])&(panel.date<f['test_first'])&(panel.exit_date<f['test_first'])&panel.target.notna()]
    train=panel[(panel.date>=f['train_first'])&(panel.date<f['validation_start'])&(panel.exit_date<f['validation_start'])&panel.target.notna()]
    refit=panel[(panel.date>=f['train_first'])&(panel.date<f['test_first'])&(panel.exit_date<f['test_first'])&panel.target.notna()]
    records.append({**{k:v for k,v in f.items() if k!='selected_features'},'qualified_count':len(qualified),'top_candidate_count':len(candidates),
                    'validation_last_signal':str(valid.date.max()),'validation_label_end':str(valid.exit_date.max()),
                    'refit_last_signal':str(refit.date.max()),'training_signal_days':train.date.nunique(),
                    'validation_signal_days':valid.date.nunique(),'extended_dedup_count':ext['deduplicated_count'],
                    'selected_q_below05':int((card.loc[selected,'by_fdr_q']<.05).sum()),
                    'alpha101_count':sum(s.startswith('x_c01') for s in selected),'alpha191_count':sum(s.startswith('x_c02') for s in selected),
                    'price_count':sum(not s.startswith(('x_c01','x_c02')) for s in selected),
                    'selected_negative_ic_count':int((card.loc[selected,'mean_ic']<0).sum()),
                    'features':selected,'decisions':pd.Series([t['decision'].split(':')[0] for t in trace]).value_counts().to_dict()})
    card=card.reset_index(); card['quarter']=q
    card['selected_order']=card.feature.map({s:i+1 for i,s in enumerate(selected)})
    cards_all.append(card)
    csv(f'selection_trace_{q}.csv',pd.DataFrame(trace))
save('original_folds.json',records)
allcards=pd.concat(cards_all,ignore_index=True);csv('all_factor_cards_8folds.csv',allcards)
csv('selected40_all_quarters.csv',allcards[allcards.selected_order.notna()].sort_values(['quarter','selected_order']))
q3=allcards[(allcards.quarter=='2026Q3')&allcards.selected_order.notna()].sort_values('selected_order').copy()
ridge=np.load(OLD/'bj/2026Q3/ridge.npz');coeff=dict(zip(ridge['features'],ridge['coef']))
q3['ridge_coefficient']=q3.feature.map(coeff)
registry=pd.read_csv(OLD/'all_factor_registry_quality.csv').fillna('')
def dictionary_row(feature):
    if feature.startswith('x_c02_'): match=registry[registry.table.str.endswith('_c02')&(registry.field==feature[6:])]
    elif feature.startswith('x_c01_'): match=registry[registry.table.str.endswith('_c01')&registry.field.str.endswith('_'+feature[-3:])]
    else: return {'meaning':'local price feature','notes':''}
    return {'meaning':match.supplier_meaning.iloc[0],'notes':match.supplier_notes.iloc[0]} if len(match)==1 else {'meaning':'unresolved','notes':''}
q3['supplier_meaning']=q3.feature.map(lambda x:dictionary_row(x)['meaning'])
q3['supplier_notes']=q3.feature.map(lambda x:dictionary_row(x)['notes'])
csv('selected40_2026Q3_details.csv',q3)
save('ridge_2026Q3.json',{'alpha':10,'intercept':float(ridge['intercept']),'coefficients':coeff})
f=records[-1];cols=f['features']
rp=pd.read_parquet(ROOT/'training_data/research_20260930/bj_model_panel.parquet',columns=['date','exit_date','target']+cols)
rp=rp[(rp.date>=f['train_first'])&(rp.date<f['test_first'])&(rp.exit_date<f['test_first'])&rp.target.notna()]
x=rp[cols].fillna(0).to_numpy(dtype=float);x-=x.mean(axis=0);e=np.linalg.eigvalsh(x.T@x)
save('ridge_condition.json',{'refit_rows':len(rp),'missing_fraction':float(rp[cols].isna().to_numpy().mean()),
 'gram_condition':float(e[-1]/e[0]),'ridge_gram_condition':float((e[-1]+10)/(e[0]+10))})
del rp,x
sets=[set(f['selected_features']) for f in folds]
save('selection_summary.json',{'total_distinct_selected':len(set.union(*sets)),'always_selected':sorted(set.intersection(*sets)),
 'adjacent_overlap':[len(a&b) for a,b in zip(sets,sets[1:])],'q3_q_significant':int((q3.by_fdr_q<.05).sum())})

models=[f'{prefix}_{n}' for n in [20,40,60,80,100] for prefix in ['count','ridge']]
metrics=pd.read_csv(EXT/'extension_results.csv');diag=[]
benchmark=pd.read_csv(OLD/'bj_pred_selected_account.csv',parse_dates=['date']).set_index('date').benchmark_nav
for name in models:
    d=pd.read_csv(EXT/f'{name}_account.csv',parse_dates=['date'])
    dd=d.equity/d.equity.cummax()-1; trough=dd.idxmin();peak=d.loc[:trough,'equity'].idxmax()
    post=d.loc[trough:]; rec=post[post.equity>=d.loc[peak,'equity']]
    b=benchmark.reindex(d.date).to_numpy();rets=d.equity.pct_change();br=pd.Series(b).pct_change()
    beta=float(pd.DataFrame({'a':rets,'b':br}).cov().loc['a','b']/br.var())
    m=metrics[metrics.model==name].iloc[0].to_dict()
    same=d.set_index('date').equity
    diag.append({**m,'peak_date':str(d.loc[peak,'date'].date()),'trough_date':str(d.loc[trough,'date'].date()),
                 'peak_equity':d.loc[peak,'equity'],'trough_equity':d.loc[trough,'equity'],
                 'recovery_date':str(rec.iloc[0].date.date()) if len(rec) else None,
                 'benchmark_return_same_drawdown_window':b[trough]/b[peak]-1,'full_sample_beta_descriptive':beta,
                 'worst_daily_return':float(rets.min()),'worst_daily_date':str(d.loc[rets.idxmin(),'date'].date()),
                 'return_common_20260123_20260724':same.loc['2026-07-24']/same.loc['2026-01-23']-1})
csv('drawdown_diagnostics.csv',pd.DataFrame(diag))

icrows=[]; ic_summary=[]
for name in models:
    d=pd.read_csv(EXT/f'{name}_daily_ic.csv',parse_dates=['date'])
    for quarter,g in d.groupby(d.date.dt.to_period('Q')):
        icrows.append({'version':'old','model':name,'quarter':str(quarter),'n_days':int(g.ic.count()),
                       'mean_ic':g.ic.mean(),'daily_ic_std':g.ic.std(),'daily_icir':g.ic.mean()/g.ic.std()})
latest=pd.read_parquet(NEW/'h5/predictions.parquet')
for name in ['ridge60','lgbm100','ic','score']:
    temp=latest[['date','target_5',name]].dropna().copy()
    temp[['yr','pr']]=temp.groupby('date')[['target_5',name]].rank()
    g=temp.groupby('date')[['yr','pr']]; daily=g.apply(lambda x:x.yr.corr(x.pr) if len(x)>=30 else np.nan)
    for quarter,v in daily.groupby(daily.index.to_period('Q')):
        icrows.append({'version':'latest_h5','model':name,'quarter':str(quarter),'n_days':int(v.count()),
                      'mean_ic':v.mean(),'daily_ic_std':v.std(),'daily_icir':v.mean()/v.std()})
icdf=pd.DataFrame(icrows);csv('quarterly_ic.csv',icdf)
for (version,name),g in icdf[icdf.quarter!='2026Q4'].groupby(['version','model']):
    ic_summary.append({'version':version,'model':name,'folds':len(g),'mean_of_fold_ic':g.mean_ic.mean(),
       'std_of_fold_ic':g.mean_ic.std(),'worst_fold':g.loc[g.mean_ic.idxmin(),'quarter'],'worst_fold_ic':g.mean_ic.min(),
       'positive_folds':int((g.mean_ic>0).sum()),'pooled_daily_mean_ic':np.average(g.mean_ic,weights=g.n_days)})
csv('quarterly_ic_stability.csv',pd.DataFrame(ic_summary))

sources=[ROOT/'1012老师提出的新问题.pdf',ROOT/'tools/research_20260930/train.py',ROOT/'tools/research_20260930/prepare.py',
 ROOT/'tools/research_20260930/accounts.py',ROOT/'tools/research_20261008/experiments.py',ROOT/'tools/research_20261011/train_models.py',
 ROOT/'tools/research_20261011/prepare_data.py',EXT/'extension_results.csv',OLD/'bj_folds.json']
save('source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
save('checks.json',checks)
assert all(x['passed'] for x in checks)
print(json.dumps({'checks':len(checks),'passed':sum(x['passed'] for x in checks),'folds':len(records),
                  'latest_q3':{k:v for k,v in records[-1].items() if k not in ['features']},
                  'selection':load(OUT/'selection_summary.json')},ensure_ascii=False,indent=2))
