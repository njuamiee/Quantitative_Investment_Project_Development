"""只读复核既有研究结果，导出逐日时间边界和一个真实信号日的证据。"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import joblib
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'training_data/research_20261011'
RUN = ROOT / 'training_runs/research_20261011'
OUT = ROOT / '学习笔记/20261015_真实日期逐步解释'
OUT.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(ROOT / 'tools/research_20261011'))
from ledger import metrics

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso', force_ascii=False))

panel = pd.read_parquet(DATA / 'model_panel.parquet').sort_values(['date','stockid'])
prices = pd.read_parquet(DATA / 'prices.parquet')
dates = pd.DatetimeIndex(sorted(prices.date.unique()))
day = pd.Timestamp('2026-04-01')
cut = day
names = {'BJ'+p.name[:6]:p.stem[7:].removeprefix('BJ-') for p in (DATA/'bj/raw').glob('*.csv')}
u = prices[prices.date == day].copy()
u['name'] = u.stockid.map(names)
u['sequential_reason'] = np.select([
    u.observed_days < 60, u.active20 < 16, ~u.observed,
    ~(u.adv20 > 0), u.amount_rank > 100],
    ['有效交易记录不足60天','最近20日有效交易不足16天','当天交易记录无效','20日均成交额无效','成交额排名在100名以后'], default='入选')
save(u[['date','stockid','name','observed_days','active20','observed','adv20','amount_rank','in_pool','sequential_reason']], '01_20260401全部股票筛选过程.csv')
f = pd.read_csv(RUN/'h5/2026Q2_deployment_factors.csv')
train = panel[panel.date.isin(dates[dates < cut][-504:]) & (panel.label_exit_5 < cut)].dropna(subset=['target_5'])
selected = f.loc[f.selected,'factor'].tolist()
ranked = f.factor.tolist()
# 复现现有筛选函数的整张表相关性定义，解释淘汰的具体原因。
corr = train[ranked].fillna(0).corr().abs()
chosen = []; trace = []
for rank, row in enumerate(f.itertuples(index=False),1):
    related = corr.loc[row.factor,chosen] if chosen else pd.Series(dtype=float)
    maximum = float(related.max()) if len(related) else 0.
    blocker = related.idxmax() if len(related) else ''
    if len(chosen) >= 100: reason = '已达到100列上限'
    elif maximum >= .92: reason = '与已入选因子过于相似'
    else: reason = '入选'; chosen.append(row.factor)
    trace.append({'rank':rank, **row._asdict(), 'max_abs_correlation':maximum,
                  'closest_selected':blocker, 'reason':reason})
assert chosen == selected
trace = pd.DataFrame(trace)
all_features=[c for c in panel if c.startswith('x_')]
low_coverage=[]
for c in all_features:
    if c not in ranked:
        low_coverage.append({'factor':c,'coverage':float(train[c].notna().mean()),'selected':False,'reason':'有效数据比例不足70%'})
trace=pd.concat([trace,pd.DataFrame(low_coverage)],ignore_index=True)
save(trace, '02_2026Q2五天模型全部因子筛选过程.csv')

q = pd.read_parquet(RUN/'h5/2026Q2_predictions.parquet')
today = q[q.date == day].sort_values(['score','stockid'],ascending=[False,True]).copy()
today.insert(2,'name',today.stockid.map(names)); today.insert(3,'rank',range(1,len(today)+1))
save(today, '03_20260401一百只股票最终评分.csv')
model_bundle = joblib.load(RUN/'h5/2026Q2_models.joblib')
obj = model_bundle['lgbm100']; cols=obj['features']
today_x = panel[panel.date == day].copy()
actual_pred = obj['model'].predict(today_x[cols].fillna(0).to_numpy(dtype=np.float32))
stored = q[q.date == day].set_index('stockid').loc[today_x.stockid,'lgbm100'].to_numpy()
assert np.max(np.abs(actual_pred-stored)) < 1e-10
save(today_x[['date','stockid']+cols], '04_20260401实际交给LightGBM的100列.csv')
save(train[['date','stockid','label_exit_5','label_5','target_5']+cols].head(5), '05_训练表前五行含答案.csv')

fold_rows=[]; daily_rows=[]; inner_rows=[]
for h in [1,2,5,10,20]:
    folds=json.loads((RUN/f'h{h}/folds.json').read_text(encoding='utf-8'))
    mapping={}
    for fold in folds:
        record={k:fold[k] for k in ['quarter','horizon','training_start','training_last_signal','training_last_label_exit','test_start','test_end','training_rows']}
        record.update(model=fold['selected']['model'],topn=fold['selected']['topn'],policy=fold['selected']['policy'])
        fold_rows.append(record)
        for d in dates[(dates>=pd.Timestamp(fold['test_start'])) & (dates<=pd.Timestamp(fold['test_end']))]: mapping[d]=fold
    for d in dates[(dates>='2024-01-01') & (dates<='2026-10-09')]:
        fold=mapping.get(d)
        row={'date':d,'horizon':h,'status':'只有历史材料；本版本尚未开始逐日前向预测' if fold is None else '使用当季冻结模型预测', 'refit_today':fold is not None and d==pd.Timestamp(fold['test_start'])}
        if fold:
            row.update({k:fold[k] for k in ['quarter','training_start','training_last_signal','training_last_label_exit','training_rows']})
            row['factor_file']=f'h{h}/{fold["quarter"]}_deployment_factors.csv'
            row['selected_model']=fold['selected']['model']
        daily_rows.append(row)
    past=dates[dates<cut];vd=past[-126:];td=past[-630:-126]
    inner=panel[panel.date.isin(td)&(panel[f'label_exit_{h}']<vd[0])].dropna(subset=[f'target_{h}'])
    valid=panel[panel.date.isin(vd)&(panel[f'label_exit_{h}']<cut)].dropna(subset=[f'target_{h}'])
    outer=panel[panel.date.isin(past[-504:])&(panel[f'label_exit_{h}']<cut)].dropna(subset=[f'target_{h}'])
    inner_rows.append({'horizon':h,'inner_start':inner.date.min(),'inner_last_signal':inner.date.max(),'inner_last_exit':inner[f'label_exit_{h}'].max(),
        'inner_rows':len(inner),'inner_signal_days':inner.date.nunique(),'validation_start':vd[0],'validation_end':vd[-1],
        'validation_last_mature_signal':valid.date.max(),'validation_mature_days':valid.date.nunique(),'outer_start':outer.date.min(),
        'outer_last_signal':outer.date.max(),'outer_last_exit':outer[f'label_exit_{h}'].max(),'outer_days':outer.date.nunique(),'outer_rows':len(outer)})
save(pd.DataFrame(fold_rows),'06_所有季度所有期限训练时间表.csv')
save(pd.DataFrame(daily_rows),'07_2024至2026每个交易日到底用哪次训练.csv')
save(pd.DataFrame(inner_rows),'08_20260401训练和内部考试完整边界.csv')

ic_col='x_c02_gtalpha046'
ic_day=pd.Timestamp('2026-03-24')
ex=panel[panel.date==ic_day][['date','stockid',ic_col,'label_5','label_exit_5','target_5']].copy()
ex['factor_rank']=ex[ic_col].rank();ex['return_rank']=ex.label_5.rank()
save(ex,'09_20260324单日IC逐股票计算材料.csv')
def actual_ic(d):
    xx=d[ic_col]-d[ic_col].mean();yy=d.target_5-d.target_5.mean()
    return (xx*yy).sum()/np.sqrt(xx.pow(2).sum()*yy.pow(2).sum())
icseries=train.groupby('date').apply(lambda d:d[ic_col].corr(d.target_5))
code_ics=train.groupby('date').apply(actual_ic)
save(pd.DataFrame({'code_IC':code_ics,'pairwise_complete_IC':icseries}).reset_index(),'10_用于选因子的每一天IC.csv')
old_card=pd.read_csv(ROOT/'training_runs/teacher9_20260930/bj/2026Q2/factor_cards.csv')
old_trace=pd.read_csv(ROOT/'training_runs/teacher_answers_20261012/selection_trace_2026Q2.csv')
save(old_trace,'11_旧40因子版本的真实筛选过程.csv')

daily=pd.read_csv(RUN/'h5/selected_daily.csv',parse_dates=['date'])
e=daily.equity.to_numpy();ret=np.r_[e[0]/1e6-1,e[1:]/e[:-1]-1];peaks=np.maximum.accumulate(np.r_[1e6,e])[1:]
di=int(np.argmin(e/peaks-1));pi=int(np.argmax(np.r_[1e6,e[:di+1]]))-1
daily['daily_return']=ret;daily['historical_peak']=peaks;daily['drawdown']=e/peaks-1
save(daily,'12_五天策略每日资产夏普回撤计算表.csv')
orders=pd.read_csv(RUN/'h5/selected_orders.csv',parse_dates=['date'])
save(orders[orders.date.between('2026-04-01','2026-04-03')],'13_真实订单4月1日至3日.csv')
reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date')
sample_dates=['2024-01-02','2024-01-03','2024-01-04','2024-12-31','2025-01-02','2025-12-31','2026-03-24','2026-03-25','2026-03-31','2026-04-01']
label_examples=panel[(panel.stockid=='BJ920001')&panel.date.isin(pd.to_datetime(sample_dates))][['date','stockid']+[f'label_exit_{h}' for h in [1,2,5,10,20]]]
save(label_examples,'14_每个出题日对应哪一天才知道答案.csv')
tree=obj['model'].booster_.dump_model()['tree_info'][0]['tree_structure']
def tree_leaf(tree, row):
    while 'split_index' in tree:
        tree=tree['left_child'] if row[cols[tree['split_feature']]]<=float(tree['threshold']) else tree['right_child']
    return tree['leaf_value']
one_row=today_x.set_index('stockid').loc['BJ920249',cols].fillna(0)
tree_outputs=[tree_leaf(t['tree_structure'],one_row) for t in obj['model'].booster_.dump_model()['tree_info']]
top5=today.head(5).stockid.tolist()
feature_example=today_x.set_index('stockid').loc[top5,['x_c02_gtalpha046','x_c02_gtalpha063','x_mom5']].reset_index()
price_examples=prices[(prices.stockid.isin(top5))&prices.date.isin(pd.to_datetime(['2026-04-01','2026-04-02','2026-04-03','2026-04-07','2026-04-08','2026-04-09']))][['date','stockid','open','close','adj_open','adj_close']]
save(price_examples,'15_真实买卖日期的原始与复权价格.csv')
summary={
 'signal_date':day,'universe_file_count':len(u),'ever_observed_by_day':int((u.observed_days>0).sum()),'observed_today':int(u.observed.sum()),'eligible':int(u.eligible.sum()),'pool':int(u.in_pool.sum()),
 'excluded':u.sequential_reason.value_counts().to_dict(),
 'pool_cutoff_examples':records(u[u.amount_rank.between(98,103)][['stockid','name','adv20','amount_rank']].sort_values('amount_rank')),
 'excluded_examples':records(u[u.sequential_reason!='入选'].groupby('sequential_reason').head(2)[['stockid','name','observed_days','active20','adv20','sequential_reason']]),
 'top10':records(today.head(10)[['stockid','name','rank','lgbm100','ensemble','score','label_5']]),
 'feature_example':records(feature_example),'factor_trace_first15':records(trace.head(15)),
 'old_factors_example':records(old_card[old_card.feature.isin(['x_c02_gtalpha046','x_c02_gtalpha067','x_c02_gtalpha063'])]),
 'boundaries':records(pd.DataFrame(inner_rows)),
 'label_dates':records(label_examples),'factor_count':len(f),'selected_count':len(cols),
 'reloaded_prediction_max_difference':float(np.max(np.abs(actual_pred-stored))),
 'first_tree_split':{'feature':cols[tree['split_feature']],'threshold':tree['threshold'],'decision_type':tree['decision_type']},
 'lgbm_tree_count':len(tree_outputs),'lgbm_first_tree_output_BJ920249':tree_outputs[0],'lgbm_sum_tree_outputs_BJ920249':sum(tree_outputs),
 'training_first5':records(train[['date','stockid','x_c02_gtalpha046','x_c02_gtalpha063','x_mom5','label_5','target_5']].head(5)),
 'single_day_ic':float(ex[ic_col].corr(ex.target_5)),'single_day_ic_stocks':len(ex.dropna()),'single_day_label_exit':ex.label_exit_5.max(),
 'single_day_code_ic':float(code_ics.loc[ic_day]),'code_ic_mean':float(code_ics.mean()),'code_ic_std':float(code_ics.std()),
 'factor_ic_mean_pairwise':float(icseries.mean()),'factor_ic_std_pairwise':float(icseries.std()),'factor_ic_days':len(icseries),
 'account_metrics':metrics(daily),'mean_daily_return':float(ret.mean()),'daily_return_sd':float(ret.std(ddof=1)),
 'maxdd_peak_date':str(daily.date.iloc[pi]) if pi>=0 else 'initial capital','maxdd_peak_value':float(peaks[di]),'maxdd_trough_date':daily.date.iloc[di],'maxdd_trough_value':float(e[di]),
 'april1_risk_cap':float(reg.loc[day,'cap_hmm3_defensive']),
 'april2_buys':records(orders[(orders.date=='2026-04-02')&(orders.side=='buy')]),
 'april1_account':records(daily[daily.date=='2026-04-01']),
 'h5_folds': [r for r in fold_rows if r['horizon']==5],
}
(OUT/'核实结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
