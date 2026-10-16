"""只读展开1天模型的1187个日期，以及保存模型中180次加法。"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training_data/research_20261011'
RUN=ROOT/'training_runs/research_20261011'
OUT=ROOT/'学习笔记/20261015_真实日期逐步解释'
prices=pd.read_parquet(DATA/'prices.parquet')
panel=pd.read_parquet(DATA/'model_panel.parquet').sort_values(['date','stockid'])
dates=pd.DatetimeIndex(sorted(prices.date.unique()))
daily=prices.groupby('date').agg(observed=('observed','sum'),pool=('in_pool','sum'))
folds=json.loads((RUN/'h1/folds.json').read_text(encoding='utf-8'))
pred=pd.read_parquet(RUN/'h1/predictions.parquet')
valid=pd.read_parquet(RUN/'h1/2024Q4_validation_predictions.parquet')
allpred=pd.concat([valid,pred]).sort_values(['date','stockid'])
groups={d:g for d,g in allpred.groupby('date')}
mature=panel.dropna(subset=['target_1','label_exit_1']).groupby('label_exit_1').agg(count=('stockid','size'),signal=('date','min'))
past=dates[dates<pd.Timestamp('2024-10-08')]
inner=panel[panel.date.isin(past[-630:-126])&(panel.label_exit_1<past[-126])].dropna(subset=['target_1'])
lines=[
'这张表按历史日期解释现有1天模型能使用哪些材料。实际脚本先批量生成历史表，再按季度调用训练和预测；它不是下面这样逐日修改模型的程序。',
'',
'范围：2021-11-15至2026-10-09，共1187个交易日。1天答案定义：当天收盘记录输入，下一交易日开盘作为买入价，再下一交易日开盘作为卖出价。',
'',
'“新答案”只统计当天可以结束、且收益数据有效的股票池记录。“输出”指LightGBM给每只股票的相对排名分数，不是实际收益百分比，不是买单。“示例”取当日预测表中股票代码最小的一只。',
'',
'“启用”是历史回放的信息边界，不是声称程序在历史当天实际运行。“初训模型”是为2024-03-27至2024-09-30内部比较产生的模型，其学习记录是2022-03-01至2024-03-22，答案最晚2024-03-26结束。',
'',
'当日固定顺序：开盘取得新价格，使更早记录的答案可以计算 → 收盘取得当日六项行情、读取供应商因子 → 判断是否有60天历史、是否活跃、成交额排名是否前100 → 计算当日335列候选值并按当天股票排名 → 有模型则仅取它指定的100列输出预测；没有模型则只保留记录。答案变成可知不会立即更新LightGBM。仅下表列出的启用日更换模型。',
'',
'| 日期 | 有效行情股票数 | 池内股票数 | 今天结束的旧记录 | 模型动作 | 训练使用记录数 | 输出数 | 一只股票的真实输出 | 新答案怎样反馈 |',
'|---|---:|---:|---|---|---:|---:|---|---|',
]
for day,row in daily.iterrows():
    ds=str(day.date());m=mature.loc[day] if day in mature.index else None
    mtext=f'{str(m.signal.date())}的{int(m["count"])}条记录' if m is not None else '无有效股票池答案'
    fold=next((f for f in folds if pd.Timestamp(f['test_start'])<=day<=pd.Timestamp(f['test_end'])),None)
    g=groups.get(day)
    if fold:
        new=day==pd.Timestamp(fold['test_start'])
        action=('重新训练并启用' if new else '沿用')+fold['quarter']+'模型'
        n=fold['training_rows']
        future=next((f for f in folds if pd.Timestamp(f['test_start'])>day),None)
        feedback='当天不修改规则；到'+str(pd.Timestamp(future['test_start']).date())+'重训时再筛入' if future else '当天不修改规则；样本结束前无下一次重训'
    elif day>=pd.Timestamp('2024-03-27'):
        action='启用内部比较模型' if day==pd.Timestamp('2024-03-27') else '沿用内部比较模型'
        n=len(inner);feedback='用于内部比较；10月8日启用的模型重训时再筛入'
    else:
        action='没有预测模型，只准备历史材料';n=0;feedback='此时没有模型可修改；留待后续训练筛入'
    example=f'{g.iloc[0].stockid}: {g.iloc[0].lgbm100:.8f}' if g is not None else '无'
    lines.append(f'| {ds} | {int(row.observed)} | {int(row.pool)} | {mtext} | {action} | {n:,} | {len(g) if g is not None else 0} | {example} | {feedback} |')
(OUT/'16_从第一天到最后一天逐日动作.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

obj=joblib.load(RUN/'h1/2024Q4_models.joblib')['lgbm100']
cols=obj['features'];train=panel[panel.date.isin(past[-504:])&(panel.label_exit_1<'2024-10-08')].dropna(subset=['target_1'])
X=train[cols].fillna(0).to_numpy(dtype=np.float32);y=train.target_1.to_numpy()
trees=obj['model'].booster_.dump_model()['tree_info']
def evaluate_tree(tree,indices,out):
    if 'leaf_value' in tree:
        out[indices]=tree['leaf_value'];return
    left=X[indices,tree['split_feature']]<=float(tree['threshold'])
    evaluate_tree(tree['left_child'],indices[left],out)
    evaluate_tree(tree['right_child'],indices[~left],out)

base=y.mean();prediction=np.zeros(len(y));idx=np.arange(len(y));rows=[]
for i,tree in enumerate(trees,1):
    contribution=np.zeros(len(y));evaluate_tree(tree['tree_structure'],idx,contribution)
    prediction+=contribution
    rows.append({'step':i,'sse':float(((y-prediction)**2).sum()),'first_example_prediction':float(prediction[0]),'first_example_answer':float(y[0]),'first_example_error':float(y[0]-prediction[0])})
expected=obj['model'].predict(X);assert np.max(abs(expected-prediction))<1e-10
log=['这是2024年第四季度1天LightGBM保存模型的实际180棵树，读取每棵树并逐个加总得到。没有重新训练，也没有用测试答案来改树。',
'','训练材料为2022-09-01至2024-09-26的50,136条记录；答案最晚2024-09-30结束。每一轮同时检查这批全部旧记录。轮数不是日期。',
'',f'所有记录先给同一个基础分数：{base:.10f}。起始平方误差总和：{((y-base)**2).sum():.8f}。',
'','每轮：读取上一轮对全部旧记录的预测 → 答案减去预测，计算误差 → 形成一套新的if/else判断 → 每条记录按判断进入某个末端组 → 给组内记录加一个修正数 → 重新计算误差，进入下一轮。',
'',f'表中示例是{train.iloc[0].date.date()}的{train.iloc[0].stockid}；答案是{y[0]:.10f}。该示例的误差可能变大，模型优化的是整批记录，不保证每条都改善。',
'','| 轮次 | 全部旧记录平方误差总和 | 示例股票当前预测 | 示例股票真实答案 | 答案减预测 |','|---:|---:|---:|---:|---:|']
for r in rows:log.append(f'| {r["step"]} | {r["sse"]:.8f} | {r["first_example_prediction"]:.10f} | {r["first_example_answer"]:.10f} | {r["first_example_error"]:.10f} |')
(OUT/'17_LightGBM一次训练内部180轮反馈.md').write_text('\n'.join(log)+'\n',encoding='utf-8')
checks={'first_date':str(dates.min().date()),'last_date':str(dates.max().date()),'dates':len(dates),'earliest_pool':str(panel.date.min().date()),'trees':len(rows),'max_prediction_difference':float(np.max(abs(expected-prediction))),'initial_inner_rows':len(inner),'outer_training_rows':len(train)}
(OUT/'逐日与训练轮次核验.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
