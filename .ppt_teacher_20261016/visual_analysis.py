from pathlib import Path
import sys,json,math,zipfile,xml.etree.ElementTree as ET
import pandas as pd,numpy as np
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parents[1]; B=Path(__file__).parent/'visual_revision';B.mkdir(exist_ok=True)
D=R/'training_data/research_20261011'; H=R/'training_runs/research_20261011/h5'
p=pd.read_parquet(D/'prices.parquet');p.date=pd.to_datetime(p.date)
start=pd.Timestamp('2024-10-08'); end=pd.Timestamp('2026-09-30')
dates=pd.DatetimeIndex(sorted(p.loc[p.date.between(start,end),'date'].unique()))
initial=p[(p.date==start)&p.in_pool].sort_values('stockid').set_index('stockid');ids=initial.index.tolist()
assert len(ids)==100
pp=p[p.stockid.isin(ids)&p.date.between(start,end)]
op=pp.pivot(index='date',columns='stockid',values='open').reindex(index=dates,columns=ids)
cl=pp.pivot(index='date',columns='stockid',values='close').reindex(index=dates,columns=ids).ffill()
obs=pp.pivot(index='date',columns='stockid',values='observed').reindex(index=dates,columns=ids)
act=pd.read_csv(D/'actions.csv',parse_dates=['date']);q=pd.Series(0.,index=ids);cash=1e6;rows=[];buys=[]
for i,date in enumerate(dates):
    div=0.;cost=0.
    for a in act[act.date.eq(date)&act.stockid.isin(ids)].itertuples():
        div+=q[a.stockid]*a.dividend;cash+=q[a.stockid]*a.dividend;q[a.stockid]*=a.split
    if i==1:
        for s in ids:
            pr=op.loc[date,s];ref=initial.loc[s,'close']
            for a in act[act.date.eq(date)&act.stockid.eq(s)].itertuples():ref=(ref-a.dividend)/a.split
            ok=bool(obs.loc[date,s]) and np.isfinite(pr) and pr>0 and not(pr>=ref*1.295)
            n=math.floor(min(1e4,initial.loc[s,'adv20']*.01,cash/1.0005)/(pr*1.0005)) if ok else 0
            if n<100:n=0
            fee=n*pr*.0005 if n else 0.;value=n*pr if n else 0.
            q[s]=n;cash-=value+fee;cost+=fee
            buys.append({'stockid':s,'signal_date':str(start.date()),'buy_date':str(date.date()),'price':pr,'shares':n,'fee':fee,'target_budget':1e4,'reason':'filled' if n else 'unfilled'})
    value=float((q*cl.loc[date]).sum());equity=cash+value
    assert cash>=0 and np.isfinite(equity)
    rows.append({'date':date,'equity':equity,'exposure':value/equity,'cash':cash,'cost':cost,'dividend':div})
bh=pd.DataFrame(rows).set_index('date');bh.to_csv(B/'buy_hold_daily.csv');pd.DataFrame(buys).to_csv(B/'buy_hold_initial_orders.csv',index=False)
selected=pd.read_csv(H/'selected_daily.csv',parse_dates=['date']).set_index('date').loc[dates]
equal=pd.read_csv(H/'equal_pool_daily.csv',parse_dates=['date']).set_index('date').loc[dates]
idx=pd.read_csv(H.parent/'bj50_benchmark.csv',parse_dates=['date']).set_index('date').loc[dates]
fixed=pd.read_parquet(H/'fixed_daily.parquet');lg=fixed[fixed.strategy.eq('lgbm100__none')].set_index('date').loc[dates]
nav=pd.DataFrame({'主流程':selected.equity/1e6,'期初等权持有':bh.equity/1e6,'北证50价格':idx.equity/idx.equity.iloc[0],'动态等权':equal.equity/1e6,'纯LightGBM':lg.equity/1e6})
assert nav.notna().all().all() and (nav.iloc[0]==1).all()
nav.to_csv(B/'comparison_daily.csv')
def stats(n):
    v=n/n.iloc[0];ret=v.iloc[-1]-1;dd=(v/v.cummax()-1).min();r=v.pct_change().dropna()
    annual=v.iloc[-1]**(365.25/(n.index[-1]-n.index[0]).days)-1
    return {'return':ret,'max_drawdown':-dd,'annual_return':annual,'sharpe':(r.mean()-(1.015**(1/252)-1))/r.std(ddof=1)*np.sqrt(252),'end':v.iloc[-1]}
periods={}
for k,a,b in [('全期',start,end),('前一年',start,pd.Timestamp('2025-09-30')),('后一年',pd.Timestamp('2025-09-30'),end),('最近半年',pd.Timestamp('2026-03-31'),end)]:
    n=nav.loc[a:b];st={c:stats(n[c]) for c in n};st['超额']={c:st['主流程']['return']-st[c]['return'] for c in ['期初等权持有','北证50价格','动态等权','纯LightGBM']};periods[k]=st
quarter=[]
for k,g in nav.groupby(nav.index.to_period('Q')):
    prev=nav.loc[nav.index<g.index[0]];base=prev.iloc[-1] if len(prev) else nav.iloc[0]
    vals=g.iloc[-1]/base-1;quarter.append({'quarter':str(k),**vals.to_dict(),'超额':vals['主流程']-vals['期初等权持有']})
out={'periods':periods,'quarterly':quarter,'buy_hold':{'pool':100,'filled':sum(x['shares']>0 for x in buys),'initial_cost':bh.cost.sum(),'initial_exposure':bh.exposure.iloc[1],'mean_exposure':bh.exposure.mean(),'dividend_total':bh.dividend.sum(),'end_cash':bh.cash.iloc[-1]},'mean_exposure':{'主流程':selected.exposure.mean(),'期初等权持有':bh.exposure.mean(),'动态等权':equal.exposure.mean(),'纯LightGBM':lg.exposure.mean()},'days':len(dates)}
ev=json.loads((B.parent/'evidence.json').read_text(encoding='utf-8'));out['old_evidence']=ev
for name in ['02_2026Q2五天模型全部因子筛选过程.csv','03_20260401一百只股票最终评分.csv','05_训练表前五行含答案.csv','08_20260401训练和内部考试完整边界.csv']:
    df=pd.read_csv(R/'学习笔记/20261015_真实日期逐步解释'/name);print(name,list(df.columns));print(df.head(5).to_string(index=False))
out['curves']={'dates':[str(d.date()) for d in dates],'series':{c:nav[c].tolist() for c in nav}}
out['folds']=json.loads((H/'folds.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(R/'tools/research_20261011'))
from ledger import Market
reg=pd.read_csv(H.parent/'regime_probabilities.csv',parse_dates=['date']).set_index('date')
market=Market(p,act,reg)
schedule=pd.read_csv(H/'schedule.csv',parse_dates=['date']).set_index('date');schedule['topn']=100
equal_signals=p.loc[p.in_pool & p.date.between(start,end),['date','stockid']].copy();equal_signals['score']=0.
matched,matched_metrics,*_=market.run(equal_signals,horizon=5,start=start,end=end,schedule=schedule)
matched.to_csv(B/'same_policy_equal_daily.csv',index=False)
out['same_policy_equal']={**stats(matched.set_index('date').equity),'mean_exposure':matched.exposure.mean()}
out['curves']['series']['同仓位规则等权']=(matched.equity/1e6).tolist()
import joblib
models=joblib.load(H/'2026Q2_models.joblib');m=models['lgbm100']['model'];cols=models['lgbm100']['features']
panel=pd.read_parquet(D/'model_panel.parquet');cut=pd.Timestamp('2026-04-01');past=pd.DatetimeIndex(sorted(p.loc[p.date<cut,'date'].unique()))[-504:]
train=panel[panel.date.isin(past)&(panel.label_exit_5<cut)].dropna(subset=['target_5']).sort_values(['date','stockid'])
X=train[cols].fillna(0).to_numpy(dtype=np.float32);y=train.target_5.to_numpy();stages=[]
for k in [0,1,2,5,10,20,40,60,90,120,150,180]:
    pred=np.repeat(y.mean(),len(y)) if k==0 else m.predict(X,num_iteration=k)
    stages.append({'round':k,'sse':float(np.square(y-pred).sum()),'sample_prediction':float(pred[0]),'sample_error':float(y[0]-pred[0])})
out['training_stages']=stages
out['first_tree']=m.booster_.dump_model()['tree_info'][0]
out['training_shape']=[len(train),len(cols)]
sample=train[['date','stockid','x_c02_gtalpha046','x_c02_gtalpha063','target_5','label_5']].head(2).to_dict('records');out['train_samples']=sample
preds=pd.read_parquet(H/'2026Q2_predictions.parquet');preds=preds[preds.date.eq(cut)].sort_values('lgbm100',ascending=False);out['prediction_sample']=preds[['stockid','lgbm100','label_5','ensemble']].head(5).to_dict('records')
fac=pd.read_csv(R/'学习笔记/20261015_真实日期逐步解释/02_2026Q2五天模型全部因子筛选过程.csv');out['factor_example']=fac.head(5).fillna('').to_dict('records')
ic=pd.read_csv(R/'学习笔记/20261015_真实日期逐步解释/10_用于选因子的每一天IC.csv');out['factor_ic_curve']={'dates':ic.date.tolist(),'ic21':ic.code_IC.rolling(21).mean().fillna(0).tolist()}
(B/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k in ['periods','buy_hold','mean_exposure','days']},ensure_ascii=False,indent=2))
z=zipfile.ZipFile(R/'组会汇报_20261016/量化选股项目_老师意见修订_25页.pptx');ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
text=[]
for i in range(1,26):
    root=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'));t=[x.text for x in root.findall('.//a:t',ns) if x.text];text.append({'page':i,'text':t})
(B/'current_source_text.json').write_text(json.dumps(text,ensure_ascii=False,indent=2),encoding='utf-8')
