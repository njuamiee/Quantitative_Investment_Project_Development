"""Registered common-account comparisons, validation-only risk selection."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,hashlib
import numpy as np
import pandas as pd
from execution import CohortMarket,Rule,RULES,CASH,performance,period_metrics

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261017'
DATA=ROOT/'training_data/research_20261011'
START='2024-10-08';END='2026-09-30'


def dump(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def save_account(name,result):
    p=RUN/'accounts'/name;p.mkdir(parents=True,exist_ok=True)
    for key,frame in result.items():
        if isinstance(frame,pd.DataFrame):frame.to_csv(p/(key+'.csv'),index=False,encoding='utf-8-sig')
    dump(result['metrics'],p/'metrics.json')


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    prices=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','adv20','observed','mom20','vol20','in_pool','adj_ret1'])
    actions=pd.read_csv(DATA/'actions.csv');metadata=pd.read_parquet(RUN/'risk_metadata.parquet')
    market=CohortMarket(prices,actions,metadata)
    predictions=pd.read_parquet(RUN/'models/daily_predictions.parquet')
    folds=pd.read_csv(RUN/'models/folds.csv')
    riskgrid=[];schedule={};choices=[]
    for f in folds.itertuples(index=False):
        valid=pd.read_parquet(RUN/'models'/f.quarter/'valid_predictions.parquet')[['date','stockid','score']]
        rows=[]
        for rule in RULES:
            r=market.run(valid,rule,start=f.valid_start,end=f.valid_end)
            m=r['metrics'];cal=m['calmar'];objective=min(m['sharpe_rf15'],cal/2) if cal is not None else -1e9
            rows.append(dict(quarter=f.quarter,rule=rule.name,objective=objective,**{k:v for k,v in m.items() if k not in ['rule','schedule']}))
        viable=[x for x in rows if x['total_return']>0 and x['completed_sale_parts']>=5]
        chosen=max(viable,key=lambda x:(x['objective'],x['annual_return'],x['rule'])) if viable else None
        pick=next(r for r in RULES if r.name==chosen['rule']) if chosen else CASH
        schedule[f.quarter]=pick
        choices.append(dict(quarter=f.quarter,valid_start=f.valid_start,valid_end=f.valid_end,
                            selected_rule=pick.name,validation_objective=chosen['objective'] if chosen else None,
                            reason='positive_validation_return_max_min_sharpe_calmar' if chosen else 'no_positive_validation_candidate_cash',
                            model=f.selected_name))
        riskgrid.extend(rows)
        print('risk choice',f.quarter,pick.name,flush=True)
    pd.DataFrame(riskgrid).to_csv(RUN/'risk_validation_grid.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(choices).to_csv(RUN/'risk_choices.csv',index=False,encoding='utf-8-sig')
    dump({k:asdict(v) for k,v in schedule.items()},RUN/'risk_schedule.json')
    def scores(col):return predictions[['date','stockid',col]].rename(columns={col:'score'})
    accounts={}
    for name,col,rule,selected in [
        ('new_main_monthly','score_monthly',Rule(),True),
        ('new_quarterly_same_risk','score_quarterly',Rule(),True),
        ('new_monthly_equal','score_monthly',Rule(),False),
        ('new_quarterly_equal','score_quarterly',Rule(),False),
        ('new_monthly_stop10','score_monthly',RULES[1],False),
        ('new_monthly_diversified','score_monthly',RULES[3],False),
        ('simple_ic20_equal','score_simple',Rule(),False),
        ('simple_ic20_same_risk','score_simple',Rule(),True),
    ]:
        r=market.run(scores(col),rule,start=START,end=END,schedule=schedule if selected else None,save_details=True)
        accounts[name]=r;save_account(name,r);print(name,round(r['metrics']['total_return'],4),round(r['metrics']['max_drawdown'],4),flush=True)
    pool=prices.loc[prices.in_pool & prices.date.between(START,END),['date','stockid']].copy();pool['score']=0.
    for name,rule in [('equal_pool_5day',Rule('equal_pool_5day',topn=100)),('equal_buy_hold',Rule('equal_buy_hold',topn=100,buy_hold=True))]:
        r=market.run(pool,rule,start=START,end=END,save_details=True);accounts[name]=r;save_account(name,r)
    old=pd.read_parquet(ROOT/'training_runs/research_20261011/h5/predictions.parquet')
    r=market.run(old[['date','stockid','lgbm100']].rename(columns={'lgbm100':'score'}),Rule(),start=START,end=END,save_details=True)
    accounts['old_lgbm_same_execution']=r;save_account('old_lgbm_same_execution',r)
    olddaily=pd.read_csv(ROOT/'training_runs/research_20261011/h5/selected_daily.csv',parse_dates=['date'])
    olddaily=olddaily[olddaily.date.between(START,END)].copy()
    accounts['old_main']={'daily':olddaily,'metrics':performance(olddaily)}
    benchmark=pd.read_csv(ROOT/'training_runs/research_20261011/bj50_benchmark.csv',parse_dates=['date'])
    benchmark=benchmark[benchmark.date.between(START,END)].copy();benchmark.equity=benchmark.equity/benchmark.equity.iloc[0]*1e6
    accounts['bj50_price_index']={'daily':benchmark,'metrics':performance(benchmark)}
    summary=[];periods=[];daily=[];quarterrows=[]
    for name,r in accounts.items():
        summary.append(dict(strategy=name,**r['metrics']))
        d=r['daily'].copy();d['strategy']=name;daily.append(d)
        for label,a,b in [('full',START,END),('recent_year','2025-10-01',END),('recent_half','2026-04-01',END)]:
            periods.append(dict(strategy=name,period=label,**period_metrics(r['daily'],a,b)))
        for quarter in sorted(r['daily'].date.dt.to_period('Q').unique()):
            q=r['daily'][r['daily'].date.dt.to_period('Q')==quarter]
            quarterrows.append(dict(strategy=name,quarter=str(quarter),**period_metrics(r['daily'],q.date.min(),q.date.max())))
    pd.DataFrame(summary).to_csv(RUN/'strategy_summary.csv',index=False,encoding='utf-8-sig')
    period=pd.DataFrame(periods)
    for reference in ['equal_buy_hold','equal_pool_5day','bj50_price_index','old_main']:
        baseline=period[period.strategy==reference].set_index('period').total_return
        period['excess_vs_'+reference]=period.total_return-period.period.map(baseline)
    period.to_csv(RUN/'period_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(quarterrows).to_csv(RUN/'quarter_comparison.csv',index=False,encoding='utf-8-sig')
    pd.concat(daily,ignore_index=True).to_parquet(RUN/'all_daily.parquet',index=False)
    # Material implementation stress: unchanged historical choices, higher fees
    # or one-session delayed ranking. These are diagnostics, never winner selection.
    stress=[]
    for fee,delay in [(0.,0),(5.,0),(15.,0),(5.,1)]:
        r=market.run(scores('score_monthly'),Rule(),start=START,end=END,schedule=schedule,cost_bp=fee,signal_delay=delay)
        stress.append(dict(scenario=f'cost{fee:g}_delay{delay}',**r['metrics']))
    pd.DataFrame(stress).to_csv(RUN/'stress_tests.csv',index=False,encoding='utf-8-sig')
    mainaccount=accounts['new_main_monthly']
    t=mainaccount['trades']
    if len(t):
        t.groupby('stockid').agg(realized_pnl=('pnl','sum'),sale_parts=('pnl','size'),mean_intervals=('holding_session_intervals','mean')).sort_values('realized_pnl').to_csv(RUN/'realized_stock_pnl.csv',encoding='utf-8-sig')
        t.groupby(['holding_session_intervals','reason']).size().rename('sale_parts').to_csv(RUN/'holding_distribution.csv',encoding='utf-8-sig')
    o=mainaccount['orders']
    if len(o):o.fillna({'reject_reason':'filled'}).groupby(['side','reject_reason']).agg(orders=('date','size'),requested_shares=('requested_shares','sum'),filled_shares=('filled_shares','sum')).to_csv(RUN/'order_diagnosis.csv',encoding='utf-8-sig')
    manifest={'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'models/daily_predictions.parquet',RUN/'risk_metadata.parquet']},
              'period':[START,END],'accounts':list(accounts),'rules':[asdict(r) for r in RULES],
              'warning':'retrospective inspected history; no real trades; corporate-action and price-limit proxies documented'}
    dump(manifest,RUN/'run_manifest.json')
    print('COMPLETE',len(accounts),'accounts',flush=True)


if __name__=='__main__':main()
