"""Predetermined causal exit/entry comparisons on frozen BJ model forecasts."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from adaptive_execution import CohortMarket,Rule,performance,period_metrics

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'training_runs/research_20261017'
OUT=ROOT/'training_runs/research_20261018/deep_risk'
RULES=[Rule('equal'),Rule('allocation50',allocation=.5),Rule('allocation75',allocation=.75),
       Rule('stop06',stop=.06),Rule('stop10',stop=.10),
       Rule('trend20',market_gate='trend20'),Rule('trend60',market_gate='trend60'),
       Rule('breadth',market_gate='breadth'),Rule('hmm3_gate',market_gate='hmm'),
       Rule('vol10',volatility_target=.10),Rule('vol12',volatility_target=.12),
       Rule('breadth_vol12_stop06',market_gate='breadth',volatility_target=.12,stop=.06),
       Rule('trend20_vol12_stop10',market_gate='trend20',volatility_target=.12,stop=.10),
       Rule('hmm3_vol12_stop10',market_gate='hmm',volatility_target=.12,stop=.10),
       Rule('circuit08',circuit_drawdown=.08),Rule('circuit12',circuit_drawdown=.12),
       Rule('breadth_circuit08_vol12',market_gate='breadth',circuit_drawdown=.08,volatility_target=.12),
       Rule('allocation50_circuit08',allocation=.5,circuit_drawdown=.08),
       Rule('allocation75_stop06',allocation=.75,stop=.06)]
CASH=Rule('cash',allocation=0.)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from dataclasses import asdict
    design={'period':['2024-10-08','2026-09-30'],'history_previously_inspected':True,
            'rules':[asdict(r) for r in RULES],'model_streams':['quarterly','monthly','simple','quarterly_simple_50_50'],
            'selection':'only previous 126-session validation; reject nonpositive return, fewer than5 sales, MDD>20%; maximize min(Sharpe_rf15,Calmar/2); else cash',
            'cost_per_side':.0005,'evaluation_targets':{'max_drawdown_lte':.20,'sharpe_gt':1.,'calmar_gt':2.}}
    (OUT/'design.json').write_text(json.dumps(design,ensure_ascii=False,indent=2),encoding='utf-8')
    prices=pd.read_parquet(ROOT/'training_data/research_20261011/prices.parquet',columns=['date','stockid','open','close','adv20','observed','mom20','vol20','in_pool','adj_ret1'])
    actions=pd.read_csv(ROOT/'training_data/research_20261011/actions.csv')
    regime=pd.read_csv(ROOT/'training_runs/research_20261011/regime_probabilities.csv',parse_dates=['date']).set_index('date')
    market=CohortMarket(prices,actions,hmm_caps=regime.cap_hmm3_defensive)
    predictions=pd.read_parquet(OLD/'models/daily_predictions.parquet')
    predictions['score_blend']=predictions[['score_quarterly','score_simple']].groupby(predictions.date).rank(pct=True).mean(axis=1)-.5
    folds=pd.read_csv(OLD/'models/folds.csv')
    grid=[];schedules={};choices=[]
    for model in ['quarterly','simple','blend']:
        schedule={}
        for f in folds.itertuples(index=False):
            valid=pd.read_parquet(OLD/'models'/f.quarter/'valid_predictions.parquet')[['date','stockid','score','score_simple']].copy()
            if model=='simple':valid['score']=valid.score_simple
            elif model=='blend':valid['score']=valid[['score','score_simple']].groupby(valid.date).rank(pct=True).mean(axis=1)-.5
            candidates=[]
            for rule in RULES:
                result=market.run(valid[['date','stockid','score']],rule,start=f.valid_start,end=f.valid_end)
                m=result['metrics'];objective=min(m['sharpe_rf15'],m['calmar']/2) if m['calmar'] is not None else -1e9
                record={'model':model,'quarter':f.quarter,'objective':objective,**m};grid.append(record)
                if m['total_return']>0 and m['completed_sale_parts']>=5 and m['max_drawdown']<=.20:
                    candidates.append((objective,m['annual_return'],rule.name,rule))
            pick=max(candidates,key=lambda x:x[:3])[-1] if candidates else CASH
            schedule[f.quarter]=pick;choices.append({'model':model,'quarter':f.quarter,'selected':pick.name,'qualifying':len(candidates)})
            print('selected',model,f.quarter,pick.name,flush=True)
        schedules[model]=schedule
    pd.DataFrame(grid).to_csv(OUT/'validation_all_candidates.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(choices).to_csv(OUT/'validation_choices.csv',index=False,encoding='utf-8-sig')
    summaries=[];periods=[];all_daily=[]
    def save(name,result):
        m=result['metrics'];summaries.append({'strategy':name,**m})
        d=result['daily'].copy();d['strategy']=name;all_daily.append(d)
        for label,a,b in [('full','2024-10-08','2026-09-30'),('recent_year','2025-10-01','2026-09-30'),('recent_half','2026-04-01','2026-09-30')]:
            p=period_metrics(d,a,b);p['meets_targets']=bool(p['max_drawdown']<=.20 and p['sharpe_rf15']>1 and p['calmar'] is not None and p['calmar']>2)
            periods.append({'strategy':name,'period':label,**p})
        if len(result['orders']):
            folder=OUT/'accounts'/name;folder.mkdir(parents=True,exist_ok=True)
            for k in ['daily','orders','trades','holdings']:result[k].to_csv(folder/f'{k}.csv',index=False,encoding='utf-8-sig')
        print(name,round(m['total_return'],4),round(m['max_drawdown'],4),round(m['sharpe_rf15'],3),flush=True)
    for model,col in [('quarterly','score_quarterly'),('monthly','score_monthly'),('simple','score_simple'),('blend','score_blend')]:
        scores=predictions[['date','stockid',col]].rename(columns={col:'score'})
        for rule in RULES:
            result=market.run(scores,rule,save_details=True)
            save(model+'__'+rule.name,result)
        schedule=schedules['quarterly' if model=='monthly' else model]
        save(model+'__past_validation_selected',market.run(scores,Rule(),schedule=schedule,save_details=True))
    pd.DataFrame(summaries).to_csv(OUT/'all_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(periods).to_csv(OUT/'all_period_results.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_daily).to_parquet(OUT/'all_daily.parquet',index=False)
    (OUT/'input_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'models/daily_predictions.parquet',ROOT/'training_runs/research_20261011/regime_probabilities.csv']},indent=2),encoding='utf-8')
    print('DEEP_RISK_COMPLETE',len(summaries),flush=True)

if __name__=='__main__':main()
