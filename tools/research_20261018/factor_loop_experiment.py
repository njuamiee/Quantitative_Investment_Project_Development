"""Two bounded AI proposal rounds -> frozen factors -> controlled account test."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from auto_factor_lab import Protocol,split_mature,inspect_expression,_score,daily_rank_ic,allowed_inputs
from adaptive_execution import CohortMarket,Rule,period_metrics
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/research_20261017'))
from models import screen,lgbm_params,rank_ic_daily

OUT=ROOT/'training_runs/research_20261018/factor_loop'
PANEL=ROOT/'training_data/research_20261011/model_panel.parquet'
PROTOCOL=Protocol('2023-01-01','2025-07-01','2026-01-01','2026-10-01')

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def engineered(frame,formulas):
    out=frame.copy()
    for r in formulas:
        parents,_=inspect_expression(r['expression'],allowed_inputs(out.columns))
        value=_score(r['expression'],out,parents)*int(r['orientation_from_train'])
        out['x_new_'+r['id']]=value.groupby(out.date).rank(pct=True)-.5
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # This read cannot expose the independent period during formula selection.
    past=pd.read_parquet(PANEL,filters=[('date','<',pd.Timestamp(PROTOCOL.test_start))])
    past=past.sort_values(['date','stockid']).reset_index(drop=True)
    train,valid,_=split_mature(past,PROTOCOL)
    cards=[]
    for round_no in [1,2]:
        c=pd.read_csv(OUT/f'round{round_no}/screening.csv');c['round']=round_no;cards.append(c)
    cards=pd.concat(cards,ignore_index=True)
    eligible=cards[cards.accepted_before_independent_test].sort_values(
        ['valid_rank_ic_after_train_orientation','id'],ascending=[False,True])
    chosen=[];values=[];decisions=[]
    for r in eligible.to_dict('records'):
        parents,_=inspect_expression(r['expression'],allowed_inputs(train.columns))
        s=_score(r['expression'],train,parents)
        corr=max([abs(s.corr(v)) for v in values],default=0.)
        reason='selected'
        if len(chosen)>=3:reason='after_maximum_3'
        elif corr>=.92:reason='abs_corr_ge_0.92_with_already_selected'
        else:chosen.append(r);values.append(s)
        decisions.append({'id':r['id'],'max_corr_to_selected':corr,'reason':reason})
    pd.DataFrame(decisions).to_csv(OUT/'deduplication.csv',index=False,encoding='utf-8-sig')
    # Reject old AI inputs in BOTH arms so this is genuinely incremental.
    base=[c for c in past.columns if c.startswith('x_') and not c.startswith('x_llm_')]
    train['target_now']=train.groupby('date').label_5.rank(pct=True)-.5
    valid['target_now']=valid.groupby('date').label_5.rank(pct=True)-.5
    features,card=screen(train,base,None,maximum=100)
    card.to_csv(OUT/'baseline_feature_screen.csv',index=False,encoding='utf-8-sig')
    baseline=lgb.LGBMRegressor(**lgbm_params(300))
    baseline.fit(train[features].fillna(0),train.target_now,
                 eval_set=[(valid[features].fillna(0),valid.target_now)],
                 callbacks=[lgb.early_stopping(30,verbose=False)])
    rounds=int(baseline.best_iteration_ or baseline.n_estimators_)
    valid_ic,_=rank_ic_daily(valid,baseline.predict(valid[features].fillna(0)))
    formal=past[(past.date>=pd.Timestamp(PROTOCOL.train_start)) &
                (past.label_exit_5<pd.Timestamp(PROTOCOL.test_start)) & past.label_5.notna()].copy()
    last_dates=sorted(formal.date.unique())[-504:]
    formal=formal[formal.date.isin(last_dates)].copy()
    formal['target_now']=formal.groupby('date').label_5.rank(pct=True)-.5
    freeze={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'proposal_total':len(cards),
            'proposal_rounds':2,'accepted_before_dedup':len(eligible),'chosen':chosen,
            'features':features,'shared_model_rounds':rounds,'baseline_valid_rank_ic':valid_ic,
            'train_selection_last_exit':train.label_exit_5.max(),'valid_selection_last_exit':valid.label_exit_5.max(),
            'formal_train_first_signal':formal.date.min(),'formal_train_last_signal':formal.date.max(),
            'formal_train_last_exit':formal.label_exit_5.max(),'formal_train_rows':len(formal),
            'test_start':PROTOCOL.test_start,'test_end_exclusive':PROTOCOL.test_end,
            'history_already_used_for_research':True,'independent_rows_loaded_before_freeze':False,
            'selection_uses_test':False,'separate_local_llm_called':False,
            'source_proposals_sha256':{str(i):hashlib.sha256((ROOT/f'tools/research_20261018/loop_round{i}.json').read_bytes()).hexdigest() for i in [1,2]}}
    dump('frozen_selection.json',freeze)
    frozen_sha=hashlib.sha256((OUT/'frozen_selection.json').read_bytes()).hexdigest()
    # No more selection from this point. Load future INPUTS and report answers
    # only after a fixed selection has been persisted above.
    test=pd.read_parquet(PANEL,filters=[('date','>=',pd.Timestamp(PROTOCOL.test_start)),('date','<',pd.Timestamp(PROTOCOL.test_end))])
    formal=engineered(formal,chosen);test=engineered(test,chosen)
    mature_test=test[(test.label_exit_5<pd.Timestamp(PROTOCOL.test_end)) & test.label_5.notna()].copy()
    mature_test['target_now']=mature_test.groupby('date').label_5.rank(pct=True)-.5
    factor_test=[]
    for r in chosen:
        d=daily_rank_ic(mature_test,mature_test['x_new_'+r['id']],10)
        factor_test.append({'id':r['id'],'test_rank_ic':d.rank_ic.mean(),'days':len(d)})
    pd.DataFrame(factor_test).to_csv(OUT/'frozen_factors_test.csv',index=False,encoding='utf-8-sig')
    preds=test[['date','stockid']].copy();model_cards=[]
    for name,cols in [('baseline',features),('augmented',features+['x_new_'+r['id'] for r in chosen])]:
        model=lgb.LGBMRegressor(**lgbm_params(rounds))
        model.fit(formal[cols].fillna(0),formal.target_now)
        preds['score_'+name]=model.predict(test[cols].fillna(0))
        ic,d=rank_ic_daily(mature_test,model.predict(mature_test[cols].fillna(0)))
        model_cards.append({'model':name,'features':len(cols),'rounds':rounds,'test_rank_ic':ic})
        d.to_csv(OUT/(name+'_daily_ic.csv'),index=False,encoding='utf-8-sig')
        joblib.dump({'model':model,'factors':cols},OUT/(name+'.joblib'))
    preds.to_parquet(OUT/'test_predictions.parquet',index=False)
    pd.DataFrame(model_cards).to_csv(OUT/'model_comparison.csv',index=False,encoding='utf-8-sig')
    prices=pd.read_parquet(ROOT/'training_data/research_20261011/prices.parquet',columns=['date','stockid','open','close','adv20','observed','mom20','vol20','in_pool','adj_ret1'])
    actions=pd.read_csv(ROOT/'training_data/research_20261011/actions.csv')
    market=CohortMarket(prices,actions)
    summaries=[];daily=[]
    a=str(test.date.min().date());b=str(test.date.max().date())
    for name in ['baseline','augmented']:
        scores=preds[['date','stockid','score_'+name]].rename(columns={'score_'+name:'score'})
        for rule in [Rule('equal'),Rule('breadth',market_gate='breadth')]:
            result=market.run(scores,rule,start=a,end=b,save_details=True)
            strategy=name+'__'+rule.name
            folder=OUT/'accounts'/strategy;folder.mkdir(parents=True,exist_ok=True)
            for k in ['daily','orders','trades','holdings']:result[k].to_csv(folder/(k+'.csv'),index=False,encoding='utf-8-sig')
            d=result['daily'].assign(strategy=strategy);daily.append(d)
            summaries.append({'strategy':strategy,**result['metrics']})
            print(strategy,result['metrics']['total_return'],flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'account_results.csv',index=False,encoding='utf-8-sig')
    pd.concat(daily).to_parquet(OUT/'account_daily.parquet',index=False)
    dump('completion.json',{'selection_sha256':frozen_sha,'formula_count':len(chosen),'evaluation_start':a,'evaluation_end':b,
                            'test_rows':len(test),'mature_test_rows':len(mature_test),'model_comparison':model_cards,
                            'same_rounds_and_inputs_except_added_factors':True,'third_round_after_test':False})
    print('FACTOR_LOOP_COMPLETE',flush=True)

if __name__=='__main__':main()
