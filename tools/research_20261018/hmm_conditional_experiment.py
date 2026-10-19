"""Past-only monthly conditional factor weights, compared with identical global factors."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from adaptive_execution import CohortMarket,Rule,period_metrics
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/research_20261017'))
from models import screen,daily_factor_ic
OUT=ROOT/'training_runs/research_20261018/hmm_conditional'

def weights(mean):
    return mean/max(np.abs(mean).sum(),1e-12)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'window_sessions':504,'factor_count':20,'refit':'monthly','states':[2,3],
              'shrink_state_mean_toward_global_days':60,'period':['2024-10-08','2026-09-30'],
              'labels_mature_strictly_before_month_start':True,'test_used_to_select_hyperparameters':False,
              'history_previously_inspected':True,'probabilities':'existing monthly past-only forward filter then next-step prediction; states ordered by training volatility',
              'account_rules':['equal','fixed50','hmm3_gate'],'cost_per_side':.0005}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    p=pd.read_parquet(ROOT/'training_data/research_20261011/model_panel.parquet').sort_values(['date','stockid'])
    reg=pd.read_csv(ROOT/'training_runs/research_20261011/regime_probabilities.csv',parse_dates=['date']).set_index('date')
    features=[c for c in p if c.startswith('x_')]
    outer=p[p.date.between('2024-10-08','2026-09-30')]
    all_pred=[];diagnostics=[];factor_rows=[]
    for month,block in outer.groupby(outer.date.dt.to_period('M')):
        cutoff=pd.Timestamp(month.start_time)
        dates=sorted(p.loc[p.date<cutoff,'date'].unique())[-504:]
        train=p[p.date.isin(dates)&(p.label_exit_5<cutoff)&p.label_5.notna()].copy()
        train['target_now']=train.groupby('date').label_5.rank(pct=True)-.5
        selected,card=screen(train,features,None,maximum=20)
        d,ic=daily_factor_ic(train,selected)
        global_mean=np.nanmean(ic,axis=0);global_weight=weights(global_mean)
        pred=block[['date','stockid']].copy();x=block[selected].fillna(0).to_numpy()
        pred['score_global']=x@global_weight
        for k in [2,3]:
            cols=[f'hmm{k}_p_{j}' for j in range(k)]
            old=reg[cols].reindex(pd.DatetimeIndex(d)).to_numpy(float)
            known=np.isfinite(old).all(axis=1)
            w=np.where(known[:,None],old,0)
            cond=[]
            for j in range(k):
                mask=np.isfinite(ic)
                mass=(w[:,j,None]*mask).sum(axis=0)
                mean=(np.nansum(ic*w[:,j,None],axis=0)+60*global_mean)/(mass+60)
                cond.append(weights(mean))
                for f,mu,wt in zip(selected,mean,cond[-1]):
                    factor_rows.append({'month':str(month),'states':k,'state_order_low_to_high_vol':j,
                                        'factor':f,'signed_mean_ic':mu,'signed_weight':wt})
            now=reg[cols].reindex(block.date).to_numpy(float)
            available=np.isfinite(now).all(axis=1)
            effective=np.nan_to_num(now)@np.array(cond)
            effective[~available]=global_weight
            pred[f'score_hmm{k}']=(x*effective).sum(axis=1)
            diagnostics.append({'month':str(month),'states':k,'train_rows':len(train),'train_days':len(d),
                                'train_first_signal':str(train.date.min().date()),'train_last_signal':str(train.date.max().date()),
                                'train_last_label_exit':str(train.label_exit_5.max().date()),'first_score':str(block.date.min().date()),
                                'probability_train_days':int(known.sum()),'fallback_score_rows':int((~available).sum()),'selected_factors':selected})
        all_pred.append(pred);print('conditional',str(month),flush=True)
    preds=pd.concat(all_pred,ignore_index=True);preds.to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(factor_rows).to_csv(OUT/'conditional_factor_weights.csv',index=False,encoding='utf-8-sig')
    (OUT/'monthly_training_audit.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    prices=pd.read_parquet(ROOT/'training_data/research_20261011/prices.parquet',columns=['date','stockid','open','close','adv20','observed','mom20','vol20','in_pool','adj_ret1'])
    actions=pd.read_csv(ROOT/'training_data/research_20261011/actions.csv')
    market=CohortMarket(prices,actions,hmm_caps=reg.cap_hmm3_defensive)
    summaries=[];periods=[];alldaily=[]
    for name in ['global','hmm2','hmm3']:
        scores=preds[['date','stockid','score_'+name]].rename(columns={'score_'+name:'score'})
        for rule in [Rule('equal'),Rule('fixed50',allocation=.5),Rule('hmm3_gate',market_gate='hmm')]:
            strategy=name+'__'+rule.name;result=market.run(scores,rule,save_details=True)
            folder=OUT/'accounts'/strategy;folder.mkdir(parents=True,exist_ok=True)
            for key in ['daily','orders','trades','holdings']:result[key].to_csv(folder/(key+'.csv'),index=False,encoding='utf-8-sig')
            d=result['daily'].assign(strategy=strategy);alldaily.append(d)
            summaries.append({'strategy':strategy,**result['metrics']})
            for period,a,b in [('full','2024-10-08','2026-09-30'),('recent_year','2025-10-01','2026-09-30'),('recent_half','2026-04-01','2026-09-30')]:
                periods.append({'strategy':strategy,'period':period,**period_metrics(d,a,b)})
            print(strategy,result['metrics']['total_return'],flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'account_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(periods).to_csv(OUT/'period_results.csv',index=False,encoding='utf-8-sig')
    pd.concat(alldaily).to_parquet(OUT/'account_daily.parquet',index=False)
    print('HMM_CONDITIONAL_COMPLETE',flush=True)

if __name__=='__main__':main()
