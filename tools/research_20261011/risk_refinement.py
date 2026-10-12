"""Explicitly exploratory risk-budget scale study, after initial fixed comparisons.

All configurations retained. Does not overwrite the preregistered nested selector.
"""
import argparse
import numpy as np
import pandas as pd
from common import DATA,RUN,dump
from ledger import Market,metrics
from evaluate_models import block_interval

def main(horizons):
    p=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','amount','adv20','observed','vol20','mom20']);a=pd.read_csv(DATA/'actions.csv')
    reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date');market=Market(p,a,reg)
    dump({'status':'exploratory after seeing initial h5 fixed comparisons','models':['ic','conditional_ic','ridge60'],
          'policies':['hmm2_defensive','hmm3_defensive','hmm3_growth'],'risk_scales':[.25,.5,.75,1.],
          'guards':['none','drawdown','cppi'],'topn':20,'decision_rule':'among numeric passers prefer lowest drawdown, then highest Calmar; historical candidate only'},RUN/'risk_refinement_registry.json')
    for h in horizons:
        folder=RUN/f'h{h}';q=pd.read_parquet(folder/'predictions.parquet');rows=[];navs=[]
        for model in ['ic','conditional_ic','ridge60']:
            for policy in ['hmm2_defensive','hmm3_defensive','hmm3_growth']:
                for scale in [.25,.5,.75,1.]:
                    reg['cap_scaled']=reg['cap_'+policy]*scale
                    for guard in ['none','drawdown','cppi']:
                        d,m,*_=market.run(q,horizon=h,score=model,topn=20,policy='scaled',guard=guard)
                        m.update(model=model,base_policy=policy,risk_scale=scale,selection='exploratory historical risk refinement');rows.append(m)
                        d['strategy']=f'{model}__{policy}__{scale}__{guard}';navs.append(d)
            print('risk refined',h,model,flush=True)
        result=pd.DataFrame(rows);result.to_csv(folder/'risk_refinement.csv',index=False);pd.concat(navs).to_parquet(folder/'risk_refinement_daily.parquet',index=False)
        passing=result[result.pass_target]
        chosen=(passing.sort_values(['max_drawdown','calmar'],ascending=[True,False]).iloc[0] if len(passing) else result.sort_values('calmar',ascending=False).iloc[0]).to_dict()
        reg['cap_candidate']=reg['cap_'+chosen['base_policy']]*chosen['risk_scale']
        d,m,o,t,hold=market.run(q,horizon=h,score=chosen['model'],topn=20,policy='candidate',guard=chosen['guard'],save_details=True)
        m.update(model=chosen['model'],base_policy=chosen['base_policy'],risk_scale=chosen['risk_scale'],
                 selection='Chosen after inspecting historical comparisons; requires prospective frozen validation',
                 tested_refinement_configurations=len(result),passing_refinement_configurations=len(passing))
        dump(m,folder/'candidate_metrics.json');dump(block_interval(d),folder/'candidate_bootstrap.json')
        for name,frame in [('daily',d),('orders',o),('trades',t),('holdings',hold)]:frame.to_csv(folder/('candidate_'+name+'.csv'),index=False)
        scenarios=[]
        for fee in [0,2,5,15]:
            _,sens,*_=market.run(q,horizon=h,score=chosen['model'],topn=20,policy='candidate',guard=chosen['guard'],cost_bp=fee)
            sens['scenario']='same candidate, cost sensitivity';scenarios.append(sens)
        _,sens,*_=market.run(q,horizon=h,score=chosen['model'],topn=20,policy='candidate',guard=chosen['guard'],actions=False)
        sens['scenario']='same candidate, raw prices only';scenarios.append(sens)
        # Entire ranking is delayed by one session, using calendar mapping, as a conservative availability stress.
        delayed=q.copy();calendar=market.dates;mapping={calendar[i]:calendar[i+1] for i in range(len(calendar)-1)};delayed['date']=delayed.date.map(mapping);delayed=delayed.dropna(subset=['date'])
        _,sens,*_=market.run(delayed,horizon=h,score=chosen['model'],topn=20,policy='candidate',guard=chosen['guard'],start=q.date.min(),end=q.date.max())
        sens['scenario']='entire stock ranking delayed one session; risk observations current';scenarios.append(sens)
        dump(scenarios,folder/'candidate_sensitivity.json')
        print('candidate',h,m['model'],m['base_policy'],m['risk_scale'],m['guard'],'Sharpe',m['sharpe_rf15'],'Calmar',m['calmar'],'MDD',m['max_drawdown'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--horizons',default='1,2,5,10,20');args=parser.parse_args();main([int(x) for x in args.horizons.split(',')])
