"""Forecast length and execution holding period are distinct design choices.
Exploratory extension for longer forecasts; every tested configuration is retained.
"""
import argparse,json,shutil
import numpy as np
import pandas as pd
from common import DATA,RUN,dump
from ledger import Market
from evaluate_models import block_interval

def main(horizons):
    p=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','amount','adv20','observed','vol20','mom20']);reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date');market=Market(p,pd.read_csv(DATA/'actions.csv'),reg)
    dump({'status':'exploratory extension after aligned 20-day candidate failed Calmar','forecast_horizons':horizons,
          'models':['ic','conditional_ic','ridge60'],'topn':[10,20],'holding_periods':[5,10,20],
          'policies':['hmm2_defensive','hmm3_defensive','hmm3_growth'],'risk_scales':[.25,.5],
          'selection':'lowest drawdown among numeric passers; no assertion of untouched holdout'},RUN/'portfolio_extension_registry.json')
    for h in horizons:
        folder=RUN/f'h{h}';q=pd.read_parquet(folder/'predictions.parquet');rows=[]
        for model in ['ic','conditional_ic','ridge60']:
            for top in [10,20]:
                for holding in [5,10,20]:
                    for policy in ['hmm2_defensive','hmm3_defensive','hmm3_growth']:
                        for scale in [.25,.5]:
                            reg['cap_extension']=reg['cap_'+policy]*scale
                            _,m,*_=market.run(q,horizon=holding,score=model,topn=top,policy='extension')
                            m.update(forecast_horizon=h,holding_period=holding,model=model,base_policy=policy,risk_scale=scale);rows.append(m)
            print('portfolio extension',h,model,flush=True)
        results=pd.DataFrame(rows);results.to_csv(folder/'portfolio_extension.csv',index=False)
        previous=json.loads((folder/'candidate_metrics.json').read_text(encoding='utf-8'))
        passers=results[results.pass_target];chosen=passers.sort_values(['max_drawdown','calmar'],ascending=[True,False]).iloc[0].to_dict() if len(passers) else None
        if chosen is None or (previous['pass_target'] and previous['max_drawdown']<=chosen['max_drawdown']):
            print('extension retains aligned candidate',h,'passers',len(passers),flush=True);continue
        archive=folder/'aligned_holding';archive.mkdir(exist_ok=True)
        for file in folder.glob('candidate_*'):
            if file.is_file():shutil.copy2(file,archive/file.name)
        reg['cap_candidate']=reg['cap_'+chosen['base_policy']]*chosen['risk_scale']
        d,m,o,t,hold=market.run(q,horizon=chosen['holding_period'],score=chosen['model'],topn=int(chosen['topn']),policy='candidate',save_details=True)
        m.update(horizon=h,forecast_horizon=h,holding_period=chosen['holding_period'],model=chosen['model'],base_policy=chosen['base_policy'],risk_scale=chosen['risk_scale'],
                 selection='Exploratory historical selection with forecast and execution periods separated; prospective validation pending',
                 tested_refinement_configurations=previous['tested_refinement_configurations']+len(results),passing_extension_configurations=len(passers))
        dump(m,folder/'candidate_metrics.json');dump(block_interval(d),folder/'candidate_bootstrap.json')
        for name,frame in [('daily',d),('orders',o),('trades',t),('holdings',hold)]:frame.to_csv(folder/('candidate_'+name+'.csv'),index=False)
        scenarios=[]
        for fee in [0,2,5,15]:
            _,x,*_=market.run(q,horizon=chosen['holding_period'],score=chosen['model'],topn=int(chosen['topn']),policy='candidate',cost_bp=fee)
            x.update(forecast_horizon=h,holding_period=chosen['holding_period'],scenario='same candidate, cost sensitivity');scenarios.append(x)
        _,x,*_=market.run(q,horizon=chosen['holding_period'],score=chosen['model'],topn=int(chosen['topn']),policy='candidate',actions=False)
        x['scenario']='same candidate, raw prices only';scenarios.append(x)
        delayed=q.copy();cal=market.dates;mapping={cal[i]:cal[i+1] for i in range(len(cal)-1)};delayed.date=delayed.date.map(mapping);delayed=delayed.dropna(subset=['date'])
        _,x,*_=market.run(delayed,horizon=chosen['holding_period'],score=chosen['model'],topn=int(chosen['topn']),policy='candidate',start=q.date.min(),end=q.date.max())
        x['scenario']='entire ranking delayed one session';scenarios.append(x);dump(scenarios,folder/'candidate_sensitivity.json')
        print('extended candidate',h,'holding',chosen['holding_period'],m['sharpe_rf15'],m['calmar'],m['max_drawdown'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--horizons',default='20,10');args=parser.parse_args();main([int(x) for x in args.horizons.split(',')])
