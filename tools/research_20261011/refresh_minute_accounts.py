"""Reprice cached minute-feature predictions with the current continuous ledger."""
import json
import pandas as pd
from common import DATA,RUN,dump
from ledger import Market

def main():
    p=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','amount','adv20','observed','vol20','mom20']);reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date');market=Market(p,pd.read_csv(DATA/'actions.csv'),reg)
    old=json.loads((RUN/'minute_factor_results.json').read_text(encoding='utf-8'));rows=[]
    for h in [1,5,20]:
        q=pd.read_parquet(RUN/f'minute_factor_h{h}_predictions.parquet')
        for policy in ['none','vol12','hmm3_volatility']:
            d,m,*_=market.run(q,horizon=h,topn=20,policy=policy)
            m.update(train_days=next(x['train_days'] for x in old if x['horizon']==h),label='2025Q4 minute-factor train, 2026Q3 evaluation; terminal holdings marked, small sample')
            rows.append(m);d.to_csv(RUN/f'minute_factor_h{h}_{policy}_daily.csv',index=False)
    dump(rows,RUN/'minute_factor_results.json')

if __name__=='__main__':main()
