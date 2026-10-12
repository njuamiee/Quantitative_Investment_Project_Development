"""Rerun existing SH/SZ and minute execution experiments at 5bp, retaining their limits."""
import sys
import pandas as pd
from common import ROOT,DATA,RUN,dump
sys.path.insert(0,str(ROOT/'tools/research_20260930'))
import accounts
sys.path.insert(0,str(ROOT/'tools/research_20261008'))
from minute_account import minute_simulate

def main():
    source=ROOT/'training_data/research_20261008/shsz_alpha'
    p=pd.read_parquet(source/'shsz_prices.parquet');q=pd.read_parquet(ROOT/'training_runs/teacher9_20261008/shsz_alpha/shsz_predictions.parquet')
    result=[]
    for score in ['pred_price','pred_selected','pred_ridge']:
        d,_,_,m=accounts.simulate(q,p,score=score,cost_bp=5);m['model']=score
        m['scope']='Existing 31-quarter SH/SZ experiment rerun at 5bp; historical 496-symbol sample; original ledger constraints; price-only'
        result.append(m);d.to_csv(RUN/('shsz_'+score+'_daily.csv'),index=False)
    dump(result,RUN/'shsz_cost5_results.json')
    scored=pd.read_parquet(RUN/'h5/predictions.parquet')
    prices=pd.read_parquet(DATA/'prices.parquet');start,end=pd.Timestamp('2026-06-30'),pd.Timestamp('2026-09-29')
    p=prices[prices.date.between(start,end)].copy();q=scored[scored.date.between(start,end)].copy();q['pred_selected']=q.ic
    q=q.merge(p[['date','stockid','adv20']],on=['date','stockid'],validate='one_to_one')
    result=[]
    for participation in [.005,.01,.02]:
        d,f,o,m=minute_simulate(q,p,participation=participation,cost_bp=5)
        m['scope']='IC stock ranking, original top5 execution stress ledger, 2026Q3; no HMM risk scale; not candidate NAV'
        result.append(m);label=str(participation).replace('.','p')
        d.to_csv(RUN/f'minute_execution_{label}_daily.csv',index=False);f.to_csv(RUN/f'minute_execution_{label}_fills.csv',index=False);o.to_csv(RUN/f'minute_execution_{label}_orders.csv',index=False)
    dump(result,RUN/'minute_execution_results.json');print('SHSZ and minute execution extensions complete',flush=True)

if __name__=='__main__':main()
