"""Never hide recent degradation behind a passing full-history ratio."""
import pandas as pd
from common import RUN,HORIZONS
from ledger import metrics

def main():
    rows=[]
    for h in HORIZONS:
        for family in ['fixed_daily','risk_refinement_daily']:
            allnav=pd.read_parquet(RUN/f'h{h}/{family}.parquet')
            for name,d in allnav.groupby('strategy'):
                d=d.sort_values('date').reset_index(drop=True);r={'horizon':h,'family':family,'strategy':name}
                for start,end,label in [('2024-10-08','2026-10-09','full'),('2025-10-01','2026-09-30','recent_year'),('2026-04-01','2026-09-30','recent_half')]:
                    g=d[d.date.between(start,end)];i=g.index[0];initial=d.loc[i-1,'equity'] if i>0 else 1e6;m=metrics(g,initial)
                    for key in ['annual_return','sharpe_rf15','calmar','max_drawdown','pass_target']:r[label+'_'+key]=m[key]
                rows.append(r)
    q=pd.DataFrame(rows);q.to_csv(RUN/'all_strategy_period_audit.csv',index=False)
    for cols in [['full_pass_target','recent_year_pass_target'],['full_pass_target','recent_half_pass_target'],['full_pass_target','recent_year_pass_target','recent_half_pass_target']]:
        x=q[q[cols].all(axis=1)];print(cols,'passed',len(x),flush=True)
        print(x.sort_values('recent_year_calmar',ascending=False)[['horizon','strategy','full_calmar','recent_year_sharpe_rf15','recent_year_calmar','recent_half_calmar']].head(8).to_string(index=False),flush=True)

if __name__=='__main__':main()
