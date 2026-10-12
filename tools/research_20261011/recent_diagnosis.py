"""Explain recent failure using frozen scores; no new parameter selection."""
import json
import pandas as pd
from common import RUN,DATA,dump
from ledger import metrics

def main():
    exits=pd.read_parquet(DATA/'model_panel.parquet',columns=['date','stockid','label_exit_1','label_exit_5','label_exit_20'])
    rows=[]
    periods=[('first_year','2024-10-08','2025-09-30'),('recent_year','2025-10-01','2026-09-30')]
    for h in [1,5,20]:
        m=json.loads((RUN/f'h{h}/candidate_metrics.json').read_text())
        q=pd.read_parquet(RUN/f'h{h}/predictions.parquet').merge(exits[['date','stockid',f'label_exit_{h}']],on=['date','stockid'])
        for period,start,end in periods:
            g=q[q.date.between(start,end)&(q[f'label_exit_{h}']<=pd.Timestamp(end))].dropna(subset=[m['model'],f'label_{h}',f'target_{h}'])
            ic=g.groupby('date').apply(lambda d:d[m['model']].corr(d[f'target_{h}']))
            top=g.sort_values(['date',m['model'],'stockid'],ascending=[True,False,True]).groupby('date').head(m['topn']).groupby('date')[f'label_{h}'].mean()
            pool=g.groupby('date')[f'label_{h}'].mean()
            rows.append({'horizon':h,'period':period,'model':m['model'],'mean_daily_rank_ic':float(ic.mean()),
                         'mean_top_label_return':float(top.mean()),'mean_pool_label_return':float(pool.mean()),
                         'top_minus_pool':float((top-pool).mean()),'days':len(ic),
                         'scope':'Overlapping forecast labels, before costs/actual fills; not portfolio returns or independent observations'})
    pd.DataFrame(rows).to_csv(RUN/'recent_signal_diagnosis.csv',index=False)
    controls=[]
    for name,path in [('pool_equal','h5/equal_pool_daily.csv'),('bj50_price','bj50_benchmark.csv')]:
        d=pd.read_csv(RUN/path,parse_dates=['date'])
        for period,start,end in periods:
            part=d[d.date.between(start,end)];i=part.index.min();capital=d.loc[i-1,'equity'] if i>0 else 1e6
            m=metrics(part,capital);m.update(control=name,period=period);controls.append(m)
    dump(controls,RUN/'recent_benchmark_diagnosis.json')
    print(pd.DataFrame(rows)[['horizon','period','mean_daily_rank_ic','mean_top_label_return','mean_pool_label_return']].to_string(index=False))
    print('Benchmark recent year',[(m['control'],m['total_return']) for m in controls if m['period']=='recent_year'])

if __name__=='__main__':main()
