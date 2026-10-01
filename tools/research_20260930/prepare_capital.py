"""Daily supplier snapshots of share capital; no hindsight backfill or action inference."""
import json
import pandas as pd
import numpy as np
from prepare import DATA,RUN,read_zip,dump

def main():
    sources=[]
    for table,mapping in [('071',{'t002':'free_float_shares','t005':'adjusted_free_float_shares'}),
                          ('601',{'t005':'total_shares','t007':'circulating_shares','rdate':'report_date'})]:
        frames=[]
        for p in sorted((DATA/'factors'/table).glob('*/*.zip')):
            x=read_zip(p,'BJ')[['stockid','enddate',*mapping]].rename(columns=mapping)
            x['date']=pd.to_datetime(p.stem)
            x=x[x.stockid.str.fullmatch(r'BJ920\d{3}')].copy()
            if x.duplicated(['date','stockid']).any():raise ValueError('duplicate share-capital keys')
            for c in mapping.values():x[c]=pd.to_numeric(x[c],errors='coerce')
            x=x.rename(columns={'enddate':'supplier_enddate_'+table})
            frames.append(x)
        sources.append(pd.concat(frames,ignore_index=True))
    x=sources[0].merge(sources[1],on=['date','stockid'],how='outer',validate='one_to_one')
    prices=pd.read_parquet(DATA/'bj_prices.parquet',columns=['date','stockid','volume','in_pool'])
    x=x.merge(prices,on=['date','stockid'],how='left',validate='one_to_one')
    replay=json.loads((RUN/'minute_replay_summary.json').read_text(encoding='utf-8'))
    ratio=replay['volume_ratio_median']
    if not 99<ratio<101:raise ValueError('Daily-volume unit not supported by minute reconciliation')
    x['capital_ordering_warning']=(x.free_float_shares>x.circulating_shares)|(x.circulating_shares>x.total_shares)
    x['turnover_circulating_proxy']=x.volume*100/x.circulating_shares.where((x.circulating_shares>0)&~x.capital_ordering_warning)
    x['snapshot_available_at_assumption']=x.date+pd.Timedelta(hours=18)
    x.to_parquet(DATA/'bj_share_capital.parquet',index=False)
    pool=prices[prices.in_pool].merge(x[['date','stockid','circulating_shares','free_float_shares']],on=['date','stockid'],how='left',validate='one_to_one')
    summary={'rows':len(x),'snapshot_dates':x.date.nunique(),'stocks':x.stockid.nunique(),
        'first_date':str(x.date.min().date()),'last_date':str(x.date.max().date()),
        'pool_circulating_share_coverage':float((pool.circulating_shares>0).mean()),
        'pool_free_float_share_coverage':float((pool.free_float_shares>0).mean()),
        'minute_daily_volume_ratio_median':ratio,'daily_volume_to_shares_multiplier':100,
        'turnover_proxy_median':float(x.turnover_circulating_proxy.median()),
        'capital_ordering_violation_rows':int(((x.free_float_shares>x.circulating_shares)|(x.circulating_shares>x.total_shares)).sum()),
        'status':'auxiliary candidate only; not added to already evaluated primary models',
        'limitations':'Supplier snapshot availability assumed 18:00 on file date; historical publication/revision timing unverified. No backfill. Report-date share count may lag. Not a verified corporate-action cash/share ledger.'}
    dump(summary,RUN/'share_capital_summary.json');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
