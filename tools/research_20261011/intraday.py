"""Past-pool minute IC and next-session daily selection using aggregated minute factors."""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from common import ROOT,DATA,RUN,dump
from ledger import Market

def main():
    p=pd.read_parquet(DATA/'prices.parquet');calendar=sorted(p.date.unique());prior={calendar[i]:calendar[i-1] for i in range(1,len(calendar))}
    pool={d:set(g.stockid) for d,g in p[p.in_pool].groupby('date')}
    paths={}
    for folder in [ROOT/'training_data/research_20260930/minute',ROOT/'training_data/research_20261008/minute',DATA/'minute']:
        for path in folder.glob('*.parquet'):paths[path.stem]=path
    daily=[];icrows=[];quality=[]
    for j,(name,path) in enumerate(sorted(paths.items())):
        date=pd.to_datetime(name,format='%Y%m%d');allowed=pool.get(pd.Timestamp(prior.get(date)),set())
        if not allowed:continue
        x=pd.read_parquet(path);x['stockid']='BJ'+x.code.astype(str).str.zfill(6);x=x[x.stockid.isin(allowed)].copy()
        x=x.sort_values(['stockid','datetime']);duplicate=int(x.duplicated(['stockid','datetime']).sum())
        x=x.drop_duplicates(['stockid','datetime'])
        numeric=['open','high','low','close','volume','amount'];x[numeric]=x[numeric].astype(float)
        x=x[(x.close>0)&(x.open>0)&(x.volume>0)&(x.amount>0)].copy();g=x.groupby('stockid',sort=False)
        x['ret1']=g.close.pct_change(fill_method=None);x['mom5']=g.close.pct_change(5,fill_method=None)
        x['mom20']=g.close.pct_change(20,fill_method=None)
        x['volume_ratio']=x.volume/g.volume.transform(lambda a:a.rolling(20,min_periods=15).mean())
        x['pressure']=(x.close-x.open)/(x.high-x.low).replace(0,np.nan)
        x['reversal']=-x['mom5'];x['volume_pressure']=x.pressure*x.volume_ratio
        # Continuity gate excludes missing-bar gaps and the lunch break from forward IC labels.
        for h in [1,5,20]:
            endtime=g.datetime.shift(-h);x['forward']=g.close.shift(-h)/x.close-1
            good=(endtime-x.datetime).dt.total_seconds()==60*h
            sampled=x[good & (x.datetime.dt.minute%5==0)]
            for factor in ['mom5','mom20','reversal','volume_pressure']:
                values=sampled.groupby('datetime').apply(lambda a:a[factor].corr(a.forward,method='spearman') if len(a)>=30 else np.nan).dropna()
                if len(values):icrows.append({'date':date,'horizon_minutes':h,'factor':factor,'daily_mean_ic':values.mean(),'cross_sections':len(values)})
        for sid,a in x.groupby('stockid',sort=False):
            if len(a)<180:continue
            afternoon=a[a.datetime.dt.hour>=13];morning=a[a.datetime.dt.hour<12];tail=a.tail(30)
            r=a.ret1.dropna();rv=np.sqrt((r*r).sum())
            daily.append({'date':date,'stockid':sid,'minute_return':a.close.iloc[-1]/a.open.iloc[0]-1,
                          'late_return':tail.close.iloc[-1]/tail.open.iloc[0]-1,
                          'realized_vol':rv,'downside_vol':np.sqrt((r.clip(upper=0)**2).sum()),
                          'return_skew':r.skew(),'volume_concentration':(a.volume/a.volume.sum()).pow(2).sum(),
                          'late_volume_share':tail.volume.sum()/a.volume.sum(),
                          'close_vwap_gap':a.close.iloc[-1]/(a.amount.sum()/a.volume.sum())-1,
                          'afternoon_minus_morning':afternoon.close.iloc[-1]/afternoon.open.iloc[0]-morning.close.iloc[-1]/morning.open.iloc[0] if len(afternoon) and len(morning) else np.nan,
                          'price_volume_corr':r.corr(a.volume),'bars':len(a)})
        quality.append({'date':date,'source':str(path.relative_to(ROOT)),'rows':len(x),'stocks':x.stockid.nunique(),'duplicates_removed':duplicate})
        if j%20==0:print('minute factors',j,'/',len(paths),flush=True)
    f=pd.DataFrame(daily);f.to_parquet(DATA/'minute_factors.parquet',index=False);pd.DataFrame(quality).to_csv(RUN/'minute_quality.csv',index=False)
    ic=pd.DataFrame(icrows);ic['split']=np.where(ic.date<'2026-01-01','train_2025Q4','test_2026Q3')
    ic.to_csv(RUN/'minute_ic_daily.csv',index=False)
    summary=ic.groupby(['split','horizon_minutes','factor']).agg(mean_ic=('daily_mean_ic','mean'),daily_ic_std=('daily_mean_ic','std'),days=('date','nunique')).reset_index()
    summary['icir']=summary.mean_ic/summary.daily_ic_std;summary.to_csv(RUN/'minute_ic_summary.csv',index=False)
    features=[c for c in f if c not in ['date','stockid','bars']]
    f[features]=f[features].replace([np.inf,-np.inf],np.nan).groupby(f.date).rank(pct=True).sub(.5)
    q=f.merge(p[['date','stockid']+[f'target_{h}' for h in []]+[f'label_{h}' for h in [1,5,20]]+[f'label_exit_{h}' for h in [1,5,20]]],on=['date','stockid'])
    regimes=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date');market=Market(p,pd.read_csv(DATA/'actions.csv'),regimes)
    results=[];factorcards=[]
    for h in [1,5,20]:
        train=q[(q.date<'2026-01-01')&(q[f'label_exit_{h}']<'2026-01-01')].dropna(subset=[f'label_{h}']).copy()
        test=q[q.date>='2026-07-01'].copy()
        train['target']=train.groupby('date')[f'label_{h}'].rank(pct=True).sub(.5)
        model=Ridge(alpha=100).fit(train[features].fillna(0),train.target);test['score']=model.predict(test[features].fillna(0))
        for c,coef in zip(features,model.coef_):factorcards.append({'horizon':h,'factor':c,'train_coefficient':coef})
        for policy in ['none','vol12','hmm3_volatility']:
            d,m,*_=market.run(test,horizon=h,topn=20,policy=policy,start=test.date.min(),end=test.date.max())
            m.update(train_days=train.date.nunique(),label='Intraday aggregated features; frozen 2025Q4 train, 2026Q3 evaluation; small sample')
            results.append(m);d.to_csv(RUN/f'minute_factor_h{h}_{policy}_daily.csv',index=False)
        test[['date','stockid','score',f'label_{h}']].to_parquet(RUN/f'minute_factor_h{h}_predictions.parquet',index=False)
    pd.DataFrame(factorcards).to_csv(RUN/'minute_factor_coefficients.csv',index=False);dump(results,RUN/'minute_factor_results.json')
    print('minute complete',len(paths),'files',len(f),'factor rows',flush=True)

if __name__=='__main__':
    with warnings.catch_warnings():warnings.simplefilter('ignore',FutureWarning);main()
