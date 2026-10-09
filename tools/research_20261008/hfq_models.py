"""Stable-base adjusted-price target/feature sensitivity; not a settlement-cash ledger."""
import sys,json
from pathlib import Path
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/research_20260930'))
from prepare import DATA,PRICE_FEATURES,read_csv,RENAME,dump
from train import fit_lgb,model_metrics
from accounts import simulate
D=ROOT/'training_data/research_20261008';R=ROOT/'training_runs/teacher9_20261008';OLD=ROOT/'training_runs/teacher9_20260930'

def main():
    panel=pd.read_parquet(DATA/'bj_model_panel.parquet');prices=pd.read_parquet(DATA/'bj_prices.parquet');cal=pd.DatetimeIndex(sorted(prices.date.unique()));parts=[]
    for p in sorted((D/'bj/hfq').glob('*.csv')):
        x=read_csv(p).rename(columns=RENAME);x['date']=pd.to_datetime(x.date);x=x.set_index('date').reindex(cal);x.index.name='date'
        for c in ['open','high','low','close']:x[c]=pd.to_numeric(x[c],errors='coerce')
        x['stockid']='BJ'+p.name[:6];x['hfq_label']=x.close.shift(-5)/x.open.shift(-1)-1
        x['ret1']=x.close.pct_change(fill_method=None)
        for h in [5,10,20,60]:x[f'h_mom{h}']=x.close/x.close.shift(h)-1
        for h in [10,20,60]:x[f'h_vol{h}']=x.ret1.rolling(h,min_periods=int(h*.8)).std()
        x['h_range20']=((x.high-x.low)/x.close.shift(1)).rolling(20,min_periods=16).mean()
        rawamount=prices[prices.stockid==x.stockid.iloc[0]].set_index('date').amount.reindex(cal)
        x['h_log_illiq']=np.log((x.ret1.abs()/rawamount.where(rawamount>0)).rolling(20,min_periods=16).mean().clip(lower=1e-16))
        parts.append(x[['stockid','hfq_label']+[c for c in x if c.startswith('h_')]].reset_index())
    hfq=pd.concat(parts,ignore_index=True).replace([np.inf,-np.inf],np.nan);panel=panel.merge(hfq,on=['date','stockid'],how='left',validate='one_to_one')
    hcols=[c for c in hfq if c.startswith('h_')];ranked=panel.groupby('date')[hcols].rank(pct=True).sub(.5);ranked.columns=['xh_'+c[2:] for c in hcols];panel=pd.concat([panel,ranked],axis=1)
    panel['target']=panel.groupby('date').hfq_label.rank(pct=True).sub(.5)
    folds=json.loads((OLD/'bj_folds.json').read_text('utf-8'));dates=pd.DatetimeIndex(sorted(panel.date.unique()));frames=[]
    for fold in folds:
        q=fold['quarter'];test=panel[panel.date.dt.to_period('Q')==pd.Period(q)].copy();begin=test.date.min();history=dates[dates<begin];valid_start=history[-126];train_start=history[-630]
        a=panel[(panel.date>=train_start)&(panel.date<valid_start)&(panel.exit_date<valid_start)&panel.target.notna()]
        b=panel[(panel.date>=valid_start)&(panel.date<begin)&(panel.exit_date<begin)&panel.target.notna()]
        refit=panel[(panel.date>=train_start)&(panel.date<begin)&(panel.exit_date<begin)&panel.target.notna()]
        featurefile=OLD/'bj'/q/'selected_lgbm.txt'
        import lightgbm as lgb
        selected=lgb.Booster(model_str=featurefile.read_text('utf-8')).feature_name();directory=R/'hfq'/q;directory.mkdir(parents=True,exist_ok=True)
        test['hfq_target_only'],_=fit_lgb(a,b,refit,test,selected,directory,'target_only')
        adjusted=[c.replace('x_','xh_',1) if c[2:] in [v[2:] for v in hcols] else c for c in selected]
        test['hfq_target_features'],_=fit_lgb(a,b,refit,test,adjusted,directory,'target_features')
        frames.append(test[['date','stockid','entry_date','exit_date','label_raw','hfq_label','target','adv20','close','hfq_target_only','hfq_target_features']]);print('HFQ',q,flush=True)
    scored=pd.concat(frames,ignore_index=True);scored.to_parquet(R/'hfq_predictions.parquet',index=False);results=[]
    for name in ['hfq_target_only','hfq_target_features']:
        daily,_,_,m=simulate(scored,prices,score=name);daily.to_csv(R/(name+'_account.csv'),index=False)
        metricdata=scored.copy();metricdata['label_raw']=scored.hfq_label;pm,_=model_metrics(metricdata,name)
        m.update({'model':name,'hfq_rank_ic':pm['mean_ic'],'return_basis':'unchanged raw-price-only account to isolate changed predictions; not hfq as execution prices'});results.append(m)
    dump({'input_files':len(parts),'label_coverage_among_raw_labeled':float(panel.loc[panel.label_raw.notna(),'hfq_label'].notna().mean()),'label_pairs_differ_over_1pp':int(((panel.hfq_label-panel.label_raw).abs()>.01).sum()),'models':results,'limitations':'Current supplier proportional hfq series is a return-continuity proxy. Historical revisions, reinvestment/tax convention and event completeness remain unverified. Original training-only feature identities frozen; do not call this a newly selected optimal model.'},R/'hfq_results.json')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
