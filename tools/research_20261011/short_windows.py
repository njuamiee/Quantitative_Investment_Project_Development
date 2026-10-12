"""Explicit follow-up to recent-year degradation: monthly short-window adaptation.
Registered before these predictions: 63/126/252 sessions, Ridge and shallow LGBM.
"""
import json,warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
from common import DATA,RUN,dump,SEED
from train_models import screening
from ledger import Market,metrics

def main():
    folder=RUN/'short_windows';folder.mkdir(exist_ok=True)
    fingerprint=json.loads((RUN/'data_summary.json').read_text(encoding='utf-8'))['panel_sha256']
    dump({'status':'follow-up exploratory study after recent-year failure, not untouched validation','forecast_horizons':[1,5,20],
          'training_sessions':[63,126,252],'refit':'monthly','bank':'initial 2024Q4 training-selected 100 plus causal price/LLM features',
          'models':['ridge20','ridge180','lgbm','ic'],'input_sha256':fingerprint,'risk_grid':'w504/w126 HMM3 defensive/growth, scales .25/.5, plus none and vol12',
          'acceptance':'Full interval AND recent year AND recent half-year must jointly pass Sharpe>1, Calmar>2 for a stability claim'},folder/'registry.json')
    panel=pd.read_parquet(DATA/'model_panel.parquet').sort_values(['date','stockid']);dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    for h in [1,5,20]:
        initial=pd.read_csv(RUN/f'h{h}/2024Q4_deployment_factors.csv');bank=initial.loc[initial.selected,'factor'].head(100).tolist()
        bank=list(dict.fromkeys(bank+[c for c in panel if c.startswith('x_') and not c.startswith(('x_c01','x_c02'))]))
        for window in [63,126,252]:
            path=folder/f'pred_h{h}_w{window}.parquet'
            cache=folder/f'cache_h{h}_w{window}.json'
            if path.exists() and cache.exists() and json.loads(cache.read_text(encoding='utf-8')).get('input_sha256')==fingerprint:continue
            parts=[];cards=[]
            for month,test in panel[panel.date>='2024-10-01'].groupby(panel.date.dt.to_period('M')):
                cutoff=test.date.min();past=dates[dates<cutoff][-window:];train=panel[panel.date.isin(past)&(panel[f'label_exit_{h}']<cutoff)].dropna(subset=[f'target_{h}'])
                cols,card=screening(train,h,bank,count=40);cards.append({'month':str(month),'training_last_label_exit':str(train[f'label_exit_{h}'].max()),'test_start':str(cutoff),'features':cols})
                x=train[cols].fillna(0).to_numpy(dtype=np.float32);y=train[f'target_{h}'].to_numpy();z=test[cols].fillna(0).to_numpy(dtype=np.float32)
                q=test[['date','stockid',f'target_{h}']].copy()
                for alpha in [20,180]:q['ridge'+str(alpha)]=Ridge(alpha=alpha).fit(x,y).predict(z)
                model=LGBMRegressor(n_estimators=120,num_leaves=7,max_depth=3,min_child_samples=100,reg_lambda=20,learning_rate=.025,n_jobs=3,verbosity=-1,random_state=SEED)
                q['lgbm']=model.fit(x,y).predict(z);w=card.set_index('factor').loc[cols,'mean_ic'].to_numpy();w/=max(abs(w).sum(),1e-12);q['ic']=z@w;parts.append(q)
            pd.concat(parts).to_parquet(path,index=False);dump(cards,folder/f'folds_h{h}_w{window}.json');dump({'input_sha256':fingerprint},cache);print('monthly forecasts',h,window,flush=True)
    del panel
    p=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','amount','adv20','observed','vol20','mom20'])
    reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date');short=pd.read_csv(RUN/'w126_regime_probabilities.csv',parse_dates=['date']).set_index('date')
    for c in short:
        if c.startswith('cap_hmm'):reg[c.replace('cap_','cap_w126_')]=short[c]
    policies=['none','vol12']
    for family in ['hmm3_defensive','hmm3_growth','w126_hmm3_defensive','w126_hmm3_growth']:
        for scale in [.25,.5]:
            name=family+'_'+str(scale);reg['cap_'+name]=reg['cap_'+family]*scale;policies.append(name)
    market=Market(p,pd.read_csv(DATA/'actions.csv'),reg);results=[];nav=[]
    for h in [1,5,20]:
        for window in [63,126,252]:
            q=pd.read_parquet(folder/f'pred_h{h}_w{window}.parquet')
            for model in ['ridge20','ridge180','lgbm','ic']:
                for holding in ([h] if h!=20 else [5,20]):
                    for policy in policies:
                        d,m,*_=market.run(q,horizon=holding,score=model,topn=20,policy=policy)
                        m.update(forecast_horizon=h,holding_period=holding,window=window,model=model)
                        for start,end,label in [('2025-10-01','2026-09-30','recent_year'),('2026-04-01','2026-09-30','recent_half')]:
                            g=d[d.date.between(start,end)];i=g.index[0];initial=d.loc[i-1,'equity'] if i>0 else 1e6;s=metrics(g,initial)
                            for k in ['annual_return','sharpe_rf15','calmar','max_drawdown','pass_target']:m[label+'_'+k]=s[k]
                        m['all_periods_pass']=bool(m['pass_target'] and m['recent_year_pass_target'] and m['recent_half_pass_target']);results.append(m)
                        d['strategy']=f'h{h}_w{window}_{model}_hold{holding}_{policy}';nav.append(d)
            print('monthly accounts',h,window,flush=True)
    result=pd.DataFrame(results);result.to_csv(folder/'all_results.csv',index=False);pd.concat(nav).to_parquet(folder/'all_daily.parquet',index=False)
    dump({'tested':len(result),'full_pass':int(result.pass_target.sum()),'recent_year_pass':int(result.recent_year_pass_target.sum()),
          'recent_half_pass':int(result.recent_half_pass_target.sum()),'all_periods_pass':int(result.all_periods_pass.sum())},folder/'summary.json')
    print('short windows complete',len(result),'all periods pass',int(result.all_periods_pass.sum()),flush=True)

if __name__=='__main__':
    with warnings.catch_warnings(),threadpool_limits(limits=3):warnings.simplefilter('ignore',FutureWarning);main()
