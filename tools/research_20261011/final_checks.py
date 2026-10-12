"""Artifact invariants, candidate controls and frozen prospective release metadata."""
import argparse,json,hashlib,platform,importlib.metadata
import numpy as np
import pandas as pd
from common import DATA,RUN,ROOT,HORIZONS,dump,sha
from ledger import Market,metrics

def main(horizons):
    p=pd.read_parquet(DATA/'prices.parquet',columns=['date','stockid','open','close','amount','adv20','observed','vol20','mom20']);a=pd.read_csv(DATA/'actions.csv');reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']).set_index('date');market=Market(p,a,reg)
    checks=[];releases=[]
    for h in horizons:
        folder=RUN/f'h{h}';q=pd.read_parquet(folder/'predictions.parquet');m=json.loads((folder/'candidate_metrics.json').read_text(encoding='utf-8'))
        daily=pd.read_csv(folder/'candidate_daily.csv',parse_dates=['date']);o=pd.read_csv(folder/'candidate_orders.csv',parse_dates=['date']);t=pd.read_csv(folder/'candidate_trades.csv',parse_dates=['entry_date','exit_date'])
        assert (daily.cash>=-.01).all();identity=float((daily.equity-daily.cash-daily.market_value).abs().max());assert identity<1e-6
        assert (t.exit_date>t.entry_date).all();assert not q.duplicated(['date','stockid']).any()
        o['notional']=o.shares*o.price;used=o.groupby(['date','stockid']).notional.sum().reset_index()
        prior=p.sort_values(['stockid','date']).copy();prior['capacity']=prior.groupby('stockid').adv20.shift(1)*.01
        used=used.merge(prior[['date','stockid','capacity']],on=['date','stockid']);overrun=float((used.notional-used.capacity).max());assert overrun<=1e-5
        folds=json.loads((folder/'folds.json').read_text(encoding='utf-8'));assert all(pd.Timestamp(x['training_last_label_exit'])<pd.Timestamp(x['test_start']) for x in folds)
        # Final trained files are serialized only from local trusted training, never remote pickle.
        latest=sorted(q.quarter.unique())[-1];last=q.date.max();release=q[q.date==last].sort_values([m['model'],'stockid'],ascending=[False,True]).copy()
        release['rank']=np.arange(1,len(release)+1);release['target_cap']=float(reg.loc[last,'cap_'+m['base_policy']])*m['risk_scale']
        release[['date','stockid',m['model'],'rank','target_cap']].to_csv(folder/'frozen_latest_ranking.csv',index=False)
        metadata={'horizon':h,'status':'frozen research candidate; historical exploration, prospective validation pending',
                  'frozen_asof':'2026-10-12','last_input_session':str(last.date()),'forecast_for_next_session':'2026-10-12',
                  'score_model':m['model'],'risk_policy':m['base_policy'],'risk_scale':m['risk_scale'],'guard':m['guard'],'topn':m['topn'],'cost_bp':5,
                  'forecast_horizon':h,'holding_period':m.get('holding_period',h),
                  'model_file':str((folder/(latest+'_models.joblib')).relative_to(ROOT)),
                  'model_sha256':sha(folder/(latest+'_models.joblib')),'input_sha256':sha(DATA/'model_panel.parquet'),
                  'training':next(x for x in folds if x['quarter']==latest),'minimum_prospective_sessions':126,
                  'review_rule':'Do not retune during prospective evaluation; update observations and predeclared monthly/quarterly refits only. Reopened tuning starts a new version.'}
        dump(metadata,folder/'frozen_release.json');releases.append(metadata)
        main=daily[daily.date<='2026-09-30'];z=metrics(main);z['horizon']=h;z['scope']='Full 2024Q4-2026Q3; includes initial capital in drawdown';dump(z,folder/'candidate_main_period_metrics.json')
        segments=[]
        for start,end,label in [('2024-10-08','2025-09-30','first_year'),('2025-10-01','2026-09-30','second_year'),('2026-04-01','2026-09-30','recent_half_year')]:
            part=daily[daily.date.between(start,end)];i=part.index.min();initial=daily.loc[i-1,'equity'] if i>0 else 1e6
            sm=metrics(part,initial);sm['period']=label;segments.append(sm)
        dump(segments,folder/'candidate_subperiods.json')
        checks.append({'horizon':h,'cash_identity_error':identity,'capacity_overrun_max':overrun,'same_day_sales':int((t.exit_date<=t.entry_date).sum()),
                       'duplicate_predictions':int(q.duplicated(['date','stockid']).sum()),'purged_folds':len(folds),'ledger_rows':len(daily),'orders':len(o),'passed':True})
        if h==5:
            control=[]
            for policy in ['none','vol12','hmm2_defensive','hmm3_defensive','hmm3_growth']:
                reg['cap_control']=reg['cap_'+policy]*m['risk_scale']
                _,cm,*_=market.run(q,horizon=h,score=m['model'],topn=20,policy='control',guard=m['guard']);cm['control']=policy+' scaled by same factor';control.append(cm)
            lo,hi=0.,1.
            for _ in range(12):
                cap=(lo+hi)/2;reg['cap_control']=cap
                _,cm,*_=market.run(q,horizon=h,score=m['model'],topn=20,policy='control',guard=m['guard'])
                if cm['mean_exposure']<m['mean_exposure']:lo=cap
                else:hi=cap
            cm.update(control='constant cap matched to realized candidate exposure; retrospective diagnostic',constant_cap=cap)
            control.append(cm);dump(control,folder/'candidate_hmm_controls.json')
    dump(checks,RUN/'artifact_checks.json');dump(releases,RUN/'frozen_releases.json')
    packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','catboost','xgboost','hmmlearn','torch','paramiko','joblib','pyarrow','matplotlib','threadpoolctl','requests','einops','safetensors','huggingface-hub','tqdm']}
    dump({'python':platform.python_version(),'packages':packages},RUN/'environment.json')
    print('artifact checks passed',len(checks),'horizons',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--horizons',default='1,2,5,10,20');args=parser.parse_args();main([int(x) for x in args.horizons.split(',')])
