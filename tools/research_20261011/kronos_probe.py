"""Local public-checkpoint inference; frozen weekly 2026Q3 ranking diagnostic.
Not added to validation-selected production grid; no pretraining-vintage claim.
"""
import os,sys,json,subprocess
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi,snapshot_download
from common import ROOT,DATA,RUN,HORIZONS,dump,sha,SEED
sys.path.insert(0,str(DATA/'vendor/Kronos'))
from model import Kronos,KronosTokenizer,KronosPredictor

def main():
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED)
    commits={};paths={}
    for name in ['Kronos-mini','Kronos-Tokenizer-2k']:
        repo='NeoQuasar/'+name;info=HfApi().model_info(repo,token=False);commits[repo]=info.sha
        paths[name]=snapshot_download(repo,revision=info.sha,allow_patterns=['config.json','model.safetensors'],cache_dir=str(DATA/'hf_cache'),token=False)
    model=Kronos.from_pretrained(paths['Kronos-mini']);tokenizer=KronosTokenizer.from_pretrained(paths['Kronos-Tokenizer-2k'])
    predictor=KronosPredictor(model,tokenizer,device='cpu',max_context=128)
    p=pd.read_parquet(DATA/'prices.parquet');calendar=pd.DatetimeIndex(sorted(p.date.unique()))
    dates=calendar[(calendar>='2026-07-01')&(calendar<='2026-08-31')][::5]
    stocks={s:g.set_index('date').sort_index() for s,g in p.groupby('stockid')};rows=[]
    for d in dates:
        eligible=sorted(p.loc[(p.date==d)&p.in_pool,'stockid']);frames=[];timestamps=[];future=[];ids=[]
        i=calendar.get_loc(d);hist=calendar[i-127:i+1];outdates=calendar[i+1:i+21]
        if len(hist)!=128 or len(outdates)!=20:continue
        for sid in eligible:
            block=stocks[sid].reindex(hist)
            cols=['adj_open','adj_high','adj_low','adj_close','volume','amount']
            x=block[cols].copy();x.columns=['open','high','low','close','volume','amount']
            # Adjust volume to keep price*volume economically consistent with the adjusted price scale.
            ratio=(block.adj_close/block.close);x['volume']=x.volume/ratio
            if not np.isfinite(x.to_numpy(dtype=float)).all():continue
            frames.append(x.reset_index(drop=True));timestamps.append(pd.Series(hist));future.append(pd.Series(outdates));ids.append(sid)
        for start in range(0,len(ids),8):
            stop=start+8
            forecasts=predictor.predict_batch(frames[start:stop],timestamps[start:stop],future[start:stop],pred_len=20,T=1.,top_p=.9,sample_count=1,verbose=False)
            for sid,f in zip(ids[start:stop],forecasts):
                actual=stocks[sid].loc[d]
                row={'date':d,'stockid':sid}
                for h in HORIZONS:
                    row['score_'+str(h)]=(f.open.iloc[1] if h==1 else f.close.iloc[h-1])/f.open.iloc[0]-1
                    row['label_'+str(h)]=actual['label_'+str(h)]
                rows.append(row)
        print('Kronos local inference',str(d.date()),len(ids),'stocks',flush=True)
    q=pd.DataFrame(rows);q.to_parquet(RUN/'kronos_predictions.parquet',index=False);summary=[]
    for h in HORIZONS:
        daily=q.groupby('date').apply(lambda x:x['score_'+str(h)].corr(x['label_'+str(h)],method='spearman'))
        summary.append({'horizon':h,'mean_rank_ic':float(daily.mean()),'dates':len(daily),'stocks_per_date':float(q.groupby('date').size().mean()),'diagnostic_only':True})
    dump({'status':'local inference completed','checkpoint_revisions':commits,'model':'Kronos-mini','context':128,'sampling_paths':1,
          'repository_commit':subprocess.check_output(['git','-C',str(DATA/'vendor/Kronos'),'rev-parse','HEAD'],text=True).strip(),
          'results':summary,'limits':'Sparse weekly cross-section diagnostic, no full portfolio gate; checkpoint training data vintage not independently verified; no fine-tuning'},RUN/'kronos_results.json')

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump({'status':'blocked','error_type':type(e).__name__,'error':str(e),'paid_calls':0},RUN/'kronos_results.json');raise
