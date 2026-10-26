"""Q4-trained minute signal increment and actual first-30-minute participation simulation.

Order quantities are fixed before the execution window. Realized minute volume
limits fills; it is never a predictor or a pre-trade ranking feature.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(v,'1')
import json,itertools,warnings
import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import Ridge
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from config import ROOT,RUN,REPORT,HORIZONS,key,dump
from account import market_inputs,score_order,metrics
from train import run_job

@njit(cache=True)
def minute_ledger(order,barprice,baramount,cl,div,split,begin,end,h,topn,target,stop,participation,fee=.0005):
    ns=cl.shape[1];q=np.zeros(ns);basis=np.zeros(ns);last=cl[begin-1].copy();ban=np.zeros(ns,dtype=np.int64)
    bought=np.full(ns,-1,dtype=np.int64);pending=np.zeros(ns,dtype=np.int64);cash=1e6;out=np.zeros((end-begin+1,8));maxerr=0.
    for i in range(begin,end+1):
        opening=cash;dcash=0.;buy=0.;sell=0.;cost=0.;prev=i-1
        stopnow=np.zeros(ns,dtype=np.int64)
        for s in range(ns):
            if q[s]>0:
                if stop and last[s]<basis[s]*(1-stop):pending[s]=1;ban[s]=max(ban[s],i+10)
                dcash+=q[s]*div[i,s];q[s]*=split[i,s];basis[s]=(basis[s]-div[i,s])/split[i,s]
            if np.isfinite(last[s]):last[s]=(last[s]-div[i,s])/split[i,s]
        cash+=dcash;equity=cash
        for s in range(ns):
            if q[s]>0 and np.isfinite(last[s]):equity+=q[s]*last[s]
        wanted=q.copy();rebal=(i-begin)%h==0
        if rebal:
            wanted[:]=0.;n=0
            for r in range(order.shape[1]):
                s=order[prev,r]
                if s<0 or ban[s]>i or pending[s] or not np.isfinite(last[s]) or last[s]<=0:continue
                wanted[s]=np.floor(min(.10,target/topn)*equity/last[s]);n+=1
                if n>=topn:break
        for s in range(ns):
            if pending[s]:wanted[s]=0.
        buyq=np.maximum(wanted-q,0);sellq=np.maximum(q-wanted,0);requested=buyq.sum()+sellq.sum();filled=0.
        # Shares were fixed using previous close, before any execution-window volumes.
        for b in range(barprice.shape[1]):
            for side in [0,1]:
                for s in range(ns):
                    desired=sellq[s] if side==0 else buyq[s]
                    if desired<=0:continue
                    price=barprice[i,b,s];amount=baramount[i,b,s]
                    if not np.isfinite(price) or price<=0 or amount<=0:continue
                    if (side==0 and (bought[s]>=i or price<=last[s]*.705)) or (side==1 and price>=last[s]*1.295):continue
                    capacity=np.floor(amount*participation/price)
                    quantity=min(desired,capacity)
                    if side==1:quantity=min(quantity,np.floor(cash/(price*(1+fee))))
                    if quantity<100 and (side==1 or quantity<q[s]-1e-8):continue
                    if quantity<=0:continue
                    value=quantity*price;charge=value*fee;cost+=charge;filled+=quantity
                    # One side can consume the bar, and its amount is debited below.
                    baramount[i,b,s]-=value/participation
                    if side==0:
                        cash+=value-charge;sell+=value;q[s]-=quantity;sellq[s]-=quantity
                        if q[s]<1e-8:pending[s]=0
                    else:
                        cash-=value+charge;buy+=value;basis[s]=(basis[s]*q[s]+value)/(q[s]+quantity)
                        q[s]+=quantity;buyq[s]-=quantity;bought[s]=i
        mv=0.
        for s in range(ns):
            if np.isfinite(cl[i,s]) and cl[i,s]>0:last[s]=cl[i,s]
            if q[s]>0:mv+=q[s]*last[s]
        eq=cash+mv;maxerr=max(maxerr,abs(cash-(opening+dcash+sell-buy-cost)))
        if cash<-.001 or not np.isfinite(eq):raise ValueError('minute cash invariant')
        out[i-begin]=np.array([eq,cash,mv,cost,mv/eq,requested,filled,buy+sell])
    return out,maxerr

def prepare_minutes(cal,ids):
    dest=RUN/'minute';dest.mkdir(exist_ok=True);cache=dest/'first30.npz'
    if cache.exists():a=np.load(cache);return a['price'],a['amount']
    paths={}
    for folder in ['research_20260930','research_20261008','research_20261011']:
        for p in (ROOT/'training_data'/folder/'minute').glob('*.parquet'):paths[p.stem]=p
    price=np.full((len(cal),30,len(ids)),np.nan,np.float32);amount=np.zeros_like(price);logs=[]
    for name,path in sorted(paths.items()):
        date=pd.to_datetime(name,format='%Y%m%d')
        if date<pd.Timestamp('2026-07-01') or date>pd.Timestamp('2026-09-30') or date not in cal:continue
        x=pd.read_parquet(path);x['stockid']='BJ'+x.code.astype(str).str.zfill(6)
        x=x[x.stockid.isin(ids) & (x.datetime.dt.hour==9) & (x.datetime.dt.minute>=30)].copy()
        if x.duplicated(['stockid','datetime']).any():raise ValueError('duplicate minute keys')
        x=x[(x.amount>0)&(x.volume>0)&(x.close>0)]
        scale=((x.amount/x.volume)/x.close).median()
        if not .9<scale<1.1:raise ValueError('minute volume unit mismatch')
        di=cal.get_loc(date);si=ids.get_indexer(x.stockid);bi=x.datetime.dt.minute.to_numpy()-30
        price[di,bi,si]=(x.amount/x.volume).to_numpy();amount[di,bi,si]=x.amount.to_numpy()
        logs.append(dict(date=str(date.date()),stocks=x.stockid.nunique(),bars=len(x),volume_scale=float(scale)))
    np.savez_compressed(cache,price=price,amount=amount);dump(logs,dest/'coverage.json');return price,amount

def main():
    cal,m,args=market_inputs();ids=pd.Index(json.loads((RUN/'data/stocks.json').read_text()));dest=RUN/'minute';dest.mkdir(exist_ok=True)
    bp,ba=prepare_minutes(cal,ids);begin=cal.get_loc('2026-07-01');end=cal.get_loc('2026-09-30')
    meta=pd.read_parquet(RUN/'data/meta.parquet');xx=np.load(RUN/'data/minute.npy');y=np.load(RUN/'data/targets.npy');ex=np.load(RUN/'data/label_exit_index.npy')
    signal=cut=cal.get_loc('2026-01-05');fitcut=cal.get_loc('2025-12-01')
    available=np.isfinite(xx).all(axis=1);di=meta.date_index.to_numpy();rows=[];choices=[];curves=[];audits=[]
    for hi,h in enumerate(HORIZONS):
        c=dict(horizon=h,count=20,model='lightgbm',frequency='quarterly',window=252,library='daily')
        r=run_job(c);pred=pd.read_parquet(RUN/'models'/r['job']/'predictions.parquet')
        join=meta.merge(pred[['date','stockid','score']],on=['date','stockid'],how='left');daily=join.groupby('date').score.rank(pct=True).to_numpy()-.5
        tr=available&(di>=cal.get_loc('2025-10-09'))&(di<fitcut)&(ex[:,hi]<fitcut)&np.isfinite(y[:,hi])
        va=available&(di>=fitcut)&(di<cut)&(ex[:,hi]<cut)&np.isfinite(y[:,hi])
        final=available&(di>=cal.get_loc('2025-10-09'))&(di<cut)&(ex[:,hi]<cut)&np.isfinite(y[:,hi])
        chosen=0.;val=[]
        if tr.sum()>=500 and np.unique(di[va]).size>=5:
            with threadpool_limits(1):model=Ridge(alpha=100).fit(xx[tr],y[tr,hi]);vp=model.predict(xx[va])
            vm=meta.loc[va,['date']].copy();vm['minute']=vp;vm['minute']=vm.groupby('date').minute.rank(pct=True)-.5
            for w in [0.,.25,.5,1.]:
                score=(1-w)*daily[va]+w*vm.minute.to_numpy();a=pd.DataFrame(dict(date=meta.loc[va,'date'],score=score,y=y[va,hi]))
                ic=a.groupby('date').apply(lambda z:z.score.corr(z.y,method='spearman')).mean();val.append((w,float(ic)))
            chosen=max(val,key=lambda z:(z[1],-z[0]))[0]
        with threadpool_limits(1):model=Ridge(alpha=100).fit(xx[final],y[final,hi])
        mp=np.full(len(meta),np.nan);mp[available]=model.predict(xx[available]);mps=pd.Series(mp).groupby(meta.date).rank(pct=True).to_numpy()-.5
        choices.append(dict(horizon=h,chosen_weight=chosen,validation=val,train_rows=int(tr.sum()),valid_rows=int(va.sum()),final_train_rows=int(final.sum())))
        for w in [0.,.25,.5,1.]:
            scores=meta[['date','stockid','date_index','stock_index']].copy();scores['score']=np.where(np.isfinite(mps),(1-w)*daily+w*mps,daily)
            order=score_order(scores[scores.date.between('2026-06-30','2026-09-30')],cal,len(ids))
            for topn,target,stop,participation in itertools.product([5,20,40],[.25,.5,1.],[0.,.08],[.01,.05]):
                d,error=minute_ledger(order,bp,ba.copy(),m['close'],m['dividend'],m['split'],begin,end,h,topn,target,stop,participation)
                mm=metrics(d[:,0],cal[begin:end+1]);rows.append(dict(horizon=h,minute_weight=w,topn=topn,target=target,stop=stop,
                    participation=participation,chosen_on_Q4=w==chosen,cash_error=error,fill_share_fraction=float(d[:,6].sum()/max(d[:,5].sum(),1)),
                    mean_exposure=float(d[:,4].mean()),**mm))
                if topn==20 and target==.5 and stop==0 and participation==.05:
                    curves.append(pd.DataFrame(dict(date=cal[begin:end+1],horizon=h,minute_weight=w,equity=d[:,0],exposure=d[:,4])))
        print('minute horizon',h,'accounts',len(rows),'chosen',chosen,flush=True)
    pd.DataFrame(rows).to_csv(dest/'results.csv',index=False,encoding='utf-8-sig');pd.concat(curves).to_parquet(dest/'curves.parquet',index=False)
    dump(choices,dest/'Q4_selection.json');dump(dict(accounts=len(rows),period=['2026-07-01','2026-09-30'],
        price='each actual minute amount/volume benchmark; not a guaranteed queue fill',
        capacity='1% or 5% of realized per-minute amount, shared across sides',
        decision='fixed quantities from previous close, only first 30 minutes fill',
        minute_signal='Q4 train/validation, Q3 held frozen; missing prior-minute factor falls back to daily score',
        caveats=['no bid/ask queue','no live execution claim','same historical period previously examined'],
        max_cash_error=max(r['cash_error'] for r in rows)),dest/'complete.json')

if __name__=='__main__':
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);main()
