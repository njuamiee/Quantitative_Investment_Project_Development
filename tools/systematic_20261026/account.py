"""Compiled share/cash ledger: causal orders, T+1, costs, shared ADV capacity.

This is daily-bar execution research, not a claim of actual auction fills.
All stop and allocation decisions refer to the preceding close.
"""
import itertools,json
import numpy as np
import pandas as pd
from numba import njit
from config import RUN,START,END,dump
from features import load_meta

RISK_AXES=dict(topn=[5,20,40],weighting=['equal','inverse_vol'],
 exposure=['full','half','vol15'],regime=['none','hmm2','hmm3'],
 exit=['scheduled','entry_stop8','trailing10','portfolio_dd'],
 concentration=['none','industry_size'],entry=['rank','rebound_confirm'],
 pool=[50,100],rebalance=['horizon','daily'])

def risk_grid(h):
    names=list(RISK_AXES);result=[];seen=set()
    for vals in itertools.product(*(RISK_AXES[n] for n in names)):
        c=dict(zip(names,vals));c['rebalance_days']=h if c['rebalance']=='horizon' else 1
        # h=1 daily/horizon are exactly the same strategy; do not inflate counts.
        signature=tuple(v for k,v in c.items() if k!='rebalance')
        if signature in seen:continue
        seen.add(signature);c['risk_id']=len(result);result.append(c)
    return result

@njit(cache=True)
def ledger(order,op,cl,adv,observed,div,split,vol,mom2,mom5,mom20,industry,size,amount_rank,
           market_vol,trend5,trend20,breadth,market_dd,hmm2,hmm3,start,finish,
           topn,weighting,expo,regime,exit_rule,concentration,entry,pool,rebal,fee=.0005,capital=1e6,schedule=None):
    ns=op.shape[1];nd=finish-start+1
    q=np.zeros(ns);last=np.zeros(ns);basis=np.zeros(ns);peakprice=np.zeros(ns)
    bought=np.full(ns,-999,dtype=np.int64);ban=np.full(ns,-1,dtype=np.int64)
    pending=np.zeros(ns,dtype=np.int64)
    daily=np.zeros((nd,10));cash=capital;peak=capital;previous=capital;dd=0.;maxerr=0.
    for s in range(ns):
        if start>0 and np.isfinite(cl[start-1,s]):last[s]=cl[start-1,s]
    for i in range(start,finish+1):
        prev=i-1;j=i-start;opening=cash;cost=0.;buys=0.;sells=0.;divcash=0.;limited=0.;attempts=0
        changed=False
        if schedule is not None:
            row=schedule[prev]
            topn=int(row[0]);weighting=int(row[1]);expo=int(row[2]);regime=int(row[3]);exit_rule=int(row[4])
            concentration=int(row[5]);entry=int(row[6]);pool=int(row[7]);rebal=int(row[8])
            changed=prev>0 and schedule[prev,9]!=schedule[prev-1,9]
        used=np.zeros(ns);stop=np.zeros(ns,dtype=np.int64)
        for s in range(ns):
            if q[s]>0:
                if exit_rule==1 and last[s]<basis[s]*.92:stop[s]=1
                if exit_rule==2 and last[s]<peakprice[s]*.90:stop[s]=1
                divcash+=q[s]*div[i,s];q[s]*=split[i,s]
                basis[s]=(basis[s]-div[i,s])/split[i,s];peakprice[s]=(peakprice[s]-div[i,s])/split[i,s]
                if stop[s]:pending[s]=1;ban[s]=i+10
        cash+=divcash
        target=1. if expo==0 else .5 if expo==1 else 0. if expo==3 else min(1.,.15/max(.03,market_vol[prev]))
        cap=1. if regime==0 else hmm2[prev] if regime==1 else hmm3[prev]
        target=min(target,cap)
        if exit_rule==3:
            multiplier=1. if dd<.08 else .5 if dd<.15 else .1
            if trend5[prev]>0 and breadth[prev]>.6:multiplier=max(multiplier,.5)
            target*=multiplier
        # Rebound ladder: only observable drawdown, breadth and recent recovery.
        # A bearish phase can hold zero; one stock never receives the whole portfolio.
        ntop=topn
        if regime and cap<.25:ntop=1;target=min(target,.05)
        elif regime and cap<.6:ntop=max(3,topn//2)
        if entry==1:
            recovery=trend5[prev]>0 and breadth[prev]>.35
            if trend20[prev]<0:
                target=min(target,.10 if recovery and market_dd[prev]<-.08 else 0.)
            elif breadth[prev]<.55:target=min(target,.5)
        mv=0.
        for s in range(ns):
            price=op[i,s] if np.isfinite(op[i,s]) and op[i,s]>0 else last[s]
            mv+=q[s]*price
        equity=cash+mv
        rebalance=prev>=start and ((prev-start)%rebal==0 or changed)
        wanted=np.zeros(ns); chosen=np.full(100,-1,dtype=np.int64);cw=np.zeros(100);n=0;wsum=0.
        if rebalance and target>0:
            for rr in range(order.shape[1]):
                s=order[prev,rr]
                if s<0:continue
                if not np.isfinite(amount_rank[prev,s]) or amount_rank[prev,s]>pool:continue
                if ban[s]>i or pending[s]:continue
                if entry==1 and not (mom2[prev,s]>0 and mom5[prev,s]>-.02):continue
                if concentration and (industry[prev,s]<=0 or size[prev,s]<=0):continue
                if concentration:
                    ni=0;nsz=0
                    for z in range(n):
                        ni+=industry[prev,chosen[z]]==industry[prev,s]
                        nsz+=size[prev,chosen[z]]==size[prev,s]
                    if ni>=max(1,int(ntop*.3)) or nsz>=max(1,int(ntop*.5)):continue
                chosen[n]=s;cw[n]=1. if weighting==0 else 1/max(.005,vol[prev,s])
                wsum+=cw[n];n+=1
                if n>=ntop:break
            for z in range(n):
                s=chosen[z]
                # Incomplete availability leaves cash; never rescales missing picks to full target.
                weight=(1./ntop if weighting==0 else cw[z]/max(wsum,1e-12)*n/ntop)
                wanted[s]=min(.10,weight*target)*equity
            if concentration:
                for z in range(n):
                    s=chosen[z];ivals=0.;svals=0.
                    for zz in range(n):
                        ss=chosen[zz]
                        if industry[prev,ss]==industry[prev,s]:ivals+=wanted[ss]
                        if size[prev,ss]==size[prev,s]:svals+=wanted[ss]
                    ratio=min(1.,.30*target*equity/max(ivals,1e-12),.50*target*equity/max(svals,1e-12))
                    if ratio<1:
                        # Proportional group reduction only; no renormalization to other names.
                        for zz in range(n):
                            ss=chosen[zz]
                            if industry[prev,ss]==industry[prev,s] or size[prev,ss]==size[prev,s]:wanted[ss]*=ratio
        else:
            reduce=min(1.,target*equity/max(mv,1e-12))
            for s in range(ns):
                price=op[i,s] if np.isfinite(op[i,s]) and op[i,s]>0 else last[s]
                wanted[s]=q[s]*price*reduce
        # Sell before buy, sharing the same prior-ADV limit on both sides.
        for s in range(ns):
            if q[s]<=0:continue
            price=op[i,s];reference=(last[s]-div[i,s])/split[i,s]
            if pending[s]:wanted[s]=0.
            requested=max(0.,q[s]-wanted[s]/price) if np.isfinite(price) and price>0 else q[s]
            if requested<1:continue
            attempts+=1
            if bought[s]>=i or observed[i,s]<.5 or not np.isfinite(price) or price<=0 or price<=reference*.705:
                limited+=1;continue
            available=max(0.,adv[prev,s]*.01-used[s]) if np.isfinite(adv[prev,s]) else 0.
            quantity=min(q[s],np.floor(requested),np.floor(available/price))
            # A residual <100 may be sold only in full; otherwise normal sell minimum100.
            if quantity<100 and quantity<q[s]-1e-8:quantity=0.
            if quantity<requested-1:limited+=1
            if quantity<=0:continue
            value=quantity*price;charge=value*fee;cash+=value-charge;cost+=charge;sells+=value;used[s]+=value;q[s]-=quantity
            if q[s]<1e-8:q[s]=0.;pending[s]=0
        if rebalance:
            mv=0.
            for s in range(ns):
                price=op[i,s] if np.isfinite(op[i,s]) and op[i,s]>0 else last[s]
                mv+=q[s]*price
            for z in range(n):
                s=chosen[z];price=op[i,s];reference=(last[s]-div[i,s])/split[i,s]
                requested=max(0.,wanted[s]-q[s]*price) if np.isfinite(price) and price>0 else 0.
                if requested<=0:continue
                attempts+=1
                if observed[i,s]<.5 or not np.isfinite(price) or price<=0 or price>=reference*1.295:
                    limited+=1;continue
                available=max(0.,adv[prev,s]*.01-used[s]) if np.isfinite(adv[prev,s]) else 0.
                room=max(0.,target*equity-mv)
                if concentration:
                    iv=0.;sv=0.
                    for ss in range(ns):
                        pp=op[i,ss] if np.isfinite(op[i,ss]) and op[i,ss]>0 else last[ss]
                        if industry[prev,ss]==industry[prev,s]:iv+=q[ss]*pp
                        if size[prev,ss]==size[prev,s]:sv+=q[ss]*pp
                    room=min(room,max(0.,.30*target*equity-iv),max(0.,.50*target*equity-sv))
                spend=min(requested,cash/(1+fee),available,room)
                quantity=np.floor(spend/(price*(1+fee)))
                if quantity<100:limited+=1;continue
                if quantity*price<requested-1:limited+=1
                value=quantity*price;charge=value*fee
                oldq=q[s];basis[s]=(basis[s]*oldq+value)/(oldq+quantity)
                peakprice[s]=max(peakprice[s],price) if oldq>0 else price
                q[s]+=quantity;bought[s]=i;cash-=value+charge;cost+=charge;buys+=value;used[s]+=value;mv+=value
        value=0.
        for s in range(ns):
            if np.isfinite(cl[i,s]) and cl[i,s]>0:last[s]=cl[i,s]
            value+=q[s]*last[s]
            if q[s]>0:peakprice[s]=max(peakprice[s],last[s])
        eq=cash+value;peak=max(peak,eq);dd=1-eq/peak
        maxerr=max(maxerr,abs(cash-(opening+divcash+sells-buys-cost)))
        if cash<-.001 or not np.isfinite(eq):raise ValueError('cash/equity invariant')
        daily[j,0]=eq;daily[j,1]=cash;daily[j,2]=value;daily[j,3]=cost;daily[j,4]=value/eq
        daily[j,5]=buys;daily[j,6]=sells;daily[j,7]=attempts;daily[j,8]=limited;daily[j,9]=target
        previous=eq
    return daily,maxerr

def market_inputs():
    meta,cal=load_meta();m=dict(np.load(RUN/'data/market.npz'));r=pd.read_parquet(RUN/'data/regimes.parquet')
    cols=['open','close','adv20','observed','dividend','split','vol20','mom2','mom5','mom20','industry','size','amount_rank']
    args=[np.ascontiguousarray(m[c],dtype=np.float64) for c in cols]
    args += [np.nan_to_num(r[c].to_numpy(float),nan=.2 if c=='vol' else 0.) for c in ['vol','trend5','trend20','breadth','market_drawdown','hmm2_cap','hmm3_cap']]
    return cal,m,args

def score_order(pred,cal,ns):
    order=np.full((len(cal),100),-1,dtype=np.int32)
    for d,g in pred.groupby('date_index'):
        ordered=g.sort_values(['score','stockid'],ascending=[False,True]).head(100)
        order[int(d),:len(ordered)]=ordered.stock_index.to_numpy()
    return order

def run_account(order,args,cal,risk,fee=5.,start=START,end=END):
    idx=lambda a,v:RISK_AXES[a].index(v)
    params=[risk['topn']]+[idx(a,risk[a]) for a in ['weighting','exposure','regime','exit','concentration','entry']]+[risk['pool'],risk['rebalance_days']]
    return ledger(order,*args,cal.get_loc(start),cal.get_loc(end),*params,fee/1e4)

def metrics(equity,dates,capital=1e6,initial_date=None):
    r=np.r_[equity[0]/capital-1,equity[1:]/equity[:-1]-1]
    peak=np.maximum.accumulate(np.r_[capital,equity])[1:];dd=1-equity/peak
    years=max((pd.Timestamp(dates[-1])-pd.Timestamp(initial_date if initial_date is not None else dates[0])).days/365.25,1/252)
    annual=(equity[-1]/capital)**(1/years)-1;sd=np.std(r,ddof=1);mdd=float(dd.max())
    sr=(r.mean()-(1.015**(1/252)-1))/sd*np.sqrt(252) if sd>1e-12 else np.nan
    cm=annual/mdd if mdd>1e-12 else np.nan
    return dict(total_return=float(equity[-1]/capital-1),annual_return=float(annual),max_drawdown=mdd,
                sharpe=float(sr),calmar=float(cm),pass_target=bool(sr>1 and cm>2 and mdd<.30))
