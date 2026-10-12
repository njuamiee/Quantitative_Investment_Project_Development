"""Generalized share/cash ledger. All risk decisions use information through prior close."""
import math
import numpy as np
import pandas as pd

def metrics(daily,capital=1e6,trades=None):
    e=daily.equity.to_numpy(dtype=float);r=np.r_[e[0]/capital-1,e[1:]/e[:-1]-1]
    peak=np.maximum.accumulate(np.r_[capital,e])[1:];dd=e/peak-1
    years=max((pd.Timestamp(daily.date.iloc[-1])-pd.Timestamp(daily.date.iloc[0])).days/365.25,1/252)
    annual=(e[-1]/capital)**(1/years)-1;sd=np.std(r,ddof=1)
    sharpe=np.mean(r)/sd*np.sqrt(252) if sd>0 else 0.
    sharpe_rf=(np.mean(r)-((1.015)**(1/252)-1))/sd*np.sqrt(252) if sd>0 else 0.
    mdd=abs(float(dd.min()));calmar=annual/mdd if mdd>1e-12 else None
    m={'total_return':float(e[-1]/capital-1),'annual_return':float(annual),'sharpe':float(sharpe),'sharpe_rf15':float(sharpe_rf),
       'max_drawdown':mdd,'calmar':calmar,'pass_target':bool(sharpe_rf>1 and calmar is not None and calmar>2),
       'mean_exposure':float(daily.exposure.mean()),'cash_days_fraction':float((daily.exposure<.01).mean()),
       'turnover_sum':float(daily.turnover.sum()),'total_cost':float(daily.cost.sum()),'days':len(daily),
       'start':str(daily.date.iloc[0]),'end':str(daily.date.iloc[-1]),'worst_day':float(r.min())}
    if trades is not None and len(trades):
        t=pd.DataFrame(trades);w=t.loc[t.pnl>0,'pnl'];loss=t.loc[t.pnl<0,'pnl']
        m.update(completed_sales=len(t),trade_win_rate=float((t.pnl>0).mean()),
                 payoff_ratio=float(w.mean()/abs(loss.mean())) if len(w) and len(loss) else None)
    return m

class Market:
    def __init__(self,prices,actions=None,regimes=None):
        self.dates=pd.DatetimeIndex(sorted(prices.date.unique()));self.ids=sorted(prices.stockid.unique())
        self.di={d:i for i,d in enumerate(self.dates)};self.si={s:i for i,s in enumerate(self.ids)}
        self.a={c:prices.pivot(index='date',columns='stockid',values=c).reindex(index=self.dates,columns=self.ids).to_numpy()
                for c in ['open','close','amount','adv20','observed','vol20','mom20']}
        self.actions={}
        if actions is not None:
            for row in actions.itertuples(index=False):
                if row.stockid in self.si and pd.Timestamp(row.date) in self.di:
                    self.actions.setdefault(self.di[pd.Timestamp(row.date)],[]).append((self.si[row.stockid],float(row.dividend),float(row.split)))
        self.regimes=regimes

    def run(self,pred,horizon=5,score='score',topn=5,policy='none',cost_bp=5,guard='none',capital=1e6,
            start=None,end=None,actions=True,save_details=False,schedule=None):
        d0=pd.Timestamp(start) if start is not None else pred.date.min()
        d1=pd.Timestamp(end) if end is not None else pred.date.max()
        begin=self.di[d0];finish=self.di[d1];fee=cost_bp/10000
        signals={}
        for d,g in pred.dropna(subset=[score]).groupby('date'):
            if d not in self.di:continue
            i=self.di[d]
            daytop=int(schedule.loc[d,'topn']) if schedule is not None and d in schedule.index else topn
            selected=g.sort_values([score,'stockid'],ascending=[False,True]).head(daytop)
            signals[i]=[self.si[s] for s in selected.stockid if s in self.si]
        cap=np.ones(len(self.dates));markettrend=np.zeros(len(cap))
        if self.regimes is not None:
            rc=self.regimes.reindex(self.dates)
            if 'cap_'+policy in rc:cap=rc['cap_'+policy].fillna(.5).to_numpy()
            markettrend=rc.trend20.fillna(0).to_numpy()
        cash=float(capital);lots=[];last=np.full(len(self.ids),np.nan);peak=capital;previous_equity=capital;previous_dd=0.
        orders=[];trades=[];rows=[];holdings=[];failed=0;attempted=0
        def price(s,i,when):
            v=self.a[when][i,s]
            return float(v) if np.isfinite(v) and v>0 else float(last[s]) if np.isfinite(last[s]) else 0.
        for i in range(begin,finish+1):
            date=self.dates[i];turn=cost=divcash=0.;prev=i-1
            used=np.zeros(len(self.ids))
            dayguard=guard;daypolicy=policy
            if schedule is not None and prev>=0 and self.dates[prev] in schedule.index:
                setting=schedule.loc[self.dates[prev]];dayguard=setting.guard;daypolicy=setting.policy
            if actions:
                for s,div,split in self.actions.get(i,[]):
                    for lot in lots:
                        if lot['s']==s:
                            cash+=lot['q']*div;divcash+=lot['q']*div;lot['dividend']+=lot['q']*div;lot['q']*=split
                            lot['peak_price']=max(.001,(lot['peak_price']-div)/split)
            def fillable(s,side,when):
                v=self.a[when][i,s]
                if self.a['observed'][i,s]!=True or not np.isfinite(v) or v<=0:return False
                reference=self.a['close'][prev,s] if prev>=0 else np.nan
                for sid,div,split in self.actions.get(i,[]):
                    if sid==s and reference>0:reference=(reference-div)/split
                # Approximate historical limit check; true exchange reference/status is a documented missing input.
                if np.isfinite(reference) and (v>=reference*1.295 if side=='buy' else v<=reference*.705):return False
                return True
            def sell(lot,q,when,reason):
                nonlocal cash,turn,cost,failed
                if i<=lot['entry'] or not fillable(lot['s'],'sell',when):failed+=1;return
                pr=price(lot['s'],i,when)
                adv=self.a['adv20'][prev,lot['s']] if prev>=0 else 0
                available=max(0,adv*.01-used[lot['s']]) if np.isfinite(adv) else 0
                q=min(float(q),lot['q'],math.floor(available/pr) if pr>0 else 0)
                if q<lot['q']:q=math.floor(q)
                if q<=0:return
                notional=q*pr;f=fee*notional;used[lot['s']]+=notional
                alloc=lot['basis']*q/lot['q'];divalloc=lot['dividend']*q/lot['q']
                cash+=notional-f;turn+=notional;cost+=f
                if save_details:orders.append({'date':date,'stockid':self.ids[lot['s']],'side':'sell','shares':q,'price':pr,'when':when,'reason':reason,'fee':f})
                trades.append({'entry_date':self.dates[lot['entry']],'exit_date':date,'stockid':self.ids[lot['s']],
                               'pnl':notional-f+divalloc-alloc,'return':(notional-f+divalloc)/alloc-1,'reason':reason})
                lot['q']-=q;lot['basis']-=alloc;lot['dividend']-=divalloc
                if lot['q']<1e-8:lots.remove(lot)
            # One-day labels require next-day open exit, before opening a new cohort.
            if horizon==1:
                for lot in list(lots):
                    if i>=lot['due']:sell(lot,lot['q'],'open','scheduled')
            target=float(cap[prev]) if prev>=0 else .5
            if schedule is not None and self.regimes is not None and prev>=0:
                target=float(rc['cap_'+daypolicy].iloc[prev])
                if not np.isfinite(target):target=.5
            if dayguard=='drawdown':
                multiplier=1. if previous_dd<.06 else .5 if previous_dd<.10 else .15
                if prev>=0 and markettrend[prev]>.03:multiplier=max(multiplier,.5)
                target*=multiplier
            elif dayguard=='cppi':target=min(target,max(0,4*(previous_equity-.85*peak)/previous_equity))
            open_value=sum(lot['q']*price(lot['s'],i,'open') for lot in lots);equity=cash+open_value
            if dayguard=='stop8':
                for lot in list(lots):
                    if prev>=0 and np.isfinite(last[lot['s']]) and last[lot['s']]<lot['peak_price']*.92:
                        sell(lot,lot['q'],'open','trailing_stop8')
            open_value=sum(lot['q']*price(lot['s'],i,'open') for lot in lots);equity=cash+open_value
            excess=max(0,open_value-target*equity)
            for lot in list(lots):
                if excess<=1:break
                before=lot['q'];pr=price(lot['s'],i,'open');sell(lot,min(before,math.ceil(excess/pr)),'open','risk_budget')
                excess-=(before-lot['q'])*pr
            open_value=sum(lot['q']*price(lot['s'],i,'open') for lot in lots);equity=cash+open_value
            picks=signals.get(prev,[])
            due=prev+(2 if horizon==1 else horizon)
            if picks:
                budget=max(0,min(equity/max(horizon,1),cash,target*equity-open_value));per=budget/len(picks)
                for s in picks:
                    attempted+=1
                    if not fillable(s,'buy','open'):failed+=1;continue
                    pr=price(s,i,'open');existing=sum(lot['q']*pr for lot in lots if lot['s']==s)
                    adv=self.a['adv20'][prev,s]
                    amount=min(per,max(0,.1*equity-existing),max(0,adv*.01-used[s]) if np.isfinite(adv) else 0,cash/(1+fee))
                    q=math.floor(amount/(pr*(1+fee)))
                    if q<100:failed+=1;continue
                    n=q*pr;f=n*fee;cash-=n+f;turn+=n;cost+=f;used[s]+=n
                    lots.append({'s':s,'q':float(q),'basis':n+f,'entry':i,'due':due,'dividend':0.,'peak_price':pr})
                    if save_details:orders.append({'date':date,'stockid':self.ids[s],'side':'buy','shares':q,'price':pr,'when':'open','reason':'signal','fee':f})
            if horizon!=1:
                for lot in list(lots):
                    if i>=lot['due']:sell(lot,lot['q'],'close','scheduled')
            close=self.a['close'][i];valid=np.isfinite(close)&(close>0);last[valid]=close[valid]
            value=sum(lot['q']*last[lot['s']] for lot in lots);equity=cash+value
            for lot in lots:lot['peak_price']=max(lot['peak_price'],last[lot['s']])
            if cash < -.01 or not np.isfinite(equity):raise AssertionError('Invalid ledger equity/cash')
            peak=max(peak,equity);previous_dd=1-equity/peak;previous_equity=equity
            rows.append({'date':date,'cash':cash,'market_value':value,'equity':equity,'exposure':value/equity,'target_cap':target,
                         'turnover':turn/equity,'cost':cost,'gross_dividend_cash':divcash,'lots':len(lots)})
            if save_details:
                for s in sorted(set(lot['s'] for lot in lots)):
                    q=sum(lot['q'] for lot in lots if lot['s']==s);holdings.append({'date':date,'stockid':self.ids[s],'shares':q,'mark':last[s],'value':q*last[s]})
        daily=pd.DataFrame(rows);m=metrics(daily,capital,trades)
        m.update(horizon=horizon,topn=topn,policy=policy,guard=guard,cost_bp=cost_bp,attempted=attempted,failed_orders=failed,
                 open_lots_at_end=len(lots),terminal_rule='continue entries; mark open holdings at sample end; no future-price liquidation',
                 corporate_action_scenario='gross cash and all shares usable on ex-date' if actions else 'raw price only',
                 trading_limits='BJ 30% price proxy; minimum 100 shares; integer increments; T+1; past ADV capacity 1%; no leverage')
        return daily,m,pd.DataFrame(orders),pd.DataFrame(trades),pd.DataFrame(holdings)
