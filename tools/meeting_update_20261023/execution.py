"""Five-session cohorts with one cash balance and fully explained order attempts.

Signal close t -> open t+1 entry -> predetermined close t+5 exit. Stops known
through previous close execute at the next open. Never borrow or refill a
stopped position inside a cohort. Prices/corporate actions are scenario inputs.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str = 'equal'
    topn: int = 20
    weighting: str = 'equal'
    stop: float = 0.0
    diversify: bool = False
    exposure: str = 'full'
    buy_hold: bool = False
    allow_residual: bool = False
    take_profit: float = 0.0
    cooldown: int = 0
    stop_basis: str = 'entry'
    group_keys: tuple = ('industry', 'size_group')


RULES = [Rule(), Rule('stop10', stop=.10),
         Rule('rank_stop10', weighting='rank', stop=.10),
         Rule('diversified_stop10', stop=.10, diversify=True),
         Rule('diversified_trend', stop=.10, diversify=True, exposure='trend'),
         Rule('diversified_vol12', stop=.10, diversify=True, exposure='vol12')]
CASH = Rule('cash', exposure='cash')


def performance(daily, capital=1e6, initial_date=None):
    if daily.empty:
        return {}
    e=daily.equity.to_numpy(float)
    r=np.r_[e[0]/capital-1, e[1:]/e[:-1]-1]
    dd=e/np.maximum.accumulate(np.r_[capital,e])[1:]-1
    start=pd.Timestamp(initial_date) if initial_date is not None else pd.Timestamp(daily.date.iloc[0])
    years=max((pd.Timestamp(daily.date.iloc[-1])-start).days/365.25,1/252)
    annual=(e[-1]/capital)**(1/years)-1
    sd=np.std(r,ddof=1) if len(r)>1 else 0
    rf=1.015**(1/252)-1
    mdd=-float(dd.min())
    return dict(start=str(daily.date.iloc[0])[:10], end=str(daily.date.iloc[-1])[:10],
                days=len(daily), total_return=e[-1]/capital-1, annual_return=annual,
                max_drawdown=mdd, sharpe_rf15=(r.mean()-rf)/sd*np.sqrt(252) if sd>1e-12 else 0.,
                sharpe_rf0=r.mean()/sd*np.sqrt(252) if sd>1e-12 else 0.,
                calmar=annual/mdd if mdd>1e-12 else None,
                mean_exposure=float(daily.exposure.mean()), total_cost=float(daily.cost.sum()),
                turnover=float(daily.turnover.sum()), end_equity=float(e[-1]),
                cash_days_fraction=float((daily.exposure<.01).mean()))


def period_metrics(daily, start, end, capital=1e6):
    part=daily[daily.date.between(pd.Timestamp(start),pd.Timestamp(end))]
    before=daily[daily.date<part.date.iloc[0]] if len(part) else daily.iloc[:0]
    return performance(part, float(before.equity.iloc[-1]) if len(before) else capital,
                       before.date.iloc[-1] if len(before) else None)


class CohortMarket:
    def __init__(self, prices, actions=None, metadata=None):
        prices=prices.copy(); prices.date=pd.to_datetime(prices.date)
        self.dates=pd.DatetimeIndex(sorted(prices.date.unique()))
        self.ids=sorted(prices.stockid.unique()); self.di={d:i for i,d in enumerate(self.dates)}
        self.si={s:i for i,s in enumerate(self.ids)}
        cols=['open','close','adv20','observed','mom20','vol20']
        self.a={c:prices.pivot(index='date',columns='stockid',values=c).reindex(index=self.dates,columns=self.ids).to_numpy() for c in cols}
        self.actions={}
        if actions is not None:
            for r in actions.itertuples(index=False):
                d=pd.Timestamp(r.date)
                if d in self.di and r.stockid in self.si:
                    self.actions.setdefault(self.di[d],[]).append((self.si[r.stockid],float(r.dividend),float(r.split)))
        self.meta={}
        if metadata is not None:
            metadata=metadata.copy(); metadata.date=pd.to_datetime(metadata.date)
            self.meta={(pd.Timestamp(r['date']),r['stockid']):r for r in metadata.to_dict('records')}
        # Only close-to-close past returns; no leading shifts or full-period fit.
        if 'in_pool' in prices:
            z=prices.loc[prices.in_pool].groupby('date')['adj_ret1' if 'adj_ret1' in prices else 'ret1'].mean()
        else:
            z=prices.groupby('date').apply(lambda g: 0.,include_groups=False)
        self.market_return=z.reindex(self.dates).fillna(0.)
        self.trend=(1+self.market_return).rolling(20,min_periods=10).apply(np.prod,raw=True)-1
        self.vol=self.market_return.rolling(20,min_periods=10).std()*np.sqrt(252)
        retcol='adj_ret1' if 'adj_ret1' in prices else 'ret1'
        self.asset_returns=prices.pivot(index='date',columns='stockid',values=retcol).reindex(index=self.dates,columns=self.ids).to_numpy()
        self.breadth=prices.loc[prices.in_pool].assign(positive=lambda f:f.mom20>0).groupby('date').positive.mean().reindex(self.dates).fillna(.5)


    def _groups(self,date,s):
        row=self.meta.get((date,self.ids[s]),{})
        aliases={'size_group':['size_group','size_bucket','size_tertile'], 'momentum_group':['momentum_group','mom_group','momentum_bucket','momentum_tertile'],
                 'volatility_group':['volatility_group','vol_group','volatility_bucket','volatility_tertile'], 'industry':['industry']}
        out={}
        for canonical,keys in aliases.items():
            val=next((row[k] for k in keys if k in row and pd.notna(row[k])), 'unknown')
            out[canonical]=str(val)
        return out

    def _picks(self,frame,date,rule):
        ordered=frame.sort_values(['score','stockid'],ascending=[False,True])
        picks=[]; counts={}; exclusions=[]
        for r in ordered.itertuples(index=False):
            if r.stockid not in self.si or not np.isfinite(r.score):continue
            s=self.si[r.stockid];groups=self._groups(date,s)
            rejected=''
            if rule.diversify:
                # Unknown industry is disclosed, never relabelled as a real sector.
                for key,val in groups.items():
                    if key not in rule.group_keys:continue
                    if val.lower() in ['unknown','nan','none','missing','未知']:
                        if key in ['size_group','industry']:
                            rejected='missing_historical_'+('size' if key=='size_group' else 'industry');break
                        continue
                    limit=.30 if key=='industry' else .50
                    if counts.get((key,val),0)>=max(1,math.floor(rule.topn*limit)):
                        rejected=key+'_quota';break
            if rejected:
                exclusions.append({'stockid':r.stockid,'reason':rejected});continue
            picks.append((s,float(r.score),groups))
            for key,val in groups.items():counts[(key,val)]=counts.get((key,val),0)+1
            if len(picks)>=rule.topn:break
        if not picks:return [],exclusions
        # Rank weighting's scale depends on requested TopN, so missing picks
        # deliberately leave cash instead of inflating the surviving weights.
        if rule.weighting=='rank':
            w=np.arange(rule.topn,rule.topn-len(picks),-1,dtype=float)/(rule.topn*(rule.topn+1)/2)
        else:w=np.full(len(picks),1/rule.topn)
        w=np.minimum(w,.10 if rule.topn<=20 else 1/rule.topn)
        if rule.diversify:
            for key in rule.group_keys:
                for val in sorted({x[2][key] for x in picks}):
                    if val.lower() in ['unknown','nan','none','missing','未知']:continue
                    idx=np.array([x[2][key]==val for x in picks]);limit=.30 if key=='industry' else .50
                    if w[idx].sum()>limit:w[idx]*=limit/w[idx].sum()
        return [(s,score,groups,float(weight)) for (s,score,groups),weight in zip(picks,w)],exclusions

    def target_exposure(self,i,rule,frame):
        if rule.exposure=='cash':return 0.
        if rule.exposure=='fixed25':return .25
        if rule.exposure=='fixed50':return .50
        if rule.exposure=='full':return 1.
        if rule.exposure=='ma20':
            level=(1+self.market_return.iloc[:i+1]).cumprod()
            return 1. if len(level)>=20 and level.iloc[-1]>=level.iloc[-20:].mean() else .5
        if rule.exposure=='trend':return .95 if float(self.trend.iloc[i])>0 else .25
        if rule.exposure=='vol12':
            v=float(self.vol.iloc[i]);return float(np.clip(.12/v,.10,.95)) if np.isfinite(v) and v>0 else .25
        if rule.exposure not in ['soft_trend','portfolio_vol','combined']:
            raise ValueError('Unknown exposure rule '+rule.exposure)
        history=self.market_return.iloc[max(0,i-59):i+1]
        v20=float(history.iloc[-20:].std()*np.sqrt(252));v60=float(history.std()*np.sqrt(252))
        v=max(v20,v60,1e-4)
        t20=float(np.prod(1+history.iloc[-20:])-1);t60=float(np.prod(1+history)-1)
        strength20=.5+.5*np.tanh(t20/(v*np.sqrt(20/252)+1e-8))
        strength60=.5+.5*np.tanh(t60/(v*np.sqrt(60/252)+1e-8))
        breadth=float(self.breadth.iloc[i])
        trend=float(np.clip(.25+.70*(strength20+strength60+breadth)/3,.25,.95))
        if rule.exposure=='soft_trend':return trend
        picks,_=self._picks(frame,self.dates[i],rule) if frame is not None else ([],[])
        if not picks:return .25
        sids=[s for s,score,groups,w in picks];weights=np.array([w for s,score,groups,w in picks]);weights/=weights.sum()
        past=self.asset_returns[max(0,i-59):i+1,sids]
        # Only trailing observed returns; zero for an unchanged/missing quote is a proxy disclosed in the report.
        rp=np.nan_to_num(past)@weights
        est=max(np.std(rp[-20:],ddof=1),np.std(rp,ddof=1))*np.sqrt(252)
        vol=float(np.clip(.15/max(est,1e-4),.25,.95))
        return min(trend,vol) if rule.exposure=='combined' else vol

    def run(self,scores,rule=Rule(),start='2024-10-08',end='2026-09-30',cost_bp=5.,
            capital=1e6,schedule=None,save_details=False,signal_delay=0):
        start=pd.Timestamp(start);end=pd.Timestamp(end)
        begin=self.di[start];finish=self.di[end];fee=cost_bp/1e4
        scoremap={pd.Timestamp(d):g for d,g in scores.dropna(subset=['score']).groupby('date')}
        # All accounts use the same phase for their own evaluation window.
        signals=set(range(begin,finish+1,5)); first_buy=True
        cash=float(capital);lots=[];next_id=0;last=np.full(len(self.ids),np.nan);banned={}
        rows=[];orders=[];trades=[];holds=[];selection=[];exposure_rows=[]
        if begin>0:
            last=np.where(np.isfinite(self.a['close'][begin-1]),self.a['close'][begin-1],np.nan)
        def mark(s,i,when):
            v=self.a[when][i,s]
            return float(v) if np.isfinite(v) and v>0 else float(last[s]) if np.isfinite(last[s]) else 0.
        def choose_rule(signal_i):
            if schedule is None:return rule
            eligible=[k for k in schedule if pd.Timestamp(k)<=self.dates[signal_i]]
            return schedule[max(eligible)] if eligible else rule
        for i in range(begin,finish+1):
            date=self.dates[i];prev=i-1;day_cost=day_turn=divcash=buyvalue=sellvalue=0.
            opening_cash=cash;used=np.zeros(len(self.ids));attempts=rejected=0
            # Freeze prior-close decisions BEFORE today's corporate action adjusts
            # cost bases. Comparing an unadjusted prior close to a split-adjusted
            # entry would invent a take-profit signal on the ex-date.
            triggers={}
            for lot in lots:
                ref=lot['peak'] if lot.get('stop_basis')=='peak' else lot['entry_price']
                previous=last[lot['s']]
                triggers[lot['id']]=(lot['stop']>0 and np.isfinite(previous) and previous<=ref*(1-lot['stop']),
                    lot.get('take_profit',0)>0 and np.isfinite(previous) and previous>=lot['entry_price']*(1+lot['take_profit']))
            for s,div,split in self.actions.get(i,[]):
                for lot in lots:
                    if lot['s']==s:
                        val=lot['q']*div;cash+=val;divcash+=val;lot['dividend']+=val
                        lot['q']*=split;lot['peak']=(lot['peak']-div)/split
                        lot['entry_price']=(lot['entry_price']-div)/split
            def availability(s,side,when):
                pr=self.a[when][i,s]
                if self.a['observed'][i,s]!=True:return 0.,'no_observed_bar'
                if not np.isfinite(pr) or pr<=0:return 0.,'invalid_price'
                ref=self.a['close'][prev,s] if prev>=0 else np.nan
                for sid,div,split in self.actions.get(i,[]):
                    if sid==s and np.isfinite(ref):ref=(ref-div)/split
                if np.isfinite(ref) and ((side=='buy' and pr>=ref*1.295) or (side=='sell' and pr<=ref*.705)):
                    return float(pr),'limit_price_proxy'
                return float(pr),''
            def log_order(s,side,when,reason,requested,filled,pr,fee_paid,reject,lot_id=None,**extra):
                nonlocal attempts,rejected
                attempts+=1;rejected+=int(filled<requested-1e-8)
                if save_details:
                    orders.append(dict(date=date,stockid=self.ids[s],side=side,when=when,reason=reason,
                                       requested_shares=requested,filled_shares=filled,price=pr,fee=fee_paid,
                                       reject_reason=reject,lot_id=lot_id,cash_after=cash,**extra))
            def sell(lot,when,reason,max_shares=None):
                nonlocal cash,day_cost,day_turn,sellvalue
                s=lot['s'];original_q=lot['q'];requested=original_q if max_shares is None else min(original_q,max_shares);pr,why=availability(s,'sell',when)
                if i<=lot['entry']:why='t_plus_one'
                adv=self.a['adv20'][prev,s] if prev>=0 else np.nan
                available=max(0,float(adv)*.01-used[s]) if np.isfinite(adv) else 0.
                q=0. if why else min(requested,math.floor(available/pr))
                if q<requested and not why:why='past_adv_capacity'
                if q>0:
                    value=q*pr;f=value*fee;basis=lot['basis']*q/original_q;div=lot['dividend']*q/original_q
                    cash+=value-f;day_cost+=f;day_turn+=value;sellvalue+=value;used[s]+=value
                    trades.append(dict(lot_id=lot['id'],stockid=self.ids[s],signal_date=self.dates[lot['signal']],
                                       entry_date=self.dates[lot['entry']],exit_date=date,entry_shares=q,
                                       exit_price=pr,entry_price=lot['entry_price'],pnl=value-f+div-basis,
                                       holding_session_intervals=i-lot['entry'],holding_sessions_inclusive=i-lot['entry']+1,
                                       reason=reason,basis_allocated=basis))
                    lot['q']-=q;lot['basis']-=basis;lot['dividend']-=div
                    if lot['q']<1e-8:lots.remove(lot)
                else:f=0.
                log_order(s,'sell',when,reason,requested,q,pr,f,why,lot['id'])
            # Only information already observed by the previous close triggers a stop.
            for lot in list(lots):
                if not lot.get('pending') and i>lot['entry'] and np.isfinite(last[lot['s']]):
                    if triggers[lot['id']][0]:
                        lot['pending']='stop_previous_close'
                        banned[lot['s']]=max(banned.get(lot['s'],-1),i+lot.get('cooldown',0))
                    elif triggers[lot['id']][1]:
                        lot['pending']='take_profit_previous_close'
                if lot.get('pending'):
                    sell(lot,'open',lot['pending'])
                elif i>lot['due']:
                    sell(lot,'open','overdue_exit')
            signal_i=prev
            rule_now=choose_rule(signal_i) if signal_i>=begin else rule
            # Daily market risk reduction uses the previous close; additions wait
            # for the next scheduled cohort. Partial sales allocate basis pro rata.
            daily_cap=self.target_exposure(prev,rule_now,None) if rule_now.exposure=='ma20' and prev>=0 else 1.
            if daily_cap<1 and lots:
                mv=sum(l['q']*mark(l['s'],i,'open') for l in lots)
                excess=max(0,mv-daily_cap*(cash+mv))
                fraction=excess/mv if mv>0 else 0
                if fraction>1e-6:
                    for lot in list(lots):sell(lot,'open','market_ma20_reduce',math.ceil(lot['q']*fraction))
            target=0.;desired_total=0.;unallocated=0.;planned_count=0;blocked_by_unsold=False
            enter=signal_i in signals and (not rule_now.buy_hold or first_buy)
            if enter:
                signal_date=self.dates[signal_i]
                lookup_i=signal_i-signal_delay
                frame=scoremap.get(self.dates[lookup_i]) if lookup_i>=0 else None
                if frame is not None:
                    blocked=frame.stockid.map(lambda s:banned.get(self.si.get(s,-1),-1)>i)
                    if save_details:
                        selection.extend(dict(signal_date=signal_date,rule=rule_now.name,stockid=s,reason='stop_cooldown') for s in frame.loc[blocked,'stockid'])
                    frame=frame.loc[~blocked]
                target=self.target_exposure(signal_i,rule_now,frame)
                blocked_by_unsold=bool(lots) and not rule_now.allow_residual
                if frame is not None and target>0 and not blocked_by_unsold:
                    picks,excluded=self._picks(frame,signal_date,rule_now)
                    if save_details:
                        selection.extend(dict(signal_date=signal_date,rule=rule_now.name,**e) for e in excluded)
                    open_value=sum(l['q']*mark(l['s'],i,'open') for l in lots);equity=cash+open_value
                    desired_total=target*equity;planned_count=len(picks)
                    for rank,(s,score,groups,weight) in enumerate(picks,1):
                        existing=sum(l['q']*mark(s,i,'open') for l in lots if l['s']==s)
                        desired=max(0,weight*target*equity-existing)
                        pr,why=availability(s,'buy','open')
                        requested=math.floor(desired/(pr*(1+fee))) if pr>0 else 0
                        adv=self.a['adv20'][signal_i,s]
                        capacity=max(0,float(adv)*.01-used[s]) if np.isfinite(adv) else 0.
                        spendable=max(0,min(desired,cash/(1+fee),capacity,
                                              max(0,target*equity-sum(l['q']*mark(l['s'],i,'open') for l in lots))))
                        q=0 if why else math.floor(spendable/(pr*(1+fee))) if pr>0 else 0
                        if rule_now.allow_residual and rule_now.diversify and pr>0:
                            for group_key,group_value in groups.items():
                                if group_key not in rule_now.group_keys:continue
                                if str(group_value).lower() in ['unknown','nan','none','missing','未知']:continue
                                cap=.30 if group_key=='industry' else .50
                                invested=sum(l['q']*mark(l['s'],i,'open') for l in lots if self._groups(signal_date,l['s'])[group_key]==group_value)
                                room=max(0.,cap*target*equity-invested)
                                q=min(q,math.floor(room/(pr*(1+fee))))
                        if q<100:
                            q=0
                            if not why:
                                why='cash_or_target_exhausted' if cash<100*max(pr,0)*(1+fee) or target<=0 else 'below_100_shares_after_limits'
                        elif q<requested and not why:why='capacity_cash_or_target_partial'
                        if q>0:
                            value=q*pr;f=value*fee;cash-=value+f;day_cost+=f;day_turn+=value;buyvalue+=value;used[s]+=value
                            lots.append(dict(id=next_id,s=s,q=float(q),basis=value+f,dividend=0.,entry=i,signal=signal_i,
                                             due=finish+1000 if rule_now.buy_hold else signal_i+5,
                                             entry_price=pr,peak=pr,stop=rule_now.stop,groups=groups,
                                             stop_basis=rule_now.stop_basis,take_profit=rule_now.take_profit,cooldown=rule_now.cooldown,pending=''))
                            next_id+=1
                        else:f=0.
                        log_order(s,'buy','open','new_cohort',requested,q,pr,f,why,next_id-1 if q else None,
                                  signal_date=signal_date,rank=rank,score=score,target_weight=weight*target,
                                  target_value=desired,rule=rule_now.name)
                    unallocated=max(0,desired_total-sum(l['q']*mark(l['s'],i,'open') for l in lots))
                    first_buy=False
            # Expiry was determined on the signal date, before this close existed.
            for lot in list(lots):
                if i>=lot['due']:sell(lot,'close','scheduled_fifth_session')
            closes=self.a['close'][i];ok=np.isfinite(closes)&(closes>0);last[ok]=closes[ok]
            value=sum(l['q']*last[l['s']] for l in lots);equity=cash+value
            for lot in lots:lot['peak']=max(lot['peak'],last[lot['s']])
            if cash<-.001 or not np.isfinite(equity):raise AssertionError('cash/equity invariant')
            if abs(cash-(opening_cash+divcash+sellvalue-buyvalue-day_cost))>1e-5:raise AssertionError('cash flow does not reconcile')
            rows.append(dict(date=date,cash=cash,market_value=value,equity=equity,exposure=value/equity,
                             opening_cash=opening_cash,buy_value=buyvalue,sell_value=sellvalue,cost=day_cost,
                             dividend_cash=divcash,turnover=day_turn/equity,orders_attempted=attempts,orders_limited=rejected,
                             signal_used=self.dates[signal_i] if enter else pd.NaT,target_exposure_on_entry=target if enter else np.nan,
                             planned_positions=planned_count,unallocated_target_value=unallocated,lots=len(lots),
                             effective_rule=rule_now.name,effective_topn=rule_now.topn,blocked_by_unsold=blocked_by_unsold,
                             market_cap_previous_close=daily_cap,active_bans=sum(v>i for v in banned.values())))
            if save_details:
                for lot in lots:
                    holds.append(dict(date=date,lot_id=lot['id'],stockid=self.ids[lot['s']],shares=lot['q'],mark=last[lot['s']],
                                      value=lot['q']*last[lot['s']],entry_date=self.dates[lot['entry']],due_index=lot['due']))
                by_group={}
                for lot in lots:
                    current=self._groups(date,lot['s'])
                    for key,v in current.items():by_group[(key,v)]=by_group.get((key,v),0.)+lot['q']*last[lot['s']]/equity
                exposure_rows.extend(dict(date=date,dimension=k,group=v,weight=w) for (k,v),w in by_group.items())
        d=pd.DataFrame(rows);t=pd.DataFrame(trades);m=performance(d,capital)
        m.update(rule='quarterly_validation_selected' if schedule is not None else rule.name,
                 topn=sorted({r.topn for r in schedule.values()}) if schedule is not None else rule.topn,
                 schedule_applied=schedule is not None,
                 schedule={k:asdict(v) for k,v in schedule.items()} if schedule is not None else None,
                 cost_bp=cost_bp,signal_delay=signal_delay,
                 completed_sale_parts=len(t),mean_holding_intervals=float(t.holding_session_intervals.mean()) if len(t) else None,
                 mean_holding_sessions_inclusive=float(t.holding_sessions_inclusive.mean()) if len(t) else None,
                 trade_win_rate=float((t.pnl>0).mean()) if len(t) else None,
                 ending_open_lots=len(lots),cash_identity_max_error=float((d.equity-d.cash-d.market_value).abs().max()))
        return dict(daily=d,metrics=m,orders=pd.DataFrame(orders),trades=t,holdings=pd.DataFrame(holds),
                    rejected_selection=pd.DataFrame(selection),risk_exposure=pd.DataFrame(exposure_rows))
