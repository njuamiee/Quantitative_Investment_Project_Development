"""Post-search diagnostic controls; never enter the monthly candidate selector."""
import numpy as np
import pandas as pd
from config import RUN,START,END,dump
from account import market_inputs,score_order,ledger,metrics

def lagged_exposure_control(days,curves):
    cal,m,args=market_inputs();begin=cal.get_loc(START);finish=cal.get_loc(END)
    meta=pd.read_parquet(RUN/'data/meta.parquet');pool=meta[meta.date.between(START,END)].copy();pool['score']=0.
    order=score_order(pool,cal,m['open'].shape[1])
    # The existing volatility-budget input maps deterministically to a desired
    # exposure. Its value at t uses only the primary account's close at t;
    # ledger reads it at t+1. No forward return or future position is used.
    exposure=curves['rolling_cash_fallback_fee5'][:,4]
    budget=np.full(len(cal),.15/1e-12)
    budget[begin:finish+1]=.15/np.maximum(exposure,1e-12)
    control_args=list(args);control_args[13]=budget
    # Equal weight, no HMM/stop/entry/concentration filter, dynamic100, daily rebalance.
    params=[100,0,2,0,0,0,0,100,1]
    extra={};rows=[]
    for fee in [5.,15.]:
        daily,error=ledger(order,*control_args,begin,finish,*params,fee/1e4,1e6)
        name=f'lagged_exposure_equal_fee{int(fee)}';extra[name]=daily
        for label,a,b in [('common',0,len(days)),('recent',int(days.searchsorted('2026-05-22')),len(days))]:
            capital=1e6 if a==0 else daily[a-1,0]
            row=metrics(daily[a:b,0],days[a:b],capital,days[a-1] if a else None)
            row.update(strategy=name,period=label,mean_exposure=float(daily[a:b,4].mean()),cash_flow_error=error);rows.append(row)
        frame=pd.DataFrame(daily,columns=['equity','cash','market_value','cost','exposure','buy_value','sell_value','attempts','limited','target'])
        frame.insert(0,'date',days);frame.to_parquet(RUN/'selection'/(name+'.parquet'),index=False)
    # An explicit causal mutation check for this added diagnostic.
    changed=list(control_args);changed[13]=budget.copy();cut=begin+len(days)//2;changed[13][cut:]*=.5
    check,_=ledger(order,*changed,begin,finish,*params,.0005,1e6)
    np.testing.assert_array_equal(check[:cut-begin+1],extra['lagged_exposure_equal_fee5'][:cut-begin+1])
    dump(dict(added_after_primary_search=True,not_a_selector_candidate=True,additional_accounts=2,
        target='primary fee5 account previous-close actual equity exposure; same target for both fee scenarios',
        actual_exposure_can_differ_due_to_fills=True,future_budget_mutation_check=True,
        causal_unchanged_days=cut-begin+1),RUN/'selection/exposure_diagnostic.json')
    return extra,rows
