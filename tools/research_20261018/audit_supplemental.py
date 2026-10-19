"""Independent reconciliation of new model experiments and evidence tables."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from audit_deep_risk import metric_calc
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261018'

def main():
    errors=[];counts={'accounts':0,'days':0,'orders':0};maxgap=0.
    dates=pd.DatetimeIndex(pd.read_parquet(ROOT/'training_data/research_20261011/prices.parquet',columns=['date']).date.unique()).sort_values()
    di={d:i for i,d in enumerate(dates)}
    actions=pd.read_csv(ROOT/'training_data/research_20261011/actions.csv',parse_dates=['date'])
    am={(r.date,r.stockid):(float(r.dividend),float(r.split)) for r in actions.itertuples(index=False)}
    cases=[]
    for exp in ['factor_loop','hmm_conditional','minute_incremental']:
        summary=pd.read_csv(RUN/exp/'account_results.csv')
        for row in summary.to_dict('records'):
            name=row['strategy']
            paths={k:(RUN/exp/(name+'_'+k+'.csv') if exp=='minute_incremental' else RUN/exp/'accounts'/name/(k+'.csv')) for k in ['daily','orders','holdings','trades']}
            cases.append((exp+'/'+name,paths,row))
    for d in (RUN/'evidence/benchmarks').glob('*/*'):
        cases.append(('benchmark/'+str(d.relative_to(RUN/'evidence/benchmarks')),{k:d/(k+'.csv') for k in ['daily','orders','holdings','trades']},None))
    def check(name,condition,detail):
        if not condition:errors.append({'case':name,'error':detail})
    for name,paths,summary in cases:
        d=pd.read_csv(paths['daily'],parse_dates=['date'])
        o=pd.read_csv(paths['orders'],parse_dates=['date','signal_date'])
        h=pd.read_csv(paths['holdings'],parse_dates=['date'])
        counts['accounts']+=1;counts['days']+=len(d);counts['orders']+=len(o)
        gaps=[float((d.equity-d.cash-d.market_value).abs().max()),
              float((d.cash-(d.opening_cash+d.dividend_cash+d.sell_value-d.buy_value-d.cost)).abs().max()),
              float((d.market_value-h.groupby('date').value.sum().reindex(d.date,fill_value=0).to_numpy()).abs().max())]
        maxgap=max(maxgap,*gaps);check(name,max(gaps)<.01,'daily identities')
        lots={};last_cash=1e6;og={day:g for day,g in o.groupby('date')};hg={day:g for day,g in h.groupby('date')}
        for row in d.itertuples(index=False):
            dividend=0.
            for state in lots.values():
                dv,sp=am.get((row.date,state['stockid']),(0.,1.))
                dividend+=state['qty']*dv;state['qty']*=sp
            check(name,abs(dividend-row.dividend_cash)<.01,'dividend '+str(row.date))
            check(name,abs(last_cash-row.opening_cash)<.01,'opening cash '+str(row.date))
            cash=last_cash+dividend
            for order in og.get(row.date,o.iloc[:0]).itertuples(index=False):
                qty=float(order.filled_shares);value=qty*order.price
                cash+=(value-order.fee) if order.side=='sell' else (-value-order.fee)
                check(name,abs(cash-order.cash_after)<.01,'order cash '+str(row.date))
                if qty<=0:continue
                lid=int(order.lot_id)
                if order.side=='buy':
                    check(name,di[row.date]-di[order.signal_date]==1 and order.when=='open','next open')
                    check(name,(di[order.signal_date]-di[d.date.iloc[0]])%5==0,'five-session signal phase')
                    lots[lid]={'stockid':order.stockid,'qty':qty,'entry':row.date}
                else:
                    check(name,lid in lots,'sale position exists')
                    state=lots[lid];check(name,di[row.date]>di[state['entry']],'T+1')
                    state['qty']-=qty;check(name,state['qty']>=-.00001,'no overselling')
                    if state['qty']<1e-8:del lots[lid]
            check(name,abs(cash-row.cash)<.01,'close cash');last_cash=cash
            dayh=hg.get(row.date,h.iloc[:0]);check(name,set(lots)==set(dayh.lot_id.astype(int)),'holding lots')
            for held in dayh.itertuples(index=False):
                check(name,abs(lots[int(held.lot_id)]['qty']-held.shares)<.00001,'holding quantity')
        if summary:
            m=metric_calc(d,1e6,d.date.iloc[0])
            for key in ['total_return','max_drawdown','sharpe_rf15','calmar','annual_return','end_equity']:
                check(name,np.isclose(m[key],summary[key],rtol=1e-9,atol=1e-8),'metric '+key)
            check(name,summary['cash_identity_max_error']<.01,'reported identity diagnostic')
    extra=pd.read_csv(RUN/'evidence/all_periods_with_excess.csv')
    for ref in ['equal_buy_hold','equal_pool_5day','bj50_price_index']:
        check('all excess',np.allclose(extra['excess_pp_'+ref],extra.total_return-extra['benchmark_return_'+ref]),ref)
    attrib=json.loads((RUN/'evidence/drawdown_attribution.json').read_text(encoding='utf-8'))
    for item in attrib:check(item['strategy'],abs(item['identity_error'])<.01,'attribution')
    freeze=json.loads((RUN/'factor_loop/frozen_selection.json').read_text(encoding='utf-8'))
    check('factor_loop',pd.Timestamp(freeze['formal_train_last_exit'])<pd.Timestamp(freeze['test_start']),'formal maturity')
    check('factor_loop',freeze['independent_rows_loaded_before_freeze'] is False,'freeze ordering record')
    monthly=json.loads((RUN/'hmm_conditional/monthly_training_audit.json').read_text(encoding='utf-8'))
    check('hmm_conditional',all(pd.Timestamp(x['train_last_label_exit'])<pd.Timestamp(x['first_score']) for x in monthly),'mature labels before predictions')
    report={'status':'pass' if not errors else 'issues',**counts,'max_daily_identity_error':maxgap,
            'excess_rows':len(extra),'reconciled_drawdown_intervals':len(attrib),'errors':errors,
            'scope':'Independent saved-ledger reconciliation; not source publication/vintage verification.'}
    (RUN/'evidence/supplemental_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if errors:raise AssertionError('supplemental audit failed')

if __name__=='__main__':main()
