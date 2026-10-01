"""Check saved real-run ledgers, split boundaries, feature filters and audit hashes."""
import hashlib,json
import numpy as np
import pandas as pd
from prepare import RUN,ROOT,dump

def main():
    checked=0;max_error=0.;fold_count=0;max_correlation=0.
    for p in RUN.glob('*_account.csv'):
        d=pd.read_csv(p)
        error=float((d.cash+d.market_value-d.equity).abs().max())
        assert error<1e-6 and d.cash.min()>=-1e-6 and not d.date.duplicated().any()
        max_error=max(max_error,error);checked+=1
        prefix=p.name.removesuffix('_account.csv')
        order=RUN/(prefix+'_orders.csv');trade=RUN/(prefix+'_trades.csv')
        if order.exists():
            o=pd.read_csv(order);assert abs(o.cost.sum()-d.cost.sum())<1e-6
        if trade.exists():
            t=pd.read_csv(trade,parse_dates=['entry_date','exit_date']);assert (t.exit_date>t.entry_date).all()
    for market in ['bj','shsz']:
        folds=json.loads((RUN/(market+'_folds.json')).read_text(encoding='utf-8'))
        for f in folds:
            if 'skipped' in f:continue
            assert pd.Timestamp(f['train_label_end'])<pd.Timestamp(f['validation_start'])
            assert pd.Timestamp(f['refit_label_end'])<pd.Timestamp(f['test_first'])
            directory=RUN/market/f['quarter'];cards=pd.read_csv(directory/'factor_cards.csv',index_col=0)
            selected=f['selected_features'];assert len(selected)<=40
            assert (cards.loc[selected,'coverage']>=.7).all()
            corr=pd.read_csv(directory/'mean_daily_spearman.csv',index_col=0).loc[selected,selected].to_numpy()
            np.fill_diagonal(corr,0);value=float(np.abs(corr).max());assert value<.85+1e-10
            max_correlation=max(max_correlation,value);fold_count+=1
    fits=json.loads((RUN/'hmm_fits.json').read_text(encoding='utf-8'))
    assert all(pd.Timestamp(f['train_end'])<pd.Period(f['month']).start_time for f in fits)
    signals=pd.read_csv(RUN/'execution_signals.csv',parse_dates=['date','signal_available_at','earliest_execution_date'])
    assert not signals.duplicated(['date','stockid']).any()
    assert (signals.earliest_execution_date>signals.signal_available_at).all()
    frozen=json.loads((RUN/'frozen_old_run_hashes.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/'training_runs/lgbm_hmm_20260920'/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
    result={'passed':True,'actual_account_files_checked':checked,'max_cash_plus_value_error':max_error,
        'rolling_folds_checked':fold_count,'maximum_selected_pair_abs_mean_daily_spearman':max_correlation,
        'monthly_hmm_train_boundaries_checked':len(fits),'research_signals_checked':len(signals),
        'frozen_old_files_unchanged':len(frozen),'limitations':'Checks validate implementation, not historical source availability, real executions, corporate actions or profitability.'}
    dump(result,RUN/'output_verification.json');print(json.dumps(result))

if __name__=='__main__':main()
