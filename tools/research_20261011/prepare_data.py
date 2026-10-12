"""New input vintage, causal price/LLM-proposed features and multiple labels."""
import io,json,re,sys
import numpy as np
import pandas as pd
from common import ROOT,DATA,RUN,HORIZONS,dump,sha
sys.path.insert(0,str(ROOT/'tools/research_20260930'))
import prepare as legacy

def main():
    legacy.DATA=DATA;legacy.RUN=RUN
    source=DATA/'benchmark'
    if not list(source.glob('000001.SH*')):raise RuntimeError('New benchmark input missing: finish audit_sync first')
    cal=pd.DatetimeIndex(pd.to_datetime(legacy.read_csv(next(source.glob('000001.SH*')))['日期'])).sort_values().unique()
    cal=cal[(cal>='2021-11-15')&(cal<='2026-10-09')]
    prices=legacy.prices('bj',cal).sort_values(['stockid','date']).reset_index(drop=True)
    parts=[]
    for p in sorted((DATA/'bj/hfq').glob('*.csv')):
        x=legacy.read_csv(p).rename(columns=legacy.RENAME);x['date']=pd.to_datetime(x.date)
        if x.duplicated('date').any():raise ValueError('duplicate adjusted price date')
        x['stockid']='BJ'+p.name[:6]
        parts.append(x[['date','stockid','open','high','low','close']].rename(columns={c:'adj_'+c for c in ['open','high','low','close']}))
    if not parts:raise RuntimeError('Adjusted price input missing')
    prices=prices.merge(pd.concat(parts),on=['date','stockid'],how='left',validate='one_to_one')
    # No backfill across dates, no unadjusted substitute for missing adjusted targets.
    g=prices.groupby('stockid',sort=False);a=prices.adj_close
    prices['adj_ret1']=a/g.adj_close.shift(1)-1
    prices['intraday_ret']=prices.adj_close/prices.adj_open-1
    prices['overnight_ret']=prices.adj_open/g.adj_close.shift(1)-1
    for h in [2,5,10,20,60,120]:
        prices[f'mom{h}']=a/g.adj_close.shift(h)-1
    for h in [5,10,20,60]:
        prices[f'vol{h}']=g.adj_ret1.transform(lambda s:s.rolling(h,min_periods=max(3,int(h*.8))).std())
    for h in [5,20,60]:
        prices[f'ma{h}']=g.adj_close.transform(lambda s:s.rolling(h,min_periods=int(h*.8)).mean())
    prices['drawdown60']=a/g.adj_close.transform(lambda s:s.rolling(60,min_periods=40).max())-1
    prices['range_today']=(prices.adj_high-prices.adj_low)/prices.adj_close
    prices['close_location']=(2*prices.adj_close-prices.adj_high-prices.adj_low)/(prices.adj_high-prices.adj_low).replace(0,np.nan)
    prices['downside20']=g.adj_ret1.transform(lambda s:s.clip(upper=0).pow(2).rolling(20,min_periods=16).mean().pow(.5))
    prices['amount_ratio']=g.amount.transform(lambda s:s.rolling(5,min_periods=4).mean())/prices.adv20.replace(0,np.nan)
    prices['log_illiq']=np.log((prices.adj_ret1.abs()/prices.amount.where(prices.amount>0)).groupby(prices.stockid).transform(lambda s:s.rolling(20,min_periods=16).mean()).clip(lower=1e-16))
    proposals=[
      ('risk_adjusted_momentum20','mom20 / (vol20 + 0.001)','Trend strength scaled by observable volatility'),
      ('risk_adjusted_momentum60','mom60 / (vol60 + 0.001)','Medium horizon trend risk adjustment'),
      ('short_reversal','-mom2','Short term overreaction hypothesis'),
      ('reversal_liquid','-mom5 * amount_ratio','Volume conditioned short term reversal'),
      ('quiet_trend','mom20 / (range_today + 0.01)','Trend in relatively quiet price ranges'),
      ('momentum_acceleration','mom5 - mom20 / 4','Change in recent trend speed'),
      ('momentum_curvature','mom20 - mom60 / 3','Intermediate trend acceleration'),
      ('low_downside','-downside20','Lower historical downside risk'),
      ('low_volatility','-vol20','Lower historical total volatility'),
      ('volatility_expansion','vol5 / (vol20 + 0.001)','Changes in short term volatility'),
      ('drawdown_recovery','mom5 - drawdown60','Recovery relative to trailing drawdown'),
      ('close_pressure','close_location * amount_ratio','Price location and volume pressure'),
      ('overnight_reversal','-overnight_ret','Gap reversal hypothesis'),
      ('intraday_strength','intraday_ret / (range_today + 0.01)','Within-day buying pressure'),
      ('trend_break','ma5 / ma20 - 1','Recent versus intermediate moving average'),
      ('medium_trend','ma20 / ma60 - 1','Intermediate versus long moving average'),
      ('volume_momentum','mom10 * amount_ratio','Trend with volume confirmation'),
      ('liquidity_stability','-log_illiq','Amihud based liquidity'),
      ('downside_adjusted_momentum','mom20 / (downside20 + 0.001)','Trend adjusted for downside volatility'),
      ('gap_pressure','overnight_ret - intraday_ret','Gap versus intraday price pressure'),
      ('reversal_in_trend','mom60 - mom5','Longer trend following with short reversal'),
      ('range_reversal','-intraday_ret * range_today','Large range overreaction'),
      ('trend_drawdown','mom20 + drawdown60','Penalize trend with persistent drawdowns'),
      ('momentum_balance','mom5 / (vol5 + 0.001) - mom20 / (vol20 + 0.001)','Change in risk-adjusted trend')]
    from factor_dsl import evaluate
    for name,expression,_ in proposals:prices['llm_'+name]=evaluate(expression,prices)
    registry=[{'id':'llm_'+n,'expression':e,'hypothesis':h,'proposed_by':'Codex during this research turn; not imported RD-Agent output',
               'information_cutoff':'close t','screening':'training-period only, future-window purge'} for n,e,h in proposals]
    dump(registry,RUN/'llm_factor_proposals.json')
    for h in HORIZONS:
        entry=g.adj_open.shift(-1)
        exit_price=g.adj_open.shift(-2) if h==1 else g.adj_close.shift(-h)
        prices[f'label_{h}']=exit_price/entry-1
        prices[f'label_exit_{h}']=prices.date.groupby(prices.stockid).shift(-2 if h==1 else -h)
        prices[f'raw_label_{h}']=(g.open.shift(-2) if h==1 else g.close.shift(-h))/g.open.shift(-1)-1
    factors=pd.read_parquet(ROOT/'training_data/research_20261008/bj_factor_panel.parquet')
    updated=[]
    for suffix in ['c01','c02']:
        chunks=[]
        for p in sorted((DATA/'factors'/suffix).glob('*/*.zip')):
            x=legacy.read_zip(p,'BJ');x=x[x.stockid.str.fullmatch(r'BJ\d{6}')].copy();x['date']=pd.to_datetime(p.stem)
            cols=[c for c in x if c not in ['date','stockid','enddate','rdate']]
            ren={c:'c01_Alpha'+c.rsplit('_',1)[-1] if suffix=='c01' else suffix+'_'+c for c in cols}
            x=x[['date','stockid']+cols].rename(columns=ren);x[list(ren.values())]=x[list(ren.values())].apply(pd.to_numeric,errors='coerce')
            chunks.append(x.set_index(['date','stockid']))
        if chunks:updated.append(pd.concat(chunks))
    if updated:
        new=pd.concat(updated,axis=1).reset_index()
        factors=pd.concat([factors,new]).drop_duplicates(['date','stockid'],keep='last')
    factorcols=[c for c in factors if c.startswith('c0')]
    factors=factors.replace([np.inf,-np.inf],np.nan)
    full=prices.merge(factors,on=['date','stockid'],how='left',validate='one_to_one')
    # Store full-calendar lagged factors before selecting dynamic pool; no sparse-membership lag.
    for c in factorcols:full['lag1_'+c]=full.groupby('stockid')[c].shift(1)
    panel=full[full.in_pool].copy()
    causal=['mom2','mom5','mom10','mom20','mom60','mom120','vol5','vol10','vol20','vol60','adv20','amount_ratio','range_today','log_illiq','drawdown60','close_location','overnight_ret','intraday_ret','downside20']
    llm=[x['id'] for x in registry]
    cols=factorcols+causal+llm
    ranked=panel[cols].replace([np.inf,-np.inf],np.nan).groupby(panel.date).rank(pct=True).sub(.5).astype('float32');ranked.columns=['x_'+c for c in cols]
    panel=pd.concat([panel,ranked],axis=1)
    for h in HORIZONS:panel[f'target_{h}']=panel.groupby('date')[f'label_{h}'].rank(pct=True).sub(.5)
    panel['factor_coverage']=panel[factorcols].notna().mean(axis=1)
    keep=['date','stockid','adv20','close','open','vol20','ret1','adj_ret1','mom20','mom60','drawdown60','factor_coverage']
    keep +=[f'{prefix}_{h}' for h in HORIZONS for prefix in ['label','raw_label','label_exit','target']]+list(ranked.columns)
    panel[keep].to_parquet(DATA/'model_panel.parquet',index=False)
    prices.to_parquet(DATA/'prices.parquet',index=False)
    fields=panel.groupby('date').agg(stocks=('stockid','size'),factor_coverage=('factor_coverage','mean'))
    fields.to_csv(RUN/'data_coverage.csv')
    actions=[]
    p=DATA/'dividend_BJ.parquet'
    if p.exists():
        a=pd.read_parquet(p);a['stockid']='BJ'+a.symbol.str[:6]
        a['date']=pd.to_datetime(a.exDividendDate,errors='coerce');record=pd.to_datetime(a.equityRecordDate,errors='coerce')
        a=a[(a.progress=='实施分配')&a.date.notna()&(a.date<=cal[-1])&(record<a.date)].copy()
        a['dividend']=a.cashDividendPretaxPer10.fillna(0)/10
        a['split']=1+a.bonusTotalPer10.fillna(a.bonusPer10.fillna(0)+a.transferPer10.fillna(0))/10
        if a.duplicated(['date','stockid']).any():raise ValueError('Duplicate corporate actions require review')
        a[['date','stockid','dividend','split']].to_csv(DATA/'actions.csv',index=False)
    dump({'last_date':str(cal[-1].date()),'price_stocks':prices.stockid.nunique(),'pool_rows':len(panel),'features':len(cols),'llm_proposals':len(proposals),
          'horizons':HORIZONS,'label_rule':'h>=2: open t+1 to close t+h; h=1: open t+1 to open t+2 (T+1)',
          'factor_vintage':'current supplier files; historical release times and revisions not verified',
          'execution':'raw prices; ex-date immediate gross cash/share scenario separately identified',
          'panel_sha256':sha(DATA/'model_panel.parquet'),'prices_sha256':sha(DATA/'prices.parquet')},RUN/'data_summary.json')
    print('prepared',len(prices),len(panel),len(cols),'last',str(cal[-1].date()),flush=True)

if __name__=='__main__':main()
