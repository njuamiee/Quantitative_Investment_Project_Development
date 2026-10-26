"""Copy the original ledger into this isolated version and expose a causal sizing hook."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'tools/research_20261017/execution.py').read_text(encoding='utf-8-sig')
old="""                target=1.
                if rule_now.exposure=='cash':target=0.
                elif rule_now.exposure=='trend':target=.95 if float(self.trend.iloc[signal_i])>0 else .25
                elif rule_now.exposure=='vol12':
                    v=float(self.vol.iloc[signal_i]);target=float(np.clip(.12/v,.10,.95)) if np.isfinite(v) and v>0 else .25"""
assert old in source
source=source.replace(old,"                target=self.target_exposure(signal_i,rule_now,frame)")
source=source.replace("    buy_hold: bool = False","    buy_hold: bool = False\n    allow_residual: bool = False")
source=source.replace("                blocked_by_unsold=bool(lots)","                blocked_by_unsold=bool(lots) and not rule_now.allow_residual")
needle="                        if q<100:"
assert needle in source
source=source.replace(needle,"""                        if rule_now.allow_residual and rule_now.diversify and pr>0:
                            for group_key,group_value in groups.items():
                                if str(group_value).lower() in ['unknown','nan','none','missing','未知']:continue
                                cap=.30 if group_key=='industry' else .50
                                invested=sum(l['q']*mark(l['s'],i,'open') for l in lots if self._groups(signal_date,l['s'])[group_key]==group_value)
                                room=max(0.,cap*target*equity-invested)
                                q=min(q,math.floor(room/(pr*(1+fee))))
"""+needle)
method='''    def target_exposure(self,i,rule,frame):
        if rule.exposure=='cash':return 0.
        if rule.exposure=='fixed25':return .25
        if rule.exposure=='fixed50':return .50
        if rule.exposure=='full':return 1.
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

'''
needle='    def run(self,scores,rule=Rule(),'
assert needle in source
source=source.replace(needle,method+needle)
needle="        self.vol=self.market_return.rolling(20,min_periods=10).std()*np.sqrt(252)"
assert needle in source
source=source.replace(needle,needle+'''
        retcol='adj_ret1' if 'adj_ret1' in prices else 'ret1'
        self.asset_returns=prices.pivot(index='date',columns='stockid',values=retcol).reindex(index=self.dates,columns=self.ids).to_numpy()
        self.breadth=prices.loc[prices.in_pool].assign(positive=lambda f:f.mom20>0).groupby('date').positive.mean().reindex(self.dates).fillna(.5)
''')
(Path(__file__).parent/'risk_execution.py').write_text(source,encoding='utf-8')
print('Built isolated ledger; original matching/fees/T+1/capacity retained')
