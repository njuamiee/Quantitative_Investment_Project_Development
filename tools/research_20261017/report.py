"""Build auditable Chinese explanations, figures and a self-contained HTML report."""
from pathlib import Path
import base64,json,html,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from execution import performance

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'training_runs/research_20261017';OUT=ROOT/'新项目_20261017'
NAMES={'new_main_monthly':'新主流程：月度重训＋过去验证选风控','new_quarterly_same_risk':'新模型：季度重训＋相同风控',
       'new_monthly_equal':'新模型：月度重训＋等权','new_quarterly_equal':'新模型：季度重训＋等权',
       'new_monthly_stop10':'新模型：等权＋10%跟踪止损','new_monthly_diversified':'新模型：分组限制＋止损',
       'simple_ic20_equal':'简单IC20组合＋等权','simple_ic20_same_risk':'简单IC20组合＋相同风控',
       'equal_pool_5day':'整个股票池：五日等权换仓','equal_buy_hold':'期初股票池：等额买入持有',
       'old_lgbm_same_execution':'旧LightGBM分数＋新交易规则','old_main':'旧主流程','bj50_price_index':'北证50价格指数'}


def fmt(x,kind='ratio'):
    if x is None or pd.isna(x):return '—'
    return f'{x:.2%}' if kind=='pct' else f'{x:,.2f}'


def mdtable(frame):
    cols=[str(c) for c in frame.columns]
    result='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
    return result+'\n'.join('| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def load(name):return pd.read_csv(RUN/name,encoding='utf-8-sig')


def stock_attribution(start='2026-04-01',end='2026-09-30',tag='recent_half'):
    folder=RUN/'accounts/new_main_monthly';d=pd.read_csv(folder/'daily.csv',parse_dates=['date'])
    h=pd.read_csv(folder/'holdings.csv',parse_dates=['date']);o=pd.read_csv(folder/'orders.csv',parse_dates=['date'])
    before=d[d.date<pd.Timestamp(start)];initial_day=before.date.iloc[-1]
    initial=h[h.date==initial_day].groupby('stockid').value.sum();terminal=h[h.date==pd.Timestamp(end)].groupby('stockid').value.sum()
    rows=o[o.date.between(start,end)].copy();rows['notional']=rows.filled_shares*rows.price
    buy=rows[rows.side=='buy'].groupby('stockid').notional.sum();sell=rows[rows.side=='sell'].groupby('stockid').notional.sum();fees=rows.groupby('stockid').fee.sum()
    div={};actions=pd.read_csv(ROOT/'training_data/research_20261011/actions.csv',parse_dates=['date'])
    dates=pd.DatetimeIndex(sorted(d.date.unique()))
    for a in actions[actions.date.between(start,end)].itertuples(index=False):
        past=dates[dates<a.date]
        if not len(past):continue
        shares=h[(h.date==past[-1])&(h.stockid==a.stockid)].shares.sum()
        if shares:div[a.stockid]=div.get(a.stockid,0.)+shares*a.dividend
    result=pd.concat({'start_value':initial,'end_value':terminal,'buys':buy,'sells':sell,'fees':fees,'dividends':pd.Series(div,dtype=float)},axis=1).fillna(0.)
    result['price_pnl']=result.end_value-result.start_value+result.sells-result.buys
    result['net_pnl']=result.price_pnl+result.dividends-result.fees
    expected=d.loc[d.date==pd.Timestamp(end),'equity'].iloc[0]-before.equity.iloc[-1]
    assert abs(result.net_pnl.sum()-expected)<.01,(result.net_pnl.sum(),expected)
    result.sort_values('net_pnl').to_csv(RUN/(tag+'_stock_attribution.csv'),encoding='utf-8-sig',index_label='stockid')
    return result.sort_values('net_pnl'),expected


def main():
    OUT.mkdir(exist_ok=True);figs=OUT/'图表';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
    all_daily=pd.read_parquet(RUN/'all_daily.parquet');all_daily.date=pd.to_datetime(all_daily.date)
    summary=load('strategy_summary.csv').set_index('strategy');period=load('period_comparison.csv');folds=load('models/folds.csv')
    choices=load('risk_choices.csv'); mainrow=summary.loc['new_main_monthly']; old=summary.loc['old_main']
    riskmeta=pd.read_parquet(RUN/'risk_metadata.parquet',columns=['date','stockid','in_pool','industry'])
    meta_scope=riskmeta[riskmeta.in_pool & riskmeta.date.between('2024-10-08','2026-09-30')]
    known_fraction=meta_scope.industry.ne('unknown').mean()
    def series(name):return all_daily[all_daily.strategy==name].set_index('date').sort_index()
    def save(name):plt.tight_layout();plt.savefig(figs/(name+'.png'));plt.close()
    colors=['#111111','#888888','#bbbbbb'];styles=['-','--',':']
    for mode,name in [('equity','01_三条资金曲线'),('drawdown','02_三条回撤曲线')]:
        fig,ax=plt.subplots(figsize=(11,4.5))
        for n,c,st in zip(['new_main_monthly','equal_buy_hold','bj50_price_index'],colors,styles):
            g=series(n);y=g.equity/1e4 if mode=='equity' else 100*(g.equity/np.maximum.accumulate(np.r_[1e6,g.equity])[1:]-1)
            ax.plot(g.index,y,label=NAMES[n],color=c,ls=st,lw=1.7)
        ax.set_ylabel('账户金额（万元）' if mode=='equity' else '相对此前最高资产的跌幅（%）')
        ax.grid(alpha=.18);ax.legend(loc='best',fontsize=9);ax.set_title('共同起点：2024-10-08；结束：2026-09-30')
        save(name)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for n,c,st in zip(['new_main_monthly','old_main','new_monthly_equal'],colors,styles):
        g=series(n);axes[0].plot(g.index,g.equity/1e4,label=NAMES[n],color=c,ls=st);axes[1].plot(g.index,g.exposure.rolling(20,min_periods=1).mean()*100,color=c,ls=st)
    axes[0].set_ylabel('万元');axes[1].set_ylabel('最近20日平均仓位（%）');axes[0].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.18)
    save('03_新旧账户与仓位')
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for ax,f in zip(axes.ravel(),folds.itertuples(index=False)):
        curve=pd.read_csv(RUN/'models'/f.quarter/'loss_curves.csv')
        # Accept the model module's stable column names after inspecting output.
        namecol='candidate' if 'candidate' in curve else 'name' if 'name' in curve else 'candidate_name'
        c=curve[curve[namecol]==f.selected_name]
        if 'dataset' in c:
            for ds,part in c.groupby('dataset'):
                ycol='l2' if 'l2' in part else 'loss';xcol='iteration' if 'iteration' in part else 'round'
                ax.plot(part[xcol],part[ycol],label=ds)
        else:
            xcol='iteration' if 'iteration' in c else 'round'
            for col in [x for x in c if 'l2' in x or 'loss' in x]:
                training='training' in col
                ax.plot(c[xcol],c[col],label='训练误差' if training else '验证误差',color='#111111' if training else '#888888',ls='-' if training else '--')
        ax.axvline(f.selected_best_iteration,color='black',ls=':',lw=.8);ax.set_title(f'{f.quarter}：选{f.selected_best_iteration}轮',fontsize=11);ax.legend(fontsize=8);ax.grid(alpha=.15)
        ax.set_xlabel('训练轮数');ax.set_ylabel('平均平方误差')
    save('04_训练与验证误差')
    import matplotlib.dates as mdates
    fig,ax=plt.subplots(figsize=(12,5))
    bars=[('内部训练：24,690行','2024-09-03','2025-09-09'),('验证：12,600评分/12,095答案','2025-09-17','2026-03-31'),
          ('选好后正式训练：24,681行','2025-03-18','2026-03-24'),('下一季度：只给当天股票评分','2026-04-01','2026-06-30')]
    for i,(label,a,b) in enumerate(bars):
        left=mdates.date2num(pd.Timestamp(a));right=mdates.date2num(pd.Timestamp(b))
        ax.broken_barh([(left,right-left)],(3-i-.22,.44),facecolors=['#333333','#888888','#555555','#bbbbbb'][i])
        ax.text((left+right)/2,3-i+.30,f'{a} 至 {b}',ha='center',fontsize=9)
    ax.set_yticks(range(4),[x[0] for x in bars[::-1]]);ax.xaxis_date();ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(axis='x',alpha=.15);ax.set_ylim(-.5,3.8)
    ax.set_title('真实例子：2026年二季度模型如何产生\n训练行的信号日期如图；对应答案必须在下一段开始前结束',fontsize=12)
    save('08_训练验证正式预测时间轴')
    exposure=pd.read_csv(RUN/'accounts/new_main_monthly/risk_exposure.csv',parse_dates=['date'])
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for ax,dimension,title in zip(axes,['size_group','momentum_group','volatility_group'],['规模分布（股票市值占总资产）','过去20日涨跌分组','过去20日波动分组']):
        g=exposure[exposure.dimension==dimension].pivot_table(index='date',columns='group',values='weight',aggfunc='sum',fill_value=0.)
        g=g.reindex(series('new_main_monthly').index,fill_value=0.)
        groupnames={'large':'大规模','middle':'中间组','small':'小规模','high':'高组','low':'低组','unknown':'资料未知'}
        ax.stackplot(g.index,g.to_numpy().T*100,labels=[groupnames.get(x,x) for x in g.columns],colors=['#202020','#777777','#bbbbbb','#eeeeee'][:len(g.columns)])
        ax.set_ylim(0,100);ax.set_title(title);ax.legend(loc='upper left',fontsize=8,ncol=4);ax.set_ylabel('%')
    save('05_每日规模与风格监测')
    industry=exposure[exposure.dimension=='industry'].pivot_table(index='date',columns='group',values='weight',aggfunc='sum',fill_value=0.)
    industry=industry.reindex(series('new_main_monthly').index,fill_value=0.)
    largest=industry.max(axis=1)*100
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(largest.index,largest,label='每日占资产最多的一个行业',color='#111111')
    ax.axhline(30,color='#888888',ls='--',label='分散方案建仓额度30%（并非所有季度启用）')
    ax.set_ylabel('占账户总资产（%）');ax.set_ylim(0,100);ax.legend(fontsize=9);ax.grid(alpha=.18)
    ax.set_title('每日行业集中度：按当日可用历史分类监控');save('07_行业集中度监测')
    half_attrib,half_delta=stock_attribution()
    account=series('new_main_monthly');dd=account.equity/account.equity.cummax()-1
    trough=dd.idxmin();peak=account.loc[:trough,'equity'].idxmax();loss_start=account.index[account.index.get_loc(peak)+1]
    attrib,delta=stock_attribution(str(loss_start.date()),str(trough.date()),'max_drawdown');toploss=attrib.head(5).copy()
    loss_table=toploss[['start_value','end_value','net_pnl']].reset_index().rename(columns={'index':'股票','stockid':'股票','start_value':'阶段起点持股市值','end_value':'阶段终点持股市值','net_pnl':'净盈亏（元）'})
    for c in loss_table.columns[1:]:loss_table[c]=loss_table[c].map(lambda v:f'{v:,.2f}')
    fig,ax=plt.subplots(figsize=(10,4));parts=pd.concat([toploss.net_pnl,pd.Series({'其他股票':attrib.iloc[5:].net_pnl.sum()})])
    ax.bar(parts.index,parts.values/1e4,color=['#333333' if v<0 else '#aaaaaa' for v in parts.values]);ax.axhline(0,color='black',lw=.7)
    ax.set_ylabel('对最大回撤阶段账户变化的贡献（万元）');ax.set_title(f'{peak:%Y-%m-%d}→{trough:%Y-%m-%d}逐股对账：合计{delta/1e4:+.2f}万元');save('06_最大回撤盈亏来源')
    # Real executed order example, not hand-invented sample numbers.
    orders=pd.read_csv(RUN/'accounts/new_main_monthly/orders.csv',parse_dates=['date','signal_date'])
    possible=orders[(orders.side=='buy')&(orders.filled_shares>0)&(orders.date>='2026-03-01')]
    if possible.empty:possible=orders[(orders.side=='buy')&(orders.filled_shares>0)]
    example_date=possible.date.iloc[0];example=orders[(orders.date==example_date)&(orders.side=='buy')]
    example.to_csv(OUT/'真实交易日_下单明细.csv',index=False,encoding='utf-8-sig')
    day=series('new_main_monthly').loc[example_date]
    # Small printable Chinese comparison file.
    show=period.copy()
    show['策略']=show.strategy.map(NAMES);show['时期']=show.period.map({'full':'全期','recent_year':'最近一年','recent_half':'最近半年'})
    cols={'total_return':'累计收益','annual_return':'年化收益','max_drawdown':'最大回撤','sharpe_rf15':'夏普（无风险1.5%）','calmar':'卡玛','mean_exposure':'平均仓位','excess_vs_equal_buy_hold':'相对买入持有超额（百分点）','excess_vs_bj50_price_index':'相对北证50超额（百分点）'}
    show[['excess_vs_equal_buy_hold','excess_vs_bj50_price_index']]*=100
    show=show[['策略','时期']+list(cols)].rename(columns=cols);show.to_csv(OUT/'实验结果_同口径指标对照.csv',index=False,encoding='utf-8-sig')
    def metric_table(keys):
        x=summary.loc[keys,['total_return','annual_return','max_drawdown','sharpe_rf15','calmar','mean_exposure']].copy()
        x.insert(0,'策略',[NAMES[k] for k in x.index])
        for c in ['total_return','annual_return','max_drawdown','mean_exposure']:x[c]=x[c].map(lambda v:fmt(v,'pct'))
        for c in ['sharpe_rf15','calmar']:x[c]=x[c].map(fmt)
        return x.rename(columns={'total_return':'累计收益','annual_return':'年化收益','max_drawdown':'最大回撤','sharpe_rf15':'夏普','calmar':'卡玛','mean_exposure':'平均仓位'})
    comparison=metric_table(['new_main_monthly','old_main','equal_buy_hold','bj50_price_index','simple_ic20_equal'])
    allcomparison=metric_table(list(NAMES))
    risknames={'equal':'等权、无止损','rank_stop10':'名次分配＋10%跟踪止损','diversified_trend':'分组限制＋止损＋趋势仓位',
               'diversified_vol12':'分组限制＋止损＋波动仓位','cash':'不再开新仓，旧仓按原规则退出'}
    risktable=choices[['quarter','valid_start','valid_end','selected_rule']].copy()
    risktable.selected_rule=risktable.selected_rule.map(risknames)
    risktable.columns=['执行季度','过去验证起日','过去验证止日','选中的交易规则']
    recent=period[(period.strategy=='new_main_monthly')&(period.period=='recent_year')].iloc[0]
    recenthalf=period[(period.strategy=='new_main_monthly')&(period.period=='recent_half')].iloc[0]
    stress=load('stress_tests.csv')[['scenario','total_return','max_drawdown','sharpe_rf15','calmar']].copy()
    stress.scenario=stress.scenario.map({'cost0_delay0':'无费用（仅作对照）','cost5_delay0':'买卖各0.05%（主口径）','cost15_delay0':'买卖各0.15%','cost5_delay1':'买卖各0.05%，改用早一日的排名'})
    for c in ['total_return','max_drawdown']:stress[c]=stress[c].map(lambda v:fmt(v,'pct'))
    for c in ['sharpe_rf15','calmar']:stress[c]=stress[c].map(fmt)
    stress.columns=['情景','累计收益','最大回撤','夏普','卡玛']
    direction=lambda a,b,higher=True:'改善' if ((a>b) if higher else (a<b)) else '下降' if higher else '变大'
    verdict=(f"相对旧主流程，新主流程累计收益{direction(mainrow.total_return,old.total_return)}：{fmt(old.total_return,'pct')}→{fmt(mainrow.total_return,'pct')}；"
             f"最大回撤{direction(mainrow.max_drawdown,old.max_drawdown,False)}：{fmt(old.max_drawdown,'pct')}→{fmt(mainrow.max_drawdown,'pct')}；"
             f"夏普{fmt(old.sharpe_rf15)}→{fmt(mainrow.sharpe_rf15)}，卡玛{fmt(old.calmar)}→{fmt(mainrow.calmar)}。")
    foldtable=folds[['quarter','selected_name','selected_best_iteration','selected_valid_rank_ic']].rename(columns={'quarter':'季度','selected_name':'此前验证选中的模型','selected_best_iteration':'树的轮数','selected_valid_rank_ic':'验证排名IC'})
    foldtable['验证排名IC']=foldtable['验证排名IC'].map(lambda x:f'{x:.4f}')
    ordercols=['stockid','rank','score','target_weight','requested_shares','filled_shares','price','fee','reject_reason']
    ex=example[ordercols].head(8).copy().rename(columns={'stockid':'股票','rank':'名次','score':'分数','target_weight':'计划仓位','requested_shares':'计划股数','filled_shares':'成交股数','price':'开盘成交价','fee':'费用','reject_reason':'限制原因'})
    for c in ['分数','计划仓位','开盘成交价','费用']:ex[c]=ex[c].map(lambda v:f'{v:.4f}')
    ex=ex.fillna('')
    guide=f'''# 新项目：从数据、训练到真实资金账本

实验规则登记于2026-10-17，文件交付日期2026-10-18。阅读顺序：先看本文件的结果和一笔交易，再看每季训练表；逐条老师意见对应另见《01_老师意见一对一答复.md》。完整账户、模型与验证记录在 `training_runs/research_20261017/`。

## 1. 先回答：这次有没有更好

{verdict}

最近一年主流程仍亏{abs(recent.total_return):.2%}。季度重训＋相同风控累计收益{summary.loc['new_quarterly_same_risk','total_return']:.2%}，高于月度主流程；简单IC20＋相同风控也达到{summary.loc['simple_ic20_same_risk','total_return']:.2%}。所以本轮不能得出“更新越快越好”或“复杂模型必然优于简单组合”的结论。

相对等额买入持有，主流程全期累计收益高出{(mainrow.total_return-summary.loc['equal_buy_hold','total_return'])*100:.2f}个百分点；相对北证50高出{(mainrow.total_return-summary.loc['bj50_price_index','total_return'])*100:.2f}个百分点。这里是两个累计收益相减，不是实际做了对冲后赚到的钱。

{mdtable(comparison)}

这是2024-10-08至2026-09-30同一期间的历史重演，初始都是100万元。主流程的参数只根据各季度开始前的训练/验证决定，但这段历史已经在前几版研究中被看过；因此它可说明改版在这些历史上怎样表现，不能当作完全未见数据的保证。

比较中的北证50是价格指数折算的参考资产，不是扣完交易费的基金；期初等权持有是实际模拟账户，和策略一样扣费、受成交限制。我们没有把简单策略或收益最高的固定方案事后冒充主流程。

## 2. 项目由哪几部分组成

1. `risk_metadata.py`准备每个历史日期已可使用的规模、行业和风格资料。
2. `models.py`先筛因子、再训练模型，保存每季选择依据和每天股票评分。
3. `execution.py`把评分变成买卖，管理唯一现金余额、股票数量、费用、到期及止损。
4. `experiment.py`在过去验证期选风险规则，再把新主流程、简单方法、旧方法和基准放到同一账本比较。
5. `report.py`从保存的结果自动生成本报告、图表、真实日例子和逐股盈亏账。

日行情从2021-11-15到2026-10-09；因子训练表从2022-02-14开始。前面最早的日期用来积累足够的历史，不假装在没有历史时就训练出模型。完整对照结束日统一到2026-09-30。

## 3. 每一行和每一列是什么

一行是“某个交易日的一只股票”，例如同一天100只股票就有100行。把很多天叠起来，得到几万行；每一列是一个因子。日期和股票代码用于对齐和切时间，不作为模型输入列。未来五日收益放在独立答案表，交易时的评分文件不带未来答案。

原始特征先在同一天各股票之间排名：某股票在100只有数据的股票中排第35，就用35÷100−0.5=−0.15。这样输入表示当日相对位置，而非价格或因子的绝对单位。缺失值送入模型前填0，代表这一列的中间位置；这不能说成“缺失肯定排最后”。

答案先算实际五日收益，再在同日已有成熟答案的股票中排名并减0.5。模型分数用于比较股票强弱，0.10并不等于预计上涨10%。每次训练先检查答案结束日，尚未结束的整条答案不参与。

## 4. 因子具体如何筛

普通模型分支的中文规则：先用训练段算每列有值比例；不足70%就不要。对每个训练日，取因子和答案同时有数值的同一批股票，计算相关程度；再把各日结果按该候选的时间权重合并。按平均相关程度的绝对值从高到低排队；与已入选因子的绝对相关达到0.92，就跳过；直到得到候选要求的40列或100列。

负相关因子可以保留。例如值越小、后面排名越高，它仍然有用；简单IC20组合给它负权重，树模型则可学习“小于某值时加分”的判断。旧代码筛选相关度时，缺失配对的计算不够严谨；本次已按同一批成对有效股票重新计算。

PCA分支用于检验老师提出的“高相关也可能保留增量信息”：先按覆盖率和训练期相关程度取前100列，**不先删除高相关列**，再只在训练段把它们组合成20列。20是这次登记的降维候选设置，不是金融定律，也没有声称最优。新列是旧列的加权组合，不是挑出几列改名。验证/测试只使用训练段确定的组合，不能用测试数据重新算组合。

实际保留比例随训练数据变化，例如2024Q4正式训练的20列保留了约85.29%的输入变化量。这不等于保留85.29%的可赚钱信息，剩余部分仍然被丢掉；能否改善必须看同协议比较。所有候选验证成绩都保存在每季 `candidate_metrics.csv`，不会只展示胜者。

## 5. 一次训练到底怎么进行

每季开始前，程序把最近126个交易日当内部验证。再往前取该候选需要的252或504个交易日训练，并删除答案跨入验证期的训练记录。模型可以一次读完这些已经结束的训练行，因为做这次预测的时点已经在它们之后；训练内部的一轮又一轮是反复调整判断规则，不是日历第一天、第二天。

LightGBM开始给各训练行一个初始分数，计算“真实答案−当前分数”。程序搜索哪些因子阈值能把误差不同的记录分组，例如“因子A≤某个数”；每个末端组获得一个小修正，所有行的新分数等于旧分数加修正，再算误差。它不是随便猜无穷大，而是在指定损失、学习步长、树深、每组最小记录数等限制下求修正。

本次每个树模型最多300轮。每轮都记录训练和验证的平均平方误差；验证误差连续30轮没有更好就停止，保留验证误差最好那一轮。随后在候选模型之间按验证期平均每日排名IC选择。最后把过去已经成熟的标签重新用于正式训练，轮数使用此前选定值，然后预测下一段。

{mdtable(foldtable)}

季度版每季重训一次；月度版沿用该季选好的超参数和轮数，每月加入最近已经结束的答案、重新筛因子和建树。训练窗口252/504日和旧样本权重也在候选中比较；半衰期63日表示63个交易日前记录的训练权重是最近记录的一半，126日前是四分之一。这是学习时的权重，不是股票仓位。

## 6. 五日交易：每天先做什么，后做什么

```
从共同起点开始，每隔5个交易日作为一次信号日。
信号日收盘：用当天已知因子，给股票排序，查本季度已选好的风险规则。
下一交易日开盘：
    如果旧一批因停牌等原因还没卖完：跳过这一次新建仓。
    否则按排序选股，计算每只股票预算，只能用当前真实现金买入。
    不足100股、停牌、涨停代理检查不通过、超过成交额容量时，记下原因。
持有期间每天收盘：记现金、股票市值、总资产，更新最高收盘价。
    如果前一日收盘已经触发所选止损规则：下一日开盘尝试卖出。
    提前卖出的现金留到下一次换仓，不凭空又建一个新批次。
信号日后的第5个交易日收盘：按预先安排卖出；卖不出则以后开盘再尝试。
期末仍持有的股票按当日市值记账，不借用期末之后的卖出价。
```

买入日到卖出日覆盖5个交易日，两个日期之间通常相隔4个交易日间隔，两种数法不能混用；提前止损或停牌会改变实际持有长度。主流程完成卖出记录的平均持有间隔为{fmt(mainrow.mean_holding_intervals)}，含首尾交易日为{fmt(mainrow.mean_holding_sessions_inclusive)}。

总资产=现金+股票数量×当日收盘价。现金变化=昨日现金+股息+卖出金额−买入金额−费用。程序每一天都检查这两条账，全部股票共享同一笔钱。

## 7. 排名怎样变成仓位：一个真实日子

{example_date:%Y-%m-%d}开盘的实际买单如下（展示前8行，完整见`真实交易日_下单明细.csv`）。信号取自之前的收盘，分数只用于排序。等权规则中，计划股票总仓位÷20就是每只股票计划份额；排名权重规则给第1至20名按20、19、…、1分配，再受单股和分组上限约束。每季选择哪套规则见`risk_choices.csv`。

{mdtable(ex)}

这天开盘前现金{day.opening_cash:,.2f}元，买入{day.buy_value:,.2f}元，卖出{day.sell_value:,.2f}元，费用{day.cost:,.2f}元，股息{day.dividend_cash:,.2f}元。收盘现金{day.cash:,.2f}元、股票市值{day.market_value:,.2f}元，总资产{day.equity:,.2f}元，仓位{day.exposure:.2%}。完整现金流与逐单余额都可复算。

## 8. 风控具体管什么

六套候选包括：等权、不同行情下仍等权但带10%跟踪止损、排名权重加止损、行业/规模/风格建仓限制加止损、再叠加趋势仓位或波动仓位。每季只用此前验证账本选规则；验证期没有净收益为正且有足够交易的规则，就把下一季选为现金。

具体if/else：先删去验证期扣费后收益≤0或完成卖出分笔不足5的方案；若一个也不剩，下季不新买股票；否则给每个方案算min(夏普，卡玛÷2)，取最高者。min表示取两者较小值，避免仅靠一项很高掩盖另一项很差。这是本次研究预先设定的偏好，不是金融定律。模型、风控都用了同一段内部验证，因此该段成绩有挑选带来的乐观偏差，正式比较看后面的季度。

{mdtable(risktable)}

2026Q2和Q3所有候选都未通过上述条件，因此没有新开仓。3月留下的一批股票在4月1日结清，此后至9月30日全是现金；这解释了最近半年{recenthalf.total_return:+.2%}来自期初旧仓退出，后面曲线横着走。不能称这半年不断选股赚到了钱。

跟踪止损示例：买入后最高收盘价是10元，某天收盘跌到9元或以下；程序下一交易日开盘尝试卖，不假装一定能成交在9元。停牌、跌停代理条件、成交量限制可能让损失继续扩大。老师举的止损/止盈比例并不是收益保证。

分组上限控制建仓时同一行业最多30%、同一规模/动量/波动组最多50%的计划资产。持仓后价格和分组可能变化，所以每天另出监控表，不承诺任意收盘时刻都精确维持上限。历史行业资料使用有发布日期的官方分类，在可用日之后生效；未分类股票及历史股本异常都在覆盖报告中记录。

复核发现，启用分散规则时，信号日计划的行业/风格额度合格；但按持仓日重新分组，规模/动量/波动某一组的收盘最高占比曾达61.35%/69.28%/79.69%。目前系统是“建仓限制＋每日监测”，没有自动为了维持分组上限每天减仓。这是现有风控的实际边界。

历史行业已补齐五期中国上市公司协会官方分类，使用发布日期之后的下一交易日起生效；本次共同研究期股票池覆盖{known_fraction:.2%}，未覆盖不向前倒填。启用分散规则时，行业或规模未知的股票不买入；其他规则仍用共同股票池。下载原件、发布时间、校验值、旧新代码对应和解析检查保存在`training_data/research_20261017/industry/`。

最早一期行业到2024-10-08才可用，因此2024Q4的前126日验证内，所有要求行业已知的分散候选都无法建仓，收益/仓位为0。早期分散方案受到资料覆盖限制，这项比较不能全解释为分散规则好坏。主流程仅2025Q3和2026Q1选择分散规则；其他季度没有行业30%约束，整个主账户单一行业最高曾达49.43%。

有无约束对照并没有显示“加限制必然更好”：月度等权收益129.08%、回撤44.73%；加10%止损后收益54.03%、回撤47.65%；再加分组限制后收益33.95%、回撤50.55%。本轮这两项固定改动没有改善全期结果，不能因为叫风控就默认有效。

## 9. 怎样看收益、回撤、夏普、卡玛和IC

累计收益=最后总资产÷最初总资产−1。年化收益根据实际经过的日历天数折算。最大回撤每天用当日资产÷截至当日最高资产−1，最深的跌幅取绝对值；最初100万元也计入最高值。

夏普用每日账户收益，先减去年化1.5%换算的日无风险收益，再用平均值÷日收益标准差×√252。卡玛=年化收益÷最大回撤。它们都来自整个账户，包括现金和费用，不是几只股票收益的简单平均。卡玛高也可能因为长期只投少量钱，所以必须同时看仓位与收益。

IC是同一天模型给股票排的名次与后来实际收益名次之间的相关程度；正数表示总体排序方向较一致，负数表示总体反向。ICIR是多个日期的IC平均值÷这些日期IC的标准差，不自动乘√252。它评价排序稳定性，不直接等于交易利润。

### 最大亏损到底发生在哪里

最大回撤从{peak:%Y-%m-%d}最高资产{account.loc[peak,'equity']:,.2f}元，到{trough:%Y-%m-%d}资产{account.loc[trough,'equity']:,.2f}元，减少{-delta:,.2f}元。逐只股票计算“期末市值−期初市值＋卖出金额−买入金额＋股息−费用”，汇总和账户减少金额逐分核对。亏损最多的五只见图，完整表是`max_drawdown_stock_attribution.csv`。这是账户亏损分解，不是简单把已平仓的交易亏损相加，因此也包括阶段边界仍持有股票的涨跌。

{mdtable(loss_table)}

上表起点没有持仓但仍亏损，表示它在这段时间内被买入；不能只把起止市值相减来算这只的盈亏。该段两个端点现金仅差+33.38元，资产下跌主要体现在所持股票的损失。

### 提高成本或让信号慢一天会怎样

{mdtable(stress)}

以上都保持原来按过去验证选出的规则，不重新找最优。延迟情景使用早一交易日的排名、仍按原定调仓时点下单；它并不是模拟所有订单迟一天成交。费用提高或信息变旧均明显降低收益。

## 10. 这些结果还有哪些边界

所有新比较都使用同一已登记历史快照。原始因子的历史发布/修订时点、完整退市样本、真实涨跌停参考价和股息股份到账日仍未完全取得；成交使用日线开盘/收盘价及过去成交额容量代理。结果是明确假设下的研究账户，不能称为逐笔实盘成绩。

另一位同学的完整代码和账户未提供，本次实现的是可复算的“简单因子组合＋相同Top20/费用/周期”对照，不能声称复现了他的具体收益。行业历史表和当前状态不能互相替代。外层历史已被研究查看，下一步真正的前瞻成绩只能在规则固定后的新数据中积累。

参考实现原理：[LightGBM官方早停回调](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.early_stopping.html)、[scikit-learn官方数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html)。本项目具体设置以本地代码与输出记录为准。
'''
    (OUT/'02_新项目从数据到交易.md').write_text(guide,encoding='utf-8')
    # One-to-one closure: executable evidence and explicit boundaries for each request.
    tasks=[
        ('查清27%仓位','重算旧账户目标均值36.02%、实际27.79%；逐季列出风控设置。低仓位主要来自策略主动压低上限，另受分批资金和成交限制。','legacy_audit/旧主流程仓位审计.md'),
        ('现金、股票和总资产三本账','新账本逐日记录三者及买卖/股息/费用，检查现金流恒等式；只有一个现金余额。','accounts/new_main_monthly/daily.csv'),
        ('钱是否被重复使用','旧账无负现金与重复本金证据；新账每单即时扣现金，旧批未清跳过新买，已做合成用例验证。','../research_20261017/legacy_audit/审计摘要.json'),
        ('五天预测为什么每天交易','改为每5交易日发一批信号、下一开盘买、第5交易日收盘卖；日内只处理提前止损和延迟卖出。','../../tools/research_20261017/execution.py'),
        ('实际拿了几天','每次成交记录持有间隔和含首尾天数，分别汇总，停牌/止损单列。','holding_distribution.csv'),
        ('排序如何分到仓位','显式比较等权/20至1的排名权重；每笔记录名次、预算、计划股数、成交数和限制原因。','accounts/new_main_monthly/orders.csv'),
        ('训练、验证、重训的区别','8个季度记录边界；训练标签早于验证首日，正式训练标签早于季度/月度切点。','models/folds.csv'),
        ('180轮谁决定、何时停','从固定180轮改为最多300轮且验证误差30轮不改善停止，保存逐轮训练/验证曲线及选定轮数。','models/2026Q2/loss_curves.csv'),
        ('输入与未来答案不要混淆','外层交易文件只有评分；未来收益放独立审核文件；训练用成熟答案，预测不输入答案。','models/daily_predictions.parquet'),
        ('因子如何选、负相关怎样用','覆盖率、成对IC、相关去重、数量限制有完整原因表；负相关按有用的反向信号保留。','models/2026Q2/deployment_factor_screen.csv'),
        ('时间漂移与因子衰减','验证比较窗口/样本半衰期；季度参数选择，另外真实运行月度与季度重训对照。','strategy_summary.csv'),
        ('IC、ICIR与快慢要有实测','保存每日和逐季IC；2026Q2因子在共同日期比较1/5/20日，另用21/63/126个已成熟信号日检查最近关系是否变弱。','model_diagnostics/模型效果诊断.md'),
        ('借鉴PCA/正交化','PCA分支不先相关去重，训练期把100列组合成20列，记录实际保留方差，与普通分支同协议竞争。','models/2026Q2/candidate_metrics.csv'),
        ('对比简单策略','IC20简单组合、动态池等权和期初等权持有均在同时间/成本/账本中计算；未冒充同学原模型复现。','period_comparison.csv'),
        ('三条资金曲线与完整指标','给主流程、等额买入持有、北证50净值和回撤，附全期/一年/半年指标。','all_daily.parquet'),
        ('行业、市值和风格限制','用历史资料生成分组，建仓配额及每日集中度监测；覆盖与不可核实数据明确记录。','accounts/new_main_monthly/risk_exposure.csv'),
        ('亏损发生在哪里','2025-03-17至04-07资产从278.41万降至199.72万，损失78.69万；63只股票的市值、买卖、股息及费用归因加总一致。','max_drawdown_stock_attribution.csv'),
        ('HMM到底有没有用在主流程','纠正原汇报：旧主流程2025Q4、2026Q1/Q2确实由HMM控制仓位。新实验用明确的趋势/波动规则作对照。','legacy_audit/分季度仓位汇总.csv'),
        ('所有数字必须能复核','保存每季模型、所有验证候选、逐单/逐日账及现金等金融约束测试；附实际一天可手算的例子。','run_manifest.json'),
    ]
    taskmd='# 1016老师意见：一对一答复\n\n规则登记2026-10-17，交付2026-10-18。此处逐项说明实际改了什么，证据路径相对 `training_runs/research_20261017/`。\n\n'+mdtable(pd.DataFrame(tasks,columns=['老师问题/建议','本次处理','可复核证据']))
    taskmd+='\n## 不能混成已完成实盘验证的事情\n\n行业分类按官方发布日接入；历史股本/供应商因子的发布时间及修订记录仍未完全核实。简单基线借鉴了同学思路，但没有其完整原始代码与账户，不能宣称完全复现。本人是否真正理解仍需用真实日例子手算演练，不能由AI代替确认。老师明确让本人暂不做的衍生品对冲，不纳入这次选股系统。\n\n'+verdict+'\n'
    taskmd+='\n月度主流程最近一年仍亏9.39%；同风控季度重训全期收益249.47%，超过月度174.79%；同风控简单IC20组合为190.07%，也超过月度。固定加入止损及分组限制没有改善本段全期结果。2026年4月1日旧仓退出后，Q2/Q3保持现金；不能用这一段平直曲线证明选股能力提高。分组额度只约束信号日建仓计划，持有期间每日重新分组后存在超限，详见完整结果。\n'
    taskmd+='\n完整因子筛选、实际日期和中文伪代码见同目录《03_训练和因子筛选_具体日子逐步解释.md》。与其他同学本人商定口径、索取其私有代码和现场讲解不能由本地代码替代；本次没有向外发消息。分钟模型、LLM与衍生品对冲不是1016会议本轮必做事项。\n'
    (OUT/'01_老师意见一对一答复.md').write_text(taskmd,encoding='utf-8')
    # Minimal self-contained HTML, visible without external assets or scripts.
    def img(name):
        data=base64.b64encode((figs/(name+'.png')).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{name}"></figure>'
    def inline(text):
        escaped=html.escape(text)
        escaped=re.sub(r'\[([^\]]+)\]\(((?:https?://|\.\./)[^\s)]+)\)',r'<a href="\2">\1</a>',escaped)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',escaped)
    def paragraph(text):return '<p>'+inline(text)+'</p>'
    def simple_md(text):
        out=[];in_code=False;table=[]
        def flush():
            if table:
                rows=[r for r in table if not re.match(r'^\|\s*[-: ]+\|',r)]
                out.append('<div class="tablewrap"><table>'+''.join('<tr>'+''.join(('<th>' if j==0 else '<td>')+inline(v.strip())+('</th>' if j==0 else '</td>') for v in r.strip().strip('|').split('|'))+'</tr>' for j,r in enumerate(rows))+'</table></div>');table.clear()
        for line in text.splitlines():
            if line.startswith('```'):
                flush();out.append('</pre>' if in_code else '<pre>');in_code=not in_code;continue
            if in_code:out.append(html.escape(line)+'\n');continue
            if line.startswith('|'):table.append(line);continue
            flush()
            if line.startswith('### '):out.append('<h3>'+html.escape(line[4:])+'</h3>')
            elif line.startswith('## '):out.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line.startswith('# '):out.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line:out.append(paragraph(line.replace('`','')))
        flush();return ''.join(out)
    body='<h1>10月16日组会后：项目改版与实验结果</h1><p>2026-10-18交付 · 全部数字由本次保存的账户记录生成</p>'+paragraph(verdict)+comparison.to_html(index=False,border=0)
    body+='<h2>1. 策略与两个基准的资金和回撤</h2>'+img('01_三条资金曲线')+img('02_三条回撤曲线')
    body+='<h2>2. 新旧变化与仓位</h2>'+img('03_新旧账户与仓位')+'<p>收益变化可能同时来自模型和投入资金的比例；因此另列相同交易规则、相同风控的比较。</p>'
    body+='<h2>3. 全部固定对照</h2>'+allcomparison.to_html(index=False,border=0)
    pretty=show[show['策略'].isin([NAMES[k] for k in ['new_main_monthly','old_main','equal_buy_hold','bj50_price_index','simple_ic20_equal']])].copy()
    for c in ['累计收益','年化收益','最大回撤','平均仓位']:pretty[c]=pretty[c].map(lambda v:fmt(v,'pct'))
    for c in ['相对买入持有超额（百分点）','相对北证50超额（百分点）']:pretty[c]=pretty[c].map(fmt)
    for c in ['夏普（无风险1.5%）','卡玛']:pretty[c]=pretty[c].map(fmt)
    body+='<h2>4. 全期、最近一年与最近半年</h2>'+pretty.to_html(index=False,border=0)
    body+='<h2>5. 每季训练与验证误差</h2>'+img('08_训练验证正式预测时间轴')+'<p>先在内部验证中选方案；到2026-04-01时，再用截至3月31日已结束的答案重新建模。正式训练可以包含此前验证日期，下一季度才用于向前检验。</p>'+img('04_训练与验证误差')+'<p>竖向点线是此前验证期选定的轮数；实线为训练误差，横向虚线为验证误差。训练误差继续下降，不代表验证效果继续改善。</p>'
    body+='<h2>6. 每日风险分布与亏损来源</h2>'+img('05_每日规模与风格监测')+img('07_行业集中度监测')+img('06_最大回撤盈亏来源')
    body+='<h2>7. 老师问题逐项对应</h2>'+simple_md(taskmd)+'<h2>8. 从头读懂新项目</h2>'+simple_md(guide)
    diagnostics=(RUN/'model_diagnostics/模型效果诊断.md').read_text(encoding='utf-8')
    (OUT/'04_IC与因子衰减_实测结果.md').write_text(diagnostics,encoding='utf-8')
    body+='<h2>9. 排序效果、因子衰减与更新频率的实测</h2>'+simple_md(diagnostics)
    detail=OUT/'03_训练和因子筛选_具体日子逐步解释.md'
    if detail.exists():body+='<h2>10. 具体日期与中文伪代码详解</h2>'+simple_md(detail.read_text(encoding='utf-8'))
    daydetail=OUT/'05_真实一天_股票池到成交.md'
    if daydetail.exists():body+='<h2>11. 真实一天：从350个代码到20笔成交</h2>'+simple_md(daydetail.read_text(encoding='utf-8'))
    body+='<h2>原始结果入口</h2><p>同目录：实验结果_同口径指标对照.csv、真实交易日_下单明细.csv、01_老师意见一对一答复.md、02_新项目从数据到交易.md。CSV的收益/回撤/仓位按小数存储（0.10表示10%），超额列已经换算为百分点。完整逐日逐单记录与候选表位于项目 training_runs/research_20261017/。</p>'
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>1016改版实验结果</title><style>body{font-family:"Microsoft YaHei",sans-serif;color:#111;background:#fff;max-width:1160px;margin:40px auto;padding:0 24px;line-height:1.8}h1{font-size:30px}h2{margin-top:48px;border-bottom:1px solid #aaa;padding-bottom:8px}p{max-width:1000px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}th{background:#f3f3f3}img{max-width:100%;height:auto}figure{margin:24px 0}pre{white-space:pre-wrap;background:#f5f5f5;padding:20px}.tablewrap{overflow-x:auto}@media print{body{margin:0;max-width:none}h2{break-after:avoid}figure,tr{break-inside:avoid}}</style><body>'+body+'</body></html>'
    (OUT/'项目改版_完整说明与实验结果.html').write_text(document,encoding='utf-8')
    (OUT/'README.md').write_text('# 1016会议后新项目\n\n先打开[完整说明与实验结果](项目改版_完整说明与实验结果.html)，包含资金曲线、指标、亏损来源、老师意见对应和从头说明。\n\n'+verdict+'\n\n代码在 `tools/research_20261017/`，详细结果在 `training_runs/research_20261017/`。\n',encoding='utf-8')
    print(verdict,flush=True);print('REPORT_COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
