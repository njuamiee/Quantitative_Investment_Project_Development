"""Streaming result charts and readable live report. Never labels pending tasks complete."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,html,base64,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import RUN,REPORT,HORIZONS,COUNTS,MODELS,FREQUENCIES,START,END,dump,digest,write_csv,temporary_path,atomic_replace
from monthly_selection import completed,select_and_replay
from account import metrics
from controlled_comparisons import compare,risk_compare
from report_details import detail_charts,decision_rows,acceptance_html
from diagnostics import lagged_exposure_control

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'figure.facecolor':'#f7f9fc','axes.facecolor':'white','font.size':10})
COLORS=['#116A7B','#D46642','#536CC5','#7D557F','#55735B']
def figsave(fig,name):
    path=REPORT/'图表'/name;temp=temporary_path(path)
    fig.tight_layout();fig.savefig(temp,dpi=160,bbox_inches='tight');plt.close(fig);atomic_replace(temp,path)

def build():
    REPORT.mkdir(exist_ok=True);(REPORT/'图表').mkdir(exist_ok=True)
    records=completed()
    if not records:return
    snapshot=time.strftime('%Y-%m-%d %H:%M:%S');rows=[];risk_rows=[];samples=[];best=None;posthoc=None;total=0;pass_counts=np.zeros(2,int);inactive=np.zeros(2,int)
    for r in records:
        a=np.load(RUN/'models'/r['job']/'risk_results.npz');v=a['metrics'];c=r['config'];total+=len(v)
        risk_rows.extend(risk_compare(r,v))
        for pi in range(2):
            pass_counts[pi]+=int(((v[:,pi,3]>1)&(v[:,pi,4]>2)&(v[:,pi,2]<.3)).sum())
            inactive[pi]+=int((v[:,pi,5]<=1e-12).sum())
        for period,pi in [('共同历史',0),('最近区间',1)]:
            rows.append(dict(**c,job=r['job'],period=period,accounts=len(v),actual_fits=r['actual_fits'],
                median_return=float(np.nanmedian(v[:,pi,0])),max_return=float(np.nanmax(v[:,pi,0])),
                median_drawdown=float(np.nanmedian(v[:,pi,2])),passes=int(((v[:,pi,3]>1)&(v[:,pi,4]>2)&(v[:,pi,2]<.3)).sum())))
        ix=np.unique(np.linspace(0,len(v)-1,min(12,len(v))).astype(int));samples.append(v[ix,1,:5])
        obj=np.minimum(v[:,0,3],v[:,0,4]/2);obj=np.where(np.isfinite(obj),obj,-np.inf);k=int(np.argmax(obj))
        if best is None or obj[k]>best[0]:best=(obj[k],r,k)
    df=pd.DataFrame(rows);write_csv(df,REPORT/'已完成模型配置汇总.csv')
    sample=np.concatenate(samples);sample=sample[::max(1,len(sample)//10000)]
    days,curves,summary,decisions=select_and_replay(records)
    extra,additional_summary=lagged_exposure_control(days,curves);curves.update(extra);summary.extend(additional_summary)
    if best is not None:
        a=np.load(RUN/'models'/best[1]['job']/'full_posthoc.npz');posthoc=a['daily']
    summary_df=pd.DataFrame(summary);write_csv(summary_df,REPORT/'逐月系统与基准结果.csv')
    write_csv(summary_df,RUN/'selection/summary.csv')
    # 1. Actual completed model coverage; zero cells are visible.
    cov=pd.DataFrame([r['config'] for r in records]);matrix=cov.groupby(['model','horizon']).size().unstack(fill_value=0).reindex(index=MODELS,columns=HORIZONS,fill_value=0)
    fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(matrix,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(HORIZONS)),HORIZONS);ax.set_yticks(range(len(MODELS)),MODELS);ax.set_xlabel('预测收益期限（交易日）')
    for i in range(len(MODELS)):
        for j in range(len(HORIZONS)):ax.text(j,i,str(matrix.iloc[i,j]),ha='center',va='center',color='black')
    ax.set_title(f'实际完成覆盖：{len(records):,} / 8,640 个模型配置（每格上限135）');fig.colorbar(im,ax=ax,label='完成的模型配置');figsave(fig,'01_实验覆盖.png')
    # 2-3. Deterministic stratified plot sample, exact complete counts separately reported.
    fig,ax=plt.subplots(figsize=(10,5));ax.scatter(sample[:,2]*100,sample[:,0]*100,c=sample[:,3],cmap='RdYlBu',s=8,alpha=.5)
    ax.axvline(30,color='#b32b36',ls='--');ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='最大回撤 %',ylabel='累计收益 %',title='最近区间：收益与回撤（均匀抽取已完成配置用于显示）');figsave(fig,'02_收益与回撤.png')
    fig,ax=plt.subplots(figsize=(10,5));valid=np.isfinite(sample[:,3:5]).all(axis=1)
    ax.scatter(sample[valid,3],sample[valid,4],s=9,alpha=.4,color=COLORS[0]);ax.axvline(1,color='#b32b36',ls='--');ax.axhline(2,color='#b32b36',ls='--')
    ax.set(xlabel='夏普（无风险利率1.5%）',ylabel='卡玛（远离0的区间压缩显示）',title='最近区间：夏普与卡玛；虚线右上方才同时达标');ax.set_yscale('symlog',linthresh=2);figsave(fig,'03_夏普与卡玛.png')
    # 4-5. These are replayed accounts, not concatenated winning curves.
    names=[('rolling_cash_fallback_fee5','逐月选择：不合格则现金'),('rolling_equal_fallback_fee5','逐月选择：不合格则股票池等权'),('dynamic_pool_equal_fee5','动态股票池等权基准'),('quarter_pool_buyhold_fee5','季度初股票池等权、季内持有'),('lagged_exposure_equal_fee5','跟随系统昨日仓位的等权参照')]
    fig,ax=plt.subplots(figsize=(11,5.3))
    for i,(k,label) in enumerate(names):ax.plot(days,curves[k][:,0]/1e6,label=label,color=COLORS[i],lw=1.8)
    if posthoc is not None:ax.plot(days,posthoc[:,0]/1e6,label='事后全历史最佳：仅描述，不可用作选型证明',color='#888888',ls=':',lw=1.2)
    ax.set(title='100万元模拟账户资金曲线｜当前已完成候选集合',ylabel='净值（起始=1）');ax.legend(fontsize=8);ax.grid(alpha=.2);figsave(fig,'04_连续资金曲线.png')
    fig,ax=plt.subplots(figsize=(11,4.5))
    for i,(k,label) in enumerate(names):
        e=curves[k][:,0];dd=e/np.maximum.accumulate(np.r_[1e6,e])[1:]-1;ax.plot(days,dd*100,label=label,color=COLORS[i])
    ax.axhline(-30,color='#b32b36',ls='--');ax.set(title='连续账户最大回撤路径',ylabel='距历史最高净值的下跌 %');ax.legend(fontsize=8);ax.grid(alpha=.2);figsave(fig,'05_回撤曲线.png')
    fig,ax=plt.subplots(figsize=(10,4.5));sub=df[df.period=='最近区间']
    distributions=[sub[sub.horizon==h].median_return.to_numpy()*100 for h in HORIZONS]
    ax.boxplot([a if len(a) else [np.nan] for a in distributions],labels=HORIZONS,showfliers=False)
    ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='预测收益期限（交易日）',ylabel='每个模型配置下风控组合收益中位数 %',title='期限比较：展示分布；其他维度尚未全部完成时不作优劣定论');figsave(fig,'06_收益期限比较.png')
    fig,ax=plt.subplots(figsize=(10,4.5));freq=cov.frequency.value_counts().reindex(FREQUENCIES,fill_value=0)
    ax.bar(['季度重训','月度重训','每日重训'],freq,color=COLORS[:3]);ax.set(ylabel='已完成模型配置数',title='重训频率覆盖（计划每种2,880个配置）')
    for i,v in enumerate(freq):ax.text(i,v+max(freq.max()*.02,1),str(v),ha='center')
    figsave(fig,'07_重训频率覆盖.png')
    fig,ax=plt.subplots(figsize=(11,4.5))
    for i,(k,label) in enumerate(names):ax.plot(days,curves[k][:,4]*100,label=label,color=COLORS[i])
    ax.set(title='实际持仓比例：资金减少与择股能力分别观察',ylabel='股票市值/账户权益 %');ax.legend(fontsize=8);figsave(fig,'08_实际仓位.png')
    # Paired moving-block bootstrap, no IID-day assumption.
    rng=np.random.default_rng(20261026);a=curves[names[0][0]][:,0];b=curves[names[2][0]][:,0]
    ra=np.r_[a[0]/1e6-1,a[1:]/a[:-1]-1];rb=np.r_[b[0]/1e6-1,b[1:]/b[:-1]-1];delta=ra-rb
    means=[];n=len(delta)
    for _ in range(2000):
        idx=np.concatenate([(s+np.arange(20))%n for s in rng.integers(0,n,size=int(np.ceil(n/20)))])[:n]
        means.append(delta[idx].mean()*252)
    lo,hi=np.quantile(means,[.025,.975]);fig,ax=plt.subplots(figsize=(9,4));ax.hist(np.asarray(means)*100,bins=45,color=COLORS[0],alpha=.8)
    ax.axvline(0,color='#b32b36',ls='--');ax.set(title=f'逐月系统相对等权：20日成组重采样\n年化日均收益差95%区间 [{lo:.1%}, {hi:.1%}]',xlabel='年化日均收益差 %（不是复合年化收益差）',ylabel='重采样次数');figsave(fig,'09_相对收益不确定性.png')
    pass_common=int(pass_counts[0]);pass_recent=int(pass_counts[1]);fits=sum(r['actual_fits'] for r in records)
    paired=compare(records);write_csv(paired,REPORT/'控制变量配对比较.csv')
    details=detail_charts(paired,risk_rows,figsave) if len(paired) else []
    decisions_html=decision_rows(decisions)
    if len(paired):
        recent=paired[paired.period=='recent'];freqpairs=recent[recent.axis=='frequency']
        fig,ax=plt.subplots(figsize=(11,4.8));data=[];labels=[]
        for alt,label in [('monthly','月度 − 季度'),('daily','每日 − 季度')]:
            z=freqpairs[freqpairs.alternative==alt].median_return_change.to_numpy()*100
            data.append(z if len(z) else [np.nan]);labels.append(label+f'\n{len(z)}对模型配置')
        ax.boxplot(data,labels=labels,showfliers=False);ax.axhline(0,color='gray',lw=.8)
        ax.set(title='控制其他参数：重训更频繁是否更好？',ylabel='相同风控规则的收益差中位数（百分点）');figsave(fig,'13_重训频率控制比较.png')
        fig,ax=plt.subplots(figsize=(11,4.8));hp=recent[recent.axis=='horizon'];data=[];labels=[]
        for h in [1,2,3,4,10,15,20]:
            z=hp[hp.alternative==str(h)].median_return_change.to_numpy()*100
            data.append(z if len(z) else [np.nan]);labels.append(str(h))
        ax.boxplot(data,labels=labels,showfliers=False);ax.axhline(0,color='gray',lw=.8)
        ax.set(title='只改变预测期限：全部固定每日换仓，与5日期限配对',xlabel='预测期限（交易日）',ylabel='相同规则下收益差中位数（百分点）');figsave(fig,'14_期限控制比较.png')
    status='全部完成' if len(records)==8640 else '运行中的阶段快照，非最终验收'
    best_card=None
    if best is not None:
        _,record,risk_id=best;folder=RUN/'models'/record['job']
        with np.load(folder/'risk_results.npz') as a:best_metrics=a['metrics'][risk_id,:,:5].tolist()
        best_card=dict(job=record['job'],model_config=record['config'],risk_id=risk_id,
            risk_config=json.loads((folder/'risk_registry.json').read_text())[risk_id],
            common_and_recent_metrics=best_metrics,
            metric_order=['total_return','annual_return','max_drawdown','sharpe','calmar'],
            selection='posthoc whole-history max(min(Sharpe,Calmar/2)); descriptive only, not a deployment recommendation')
    manifest=dict(generated=snapshot,status=status,completed_model_configs=len(records),actual_model_fits=fits,
                  completed_account_configs=total,registered_model_configs=8640,registered_account_configs=27993600,
                  pass_common=pass_common,pass_recent=pass_recent,candidate_snapshot=[r['job'] for r in records],
                  zero_equity_exposure_configs=dict(common=int(inactive[0]),recent=int(inactive[1])),
                  descriptive_best_fixed=best_card,
                  data_end=END,paired_block_bootstrap_interval=[float(lo),float(hi)],
                  code_hashes={p.name:digest(p) for p in __import__('pathlib').Path(__file__).parent.glob('*.py')})
    sources=[('Microsoft Qlib滚动研究框架','https://github.com/microsoft/qlib'),('Microsoft RD-Agent因子研究闭环','https://github.com/microsoft/RD-Agent'),
             ('Vanguard 2024：机器学习、可解释性与风控','https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/machine-learning-role-in-hunt-alpha.html'),
             ('美联储2024：因子选择与结构变化','https://www.federalreserve.gov/econres/feds/factor-selection-and-structural-breaks.htm'),
             ('AQR：趋势研究','https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing'),
             ('多重搜索的夏普偏差原始论文','https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf'),
             ('北交所交易规则','https://www.bse.cn/jygl_list/200028217.html')]
    minute_html=''
    if (RUN/'minute/complete.json').exists():
        mr=pd.read_csv(RUN/'minute/results.csv');mc=pd.read_parquet(RUN/'minute/curves.parquet')
        choices=json.loads((RUN/'minute/Q4_selection.json').read_text())
        choice={r['horizon']:r['chosen_weight'] for r in choices}
        reference=mr[(mr.topn==20)&(mr.target==.5)&(mr.stop==0)&(mr.participation==.05)]
        daily=reference[reference.minute_weight==0].set_index('horizon');chosen=reference[reference.chosen_on_Q4].set_index('horizon')
        fig,ax=plt.subplots(figsize=(10,4.8));xx=np.arange(len(HORIZONS))
        ax.bar(xx-.18,daily.reindex(HORIZONS).total_return*100,.36,label='日频信号，同一真实分钟容量',color=COLORS[0])
        ax.bar(xx+.18,chosen.reindex(HORIZONS).total_return*100,.36,label='Q4事先选择的分钟混合权重',color=COLORS[1])
        ax.set_xticks(xx,HORIZONS);ax.axhline(0,color='gray',lw=.8);ax.set(title='2026Q3分钟配对：固定Top20、目标50%、窗口成交额5%',xlabel='预测期限',ylabel='累计收益 %');ax.legend(fontsize=8);figsave(fig,'10_分钟信号配对.png')
        fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
        h=2
        for w,label,color in [(0.,'日频信号',COLORS[0]),(choice[h],f'Q4选择分钟权重{choice[h]:.0%}',COLORS[1])]:
            s=mc[(mc.horizon==h)&(mc.minute_weight==w)].sort_values('date')
            axs[0].plot(s.date,s.equity/1e6,label=label,color=color);axs[1].plot(s.date,s.exposure*100,color=color)
        axs[0].set(title='2日期限：相同分钟成交模型下的资金曲线',ylabel='净值');axs[0].legend();axs[1].set(ylabel='实际仓位 %');figsave(fig,'11_分钟账户资金曲线.png')
        # Add a compact capacity plot across the actually completed micro-grid.
        fig,ax=plt.subplots(figsize=(9,4.2));z=mr.groupby(['participation','topn']).fill_share_fraction.median().unstack()
        for i,n in enumerate([5,20,40]):ax.plot(z.index*100,z[n]*100,marker='o',label=f'Top{n}',color=COLORS[i])
        ax.set(title='成交容量：计划股数的实际模拟成交比例中位数',xlabel='允许参与真实分钟成交额的比例 %',ylabel='成交股数/计划股数 %');ax.legend();figsave(fig,'12_分钟容量限制.png')
        dump(dict(accounts=len(mr),passes=int(mr.pass_target.sum()),chosen_reference=chosen.reset_index().to_dict('records')),REPORT/'分钟实验摘要.json')
        minute_html='<h2>分钟配对实验：单独的真实覆盖区间</h2><p>已完成1,152个账户配置。2025Q4训练/验证选择分钟信号权重，2026Q3冻结使用；数量在执行窗口前按此前收盘确定，随后受每个真实分钟成交额的1%或5%限制。使用该分钟成交均价作为参与式执行基准，仍没有逐笔盘口或成交队列保证。图中的配对固定Top20、目标50%、不加止损，其余组合全部保留。普通换仓订单仅当次窗口有效，止损待卖状态持续保留。</p>'
    def img(name):
        p=REPORT/'图表'/name;return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'" alt="'+name+'">'
    if minute_html:minute_html+=''.join(img(n) for n in ['10_分钟信号配对.png','11_分钟账户资金曲线.png','12_分钟容量限制.png'])
    paired_html=''
    if len(paired):paired_html='<h2>控制变量比较</h2><p>每一对保持日期、成本、模型其他参数和具体风控规则一致，仅改变一个参数。期限比较另外固定为每日换仓，避免把“预测更长”与“持有更久”混为一谈。图显示这些配对的分布；相关组合不是独立统计样本，未完成的配对暂不纳入。</p>'+img('13_重训频率控制比较.png')+img('14_期限控制比较.png')
    paired_html+=''.join(img(n) for n in details)
    if details:paired_html+='<p>风控图逐一改变持股数、权重、仓位预算、HMM、退出、集中度、反弹入场、交易池和换仓间隔。横线是各模型配对结果的四分位范围，不是置信区间；同时展示收益、回撤和实际仓位，便于区分减少投资带来的风险变化。所有配对摘要均保留CSV，零差异与不利变化一并纳入。</p>'
    def pct(v):return f'{v:.2%}' if pd.notna(v) else '无定义'
    table=summary_df[['period','strategy','total_return','max_drawdown','sharpe','calmar','mean_exposure']].copy()
    chinese={'rolling_cash_fallback':'逐月选择，后备现金','rolling_equal_fallback':'逐月选择，后备等权',
             'dynamic_pool_equal':'动态池等权','quarter_pool_buyhold':'季度初等权、季内持有',
             'lagged_exposure_equal':'跟随昨日仓位等权（诊断）'}
    table['strategy']=table.strategy.map(lambda s:chinese[s.rsplit('_fee',1)[0]]+' / '+s.rsplit('_fee',1)[1])
    for c in ['total_return','max_drawdown','mean_exposure']:table[c]=table[c].map(pct)
    for c in ['sharpe','calmar']:table[c]=table[c].map(lambda v:f'{v:.2f}' if pd.notna(v) else '无定义')
    recent_table=table[table.period=='recent'].drop(columns='period');common_table=table[table.period=='common'].drop(columns='period')
    for t in [recent_table,common_table]:t.columns=['系统/成本（bp/边）','收益','回撤','夏普','卡玛','实际仓位']
    body=f'''<p class="eyebrow">北交所横截面选股 · 系统重设计 · {snapshot}</p><h1>让规则每月更新，<br>让每一次选择都有证据。</h1>
<p class="status">{status}。行情截至 {END}；已查看过的历史用于探索，不能称为全新独立样本。</p>
<div class="cards"><div><b>{len(records):,}</b>已完成模型配置 / 8,640</div><div><b>{fits:,}</b>实际拟合次数</div><div><b>{total:,}</b>已完成策略账户 / 27,993,600</div></div>
{acceptance_html(summary)}
<h2>这次系统怎样工作</h2><p>每日收盘：动态股票池和当时可得因子 → 按成熟答案重训或沿用模型 → 股票排名 → 市场状态与仓位上限 → 次日模拟交易。每月开始：只用此前63/126日的真实模拟账户记录，综合夏普和卡玛选择下一月规则；两个窗口中较差的表现决定评分。没有合格候选时，分别比较现金与股票池等权两种后备规则。换规则后重新运行一份连续现金和股票账本，不拼接赢家资金曲线。</p>
<h2>实验覆盖和结果</h2><p>8个期限 × 5个因子数量 × 8类模型 × 3种重训频率 × 3种训练窗口 × 3类因子库，共8,640个模型配置。风控层再独立组合持股数、权重、仓位、HMM、退出规则、行业/市值约束、入场条件、池大小与换仓频率；去除1日期限中等价的换仓重复项，登记27,993,600个策略配置。这些是配置数，不是相互独立的统计样本。</p>
{img('01_实验覆盖.png')}<p>目前共同历史中有 {pass_common:,} 个配置同时通过“夏普&gt;1、卡玛&gt;2、回撤&lt;30%”；最近区间有 {pass_recent:,} 个。事后找到通过者，不等于事前能选中。未完成配置不计入任何已完成数。</p>
<p>其中共同历史全程股票仓位为0的配置有 {int(inactive[0]):,} 个，最近区间有 {int(inactive[1]):,} 个；这些保留为风险规则拒绝入场的实验结果，不算盈利通过者。不同参数还可能产生相同的持仓路径，所以账户配置总数不等于独立策略或独立证据的数量。</p>
{img('02_收益与回撤.png')}{img('03_夏普与卡玛.png')}
<h2>真正要验收的是逐月选择系统</h2><p>5bp与15bp均为每边合并成本情景；15bp保持同一选择规则，不按后段成绩重新挑模型。现金的收益假设为0。图中灰色虚线是事后最佳，仅供观察。</p>
<h3>共同历史：2025-07-02至2026-10-19</h3><p>前63日没有足够账户记录选型，现金分支会空仓，等权分支按后备规则持股。共同历史收益包含这段预热，因此不能把它们的全部差额都归于选股。</p>{common_table.to_html(index=False,escape=True)}
<h3>最近区间：2026-05-22至2026-10-19</h3>{recent_table.to_html(index=False,escape=True)}
<p>“跟随昨日仓位等权”是在看到主结果后增加的诊断参照：下一日只使用逐月现金分支此前收盘的实际仓位，在动态池中等权投资，每日重新计算目标。同样使用现金、容量和成本账本，实际仓位仍会因成交不同而略有差异；两个成本情景共用主系统5bp账户的历史仓位目标。它帮助区分低仓位与选股效果，不是新的独立验证，也不参加月度选型或计入主网格数量。</p>
{img('04_连续资金曲线.png')}{img('05_回撤曲线.png')}{img('08_实际仓位.png')}
{decisions_html}
<h2>期限、重训频率和不确定性</h2><p>按你的最新要求，期限仅为1、2、3、4、5、10、15、20日。每日重训并不天然更好，必须在同日期、成本、因子及风控条件下比较。当前覆盖尚不均衡时，禁止把先跑完的一类宣布为最终赢家。</p>
{img('06_收益期限比较.png')}{img('07_重训频率覆盖.png')}{img('09_相对收益不确定性.png')}{paired_html}{minute_html}
<h2>熊市、抄底和仓位</h2><p>HMM按过去数据估计状态，逐日向前滤波；熊市把新增买入最多压至1只，同时把目标仓位限制至5%，单只股票不会获得整个账户资金。确认式反弹规则要求近期回升与市场广度改善：下跌阶段最高10%试探，改善后再分级增加。它不能保证买到最低价。个股止损、追踪止损和组合回撤降仓均在此前收盘触发，下一次可成交时才执行；跌停和停牌可能造成无法及时减仓。</p>
<h2>中证2000到底在比较什么</h2><p>中证2000既是一条指数曲线，也对应一个会调整的股票集合。横截面选股实验应在当时的成分股中给每只股票排序，再与同一成分股池按相同成本和成交规则等权买入比较，这检验“在这个股票池里选股有没有增益”。官方指数是另一条参照，因为权重和成交时点不同，不能直接把两者差值都称为选股能力。北交所与中证2000之间的实验检验方法能否迁移，不是把一个指数当成2000只股票的预测模型。先前的成分历史校验缺口仍在，不能用不确定的成员历史做正式迁移验收。</p>
<h2>因子、分钟数据与真实边界</h2><p>311个完整日频特征、54个Codex预先提出的公式假说，以及1601个保守供应商候选字段。供应商扩库仅154个快照日，扩库实验明确采用最多向前沿用4个交易日的历史可得值；不会向过去填入未来因子。54个公式只算候选，季度筛选中检查覆盖、成熟答案相关性和重复性。没有声称部署了无人值守外部LLM代理。分钟因子仅125个交易日，分钟成交容量需另做有数据日期的配对实验；本页主网格为日线成交近似。</p>
<p>原始行情、停牌、30%涨跌幅近似、最低100股、每股递增、前20日均成交额1%共享容量、分红拆股及现金约束均进入账本。没有逐笔盘口或真实委托队列，因此图中是研究账户，不代表实盘可以完整成交。行业/市值规则是集中度上限，不是严格行业中性。</p>
<h2>公开研究依据</h2><p>借鉴滚动任务管理、因子提案反馈、可解释性、因子随市场变化和多重搜索控制；没有把公开材料称为机构的内部策略，也没有照搬境外收益承诺。</p><ul>'''+''.join(f'<li><a href="{u}">{html.escape(t)}</a></li>' for t,u in sources)+'''</ul>
<h2>文件与复现</h2><p>完整协议见《研究协议.md》。每个模型配置保留预测、训练边界、因子清单、全部风控结果和选择评分；任务队列保留待运行、运行、完成、失败状态。结果快照.json记录本页的实际候选集合。脚本位于 tools/systematic_20261026，账本结果位于 training_runs/systematic_20261026。完成图表会由监视脚本更新，刷新本文件即可查看最新落盘快照。</p>'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>组合研究 · 实验进度与图表</title><style>body{background:#eef2f5;color:#20323e;font:16px/1.8 "Microsoft YaHei",sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:48px 36px;background:white}h1{font-size:42px;line-height:1.3}h2{font-size:25px;border-top:1px solid #dbe4e9;padding-top:32px;margin-top:42px}p{max-width:1000px}.eyebrow{color:#116a7b}.status{padding:16px;background:#fff1d6;border-left:4px solid #c48622}.cards{display:flex;gap:20px;margin:30px 0}.cards div{flex:1;padding:24px;background:#eaf4f5}.cards b{display:block;font-size:32px;color:#116a7b}img{width:100%;height:auto;margin:15px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}a{color:#116a7b}@media(max-width:700px){main{padding:20px}.cards{display:block}h1{font-size:30px}}</style><main>'+body+'</main></html>'
    dest=REPORT/'实验进度与结果.html';tmp=dest.with_suffix('.tmp.html');tmp.write_text(page,encoding='utf-8')
    atomic_replace(tmp,dest)
    dump(manifest,REPORT/'结果快照.json')
    print(json.dumps(dict(model_configs=len(records),fits=fits,accounts=total,pass_common=pass_common,pass_recent=pass_recent,generated=snapshot),ensure_ascii=False),flush=True)

if __name__=='__main__':build()
