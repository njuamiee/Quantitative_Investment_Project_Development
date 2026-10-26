from pathlib import Path
import json,shutil,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'training_runs/meeting_update_20261023';OUT=ROOT/'实验更新_20261023';FIG=OUT/'图表';FIG.mkdir(exist_ok=True)
END='2026-10-19'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
names={'original_monthly':'原月度模型','alpha101_lgbm':'Alpha101树模型','expanded_lgbm':'扩库树模型','expanded_ridge_all':'扩库全列Ridge',
       'rolling_selected':'滚动选模型/规则','pool_equal100':'动态池100只等权','initial_pool_buyhold':'起点池买入持有'}
rules={'equal20':'20只等权','equal40':'40只等权','industry_size40':'40只行业/市值分散','fixed25':'分散+固定25%','fixed50':'分散+固定50%',
       'ma20':'分散+MA20','stop5_ban10':'分散+止损5%/禁买10日','stop10_take20_ban10':'分散+止损10%/止盈20%','ma20_stop5_ban10':'分散+MA20/止损禁买'}
def label(s):
    if '__' not in s:return names.get(s,s)
    a,b=s.split('__');return names[a]+' · '+rules[b]
def daily(key,fee=5):return pd.read_parquet(RUN/f'{key}__fee{fee}_daily.parquet')
def save(fig,name):fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
def table(df):
    cols=list(df);lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    lines+=['| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)
def pct(x):return f'{x:+.2%}'

def main():
    s=pd.read_csv(RUN/'summary.csv');s['名称']=s.strategy.map(label)
    coverage=json.loads((RUN/'input_coverage.json').read_text(encoding='utf-8'))
    test=s[(s.period=='test')&(s.fee_bp==5)].copy();r=test.set_index('strategy')
    keys=['original_monthly__equal20','original_monthly__industry_size40','original_monthly__fixed50','original_monthly__ma20','original_monthly__stop5_ban10',
          'original_monthly__ma20_stop5_ban10','expanded_lgbm__industry_size40','expanded_ridge_all__equal20','rolling_selected','pool_equal100','initial_pool_buyhold']
    brief=[]
    for key in keys:
        z=r.loc[key];brief.append({'方案':label(key),'收益':pct(z.total_return),'最大回撤':f'{z.max_drawdown:.2%}',
          '夏普rf1.5%':f'{z.sharpe_rf15:.2f}','卡玛':f'{z.calmar:.2f}','平均收盘仓位':f'{z.mean_exposure:.2%}'})
    # Full 4x9 grid, preserve failures and the predeclared order.
    fig,axes=plt.subplots(1,2,figsize=(17,5.8));mods=list(names)[:4];rr=list(rules)
    for ax,col,title in zip(axes,['total_return','max_drawdown'],['测试期收益','测试期最大回撤']):
        data=np.array([[r.loc[f'{m}__{x}',col] for x in rr] for m in mods])
        im=ax.imshow(data,cmap='RdYlGn' if col=='total_return' else 'YlOrRd',aspect='auto')
        ax.set_xticks(range(9),[rules[x] for x in rr],rotation=65,ha='right');ax.set_yticks(range(4),[names[x] for x in mods]);ax.set_title(title+'（每边5bp）')
        for i in range(4):
            for j in range(9):
                rgba=im.cmap(im.norm(data[i,j]));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,f'{data[i,j]:.1%}',ha='center',va='center',fontsize=9,color='white' if lum<.45 else 'black')
    fig.suptitle('36组固定候选：降低回撤是否伴随收益改善',fontsize=17);fig.tight_layout();save(fig,'01_全部候选')
    navkeys=['original_monthly__equal20','original_monthly__fixed50','original_monthly__ma20_stop5_ban10','expanded_ridge_all__equal20','rolling_selected','initial_pool_buyhold']
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
    for key in navkeys:
        d=daily(key);base=float(d[d.date<'2026-05-22'].equity.iloc[-1]);z=d[d.date>='2026-05-22'];nav=z.equity/base
        x=pd.concat([pd.Series([pd.Timestamp('2026-05-21')]),z.date],ignore_index=True);v=np.r_[1.,nav.to_numpy()]
        axes[0].plot(x,v,label=label(key),linewidth=1.7 if key=='rolling_selected' else 1.1)
        axes[1].plot(x,v/np.maximum.accumulate(v)-1)
    axes[0].set_ylabel('净值（5月21日收盘=1）');axes[0].legend(fontsize=9,ncol=2);axes[1].set_ylabel('自该期起点回撤');axes[1].yaxis.set_major_formatter(PercentFormatter(1));fig.suptitle('已查看测试段：收益与逐日回撤');fig.tight_layout();save(fig,'02_净值与回撤')
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for key in ['original_monthly__equal20','original_monthly__ma20','rolling_selected']:
        d=daily(key);z=d[d.date>='2026-05-22'];axes[0].plot(z.date,z.exposure,label=label(key),alpha=.8)
    a=daily('rolling_selected');z=a[a.date>='2026-05-22'];axes[1].step(z.date,z.active_bans,where='post',color='#6d51ae')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].set_ylabel('收盘股票市值/总资产');axes[0].legend(fontsize=9);axes[1].set_ylabel('滚动账户禁买股票数');fig.suptitle('仓位与禁买状态：第五日收盘退出会形成周期性现金');fig.tight_layout();save(fig,'03_仓位与禁买')
    log=pd.read_csv(RUN/'selection_all_candidates.csv',parse_dates=['decision_date','validation_end']);chosen=log[log.selected].copy()
    chosen['模型']=chosen.candidate.str.split('__').str[0].map(names);chosen['规则']=chosen.candidate.str.split('__').str[1].map(rules)
    chosen.to_csv(OUT/'滚动选择记录.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(13,5));levels=list(dict.fromkeys(chosen.candidate));mapping={k:i for i,k in enumerate(levels)}
    ax.step(chosen.decision_date,chosen.candidate.map(mapping),where='post',color='#087f8c');ax.scatter(chosen.decision_date,chosen.candidate.map(mapping),color='#087f8c')
    ax.set_yticks(range(len(levels)),[label(k) for k in levels]);ax.set_title('每20交易日按过去成绩选择，隔离最近10交易日');fig.tight_layout();save(fig,'04_滚动选择')
    monthly=[]
    base=daily('initial_pool_buyhold').set_index('date').equity.pct_change().fillna(0)
    for key in navkeys:
        d=daily(key).set_index('date');ret=d.equity.pct_change().fillna(0);ret=ret[ret.index>='2026-05-22'];br=base.reindex(ret.index)
        for per,g in ret.groupby(ret.index.to_period('M')):
            b=br.reindex(g.index);monthly.append(dict(strategy=key,month=str(per),strategy_return=np.prod(1+g)-1,benchmark_return=np.prod(1+b)-1,excess_pp=(np.prod(1+g)-np.prod(1+b))*100))
    pd.DataFrame(monthly).to_csv(OUT/'测试期逐月对照.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(12,5));cost=s[(s.period=='test')&s.strategy.isin(['rolling_selected','original_monthly__equal20','original_monthly__fixed50','expanded_ridge_all__equal20','expanded_ridge_all__fixed50'])]
    ck=list(dict.fromkeys(cost.strategy));x=np.arange(len(ck))
    for off,fee in [(-.18,5),(.18,15)]:
        v=[cost[(cost.strategy==k)&(cost.fee_bp==fee)].total_return.iloc[0] for k in ck];bars=ax.bar(x+off,v,.36,label=f'每边{fee}bp')
        ax.bar_label(bars,labels=[f'{z:.1%}' for z in v],padding=3,fontsize=9)
    ax.set_xticks(x,[label(k).replace(' · ','\n') for k in ck],fontsize=9);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.set_title('成本敏感性：15bp情景保持5bp时选择的模型/规则');ax.axhline(0,color='#555',linewidth=.6);fig.tight_layout();save(fig,'05_交易成本')
    # Ledger and provenance checks, full outputs remain available.
    checks=[]
    for p in RUN.glob('*_daily.parquet'):
        d=pd.read_parquet(p);err=float((d.equity-d.cash-d.market_value).abs().max())
        flow=float((d.cash-(d.opening_cash+d.dividend_cash+d.sell_value-d.buy_value-d.cost)).abs().max())
        assert err<1e-5 and flow<1e-5 and d.cash.min()>-.001
        checks.append(dict(account=p.stem,identity_error=err,cash_flow_error=flow,min_cash=float(d.cash.min())))
    assert (log.validation_end<log.decision_date).all()
    pd.DataFrame(checks).to_csv(OUT/'账户核验.csv',index=False,encoding='utf-8-sig')
    s.to_csv(OUT/'全部实验结果.csv',index=False,encoding='utf-8-sig')
    for key in ['rolling_selected','original_monthly__equal20']:
        for suffix in ['daily','orders','trades']:
            pd.read_parquet(RUN/f'{key}__fee5_{suffix}.parquet').to_csv(OUT/f'{key}_{suffix}.csv',index=False,encoding='utf-8-sig')
    selected=r.loc['rolling_selected'];baseline=r.loc['original_monthly__equal20'];fixed=r.loc['original_monthly__fixed50']
    st=chosen[chosen.decision_date>='2026-05-22'];counts=st.groupby(['模型','规则']).size().reset_index(name='选择次数')
    month=pd.DataFrame(monthly);rollmonth=month[month.strategy=='rolling_selected'].copy();wins=int((rollmonth.excess_pp>0).sum())
    monthly_table=table(pd.DataFrame([{'月份':z.month,'滚动收益':pct(z.strategy_return),'起点池持有收益':pct(z.benchmark_return),'差值(百分点)':f'{z.excess_pp:+.2f}'} for z in rollmonth.itertuples()]))
    rr=daily('rolling_selected').set_index('date').equity.pct_change();unc=[];rng=np.random.default_rng(20261023)
    for key in ['original_monthly__equal20','original_monthly__fixed50','initial_pool_buyhold']:
        br=daily(key).set_index('date').equity.pct_change();ex=(rr-br).loc['2026-05-22':END].to_numpy();n=len(ex)
        sims=[]
        for _ in range(2000):
            starts=rng.integers(0,n,size=int(np.ceil(n/20)));ix=np.concatenate([(j+np.arange(20))%n for j in starts])[:n]
            sims.append(float(ex[ix].mean()*252))
        unc.append({'比较基准':label(key),'年化日均收益差':pct(ex.mean()*252),'区间下界':pct(np.quantile(sims,.025)),'区间上界':pct(np.quantile(sims,.975))})
    pd.DataFrame(unc).to_csv(OUT/'区块重采样诊断.csv',index=False,encoding='utf-8-sig')
    qrows=[]
    for period in ['Q2','Q3']:
        z=s[(s.strategy=='rolling_selected')&(s.fee_bp==5)&(s.period==period)].iloc[0]
        qrows.append({'季度':period,'收益':pct(z.total_return),'平均收盘仓位':f'{z.mean_exposure:.2%}','收盘仓位不足1%的日数比例':f'{z.cash_days_fraction:.2%}'})
    exits=pd.read_parquet(RUN/'rolling_selected__fee5_trades.parquet');exits=exits[exits.exit_date>='2026-05-22']
    exstats=exits.groupby('reason').agg(成交部分数=('lot_id','size'),平均持有交易日=('holding_sessions_inclusive','mean'),已实现盈亏=('pnl','sum')).round(2).reset_index()
    exstats.to_csv(OUT/'滚动账户退出归因.csv',index=False,encoding='utf-8-sig')
    groups=pd.read_parquet(RUN/'rolling_selected__fee5_risk_exposure.parquet');groups=groups[groups.date>='2026-05-22']
    groupdiag=groups.groupby('dimension').weight.max().rename('max_account_weight').reset_index();groupdiag.to_csv(OUT/'滚动账户分组暴露.csv',index=False,encoding='utf-8-sig')
    report=f'''# 根据10月23日会议更新的北交所实验

本轮已实际运行 **36组固定候选、1个滚动选模型/规则账户、2个股票池基准**，并对预定账户运行15bp费用敏感性。共保存{len(checks)}份账户。数据截至2026-10-19，历史测试段已经看过，本轮属于探索与核验。

## 先看结果

滚动组合在2026-05-22—2026-10-19累计收益 **{selected.total_return:+.2%}**，最大回撤 **{selected.max_drawdown:.2%}**，平均收盘仓位 **{selected.mean_exposure:.2%}**。同一新实验起点下，原月度分数20只等权收益 **{baseline.total_return:+.2%}**、最大回撤 **{baseline.max_drawdown:.2%}**；原月度分数分散后固定50%收益 **{fixed.total_return:+.2%}**、回撤 **{fixed.max_drawdown:.2%}**。

收益和回撤的变化不能单独证明选股增强。需要结合固定仓位、股票池基准、费用与各月结果判断；本次不按历史最好方案宣布上线。滚动组合在所列{len(rollmonth)}个自然月切片中，有{wins}个收益超过起点股票池买入持有；5月和10月是不完整月份，样本有限。

**止损并非必然改善**：同样原月度分数、40只行业市值分散，加5%止损/10日禁买后的测试收益为{r.loc['original_monthly__stop5_ban10','total_return']:+.2%}，均线加止损组合为{r.loc['original_monthly__ma20_stop5_ban10','total_return']:+.2%}。滚动组合与固定50%相比仍需看全段表现，不能只拿较少亏损证明复杂规则有效。

{table(pd.DataFrame(brief))}

## 会议要求落实了什么

| 要求 | 本轮落实 |
| --- | --- |
| 行业、市值分散 | 40只组合：行业30%、市值三组50%的建仓上限；未知信息不强行填充。名称为集中度约束，不声称严格中性。 |
| 先做简单风控 | 20日均线100%/50%、5%止损后禁买10日、10%止损/20%止盈，以及与均线组合。 |
| 避免季度极端空仓 | 每20交易日重选、每五日调仓；没有季度收益正负门槛，也没有现金候选。仍可能因到期退出、禁买或成交限制产生现金。 |
| 参数事前确定 | 固定4类分数×9条规则；每次只用之前60日结果，中间隔离10日，至少40日；所有36个候选评价留档。 |
| 滚动运行 | 保存历史已实现候选收益，按rf0夏普选择下一段，单一真实模拟账本运行，保留跨段残留；不是拼接每段赢家净值。 |
| 扩因子 | 沿用已完成的120表、2445字段目录，主版本为1601保守候选；纳入扩库树模型和全合格列Ridge，不为本次重复下载或虚称重训。 |
| 成本与交易 | 主情景每边5bp，敏感性15bp；T+1、成交额容量和涨跌停代理；止损次日尝试，失败继续持有，不凭低价假设成交。 |
| 可解释与核验 | 逐日现金/市值/权益、买卖订单、实际持有天数、退出原因、禁买记录、行业/市值暴露均保存。 |

## 全部候选及收益路径

![全部候选](图表/01_全部候选.png)

![净值和回撤](图表/02_净值与回撤.png)

![仓位和禁买](图表/03_仓位与禁买.png)

### Q2/Q3是否仍然整季空仓

{table(pd.DataFrame(qrows))}

这里的低于1%统计是**收盘时点**，第五日收盘退出也会被计入，不能解释成全天没承担风险。该滚动版本没有整季强制现金逻辑。

## 滚动究竟选了什么

最近测试段中的选择频数如下。每次选择有历史评价截止日，最新10个交易日不参与；在最早评价历史不足时，采用原月度20只等权作为固定默认。

{table(counts)}

![滚动选择](图表/04_滚动选择.png)

候选因子/模型预测复用既有滚动结果，其中扩库模型按252训练、6gap、63内部验证、6gap、21预测，按真实标签终点隔离。**此次新增的是模型与风控的外层历史选择，没有更改内层训练。**研究者早已查看过这些日期，不能把程序的时间隔离等同于全新独立测试。

## 成本与逐月结果

![成本敏感性](图表/05_交易成本.png)

{monthly_table}

### 改善有多确定

对测试段逐日收益差做20交易日循环区块重采样，共2000次；下表为年化平均日收益差及95%分位区间，**不是累计超额收益，也不是多重比较校正后的显著性结论**。这只是对100日已查看样本的不确定性诊断。

{table(pd.DataFrame(unc))}

15bp情景保持5bp时得到的选择日程，检验同一策略对更高费用的敏感性。它不是在15bp下重新选参。买入持有从2025年7月共同起点首次入场的股票池持续持有，测试期切片保留原有持仓；动态100只等权使用同样五日调仓，是另一基准。

## 数据与结论边界

- 本轮共同账户从2025-07-02开始使用预测，之前持币；保留原调仓日历相位。与10月21日从2024年起运行的旧账户有不同初始化，因此不把结果差额全归因于新增风控。
- 行业已知覆盖{coverage['industry_known']:.2%}，市值分组覆盖{coverage['size_known']:.2%}。行业使用当时已发布分类；股本快照发布时间与修订历史仍未完全独立核验。
- 30%涨跌幅、过去成交额容量只是代理；缺少完整历史ST、退市、真实涨跌停与股息到账资料，仍为研究回测。卖出不能保证止损线即最终成交损失。
- MA20参考的是动态北交所100只池等权累计收益，非北证50；行业分散不能消除全市场下跌。固定低仓位对照必须同时看。
- 本轮未连接实时数据或VNPY。模型工程对接、完整市场中性、逐事件PIT核对以及真正未看日期的纸面运行仍未完成，不声称会议所有目标已验收。
- 本地北证50参考文件仅到2026-10-12，本轮未将其不完整区间收益冒充截至10月19日的同期间基准；使用覆盖完整的两个股票池基准。
- 本轮没有新增日历日期的数据，也没有以事后最优结果替换原主流程。下一步需要固定候选和评价规则后接入新日期，持续记录真实前瞻表现。

## 如何复现和查细节

- 协议：[实验协议](实验协议.md)。结果：[全部实验结果](全部实验结果.csv)；选择：[滚动选择记录](滚动选择记录.csv)；账本检查：[账户核验](账户核验.csv)。
- 完整代码：`tools/meeting_update_20261023/`；完整Parquet：`training_runs/meeting_update_20261023/`。旧文件未覆盖。
- 顺序运行 `test_update.py`、`run_experiments.py`、`build_report.py`，使用 `.venv_research_20261011/Scripts/python.exe`。
- 8项引擎关键测试已通过（含拆股不误触发止盈）；本轮{len(checks)}个账户的逐日资金恒等和现金流均核对通过。源数据、预测、协议和脚本哈希保存在运行目录。
'''
    (OUT/'实验更新报告.md').write_text(report,encoding='utf-8')
    # Use existing Markdown renderer if available, otherwise a readable escaped text.
    try:
        import markdown
        body=markdown.markdown(report,extensions=['tables','fenced_code'])
    except ImportError:body='<pre>'+html.escape(report)+'</pre>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>会议后实验更新</title><style>body{font:16px/1.8 Microsoft YaHei,sans-serif;max-width:1180px;margin:40px auto;padding:0 25px;color:#193247}h1,h2{line-height:1.4}h2{margin-top:44px;border-bottom:2px solid #d7e4eb}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #d7e4eb;text-align:left}th{background:#e9f1f5}img{max-width:100%}a{color:#087f8c}pre{white-space:pre-wrap}</style><body>'+body+'</body></html>'
    (OUT/'实验更新报告.html').write_text(page,encoding='utf-8')
    print(table(pd.DataFrame(brief)),flush=True)

if __name__=='__main__':main()
