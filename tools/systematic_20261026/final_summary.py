"""Write the completed-batch Chinese interpretation only after final audit."""
import json
from datetime import datetime
import pandas as pd
from config import REPORT,RUN,atomic_replace,temporary_path
from report_details import NAMES

def main():
    s=json.loads((REPORT/'结果快照.json').read_text(encoding='utf-8'))
    audit=json.loads((REPORT/'审计快照.json').read_text(encoding='utf-8'))
    assert s['completed_model_configs']==audit['completed_models']==8640
    assert s['actual_model_fits']==audit['actual_fits']==967680
    assert s['completed_account_configs']==audit['completed_accounts']==27993600
    f=pd.read_csv(REPORT/'逐月系统与基准结果.csv')
    def result(strategy,period):return f[(f.strategy==strategy)&(f.period==period)].iloc[0]
    common=result('rolling_cash_fallback_fee5','common');recent=result('rolling_cash_fallback_fee5','recent')
    costly=result('rolling_cash_fallback_fee15','common');matched=result('lagged_exposure_equal_fee5','common')
    matched_cost=result('lagged_exposure_equal_fee15','common')
    quarter=result('quarter_pool_buyhold_fee5','common');quarter_recent=result('quarter_pool_buyhold_fee5','recent')
    best=s['descriptive_best_fixed'];period=best['common_and_recent_metrics'][0]
    bc=best['model_config'];br=best['risk_config']
    library='原日频库' if bc['library']=='daily' else NAMES[bc['library']]
    paired=pd.read_csv(REPORT/'控制变量配对比较.csv');paired=paired[paired.period=='recent']
    def effect(axis,alternative):
        z=paired[(paired.axis==axis)&(paired.alternative.astype(str)==str(alternative))]
        return float(z.median_return_change.median()*100),len(z)
    h1,n1=effect('horizon',1);h2,n2=effect('horizon',2);daily,nd=effect('frequency','daily')
    factors,nf=effect('count',100);vendor,nv=effect('library','vendor_asof')
    minute=pd.read_csv(RUN/'minute/results.csv')
    reference=minute[(minute.topn==20)&(minute.target==.5)&(minute.stop==0)&(minute.participation==.05)]
    base=reference[reference.minute_weight==0].set_index('horizon').total_return
    chosen=reference[reference.chosen_on_Q4].set_index('horizon').total_return
    minute_delta=chosen-base
    acceptance='达到三项历史门槛，但这仍不等于独立前瞻或实盘验证' if common.pass_target and recent.pass_target else '没有同时达到用户要求的收益风险门槛'
    body=f'''# 全组合实验最终解读

结果快照：{s['generated']}。行情截至2026-10-19。本批次的训练与账户实验全部完成，逐月选择系统{acceptance}。

## 实际完成了多少工作

- 8,640个模型配置全部完成，0个遗漏；包含1、2、3、4、5、10、15、20日期限，20/40/60/80/100个因子上限，8类模型，日/月/季重训，126/252/504日训练窗口及3类因子库。
- 完成967,680次保留的滚动拟合，以及27,993,600个主网格账户参数配置；不同账户复用预测，参数配置也可能产生相同持仓，不能解释成两千多万个独立模型或独立证据。
- 单独完成1,152个真实覆盖区间的分钟账户实验；另有2个跟随昨日实际仓位的等权诊断参照，不计入主网格数量。
- 全部{audit['fold_boundaries_checked']:,}次拟合的时间边界通过核对；抽查{audit['sampled_prediction_rows_checked']:,}行预测及文件哈希，独立复算{audit['independent_label_checks']}个收益标签；现金收支最大误差为{audit['max_cash_flow_error']:.3g}元。

## 最重要的账户结果

以下均为连续现金/股份账本。主系统每月只用以前的63/126日账户表现选择下一月规则，没有合格候选则持现金；每边5bp约为买或卖金额的0.05%。

| 评价区间 | 累计收益 | 最大回撤 | 夏普 | 卡玛 | 平均实际仓位 |
|---|---:|---:|---:|---:|---:|
| 2025-07-02至2026-10-19 | {common.total_return:.2%} | {common.max_drawdown:.2%} | {common.sharpe:.2f} | {common.calmar:.2f} | {common.mean_exposure:.2%} |
| 2026-05-22至2026-10-19 | {recent.total_return:.2%} | {recent.max_drawdown:.2%} | {recent.sharpe:.2f} | {recent.calmar:.2f} | {recent.mean_exposure:.2%} |

目标是最大回撤小于30%、夏普大于1、卡玛大于2，必须同时满足。少亏、回撤较小或跑赢满仓下跌的股票池，都不能代替这三项验收。共同历史还包含最初没有足够账户记录选型的预热期。

这批评价从2025-07-02开始，不能和旧版从2024-10-08开始的累计收益直接比较。不同版本的数据处理、因子筛选、交易账本和评价区间应分别保留，先在同区间、同成本下对齐，才能讨论哪一版更好。

## 为什么事后有好组合，逐月系统仍可能不达标

共同历史有{s['pass_common']:,}个固定组合达到三项门槛，最近区间有{s['pass_recent']:,}个。事后全历史评分最高的固定组合，累计收益为{period[0]:.2%}、回撤{period[2]:.2%}、夏普{period[3]:.2f}、卡玛{period[4]:.2f}。

它使用 **{NAMES[bc['model']]}、{bc['horizon']}日预测、最多{bc['count']}个因子、{NAMES[bc['frequency']]}重训、{bc['window']}日训练窗口、{library}**。账户基础持股上限为{br['topn']}只，采用{NAMES[br['weighting']]}、{NAMES[br['exposure']]}仓位预算，市场状态规则为“{NAMES[br['regime']]}”，退出规则为“{NAMES[br['exit']]}”，入场条件为“{NAMES[br['entry']]}”，在{br['pool']}只交易池中每{br['rebalance_days']}日重新计算目标。实际持仓仍受个股10%上限、状态和成交约束。完整参数可按配置ID `{best['job']}`、风控ID `{best['risk_id']}` 查询。

这是观察完完整历史后挑出的结果，不能倒过来宣称当时已经知道应该用它，也不作为下个月部署推荐。逐月选择器只能看过去，短窗口中的赢家可能在之后失效；大规模搜索放大了碰到偶然赢家的机会。这里是对结果的可能解释，不是已证明唯一原因。

## 少亏是否只是因为少买股票

补充诊断让一个同池等权账户每天跟随主系统上一日收盘的实际股票仓位。共同历史中，主系统收益{common.total_return:.2%}，该等权参照收益{matched.total_return:.2%}，差额为{(common.total_return-matched.total_return)*100:.2f}个百分点。

当每边成本从5bp提高到15bp，主系统共同历史收益变为{costly.total_return:.2%}；跟随仓位的等权参照为{matched_cost.total_return:.2%}。成本敏感性反映交易频繁的代价，必须和低成本曲线一起看。这个诊断是在看到主结果后增加的，没有参与选型；实际仓位也会因成交限制不同而有差异，因此不是严格的因果分解或新的独立样本验证。

“季度初等权、季内持有”的5bp基准在共同历史收益为{quarter.total_return:.2%}，最近区间为{quarter_recent.total_return:.2%}。本轮结果不支持“把空仓改成一直持股就一定挣钱”；持续持股也会持续承担市场下跌。是否保持仓位需要看收益与回撤，而不是只看资金有没有闲着。

## 怎么查每一类设置的影响

在最近区间的完整控制变量比较中：

- 固定每日换仓，1日预测相对5日预测的配对收益差中位数为{h1:.3f}个百分点（{n1:,}对模型配置）；2日相对5日为{h2:.3f}个百分点（{n2:,}对）。这是期限这一项的比较，不能把“1日模型”当作一套自动满足全部指标的方案。
- 每日重训相对季度重训的配对收益差中位数为{daily:.3f}个百分点（{nd:,}对）。更频繁学习不自动等于明显更高收益。
- 因子上限100相对20的配对收益差中位数为{factors:.3f}个百分点（{nf:,}对）；加入供应商扩库相对原日频库为{vendor:.3f}个百分点（{nv:,}对）。更多字段和更大模型需要用增量结果证明价值。
- 分钟实验有{int(minute.pass_target.sum())}个事后组合达到季度区间的三项门槛，{int((minute.total_return>0).sum())}个组合盈利。固定Top20、基础50%、不加止损、分钟参与率5%的配对中，按2025Q4事先选择的分钟权重有{int((minute_delta>1e-12).sum())}个期限改善、{int((minute_delta < -1e-12).sum())}个恶化、{int((minute_delta.abs()<=1e-12).sum())}个相同；事后组合通过数不代表已事先选中这些组合。

打开《实验进度与结果.html》中的18张图：先看连续资金、回撤和实际仓位，再看期限、重训频率、因子数量、模型、训练窗口、因子库与风险规则的控制变量比较。每对只改变一个设置，图中保留改善、变差和没有变化的情况。分钟增量和分钟成交容量使用独立覆盖区间，单独展示。

“每个月具体用了哪套规则”可以逐月展开；《逐月选择规则明细.csv》保留完整参数。全量账户指标按模型配置存放在本地 `training_runs/systematic_20261026/models/` 中的 `risk_results.npz` 和 `risk_registry.json`，不把大型可重建缓存上传到代码仓库。

## 能得出和不能得出的结论

这轮已经完成系统设计、全组合计算、对照图、时间边界审计和可复查结果保存。固定组合的好成绩与逐月系统的实际成绩必须分开陈述。当前历史已经多次查看，不能叫全新独立测试；不能为了达到目标再反复改规则后仍把同一段历史当成独立验收。

本轮没有完成可靠的11月实盘部署包：10月19日之后的数据、完整的历史发布时点、真实委托队列及生产截止时间仍需验收。下一步应冻结从本批次提取的明确假说，在新增数据上向前检查；若继续改选择器，必须作为新版本登记，并保留本批次未达标的原始结论。

完整方法、机构公开资料的借鉴范围和中证2000迁移实验的边界，见《研究协议.md》；零基础读者先看《先读这份说明.md》。
'''
    path=REPORT/'全组合实验最终解读.md';tmp=temporary_path(path);tmp.write_text(body,encoding='utf-8');atomic_replace(tmp,path)
    print('Final interpretation written after matching complete report and audit receipts.')

if __name__=='__main__':main()
