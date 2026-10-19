# 2026-10-18 深入实验：复现入口

用户文件：`../../新项目_20261018_深入实验/深入实验_完整结果与解释.html`。本轮96组账户、10张资金/风险/损失图、中文逐项答复和完整时段表。所有历史均已进入研究，不称全新未见样本。

环境建议 Python 3.11；本轮已核实的直接依赖与版本见 [`requirements.txt`](requirements.txt)，可在项目根目录运行 `python -m pip install -r tools/research_20261018/requirements.txt`。运行前必须按 [`DATA_INPUTS.md`](DATA_INPUTS.md) 准备获授权的本地快照。公开代码和汇总结果本身不包含逐股行情、分钟原始数据或模型大文件，因此缺少这些输入时无法完整重跑。

## 运行顺序（工作目录为项目根目录）

本机已有环境可用 `.venv_research_20261011/Scripts/python.exe`；其他环境使用安装上述依赖后的 `python`。前置：1017 模型预测、1011 价格/公司行动/分钟数据和状态概率均已存在。

1. `tools/research_20261018/deep_risk_experiment.py`：4种固定评分×19退出规则，另4条过去验证选规则，共80账户。登记规则在`PROTOCOL.md`及结果`design.json`。
2. 因子反馈第一轮：`auto_factor_lab.py --proposals tools/research_20261018/loop_round1.json --train-start 2023-01-01 --valid-start 2025-07-01 --test-start 2026-01-01 --test-end 2026-10-01 --screen-only --out training_runs/research_20261018/factor_loop/round1`。
3. 提案者只看第一轮训练/验证反馈，写第二轮；此次已冻结为`loop_round2.json`，相同命令换成round2。重放JSON只复现既有两轮，不声称重新调用了另一台LLM。
4. `factor_loop_experiment.py`：最多3个新增公式，100/103列共同172轮模型，4账户。协议`FACTOR_LOOP_PROTOCOL.md`，反馈`loop_feedback.md`。
5. `hmm_conditional_experiment.py`：每月504日已成熟标签，20共同因子，统一/2状态/3状态权重×3种资金规则，9账户。
6. `minute_incremental_experiment.py`：2025Q4训练和验证，2026Q3三条同池账户；Q4选出0%分钟权重，固定25%只敏感度。
7. `explain_new_model.py`：保存两天真实分数的逐因子贡献，合计复原预测。
8. `assemble_evidence.py`：统一基准、时段超额、季度表、5段最大回撤逐股归因。
9. `audit_deep_risk.py`和`audit_supplemental.py`：分别复核80账户以及16新增账户+6基准，不依赖执行引擎重算资产等式。
10. `python -m unittest discover -s tools/research_20261018 -p "test_*.py"`：12项测试。
11. `build_delivery.py`：生成10张图和白底中文HTML、复制证据表。

## 中证2000数据

`sync_market.py --phase factors|raw|hfq`仅从用户已有授权服务器只读下载；凭证仍通过既有连接函数读取，不在报告中复制。`--parse-only`只重建本地派生表。`audit_market_coverage.py`审计实际股票日键，检查BJ因子不能再漏掉。670天原件全部落盘，最终因子3,496,511行；覆盖表与缺配键已保存。

**真实动态历史成员仍缺，因此没有中证2000正式比较结果，也没有将代理池标为真实指数池的运行入口。**固定日期、Alpha101、零费用零滑点、本人Top20和五日规则见`PROTOCOL.md`。详细已查来源和缺口见`中证2000数据核查.md`。后续拿到正式成员还需按生效/可得时间审计，并补查两只BJ复权缺口；不能仅插入当前成分就跑收益。

## 核心限制

- 账本正确不代表上游历史发布/修订时点已经全部证实。
- 北交所涨跌停/停牌/容量用日线代理，分红按既有公司行动假设；不适合直接搬到沪深执行。
- 普通固定策略全期胜出属于历史发现，不能冒称每个季度事前都会选它。
- Sharpe>1、Calmar>2和回撤≤20%的近期验收未通过；12项测试通过也不等于业绩目标通过。
- 未操作真实账户、未发邮件、未购买数据或部署外部服务。
