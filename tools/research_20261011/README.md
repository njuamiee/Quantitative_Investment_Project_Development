# 10月9日会议后：多期限选股研究

输入快照于2026-10-11采集，行情截至2026-10-09；文档交付日期2026-10-12。
用户已确认飞书没有更新，无需再补正文。

## 结果位置

- `training_runs/research_20261011/candidate_summary.csv`：各预测期限的历史筛选候选。
- `validation_selected_summary.csv`：按预先写定的验证规则自动选型的独立结果，不与探索候选混淆。
- `h*/candidate_metrics.json`：执行持有期、成本、收益、回撤、利用率、胜率等。
- `h*/candidate_sensitivity.json`：15bp、无分红、延迟一天等压力情景。
- `h*/frozen_release.json`：固定候选参数、模型文件与输入哈希；不能在后续评价期间悄悄改参数。
- `missing_files.csv`、`missing_fields_and_documents.csv`：已识别的文件/字段缺口。
- `requirements_matrix.csv`：30项老师/项目/用户要求与证据。

历史探索候选是在比较后选出的，不能称为全新留出集验证。10/20日预测可采用5日持有，两个期限都明确记录；20日原始持有对照也保留。

**关键限制：5日和20日历史候选最近一年、最近半年亏损。预定验证选型也未通过Calmar门槛。因此尚未完成稳定达标的验收。** `all_strategy_period_audit.csv`保留500组固定方案及540组风险研究的分段检查；`short_windows/`保留发现该问题后补做的短窗口月度更新实验，仍是探索研究。

后续480组短窗口实验已完成：完整区间41组通过，最近一年0组、最近半年0组通过双门槛。`recent_signal_diagnosis.csv`显示相对排序IC仍为正，但近期5/20日入选股票的平均绝对标签收益为负；诊断数据不是账户收益。
完整24页Word/PDF、30项要求状态表、逐文件缺口与全部短窗口结果放在 `学习笔记/20261012_多期限选股项目/`；`document_qa.json`记录逐页检查，目录内manifest提供交付文件哈希。

## 复现

研究环境：`.venv_research_20261011/Scripts/python.exe`；软件版本见 `environment.json`。
在工作区运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools/research_20261011/run_local.ps1
```

默认只使用本地数据。加 `-Sync` 才只读盘点/下载SFTP，脚本数据截止仍为登记的10月9日。
不对服务器写文件，不下单。Jev脚本只生成本地示例，不收费调用。
加 `-ShortWindows` 可复现63/126/252日训练窗口、月度更新与126日HMM的后续研究。单独执行：先运行 `regimes.py --window 126`，再运行 `short_windows.py`。该研究缓存仅适用于已登记的本次输入快照；更换输入必须使用新输出目录并重新训练。
Kronos另运行 `kronos_probe.py`，下载公开权重并在本机推理；远程仅涉及公开模型下载，不发送行情。

完整重跑依赖工作区原有9月30日与10月8日数据快照，以及被引用的旧账本实现；本目录不是脱离整个工作区即可运行的独立包。
若输入哈希变化，训练缓存自动失效。重新调参的结果必须建立新的研究版本，不能覆盖原先前瞻冻结评价。

## 远程仓库保存范围

仓库保存研究代码、完整报告、全部方案指标、因子筛选记录、关键日账户/成交记录、验证结果和五个期限的 `2026Q4_models.joblib`。模型哈希见各自的 `frozen_release.json`。
原始行情与因子快照、本地环境、逐股预测Parquet缓存、较早季度模型及已失效账本中间产物在本机保留；需要这些缓存时，按上述复现流程重新生成。远程仓库单独克隆后仍需配置数据路径及取得相应数据，不能把摘要文件当成原始训练数据。

## 关键定义

1. h≥2：t收盘打分，t+1开盘入场，预测t+h收盘相对入场的复权收益。h=1：预测t+1开盘至t+2开盘，避免同日卖出。
2. 一百万初始资金；5bp每边的总成本假设；不加杠杆；北交所初始买入至少100股，整数增量；单股初始10%上限；买卖共用过去20日成交额1%的日容量。
3. 训练/验证按未来标签终点剔除重叠。模型季度更新，股票池每日更新，HMM月度更新。
4. HMM只使用前向滤波与一步预测；状态按训练内波动排序。现有资料下的除权现金和股份可用性是明确的研究情景。
5. Sharpe使用252交易日，报告无风险0和1.5%；Calmar使用日历年化收益/最大回撤。初始资金纳入峰值。现金收益记0。

## 文件说明

`ledger.py`连续账本；`train_models.py`模型与验证；`regimes.py`HMM；`factor_timing.py`因子收益择时；`intraday.py`日内因子；`jev_adapter.py`离线文本决策接入；`verify.py`金融约束测试。
`portfolio_extension.py`与`risk_refinement.py`均为明确标记的历史探索，不回写成预定验证结果。

文档用Codex托管Python生成，调用文档插件的规范渲染器进行逐页检查。研究与文档环境分开，避免修改系统Python依赖。

文档生成：使用托管Python运行 `build_report.py`；随后用同一Python运行 `tools/research_20260930/render_report_windows.py`，参数为生成的DOCX路径、`--output_dir` 和新的QA目录、`--emit_pdf`。LibreOffice的program目录需加入PATH。研究运行环境的已用版本记录在 `environment.json`；原始数据仍留在本机。
