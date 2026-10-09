# 10月8日扩展研究与10月9日汇报

本轮在提交并推送旧成果`3407076`之后开始。结果目录为`training_runs/teacher9_20261008/`，原始输入快照为`training_data/research_20261008/`。旧研究结果保持冻结；当前学习材料见[学习笔记](../../学习笔记/README.md)。所有脚本从仓库根目录执行。

## 已完成范围

- 北交所304候选特征，训练内覆盖筛选及每日横截面Spearman去重后，每季217—223个合格候选；20/40/60/80/100五档均实际选足，各8季度。
- LightGBM和Ridge五档对照；按每轮过去126日验证Rank IC选数量，清除标签跨界样本；固定窗口2024-10-08—2026-09-29。
- 7/15/31叶子、Alpha101/191分组、前日股本代理、额外2日Alpha延迟、相同抽样规则的两个随机种子、5/15/30基点成本。
- 独立连续分钟资金账户：2025Q4与2026Q3至9/29，各自100万元；0.5%/1%/2%参与率，共享分钟容量、部分成交、延期卖出、T+1及逐批持股。
- 348只比例后复权文件用于两种学习目标/特征对照；2147个有效非未来分红事件核对、即时可用情景和单列现金应收。
- 严格无转移先验HMM拟合筛选及简单仓位对照；496只来源股票的沪深Alpha101研究，113候选、31季度。
- 8张实际实验图、59项研究核对、41页入门Word和23页科研PPT。

以上是历史研究，并不构成真实收益验收。选择过的历史不变成全新测试；所有探索尚未做全族显著性校正。当前股票来源样本仍受幸存者偏差影响，分红结算日、股份可卖日、税费及发布时间仍待补。

## 输入与运行依赖

研究解释器：`.venv_quant/Scripts/python.exe`，沿用9月30日的pandas、numpy、scipy、LightGBM、scikit-learn、hmmlearn、pyarrow、matplotlib、paramiko等环境。脚本复用`tools/research_20260930/`的准备、训练和账户函数，并读取那一轮冻结的面板、预测和账户。

`sync.py`仅从SFTP读取：日线、指数、分红、指定Alpha快照、9月底新分钟，以及待核实的两份原始归档。它依赖已保存的`training_data/audit_20261008/inventory.json`和10月2日清单，修正分钟目录改为年份子目录后的扫描结果。`audit_summary.py`汇总目录、日期与归档内容证据。它们是本次快照工具，不应把运行时最新数据默认为原实验输入。

2026Q3的65个分钟输入已另外按年目录抓取至本轮`minute/`，Q4分钟复用旧快照；`minute_account.py`优先读本轮文件，缺失时读旧快照。本轮原始文件、抓取清单和审计快照留在本地且被Git忽略，仓库保存代码、模型及研究产物。只下载仓库而没有这些输入，不能从零重训。

## 复现顺序

下列命令读取已有输入快照，会重写**本轮**同名输出；需留存另一组版本时先复制结果或修改输出目录。训练脚本存在缓存，修改参数时也必须使用独立目录，不能把旧缓存当新训练。

```powershell
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/experiments.py'
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/new_data.py'
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/hfq_models.py'
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/hmm_robustness.py'
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/minute_account.py'
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/minute_account.py' --q3
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/shsz_alpha.py'
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/analyze.py'
& '.venv_quant/Scripts/python.exe' 'tools/research_20261008/verify.py'
```

`experiments.py --accounts`只重建缺失的账户，不重新训练。`shsz_alpha.py`在缺少本轮清单时会下载所需101个Alpha及对应股票日线；已有输入时直接使用。`analyze.py`生成汇总、图表和按日配对的10日块自助区间；`verify.py`检查资金和时间逻辑，不训练。

`late_snapshot.py`另读取晚间终检快照，通过SFTP下载10月8日c01/c02，保存100只当日池内股票的研究评分。文件明确记录实际抓取时间、使用2026Q3冻结模型、未按Q4重训、未来标签未成熟。它不是实盘订单或已获得结果的前瞻验证；重复运行会重新抓取供应商当前版本。

Word生成：用文档运行时Python执行`build_guide.py`，再用`tools/research_20260930/render_report_windows.py`渲染到`training_runs/beginner_guide_20261008/docx_qa_final/`，逐页检查后执行`verify_deliverables.py`。`build_results_note.py`生成最新结果清单。PPT由artifact-tool JSX生成，图表来自本轮`figures/`，正文和表格为可编辑对象，23页均含讲者备注。

## 关键产物与核对

| 文件 | 用途 |
|---|---|
| `final_summary.json`、`extension_results.csv` | 当前数字总表与21种模型/选择方案 |
| `extension_protocol.json`、`folds.json`、`validation_models.csv` | 固定规则、训练/验证/测试边界与事前选择 |
| `bj/*/selection.json`及模型文件 | 各轮训练内筛选、实际因子与拟合产物 |
| `*_account.csv`、`*_fills.csv`、`*_orders.csv` | 重建现金、持股与分钟成交；并非每个日线方案都重复输出成交明细 |
| `shsz_alpha/` | 新沪深同样本对照；不能直接与旧300只样本归因比较 |
| `dividend_summary.json`、`dividend_event_audit.csv`、`hfq_results.json` | 已有事件、缺字段及两类不同的复权对照 |
| `sftp_update_summary.json`、`late_server_check.json` | 下午检查及10月8日23:59终检；以晚间状态为当前状态 |
| `oct8_frozen_model_monitoring.csv` | 实际留存的冻结模型观察评分及抓取时点 |
| `verification.json` | 59项研究检查通过；不等于59个金融有效性证明 |
| `../beginner_guide_20261008/document_qa.json` | 最终交付文档、PPT的结构、数字与文件哈希核对 |

40因子预测及价格账户逐值复现旧版本。相关矩阵实现改用`fast_correlation.daily_spearman`：按相同缺失模式分组，在每对共同样本上重新排名，保持原pandas逐对Spearman定义；并列值/不同缺失模式测试最大差异为0。

## 当前服务与必要缺口

10月8日23:59：SFTP可读，健康接口HTTP 200；120张因子表到10/8，任务120成功、0失败；分钟任务新增0且文件仍到9/30，BJ/SH/SZ当日原始tick均未见。分钟目录为`bar1m/BJ/YYYY/YYYYMMDD.parquet`，清洗tick仍是平铺目录。

需要连续全历史BJ分钟时补2025-06-30、2026-05-18和2026-10-08。6/30原始BJ tick存在，可按上游流程重建；另两天需要上游原始资料。本轮两段分钟实验无需前两天的缺口。SZ的20260105归档与BJ逐字节相同且内容全为BJ股票/指数，深圳高频研究应更正；本轮沪深日线没有读取它。

最重要的剩余证据仍是历史上市/退市/停牌/特殊状态、代码历史、因子和财报真实发布时间及修订、分红支付/股份可卖日/税费，以及商业平台环境与验收规则。完整解释和精确路径见当前讲义及清单。
