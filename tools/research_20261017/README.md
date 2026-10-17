# 1016会议改版：运行与结果入口

本轮规则登记于2026-10-17，模型、账户和说明于2026-10-18完成。当前入口是项目根目录的 `新项目_20261017/项目改版_完整说明与实验结果.html`，逐条老师答复和具体日期教学文件在同目录。

## 本轮实际构成

1. `fetch_industry.py`、`parse_industry.py`：下载/解析官方历史行业。原件、发布日期和代码映射留在 `training_data/research_20261017/industry/`，按公开后的下一交易日启用。
2. `risk_metadata.py`：生成历史行业、规模、动量、波动分组。规模数据有历史发布/修订未验证的边界，见覆盖报告。
3. `models.py`：九个候选、八个季度；过去验证选择候选和树轮数；正式月度/季度重训；答案与交易评分分离。
4. `execution.py`：不重叠五日批次，单一现金账户，成交限制，股息/拆股、止损、到期、风险监测与逐单账。
5. `experiment.py`：过去验证选交易规则；13条同时间曲线及成本/信息延迟情景。主流程月度版是事前指定，不能根据全期成绩换成赢家。
6. `diagnose_models.py`：冻结预测后的IC、ICIR、因子更换、持有期/近期IC检查，不改变模型选择。
7. `report.py`：由冻结结果生成图表、HTML、主要说明、逐股亏损分解。
8. `test_models.py`、`test_execution.py`、`audit_accounts_ledger.py`、`audit_metadata_reproduction.py`：模型边界、资金约束、指标复算与风险数据复现核对。

## 复现命令

在项目根目录用 PowerShell 运行。`run_local.ps1` 默认优先使用现有 `.venv_research_20261011`，若不存在则使用 `PATH` 中的 `python`；也可以明确传入 `-PythonPath`。建议 Python 3.11，并按 [`requirements.txt`](../research_20261018/requirements.txt) 安装本轮实际依赖。运行前按 [`DATA_INPUTS.md`](../research_20261018/DATA_INPUTS.md) 准备有权使用的本地快照。

```powershell
# 只重新生成图表与报告，不训练、不改预测
& './tools/research_20261017/run_local.ps1' -Mode Report

# 检查已有模型/账户并生成报告
& './tools/research_20261017/run_local.ps1' -Mode Verify

# 使用本地冻结快照重算全部；会重新生成本轮模型及结果
& './tools/research_20261017/run_local.ps1' -Mode All

# 显式指定 Python（示例使用 PATH 中的 python）
& './tools/research_20261017/run_local.ps1' -Mode Verify -PythonPath (Get-Command python).Source
```

不需要联网即可复算已备齐的本地快照；上游模型输入是 `training_data/research_20261011/model_panel.parquet`，价格与企业行动在同目录。上述数据和虚拟环境不随代码发布，因此新机器必须先按输入说明备齐文件。官方原件已下载，不在每次运行时自动更新，以免改变研究输入。具体日期教学说明针对这份固定快照；换数据后须重新核对例子。

## 比较和证据

共同期间2024-10-08至2026-09-30，100万元开始。所有结果在 `training_runs/research_20261017/`：`strategy_summary.csv` 全部方案，`period_comparison.csv` 全期/一年/半年，`accounts/` 逐日逐单，`models/` 每季度因子筛选、验证分数和模型，`verification/` 独立核查。

主流程收益174.79%，最大回撤28.26%，夏普1.54、卡玛2.36；旧主流程相应49.23%、23.87%、1.03、0.94。主流程近一年仍亏9.39%，且月度更新未胜过季度更新。Q2/Q3空仓让后段曲线平直，不能作为选股能力增强的证据。

`run_manifest.json` 保留账户运行时的代码/输入哈希；`final_delivery_manifest.json` 另存交付时哈希及差异，不把后续文档变化伪装成当时运行版本。风险输入生成代码后续变化已在内存精确重建验证。当前历史曾被研究查看，所有曲线均属历史重演；上游因子发布时点、退市样本、真实成交与企业行动到账仍未达到实盘核验标准。
