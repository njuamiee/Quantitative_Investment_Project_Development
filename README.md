# 量化选股项目：当前入口

整理日期：2026-09-23。当前主线是基于历史时序因子的横截面选股。

优先阅读：[老师九项意见——分析与解决方案](学习笔记/老师意见分析_20260923/老师九项意见_分析与解决方案.md)。其中包含现状、专业方案、实施顺序，以及资金曲线和 HMM 诊断图。

| 内容 | 唯一工作入口 |
|---|---|
| 最新完整代码 | `_review_ts_factors_master_20260920/ts-factors-master/`（目录虽带 review，仍是本地最新代码） |
| LightGBM 与 HMM 首轮训练 | `_review_ts_factors_master_20260920/ts-factors-master/modeling/train_lightgbm_hmm.py` |
| 首轮模型与研究结果 | `training_runs/lgbm_hmm_20260920/`，冻结保留，不覆盖 |
| 101 个 Alpha 因子 | `factor_demo_run/factor_lib/`；该父目录名称沿用历史，仅保留数据及样例 |
| 未复权行情快照 | `training_data/snapshot_20260920/unadjusted_daily/`，333 个行情文件 |
| 本次评审与图表 | `学习笔记/老师意见分析_20260923/` |
| 清理记录 | `cleanup_20260923/清理说明.md` 及逐文件 manifest |
| 本地运行环境 | `.venv_quant/Scripts/python.exe`，64 位 Python 3.11 |

当前事实：2025 年每日评分 245～263 只股票，尚未限制 Top100；LightGBM 尚未多轮滚动；最新训练用未复权行情；当前 HMM 未通过验收。按老师意见，后续因子研究暂按当日收盘后可用处理。新的价格、股票池和训练方案应输出新的 run。

最新版里的 `factors/` 和 `backtest/` 仍包含可复用的旧因子评价、滚动规则、绘图及测试。它们不是 LightGBM 滚动训练的替代品；其中默认相对数据路径和静态 Top100 名单不能直接代表本机正式数据配置。使用时显式指定实际因子与行情路径。旧源代码快照中的 AGENTS/README 保留了原作者环境描述，本机使用上述虚拟环境。

本次图表复现（从本目录执行）：

```powershell
& '.venv_quant/Scripts/python.exe' 'tools/review_latest_results.py'
```

该命令只读取冻结产物并生成图表和核查表，不训练模型、不改写首轮结果。

论文、老师任务原件、供应商说明、样例输入及唯一的前复权行情样本继续保留；它们没有已确认的替代品，文件较早不等于已经失效。连接资料不作为报告内容传播。
