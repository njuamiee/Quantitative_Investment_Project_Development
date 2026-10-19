# 复现所需的本地输入

本仓库的代码、协议、汇总结果和教学报告可以随项目发布；行情、逐股因子、分钟原始数据、公司行动及模型大文件不作为公开下载包。`training_data/` 和研究用 Python 虚拟环境被 `.gitignore` 排除。**只下载代码不能直接重跑历史收益。**完整复算需要在项目根目录按下列路径放入有权使用的本地快照，保持日期和字段口径不变；换一版数据要重新核对报告中的具体数字。现有汇总结果属于已经查看过的历史研究，不是未见样本。

| 路径 | 作用和至少需要的字段 |
|---|---|
| `training_data/research_20261011/model_panel.parquet` | 1017 模型及 1018 新因子、状态条件实验的股票日输入；`date`、`stockid`、`label_5`、`label_exit_5`、`x_*` 等因子。 |
| `training_data/research_20261011/prices.parquet` | 两版交易账本的逐日行情和股票池；`date`、`stockid`、`open`、`close`、`adv20`、`observed`、`mom20`、`vol20`、`in_pool`、`adj_ret1`。1017 账本还会用到该文件中的其他日线字段，具体以源代码读取列为准。 |
| `training_data/research_20261011/actions.csv` | 逐股、逐日分红拆股；`date`、`stockid`、`dividend`、`split`。 |
| `training_data/research_20261011/minute_factors.parquet` | 分钟线汇总成的 10 列日频因子及 `bars`；由 `tools/research_20261011/intraday.py` 生成。本轮实际仅有 2025Q4 和 2026Q3 覆盖。 |
| `training_runs/research_20261011/regime_probabilities.csv` | 过去数据逐月训练的 HMM 次日状态概率；状态条件实验和风险门槛使用。由 `tools/research_20261011/regimes.py` 生成。 |
| `training_runs/research_20261017/models/daily_predictions.parquet`、`models/folds.csv`、各季度 `valid_predictions.parquet` 和 `deployment_info.json` | 1018 风险比较与分钟增量实验使用 1017 已冻结的日频预测及其历史验证记录。分钟实验会核对 2025Q4、2026Q3 的最后训练标签结束日。 |
| `training_runs/research_20261011/minute_factor_h5_predictions.parquet` | 旧分钟预测，仅用于新分钟实验的一致性核查；测试标签列不会读入模型。 |
| `training_data/research_20261011/bj_prices.parquet`、`training_data/research_20260930/bj_share_capital.parquet`、`training_data/research_20261017/industry/industry_history.parquet` | 若从头重建 1017 风险分组和行业资料，需要这些历史快照；重新下载官方资料可能改变输入版本。 |

1018 还直接复用 `tools/research_20261017/models.py` 的因子筛选函数、`tools/research_20261011/factor_dsl.py` 的受限公式计算，以及旧 `intraday.py`、`regimes.py` 生成的派生输入。若要从更早的原始供应商数据开始重建，需要对应的 1011 数据处理模块、合法原始数据和当时可得日期资料；本目录不声称提供了无需数据授权的端到端重建。

在 Python 3.11 环境安装本轮已观察到的直接依赖：`python -m pip install -r tools/research_20261018/requirements.txt`。1017 可运行 `./tools/research_20261017/run_local.ps1 -Mode Verify`，或用 `-PythonPath` 指定解释器；1018 的顺序见同目录 `README.md`。远程同步脚本只用于用户已有授权服务器，连接资料和凭证必须留在本地，不写进仓库或命令行示例。
