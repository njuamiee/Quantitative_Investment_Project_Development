# 老师九项意见：2026-09-30 实施版

本目录为独立研究流水线，使用工作区根目录的 `.venv_quant/Scripts/python.exe`。结果在 `training_runs/teacher9_20260930/`，输入冻结在 `training_data/research_20260930/`。不覆盖首轮 `lgbm_hmm_20260920`。

## 已运行的内容

- 每日成交额 Top100、历史可交易性代理、独立入池及因子覆盖记录。
- 120 表、2508 非键字段的数据字典映射：2447 数值候选，61 辅助字段。最新快照审计不等于全历史验收。
- 北交所 Alpha101 + Alpha191 + 12 价格特征共 304 候选，每轮训练内 IC、HAC、BY-FDR 报告及日截面 Spearman 去重，保留最多 40 个。
- 北交所 8 季度、沪深 31 季度滚动，隔离训练/验证/测试标签；LightGBM、Ridge、单因子、价格模型对照。
- 现金/股数账户、T+1、成本和流动性约束，指数、随机选股、额外一天因子延迟、HMM 和简单减仓对照。
- 61 个分钟文件检查及 2025Q4 的 558 笔订单级压力回放；历史信号时间门控与 vn.py 离线适配器。
- 071/601 历史股本面板，含 22954 条口径顺序异常标记；异常行不生成换手率代理，股本未加入已评估主模型。

## 复现顺序

以下命令从工作区根目录运行；会重建本轮产物，不建议只为查看报告而重复训练。SFTP 下载使用现有连接资料及固定服务器身份校验，脚本不上传或删除服务器文件。字典映射和 Word 构建使用已配置的文档运行时，不能用缺少 openpyxl/docx 的研究环境替代。

```powershell
$py = '.venv_quant/Scripts/python.exe'
& $py tools/research_20260930/fetch_data.py prices
& $py tools/research_20260930/fetch_data.py factors
& $py tools/research_20260930/fetch_data.py samples
& $py tools/research_20260930/fetch_data.py minute
& $py tools/research_20260930/fetch_data.py capital
& $py tools/research_20260930/prepare.py
$docpy = 'C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $docpy tools/research_20260930/map_dictionary.py
& $py tools/research_20260930/train.py
& $py tools/research_20260930/accounts.py
& $py tools/research_20260930/replay.py
& $py tools/research_20260930/prepare_capital.py
& $py tools/research_20260930/test_research.py
& $py tools/research_20260930/verify_outputs.py
& $py tools/research_20260930/analyze.py
& $docpy tools/research_20260930/build_report.py
```

`prepare.py` 还使用既有 `factor_demo_run/factor_lib` 中的 2025 年及以前 Alpha101；全部来源见源码与各下载 manifest。Word 渲染命令见构建脚本说明及 documents skill。

## 结果的使用边界

主账户是**未复权价格研究账户**：真实分红送转台账、历史股票资格、逐日涨跌停及实际费率仍不完整。15bp 为单边合计成本假设；日线开盘/收盘成交为参考撮合。最后五个交易日停止开新批次，用于统一评价完整五日持有期；每日最新信号另行导出。Top100 等权是目标分配，100 股门槛等约束使实际仓位低于等权基准。

沪深是从当前可用文件确定性抽取的 300 只股票，含幸存者偏差，使用 12 个价格特征，不能声称是完整沪深 Alpha 扩展验证。历史供应商文件可能被修订，按文件日收盘后 18:00 可用是研究假设，未获得真实发布时间证据。分钟回放只重放日线已成交参考订单，不产生独立资金收益曲线，也没有订单队列模型。vn.py 商业版环境未安装验收。

结果未用于实盘或模型上线。参数在本轮测试前固定，未按外层测试收益调参；研究结论仍需新的未来样本确认。
