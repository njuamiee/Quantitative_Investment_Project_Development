# 股票策略的消息驱动迁移

本目录是独立分支 `codex/vnpy-message-driven` 的执行系统。研究代码及原有实验结果保留不变。

用户在 2026-10-28 明确：目前只需要模拟交易，不接真实交易账户。交付重点是历史 Tick 回放、原生 DeltaOne 引擎验证、前视与成交规则测试，以及未来实时行情的统一接入口。

2026-10-29 起，原生回放已改为 **ClickHouse → vn.py Database → DeltaOne 原始加载器 → 引擎 → 策略回调**。不再用函数替换把Parquet文件直接喂给引擎。历史原始文件只在一次性入库时读取；回放中的每个“当日”仍来自历史数据库，不是当日实时行情。行情订阅仍未接通，vn.py本身不产生行情。

## 从研究到消息驱动

研究模型先产生带时间戳的股票评分。执行系统依次接收信号、行情、委托回报和成交回报，每一步只使用当时已经收到的数据。发出委托不等于成交，只有成交消息才改变持仓。

```
已到时点的特征 → 模型文件 → 带有效期的信号
                                 ↓
历史 Tick / 未来实时行情 → 策略 → 风控 → 委托
                                 ↑          ↓
                            持仓和资金 ← 成交回报
```

第一版 `SimpleMomentumStrategy` 只有一只股票，以已经结束的 3/8 个分钟观察值计算均线，目标仓位上限 10%，只做多。缺失分钟不虚构价格；Tick 模式收到下一分钟首条行情后，才确认上一分钟的收盘观察值。这是执行检验基线，不是收益优化结果。

完整执行版 `TargetStrategy` 接收横截面评分，支持股票池、选股数、每日最多买入股票数、等权/逆波动权重、仓位和波动率目标、HMM 市场上限、行业与规模分组预算、反弹条件、分钟信号增量、个股止损/跟踪止损、组合回撤停机和冷却期。行业/规模约束是生成目标时的预算约束，并非每一笔行情后强制行业中性；行情跳空、停牌和成交限制会使实际权重偏离目标。

`models.py` 支持 Ridge、LightGBM、LightGBM Ranker、XGBoost、ExtraTrees、CatBoost、IC、条件 IC 的模型文件保存和加载。`production.py` 提供日/月/季重训边界判断以及训练、信号发布命令；并未创建定时任务。模型输入必须包含特征可用时间、收益标签成熟时间，实时特征生产器和供应商发布时间仍需接通。该模块的因子筛选是训练内逐日 IC，不能冒充旧研究的所有规则一模一样；模型卡明确记录差异。

## 三种运行方式

| mode | 行情输入 | 成交由谁决定 | 真实下单 |
|---|---|---|---|
| paper | 未来盘中 Tick | 本项目 PaperBroker | 否 |
| backtest | 历史 Tick | 已安装的 DeltaOne BacktestingEngine | 否 |
| live | 未来真实行情 | 券商成交回报 | 本阶段不启用 |

Windows 自建回放测试与 Linux 原生引擎测试是两组不同证据，不能混为一谈。分钟数据回放中的买卖盘是明确标记的人工价差；原生 Tick 回放使用数据中实际的五档买卖盘及涨跌停价，但仍是快照撮合，不是逐笔队列重建。

## 已识别的平台和数据问题

1. 使用量化账号连接后，通过 `sudo -iu vnpy` 运行 `/home/vnpy/.venv/bin/python`，不切换成 root 运行策略。sudo 密码仅从既有本地配置读取，通过标准输入传递。
2. 原生运行时为 Python 3.13，vn.py 4.5.0，包含二进制 DeltaOne 和 Level2。项目独立放在 `/home/vnpy/quant_message/`。
3. 原生 `is_equity('920000.BSE')` 返回 False。`native_validation.enable_bse_equities()` 仅在当前回放进程补充 BSE 分类，不修改商业版二进制，不把北交所伪装成上交所。
4. 初查公共 ClickHouse 的 26,805 条 Tick 概览均标为 SSE，BSE 查询为空。新样本通过已安装驱动的 `save_tick_data` 导入独立库 `quant_message_20261029_base`；扰动样本在 `quant_message_20261029_mutated`。107只股票逐字段校验读回散列、共1,300,881条。公共库不改动，数据库选择只影响回放进程的内存配置。原始SQL调用记录随结果保存。
5. 原生环境缺少 Arrow，项目依赖目录单独安装 `pyarrow==21.0.0`。Arrow仅用于数据整理和入库，回放函数没有Parquet/CSV行情入口。
6. 原生报表函数含链式赋值，在 pandas 3 下触发兼容性告警。回放通过独立依赖目录使用 `pandas==2.3.3`，关闭 Copy-on-Write，避免静默使用可能没更新成功的收益表；不降级整个 vn.py 环境。
7. 分钟数据存在全空占位行和成交额/成交量不符合 OHLC 的行，回放隔离这些记录并生成质量清单。Tick 导入也过滤无有效成交价、缺失字段或买卖盘倒挂行。隔离行数量必须随结果一起阅读。
8. 在 pandas 2.3.3 下，多股原生资金汇总仍只返回两个日期且各字段为空；因此不能把兼容性修复视为财务报表正确的证明。`portfolio_audit.py` 从成交、每日末笔有效报价和公司行为重建全部交易日，复算手续费、现金、持股和净值，并与执行账本交叉核对。异常原表保留，`native_statistics_valid=false`；正式图表使用 `audited_equity.csv`。单股同费用、无股息口径的重建曲线与原生表完全一致。

## 时间与账户约束

- 信号同时记录观察时间、可用时间、最早执行时间、失效时间、模型训练时间和最后成熟标签时间。历史假设发布时间的研究信号始终标为 `research_only`。
- 自建撮合不允许在触发下单的同一 Tick 成交；分钟完成后才能用于决策，成交量也只能限制后续成交。
- 现金/股数冻结、T+1、部分成交、累计手续费和同一行情的共享流动性预算均纳入自建账本。
- 限价不能被滑点模型突破。原生成交额外检查成交价是否越过委托价、成交时间是否晚于提交时间。
- 支持订单回报先到、成交回报后到；尚未核对的成交不会提前释放全部冻结资金。重复成交不重复记账。
- 原生回测引擎的费率与绝对价格滑点统计、自建账本的最低佣金/卖出税费是两个明确列示的口径，不能直接把两条资金曲线解释成策略差异。
- 纸面账户支持除权除息及停牌期间估值调整；真实账户的公司行为、外部手工交易和断线后持仓恢复尚不在本阶段验收范围。
- 本次原生回放的税前股息仅记应收款，不增加可买现金；没有已确认的到账日和持有期税款，故净值仍是税前口径。原生股份拆分遇到非1比例时会拒绝运行，防止外部引擎持仓与项目账本不一致。历史分钟对照保留原研究的除息日立即到账假设，不能与本次原生股息处理混淆。
- 原生图表用双边0.05%费率，加每股0.01/0.05元绝对滑点；滑点作为额外成本计入曲线。下单预算和`shadow_equity`用本项目最低佣金/卖出税费。两种口径均有独立列；后者与运行账户核对，前者用于引擎滑点敏感性对照。

## 运行

本地核心和模型测试：

```powershell
.\.venv_research_20261011\Scripts\python.exe -m pytest tests/test_message_execution.py tests/test_message_data_models.py tests/test_message_production.py tests/test_message_portfolio.py -q
.\.venv_message\Scripts\python.exe -m pytest tests/test_message_deltaone.py -q
```

一键上传并执行原生模拟验收（依赖和样本已在服务器准备好后）：

```powershell
.\.venv_research_20261011\Scripts\python.exe tools/vnpy_message/acceptance.py
```

脚本执行简单版的延迟/滑点对照和完整信号版回放，将机器报告、成交单和资金表取回 `training_runs/vnpy_message/native/`。它不会连接交易柜台。原始数据、模型文件、环境、运行日志不推送 Git。

重建输入时依次运行 `tools/vnpy_message/prepare_native_inputs.py` 和 `tools/vnpy_message/stage_inputs.py`，然后在服务器调用 `python -m quant_message.database_io --data-dir <整理后的输入目录> --database <新的任务专属库> --output <新清单目录>` 一次性入库。库名必须以 `quant_message_20261029_` 开头；非空库拒绝再次导入，避免重复数据。项目模型在本地研究环境训练并导出带时间戳的信号；未声称所有机器学习依赖已在商业版Python3.13里安装或不同Python环境的pickle可以互换。

验收样本：2026-09-23、24、28、29、30日，107只股票，1,300,881条有效Tick。完整信号版产生43条成交回报；简单单股版产生11条。另有2026年第三季度65个交易日的六组分钟代理对照。这些是技术验收，不是新的参数搜索，也未达到或证明此前设定的收益/夏普/卡玛目标。

一键模拟验收已经提供；一键真实交易不在当前交付结论内。未来仍需行情授权及适配、真实特征发布时间、券商回报契约、账户恢复和实盘前联调。

在服务器上直接运行，先 `sudo -iu vnpy`，以实际发布目录替换 `<release>`：

```bash
export PYTHONPATH=/home/vnpy/quant_message/deps:/home/vnpy/quant_message/releases/<release>
/home/vnpy/.venv/bin/python -m quant_message.native_validation --data-dir /home/vnpy/quant_message/data/database_base --output /home/vnpy/quant_message/results/new_run --kind simple
```

`--data-dir`在回放时只指向数据库清单、日期、合约集合、已生成信号和公司行为文件；其中没有行情文件。策略既不直接查数据库，也不读取历史Tick文件。`--execution paper`仅切换成自建撮合，仍由同一原生引擎从同一数据库推送Tick，且只允许在BacktestingEngine中使用历史时钟。

严格对照命令：本地运行 `tools/vnpy_message/database_experiments.py`（数据库样本已准备），完成14组回放后运行 `tools/vnpy_message/compare_database.py`。输出到 `vnpy数据库对照_20261029`。包含两种策略、两种撮合、基准/前缀截断/未来扰动，以及旧文件加载与新数据库加载的回归比较。`tick_depth`和`queue_depth`是原生引擎设置，不将其未经文档验证地解释为同等的盘口成交量约束。

未来扰动检查截至2026-09-24的消息序列、信号目标、委托意图、成交和逐日净值；未来行情乘1.15、未来评分反转并改变仓位上限，未来结果必须实际改变。八类模型另测试未成熟标签及未来特征被篡改后拟合输出不变。历史因子来源、真实发布时间和样本股票池的可得性并未由这些检查自动证明。

这次对照修复了自建撮合将float32报价噪声误取整为额外一档价格的问题；先归一化报价的微小误差，再施加真实滑点。原2026-10-28分钟代理报告保留为修复前证据，不能当成本次版本的新运行结果。

## 实时行情该如何选择

实时行情是交易时持续送来的市场消息；历史 Tick 是过去消息的录制。只接行情、模拟成交无需真实下单账户。优先询问现有供应商是否提供北交所实时 Level-1、明确时间戳/股数单位/推送频率/涨跌停字段，以及 Linux 接入方式。若已有服务不支持，再比较迅投研、米筐或其他持有相应权限的数据服务。vn.py 已安装某个插件，不代表账号已开通所有数据权限，也不证明历史查询接口能直接连续推送实时消息。

可直接向供应方表述：**目前只做模拟交易，需要 Linux 可接收的北交所股票实时 Level-1；优先复用现有服务。请确认覆盖范围、五档盘口、成交量单位、时间戳、延迟、断线补发及费用。** 当前回放可继续完成；没有已确认的数据订阅时，不宣称今天的实时模拟已联通。

`python -m quant_message.streaming --symbols 920000.BSE --output runtime/new_paper_run` 可从标准输入接收供应商适配器输出的 JSONL，运行纸面账户。输入字段遵循 `domain.Tick`，必须提供当天真实接收时间和增量股数；此入口没有任何交易网关。当前没有虚构供应商连接或把历史回放当成实时接收。

依据：[vn.py 数据服务](https://www.vnpy.com/docs/cn/community/info/datafeed.html)、[事件引擎源码](https://github.com/vnpy/vnpy/blob/master/vnpy/event/engine.py)、[Tick/Order/Trade 数据结构](https://github.com/vnpy/vnpy/blob/master/vnpy/trader/object.py)、[北交所行情授权说明](https://www.bseinfo.net/application/guide.html)。商业版接口以服务器实际签名和用户提供模板为准。
