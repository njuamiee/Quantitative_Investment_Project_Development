# 项目说明（AGENTS.md）2026/9/16

本目录 `d:\jzhang\data-kyts\src` 是量化研究代码目录，主要包含三部分：

1. **`utils.py`** — 通用工具函数：期货交易模拟 (`simulate_futures`) 与绩效评估
   (`evaluate_futures`)、分钟线重采样 (`resample_minute_bars`)、K 线前复权
   (`forward_adjust_daily_bars`)、K 线/信号绘图 (`plot_candles*` 等)。
2. **`factors/`** — 北交所股票因子库管理系统 + 因子评价（IC 分析）+ 因子绘图（详见下文）。
3. **`backtest/`** — 分桶信号模拟交易（K 线/因子/持仓/总资产四子图出图），
   独立 CLI + 已集成进因子评价流程；另有滚动回测与 rolling IC+OU 概率信号回测
   （bactest_ic.py，6 子图）（详见下文）。

目录结构参考 A 工程（`D:\jzhang\work\options\superTrends`）的 factors 包：

```
src/factors/
├── registry.py     # 因子注册表：factor_list.xlsx → 唯一英文标识
├── builder.py      # 因子库构建：日期外层循环 + 内存累积 + 一次性写出
├── library.py      # FactorLibrary：init/update/list_factors/list_stocks/get
├── price_data.py   # BJ 股票日线加载（day-data/前复权，GBK），供评价与绘图
├── eval_config.py  # 【移植自 A】EvalConfig + Newey-West 滞后阶数规则
│                   #   （本项目新增 run_backtest/backtest_k/backtest_fee 三字段）
├── stats_utils.py  # 【移植自 A】Spearman IC、OLS + NW HAC（statsmodels 可选）
├── evaluator.py    # 【移植自 A】evaluate_factor / forward_returns
│                   #   （本项目新增：_evaluate_symbol 末尾跑分桶模拟交易）
├── report.py       # 【移植自 A】FactorReport：summary()/to_markdown()/plot()
│                   #   （本项目新增：BT_* 汇总列、模拟交易 markdown 节、出图与 metrics CSV、
│                   #     save_rolling_ic_csv 滚动 IC 数值 CSV 导出）
├── plotting.py     # 因子绘图：个股日 K 线（叠加 EMA）+ 因子曲线，复用 utils.py
├── main_eval.py    # 因子评价 CLI（对应 A 的 factors/main_eval.py）
├── main_batch.py   # 批量评价 + 模拟回测（多进程并行）：成交额 TopN 股票 × 全部因子，
│                   #   全 k × 全桶 × 多空回测 + 买入持有基线；每股票×因子一个子目录
├── sign_forecast.py # rolling IC 正负预测：4 模型（base/persist/ou/logit）× k×h×W
│                   #   网格评估 + 面板热力图（复用 report.rolling_ic_frame 的 IC 序列）
├── main_sign.py    # 上述 CLI：python -m factors sign <factor> <stock>
├── tests/          # 评价模块 pytest（24 个用例，移植自 A 并适配；含 main_batch/sign 用例）

src/backtest/
├── engine.py       # 纯计算核心：BacktestConfig/BacktestResult/run_backtest
│                   #   + buy_and_hold_equity/buy_and_hold_metrics 买入持有基线
├── plotting.py     # 四子图绘制（复用 utils.py 的 K 线与时间刻度）
├── bt_analysis.py  # 回测指标分析：直方图/RankIC 散点 + filter_high_performance_combos
│                   #   高绩效组合筛选（滚动回训的选股池规则复用该函数）
├── rolling.py      # 滚动回测：每 M 日重训候选池（窗口 N 日 IC+回测筛选）+ 逐日模拟，
│                   #   现金均分开仓只做多；CLI python -m backtest.rolling（详见下文）
├── bactest_ic.py   # rolling Rank IC + AR(1)/OU 概率信号回测（两档仓位只做多）；
│                   #   CLI python -m backtest.bactest_ic（详见下文"IC 信号回测"一节）
├── __init__.py     # 公共 API + run_factor_backtest / sweep_backtests 遍历包装
├── __main__.py     # 独立 CLI：python -m backtest 920725 Alpha098（缺省全桶×多空扫描；--side 3 --direction short 单跑）
└── tests/          # pytest（引擎 10 + bt_analysis 5 + rolling 9 + bactest_ic 4，确定性合成数据）
```

输出目录 `<项目根>/reports/`（评价报告 markdown/summary CSV/图，运行时生成；
**与 src 平级**，同步 src 目录时不会带上这些大体积产物）。

## 运行环境（重要）

- **必须使用 64 位 Python 3.11**：`D:\jzhang\setup\python64\Python311\python.exe`
  （已装 pandas 2.3.3、numpy 2.4.4、pyarrow 23.0.1、openpyxl 3.1.5、
  scipy 1.17.1、matplotlib 3.10.8、pytest 9.0.3；statsmodels 未装，
  评价模块的 HAC 自动回退到自实现）。
- 系统默认的 `python` 是 32 位 Python 3.8，**没有也无法安装 pyarrow**，
  不能用于因子库代码（读 xlsx 可以用它，但没必要）。
- Windows 控制台中文可能乱码，运行时建议设 `PYTHONIOENCODING=utf-8`。

## 数据目录结构（运行本项目所需）

```
d:\jzhang\data-kyts\                  # 项目根
├── src\                              # 代码目录（本文件所在）
│   ├── factor_list.xlsx              # 因子定义表（表名/序号/字段/因子名/类型/备注，101 行）
│   ├── config.json                   # 全局配置：[batch] njobs/min_bars/plot_sides/plot_ks
│   ├── factors\                      # 因子库 + 评价 + 绘图代码包
│   └── utils.py                      # 通用工具（模拟/评估/重采样/绘图）
│
├── reports\                          # 【输出】评价/回测报告（与 src 平级，运行时生成，
│                                     #   markdown/summary CSV/png 图，不进 git）
│
├── factors\kyzqsh11\                 # 【输入】原始因子数据（天软科技下载，全部约 3.6GB）
│   └── ts_stockfactordata_c01\       # 表目录，对应 factor_list.xlsx 的"表名"列
│       └── <年份>\                   # 2016 .. 2025（目前仅用到 c01 这一个表）
│           └── <YYYYMMDD>.zip        # 每交易日一个 zip，内嵌同名 csv，约 2430 个文件：
│                                     #   列为 stockid / enddate / oho_alpha_001..101
│                                     #   每行一只股票（stockid 如 SH600051 / SZ000001 / BJ920010）
│
├── day-data\                         # 【输入】A 股股票日线数据（5868 只股票）
│   ├── 前复权\                       # ← 本项目使用此前复权基线（评价与绘图共用）
│   │   └── <6位代码>-<名称>.csv      # GBK 编码；列为 日期/股票代码/开盘/收盘/最高/最低/
│   │                                 #   成交量/成交额/振幅/涨跌幅/涨跌额/换手率
│   ├── 后复权\  不复权\              # 备用，本项目代码不读
│   └── A股-股票日线数据.zip          # 原始打包文件
│
└── factor_lib\                       # 【输出】构建产出的因子库（init/update 写入，约 100MB）
    ├── registry.csv                  # 因子注册表（factor_id/因子名/字段/表名/备注，utf-8-sig）
    └── <factor_id>.parquet           # 每因子一个文件：date/stockid/value 长表
```

关键映射关系：

- **因子字段**：`factor_list.xlsx` 的"字段"列 `oho_alpha_001` ↔ 因子 `Alpha001`，
  数据位于 `factors\kyzqsh11\ts_stockfactordata_c01\<年份>\<日期>.zip` 的同名 csv 列中。
- **股票代码**：因子库 stockid `BJ920010` → 日线文件 `day-data\前复权\920010-凯添燃气.csv`
  （去掉 `BJ` 前缀匹配 `<代码>-*.csv`，见 `factors/price_data.py`）。
- 原始 zip 目录中可能混有已解压的同名子目录（如 `20251231\`），扫描时只认 `*.zip`。
- 日线数据起点可能早于因子库（如 920010 自 2020-07 新三板精选层起），
  评价/绘图按两者日期交集对齐；北交所因子自 2021-11-15 开市起才有。

## 因子库（src/factors/）

### 数据流

```
factor_list.xlsx (因子定义: 表名/字段/因子名/备注)
        │
        ▼
registry.py ──► registry.csv (factor_id 唯一英文标识, 如 Alpha001)
        │
原始数据: d:\jzhang\data-kyts\factors\kyzqsh11\<表名>\<年份>\<YYYYMMDD>.zip
        │   每个 zip 解压为 csv: stockid / enddate / oho_alpha_001..101 等列
        ▼
builder.py  日期外层循环 → 过滤 BJ 开头股票 → 内存累积 → 每因子一次性写出
        │
        ▼
因子库: d:\jzhang\data-kyts\factor_lib\<factor_id>.parquet
        长表格式, 列: date / stockid / value, 按 (stockid, date) 排序
        （101 个因子 = 101 个文件 + registry.csv，共约 100MB）
```

当前库内容：101 个 Alpha 因子（Alpha001..Alpha101）、263 只北交所股票、
2021-11-15 ~ 2025-12-31，每因子约 17.6 万行。

### 命令行用法

在 `src` 目录下运行（**必须在 src 下**，否则 `python -m factors` 会误匹配到
项目根的数据目录 `d:\jzhang\data-kyts\factors\`）：

```bash
cd d:\jzhang\data-kyts\src
PY="D:/jzhang/setup/python64/Python311/python.exe"

$PY -m factors init      # 全量重建因子库（约 10~20 分钟，覆盖写）
$PY -m factors update    # 增量更新：只处理库中最大日期之后的新 zip，幂等
$PY -m factors list      # 列出所有因子（factor_id/因子名/字段/表名/备注）

# 因子评价（IC 分析，详见下文）
$PY -m factors eval Alpha001                        # 全部 BJ 股票汇总评价（约 1~2 分钟）
$PY -m factors eval Alpha001 --stock BJ920010       # 单只股票详细评价 + 出图
$PY -m factors eval all --stock BJ920725            # 单只股票 × 全部因子（all 表示所有因子）
$PY factors/main_eval.py Alpha001 --horizons 1,5,10 # 等价的脚本入口（与 A 工程一致）

# rolling IC 正负预测模型对比（详见下文"Rolling IC 正负预测"一节）
$PY -m factors sign Alpha098 BJ920725               # 4 模型 × k×h×W 命中率面板热力图
$PY -m factors sign Alpha098 BJ920725 --horizons 1,5,10 --hs 0,5,10,20

# 批量评价 + 模拟回测（多进程并行）：成交额 TopN 股票 × 全部因子
#   每组合 = IC 评价（horizons 默认 1,2,3,5,10,15,20）+ 5桶×多空×全k 共 70 组回测
#   + 买入持有基线；对齐样本 < min_bars（默认 250）的组合跳过
$PY factors/main_batch.py                                # 101 因子 × top20
$PY factors/main_batch.py --top 2 --factors Alpha001,Alpha098   # 子集冒烟
$PY factors/main_batch.py --njobs 8 --horizons 1,5,20          # 覆盖并行度与 k 值
#   并行度/最小样本/出图子集默认取 src/config.json 的 [batch]（njobs=6 等），CLI 优先
#   输出 <项目根>/reports/batch_top<N>[_i]/（与 src 平级）：
#     <fid>_<sid>/                每股票×因子子目录：data_<fid>_<sid>.csv（date/OHLCV/
#                                 factor + 每组合 pos/eq 两列 + eq_buyhold）、
#                                 ic_decay.png/.csv（行=k 的 IC 表）、
#                                 bucket_mean_ret.png/.csv（行=(k,bucket) 的分桶收益表）、
#                                 backtest_s*_k*.png（仅出图子集
#                                 side∈{1,5}×多空×k∈{1,5,20}=12 张，config.json 可改）
#     IC_top<N>_detail.csv          明细表（行=因子×股票）：RankIC(k)/IC_std(k)/ICIR(k)/
#                                 n_bars/n_warnings/n_errors/error
#     backtest_metrics_top<N>.csv 回测总表（行=因子×股票×k×桶×方向 + buyhold 基线行）
#     summary_by_factor.csv / summary_by_stock.csv（IC 指标透视表）
#     run_params.json             运行参数快照（断点续算时校验一致性用）
#   断点续算：运行中追加写 summary_partial_detail/long.csv；中断后重跑自动检测最新
#   未完成目录并提示确认（--resume 直接续算不询问），已完成组合跳过；参数与
#   run_params.json 不一致则报错退出（不新开目录）。全部完成后 partial 文件自动删除

# 因子绘图：个股日 K 线（叠加 EMA5/20）+ 因子曲线，弹窗显示
$PY -m factors plot Alpha001 BJ920010                        # 全历史
$PY -m factors plot Alpha001 BJ920010 20250601 20251231      # 指定日期范围
$PY -m factors plot Alpha001 BJ920010 --save out.png         # 同时存盘

# 分桶信号模拟交易（详见下文"模拟交易"一节）
$PY -m backtest 920725 Alpha098                              # 全扫描：1..5桶 × 多/空，逐一出图 + summary.csv
$PY -m backtest BJ920725 Alpha098 --side 1 --k 10            # 单跑：桶底信号做多/持有上限10天
$PY -m backtest 920725 Alpha098 --side 5 --direction short   # 单跑：桶顶信号做空
$PY -m backtest 920725 Alpha098 --start 2023-01-01 --show    # 弹窗显示

# 滚动回测（详见下文"滚动回测"一节）
$PY -m backtest.rolling                                      # 全周期两阶段（top20 × 全因子）
$PY -m backtest.rolling --top 2 --factors Alpha001,Alpha098 \
    --start 2023-01-01 --end 2023-09-30                      # 冒烟（子集+区间）
$PY -m backtest.rolling --simulate-only                      # 复用已有重训产物只跑实盘

# IC 信号回测（rolling Rank IC + AR(1)/OU 概率信号，详见下文"IC 信号回测"一节）
$PY -m backtest.bactest_ic 920725 Alpha098 --k 5 --W 50
$PY -m backtest.bactest_ic BJ920725 Alpha098 --k 5 --W 50 --start 2023-01-01 --show

# 单元测试（factors 24 个 + backtest 28 个用例）
$PY -m pytest factors/tests backtest/tests -v
```

### Python API

```python
from factors import FactorLibrary

lib = FactorLibrary()        # 默认参数见下
lib.init()                   # 全量重建
lib.update()                 # 增量追加新日期
lib.list_factors()           # -> DataFrame(factor_id, name, field, table, note)
lib.list_stocks()            # -> ['BJ920010', ...]

# 查询：factor 可用 factor_id 或原始因子名
s = lib.get("Alpha001", "BJ920010")                     # 单股时序 (Series, date 索引)
df = lib.get("Alpha001", start="2025-01-01")            # 长表 (date/stockid/value)
df = lib.get("Alpha001", start="2025-01-01", end="2025-06-30")

# 非默认路径时：
FactorLibrary(
    lib_dir=r"d:\jzhang\data-kyts\factor_lib",          # 因子库目录
    factor_list_path=r"d:\jzhang\data-kyts\src\factor_list.xlsx",
    src_dir=r"d:\jzhang\data-kyts\factors\kyzqsh11",    # 原始 zip 根目录
    prefix="BJ",                                        # 股票代码前缀过滤
)
```

### 设计要点（改动代码前必读）

- **循环优先级**：构建按"日期外层 + 内存累积 + 末尾一次性写出"。每个 zip 只
  解压一次（共约 2430 个），避免"因子外层"导致的 24.6 万次重复解压，也避免
  "每日落盘"导致的 24.6 万次 parquet 重写。增量 `update()` 复用同一流程，
  合并阶段按 `(stockid, date)` 去重（新数据优先），因此重复执行是幂等的。
- **日期取自文件名** `YYYYMMDD`，不要用 csv 里的 `enddate` 列（Excel 序列号）。
- **扫描只认 `YYYYMMDD.zip`**；数据目录里混有的已解压同名子目录要忽略。
- 原始 zip 里 2016 年起就有数据，但 BJ 股票 2021-11-15（北交所开市）才出现，
  此前年份过滤后为空属正常现象。
- 因子标识规则：ASCII 因子名清洗后直接用；中文名走
  `registry._translate_to_english()` 钩子（当前数据全英文，不会触发）；
  重名自动加 `_2`、`_3` 后缀。
- 某 zip 缺某因子列或文件损坏：warning 后跳过，不中断构建。

## 因子评价（IC 分析，移植自 A 工程）

`eval_config.py` / `stats_utils.py` / `evaluator.py` / `report.py` 四个文件从
A 工程（`D:\jzhang\work\options\superTrends\factors\`）**原样复制**，自包含
（仅依赖 pandas/numpy/scipy/matplotlib），口径与 A 完全一致。
**本项目在此基础上的新增**（不影响原有 IC 口径）：`EvalConfig` 增加
`run_backtest`（默认 True）/`backtest_k`（默认 5）/`backtest_fee`（默认 5e-5）
三字段；`evaluator._evaluate_symbol` 末尾对 side=1..n_buckets 全部桶 × 多/空
逐组合扫描跑分桶模拟交易（`backtest` 包，key 形如 `side5_long`/`side1_short`），
结果存 `SymbolReport.backtests`；`FactorReport.summary()` 追加
`BT_*(<key>)` 列，`to_markdown()` 追加"模拟交易"小节，`plot()` 追加
`backtest_<key>_<标的>.png` 四子图，`backtest_metrics()` 导出指标长表 CSV。
**时间序列评价**
（非横截面）：对每只股票单独沿时间评价，再跨股票汇总。

### 口径说明（与报告头部的"口径说明"一致）

- **前向收益**（`return_convention="next_open"`，防未来函数）：t 日收盘生成信号，
  t+1 开盘价进场，r(t→t+k) = Close[t+1+k]/Open[t+1] − 1。horizons 默认
  (1,3,5,10,20,50) 个交易日（CLI 默认；`EvalConfig` 类默认为
  (1,2,3,5,10,15,20,30,50)，直接调 API 不传 config 时才用它）。
- **Rank IC 与 IC 衰减**：对每个 k，全样本一次性计算信号与前向收益的 Spearman
  秩相关（`evaluator._eval_ic_table`）；IC 衰减图即 rank_ic 对 k 的折线
  （`FactorReport.plot`）。
- **滚动 IC**：窗口 W=`EvalConfig.rolling_window`（默认 50）bar、步长 1 bar，
  **窗口内独立编秩**的 Rank IC 时间序列（`evaluator._rolling_spearman`），对每个
  k 分别计算；跨标的汇总表中 IC_std/ICIR/ICIR_annual/IC>0 列取主 horizon
  k=horizons[0]（CLI 默认 k=1）。
- **ICIR 计算**（`evaluator._eval_rolling_ic`）：ICIR = 滚动IC均值 / 滚动IC标准差；
  年化 ICIR = ICIR·√(252/W)（严格按此公式）；IC>0 占比为滚动 IC 取正值的比例。
- **预测回归**：r(t→t+k)/σ̂_t = α + β·s_t + ε，`vol_scaling=True`（σ̂_t 为 1-bar
  close/close 收益的 ewm(span=20) 标准差）。表列含义：beta 为一元 OLS 斜率
  （含常数项 α），衡量信号每单位取值对波动率调整后前向收益的影响；r2 为判定
  系数；beta_t/beta_p 为检验 β=0 的 NW 修正 t 值与对应 p 值，滞后阶数见
  nw_lag 列。
- **Newey-West 滞后规则**（`nw_lag=None` 表示自动）：预测回归（含顶底价差、
  多头超额 t 检验）L = max(k, ⌊4·(T/100)^(2/9)⌋)；滚动 IC 均值检验
  L = max(W, ⌊4·(T/100)^(2/9)⌋)；各检验实际使用的 L 见结果表。
- **NW t 值**：报告中所有 t 值（beta_t、滚动 IC 均值、顶底价差、多头超额的
  t 检验）均为 Newey-West HAC 修正后的 t 值——点估计（OLS 系数或序列均值）
  不变，仅用 Bartlett 核（权重 1−l/(L+1)）对前 L 阶自协方差加权累加重估
  标准误，以校正序列自相关与异方差；含 T/(T−p) 自由度修正，L=0 时退化为
  普通同方差标准误；p 值基于 t 分布（df=T−p）。statsmodels 缺失时自动回退到
  自实现（`stats_utils.py`）。
- **年化约定**：`bars_per_year=252`（main_eval 已固定）；分层年化收益
  ann_ret = mean(r_k)·(bars_per_year/k)，Sharpe = mean/std·√(bars_per_year/k)。
- **分桶规则**：按信号值的全样本分位数等频（pd.qcut）分成 `n_buckets=5` 桶，
  每桶约占总有效样本的 1/N；分桶依据是信号 s_t 的取值（非收益），桶序号
  1 = 信号最低，N = 信号最高；信号有效取值数 < N 时自动降桶并告警，常数信号
  跳过分层测试；分位点基于全样本，属样本内诊断口径。另报告单调性
  （桶序号与桶均收益的 Spearman 相关）。
- **顶底价差**：逐 t 构造多空序列——t 属于顶桶（信号最高桶）取 +r(t→t+k)，
  属于底桶取 −r(t→t+k)，其余桶不计入；报告该序列的均值、NW t 值、年化
  Sharpe（mean/std·√(bars_per_year/k)）与累计净值最大回撤。
- **多头超额口径**：顶桶前向收益 − 全样本同 k 前向收益均值（即相对买入持有
  的超额）。
- **NAV**（`top_bottom_nav_k1_<标的>.png` 三条曲线，只绘制 k=1、其它 k 不出图；
  数据来自 `evaluator._eval_buckets` 的 `BucketResult.nav`）：均为对应收益序列按时间复利累乘的累计净值
  NAV_t = ∏_{i≤t}(1+r_i)。top = 顶桶日子的前向收益序列；bottom = 底桶日子的
  前向收益序列（**未取负**，下跌即底桶日亏钱）；spread = 顶底多空序列（顶桶日 +r、
  底桶日 −r、其余日子剔除），即"做多顶桶信号日、做空底桶信号日"的组合收益
  （**不是 top、bottom 两条净值直接相减**）。**注意 r 是 k 日前向收益，相邻信号日的**
  **持有期互相重叠，NAV 属诊断口径，不是可实现的逐日调仓回测净值**。
- **最小有效样本数** `min_periods=252`，不足的 (标的,k) 在结果中标记
  `insufficient` 并告警。

Python API：

```python
from factors import evaluate_factor, EvalConfig, FactorLibrary
from factors.price_data import load_daily_bars, load_prices

lib = FactorLibrary()
s = lib.get("Alpha001", "BJ920010")          # 单股信号序列
prices = load_daily_bars("BJ920010")         # open/high/low/close/volume 日线
report = evaluate_factor(s, prices, EvalConfig(bars_per_year=252))
print(report.symbols["Alpha001"].ic_table)   # 各 horizon 的 RankIC + 回归表
report.plot("reports/")                      # 出图；report.to_markdown() 出报告
# 多标的：signal 传 date×stockid 宽表，prices 传 load_prices(stockids) 的 dict
```

`main_eval.py` 三种模式：不带 `--stock` 时逐股票评价全部 263 只并输出跨股票汇总
（summary CSV + markdown + 模拟交易指标长表 `backtest_metrics_<fid>.csv`，
**不出单股图**）；带 `--stock` 时输出该股票的完整
IC 表、滚动 IC、分层测试、全部图表（含每个 side×方向组合一张回测四子图）、
markdown 与 `backtest_metrics_<fid>_<stock>.csv`，以及滚动 IC 数值 CSV
`rolling_ic_<标的>.csv`（第一列时间、第二列原始因子值，其后每列一个 k×W 组合的
滚动 Rank IC，列名 `rank_ic_k{k}_W{w}`，W∈{10,20,30,50,100}，与滚动 IC 图同口径）；
factor 取 `all` 且带 `--stock`
时对该股票遍历库中全部因子（行情只加载一次，单因子失败不中断整批），每因子一个
子目录（全部图表 + markdown + backtest metrics CSV + 滚动 IC 数值 CSV），另输出跨因子汇总
`summary_all_<stock>.csv`
（行=factor_id，控制台按 |RankIC(k=主horizon)| 降序打印前 20）。输出在
`<项目根>/reports/<名称>/`（all 模式为 `all_<stock>/`，自动加 `_1` 等后缀避免覆盖）。

### Rolling IC 正负预测（sign_forecast.py）

`python -m factors sign <factor> <stock>`（等价脚本入口 `factors/main_sign.py`）：
复用评价对齐结果（`report.rolling_ic_frame` 由 plot_data 现场重算各 k×W 滚动
Rank IC，与滚动 IC 图、`rolling_ic_*.csv` 同一口径），对每条 (k,W) IC 序列 ×
h ∈ `--hs`（默认 0,5,10,20）评估 4 个符号预测模型的测试集命中率
（按时间前 80% 训练、末 20% 测试，`--test-frac` 可调）：
base=训练集多数类、persist=最近已知 IC 符号延续、
ou=AR(1)/OU 条件概率解析解（P(IC>0)=Φ(m/σ_gap)，随跨度自动衰减到均值方向）、
logit=L2 逻辑回归（IRLS，特征=[最近已知 IC, 再滞后1, 近5根均值]）。
**信息可得性口径**：IC_t 的窗口含 Close[t+1+k]，t 日收盘时最新可用 IC 下标为
t−k−1，故"预测未来第 h 天"的实际跨度 gap = h+k+1 个 bar。
输出 `<项目根>/reports/sign_<fid>_<stock>[_N]/`：
`sign_forecast_<fid>_<stock>.csv`（k/W/h × 4 模型命中率长表）+
`sign_heatmap_<fid>_<stock>.png`（行=k、列=模型的面板热力图，面板内横轴 h、
纵轴 W，色标以 0.50 随机水平为中心，绿=优于随机）。
可调 `--horizons/--windows/--hs/--test-frac/--start/--end`。

## 模拟交易（src/backtest/）

分桶信号模拟交易：按信号全样本分位数分桶（与评价的分桶规则一致），`side` 指定
触发桶号（**1..n_buckets 整数**，1=桶底、n_buckets=桶顶，缺省=顶桶），
`direction`="long"/"short" 做多/做空；t 日收盘触发 → **t+1 开盘调仓**。
**仓位 3 档**：tier 0/1/2/3 → 目标持仓市值占当前总资产 0%/20%/50%/100%；
每次触发升一档并调仓至该档目标占比，满仓（tier=3）再触发则跳过（n_skipped
计数），无信号则无动作。**整仓计时**：首次建仓（tier 0→1）起满 k 天，于到期日
**收盘整仓平仓**（加仓不重置计时，trades 中 forced=False）；数据末端未平仓持仓
按最后一日收盘价强制平仓（forced=True）。做空按 signed shares 记账
（equity = cash + shares·price，shares 空为负；gross_ret = entry/exit − 1），
不考虑融券保证金/利息。手续费单边 5e-5（万 0.5，双边收取）；初始资金 1.0
（净值口径）。trades 每个建仓-平仓周期一行（entry_px 为多次加仓的成交量加权
均价，n_adds 为加仓次数）。
绩效指标：总收益率/年化收益率（按 252 复利年化）/最大回撤/夏普率
（日收益 mean/std·√252）/卡玛比率（年化/|回撤|）。

- `engine.py`：`run_backtest(signal, prices, BacktestConfig)` 纯计算
  （仅依赖 pandas/numpy），返回 `BacktestResult`（trades/equity/position/metrics）；
  另有 `buy_and_hold_equity(prices)` / `buy_and_hold_metrics(prices)` 买入持有基线
  （equity = init·close/close_0，不收手续费，metrics 键与 run_backtest 一致，
  n_trades=1），供批量回测的基线对比使用。
- `plotting.py`：`plot_backtest(result, save_path=None, show=False)` ——
  K 线（复用 `utils.plot_candles_with_ema`，叠加 EMA5/20 与买^/卖v 标记）/
  因子值/持仓（仓位）/总资产四子图（总资产子图叠加绿色的买入并持有基准，
  按首日收盘归一到初始资金），横轴时间对齐（`utils._apply_time_xticks`），
  suptitle 含主要参数与指标；图内文字英文（与 report.py 出图惯例一致）。
- `__init__.py`：`run_factor_backtest(stock, factor, side, direction, k,
  n_buckets, ...)` 高层包装（自动补 BJ 前缀、取信号与行情）；
  `sweep_backtests(stocks, factors, sides, directions, ks, ...)` 参数网格遍历，
  返回每行一个组合的指标 DataFrame，单组合失败不中断（error 列记录原因）。
- CLI：`python -m backtest 920725 Alpha098 [--side 5 --direction long --k 5
  --buckets 5]`（`--fee/--start/--end/--show`）。**不指定 --side 与 --direction 时
  全扫描**：side=1..buckets × long/short 共 2×buckets 个组合逐一出图，输出到
  `<项目根>/reports/backtest_<factor>_<stock>_all_k<k>[_N]/`（每组合一张 png + 一份
  `<name>_trades.csv`，另有按年化降序的 summary.csv）；指定其一即单跑（--side
  缺省=顶桶 buckets、--direction 缺省=long），输出到
  `<项目根>/reports/backtest_<factor>_<stock>_side<side>_<direction>_k<k>[_N]/`。
- 评价集成：`EvalConfig.run_backtest=True`（默认）时 `_evaluate_symbol` 对
  side=1..n_buckets 全部桶 × 多/空逐组合各跑一次（k=cfg.backtest_k，
  费率=cfg.backtest_fee），结果 key 形如 `side5_long`/`side1_short`，
  指标进 summary CSV（`BT_*(<key>)` 等列）、markdown，并经
  `FactorReport.backtest_metrics()` 导出长表 CSV（`backtest_metrics_*.csv`，
  main_eval 三种模式都会写）；图随单股票模式的
  `report.plot()` 保存（文件名 `backtest_<key>_<标的>.png`；多标的 263 只模式不出图，
  与现有多标的"不出单股图"行为一致）。

## 滚动回测（src/backtest/rolling.py）

每 M 日重训候选股票池 + 逐日模拟交易（**只做多**）的滚动回测，CLI：
`python -m backtest.rolling`（`--top/--factors/--lookback/--rebalance/--ks/
--init-capital/--fee/--min-window-bars/--start/--end/--njobs/--outdir/
--retrain-only/--simulate-only`）。输出目录 `<项目根>/reports/rolling_backtest/`。

**两阶段**：

- **阶段A（重训，多进程）**：全局交易日历（所选股票日线日期并集）上调仓日
  T = 第 lookback 根 bar 起、每隔 rebalance 根一次（默认 N=63≈3 个月、M=5）。
  每个 T 用窗口 [T−62 .. T] 对 全部因子 × top20 股票（`BJ_stocks_top100.csv`
  前 20 名，`--top` 可改）逐组合：IC 分析（k=1,2,3,5,10,15，next_open 口径
  RankIC）、窗口回测（复用 `engine.run_backtest`，side∈{1,5} 多头 × 全 k，
  指标行仿 backtest_metrics_top20.csv）、买入持有基线；桶边界（信号值的
  q(1/N)/q((N−1)/N) 分位，即 side=1/side=5 的触发阈值）记 json；
  候选池筛选**直接调用** `bt_analysis.filter_high_performance_combos`
  （ann_ret>0.3、sharpe>1、calmar>2、maxdd>−0.2、ann_ret>同股票 buyhold 年化）。
  每调仓日产出 4 文件：`bucket_<YYYYMMDD>.json`、`backtest_metrics_<YYYYMMDD>.csv`、
  `ic_detail_<YYYYMMDD>.csv`、`high_performance_combos_<YYYYMMDD>.csv`。
  并行粒度 = 一个因子 × 全部股票（worker 内该因子 parquet 只读一次）；
  窗口对齐样本 < min_window_bars（默认 40）的组合跳过。
- **阶段B（实盘模拟，单进程）**：自首个调仓日次日起逐日循环——
  ① 开仓：前一交易日收盘产生的信号，当日有 bar 且开盘价有效者按组合均分
  **当前现金**成交（n=当日实际成交笔数，每笔金额 = cash/n，**含费口径**：
  现金支出恰好 cash/n，股数 = (cash/n)/(开盘价×(1+fee))，允许小数股）；
  现金耗尽（cash≤0）时当日信号全部跳过（n_skipped）；该股当日无 bar 则顺延，
  超过 max_entry_wait（默认 5）个全局交易日放弃（n_dropped）；
  ② 平仓：持仓满 k 日（**按个股自身 K 线位置计数**，停牌自然顺延）当日收盘
  整仓平仓，回款当日加入现金、次日可用；③ 信号检测：用**严格早于当日的
  最近一次调仓日**的池与边界，side=5 需因子值 ≥ q_high、side=1 需 ≤ q_low；
  组合已有持仓/待开仓时信号跳过（n_skipped）；④ 收盘记账：现金、占用资金
  （持仓市值）、总资产 = 现金 + 占用资金；数据末端未平仓按最后可得收盘价
  强平（forced=True），末日总资产=现金。
  资金口径：**初始资金固定 1000 万元**（`--init-capital`），初始现金=初始资金；
  候选池中股票若未加载行情（如 --simulate-only 与重训 --top 不一致）会自动
  补充加载并并入全局日历。

**汇总产物**：`trades.csv`（每笔 stock/factor/k/side/开平仓价/股数/entry_value
（当日分得金额）/收益/费用/forced）、`daily_equity.csv`（date/cash/occupied/
equity/n_positions）、`equity_curve.png`（占用资金+现金 / 总资产双子图，
叠加初始资金线）、`summary.json`（总收益率/年化/最大回撤/Sharpe/卡玛 +
n_trades/n_skipped/n_dropped/init_capital/cash_end/max_occupied，控制台同印）。
`--simulate-only` 复用已有调仓日产物直接跑实盘，便于调参迭代。

## IC 信号回测（src/backtest/bactest_ic.py）

rolling Rank IC + AR(1)/OU 条件概率信号的单股做多模拟回测，CLI：
`python -m backtest.bactest_ic <stock> <factor> --k 5 --W 50`
（`--p-threshold/--quantile/--min-obs/--fee/--init-capital/--start/--end/--show`）。
口径：rolling Rank IC（窗口 W，与评价/rolling_ic CSV 同口径，复用
`evaluator._rolling_spearman`）；IC_t 的窗口含 Close[t+1+k]，t 日收盘时最新
可用下标为 t−k−1——第 t 日的预测取 ≤ t−k−1 的最后一条非 NaN IC（下标 j），
用 j 及之前全部非 NaN IC（扩展窗）现场拟合 AR(1)/OU（复用
`sign_forecast._ou_fit/_ou_p_pos`），P(IC_t>0) = Φ(m/σ_gap)，gap = t−j ≥ k+1；
可用 IC 条数 < min_obs（默认 60）不预测。开仓信号：P > 0.7（--p-threshold）且
s_t ≥ 过去 W 根因子值的 0.7 分位（前 30%，--quantile）→ t+1 开盘价开仓。
仓位两档 30%/100%（目标持仓市值占当前总资产），满仓再触发跳过（n_skipped）；
首次建仓起满 k 日收盘整仓平仓（加仓不重置计时），数据末端强平（forced）；
费率默认 5e-5 双边，初始资金默认 100 万元，只做多。IC 预测评估：P>0.5 判正，
对照真实 IC_t 正负报告 sensitivity/precision（与开仓阈值 0.7 是两个口径）。
绩效指标复用 `engine._calc_metrics`。输出
`<项目根>/reports/bactest_ic_<fid>_<stock>_k<k>_W<W>[_N]/`：6 子图 png
（日 K 线/因子值/rolling IC/开仓信号/仓位/总资产）+ 数据 CSV
（首列 date，含 OHLCV/factor/q_thr/rolling_ic/prob_pos/signal/position/equity）
+ trades CSV；控制台打印绩效与 sensitivity/precision。

## 因子绘图（plotting.py）

`plot_factor(factor, stockid, start, end, save_path=None, show=True)`：
主图为个股日 K 线（`utils.plot_candles_with_ema`，叠加现场计算的 EMA5/EMA20），
子图为因子曲线；x 轴行号经 `utils._apply_time_xticks` 显示日期，
右键点击可在双子图添加对齐竖线。行情数据来自 `price_data.load_daily_bars`
（`day-data/前复权/<代码>-<名称>.csv`，GBK 编码；stockid 'BJ920010' →
文件 '920010-凯添燃气.csv'）。

注意：因子库覆盖 2021-11-15 起，日线行情可能更早（如新三板/精选层时期），
绘图与评价均按两者交集对齐。

## 代码风格约定

- 注释、docstring 用中文（与 `utils.py` 一致）；标识符用英文。
- 类型标注用 `from __future__ import annotations` + 现代写法（`str | Path`）。
- 依赖只使用已安装包（pandas / numpy / pyarrow / openpyxl / scipy / matplotlib / pytest），
  新增依赖前先确认 64 位 Python 3.11 环境可装。
