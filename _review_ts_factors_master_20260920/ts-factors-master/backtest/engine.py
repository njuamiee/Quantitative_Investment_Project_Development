"""分桶信号模拟交易引擎（纯计算，仅依赖 pandas/numpy）。

口径：
- 信号按全样本分位数等频分桶（pd.qcut，与 factors.evaluator._eval_buckets 相同）；
- side 指定触发桶号（1=桶底 … n_buckets=桶顶），direction="long"/"short" 做多/做空；
- 仓位 3 档：tier 0/1/2/3 → 目标持仓市值占当前总资产 0%/20%/50%/100%；
- t 日收盘触发 → t+1 开盘升一档（按开盘价把 |持仓市值| 调到该档目标占比）；
  满仓（tier=3）再出信号则无动作（n_skipped 计数）；无信号则无动作；
- 整仓计时：首次建仓（tier 0→1）bar 起算，加仓不重置；满 k 天到期日当天收盘
  整仓平仓（trades 中 forced=False）；数据末端未平仓按最后一日收盘强平（forced=True）；
- 做空：shares 带符号（空为负），equity = cash + shares·price；做空 gross_ret =
  entry_px/exit_px − 1；简化口径，不考虑融券保证金与利息；
- 手续费双边收取（买入：成交金额 X，费用 X·fee；卖出/卖空：到账 X·(1−fee)）。

另提供 buy_and_hold_metrics：买入持有基线指标（首日收盘买入持有到末日，不收
手续费），返回与 run_backtest metrics 相同的键，便于与策略组合直接对比。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_TRADE_COLS = ["entry_date", "entry_px", "exit_date", "exit_px",
               "gross_ret", "net_ret", "n_adds", "forced"]

# 档位 → 目标持仓市值占总资产比例
_TIER_RATIO = {1: 0.20, 2: 0.50, 3: 1.00}


@dataclass
class BacktestConfig:
    """模拟交易配置。

    Attributes:
        n_buckets: 分桶数（按信号值全样本分位数等频划分）。
        side: 触发桶号 1..n_buckets（1=桶底，n_buckets=桶顶）；
            None 时取 n_buckets（顶桶）。
        direction: "long"=做多；"short"=做空。
        k: 持仓天数上限：首次建仓起满 k 天，到期日收盘整仓平仓（加仓不重置计时）。
        fee_rate: 单边费率（买卖双边各收一次），默认万分之 0.5。
        init_capital: 初始资金（默认 1.0，净值口径）。
        bars_per_year: 年化用每年 bar 数（日线 252）。
    """

    n_buckets: int = 5
    side: int | None = None
    direction: str = "long"
    k: int = 5
    fee_rate: float = 5e-5
    init_capital: float = 1.0
    bars_per_year: int = 252

    def __post_init__(self):
        if self.side is None:
            self.side = self.n_buckets
        if not (1 <= int(self.side) <= self.n_buckets):
            raise ValueError(
                f"side 须在 1..n_buckets={self.n_buckets} 之间，收到 {self.side!r}")
        self.side = int(self.side)
        if self.direction not in ("long", "short"):
            raise ValueError(
                f"direction 只能是 'long' 或 'short'，收到 {self.direction!r}")
        if self.n_buckets < 2:
            raise ValueError("n_buckets 至少为 2")
        if self.k < 1:
            raise ValueError("k 至少为 1")
        if self.fee_rate < 0:
            raise ValueError("fee_rate 不能为负")


@dataclass
class BacktestResult:
    """模拟交易结果。"""

    config: BacktestConfig
    data: pd.DataFrame  # 对齐后的 s/open/close（含 high/low 若输入提供）
    trades: pd.DataFrame  # 每个建仓-平仓周期一行，列见 _TRADE_COLS
    equity: pd.Series  # 逐日总资产（date 索引，起始 init_capital）
    position: pd.Series  # 逐日仓位 = |持仓市值| / 总资产（0~1，做空也为正值）
    metrics: dict  # total_ret/ann_ret/maxdd/sharpe/calmar/n_trades/n_skipped
    label: str = ""  # 图题用标识（如 "Alpha098 BJ920725"），由调用方填写


def _nan_metrics() -> dict:
    """无法交易时的空指标。"""
    return {"total_ret": np.nan, "ann_ret": np.nan, "maxdd": np.nan,
            "sharpe": np.nan, "calmar": np.nan, "n_trades": 0, "n_skipped": 0}


def _calc_metrics(equity: pd.Series, init: float, bars_per_year: int,
                  n_trades: int, n_skipped: int) -> dict:
    """由逐日总资产序列计算绩效指标。"""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        m = _nan_metrics()
        m["n_trades"], m["n_skipped"] = n_trades, n_skipped
        return m
    daily = equity.pct_change().dropna()
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_days = len(daily)
    # 权益为负（理论极端情形）时复利年化为复数，按 NaN 处理
    ann_ret = float((1.0 + total_ret) ** (bars_per_year / n_days) - 1.0) \
        if n_days > 0 and 1.0 + total_ret > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    maxdd = float(dd.min())
    std = float(daily.std())
    sharpe = float(daily.mean() / std * np.sqrt(bars_per_year)) if std > 0 else np.nan
    calmar = float(ann_ret / abs(maxdd)) if maxdd < 0 else np.nan
    return {"total_ret": total_ret, "ann_ret": ann_ret, "maxdd": maxdd,
            "sharpe": sharpe, "calmar": calmar,
            "n_trades": int(n_trades), "n_skipped": int(n_skipped)}


def _flat_result(data: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """无交易时的平直净值结果（指标 NaN）。"""
    equity = pd.Series(cfg.init_capital, index=data.index, name="equity")
    position = pd.Series(0.0, index=data.index, name="position")
    trades = pd.DataFrame(columns=_TRADE_COLS)
    return BacktestResult(config=cfg, data=data, trades=trades, equity=equity,
                          position=position, metrics=_nan_metrics())


def run_backtest(signal: pd.Series, prices: pd.DataFrame,
                 config: BacktestConfig | None = None) -> BacktestResult:
    """分桶信号模拟交易主入口。

    Args:
        signal: datetime 索引的因子信号序列（NaN 不参与分桶与触发）。
        prices: 含 open/close 列的日线 DataFrame（有 high/low 列时一并保留，供画 K 线）。
        config: BacktestConfig，缺省用默认参数。

    Returns:
        BacktestResult: 交易明细、逐日总资产/仓位与绩效指标。
    """
    cfg = config or BacktestConfig()
    if "open" not in prices.columns or "close" not in prices.columns:
        raise ValueError("prices 必须包含 open/close 列")
    px_cols = [c for c in ("open", "high", "low", "close") if c in prices.columns]
    data = pd.concat([signal.rename("s"), prices[px_cols]], axis=1,
                     join="inner").sort_index()
    if data.index.has_duplicates:
        raise ValueError("对齐后索引存在重复时间戳，请检查输入数据")
    n = len(data)
    if n < 2:
        warnings.warn("对齐后样本不足 2 条，无法模拟交易", stacklevel=2)
        return _flat_result(data, cfg)

    # ---- 分桶（与 factors.evaluator._eval_buckets 同口径） ----
    s_valid = data["s"].dropna()
    n_buckets = min(cfg.n_buckets, int(s_valid.nunique()))
    if n_buckets < cfg.n_buckets:
        warnings.warn(
            f"信号有效取值数 {s_valid.nunique()} < n_buckets={cfg.n_buckets}，"
            f"实际桶数降为 {n_buckets}", stacklevel=2)
    if n_buckets < 2:
        warnings.warn("常数信号（或取值不足），无法分桶，模拟交易跳过", stacklevel=2)
        return _flat_result(data, cfg)
    if cfg.side > n_buckets:
        warnings.warn(
            f"side={cfg.side} 超出实际桶数 {n_buckets}，该桶不存在，模拟交易跳过",
            stacklevel=2)
        return _flat_result(data, cfg)
    bucket = pd.qcut(s_valid, n_buckets, labels=False, duplicates="drop") + 1
    # qcut 的 duplicates="drop" 可能进一步减少实际桶数（大量重复取值时），
    # 名义桶数检查在上一步，这里须按实际桶数复查 side 是否可达
    n_buckets_actual = int(bucket.max())
    if n_buckets_actual < n_buckets:
        warnings.warn(
            f"分位数边界重复，实际桶数由 {n_buckets} 降为 {n_buckets_actual}",
            stacklevel=2)
        if cfg.side > n_buckets_actual:
            warnings.warn(
                f"side={cfg.side} 超出实际桶数 {n_buckets_actual}，该桶不存在，模拟交易跳过",
                stacklevel=2)
            return _flat_result(data, cfg)
    bucket_arr = bucket.reindex(data.index).to_numpy(dtype=float)
    # t 日收盘触发 → t+1 开盘调仓；最后一根 bar 无下一根，不触发
    trigger = np.zeros(n, dtype=bool)
    trigger[:-1] = bucket_arr[:-1] == cfg.side

    # ---- 逐 bar 模拟（3 档仓位 + 整仓 k 天到期） ----
    open_arr = data["open"].to_numpy(dtype=float)
    close_arr = data["close"].to_numpy(dtype=float)
    dates = data.index
    k = cfg.k
    fee = cfg.fee_rate
    is_long = cfg.direction == "long"
    sgn = 1.0 if is_long else -1.0

    cash = float(cfg.init_capital)
    shares = 0.0  # 带符号：做空为负
    tier = 0
    entry_bar = -1  # 首次建仓 bar（整仓计时起点）
    # 本周期统计：entry_date / 加权入场价 / 现金流基数 / 加仓次数
    cyc_entry_date = None
    cyc_qty = 0.0       # 累计 |成交股数|
    cyc_notional = 0.0  # 累计成交金额（不含费），做空即名义本金
    cyc_cashflow = 0.0  # long: 累计买入总花费（含费）；short: 累计卖空净所得
    cyc_adds = 0
    trades: list[dict] = []
    n_skipped = 0
    equity = np.empty(n)
    position = np.empty(n)

    def _close_all(i: int, forced: bool) -> None:
        """按 close[i] 整仓平仓并记录一个交易周期。"""
        nonlocal cash, shares, tier, entry_bar
        nonlocal cyc_entry_date, cyc_qty, cyc_notional, cyc_cashflow, cyc_adds
        px = close_arr[i]
        if is_long:
            proceeds = shares * px * (1.0 - fee)
            cash += proceeds
            net_ret = proceeds / cyc_cashflow - 1.0
            gross_ret = px / (cyc_notional / cyc_qty) - 1.0
        else:
            cost = -shares * px * (1.0 + fee)
            cash -= cost
            net_ret = (cyc_cashflow - cost) / cyc_notional
            gross_ret = (cyc_notional / cyc_qty) / px - 1.0
        trades.append({
            "entry_date": cyc_entry_date, "entry_px": float(cyc_notional / cyc_qty),
            "exit_date": dates[i], "exit_px": float(px),
            "gross_ret": float(gross_ret), "net_ret": float(net_ret),
            "n_adds": cyc_adds, "forced": forced,
        })
        shares = 0.0
        tier = 0
        entry_bar = -1
        cyc_entry_date = None
        cyc_qty = cyc_notional = cyc_cashflow = 0.0
        cyc_adds = 0

    for i in range(n):
        # 1) 上一 bar 收盘触发 → 本 bar 开盘升一档（到期平仓发生在本 bar 收盘，
        #    故此刻仍持满仓时新信号算 skipped）
        if i > 0 and trigger[i - 1]:
            eq_open = cash + shares * open_arr[i]
            if tier >= 3:
                n_skipped += 1
            elif eq_open <= 0.0:  # 极端情形（如满仓做空遇暴涨），防御性跳过
                n_skipped += 1
            else:
                tier += 1
                target = _TIER_RATIO[tier] * eq_open
                delta = target - sgn * shares * open_arr[i]  # 需补的名义市值
                if delta > 0.0:
                    if is_long:
                        cash -= delta * (1.0 + fee)
                        shares += delta / open_arr[i]
                        cyc_cashflow += delta * (1.0 + fee)
                    else:
                        cash += delta * (1.0 - fee)
                        shares -= delta / open_arr[i]
                        cyc_cashflow += delta * (1.0 - fee)
                    cyc_qty += delta / open_arr[i]
                    cyc_notional += delta
                    cyc_adds += 1
                # tier 0→1（首次建仓，必然成交）启动整仓计时
                if tier == 1:
                    entry_bar = i
                    cyc_entry_date = dates[i]
        # 2) 整仓到期：首次建仓起满 k 天，本 bar 收盘全平
        if tier > 0 and entry_bar >= 0 and i == entry_bar + k:
            _close_all(i, forced=False)
        # 3) 收盘记账
        mkt_val = shares * close_arr[i]  # 带符号
        equity[i] = cash + mkt_val
        position[i] = abs(mkt_val) / equity[i] if equity[i] > 0 else 0.0

    # ---- 数据结束：未平仓持仓按最后一日收盘价强制平仓 ----
    if tier > 0 and shares != 0.0:
        _close_all(n - 1, forced=True)
        # 强平费用反映到最后一日的权益与仓位
        equity[-1] = cash
        position[-1] = 0.0

    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS)
    equity_s = pd.Series(equity, index=dates, name="equity")
    position_s = pd.Series(position, index=dates, name="position")
    metrics = _calc_metrics(equity_s, cfg.init_capital, cfg.bars_per_year,
                            len(trades_df), n_skipped)
    return BacktestResult(config=cfg, data=data, trades=trades_df,
                          equity=equity_s, position=position_s, metrics=metrics)


def buy_and_hold_equity(prices: pd.DataFrame, init_capital: float = 1.0) -> pd.Series:
    """买入持有基线的逐日总资产：首日收盘买入并持有，equity = init·close/close_0。"""
    if "close" not in prices.columns:
        raise ValueError("prices 必须包含 close 列")
    close = prices["close"].dropna()
    if close.empty or close.iloc[0] <= 0:
        raise ValueError("close 序列为空或首日收盘非正，无法构造买入持有基线")
    return pd.Series(init_capital * close / close.iloc[0], name="equity")


def buy_and_hold_metrics(prices: pd.DataFrame, init_capital: float = 1.0,
                         bars_per_year: int = 252) -> dict:
    """买入持有基线指标（不收手续费），键与 run_backtest 的 metrics 一致。

    n_trades 固定为 1（首日建仓持有到末日），n_skipped 固定为 0。
    """
    equity = buy_and_hold_equity(prices, init_capital)
    return _calc_metrics(equity, init_capital, bars_per_year, 1, 0)
