"""分桶信号模拟交易（backtest 包）。

- engine: 纯计算核心（run_backtest / BacktestConfig / BacktestResult /
  buy_and_hold_metrics 买入持有基线），仅依赖 pandas/numpy；
- plotting: K 线 / 因子值 / 持仓 / 总资产四子图；
- run_factor_backtest / sweep_backtests: 从因子库取数的高层包装，
  便于对不同股票、因子、参数组合做遍历；
- rolling: 滚动回测（每 M 日重训候选股票池 + 逐日模拟，现金均分开仓，只做多），
  独立 CLI：`python -m backtest.rolling`（详见 rolling.py 头部 docstring）。

用法::

    from backtest import run_factor_backtest, sweep_backtests
    res = run_factor_backtest("920725", "Alpha098", side=5, direction="long", k=5)
    print(res.metrics)

    # 参数遍历（每行一个组合）
    df = sweep_backtests(stocks=["920725", "BJ920010"],
                         factors=["Alpha098", "Alpha001"],
                         sides=(5, 1), directions=("long", "short"), ks=(5, 10))

CLI（在 src 目录下运行）::

    python -m backtest 920725 Alpha098 --side 5 --direction long --k 5 --buckets 5
"""

from __future__ import annotations

import pandas as pd

from .engine import (
    BacktestConfig,
    BacktestResult,
    buy_and_hold_equity,
    buy_and_hold_metrics,
    run_backtest,
)
from .bt_analysis import (
    analyze_and_plot_metrics,
    analyze_and_plot_rankic_vs_return,
    filter_high_performance_combos,
    run_high_performance_backtests,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "buy_and_hold_equity",
    "buy_and_hold_metrics",
    "run_backtest",
    "run_factor_backtest",
    "sweep_backtests",
    "analyze_and_plot_metrics",
    "analyze_and_plot_rankic_vs_return",
    "filter_high_performance_combos",
    "run_high_performance_backtests",
]


def normalize_stockid(stock: str) -> str:
    """股票代码归一化：'920725' → 'BJ920725'；已带 BJ 前缀的原样返回。"""
    s = str(stock).strip().upper()
    return s if s.startswith("BJ") else f"BJ{s}"


def run_factor_backtest(
    stock: str,
    factor: str,
    side: int | None = None,
    direction: str = "long",
    k: int = 5,
    n_buckets: int = 5,
    fee_rate: float = 5e-5,
    start: str | None = None,
    end: str | None = None,
    lib=None,
) -> BacktestResult:
    """高层包装：因子库取信号 + 前复权日线取行情 → run_backtest。

    Args:
        stock: 股票代码（'920725' 或 'BJ920725'）。
        factor: 因子标识或因子名（如 'Alpha098'）。
        side: 触发桶号 1..n_buckets（1=桶底，n_buckets=桶顶）；None=顶桶。
        direction: "long"=做多；"short"=做空。
        k: 持仓天数上限（首次建仓起满 k 天，到期日收盘整仓平仓）。
        n_buckets: 分桶数。
        fee_rate: 单边费率（默认万 0.5，双边收取）。
        start, end: 可选日期范围（'YYYYMMDD' 或 'YYYY-MM-DD'）。
        lib: FactorLibrary 实例，缺省按默认路径构造。

    Returns:
        BacktestResult（label 置为 "<factor_id> <stockid>"）。
    """
    from factors.library import FactorLibrary  # 延迟导入，避免循环依赖
    from factors.price_data import load_daily_bars

    lib = lib or FactorLibrary()
    stockid = normalize_stockid(stock)
    fid = lib._resolve_factor_id(factor)  # 未知因子在此抛 KeyError
    s = lib.get(fid, stockid, start=start, end=end)
    prices = load_daily_bars(stockid)
    cfg = BacktestConfig(n_buckets=n_buckets, side=side, direction=direction,
                         k=k, fee_rate=fee_rate)
    res = run_backtest(s, prices, cfg)
    res.label = f"{fid} {stockid}"
    return res


def sweep_backtests(
    stocks: list[str],
    factors: list[str],
    sides: tuple[int | None, ...] = (None,),
    directions: tuple[str, ...] = ("long",),
    ks: tuple[int, ...] = (5,),
    n_buckets: int = 5,
    fee_rate: float = 5e-5,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """参数网格遍历：行 = (stock, factor, side, direction, k) 组合，列 = 参数 + 绩效指标。

    信号与行情按 (stock, factor) 只加载一次；单个组合失败不中断，
    该行指标为 NaN 且 error 列记录原因。
    """
    from factors.library import FactorLibrary  # 延迟导入，避免循环依赖
    from factors.price_data import load_daily_bars

    # None（顶桶）先解析为实际桶号，结果表中记录实际使用的 side
    sides = tuple(n_buckets if s is None else int(s) for s in sides)
    lib = FactorLibrary()
    rows: list[dict] = []
    for stock in stocks:
        stockid = normalize_stockid(stock)
        try:
            prices = load_daily_bars(stockid)
        except Exception as exc:
            for factor in factors:
                for side in sides:
                    for direction in directions:
                        for k in ks:
                            rows.append({"stock": stockid, "factor": factor,
                                         "side": side, "direction": direction,
                                         "k": k, "n_buckets": n_buckets,
                                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        for factor in factors:
            base = {"stock": stockid, "factor": factor, "n_buckets": n_buckets}
            try:
                fid = lib._resolve_factor_id(factor)
                s = lib.get(fid, stockid, start=start, end=end)
            except Exception as exc:
                for side in sides:
                    for direction in directions:
                        for k in ks:
                            rows.append({**base, "side": side,
                                         "direction": direction, "k": k,
                                         "error": f"{type(exc).__name__}: {exc}"})
                continue
            for side in sides:
                for direction in directions:
                    for k in ks:
                        row = {**base, "side": side, "direction": direction, "k": k}
                        try:
                            cfg = BacktestConfig(n_buckets=n_buckets, side=side,
                                                 direction=direction, k=k,
                                                 fee_rate=fee_rate)
                            row.update(run_backtest(s, prices, cfg).metrics)
                            row["error"] = ""
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        rows.append(row)
    return pd.DataFrame(rows)
