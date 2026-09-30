"""因子评价配置：EvalConfig 数据类与 Newey-West 滞后阶数规则。

滞后阶数统一规则（nw_lag 配置可覆盖，None 表示自动）：
  - 自动值: L_auto = floor(4 * (T/100)^(2/9))
  - 预测回归（含顶底价差、多头超额 t 检验）: L = max(k, L_auto)，k 为对应 horizon
  - 滚动 IC 均值检验: L = max(W, L_auto)，W 为滚动窗口长度
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """时间序列因子评价配置。

    Attributes:
        horizons: 前向收益 horizon 列表（单位：bar）。
        n_buckets: 分层测试的桶数（按信号值全样本分位数划分）。
        rolling_window: 滚动 IC 窗口 W（单位：bar），步长固定为 1 bar。
        nw_lag: Newey-West 滞后阶数；None 表示按统一规则自动确定。
        vol_scaling: 预测回归中是否用事前波动率 σ̂_t 缩放前向收益。
        vol_ewm_span: σ̂_t 的 EWMA 周期（1-bar close/close 收益的 ewm(span) 标准差）。
        return_convention: 前向收益口径，目前仅实现 "next_open"
            （t bar 收盘生成信号，t+1 开盘价进场，r(t→t+k) = Close[t+1+k]/Open[t+1] − 1）。
        min_periods: 每个 (标的, k) 的最小有效样本数，不足时告警并在结果中标记。
        bars_per_year: 每年 bar 数，用于年化（如日线 252，15 分钟线 16×252=4032）。
        run_backtest: 【本项目新增】评价时是否同时跑分桶信号模拟交易
            （side=1..n_buckets 全部桶 × 多/空逐组合扫描，见 src/backtest/engine.py 口径）。
        backtest_k: 【本项目新增】模拟交易持仓天数。
        backtest_fee: 【本项目新增】模拟交易单边费率（双边收取）。
    """

    horizons: tuple = (1, 2, 3, 5, 10, 15, 20, 30, 50)
    n_buckets: int = 5
    rolling_window: int = 50 #10 weeks
    nw_lag: int | None = None
    vol_scaling: bool = True
    vol_ewm_span: int = 20
    return_convention: str = "next_open"
    min_periods: int = 252
    bars_per_year: int = 252
    run_backtest: bool = True
    backtest_k: int = 5
    backtest_fee: float = 5e-5

    def __post_init__(self):
        self.horizons = tuple(int(k) for k in self.horizons)
        if self.return_convention != "next_open":
            raise ValueError(
                f"return_convention 目前仅支持 'next_open'，收到 {self.return_convention!r}"
            )
        if self.n_buckets < 2:
            raise ValueError("n_buckets 至少为 2")
        if self.rolling_window < 5:
            raise ValueError("rolling_window 过小，窗口内秩相关无意义")


def nw_lag_auto(T: int) -> int:
    """自动 Newey-West 滞后阶数: floor(4 * (T/100)^(2/9))。"""
    if T <= 0:
        raise ValueError(f"样本量 T 必须为正，收到 {T}")
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def nw_lag_regression(T: int, k: int, override: int | None = None) -> int:
    """预测回归的 NW 滞后阶数: L = max(k, L_auto)，override 可覆盖。"""
    if override is not None:
        return int(override)
    return max(int(k), nw_lag_auto(T))


def nw_lag_rolling(T: int, W: int, override: int | None = None) -> int:
    """滚动 IC 均值检验的 NW 滞后阶数: L = max(W, L_auto)，override 可覆盖。"""
    if override is not None:
        return int(override)
    return max(int(W), nw_lag_auto(T))
