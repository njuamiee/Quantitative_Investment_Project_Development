"""backtest.engine 的确定性合成数据测试。

主场景：60 根日线，open=100+i 严格递增，close=open+1；信号 s=i 严格递增，
5 桶等频分桶后桶序号 = i//12+1（每桶 12 天）。

side=5、long、k=5、fee=0 时（顶桶 = 第 48~59 天，触发日 48..58）：
- 周期一：bar 49/50/51 开盘依次升档 20%→50%→100%（加仓 3 次）；bar 52~54 信号满仓
  跳过 3 次；首次建仓 bar 49，满 k=5 天 → bar 54 收盘整仓平仓（n_adds=3，forced=False）；
- 周期二：bar 54 收盘信号 → bar 55 重建 tier1，bar 56/57 升 2/3 档，bar 58/59 满仓
  跳过 2 次；到期 bar 60 超出数据末端 → 末日 bar 59 收盘强平（forced=True）；
- 共 2 个交易周期，n_skipped=5。
"""

import numpy as np
import pandas as pd
import pytest

from backtest import (
    BacktestConfig,
    buy_and_hold_equity,
    buy_and_hold_metrics,
    run_backtest,
)

N = 60


def _make_data():
    idx = pd.date_range("2024-01-01", periods=N, freq="D")
    open_ = 100.0 + np.arange(N, dtype=float)
    close = open_ + 1.0
    prices = pd.DataFrame({
        "open": open_, "high": close, "low": open_ - 1.0, "close": close,
    }, index=idx)
    signal = pd.Series(np.arange(N, dtype=float), index=idx, name="synth")
    return signal, prices


def _make_flat_price_data():
    """价格恒为 100（open=close），用于精确核对 3 档目标占比。"""
    idx = pd.date_range("2024-01-01", periods=N, freq="D")
    px = np.full(N, 100.0)
    prices = pd.DataFrame({"open": px, "high": px, "low": px, "close": px},
                          index=idx)
    signal = pd.Series(np.arange(N, dtype=float), index=idx, name="synth")
    return signal, prices


def test_entry_tiers_and_expiry_timing():
    """t 日触发 → t+1 开盘升档；首次建仓满 k 天于到期日收盘整仓平仓。"""
    signal, prices = _make_data()
    res = run_backtest(signal, prices,
                       BacktestConfig(n_buckets=5, side=5, direction="long",
                                      k=5, fee_rate=0.0))
    tr = res.trades
    assert res.metrics["n_trades"] == 2
    assert res.metrics["n_skipped"] == 5

    dates = prices.index
    first = tr.iloc[0]
    assert first["entry_date"] == dates[49]         # 第 48 天触发，次日开盘建仓
    assert first["exit_date"] == dates[54]          # 建仓 bar 49，k=5 → bar 54 收盘平
    assert first["exit_px"] == prices["close"].iloc[54]
    assert first["n_adds"] == 3                     # 20% → 50% → 100% 三次加仓
    assert bool(first["forced"]) is False

    # 加权平均入场价：按各次加仓的成交金额/股数加权
    o = prices["open"].to_numpy()
    q1 = 0.2 / o[49]                                # tier1：目标 20% × 权益 1.0
    eq50 = 0.8 + q1 * o[50]
    d2 = 0.5 * eq50 - q1 * o[50]                    # tier2：补到 50%
    q2 = d2 / o[50]
    eq51 = (1.0 - 0.2 - d2) + (q1 + q2) * o[51]
    d3 = eq51 - (q1 + q2) * o[51]                   # tier3：补到 100%（现金归零）
    q3 = d3 / o[51]
    entry_px = (0.2 + d2 + d3) / (q1 + q2 + q3)
    assert first["entry_px"] == pytest.approx(entry_px)
    assert o[49] < first["entry_px"] < o[51]
    assert first["gross_ret"] == pytest.approx(
        prices["close"].iloc[54] / entry_px - 1.0)
    assert first["net_ret"] == pytest.approx(first["gross_ret"])  # fee=0

    # 周期二：到期日超出数据末端 → 末日强平
    second = tr.iloc[1]
    assert second["entry_date"] == dates[55]        # bar 54 收盘信号 → 次日重建
    assert second["exit_date"] == dates[-1]
    assert second["exit_px"] == prices["close"].iloc[-1]
    assert second["n_adds"] == 3
    assert bool(second["forced"]) is True
    # 强平后权益=现金、仓位归零
    assert res.position.iloc[-1] == 0.0


def test_position_ratios_exact_on_flat_price():
    """恒价数据 + fee=0：3 档目标占比精确为 20%/50%/100%（收盘口径）。"""
    signal, prices = _make_flat_price_data()
    res = run_backtest(signal, prices,
                       BacktestConfig(n_buckets=5, side=5, direction="long",
                                      k=100, fee_rate=0.0))  # k 大 → 不到期
    pos = res.position
    assert pos.iloc[48] == 0.0                      # 尚未建仓
    assert pos.iloc[49] == pytest.approx(0.20)      # tier1
    assert pos.iloc[50] == pytest.approx(0.50)      # tier2
    assert pos.iloc[51] == pytest.approx(1.00)      # tier3
    assert pos.iloc[53] == pytest.approx(1.00)      # 无动作保持满仓
    assert res.equity.iloc[53] == pytest.approx(1.0)  # 价格不动，权益不变
    # 触发 11 次（bar 48..58）：3 次升档 + 8 次满仓跳过
    assert res.metrics["n_skipped"] == 8
    # 数据末端强平：一个周期、毛收益 0、仓位归零
    assert res.metrics["n_trades"] == 1
    assert bool(res.trades.iloc[0]["forced"]) is True
    assert res.trades.iloc[0]["gross_ret"] == pytest.approx(0.0)
    assert pos.iloc[-1] == 0.0


def test_fee_deducted_both_sides():
    """双边费用：net_ret = 卖出净得 / 累计买入总花费（含费）− 1。"""
    signal, prices = _make_data()
    fee = 0.01
    res = run_backtest(signal, prices,
                       BacktestConfig(n_buckets=5, side=5, direction="long",
                                      k=5, fee_rate=fee))
    o = prices["open"].to_numpy()
    cash = 1.0
    d1 = 0.2
    cash -= d1 * (1 + fee)
    q1 = d1 / o[49]
    eq50 = cash + q1 * o[50]
    d2 = 0.5 * eq50 - q1 * o[50]
    cash -= d2 * (1 + fee)
    q2 = d2 / o[50]
    eq51 = cash + (q1 + q2) * o[51]
    d3 = eq51 - (q1 + q2) * o[51]
    cash -= d3 * (1 + fee)
    q3 = d3 / o[51]
    cost = (d1 + d2 + d3) * (1 + fee)
    proceeds = (q1 + q2 + q3) * prices["close"].iloc[54] * (1 - fee)

    first = res.trades.iloc[0]
    assert first["net_ret"] == pytest.approx(proceeds / cost - 1.0)
    assert first["net_ret"] < first["gross_ret"]
    assert res.equity.iloc[54] == pytest.approx(cash + proceeds)


def test_short_direction_profits_when_price_falls():
    """做空：gross_ret = entry/exit − 1；上涨行情中做空亏钱；仓位为 |占比|。"""
    signal, prices = _make_data()
    res = run_backtest(signal, prices,
                       BacktestConfig(n_buckets=5, side=5, direction="short",
                                      k=5, fee_rate=0.0))
    tr = res.trades
    assert res.metrics["n_trades"] == 2
    assert res.metrics["n_skipped"] == 5
    first = tr.iloc[0]
    assert first["entry_date"] == prices.index[49]
    assert first["exit_date"] == prices.index[54]
    assert first["gross_ret"] == pytest.approx(
        first["entry_px"] / prices["close"].iloc[54] - 1.0)
    assert first["gross_ret"] < 0.0                 # 上涨中做空亏损
    assert res.metrics["total_ret"] < 0.0
    # 空仓仓位按 |市值|/总资产 计，介于 0~1 余
    assert res.position.max() <= 1.05
    assert res.position.iloc[50] > 0.0
    # 做空记账自洽：权益 = 现金 + shares·价（shares 为负）
    o = prices["open"].to_numpy()
    cash = 1.0 + 0.2                                # 卖空 20% 所得进现金
    q1 = 0.2 / o[49]
    eq49_close = cash - q1 * prices["close"].iloc[49]
    assert res.equity.iloc[49] == pytest.approx(eq49_close)
    assert res.position.iloc[49] == pytest.approx(
        q1 * prices["close"].iloc[49] / eq49_close)


def test_side1_uses_bottom_bucket():
    """side=1 = 桶底信号（i<=11 触发），首个建仓在 bar 1。"""
    signal, prices = _make_data()
    res = run_backtest(signal, prices,
                       BacktestConfig(n_buckets=5, side=1, direction="long",
                                      k=5, fee_rate=0.0))
    tr = res.trades
    assert res.metrics["n_trades"] == 2
    first = tr.iloc[0]
    assert first["entry_date"] == prices.index[1]   # bar 0 收盘触发 → bar 1 建仓
    assert first["exit_date"] == prices.index[6]    # 1 + k=5 → bar 6 收盘平
    assert first["n_adds"] == 3
    second = tr.iloc[1]
    assert second["entry_date"] == prices.index[7]  # bar 6 收盘信号 → bar 7 重建
    assert second["exit_date"] == prices.index[12]
    assert bool(second["forced"]) is False          # 到期平仓，非强平


def test_constant_signal_no_trades():
    """常数信号无法分桶：告警、无交易、净值平直，不抛异常。"""
    signal, prices = _make_data()
    with pytest.warns(UserWarning):
        res = run_backtest(signal * 0.0, prices, BacktestConfig())
    assert res.trades.empty
    assert res.metrics["n_trades"] == 0
    assert np.isnan(res.metrics["total_ret"])
    assert (res.equity == 1.0).all()
    assert (res.position == 0.0).all()


def test_side_beyond_reduced_buckets_no_trades():
    """降桶后 side 超出实际桶数：告警、无交易、净值平直。"""
    signal, prices = _make_data()
    sig = pd.Series(np.tile([0.0, 1.0, 2.0], 20), index=signal.index)
    with pytest.warns(UserWarning):
        res = run_backtest(sig, prices,
                           BacktestConfig(n_buckets=5, side=5, direction="long"))
    assert res.trades.empty
    assert (res.equity == 1.0).all()


def test_config_validation():
    """side 缺省=顶桶；side 越界 / direction 非法 → ValueError。"""
    assert BacktestConfig(n_buckets=5).side == 5    # None → 顶桶
    assert BacktestConfig(n_buckets=10).side == 10
    with pytest.raises(ValueError):
        BacktestConfig(n_buckets=5, side=0)
    with pytest.raises(ValueError):
        BacktestConfig(n_buckets=5, side=6)
    with pytest.raises(ValueError):
        BacktestConfig(n_buckets=5, side=3, direction="neutral")


def test_buy_and_hold_metrics_up_and_down():
    """买入持有基线：equity = close/close_0，n_trades=1，不收手续费。"""
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    up = pd.DataFrame({"close": 100.0 + np.arange(60, dtype=float)}, index=idx)
    m = buy_and_hold_metrics(up, bars_per_year=252)
    assert m["n_trades"] == 1
    assert m["n_skipped"] == 0
    assert m["total_ret"] == pytest.approx(159.0 / 100.0 - 1.0)
    assert m["maxdd"] == pytest.approx(0.0)         # 单调上涨无回撤
    assert m["ann_ret"] > m["total_ret"] > 0.0      # 60 天收益复利年化放大
    assert m["sharpe"] > 0.0

    eq = buy_and_hold_equity(up)
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] == pytest.approx(1.59)

    down = pd.DataFrame({"close": 100.0 - np.arange(60, dtype=float)}, index=idx)
    m2 = buy_and_hold_metrics(down)
    assert m2["total_ret"] == pytest.approx(41.0 / 100.0 - 1.0)
    # 单调下跌：最大回撤终点即末日
    assert m2["maxdd"] == pytest.approx(41.0 / 100.0 - 1.0)


def test_buy_and_hold_metrics_invalid_input():
    """缺 close 列或空序列 → ValueError。"""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.raises(ValueError):
        buy_and_hold_metrics(pd.DataFrame({"open": [1.0, 2.0, 3.0]}, index=idx))
    with pytest.raises(ValueError):
        buy_and_hold_equity(pd.DataFrame({"close": [np.nan] * 3}, index=idx))
