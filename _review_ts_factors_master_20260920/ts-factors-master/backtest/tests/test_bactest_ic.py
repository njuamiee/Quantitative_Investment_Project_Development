"""backtest.bactest_ic 的确定性合成数据测试。

主场景构造：open_t = 100·exp(0.001t + 0.0001t²)（加速上升），close_t = 1.001·open_t，
信号 s_t = t 严格递增。此时 r(t→t+k) = 1.001·exp(0.001k + 0.0001k(2(t+1)+k)) − 1
严格递增，任意窗口 rolling Rank IC 恒为 +1，OU 预测 P(IC>0)=1 恒超阈值；
s_t 严格递增故必位于其 rolling 窗口的前 30% 分位 → 每日触发。

k=5、W=10、min_obs=20、fee=0、init=1e6、N=200 时：
- IC 自 bar 9（W−1）起非 NaN；首个预测日 t 需 j=t−k−1 ≥ 9+19=28 → t=34；
- 周期一：bar 35 开盘建仓 30%、bar 36 加满 100%（n_adds=2）；bar 37~40 的信号
  满仓跳过 4 次；首次建仓 bar 35，满 k=5 天 → bar 40 收盘整仓平仓（forced=False）；
- 之后每 6 根 bar 一个周期（entry 35+6m，exit 40+6m）；最后一个周期 bar 197 建仓、
  bar 198 加满，到期日超出数据末端 → 末日 bar 199 收盘强平（forced=True）；
- 共 28 个交易周期，n_skipped = 4×27 + 1 = 109；
- 预测日 t=34..199 中 IC_t 非 NaN 者止于 bar 193（r 需 Close[t+1+k]）→ n_ic_pred=160，
  预测全正、实际全正 → sensitivity=precision=1.0。
"""

import numpy as np
import pandas as pd
import pytest

from backtest.bactest_ic import ICBacktestConfig, run_ic_backtest

N = 200


def _make_trending_data(n=N):
    """价格加速上升、信号严格递增 → rolling Rank IC 恒 +1。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    open_ = 100.0 * np.exp(0.001 * t + 0.0001 * t * t)
    close = 1.001 * open_
    prices = pd.DataFrame({
        "open": open_, "high": close, "low": open_, "close": close,
    }, index=idx)
    signal = pd.Series(t, index=idx, name="synth")
    return signal, prices


def _make_random_data(n=N, seed=42):
    """ seeded 随机游走价格 + 随机信号 → IC 序列有正有负。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    open_ = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    close = open_ * np.exp(rng.normal(0.0003, 0.008, n))
    prices = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close),
        "low": np.minimum(open_, close),
        "close": close,
    }, index=idx)
    signal = pd.Series(rng.normal(0.0, 1.0, n), index=idx, name="synth")
    return signal, prices


def _cfg(**kw):
    base = dict(k=5, W=10, min_obs=20, fee_rate=0.0, init_capital=1e6)
    base.update(kw)
    return ICBacktestConfig(**base)


def test_signals_and_trading_cycle():
    """IC 恒正 → 每日触发；两档加仓、满 k 天收盘整仓平仓、周期循环。"""
    signal, prices = _make_trending_data()
    res = run_ic_backtest(signal, prices, _cfg())
    m = res.metrics
    assert m["n_trades"] == 28
    assert m["n_skipped"] == 109
    assert m["total_ret"] > 0.0

    tr = res.trades
    dates = prices.index
    assert (tr["n_adds"] == 2).all()                # 30% → 100% 两次成交
    assert not tr["forced"].iloc[:-1].any()         # 仅最后一个周期强平
    assert bool(tr["forced"].iloc[-1]) is True

    first = tr.iloc[0]
    assert first["entry_date"] == dates[35]         # t=34 触发，次日开盘建仓
    assert first["exit_date"] == dates[40]          # 建仓 bar 35，k=5 → bar 40 收盘平
    assert first["exit_px"] == prices["close"].iloc[40]
    o35, o36 = prices["open"].iloc[35], prices["open"].iloc[36]
    entry_px = 1e6 / (3e5 / o35 + 7e5 / o36)        # 两次加仓的加权均价
    assert first["entry_px"] == pytest.approx(entry_px)
    assert first["gross_ret"] == pytest.approx(
        prices["close"].iloc[40] / entry_px - 1.0)
    assert first["net_ret"] == pytest.approx(first["gross_ret"])  # fee=0

    # 满仓后仓位达到 100%（fee=0 精确口径）
    assert res.position.max() == pytest.approx(1.0)

    # IC 预测：预测全正、真实 IC 全正 → sensitivity=precision=1
    assert m["n_ic_pred"] == 160                    # t=34..193
    assert m["ic_sensitivity"] == pytest.approx(1.0)
    assert m["ic_precision"] == pytest.approx(1.0)


def test_insufficient_history_flat():
    """可用 IC 条数 < min_obs → 全程无预测无交易，净值平直。"""
    signal, prices = _make_trending_data(n=60)
    res = run_ic_backtest(signal, prices, _cfg(min_obs=60))
    m = res.metrics
    assert m["n_trades"] == 0
    assert m["n_ic_pred"] == 0
    assert np.isnan(m["ic_sensitivity"]) and np.isnan(m["ic_precision"])
    assert res.data["prob_pos"].isna().all()
    assert (res.data["signal"] == 0).all()
    assert (res.equity == 1e6).all()
    assert (res.position == 0.0).all()


def test_prediction_no_lookahead():
    """改动 bar m 之后的价格，不影响 t ≤ m 的预测（信息可得性 gap ≥ k+1）。"""
    m_bar = 150
    signal, prices_a = _make_random_data()
    prices_b = prices_a.copy()
    close_b = prices_b["close"].to_numpy()
    open_b = prices_b["open"].to_numpy()
    close_b[m_bar + 1:] *= 1.5                      # 末日之后的走势完全改变
    prices_b["close"] = close_b
    prices_b["high"] = np.maximum(open_b, close_b)
    prices_b["low"] = np.minimum(open_b, close_b)

    res_a = run_ic_backtest(signal, prices_a, _cfg())
    res_b = run_ic_backtest(signal, prices_b, _cfg())
    pa = res_a.data["prob_pos"].to_numpy()
    pb = res_b.data["prob_pos"].to_numpy()
    assert np.isfinite(pa[m_bar])                   # 确保 m 处确有预测值可比对
    assert np.allclose(pa[:m_bar + 1], pb[:m_bar + 1], equal_nan=True)
    # 尾部 IC 变了，m 之后的预测理应随之改变
    assert not np.allclose(pa[m_bar + 1:], pb[m_bar + 1:], equal_nan=True)


def test_sens_prec_aggregation():
    """sensitivity/precision 与 data 列（prob_pos/rolling_ic）手工重算一致。"""
    signal, prices = _make_random_data()
    res = run_ic_backtest(signal, prices, _cfg())
    d = res.data
    mask = d["prob_pos"].notna() & d["rolling_ic"].notna()
    assert int(mask.sum()) == res.metrics["n_ic_pred"] > 0
    pred = d.loc[mask, "prob_pos"] > 0.5
    act = d.loc[mask, "rolling_ic"] > 0
    tp = int((pred & act).sum())
    fp = int((pred & ~act).sum())
    fn = int((~pred & act).sum())
    exp_sens = tp / (tp + fn) if tp + fn else np.nan
    exp_prec = tp / (tp + fp) if tp + fp else np.nan
    if np.isfinite(exp_sens):
        assert res.metrics["ic_sensitivity"] == pytest.approx(exp_sens)
    else:
        assert np.isnan(res.metrics["ic_sensitivity"])
    if np.isfinite(exp_prec):
        assert res.metrics["ic_precision"] == pytest.approx(exp_prec)
    else:
        assert np.isnan(res.metrics["ic_precision"])
