"""符号预测模块：网格评估结构、OU 概率取值范围、面板图落盘。"""

import os

import numpy as np
import pandas as pd

from factors import EvalConfig, evaluate_factor
from factors.report import rolling_ic_frame
from factors.sign_forecast import (MODELS, _ou_fit, _ou_p_pos, eval_sign_column,
                                   evaluate_sign_grid, plot_sign_heatmap)


def _make_data(T: int = 400):
    rng = np.random.default_rng(7)
    n = T + 60  # 末尾预留前向收益所需的 bar
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
    prices = pd.DataFrame({"open": open_, "close": close}, index=idx)
    signal = pd.Series(rng.normal(size=T), index=idx[:T], name="synth")
    return signal, prices


def test_eval_sign_column_bounds():
    """AR(1) 序列上的符号命中率在 [0,1]，且高持续序列 h=0 应显著优于随机。"""
    rng = np.random.default_rng(1)
    e = rng.normal(size=600)
    x = np.empty(600)
    x[0] = e[0]
    for t in range(1, 600):
        x[t] = 0.98 * x[t - 1] + e[t]  # 高持续 AR(1)
    r = eval_sign_column(x, k=1, h=0)
    assert set(MODELS) <= set(r)
    for m in MODELS:
        assert 0.0 <= r[m] <= 1.0
    assert r["ou"] > 0.8  # b=0.98、gap=2 时符号高度可预测
    assert 0.9 < r["b"] < 1.0


def test_ou_p_pos_range():
    """OU 条件概率在 (0,1) 内，且 gap 越大越趋向长期均值方向的先验。"""
    rng = np.random.default_rng(2)
    e = rng.normal(size=800)
    x = np.empty(800)
    x[0] = e[0]
    for t in range(1, 800):
        x[t] = 0.9 * x[t - 1] + e[t]  # 有持续性的 AR(1)，衰减才可检验
    mu, b, sig = _ou_fit(x)
    p1 = _ou_p_pos(mu, b, sig, np.array([mu + 3 * sig]), gap=1)
    p99 = _ou_p_pos(mu, b, sig, np.array([mu + 3 * sig]), gap=99)
    assert 0.5 < p1[0] <= 1.0
    assert abs(p99[0] - 0.5) < abs(p1[0] - 0.5)  # 长期衰减到基准


def test_grid_and_heatmap(tmp_path):
    """evaluate_factor → rolling_ic_frame → evaluate_sign_grid → 面板图。"""
    signal, prices = _make_data()
    cfg = EvalConfig(horizons=(1, 5), rolling_window=50, min_periods=30,
                     run_backtest=False)
    report = evaluate_factor(signal, prices, cfg)
    rep = report.symbols["synth"]
    frame = rolling_ic_frame(rep, cfg.horizons, windows=(10, 50))
    assert list(frame.columns) == [
        "factor", "rank_ic_k1_W10", "rank_ic_k1_W50",
        "rank_ic_k5_W10", "rank_ic_k5_W50",
    ]

    res = evaluate_sign_grid(frame, hs=(0, 5))
    assert len(res) == 2 * 2 * 2  # 2 个 k × 2 个 W × 2 个 h
    assert set(res.columns) >= {"col", "k", "W", "h", *MODELS}
    assert res[list(MODELS)].apply(lambda s: s.between(0, 1)).all().all()

    out = plot_sign_heatmap(res, str(tmp_path / "hm.png"), title="test")
    assert os.path.exists(out) and os.path.getsize(out) > 0
