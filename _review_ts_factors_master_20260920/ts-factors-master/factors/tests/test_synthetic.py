"""合成数据验证：已知强度的预测关系应给出显著为正的 RankIC。"""

import numpy as np
import pandas as pd

from factors import EvalConfig, evaluate_factor


def _make_prices_from_signal(s: np.ndarray, eps: np.ndarray, rho: float):
    """构造 open/close 使 r(t→t+1) = close[t+2]/open[t+1] − 1 = rho·s_t + eps_t。

    open 取任意正数（此处为 100 加微小扰动）；close[t+2] 由上式反解，
    close[0]/close[1] 任意赋值，不与 t、t+1 的约束冲突。
    """
    T = len(s)
    n = T + 2
    open_ = np.full(n, 100.0) + np.arange(n) * 1e-6
    close = np.empty(n)
    close[0] = close[1] = 100.0
    for t in range(T):
        close[t + 2] = open_[t + 1] * (1.0 + rho * s[t] + eps[t])
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    prices = pd.DataFrame({"open": open_, "close": close}, index=idx)
    signal = pd.Series(s, index=idx[:T], name="synth")
    return signal, prices


def test_rank_ic_significant_positive():
    T, rho = 5000, 0.3
    rng = np.random.default_rng(123)
    s = rng.normal(size=T)
    eps = rng.normal(size=T)
    signal, prices = _make_prices_from_signal(s, eps, rho)

    cfg = EvalConfig(horizons=(1,), bars_per_year=4032)
    report = evaluate_factor(signal, prices, cfg)
    rep = report.symbols["synth"]

    rank_ic = rep.ic_table.loc[1, "rank_ic"]
    ic_p = rep.ic_table.loc[1, "ic_pvalue"]
    # 大样本下 Spearman ≈ (6/π)·arcsin(ρ_pearson/2)；
    # 此处 ρ_pearson ≈ 0.3/√(1+0.09) ≈ 0.287 → Spearman ≈ 0.275，取宽松区间
    assert 0.15 < rank_ic < 0.40
    assert ic_p < 0.01
    # 预测回归 β 显著为正
    assert rep.ic_table.loc[1, "beta"] > 0
    assert rep.ic_table.loc[1, "beta_p"] < 0.01
