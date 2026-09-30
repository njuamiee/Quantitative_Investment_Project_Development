"""纯随机信号（与收益独立）：各 k 的 IC 不应显著。"""

import numpy as np
import pandas as pd

from factors import EvalConfig, evaluate_factor


def test_random_signal_ic_not_significant():
    T = 4000
    rng = np.random.default_rng(20240601)
    # 价格：独立增量的随机游走；信号：与价格完全独立的随机序列
    idx = pd.date_range("2024-01-01", periods=T, freq="15min")
    ret = rng.normal(scale=0.001, size=T)
    close = 100.0 * np.cumprod(1.0 + ret)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    prices = pd.DataFrame({"open": open_, "close": close}, index=idx)
    signal = pd.Series(rng.normal(size=T), index=idx, name="noise")

    cfg = EvalConfig(horizons=(1, 2, 5, 10, 20), bars_per_year=4032)
    report = evaluate_factor(signal, prices, cfg)
    rep = report.symbols["noise"]

    for k in cfg.horizons:
        rank_ic = rep.ic_table.loc[k, "rank_ic"]
        ic_p = rep.ic_table.loc[k, "ic_pvalue"]
        # 宽松断言：|IC| 小且 p 不显著（se ≈ 1/√T ≈ 0.016）
        assert abs(rank_ic) < 0.05, f"k={k}: RankIC={rank_ic}"
        assert ic_p > 0.01, f"k={k}: p={ic_p}"
