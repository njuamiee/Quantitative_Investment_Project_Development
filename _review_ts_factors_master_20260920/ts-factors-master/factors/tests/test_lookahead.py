"""未来函数自检：信号等于次日 1-bar 收益时，正确口径下 IC 应≈0。

构造：close 随机游走（增量 u_t iid），open[t] = close[t]·(1+g_t)（g_t iid，与 u 独立）。
信号 s_t = close[t+1]/close[t] − 1 = u_{t+1}（即"明天"的收益）。

- 错误口径"t 收盘进场"（r = close[t+1]/close[t]−1）下 IC≈1 —— 说明信号确实"知道未来"；
- 正确口径"t+1 open 进场"（r = close[t+2]/open[t+1]−1 = (1+u_{t+2})/(1+g_{t+1})−1）
  与 s_t = u_{t+1} 独立，IC 应≈0。
"""

import numpy as np
import pandas as pd
from scipy import stats as _stats

from factors import EvalConfig, evaluate_factor


def test_no_lookahead_bias():
    T = 5000
    rng = np.random.default_rng(999)
    idx = pd.date_range("2024-01-01", periods=T, freq="15min")
    u = rng.normal(scale=0.01, size=T)
    g = rng.normal(scale=0.005, size=T)
    close = 100.0 * np.cumprod(1.0 + u)
    open_ = close * (1.0 + g)
    prices = pd.DataFrame({"open": open_, "close": close}, index=idx)

    # 信号 = 次日 1-bar 收益（站在 t 时刻不可知，用于检验实现是否偷看未来）
    signal = (prices["close"].shift(-1) / prices["close"] - 1.0).rename("peek")

    #  sanity：错误口径（t 收盘进场）下 IC≈1，证明该信号确实包含未来信息
    wrong_ret = prices["close"].shift(-1) / prices["close"] - 1.0
    pair = pd.concat([signal, wrong_ret], axis=1).dropna()
    wrong_ic = _stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
    assert wrong_ic > 0.99

    # 正确实现：t+1 open 进场，IC 应≈0
    cfg = EvalConfig(horizons=(1,), bars_per_year=4032)
    report = evaluate_factor(signal, prices, cfg)
    rank_ic = report.symbols["peek"].ic_table.loc[1, "rank_ic"]
    assert abs(rank_ic) < 0.05
