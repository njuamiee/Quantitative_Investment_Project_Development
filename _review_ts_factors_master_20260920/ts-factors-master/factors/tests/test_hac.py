"""自实现 Newey-West HAC 的单元测试。

- L=0 时退化为普通 OLS 标准误（常数回归情形下严格相等）；
- 残差正自相关时 HAC 标准误 > OLS 标准误；
- 与手算小样本例子对比。
"""

import numpy as np

from factors.stats_utils import mean_test_hac, ols_hac


def test_lag0_reduces_to_ols_se():
    # 常数回归（均值检验）：L=0 时 HAC 标准误应严格等于普通 OLS 标准误 s/√T
    rng = np.random.default_rng(42)
    y = rng.normal(0.5, 2.0, size=200)
    out = mean_test_hac(y, L=0, backend="self")
    T = len(y)
    ols_se = y.std(ddof=1) / np.sqrt(T)
    assert abs(out["se"] - ols_se) < 1e-12
    assert abs(out["mean"] - y.mean()) < 1e-12


def test_hac_se_exceeds_ols_se_under_positive_autocorrelation():
    # AR(1) 正自相关序列：HAC(L>0) 标准误应大于 OLS(L=0) 标准误
    rng = np.random.default_rng(7)
    T, phi = 500, 0.8
    e = rng.normal(size=T)
    y = np.empty(T)
    y[0] = e[0]
    for t in range(1, T):
        y[t] = phi * y[t - 1] + e[t]
    se0 = mean_test_hac(y, L=0, backend="self")["se"]
    se10 = mean_test_hac(y, L=10, backend="self")["se"]
    assert se10 > se0


def test_hand_computed_small_sample():
    # 手算小样本：y = [1, 2, 3]，常数回归，L=1
    #   均值=2，残差 e = [-1, 0, 1]
    #   S = Σe² + 2·w1·Σ e_t·e_{t-1}，w1 = 1 - 1/(L+1) = 0.5
    #     = (1+0+1) + 2·0.5·((-1)·0 + 0·1) = 2
    #   cov = (T/(T-1)) · S/T² = 3/2 · 2/9 = 1/3，se = √(1/3)
    y = np.array([1.0, 2.0, 3.0])
    out = mean_test_hac(y, L=1, backend="self")
    assert abs(out["mean"] - 2.0) < 1e-12
    assert abs(out["se"] - np.sqrt(1.0 / 3.0)) < 1e-10


def test_ols_hac_beta_recovers_slope():
    # 无线性关系的随机数据上验证回归基本正确性：构造 y = 1 + 2x + 小噪声
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.1, size=300)
    out = ols_hac(y, x, L=5, backend="self")
    assert abs(out["beta"] - 2.0) < 0.05
    assert abs(out["alpha"] - 1.0) < 0.05
    assert out["beta_p"] < 1e-10
    assert out["r2"] > 0.99
