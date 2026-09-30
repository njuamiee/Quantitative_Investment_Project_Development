"""Newey-West 滞后阶数规则的数值用例测试。"""

from factors.eval_config import nw_lag_auto, nw_lag_regression, nw_lag_rolling


def test_nw_lag_auto():
    # L_auto = floor(4 * (T/100)^(2/9))
    assert nw_lag_auto(100) == 4        # 4 * 1^(2/9) = 4
    assert nw_lag_auto(1000) == 6       # 4 * 10^(2/9) ≈ 6.67
    assert nw_lag_auto(10000) == 11     # 4 * 100^(2/9) ≈ 11.13


def test_nw_lag_regression_rule():
    # 预测回归: L = max(k, L_auto)
    assert nw_lag_regression(10000, 1) == 11    # max(1, 11)
    assert nw_lag_regression(10000, 20) == 20   # max(20, 11)
    assert nw_lag_regression(10000, 11) == 11
    assert nw_lag_regression(1000, 10) == 10    # max(10, 6)


def test_nw_lag_rolling_rule():
    # 滚动 IC 均值检验: L = max(W, L_auto)
    assert nw_lag_rolling(10000, 30) == 30      # max(30, 11)
    assert nw_lag_rolling(10000, 5) == 11       # max(5, 11)


def test_nw_lag_override():
    # nw_lag 配置可覆盖自动规则
    assert nw_lag_regression(10000, 1, override=7) == 7
    assert nw_lag_rolling(10000, 30, override=3) == 3
