"""滚动 IC 数值 CSV 导出：列结构正确，W=rolling_window 列与 RollingICResult 数值一致。"""

import numpy as np
import pandas as pd

from factors import EvalConfig, evaluate_factor


def _make_data(T: int = 400):
    rng = np.random.default_rng(7)
    n = T + 60  # 末尾预留前向收益所需的 bar
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
    prices = pd.DataFrame({"open": open_, "close": close}, index=idx)
    signal = pd.Series(rng.normal(size=T), index=idx[:T], name="synth")
    return signal, prices


def test_save_rolling_ic_csv(tmp_path):
    signal, prices = _make_data()
    cfg = EvalConfig(horizons=(1, 5), rolling_window=50, min_periods=30,
                     run_backtest=False)
    report = evaluate_factor(signal, prices, cfg)
    paths = report.save_rolling_ic_csv(str(tmp_path))
    assert len(paths) == 1

    df = pd.read_csv(paths[0], index_col=0, parse_dates=True)
    assert df.index.name == "date"
    expected_cols = ["factor"] + [
        f"rank_ic_k{k}_W{w}" for k in (1, 5) for w in (10, 20, 30, 50, 100)
    ]
    assert list(df.columns) == expected_cols
    # 第二列为原始因子值（与输入信号一致）
    np.testing.assert_allclose(df["factor"].dropna().values, signal.values)
    # 前 W-1 个窗口期位置为 NaN
    assert df["rank_ic_k1_W10"].isna().sum() >= 9
    # W=rolling_window=50 的列与评价结果的滚动 IC 序列数值一致
    rep = report.symbols["synth"]
    pd.testing.assert_series_equal(
        df["rank_ic_k1_W50"].dropna(), rep.rolling[1].series,
        check_names=False, check_freq=False,
    )
    pd.testing.assert_series_equal(
        df["rank_ic_k5_W50"].dropna(), rep.rolling[5].series,
        check_names=False, check_freq=False,
    )
