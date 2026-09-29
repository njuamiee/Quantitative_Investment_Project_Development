"""backtest.rolling 的确定性合成数据测试。

不依赖真实因子库与行情文件：_retrain_one / _simulate 均为显式数据注入的
纯计算函数，测试直接构造合成 Series/DataFrame 与调仓日产物。
"""

import json

import numpy as np
import pandas as pd
import pytest

from backtest.rolling import (
    RollingConfig,
    _load_rebal_artifacts,
    _rebalance_indices,
    _retrain_one,
    _simulate,
    _write_day_outputs,
)


# ----------------------------------------------------------------------
# _retrain_one：窗口 IC + 桶边界 + 回测指标行
# ----------------------------------------------------------------------
def _make_window(n: int = 63):
    """n 根日线：open=100+i 递增，close=open+1；信号 s=i 严格递增。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    open_ = 100.0 + np.arange(n, dtype=float)
    px = pd.DataFrame({"open": open_, "high": open_ + 1.5,
                       "low": open_ - 0.5, "close": open_ + 1.0}, index=idx)
    s = pd.Series(np.arange(n, dtype=float), index=idx, name="F1")
    return s, px


def test_retrain_one_outputs():
    s, px = _make_window()
    cfg = RollingConfig(ks=(1, 5), sides=(1, 5), min_window_bars=10, fee_rate=0.0)
    r = _retrain_one("F1", "BJA", s, px, cfg)

    # 桶边界：0..62 的 20%/80% 分位数
    assert r["boundary"]["q_low"] == pytest.approx(np.quantile(np.arange(63), 0.2))
    assert r["boundary"]["q_high"] == pytest.approx(np.quantile(np.arange(63), 0.8))
    assert r["boundary"]["n_unique"] == 63

    # IC：信号递增、前向收益 r=2/(101+t) 递减 → RankIC(1) = -1
    assert r["ic_row"]["n_bars"] == 63
    assert r["ic_row"]["RankIC(1)"] == pytest.approx(-1.0)
    assert r["ic_row"]["RankIC(5)"] == pytest.approx(-1.0)

    # 回测行：2 side × 2 k + 1 buyhold = 5 行，列与 batch 口径一致
    rows = r["metric_rows"]
    assert len(rows) == 5
    combos = sorted(row["combo"] for row in rows)
    assert combos == ["buyhold", "s1_long", "s1_long", "s5_long", "s5_long"]
    for row in rows:
        for col in ("duration_years", "flat_frac", "total_ret", "ann_ret",
                    "maxdd", "sharpe", "calmar", "n_trades", "n_skipped"):
            assert col in row
    bh = [row for row in rows if row["combo"] == "buyhold"][0]
    assert bh["direction"] == "buyhold" and bh["n_trades"] == 1


def test_retrain_one_min_bars_skip():
    s, px = _make_window(20)
    cfg = RollingConfig(ks=(1,), min_window_bars=40)
    r = _retrain_one("F1", "BJA", s, px, cfg)
    assert r["metric_rows"] == [] and r["boundary"] is None
    assert "min_window_bars" in r["ic_row"]["error"]


def test_rebalance_indices():
    cal = pd.date_range("2024-01-01", periods=100, freq="D")
    cfg = RollingConfig(lookback=10, rebalance=5)
    idx = _rebalance_indices(cal, cfg)
    assert idx[:3] == [9, 14, 19] and idx[-1] == 99

    cfg2 = RollingConfig(lookback=10, rebalance=5,
                         start="2024-01-20", end="2024-02-15")
    idx2 = _rebalance_indices(cal, cfg2)
    assert all(cal[i] >= pd.Timestamp("2024-01-20") for i in idx2)
    assert all(cal[i] <= pd.Timestamp("2024-02-15") for i in idx2)
    assert idx2[0] > 9  # 起点被 start 裁剪


# ----------------------------------------------------------------------
# _write_day_outputs / _load_rebal_artifacts：产物落盘与回读
# ----------------------------------------------------------------------
def test_day_outputs_roundtrip(tmp_path):
    cfg = RollingConfig(ks=(1, 5), sides=(1, 5))
    T = pd.Timestamp("2024-03-01")
    win_start = pd.Timestamp("2024-01-01")
    metric_rows = [
        # 通过过滤规则的组合（ann_ret>0.3, sharpe>1, calmar>2, maxdd>-0.2, > buyhold）
        {"factor_id": "F1", "stock": "BJA", "combo": "s5_long", "k": 5,
         "side": 5, "direction": "long", "duration_years": 0.17,
         "flat_frac": 0.5, "total_ret": 0.08, "ann_ret": 0.6, "maxdd": -0.05,
         "sharpe": 2.0, "calmar": 3.0, "n_trades": 3, "n_skipped": 0,
         "n_warnings": 0, "error": ""},
        # 不达标组合（sharpe 不足）
        {"factor_id": "F1", "stock": "BJA", "combo": "s1_long", "k": 1,
         "side": 1, "direction": "long", "duration_years": 0.17,
         "flat_frac": 0.5, "total_ret": 0.01, "ann_ret": 0.5, "maxdd": -0.05,
         "sharpe": 0.5, "calmar": 3.0, "n_trades": 3, "n_skipped": 0,
         "n_warnings": 0, "error": ""},
        {"factor_id": "F1", "stock": "BJA", "combo": "buyhold", "k": None,
         "side": None, "direction": "buyhold", "duration_years": 0.17,
         "flat_frac": 0.0, "total_ret": 0.02, "ann_ret": 0.1, "maxdd": -0.1,
         "sharpe": 1.0, "calmar": 1.0, "n_trades": 1, "n_skipped": 0,
         "n_warnings": 0, "error": ""},
    ]
    ic_rows = [{"factor_id": "F1", "stock": "BJA", "n_bars": 63,
                "RankIC(1)": 0.1, "RankIC(5)": 0.2, "error": ""}]
    boundaries = {"F1": {"BJA": {"q_low": 1.0, "q_high": 9.0, "n_unique": 50}}}

    pool_path = _write_day_outputs(tmp_path, T, win_start, cfg,
                                   ic_rows, metric_rows, boundaries)
    assert pool_path.name == "high_performance_combos_20240301.csv"
    assert (tmp_path / "bucket_20240301.json").is_file()
    assert (tmp_path / "backtest_metrics_20240301.csv").is_file()
    assert (tmp_path / "ic_detail_20240301.csv").is_file()

    with open(tmp_path / "bucket_20240301.json", encoding="utf-8") as f:
        bj = json.load(f)
    assert bj["boundaries"]["F1"]["BJA"]["q_high"] == 9.0

    rebal = _load_rebal_artifacts(tmp_path, cfg)
    assert len(rebal) == 1 and rebal[0]["date"] == T
    # 只有 s5_long k=5 一行通过过滤
    assert rebal[0]["combos"] == [("F1", "BJA", 5, 5)]
    assert rebal[0]["boundaries"]["F1"]["BJA"]["q_low"] == 1.0


# ----------------------------------------------------------------------
# _simulate：逐日模拟（现金均分开仓 / 按个股 K 线位置 k 日平仓 / 持仓跳过）
# ----------------------------------------------------------------------
def _make_sim_world():
    """两只股票 10 个全局交易日，init_capital=2400、fee=0、价格恒 100：

    - BJA：每天有 bar；因子 F1 在 d3/d4/d8 取 10（≥q_high=9），其余为 5。
      d3 收盘触发 → d4 开盘进场（当日仅 1 笔 → 金额=全部现金 2400，24 股），
      k=2 → d6 收盘离场（回款 2400）；d4 收盘再触发（持仓中）→ 计 1 次跳过；
      d8 收盘触发 → d9 开盘进场，数据末端强平。
    - BJB：缺 d5 的 bar（停牌）；因子 F2 在 d4 取 1（≤q_low=2）。d4 收盘触发 →
      d5 无 bar 等待，d6 有 bar 但当日现金为 0（BJA 未平仓）→ 跳过记 n_skipped。
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="D")  # d0..d9

    def _px(active):
        sub = dates[list(active)]
        return pd.DataFrame({"open": 100.0, "high": 101.0,
                             "low": 99.0, "close": 100.0}, index=sub)

    prices = {"BJA": _px(range(10)),
              "BJB": _px([0, 1, 2, 3, 4, 6, 7, 8, 9])}  # 缺 d5

    f1 = pd.Series(5.0, index=dates)
    f1[dates[3]] = f1[dates[4]] = f1[dates[8]] = 10.0
    f2 = pd.Series(5.0, index=dates)
    f2[dates[4]] = 1.0
    factor_map = {("F1", "BJA"): f1, ("F2", "BJB"): f2}

    rebal = [{"date": dates[2],
              "combos": [("F1", "BJA", 2, 5), ("F2", "BJB", 1, 1)],
              "boundaries": {
                  "F1": {"BJA": {"q_low": 1.0, "q_high": 9.0, "n_unique": 10}},
                  "F2": {"BJB": {"q_low": 2.0, "q_high": 8.0, "n_unique": 10}},
              }}]
    cfg = RollingConfig(ks=(1, 2), init_capital=2400.0, fee_rate=0.0)
    return dates, prices, factor_map, rebal, cfg


def test_simulate_entries_exits_and_skip():
    dates, prices, factor_map, rebal, cfg = _make_sim_world()
    calendar = pd.DatetimeIndex(dates)
    res = _simulate(cfg, calendar, rebal, prices,
                    lambda fid, sid: factor_map.get((fid, sid)))
    tr = res["trades"].set_index(["stock", "entry_date"])

    # 只有 BJA 的 2 个周期成交；BJB 因现金为 0 被跳过
    assert res["metrics"]["n_trades"] == 2
    assert set(tr.index.get_level_values("stock")) == {"BJA"}

    # BJA 周期一：d4 开盘进（当日 1 笔 → 分得全部现金 2400，24 股），
    # d6 收盘出（按自身 K 线位置 +2）
    t1 = tr.loc[("BJA", dates[4])]
    assert t1["exit_date"] == dates[6]
    assert bool(t1["forced"]) is False
    assert t1["entry_value"] == pytest.approx(2400.0)
    assert t1["entry_px"] == 100.0 and t1["shares"] == pytest.approx(24.0)
    assert t1["exit_px"] == 100.0 and t1["net_ret"] == pytest.approx(0.0)
    # BJA 周期二：d9 开盘进（d6 平仓回款后现金恢复 2400），数据末端强平
    t2 = tr.loc[("BJA", dates[9])]
    assert bool(t2["forced"]) is True
    assert t2["entry_value"] == pytest.approx(2400.0)

    # 跳过 2 次：d4 收盘 BJA 持仓中再触发；d6 BJB 可成交但现金为 0
    assert res["metrics"]["n_skipped"] == 2
    assert res["metrics"]["n_dropped"] == 0

    daily = res["daily"]
    # 价格不变且无费 → 总资产恒为初始资金；d4/d5 持仓中 cash=0、占用 2400
    assert daily["equity"].eq(2400.0).all()
    assert daily.loc[dates[4], "cash"] == pytest.approx(0.0)
    assert daily.loc[dates[4], "occupied"] == pytest.approx(2400.0)
    # d6 收盘平仓回款 → cash 恢复 2400
    assert daily.loc[dates[6], "cash"] == pytest.approx(2400.0)
    assert daily.loc[dates[6], "occupied"] == pytest.approx(0.0)
    assert res["metrics"]["total_ret"] == pytest.approx(0.0)
    assert res["metrics"]["init_capital"] == pytest.approx(2400.0)
    assert res["metrics"]["max_occupied"] == pytest.approx(2400.0)
    assert res["metrics"]["cash_end"] == pytest.approx(2400.0)


def test_simulate_fee_and_pnl():
    """价格变动 + 手续费：含费开仓、平仓回款加回现金、equity=cash+持仓市值。"""
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    n = len(dates)
    open_ = 100.0 + np.arange(n, dtype=float)  # 100..107
    close = open_ + 1.0                        # 101..108
    prices = {"BJA": pd.DataFrame({"open": open_, "high": close, "low": open_ - 1,
                                   "close": close}, index=dates)}
    f1 = pd.Series(5.0, index=dates)
    f1[dates[3]] = 10.0  # d3 收盘触发 → d4 开盘进，k=2 → d6 收盘出
    rebal = [{"date": dates[2], "combos": [("F1", "BJA", 2, 5)],
              "boundaries": {"F1": {"BJA": {"q_low": 1.0, "q_high": 9.0,
                                            "n_unique": 10}}}}]
    fee = 0.01
    cfg = RollingConfig(ks=(2,), init_capital=1000.0, fee_rate=fee)
    res = _simulate(cfg, pd.DatetimeIndex(dates), rebal, prices,
                    lambda fid, sid: f1)
    tr = res["trades"]
    assert len(tr) == 1
    t0 = tr.iloc[0]
    entry_px, exit_px = open_[4], close[6]          # 104, 107
    amount = 1000.0                                  # 当日 1 笔 → 分得全部现金
    shares = amount / (entry_px * (1 + fee))         # 含费口径
    proceeds = shares * exit_px * (1 - fee)
    assert t0["entry_value"] == pytest.approx(amount)
    assert t0["entry_px"] == pytest.approx(entry_px)
    assert t0["exit_px"] == pytest.approx(exit_px)
    assert t0["shares"] == pytest.approx(shares)
    assert t0["net_ret"] == pytest.approx(proceeds / amount - 1.0)
    # 费用 = 买入费（amount − 净买额）+ 卖出费
    exp_fee = (amount - shares * entry_px) + shares * exit_px * fee
    assert t0["fee_paid"] == pytest.approx(exp_fee)

    daily = res["daily"]
    # d4 收盘（持仓第一日）：cash≈0，equity = 持仓市值
    assert daily.loc[dates[4], "cash"] == pytest.approx(0.0, abs=1e-6)
    assert daily.loc[dates[4], "occupied"] == pytest.approx(shares * close[4])
    assert daily.loc[dates[4], "equity"] == pytest.approx(shares * close[4])
    # d6 收盘平仓回款：cash = proceeds，占用清零
    assert daily.loc[dates[6], "cash"] == pytest.approx(proceeds)
    assert daily.loc[dates[6], "occupied"] == pytest.approx(0.0)
    assert daily["equity"].iloc[-1] == pytest.approx(proceeds)
    assert res["metrics"]["cash_end"] == pytest.approx(proceeds)
    assert res["metrics"]["total_ret"] == pytest.approx(proceeds / 1000.0 - 1.0)


def test_simulate_cash_split_two_entries():
    """同日 2 个组合触发 → 各得现金一半（n=当日成交笔数）。"""
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    px = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0}, index=dates)
    prices = {"BJA": px, "BJB": px.copy()}
    f = pd.Series(5.0, index=dates)
    f[dates[3]] = 10.0  # d3 收盘两只同时触发 → d4 开盘各进一笔
    rebal = [{"date": dates[2],
              "combos": [("F1", "BJA", 2, 5), ("F1", "BJB", 2, 5)],
              "boundaries": {"F1": {"BJA": {"q_low": 1.0, "q_high": 9.0,
                                            "n_unique": 10},
                                    "BJB": {"q_low": 1.0, "q_high": 9.0,
                                            "n_unique": 10}}}}]
    cfg = RollingConfig(ks=(2,), init_capital=1000.0, fee_rate=0.0)
    res = _simulate(cfg, pd.DatetimeIndex(dates), rebal, prices,
                    lambda fid, sid: f)
    tr = res["trades"]
    assert len(tr) == 2
    for _, row in tr.iterrows():
        assert row["entry_value"] == pytest.approx(500.0)   # 均分现金
        assert row["shares"] == pytest.approx(5.0)
        assert row["entry_date"] == dates[4]
        assert row["exit_date"] == dates[6]                  # k=2
        assert bool(row["forced"]) is False
    daily = res["daily"]
    # d4 开仓后现金≈0、占用 1000；d6 平仓后现金回到 1000
    assert daily.loc[dates[4], "cash"] == pytest.approx(0.0, abs=1e-6)
    assert daily.loc[dates[4], "occupied"] == pytest.approx(1000.0)
    assert daily.loc[dates[6], "cash"] == pytest.approx(1000.0)
    assert daily["equity"].eq(1000.0).all()


def test_simulate_no_cash_skip():
    """现金被持仓占满时，新信号跳过记 n_skipped，不开仓。"""
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    px = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0}, index=dates)
    prices = {"BJA": px, "BJB": px.copy()}
    fa = pd.Series(5.0, index=dates)
    fa[dates[3]] = 10.0   # BJA d3 触发 → d4 全仓（现金 500 用尽），k=3 → d7 才平仓
    fb = pd.Series(5.0, index=dates)
    fb[dates[4]] = 10.0   # BJB d4 触发 → d5 开盘时现金为 0 → 跳过
    fmap = {("F1", "BJA"): fa, ("F2", "BJB"): fb}
    rebal = [{"date": dates[2],
              "combos": [("F1", "BJA", 3, 5), ("F2", "BJB", 3, 5)],
              "boundaries": {"F1": {"BJA": {"q_low": 1.0, "q_high": 9.0,
                                            "n_unique": 10}},
                             "F2": {"BJB": {"q_low": 1.0, "q_high": 9.0,
                                            "n_unique": 10}}}}]
    cfg = RollingConfig(ks=(3,), init_capital=500.0, fee_rate=0.0)
    res = _simulate(cfg, pd.DatetimeIndex(dates), rebal, prices,
                    lambda fid, sid: fmap.get((fid, sid)))
    tr = res["trades"]
    assert len(tr) == 1 and tr.iloc[0]["stock"] == "BJA"
    assert tr.iloc[0]["entry_value"] == pytest.approx(500.0)
    assert res["metrics"]["n_skipped"] == 1   # BJB 现金不足跳过
    assert res["metrics"]["n_dropped"] == 0


def test_simulate_no_pool_no_trade():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = {"BJA": pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                                   "close": 100.0}, index=dates)}
    rebal = [{"date": dates[2], "combos": [], "boundaries": {}}]
    cfg = RollingConfig(init_capital=12345.0, fee_rate=0.0)
    res = _simulate(cfg, pd.DatetimeIndex(dates), rebal, prices,
                    lambda fid, sid: None)
    assert res["trades"].empty
    assert res["metrics"]["n_trades"] == 0
    assert res["metrics"]["total_ret"] == pytest.approx(0.0)
    # 全程无交易：现金与总资产恒为初始资金
    assert res["daily"]["cash"].eq(12345.0).all()
    assert res["daily"]["equity"].eq(12345.0).all()
    assert res["metrics"]["cash_end"] == pytest.approx(12345.0)
