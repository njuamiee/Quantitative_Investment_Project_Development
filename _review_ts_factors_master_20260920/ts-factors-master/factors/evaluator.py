"""时间序列因子评价核心：evaluate_factor(signal, prices, config)。

口径（严格遵守，防未来函数）：
  信号 s_t 于第 t 根 K 线收盘后计算完毕；t+1 开盘价进场；
  前向收益 r(t→t+k) = Close[t+1+k] / Open[t+1] − 1。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as _stats

from .eval_config import EvalConfig, nw_lag_regression, nw_lag_rolling
from .report import BucketResult, FactorReport, RollingICResult, SymbolReport
from .stats_utils import mean_test_hac, ols_hac, spearman_ic

# 模拟交易（src/backtest 包，纯计算模块，无循环依赖）
from backtest.engine import BacktestConfig, run_backtest


def forward_returns(open_: pd.Series, close: pd.Series, k: int) -> pd.Series:
    """前向收益 r(t→t+k) = Close[t+1+k] / Open[t+1] − 1，索引对齐到 t。"""
    return close.shift(-(k + 1)) / open_.shift(-1) - 1.0


# ----------------------------------------------------------------------
# 输入规范化
# ----------------------------------------------------------------------
def _normalize_signal(signal) -> pd.DataFrame:
    """signal 兼容 Series（单标的）与 DataFrame（列=标的）。"""
    if isinstance(signal, pd.Series):
        name = signal.name if signal.name is not None else "factor"
        return signal.to_frame(name)
    if isinstance(signal, pd.DataFrame):
        if signal.shape[1] == 0:
            raise ValueError("signal DataFrame 没有列")
        return signal
    raise TypeError(f"signal 类型不支持: {type(signal)!r}，应为 Series 或 DataFrame")


def _normalize_prices(prices, symbols: list) -> dict:
    """prices 兼容单 DataFrame（含 open/close 列，仅用于单标的）与 dict[symbol -> DataFrame]。"""
    if isinstance(prices, pd.DataFrame):
        if len(symbols) != 1:
            raise ValueError(
                "prices 为单个 DataFrame 时仅支持单标的 signal；"
                "多标的请传入 dict[symbol -> DataFrame(open/close)]"
            )
        prices = {symbols[0]: prices}
    if not isinstance(prices, dict):
        raise TypeError(f"prices 类型不支持: {type(prices)!r}")
    out = {}
    for sym in symbols:
        if sym not in prices:
            raise ValueError(f"prices 缺少标的 {sym!r} 的数据")
        df = prices[sym]
        missing = {"open", "close"} - set(df.columns)
        if missing:
            raise ValueError(f"标的 {sym!r} 的 prices 缺少列: {sorted(missing)}")
        out[sym] = df
    return out


# ----------------------------------------------------------------------
# 各评价组件
# ----------------------------------------------------------------------
def _eval_ic_table(s: pd.Series, open_: pd.Series, close: pd.Series,
                   sigma: pd.Series | None, cfg: EvalConfig,
                   sym: str, warn_list: list) -> pd.DataFrame:
    """每个 horizon 的 Rank IC 与预测回归（r/σ̂ = α + β·s + ε，NW HAC）。"""
    rows = {}
    for k in cfg.horizons:
        fwd = forward_returns(open_, close, k)
        pair = pd.concat([s.rename("s"), fwd.rename("r")], axis=1).dropna()
        T = len(pair)
        insufficient = T < cfg.min_periods
        if insufficient:
            msg = f"{sym} k={k}: 有效样本 {T} < min_periods={cfg.min_periods}，结果仅供参考"
            warnings.warn(msg, stacklevel=2)
            warn_list.append(msg)
        rank_ic, ic_p = spearman_ic(pair["s"].values, pair["r"].values)

        beta = beta_t = beta_p = r2 = np.nan
        L = np.nan
        if T >= 5 and np.unique(pair["s"].values).size >= 2:
            y = pair["r"]
            xs = pair["s"]
            if sigma is not None:
                sig = sigma.reindex(pair.index)
                ok = sig.notna() & (sig > 0)
                y = y[ok] / sig[ok]
                xs = xs[ok]
            if len(y) >= 5:
                L = nw_lag_regression(len(y), k, cfg.nw_lag)
                reg = ols_hac(y.values, xs.values, L)
                beta, beta_t, beta_p, r2 = reg["beta"], reg["beta_t"], reg["beta_p"], reg["r2"]
        rows[k] = {
            "rank_ic": rank_ic,
            "ic_pvalue": ic_p,
            "beta": beta,
            "beta_t": beta_t,
            "beta_p": beta_p,
            "r2": r2,
            "n_obs": T,
            "nw_lag": L,
            "insufficient": insufficient,
        }
    table = pd.DataFrame(rows).T
    for c in ("rank_ic", "ic_pvalue", "beta", "beta_t", "beta_p", "r2", "nw_lag"):
        table[c] = pd.to_numeric(table[c], errors="coerce")
    table["n_obs"] = pd.to_numeric(table["n_obs"], errors="coerce").astype("Int64")
    table["insufficient"] = table["insufficient"].astype(bool)
    table.index.name = "k"
    return table


def _rolling_spearman(a: np.ndarray, b: np.ndarray, W: int) -> np.ndarray:
    """滚动窗口 Spearman 秩相关（窗口内独立编秩），长度与输入一致，前 W-1 为 NaN。"""
    n = len(a)
    out = np.full(n, np.nan)
    for i in range(W - 1, n):
        wa = a[i - W + 1: i + 1]
        wb = b[i - W + 1: i + 1]
        if np.unique(wa).size < 2 or np.unique(wb).size < 2:
            continue
        ra = _stats.rankdata(wa)
        rb = _stats.rankdata(wb)
        da = ra - ra.mean()
        db = rb - rb.mean()
        denom = np.sqrt((da @ da) * (db @ db))
        if denom > 0:
            out[i] = float(da @ db / denom)
    return out


def _eval_rolling_ic(s: pd.Series, open_: pd.Series, close: pd.Series,
                     cfg: EvalConfig) -> dict:
    """滚动 IC（窗口 W、步长 1，对每个 horizon 分别计算）及 ICIR 统计。"""
    W = int(cfg.rolling_window)
    results = {}
    for k in cfg.horizons:
        fwd = forward_returns(open_, close, k)
        pair = pd.concat([s.rename("s"), fwd.rename("r")], axis=1).dropna()
        ic_vals = _rolling_spearman(pair["s"].values, pair["r"].values, W)
        ic_series = pd.Series(ic_vals, index=pair.index, name="rolling_ic").dropna()

        if len(ic_series) >= 5:
            mean = float(ic_series.mean())
            std = float(ic_series.std())
            icir = mean / std if std > 0 else np.nan
            icir_annual = icir * np.sqrt(252.0 / W)  # 严格按此公式
            frac_pos = float((ic_series > 0).mean())
            L = nw_lag_rolling(len(ic_series), W, cfg.nw_lag)
            test = mean_test_hac(ic_series.values, L)
            t_nw, p_nw = test["t"], test["p"]
        else:
            mean = std = icir = icir_annual = frac_pos = t_nw = p_nw = np.nan
            L = nw_lag_rolling(max(len(ic_series), 1), W, cfg.nw_lag)
        results[k] = RollingICResult(
            horizon=k, window=W, series=ic_series, mean=mean, std=std, icir=icir,
            icir_annual=icir_annual, frac_pos=frac_pos, t_nw=t_nw, p_nw=p_nw, nw_lag=L,
        )
    return results


def _max_drawdown(ret: pd.Series) -> float:
    """累计净值的最大回撤（负数）。"""
    nav = (1.0 + ret).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _eval_buckets(s: pd.Series, open_: pd.Series, close: pd.Series,
                  cfg: EvalConfig, sym: str, warn_list: list) -> dict:
    """分层测试：按信号全样本分位数把时间 t 分桶，逐 horizon 统计。"""
    s_valid = s.dropna()
    n_buckets = min(cfg.n_buckets, int(s_valid.nunique()))
    if n_buckets < cfg.n_buckets:
        msg = (f"{sym}: 信号有效取值数 {s_valid.nunique()} < n_buckets={cfg.n_buckets}，"
               f"实际桶数降为 {n_buckets}")
        warnings.warn(msg, stacklevel=2)
        warn_list.append(msg)
    if n_buckets < 2:
        msg = f"{sym}: 常数信号（或取值不足），分层测试跳过"
        warnings.warn(msg, stacklevel=2)
        warn_list.append(msg)
        return {}
    bucket = pd.qcut(s_valid, n_buckets, labels=False, duplicates="drop") + 1  # 桶序号 1..N

    results = {}
    for k in cfg.horizons:
        fwd = forward_returns(open_, close, k)
        df = pd.concat([bucket.rename("b"), fwd.rename("r")], axis=1).dropna()
        if len(df) == 0:
            continue
        df["b"] = df["b"].astype(int)

        grp = df.groupby("b")["r"]
        table = pd.DataFrame({
            "count": grp.size(),
            "mean": grp.mean(),
            "std": grp.std(),
        })
        factor = cfg.bars_per_year / k
        table["ann_ret"] = table["mean"] * factor
        table["sharpe"] = np.where(
            table["std"] > 0, table["mean"] / table["std"] * np.sqrt(factor), np.nan
        )
        table = table[["count", "mean", "ann_ret", "sharpe"]]
        table.index.name = "bucket"

        # 单调性：桶序号 1..N 与桶均收益的 Spearman 相关
        mono_rho, mono_p = spearman_ic(
            table.index.values.astype(float), table["mean"].values
        )

        # 顶底价差：顶桶收益序列 − 底桶收益序列（按 t 对齐成对，逐 t 构造多空序列）
        top_b, bot_b = int(bucket.max()), int(bucket.min())
        r_top = df.loc[df["b"] == top_b, "r"]
        r_bot = df.loc[df["b"] == bot_b, "r"]
        spread = pd.concat([r_top, -r_bot]).sort_index()
        if len(spread) >= 5:
            L = nw_lag_regression(len(spread), k, cfg.nw_lag)
            tb_test = mean_test_hac(spread.values, L)
            tb_std = float(spread.std())
            tb = {
                "mean": float(spread.mean()),
                "t": tb_test["t"],
                "p": tb_test["p"],
                "ann_sharpe": (float(spread.mean()) / tb_std * np.sqrt(factor)
                               if tb_std > 0 else np.nan),
                "maxdd": _max_drawdown(spread),
                "n": len(spread),
                "nw_lag": L,
            }
        else:
            tb = {"mean": np.nan, "t": np.nan, "p": np.nan, "ann_sharpe": np.nan,
                  "maxdd": np.nan, "n": len(spread), "nw_lag": np.nan}

        # 多头桶超额：顶桶前向收益 − 全样本同 k 前向收益均值
        excess = r_top - float(df["r"].mean())
        if len(excess) >= 5:
            L = nw_lag_regression(len(excess), k, cfg.nw_lag)
            le_test = mean_test_hac(excess.values, L)
            le = {"mean": float(excess.mean()), "t": le_test["t"], "p": le_test["p"],
                  "n": len(excess), "nw_lag": L}
        else:
            le = {"mean": np.nan, "t": np.nan, "p": np.nan, "n": len(excess),
                  "nw_lag": np.nan}

        # NAV 不是三份“绝对价格”曲线，而是对应收益序列的累计复利净值：
        # - top: 最高信号桶前向收益 r_top 的累计净值；
        # - bottom: 最低信号桶前向收益 r_bot 的累计净值；
        # - spread: 逐 t 构造的多空价差收益序列（顶桶取 +r_t、底桶取 -r_t）后做复利，
        #   代表“多头顶桶 / 空头底桶”的组合收益；它不是 top 与 bottom 两条净值直接相减。
        nav = pd.DataFrame({
            "top": (1.0 + r_top).cumprod(),
            "bottom": (1.0 + r_bot).cumprod(),
            "spread": (1.0 + spread).cumprod(),
        })
        results[k] = BucketResult(
            k=k, table=table, monotonic_rho=mono_rho, monotonic_p=mono_p,
            top_bottom=tb, long_excess=le, nav=nav,
        )
    return results


# ----------------------------------------------------------------------
# 单标的评价
# ----------------------------------------------------------------------
def _evaluate_symbol(sym: str, s: pd.Series, prices: pd.DataFrame,
                     cfg: EvalConfig) -> SymbolReport:
    warn_list: list = []
    # high/low 若有则一并保留（供模拟交易的 K 线图使用），评价计算只用 open/close
    px_cols = [c for c in ("open", "high", "low", "close") if c in prices.columns]
    data = pd.concat(
        [s.rename("s"), prices[px_cols]], axis=1, join="inner"
    ).sort_index()
    if data.index.has_duplicates:
        raise ValueError(f"{sym}: 对齐后索引存在重复时间戳，请检查输入数据")
    if len(data) == 0:
        raise ValueError(f"{sym}: signal 与 prices 按索引对齐后无重叠样本")
    if data["s"].dropna().empty:
        warnings.warn(f"{sym}: 信号全为 NaN", stacklevel=2)
        warn_list.append(f"{sym}: 信号全为 NaN")

    s_vol = None
    if cfg.vol_scaling:
        ret1 = data["close"].pct_change()
        s_vol = ret1.ewm(span=cfg.vol_ewm_span).std()

    ic_table = _eval_ic_table(data["s"], data["open"], data["close"], s_vol,
                              cfg, sym, warn_list)
    rolling = _eval_rolling_ic(data["s"], data["open"], data["close"], cfg)
    buckets = _eval_buckets(data["s"], data["open"], data["close"], cfg, sym, warn_list)

    # 分桶信号模拟交易（side=1..n_buckets × 多/空全部组合；单组合失败只告警不中断评价）
    backtests: dict = {}
    if cfg.run_backtest:
        for side in range(1, cfg.n_buckets + 1):
            for direction in ("long", "short"):
                key = f"side{side}_{direction}"
                try:
                    bt_cfg = BacktestConfig(
                        n_buckets=cfg.n_buckets, side=side, direction=direction,
                        k=cfg.backtest_k, fee_rate=cfg.backtest_fee,
                        bars_per_year=cfg.bars_per_year,
                    )
                    res = run_backtest(data["s"], data[px_cols], bt_cfg)
                    res.label = sym
                    backtests[key] = res
                except Exception as exc:
                    msg = f"{sym}: 模拟交易({key})失败: {exc}"
                    warnings.warn(msg, stacklevel=2)
                    warn_list.append(msg)

    return SymbolReport(
        symbol=sym, start=data.index[0], end=data.index[-1], n_bars=len(data),
        ic_table=ic_table, rolling=rolling, buckets=buckets, warnings=warn_list,
        plot_data=data, backtests=backtests,
    )


# ----------------------------------------------------------------------
# 对外接口
# ----------------------------------------------------------------------
def evaluate_factor(signal, prices, config: EvalConfig | None = None) -> FactorReport:
    """时间序列因子评价主入口。

    Args:
        signal: datetime 索引的信号。Series（单标的）或 DataFrame（列=标的）。
        prices: 单标的时为含 open/close 列的 DataFrame；
            多标的时为 dict[symbol -> DataFrame(open/close)]。
        config: EvalConfig，缺省使用默认配置。

    Returns:
        FactorReport: 结构化评价结果，支持 summary()/to_markdown()/plot()。
    """
    cfg = config or EvalConfig()
    sig_df = _normalize_signal(signal)
    prices_map = _normalize_prices(prices, list(sig_df.columns))

    symbols = {}
    for sym in sig_df.columns:
        symbols[str(sym)] = _evaluate_symbol(str(sym), sig_df[sym], prices_map[sym], cfg)
    return FactorReport(cfg, symbols)
