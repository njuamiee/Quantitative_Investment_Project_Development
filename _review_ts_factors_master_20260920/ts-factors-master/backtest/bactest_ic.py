"""Rolling Rank IC + AR(1)/OU 条件概率信号的模拟回测（纯计算 + CLI + 出图）。

口径（严格防未来函数）：
- 前向收益 r(t→t+k) = Close[t+1+k]/Open[t+1] − 1（复用 factors.evaluator.forward_returns）；
- rolling Rank IC：窗口 W、步长 1、窗口内独立编秩的 Spearman（复用
  factors.evaluator._rolling_spearman），与评价模块同口径——先 (s, r) dropna 再滚动，
  结果按日期对齐回行情索引（停牌等缺行日期 IC 为 NaN）；
- 信息可得性：IC_t 的窗口含 Close[t+1+k]，t 日收盘时最新可用 IC 下标为 t−k−1。
  第 t 日的预测取 ≤ t−k−1 的最后一条非 NaN IC（下标记 j），用 j 及之前的全部
  非 NaN IC（扩展窗）现场做 AR(1)/OU 拟合（factors.sign_forecast._ou_fit），
  P(IC_t>0) = _ou_p_pos(mu, b, sig, ic[j], gap=t−j)（gap ≥ k+1，随跨度自动衰减
  到长期均值方向）；可用 IC 条数 < min_obs 时不预测；
- 开仓信号（t 日收盘判定）：P(IC_t>0) > p_threshold（默认 0.7）且
  s_t ≥ 过去 W 根因子值的 q_quantile 分位（默认 0.7，即"前 30% 分位"，窗口含当日）
  → t+1 开盘价开仓；
- 仓位 2 档：tier 0/1/2 → 目标持仓市值占当前总资产 0%/30%/100%；每次触发升一档，
  满仓（tier=2）再触发跳过（n_skipped 计数）；首次建仓（tier 0→1）bar 起满 k 天
  到期日收盘整仓平仓（加仓不重置计时）；数据末端未平仓按最后一日收盘强平（forced）；
- 手续费双边收取（默认单边 5e-5）；初始资金默认 100 万元；只做多，允许小数股。

IC 正负预测评估：在所有"有 OU 预测且真实 IC_t 非 NaN"的 t 上，
pred_pos = (P > 0.5)、actual_pos = (IC_t > 0)，报告 sensitivity=TP/(TP+FN) 与
precision=TP/(TP+FP)（注意评估阈值 0.5 与开仓阈值 p_threshold=0.7 是两个口径）。

CLI（在 src 目录下运行）::

    python -m backtest.bactest_ic 920725 Alpha098 --k 5 --W 50
    python -m backtest.bactest_ic BJ920725 Alpha098 --k 5 --W 50 --start 2023-01-01 --show

输出 <项目根>/reports/bactest_ic_<fid>_<stock>_k<k>_W<W>[_N]/：
6 子图 png（日 K 线/因子值/rolling IC/开仓信号/仓位/总资产）+
数据 CSV（首列 date）+ trades CSV；控制台打印绩效指标与 IC 预测
sensitivity/precision。
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 允许以脚本方式直接运行（python backtest/bactest_ic.py）
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from backtest import normalize_stockid  # noqa: E402
from backtest.engine import _calc_metrics  # noqa: E402
from factors.evaluator import _rolling_spearman, forward_returns  # noqa: E402
from factors.sign_forecast import _ou_fit, _ou_p_pos  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(_SRC_DIR), "reports")  # 项目根/reports，与 src 平级

_TRADE_COLS = ["entry_date", "entry_px", "exit_date", "exit_px",
               "gross_ret", "net_ret", "n_adds", "forced"]

# 档位 → 目标持仓市值占总资产比例（两档：30% / 100%）
_TIER_RATIO = {1: 0.30, 2: 1.00}


@dataclass
class ICBacktestConfig:
    """IC 信号模拟交易配置。

    Attributes:
        k: 持仓天数（前向收益 horizon；首次建仓起满 k 天到期日收盘整仓平仓）。
        W: rolling Rank IC 窗口宽度，同时是因子 rolling 分位窗口宽度。
        p_threshold: 开仓概率阈值，P(IC_t>0) 大于它才允许触发（默认 0.7）。
        q_quantile: 因子 rolling 分位阈值（默认 0.7，即前 30% 分位）。
        min_obs: 开始做 OU 预测所需的最少历史 IC 条数（默认 60）。
        fee_rate: 单边费率（默认 5e-5，双边收取）。
        init_capital: 初始资金（默认 100 万元）。
        bars_per_year: 年化用每年 bar 数（日线 252）。
    """

    k: int = 5
    W: int = 50
    p_threshold: float = 0.7
    q_quantile: float = 0.7
    min_obs: int = 60
    fee_rate: float = 5e-5
    init_capital: float = 1e6
    bars_per_year: int = 252

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k 至少为 1")
        if self.W < 2:
            raise ValueError("W 至少为 2")
        if not (0.0 < self.p_threshold < 1.0):
            raise ValueError("p_threshold 须在 (0, 1) 之间")
        if not (0.0 < self.q_quantile < 1.0):
            raise ValueError("q_quantile 须在 (0, 1) 之间")
        if self.min_obs < 5:
            raise ValueError("min_obs 至少为 5")
        if self.fee_rate < 0:
            raise ValueError("fee_rate 不能为负")
        if self.init_capital <= 0:
            raise ValueError("init_capital 必须为正")


@dataclass
class ICBacktestResult:
    """IC 信号模拟交易结果。"""

    config: ICBacktestConfig
    data: pd.DataFrame  # OHLCV + s/q_thr/rolling_ic/prob_pos/signal/position/equity
    trades: pd.DataFrame  # 每个建仓-平仓周期一行，列见 _TRADE_COLS
    equity: pd.Series  # 逐日总资产（date 索引，起始 init_capital）
    position: pd.Series  # 逐日仓位 = 持仓市值 / 总资产（0~1）
    metrics: dict  # total_ret/ann_ret/maxdd/sharpe/calmar/n_trades/n_skipped
    #              + ic_sensitivity/ic_precision/n_ic_pred
    label: str = ""  # 图题用标识（如 "Alpha098 BJ920725"），由调用方填写


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%" if np.isfinite(v) else "NaN"


def _fmt_num(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "NaN"


def _flat_result(data: pd.DataFrame, cfg: ICBacktestConfig) -> ICBacktestResult:
    """样本不足时的平直净值结果（指标 NaN，无交易无预测）。"""
    n = len(data)
    out = data.copy()
    out["q_thr"] = np.nan
    out["rolling_ic"] = np.nan
    out["prob_pos"] = np.nan
    out["signal"] = np.zeros(n, dtype=int)
    equity = pd.Series(cfg.init_capital, index=data.index, name="equity")
    position = pd.Series(0.0, index=data.index, name="position")
    out["position"] = position
    out["equity"] = equity
    trades = pd.DataFrame(columns=_TRADE_COLS)
    metrics = _calc_metrics(equity, cfg.init_capital, cfg.bars_per_year, 0, 0)
    metrics.update(ic_sensitivity=np.nan, ic_precision=np.nan, n_ic_pred=0)
    return ICBacktestResult(config=cfg, data=out, trades=trades, equity=equity,
                            position=position, metrics=metrics)


def run_ic_backtest(signal: pd.Series, prices: pd.DataFrame,
                    config: ICBacktestConfig | None = None) -> ICBacktestResult:
    """Rolling Rank IC + AR(1)/OU 概率信号的模拟交易主入口。

    Args:
        signal: datetime 索引的因子序列（NaN 不参与 IC 与触发）。
        prices: 含 open/close 列的日线 DataFrame（有 high/low/volume 时一并保留）。
        config: ICBacktestConfig，缺省用默认参数。

    Returns:
        ICBacktestResult: 交易明细、逐日数据（含 IC/概率/信号/仓位/总资产）与指标。
    """
    cfg = config or ICBacktestConfig()
    if "open" not in prices.columns or "close" not in prices.columns:
        raise ValueError("prices 必须包含 open/close 列")
    px_cols = [c for c in ("open", "high", "low", "close", "volume")
               if c in prices.columns]
    data = pd.concat([signal.rename("s"), prices[px_cols]], axis=1,
                     join="inner").sort_index()
    if data.index.has_duplicates:
        raise ValueError("对齐后索引存在重复时间戳，请检查输入数据")
    n = len(data)
    if n < 2:
        warnings.warn("对齐后样本不足 2 条，无法模拟交易", stacklevel=2)
        return _flat_result(data, cfg)

    # ---- 前向收益与 rolling Rank IC（与 factors.evaluator._eval_rolling_ic 同口径） ----
    k = int(cfg.k)
    fwd = forward_returns(data["open"], data["close"], k)
    pair = pd.concat([data["s"], fwd.rename("r")], axis=1).dropna()
    ic = pd.Series(_rolling_spearman(pair["s"].values, pair["r"].values, cfg.W),
                   index=pair.index, name="rolling_ic").reindex(data.index)
    # ---- 因子 rolling 分位阈值（窗口 [t−W+1, t]，含当日） ----
    q_thr = data["s"].rolling(cfg.W).quantile(cfg.q_quantile)

    ic_arr = ic.to_numpy(dtype=float)
    s_arr = data["s"].to_numpy(dtype=float)
    q_arr = q_thr.to_numpy(dtype=float)
    open_arr = data["open"].to_numpy(dtype=float)
    close_arr = data["close"].to_numpy(dtype=float)
    dates = data.index

    # last_valid[i] = ≤ i 的最后一条非 NaN IC 下标（无则 -1），供信息可得性定位用
    last_valid = np.full(n, -1, dtype=int)
    last = -1
    for i in range(n):
        if not np.isnan(ic_arr[i]):
            last = i
        last_valid[i] = last

    # ---- 逐日 OU 预测：第 t 日只用 ≤ t−k−1 的非 NaN IC（扩展窗重估），gap = t−j ----
    prob = np.full(n, np.nan)
    for t in range(n):
        cut = t - k - 1
        if cut < 0:
            continue
        j = last_valid[cut]
        if j < 0:
            continue
        hist = ic_arr[:j + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < cfg.min_obs:
            continue
        mu, b, sig = _ou_fit(hist)
        prob[t] = float(_ou_p_pos(mu, b, sig, ic_arr[j], gap=t - j))

    # ---- 开仓信号：P > p_threshold 且 s_t ≥ rolling 分位阈值；t 日收盘判定 ----
    trigger = np.zeros(n, dtype=bool)
    trigger[:-1] = (prob[:-1] > cfg.p_threshold) & (s_arr[:-1] >= q_arr[:-1])

    # ---- 逐 bar 模拟（两档仓位 30%/100% + 整仓 k 天到期，只做多） ----
    fee = cfg.fee_rate
    cash = float(cfg.init_capital)
    shares = 0.0
    tier = 0
    entry_bar = -1  # 首次建仓 bar（整仓计时起点）
    cyc_entry_date = None
    cyc_qty = 0.0       # 累计成交股数
    cyc_notional = 0.0  # 累计成交金额（不含费）
    cyc_cashflow = 0.0  # 累计买入总花费（含费）
    cyc_adds = 0
    trades: list[dict] = []
    n_skipped = 0
    equity = np.empty(n)
    position = np.empty(n)

    def _close_all(i: int, forced: bool) -> None:
        """按 close[i] 整仓平仓并记录一个交易周期。"""
        nonlocal cash, shares, tier, entry_bar
        nonlocal cyc_entry_date, cyc_qty, cyc_notional, cyc_cashflow, cyc_adds
        px = close_arr[i]
        proceeds = shares * px * (1.0 - fee)
        cash += proceeds
        entry_px = cyc_notional / cyc_qty
        trades.append({
            "entry_date": cyc_entry_date, "entry_px": float(entry_px),
            "exit_date": dates[i], "exit_px": float(px),
            "gross_ret": float(px / entry_px - 1.0),
            "net_ret": float(proceeds / cyc_cashflow - 1.0),
            "n_adds": cyc_adds, "forced": forced,
        })
        shares = 0.0
        tier = 0
        entry_bar = -1
        cyc_entry_date = None
        cyc_qty = cyc_notional = cyc_cashflow = 0.0
        cyc_adds = 0

    for i in range(n):
        # 1) 上一 bar 收盘触发 → 本 bar 开盘升一档（到期平仓发生在本 bar 收盘，
        #    故此刻仍持满仓时新信号算 skipped）
        if i > 0 and trigger[i - 1]:
            eq_open = cash + shares * open_arr[i]
            if tier >= 2 or eq_open <= 0.0:
                n_skipped += 1
            else:
                tier += 1
                target = _TIER_RATIO[tier] * eq_open
                delta = target - shares * open_arr[i]  # 需补的名义市值
                if delta > 0.0:
                    cash -= delta * (1.0 + fee)
                    shares += delta / open_arr[i]
                    cyc_cashflow += delta * (1.0 + fee)
                    cyc_qty += delta / open_arr[i]
                    cyc_notional += delta
                    cyc_adds += 1
                # tier 0→1（首次建仓，必然成交）启动整仓计时
                if tier == 1:
                    entry_bar = i
                    cyc_entry_date = dates[i]
        # 2) 整仓到期：首次建仓起满 k 天，本 bar 收盘全平
        if tier > 0 and entry_bar >= 0 and i == entry_bar + k:
            _close_all(i, forced=False)
        # 3) 收盘记账
        mkt_val = shares * close_arr[i]
        equity[i] = cash + mkt_val
        position[i] = abs(mkt_val) / equity[i] if equity[i] > 0 else 0.0

    # ---- 数据结束：未平仓持仓按最后一日收盘价强制平仓 ----
    if tier > 0 and shares != 0.0:
        _close_all(n - 1, forced=True)
        equity[-1] = cash
        position[-1] = 0.0

    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS)
    equity_s = pd.Series(equity, index=dates, name="equity")
    position_s = pd.Series(position, index=dates, name="position")
    metrics = _calc_metrics(equity_s, cfg.init_capital, cfg.bars_per_year,
                            len(trades_df), n_skipped)

    # ---- IC 正负预测评估：P>0.5 判正，对照真实 IC_t 的正负 ----
    mask = ~np.isnan(prob) & ~np.isnan(ic_arr)
    n_pred = int(mask.sum())
    sens = prec = np.nan
    if n_pred:
        pred_pos = prob[mask] > 0.5
        actual_pos = ic_arr[mask] > 0
        tp = int(np.sum(pred_pos & actual_pos))
        fp = int(np.sum(pred_pos & ~actual_pos))
        fn = int(np.sum(~pred_pos & actual_pos))
        sens = tp / (tp + fn) if tp + fn > 0 else np.nan
        prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    metrics.update(ic_sensitivity=sens, ic_precision=prec, n_ic_pred=n_pred)

    out = data.copy()
    out["q_thr"] = q_thr
    out["rolling_ic"] = ic
    out["prob_pos"] = prob
    out["signal"] = trigger.astype(int)
    out["position"] = position_s
    out["equity"] = equity_s
    return ICBacktestResult(config=cfg, data=out, trades=trades_df,
                            equity=equity_s, position=position_s,
                            metrics=metrics)


# ----------------------------------------------------------------------
# 绘图（6 子图，横轴时间对齐，图内文字英文）
# ----------------------------------------------------------------------
def plot_ic_backtest(
    result: ICBacktestResult,
    save_path: str | os.PathLike | None = None,
    show: bool = False,
    ema_periods: tuple[int, ...] = (5, 20),
) -> str | None:
    """绘制 IC 信号回测 6 子图：日 K 线/因子值/rolling IC/开仓信号/仓位/总资产。

    Args:
        result: run_ic_backtest 的返回结果（label 非空时用于标题）。
        save_path: 非空时保存图片到该路径。
        show: 是否弹窗阻塞显示。
        ema_periods: K 线主图叠加的 EMA 周期（现场按收盘价计算）。

    Returns:
        保存路径；未保存时返回 None。
    """
    import matplotlib.pyplot as plt

    import utils

    cfg = result.config
    m = result.metrics
    data = result.data.copy()
    n = len(data)
    if n == 0:
        raise ValueError("回测结果为空，无法绘图")
    x = np.arange(n, dtype=float)

    ema_cols: list[str] = []
    for p in ema_periods:
        col = f"EMA{p}"
        data[col] = data["close"].ewm(span=p, adjust=False).mean()
        ema_cols.append(col)

    fig, (ax_k, ax_f, ax_ic, ax_sig, ax_p, ax_e) = plt.subplots(
        6, 1, figsize=(18, 16), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0, 1.0, 1.0, 1.2]},
    )

    # 1) K 线 + EMA
    if {"high", "low"}.issubset(data.columns):
        utils.plot_candles_with_ema(
            data, ema_cols=ema_cols, title="Candles with EMA", ax=ax_k)
    else:  # 缺 high/low 时退化为收盘折线
        ax_k.plot(x, data["close"].to_numpy(), linewidth=1.0, label="close")
        for col in ema_cols:
            ax_k.plot(x, data[col].to_numpy(), linewidth=1.2, label=col)
        ax_k.set_title("Close line with EMA (no high/low)")
        ax_k.legend(loc="best")
        ax_k.grid(True, linestyle="--", alpha=0.25)

    # 2) 因子值 + rolling 分位阈值
    ax_f.plot(x, data["s"].to_numpy(), linewidth=1.0, color="#4169e1",
              label="factor")
    ax_f.plot(x, data["q_thr"].to_numpy(), linewidth=0.8, linestyle="--",
              color="#ff7f0e", label=f"rolling q{cfg.q_quantile:.2f} (W={cfg.W})")
    ax_f.set_ylabel("factor")
    ax_f.set_title("Factor value vs rolling quantile threshold")
    ax_f.legend(loc="best")
    ax_f.grid(True, linestyle="--", alpha=0.3)

    # 3) rolling Rank IC
    ax_ic.plot(x, data["rolling_ic"].to_numpy(), linewidth=1.0, color="#9467bd")
    ax_ic.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_ic.set_ylabel("rank IC")
    ax_ic.set_title(f"Rolling Rank IC (W={cfg.W}, k={cfg.k})")
    ax_ic.grid(True, linestyle="--", alpha=0.3)

    # 4) 开仓信号：P(IC>0) 曲线 + 阈值线 + 触发标记
    ax_sig.plot(x, data["prob_pos"].to_numpy(), linewidth=1.0, color="#2ca02c",
                label="P(IC>0)")
    ax_sig.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax_sig.axhline(cfg.p_threshold, color="#d62728", linewidth=0.8,
                   linestyle="--", label=f"thr={cfg.p_threshold}")
    sig_idx = np.flatnonzero(data["signal"].to_numpy() > 0)
    if len(sig_idx):
        ax_sig.scatter(x[sig_idx], data["prob_pos"].to_numpy()[sig_idx],
                       marker="^", color="#ff00ff", edgecolors="black", s=45,
                       zorder=5, label=f"entry signal ({len(sig_idx)})")
    ax_sig.set_ylim(-0.05, 1.05)
    ax_sig.set_ylabel("prob")
    ax_sig.set_title("P(IC>0) by AR(1)/OU & entry signals")
    ax_sig.legend(loc="best")
    ax_sig.grid(True, linestyle="--", alpha=0.3)

    # 5) 仓位（持仓市值/总资产）
    ax_p.fill_between(x, 0.0, result.position.to_numpy(), color="#ff7f0e",
                      alpha=0.6)
    ax_p.set_ylim(0.0, 1.05)
    ax_p.set_ylabel("position")
    ax_p.set_title("Position (market value / total asset)")
    ax_p.grid(True, linestyle="--", alpha=0.3)

    # 6) 总资产曲线（叠加买入并持有基准：首日收盘买入并持有，绿色）
    ax_e.plot(x, result.equity.to_numpy(), linewidth=1.2, color="#d62728",
              label="strategy")
    bh = data["close"].to_numpy(dtype=float)
    if len(bh) and bh[0] > 0:
        ax_e.plot(x, cfg.init_capital * bh / bh[0], linewidth=1.0,
                  color="#2ca02c", label="buy&hold")
    ax_e.axhline(cfg.init_capital, color="gray", linewidth=0.8, linestyle="--")
    ax_e.legend(loc="best")
    ax_e.set_ylabel("equity")
    ax_e.set_title("Total asset (vs buy & hold)")
    ax_e.set_xlabel("time")
    ax_e.grid(True, linestyle="--", alpha=0.3)

    label = result.label or "ic_backtest"
    fig.suptitle(
        f"{label} | k={cfg.k} W={cfg.W} p>{cfg.p_threshold} "
        f"q{cfg.q_quantile} fee={cfg.fee_rate * 1e4:.1f}bp | "
        f"Ret={_fmt_pct(m['total_ret'])} Ann={_fmt_pct(m['ann_ret'])} "
        f"MaxDD={_fmt_pct(m['maxdd'])} Sharpe={_fmt_num(m['sharpe'])} "
        f"Calmar={_fmt_num(m['calmar'])} Sens={_fmt_num(m['ic_sensitivity'])} "
        f"Prec={_fmt_num(m['ic_precision'])} trades={m['n_trades']}",
        fontsize=13,
    )
    utils._apply_time_xticks(ax_e, data.index)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    saved = None
    if save_path:
        save_path = str(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        saved = save_path
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return saved


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _make_unique_output_dir(base_outdir: str, name: str) -> str:
    """在 base_outdir 下生成唯一的 name 子目录，避免覆盖上次输出。"""
    candidate = os.path.join(base_outdir, name)
    idx = 1
    while os.path.exists(candidate) and os.listdir(candidate):
        candidate = os.path.join(base_outdir, f"{name}_{idx}")
        idx += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling Rank IC + AR(1)/OU 概率信号模拟回测：t 日收盘判定"
                    "（P(IC_t>0)>p_threshold 且因子位于 rolling 前 30% 分位），"
                    "t+1 开盘升档（30%/100%），首次建仓起满 k 天到期日收盘整仓平仓。")
    parser.add_argument("stock", help="股票代码（如 920725 或 BJ920725）")
    parser.add_argument("factor", help="因子标识或因子名（如 Alpha098）")
    parser.add_argument("--k", type=int, default=5,
                        help="持仓天数（前向收益 horizon，默认 5）")
    parser.add_argument("--W", type=int, default=50,
                        help="rolling IC 窗口宽度（兼作因子分位窗口，默认 50）")
    parser.add_argument("--p-threshold", type=float, default=0.7,
                        help="开仓概率阈值（默认 0.7）")
    parser.add_argument("--quantile", type=float, default=0.7,
                        help="因子 rolling 分位阈值（默认 0.7，即前 30%% 分位）")
    parser.add_argument("--min-obs", type=int, default=60,
                        help="开始 OU 预测所需的最少历史 IC 条数（默认 60）")
    parser.add_argument("--fee", type=float, default=5e-5,
                        help="单边费率（默认 5e-5，即万 0.5，双边收取）")
    parser.add_argument("--init-capital", type=float, default=1e6,
                        help="初始资金（默认 1000000，即 100 万元）")
    parser.add_argument("--start", default=None,
                        help="起始日期，如 2023-01-01 或 20230101")
    parser.add_argument("--end", default=None, help="结束日期")
    parser.add_argument("--show", action="store_true",
                        help="弹窗显示图（缺省只保存文件）")
    return parser.parse_args(argv)


def _print_detail(res: ICBacktestResult) -> None:
    """控制台打印绩效指标与 IC 预测 sensitivity/precision。"""
    m = res.metrics
    cfg = res.config
    print("=" * 78)
    print(f"IC 信号回测: {res.label} | k={cfg.k} W={cfg.W} "
          f"p_threshold={cfg.p_threshold} quantile={cfg.q_quantile} "
          f"fee={cfg.fee_rate * 1e4:.1f}bp(双边) init={cfg.init_capital:,.0f}")
    print(f"样本区间: {res.data.index[0].date()} ~ {res.data.index[-1].date()}, "
          f"bar 数: {len(res.data)}")
    print("口径: t 日收盘判定（P(IC_t>0)>阈值 且 因子 ≥ rolling 分位阈值）→ t+1 开盘"
          "升档(30%/100%)，首次建仓起满 k 天到期日收盘整仓平仓")
    print("=" * 78)
    print(f"  总收益率:   {_fmt_pct(m['total_ret'])}")
    print(f"  年化收益率: {_fmt_pct(m['ann_ret'])}")
    print(f"  最大回撤:   {_fmt_pct(m['maxdd'])}")
    print(f"  夏普率:     {_fmt_num(m['sharpe'])}")
    print(f"  卡玛比率:   {_fmt_num(m['calmar'])}")
    n_forced = int(res.trades["forced"].sum()) if len(res.trades) else 0
    print(f"  交易周期数: {m['n_trades']}（满仓信号跳过 {m['n_skipped']} 次，"
          f"强制平仓 {n_forced} 个周期）")
    print(f"  IC 预测:    sensitivity={_fmt_num(m['ic_sensitivity'])} "
          f"precision={_fmt_num(m['ic_precision'])}"
          f"（预测样本 {m['n_ic_pred']} 日，P>0.5 判正）")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from factors.library import FactorLibrary
    from factors.price_data import load_daily_bars

    stockid = normalize_stockid(args.stock)
    try:
        lib = FactorLibrary()
        fid = lib._resolve_factor_id(args.factor)  # 未知因子在此抛 KeyError
        s = lib.get(fid, stockid, start=args.start, end=args.end)
        prices = load_daily_bars(stockid)
        cfg = ICBacktestConfig(
            k=args.k, W=args.W, p_threshold=args.p_threshold,
            q_quantile=args.quantile, min_obs=args.min_obs,
            fee_rate=args.fee, init_capital=args.init_capital)
        res = run_ic_backtest(s, prices, cfg)
    except KeyError as exc:
        print(f"ERROR: 未知因子 {args.factor!r}，请用 python -m factors list 查看",
              file=sys.stderr)
        raise SystemExit(1) from exc
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    res.label = f"{fid} {stockid}"
    _print_detail(res)

    name = f"bactest_ic_{fid}_{stockid}_k{cfg.k}_W{cfg.W}"
    outdir = _make_unique_output_dir(OUTDIR, name)
    png_path = plot_ic_backtest(res, save_path=os.path.join(outdir, f"{name}.png"),
                                show=args.show)
    # 数据 CSV：第一列 date，其后 行情/因子/IC/概率/信号/仓位/总资产
    out = res.data.rename_axis("date").reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.rename(columns={"s": "factor"})
    px_cols = [c for c in ("open", "high", "low", "close", "volume")
               if c in out.columns]
    cols = ["date", *px_cols, "factor", "q_thr", "rolling_ic", "prob_pos",
            "signal", "position", "equity"]
    data_path = os.path.join(outdir, f"{name}_data.csv")
    out[cols].to_csv(data_path, encoding="utf-8-sig", index=False)
    trades_path = os.path.join(outdir, f"{name}_trades.csv")
    res.trades.to_csv(trades_path, encoding="utf-8-sig", index=False)

    print(f"\n图已保存: {png_path}")
    print(f"数据 CSV: {data_path}")
    print(f"交易明细: {trades_path}")
    return 0


if __name__ == "__main__":
    main()
