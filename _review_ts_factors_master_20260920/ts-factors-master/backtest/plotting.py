"""模拟交易结果绘图：K 线（叠加 EMA 与买卖点）/ 因子值 / 持仓 / 总资产四子图。

四个子图横轴均为时间并对齐（行号绘制，刻度经 utils._apply_time_xticks 显示日期）。
K 线复用 src/utils.py 的 plot_candles_with_ema；输入缺 high/low 列时退化为收盘折线。
图内文字用英文（与 factors/report.py 出图惯例一致，避免中文字体缺字形）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import utils  # noqa: E402

from .engine import BacktestResult  # noqa: E402


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%" if np.isfinite(v) else "NaN"


def _fmt_num(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "NaN"


def plot_backtest(
    result: BacktestResult,
    save_path: str | Path | None = None,
    show: bool = False,
    ema_periods: tuple[int, ...] = (5, 20),
) -> str | None:
    """绘制模拟交易结果四子图。

    子图自上而下：K 线（叠加 EMA 与买 ^ / 卖 v 标记）、因子值、持仓（仓位）、
    总资产曲线。suptitle 给出主要参数与绩效指标。

    Args:
        result: run_backtest 的返回结果（label 非空时用于标题）。
        save_path: 非空时保存图片到该路径。
        show: 是否弹窗阻塞显示。
        ema_periods: K 线主图叠加的 EMA 周期（现场按收盘价计算）。

    Returns:
        保存路径；未保存时返回 None。
    """
    import matplotlib.pyplot as plt

    cfg = result.config
    m = result.metrics
    data = result.data.copy()
    n = len(data)
    if n == 0:
        raise ValueError("backtest 结果为空，无法绘图")
    x = np.arange(n, dtype=float)

    ema_cols: list[str] = []
    for p in ema_periods:
        col = f"EMA{p}"
        data[col] = data["close"].ewm(span=p, adjust=False).mean()
        ema_cols.append(col)

    fig, (ax_k, ax_f, ax_p, ax_e) = plt.subplots(
        4, 1, figsize=(18, 12), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0, 1.2]},
    )

    # 1) K 线 + EMA + 买卖点
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
    trades = result.trades
    #暂时disable画买卖点的功能，以节省时间
    # if not trades.empty:
    #     pos = data.index.get_indexer(trades["entry_date"])
    #     ax_k.scatter(x[pos], trades["entry_px"].to_numpy(), marker="^",
    #                  color="#ff00ff", edgecolors="black", s=45, zorder=5,
    #                  label=f"buy ({len(trades)})")
    #     pos = data.index.get_indexer(trades["exit_date"])
    #     ax_k.scatter(x[pos], trades["exit_px"].to_numpy(), marker="v",
    #                  color="#00bcd4", edgecolors="black", s=45, zorder=5,
    #                  label="sell")
    #     ax_k.legend(loc="best")

    # 2) 因子值
    ax_f.plot(x, data["s"].to_numpy(), linewidth=1.0, color="#4169e1")
    ax_f.set_ylabel("factor")
    ax_f.set_title("Factor value")
    ax_f.grid(True, linestyle="--", alpha=0.3)

    # 3) 持仓（仓位 = 持仓市值/总资产）
    ax_p.fill_between(x, 0.0, result.position.to_numpy(), color="#ff7f0e",
                      alpha=0.6, step=None)
    ax_p.set_ylim(0.0, 1.05)
    ax_p.set_ylabel("position")
    ax_p.set_title("Position (market value / total asset)")
    ax_p.grid(True, linestyle="--", alpha=0.3)

    # 4) 总资产曲线（叠加买入并持有基准：首日收盘买入并持有，绿色）
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

    label = result.label or "backtest"
    fig.suptitle(
        f"{label} | side={cfg.side}/{cfg.direction} buckets={cfg.n_buckets} k={cfg.k} "
        f"fee={cfg.fee_rate * 1e4:.1f}bp | "
        f"Ret={_fmt_pct(m['total_ret'])} Ann={_fmt_pct(m['ann_ret'])} "
        f"MaxDD={_fmt_pct(m['maxdd'])} Sharpe={_fmt_num(m['sharpe'])} "
        f"Calmar={_fmt_num(m['calmar'])} trades={m['n_trades']}",
        fontsize=13,
    )
    utils._apply_time_xticks(ax_e, data.index)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    saved = None
    if save_path:
        save_path = str(save_path)
        fig.savefig(save_path, dpi=120)
        saved = save_path
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return saved
