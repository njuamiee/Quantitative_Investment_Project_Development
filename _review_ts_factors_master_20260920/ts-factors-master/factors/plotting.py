"""因子曲线绘图：个股日 K 线（叠加 EMA）+ 因子曲线双子图。

复用项目根 utils.py 的绘图工具（plot_candles_with_ema / _apply_time_xticks /
add_right_click_vertical_lines），与 A 工程 factor_lib.py 的 plot 子命令对应。

用法::

    python -m factors plot Alpha001 BJ920010 [start] [end] [--save out.png]
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

from .library import FactorLibrary  # noqa: E402
from .price_data import DEFAULT_DAILY_DIR, load_daily_bars  # noqa: E402


def plot_factor(
    factor: str,
    stockid: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    lib: FactorLibrary | None = None,
    daily_dir: str | Path = DEFAULT_DAILY_DIR,
    ema_periods: tuple[int, ...] = (5, 20),
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制个股的因子曲线与对应日 K 线行情。

    参数
    ----------
    factor : 因子标识或原始因子名（如 'Alpha001'）。
    stockid : 股票代码（如 'BJ920010'）。
    start, end : 可选日期范围（'YYYYMMDD' 或 'YYYY-MM-DD'）。
    lib : FactorLibrary 实例，缺省按默认路径构造。
    daily_dir : 日线数据目录。
    ema_periods : 主图叠加的 EMA 周期（现场按收盘价计算）。
    save_path : 非空时保存图片到该路径。
    show : 是否弹窗阻塞显示（存盘后仍可显示）。
    """
    import matplotlib.pyplot as plt

    lib = lib or FactorLibrary()
    s = lib.get(factor, stockid)  # date 索引的因子序列
    # 打印因子序列的时间范围
    print('plot_factor: factor range: ', s.index.min(), '->', s.index.max())
    fid = s.name
    bars = load_daily_bars(stockid, daily_dir)
    print('plot_factor: bars range: ', bars.index.min(), '->', bars.index.max())

    combined = bars.join(s.rename("value"), how="inner")
    if start is not None:
        combined = combined[combined.index >= pd.Timestamp(start)]
    if end is not None:
        combined = combined[combined.index <= pd.Timestamp(end)]
    if combined.empty:
        raise ValueError(
            f"{fid}/{stockid} 在指定范围内没有可绘制数据: {start or 'start'} ~ {end or 'end'}"
        )

    ema_cols: list[str] = []
    for p in ema_periods:
        col = f"EMA{p}"
        combined[col] = combined["close"].ewm(span=p, adjust=False).mean()
        ema_cols.append(col)

    title = f"{fid} - {stockid}"
    if start or end:
        title = f"{title} | {start or 'start'} ~ {end or 'end'}"

    fig, (ax_main, ax_factor) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    utils.plot_candles_with_ema(combined, ema_cols=ema_cols, title=title, ax=ax_main)
    ax_main.legend(loc="best")

    x = np.arange(len(combined))
    ax_factor.plot(x, combined["value"].to_numpy(), label=fid, color="#4169e1", linewidth=1.6)
    ax_factor.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax_factor.set_title(f"Factor curve: {fid}")
    ax_factor.set_ylabel(fid)
    ax_factor.grid(True, linestyle="--", alpha=0.3)
    ax_factor.legend(loc="best")
    utils._apply_time_xticks(ax_factor, combined.index)

    utils.add_right_click_vertical_lines(fig, [ax_main, ax_factor])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"图已保存: {save_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
