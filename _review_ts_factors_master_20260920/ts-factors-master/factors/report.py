"""因子评价结果的数据结构与报告导出（markdown / 图）。

FactorReport 为 evaluate_factor 的返回类型：
  - symbols: 每个标的一个 SymbolReport（IC 表、滚动 IC、分层结果等）；
  - summary(): 跨标的汇总表；
  - to_markdown(): 导出 markdown 报告（头部附口径说明）；
  - plot(outdir): 保存图（IC 衰减曲线、滚动 IC、每桶柱状图、顶/底桶净值）；
  - save_rolling_ic_csv(outdir): 导出滚动 IC 数值 CSV（与滚动 IC 图同口径）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from .eval_config import EvalConfig

# 滚动 IC 图绘制的窗口列表（仅影响 plot 出图；ICIR 等统计仍用 EvalConfig.rolling_window）
_ROLLING_PLOT_WINDOWS = (10, 20, 30, 50, 100)


def rolling_ic_frame(rep: "SymbolReport", horizons,
                     windows=_ROLLING_PLOT_WINDOWS) -> pd.DataFrame | None:
    """由 SymbolReport.plot_data 现场重算各 k×W 滚动 Rank IC 并合并成宽表。

    列：factor（原始因子值）+ rank_ic_k{k}_W{w}，索引为对齐后的日期，
    与滚动 IC 图 / rolling_ic_*.csv 同一口径；窗口期不足处为 NaN。
    rep.rolling 中缺失的 k 跳过；plot_data 缺失时返回 None。
    """
    from .evaluator import _rolling_spearman, forward_returns

    if rep.plot_data is None:
        return None
    cols: dict[str, pd.Series] = {"factor": rep.plot_data["s"]}
    for k in horizons:
        if rep.rolling.get(k) is None:
            continue
        fwd = forward_returns(rep.plot_data["open"], rep.plot_data["close"], k)
        pair = pd.concat(
            [rep.plot_data["s"].rename("s"), fwd.rename("r")], axis=1
        ).dropna()
        for w in windows:
            vals = _rolling_spearman(pair["s"].values, pair["r"].values, w)
            cols[f"rank_ic_k{k}_W{w}"] = pd.Series(vals, index=pair.index)
    return pd.DataFrame(cols).reindex(rep.plot_data.index)


@dataclass
class BucketResult:
    """单个 horizon 的分层测试结果。"""

    k: int
    table: pd.DataFrame  # 行=桶序号 1..N，列=count/mean/ann_ret/sharpe
    monotonic_rho: float  # 桶序号与桶均收益的 Spearman 相关
    monotonic_p: float
    top_bottom: dict  # mean/t/p/ann_sharpe/maxdd/n/nw_lag
    long_excess: dict  # mean/t/p/n/nw_lag（顶桶相对全样本均值的超额）
    nav: pd.DataFrame  # 顶/底/价差累计净值（绘图用），列=top/bottom/spread


@dataclass
class RollingICResult:
    """滚动 IC 结果（horizon 固定为 horizons[0]）。"""

    horizon: int
    window: int
    series: pd.Series
    mean: float
    std: float
    icir: float
    icir_annual: float  # ICIR × sqrt(252/W)（严格按此公式）
    frac_pos: float
    t_nw: float  # IC 均值的 Newey-West 修正 t 值
    p_nw: float
    nw_lag: int


@dataclass
class SymbolReport:
    """单个标的的因子评价结果。"""

    symbol: str
    start: object
    end: object
    n_bars: int
    ic_table: pd.DataFrame  # 行=k，列=rank_ic/ic_pvalue/beta/beta_t/beta_p/r2/n_obs/nw_lag/insufficient
    rolling: dict  # k -> RollingICResult（每个 horizon 一份）
    buckets: dict  # k -> BucketResult
    warnings: list = field(default_factory=list)
    plot_data: pd.DataFrame | None = None  # 对齐后的 s/open/close，供评价图使用
    backtests: dict = field(default_factory=dict)  # key("side{s}_{direction}") -> BacktestResult


class FactorReport:
    """因子评价报告：结构化结果 + markdown / 图导出。"""

    def __init__(self, config: EvalConfig, symbols: dict[str, SymbolReport]):
        self.config = config
        self.symbols = symbols

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """跨标的汇总表：行=标的，列=各 k 的 RankIC、各 k 的滚动 IC 均值，
        以及主 horizon（horizons[0]）的滚动 IC 详细统计。"""
        rows = {}
        for sym, rep in self.symbols.items():
            row = {}
            for k, r in rep.ic_table.iterrows():
                row[f"RankIC({int(k)})"] = r["rank_ic"]
            for k, ro in rep.rolling.items():
                row[f"RollingIC({int(k)})"] = ro.mean
            ro0 = rep.rolling[self.config.horizons[0]]
            row["IC_std"] = ro0.std
            row["ICIR"] = ro0.icir
            row["ICIR_annual"] = ro0.icir_annual
            row["IC>0"] = ro0.frac_pos
            row["n_bars"] = rep.n_bars
            for key, bt in rep.backtests.items():
                m = bt.metrics
                row[f"BT_TotalRet({key})"] = m["total_ret"]
                row[f"BT_AnnRet({key})"] = m["ann_ret"]
                row[f"BT_MaxDD({key})"] = m["maxdd"]
                row[f"BT_Sharpe({key})"] = m["sharpe"]
                row[f"BT_Calmar({key})"] = m["calmar"]
                row[f"BT_NTrades({key})"] = m["n_trades"]
            rows[sym] = row
        df = pd.DataFrame(rows).T
        df["n_bars"] = df["n_bars"].astype(int)
        return df

    def backtest_metrics(self) -> pd.DataFrame:
        """模拟交易指标长表：每 (标的, 组合) 一行，列含 side/direction 与全部指标。

        组合 key 形如 side{s}_{direction}（side=1..n_buckets，direction=long/short）；
        无任何模拟交易结果时返回空表。
        """
        rows = []
        for sym, rep in self.symbols.items():
            for key, bt in rep.backtests.items():
                cfg = bt.config
                rows.append({"symbol": sym, "key": key, "side": cfg.side,
                             "direction": cfg.direction, "k": cfg.k,
                             **bt.metrics})
        return pd.DataFrame(rows)

    def save_rolling_ic_csv(self, outdir: str) -> list[str]:
        """导出滚动 IC 数值 CSV：每个标的一份 rolling_ic_<标的>.csv，返回路径列表。

        第一列时间（date），第二列原始因子值（factor），其后每列一个 (k, W) 组合的
        滚动 Rank IC（列名 rank_ic_k{k}_W{w}，W 取 _ROLLING_PLOT_WINDOWS，
        与滚动 IC 图同一口径、由 rolling_ic_frame 现场重算）；不足窗口期的位置为 NaN。
        plot_data 缺失时退化为仅 config.rolling_window 单窗口、无因子值列。
        """
        os.makedirs(outdir, exist_ok=True)
        paths = []
        for sym, rep in self.symbols.items():
            safe = sym.replace(".", "_").replace("/", "_")
            frame = rolling_ic_frame(rep, self.config.horizons)
            if frame is None:
                cols = {f"rank_ic_k{k}_W{int(ro.window)}": ro.series
                        for k, ro in rep.rolling.items()}
                if not cols:
                    continue
                frame = pd.DataFrame(cols)
            frame = frame.copy()
            frame.index.name = "date"
            p = os.path.join(outdir, f"rolling_ic_{safe}.csv")
            frame.to_csv(p, encoding="utf-8-sig")
            paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # markdown 报告
    # ------------------------------------------------------------------
    @staticmethod
    def _df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
        """不依赖 tabulate 的简易 markdown 表格输出。"""
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "NaN")
        cols = [str(c) for c in d.columns]
        idx_name = d.index.name or ""
        lines = ["| " + idx_name + " | " + " | ".join(cols) + " |"]
        lines.append("|" + "---|" * (len(cols) + 1))
        for idx, row in d.iterrows():
            lines.append("| " + str(idx) + " | " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)

    def _header(self) -> str:
        cfg = self.config
        lines = [
            "# 因子评价报告",
            "",
            "## 口径说明",
            "",
            f"- 前向收益（return_convention={cfg.return_convention}）：t bar 收盘生成信号，"
            "t+1 开盘价进场，r(t→t+k) = Close[t+1+k] / Open[t+1] − 1（无未来函数）。",
            f"- horizons: {list(cfg.horizons)}（单位: bar）。",
            f"- 滚动 IC：窗口 W={cfg.rolling_window} bar、步长 1 bar 的 Rank IC 时间序列，"
            "对 horizons 中每个 k 分别计算；跨标的汇总表中 IC_std/ICIR/ICIR_annual/IC>0 "
            f"列为主 horizon k={cfg.horizons[0]} 的统计。",
            "- Newey-West 滞后规则：预测回归（含顶底价差、多头超额 t 检验）"
            "L = max(k, ⌊4·(T/100)^(2/9)⌋)；滚动 IC 均值检验 L = max(W, ⌊4·(T/100)^(2/9)⌋)；"
            f"nw_lag 配置 = {cfg.nw_lag}（None 表示自动），各检验实际使用的 L 见结果表。",
            "- NW t 值说明：本报告中所有 t 值（beta_t、滚动 IC 均值、顶底价差、多头超额的 t 检验）"
            "均为 Newey-West HAC 修正后的 t 值——点估计（OLS 系数或序列均值）不变，"
            "仅用 Bartlett 核（权重 1−l/(L+1)）对前 L 阶自协方差加权累加重估标准误，"
            "以校正序列自相关与异方差；含 T/(T−p) 自由度修正，L=0 时退化为普通同方差标准误；"
            "p 值基于 t 分布（df=T−p）。",
            f"- 预测回归：r(t→t+k)/σ̂_t = α + β·s_t + ε，vol_scaling={cfg.vol_scaling}"
            f"（σ̂_t 为 1-bar close/close 收益的 ewm(span={cfg.vol_ewm_span}) 标准差）。",
            "- 预测回归表列含义：beta 为上述一元 OLS 的斜率（含常数项 α），"
            "衡量信号每单位取值对波动率调整后前向收益的影响；r2 为判定系数；"
            "beta_t / beta_p 为检验 β=0 的 NW 修正 t 值与对应 p 值，滞后阶数见 nw_lag 列。",
            f"- 年化约定：bars_per_year={cfg.bars_per_year}；"
            "分层年化收益 ann_ret = mean(r_k)·(bars_per_year/k)，"
            "Sharpe = mean/std·√(bars_per_year/k)；"
            "年化 ICIR = ICIR·√(252/W)。",
            f"- 分桶规则：按信号值的全样本分位数等频（pd.qcut）分成 n_buckets={cfg.n_buckets} 桶，"
            "每桶样本约占总有效样本的 1/N；分桶依据是信号 s_t 的取值（非收益），"
            "桶序号 1 = 信号最低，N = 信号最高；信号有效取值数 < N 时自动降桶并告警，"
            "常数信号跳过分层测试；分位点基于全样本，属样本内诊断口径。",
            "- 顶底价差：逐 t 构造多空序列——t 属于顶桶（信号最高桶）取 +r(t→t+k)，"
            "属于底桶取 −r(t→t+k)，其余桶不计入；报告该序列的均值、NW t 值、"
            "年化 Sharpe（mean/std·√(bars_per_year/k)）与累计净值最大回撤。",
            "- 多头超额口径：顶桶前向收益 − 全样本同 k 前向收益均值（即相对买入持有的超额）。",
            "- NAV = Net Asset Value（净值）：图中 cumulative NAV 为对应收益序列按时间复利累乘的"
            "累计净值 NAV_t = ∏_{i≤t}(1+r_i)；顶/底桶 NAV 分别用顶桶、底桶的前向收益序列，"
            "价差 NAV 用上述顶底多空序列。",
            f"- 最小有效样本数 min_periods={cfg.min_periods}，不足的 (标的,k) 已在结果中标记。",
        ]
        if cfg.run_backtest:
            lines += [
                "- 模拟交易（src/backtest 包）：按信号全样本分位数分桶（同上），"
                f"对 side=1..{cfg.n_buckets} 全部桶 × 做多/做空两个方向逐组合扫描"
                "（key 形如 side5_long=桶顶做多、side1_short=桶底做空）；"
                "t 日收盘触发 → t+1 开盘升一档并调仓至目标市值占比（3 档：20%/50%/100%），"
                "满仓再出信号则跳过、无信号则无动作；"
                f"首次建仓起满 k={cfg.backtest_k} 天于到期日收盘整仓平仓（加仓不重置计时）；"
                f"手续费单边 {cfg.backtest_fee}（双边收取）；初始资金 1.0（净值口径）；"
                "数据末端未平仓持仓按最后一日收盘价强制平仓。"
                "指标：总收益率/年化收益率（按 bars_per_year 复利年化）/"
                "最大回撤/夏普率（日收益 mean/std·√bars_per_year）/卡玛比率（年化/|回撤|）。",
            ]
        lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """导出完整 markdown 报告文本。"""
        parts = [self._header()]
        parts.append("## 跨标的汇总\n")
        parts.append(self._df_to_md(self.summary()))
        parts.append("")
        for sym, rep in self.symbols.items():
            parts.append(f"## 标的 {sym}")
            parts.append("")
            parts.append(
                f"样本区间: {rep.start} ~ {rep.end}，对齐后 bar 数: {rep.n_bars}"
            )
            if rep.warnings:
                parts.append("")
                parts.append("告警:")
                for w in rep.warnings:
                    parts.append(f"- {w}")
            parts.append("")
            parts.append("### Rank IC 与预测回归(全样本一次性，非滚动)")
            parts.append("")
            parts.append(self._df_to_md(rep.ic_table))
            parts.append("")
            parts.append("### 滚动 IC")
            parts.append("")
            for k in self.config.horizons:
                ro = rep.rolling.get(k)
                if ro is None:
                    continue
                parts.append(
                    f"k={ro.horizon}, W={ro.window}：IC 均值={ro.mean:.4f}，"
                    f"IC 标准差={ro.std:.4f}，ICIR={ro.icir:.4f}，"
                    f"年化 ICIR={ro.icir_annual:.4f}，IC>0 占比={ro.frac_pos:.4f}，"
                    f"NW t 值={ro.t_nw:.3f}（p={ro.p_nw:.4f}，L={ro.nw_lag}）"
                )
            parts.append("")
            parts.append("### 分层测试")
            for k in self.config.horizons:
                bres = rep.buckets.get(k)
                if bres is None:
                    continue
                parts.append("")
                parts.append(f"#### k = {k}")
                parts.append("")
                parts.append(self._df_to_md(bres.table))
                parts.append("")
                parts.append(
                    f"单调性: Spearman(桶序号, 桶均收益) = {bres.monotonic_rho:.4f}"
                    f"（p={bres.monotonic_p:.4f}）"
                )
                tb = bres.top_bottom
                parts.append(
                    f"顶底价差: 均值={tb['mean']:.6f}，NW t={tb['t']:.3f}"
                    f"（p={tb['p']:.4f}，L={tb['nw_lag']}），"
                    f"年化 Sharpe={tb['ann_sharpe']:.3f}，最大回撤={tb['maxdd']:.4f}"
                )
                le = bres.long_excess
                parts.append(
                    f"多头桶超额（相对全样本均值）: 均值={le['mean']:.6f}，"
                    f"NW t={le['t']:.3f}（p={le['p']:.4f}，L={le['nw_lag']}）"
                )
            if rep.backtests:
                parts.append("")
                parts.append(
                    f"### 模拟交易（分桶信号全桶多/空扫描，3 档仓位 20/50/100%，"
                    f"k={self.config.backtest_k}，费率 {self.config.backtest_fee} 双边）"
                )
                parts.append("")
                bt_df = pd.DataFrame(
                    [bt.metrics for bt in rep.backtests.values()],
                    index=list(rep.backtests.keys()),
                )
                bt_df = bt_df.rename(columns={
                    "total_ret": "总收益率", "ann_ret": "年化收益率",
                    "maxdd": "最大回撤", "sharpe": "夏普率",
                    "calmar": "卡玛比率", "n_trades": "交易周期数",
                    "n_skipped": "满仓跳过次数",
                })
                bt_df.index.name = "side"
                parts.append(self._df_to_md(bt_df))
            parts.append("")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 绘图（Agg 后端，图内文字用英文避免中文字体缺字形）
    # ------------------------------------------------------------------
    def plot(self, outdir: str = "reports/") -> list[str]:
        """保存全部图，返回生成的文件路径列表。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 方法内延迟导入 evaluator，避免与 evaluator → report 的模块级导入形成循环
        from .evaluator import _rolling_spearman, forward_returns

        os.makedirs(outdir, exist_ok=True)
        paths = []
        for sym, rep in self.symbols.items():
            safe = sym.replace(".", "_").replace("/", "_")

            # 1) IC 衰减曲线
            fig, ax = plt.subplots(figsize=(7, 4))
            ic = rep.ic_table
            ax.plot(ic.index, ic["rank_ic"], marker="o", label="RankIC")
            sig = ic["ic_pvalue"] < 0.05
            if sig.any():
                ax.plot(
                    ic.index[sig],
                    ic["rank_ic"][sig],
                    linestyle="none",
                    marker="*",
                    markersize=14,
                    color="red",
                    label="p<0.05",
                )
            ax.axhline(0.0, color="gray", linewidth=0.8)
            ax.set_xlabel("horizon k (bars)")
            ax.set_ylabel("Rank IC")
            ax.set_title(f"IC decay - {sym}")
            ax.legend()
            fig.tight_layout()
            p = os.path.join(outdir, f"ic_decay_{safe}.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)

            # 2) 滚动 IC：每个 horizon 一张图；首个子图为因子值随时间变化，
            #    其后从上往下按窗口 W 从小到大（10/20/30/50/100）各一个子图，
            #    时间轴对齐。多窗口序列由 plot_data 现场重算（与 evaluator
            #    同一口径），统计表中的 ICIR 等仍基于 config.rolling_window。
            for k in self.config.horizons:
                ro = rep.rolling.get(k)
                if ro is None:
                    continue
                factor_series = None
                if rep.plot_data is not None:
                    fwd = forward_returns(
                        rep.plot_data["open"], rep.plot_data["close"], k
                    )
                    pair = pd.concat(
                        [rep.plot_data["s"].rename("s"), fwd.rename("r")], axis=1
                    ).dropna()
                    series_by_w = {}
                    for w in _ROLLING_PLOT_WINDOWS:
                        vals = _rolling_spearman(pair["s"].values, pair["r"].values, w)
                        series_by_w[w] = pd.Series(vals, index=pair.index).dropna()
                    factor_series = rep.plot_data["s"].dropna()
                else:
                    series_by_w = {int(ro.window): ro.series}
                n_rows = len(series_by_w) + (1 if factor_series is not None else 0)
                fig, axes = plt.subplots(
                    n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True
                )
                if n_rows == 1:
                    axes = [axes]
                fig.suptitle(f"Rolling IC (k={k}) - {sym}")
                ic_axes = axes
                if factor_series is not None:
                    axes[0].plot(
                        factor_series.index, factor_series.values, linewidth=0.8
                    )
                    axes[0].set_ylabel("factor")
                    axes[0].set_title(f"Factor value - {sym}")
                    ic_axes = axes[1:]
                for ax, (w, ic_series) in zip(ic_axes, series_by_w.items()):
                    ax.plot(ic_series.index, ic_series.values, linewidth=0.8)
                    ax.axhline(0.0, color="gray", linewidth=0.8)
                    if len(ic_series) > 0:
                        mean = float(ic_series.mean())
                        ax.axhline(mean, color="red", linewidth=1.0)
                        ax.set_title(f"W = {w}, mean = {mean:.3f}")
                    else:
                        ax.set_title(f"W = {w} (no data)")
                    ax.set_ylabel("Rank IC")
                axes[-1].set_xlabel("time (window end)")
                fig.tight_layout(rect=(0, 0, 1, 0.98))
                p = os.path.join(outdir, f"rolling_ic_k{k}_{safe}.png")
                fig.savefig(p, dpi=120)
                plt.close(fig)
                paths.append(p)

            # 3) 因子与各 k 前向收益：合并为一张多子图（时间轴对齐，
            #    首个子图为因子值，其后按 k 从小到大每 k 一个子图）。
            #    前向收益使用与评价阶段相同的 next_open 口径。
            if rep.plot_data is not None:
                plot_data = rep.plot_data
                pairs = {}
                for k in sorted(self.config.horizons):
                    forward = (
                        plot_data["close"].shift(-(k + 1))
                        / plot_data["open"].shift(-1)
                        - 1.0
                    )
                    pair = pd.concat(
                        [plot_data["s"].rename("signal"), forward.rename("return")],
                        axis=1,
                    ).dropna()
                    if not pair.empty:
                        pairs[k] = pair
                signal = plot_data["s"].dropna()
                if pairs and not signal.empty:
                    n_rows = 1 + len(pairs)
                    fig, axes = plt.subplots(
                        n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True
                    )
                    axes[0].plot(signal.index, signal.values, linewidth=0.8)
                    axes[0].set_ylabel("factor")
                    axes[0].set_title(f"Factor and forward returns - {sym}")
                    for ax, k in zip(axes[1:], pairs):
                        ax.plot(
                            pairs[k].index,
                            pairs[k]["return"],
                            linewidth=0.8,
                            color="tab:orange",
                        )
                        ax.axhline(0.0, color="gray", linewidth=0.8)
                        ax.set_ylabel("forward return")
                        ax.set_title(f"k = {k}")
                    axes[-1].set_xlabel("time")
                    fig.tight_layout()
                    p = os.path.join(outdir, f"factor_return_{safe}.png")
                    fig.savefig(p, dpi=120)
                    plt.close(fig)
                    paths.append(p)

            # 4) 桶均收益柱状图：合并为一张多子图（横轴统一为桶编号，
            #    按 k 从小到大每 k 一个子图）
            ks_bucket = [
                k for k in sorted(self.config.horizons)
                if rep.buckets.get(k) is not None
            ]
            if ks_bucket:
                n_rows = len(ks_bucket)
                fig, axes = plt.subplots(
                    n_rows, 1, figsize=(7, 2.5 * n_rows), sharex=True
                )
                if n_rows == 1:
                    axes = [axes]
                for ax, k in zip(axes, ks_bucket):
                    bres = rep.buckets[k]
                    ax.bar(bres.table.index.astype(int), bres.table["mean"].values)
                    ax.axhline(0.0, color="gray", linewidth=0.8)
                    ax.set_ylabel("mean fwd ret")
                    ax.set_title(f"k = {k}")
                axes[-1].set_xlabel("bucket (1=lowest signal)")
                fig.tight_layout()
                p = os.path.join(outdir, f"buckets_{safe}.png")
                fig.savefig(p, dpi=120)
                plt.close(fig)
                paths.append(p)

            # 5) 顶/底桶累计净值：只画 k=1（其它 k 不绘制）
            k_nav = 1
            bres = rep.buckets.get(k_nav)
            if bres is not None:
                fig, ax = plt.subplots(figsize=(10, 4))
                for col, label in (("top", "top bucket"), ("bottom", "bottom bucket"),
                                   ("spread", "top-bottom")):
                    if col in bres.nav.columns:
                        ax.plot(bres.nav.index, bres.nav[col], linewidth=0.9, label=label)
                ax.set_xlabel("time")
                ax.set_ylabel("cumulative NAV")
                ax.set_title(f"Top/Bottom bucket NAV (k={k_nav}) - {sym}")
                ax.legend()
                fig.tight_layout()
                p = os.path.join(outdir, f"top_bottom_nav_k{k_nav}_{safe}.png")
                fig.savefig(p, dpi=120)
                plt.close(fig)
                paths.append(p)

            # 6) 模拟交易图（K 线/因子/持仓/总资产四子图），桶顶/桶底（key 形如
            #    side5_long/side1_long）各一张，文件名随 key：backtest_<key>_<标的>.png。
            #    延迟导入 backtest.plotting（其依赖 utils/matplotlib）。
            if rep.backtests:
                from backtest.plotting import plot_backtest

                for key, bt in rep.backtests.items():
                    p = os.path.join(outdir, f"backtest_{key}_{safe}.png")
                    if plot_backtest(bt, save_path=p, show=False):
                        paths.append(p)
        return paths
