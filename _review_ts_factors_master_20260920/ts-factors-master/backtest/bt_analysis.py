"""回测指标统计分析与相关性散点图绘制模块。

只读打开并分析回测指标文件 (backtest_metrics_top20.csv) 及 IC 明细文件 (IC_top20_detail.csv)：
1. 计算回测指标 (ann_ret, maxdd, sharpe, calmar, n_trades, flat_frac) 的分布直方图并绘图。
2. 提取每个因子*股票*k值组合的桶顶做多收益 (s5_long 的 ann_ret) 和对应 RankIC(k) 值，
   计算其 Pearson 与 Spearman 相关性并绘制散点图。

图像保存至 CSV 所在同一目录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保 src 目录在 sys.path 中，便于直接以脚本形式执行
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 配置绘图字体以正常显示中文及负号
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 默认 CSV 文件路径
DEFAULT_CSV_PATH = Path(r"D:\jzhang\data-kyts\reports\batch_top20_1\backtest_metrics_top20.csv")

# 待分析的指标及其显示名称
TARGET_METRICS = {
    "ann_ret": "年化收益率 (ann_ret)",
    "maxdd": "最大回撤 (maxdd)",
    "sharpe": "夏普比率 (sharpe)",
    "calmar": "卡玛比率 (calmar)",
    "n_trades": "交易次数 (n_trades)",
    "flat_frac": "空仓时间比例 (flat_frac)",
}


def compute_histogram_stats(
    series: pd.Series, bins: int = 30
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """计算单个指标序列的描述统计量及直方图区间频数。

    Args:
        series: 原始指标数据序列
        bins: 直方图分箱数量

    Returns:
        (stats_series, counts, bin_edges):
            - stats_series: 包含 count, mean, std, median, min, max, q25, q75 的统计序列
            - counts: 各 bin 的频数数组
            - bin_edges: 各 bin 的边界数组
    """
    # 过滤非有限值 (NaN, +inf, -inf)
    valid_data = series.replace([np.inf, -np.inf], np.nan).dropna()

    if valid_data.empty:
        stats = pd.Series(
            {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "q25": np.nan,
                "q75": np.nan,
            }
        )
        return stats, np.array([]), np.array([])

    stats = pd.Series(
        {
            "count": len(valid_data),
            "mean": valid_data.mean(),
            "std": valid_data.std(),
            "median": valid_data.median(),
            "min": valid_data.min(),
            "max": valid_data.max(),
            "q25": valid_data.quantile(0.25),
            "q75": valid_data.quantile(0.75),
        }
    )

    counts, bin_edges = np.histogram(valid_data, bins=bins)
    return stats, counts, bin_edges


def analyze_and_plot_metrics(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    bins: int = 30,
    save_filename: str = "metrics_histogram.png",
    show: bool = False,
) -> Path:
    """分析 CSV 文件中的指标并绘制分布直方图。

    Args:
        csv_path: 回测指标 CSV 文件路径 (只读模式打开)
        bins: 直方图分箱数 (默认 30)
        save_filename: 输出图片文件名 (保存至 csv_path 所在同一目录)
        show: 是否弹窗显示图片

    Returns:
        save_path: 图片保存的完整绝对路径
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到指定的 CSV 文件: {csv_path}")

    print(f"正在读取指标数据文件: {csv_path}")
    # 以只读方式打开并读取 CSV
    with open(csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        df = pd.read_csv(f)

    output_dir = csv_path.parent
    print(f"数据总行数: {len(df)}")
    print("=" * 60)
    print("各指标描述性统计及直方图计算结果:")
    print("=" * 60)

    # 创建 2x3 子图画布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    axes_flat = axes.flatten()

    for idx, (col, col_name) in enumerate(TARGET_METRICS.items()):
        ax = axes_flat[idx]

        if col not in df.columns:
            print(f"警告: 列 '{col}' 不在 CSV 文件中，跳过。")
            ax.text(0.5, 0.5, f"Column '{col}'\nNot Found", ha="center", va="center")
            continue

        series = df[col]
        valid_data = series.replace([np.inf, -np.inf], np.nan).dropna()
        stats, counts, bin_edges = compute_histogram_stats(series, bins=bins)

        # 打印控制台统计信息
        print(f"\n【指标: {col} ({col_name})】")
        print(
            f"  有效样本数: {int(stats['count'])} | 均值: {stats['mean']:.4f} | "
            f"标准差: {stats['std']:.4f} | 中位数: {stats['median']:.4f}"
        )
        print(
            f"  最小值: {stats['min']:.4f} | 25%分位数: {stats['q25']:.4f} | "
            f"75%分位数: {stats['q75']:.4f} | 最大值: {stats['max']:.4f}"
        )

        if not valid_data.empty:
            # 绘制频数直方图
            n, bins_out, patches = ax.hist(
                valid_data,
                bins=bins,
                color="#3498db",
                edgecolor="white",
                alpha=0.75,
                linewidth=0.8,
            )

            # 绘制均值与中位数指示线
            mean_val = stats["mean"]
            median_val = stats["median"]

            ax.axvline(
                mean_val,
                color="#e74c3c",
                linestyle="--",
                linewidth=1.5,
                label=f"均值 (Mean): {mean_val:.3f}",
            )
            ax.axvline(
                median_val,
                color="#2ecc71",
                linestyle=":",
                linewidth=1.8,
                label=f"中位数 (Median): {median_val:.3f}",
            )

            # 添加统计参数信息框
            textstr = (
                f"N = {int(stats['count'])}\n"
                f"Mean = {stats['mean']:.3f}\n"
                f"Std = {stats['std']:.3f}\n"
                f"Median = {stats['median']:.3f}\n"
                f"Min = {stats['min']:.3f}\n"
                f"Max = {stats['max']:.3f}"
            )
            ax.text(
                0.95,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
            )

            ax.set_title(col_name, fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel("频数 (Frequency)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    plt.suptitle(
        f"回测指标分布直方图 ({csv_path.name})",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    save_path = output_dir / save_filename
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"直方图分析完成！图形已成功保存至:\n  {save_path}")
    print("=" * 60)

    if show:
        plt.show()

    return save_path


def analyze_and_plot_rankic_vs_return(
    metrics_csv_path: str | Path = DEFAULT_CSV_PATH,
    ic_csv_path: str | Path | None = None,
    ic_metric: str = "RankIC",
    bt_metric: str = "ann_ret",
    combo: str = "s5_long",
    save_filename: str | None = None,
    show: bool = False,
) -> Path:
    """提取每个(因子*股票*k值)组合的桶顶做多回测指标(如 s5_long 的 ann_ret / sharpe)

    和指定 IC 指标(RankIC(k)或ICIR(k))，计算相关性并绘制散点图。

    Args:
        metrics_csv_path: 回测指标 CSV 文件路径
        ic_csv_path: IC 明细 CSV 文件路径，若为 None 则默认从 metrics_csv_path 所在目录同名逻辑查找
        ic_metric: 要分析的 IC 指标名称，"RankIC" 或 "ICIR" (默认 "RankIC")
        bt_metric: 要分析的回测绩效指标名称，如 "ann_ret", "sharpe", "calmar" (默认 "ann_ret")
        combo: 筛选的回测组合类型 (默认 "s5_long")
        save_filename: 输出散点图文件名 (若为 None 自动使用 {ic_metric.lower()}_vs_{bt_metric.lower()}_scatter.png)
        show: 是否弹窗显示图片

    Returns:
        save_path: 图片保存的完整绝对路径
    """
    metrics_csv_path = Path(metrics_csv_path).resolve()
    if not metrics_csv_path.is_file():
        raise FileNotFoundError(f"未找到指定的回测指标 CSV 文件: {metrics_csv_path}")

    output_dir = metrics_csv_path.parent

    if ic_csv_path is None:
        ic_csv_path = output_dir / "IC_top20_detail.csv"
    else:
        ic_csv_path = Path(ic_csv_path).resolve()

    if not ic_csv_path.is_file():
        raise FileNotFoundError(f"未找到指定的 IC 明细 CSV 文件: {ic_csv_path}")

    if save_filename is None:
        save_filename = f"{ic_metric.lower()}_vs_{bt_metric.lower()}_scatter.png"

    bt_metric_label = TARGET_METRICS.get(bt_metric, bt_metric)

    print(f"\n正在读取数据以分析 {ic_metric} 与 桶顶做多{bt_metric_label}({combo}) 相关性...")
    print(f"  回测指标文件: {metrics_csv_path}")
    print(f"  IC 明细文件:   {ic_csv_path}")

    # 1. 只读读取回测指标文件，过滤 combo
    with open(metrics_csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        df_metrics = pd.read_csv(f)

    if "combo" not in df_metrics.columns:
        raise ValueError("回测指标 CSV 中缺少 'combo' 列。")

    if bt_metric not in df_metrics.columns:
        raise ValueError(f"回测指标 CSV 中缺少 '{bt_metric}' 列。")

    df_combo = df_metrics[df_metrics["combo"] == combo].copy()
    if df_combo.empty:
        raise ValueError(f"回测指标 CSV 中未匹配到 combo == '{combo}' 的记录。")

    # 2. 只读读取 IC 明细文件，并转为 (factor_id, stock, k, {ic_metric}) 长表
    with open(ic_csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        df_ic = pd.read_csv(f)

    col_prefix = f"{ic_metric}("
    ic_records = []
    for col in df_ic.columns:
        if col.startswith(col_prefix):
            k_str = col.split("(")[1].rstrip(")")
            if k_str.isdigit():
                k_val = int(k_str)
                sub = df_ic[["factor_id", "stock", col]].copy()
                sub = sub.rename(columns={col: ic_metric})
                sub["k"] = k_val
                ic_records.append(sub)

    if not ic_records:
        raise ValueError(f"在 IC 明细 CSV 中未解析到 '{col_prefix}k)' 格式的列。")

    df_ic_long = pd.concat(ic_records, ignore_index=True)

    # 3. 按 (factor_id, stock, k) 匹配
    merged = pd.merge(
        df_combo[["factor_id", "stock", "k", bt_metric]],
        df_ic_long[["factor_id", "stock", "k", ic_metric]],
        on=["factor_id", "stock", "k"],
        how="inner",
    )

    # 清理非有限数值
    valid_data = merged.dropna(subset=[bt_metric, ic_metric]).copy()
    valid_data = valid_data[
        np.isfinite(valid_data[bt_metric]) & np.isfinite(valid_data[ic_metric])
    ]

    if valid_data.empty:
        raise ValueError(f"对齐后未获取到有效的 ({ic_metric}, {bt_metric}) 样本数据。")

    # 4. 计算相关性
    p_res = stats.pearsonr(valid_data[ic_metric], valid_data[bt_metric])
    s_res = stats.spearmanr(valid_data[ic_metric], valid_data[bt_metric])

    p_corr, p_pval = p_res.statistic, p_res.pvalue
    s_corr, s_pval = s_res.statistic, s_res.pvalue

    print("=" * 60)
    print(f"{ic_metric} 与 桶顶做多{bt_metric_label}({combo}) 相关性分析结果:")
    print("=" * 60)
    print(f"  匹配有效样本数 (N): {len(valid_data)}")
    print(f"  Pearson 相关系数:   {p_corr:.4f} (p-value: {p_pval:.4e})")
    print(f"  Spearman 秩相关系数: {s_corr:.4f} (p-value: {s_pval:.4e})")
    print("=" * 60)

    # 5. 绘制散点图
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    x = valid_data[ic_metric].values
    y = valid_data[bt_metric].values

    ax.scatter(x, y, alpha=0.35, s=16, color="#2980b9", edgecolors="none", label="组合样本点")

    # 拟合回归趋势线
    poly = np.polyfit(x, y, deg=1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(poly, x_line)
    ax.plot(
        x_line,
        y_line,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.8,
        label=f"趋势线 (y = {poly[0]:.3f}x + {poly[1]:.3f})",
    )

    textstr = (
        f"样本数 (N) = {len(valid_data)}\n"
        f"Pearson r = {p_corr:.4f} (p = {p_pval:.2e})\n"
        f"Spearman ρ = {s_corr:.4f} (p = {s_pval:.2e})"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    ax.set_xlabel(f"{ic_metric}(k)", fontsize=11)
    ax.set_ylabel(f"桶顶做多 {bt_metric_label} ({combo})", fontsize=11)
    ax.set_title(
        f"{ic_metric} 与 桶顶做多 {bt_metric_label} ({combo}) 散点图",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)

    plt.tight_layout()

    save_path = output_dir / save_filename
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"散点图成功保存至:\n  {save_path}")
    print("=" * 60)

    if show:
        plt.show()

    return save_path


def filter_high_performance_combos(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    min_ann_ret: float = 0.3,
    min_sharpe: float = 1.0,
    min_calmar: float = 2.0,
    max_maxdd: float = 0.20,
    combos: tuple[str, ...] | list[str] = ("s1_long", "s5_long"),
    save_filename: str = "high_performance_combos.csv",
) -> Path:
    """挑选满足特定条件的高绩效回测组合并保存至 CSV 文件。

    筛选条件：
    - combo 包含在 specified combos 中 (默认 's1_long' 或 's5_long')
    - ann_ret > min_ann_ret (默认 > 0.3)
    - sharpe > min_sharpe (默认 > 1.0)
    - calmar > min_calmar (默认 > 2.0)
    - 最大回撤幅度 < max_maxdd (默认 < 0.20，即 maxdd > -0.20)
    - 对于每个 (factor_id, stock) 组合，策略年化收益率 ann_ret 必须高于该组合买入持有策略 (combo='buyhold') 的年化收益率

    Args:
        csv_path: 回测指标 CSV 文件路径
        min_ann_ret: 最小年化收益率阈值
        min_sharpe: 最小夏普比率阈值
        min_calmar: 最小卡玛比率阈值
        max_maxdd: 最大允许的回撤幅度上限 (默认 0.20，即 20%)
        combos: 目标组合类型列表
        save_filename: 筛选结果输出文件名 (保存于 CSV 所在同级目录)

    Returns:
        save_path: 筛选 CSV 文件的完整保存路径
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到指定的回测指标 CSV 文件: {csv_path}")

    output_dir = csv_path.parent

    print(f"\n正在筛选高绩效组合数据...")
    print(f"  输入文件: {csv_path}")
    print(
        f"  筛选条件: combo ∈ {list(combos)}, "
        f"ann_ret > {min_ann_ret}, sharpe > {min_sharpe}, calmar > {min_calmar}, "
        f"maxdd > -{max_maxdd} (回撤<20%), 且 ann_ret > buyhold_ann_ret"
    )

    with open(csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        df = pd.read_csv(f)

    required_cols = {"factor_id", "stock", "combo", "ann_ret", "sharpe", "calmar", "maxdd"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"回测指标 CSV 中缺少列: {missing}")

    # 提取买入持有策略 (combo == 'buyhold') 的年化收益率作为对齐比较基准
    buyhold_df = df[df["combo"] == "buyhold"][["factor_id", "stock", "ann_ret"]].copy()
    buyhold_df = buyhold_df.rename(columns={"ann_ret": "buyhold_ann_ret"})

    if not buyhold_df.empty:
        merged_df = pd.merge(df, buyhold_df, on=["factor_id", "stock"], how="left")
        buyhold_cond = merged_df["buyhold_ann_ret"].isna() | (
            merged_df["ann_ret"] > merged_df["buyhold_ann_ret"]
        )
    else:
        merged_df = df.copy()
        buyhold_cond = pd.Series(True, index=df.index)

    cond = (
        merged_df["combo"].isin(combos)
        & (merged_df["ann_ret"] > min_ann_ret)
        & (merged_df["sharpe"] > min_sharpe)
        & (merged_df["calmar"] > min_calmar)
        & (merged_df["maxdd"] > -max_maxdd)
        & buyhold_cond
    )

    #filtered_df = merged_df[cond].drop(columns=["buyhold_ann_ret"], errors="ignore").copy()
    filtered_df = merged_df[cond].copy()

    save_path = output_dir / save_filename
    try:
        if save_path.exists():
            try:
                save_path.unlink()
            except OSError:
                pass
        filtered_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(f"警告: {save_path} 被外部程序占用，无法覆盖写入。将写入临时文件...")
        save_path = output_dir / f"new_{save_filename}"
        filtered_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"高绩效组合筛选完成！")
    print(f"  原始总行数:   {len(df)}")
    print(f"  筛选保留行数: {len(filtered_df)}")
    print(f"  结果已存至:   {save_path}")
    print("=" * 60)

    return save_path


def run_high_performance_backtests(
    combos_csv_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> Path:
    """读取高绩效组合 CSV 文件的每一行并执行模拟回测，保存资金曲线图和交易数据 CSV。

    Args:
        combos_csv_path: 高绩效组合 CSV 路径 (缺省从 DEFAULT_CSV_PATH 同级目录查找 high_performance_combos.csv)
        output_dir: 输出结果目录 (缺省为 CSV 所在目录下的 high_performance_combos 子目录)
        show: 是否弹窗显示产生的每个图形

    Returns:
        output_dir: 结果保存目录路径
    """
    from backtest.engine import BacktestConfig, run_backtest
    from backtest.plotting import plot_backtest
    from factors.library import FactorLibrary
    from factors.price_data import load_daily_bars

    if combos_csv_path is None:
        combos_csv_path = DEFAULT_CSV_PATH.parent / "high_performance_combos.csv"
    else:
        combos_csv_path = Path(combos_csv_path).resolve()

    if not combos_csv_path.is_file():
        raise FileNotFoundError(f"未找到高绩效组合 CSV 文件: {combos_csv_path}")

    if output_dir is None:
        output_dir = combos_csv_path.parent / "high_performance_combos"
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n正在对高绩效组合逐行运行模拟回测与图像/CSV保存...")
    print(f"  组合文件: {combos_csv_path}")
    print(f"  保存目录: {output_dir}")

    with open(combos_csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        df = pd.read_csv(f)

    if df.empty:
        print("高绩效组合文件为空，跳过回测。")
        return output_dir

    lib = FactorLibrary()
    prices_cache: dict[str, pd.DataFrame] = {}
    signal_cache: dict[tuple[str, str], pd.Series] = {}

    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        factor_id = str(row["factor_id"]).strip()
        stock = str(row["stock"]).strip()
        k_val = int(row["k"]) if "k" in row and pd.notna(row["k"]) else 5

        # 确定 side 与 direction
        if "side" in row and pd.notna(row["side"]):
            side_val = int(row["side"])
        else:
            side_val = int(str(row["combo"]).split("_")[0].replace("s", ""))

        if "direction" in row and pd.notna(row["direction"]):
            direction_val = str(row["direction"]).strip()
        else:
            direction_val = str(row["combo"]).split("_")[1].strip()

        # 加载或取缓存行情数据
        if stock not in prices_cache:
            prices_cache[stock] = load_daily_bars(stock)
        prices = prices_cache[stock]

        # 加载或取缓存信号数据
        key = (factor_id, stock)
        if key not in signal_cache:
            try:
                fid = lib._resolve_factor_id(factor_id)
                signal_cache[key] = lib.get(fid, stock)
            except Exception as exc:
                print(f"  [{idx+1}/{len(df)}] 错误: 因子/股票 {key} 加载失败 ({exc})")
                fail_count += 1
                continue

        signal = signal_cache[key]

        name = f"backtest_{factor_id}_{stock}_side{side_val}_{direction_val}_k{k_val}"
        png_path = output_dir / f"{name}.png"
        csv_path = output_dir / f"{name}.csv"

        try:
            cfg = BacktestConfig(
                n_buckets=5,
                side=side_val,
                direction=direction_val,
                k=k_val,
                fee_rate=5e-5,
            )
            res = run_backtest(signal, prices, cfg)
            res.label = f"{factor_id} {stock}"

            # 绘制四子图并存盘
            plot_backtest(res, save_path=png_path, show=show)
            if not show:
                plt.close("all")

            # 保存交易数据明细 CSV
            res.trades.to_csv(csv_path, index=False, encoding="utf-8-sig")

            success_count += 1
            if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
                print(f"  进度: [{idx+1}/{len(df)}] 已完成 {success_count} 组回测...")
        except Exception as exc:
            print(f"  [{idx+1}/{len(df)}] 回测执行失败 {name}: {exc}")
            fail_count += 1

    print("=" * 60)
    print(f"高绩效组合模拟回测批量执行完毕！")
    print(f"  成功: {success_count} 组 | 失败: {fail_count} 组")
    print(f"  图像与数据 CSV 存储目录: {output_dir}")
    print("=" * 60)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析回测指标 CSV 数据并绘制直方图与 RankIC vs 收益率散点图。"
    )
    parser.add_argument(
        "-f",
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"输入的回测指标 CSV 文件路径，默认: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--ic-csv",
        type=str,
        default=None,
        help="输入的 IC 明细 CSV 文件路径（默认在回测指标同目录下自动查找 IC_top20_detail.csv）",
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=30,
        help="直方图 Bin 数量，默认 30",
    )
    parser.add_argument(
        "-o",
        "--hist-output",
        type=str,
        default="metrics_histogram.png",
        help="直方图输出图片文件名（保存于 CSV 同级目录），默认 metrics_histogram.png",
    )
    parser.add_argument(
        "-s",
        "--scatter-output",
        type=str,
        default="rankic_vs_ann_ret_scatter.png",
        help="散点图输出图片文件名（保存于 CSV 同级目录），默认 rankic_vs_ann_ret_scatter.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否弹窗显示图形",
    )

    args = parser.parse_args()

    # 1. 指标直方图分析
    analyze_and_plot_metrics(
        csv_path=args.csv,
        bins=args.bins,
        save_filename=args.hist_output,
        show=args.show,
    )

    # 2. RankIC vs ann_ret 散点图与相关性分析
    analyze_and_plot_rankic_vs_return(
        metrics_csv_path=args.csv,
        ic_csv_path=args.ic_csv,
        ic_metric="RankIC",
        bt_metric="ann_ret",
        save_filename=args.scatter_output,
        show=args.show,
    )

    # 3. ICIR vs ann_ret 散点图与相关性分析
    icir_output = (
        "icir_vs_ann_ret_scatter.png"
        if args.scatter_output == "rankic_vs_ann_ret_scatter.png"
        else f"icir_{args.scatter_output}"
    )
    analyze_and_plot_rankic_vs_return(
        metrics_csv_path=args.csv,
        ic_csv_path=args.ic_csv,
        ic_metric="ICIR",
        bt_metric="ann_ret",
        save_filename=icir_output,
        show=args.show,
    )

    # 4. ICIR vs sharpe 散点图与相关性分析
    analyze_and_plot_rankic_vs_return(
        metrics_csv_path=args.csv,
        ic_csv_path=args.ic_csv,
        ic_metric="ICIR",
        bt_metric="sharpe",
        save_filename="icir_vs_sharpe_scatter.png",
        show=args.show,
    )

    # 5. 筛选高绩效组合 (s1_long/s5_long, ann_ret>0.3, sharpe>1, calmar>2, maxdd>-0.2) 并导出 CSV
    filter_high_performance_combos(
        csv_path=args.csv,
        min_ann_ret=0.3,
        min_sharpe=1.0,
        min_calmar=2.0,
        max_maxdd=0.20,
        combos=("s1_long", "s5_long"),
        save_filename="high_performance_combos.csv",
    )
    #raise Exception('debug stop here')

    # 6. 对筛选出的高绩效组合批量执行模拟回测，输出资金曲线图和数据明细 CSV
    run_high_performance_backtests(
        combos_csv_path=DEFAULT_CSV_PATH.parent / "high_performance_combos.csv",
        output_dir=DEFAULT_CSV_PATH.parent / "high_performance_combos",
        show=args.show,
    )


if __name__ == "__main__":
    main()
