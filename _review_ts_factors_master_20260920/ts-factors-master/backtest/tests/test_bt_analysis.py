"""backtest/bt_analysis.py 的单元测试。"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.bt_analysis import (
    analyze_and_plot_metrics,
    analyze_and_plot_rankic_vs_return,
    compute_histogram_stats,
    filter_high_performance_combos,
    run_high_performance_backtests,
)


def test_compute_histogram_stats():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.inf])
    stats, counts, bin_edges = compute_histogram_stats(series, bins=5)

    assert stats["count"] == 5
    assert pytest.approx(stats["mean"]) == 3.0
    assert pytest.approx(stats["median"]) == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert len(counts) == 5
    assert len(bin_edges) == 6


def test_analyze_and_plot_metrics_tmp(tmp_path: Path):
    # 创建临时的 csv 文件
    csv_file = tmp_path / "backtest_metrics_top20.csv"
    df = pd.DataFrame(
        {
            "ann_ret": [0.1, 0.2, -0.05, 0.15, np.nan],
            "maxdd": [-0.1, -0.2, -0.3, -0.15, -0.05],
            "sharpe": [1.0, 1.5, -0.5, 0.8, 0.2],
            "calmar": [0.5, 0.8, -0.2, 0.6, 0.1],
            "n_trades": [10, 20, 30, 40, 0],
            "flat_frac": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )
    df.to_csv(csv_file, index=False)

    save_path = analyze_and_plot_metrics(csv_path=csv_file, bins=5, save_filename="test_hist.png")

    assert save_path.exists()
    assert save_path.name == "test_hist.png"
    assert save_path.parent == tmp_path


def test_analyze_and_plot_rankic_vs_return_tmp(tmp_path: Path):
    metrics_csv = tmp_path / "backtest_metrics_top20.csv"
    ic_csv = tmp_path / "IC_top20_detail.csv"

    df_metrics = pd.DataFrame(
        {
            "factor_id": ["F1", "F1", "F2"],
            "stock": ["S1", "S2", "S1"],
            "combo": ["s5_long", "s5_long", "s5_long"],
            "k": [1, 5, 1],
            "ann_ret": [0.2, 0.15, -0.05],
            "sharpe": [1.2, 0.9, -0.3],
        }
    )
    df_metrics.to_csv(metrics_csv, index=False)

    df_ic = pd.DataFrame(
        {
            "factor_id": ["F1", "F1", "F2"],
            "stock": ["S1", "S2", "S1"],
            "RankIC(1)": [0.05, 0.01, -0.02],
            "RankIC(5)": [0.08, 0.04, -0.01],
            "ICIR(1)": [0.25, 0.10, -0.12],
            "ICIR(5)": [0.40, 0.20, -0.05],
        }
    )
    df_ic.to_csv(ic_csv, index=False)

    # 1. RankIC vs ann_ret 测试
    save_path_rankic = analyze_and_plot_rankic_vs_return(
        metrics_csv_path=metrics_csv,
        ic_csv_path=ic_csv,
        ic_metric="RankIC",
        bt_metric="ann_ret",
        save_filename="test_scatter_rankic.png",
    )

    assert save_path_rankic.exists()
    assert save_path_rankic.name == "test_scatter_rankic.png"
    assert save_path_rankic.parent == tmp_path

    # 2. ICIR vs ann_ret 测试
    save_path_icir = analyze_and_plot_rankic_vs_return(
        metrics_csv_path=metrics_csv,
        ic_csv_path=ic_csv,
        ic_metric="ICIR",
        bt_metric="ann_ret",
        save_filename="test_scatter_icir.png",
    )

    assert save_path_icir.exists()
    assert save_path_icir.name == "test_scatter_icir.png"
    assert save_path_icir.parent == tmp_path

    # 3. ICIR vs sharpe 测试
    save_path_icir_sharpe = analyze_and_plot_rankic_vs_return(
        metrics_csv_path=metrics_csv,
        ic_csv_path=ic_csv,
        ic_metric="ICIR",
        bt_metric="sharpe",
        save_filename="test_scatter_icir_sharpe.png",
    )

    assert save_path_icir_sharpe.exists()
    assert save_path_icir_sharpe.name == "test_scatter_icir_sharpe.png"
    assert save_path_icir_sharpe.parent == tmp_path


def test_filter_high_performance_combos_tmp(tmp_path: Path):
    metrics_csv = tmp_path / "backtest_metrics_top20.csv"
    df = pd.DataFrame(
        {
            "factor_id": ["F1", "F1", "F1", "F2", "F2", "F3"],
            "stock": ["S1", "S1", "S1", "S2", "S2", "S3"],
            "combo": ["s1_long", "s5_long", "buyhold", "s1_short", "s5_long", "s5_long"],
            "ann_ret": [0.35, 0.45, 0.40, 0.50, 0.10, 0.50],
            "sharpe": [1.5, 2.0, 0.8, 1.8, 0.5, 2.2],
            "calmar": [2.5, 3.0, 1.2, 2.1, 0.2, 2.5],
            "maxdd": [-0.15, -0.10, -0.25, -0.12, -0.30, -0.25], # F3 的 maxdd 为 -0.25，回撤25%将被排除
        }
    )
    df.to_csv(metrics_csv, index=False)

    save_path = filter_high_performance_combos(
        csv_path=metrics_csv,
        min_ann_ret=0.3,
        min_sharpe=1.0,
        min_calmar=2.0,
        max_maxdd=0.20,
        combos=("s1_long", "s5_long"),
        save_filename="filtered_top.csv",
    )

    assert save_path.exists()
    assert save_path.name == "filtered_top.csv"

    res_df = pd.read_csv(save_path)
    # s1_long (0.35) 不大于 buyhold (0.40)；F3 (maxdd=-0.25) > -0.20 不成立；仅 F1 的 s5_long 被保留
    assert len(res_df) == 1
    assert list(res_df["combo"]) == ["s5_long"]


def test_run_high_performance_backtests_tmp(tmp_path: Path, monkeypatch):
    combos_csv = tmp_path / "high_performance_combos.csv"
    df = pd.DataFrame(
        {
            "factor_id": ["Alpha001"],
            "stock": ["BJ920592"],
            "combo": ["s1_long"],
            "k": [2],
            "side": [1],
            "direction": ["long"],
        }
    )
    df.to_csv(combos_csv, index=False)

    out_dir = tmp_path / "high_performance_combos"

    res_dir = run_high_performance_backtests(
        combos_csv_path=combos_csv,
        output_dir=out_dir,
    )

    assert res_dir.exists()
    expected_png = out_dir / "backtest_Alpha001_BJ920592_side1_long_k2.png"
    expected_csv = out_dir / "backtest_Alpha001_BJ920592_side1_long_k2.csv"
    assert expected_png.exists()
    assert expected_csv.exists()


