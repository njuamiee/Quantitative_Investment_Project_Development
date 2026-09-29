r"""按因子标识评价因子库中的因子（IC 分析 CLI）。

用法（在 src 目录下运行）::

    python factors/main_eval.py Alpha001                     # 全部 BJ 股票汇总评价
    python factors/main_eval.py Alpha001 --stock BJ920010    # 单只股票详细评价 + 出图
    python factors/main_eval.py all --stock BJ920725         # 单只股票 × 全部因子
    python factors/main_eval.py Alpha001 --start 2023-01-01 --end 2025-12-31
    python factors/main_eval.py Alpha001 --horizons 1,5,10,20

说明：
- 因子数据取自 FactorLibrary（factor_lib/*.parquet），行情取 day-data/前复权 日线；
- 评价口径与 A 工程一致（时间序列评价，逐股票进行后汇总）：
  t 日收盘生成信号，t+1 开盘价进场，r(t→t+k) = Close[t+1+k]/Open[t+1] − 1；
- 多标的（默认）模式下逐股票评价并汇总，输出 summary CSV 与 markdown 报告，
  不为每只股票出图（图太多）；单股票模式输出全部图表与 markdown；
- 模拟交易（EvalConfig.run_backtest 默认开）对每个标的按 side=1..n_buckets ×
  多/空逐组合扫描：指标写 backtest_metrics_*.csv，单股票模式逐组合出四子图；
- factor 取 all 时（需配合 --stock）对单只股票遍历库中全部因子：每因子一个
  子目录（全部图表 + markdown + backtest metrics CSV），另输出跨因子汇总 summary CSV；
- 输出目录：<项目根>/reports/<factor_id>[_<stockid>]/ 或 all_<stockid>/（与 src 平级，
  自动加后缀避免覆盖）。
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd

# 允许以脚本方式直接运行（python factors/main_eval.py）
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from factors import EvalConfig, FactorReport, evaluate_factor  # noqa: E402
from factors.library import FactorLibrary  # noqa: E402
from factors.price_data import load_daily_bars, load_prices  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(_SRC_DIR), "reports")  # 项目根/reports，与 src 平级


def load_factor_signal(
    lib: FactorLibrary,
    factor: str,
    stock: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.Series | pd.DataFrame, str]:
    """从因子库取信号：单股票返回 Series，否则返回 date × stockid 宽表。

    返回 (signal, factor_id)；因子不存在时抛 KeyError。
    """
    fid = lib._resolve_factor_id(factor)  # 未知因子在此抛 KeyError
    if stock is not None:
        s = lib.get(fid, stock, start=start, end=end)
        return s.rename(fid), fid
    df = lib.get(fid, start=start, end=end)
    wide = df.pivot(index="date", columns="stockid", values="value").sort_index()
    return wide, fid


def _make_unique_output_dir(base_outdir: str, name: str) -> str:
    """在 base_outdir 下生成唯一的 name 子目录，避免覆盖上次输出。"""
    candidate = os.path.join(base_outdir, name)
    idx = 1
    while os.path.exists(candidate) and os.listdir(candidate):
        candidate = os.path.join(base_outdir, f"{name}_{idx}")
        idx += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate


def _eval_single(lib: FactorLibrary, args) -> None:
    """单只股票：详细打印 + 全部图表 + markdown（对应 A 工程 main_eval 的行为）。"""
    signal, fid = load_factor_signal(lib, args.factor, args.stock, args.start, args.end)
    prices = load_daily_bars(args.stock)
    cfg = EvalConfig(
        horizons=tuple(args.horizons),
        bars_per_year=252,  # 日线
    )
    report = evaluate_factor(signal, prices, cfg)
    rep = report.symbols[fid]

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("=" * 78)
    print(f"因子评价: {fid} 股票={args.stock}")
    print(f"样本区间: {rep.start} ~ {rep.end}, 对齐后 bar 数: {rep.n_bars}")
    print("口径: t 日收盘生成信号, t+1 开盘价进场, r(t->t+k)=Close[t+1+k]/Open[t+1]-1")
    print("=" * 78)

    print("\n[1] Rank IC 与预测回归（r/sigma = alpha + beta·s + eps, Newey-West HAC）")
    print(rep.ic_table.to_string())

    print(f"\n[2] 滚动 IC（W={cfg.rolling_window}，各 horizon 分别计算）")
    for k in cfg.horizons:
        ro = rep.rolling[k]
        print(f"  k={k}: IC 均值={ro.mean:.4f}  IC 标准差={ro.std:.4f}  ICIR={ro.icir:.4f}  "
              f"年化 ICIR={ro.icir_annual:.4f}  IC>0 占比={ro.frac_pos:.4f}  "
              f"NW t={ro.t_nw:.3f} (p={ro.p_nw:.4f}, L={ro.nw_lag})")

    print(f"\n[3] 分层测试（按信号全样本分位数分 {cfg.n_buckets} 桶）")
    for k in cfg.horizons:
        bres = rep.buckets.get(k)
        if bres is None:
            continue
        print(f"\n  -- k = {k} --")
        print(bres.table.to_string())
        print(f"  单调性 Spearman={bres.monotonic_rho:.4f} (p={bres.monotonic_p:.4f})")
        tb = bres.top_bottom
        print(f"  顶底价差: mean={tb['mean']:.6f}  NW t={tb['t']:.3f} (p={tb['p']:.4f})  "
              f"年化 Sharpe={tb['ann_sharpe']:.3f}  最大回撤={tb['maxdd']:.4f}")
        le = bres.long_excess
        print(f"  多头桶超额(相对全样本均值): mean={le['mean']:.6f}  "
              f"NW t={le['t']:.3f} (p={le['p']:.4f})")

    outdir = _make_unique_output_dir(OUTDIR, f"{fid}_{args.stock}")
    paths = report.plot(outdir)
    print(f"\n[4] 图已保存到 {outdir}（共 {len(paths)} 张）")

    md_path = os.path.join(outdir, f"factor_eval_{fid}_{args.stock}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    print(f"[5] markdown 报告: {md_path}")

    bt_df = report.backtest_metrics()
    if not bt_df.empty:
        bt_path = os.path.join(outdir, f"backtest_metrics_{fid}_{args.stock}.csv")
        bt_df.to_csv(bt_path, encoding="utf-8-sig", index=False)
        print(f"[6] 模拟交易指标 CSV: {bt_path}（{len(bt_df)} 个组合）")

    ric_paths = report.save_rolling_ic_csv(outdir)
    if ric_paths:
        print(f"[7] 滚动 IC 数值 CSV: {ric_paths[0]}")


def _eval_one_stock_all_factors(lib: FactorLibrary, args) -> None:
    """单只股票 × 全部因子：逐因子详细评价（每因子一个子目录，出图 + markdown），
    并输出跨因子汇总 summary CSV（行=factor_id）。单个因子失败不中断整批。"""
    stock = args.stock
    factor_ids = lib.list_factors()["factor_id"].tolist()
    prices = load_daily_bars(stock)
    cfg = EvalConfig(horizons=tuple(args.horizons), bars_per_year=252)
    outdir = _make_unique_output_dir(OUTDIR, f"all_{stock}")
    print(f"股票 {stock}: 共 {len(factor_ids)} 个因子待评价，输出目录 {outdir}")

    k0 = cfg.horizons[0]
    rows: dict[str, pd.Series] = {}
    skipped: list[tuple[str, str]] = []
    n_warn = 0
    for i, fid in enumerate(factor_ids):
        try:
            s = lib.get(fid, stock, start=args.start, end=args.end)
            if s.dropna().empty:
                raise ValueError("该股无有效因子数据")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = evaluate_factor(s.rename(fid), prices, cfg)
            n_warn += len(caught)
            rep = report.symbols[fid]
            rows[fid] = report.summary().iloc[0]

            sub = os.path.join(outdir, fid)
            os.makedirs(sub, exist_ok=True)
            report.plot(sub)
            with open(os.path.join(sub, f"factor_eval_{fid}_{stock}.md"),
                      "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            bt_df = report.backtest_metrics()
            if not bt_df.empty:
                bt_df.to_csv(os.path.join(sub, f"backtest_metrics_{fid}_{stock}.csv"),
                             encoding="utf-8-sig", index=False)
            report.save_rolling_ic_csv(sub)

            ro0 = rep.rolling[k0]
            print(f"[eval] {i + 1}/{len(factor_ids)} {fid}: "
                  f"RankIC(k={k0})={rep.ic_table.loc[k0, 'rank_ic']:.4f}, "
                  f"ICIR={ro0.icir:.4f}, n_bars={rep.n_bars}", flush=True)
        except Exception as exc:
            skipped.append((fid, f"{type(exc).__name__}: {exc}"))
            print(f"[eval] {i + 1}/{len(factor_ids)} {fid}: 跳过（{exc}）", flush=True)

    if not rows:
        print("没有任何因子评价成功。")
        return

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "factor_id"
    csv_path = os.path.join(outdir, f"summary_all_{stock}.csv")
    summary.to_csv(csv_path, encoding="utf-8-sig")

    rank_col = f"RankIC({k0})"
    ordered = summary.reindex(summary[rank_col].abs().sort_values(ascending=False).index)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n===== 跨因子汇总（按 |RankIC(k={k0})| 降序，前 20） =====")
    print(ordered.head(20).to_string())
    if n_warn:
        print(f"\n（评价过程中共 {n_warn} 条告警，多为短样本/降桶，详见各因子 markdown）")
    if skipped:
        print(f"跳过 {len(skipped)} 个因子：")
        for fid, reason in skipped:
            print(f"  - {fid}: {reason}")
    print(f"\nsummary CSV: {csv_path}")
    print(f"各因子图与 markdown: {outdir}/<factor_id>/")


def _eval_all_stocks(lib: FactorLibrary, args) -> None:
    """全部 BJ 股票：逐股票评价（带进度与告警计数），跨股票汇总，不出单股图。"""
    wide, fid = load_factor_signal(lib, args.factor, None, args.start, args.end)
    prices = load_prices(list(wide.columns))
    wide = wide[[c for c in wide.columns if c in prices]]
    print(f"因子 {fid}: {len(wide.columns)} 只股票, "
          f"{len(wide)} 个交易日 ({wide.index[0].date()} ~ {wide.index[-1].date()})")

    cfg = EvalConfig(horizons=tuple(args.horizons), bars_per_year=252)
    symbols = {}
    n_warn = 0
    for i, sid in enumerate(wide.columns):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rep_i = evaluate_factor(wide[[sid]], {sid: prices[sid]}, cfg)
        n_warn += len(caught)
        symbols[sid] = rep_i.symbols[sid]
        if (i + 1) % 50 == 0 or i + 1 == len(wide.columns):
            print(f"[eval] {i + 1}/{len(wide.columns)} 只股票完成", flush=True)
    report = FactorReport(cfg, symbols)

    summary = report.summary()
    ic_cols = [c for c in summary.columns if c.startswith("RankIC(")]
    print("\n===== 跨股票汇总（RankIC 均值/中位数/正值占比） =====")
    rows = {}
    for c in ic_cols:
        col = summary[c]
        rows[c] = {"mean": col.mean(), "median": col.median(), "frac>0": (col > 0).mean()}
    stat = pd.DataFrame(rows).T
    ro_cols = [c for c in summary.columns if c.startswith("RollingIC(")]
    for c in ro_cols:
        stat.loc[c.replace("RollingIC(", "RollingICMean(")] = {
            "mean": summary[c].mean(),
            "median": summary[c].median(),
            "frac>0": (summary[c] > 0).mean(),
        }
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(stat.to_string())
    print(f"\n主 horizon k={cfg.horizons[0]}: ICIR 均值={summary['ICIR'].mean():.4f}, "
          f"年化 ICIR 均值={summary['ICIR_annual'].mean():.4f}")
    if n_warn:
        print(f"（评价过程中共 {n_warn} 条告警，多为短样本股票，详见 markdown 报告）")

    outdir = _make_unique_output_dir(OUTDIR, fid)
    summary.to_csv(os.path.join(outdir, f"summary_{fid}.csv"), encoding="utf-8-sig")
    md_path = os.path.join(outdir, f"factor_eval_{fid}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    print(f"\nsummary CSV: {os.path.join(outdir, f'summary_{fid}.csv')}")
    print(f"markdown 报告: {md_path}")
    bt_df = report.backtest_metrics()
    if not bt_df.empty:
        bt_path = os.path.join(outdir, f"backtest_metrics_{fid}.csv")
        bt_df.to_csv(bt_path, encoding="utf-8-sig", index=False)
        print(f"模拟交易指标 CSV: {bt_path}（{len(bt_df)} 行 = 股票×组合）")
    print("（多标的模式不出单股图；如需图表请用 --stock 指定单只股票）")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按因子标识评价因子库中的因子（IC 分析）。")
    parser.add_argument("factor", help="因子标识或因子名（如 Alpha001）；取 all 表示全部因子（需配合 --stock）")
    parser.add_argument("--stock", default=None, help="股票代码（如 BJ920010）；缺省为全部 BJ 股票")
    parser.add_argument("--start", default=None, help="起始日期，如 2023-01-01 或 20230101")
    parser.add_argument("--end", default=None, help="结束日期")
    parser.add_argument("--horizons", default="1,2,3,5,10,15,20,30,50",
                        help="前向收益 horizon（交易日）列表，逗号分隔")
    args = parser.parse_args(argv)
    args.horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    lib = FactorLibrary()
    try:
        if str(args.factor).lower() == "all":
            if not args.stock:
                print("ERROR: factor=all 需配合 --stock 指定单只股票，"
                      "如 python -m factors eval all --stock BJ92   0725", file=sys.stderr)
                raise SystemExit(1)
            _eval_one_stock_all_factors(lib, args)
        elif args.stock:
            _eval_single(lib, args)
        else:
            _eval_all_stocks(lib, args)
    except KeyError as exc:
        print(f"ERROR: 未知因子 {args.factor!r}，请用 python -m factors list 查看", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
