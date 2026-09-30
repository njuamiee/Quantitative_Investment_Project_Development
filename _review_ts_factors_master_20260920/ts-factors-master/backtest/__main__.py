"""分桶信号模拟交易 CLI。

用法（在 src 目录下运行）::

    python -m backtest 920725 Alpha098                                # 全扫描：1..buckets 桶 × 多/空
    python -m backtest BJ920725 Alpha098 --side 1 --k 10              # 单跑：桶底信号做多
    python -m backtest 920725 Alpha098 --side 5 --direction short     # 单跑：桶顶信号做空
    python -m backtest 920725 Alpha098 --buckets 10 --fee 5e-5
    python -m backtest 920725 Alpha098 --start 2023-01-01 --end 2025-12-31 --show

不指定 --side 且不指定 --direction 时进入全扫描模式：对 side=1..buckets 与
direction=long/short 的全部 2×buckets 个组合逐一出图存盘，并打印/保存汇总表
（summary.csv，按年化收益率降序）；只指定其一则单跑（side 缺省=顶桶，
direction 缺省=long）。

输出：控制台打印绩效指标；单跑模式图与 trades.csv 保存到
<项目根>/reports/backtest_<factor>_<stock>_side<side>_<direction>_k<k>[_N]/；
全扫描模式统一保存到 <项目根>/reports/backtest_<factor>_<stock>_all_k<k>[_N]/
（reports 与 src 平级；每组合一张 png + 一份 trades csv，另有 summary.csv；
均自动加后缀避免覆盖）。
"""

from __future__ import annotations

import argparse
import os
import sys

# 允许以脚本方式直接运行（python backtest/__main__.py）
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from backtest import normalize_stockid, run_factor_backtest  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(_SRC_DIR), "reports")  # 项目根/reports，与 src 平级


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
        description="分桶信号模拟交易：t 日收盘信号触发，t+1 开盘升档调仓，"
                    "首次建仓起满 k 天到期日收盘整仓平仓。"
                    "不指定 --side 与 --direction 时扫描全部桶 × 多/空。")
    parser.add_argument("stock", help="股票代码（如 920725 或 BJ920725）")
    parser.add_argument("factor", help="因子标识或因子名（如 Alpha098）")
    parser.add_argument("--side", type=int, default=None,
                        help="触发桶号 1..buckets（1=桶底；单跑缺省=顶桶 buckets；"
                             "与 --direction 都不给时扫描全部桶）")
    parser.add_argument("--direction", choices=("long", "short"), default=None,
                        help="long=做多；short=做空（单跑缺省=long；"
                             "与 --side 都不给时扫描多/空两个方向）")
    parser.add_argument("--buckets", type=int, default=5, help="分桶数（默认 5）")
    parser.add_argument("--k", type=int, default=5, help="持仓天数上限（默认 5）")
    parser.add_argument("--fee", type=float, default=5e-5,
                        help="单边费率（默认 5e-5，即万 0.5，双边收取）")
    parser.add_argument("--start", default=None, help="起始日期，如 2023-01-01 或 20230101")
    parser.add_argument("--end", default=None, help="结束日期")
    parser.add_argument("--show", action="store_true", help="弹窗显示图（缺省只保存文件）")
    return parser.parse_args(argv)


def _save_result(res, outdir: str, name: str, show: bool) -> str:
    """保存单个组合的四子图与 trades.csv，返回图路径。"""
    from backtest.plotting import plot_backtest

    png_path = plot_backtest(res, save_path=os.path.join(outdir, f"{name}.png"),
                             show=show)
    res.trades.to_csv(os.path.join(outdir, f"{name}_trades.csv"),
                      encoding="utf-8-sig", index=False)
    return png_path


def _print_detail(res) -> None:
    """单跑模式的详细指标打印。"""
    m = res.metrics
    cfg = res.config
    print("=" * 78)
    print(f"模拟交易: {res.label} | side={cfg.side}/{cfg.direction} "
          f"buckets={cfg.n_buckets} k={cfg.k} fee={cfg.fee_rate * 1e4:.1f}bp(双边)")
    print(f"样本区间: {res.data.index[0].date()} ~ {res.data.index[-1].date()}, "
          f"bar 数: {len(res.data)}")
    print("口径: t 日收盘信号触发, t+1 开盘升档(20%/50%/100% 目标市值占比), "
          "首次建仓起满 k 天到期日收盘整仓平仓")
    print("=" * 78)
    print(f"  总收益率:   {m['total_ret'] * 100:.2f}%")
    print(f"  年化收益率: {m['ann_ret'] * 100:.2f}%")
    print(f"  最大回撤:   {m['maxdd'] * 100:.2f}%")
    print(f"  夏普率:     {m['sharpe']:.3f}")
    print(f"  卡玛比率:   {m['calmar']:.3f}")
    print(f"  交易周期数: {m['n_trades']}（满仓信号跳过 {m['n_skipped']} 次，"
          f"强制平仓 {int(res.trades['forced'].sum()) if len(res.trades) else 0} 个周期）")


def _run_single(args: argparse.Namespace, stockid: str) -> int:
    """单跑模式：一个 (side, direction) 组合。"""
    res = run_factor_backtest(
        stockid, args.factor, side=args.side, direction=args.direction or "long",
        k=args.k, n_buckets=args.buckets, fee_rate=args.fee,
        start=args.start, end=args.end,
    )
    _print_detail(res)
    cfg = res.config
    name = (f"backtest_{args.factor}_{stockid}"
            f"_side{cfg.side}_{cfg.direction}_k{cfg.k}")
    outdir = _make_unique_output_dir(OUTDIR, name)
    png_path = _save_result(res, outdir, name, args.show)
    print(f"\n图已保存: {png_path}")
    print(f"交易明细: {os.path.join(outdir, name + '_trades.csv')}")
    return 0


def _run_sweep_all(args: argparse.Namespace, stockid: str) -> int:
    """全扫描模式：side=1..buckets × direction=long/short，逐一出图并汇总。"""
    import pandas as pd

    from backtest import BacktestConfig, run_backtest
    from factors.library import FactorLibrary
    from factors.price_data import load_daily_bars

    lib = FactorLibrary()
    fid = lib._resolve_factor_id(args.factor)  # 未知因子在此抛 KeyError
    s = lib.get(fid, stockid, start=args.start, end=args.end)
    prices = load_daily_bars(stockid)
    n_buckets = args.buckets

    name = f"backtest_{args.factor}_{stockid}_all_k{args.k}"
    outdir = _make_unique_output_dir(OUTDIR, name)
    print("=" * 78)
    print(f"全扫描: {fid} {stockid} | side=1..{n_buckets} × long/short "
          f"k={args.k} fee={args.fee * 1e4:.1f}bp(双边)")
    print(f"输出目录: {outdir}")
    print("=" * 78)

    rows: list[dict] = []
    for side in range(1, n_buckets + 1):
        for direction in ("long", "short"):
            tag = f"side{side}_{direction}"
            try:
                cfg = BacktestConfig(n_buckets=n_buckets, side=side,
                                     direction=direction, k=args.k,
                                     fee_rate=args.fee)
                res = run_backtest(s, prices, cfg)
                res.label = f"{fid} {stockid}"
                m = res.metrics
                rows.append({"side": side, "direction": direction, **m})
                print(f"  {tag:>13}: Ret={m['total_ret'] * 100:7.2f}% "
                      f"Ann={m['ann_ret'] * 100:7.2f}% "
                      f"MaxDD={m['maxdd'] * 100:7.2f}% "
                      f"Sharpe={m['sharpe']:6.3f} trades={m['n_trades']:3d} "
                      f"skip={m['n_skipped']:3d}", flush=True)
                sub = f"backtest_{args.factor}_{stockid}_{tag}_k{args.k}"
                _save_result(res, outdir, sub, args.show)
            except Exception as exc:  # 单组合失败不中断扫描
                rows.append({"side": side, "direction": direction,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"  {tag:>13}: 失败（{exc}）", flush=True)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(outdir, "summary.csv")
    summary.to_csv(summary_path, encoding="utf-8-sig", index=False)
    ok = summary[summary.get("error", pd.Series(dtype=object)).isna()] \
        if "error" in summary.columns else summary
    print("\n----- 汇总（按年化收益率降序） -----")
    if not ok.empty:
        ok = ok.sort_values("ann_ret", ascending=False)
        print(ok.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("（无成功组合）")
    print(f"\n汇总表: {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stockid = normalize_stockid(args.stock)
    try:
        if args.side is None and args.direction is None:
            return _run_sweep_all(args, stockid)
        return _run_single(args, stockid)
    except KeyError as exc:
        print(f"ERROR: 未知因子 {args.factor!r}，请用 python -m factors list 查看",
              file=sys.stderr)
        raise SystemExit(1) from exc
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
