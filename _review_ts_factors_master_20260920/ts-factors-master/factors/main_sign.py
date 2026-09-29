"""Rolling Rank IC 正负预测模型对比 CLI（面板热力图）。

用法（在 src 目录下运行）::

    python factors/main_sign.py Alpha098 BJ920725
    python -m factors sign Alpha098 BJ920725
    python -m factors sign Alpha098 BJ920725 --horizons 1,5,10 --hs 0,5,10,20

说明：
- 复用因子评价口径：evaluate_factor 对齐信号与行情后，由 plot_data 现场重算
  各 k×W 滚动 Rank IC（与滚动 IC 图、rolling_ic_*.csv 同一来源 rolling_ic_frame）；
- 对每条 (k, W) IC 序列 × h 评估 4 个符号预测模型（base/persist/ou/logit），
  信息可得性口径 gap = h+k+1（详见 factors/sign_forecast.py 模块 docstring）；
- 输出 <项目根>/reports/sign_<fid>_<stock>[_N]/（与 src 平级，自动加后缀避免覆盖）：
  sign_forecast_<fid>_<stock>.csv（k/W/h×4 模型命中率长表）、
  sign_heatmap_<fid>_<stock>.png（模型×k 面板热力图，横轴 h、纵轴 W）。
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# 允许以脚本方式直接运行（python factors/main_sign.py）
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from factors import EvalConfig, evaluate_factor  # noqa: E402
from factors.library import FactorLibrary  # noqa: E402
from factors.main_eval import OUTDIR, _make_unique_output_dir, load_factor_signal  # noqa: E402
from factors.price_data import load_daily_bars  # noqa: E402
from factors.report import rolling_ic_frame  # noqa: E402
from factors.sign_forecast import MODELS, evaluate_sign_grid, plot_sign_heatmap  # noqa: E402


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rolling Rank IC 正负预测模型对比（面板热力图）。")
    p.add_argument("factor", help="因子标识或因子名（如 Alpha098）")
    p.add_argument("stock", help="股票代码（如 BJ920725）")
    p.add_argument("--start", default=None, help="起始日期，如 2023-01-01 或 20230101")
    p.add_argument("--end", default=None, help="结束日期")
    p.add_argument("--horizons", default="1,2,3,5,10,15,20,30,50",
                   help="k 值列表（前向收益 horizon，交易日），逗号分隔")
    p.add_argument("--windows", default="10,20,30,50,100",
                   help="滚动窗口 W 列表，逗号分隔")
    p.add_argument("--hs", default="0,5,10,20",
                   help="预测未来第 h 天的 h 列表（0=预测最近可得 IC 值），逗号分隔")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="测试集占比（按时间先后切分，默认 0.2）")
    args = p.parse_args(argv)
    args.horizons = _parse_ints(args.horizons)
    args.windows = _parse_ints(args.windows)
    args.hs = _parse_ints(args.hs)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    lib = FactorLibrary()
    try:
        signal, fid = load_factor_signal(lib, args.factor, args.stock,
                                         args.start, args.end)
    except KeyError as exc:
        print(f"ERROR: 未知因子 {args.factor!r}，请用 python -m factors list 查看",
              file=sys.stderr)
        raise SystemExit(1) from exc
    prices = load_daily_bars(args.stock)

    cfg = EvalConfig(horizons=tuple(args.horizons), bars_per_year=252,
                     run_backtest=False)
    report = evaluate_factor(signal, prices, cfg)
    rep = report.symbols[fid]
    frame = rolling_ic_frame(rep, cfg.horizons, windows=tuple(args.windows))
    if frame is None:
        print("ERROR: 无 plot_data，无法重算滚动 IC", file=sys.stderr)
        raise SystemExit(1)

    res = evaluate_sign_grid(frame, hs=tuple(args.hs), test_frac=args.test_frac)
    if res.empty:
        print("ERROR: 样本过短，没有任何 (k, W, h) 组合可评估", file=sys.stderr)
        raise SystemExit(1)

    outdir = _make_unique_output_dir(OUTDIR, f"sign_{fid}_{args.stock}")
    csv_path = os.path.join(outdir, f"sign_forecast_{fid}_{args.stock}.csv")
    res.to_csv(csv_path, encoding="utf-8-sig", float_format="%.4f", index=False)
    png_path = plot_sign_heatmap(
        res, os.path.join(outdir, f"sign_heatmap_{fid}_{args.stock}.png"),
        title=f"Sign-prediction hit rate - {fid} {args.stock} "
              f"(center 0.50 = chance level)")

    print(f"{fid} {args.stock}: {rep.n_bars} bars, "
          f"{len(frame.columns) - 1} 条 (k,W) IC 序列 × {len(args.hs)} 个 h")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n=== 符号命中率（按 h 汇总，跨全部 (k,W) 平均） ===")
    print(res.groupby("h")[list(MODELS)].mean().to_string())
    if 0 in args.hs:
        print("\n=== h=0（预测最近可得 IC 值）按 k 分组 ===")
        print(res[res["h"] == 0].groupby("k")[list(MODELS)].mean().to_string())
    print(f"\n长表 CSV: {csv_path}")
    print(f"面板图: {png_path}")


if __name__ == "__main__":
    main()
