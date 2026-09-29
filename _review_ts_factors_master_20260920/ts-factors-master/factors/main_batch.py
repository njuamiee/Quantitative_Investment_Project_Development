r"""批量因子评价 + 模拟回测（并行版）：成交额 TopN 股票 × 全部（或指定）因子。

用法（在 src 目录下运行）::

    python factors/main_batch.py                                     # 101 因子 × top20
    python factors/main_batch.py --top 2 --factors Alpha001,Alpha098  # 冒烟测试
    python factors/main_batch.py --horizons 1,2,3,5,10,15,20 --njobs 8

说明：
- 股票名单来自 BJ_stocks_top100.csv（按成交额降序），取前 --top 名；
- 每个 (因子, 股票) 组合：IC 评价（evaluate_factor，run_backtest=False）+
  全分桶 × 多/空 × 全部 k（默认 1,2,3,5,10,15,20）的分桶模拟交易
  （5 桶 × 2 方向 × 7 k = 70 组合），外加买入持有基线（buy_and_hold_metrics）；
- 信号与行情对齐后样本 < min_bars（默认 250，约一年）的组合整体跳过；
- 并行：multiprocessing.Pool，默认核数 njobs=6 存于 src/config.json
  （[batch] njobs/min_bars/plot_sides/plot_ks），CLI 参数优先于配置文件；
- 断点续算：运行中每完成一个组合把结果追加写入 summary_partial_detail/long.csv，
  并在输出目录记录 run_params.json 参数快照；中断后重跑会自动检测最新未完成目录
  （--resume 直接续算不询问，否则提示确认），已完成组合跳过；参数与断点不一致时
  报错退出（不新开目录）。全部完成、最终总表写盘后 partial 文件自动删除；
- 输出（<项目根>/reports/batch_top<N>[_i]/，与 src 平级，utf-8-sig）：
    <fid>_<sid>/                    每股票×因子一个子目录：
        data_<fid>_<sid>.csv        date/open/high/low/close/volume/factor
                                    + 每组合 pos_s{side}_{dir}_k{k}/eq_... 两列 + eq_buyhold
        ic_decay.png / ic_decay.csv              IC 衰减图及其数据（行=k）
        bucket_mean_ret.png / .csv               分桶 mean fwd ret（不同 k 为子图）
                                                 及其数据（行=(k, bucket)）
        backtest_s*_k*.png          回测四子图（仅 plot_sides × 多空 × plot_ks 子集，
                                    默认 side∈{1,5}×k∈{1,5,20}=12 张）
    IC_top<N>_detail.csv            明细表（行=因子×股票）：RankIC(k)/IC_std(k)/ICIR(k)/
                                    n_bars/n_warnings/n_errors/error
    backtest_metrics_top<N>.csv     回测总表（行=因子×股票×k×桶×方向 + buyhold 基线行；
                                    列含 duration_years=总数据时长(年)、flat_frac=空仓比例）
    summary_by_factor.csv           按因子跨股票透视（选因子用，按 |mean RankIC(k0)| 降序）
    summary_by_stock.csv            按股票跨因子透视（选股用）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

# 允许以脚本方式直接运行（python factors/main_batch.py）
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from factors import EvalConfig, evaluate_factor  # noqa: E402
from factors.library import FactorLibrary  # noqa: E402
from factors.main_eval import OUTDIR, _make_unique_output_dir  # noqa: E402
from factors.price_data import load_daily_bars  # noqa: E402
from backtest.engine import (  # noqa: E402
    BacktestConfig,
    buy_and_hold_equity,
    buy_and_hold_metrics,
    run_backtest,
)

DEFAULT_LIST_FILE = os.path.join(_SRC_DIR, "BJ_stocks_top100.csv")
CONFIG_PATH = os.path.join(_SRC_DIR, "config.json")

_DEFAULT_BATCH_CONFIG = {
    "njobs": 6,
    "min_bars": 250,
    "plot_sides": [1, 5],
    "plot_ks": [1, 5, 20],
}

_BT_METRIC_KEYS = ("total_ret", "ann_ret", "maxdd", "sharpe", "calmar",
                   "n_trades", "n_skipped")


def _load_batch_config(path: str = CONFIG_PATH) -> dict:
    """读取全局配置 src/config.json 的 [batch] 节；缺失/损坏时回退默认值。"""
    cfg = dict(_DEFAULT_BATCH_CONFIG)
    try:
        with open(path, encoding="utf-8") as f:
            user = json.load(f).get("batch", {})
        for k, v in user.items():
            if k in cfg:
                cfg[k] = v
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


def load_stock_list(list_file: str, top: int) -> list[str]:
    """从成交额 top100 csv 取前 top 名，返回 ['BJ920725', ...]。"""
    df = pd.read_csv(list_file, encoding="utf-8-sig", dtype={"股票代码": str})
    codes = df["股票代码"].head(top).str.strip()
    return [f"BJ{c}" for c in codes]


def _clip_plot_subset(plot_sides, plot_ks, n_buckets: int, horizons) -> tuple:
    """出图子集裁剪：side 限制在 1..n_buckets（超出取 n_buckets），k 须在 horizons 中。"""
    sides = sorted({min(max(int(s), 1), n_buckets) for s in plot_sides})
    ks = [k for k in horizons if k in {int(x) for x in plot_ks}]
    return sides, ks


# ----------------------------------------------------------------------
# 断点续算
# ----------------------------------------------------------------------
_RUN_PARAMS_FILE = "run_params.json"


def _find_resume_dir(top: int, base_outdir: str = OUTDIR) -> str | None:
    """在 base_outdir 下找最新的未完成 batch_top<N>* 目录。

    目录内含 summary_partial_detail.csv 即视为未完成（该文件在全部跑完、
    最终总表写盘成功后被删除）。多个候选时按 partial 文件 mtime 取最新。
    """
    pat = re.compile(rf"batch_top{top}(_\d+)?$")
    try:
        names = os.listdir(base_outdir)
    except OSError:
        return None
    cands = []
    for name in names:
        d = os.path.join(base_outdir, name)
        partial = os.path.join(d, "summary_partial_detail.csv")
        if pat.fullmatch(name) and os.path.isdir(d) and os.path.exists(partial):
            cands.append(d)
    if not cands:
        return None
    return max(cands, key=lambda d: os.path.getmtime(
        os.path.join(d, "summary_partial_detail.csv")))


def _check_run_params(resume_dir: str, run_params: dict) -> None:
    """校验断点目录的运行参数与本次一致；不一致（或参数文件缺失）直接报错退出。"""
    path = os.path.join(resume_dir, _RUN_PARAMS_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"断点目录 {resume_dir} 的 {_RUN_PARAMS_FILE} 读取失败（{exc}），"
            "无法校验参数一致性，拒绝续算。")
    if old != run_params:
        keys = sorted(set(old) | set(run_params))
        diff = {k: {"旧": old.get(k), "新": run_params.get(k)}
                for k in keys if old.get(k) != run_params.get(k)}
        raise SystemExit(
            f"断点目录 {resume_dir} 的运行参数与本次不一致，拒绝续算（也不新开目录）。\n"
            f"差异: {json.dumps(diff, ensure_ascii=False, default=str)}\n"
            "如确需全新运行，请删除或移走该目录后重试。")


def _load_partial(resume_dir: str) -> tuple[list[dict], list[dict]]:
    """载入断点 partial 结果，返回 (detail_rows, long_rows)。"""
    det = pd.read_csv(os.path.join(resume_dir, "summary_partial_detail.csv"),
                      encoding="utf-8-sig")
    det["error"] = det["error"].fillna("")
    long_rows: list[dict] = []
    long_path = os.path.join(resume_dir, "summary_partial_long.csv")
    if os.path.exists(long_path):
        lon = pd.read_csv(long_path, encoding="utf-8-sig")
        lon["error"] = lon["error"].fillna("")
        long_rows = lon.to_dict("records")
    return det.to_dict("records"), long_rows


def _worker_init() -> None:
    """multiprocessing worker 初始化：matplotlib 切 Agg 后端（须在绘图前调用）。"""
    import matplotlib

    matplotlib.use("Agg")


# ----------------------------------------------------------------------
# 绘图（每股票×因子：IC 衰减 + 分桶 mean fwd ret 汇总图）
# ----------------------------------------------------------------------
def _plot_ic_decay(ic_table: pd.DataFrame, title: str, path: str) -> str:
    """IC 衰减曲线（rank_ic 对 k），p<0.05 的点加红星。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ic_table.index, ic_table["rank_ic"], marker="o", label="RankIC")
    sig = ic_table["ic_pvalue"] < 0.05
    if sig.any():
        ax.plot(ic_table.index[sig], ic_table["rank_ic"][sig], linestyle="none",
                marker="*", markersize=14, color="red", label="p<0.05")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("horizon k (bars)")
    ax.set_ylabel("Rank IC")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_bucket_mean_ret(buckets: dict, horizons, title: str, path: str) -> str | None:
    """分桶 mean fwd ret 汇总图：不同 k 各一个子图（横轴=桶号）。"""
    import matplotlib.pyplot as plt

    ks = [k for k in sorted(horizons) if buckets.get(k) is not None]
    if not ks:
        return None
    n_rows = len(ks)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        bres = buckets[k]
        ax.bar(bres.table.index.astype(int), bres.table["mean"].values)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_ylabel("mean fwd ret")
        ax.set_title(f"k = {k}")
    axes[-1].set_xlabel("bucket (1=lowest signal)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# worker：单个 (因子, 股票) 组合的评价 + 回测 + 子目录输出
# ----------------------------------------------------------------------
def _empty_detail(fid: str, sid: str, horizons) -> dict:
    row = {"factor_id": fid, "stock": sid, "n_bars": 0}
    for k in horizons:
        row[f"RankIC({k})"] = np.nan
        row[f"IC_std({k})"] = np.nan
        row[f"ICIR({k})"] = np.nan
    row.update(n_warnings=0, n_errors=0, error="")
    return row


def _empty_long_row(fid: str, sid: str, combo: str, k, side, direction) -> dict:
    return {"factor_id": fid, "stock": sid, "combo": combo, "k": k, "side": side,
            "direction": direction, "duration_years": np.nan, "flat_frac": np.nan,
            "total_ret": np.nan, "ann_ret": np.nan,
            "maxdd": np.nan, "sharpe": np.nan, "calmar": np.nan,
            "n_trades": 0, "n_skipped": 0, "n_warnings": 0, "error": ""}


def run_one(task: dict) -> tuple[dict, list[dict], str]:
    """单个 (因子, 股票) 组合：评价 + 70 组回测 + 基线，写子目录，返回汇总行。

    Args:
        task: {factor_id, stock, outdir, params}；params 含 horizons/n_buckets/fee/
            start/end/min_bars/plot_sides/plot_ks。

    Returns:
        (detail_row, long_rows, log_msg)：明细行、回测总表行列表、进度日志。
    """
    fid, sid = task["factor_id"], task["stock"]
    outdir, params = task["outdir"], task["params"]
    horizons = list(params["horizons"])
    n_buckets = params["n_buckets"]
    k0 = horizons[0]
    t0 = time.perf_counter()
    detail = _empty_detail(fid, sid, horizons)
    long_rows: list[dict] = []
    log = f"{fid} {sid}"

    try:
        prices = load_daily_bars(sid)
        s = FactorLibrary().get(fid, sid, start=params["start"], end=params["end"])
        px_cols = [c for c in ("open", "high", "low", "close", "volume")
                   if c in prices.columns]
        data = pd.concat([s.rename("factor"), prices[px_cols]], axis=1,
                         join="inner").sort_index()
        if data.index.has_duplicates:
            raise ValueError("对齐后索引存在重复时间戳")
        n = len(data)
        detail["n_bars"] = n
        if n < params["min_bars"]:
            detail["error"] = f"skipped: n_bars={n} < min_bars={params['min_bars']}"
            return detail, long_rows, f"{log} 跳过（样本 {n} < {params['min_bars']}）"

        # 总数据时长（年）：对齐样本首末日的日历年跨度；同一股票×因子的所有组合共用
        duration_years = (data.index[-1] - data.index[0]).days / 365.25

        px = data[[c for c in ("open", "high", "low", "close") if c in data.columns]]
        cfg = EvalConfig(horizons=tuple(horizons), n_buckets=n_buckets,
                         run_backtest=False, bars_per_year=252)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = evaluate_factor(data["factor"].rename(fid), px, cfg)
        n_warnings = len(caught)
        rep = report.symbols[fid]
        for k in horizons:
            if k in rep.ic_table.index:
                detail[f"RankIC({k})"] = rep.ic_table.loc[k, "rank_ic"]
            ro = rep.rolling.get(k)
            if ro is not None:
                detail[f"IC_std({k})"] = ro.std
                detail[f"ICIR({k})"] = ro.icir

        # ---- 全分桶 × 多/空 × 全 k 回测 ----
        bt_results: dict = {}
        n_errors = 0
        for k in horizons:
            for side in range(1, n_buckets + 1):
                for direction in ("long", "short"):
                    combo = f"s{side}_{direction}"
                    row = _empty_long_row(fid, sid, combo, k, side, direction)
                    try:
                        with warnings.catch_warnings(record=True) as cw:
                            warnings.simplefilter("always")
                            res = run_backtest(
                                data["factor"], px,
                                BacktestConfig(n_buckets=n_buckets, side=side,
                                               direction=direction, k=k,
                                               fee_rate=params["fee"],
                                               bars_per_year=252))
                        res.label = f"{fid} {sid}"
                        row.update(res.metrics)
                        row["n_warnings"] = len(cw)
                        row["duration_years"] = duration_years
                        # 空仓比例：仓位为 0 的 bar 占比（position=|持仓市值|/总资产）
                        row["flat_frac"] = float((res.position == 0.0).mean())
                        bt_results[(side, direction, k)] = res
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        n_errors += 1
                    long_rows.append(row)

        # ---- 买入持有基线（始终满仓，空仓比例 0） ----
        bh_row = _empty_long_row(fid, sid, "buyhold", None, None, "buyhold")
        bh_row["duration_years"] = duration_years
        bh_row["flat_frac"] = 0.0
        try:
            bh_row.update(buy_and_hold_metrics(px, bars_per_year=252))
        except Exception as exc:
            bh_row["error"] = f"{type(exc).__name__}: {exc}"
            n_errors += 1
        long_rows.append(bh_row)

        detail["n_warnings"] = n_warnings + sum(r["n_warnings"] for r in long_rows)
        detail["n_errors"] = n_errors

        # ---- 子目录输出：数据 CSV + 图（失败不影响已算出的指标） ----
        n_figs = 0
        try:
            subdir = os.path.join(outdir, f"{fid}_{sid}")
            os.makedirs(subdir, exist_ok=True)
            base_cols = [c for c in ("open", "high", "low", "close", "volume")
                         if c in data.columns] + ["factor"]
            out = data[base_cols].copy()
            extra = {}
            for (side, direction, k), res in bt_results.items():
                extra[f"pos_s{side}_{direction}_k{k}"] = res.position
                extra[f"eq_s{side}_{direction}_k{k}"] = res.equity
            extra["eq_buyhold"] = buy_and_hold_equity(px)
            out = pd.concat([out, pd.DataFrame(extra, index=data.index)], axis=1)
            out.index.name = "date"
            out.reset_index().to_csv(
                os.path.join(subdir, f"data_{fid}_{sid}.csv"),
                index=False, encoding="utf-8-sig")

            _plot_ic_decay(rep.ic_table, f"IC decay - {fid} {sid}",
                           os.path.join(subdir, "ic_decay.png"))
            n_figs += 1
            # ic_decay.png 的数据：行=k，列=rank_ic/ic_pvalue/beta/.../insufficient
            rep.ic_table.to_csv(os.path.join(subdir, "ic_decay.csv"),
                                encoding="utf-8-sig")
            if _plot_bucket_mean_ret(
                    rep.buckets, horizons,
                    f"Mean forward return by bucket - {fid} {sid}",
                    os.path.join(subdir, "bucket_mean_ret.png")):
                n_figs += 1
            # bucket_mean_ret.png 的数据：行=(k, bucket)，列=count/mean/ann_ret/sharpe
            bkt = [rep.buckets[k].table.assign(k=k)
                   for k in sorted(horizons) if rep.buckets.get(k) is not None]
            if bkt:
                pd.concat(bkt).reset_index().rename(columns={"index": "bucket"})[
                    ["k", "bucket", "count", "mean", "ann_ret", "sharpe"]
                ].to_csv(os.path.join(subdir, "bucket_mean_ret.csv"),
                         index=False, encoding="utf-8-sig")

            from backtest.plotting import plot_backtest

            sides, ks = _clip_plot_subset(params["plot_sides"], params["plot_ks"],
                                          n_buckets, horizons)
            for side in sides:
                for direction in ("long", "short"):
                    for k in ks:
                        res = bt_results.get((side, direction, k))
                        if res is None:
                            continue
                        p = os.path.join(
                            subdir, f"backtest_s{side}_{direction}_k{k}.png")
                        if plot_backtest(res, save_path=p, show=False):
                            n_figs += 1
        except Exception as exc:
            detail["n_errors"] += 1
            detail["error"] = f"输出失败: {type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - t0
        ric = detail[f"RankIC({k0})"]
        ric_s = f"{ric:.4f}" if pd.notna(ric) else "NaN"
        log = (f"{log} n_bars={n} RankIC({k0})={ric_s} "
               f"bt_err={n_errors} figs={n_figs} {elapsed:.1f}s")
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        log = f"{log} 失败（{exc}）"
    return detail, long_rows, log


# ----------------------------------------------------------------------
# 汇总透视
# ----------------------------------------------------------------------
def _pivot_by_factor(detail: pd.DataFrame, horizons) -> pd.DataFrame:
    """按因子跨股票透视：RankIC 各 k 的 mean/median/正值占比 + ICIR(k0) 均值。"""
    valid = detail[detail["error"] == ""]
    g = valid.groupby("factor_id")
    out = pd.DataFrame({"n_valid": g.size()})
    for k in horizons:
        col = f"RankIC({k})"
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_median"] = g[col].median()
        out[f"{col}_frac>0"] = g[col].apply(lambda s: (s > 0).mean())
    out["ICIR_mean"] = g[f"ICIR({horizons[0]})"].mean()
    rank_col = f"RankIC({horizons[0]})_mean"
    return out.reindex(out[rank_col].abs().sort_values(ascending=False).index)


def _pivot_by_stock(detail: pd.DataFrame, horizons) -> pd.DataFrame:
    """按股票跨因子透视：RankIC(k0)/ICIR(k0) 均值与 RankIC 正值占比。"""
    valid = detail[detail["error"] == ""]
    g = valid.groupby("stock")
    k0 = horizons[0]
    col = f"RankIC({k0})"
    out = pd.DataFrame({
        "n_valid": g.size(),
        f"{col}_mean": g[col].mean(),
        f"{col}_median": g[col].median(),
        f"{col}_frac>0": g[col].apply(lambda s: (s > 0).mean()),
        "ICIR_mean": g[f"ICIR({k0})"].mean(),
        "n_bars": g["n_bars"].first(),
    })
    return out.reindex(out[f"{col}_mean"].abs().sort_values(ascending=False).index)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_batch(args: argparse.Namespace) -> pd.DataFrame:
    """主流程：并行跑全部 (因子, 股票) 组合，返回明细 DataFrame。"""
    file_cfg = _load_batch_config()
    stock_list = load_stock_list(args.list_file, args.top)
    lib = FactorLibrary()
    factor_ids = lib.list_factors()["factor_id"].tolist()
    if args.factors:
        wanted = [f.strip() for f in args.factors.split(",") if f.strip()]
        unknown = [f for f in wanted if f not in factor_ids]
        if unknown:
            raise KeyError(f"未知因子: {unknown}，请用 python -m factors list 查看")
        factor_ids = wanted

    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    njobs = args.njobs or int(file_cfg["njobs"])
    params = {
        "horizons": horizons,
        "n_buckets": args.buckets,
        "fee": args.fee,
        "start": args.start,
        "end": args.end,
        "min_bars": args.min_bars or int(file_cfg["min_bars"]),
        "plot_sides": list(file_cfg["plot_sides"]),
        "plot_ks": list(file_cfg["plot_ks"]),
    }
    # 本次运行的完整参数快照（写入输出目录，供断点续算时校验一致性）
    run_params = {
        "top": args.top,
        "list_file": os.path.abspath(args.list_file),
        "factors": factor_ids,
        "stocks": stock_list,
        "params": params,
    }

    # ---- 断点续算：--resume 显式续算（不询问）；缺省自动检测并提示确认 ----
    detail_rows: list[dict] = []
    long_rows_all: list[dict] = []
    done = 0
    resume_dir = _find_resume_dir(args.top)
    outdir = None
    if args.resume:
        if resume_dir is None:
            print("未找到可续算的断点目录，从头开始。")
        else:
            _check_run_params(resume_dir, run_params)  # 不一致直接退出
            outdir = resume_dir
    elif resume_dir is not None:
        _check_run_params(resume_dir, run_params)      # 不一致直接退出
        try:
            ans = input(f"检测到未完成的批量运行目录 {resume_dir}\n"
                        "是否断点续算？[Y/n] ")
        except EOFError:
            raise SystemExit(
                "非交互环境无法确认续算；请显式加 --resume 续算，"
                "或删除/移走该断点目录后全新运行。")
        if ans.strip().lower() in ("", "y", "yes"):
            outdir = resume_dir
        else:
            print("已放弃续算，将新建目录从头开始。")

    if outdir is None:  # 全新运行：建目录并记录参数快照
        outdir = _make_unique_output_dir(OUTDIR, f"batch_top{args.top}")
        with open(os.path.join(outdir, _RUN_PARAMS_FILE), "w",
                  encoding="utf-8") as f:
            json.dump(run_params, f, ensure_ascii=False, indent=2)
    else:  # 续算：载入已完成结果，剩余组合继续
        detail_rows, long_rows_all = _load_partial(outdir)
        done = len(detail_rows)
        print(f"断点续算 {outdir}：已完成 {done} 个组合，结果已载入。")
    remaining = [(fid, sid) for fid in factor_ids for sid in stock_list]
    if done:
        done_set = {(r["factor_id"], r["stock"]) for r in detail_rows}
        remaining = [t for t in remaining if t not in done_set]
        print(f"剩余 {len(remaining)} 个组合待算。")
    tasks = [{"factor_id": fid, "stock": sid, "outdir": outdir, "params": params}
             for fid, sid in remaining]
    total = len(factor_ids) * len(stock_list)
    print(f"共 {len(factor_ids)} 因子 × {len(stock_list)} 股票 = {total} 个组合；"
          f"horizons={horizons}, buckets={args.buckets}, fee={args.fee}, "
          f"min_bars={params['min_bars']}, njobs={njobs}；输出目录 {outdir}",
          flush=True)

    partial_detail = os.path.join(outdir, "summary_partial_detail.csv")
    partial_long = os.path.join(outdir, "summary_partial_long.csv")
    def _consume(result: tuple[dict, list[dict], str]) -> None:
        nonlocal done
        done += 1
        drow, lrows, log = result
        detail_rows.append(drow)
        long_rows_all.extend(lrows)
        pd.DataFrame([drow]).to_csv(
            partial_detail, mode="a",
            header=not os.path.exists(partial_detail),
            index=False, encoding="utf-8-sig")
        if lrows:
            pd.DataFrame(lrows).to_csv(
                partial_long, mode="a",
                header=not os.path.exists(partial_long),
                index=False, encoding="utf-8-sig")
        print(f"[{done}/{total}] {log}", flush=True)

    if njobs <= 1:
        _worker_init()
        for t in tasks:
            _consume(run_one(t))
    else:
        import multiprocessing as mp

        with mp.Pool(processes=njobs, initializer=_worker_init) as pool:
            for result in pool.imap_unordered(run_one, tasks, chunksize=1):
                _consume(result)

    # ---- 明细总表（行=因子×股票，IC 指标 + 告警/错误计数） ----
    detail = pd.DataFrame(detail_rows)
    lead = ["factor_id", "stock", "n_bars"]
    rest = [c for c in detail.columns if c not in lead]
    detail = detail[lead + rest].sort_values(["factor_id", "stock"])
    detail_path = os.path.join(outdir, f"IC_top{args.top}_detail.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    # ---- 回测总表（行=因子×股票×k×桶×方向 + buyhold 基线行） ----
    long_df = pd.DataFrame(long_rows_all)
    long_path = os.path.join(outdir, f"backtest_metrics_top{args.top}.csv")
    if not long_df.empty:
        # buyhold 基线行的 k/side 为空，转可空 Int64 后才能与 int 混排
        long_df["k"] = long_df["k"].astype("Int64")
        long_df["side"] = long_df["side"].astype("Int64")
        long_df = long_df.sort_values(
            ["factor_id", "stock", "k", "side", "direction"], na_position="last")
        long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    # 最终总表写盘成功后，断点保存的 partial 文件即完成使命，删除避免重复输出；
    # 若中途崩溃则 partial 保留，可用于抢救已算完的组合
    for p in (partial_detail, partial_long):
        try:
            os.remove(p)
        except OSError:
            pass

    n_err = int((detail["error"] != "").sum())
    print(f"\n完成 {len(detail)} 个组合，其中失败/跳过 {n_err} 个。")
    if n_err:
        cols = ["factor_id", "stock", "error"]
        print(detail.loc[detail["error"] != "", cols].to_string(index=False))

    valid = detail[detail["error"] == ""]
    if not valid.empty:
        by_factor = _pivot_by_factor(detail, horizons)
        by_stock = _pivot_by_stock(detail, horizons)
        pf = os.path.join(outdir, "summary_by_factor.csv")
        ps = os.path.join(outdir, "summary_by_stock.csv")
        by_factor.to_csv(pf, encoding="utf-8-sig")
        by_stock.to_csv(ps, encoding="utf-8-sig")

        k0 = horizons[0]
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda v: f"{v:.4f}")
        print(f"\n===== 因子 top20（按 |mean RankIC({k0})|，跨 {len(stock_list)} 股） =====")
        print(by_factor.head(20).to_string())
        print(f"\n按股票透视: {ps}")
        print(f"按因子透视: {pf}")

    if not long_df.empty:
        ok = long_df[(long_df["error"] == "") & (long_df["combo"] != "buyhold")]
        if not ok.empty:
            print("\n===== 回测组合 top20（按年化收益） =====")
            cols = ["factor_id", "stock", "combo", "k", "total_ret", "ann_ret",
                    "maxdd", "sharpe", "calmar", "n_trades"]
            top_bt = ok.reindex(ok["ann_ret"].sort_values(ascending=False).index)
            print(top_bt[cols].head(20).to_string(index=False))
        print(f"\n回测总表: {long_path}")
    print(f"明细总表: {detail_path}")
    return detail


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="批量因子评价 + 模拟回测（并行）：成交额 TopN 股票 × 全部因子。"
    )
    p.add_argument("--top", type=int, default=20,
                   help="取成交额排名前 N 的股票（默认 20）")
    p.add_argument("--factors", default=None,
                   help="逗号分隔的因子子集（默认库中全部因子），如 Alpha001,Alpha098")
    p.add_argument("--horizons", default="1,2,3,5,10,15,20",
                   help="前向收益 horizon（同时是回测持仓天数 k）列表，逗号分隔"
                        "（默认 1,2,3,5,10,15,20）")
    p.add_argument("--start", default=None, help="起始日期，如 2023-01-01 或 20230101")
    p.add_argument("--end", default=None, help="结束日期")
    p.add_argument("--buckets", type=int, default=5, help="分桶数（默认 5）")
    p.add_argument("--fee", type=float, default=5e-5,
                   help="模拟交易单边费率（默认 5e-5，双边收取）")
    p.add_argument("--njobs", type=int, default=None,
                   help="并行进程数（默认取 src/config.json 的 njobs=6；1 表示串行）")
    p.add_argument("--min-bars", type=int, default=None,
                   help="因子×股票对齐后最小样本数，不足则跳过"
                        "（默认取 src/config.json 的 min_bars=250）")
    p.add_argument("--resume", action="store_true",
                   help="断点续算：直接续跑最新的未完成 batch_top<N>* 目录，不再询问。"
                        "缺省时检测到断点会提示确认；运行参数与断点不一致则报错退出")
    p.add_argument("--list-file", default=DEFAULT_LIST_FILE,
                   help="成交额排名 csv（默认 src/BJ_stocks_top100.csv）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_batch(_parse_args(argv))


if __name__ == "__main__":
    main()
