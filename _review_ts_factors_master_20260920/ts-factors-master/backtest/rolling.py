r"""滚动回测：定期重训候选股票池 + 逐日模拟交易（只做多）。

两阶段：
- 阶段A（重训，多进程）：全局交易日历上每个调仓日 T（首个 T 为第 lookback 根 bar，
  之后每 rebalance 日一次），用截至 T 的 lookback 根 K 线窗口对全部
  股票×因子组合做 IC 分析（k=1,2,3,5,10,15，next_open 口径 RankIC）与窗口回测
  （side∈{1,5} 多头 × 全部 k），并计算买入持有基线；按
  bt_analysis.filter_high_performance_combos 的规则筛出候选池。
  每调仓日产出 4 个文件：bucket_<T>.json（桶顶/桶底边界 q_high/q_low）、
  backtest_metrics_<T>.csv、ic_detail_<T>.csv、high_performance_combos_<T>.csv。
- 阶段B（实盘模拟，单进程）：逐交易日循环。持仓满 k 日（按个股自身 K 线位置计数，
  停牌自然顺延）当日收盘平仓，回款加入现金；每日收盘用最近调仓日的池与桶边界
  检测信号（side=5 需因子值 ≥ q_high，side=1 需 ≤ q_low），触发组合若无在持/
  待开仓则记录信号，下一个该股有 K 线的交易日以开盘价开仓。

资金口径：初始资金固定 1000 万元（--init-capital），初始现金=初始资金；
第 t 日开盘对当日实际成交的买入信号按组合均分当前现金（每笔金额 = cash/n，
含费口径：现金支出恰好 cash/n，股数 = (cash/n)/(开盘价×(1+fee))）；现金耗尽
（cash≤0）时当日新信号全部跳过（n_skipped）；收盘平仓回款当日加入现金，
次日开盘可用。总资产 = 现金 + 持仓市值。结束后绘制 占用资金+现金/总资产 双子图
并打印总收益率/年化/最大回撤/Sharpe/卡玛。

用法（在 src 目录下运行）::

    python -m backtest.rolling                          # 全周期两阶段
    python -m backtest.rolling --top 2 --factors Alpha001,Alpha098 \
        --start 2023-01-01 --end 2023-09-30             # 冒烟
    python -m backtest.rolling --retrain-only           # 只跑重训
    python -m backtest.rolling --simulate-only          # 复用已有产物只跑实盘

输出目录：<项目根>/reports/rolling_backtest/。
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 允许以脚本方式直接运行（python backtest/rolling.py）
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from backtest.engine import (  # noqa: E402
    BacktestConfig,
    _calc_metrics,
    buy_and_hold_metrics,
    run_backtest,
)
from backtest.bt_analysis import filter_high_performance_combos  # noqa: E402
from factors.evaluator import forward_returns  # noqa: E402
from factors.main_batch import _load_batch_config, load_stock_list  # noqa: E402
from factors.main_eval import OUTDIR  # noqa: E402
from factors.price_data import load_daily_bars  # noqa: E402

DEFAULT_OUTDIR = str(Path(OUTDIR) / "rolling_backtest")

_METRIC_COLS = ["factor_id", "stock", "combo", "k", "side", "direction",
                "duration_years", "flat_frac", "total_ret", "ann_ret", "maxdd",
                "sharpe", "calmar", "n_trades", "n_skipped", "n_warnings", "error"]

_TRADE_COLS = ["stock", "factor", "k", "side", "entry_date", "entry_px",
               "exit_date", "exit_px", "shares", "entry_value", "gross_ret",
               "net_ret", "fee_paid", "forced"]


@dataclass
class RollingConfig:
    """滚动回测配置。

    Attributes:
        top: 股票池范围——成交额排名 csv 前 N 名。
        list_file: 成交额排名 csv 路径。
        factors: 因子子集（None=库中全部因子）。
        lookback: 回看窗口长度 N（交易日 bar 数，默认 63 ≈ 3 个月）。
        rebalance: 股票池更新间隔 M（交易日，默认 5）。
        ks: 前向收益 horizon（同时是回测/实盘的持仓天数）列表。
        n_buckets: 分桶数（默认 5；side=1 桶底 / side=n_buckets 桶顶）。
        sides: 参与回测与实盘的桶号（默认 (1, 5)，均做多）。
        init_capital: 初始资金（默认 1000 万元；初始现金=初始资金）。
        fee_rate: 单边费率（双边收取）。
        min_window_bars: 窗口内因子×行情对齐样本下限，不足跳过该组合。
        max_entry_wait: 信号后等待开仓 bar 的最大全局交易日数，超时放弃。
        bars_per_year: 年化用每年 bar 数（日线 252）。
        start, end: 回测区间（调仓日与实盘日均限制在此区间）。
        njobs: 重训并行进程数（默认取 config.json [batch].njobs）。
        outdir: 输出目录。
    """

    top: int = 20
    list_file: str = str(Path(_SRC_DIR) / "BJ_stocks_top100.csv")
    factors: list[str] | None = None
    lookback: int = 63
    rebalance: int = 5
    ks: tuple[int, ...] = (1, 2, 3, 5, 10, 15)
    n_buckets: int = 5
    sides: tuple[int, ...] = (1, 5)
    init_capital: float = 10_000_000.0
    fee_rate: float = 5e-5
    min_window_bars: int = 40
    max_entry_wait: int = 5
    bars_per_year: int = 252
    start: str | None = None
    end: str | None = None
    njobs: int = 6
    outdir: str = DEFAULT_OUTDIR

    def __post_init__(self):
        if self.lookback < 2:
            raise ValueError("lookback 至少为 2")
        if self.rebalance < 1:
            raise ValueError("rebalance 至少为 1")
        if not self.ks:
            raise ValueError("ks 不能为空")
        for s in self.sides:
            if not (1 <= s <= self.n_buckets):
                raise ValueError(f"sides 须在 1..{self.n_buckets} 之间，收到 {s}")
        if self.init_capital <= 0:
            raise ValueError("init_capital 必须为正")


# ----------------------------------------------------------------------
# 阶段A：调仓日重训
# ----------------------------------------------------------------------
def _empty_metric_row(fid: str, sid: str, combo: str, k, side, direction) -> dict:
    return {"factor_id": fid, "stock": sid, "combo": combo, "k": k, "side": side,
            "direction": direction, "duration_years": np.nan, "flat_frac": np.nan,
            "total_ret": np.nan, "ann_ret": np.nan, "maxdd": np.nan,
            "sharpe": np.nan, "calmar": np.nan, "n_trades": 0, "n_skipped": 0,
            "n_warnings": 0, "error": ""}


def _retrain_one(fid: str, sid: str, s: pd.Series | None, px: pd.DataFrame | None,
                 cfg: RollingConfig) -> dict:
    """单个 (因子, 股票) 在一个调仓窗口内的 IC 分析 + 回测 + 桶边界。

    Args:
        s: 窗口内（含调仓日 T 共 lookback 根 bar）的因子序列；None 表示无数据。
        px: 窗口内行情（含 open/close 列）；None 表示无数据。

    Returns:
        {"ic_row": dict, "metric_rows": list[dict], "boundary": dict | None}
    """
    ks = list(cfg.ks)
    ic_row = {"factor_id": fid, "stock": sid, "n_bars": 0, "error": ""}
    for k in ks:
        ic_row[f"RankIC({k})"] = np.nan
    metric_rows: list[dict] = []
    boundary = None

    if s is None or px is None or px.empty:
        ic_row["error"] = "no data"
        return {"ic_row": ic_row, "metric_rows": metric_rows, "boundary": boundary}

    data = pd.concat([s.rename("factor"), px[["open", "close"]]], axis=1,
                     join="inner").sort_index()
    n = len(data)
    ic_row["n_bars"] = n
    if n < cfg.min_window_bars:
        ic_row["error"] = f"skipped: n_bars={n} < min_window_bars={cfg.min_window_bars}"
        return {"ic_row": ic_row, "metric_rows": metric_rows, "boundary": boundary}

    # ---- IC 分析：RankIC(k)，next_open 口径 ----
    for k in ks:
        fr = forward_returns(data["open"], data["close"], k)
        valid = pd.concat([data["factor"], fr.rename("fr")], axis=1).dropna()
        if len(valid) >= 20:
            ic_row[f"RankIC({k})"] = float(
                spearmanr(valid["factor"], valid["fr"]).statistic)

    # ---- 桶边界（qcut 5 桶的 side=1 ⟺ s≤q(1/N)、side=N ⟺ s≥q((N-1)/N)） ----
    s_valid = data["factor"].dropna()
    q_low = float(s_valid.quantile(1.0 / cfg.n_buckets))
    q_high = float(s_valid.quantile((cfg.n_buckets - 1) / cfg.n_buckets))
    boundary = {"q_low": q_low, "q_high": q_high, "n_unique": int(s_valid.nunique())}

    # ---- 窗口回测：side ∈ sides × 全 k，只做多 ----
    duration_years = (data.index[-1] - data.index[0]).days / 365.25
    for k in ks:
        for side in cfg.sides:
            combo = f"s{side}_long"
            row = _empty_metric_row(fid, sid, combo, k, side, "long")
            try:
                with warnings.catch_warnings(record=True) as cw:
                    warnings.simplefilter("always")
                    res = run_backtest(
                        data["factor"], data[["open", "close"]],
                        BacktestConfig(n_buckets=cfg.n_buckets, side=side,
                                       direction="long", k=k,
                                       fee_rate=cfg.fee_rate,
                                       bars_per_year=cfg.bars_per_year))
                row.update(res.metrics)
                row["n_warnings"] = len(cw)
                row["duration_years"] = duration_years
                row["flat_frac"] = float((res.position == 0.0).mean())
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            metric_rows.append(row)

    # ---- 买入持有基线（每 factor×stock 一行，与 batch 口径一致） ----
    bh_row = _empty_metric_row(fid, sid, "buyhold", None, None, "buyhold")
    bh_row["duration_years"] = duration_years
    bh_row["flat_frac"] = 0.0
    try:
        bh_row.update(buy_and_hold_metrics(data[["open", "close"]],
                                           bars_per_year=cfg.bars_per_year))
    except Exception as exc:
        bh_row["error"] = f"{type(exc).__name__}: {exc}"
    metric_rows.append(bh_row)

    return {"ic_row": ic_row, "metric_rows": metric_rows, "boundary": boundary}


# ---- 多进程 worker（全局缓存行情与因子数据，避免重复读盘） ----
_W_PRICES: dict = {}
_W_FACTORS: dict = {}
_W_LIB = None


def _worker_init(stocks: list[str]) -> None:
    """multiprocessing worker 初始化：预载全部股票行情。"""
    global _W_PRICES
    for sid in stocks:
        try:
            _W_PRICES[sid] = load_daily_bars(sid)
        except Exception:
            pass


def _get_factor_map(fid: str, stocks: list[str]) -> dict:
    """读取某因子的长表并切分为 {stockid: Series}（worker 内缓存，每因子只读一次）。"""
    global _W_LIB
    if fid not in _W_FACTORS:
        if _W_LIB is None:
            from factors.library import FactorLibrary
            _W_LIB = FactorLibrary()
        df = _W_LIB.get(fid)
        df = df[df["stockid"].isin(stocks)]
        _W_FACTORS[fid] = {sid: g.set_index("date")["value"].sort_index()
                           for sid, g in df.groupby("stockid")}
    return _W_FACTORS[fid]


def _retrain_factor_day(task: dict) -> dict:
    """worker 任务：一个因子 × 全部股票在调仓日 T 的窗口重训。"""
    fid = task["factor_id"]
    stocks = task["stocks"]
    win_start, win_end = task["win_start"], task["win_end"]
    cfg: RollingConfig = task["cfg"]
    fmap = _get_factor_map(fid, stocks)
    ic_rows, metric_rows, boundaries = [], [], {}
    for sid in stocks:
        s = fmap.get(sid)
        s_win = s.loc[win_start:win_end] if s is not None else None
        px = _W_PRICES.get(sid)
        px_win = px.loc[win_start:win_end] if px is not None else None
        r = _retrain_one(fid, sid, s_win, px_win, cfg)
        ic_rows.append(r["ic_row"])
        metric_rows.extend(r["metric_rows"])
        if r["boundary"] is not None:
            boundaries[sid] = r["boundary"]
    return {"factor_id": fid, "ic_rows": ic_rows,
            "metric_rows": metric_rows, "boundaries": boundaries}


def _rebalance_indices(calendar: pd.DatetimeIndex, cfg: RollingConfig) -> list[int]:
    """调仓日在全局日历上的索引：首个为第 lookback 根 bar，之后每隔 rebalance 根。"""
    idx = list(range(cfg.lookback - 1, len(calendar), cfg.rebalance))
    if cfg.start is not None:
        start = pd.Timestamp(cfg.start)
        idx = [i for i in idx if calendar[i] >= start]
    if cfg.end is not None:
        end = pd.Timestamp(cfg.end)
        idx = [i for i in idx if calendar[i] <= end]
    return idx


def _write_day_outputs(outdir: Path, T: pd.Timestamp, win_start: pd.Timestamp,
                       cfg: RollingConfig, ic_rows: list[dict],
                       metric_rows: list[dict], boundaries: dict) -> Path:
    """写出一个调仓日的 4 类产物，返回候选池 CSV 路径。"""
    dstr = T.strftime("%Y%m%d")

    ic_df = pd.DataFrame(ic_rows)
    lead = ["factor_id", "stock", "n_bars"]
    ic_df = ic_df[lead + [c for c in ic_df.columns if c not in lead]]
    ic_df = ic_df.sort_values(["factor_id", "stock"])
    ic_df.to_csv(outdir / f"ic_detail_{dstr}.csv", index=False, encoding="utf-8-sig")

    m_df = pd.DataFrame(metric_rows, columns=_METRIC_COLS)
    if not m_df.empty:
        m_df["k"] = m_df["k"].astype("Int64")
        m_df["side"] = m_df["side"].astype("Int64")
        m_df = m_df.sort_values(["factor_id", "stock", "k", "side"],
                                na_position="last")
    metrics_path = outdir / f"backtest_metrics_{dstr}.csv"
    m_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    bucket = {"date": str(T.date()),
              "window": [str(win_start.date()), str(T.date())],
              "lookback": cfg.lookback, "n_buckets": cfg.n_buckets,
              "ks": list(cfg.ks), "sides": list(cfg.sides),
              "boundaries": boundaries}
    with open(outdir / f"bucket_{dstr}.json", "w", encoding="utf-8") as f:
        json.dump(bucket, f, ensure_ascii=False, indent=1)

    # 直接复用 bt_analysis 的筛选规则产出候选股票池
    pool_path = filter_high_performance_combos(
        csv_path=metrics_path,
        save_filename=f"high_performance_combos_{dstr}.csv")
    return pool_path


def run_retrain(cfg: RollingConfig, stocks: list[str], calendar: pd.DatetimeIndex,
                factor_ids: list[str]) -> None:
    """阶段A 主流程：逐调仓日并行重训并写盘。"""
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    idx_list = _rebalance_indices(calendar, cfg)
    if not idx_list:
        raise SystemExit("回测区间内没有可调仓的日期（数据不足一个 lookback 窗口）。")
    print(f"[retrain] 共 {len(idx_list)} 个调仓日，"
          f"{len(factor_ids)} 因子 × {len(stocks)} 股票/日，njobs={cfg.njobs}",
          flush=True)

    pool = None
    if cfg.njobs > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=cfg.njobs, initializer=_worker_init,
                       initargs=(stocks,))
    else:
        _worker_init(stocks)
    try:
        for di, i in enumerate(idx_list):
            T = calendar[i]
            win_start = calendar[i - cfg.lookback + 1]
            tasks = [{"factor_id": fid, "stocks": stocks,
                      "win_start": win_start, "win_end": T, "cfg": cfg}
                     for fid in factor_ids]
            if pool is None:
                results = [_retrain_factor_day(t) for t in tasks]
            else:
                results = pool.map(_retrain_factor_day, tasks, chunksize=4)

            ic_rows: list[dict] = []
            metric_rows: list[dict] = []
            boundaries: dict = {}
            for r in results:
                ic_rows.extend(r["ic_rows"])
                metric_rows.extend(r["metric_rows"])
                if r["boundaries"]:
                    boundaries[r["factor_id"]] = r["boundaries"]
            pool_path = _write_day_outputs(outdir, T, win_start, cfg,
                                           ic_rows, metric_rows, boundaries)
            with open(pool_path, encoding="utf-8-sig") as f:
                n_pool = max(sum(1 for _ in f) - 1, 0)
            print(f"[retrain {di + 1}/{len(idx_list)}] {T.date()} "
                  f"窗口 {win_start.date()}~{T.date()} 候选池 {n_pool} 行",
                  flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


# ----------------------------------------------------------------------
# 阶段B：实盘逐日模拟
# ----------------------------------------------------------------------
def _load_rebal_artifacts(outdir: Path, cfg: RollingConfig) -> list[dict]:
    """读取各调仓日的候选池 CSV 与桶边界 json，按日期升序返回。"""
    rebal = []
    for jp in sorted(outdir.glob("bucket_*.json")):
        dstr = jp.stem.split("_")[1]
        T = pd.Timestamp(dstr)
        if cfg.start is not None and T < pd.Timestamp(cfg.start):
            continue
        if cfg.end is not None and T > pd.Timestamp(cfg.end):
            continue
        with open(jp, encoding="utf-8") as f:
            bucket = json.load(f)
        pool_csv = jp.parent / f"high_performance_combos_{dstr}.csv"
        combos = []
        if pool_csv.is_file():
            df = pd.read_csv(pool_csv, encoding="utf-8-sig")
            for row in df.itertuples(index=False):
                combos.append((str(row.factor_id), str(row.stock),
                               int(row.k), int(row.side)))
        rebal.append({"date": T, "combos": combos,
                      "boundaries": bucket.get("boundaries", {})})
    return rebal


def _simulate(cfg: RollingConfig, calendar: pd.DatetimeIndex, rebal: list[dict],
              prices: dict[str, pd.DataFrame], get_factor_series) -> dict:
    """实盘逐日模拟主循环。

    Args:
        calendar: 全局交易日历（升序）。
        rebal: _load_rebal_artifacts 的结果（按日期升序）。
        prices: {stockid: 日线 DataFrame}。
        get_factor_series: (factor_id, stockid) -> 因子 Series（date 索引）。

    Returns:
        {"trades": DataFrame, "daily": DataFrame, "metrics": dict}
    """
    rebal_dates = [r["date"] for r in rebal]
    if not rebal_dates:
        raise SystemExit("实盘模拟需要至少一个调仓日产物（bucket_*.json）。")

    # 每股：date -> bar 位置、bar 位置 -> date、日期数组（自身 K 线位置计数用）
    stock_dates: dict[str, pd.DatetimeIndex] = {}
    stock_pos: dict[str, dict] = {}
    for sid, px in prices.items():
        idx = px.index
        stock_dates[sid] = idx
        stock_pos[sid] = {d: i for i, d in enumerate(idx)}

    # 实盘区间：首个调仓日次日 ~ 区间末端
    sim_days = [d for d in calendar if d > rebal_dates[0]
                and (cfg.end is None or d <= pd.Timestamp(cfg.end))]
    if not sim_days:
        raise SystemExit("首个调仓日之后没有可模拟的交易日。")

    positions: dict = {}   # key=(stock,factor,k,side) -> 持仓信息
    pending: dict = {}     # key -> 待开仓信号 {"signal_date": ..., "wait": ...}
    trades: list[dict] = []
    daily_rows: list[dict] = []
    cash = float(cfg.init_capital)  # 可用来买股票的现金，初始=初始资金
    n_skipped = 0
    n_dropped = 0
    fee = cfg.fee_rate

    for t in sim_days:
        # ---- 1) 开仓：pending 中当日可成交的信号均分当前现金（含费口径：
        #    每笔现金支出恰好 cash/n，股数 = (cash/n)/(open×(1+fee))） ----
        ready: list[tuple] = []
        for key, sig in list(pending.items()):
            sid = key[0]
            px = prices.get(sid)
            has_bar = px is not None and t in stock_pos[sid]
            if not has_bar:
                sig["wait"] += 1
                if sig["wait"] > cfg.max_entry_wait:  # 长期停牌，放弃该信号
                    n_dropped += 1
                    del pending[key]
                continue
            open_px = float(px.at[t, "open"])
            if not np.isfinite(open_px) or open_px <= 0:
                n_dropped += 1
                del pending[key]
                continue
            ready.append((key, open_px))
        if ready:
            if cash <= 0.0:  # 现金耗尽：当日信号全部跳过
                for key, _ in ready:
                    n_skipped += 1
                    del pending[key]
            else:
                amount = cash / len(ready)
                for key, open_px in ready:
                    sid = key[0]
                    shares = amount / (open_px * (1.0 + fee))
                    positions[key] = {
                        "entry_date": t, "entry_px": open_px, "shares": shares,
                        "entry_cost": amount,      # 现金支出（含买入费）
                        "entry_value": amount,     # 当日分得的买入金额 cash/n
                        "entry_pos": stock_pos[sid][t], "last_px": open_px,
                    }
                    cash -= amount
                    del pending[key]

        # ---- 2) 估值更新 ----
        for key, pos in positions.items():
            p = stock_pos[key[0]].get(t)
            if p is not None:
                pos["last_px"] = float(prices[key[0]].at[t, "close"])

        # ---- 3) 平仓：持仓满 k 日（按个股自身 K 线位置计数），t 日收盘平仓，
        #    回款加入 cash（当日收盘到账，次日开盘可用） ----
        for key, pos in list(positions.items()):
            sid, fid, k, side = key
            p = stock_pos[sid].get(t)
            if p is not None and p >= pos["entry_pos"] + k:
                exit_px = float(prices[sid].at[t, "close"])
                proceeds = pos["shares"] * exit_px * (1.0 - fee)
                cash += proceeds
                trades.append({
                    "stock": sid, "factor": fid, "k": k, "side": side,
                    "entry_date": pos["entry_date"], "entry_px": pos["entry_px"],
                    "exit_date": t, "exit_px": exit_px, "shares": pos["shares"],
                    "entry_value": pos["entry_value"],
                    "gross_ret": exit_px / pos["entry_px"] - 1.0,
                    "net_ret": proceeds / pos["entry_cost"] - 1.0,
                    "fee_paid": (pos["entry_cost"] - pos["shares"] * pos["entry_px"])
                    + pos["shares"] * exit_px * fee,
                    "forced": False,
                })
                del positions[key]

        # ---- 4) 信号检测（t 收盘）：用最近一次调仓日（严格早于 t）的池与边界 ----
        ri = bisect.bisect_left(rebal_dates, t) - 1
        if ri >= 0:
            rb = rebal[ri]
            bnd = rb["boundaries"]
            for fid, sid, k, side in rb["combos"]:
                b = bnd.get(fid, {}).get(sid)
                if b is None:
                    continue
                s = get_factor_series(fid, sid)
                if s is None:
                    continue
                v = s.get(t)
                if v is None or not np.isfinite(v):
                    continue
                if side >= cfg.n_buckets:
                    triggered = v >= b["q_high"]
                elif side == 1:
                    triggered = v <= b["q_low"]
                else:  # 中间桶无边界定义，不支持
                    continue
                if not triggered:
                    continue
                key = (sid, fid, k, side)
                if key in positions or key in pending:
                    n_skipped += 1  # 持仓期间（或已有待开仓）不再开新仓
                    continue
                pending[key] = {"signal_date": t, "wait": 0}

        # ---- 5) 收盘记账：总资产 = 现金 + 持仓市值（平仓回款已在 cash 中） ----
        occupied = float(sum(pos["shares"] * pos["last_px"]
                             for pos in positions.values()))
        daily_rows.append({"date": t, "cash": cash, "occupied": occupied,
                           "equity": cash + occupied,
                           "n_positions": len(positions)})

    # ---- 数据末端：未平仓持仓按各自最后可得收盘价（不晚于模拟末日）强平 ----
    sim_end = sim_days[-1]
    for key, pos in positions.items():
        sid, fid, k, side = key
        dates = stock_dates[sid]
        eligible = dates[dates <= sim_end]
        last_date = eligible[-1] if len(eligible) else dates[-1]
        exit_px = float(prices[sid].at[last_date, "close"])
        proceeds = pos["shares"] * exit_px * (1.0 - fee)
        cash += proceeds
        trades.append({
            "stock": sid, "factor": fid, "k": k, "side": side,
            "entry_date": pos["entry_date"], "entry_px": pos["entry_px"],
            "exit_date": last_date, "exit_px": exit_px, "shares": pos["shares"],
            "entry_value": pos["entry_value"],
            "gross_ret": exit_px / pos["entry_px"] - 1.0,
            "net_ret": proceeds / pos["entry_cost"] - 1.0,
            "fee_paid": (pos["entry_cost"] - pos["shares"] * pos["entry_px"])
            + pos["shares"] * exit_px * fee,
            "forced": True,
        })
    if daily_rows:
        # 强平发生在末日收盘：现金更新、持仓清零，末日总资产 = 现金
        daily_rows[-1]["cash"] = cash
        daily_rows[-1]["occupied"] = 0.0
        daily_rows[-1]["equity"] = cash

    daily = pd.DataFrame(daily_rows).set_index("date") if daily_rows else \
        pd.DataFrame(columns=["cash", "occupied", "equity", "n_positions"])
    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS)

    # ---- 绩效指标（初始资金固定，equity 为现金+持仓市值的逐日序列） ----
    max_occupied = float(daily["occupied"].max()) if len(daily) else 0.0
    metrics = _calc_metrics(daily["equity"], cfg.init_capital, cfg.bars_per_year,
                            len(trades_df), n_skipped)
    metrics.update(init_capital=cfg.init_capital, max_occupied=max_occupied,
                   cash_end=cash,
                   n_dropped=n_dropped,
                   n_rebalance=len(rebal_dates))
    return {"trades": trades_df, "daily": daily, "metrics": metrics}


def _plot_equity(daily: pd.DataFrame, metrics: dict, save_path: Path) -> None:
    """双子图：上=占用资金（持仓市值）与现金，下=总资产（叠加初始资金线），横轴日期。"""
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(daily))
    axes[0].fill_between(x, daily["occupied"].to_numpy(), step="mid",
                         alpha=0.5, color="#2980b9",
                         label="Occupied capital (position value)")
    axes[0].plot(x, daily["cash"].to_numpy(), color="#27ae60", lw=1.0,
                 label="Cash")
    axes[0].set_ylabel("Capital")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(x, daily["equity"].to_numpy(), color="#c0392b", lw=1.2,
                 label="Total assets")
    axes[1].axhline(metrics["init_capital"], color="gray", linestyle="--",
                    lw=1.0, label=f"Init capital ({metrics['init_capital']:,.0f})")
    axes[1].set_ylabel("Total assets")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # x 轴按行号绘制，刻度格式化为 yy-MM-dd（日线无 intraday 时分）
    n = len(daily)
    if n > 0:
        idx = daily.index
        step = max(1, n // 12)
        ticks = np.arange(0, n, step, dtype=int)
        if ticks[-1] != n - 1:
            ticks = np.append(ticks, n - 1)
        axes[1].set_xticks(ticks)
        axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda pos, _: "" if pos < 0 or pos >= n
            else str(idx[int(pos)])[2:10]))

    m = metrics
    fig.suptitle(
        f"Rolling backtest  total_ret={m['total_ret']:.2%}  "
        f"ann_ret={m['ann_ret']:.2%}  maxdd={m['maxdd']:.2%}  "
        f"sharpe={m['sharpe']:.2f}  calmar={m['calmar']:.2f}  "
        f"trades={m['n_trades']}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_simulate(cfg: RollingConfig, stocks: list[str], calendar: pd.DatetimeIndex,
                 prices: dict[str, pd.DataFrame]) -> dict:
    """阶段B 主流程：读重训产物 → 逐日模拟 → 写 trades/daily/图/summary。"""
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rebal = _load_rebal_artifacts(outdir, cfg)
    print(f"[simulate] 载入 {len(rebal)} 个调仓日产物；"
          f"候选池合计 {sum(len(r['combos']) for r in rebal)} 组合", flush=True)

    # 防御：候选池中股票若不在本次加载的行情里（如 --simulate-only 与重训时
    # 的 --top 不一致），补充加载其行情并并入全局日历，避免信号被静默放弃
    pool_stocks = {sid for r in rebal for (_, sid, _, _) in r["combos"]}
    missing = sorted(pool_stocks - set(prices))
    if missing:
        print(f"[simulate] 候选池含 {len(missing)} 只未加载行情的股票，补充加载",
              flush=True)
        extra = load_prices_cached(missing)
        prices = {**prices, **extra}
        if extra:
            calendar = pd.DatetimeIndex(sorted(
                set(calendar).union(*(px.index for px in extra.values()))))

    # 因子值查询（含缓存：每 (factor, stock) 只读一次）
    from factors.library import FactorLibrary
    lib = FactorLibrary()
    cache: dict = {}

    def get_factor_series(fid: str, sid: str):
        key = (fid, sid)
        if key not in cache:
            try:
                cache[key] = lib.get(fid, sid)
            except Exception:
                cache[key] = None
        return cache[key]

    result = _simulate(cfg, calendar, rebal, prices, get_factor_series)

    result["trades"].to_csv(outdir / "trades.csv", index=False,
                            encoding="utf-8-sig")
    result["daily"].reset_index().to_csv(outdir / "daily_equity.csv",
                                         index=False, encoding="utf-8-sig")
    if len(result["daily"]):
        _plot_equity(result["daily"], result["metrics"],
                     outdir / "equity_curve.png")
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=1,
                  default=str)

    m = result["metrics"]
    print("=" * 60)
    print("滚动回测完成：")
    print(f"  总收益率   : {m['total_ret']:.2%}")
    print(f"  年化收益率 : {m['ann_ret']:.2%}" if pd.notna(m["ann_ret"]) else
          "  年化收益率 : NaN")
    print(f"  最大回撤   : {m['maxdd']:.2%}")
    print(f"  Sharpe     : {m['sharpe']:.3f}" if pd.notna(m["sharpe"]) else
          "  Sharpe     : NaN")
    print(f"  卡玛比率   : {m['calmar']:.3f}" if pd.notna(m["calmar"]) else
          "  卡玛比率   : NaN")
    print(f"  交易次数   : {m['n_trades']}（跳过信号 {m['n_skipped']}，"
          f"放弃开仓 {m['n_dropped']}）")
    print(f"  初始资金   : {m['init_capital']:,.0f}"
          f"（期末现金 {m['cash_end']:,.0f}，最大占用 {m['max_occupied']:,.0f}）")
    print(f"  输出目录   : {outdir}")
    print("=" * 60)
    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="滚动回测：每 M 日重训候选股票池（窗口 N 日 IC+回测筛选），"
                    "逐日模拟交易（当日买入信号均分现金开仓，持仓 k 日平仓）。")
    p.add_argument("--top", type=int, default=20,
                   help="股票范围：成交额排名 csv 前 N 名（默认 20）")
    p.add_argument("--list-file", default=str(Path(_SRC_DIR) / "BJ_stocks_top100.csv"),
                   help="成交额排名 csv（默认 src/BJ_stocks_top100.csv）")
    p.add_argument("--factors", default=None,
                   help="逗号分隔的因子子集（默认库中全部因子），如 Alpha001,Alpha098")
    p.add_argument("--lookback", type=int, default=63,
                   help="回看窗口长度 N（交易日 bar 数，默认 63 ≈ 3 个月）")
    p.add_argument("--rebalance", type=int, default=5,
                   help="股票池更新间隔 M（交易日，默认 5）")
    p.add_argument("--ks", default="1,2,3,5,10,15",
                   help="前向收益/持仓天数 k 列表（默认 1,2,3,5,10,15）")
    p.add_argument("--init-capital", type=float, default=10_000_000.0,
                   help="初始资金（默认 1000 万元；初始现金=初始资金）")
    p.add_argument("--fee", type=float, default=5e-5,
                   help="单边费率（默认 5e-5，双边收取）")
    p.add_argument("--min-window-bars", type=int, default=40,
                   help="窗口内对齐样本下限（默认 40），不足跳过该组合")
    p.add_argument("--start", default=None, help="回测起始日期，如 2023-01-01")
    p.add_argument("--end", default=None, help="回测结束日期")
    p.add_argument("--njobs", type=int, default=None,
                   help="重训并行进程数（默认取 src/config.json 的 [batch].njobs）")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR,
                   help=f"输出目录（默认 {DEFAULT_OUTDIR}）")
    p.add_argument("--retrain-only", action="store_true", help="只跑阶段A（重训）")
    p.add_argument("--simulate-only", action="store_true",
                   help="只跑阶段B（复用 outdir 已有重训产物）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip())
    njobs = args.njobs or int(_load_batch_config()["njobs"])
    cfg = RollingConfig(
        top=args.top, list_file=args.list_file,
        factors=[f.strip() for f in args.factors.split(",") if f.strip()]
        if args.factors else None,
        ks=ks, lookback=args.lookback, rebalance=args.rebalance,
        init_capital=args.init_capital, fee_rate=args.fee,
        min_window_bars=args.min_window_bars,
        start=args.start, end=args.end, njobs=njobs, outdir=args.outdir,
    )

    stocks = load_stock_list(cfg.list_file, cfg.top)
    prices = load_prices_cached(stocks)
    calendar = pd.DatetimeIndex(sorted(set().union(
        *(px.index for px in prices.values()))))
    print(f"股票 {len(stocks)} 只；全局日历 {calendar[0].date()} ~ "
          f"{calendar[-1].date()} 共 {len(calendar)} 个交易日", flush=True)

    if not args.simulate_only:
        from factors.library import FactorLibrary
        lib = FactorLibrary()
        factor_ids = lib.list_factors()["factor_id"].tolist()
        if cfg.factors:
            unknown = [f for f in cfg.factors if f not in factor_ids]
            if unknown:
                raise KeyError(f"未知因子: {unknown}，请用 python -m factors list 查看")
            factor_ids = cfg.factors
        run_retrain(cfg, stocks, calendar, factor_ids)
    if not args.retrain_only:
        run_simulate(cfg, stocks, calendar, prices)


def load_prices_cached(stocks: list[str]) -> dict[str, pd.DataFrame]:
    """加载全部股票行情（缺文件的告警跳过）。"""
    out: dict[str, pd.DataFrame] = {}
    for sid in stocks:
        try:
            out[sid] = load_daily_bars(sid)
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
    return out


if __name__ == "__main__":
    main()
