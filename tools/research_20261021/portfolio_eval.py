"""Top20 五日轮换组合评估（零费用零滑点，张子晨口径）。

信号日每 5 个交易日一次；按各模型分数取 Top20 等权，t+1 开盘买、t+5 收盘卖；
批与批之间有一天现金空窗（本项目惯例），费用滑点为 0。
基准：中证2000全收益指数（官方日线，close(t)→close(t+5)）、全体成员等权同规则轮换。
输出：training_runs/research_20261021/portfolio_*.csv, portfolio_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "training_runs" / "research_20261021"
MKT = ROOT / "training_data" / "research_20261021" / "market"

SEGMENTS = {
    "validation": ("2025-07-02", "2026-05-22"),
    "test": ("2026-05-25", "2026-10-19"),
    "full_eval": ("2025-07-02", "2026-10-19"),
}


def metrics(rets: np.ndarray, periods_per_year: float = 252 / 5) -> dict:
    rets = np.asarray(rets, dtype=float)
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    dd = np.r_[1.0, eq] / peak - 1
    n = len(rets)
    total = eq[-1] - 1
    ann = (eq[-1]) ** (periods_per_year / n) - 1 if n else np.nan
    sd = rets.std(ddof=1)
    return {"batches": int(n), "total_return": float(total), "annual_return": float(ann),
            "max_drawdown": float(-dd.min()), "sharpe": float(rets.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else None,
            "calmar": float(ann / -dd.min()) if dd.min() < 0 else None,
            "win_rate": float((rets > 0).mean())}


def main() -> None:
    preds = pd.read_parquet(RUN / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    idx = pd.read_csv(MKT / "932000CNY010_official_daily_ext.csv", dtype={"tradeDate": str})
    idx["date"] = pd.to_datetime(idx["tradeDate"], format="%Y%m%d")
    idx = idx.set_index("date").sort_index()

    dates = np.array(sorted(preds.date.unique()))
    by_date = {d: g for d, g in preds.groupby("date", sort=True)}

    start_pos = int(np.searchsorted(dates, np.datetime64("2025-07-02")))
    rows = []
    s = start_pos
    while s < len(dates):
        d = dates[s]
        g = by_date[d]
        g = g[np.isfinite(g.label_5)]
        # 指数同期收益：close(t) -> close(t+5)
        end_pos = min(s + 5, len(dates) - 1)
        d_end = dates[end_pos]
        try:
            idx_ret = float(idx.loc[d_end, "close"] / idx.loc[d, "close"] - 1)
        except KeyError:
            idx_ret = np.nan
        row = {"signal_date": d, "exit_date": d_end, "n_members_with_label": len(g),
               "bench_index_tr": idx_ret,
               "bench_equal_members": float(g.label_5.mean())}
        for col in ("score_icir", "score_ridge", "score_lgbm"):
            top = g.nlargest(20, col)
            row[f"ret_{col[6:]}"] = float(top.label_5.mean())
            row[f"n_{col[6:]}"] = int(len(top))
        rows.append(row)
        s += 5
    batches = pd.DataFrame(rows)
    batches.to_csv(RUN / "portfolio_batches.csv", index=False, encoding="utf-8-sig")

    summary = {}
    for seg, (a, b) in SEGMENTS.items():
        sub = batches[(batches.signal_date >= a) & (batches.signal_date <= b)]
        seg_out = {}
        for col in ("ret_icir", "ret_ridge", "ret_lgbm", "bench_equal_members", "bench_index_tr"):
            v = sub[col].dropna().to_numpy()
            if len(v):
                seg_out[col] = metrics(v)
        summary[seg] = seg_out
    (RUN / "portfolio_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
