"""中证2000 真实动态池：模型梯队 + walk-forward 实验（张子晨图片要求）。

梯队：
  第0级 ICIR 滚动加权合成（零训练基准）
  第1级 Ridge（L2 线性合成，alpha 在 fold 内验证段选）
  第2级 LightGBM（非线性，fold 内早停）

预处理（图片要求）：因子截面 winsorize(1%,99%) → 截面 z-score；标签减截面均值。
滚动（图片要求）：对预测区间 [t, t+ntest)，训练 [t-ntrain-nval-gap, t-nval-gap)，
  验证 [t-nval-gap, t-gap)，gap=6 个交易日隔开标签窗口，前闭后开，不随机切分。

分段（1018 已登记协议 + 用户新增）：
  隔离 2025-06-04~2025-07-01（不使用）；验证 2025-07-02~2026-05-22；
  新增测试 2026-05-25~2026-10-19（05-22 归属验证段，避免两段重叠）。
  验证与测试均为样本外(OOS)。
费用滑点：0（张子晨指定口径）。标签：t+1 开盘买 → t+5 收盘卖（本项目口径，与张子晨
  PPT 的 open(t+1)→open(t+1+5) 差一个收盘，差异单列说明）。

输出：training_runs/research_20261021/
  daily_factor_ic.parquet     逐因子逐日 rank IC
  predictions.parquet         逐日逐股三模型分数
  model_daily_ic.parquet      三模型逐日 rank IC
  portfolio_batches.csv       Top20 五日轮换逐批收益
  summary.json / report 中间结果
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "training_data" / "research_20261021" / "market" / "panel_member.parquet"
OUT = ROOT / "training_runs" / "research_20261021"
FACTOR_COLUMNS = [f"oho_alpha_{i:03d}" for i in range(1, 102)]

N_TRAIN, N_VAL, GAP, N_TEST = 252, 63, 6, 21
SEGMENTS = {
    "roll_oos_before_validation": ("1900-01-01", "2025-07-01"),
    "validation": ("2025-07-02", "2026-05-22"),
    "test": ("2026-05-25", "2026-10-19"),
}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
SEED = 20261021


def winsorize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """逐日对因子做截面 winsorize + z-score；标签减截面均值。按日循环避免大中间表。"""
    df = df.sort_values(["date", "stockid"]).reset_index(drop=True)
    z = np.full((len(df), len(FACTOR_COLUMNS)), np.nan, dtype=np.float32)
    y = np.full(len(df), np.nan, dtype=np.float64)
    pos = 0
    for _, idx in df.groupby("date", sort=True).indices.items():
        idx = np.asarray(idx)
        f = df.iloc[idx][FACTOR_COLUMNS].to_numpy(dtype=np.float64)
        lo = np.nanquantile(f, 0.01, axis=0)
        hi = np.nanquantile(f, 0.99, axis=0)
        f = np.clip(f, lo, hi)
        mu = np.nanmean(f, axis=0)
        sd = np.nanstd(f, axis=0)
        sd[sd == 0] = np.nan
        z[idx] = ((f - mu) / sd).astype(np.float32)
        lab = df.iloc[idx]["label_5"].to_numpy(dtype=np.float64)
        y[idx] = lab - np.nanmean(lab)
        pos += 1
    zdf = pd.DataFrame(z, columns=[c + "_z" for c in FACTOR_COLUMNS], index=df.index)
    df = pd.concat([df, zdf, pd.Series(y, name="label_dm", index=df.index)], axis=1)
    return df


def daily_rank_ic(scores: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < 20:
        return np.nan
    s = pd.Series(scores[mask]).rank().to_numpy()
    l = pd.Series(labels[mask]).rank().to_numpy()
    s = (s - s.mean()) / s.std()
    l = (l - l.mean()) / l.std()
    return float((s * l).mean())


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel.date >= "2023-08-11"].reset_index(drop=True)
    print(f"panel {len(panel)} rows", flush=True)

    print("preprocessing winsorize/zscore ...", flush=True)
    panel = winsorize_zscore(panel)
    zcols = [c + "_z" for c in FACTOR_COLUMNS]
    panel[zcols] = panel[zcols].fillna(0.0)

    # ---------- 逐因子逐日 rank IC（全历史，供 ICIR 权重） ----------
    print("daily factor IC ...", flush=True)
    ic_rows = []
    for d, idx in panel.groupby("date", sort=True).indices.items():
        idx = np.asarray(idx)
        lab = panel.iloc[idx]["label_5"].to_numpy()
        if np.isfinite(lab).sum() < 50:
            continue
        f = panel.iloc[idx][FACTOR_COLUMNS].to_numpy(dtype=np.float64)
        fr = pd.DataFrame(np.where(np.isfinite(f), f, np.nan)).rank().to_numpy()
        lr = pd.Series(lab).rank().to_numpy().astype(float).copy()
        lr[~np.isfinite(lab)] = np.nan

        def col_ic(fc):
            m = np.isfinite(fc) & np.isfinite(lr)
            if m.sum() < 50:
                return np.nan
            a = fc[m]; b = lr[m]
            sa = a.std(); sb = b.std()
            if sa == 0 or sb == 0:
                return np.nan
            a = (a - a.mean()) / sa
            b = (b - b.mean()) / sb
            return float((a * b).mean())
        ic_rows.append([d] + [col_ic(fr[:, j]) for j in range(len(FACTOR_COLUMNS))])
    daily_ic = pd.DataFrame(ic_rows, columns=["date"] + FACTOR_COLUMNS)
    daily_ic.to_parquet(OUT / "daily_factor_ic.parquet", index=False)
    print(f"factor IC done {time.time()-t0:.0f}s", flush=True)

    # ---------- walk-forward ----------
    dates = np.array(sorted(panel.date.unique()))
    date_pos = {d: i for i, d in enumerate(dates)}
    by_date = {d: g for d, g in panel.groupby("date", sort=True)}
    ic_by_date = daily_ic.set_index("date")

    folds = []
    i = N_TRAIN + N_VAL + GAP
    while i < len(dates):
        test_end = min(i + N_TEST, len(dates))
        folds.append((i - N_TRAIN - N_VAL - GAP, i - N_VAL - GAP, i - GAP, i, test_end))
        i = test_end
    print(f"{len(folds)} folds", flush=True)

    pred_frames = []
    fold_log = []
    for k, (tr0, tr1, va1, te0, te1) in enumerate(folds):
        train_dates = dates[tr0:tr1]
        val_dates = dates[tr1:va1]
        test_dates = dates[te0:te1]
        train = pd.concat([by_date[d] for d in train_dates])
        val = pd.concat([by_date[d] for d in val_dates])
        test = pd.concat([by_date[d] for d in test_dates])
        train = train[np.isfinite(train.label_dm)]
        val = val[np.isfinite(val.label_dm)]

        Xtr = train[zcols].to_numpy(np.float32)
        ytr = train.label_dm.to_numpy()
        Xva = val[zcols].to_numpy(np.float32)
        yva = val.label_dm.to_numpy()
        Xte = test[zcols].to_numpy(np.float32)

        # 第0级：ICIR（用 fold 内验证段的逐因子IC）
        vic = ic_by_date.loc[val_dates]
        mu = vic.mean().to_numpy()
        sd = vic.std().to_numpy()
        w = np.where(sd > 1e-12, mu / sd, 0.0)
        w = np.nan_to_num(w)
        score_icir = Xte.astype(np.float64) @ w

        # 第1级：Ridge，alpha 在 fold 内验证段按 rank IC 选
        best_alpha, best_vic = None, -np.inf
        best_model = None
        for a in RIDGE_ALPHAS:
            m = Ridge(alpha=a)
            m.fit(Xtr, ytr)
            preds = m.predict(Xva)
            vdates = val.date.to_numpy()
            ics = [daily_rank_ic(preds[vdates == d], val.label_5.to_numpy()[vdates == d])
                   for d in val_dates]
            mic = np.nanmean(ics)
            if mic > best_vic:
                best_vic, best_alpha, best_model = mic, a, m
        score_ridge = best_model.predict(Xte)

        # 第2级：LightGBM，fold 内早停
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xva, label=yva, reference=dtrain)
        params = dict(objective="regression", learning_rate=0.05, num_leaves=31,
                      min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, seed=SEED, verbose=-1, num_threads=8)
        booster = lgb.train(params, dtrain, num_boost_round=300, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
        score_lgbm = booster.predict(Xte, num_iteration=booster.best_iteration or 300)

        pf = test[["date", "stockid", "label_5", "close", "open", "amount"]].copy()
        pf["score_icir"] = score_icir
        pf["score_ridge"] = score_ridge
        pf["score_lgbm"] = score_lgbm
        pred_frames.append(pf)
        fold_log.append({"fold": k, "train": [str(dates[tr0])[:10], str(dates[tr1 - 1])[:10]],
                         "val": [str(dates[tr1])[:10], str(dates[va1 - 1])[:10]],
                         "test": [str(dates[te0])[:10], str(dates[te1 - 1])[:10]],
                         "ridge_alpha": best_alpha, "ridge_val_ic": float(best_vic),
                         "lgbm_best_iter": int(booster.best_iteration or 300),
                         "train_rows": int(len(train))})
        print(f"fold {k+1}/{len(folds)} test {str(dates[te0])[:10]}..{str(dates[te1-1])[:10]} "
              f"alpha={best_alpha} iter={booster.best_iteration} {time.time()-t0:.0f}s", flush=True)

    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_parquet(OUT / "predictions.parquet", index=False)
    (OUT / "folds.json").write_text(json.dumps(fold_log, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 三模型逐日 rank IC ----------
    rows = []
    for d, g in preds.groupby("date", sort=True):
        lab = g.label_5.to_numpy()
        rows.append({"date": d,
                     "ic_icir": daily_rank_ic(g.score_icir.to_numpy(), lab),
                     "ic_ridge": daily_rank_ic(g.score_ridge.to_numpy(), lab),
                     "ic_lgbm": daily_rank_ic(g.score_lgbm.to_numpy(), lab)})
    model_ic = pd.DataFrame(rows)
    model_ic.to_parquet(OUT / "model_daily_ic.parquet", index=False)

    # ---------- 分段汇总 ----------
    summary = {}
    for seg, (s, e) in SEGMENTS.items():
        sub = model_ic[(model_ic.date >= s) & (model_ic.date <= e)]
        seg_out = {"days": int(len(sub))}
        for col in ("ic_icir", "ic_ridge", "ic_lgbm"):
            v = sub[col].dropna()
            seg_out[col] = {"mean_ic": float(v.mean()), "ic_std": float(v.std()),
                            "icir": float(v.mean() / v.std()) if v.std() > 0 else None,
                            "ic_pos_ratio": float((v > 0).mean())}
        summary[seg] = seg_out
    (OUT / "ic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
