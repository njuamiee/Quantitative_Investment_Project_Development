"""Audited historical score bridge. These packets are NEVER marked live validated."""
from __future__ import annotations
import json
from pathlib import Path

from .data import file_sha


def export(root, output, start="2026-07-01", end="2026-09-30"):
    import numpy as np
    import pandas as pd
    root = Path(root)
    run = root / "training_runs/systematic_20261026"
    config = dict(horizon=2, count=20, model="ridge", frequency="daily", window=126, library="daily")
    import hashlib
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    folder = run / "models" / key
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if complete["score_sha256"] != file_sha(folder / "predictions.parquet"):
        raise ValueError("source prediction hash mismatch")
    audit = json.loads((folder / "fit_audit.json").read_text(encoding="utf-8"))
    for row in audit:
        if row["last_mature_outcome_index"] >= row["fit_index"]:
            raise ValueError("source fit leaked labels")
    with np.load(run / "data/market.npz") as archive:
        m = {k: archive[k] for k in archive.files}
    cal = pd.DatetimeIndex(json.loads((run / "data/calendar.json").read_text(encoding="utf-8")))
    regimes = pd.read_parquet(run / "data/regimes.parquet")
    pred = pd.read_parquet(folder / "predictions.parquet")
    fits = {r["fit_index"]: r for r in audit}
    records = []
    for date, f in pred.groupby("date", sort=True):
        di = int(f.date_index.iloc[0])
        if di + 1 >= len(cal): continue
        trading_date = str(cal[di + 1].date())
        if not start <= trading_date <= end: continue
        signal_date = str(pd.Timestamp(date).date())
        a = fits[di]
        candidates = []
        for row in f.itertuples():
            si = int(row.stock_index)
            if not np.isfinite(m["close"][di, si]) or not np.isfinite(m["vol20"][di, si]): continue
            candidates.append(dict(symbol=row.stockid[2:] + ".BSE", score=float(row.score),
                reference=float(m["close"][di, si]), volatility=max(.001, float(m["vol20"][di, si])),
                industry=str(m["industry"][di, si]), size_group=str(m["size"][di, si]),
                pool_rank=float(m["amount_rank"][di, si]) / 100., rebound=bool(m["mom2"][di, si] > 0), minute_increment=0))
        records.append(dict(id=f"research:{key}:{signal_date}", observed_at=signal_date + "T15:00:00+08:00",
            available_at=signal_date + "T16:00:00+08:00", valid_from=trading_date + "T09:30:00+08:00",
            expires_at=trading_date + "T14:57:00+08:00", model_trained_at=signal_date + "T08:00:00+08:00",
            labels_end=str(cal[a["last_mature_outcome_index"]].date()) + "T15:00:00+08:00",
            candidates=candidates, market_cap=float(regimes.iloc[di].hmm3_cap),
            market_vol=max(.001, float(regimes.iloc[di].vol)), model_id=key, research_only=True))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in records) + "\n", encoding="utf-8")
    return dict(packets=len(records), source_job=key, config=config, sha256=file_sha(output),
        timestamps="historical assumed daily availability, NOT verified real delivery times",
        research_only=True)
