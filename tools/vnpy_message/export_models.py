"""Real project-data fit/serialization verification, not performance optimization."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quant_message.models import FAMILIES, ModelBundle, fit


def main():
    p = ROOT / "training_runs/systematic_20261026/data"
    out = ROOT / "training_runs/vnpy_message/model_bundles"
    out.mkdir(parents=True, exist_ok=True)
    cal = pd.DatetimeIndex(json.loads((p / "calendar.json").read_text()))
    meta = pd.read_parquet(p / "meta.parquet")
    names = json.loads((p / "names_daily.json").read_text())
    x = np.load(p / "X_daily.npy", mmap_mode="r")
    y = np.load(p / "targets.npy", mmap_mode="r")[:, 1]  # 2-session target.
    exits = np.load(p / "label_exit_index.npy", mmap_mode="r")[:, 1]
    regimes = pd.read_parquet(p / "regimes.parquet")
    fit_index = cal.get_loc(pd.Timestamp("2026-06-30"))
    mask = (meta.date_index >= fit_index - 140) & (meta.date_index <= fit_index)
    ix = np.flatnonzero(mask)
    f = pd.DataFrame(np.asarray(x[ix]), columns=names)
    f["date"] = meta.iloc[ix].date.to_numpy()
    f["available_at"] = pd.DatetimeIndex(f.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    f["label_available_at"] = cal[np.maximum(exits[ix], 0)].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    f["target"] = y[ix]
    for i in range(3): f[f"hmm_p{i}"] = regimes.iloc[meta.iloc[ix].date_index.to_numpy()][f"hmm3_p{i}"].to_numpy()
    at = "2026-06-30T17:00:00+08:00"
    serving = f.iloc[-100:].copy()
    records = []
    for family in FAMILIES:
        started = time.time()
        bundle = fit(f, names, family, at, 20, 126)
        bundle.audit.update(horizon=2, source="existing daily feature matrix; historical PIT assumption",
                            fixed_export_cutoff="2026-06-30", reason="execution migration acceptance, not model selection")
        card = bundle.save(out / family)
        a = bundle.predict(serving, at, serving[[f"hmm_p{i}" for i in range(3)]])
        b = ModelBundle.load_trusted(out / family).predict(serving, at, serving[[f"hmm_p{i}" for i in range(3)]])
        if not np.array_equal(a, b): raise AssertionError("reload prediction mismatch")
        records.append(dict(family=family, training_rows=card["training_rows"],
            training_sessions=card["training_sessions"], factors=len(bundle.features),
            inference_rows=len(a), reload_max_abs_error=float(abs(a-b).max()),
            labels_end=card["labels_end"], trained_at=card["trained_at"],
            model_sha256=card["model_sha256"], research_only=True, seconds=time.time()-started))
        print(json.dumps(records[-1]), flush=True)
        (out / "verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
