"""Training/inference file contracts and repeatable model release boundaries."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import argparse
import json
from pathlib import Path

from .data import file_sha
from .domain import Candidate, Signal, stamp
from .models import ModelBundle, fit


def refit_due(now, last_fit, frequency):
    now = stamp(now)
    if last_fit is None: return True
    last = stamp(last_fit)
    if last > now: raise ValueError("future fit timestamp")
    if frequency == "daily": return last.date() != now.date()
    if frequency == "monthly": return (last.year, last.month) != (now.year, now.month)
    if frequency == "quarterly": return (last.year, (last.month-1)//3) != (now.year, (now.month-1)//3)
    raise ValueError("unknown frequency")


def make_signal(bundle, frame, manifest, now, valid_from, expires_at, identifier):
    """manifest attests the entire feature/signal snapshot, including HMM inputs.

    Receiving a research matrix cannot make it point-in-time verified. Defaults
    remain research-only until a real data producer supplies release timestamps.
    """
    import numpy as np
    import pandas as pd
    now = stamp(now)
    required = {"symbol", "reference", "volatility", "industry", "size_group", "pool_rank", "rebound"}
    if not required.issubset(frame): raise ValueError("missing portfolio snapshot fields")
    observed = stamp(manifest["observed_at"])
    released = stamp(manifest["available_at"])
    if observed > released or released > now:
        raise ValueError("portfolio/HMM snapshot unavailable")
    if frame.symbol.duplicated().any(): raise ValueError("duplicate feature symbols")
    p = frame[["hmm_p0", "hmm_p1", "hmm_p2"]] if bundle.family == "conditional_ic" else None
    scores = bundle.predict(frame, now, p)
    candidates = []
    for r, score in zip(frame.itertuples(), scores):
        candidates.append(Candidate(r.symbol, float(score), float(r.reference), float(r.volatility),
            str(r.industry), str(r.size_group), float(r.pool_rank), bool(r.rebound),
            float(getattr(r, "minute_increment", 0))))
    s = Signal(identifier, observed, now, stamp(valid_from), stamp(expires_at),
               stamp(bundle.audit["trained_at"]), stamp(bundle.audit["labels_end"]), tuple(candidates),
               float(manifest["market_cap"]), float(manifest["market_vol"]), manifest["model_id"],
               bool(bundle.audit.get("research_only", True) or not manifest.get("pit_verified", False)))
    s.validate(s.valid_from)
    return s


def publish(signal, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(asdict(signal), default=str, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description="Model fit / signal publication; does not connect brokers")
    sub = p.add_subparsers(dest="action", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--frame", required=True)
    train.add_argument("--features", required=True, help="JSON list of ordered feature names")
    train.add_argument("--family", default="ridge")
    train.add_argument("--at", required=True)
    train.add_argument("--count", type=int, default=20)
    train.add_argument("--window", type=int, default=126)
    train.add_argument("--output", required=True)
    serve = sub.add_parser("signal")
    for key in ("bundle", "frame", "manifest", "at", "valid-from", "expires-at", "id", "output"):
        serve.add_argument("--" + key, required=True)
    args = p.parse_args()
    import pandas as pd
    f = pd.read_parquet(args.frame)
    if args.action == "fit":
        names = json.loads(Path(args.features).read_text(encoding="utf-8"))
        model = fit(f, names, args.family, args.at, args.count, args.window)
        model.audit["input_sha256"] = file_sha(args.frame)
        print(json.dumps(model.save(args.output), ensure_ascii=False))
    else:
        bundle = ModelBundle.load_trusted(args.bundle)
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if manifest.get("feature_file_sha256") != file_sha(args.frame):
            raise ValueError("feature snapshot digest mismatch")
        s = make_signal(bundle, f, manifest, args.at, args.valid_from, args.expires_at, args.id)
        publish(s, args.output)
        print(json.dumps(dict(signal_id=s.id, stocks=len(s.candidates), research_only=s.research_only)))


if __name__ == "__main__": main()
