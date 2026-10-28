"""Explicit market-data contracts; no fabricated order book presented as real ticks."""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from pathlib import Path

from .domain import Candidate, SH, Signal, Tick, stamp


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_signal(record):
    record = dict(record)
    record["candidates"] = tuple(Candidate(**v) for v in record["candidates"])
    for k in ("observed_at", "available_at", "valid_from", "expires_at", "model_trained_at", "labels_end"):
        record[k] = stamp(record[k])
    return Signal(**record)


def quarantine_minutes(frame):
    """Explicit import quality report; never silently repair OHLC or infer units."""
    import numpy as np
    import pandas as pd
    cols = ["open", "high", "low", "close", "volume", "amount"]
    f = frame.copy()
    values = f[cols].to_numpy(float, na_value=np.nan)
    missing = f[cols].isna().all(axis=1).to_numpy()
    bad = ~np.isfinite(values).all(axis=1) | (values[:, :4] <= 0).any(axis=1) | (values[:, 4:] < 0).any(axis=1)
    bad |= (values[:, 1] < np.max(values[:, [0, 2, 3]], axis=1)) | (values[:, 2] > np.min(values[:, [0, 1, 3]], axis=1))
    vwap = np.divide(values[:, 5], values[:, 4], out=np.zeros(len(f)), where=values[:, 4] > 0)
    mismatch = (values[:, 4] > 0) & ((vwap < values[:, 2] * .995) | (vwap > values[:, 1] * 1.005))
    mismatch |= (values[:, 4] == 0) & (values[:, 5] != 0)
    duplicate = f.duplicated(["code", "datetime"], keep=False).to_numpy()
    rejected = bad | mismatch | duplicate
    reasons = np.where(missing, "all_fields_missing", np.where(duplicate, "duplicate",
                      np.where(bad, "invalid_numeric_or_ohlc", "amount_volume_inconsistent")))
    audit = f.loc[rejected, ["code", "datetime"]].copy()
    audit["reason"] = reasons[rejected]
    return f.loc[~rejected].copy(), audit


def minute_events(frame, symbols=None, timestamp_convention="start", volume_unit="shares"):
    """Proxy at minute END, with explicitly synthetic bid/ask. Not an L2 backtest.

    Raw input timestamp convention and volume unit must be specified by importer.
    Index rows are excluded. Existing BJ code mapping must be vendor-verified
    before live use (historical renumbering is not guessed here).
    """
    import numpy as np
    import pandas as pd
    if timestamp_convention not in ("start", "end") or volume_unit not in ("shares", "lots100"):
        raise ValueError("timestamp convention and volume units must be explicit")
    required = {"code", "datetime", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing minute columns: {required - set(frame.columns)}")
    f = frame.copy()
    f["symbol"] = f.code.astype(str).str.zfill(6) + ".BSE"
    f = f[f.code.astype(str).str.fullmatch(r"920\d{3}")]
    if symbols is not None:
        f = f[f.symbol.isin(symbols)]
    if f.duplicated(["symbol", "datetime"]).any():
        raise ValueError("duplicate minute messages")
    f = f.sort_values(["datetime", "symbol"], kind="stable")
    # Vendor grid placeholders with ALL six fields missing are gaps, not zero
    # prices or executable minutes. Partial missing rows remain a hard error.
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    f = f[~f[numeric].isna().all(axis=1)]
    values = f[["open", "high", "low", "close", "volume", "amount"]].to_numpy(float, na_value=np.nan)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise ValueError("invalid price/volume rows")
    if (f.high < f[["open", "close", "low"]].max(axis=1)).any() or (f.low > f[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("invalid OHLC ordering")
    multiplier = 100 if volume_unit == "lots100" else 1
    for r in f.itertuples():
        at = pd.Timestamp(r.datetime).to_pydatetime()
        at = at.replace(tzinfo=SH) if at.tzinfo is None else at.astimezone(SH)
        if timestamp_convention == "start":
            at += timedelta(minutes=1)
        # Zero volume bars are valid observations but cannot fill anything.
        quantity = float(r.volume) * multiplier
        if quantity:
            vwap = float(r.amount) / quantity
            if not r.low * .995 <= vwap <= r.high * 1.005:
                raise ValueError(f"volume/amount unit mismatch: {r.symbol} {at}")
        close = float(r.close)
        yield Tick(r.symbol, at, at, close, round(close - .01, 2), round(close + .01, 2),
                   quantity, quantity, quantity, f"bar:{r.symbol}:{at.isoformat()}", synthetic=True)


def validate_ticks(records):
    """Input JSONL contract: incremental shares, aware exchange/receipt timestamps."""
    previous = None
    seen = set()
    for record in records:
        row = dict(record)
        row["at"], row["available_at"] = stamp(row["at"]), stamp(row["available_at"])
        t = Tick(**row)
        if t.synthetic:
            raise ValueError("tick feed cannot contain synthetic minute quotes")
        if previous and t.available_at < previous:
            raise ValueError("unordered tick input")
        key = (t.symbol, t.sequence)
        if key in seen:
            raise ValueError("duplicate tick identity")
        seen.add(key)
        previous = t.available_at
        yield t


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
