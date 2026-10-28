"""Run V1 and the full signal/risk controller on timestamped real minute files."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import argparse
import json
from pathlib import Path

from .data import file_sha, minute_events, read_jsonl, read_signal, quarantine_minutes
from .domain import Costs, Instrument, SH, Tick
from .paper import PaperBroker
from .strategy import RiskRules, SimpleMomentumStrategy, TargetStrategy


def minute_paths(root, start, end):
    found = {}
    for folder in ("research_20260930", "research_20261008", "research_20261011"):
        for path in (root / "training_data" / folder / "minute").glob("*.parquet"):
            date = f"{path.stem[:4]}-{path.stem[4:6]}-{path.stem[6:]}"
            if start <= date <= end:
                found[date] = path
    return found


def metrics(curve, initial):
    import numpy as np
    nav = np.array([initial] + [r["equity"] for r in curve])
    returns = nav[1:] / nav[:-1] - 1
    dd = 1 - nav / np.maximum.accumulate(nav)
    ann = (nav[-1] / nav[0]) ** (252 / max(1, len(returns))) - 1
    vol = returns.std(ddof=1) if len(returns) > 1 else 0
    return dict(return_pct=float((nav[-1] / nav[0] - 1) * 100),
        max_drawdown_pct=float(dd.max() * 100), annualized_return_pct=float(ann * 100),
        sharpe=float((returns.mean() - .015 / 252) / vol * np.sqrt(252)) if vol > 0 else None,
        calmar=float(ann / dd.max()) if dd.max() else None, sessions=len(returns))


def run(root, output, start="2026-07-01", end="2026-09-30", strategy="simple", slippage=5,
        signals_path=None, journal=False):
    import numpy as np
    import pandas as pd
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("use a new run directory")
    output.mkdir(parents=True)
    paths = minute_paths(root, start, end)
    if not paths:
        raise ValueError("no real minute data")
    packets = {}
    if strategy == "full":
        if not signals_path:
            raise ValueError("full strategy needs explicit timestamped signals")
        for r in read_jsonl(signals_path):
            s = read_signal(r)
            date = s.valid_from.date().isoformat()
            if date in packets:
                raise ValueError("duplicate daily signal")
            packets[date] = s
        missing = set(paths) - set(packets)
        if missing:
            raise ValueError(f"missing signals for replay dates: {sorted(missing)}")
        symbols = sorted({c.symbol for s in packets.values() for c in s.candidates})
    else:
        symbols = ["920000.BSE"]
    b = PaperBroker([Instrument(s) for s in symbols], costs=Costs(slippage_bps=slippage),
                    journal=output / "events.jsonl" if journal else None)
    controller = SimpleMomentumStrategy(b, symbols[0]) if strategy == "simple" else TargetStrategy(b)
    data = root / "training_runs/systematic_20261026/data"
    with np.load(data / "market.npz") as archive:
        m = {k: archive[k] for k in archive.files}
    cal = json.loads((data / "calendar.json").read_text(encoding="utf-8"))
    ids = json.loads((data / "stocks.json").read_text(encoding="utf-8"))
    sid = {s[2:] + ".BSE": i for i, s in enumerate(ids)}
    sources, counts, missing_quotes = [], 0, []
    empty_minutes = 0
    quarantined = []
    for day, path in sorted(paths.items()):
        at = datetime.fromisoformat(day + "T09:00:00+08:00")
        b.advance(at)
        di = cal.index(day)
        for s in symbols:
            si = sid[s]
            dividend, ratio = float(m["dividend"][di, si]), float(m["split"][di, si])
            if dividend or ratio != 1:
                b.corporate_action(day + ":" + s, s, at, dividend, ratio)
            # Previous day's actual close is only used as the 09:00 valuation.
            # It has no executable bid/ask and cannot create/fill any order.
            close = float(m["close"][di - 1, si])
            if s not in b.quotes and np.isfinite(close) and close > 0:
                b.quotes[s] = Tick(s, at, at, close, 0, 0, 0, 0, 0, "initial:" + day)
        b.advance(datetime.fromisoformat(day + "T09:30:00+08:00"))
        if strategy == "full":
            controller.on_signal(packets[day])
        active_symbols = symbols if strategy == "simple" else [s for s in symbols
            if controller.targets.get(s, 0) > 0 or b.position(s).quantity]
        frame = pd.read_parquet(path)
        subset = frame[frame.code.astype(str).isin([s.split('.')[0] for s in active_symbols])]
        empty_minutes += int(subset[["open", "high", "low", "close", "volume", "amount"]].isna().all(axis=1).sum())
        frame, issues = quarantine_minutes(subset)
        quarantined.append(issues)
        seen_today = set()
        for event in minute_events(frame, active_symbols, "start", "shares"):
            b.on_tick(event)
            counts += 1
            seen_today.add(event.symbol)
        # This simulator has no verified official daily limit prices. Synthetic
        # bars do NOT infer them from a flat 30% rule or a rounded prev-close.
        missing_quotes.extend(dict(date=day, symbol=s) for s in symbols
                              if b.position(s).quantity and s not in seen_today)
        b.advance(datetime.fromisoformat(day + "T15:01:00+08:00"))
        b.cancel_all(); b.flush(); b.mark()
        sources.append(dict(date=day, file=str(path.relative_to(root)), sha256=file_sha(path)))
    pd.DataFrame(b.curve).to_csv(output / "equity.csv", index=False)
    pd.DataFrame([asdict(f) for f in b.fills]).to_csv(output / "fills.csv", index=False)
    pd.DataFrame([asdict(o) for o in b.orders.values()]).to_csv(output / "orders.csv", index=False)
    quality = pd.concat(quarantined, ignore_index=True)
    quality.to_csv(output / "quarantined_minutes.csv", index=False)
    report = dict(strategy=strategy, costs=asdict(b.costs), rules=asdict(controller.rules),
        metrics=metrics(b.curve, b.initial), messages=counts, orders=len(b.orders), fills=len(b.fills),
        fees=sum(f.fee for f in b.fills), final_cash=b.cash,
        cash_identity_error=abs(b.cash - b.initial - b.cash_flows), input_files=sources,
        empty_minute_placeholders=empty_minutes, quarantine_counts={str(k): int(v) for k, v in quality.reason.value_counts().items()},
        missing_held_quotes=missing_quotes, signal_sha256=file_sha(signals_path) if signals_path else None,
        data_mode="REAL_MINUTE_SYNTHETIC_QUOTES", native_deltaone_tested=False, live_ready=False,
        limitations=["Minute start timestamps assumed from existing dataset; vendor confirmation pending",
            "Artificial one-cent spread; no real depth, limit prices, queue priority or cancelled-trade corrections",
            "Conservative first fill is second later minute; source minute start is unavailable until end",
            "Corporate actions from historical research snapshot; intraday availability not vendor-attested",
            "Research signals have assumed delivery timestamps; no untouched profitability test",
            "Missing held quotes carry last known valuation; not evidence of a tradable market"])
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--start", default="2026-07-01")
    p.add_argument("--end", default="2026-09-30")
    p.add_argument("--strategy", choices=["simple", "full"], default="simple")
    p.add_argument("--slippage", type=float, default=5)
    p.add_argument("--signals", dest="signals_path")
    p.add_argument("--journal", action="store_true")
    args = vars(p.parse_args())
    r = run(**args)
    print(json.dumps({k: r[k] for k in ("strategy", "messages", "orders", "fills", "metrics", "live_ready")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
