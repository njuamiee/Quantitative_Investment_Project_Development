"""Actual installed DeltaOne engine validation; isolated input store and BSE fix."""
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import math


def json_safe(value):
    if isinstance(value, dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def enable_bse_equities():
    import vnpy_deltaone.backtesting as backtesting
    import vnpy_deltaone.utility as utility
    original = backtesting.is_equity
    if original("920000.BSE"):
        return False
    def equity(symbol):
        return symbol.endswith(".BSE") or original(symbol)
    # Process-local compatibility patch; no proprietary binary or other service changes.
    backtesting.is_equity = equity
    utility.is_equity = equity
    if not backtesting.is_equity("920000.BSE"):
        raise RuntimeError("BSE compatibility patch failed")
    return True


def run(data_dir, output, kind="simple", latency=100, slippage=.01):
    import pandas as pd
    import vnpy_deltaone.backtesting as native
    from vnpy.trader.object import TickData
    from vnpy.trader.constant import Exchange
    from .deltaone import MessageStockStrategy
    if int(pd.__version__.split('.')[0]) >= 3:
        raise RuntimeError("installed DeltaOne statistics require project pandas 2.3.3; add project deps to PYTHONPATH")
    pd.options.mode.copy_on_write = False
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    data_dir = Path(data_dir)
    patch = enable_bse_equities()
    # CSV+gzip is an explicit portable contract for proprietary Python runtimes
    # without Arrow wheels; identical normalized rows, no lossy type guessing.
    csv_path = data_dir / "ticks.csv.gz"
    parquet_path = data_dir / "ticks.parquet"
    if parquet_path.exists():
        f = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        f = pd.read_csv(csv_path, dtype={"code": str}, parse_dates=["datetime"])
    else:
        raise FileNotFoundError('normalized tick input required')
    if kind == "simple": f = f[f.code == "920000"]
    symbols = sorted(f.code.unique())
    lookup = {}
    for day, g in f.groupby(f.datetime.dt.date):
        ticks = []
        for r in g.sort_values(["datetime", "code"], kind="stable").itertuples():
            t = TickData(symbol=r.code, exchange=Exchange.BSE, datetime=r.datetime.to_pydatetime(),
                last_price=float(r.price), volume=float(r.accvolume), turnover=float(r.accturnover),
                pre_close=float(r.pre_close), limit_up=float(r.HighLimit), limit_down=float(r.LowLimit),
                gateway_name="RESEARCH_RAW")
            for i in range(1, 6):
                setattr(t, f"bid_price_{i}", float(getattr(r, f"bid{i}")))
                setattr(t, f"ask_price_{i}", float(getattr(r, f"ask{i}")))
                setattr(t, f"bid_volume_{i}", float(getattr(r, f"bsize{i}")))
                setattr(t, f"ask_volume_{i}", float(getattr(r, f"asize{i}")))
            ticks.append(t)
        lookup[day] = ticks
    original_loader = native.load_tick_data
    native.load_tick_data = lambda requested, dt: [t for t in lookup.get(dt.date(), []) if t.vt_symbol in requested]
    e = native.BacktestingEngine()
    vt_symbols = [s + ".BSE" for s in symbols]
    e.set_parameters(vt_symbols=vt_symbols, start=min(f.datetime).normalize().to_pydatetime(),
        end=(max(f.datetime).normalize() + pd.Timedelta(days=1)).to_pydatetime(),
        rates={s: .0005 for s in vt_symbols}, slippages={s: slippage for s in vt_symbols},
        sizes={s: 1 for s in vt_symbols}, priceticks={s: .01 for s in vt_symbols}, capital=1_000_000,
        use_level2=False, order_latency=latency, tick_depth=5, queue_depth=5)
    settings = dict(mode="backtest", strategy_kind=kind, signal_path=str(data_dir / "signals.jsonl"),
                    corporate_action_path=str(data_dir / "actions.json"),
                    risk_json=json.dumps(dict(top_n=5, max_new_buys=5)))
    try:
        e.add_strategy(MessageStockStrategy, settings)
        e.run_backtesting()
        frame = e.calculate_result()
        stats = e.calculate_statistics(output=False) if frame is not None and not frame.empty else {}
    finally:
        native.load_tick_data = original_loader
    port = e.strategy.port
    if e.strategy.error: raise RuntimeError(e.strategy.error)
    if e.exceptions: raise RuntimeError(str(e.exceptions))
    checks = []
    for fill in port.fills:
        order = port.orders[fill.order_id]
        if fill.at <= order.submitted_at:
            raise AssertionError("native matching filled on decision tick")
        if fill.side == "buy" and fill.price > order.limit + 1e-6:
            raise AssertionError("native fill exceeded buy limit")
        if fill.side == "sell" and fill.price < order.limit - 1e-6:
            raise AssertionError("native fill exceeded sell limit")
    port.check()
    fills = pd.DataFrame([asdict(t) for t in port.fills])
    fills.to_csv(output / "fills.csv", index=False)
    pd.DataFrame([asdict(o) for o in port.orders.values()]).to_csv(output / "orders.csv", index=False)
    if frame is not None: frame.to_csv(output / "native_equity.csv")
    from .portfolio_audit import reconstruct
    marks = f.sort_values('datetime').groupby([f.datetime.dt.date, 'code']).tail(1).copy()
    marks['date'] = marks.datetime.dt.strftime('%Y-%m-%d')
    marks['symbol'] = marks.code + '.BSE'
    actions = json.loads((data_dir/'actions.json').read_text())
    audited, metrics = reconstruct(fills, marks[['date','symbol','price']], actions,
                                   slippage=slippage)
    cash_error = abs(float(audited.cash.iloc[-1]) - port.cash)
    nav_error = abs(float(audited.shadow_equity.iloc[-1]) - port.equity())
    if cash_error > 1e-5 or nav_error > 1e-5:
        raise AssertionError(f'independent ledger mismatch: cash={cash_error}, NAV={nav_error}')
    audited.to_csv(output/'audited_equity.csv',index=False)
    import numpy as np
    valid = (frame is not None and len(frame)==len(audited) and 'balance' in frame
             and np.isfinite(frame.balance).all())
    vendor_error = None
    if valid:
        vendor_error=float(np.max(np.abs(frame.balance.to_numpy()-audited.engine_cost_price_equity.to_numpy())))
        valid=vendor_error<.01
    result = dict(kind=kind, engine="installed vnpy_deltaone.BacktestingEngine", bse_patch=patch,
        symbols=len(vt_symbols), input_ticks=len(f), sessions=len(lookup), native_orders=len(e.all_orders),
        native_trades=len(e.trades), reconciled_fills=len(port.fills), error=e.strategy.error,
        cash_identity_error=abs(port.cash - port.initial - port.cash_flows),
        latency_ms=latency, engine_slippage_absolute=slippage, native_statistics=stats,
        native_statistics_valid=bool(valid), native_statistics_max_error=vendor_error,
        audited_metrics=metrics, independent_cash_error=cash_error, independent_nav_error=nav_error,
        same_tick_fills=0, real_broker_connected=False, live_ready=False,
        limitations=["L1 book snapshot matching, not true L2 queue replay",
            "Native fee/slippage accounting and shadow ledger fees are separate and intentionally labelled",
            "Use audited_equity.csv: vendor multi-symbol aggregate can be NaN/missing sessions",
            "Gross dividends are receivables only; payment date and withholding are not verified",
            "Historical receive timestamps unavailable; use exchange timestamps plus configured order latency"])
    result=json_safe(result)
    (output / "report.json").write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2,allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "native_statistics"}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["simple", "full"], default="simple")
    p.add_argument("--latency", type=int, default=100)
    p.add_argument("--slippage", type=float, default=.01)
    run(**vars(p.parse_args()))
