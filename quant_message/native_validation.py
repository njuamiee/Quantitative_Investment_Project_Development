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


def run(data_dir, output, kind="simple", latency=100, slippage=.01, execution="native", end_day=None, depth=5):
    import pandas as pd
    from .database_io import connect_database
    data_dir=Path(data_dir)
    manifest=json.loads((data_dir/'database_manifest.json').read_text())
    db=connect_database(manifest['database'])
    import vnpy_deltaone.backtesting as native
    from .deltaone import MessageStockStrategy
    if int(pd.__version__.split('.')[0])>=3:
        raise RuntimeError('project pandas 2.3.3 is required')
    pd.options.mode.copy_on_write=False
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    patch=enable_bse_equities()
    vt_symbols=['920000.BSE'] if kind=='simple' else manifest['symbols']
    days=[d for d in manifest['sessions'] if not end_day or d<=end_day]
    if not days:raise ValueError('empty replay period')
    # The installed function and its original database binding stay intact.
    loader=native.load_tick_data
    if loader.__globals__['database'] is not db:
        raise RuntimeError('commercial loader is bound to a different database')
    query_audit=[]
    execute=db.client.execute
    def traced_execute(query,*args,**kwargs):
        result=execute(query,*args,**kwargs)
        if isinstance(query,str) and query.lstrip().upper().startswith('SELECT') and 'tick_data' in query:
            params=kwargs.get('params',args[0] if args and isinstance(args[0],dict) else {})
            query_audit.append(dict(query=query,rows=len(result),parameters={k:v for k,v in params.items()
                if k in ('symbol','exchange','start','end','timezone')}))
        return result
    db.client.execute=traced_execute
    e=native.BacktestingEngine()
    e.set_parameters(vt_symbols=vt_symbols,start=datetime.fromisoformat(days[0]),
        end=datetime.fromisoformat(days[-1])+timedelta(days=1),
        rates={s:.0005 for s in vt_symbols},slippages={s:slippage for s in vt_symbols},
        sizes={s:1 for s in vt_symbols},priceticks={s:.01 for s in vt_symbols},capital=1_000_000,
        use_level2=False,order_latency=latency,tick_depth=depth,queue_depth=depth)
    settings = dict(mode="backtest" if execution=="native" else "paper_backtest",
                    slippage_bps=0.,replay_latency_ms=latency,replay_participation=1., strategy_kind=kind, signal_path=str(data_dir / "signals.jsonl"),
                    corporate_action_path=str(data_dir / "actions.json"),
                    risk_json=json.dumps(dict(top_n=5, max_new_buys=5)))
    try:
        e.add_strategy(MessageStockStrategy, settings)
        e.run_backtesting()
        frame = e.calculate_result() if execution=="native" else None
        stats = e.calculate_statistics(output=False) if frame is not None and not frame.empty else {}
    finally:
        db.client.execute=execute
    if native.load_tick_data is not loader:raise AssertionError("native loader was replaced")
    (output/"database_queries.json").write_text(json.dumps(query_audit,default=str,indent=2))
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
    # Independent end-of-day valuation also uses vn.py's original database
    # loader. These queries occur AFTER replay and never reach the strategy.
    mark_rows=[]
    for day in days:
        last={}
        for tick in loader(vt_symbols,datetime.fromisoformat(day)):
            last[tick.vt_symbol]=tick
        mark_rows.extend(dict(date=day,symbol=s,price=t.last_price) for s,t in last.items())
    marks=pd.DataFrame(mark_rows)
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
        symbols=len(vt_symbols), input_ticks=sum(a["count"] for a in e.strategy.tick_audit.values()), sessions=len(days), native_orders=len(e.all_orders),
        orders=len(port.orders), execution=execution, tick_depth=depth, database=manifest["database"],
        data_route="ClickHouse -> vn.py Database -> original DeltaOne loader -> engine -> strategy",
        original_loader_preserved=True, database_selects=len(query_audit),
        strategy_rules=asdict(e.strategy.controller.rules), shadow_costs=asdict(port.costs),
        source_sha256=manifest["source_sha256"],
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
    (output/"signal_audit.json").write_text(json.dumps(port.signal_audit,default=str,indent=2))
    (output/"tick_audit.json").write_text(json.dumps({d:dict(count=b["count"],sha256=b["hash"].hexdigest()) for d,b in e.strategy.tick_audit.items()},indent=2))
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
    p.add_argument("--execution",choices=["native","paper"],default="native")
    p.add_argument("--end-day");p.add_argument("--depth",type=int,default=5)
    run(**vars(p.parse_args()))
