"""Read-only platform inspection and genuine community EventEngine smoke test."""
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys


def inspect():
    packages = {}
    for name in ("vnpy", "vnpy_deltaone", "vnpy_level2", "vnpy_xtp", "vnpy_tora", "vnpy_hxsec"):
        try:
            spec = importlib.util.find_spec(name)
            packages[name] = dict(importable=spec is not None, location=spec.origin if spec else None)
            try: packages[name]["version"] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: pass
        except Exception as exc:
            packages[name] = dict(importable=False, error=type(exc).__name__)
    native_contract = False
    native_error = ""
    if packages["vnpy_deltaone"]["importable"]:
        try:
            from vnpy_deltaone import StrategyTemplate
            from .deltaone import MessageStockStrategy
            required = ("send_order", "cancel_order", "get_position", "on_tick", "on_order", "on_trade")
            native_contract = all(callable(getattr(StrategyTemplate, k, None)) for k in required)
            native_contract &= issubclass(MessageStockStrategy, StrategyTemplate)
        except Exception as exc:
            native_error = type(exc).__name__ + ": " + str(exc)
    community_smoke = False
    smoke_error = ""
    try:
        from threading import Event as Latch
        from datetime import datetime
        from vnpy.event import Event, EventEngine
        from vnpy.trader.constant import Exchange
        from vnpy.trader.object import TickData
        from .domain import SH
        latch, received = Latch(), []
        engine = EventEngine()
        def handler(event):
            received.append(event.data.vt_symbol)
            latch.set()
        engine.register("eTick", handler)
        engine.start()
        try:
            engine.put(Event("eTick", TickData(symbol="920000", exchange=Exchange.BSE,
                datetime=datetime.now(SH), last_price=10., gateway_name="CONTRACT_SMOKE")))
            community_smoke = latch.wait(5) and received == ["920000.BSE"]
        finally: engine.stop()
    except Exception as exc:
        smoke_error = type(exc).__name__ + ": " + str(exc)
    return dict(python=sys.executable, python_version=platform.python_version(), system=platform.system(),
        packages=packages, community_event_tick_smoke=community_smoke, community_error=smoke_error,
        deltaone_template_import=native_contract, deltaone_error=native_error,
        commercial_directory_readable=os.access("/home/vnpy", os.R_OK | os.X_OK) if os.name != "nt" else None,
        connected_to_gateway=False, live_ready=False)


if __name__ == "__main__":
    print(json.dumps(inspect(), ensure_ascii=False, indent=2))
