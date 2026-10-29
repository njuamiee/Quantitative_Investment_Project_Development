"""VeighNa plugin matching the user-provided vnpy_deltaone.StrategyTemplate API.

No community CTA/portfolio template substitution is used in production.
Import create_strategy_class for contract testing without the proprietary package.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .data import read_jsonl, read_signal
from .domain import Costs, Instrument, SH, Tick, stamp
from .paper import PaperBroker
from .strategy import RiskRules, SimpleMomentumStrategy, TargetStrategy


def create_strategy_class(base):
    class MessageStockStrategy(base):
        author = "Quantitative Investment Project"
        mode = "paper"
        strategy_kind = "simple"
        capital = 1_000_000.
        signal_path = ""
        corporate_action_path = ""
        state_path = "runtime/stock_strategy.json"
        account_id = ""
        risk_json = "{}"
        slippage_bps = 5.
        replay_latency_ms = 100
        replay_participation = 1.
        naive_gateway_timezone = "Asia/Shanghai"
        equity = 0.
        error = ""
        parameters = ["mode", "strategy_kind", "capital", "signal_path", "corporate_action_path", "state_path", "account_id",
                      "risk_json", "slippage_bps", "naive_gateway_timezone", "replay_latency_ms", "replay_participation"]
        variables = ["equity", "error"]

        def __init__(self, name, engine, vt_symbols, parameters):
            # Supplied template mutates self.variables with insert(). Instance
            # copies prevent duplicate inited/trading entries across strategies.
            self.variables = list(type(self).variables)
            self.parameters = list(type(self).parameters)
            super().__init__(name, engine, vt_symbols, parameters)
            self.volume_cache = {}
            self.packets = []
            self.timer_engine = None
            self.signal_file_version = None
            self.signal_checked_second = None
            self.tick_audit = {}

        def on_init(self):
            if self.mode not in ("paper", "live", "backtest", "paper_backtest") or self.strategy_kind not in ("simple", "full"):
                raise ValueError("invalid strategy mode")
            instruments = [Instrument(s, step=1 if s.endswith(".BSE") else 100) for s in self.vt_symbols]
            if self.strategy_kind == "full":
                self.packets = sorted([read_signal(r) for r in read_jsonl(self.signal_path)], key=lambda s: s.valid_from)
                if not self.packets:
                    raise ValueError("empty signal file")
                if self.mode == "live" and any(s.research_only for s in self.packets):
                    raise RuntimeError("historical research packets are forbidden in live mode")
            if self.mode == "paper":
                self.port = PaperBroker(instruments, self.capital, Costs(slippage_bps=self.slippage_bps))
            elif self.mode == 'paper_backtest':
                from .history import PaperReplayBroker
                self.port = PaperReplayBroker(self, instruments, self.capital,
                    Costs(slippage_bps=self.slippage_bps,latency_ms=self.replay_latency_ms,participation=self.replay_participation))
            elif self.mode == "backtest":
                from .live import NativeReplayBroker
                self.port = NativeReplayBroker(self, instruments, self.capital, Costs(slippage_bps=self.slippage_bps,
                    latency_ms=self.replay_latency_ms,participation=self.replay_participation))
            else:
                from .live import LiveBroker
                self.port = LiveBroker(self, instruments, self.state_path, self.account_id, self.capital)
            self.controller = SimpleMomentumStrategy(self.port, self.vt_symbols[0]) if self.strategy_kind == "simple" else TargetStrategy(
                self.port, RiskRules(**json.loads(self.risk_json)))
            self.inited = True
            self.write_log("消息驱动股票策略初始化；mode=" + self.mode)

        def on_start(self):
            if not self.inited:
                raise RuntimeError("strategy not initialized")
            if self.mode == "live":
                # Timer watchdog is mandatory: expiry must work when quotes stop.
                self.timer_engine = getattr(self.engine, "event_engine", None)
                if self.timer_engine is None:
                    main = getattr(self.engine, "main_engine", None)
                    self.timer_engine = getattr(main, "event_engine", None)
                if self.timer_engine is None:
                    raise RuntimeError("event timer unavailable")
                from vnpy.event import EVENT_TIMER
                self.timer_engine.register(EVENT_TIMER, self.on_timer)
            self.trading = True
            self.controller.active = True

        def on_stop(self):
            self.controller.stop()
            if self.timer_engine:
                from vnpy.event import EVENT_TIMER
                self.timer_engine.unregister(EVENT_TIMER, self.on_timer)
            self.trading = False
            if self.mode in ("live", "backtest"):
                self.port.close()

        def on_timer(self, event):
            if not self.trading:
                return
            now = datetime.now(SH)
            try:
                self.port.advance(now)
                for symbol, q in self.port.quotes.items():
                    if (now - q.at).total_seconds() > self.controller.rules.max_quote_age_seconds:
                        for o in self.port.pending(symbol):
                            self.port.cancel(o.id)
            except Exception as exc:
                self.fail(exc)

        def fail(self, exc):
            self.error = f"{type(exc).__name__}: {exc}"
            self.controller.active = False
            self.port.cancel_all()
            self.write_log(self.error)
            self.put_event()

        def on_tick(self, tick):
            if not self.trading or self.error:
                return
            try:
                at = tick.datetime
                if at.tzinfo is None:
                    if self.naive_gateway_timezone != "Asia/Shanghai":
                        raise ValueError("unverified gateway timezone")
                    at = at.replace(tzinfo=SH)
                at = stamp(at)
                received = at if self.mode in ("backtest", "paper_backtest") else datetime.now(SH)
                if at > received:
                    raise ValueError("exchange timestamp is ahead of receiver clock")
                key = (tick.vt_symbol, at.date())
                previous = self.volume_cache.get(key, float(tick.volume))
                if tick.volume < previous:
                    raise ValueError("intraday cumulative volume reset")
                increment = float(tick.volume) - previous
                self.volume_cache[key] = float(tick.volume)
                identity = hashlib.sha256(str((tick.vt_symbol, at, tick.volume, tick.last_price,
                    tick.bid_price_1, tick.ask_price_1, tick.bid_volume_1, tick.ask_volume_1)).encode()).hexdigest()[:24]
                t = Tick(tick.vt_symbol, at, received, float(tick.last_price), float(tick.bid_price_1),
                    float(tick.ask_price_1), float(tick.bid_volume_1), float(tick.ask_volume_1), increment,
                    identity, float(tick.limit_up), float(tick.limit_down))
                if self.mode in ('backtest','paper_backtest'):
                    from .database_io import tick_row
                    day=at.date().isoformat()
                    bucket=self.tick_audit.setdefault(day,dict(count=0,hash=hashlib.sha256()))
                    bucket['count']+=1
                    bucket['hash'].update((json.dumps(tick_row(tick),separators=(',',':'))+'\n').encode())
                # A file update is read as a whole atomic replacement, never while
                # training is writing it. Availability checks still gate each packet.
                if self.strategy_kind == "full":
                    second = received.replace(microsecond=0)
                    if self.signal_checked_second != second:
                        stat = Path(self.signal_path).stat()
                        version = (stat.st_mtime_ns, stat.st_size)
                        if self.signal_file_version != version:
                            self.packets = [read_signal(r) for r in read_jsonl(self.signal_path)]
                            self.signal_file_version = version
                        self.signal_checked_second = second
                    eligible = [s for s in self.packets if s.valid_from <= received < s.expires_at and s.available_at <= received]
                    if eligible:
                        s = max(eligible, key=lambda s: s.available_at)
                        if self.mode == "live" and s.research_only:
                            raise RuntimeError("research-only signal rejected")
                        self.port.advance(received)
                        if s.id not in self.controller.signal_ids:
                            self.controller.on_signal(s)
                self.port.on_tick(t)
                self.equity = self.port.equity()
                self.put_event()
            except Exception as exc:
                self.fail(exc)

        def on_order(self, order):
            if self.mode in ("live", "backtest"):
                try: self.port.on_gateway_order(order)
                except Exception as exc: self.fail(exc)

        def on_trade(self, trade):
            if self.mode in ("live", "backtest"):
                try:
                    if self.mode == "backtest" and trade.datetime:
                        at = trade.datetime
                        if at.tzinfo is None: at = at.replace(tzinfo=SH)
                        self.port.advance(at)
                    self.port.on_gateway_trade(trade)
                except Exception as exc: self.fail(exc)

    return MessageStockStrategy


try:
    from vnpy_deltaone import StrategyTemplate
except ImportError:
    # The platform must NOT accidentally discover a stand-in class as executable.
    MessageStockStrategy = None
else:
    MessageStockStrategy = create_strategy_class(StrategyTemplate)
