"""Actual vn.py objects + simulated engine. Not a proprietary DeltaOne engine test."""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vnpy.trader.constant import Direction, Exchange, Offset, Status
from vnpy.trader.object import AccountData, OrderData, TickData, TradeData
from quant_message.deltaone import create_strategy_class
from quant_message.domain import Instrument, SH
from quant_message.live import LiveBroker
from quant_message.strategy import TargetStrategy
from test_message_execution import T, tick


class ProbeBase:
    """Only the template surface supplied by the user, not a substitute engine."""
    def __init__(self, name, engine, vt_symbols, parameters):
        self.name, self.engine, self.vt_symbols = name, engine, vt_symbols
        self.inited = self.trading = False
        self.variables.insert(0, "inited"); self.variables.insert(1, "trading")
        self.get_equity_account = engine.get_equity_account
        self.get_futures_account = lambda: None
        for k, v in parameters.items():
            if k in self.parameters: setattr(self, k, v)

    def write_log(self, msg): pass
    def put_event(self): pass
    def buy(self, *args): return self.engine.send_order(*args)
    def sell(self, *args): return self.engine.send_order(*args)
    def cancel_order(self, oid): self.engine.cancels.append(oid)
    def get_position(self, symbol, direction): return None


class Engine:
    def __init__(self):
        self.sent, self.cancels = [], []
        self.main_engine = SimpleNamespace(get_all_active_orders=lambda: [])

    def get_equity_account(self):
        return AccountData(accountid="TEST", balance=1e6, frozen=0, gateway_name="TEST")

    def send_order(self, *args):
        self.sent.append(args)
        return f"TEST.{len(self.sent)}"


class AdapterTests(unittest.TestCase):
    def test_parameter_variables_do_not_accumulate_between_instances(self):
        cls = create_strategy_class(ProbeBase)
        a, b = cls("a", Engine(), ["920000.BSE"], {}), cls("b", Engine(), ["920000.BSE"], {})
        self.assertEqual(a.variables.count("inited"), 1)
        self.assertEqual(b.variables.count("inited"), 1)
        self.assertNotIn("inited", cls.variables)

    def test_paper_mode_never_calls_real_send_order(self):
        cls = create_strategy_class(ProbeBase)
        e = Engine(); s = cls("a", e, ["920000.BSE"], {})
        s.on_init(); s.on_start()
        now = datetime(2026, 7, 1, 10, tzinfo=SH)
        for i in range(15):
            at = now + timedelta(minutes=i)
            with patch("quant_message.deltaone.datetime") as clock:
                clock.now.return_value = at
                q = TickData(symbol="920000", exchange=Exchange.BSE, datetime=at,
                    last_price=10 + i * .001, bid_price_1=10, ask_price_1=10.01,
                    bid_volume_1=10000, ask_volume_1=10000, volume=i * 10000, gateway_name="TEST")
                s.on_tick(q)
        self.assertEqual(e.sent, [])
        self.assertEqual(s.error, "")
        self.assertTrue(s.port.orders)
        s.on_stop()

    def test_live_requires_explicit_enable(self):
        with patch.dict(os.environ, {"QUANT_ENABLE_LIVE": "NO"}):
            with self.assertRaises(RuntimeError):
                LiveBroker(None, [Instrument("920000.BSE")], "unused", "TEST", 1e6)


class BrokerReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name)
        readiness = dict(account_id="TEST", observed_at=datetime.now(SH).isoformat(),
            query_complete=True, cash_balance_semantics_verified=True, dedicated_account=True, gateway_bse_verified=True)
        (self.path / "ready.json").write_text(json.dumps(readiness))
        self.env = patch.dict(os.environ, {"QUANT_ENABLE_LIVE": "YES", "QUANT_BROKER_READINESS": str(self.path / "ready.json")})
        self.env.start()
        self.engine = Engine()
        self.owner = ProbeBase("probe", self.engine, ["920000.BSE"], {}) if False else SimpleNamespace(
            engine=self.engine, get_equity_account=self.engine.get_equity_account,
            get_position=lambda *args: None, buy=self.engine.send_order, sell=self.engine.send_order,
            cancel_order=lambda oid: self.engine.cancels.append(oid))
        self.b = LiveBroker(self.owner, [Instrument("920000.BSE")], self.path / "state.json", "TEST", 1e6)
        # Deterministic test clock; the gateway itself is simulated.
        self.b.now = T; self.b.day = T.date()
        self.b.on_tick(tick())
        self.oid = self.b.submit("intent", "920000.BSE", "buy", 11, 1000, T + timedelta(minutes=5))

    def tearDown(self):
        self.b.close(); self.env.stop(); self.temp.cleanup()

    def report(self, volume=1000, traded=1000, status=Status.ALLTRADED):
        return OrderData(symbol="920000", exchange=Exchange.BSE, orderid="1", volume=volume,
                         traded=traded, status=status, gateway_name="TEST")

    def fill(self, tradeid="1", quantity=1000):
        return TradeData(symbol="920000", exchange=Exchange.BSE, orderid="1", tradeid=tradeid,
            direction=Direction.LONG, offset=Offset.OPEN, price=10, volume=quantity, datetime=T,
            gateway_name="TEST")

    def test_order_report_before_trade_keeps_reservation(self):
        self.b.on_gateway_order(self.report())
        self.assertIn(self.oid, self.b.active_ids)
        self.assertGreater(self.b.reserved_cash(), 0)
        self.b.on_gateway_trade(self.fill())
        self.assertNotIn(self.oid, self.b.active_ids)
        self.assertEqual(self.b.position("920000.BSE").quantity, 1000)

    def test_duplicate_trade_is_idempotent(self):
        self.b.on_gateway_trade(self.fill()); cash = self.b.cash
        self.b.on_gateway_trade(self.fill()); self.assertEqual(self.b.cash, cash)
        self.b.on_gateway_order(self.report()); self.assertFalse(self.b.active_ids)

    def test_cancel_ack_with_partial_fill_waits_for_trade(self):
        self.b.cancel(self.oid)
        self.assertGreater(self.b.reserved_cash(), 0)
        self.b.on_gateway_order(self.report(traded=100, status=Status.CANCELLED))
        self.assertIn(self.oid, self.b.active_ids)
        self.b.on_gateway_trade(self.fill(quantity=100))
        self.assertNotIn(self.oid, self.b.active_ids)
        self.assertEqual(self.b.position("920000.BSE").quantity, 100)

    def test_restart_with_unresolved_intent_blocked(self):
        self.b.lock.close()
        with self.assertRaises(RuntimeError):
            LiveBroker(self.owner, [Instrument("920000.BSE")], self.path / "state.json", "TEST", 1e6)

    def test_synchronous_cancel_ack_keeps_terminal_status(self):
        self.owner.cancel_order=lambda oid:self.b.on_gateway_order(self.report(traded=0,status=Status.CANCELLED))
        self.b.cancel(self.oid)
        self.assertEqual(self.b.orders[self.oid].status,'cancelled')
        self.assertFalse(self.b.orders[self.oid].active)
        self.assertEqual(self.b.reserved_cash(),0)

    def test_live_quote_does_not_generate_simulated_fill(self):
        self.b.on_tick(tick(1))
        self.assertEqual(self.b.fills, [])


if __name__ == "__main__": unittest.main()
