from dataclasses import replace
from datetime import datetime, timedelta
import unittest

from quant_message.domain import Candidate, Costs, Instrument, SH, Signal, Tick
from quant_message.paper import PaperBroker
from quant_message.strategy import RiskRules, SimpleMomentumStrategy, TargetStrategy

T = datetime(2026, 7, 1, 9, 31, tzinfo=SH)
S = "920000.BSE"


def tick(n=0, price=10., volume=100000, day=0, **kwargs):
    at = T + timedelta(seconds=n, days=day)
    return Tick(S, at, at, price, price, price, volume, volume, volume,
                f"{day}:{n}", **kwargs)


def broker(capital=100000., **costs):
    return PaperBroker([Instrument(S)], capital, Costs(**costs))


def order(b, key="one", side="buy", quantity=1000, price=11.):
    return b.submit(key, S, side, price, quantity, b.now + timedelta(minutes=10))


def signal(at=T, candidates=None, **kw):
    c = (Candidate(S, 1, 10, .2, "tech", "small"),) if candidates is None else candidates
    return Signal("s:" + at.isoformat(), at - timedelta(days=1), at - timedelta(hours=1), at,
                  at + timedelta(hours=5), at - timedelta(days=1), at - timedelta(days=2), tuple(c), **kw)


class ExecutionTests(unittest.TestCase):
    def test_float32_quote_noise_does_not_add_a_price_tick(self):
        b=broker(slippage_bps=0);b.on_tick(tick(price=14.35))
        order(b,price=14.35)
        b.on_tick(tick(1,price=14.350000381469727))
        self.assertEqual(b.fills[0].price,14.35)

    def test_real_slippage_is_not_removed_by_quote_normalization(self):
        b=broker(slippage_bps=1);b.on_tick(tick(price=14.35))
        order(b,price=14.35)
        b.on_tick(tick(1,price=14.350000381469727))
        self.assertEqual(b.fills,[])

    def test_decision_cannot_fill_same_quote(self):
        b = broker(slippage_bps=0)
        b.on_tick(tick())
        oid = order(b)
        b.on_tick(tick())
        self.assertEqual(b.orders[oid].filled, 0)
        b.on_tick(tick(1))
        self.assertEqual(b.orders[oid].filled, 1000)

    def test_latency_is_respected(self):
        b = broker(latency_ms=1500, slippage_bps=0)
        b.on_tick(tick()); oid = order(b)
        b.on_tick(tick(1)); self.assertEqual(b.orders[oid].filled, 0)
        b.on_tick(tick(2)); self.assertGreater(b.orders[oid].filled, 0)

    def test_completed_bar_not_reused_for_execution(self):
        b = broker(slippage_bps=0)
        b.on_tick(tick(synthetic=True)); oid = order(b)
        b.on_tick(tick(60, synthetic=True)); self.assertEqual(b.orders[oid].filled, 0)
        b.on_tick(tick(120, synthetic=True)); self.assertGreater(b.orders[oid].filled, 0)

    def test_price_limits_and_empty_queue(self):
        b = broker(slippage_bps=0); b.on_tick(tick())
        oid = order(b, price=13.)
        b.on_tick(tick(1, price=13, limit_up=13))
        self.assertEqual(b.orders[oid].filled, 0)
        b.on_tick(replace(tick(2), ask_size=0))
        self.assertEqual(b.orders[oid].filled, 0)

    def test_all_orders_share_participation(self):
        b = broker(slippage_bps=0); b.on_tick(tick())
        order(b, "one", quantity=1000); order(b, "two", quantity=1000)
        b.on_tick(tick(1, volume=10000))
        self.assertEqual(sum(f.quantity for f in b.fills), 100)

    def test_partial_fill_min_commission_charged_once(self):
        b = broker(slippage_bps=0, commission_bps=0, transfer_bps=0)
        b.on_tick(tick()); oid = order(b, quantity=300)
        for n in (1, 2, 3): b.on_tick(tick(n, volume=10000))
        self.assertEqual(b.orders[oid].fees, 5)
        self.assertEqual(b.cash, 96995)

    def test_cash_reserved_across_symbols_and_orders(self):
        b = broker(capital=12000); b.on_tick(tick())
        first = order(b); second = order(b, "two")
        self.assertTrue(b.orders[first].active)
        self.assertEqual(b.orders[second].status, "rejected")
        b.cancel(first)
        third = order(b, "three")
        self.assertTrue(b.orders[third].active)

    def test_idempotency_and_cancel_release(self):
        b = broker(); b.on_tick(tick()); oid = order(b)
        self.assertEqual(oid, order(b)); self.assertEqual(len(b.orders), 1)
        b.cancel(oid); b.cancel(oid)
        self.assertEqual(b.reserved_cash(), 0)
        b.on_tick(tick(1)); self.assertFalse(b.fills)

    def test_t1_and_sell_freeze(self):
        b = broker(slippage_bps=0); b.on_tick(tick()); order(b)
        b.on_tick(tick(1))
        rejected = order(b, "sell_today", "sell", price=9)
        self.assertEqual(b.orders[rejected].status, "rejected")
        b.on_tick(tick(day=1))
        accepted = order(b, "sell_next", "sell", price=9)
        rejected = order(b, "double_sell", "sell", price=9)
        self.assertTrue(b.orders[accepted].active)
        self.assertEqual(b.orders[rejected].status, "rejected")
        b.on_tick(tick(1, day=1)); self.assertEqual(b.position(S).quantity, 0)

    def test_cash_invariant_slippage_and_fee(self):
        def run(slip):
            b = broker(slippage_bps=slip); b.on_tick(tick()); order(b)
            b.on_tick(tick(1)); b.on_tick(tick(day=1))
            order(b, "exit", "sell", price=9)
            b.on_tick(tick(1, day=1)); b.check()
            return b.cash
        self.assertLess(run(20), run(0))

    def test_expiry_without_market_ticks(self):
        b = broker(); b.on_tick(tick()); oid = order(b)
        b.advance(T + timedelta(minutes=11))
        self.assertEqual(b.orders[oid].status, "cancelled")

    def test_out_of_order_fails(self):
        b = broker(); b.on_tick(tick(2))
        with self.assertRaises(ValueError): b.on_tick(tick(1))

    def test_suspended_corporate_action_adjusts_valuation(self):
        b = broker(slippage_bps=0); b.on_tick(tick()); order(b); b.on_tick(tick(1))
        before = b.equity()
        b.corporate_action("ca", S, T + timedelta(days=1), .2, 1.2)
        self.assertEqual(b.position(S).quantity, 1200)
        self.assertAlmostEqual(before, b.equity())
        b.corporate_action("ca", S, T + timedelta(days=1), .2, 1.2)
        self.assertEqual(b.position(S).quantity, 1200)

    def test_lunch_no_order(self):
        b = broker(); b.on_tick(tick(3600 * 3))
        self.assertEqual(b.orders[order(b)].status, "rejected")


class StrategyTests(unittest.TestCase):
    def test_future_signal_and_label_maturity_rejected(self):
        b = broker(); b.advance(T); s = TargetStrategy(b)
        for invalid in (replace(signal(), available_at=T + timedelta(seconds=1)),
                        replace(signal(), labels_end=T), replace(signal(), expires_at=T)):
            with self.assertRaises(ValueError): s.on_signal(invalid)

    def test_simple_future_mutation_leaves_prefix_identical(self):
        def run(prices):
            b = broker(slippage_bps=0); SimpleMomentumStrategy(b, S)
            for n, p in enumerate(prices): b.on_tick(tick(n * 60, p, synthetic=True))
            return [(f.at, f.side, f.price, f.quantity) for f in b.fills]
        prefix = [10 + .01 * i for i in range(30)]
        a, z = run(prefix + [10.] * 10), run(prefix + [30.] * 10)
        cutoff = T + timedelta(minutes=29)
        self.assertTrue(a)
        self.assertEqual([v for v in a if v[0] <= cutoff], [v for v in z if v[0] <= cutoff])

    def test_hmm_bear_limits_one_stock_and_five_percent(self):
        b = PaperBroker([Instrument(S), Instrument("920001.BSE")]); b.advance(T)
        s = TargetStrategy(b)
        candidates = [Candidate(x, 1, 10, .2, "tech", "small") for x in b.instruments]
        s.on_signal(signal(candidates=candidates, market_cap=.05))
        self.assertEqual(sum(q > 0 for q in s.targets.values()), 1)
        self.assertLessEqual(sum(s.targets.values()) * 10, b.equity() * .05)

    def test_sector_size_and_single_caps(self):
        instruments = [Instrument(f"920{i:03d}.BSE") for i in range(8)]
        b = PaperBroker(instruments); b.advance(T)
        s = TargetStrategy(b, RiskRules(industry_cap=.12, size_cap=.15))
        s.on_signal(signal(candidates=[Candidate(i.symbol, 1, 10, .2, "same", "same") for i in instruments]))
        self.assertLessEqual(sum(s.targets.values()) * 10, b.equity() * .12 + 1e-6)

    def test_stop_waits_for_t1_then_exits(self):
        b = broker(slippage_bps=0); b.on_tick(tick()); s = TargetStrategy(b)
        s.on_signal(signal()); b.on_tick(tick(1)); b.on_tick(tick(2))
        self.assertGreater(b.position(S).quantity, 0)
        b.on_tick(tick(3, 8.)); self.assertIn(S, s.pending_stops)
        self.assertFalse(any(o.side == "sell" and o.active for o in b.orders.values()))
        b.on_tick(tick(0, 8., day=1)); b.on_tick(tick(1, 8., day=1))
        self.assertEqual(b.position(S).quantity, 0)

    def test_stop_cancels_before_disabling(self):
        b = broker(); b.on_tick(tick()); s = TargetStrategy(b); s.on_signal(signal())
        b.on_tick(tick(1)); s.stop()
        self.assertFalse(any(o.active for o in b.orders.values()))
        self.assertFalse(s.active)

    def test_duplicate_signal_no_rebalance(self):
        b = broker(); b.on_tick(tick()); s = TargetStrategy(b); sg = signal()
        s.on_signal(sg); b.on_tick(tick(1)); before = list(b.orders)
        s.on_signal(sg); self.assertEqual(before, list(b.orders))
        self.assertTrue(any(o.active for o in b.orders.values()))


if __name__ == "__main__":
    unittest.main()
