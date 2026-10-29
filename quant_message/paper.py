"""Deterministic message broker. A decision can fill only on a LATER market event."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from datetime import timedelta
import json
import math
from pathlib import Path

from .domain import Costs, Fill, Instrument, Order, Position, Tick, continuous_session, stamp


class PaperBroker:
    def __init__(self, instruments, capital=1_000_000., costs=Costs(), journal=None):
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError("invalid capital")
        self.instruments = {i.symbol: i for i in instruments}
        self.cash = self.initial = capital
        self.dividend_receivable = 0.
        self.costs = costs
        self.positions = {s: Position() for s in self.instruments}
        self.quotes = {}
        self.orders = {}
        self.active_ids = set()
        self.keys = {}
        self.fills = []
        self.events = deque()
        self.seen = set()
        self.last_times = {}
        self.action_ids = set()
        self.now = None
        self.day = None
        self.strategy = None
        self.journal = Path(journal) if journal else None
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            if self.journal.exists():
                raise FileExistsError("new replay journal required; never silently overwrite")
        self.curve = []
        self.signal_audit = []
        self.cash_flows = 0.

    def record(self, kind, payload):
        if kind == 'signal':
            self.signal_audit.append(dict(at=self.now.isoformat(), **payload))
        if self.journal:
            with self.journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": kind, "at": self.now, "data": payload},
                                   default=str, ensure_ascii=False, allow_nan=False) + "\n")

    def position(self, symbol):
        return self.positions[symbol]

    def equity(self):
        return self.cash + self.dividend_receivable + sum(p.quantity * self.quotes[s].last for s, p in self.positions.items()
                               if p.quantity and s in self.quotes)

    def reserved_cash(self):
        return sum(o.limit * o.remaining + max(0, self.costs.total(
            o.notional + o.limit * o.remaining, "buy") - o.fees)
            for o in (self.orders[k] for k in self.active_ids) if o.side == "buy")

    def available_cash(self):
        return max(0., self.cash - self.reserved_cash())

    def pending(self, symbol):
        return [self.orders[k] for k in sorted(self.active_ids, key=lambda k: int(k.split('.')[-1]))
                if self.orders[k].symbol == symbol]

    def submit(self, key, symbol, side, limit, quantity, expires_at):
        if key in self.keys:
            return self.keys[key]
        i = self.instruments[symbol]
        reason = ""
        if not self.now or not continuous_session(self.now):
            reason = "outside continuous session"
        elif side not in ("buy", "sell") or not math.isfinite(limit) or limit <= 0:
            reason = "invalid order"
        elif int(quantity) != quantity or quantity <= 0 or quantity % i.step:
            reason = "invalid quantity/step"
        elif abs(limit / i.tick_size - round(limit / i.tick_size)) > 1e-6:
            reason = "invalid price tick"
        elif stamp(expires_at) <= self.now:
            reason = "expired order"
        elif side == "buy" and quantity < i.min_buy:
            reason = "below minimum buy"
        elif side == "buy" and limit * quantity + self.costs.total(limit * quantity, side) > self.available_cash() + 1e-8:
            reason = "insufficient available cash"
        elif side == "sell":
            p = self.position(symbol)
            frozen = sum(o.remaining for o in self.pending(symbol) if o.side == "sell")
            if quantity > (p.sellable if i.t_plus_one else p.quantity) - frozen:
                reason = "T+1 or insufficient available shares"
        quote = self.quotes.get(symbol)
        if not reason and (not quote or (self.now - quote.at).total_seconds() > 120):
            reason = "missing/stale quote"
        if not reason and ((quote.limit_up and limit > quote.limit_up + 1e-8) or
                           (quote.limit_down and limit < quote.limit_down - 1e-8)):
            reason = "outside exchange price limits"
        oid = f"PAPER.{len(self.orders) + 1}"
        o = Order(oid, key, symbol, side, limit, int(quantity), self.now, stamp(expires_at),
                  status="rejected" if reason else "accepted", reason=reason)
        self.orders[oid] = o
        if o.active:
            self.active_ids.add(oid)
        self.keys[key] = oid
        self.events.append(("order", replace(o)))
        self.record("order", asdict(o))
        return oid

    def cancel(self, oid, reason="cancelled"):
        o = self.orders[oid]
        if o.active:
            o.status, o.reason = "cancelled", reason
            self.active_ids.discard(oid)
            self.events.append(("order", replace(o)))
            self.record("order", asdict(o))

    def cancel_all(self):
        for oid in list(self.active_ids):
            self.cancel(oid)

    def flush(self):
        # Callbacks are queued, never recursively invoked inside submit/cancel.
        while self.events:
            kind, event = self.events.popleft()
            if self.strategy:
                getattr(self.strategy, "on_" + kind)(event)

    def advance(self, now):
        now = stamp(now)
        if self.now and now < self.now:
            raise ValueError("event stream moved backwards")
        if self.day != now.date():
            self.cancel_all()
            for p in self.positions.values():
                p.today = 0
            self.day = now.date()
        self.now = now
        for o in [self.orders[k] for k in list(self.active_ids)]:
            if o.active and now >= o.expires_at:
                self.cancel(o.id, "expired")
        self.flush()

    def on_tick(self, tick: Tick):
        if tick.symbol not in self.instruments:
            raise ValueError("unknown instrument")
        identity = (tick.symbol, tick.sequence)
        if identity in self.seen:
            return
        if tick.symbol in self.last_times and tick.at < self.last_times[tick.symbol]:
            raise ValueError("out-of-order exchange observation")
        self.advance(tick.available_at)
        self.seen.add(identity)
        self.last_times[tick.symbol] = tick.at
        self.quotes[tick.symbol] = tick
        self.record("tick", asdict(tick))
        # A bar proxy is only evidence of the COMPLETED minute. It cannot fill
        # an order sent during that minute; require submission before interval start.
        earliest = tick.at - timedelta(minutes=1) if tick.synthetic else tick.at
        capacity = math.floor(tick.volume * self.costs.participation)
        book = {"buy": math.floor(tick.ask_size), "sell": math.floor(tick.bid_size)}
        tradable = continuous_session(tick.at - timedelta(microseconds=1) if tick.synthetic else tick.at)
        tradable = tradable and (self.now - tick.at).total_seconds() <= 120
        for o in list(self.pending(tick.symbol)):
            if not tradable or o.submitted_at + timedelta(milliseconds=self.costs.latency_ms) >= earliest:
                continue
            if capacity <= 0 or book[o.side] <= 0:
                continue
            askbuy = o.side == "buy"
            px = tick.ask if askbuy else tick.bid
            if px <= 0:
                continue
            if (askbuy and tick.limit_up and tick.last >= tick.limit_up) or (
                    not askbuy and tick.limit_down and tick.last <= tick.limit_down):
                continue  # Queue priority at locked limits is unknown: conservative no fill.
            i = self.instruments[tick.symbol]
            # Recorded float32 quotes such as 14.350000381 are representations
            # of an exchange tick, not a price one tick higher. Normalize only
            # tiny quote noise BEFORE applying economic slippage.
            nearest = round(px / i.tick_size) * i.tick_size
            if abs(px - nearest) <= min(i.tick_size * .005, abs(px) * 1e-7):
                px = nearest
            px *= 1 + (1 if askbuy else -1) * self.costs.slippage_bps / 10000
            px = (math.ceil(px / i.tick_size - 1e-9) if askbuy else math.floor(px / i.tick_size + 1e-9)) * i.tick_size
            if (askbuy and px > o.limit + 1e-8) or (not askbuy and px < o.limit - 1e-8):
                continue
            if (tick.limit_up and px > tick.limit_up + 1e-8) or (tick.limit_down and px < tick.limit_down - 1e-8):
                continue
            q = min(o.remaining, capacity, book[o.side])
            q = q // i.step * i.step
            if q <= 0:
                continue
            amount = q * px
            fee = self.costs.total(o.notional + amount, o.side) - o.fees
            if askbuy and amount + fee > self.cash + 1e-7:
                raise RuntimeError("cash reservation invariant broken")
            p = self.position(o.symbol)
            if askbuy:
                p.cost = (p.cost * p.quantity + amount) / (p.quantity + q)
                p.quantity += q
                p.today += q
                p.peak = max(p.peak, px)
                flow = -amount - fee
            else:
                if q > p.sellable and i.t_plus_one:
                    raise RuntimeError("T+1 invariant broken")
                p.quantity -= q
                if not p.quantity:
                    p.cost = p.peak = 0.
                flow = amount - fee
            self.cash += flow
            self.cash_flows += flow
            o.filled += q
            o.notional += amount
            o.fees += fee
            o.status = "filled" if not o.remaining else "partial"
            if not o.active:
                self.active_ids.discard(o.id)
            fill = Fill(f"PAPER.T{len(self.fills) + 1}", o.id, o.symbol, o.side, px, q, fee, self.now)
            self.fills.append(fill)
            self.events.extend((("trade", fill), ("order", replace(o))))
            self.record("trade", asdict(fill))
            capacity -= q  # ALL orders share the same event's liquidity budget.
            book[o.side] -= q
        self.flush()
        if self.strategy:
            self.strategy.on_tick(tick)
        self.flush()
        self.check()

    def corporate_action(self, identity, symbol, at, dividend=0., ratio=1., credit_cash=True):
        self.advance(at)
        if identity in self.action_ids:
            return
        if not math.isfinite(dividend) or dividend < 0 or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("invalid corporate action")
        self.action_ids.add(identity)
        for o in self.pending(symbol):
            self.cancel(o.id, "corporate action")
        p = self.position(symbol)
        shares = p.quantity * ratio
        if abs(shares - round(shares)) > 1e-6:
            raise ValueError("fractional corporate action requires cash-in-lieu record")
        payment = p.quantity * dividend
        if credit_cash:
            self.cash += payment
            self.cash_flows += payment
        else:
            # Ex-date entitlement is not proof that cash has settled. Unverified
            # gross dividends contribute to NAV, never to order buying power.
            self.dividend_receivable += payment
        p.quantity = int(round(shares))
        p.today = int(round(p.today * ratio))
        if p.cost:
            p.cost = (p.cost - dividend) / ratio
            p.peak = (p.peak - dividend) / ratio
        # Adjust stale valuation even when today's stock is suspended.
        if symbol in self.quotes:
            q = self.quotes[symbol]
            self.quotes[symbol] = replace(q, last=(q.last - dividend) / ratio,
                                         bid=0, ask=0, bid_size=0, ask_size=0)
        self.record("corporate_action", dict(id=identity, symbol=symbol, dividend=dividend, ratio=ratio,
                                             credit_cash=credit_cash))
        self.flush()
        self.check()

    def check(self):
        if abs(self.cash - self.initial - self.cash_flows) > 1e-5 or self.cash < -1e-7:
            raise RuntimeError("cash accounting invariant broken")
        if any(p.quantity < 0 or p.today < 0 or p.today > p.quantity for p in self.positions.values()):
            raise RuntimeError("share accounting invariant broken")

    def mark(self):
        row = dict(at=self.now.isoformat(), equity=self.equity(), cash=self.cash,
                   exposure=1 - self.cash / self.equity(), fills=len(self.fills))
        self.curve.append(row)
        return row
