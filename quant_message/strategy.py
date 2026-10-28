"""No broker/model dependency in strategy decisions; the same callbacks run everywhere."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
import math

from .domain import Signal, continuous_session


@dataclass(frozen=True)
class RiskRules:
    top_n: int = 20
    max_new_buys: int = 20
    target_exposure: float = .5
    single_cap: float = .1
    industry_cap: float = .3  # Fraction of strategy NAV, not fraction of invested capital.
    size_cap: float = .5
    inverse_vol: bool = False
    volatility_target: float = 0
    pool_fraction: float = 1
    use_market_cap: bool = True
    rebound_only: bool = False
    minute_weight: float = 0
    entry_stop: float = .08
    trailing_stop: float = .10
    portfolio_stop: float = .20
    cooldown_sessions: int = 10
    order_lifetime_seconds: int = 300
    limit_buffer_bps: float = 30
    max_quote_age_seconds: int = 120
    max_orders_per_session: int = 2000

    def __post_init__(self):
        if min(self.top_n, self.max_new_buys, self.order_lifetime_seconds,
               self.max_quote_age_seconds, self.max_orders_per_session) <= 0:
            raise ValueError("positive risk counts required")
        for v in (self.target_exposure, self.single_cap, self.industry_cap, self.size_cap,
                  self.pool_fraction, self.entry_stop, self.trailing_stop, self.portfolio_stop):
            if not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("invalid risk fraction")
        if not all(math.isfinite(v) and v >= 0 for v in (self.volatility_target, self.minute_weight,
                self.limit_buffer_bps, self.cooldown_sessions)):
            raise ValueError("invalid risk input")


class TargetStrategy:
    def __init__(self, broker, rules=RiskRules()):
        self.broker = broker
        broker.strategy = self
        self.rules = rules
        self.targets = {}
        self.signal = None
        self.signal_ids = set()
        self.pending_stops = set()
        self.banned_until = {}
        self.session = 0
        self.day = None
        self.new_buys = set()
        self.orders_today = 0
        self.high_water = broker.equity()
        self.halted = False
        self.active = True
        self.last_order = {}
        self.transitions = []

    def roll_day(self):
        if self.day != self.broker.now.date():
            self.day = self.broker.now.date()
            self.session += 1
            self.new_buys.clear()
            self.orders_today = 0

    def on_signal(self, signal: Signal):
        signal.validate(self.broker.now)
        if signal.id in self.signal_ids:
            return
        if self.signal and signal.available_at < self.signal.available_at:
            raise ValueError("older signal cannot replace active signal")
        if any(c.symbol not in self.broker.instruments for c in signal.candidates):
            raise ValueError("signal requires missing contract")
        self.roll_day()
        self.broker.cancel_all()
        self.signal_ids.add(signal.id)
        self.signal = signal
        r = self.rules
        capital = self.broker.equity()
        exposure = r.target_exposure
        if r.use_market_cap:
            exposure = min(exposure, signal.market_cap)
        if r.volatility_target:
            exposure = min(exposure, r.volatility_target / signal.market_vol)
        if self.halted:
            exposure = 0.
        n = min(r.top_n, 1) if r.use_market_cap and signal.market_cap <= .05 else r.top_n
        candidates = [c for c in signal.candidates if c.pool_rank <= r.pool_fraction
                      and self.banned_until.get(c.symbol, 0) <= self.session
                      and c.symbol not in self.pending_stops and (c.rebound or not r.rebound_only)]
        candidates.sort(key=lambda c: (-(c.score + r.minute_weight * c.minute_increment), c.symbol))
        candidates = candidates[:n]
        raw = [(1 / c.volatility if r.inverse_vol else 1.) for c in candidates]
        total = sum(raw) or 1
        industries, sizes = defaultdict(float), defaultdict(float)
        targets = {s: 0 for s in self.broker.instruments}
        weights = {}
        for c, v in zip(candidates, raw):
            # Unknown group values share a common bucket, never bypass constraints.
            industry, size = c.industry or "UNKNOWN", c.size_group or "UNKNOWN"
            w = max(0., min(exposure * v / total, r.single_cap,
                          r.industry_cap - industries[industry], r.size_cap - sizes[size]))
            i = self.broker.instruments[c.symbol]
            q = math.floor(capital * w / c.reference / i.step) * i.step
            if q < i.min_buy:
                q = 0
            targets[c.symbol] = q
            weights[c.symbol] = w
            industries[industry] += w
            sizes[size] += w
        self.targets = targets
        self.broker.record("signal", {"id": signal.id, "model": signal.model_id,
                                      "weights": weights, "targets": targets,
                                      "research_only": signal.research_only})

    def stop(self):
        # Cancellation is sent while the template still permits trading.
        self.broker.cancel_all()
        self.active = False

    def on_order(self, order):
        self.transitions.append((order.id, order.status, order.filled))

    def on_trade(self, trade):
        if trade.side == "buy":
            self.new_buys.add(trade.symbol)

    def on_tick(self, tick):
        if not self.active:
            return
        self.roll_day()
        b, r = self.broker, self.rules
        if (b.now - tick.at).total_seconds() > r.max_quote_age_seconds:
            for o in b.pending(tick.symbol):
                b.cancel(o.id)
            return
        equity = b.equity()
        self.high_water = max(self.high_water, equity)
        if r.portfolio_stop and equity <= self.high_water * (1 - r.portfolio_stop):
            self.halted = True
            self.pending_stops.update(s for s in b.instruments if b.position(s).quantity)
            self.targets = {s: 0 for s in b.instruments}
            b.cancel_all()
        p = b.position(tick.symbol)
        if p.quantity:
            p.peak = max(p.peak, tick.last)
            if (r.entry_stop and tick.last <= p.cost * (1 - r.entry_stop)) or (
                    r.trailing_stop and tick.last <= p.peak * (1 - r.trailing_stop)):
                self.pending_stops.add(tick.symbol)
                self.banned_until[tick.symbol] = max(self.banned_until.get(tick.symbol, 0),
                                                    self.session + r.cooldown_sessions)
        elif tick.symbol in self.pending_stops:
            self.pending_stops.remove(tick.symbol)
        stopped = tick.symbol in self.pending_stops or self.halted
        if not stopped and (not self.signal or b.now >= self.signal.expires_at):
            for o in b.pending(tick.symbol):
                b.cancel(o.id)
            return
        if not continuous_session(b.now):
            return
        target = 0 if stopped else self.targets.get(tick.symbol, 0)
        pending = b.pending(tick.symbol)
        if pending:
            if stopped:
                for o in pending:
                    if o.side == "buy":
                        b.cancel(o.id)
            return
        if self.orders_today >= r.max_orders_per_session:
            return
        # One attempt per symbol per quote. No submission/rejection callback loops.
        if self.last_order.get(tick.symbol) == tick.sequence:
            return
        delta = target - p.quantity
        if not delta:
            return
        buy = delta > 0
        if buy and (self.halted or len(self.new_buys) >= r.max_new_buys and tick.symbol not in self.new_buys):
            return
        price = tick.ask if buy else tick.bid
        if price <= 0 or (buy and tick.ask_size <= 0) or (not buy and tick.bid_size <= 0):
            return
        i = b.instruments[tick.symbol]
        price *= 1 + (1 if buy else -1) * r.limit_buffer_bps / 10000
        price = (math.ceil(price / i.tick_size - 1e-9) if buy else math.floor(price / i.tick_size + 1e-9)) * i.tick_size
        if tick.limit_up:
            price = min(price, math.floor(tick.limit_up / i.tick_size + 1e-9) * i.tick_size)
        if tick.limit_down:
            price = max(price, math.ceil(tick.limit_down / i.tick_size - 1e-9) * i.tick_size)
        if price <= 0:
            return
        if buy:
            # Prevent price gaps from increasing position size beyond NAV cap.
            cap = max(0, math.floor(equity * r.single_cap / price) - p.quantity)
            remaining_exposure = max(0., r.target_exposure * equity - (equity - b.cash) - b.reserved_cash())
            if self.signal and r.use_market_cap:
                remaining_exposure = min(remaining_exposure,
                    max(0., self.signal.market_cap * equity - (equity - b.cash) - b.reserved_cash()))
            cash = min(b.available_cash(), remaining_exposure)
            quantity = min(delta, cap, math.floor(max(0., cash - b.costs.min_commission) /
                        (price * (1 + (b.costs.commission_bps + b.costs.transfer_bps) / 10000))))
            quantity = quantity // i.step * i.step
            if quantity < i.min_buy:
                return
        else:
            quantity = min(-delta, p.sellable if i.t_plus_one else p.quantity)
            quantity = quantity // i.step * i.step
            if quantity <= 0:
                return
        expiry = b.now + timedelta(seconds=r.order_lifetime_seconds)
        if not stopped:
            expiry = min(expiry, self.signal.expires_at)
        key = f"{self.signal.id if self.signal else 'stop'}:{tick.symbol}:{tick.sequence}:{'B' if buy else 'S'}"
        oid = b.submit(key, tick.symbol, "buy" if buy else "sell", price, quantity, expiry)
        if buy and b.orders[oid].active:
            self.new_buys.add(tick.symbol)  # Reserve slot before an asynchronous fill.
        self.last_order[tick.symbol] = tick.sequence
        self.orders_today += 1


class SimpleMomentumStrategy(TargetStrategy):
    """V1: one stock, short/long moving averages of already received prices, 10% NAV."""
    def __init__(self, broker, symbol, fast=3, slow=8):
        if not 1 <= fast < slow or len(broker.instruments) != 1:
            raise ValueError("simple strategy requires one stock and fast < slow")
        super().__init__(broker, RiskRules(top_n=1, max_new_buys=1, target_exposure=.1,
            entry_stop=0, trailing_stop=0, portfolio_stop=0, use_market_cap=False))
        self.symbol, self.fast = symbol, fast
        self.history = deque(maxlen=slow)
        self.minute = None
        self.minute_last = None

    def on_tick(self, tick):
        if tick.symbol != self.symbol:
            return
        observed = None
        if tick.synthetic:
            observed = tick.last  # Already completed minute supplied by importer.
        else:
            minute = tick.at.replace(second=0, microsecond=0)
            if self.minute is not None and minute > self.minute:
                observed = self.minute_last  # Closed previous minute, no current-minute close.
            self.minute, self.minute_last = minute, tick.last
        if observed is not None:
            self.history.append(observed)
        if observed is not None and len(self.history) == self.history.maxlen:
            from .domain import Candidate, Signal
            up = sum(list(self.history)[-self.fast:]) / self.fast > sum(self.history) / len(self.history)
            at = self.broker.now
            s = Signal(f"simple:{tick.sequence}", tick.at, at, at, at + timedelta(minutes=5),
                at - timedelta(seconds=1), at - timedelta(seconds=2),
                (Candidate(tick.symbol, 1, tick.last, .2, "single", "single"),) if up else ())
            # Preserve pending orders until TTL or a direction change; repeated
            # same-side ticks must not cancel every order before its next fill.
            desired = bool(self.targets.get(tick.symbol, 0))
            if self.signal is None or desired != up or at >= self.signal.expires_at:
                self.on_signal(s)
        super().on_tick(tick)
