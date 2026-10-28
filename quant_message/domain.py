"""Messages carry observation AND availability times; prices are unadjusted CNY."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from math import isfinite
from zoneinfo import ZoneInfo

SH = ZoneInfo("Asia/Shanghai")


def stamp(value: str | datetime) -> datetime:
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError("timezone required: use +08:00 for Chinese exchange data")
    return value.astimezone(SH)


def finite_positive(*values: float) -> bool:
    return all(isfinite(v) and v > 0 for v in values)


def continuous_session(t: datetime) -> bool:
    t = stamp(t).time()
    return time(9, 30) <= t < time(11, 30) or time(13) <= t < time(14, 57)


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float = .01
    min_buy: int = 100
    step: int = 1  # BSE: >=100 shares, then increments of 1; configure other boards.
    t_plus_one: bool = True

    def __post_init__(self):
        if not self.symbol.endswith((".BSE", ".SSE", ".SZSE")):
            raise ValueError("equity exchange required")
        if not finite_positive(self.tick_size) or self.min_buy <= 0 or self.step <= 0:
            raise ValueError("invalid instrument rules")


@dataclass(frozen=True)
class Tick:
    symbol: str
    at: datetime
    available_at: datetime
    last: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    volume: float  # Incremental shares; never cumulative gateway volume.
    sequence: str
    limit_up: float = 0
    limit_down: float = 0
    synthetic: bool = False

    def __post_init__(self):
        if stamp(self.at) > stamp(self.available_at):
            raise ValueError("quote available before observation")
        if not finite_positive(self.last) or not all(isfinite(v) and v >= 0 for v in
                (self.bid, self.ask, self.bid_size, self.ask_size, self.volume, self.limit_up, self.limit_down)):
            raise ValueError("invalid quote")
        if self.bid and self.ask and self.bid > self.ask:
            raise ValueError("crossed quote")
        if not self.sequence:
            raise ValueError("message identity required")


@dataclass(frozen=True)
class Candidate:
    symbol: str
    score: float
    reference: float
    volatility: float
    industry: str
    size_group: str
    pool_rank: float = 0
    rebound: bool = True
    minute_increment: float = 0


@dataclass(frozen=True)
class Signal:
    id: str
    observed_at: datetime
    available_at: datetime
    valid_from: datetime
    expires_at: datetime
    model_trained_at: datetime
    labels_end: datetime
    candidates: tuple[Candidate, ...]
    market_cap: float = 1
    market_vol: float = .2
    model_id: str = "simple"
    research_only: bool = True

    def validate(self, now: datetime):
        dates = [stamp(x) for x in (self.observed_at, self.available_at, self.valid_from,
                                   self.expires_at, self.model_trained_at, self.labels_end, now)]
        obs, avail, start, end, fit, labels, now = dates
        if not (labels < fit <= avail and obs <= avail <= start <= now < end):
            raise ValueError("signal unavailable, expired, or trained with future outcomes")
        if len({c.symbol for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate signal symbols")
        if not isfinite(self.market_cap) or not 0 <= self.market_cap <= 1:
            raise ValueError("invalid market exposure cap")
        if not finite_positive(self.market_vol):
            raise ValueError("invalid market volatility")
        for c in self.candidates:
            if not all(isfinite(v) for v in (c.score, c.minute_increment, c.pool_rank)):
                raise ValueError("invalid scores")
            if not finite_positive(c.reference, c.volatility) or not 0 <= c.pool_rank <= 1:
                raise ValueError("invalid candidate metadata")


@dataclass
class Order:
    id: str
    key: str
    symbol: str
    side: str
    limit: float
    quantity: int
    submitted_at: datetime
    expires_at: datetime
    filled: int = 0
    notional: float = 0
    fees: float = 0
    status: str = "accepted"
    reason: str = ""

    @property
    def active(self):
        return self.status in ("accepted", "partial", "submitting", "cancel_pending")

    @property
    def remaining(self):
        return self.quantity - self.filled


@dataclass(frozen=True)
class Fill:
    id: str
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: int
    fee: float
    at: datetime


@dataclass
class Position:
    quantity: int = 0
    today: int = 0
    cost: float = 0
    peak: float = 0

    @property
    def sellable(self):
        return self.quantity - self.today


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 3
    min_commission: float = 5
    sell_tax_bps: float = 5
    transfer_bps: float = .1
    slippage_bps: float = 5
    participation: float = .01
    latency_ms: int = 1

    def __post_init__(self):
        if not all(isfinite(v) and v >= 0 for v in (self.commission_bps, self.min_commission,
                self.sell_tax_bps, self.transfer_bps, self.slippage_bps, self.latency_ms)):
            raise ValueError("invalid costs")
        if not 0 < self.participation <= 1:
            raise ValueError("invalid participation")

    def total(self, notional: float, side: str) -> float:
        if notional <= 0:
            return 0.
        return max(self.min_commission, notional * self.commission_bps / 10000) + notional * (
            self.transfer_bps + (self.sell_tax_bps if side == "sell" else 0)) / 10000
