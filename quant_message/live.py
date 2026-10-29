"""Fail-closed broker port for the supplied DeltaOne template.

Real broker accounts must be dedicated to this strategy. Queries must be complete
before start; ambiguous/recovered orders halt rather than being resubmitted.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path

from .domain import Costs, Fill, Instrument, Order, Position, SH, stamp
from .paper import PaperBroker
from .history import ReplayActions


def close_failed_start(function):
    from functools import wraps
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception:
            if hasattr(self, "lock"):
                self.lock.close()
            raise
    return wrapped


class LiveBroker(PaperBroker):
    @close_failed_start
    def __init__(self, owner, instruments, state_path, account_id, capital, costs=Costs()):
        from vnpy.trader.constant import Direction
        if os.environ.get("QUANT_ENABLE_LIVE") != "YES" or not account_id:
            raise RuntimeError("live requires explicit QUANT_ENABLE_LIVE=YES and dedicated account_id")
        self.owner = owner
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive OS lock survives crashes correctly: stale files are harmless,
        # but a live second process cannot acquire the same strategy account.
        self.lock = self.state_path.with_suffix(".lock").open("a+")
        if os.name == "nt":
            import msvcrt
            self.lock.seek(0); self.lock.write("0"); self.lock.flush(); self.lock.seek(0)
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else None
        if prior and (prior.get("unresolved_orders") or not prior.get("clean_stop")):
            raise RuntimeError("restart blocked: reconcile prior orders/trades with broker before a fresh state file")
        account = owner.get_equity_account()
        if not account or str(getattr(account, "accountid", "")) != account_id:
            raise RuntimeError("get_equity_account must return confirmed AccountData for the configured account")
        available = float(account.available)
        balance = float(account.balance)
        if not all(math.isfinite(v) and v > 0 for v in (available, balance)):
            raise RuntimeError("invalid broker account snapshot")
        if float(account.frozen) != 0:
            raise RuntimeError("startup account has frozen funds; reconcile outstanding orders first")
        main = getattr(owner.engine, "main_engine", None)
        if main is None or not callable(getattr(main, "get_all_active_orders", None)):
            raise RuntimeError("engine main_engine active-order reconciliation capability unavailable")
        if main.get_all_active_orders():
            raise RuntimeError("startup requires confirmed absence of active broker orders")
        # Gateway schemas differ: caller must attest that balance is cash, not
        # total securities NAV, and that query-completion events were received.
        readiness_path = os.environ.get("QUANT_BROKER_READINESS", "")
        readiness = json.loads(Path(readiness_path).read_text()) if readiness_path else {}
        required = ("query_complete", "cash_balance_semantics_verified", "dedicated_account", "gateway_bse_verified")
        at = datetime.now(SH)
        if readiness.get("account_id") != account_id or not all(readiness.get(k) is True for k in required):
            raise RuntimeError("missing verified broker snapshot/gateway contract")
        if not 0 <= (at - stamp(readiness["observed_at"])).total_seconds() <= 120:
            raise RuntimeError("broker readiness snapshot is stale")
        if abs(balance - available) > 1e-5:
            raise RuntimeError("account balance/available cash discrepancy")
        super().__init__(instruments, min(capital, available), costs)
        self.trade_ids = set()
        self.terminal = {}
        self.acknowledged = {}
        self.failed = False
        self.now = at
        self.day = at.date()
        for symbol in self.instruments:
            p = owner.get_position(symbol, Direction.LONG)
            if p and p.volume:
                # Do not silently adopt inherited holdings with unknown attribution,
                # dividend basis or per-strategy capital. Recovery requires a broker
                # reconciler; fresh deployments start from flat strategy inventory.
                raise RuntimeError("startup position is nonzero; reconcile/adopt holdings before deployment")
        self.persist(False)

    def persist(self, clean):
        unresolved = [asdict(o) for o in self.orders.values() if o.active or
                      self.acknowledged.get(o.id, o.filled) > o.filled]
        payload = dict(clean_stop=clean and not unresolved and not self.failed,
            unresolved_orders=unresolved, trades=sorted(self.trade_ids), cash=self.cash,
            positions={k: asdict(v) for k, v in self.positions.items()}, observed_at=self.now,
            keys=self.keys)
        temp = self.state_path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str)
            f.flush(); os.fsync(f.fileno())
        temp.replace(self.state_path)

    def pending(self, symbol):
        return [self.orders[k] for k in self.active_ids if self.orders[k].symbol == symbol]

    def submit(self, key, symbol, side, limit, quantity, expires_at):
        if self.failed:
            raise RuntimeError("broker reconciliation failed; trading halted")
        if key in self.keys:
            return self.keys[key]
        # Persist the intent BEFORE crossing the network boundary. A crash after
        # broker acceptance but before return is never retried automatically.
        self.persist(False)
        oid = super().submit(key, symbol, side, limit, quantity, expires_at)
        o = self.orders[oid]
        if not o.active:
            self.persist(False)
            return oid
        o.status = "submitting"
        self.persist(False)
        self.sending = True
        self.early_events = []
        try:
            gateway_id = self.owner.buy(symbol, limit, quantity) if side == "buy" else self.owner.sell(symbol, limit, quantity)
            if not isinstance(gateway_id, str) or not gateway_id:
                raise RuntimeError("gateway returned no order id")
        except Exception:
            self.sending = False
            self.failed = True
            self.persist(False)
            raise
        self.events.clear()  # Never expose internal PAPER ids to the controller.
        self.orders.pop(oid); self.active_ids.discard(oid)
        o.id = gateway_id
        self.orders[gateway_id] = o
        self.active_ids.add(gateway_id)
        self.keys[key] = gateway_id
        self.sending = False
        for kind, event in self.early_events:
            getattr(self, "on_gateway_" + kind)(event)
        self.early_events = []
        self.persist(False)
        return gateway_id

    def cancel(self, oid, reason="cancelled"):
        o = self.orders[oid]
        if o.active and o.status != "cancel_pending":
            o.status = "cancel_pending"
            o.reason = reason
            # Cash/shares remain frozen until BOTH order and trade reports agree.
            self.persist(False)
            # Native engines may synchronously acknowledge cancellation. Set
            # pending BEFORE dispatch so the final callback state is preserved.
            self.owner.cancel_order(oid)

    def on_tick(self, tick):
        identity = (tick.symbol, tick.sequence)
        if identity in self.seen:
            return
        if tick.symbol in self.last_times and tick.at < self.last_times[tick.symbol]:
            self.failed = True
            raise RuntimeError("out of order live quote")
        self.advance(tick.available_at)
        self.seen.add(identity); self.last_times[tick.symbol] = tick.at
        self.quotes[tick.symbol] = tick
        if self.strategy and not self.failed:
            self.strategy.on_tick(tick)
        self.persist(False)

    def _settle_terminal(self, oid):
        o = self.orders[oid]
        acknowledged = self.acknowledged.get(oid, 0)
        if oid in self.terminal and o.filled == acknowledged:
            o.status = self.terminal[oid]
            self.active_ids.discard(oid)
        elif o.filled > acknowledged and oid in self.terminal:
            self.failed = True
            raise RuntimeError("terminal order volume disagrees with trade reports")
        elif o.status != "cancel_pending":
            o.status = "partial" if o.filled else "accepted"
        if self.strategy:
            self.strategy.on_order(replace(o))

    def on_gateway_order(self, event):
        if getattr(self, "sending", False):
            self.early_events.append(("order", event)); return
        if event.vt_orderid not in self.orders:
            self.failed = True
            self.persist(False)
            raise RuntimeError("unattributed broker order; manual reconciliation required")
        oid = event.vt_orderid
        o = self.orders[oid]
        if event.volume != o.quantity or event.vt_symbol != o.symbol:
            self.failed = True
            raise RuntimeError("order identity mismatch")
        acknowledged = int(event.traded)
        if acknowledged < self.acknowledged.get(oid, 0):
            return  # delayed duplicate status cannot roll back progress
        self.acknowledged[oid] = acknowledged
        if not event.is_active():
            self.terminal[oid] = "filled" if acknowledged == o.quantity else "cancelled"
        self._settle_terminal(oid)
        self.persist(False)

    def on_gateway_trade(self, event):
        if getattr(self, "sending", False):
            self.early_events.append(("trade", event)); return
        if event.vt_tradeid in self.trade_ids:
            return
        oid = event.vt_orderid
        if oid not in self.orders:
            self.failed = True
            raise RuntimeError("unattributed trade")
        o = self.orders[oid]
        q, px = int(event.volume), float(event.price)
        if event.volume != q or q <= 0 or not math.isfinite(px) or px <= 0 or o.filled + q > o.quantity or event.vt_symbol != o.symbol:
            self.failed = True
            raise RuntimeError("invalid trade report")
        from vnpy.trader.constant import Direction
        if (event.direction == Direction.LONG) != (o.side == "buy"):
            self.failed = True
            raise RuntimeError("trade direction mismatch")
        amount = px * q
        fee = self.costs.total(o.notional + amount, o.side) - o.fees
        p = self.position(o.symbol)
        if o.side == "buy":
            p.cost = (p.cost * p.quantity + amount) / (p.quantity + q)
            p.quantity += q; p.today += q; p.peak = max(p.peak, px)
            flow = -amount - fee
        else:
            if q > p.sellable:
                self.failed = True
                raise RuntimeError("broker sell exceeds reconciled T+1 inventory")
            p.quantity -= q
            if not p.quantity: p.cost = p.peak = 0
            flow = amount - fee
        self.cash += flow; self.cash_flows += flow
        o.filled += q; o.notional += amount; o.fees += fee
        self.trade_ids.add(event.vt_tradeid)
        t = Fill(event.vt_tradeid, oid, o.symbol, o.side, px, q, fee, self.now)
        self.fills.append(t)
        if self.strategy: self.strategy.on_trade(t)
        self._settle_terminal(oid)
        self.check(); self.persist(False)

    def close(self):
        self.cancel_all()
        self.persist(True)
        self.lock.close()


class NativeReplayBroker(ReplayActions,LiveBroker):
    """Real DeltaOne backtesting orders, with the same risk/reconciliation code.

    This port is only accepted with the installed BacktestingEngine type.
    It cannot be used to bypass LiveBroker's gateway account checks.
    """
    def __init__(self, owner, instruments, capital, costs=Costs()):
        from vnpy_deltaone.backtesting import BacktestingEngine
        if not isinstance(owner.engine, BacktestingEngine):
            raise TypeError("native replay requires DeltaOne BacktestingEngine")
        PaperBroker.__init__(self, instruments, capital, costs)
        self.owner = owner
        self.trade_ids = set()
        self.terminal = {}
        self.acknowledged = {}
        self.failed = False
        self.configure_actions(owner)

    def persist(self, clean):
        pass

    def close(self):
        self.cancel_all()
