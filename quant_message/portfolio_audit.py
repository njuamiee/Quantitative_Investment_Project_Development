"""Independent long-only ledger from fills and end-of-session raw quotes.

Never fill missing vendor PnL with zero: reconstruct cash and holdings instead.
Dividend entitlement is gross/unsettled and is excluded from buying power.
"""
from collections import defaultdict
import math

from .domain import Costs


def reconstruct(fills, marks, actions=(), capital=1_000_000., rate=.0005, slippage=.01):
    import pandas as pd
    fills = fills.copy()
    if not fills.empty:
        if fills.id.duplicated().any():
            raise ValueError('duplicate fill id')
        fills['date'] = pd.to_datetime(fills['at'], utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
    marks = marks.copy()
    marks['date'] = marks['date'].astype(str)
    if marks.duplicated(['date','symbol']).any():
        raise ValueError('duplicate closing quote')
    dates = sorted(marks.date.unique())
    if not fills.empty and not set(fills.date).issubset(dates):
        raise ValueError('fill date missing from valuation sessions')
    cash = native_cash = capital
    receivable = 0.
    holdings, prices, notionals, paid = defaultdict(int), {}, defaultdict(float), defaultdict(float)
    rows = []
    for day in dates:
        today = defaultdict(int)
        for a in actions:
            if a['date'] != day:
                continue
            s, div, ratio = a['symbol'], float(a['dividend']), float(a['ratio'])
            if not math.isfinite(div) or div < 0 or not math.isfinite(ratio) or ratio <= 0:
                raise ValueError('invalid corporate action')
            receivable += holdings[s] * div
            q = holdings[s] * ratio
            if abs(q - round(q)) > 1e-7:
                raise ValueError('fractional split needs cash-in-lieu')
            holdings[s] = int(round(q))
            if s in prices:
                prices[s] = (prices[s] - div) / ratio
        daily = fills[fills.date == day] if not fills.empty else fills
        for f in daily.itertuples():
            q, px = int(f.quantity), float(f.price)
            if q != f.quantity or q <= 0 or not math.isfinite(px) or px <= 0 or f.side not in ('buy','sell'):
                raise ValueError('invalid fill')
            amount = q * px
            notionals[f.order_id] += amount
            fee = Costs().total(notionals[f.order_id], f.side) - paid[f.order_id]
            if abs(fee - f.fee) > 1e-6:
                raise ValueError('recalculated fee disagrees with fill ledger')
            paid[f.order_id] += fee
            signed = -amount if f.side == 'buy' else amount
            cash += signed - fee
            native_cash += signed - amount * rate - q * slippage
            if f.side == 'buy':
                holdings[f.symbol] += q
                today[f.symbol] += q
            else:
                if q > holdings[f.symbol] - today[f.symbol]:
                    raise ValueError('negative or T+1-invalid inventory')
                holdings[f.symbol] -= q
        current = marks[marks.date == day]
        for m in current.itertuples():
            if not math.isfinite(m.price) or m.price <= 0:
                raise ValueError('invalid close')
            prices[m.symbol] = float(m.price)
        if any(q and s not in prices for s,q in holdings.items()):
            raise ValueError('holding has no known valuation')
        value = sum(q * prices[s] for s,q in holdings.items() if q)
        stale = sum(q > 0 and s not in set(current.symbol) for s,q in holdings.items())
        rows.append(dict(date=day,cash=cash,market_value=value,dividend_receivable=receivable,
            shadow_equity=cash+value+receivable,
            engine_cost_price_equity=native_cash+value,
            audited_equity=native_cash+value+receivable,stale_held_symbols=stale))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('empty valuation calendar')
    peak = frame.audited_equity.cummax().clip(lower=capital)
    frame['drawdown_pct'] = (frame.audited_equity/peak-1)*100
    metrics = dict(return_pct=float((frame.audited_equity.iloc[-1]/capital-1)*100),
        max_drawdown_pct=float(-frame.drawdown_pct.min()),
        end_equity=float(frame.audited_equity.iloc[-1]),sessions=len(frame),
        dividend_receivable=float(receivable),tax_and_payment_date_verified=False,
        annualized_metrics_reported=False,
        cost_basis='native rate per side plus absolute slippage per filled share; gross dividend receivable')
    return frame, metrics
