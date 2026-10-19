"""Independent, read-only audit of the frozen 80 deep-risk account ledgers.

Reads account CSVs and source prices. It does not import or rerun the trading
engine for the financial reconciliations, and never rewrites experiment files.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "training_runs/research_20261018/deep_risk"
OUT_JSON = RUN / "independent_audit.json"
OUT_MD = RUN / "独立账本审计.md"
CAPITAL = 1_000_000.0
TOL = 2e-5


def clean_number(value):
    return None if value is None or pd.isna(value) else float(value)


def metric_calc(daily: pd.DataFrame, capital: float, start_date) -> dict:
    e = daily.equity.to_numpy(float)
    returns = np.r_[e[0] / capital - 1, e[1:] / e[:-1] - 1]
    peak = np.maximum.accumulate(np.r_[capital, e])[1:]
    dd = e / peak - 1
    years = max((daily.date.iloc[-1] - pd.Timestamp(start_date)).days / 365.25, 1 / 252)
    annual = (e[-1] / capital) ** (1 / years) - 1
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    mdd = -float(dd.min())
    rf = 1.015 ** (1 / 252) - 1
    return {
        "days": len(daily), "total_return": e[-1] / capital - 1,
        "annual_return": annual, "max_drawdown": mdd,
        "sharpe_rf15": (returns.mean() - rf) / sd * math.sqrt(252) if sd > 1e-12 else 0.0,
        "sharpe_rf0": returns.mean() / sd * math.sqrt(252) if sd > 1e-12 else 0.0,
        "calmar": annual / mdd if mdd > 1e-12 else None,
        "mean_exposure": float(daily.exposure.mean()),
        "total_cost": float(daily.cost.sum()), "turnover": float(daily.turnover.sum()),
        "end_equity": float(e[-1]),
        "cash_days_fraction": float((daily.exposure < .01).mean()),
    }


def compare_metrics(actual, expected, fields, errors, tag):
    for field in fields:
        a, b = clean_number(actual[field]), clean_number(expected[field])
        if a is None and b is None:
            continue
        if a is None or b is None or abs(a - b) > 1e-8 * max(1.0, abs(b)):
            errors.append({"check": tag, "field": field, "reported": a, "recomputed": b})


def main() -> None:
    summary = pd.read_csv(RUN / "all_results.csv")
    periods = pd.read_csv(RUN / "all_period_results.csv")
    choices = pd.read_csv(RUN / "validation_choices.csv")
    design = json.loads((RUN / "design.json").read_text(encoding="utf-8"))
    rules = {r["name"]: r for r in design["rules"]}
    rules["cash"] = {"stop": 0., "circuit_drawdown": 0.}
    actions = pd.read_csv(ROOT / "training_data/research_20261011/actions.csv", parse_dates=["date"])
    action_map = {(r.date, r.stockid): (float(r.dividend), float(r.split)) for r in actions.itertuples(index=False)}
    audit = {"strategies": len(summary), "period_rows": len(periods), "account_folders": len(list((RUN / "accounts").iterdir())),
             "daily_rows_checked": 0, "orders_checked": 0, "sales_checked": 0, "cash_identity_max_error": 0.,
             "cash_flow_max_error": 0., "holdings_value_max_error": 0., "order_cash_max_error": 0.,
             "metrics_disagreements": [], "invariant_errors": [], "timing_errors": [], "circuit_events": 0,
             "circuit_early_recoveries": [], "stop_attempts": 0, "stop_condition_mismatches": [],
             "rule_outcome_counts": {}, "validation_choice_past_only": None}
    assert len(summary) == 80 and len(set(summary.strategy)) == 80
    assert len(periods) == 240
    metric_fields = ["days", "total_return", "annual_return", "max_drawdown", "sharpe_rf15", "sharpe_rf0", "calmar",
                     "mean_exposure", "total_cost", "turnover", "end_equity", "cash_days_fraction"]
    date_template = None

    for rec in summary.itertuples(index=False):
        name = rec.strategy
        folder = RUN / "accounts" / name
        d = pd.read_csv(folder / "daily.csv", parse_dates=["date", "signal_used"])
        o = pd.read_csv(folder / "orders.csv", parse_dates=["date", "signal_date"])
        t = pd.read_csv(folder / "trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
        h = pd.read_csv(folder / "holdings.csv", parse_dates=["date", "entry_date"])
        if date_template is None:
            date_template = list(d.date)
        elif list(d.date) != date_template:
            audit["invariant_errors"].append({"strategy": name, "check": "date_alignment"})
        audit["daily_rows_checked"] += len(d)
        audit["orders_checked"] += len(o)
        audit["sales_checked"] += len(t)
        dates = list(d.date)
        date_index = {date: i for i, date in enumerate(dates)}
        identity = (d.equity - d.cash - d.market_value).abs().max()
        flow = (d.cash - (d.opening_cash + d.dividend_cash + d.sell_value - d.buy_value - d.cost)).abs().max()
        exposure = (d.exposure - d.market_value / d.equity).abs().max()
        hv = h.groupby("date").value.sum().reindex(d.date, fill_value=0).to_numpy()
        holding_err = np.abs(hv - d.market_value.to_numpy()).max()
        audit["cash_identity_max_error"] = max(audit["cash_identity_max_error"], float(identity))
        audit["cash_flow_max_error"] = max(audit["cash_flow_max_error"], float(flow))
        audit["holdings_value_max_error"] = max(audit["holdings_value_max_error"], float(holding_err))
        for check, value in [("equity=cash+positions", identity), ("cash_flow", flow), ("exposure_ratio", exposure),
                             ("holdings_value", holding_err)]:
            if value > TOL:
                audit["invariant_errors"].append({"strategy": name, "check": check, "error": float(value)})
        if d.cash.min() < -TOL or d.equity.min() <= 0 or (h.shares < -TOL).any():
            audit["invariant_errors"].append({"strategy": name, "check": "negative_cash_equity_or_shares"})
        if (d.lots.to_numpy() != h.groupby("date").size().reindex(d.date, fill_value=0).to_numpy()).any():
            audit["invariant_errors"].append({"strategy": name, "check": "lot_count"})
        if not np.allclose(d.opening_cash.iloc[1:].to_numpy(), d.cash.iloc[:-1].to_numpy(), atol=TOL):
            audit["invariant_errors"].append({"strategy": name, "check": "cash_carry"})

        # Rebuild all intraday cash movements from actual filled orders, including
        # zero-share failed attempts. This is separate from the daily cash formula.
        groups = {day: frame for day, frame in o.groupby("date", sort=False)}
        holding_groups = {day: frame for day, frame in h.groupby("date", sort=False)}
        empty_orders = o.iloc[:0]
        empty_holdings = h.iloc[:0]
        lot = {}
        last_holding_mark = {}
        for row in d.itertuples(index=False):
            day = row.date
            balance = float(row.opening_cash + row.dividend_cash)
            for lid, state in lot.items():
                div, split = action_map.get((day, state["stockid"]), (0., 1.))
                state["shares"] *= split
                state["peak"] = (state["peak"] - div) / split
            day_orders = groups.get(day, empty_orders)
            for order in day_orders.itertuples(index=False):
                qty = float(order.filled_shares)
                value = qty * float(order.price)
                fee = float(order.fee)
                balance += value - fee if order.side == "sell" else -value - fee
                audit["order_cash_max_error"] = max(audit["order_cash_max_error"], abs(balance - float(order.cash_after)))
                if abs(balance - float(order.cash_after)) > TOL:
                    audit["invariant_errors"].append({"strategy": name, "check": "per_order_cash", "date": str(day.date())})
                if qty <= 0:
                    continue
                lid = int(order.lot_id)
                if order.side == "buy":
                    if order.when != "open" or pd.isna(order.signal_date) or date_index[day] - date_index[order.signal_date] != 1:
                        audit["timing_errors"].append({"strategy": name, "check": "buy_next_open", "date": str(day.date())})
                    if (date_index[order.signal_date] % 5) != 0:
                        audit["timing_errors"].append({"strategy": name, "check": "buy_signal_phase", "date": str(day.date())})
                    if lid in lot:
                        audit["invariant_errors"].append({"strategy": name, "check": "duplicate_lot_id", "lot_id": lid})
                    lot[lid] = {"stockid": order.stockid, "shares": qty, "entry_date": day,
                                "peak": float(order.price), "stop": float(rules[order.rule]["stop"])}
                else:
                    state = lot.get(lid)
                    if state is None or state["stockid"] != order.stockid or state["shares"] + TOL < qty:
                        audit["invariant_errors"].append({"strategy": name, "check": "sell_without_shares", "lot_id": lid})
                        continue
                    if order.when == "open" and order.reason == "trailing_stop_previous_close":
                        audit["stop_attempts"] += 1
                        prior = last_holding_mark.get(lid)
                        if prior is None or not (prior <= state["peak"] * (1 - state["stop"]) + 1e-7):
                            audit["stop_condition_mismatches"].append({"strategy": name, "lot_id": lid, "date": str(day.date()),
                                                                        "prior_close": prior, "peak": state["peak"], "stop": state["stop"]})
                    if date_index[day] <= date_index[state["entry_date"]]:
                        audit["timing_errors"].append({"strategy": name, "check": "T+1", "date": str(day.date())})
                    state["shares"] -= qty
                    if state["shares"] < 1e-8:
                        del lot[lid]
            if abs(balance - row.cash) > TOL:
                audit["invariant_errors"].append({"strategy": name, "check": "end_cash_from_orders", "date": str(day.date())})
            day_h = holding_groups.get(day, empty_holdings)
            held_ids = set(int(x) for x in day_h.lot_id)
            if set(lot) != held_ids:
                audit["invariant_errors"].append({"strategy": name, "check": "holdings_lot_set", "date": str(day.date())})
            for held in day_h.itertuples(index=False):
                lid = int(held.lot_id)
                if lid in lot:
                    if abs(lot[lid]["shares"] - float(held.shares)) > TOL:
                        audit["invariant_errors"].append({"strategy": name, "check": "holdings_shares", "date": str(day.date()), "lot_id": lid})
                    lot[lid]["peak"] = max(lot[lid]["peak"], float(held.mark))
                    last_holding_mark[lid] = float(held.mark)
            if row.blocked_by_unsold and ((day_orders.side == "buy") & (day_orders.filled_shares > 0)).any():
                audit["invariant_errors"].append({"strategy": name, "check": "buy_despite_unsold_cohort", "date": str(day.date())})

        # Executed sales must be the trade records; failed attempts are excluded.
        sales = o[(o.side == "sell") & (o.filled_shares > 0)].copy()
        lhs = sales.groupby(["date", "lot_id"]).filled_shares.sum()
        rhs = t.groupby(["exit_date", "lot_id"]).entry_shares.sum()
        rhs.index.names = lhs.index.names
        if not lhs.sort_index().equals(rhs.sort_index()):
            audit["invariant_errors"].append({"strategy": name, "check": "sales_vs_trades"})
        if len(t):
            elapsed = t.exit_date.map(date_index).to_numpy() - t.entry_date.map(date_index).to_numpy()
            if (elapsed != t.holding_session_intervals.to_numpy()).any() or (elapsed + 1 != t.holding_sessions_inclusive.to_numpy()).any():
                audit["invariant_errors"].append({"strategy": name, "check": "holding_intervals"})
            if (elapsed < 1).any():
                audit["timing_errors"].append({"strategy": name, "check": "trade_T+1"})
            scheduled = t[t.reason == "scheduled_fifth_session"]
            if len(scheduled) and any(date_index[x.exit_date] < date_index[x.signal_date] + 5 for x in scheduled.itertuples(index=False)):
                audit["timing_errors"].append({"strategy": name, "check": "premature_scheduled_exit"})
        buy_day = o[(o.side == "buy") & (o.filled_shares > 0)].groupby("date").size()
        if buy_day.reindex(d.loc[d.blocked_by_unsold, "date"], fill_value=0).sum() > 0:
            audit["invariant_errors"].append({"strategy": name, "check": "overlapping_cohort_buy"})

        first = d.date.iloc[0]
        expected = metric_calc(d, CAPITAL, first)
        compare_metrics(rec._asdict(), expected, metric_fields, audit["metrics_disagreements"], name + "/full")
        for p in periods[periods.strategy == name].itertuples(index=False):
            part = d[d.date.between(pd.Timestamp(p.start), pd.Timestamp(p.end))]
            before = d[d.date < part.date.iloc[0]]
            prior_cap = float(before.equity.iloc[-1]) if len(before) else CAPITAL
            prior_date = before.date.iloc[-1] if len(before) else part.date.iloc[0]
            expected = metric_calc(part, prior_cap, prior_date)
            compare_metrics(p._asdict(), expected, metric_fields, audit["metrics_disagreements"], name + "/" + p.period)

        # Circuit firing can only use previous close equity and prior peak.
        circuit = float(rules[name.split("__", 1)[1]]["circuit_drawdown"]) if not name.endswith("past_validation_selected") else None
        fires = []
        for i, row in enumerate(d.itertuples(index=False)):
            if not row.circuit_fired:
                continue
            audit["circuit_events"] += 1
            fires.append(i)
            active_rule = rules.get(row.effective_rule, rules["cash"])
            threshold = float(active_rule.get("circuit_drawdown", 0.)) if circuit is None else circuit
            if i == 0 or threshold <= 0 or d.equity.iloc[i - 1] > d.risk_episode_peak.iloc[i - 1] * (1 - threshold) + TOL:
                audit["timing_errors"].append({"strategy": name, "check": "circuit_not_previous_close", "date": str(row.date.date())})
        for j in fires:
            clear = next((i for i in range(j + 1, len(d)) if not bool(d.risk_paused.iloc[i])), None)
            if clear is not None and clear < j + 5:
                audit["circuit_early_recoveries"].append({"strategy": name, "fire": str(d.date.iloc[j].date()),
                                                          "clear": str(d.date.iloc[clear].date()), "session_gap": clear - j})

    # The selector's validation windows precede the quarter being traded.
    folds = pd.read_csv(ROOT / "training_runs/research_20261017/models/folds.csv")
    choice = choices.merge(folds[["quarter", "valid_end", "test_start"]], on="quarter", validate="many_to_one")
    audit["validation_choice_past_only"] = bool((choice.valid_end < choice.test_start).all() and len(choice) == 24)
    audit["rule_outcome_counts"] = {
        "full_all_three_targets": int(((periods.period == "full") & periods.meets_targets).sum()),
        "recent_year_all_three_targets": int(((periods.period == "recent_year") & periods.meets_targets).sum()),
        "recent_half_all_three_targets": int(((periods.period == "recent_half") & periods.meets_targets).sum()),
    }
    audit["status"] = "pass" if all(not audit[key] for key in ["metrics_disagreements", "invariant_errors", "timing_errors",
                                                                "circuit_early_recoveries", "stop_condition_mismatches"]) and audit["validation_choice_past_only"] else "issues"
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "# 80组风险策略独立账本复核\n\n"
        f"结论：**{audit['status']}**。核查{audit['strategies']}个账户、{audit['daily_rows_checked']}行逐日账、"
        f"{audit['orders_checked']}笔订单尝试、{audit['sales_checked']}笔成交卖出；"
        f"现金+持仓与权益最大差{audit['cash_identity_max_error']:.8g}元，逐日现金流最大差{audit['cash_flow_max_error']:.8g}元，"
        f"持仓汇总最大差{audit['holdings_value_max_error']:.8g}元，逐笔订单现金最大差{audit['order_cash_max_error']:.8g}元。\n\n"
        f"复算绩效差异{len(audit['metrics_disagreements'])}项；账本约束错误{len(audit['invariant_errors'])}项；"
        f"时点错误{len(audit['timing_errors'])}项；止损触发不符{len(audit['stop_condition_mismatches'])}项；"
        f"熔断后不足5日恢复{len(audit['circuit_early_recoveries'])}项。详细记录见 `independent_audit.json`。\n\n"
        "判断范围：验证策略选择所用的`valid_end`早于当季`test_start`；订单/现金/仓位、T+1、"
        "信号次日开盘、上一日风险熔断与5交易日冷却经逐条核对。该检查并不证明历史数据源自身完全无修订或模型分数生成时无泄漏。\n",
        encoding="utf-8",
    )
    print(json.dumps({k: audit[k] for k in ["status", "strategies", "daily_rows_checked", "orders_checked", "sales_checked",
                                               "cash_identity_max_error", "cash_flow_max_error", "holdings_value_max_error",
                                               "order_cash_max_error", "validation_choice_past_only"]}, ensure_ascii=False))
    print("issue counts", {k: len(audit[k]) for k in ["metrics_disagreements", "invariant_errors", "timing_errors",
                                                       "circuit_early_recoveries", "stop_condition_mismatches"]})


if __name__ == "__main__":
    main()
