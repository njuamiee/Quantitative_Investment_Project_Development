"""Rebuild the 2026-03-04 signal / 2026-03-05 entry example from saved inputs.

This is an audit helper only. It does not fit a model or alter the backtest.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SIGNAL = pd.Timestamp("2026-03-04")
ENTRY = pd.Timestamp("2026-03-05")
ACCOUNT = ROOT / "training_runs/research_20261017/accounts/new_main_monthly"
PRICES = ROOT / "training_data/research_20261011/prices.parquet"
PREDICTIONS = ROOT / "training_runs/research_20261017/models/daily_predictions.parquet"
METADATA = ROOT / "training_runs/research_20261017/risk_metadata.parquet"


def plain(value: object) -> str:
    return "unknown" if pd.isna(value) else str(value)


def main() -> None:
    price_all = pd.read_parquet(PRICES)
    day = price_all.loc[price_all.date.eq(SIGNAL)].copy().sort_values("stockid")
    assert len(day) == 350 and day.stockid.is_unique

    # The four tests below are the exact conjunction in prepare.prices().
    day["history_ge_60"] = day.observed_days.ge(60)
    day["active_ge_16_of_20"] = day.active20.ge(16)
    day["valid_today_bar"] = day.observed.astype(bool)
    day["adv20_positive"] = day.adv20.gt(0)
    rebuilt_eligible = day[["history_ge_60", "active_ge_16_of_20",
                            "valid_today_bar", "adv20_positive"]].all(axis=1)
    assert rebuilt_eligible.equals(day.eligible.astype(bool))
    assert day.in_pool.sum() == 100
    assert day.eligible.sum() == 282

    market = day[["stockid", "observed", "observed_days", "active20", "amount",
                  "adv20", "history_ge_60", "active_ge_16_of_20",
                  "valid_today_bar", "adv20_positive", "eligible", "amount_rank",
                  "in_pool", "exclude_reason"]].copy()
    market.to_csv(OUT / "05_20260304_350只股票池筛选.csv", index=False,
                  encoding="utf-8-sig")

    prediction = pd.read_parquet(PREDICTIONS)
    score = prediction.loc[prediction.date.eq(SIGNAL),
                           ["stockid", "score_monthly", "quarter"]].copy()
    assert len(score) == 100 and score.stockid.is_unique
    assert set(score.stockid) == set(day.loc[day.in_pool, "stockid"])
    assert score.score_monthly.notna().all()

    metadata = pd.read_parquet(METADATA)
    meta = metadata.loc[metadata.date.eq(SIGNAL),
                        ["stockid", "industry", "size_tertile",
                         "momentum_tertile", "volatility_tertile"]].copy()
    assert meta.stockid.is_unique
    ranked = score.merge(meta, on="stockid", how="left", validate="one_to_one")
    ranked = ranked.sort_values(["score_monthly", "stockid"],
                                ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "score_rank", np.arange(1, len(ranked) + 1))

    group_cols = [("industry", "industry"), ("size_group", "size_tertile"),
                  ("momentum_group", "momentum_tertile"),
                  ("volatility_group", "volatility_tertile")]
    counts: dict[tuple[str, str], int] = {}
    statuses: list[str] = []
    reasons: list[str] = []
    selected_orders: list[int | None] = []
    selected_count = 0
    for row in ranked.itertuples(index=False):
        if selected_count >= 20:
            statuses.append("not_reached_after_top20")
            reasons.append("")
            selected_orders.append(None)
            continue
        groups = {key: plain(getattr(row, column)) for key, column in group_cols}
        reason = ""
        # _picks() checks the groups in this order, including the unknown check.
        for key in ["size_group", "momentum_group", "volatility_group", "industry"]:
            value = groups[key]
            if value.lower() in ["unknown", "nan", "none", "missing", "未知"]:
                if key in ["size_group", "industry"]:
                    reason = "missing_historical_" + (
                        "size" if key == "size_group" else "industry")
                    break
                continue
            limit = .30 if key == "industry" else .50
            if counts.get((key, value), 0) >= max(1, math.floor(20 * limit)):
                reason = key + "_quota"
                break
        if reason:
            statuses.append("rejected_by_group")
            reasons.append(reason)
            selected_orders.append(None)
        else:
            selected_count += 1
            statuses.append("selected_for_order")
            reasons.append("")
            selected_orders.append(selected_count)
            for key, value in groups.items():
                counts[(key, value)] = counts.get((key, value), 0) + 1
    ranked["selection_status"] = statuses
    ranked["selection_reason"] = reasons
    ranked["selected_rank"] = pd.array(selected_orders, dtype="Int64")

    saved_rejections = pd.read_csv(ACCOUNT / "rejected_selection.csv")
    saved_rejections = saved_rejections.loc[
        saved_rejections.signal_date.eq(str(SIGNAL.date()))]
    replay_rejections = ranked.loc[ranked.selection_status.eq("rejected_by_group"),
                                   ["stockid", "selection_reason"]]
    assert list(zip(saved_rejections.stockid, saved_rejections.reason)) == list(
        replay_rejections.itertuples(index=False, name=None))
    assert selected_count == 20 and len(replay_rejections) == 25
    assert ranked.selection_status.eq("not_reached_after_top20").sum() == 55

    all_orders = pd.read_csv(ACCOUNT / "orders.csv")
    orders = all_orders.loc[all_orders.date.eq(str(ENTRY.date())) &
                            all_orders.side.eq("buy")].copy().sort_values("rank")
    assert len(orders) == 20 and orders.filled_shares.gt(0).all()
    assert orders.filled_shares.equals(orders.requested_shares)
    assert list(orders.stockid) == list(
        ranked.loc[ranked.selection_status.eq("selected_for_order"), "stockid"])
    ranked = ranked.merge(
        orders[["stockid", "rank", "target_weight", "target_value",
                "requested_shares", "filled_shares", "price", "fee", "cash_after"]],
        on="stockid", how="left", validate="one_to_one")
    assert np.allclose(
        ranked.loc[ranked.selected_rank.notna(), "selected_rank"].astype(float),
        ranked.loc[ranked.selected_rank.notna(), "rank"].astype(float))
    ranked.to_csv(OUT / "05_20260304_100只评分及分组筛选.csv", index=False,
                  encoding="utf-8-sig")
    orders.to_csv(OUT / "05_20260305_20笔真实买单.csv", index=False,
                  encoding="utf-8-sig")

    account = pd.read_csv(ACCOUNT / "daily.csv", parse_dates=["date"])
    signal_account = account.loc[account.date.eq(SIGNAL)].iloc[0]
    entry_account = account.loc[account.date.eq(ENTRY)].iloc[0]
    assert signal_account.equity == entry_account.opening_cash
    assert entry_account.planned_positions == 20
    assert entry_account.orders_limited == 0
    assert entry_account.effective_rule == "diversified_vol12"
    assert abs(orders.filled_shares.mul(orders.price).sum() - entry_account.buy_value) < .01
    assert abs(orders.fee.sum() - entry_account.cost) < .01
    assert abs(entry_account.cash + entry_account.market_value - entry_account.equity) < .01

    market_return = price_all.loc[price_all.in_pool].groupby("date").adj_ret1.mean()
    all_dates = pd.DatetimeIndex(sorted(price_all.date.unique()))
    market_return = market_return.reindex(all_dates).fillna(0.)
    volatility = market_return.rolling(20, min_periods=10).std().loc[SIGNAL] * np.sqrt(252)
    target = float(np.clip(.12 / volatility, .10, .95))
    assert abs(target - entry_account.target_exposure_on_entry) < 1e-12
    per_stock_budget = target * signal_account.equity / 20
    assert abs(per_stock_budget - orders.target_value.iloc[0]) < .01
    first = orders.iloc[0]
    shares_formula = math.floor(per_stock_budget / (first.price * 1.0005))
    assert shares_formula == first.requested_shares

    reason_counts = replay_rejections.selection_reason.value_counts()
    lines = [
        "# 真实一天：从350只股票到20笔买单",
        "",
        "**信号日：2026-03-04收盘；下单日：2026-03-05开盘。**",
        "这一天并非编造例子。每个数字来自已保存的日行情、模型评分、历史分组和账户逐单记录。",
        "后面的三个CSV列出350只原始股票、100只池内股票和20笔实际买单；本目录的生成脚本逐项核对原始流水。",
        "",
        "## 第一步：形成当天能选的100只股票",
        "",
        f"原始行情表每天有{len(day)}个股票代码的占位行，并不表示{len(day)}只当天都成交。",
        f"3月4日真正有有效开高低收、成交量和成交额的有{int(day.observed.sum())}只。",
        "对每只股票依次检查：",
        "",
        "```text",
        "如果截至今天，有效行情累计少于60天：排除；",
        "否则如果最近20个交易日里有效行情少于16天：排除；",
        "否则如果今天没有有效行情：排除；",
        "否则如果过去20日平均成交额不是正数：排除；",
        "否则：进入可排名名单。",
        "把名单按过去20日平均成交额从高到低排；名次≤100的进入当天股票池。",
        "```",
        "",
        "这里‘有效行情’指当日成交额>0、成交量>0，且开高低收四个价格都>0。",
        "过去20日平均成交额=最近20个交易日的成交额合计÷20；无成交记录按0计，资料不足20天则暂不计算。",
        "同金额时按已保存数据顺序决定先后；这份数据按股票代码排序。股票池是**收盘后**形成，供下一交易日开盘尝试交易。",
        "",
        "| 依次完成的检查 | 还剩多少只 | 本步新排除多少只 |",
        "| --- | ---: | ---: |",
        f"| 原始代码占位行 | {len(day)} | — |",
        f"| 累计有效行情≥60日 | {int(day.history_ge_60.sum())} | {len(day)-int(day.history_ge_60.sum())} |",
        f"| 再要求最近20日有效≥16日 | {int((day.history_ge_60 & day.active_ge_16_of_20).sum())} | {int(day.history_ge_60.sum()-(day.history_ge_60 & day.active_ge_16_of_20).sum())} |",
        f"| 再要求当天有效行情 | {int((day.history_ge_60 & day.active_ge_16_of_20 & day.valid_today_bar).sum())} | {int((day.history_ge_60 & day.active_ge_16_of_20).sum()-(day.history_ge_60 & day.active_ge_16_of_20 & day.valid_today_bar).sum())} |",
        f"| 再要求20日平均成交额>0 | {int(day.eligible.sum())} | {int((day.history_ge_60 & day.active_ge_16_of_20 & day.valid_today_bar).sum()-day.eligible.sum())} |",
        "| 过去20日平均成交额排名前100 | 100 | 182 |",
        "",
        f"当天有有效行情的总数是{int(day.observed.sum())}，但这是单独统计，不能把它当成上述第一步后的剩余数；无有效行情的代码有些已经在‘不足60天’一项被排除了。",
        "当天共282只满足全部可排名条件；排名后100只进入模型评分。名单和每只的检查结果见 `05_20260304_350只股票池筛选.csv`。",
        "",
        "## 第二步：模型给100只打分，再按分组限制挑20只",
        "",
        "当时使用的是2026年一季度先前选定的月度重训模型。3月4日收盘后，",
        "它对池内100只各给一个分数；这个分数只表示相对排序，不是预计上涨百分比。",
        "从分数高到低看；同分按股票代码先后。2026年一季度之前的验证选择了 `diversified_vol12` 交易规则。",
        "这个规则需要历史行业和规模有值，并限制已选的20只中同一行业最多6只、",
        "同一规模组、过去20日涨跌组、过去20日波动组各最多10只。",
        "",
        "```text",
        "对分数最高的股票开始逐只看：",
        "    如果行业或规模历史资料未知：跳过这只；",
        "    否则如果把它加入后，同业会超过6只：跳过；",
        "    否则如果同规模/涨跌/波动组会超过10只：跳过；",
        "    否则把它放进待买名单。",
        "一旦放满20只就停，后面股票本次不检查、也不下单。",
        "```",
        "",
        f"本日实际看到分数第45名时才凑满20只。期间跳过{len(replay_rejections)}只：",
        f"行业未知{int(reason_counts.get('missing_historical_industry',0))}只、",
        f"规模配额{int(reason_counts.get('size_group_quota',0))}只、",
        f"涨跌组配额{int(reason_counts.get('momentum_group_quota',0))}只、",
        f"波动组配额{int(reason_counts.get('volatility_group_quota',0))}只。",
        "剩余55只没有轮到，不能说它们被规则判为无效。逐只评分和每只被选/跳过/未检查的原因见 `05_20260304_100只评分及分组筛选.csv`。",
        "",
        "## 第三步：把20只变成实际买单",
        "",
        f"3月4日收盘，旧批已清，账户现金和总资产均为{signal_account.equity:,.2f}元。",
        f"模型在3月4日看到的股票池每日等权收益过去20日年化波动约{volatility:.4%}。",
        f"按已登记公式：计划总仓位=把12%÷{volatility:.4%}限制在10%到95%之间，得到{target:.4%}。",
        f"这20只等权，每只计划金额=总资产×{target:.4%}÷20={per_stock_budget:,.2f}元。",
        "不是看模型分数有多大就直接乘出收益。",
        "",
        f"以第1名 {first.stockid} 为例：3月5日开盘价{first.price:.2f}元，",
        f"预计买入股数=向下取整({per_stock_budget:,.2f}÷[{first.price:.2f}×(1+0.0005)])={shares_formula:,}股。",
        "随后再核对唯一现金余额、计划总仓位、前20日平均成交额1%的当日成交容量，以及开盘价相对前收盘价的涨停代理。",
        "如果任何限制使可买股数少于100股，就不下买单；否则按整数股成交并立刻从现金扣买价和费用。",
        "北交所这套仿真约束是‘一笔至少100股，之后可以是整数股’，因此例子中的2,277股并非强制取100的倍数。",
        "当天没有一笔买单被容量、现金、最少股数或涨停代理挡住：20笔计划单全部成交。",
        f"买入股票金额合计{entry_account.buy_value:,.2f}元，费用{entry_account.cost:,.2f}元。",
        f"收盘现金{entry_account.cash:,.2f}元、股票市值{entry_account.market_value:,.2f}元、",
        f"总资产{entry_account.equity:,.2f}元，三者逐分相等；实际股票仓位{entry_account.exposure:.4%}。",
        "20笔真实买单、每笔股数和买后现金见 `05_20260305_20笔真实买单.csv`。",
        "",
        "## 这天能证明什么，不能证明什么",
        "",
        "这一整串数字说明**2026-03-04收盘时知道什么、2026-03-05开盘怎样使用**，",
        "未把3月5日收盘价或五日后收益拿来决定这20只。3月5日收盘价只用于收盘记账。",
        "模型训练仍须在各自重训切点排除尚未结束的五日答案，详见主说明和 `models/folds.csv`。",
        "现有历史上市/退市和ST变更资料不完整，日行情被重排为350个股票代码的固定行；",
        "不能把这350行说成当时真实挂牌350只，也不能声称已完全排除退市、ST或供应商后修订影响。",
        "涨跌停、停牌和真实排队成交只有日线代理；此例全部成交是模型账本结论，并非交易所逐笔成交证明。",
        "",
        "核查来源：`tools/research_20260930/prepare.py` 的 `prices()`、",
        "`tools/research_20261017/execution.py` 的 `_picks()`/`run()`、",
        "`training_data/research_20261011/prices.parquet`、",
        "`training_runs/research_20261017/models/daily_predictions.parquet`、",
        "`training_runs/research_20261017/risk_metadata.parquet`、",
        "`training_runs/research_20261017/accounts/new_main_monthly/`。",
        "",
    ]
    (OUT / "05_真实一天_股票池到成交.md").write_text("\n".join(lines),
                                                    encoding="utf-8")
    print("CHECKED", len(day), len(score), len(replay_rejections), len(orders))


if __name__ == "__main__":
    main()
