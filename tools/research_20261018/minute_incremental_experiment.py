"""Frozen 2025Q4 minute add-on vs a frozen daily model, tested in 2026Q3.

Run from anywhere with::
    python tools/research_20261018/minute_incremental_experiment.py

Q3 labels are read only by the post-fit diagnostic routine.  The deployed
minute Ridge and its mixing weight are entirely decided by Q4 mature labels.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from adaptive_execution import CohortMarket, Rule


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training_data/research_20261011"
RUN = ROOT / "training_runs/research_20261018/minute_incremental"
DAILY_PATH = ROOT / "training_runs/research_20261017/models/daily_predictions.parquet"
BASELINE_DIR = ROOT / "training_runs/research_20261017/models"
MINUTE_PATH = DATA / "minute_factors.parquet"
OLD_MINUTE_PATH = ROOT / "training_runs/research_20261011/minute_factor_h5_predictions.parquet"
PRICE_PATH = DATA / "prices.parquet"
ACTIONS_PATH = DATA / "actions.csv"
TRAIN_CUTOFF = pd.Timestamp("2025-12-01")
Q4_END = pd.Timestamp("2026-01-01")
Q3_START = pd.Timestamp("2026-07-01")
Q3_END = pd.Timestamp("2026-09-30")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FEATURES = (
    "minute_return", "late_return", "realized_vol", "downside_vol",
    "return_skew", "volume_concentration", "late_volume_share",
    "close_vwap_gap", "afternoon_minus_morning", "price_volume_corr",
)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_date(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("date", sort=False)[col].rank(pct=True).sub(0.5)


def mean_daily_rank_ic(frame: pd.DataFrame, score_col: str) -> tuple[float, pd.DataFrame]:
    rows = []
    for date, group in frame.groupby("date", sort=True):
        pair = group[[score_col, "label_5"]].dropna()
        ic = pair[score_col].corr(pair.label_5, method="spearman") if len(pair) >= 20 else np.nan
        rows.append({"date": date, "rank_ic": ic, "stock_rows": len(pair)})
    daily = pd.DataFrame(rows)
    return float(daily.rank_ic.mean()), daily


def feature_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build same-day, same-pool joined panel; no target enters this function."""
    minutes = pd.read_parquet(MINUTE_PATH)
    daily = pd.read_parquet(DAILY_PATH, columns=["date", "stockid", "score_quarterly"])
    minutes.date = pd.to_datetime(minutes.date)
    daily.date = pd.to_datetime(daily.date)
    assert not minutes.duplicated(["date", "stockid"]).any()
    assert not daily.duplicated(["date", "stockid"]).any()
    assert set(FEATURES).issubset(minutes.columns)
    minutes = minutes[minutes.date.between("2025-10-01", "2025-12-31") |
                      minutes.date.between(Q3_START, Q3_END)].copy()
    daily = daily[daily.date.between("2025-10-01", "2025-12-31") |
                  daily.date.between(Q3_START, Q3_END)].copy()
    for factor in FEATURES:
        minutes[factor] = minutes[factor].replace([np.inf, -np.inf], np.nan)
        minutes[factor] = rank_date(minutes, factor)
    common = daily.merge(minutes[["date", "stockid", "bars", *FEATURES]],
                         on=["date", "stockid"], how="inner", validate="one_to_one")
    common["score_daily"] = rank_date(common, "score_quarterly")
    assert common.groupby("date").size().min() >= 20
    coverage = daily.groupby("date").size().rename("daily_pool").to_frame()
    coverage["minute_rows_raw"] = minutes.groupby("date").size()
    coverage["common_pool"] = common.groupby("date").size()
    coverage["common_fraction"] = coverage.common_pool / coverage.daily_pool
    coverage["median_bars_common"] = common.groupby("date").bars.median()
    coverage = coverage.reset_index().sort_values("date")
    return common, coverage, minutes


def mature_q4_labels() -> pd.DataFrame:
    """Read only Q4 labels into the fit/selection path."""
    answer = pd.read_parquet(PRICE_PATH,
                             columns=["date", "stockid", "label_5", "label_exit_5"],
                             filters=[('date','>=',pd.Timestamp('2025-10-01')),('date','<',Q4_END)])
    answer.date = pd.to_datetime(answer.date)
    answer.label_exit_5 = pd.to_datetime(answer.label_exit_5)
    return answer[(answer.date < Q4_END) & (answer.date >= "2025-10-01") &
                  (answer.label_exit_5 < Q4_END)].dropna(subset=["label_5"]).copy()


def verify_daily_baseline_timing() -> dict:
    audit = {}
    for quarter, cutoff in (("2025Q4", pd.Timestamp("2025-10-01")),
                            ("2026Q3", Q3_START)):
        info_path = BASELINE_DIR / quarter / "deployment_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        quarterly = info["quarterly"]
        last_exit = pd.Timestamp(quarterly["train_last_exit"])
        assert last_exit < cutoff, (quarter, last_exit, cutoff)
        audit[quarter] = {"last_training_signal": quarterly["train_last_signal"],
                          "last_mature_label_exit": quarterly["train_last_exit"],
                          "first_test_cutoff": str(cutoff.date()),
                          "deployment_info_sha256": digest(info_path)}
    return audit


def fit_minute(panel: pd.DataFrame) -> Ridge:
    target = rank_date(panel, "label_5")
    assert len(panel) and np.isfinite(target).all()
    return Ridge(alpha=100.0).fit(panel[list(FEATURES)].fillna(0.0), target)


def add_scores(panel: pd.DataFrame, model: Ridge, weight: float) -> pd.DataFrame:
    out = panel[["date", "stockid", "score_daily"]].copy()
    out["score_minute_raw"] = model.predict(panel[list(FEATURES)].fillna(0.0))
    out["score_minute"] = rank_date(out, "score_minute_raw")
    out["score_augmented"] = (1.0-weight)*out.score_daily + weight*out.score_minute
    return out


def select_q4_weight(common: pd.DataFrame, q4_answer: pd.DataFrame):
    labelled = common[common.date < Q4_END].merge(q4_answer, on=["date", "stockid"],
                                                    how="inner", validate="one_to_one")
    fit = labelled[(labelled.date < TRAIN_CUTOFF) &
                   (labelled.label_exit_5 < TRAIN_CUTOFF)].copy()
    valid = labelled[(labelled.date >= TRAIN_CUTOFF) &
                     (labelled.label_exit_5 < Q4_END)].copy()
    assert len(fit) > 1000 and len(valid) > 1000
    assert fit.label_exit_5.max() < valid.date.min()
    model = fit_minute(fit)
    predicted = add_scores(valid, model, 0.0)
    predicted = predicted.merge(valid[["date", "stockid", "label_5"]],
                                on=["date", "stockid"], validate="one_to_one")
    choices = []
    for weight in WEIGHTS:
        predicted["blend"] = ((1-weight)*predicted.score_daily +
                              weight*predicted.score_minute)
        ic, _ = mean_daily_rank_ic(predicted, "blend")
        choices.append({"minute_weight": weight, "valid_mean_daily_rank_ic": ic,
                        "valid_days": int(predicted.date.nunique()),
                        "valid_stock_rows": len(predicted)})
    scores = pd.DataFrame(choices)
    # A fixed, documented tie rule keeps the simpler daily signal.
    best = scores.sort_values(["valid_mean_daily_rank_ic", "minute_weight"],
                              ascending=[False, True]).iloc[0]
    return float(best.minute_weight), scores, fit, valid, model


def q3_realized_diagnostic(scores: pd.DataFrame) -> pd.DataFrame:
    """This routine is called after fit/weight selection; it cannot change them."""
    answer = pd.read_parquet(PRICE_PATH,
                             columns=["date", "stockid", "label_5", "label_exit_5"])
    answer.date = pd.to_datetime(answer.date)
    answer.label_exit_5 = pd.to_datetime(answer.label_exit_5)
    answer = answer[(answer.date >= Q3_START) & (answer.date <= Q3_END) &
                    (answer.label_exit_5 <= Q3_END)].copy()
    answer = scores.merge(answer, on=["date", "stockid"],
                          how="inner", validate="one_to_one")
    out = []
    for col in ("score_daily", "score_minute", "score_augmented", "score_fixed25"):
        _, daily = mean_daily_rank_ic(answer, col)
        daily["signal"] = col
        out.append(daily)
    return pd.concat(out, ignore_index=True)


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    return obj


def save_equity_plot(paired: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    ax.plot(paired.date, paired.daily_equity/1e6, color="#1f4e79", lw=2.0,
            label="Daily baseline = Q4-selected blend (0% minute)")
    ax.plot(paired.date, paired.daily_plus_minute_fixed25_equity/1e6,
            color="#af531b", lw=2.0, label="Daily + 25% minute (sensitivity)")
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--")
    ax.set(title="2026Q3 paired cash-account equity, same pool and 5bp per side",
           xlabel="Date", ylabel="Equity / initial capital")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RUN / "q3_equity_comparison.png")
    plt.close(fig)


def write_chinese_audit(summary: dict, choices: pd.DataFrame, coverage: pd.DataFrame) -> None:
    b = summary["backtests"]["daily"]
    s = summary["backtests"]["daily_plus_minute"]
    f = summary["backtests"]["daily_plus_minute_fixed25"]
    q4cov = coverage[coverage.date < Q4_END]
    q3cov = coverage[coverage.date.between(Q3_START, Q3_END)]
    fmt = lambda x: f"{x*100:.2f}%"
    lines = [
        "# 分钟因子对日频选股的增量实验：时间和交易核查",
        "",
        "**事先规则的结论：2025Q4 内部验证选出分钟权重 0%，故正式叠加策略与日频基线完全相同；不能把 2026Q3 的改善倒填为事先选出的策略。**",
        "",
        "## 具体做法",
        "",
        "1. `intraday.py` 已从每只股票当日完整分钟线汇总 10 个数，例如最后 30 分钟涨跌、日内波动、收盘价相对全天成交均价。只在当日收盘后算完，次日开盘才可能买入。分钟文件的股票集合由前一日股票池先限制；本实验再与日频预测按日期、股票代码取交集。",
        f"2. 日频基线使用 `research_20261017/models/daily_predictions.parquet` 中的 `score_quarterly`。其 Q4 版本最后训练答案于 {summary['daily_baseline_timing']['2025Q4']['last_mature_label_exit']} 结束，Q3 版本最后训练答案于 {summary['daily_baseline_timing']['2026Q3']['last_mature_label_exit']} 结束，然后各自冻结整季；不是把 Q3 答案混进去。分钟附加模型单独用 2025Q4 数据训练。两种分数都在当日共同股票池内转成排名百分位，再加权。",
        f"3. 第一次分钟训练：{summary['q4_initial_fit']['first_signal'][:10]} 至 {summary['q4_initial_fit']['last_signal'][:10]}，{summary['q4_initial_fit']['days']} 天、{summary['q4_initial_fit']['rows']} 只股票日；最后一个真实答案在 {summary['q4_initial_fit']['last_label_exit'][:10]} 才结束。训练的目标是同一天股票的后 5 个交易日收益排名。10 列分钟数每天各自排名，空值补 0，Ridge 的收缩强度固定 100。",
        f"4. 内部验证：{summary['q4_validation']['first_signal'][:10]} 至 {summary['q4_validation']['last_signal'][:10]}，{summary['q4_validation']['days']} 天、{summary['q4_validation']['rows']} 只股票日，所有答案截至 {summary['q4_validation']['last_label_exit'][:10]} 已结束。按每日股票分数与后 5 日实际收益的排名相关系数取平均，从预先写定的 0%、25%、50%、75%、100% 中选权重，平手优先小权重。",
        f"5. 确定权重后，用截至 2025-12-31 真正结束的 Q4 答案重新训练分钟模型：{summary['q4_final_fit']['rows']} 行、{summary['q4_final_fit']['days']} 天，最后出题日 {summary['q4_final_fit']['last_signal'][:10]}。此后模型和权重冻结，直接预测 2026Q3；Q3 的后 5 日答案只在全部决定完成后用来核查排名，不参与任何因子或权重选择。",
        "",
        "## 样本和覆盖",
        "",
        "| 时段 | 分钟原始股票日 | 日频股票日 | 两边共同股票日 | 共同覆盖率（日均） | 分钟条数中位数/股票日 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 2025Q4 | {int(q4cov.minute_rows_raw.sum()):,} | {int(q4cov.daily_pool.sum()):,} | {int(q4cov.common_pool.sum()):,} | {fmt(float(q4cov.common_fraction.mean()))} | {q4cov.median_bars_common.median():.0f} |",
        f"| 2026Q3 | {int(q3cov.minute_rows_raw.sum()):,} | {int(q3cov.daily_pool.sum()):,} | {int(q3cov.common_pool.sum()):,} | {fmt(float(q3cov.common_fraction.mean()))} | {q3cov.median_bars_common.median():.0f} |",
        "",
        "只有这 2 个季度有分钟文件，不能声称历年分钟因子稳定。2026Q3 有 65 个交易日，日频股票池每日日均 100 只，交集共 6,042 个股票日；覆盖最差日仅 78%。交集外的股票，两组都不买，避免一分钟信号组因覆盖少而与不同股票池比较。",
        "",
        "## Q4 内部验证，选择发生在 Q3 之前",
        "",
        "| 分钟权重 | 12 月每日平均 RankIC | 是否选中 |",
        "|---:|---:|---|",
    ]
    for row in choices.itertuples(index=False):
        lines.append(f"| {row.minute_weight:.0%} | {row.valid_mean_daily_rank_ic:.4f} | {'是' if row.minute_weight == summary['chosen_minute_weight'] else '否'} |")
    lines += [
        "",
        "RankIC 的意思是：同一天，预测排名与后来真实收益排名是否同向；1 表示完全同向，0 表示看不出规律。Q4 只有 18 个内部验证日，权重选择很不稳，正式按它选择仍只能是 0%。",
        "",
        "## 2026Q3 真正逐日账本：同日、同股票池、同成本",
        "",
        "信号日收盘后排 Top20；下一交易日开盘买，预定第 5 个交易日收盘卖。一个现金账户，不凭空反复注资；买卖两侧各按成交额收 5 个基点。考虑整手 100 股、前日成交额 1% 容量、停牌与涨跌停近似拒单、分红拆股。三组都用相同账本代码、起始 100 万元和相同日历。末日各有 20 条未平仓的持仓记录（lot，不是 20 个五日组合批次），按当天收盘价计值，未伪装成已卖出收益。",
        "",
        "| 方案 | Q3 收益 | 最大回撤 | 夏普（无风险 0） | 卡尔玛 | 模拟交易成本 | 末日资产 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 日频基线 | {fmt(b['total_return'])} | {fmt(b['max_drawdown'])} | {b['sharpe_rf0']:.2f} | {b['calmar']:.2f} | {b['total_cost']:,.2f} 元 | {b['end_equity']:,.2f} 元 |",
        f"| Q4 选定的分钟叠加（0%） | {fmt(s['total_return'])} | {fmt(s['max_drawdown'])} | {s['sharpe_rf0']:.2f} | {s['calmar']:.2f} | {s['total_cost']:,.2f} 元 | {s['end_equity']:,.2f} 元 |",
        f"| 事先候选但 Q4 未选中：固定 25% | {fmt(f['total_return'])} | {fmt(f['max_drawdown'])} | {f['sharpe_rf0']:.2f} | {f['calmar']:.2f} | {f['total_cost']:,.2f} 元 | {f['end_equity']:,.2f} 元 |",
        "",
        f"固定 25% 的 Q3 收益比基线多 {summary['fixed25_return_difference_percentage_points']:.2f} 个百分点，回撤少 {(b['max_drawdown']-f['max_drawdown'])*100:.2f} 个百分点，零利率夏普从 {b['sharpe_rf0']:.2f} 变为 {f['sharpe_rf0']:.2f}；但二者都亏损，卡尔玛从 {b['calmar']:.2f} 到 {f['calmar']:.2f} 并未改善。固定 25% 在 Q4 反而比 0% 略差，它只是**外部测试的敏感度结果**，不能事后宣布为正式赢家。",
        "",
        "Q3 已结束答案的每日平均 RankIC：日频 0.0767，固定 25% 0.1053，分钟单独 0.1601（每组 60 天、5,616 个股票日）。这与 Q4 验证顺序相反，提示时间漂移；只靠 Q4 的 18 天很难稳定判断分钟因子。旧分钟预测与本次分钟模型在共同 6,042 行的排名相关系数为 0.9995，说明重训与旧实验基本一致，但股票池交集使系数和原分数不必逐行相等。",
        "",
        "![同股票池逐日资金曲线](q3_equity_comparison.png)",
        "",
        "## 核查边界",
        "",
        "- Q3 标签只在建模、内部选权重、保存所有预测以及跑完账本以后，才用于 `q3_ic_daily.csv` 的解释性评估。账本用真实逐日开收盘价结算，这是回测的必要结果数据，不是训练标签。",
        "- 日频模型是先前项目的季度冻结模型，使用 Q4 以外的历史数据；‘只用 Q4 训练’严格指本次新增的分钟模型与权重，并非整个旧日频模型。",
        "- 旧 `intraday.py` 的完整分钟文件只覆盖 Q4 与 Q3；分钟原始供应源或复权若有误，研究会受影响。本脚本对来源文件存 SHA-256；源文件重建后须重跑。",
        "- Ridge 的 10 个系数见 `factor_coefficients.csv`，权重检查见 `q4_weight_validation.csv`，每天三组资金见 `paired_equity.csv`，成交和拒单见对应 `*_orders.csv`、`*_trades.csv`。",
        "- Q3 只有一个季度，固定 25% 的收益改善不足以证明以后都有效；若以后继续研究，应另取有分钟数据的新季度做真正的第二次外部测试，不应据本次 Q3 调参。",
    ]
    (RUN / "中文核查与结论.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run() -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    baseline_timing = verify_daily_baseline_timing()
    common, coverage, _ = feature_panel()
    q4 = mature_q4_labels()
    weight, choices, fit, valid, initial_model = select_q4_weight(common, q4)
    final_train = common[common.date < Q4_END].merge(q4, on=["date", "stockid"],
                                                     how="inner", validate="one_to_one")
    assert final_train.date.max() < Q4_END
    assert final_train.label_exit_5.max() < Q4_END
    final_model = fit_minute(final_train)
    q3 = common[common.date.between(Q3_START, Q3_END)].copy()
    scores = add_scores(q3, final_model, weight).sort_values(["date", "stockid"])
    # A nonzero predeclared candidate is reported separately even if Q4 rejects it.
    # It is a sensitivity analysis, not an outer-test-selected deployment rule.
    scores["score_fixed25"] = 0.75*scores.score_daily+0.25*scores.score_minute
    assert scores.date.nunique() == 65 and not scores.duplicated(["date", "stockid"]).any()
    assert np.isfinite(scores[["score_daily", "score_minute", "score_augmented"]]).all().all()

    # The old minute-only artifact is used solely as an independent consistency
    # check. Its Q3 answer column is intentionally never read.
    old = pd.read_parquet(OLD_MINUTE_PATH, columns=["date", "stockid", "score"])
    old.date = pd.to_datetime(old.date)
    check = scores.merge(old, on=["date", "stockid"], validate="one_to_one")
    old_corr = float(check.score_minute_raw.corr(check.score, method="spearman"))
    old_max_diff = float((check.score_minute_raw-check.score).abs().max())

    # Same dates, same rows, same rule, same market, same start capital and costs.
    execution_prices = pd.read_parquet(PRICE_PATH,
        columns=["date", "stockid", "open", "close", "adv20", "observed",
                 "mom20", "vol20", "in_pool", "adj_ret1"])
    actions = pd.read_csv(ACTIONS_PATH)
    market = CohortMarket(execution_prices, actions)
    results = {}
    for key, col in (("daily", "score_daily"),
                     ("daily_plus_minute", "score_augmented"),
                     ("daily_plus_minute_fixed25", "score_fixed25")):
        input_scores = scores[["date", "stockid", col]].rename(columns={col: "score"})
        result = market.run(input_scores, rule=Rule(name=key, topn=20),
                            start=Q3_START, end=Q3_END, cost_bp=5.0,
                            capital=1_000_000, save_details=True)
        results[key] = result
        for table in ("daily", "orders", "trades", "holdings", "rejected_selection"):
            result[table].to_csv(RUN / f"{key}_{table}.csv", index=False)
    pd.concat([results[key]["daily"].assign(strategy=key) for key in results],
              ignore_index=True).to_parquet(RUN / "account_daily.parquet", index=False)
    pd.DataFrame([{"strategy": key, **result["metrics"]}
                  for key, result in results.items()]).to_csv(
                      RUN / "account_results.csv", index=False, encoding="utf-8-sig")

    b = results["daily"]["daily"].set_index("date")
    a = results["daily_plus_minute"]["daily"].set_index("date")
    c = results["daily_plus_minute_fixed25"]["daily"].set_index("date")
    assert b.index.equals(a.index)
    assert b.index.equals(c.index)
    paired = pd.DataFrame({"date": b.index,
                           "daily_equity": b.equity.to_numpy(),
                           "daily_plus_minute_equity": a.equity.to_numpy(),
                           "daily_plus_minute_fixed25_equity": c.equity.to_numpy(),
                           "daily_cost": b.cost.to_numpy(),
                           "daily_plus_minute_cost": a.cost.to_numpy(),
                           "daily_plus_minute_fixed25_cost": c.cost.to_numpy(),
                           "daily_exposure": b.exposure.to_numpy(),
                           "daily_plus_minute_exposure": a.exposure.to_numpy(),
                           "daily_plus_minute_fixed25_exposure": c.exposure.to_numpy()})
    paired["excess_wealth_ratio"] = paired.daily_plus_minute_equity/paired.daily_equity-1.0
    paired["fixed25_excess_wealth_ratio"] = paired.daily_plus_minute_fixed25_equity/paired.daily_equity-1.0
    paired["daily_return"] = paired.daily_equity.pct_change().fillna(paired.daily_equity.iloc[0]/1e6-1)
    paired["daily_plus_minute_return"] = paired.daily_plus_minute_equity.pct_change().fillna(paired.daily_plus_minute_equity.iloc[0]/1e6-1)
    paired["daily_plus_minute_fixed25_return"] = paired.daily_plus_minute_fixed25_equity.pct_change().fillna(paired.daily_plus_minute_fixed25_equity.iloc[0]/1e6-1)
    paired["daily_return_difference"] = paired.daily_plus_minute_return-paired.daily_return
    paired["fixed25_daily_return_difference"] = paired.daily_plus_minute_fixed25_return-paired.daily_return
    save_equity_plot(paired)

    q3_ic = q3_realized_diagnostic(scores)
    q3_ic_summary = q3_ic.groupby("signal").agg(
        mean_daily_rank_ic=("rank_ic", "mean"), sd_daily_rank_ic=("rank_ic", "std"),
        labelled_days=("rank_ic", "count"), labelled_stock_rows=("stock_rows", "sum")
    ).reset_index()
    q3_ic_summary["rank_ic_ir"] = q3_ic_summary.mean_daily_rank_ic/q3_ic_summary.sd_daily_rank_ic
    coeff = pd.DataFrame({"factor": FEATURES,
                          "q4_oct_nov_fit_coefficient": initial_model.coef_,
                          "q4_full_refit_coefficient": final_model.coef_})
    for name, frame in (("coverage_daily.csv", coverage), ("q4_weight_validation.csv", choices),
                        ("factor_coefficients.csv", coeff), ("q3_ic_daily.csv", q3_ic),
                        ("q3_ic_summary.csv", q3_ic_summary), ("paired_equity.csv", paired)):
        frame.to_csv(RUN / name, index=False)
    scores.to_parquet(RUN / "q3_same_pool_predictions.parquet", index=False)

    metrics = {k: result["metrics"] for k, result in results.items()}
    summary = {
        "protocol": "minute Ridge alpha100 fitted solely to mature 2025Q4 five-session labels; Q4 December validates mix; frozen Q3",
        "daily_baseline_timing": baseline_timing,
        "features": list(FEATURES), "candidate_weights": list(WEIGHTS),
        "chosen_minute_weight": weight,
        "q4_initial_fit": {"rows": len(fit), "days": fit.date.nunique(),
                           "first_signal": fit.date.min(), "last_signal": fit.date.max(),
                           "last_label_exit": fit.label_exit_5.max()},
        "q4_validation": {"rows": len(valid), "days": valid.date.nunique(),
                          "first_signal": valid.date.min(), "last_signal": valid.date.max(),
                          "last_label_exit": valid.label_exit_5.max()},
        "q4_final_fit": {"rows": len(final_train), "days": final_train.date.nunique(),
                         "first_signal": final_train.date.min(), "last_signal": final_train.date.max(),
                         "last_label_exit": final_train.label_exit_5.max()},
        "q3": {"days": q3.date.nunique(), "common_stock_rows": len(q3),
               "daily_pool_rows": int(coverage[coverage.date.between(Q3_START, Q3_END)].daily_pool.sum()),
               "first_signal": q3.date.min(), "last_signal": q3.date.max(),
               "mean_common_fraction": float(coverage[coverage.date.between(Q3_START, Q3_END)].common_fraction.mean())},
        "old_minute_prediction_check": {"common_rows": len(check),
                                        "spearman": old_corr,
                                        "max_absolute_raw_score_difference": old_max_diff},
        "backtests": metrics,
        "net_return_difference_percentage_points": 100*(metrics["daily_plus_minute"]["total_return"]-
                                                   metrics["daily"]["total_return"]),
        "fixed25_return_difference_percentage_points": 100*(metrics["daily_plus_minute_fixed25"]["total_return"]-
                                                   metrics["daily"]["total_return"]),
        "end_relative_wealth_gain": float(paired.excess_wealth_ratio.iloc[-1]),
        "fixed25_end_relative_wealth_gain": float(paired.fixed25_excess_wealth_ratio.iloc[-1]),
        "q3_label_ic_diagnostic_only": q3_ic_summary.to_dict("records"),
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                   [DAILY_PATH, MINUTE_PATH, OLD_MINUTE_PATH, PRICE_PATH, ACTIONS_PATH]},
        "rule": asdict(Rule(name="daily_plus_minute", topn=20)),
        "cost_bp_each_trade_side": 5.0,
        "note": "Q3 answers are used only after freezing final minute coefficients and mixing weight. The daily baseline is an earlier quarterly-frozen daily model, not trained solely on Q4. Fixed25 was in the predeclared Q4 grid but Q4 did not select it, so Q3 fixed25 is sensitivity only.",
    }
    (RUN / "metrics_and_manifest.json").write_text(
        json.dumps(clean_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_chinese_audit(clean_json(summary), choices, coverage)
    return clean_json(summary)


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({k: outcome[k] for k in
                      ["chosen_minute_weight", "q4_initial_fit", "q4_validation",
                       "q4_final_fit", "q3", "old_minute_prediction_check",
                       "net_return_difference_percentage_points",
                       "fixed25_return_difference_percentage_points"]},
                     ensure_ascii=False, indent=2))
