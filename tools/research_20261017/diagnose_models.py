"""Read-only diagnostics of the frozen 2026-10-17 model experiment.

No model fitting or outer-period selection occurs here. The outcomes are read
from separate answer files only after all saved predictions have been loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "training_runs/research_20261017/models"
OUT = ROOT / "training_runs/research_20261017/model_diagnostics"
PANEL = ROOT / "training_data/research_20261011/model_panel.parquet"
TEST_END = pd.Timestamp("2026-09-30")
CUTOFF = pd.Timestamp("2026-04-01")
SCORES = ["score_quarterly", "score_monthly", "score_simple"]


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def daily_ic(date: pd.Timestamp, x: pd.DataFrame, score_col: str,
             answer_col: str) -> dict:
    a = x[[score_col, answer_col]].replace([np.inf, -np.inf], np.nan).dropna()
    count = len(a)
    if count < 5 or a[score_col].nunique() < 2 or a[answer_col].nunique() < 2:
        ic = np.nan
    else:
        ic = a[score_col].corr(a[answer_col], method="spearman")
    return {"date": date, "score": score_col, "rank_ic": ic,
            "stock_pairs": count, "usable": bool(np.isfinite(ic))}


def aggregate_ic(x: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, day in x.groupby(groups, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        clean = day.loc[day.rank_ic.notna()]
        avg = clean.rank_ic.mean() if len(clean) else np.nan
        sd = clean.rank_ic.std(ddof=1) if len(clean) > 1 else np.nan
        rows.append({**dict(zip(groups, keys)), "mean_daily_rank_ic": avg,
                     "daily_ic_std": sd,
                     "icir_not_annualized": avg / sd if sd and np.isfinite(sd) else np.nan,
                     "evaluated_days": len(clean),
                     "stock_pairs": int(clean.stock_pairs.sum()),
                     "first_evaluated_date": clean.date.min() if len(clean) else pd.NaT,
                     "last_evaluated_date": clean.date.max() if len(clean) else pd.NaT})
    return pd.DataFrame(rows)


def outer_score_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(MODEL / "daily_predictions.parquet")
    answer_parts = [pd.read_parquet(MODEL / str(q) / "outer_answers_for_later_audit.parquet")
                    for q in pd.period_range("2024Q4", "2026Q3", freq="Q")]
    answers = pd.concat(answer_parts, ignore_index=True)
    assert not predictions.duplicated(["date", "stockid"]).any()
    assert not answers.duplicated(["date", "stockid"]).any()
    merged = predictions.merge(answers, on=["date", "stockid"],
                               how="left", validate="one_to_one", indicator=True)
    assert merged._merge.eq("both").all()
    assert len(merged) == len(predictions) == 48500
    # The comparison period ends on Sep 30. Labels exiting afterward remain
    # excluded, though they are available in the later data snapshot.
    merged.loc[merged.label_exit_5 > TEST_END, "label_5"] = np.nan
    dates = merged.date.drop_duplicates().sort_values()
    day_groups = {d: g for d, g in merged.groupby("date", sort=True)}
    daily = pd.DataFrame([daily_ic(d, day_groups[d], s, "label_5")
                          for d in dates for s in SCORES])
    daily["quarter"] = daily.date.dt.to_period("Q").astype(str)
    quarterly = aggregate_ic(daily, ["quarter", "score"])
    summary = aggregate_ic(daily, ["score"])
    daily.to_csv(OUT / "outer_daily_rankic.csv", index=False)
    quarterly.to_csv(OUT / "outer_quarterly_rankic.csv", index=False)
    summary.to_csv(OUT / "outer_summary_rankic.csv", index=False)
    return daily, quarterly, summary, merged


def pca_and_turnover() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = json.loads((MODEL / "folds.json").read_text(encoding="utf-8"))
    pca_rows = []
    turnover = []
    best = []
    previous: set[str] | None = None
    previous_top10: set[str] | None = None
    for fold in folds:
        quarter = fold["quarter"]
        qdir = MODEL / quarter
        validation = json.loads((qdir / "validation_models.json").read_text(encoding="utf-8"))
        deployment = json.loads((qdir / "deployment_info.json").read_text(encoding="utf-8"))
        chosen = deployment["quarterly"]["factors"]
        current = set(chosen)
        top10 = set(chosen[:10])
        turnover.append({"quarter": quarter, "selected_name": fold["selected_name"],
                         "factor_count": len(chosen),
                         "retained_from_previous": len(current & previous) if previous is not None else np.nan,
                         "added_vs_previous": len(current - previous) if previous is not None else np.nan,
                         "removed_vs_previous": len(previous - current) if previous is not None else np.nan,
                         "jaccard_with_previous": len(current & previous) / len(current | previous)
                         if previous is not None else np.nan,
                         "top10_retained": len(top10 & previous_top10)
                         if previous_top10 is not None else np.nan,
                         "top10_replaced": 10 - len(top10 & previous_top10)
                         if previous_top10 is not None else np.nan})
        previous = current
        previous_top10 = top10
        best.append({"quarter": quarter, "selected_name": fold["selected_name"],
                     "selected_best_iteration": fold["selected_best_iteration"],
                     "selected_validation_rank_ic": fold["selected_valid_rank_ic"],
                     "deployed_factor_count": len(chosen),
                     "deployment_train_last_exit": deployment["quarterly"]["train_last_exit"]})
        for name, info in validation.items():
            if "pca" not in info:
                continue
            data = info["pca"]
            pca_rows.append({"quarter": quarter, "stage": "inner_validation",
                             "candidate": name, "selected": name == fold["selected_name"],
                             "input_features": len(info["factors"]),
                             "pca_components": data["components"],
                             "explained_variance_fraction": data["cumulative_explained"],
                             "explained_variance_each_component_json": json.dumps(
                                 data["explained_variance_ratio"]),
                             "train_last_exit": info["train_last_exit"]})
        for stage, item in [("outer_quarter_fit", deployment["quarterly"])]:
            if "pca" in item:
                data = item["pca"]
                pca_rows.append({"quarter": quarter, "stage": stage,
                                 "candidate": fold["selected_name"], "selected": True,
                                 "input_features": len(item["factors"]),
                                 "pca_components": data["components"],
                                 "explained_variance_fraction": data["cumulative_explained"],
                                 "explained_variance_each_component_json": json.dumps(
                                     data["explained_variance_ratio"]),
                                 "train_last_exit": item["train_last_exit"]})
        for item in deployment["monthly"]:
            if "pca" not in item:
                continue
            data = item["pca"]
            pca_rows.append({"quarter": quarter, "stage": f"monthly_{item['month']}",
                             "candidate": fold["selected_name"], "selected": True,
                             "input_features": len(item["factors"]),
                             "pca_components": data["components"],
                             "explained_variance_fraction": data["cumulative_explained"],
                             "explained_variance_each_component_json": json.dumps(
                                 data["explained_variance_ratio"]),
                             "train_last_exit": item["train_last_exit"]})
    pca = pd.DataFrame(pca_rows)
    turns = pd.DataFrame(turnover)
    best_df = pd.DataFrame(best)
    pca.to_csv(OUT / "pca_trace.csv", index=False)
    turns.to_csv(OUT / "factor_turnover.csv", index=False)
    best_df.to_csv(OUT / "best_iterations.csv", index=False)
    return pca, turns, best_df


def factor_ic_by_day(frame: pd.DataFrame, factor: str, horizon: int) -> pd.DataFrame:
    label = f"label_{horizon}"
    exit_date = f"label_exit_{horizon}"
    a = frame.loc[(frame[exit_date] < CUTOFF) & frame[label].notna(),
                  ["date", "stockid", factor, label]].copy()
    rows = []
    for date, day in a.groupby("date", sort=True):
        metric = daily_ic(date, day, factor, label)
        rows.append(metric)
    return pd.DataFrame(rows)


def factor_horizons_and_recency() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deployment = json.loads((MODEL / "2026Q2/deployment_info.json").read_text(encoding="utf-8"))
    qtrain = deployment["quarterly"]
    first = pd.Timestamp(qtrain["train_first_signal"])
    last = pd.Timestamp(qtrain["train_last_signal"])
    factors = qtrain["factors"][:10]
    columns = ["date", "stockid"] + factors
    for h in (1, 5, 20):
        columns.extend([f"label_{h}", f"label_exit_{h}"])
    panel = pd.read_parquet(PANEL, columns=columns)
    panel = panel[(panel.date >= first) & (panel.date <= last) & (panel.date < CUTOFF)]
    assert panel.date.nunique() <= 252
    # Keep only dates where every horizon has a reasonably complete mature
    # cross-section; then h1/h5/h20 differences cannot arise from date choice.
    date_counts = {}
    for h in (1, 5, 20):
        good = (panel[f"label_exit_{h}"] < CUTOFF) & panel[f"label_{h}"].notna()
        date_counts[h] = panel.loc[good].groupby("date").size()
    common = set(date_counts[1][date_counts[1] >= 30].index)
    for h in (5, 20):
        common &= set(date_counts[h][date_counts[h] >= 30].index)
    common_dates = pd.DatetimeIndex(sorted(common))
    assert len(common_dates) >= 150
    common_panel = panel[panel.date.isin(common_dates)]
    horizon_rows = []
    day_parts = []
    for rank, factor in enumerate(factors, 1):
        for h in (1, 5, 20):
            d = factor_ic_by_day(common_panel, factor, h)
            d["factor"] = factor; d["horizon"] = h
            day_parts.append(d)
            clean = d[d.rank_ic.notna()]
            avg = clean.rank_ic.mean(); sd = clean.rank_ic.std(ddof=1)
            horizon_rows.append({"selected_order": rank, "factor": factor,
                                 "horizon_sessions": h,
                                 "mean_daily_rank_ic": avg, "daily_ic_std": sd,
                                 "icir_not_annualized": avg / sd if sd else np.nan,
                                 "evaluated_days": len(clean),
                                 "stock_pairs": int(clean.stock_pairs.sum()),
                                 "common_first_signal": common_dates.min(),
                                 "common_last_signal": common_dates.max()})
    horizon = pd.DataFrame(horizon_rows)
    recency_rows = []
    all_mature_h5 = panel.loc[(panel.label_exit_5 < CUTOFF) & panel.label_5.notna(),
                              "date"].drop_duplicates().sort_values()
    for span in (21, 63, 126):
        recent_dates = pd.DatetimeIndex(all_mature_h5.iloc[-span:])
        recent_panel = panel[panel.date.isin(recent_dates)]
        for rank, factor in enumerate(factors, 1):
            d = factor_ic_by_day(recent_panel, factor, 5)
            clean = d[d.rank_ic.notna()]
            avg = clean.rank_ic.mean(); sd = clean.rank_ic.std(ddof=1)
            recency_rows.append({"selected_order": rank, "factor": factor,
                                 "recent_mature_signal_days_requested": span,
                                 "mean_daily_rank_ic": avg, "daily_ic_std": sd,
                                 "icir_not_annualized": avg / sd if sd else np.nan,
                                 "evaluated_days": len(clean),
                                 "stock_pairs": int(clean.stock_pairs.sum()),
                                 "first_signal": recent_dates.min(),
                                 "last_signal": recent_dates.max(),
                                 "last_label_exit": recent_panel.loc[
                                     recent_panel.label_exit_5 < CUTOFF, "label_exit_5"].max()})
    recent = pd.DataFrame(recency_rows)
    days = pd.concat(day_parts, ignore_index=True)
    horizon.to_csv(OUT / "factor_horizon_ic_common_dates.csv", index=False)
    recent.to_csv(OUT / "factor_recent_h5_ic.csv", index=False)
    days.to_csv(OUT / "factor_horizon_daily_ic.csv", index=False)
    return horizon, recent, days


def report(daily: pd.DataFrame, quarterly: pd.DataFrame, summary: pd.DataFrame,
           pca: pd.DataFrame, turns: pd.DataFrame, best: pd.DataFrame,
           horizon: pd.DataFrame, recent: pd.DataFrame) -> None:
    s = summary.set_index("score")
    q3 = quarterly[quarterly.quarter == "2026Q3"].set_index("score")
    selected_pca = pca[(pca.stage == "outer_quarter_fit") & pca.selected]
    top = horizon[horizon.horizon_sessions == 5].sort_values("selected_order").head(3)
    lead_h = horizon[horizon.selected_order == 1].set_index("horizon_sessions")
    lead_r = recent[recent.selected_order == 1].set_index("recent_mature_signal_days_requested")
    text = ["# 五日模型效果诊断（冻结预测，仅事后评估）", "",
            "诊断日期：2026-10-18。预测文件为已冻结的 2024-10-08—2026-09-30 外层分数；本次没有重新训练、调整参数，也没有根据外层结果更改胜出方案。",
            "",
            "## 模型排序效果", "",
            "这里每天先在可比较的股票中排分数名次、再排实际五日收益名次，计算两个名次的相关程度；越接近 +1 越同向。只使用第五日实际退出日期不晚于 2026-09-30 的答案，故最后约五个信号日不计入本段评分。ICIR 是每日 IC 平均值除以每日 IC 的标准差，未做年化。这些是**选股排序**指标，不是资金收益、夏普或最大回撤。", "",
            "| 分数 | 每日IC均值 | ICIR | 可评估日 | 股票配对数 |", "|---|---:|---:|---:|---:|"]
    labels = {"score_quarterly": "季度重训", "score_monthly": "月度重训", "score_simple": "有符号IC前20因子"}
    for name in SCORES:
        r = s.loc[name]
        text.append(f"| {labels[name]} | {r.mean_daily_rank_ic:.4f} | {r.icir_not_annualized:.3f} | {int(r.evaluated_days)} | {int(r.stock_pairs)} |")
    text.extend(["", "季度结果见 `outer_quarterly_rankic.csv`。例如 2026Q3：" +
                 "、".join(f"{labels[n]}每日IC {q3.loc[n, 'mean_daily_rank_ic']:.3f}" for n in SCORES) +
                 "。这说明排序关系在该阶段的表现，仍需结合真实可成交资金账本判断收益。", "",
                 "## 验证轮数、PCA与因子更换", "",
                 "各季从验证期选出的最佳轮数依次是 " +
                 "、".join(f"{r.quarter} {int(r.selected_best_iteration)}轮" for r in best.itertuples()) +
                 "。逐轮训练/验证误差保存在模型目录各季的 `loss_curves.csv`。",
                 "",
                 f"最终被选中并正式用于 2024Q4 的 PCA 模型，从100个训练期筛出的原始因子压缩为{int(selected_pca.iloc[0].pca_components)}个主成分，训练期保留方差{selected_pca.iloc[0].explained_variance_fraction:.1%}。所有PCA候选每季实际解释比例及20个单独成分见 `pca_trace.csv`；PCA输入未按高相关阈值删列。压成20个方向后仍可能舍弃剩余信息。",
                 "",
                 "各季入选因子与上季的交集、新增、退出数量见 `factor_turnover.csv`。例如 " +
                 f"{turns.iloc[-1].quarter} 保留上季 {int(turns.iloc[-1].retained_from_previous)} 列，新增 {int(turns.iloc[-1].added_vs_previous)} 列，退出 {int(turns.iloc[-1].removed_vs_previous)} 列。PCA季度与普通模型季度的列选择方法不同，跨方法的更换数量不能全解释为市场漂移。",
                 "", "| 季度 | 最佳轮数 | 入选列数 | 比上季新增 | 比上季退出 | 前10列换掉 |",
                 "|---|---:|---:|---:|---:|---:|"])
    for i, (t, b) in enumerate(zip(turns.itertuples(), best.itertuples())):
        text.append(f"| {t.quarter} | {int(b.selected_best_iteration)} | {int(t.factor_count)} | " +
                    ("— | — | — |" if i == 0 else
                     f"{int(t.added_vs_previous)} | {int(t.removed_vs_previous)} | {int(t.top10_replaced)} |"))
    text.extend([
                 "",
                 "## 2026Q2选中因子的持有期与近期变化", "",
                 "使用2026-04-01切点前、该季正式训练窗口里的前10个因子。比较1/5/20日时，三种答案只用**同一批信号日期**且都已在切点前成熟；这样日期变化不混进持有期比较。每个因子具体每日IC、均值和ICIR见 `factor_horizon_ic_common_dates.csv`。前3个因子的五日IC为 " +
                 "、".join(f"{r.factor} {r.mean_daily_rank_ic:.3f}" for r in top.itertuples()) +
                 "。负数表示因子越小，后续收益排名通常越高。", "",
                 f"以入选第一的 x_c02_gtalpha063 为例，在相同的{int(lead_h.loc[5, 'evaluated_days'])}个信号日（" +
                 f"{pd.Timestamp(lead_h.loc[5, 'common_first_signal']).date()}至{pd.Timestamp(lead_h.loc[5, 'common_last_signal']).date()}），1/5/20日IC分别是 " +
                 f"{lead_h.loc[1, 'mean_daily_rank_ic']:.3f} / {lead_h.loc[5, 'mean_daily_rank_ic']:.3f} / {lead_h.loc[20, 'mean_daily_rank_ic']:.3f}。" +
                 "该因子在五日口径最强，20日仍有关系，不能概括成天数越长就一直变弱。", "",
                 "最近21/63/126日只比较五日答案：先在切点前找到已成熟的信号日，再向前数21、63、126个交易日；每段实际起止日期、IC与样本数见 `factor_recent_h5_ic.csv`。这用于看同一持有期的**日历时间变化**，不能把它解释成1日到20日的持有期衰减。", "",
                 f"同一因子 x_c02_gtalpha063 的最近21/63/126个成熟信号日五日IC分别为 " +
                 f"{lead_r.loc[21, 'mean_daily_rank_ic']:.3f} / {lead_r.loc[63, 'mean_daily_rank_ic']:.3f} / {lead_r.loc[126, 'mean_daily_rank_ic']:.3f}。" +
                 "最近21日明显弱于较长窗口，提示临近2026-04-01时关系变弱；21日样本较短，需看后续季度能否重复。", "",
                 "1日答案采用次日开盘至再下一日开盘；5日和20日采用次日开盘至第5/20日收盘。因退出时点也不同，三种IC差异不能全部归因于持有天数。历史后复权价格与原始因子发布时点仍有上游历史可得性限制，本诊断只核查既定模型输出和标签成熟日期。"])
    (OUT / "模型效果诊断.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily, quarterly, summary, merged = outer_score_diagnostics()
    pca, turns, best = pca_and_turnover()
    horizon, recent, _ = factor_horizons_and_recency()
    report(daily, quarterly, summary, pca, turns, best, horizon, recent)
    save_json(OUT / "diagnostic_checks.json", {
        "prediction_rows": len(merged), "prediction_dates": int(merged.date.nunique()),
        "daily_score_rows": len(daily), "quarterly_score_rows": len(quarterly),
        "pca_trace_rows": len(pca), "factor_turnover_quarters": len(turns),
        "horizon_comparison_rows": len(horizon), "recent_h5_rows": len(recent),
        "no_model_training_or_outer_selection": True,
        "answer_last_exit_for_ic": str(TEST_END.date()),
        "factor_cutoff_exclusive": str(CUTOFF.date())})
    print("COMPLETE", len(merged), "predictions", len(daily), "daily score rows", flush=True)


if __name__ == "__main__":
    main()
