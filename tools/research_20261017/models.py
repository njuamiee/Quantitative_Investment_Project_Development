"""Point-in-time, quarterly-selected five-session stock ranking experiments.

The signal uses information available after the signal day's close. A label is
usable only after its actual exit date, strictly before the next fit boundary.
This module owns model selection/predictions; portfolio accounting is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = ROOT / "training_data/research_20261011/model_panel.parquet"
DEFAULT_OUT = ROOT / "training_runs/research_20261017/models"
SEED = 20261017
OUTER_START = pd.Timestamp("2024-10-08")
OUTER_END = pd.Timestamp("2026-09-30")


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    sessions: int
    features: int
    half_life: int | None


CANDIDATES = [
    Candidate("raw_504_40_equal", "lgbm", 504, 40, None),
    Candidate("raw_504_100_equal", "lgbm", 504, 100, None),
    Candidate("raw_252_40_half63", "lgbm", 252, 40, 63),
    Candidate("raw_252_100_half63", "lgbm", 252, 100, 63),
    Candidate("raw_504_100_half126", "lgbm", 504, 100, 126),
    Candidate("pca_lgbm_504_100_equal", "pca_lgbm", 504, 100, None),
    Candidate("pca_lgbm_252_100_half63", "pca_lgbm", 252, 100, 63),
    Candidate("signed_ic20_504_equal", "signed_ic", 504, 20, None),
    Candidate("pca_ridge_504_100_equal", "pca_ridge", 504, 100, None),
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def matured(panel: pd.DataFrame, signal_dates: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rebuild cross-sectional answers from labels known at the fit boundary.

    The archived `target_5` was ranked using the full final panel. Re-ranking
    the known labels avoids even indirect use of an unfinished peer's answer.
    """
    part = panel.loc[panel.date.isin(signal_dates) & (panel.label_exit_5 < cutoff)
                     & panel.label_5.notna()].copy()
    part["target_now"] = part.groupby("date").label_5.rank(pct=True).sub(.5)
    return part


def row_weights(frame: pd.DataFrame, half_life: int | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(frame), dtype=np.float64)
    dates = np.sort(frame.date.unique())
    positions = pd.Index(dates).get_indexer(frame.date)
    age = len(dates) - 1 - positions
    weights = np.exp2(-age / half_life)
    return weights / weights.mean()


def daily_factor_ic(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise daily Pearson on already cross-sectionally ranked columns."""
    x = frame[features].to_numpy(dtype=np.float32, copy=False)
    y = frame.target_now.to_numpy(dtype=np.float64)
    dates = frame.date.to_numpy()
    unique_dates, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], len(frame)]
    values = np.full((len(unique_dates), len(features)), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(starts, ends)):
        v = x[a:b].astype(np.float64, copy=False)
        z = y[a:b]
        mask = np.isfinite(v) & np.isfinite(z[:, None])
        n = mask.sum(axis=0)
        xv = np.where(mask, v, 0)
        yv = np.where(mask, z[:, None], 0)
        sx = xv.sum(axis=0); sy = yv.sum(axis=0)
        sxx = np.square(xv).sum(axis=0); syy = np.square(yv).sum(axis=0)
        sxy = (xv * yv).sum(axis=0)
        den = np.sqrt(np.maximum(n * sxx - sx * sx, 0) * np.maximum(n * syy - sy * sy, 0))
        values[i] = np.divide(n * sxy - sx * sy, den, out=np.full(len(features), np.nan),
                              where=(n >= 5) & (den > 1e-12))
    return unique_dates, values


def screen(frame: pd.DataFrame, features: list[str], half_life: int | None,
           maximum: int = 100, deduplicate: bool = True) -> tuple[list[str], pd.DataFrame]:
    dates, ics = daily_factor_ic(frame, features)
    daily_weight = np.ones(len(dates)) if half_life is None else np.exp2(
        -(len(dates) - 1 - np.arange(len(dates))) / half_life)
    valid = np.isfinite(ics)
    den = (valid * daily_weight[:, None]).sum(axis=0)
    mean = np.divide(np.nansum(ics * daily_weight[:, None], axis=0), den,
                     out=np.full(len(features), np.nan), where=den > 0)
    cov = frame[features].notna().mean().to_numpy()
    card = pd.DataFrame({"factor": features, "coverage": cov, "mean_daily_rank_ic": mean})
    card["abs_ic"] = card.mean_daily_rank_ic.abs()
    card = card.sort_values(["abs_ic", "factor"], ascending=[False, True],
                            na_position="last").reset_index(drop=True)
    eligible = card[(card.coverage >= .7) & card.abs_ic.notna()].factor.tolist()
    if not eligible:
        raise ValueError("No factor meets historical coverage >= 0.70")
    correlations = None
    if deduplicate:
        matrix = frame[eligible].fillna(0).to_numpy(dtype=np.float32)
        # Correlation is fit solely to the historical training window.
        correlations = np.abs(np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0))
    chosen_idx: list[int] = []
    reason: dict[str, str] = {}
    for idx, name in enumerate(eligible):
        if len(chosen_idx) >= maximum:
            reason[name] = "rank_after_feature_limit"
        elif deduplicate and chosen_idx and np.max(correlations[idx, chosen_idx]) >= .92:
            peer = chosen_idx[int(np.argmax(correlations[idx, chosen_idx]))]
            reason[name] = f"abs_corr_ge_0.92_with:{eligible[peer]}"
        else:
            reason[name] = "selected_with_corr_dedup" if deduplicate else "selected_by_abs_ic_without_corr_dedup"
            chosen_idx.append(idx)
    chosen = [eligible[i] for i in chosen_idx]
    card["selection_reason"] = card.factor.map(reason).fillna("coverage_below_0.70_or_no_ic")
    card["chosen_order"] = card.factor.map({name: i + 1 for i, name in enumerate(chosen)})
    return chosen, card


def rank_ic_daily(frame: pd.DataFrame, score: np.ndarray) -> tuple[float, pd.DataFrame]:
    x = pd.DataFrame({"date": frame.date.to_numpy(), "stockid": frame.stockid.to_numpy(),
                      "score": score, "answer": frame.target_now.to_numpy()})
    x = x.dropna(subset=["score", "answer"])
    rows = []
    for date, day in x.groupby("date", sort=True):
        if len(day) < 5:
            continue
        a = day.score.rank().to_numpy(); b = day.answer.rank().to_numpy()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        rows.append({"date": date, "rank_ic": float(np.corrcoef(a, b)[0, 1]), "stocks": len(day)})
    record = pd.DataFrame(rows)
    return (float(record.rank_ic.mean()) if len(record) else float("nan"), record)


def prepare_xy(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return frame[cols].fillna(0).to_numpy(dtype=np.float32)


def lgbm_params(rounds: int) -> dict:
    return dict(n_estimators=rounds, learning_rate=.035, num_leaves=15, max_depth=5,
                min_child_samples=160, reg_lambda=12., colsample_bytree=.8,
                subsample=.8, subsample_freq=1, n_jobs=3, random_state=SEED,
                verbosity=-1, objective="regression", metric="l2")


def pca_transform(train_x: np.ndarray, apply_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA, dict]:
    components = min(20, train_x.shape[1], train_x.shape[0] - 1)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=SEED)
    train_z = pca.fit_transform(train_x)
    apply_z = pca.transform(apply_x)
    variance = {"components": components, "explained_variance_ratio":
                pca.explained_variance_ratio_.tolist(), "cumulative_explained":
                float(pca.explained_variance_ratio_.sum())}
    return train_z, apply_z, pca, variance


def fit_candidate(candidate: Candidate, train: pd.DataFrame, apply: pd.DataFrame,
                  selected: list[str], card: pd.DataFrame, *,
                  rounds: int | None = None, validate: bool = False) -> tuple[np.ndarray, dict, dict]:
    cols = selected[:candidate.features]
    if len(cols) < candidate.features:
        raise ValueError(f"{candidate.name}: only {len(cols)} screened factors")
    x = prepare_xy(train, cols)
    z = prepare_xy(apply, cols)
    weights = row_weights(train, candidate.half_life)
    info = {"factors": cols, "train_rows": len(train), "apply_rows": len(apply),
            "train_first_signal": str(train.date.min().date()),
            "train_last_signal": str(train.date.max().date()),
            "train_last_exit": str(train.label_exit_5.max().date()),
            "feature_count": len(cols), "half_life_sessions": candidate.half_life,
            "training_sessions_requested": candidate.sessions}
    if candidate.kind == "signed_ic":
        ic = card.set_index("factor").loc[cols, "mean_daily_rank_ic"].to_numpy(dtype=float)
        factor_weights = ic / np.maximum(np.abs(ic).sum(), 1e-12)
        info["signed_factor_weights"] = dict(zip(cols, factor_weights.tolist()))
        return z @ factor_weights, info, {"kind": "signed_ic", "weights": factor_weights,
                                           "factors": cols}
    pca = None
    if candidate.kind.startswith("pca_"):
        x, z, pca, variance = pca_transform(x, z)
        info["pca"] = variance
    if candidate.kind == "pca_ridge":
        model = Ridge(alpha=180)
        model.fit(x, train.target_now.to_numpy(), sample_weight=weights)
        info["ridge_alpha"] = 180
        return model.predict(z), info, {"kind": candidate.kind, "model": model,
                                         "pca": pca, "factors": cols}
    model = lgb.LGBMRegressor(**lgbm_params(300 if validate else int(rounds or 300)))
    history: dict = {}
    callbacks = [lgb.record_evaluation(history)]
    kwargs = {}
    if validate:
        # Round count is chosen by validation L2. Between models, use daily RankIC.
        kwargs.update(eval_set=[(x, train.target_now.to_numpy()),
                                (z, apply.target_now.to_numpy())],
                      eval_names=["training", "validation"],
                      callbacks=callbacks + [lgb.early_stopping(30, first_metric_only=True, verbose=False)])
    model.fit(x, train.target_now.to_numpy(), sample_weight=weights, **kwargs)
    info["best_iteration"] = int(model.best_iteration_ or model.n_estimators_)
    info["trained_iterations"] = int(model.n_estimators_)
    if validate:
        info["training_l2_by_round"] = history["training"]["l2"]
        info["validation_l2_by_round"] = history["validation"]["l2"]
    return model.predict(z), info, {"kind": candidate.kind, "model": model,
                                     "pca": pca, "factors": cols}


def predict_fitted(fitted: dict, frame: pd.DataFrame) -> np.ndarray:
    """Use the exact fitted validation model on dates with unavailable answers."""
    x = prepare_xy(frame, fitted["factors"])
    if fitted["kind"] == "signed_ic":
        return x @ fitted["weights"]
    if fitted["pca"] is not None:
        x = fitted["pca"].transform(x)
    return fitted["model"].predict(x)


def quarter_dates(dates: pd.DatetimeIndex, start: pd.Timestamp) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    history = dates[dates < start]
    if len(history) < 126 + 252:
        raise ValueError(f"Insufficient history for {start.date()}")
    return history[-126:], history


def train_slice(panel: pd.DataFrame, history: pd.DatetimeIndex,
                cutoff: pd.Timestamp, sessions: int) -> pd.DataFrame:
    frame = matured(panel, history[-sessions:], cutoff)
    return frame.sort_values(["date", "stockid"]).reset_index(drop=True)


def evaluate_quarter(panel: pd.DataFrame, dates: pd.DatetimeIndex,
                     quarter: pd.Period, out: Path, features: list[str]) -> tuple[dict, pd.DataFrame]:
    qdates = dates[dates.to_period("Q") == quarter]
    qdates = qdates[(qdates >= OUTER_START) & (qdates <= OUTER_END)]
    cutoff = qdates.min()
    valid_dates, history = quarter_dates(dates, cutoff)
    valid = matured(panel, valid_dates, cutoff).sort_values(["date", "stockid"]).reset_index(drop=True)
    valid_all = panel.loc[panel.date.isin(valid_dates)].sort_values(
        ["date", "stockid"]).reset_index(drop=True)
    if valid.date.nunique() < 100:
        raise ValueError(f"Only {valid.date.nunique()} mature validation days for {quarter}")
    qdir = out / str(quarter)
    qdir.mkdir(parents=True, exist_ok=True)
    train_cache: dict[tuple[int, int | None, bool], tuple[pd.DataFrame, list[str], pd.DataFrame]] = {}
    # All 126 signal days need predictions for a fair risk/account validation.
    # Answers for signals not mature at cutoff stay null in this output.
    prediction = valid_all[["date", "stockid"]].merge(
        valid[["date", "stockid", "target_now", "label_5", "label_exit_5"]],
        on=["date", "stockid"], how="left", validate="one_to_one")
    metrics = []
    curves = []
    summaries = {}
    fitted_models = {}
    for candidate in CANDIDATES:
        deduplicate = not candidate.kind.startswith("pca_")
        key = candidate.sessions, candidate.half_life, deduplicate
        if key not in train_cache:
            train_history = history[history < valid_dates.min()]
            train = train_slice(panel, train_history, valid_dates.min(), candidate.sessions)
            selected, card = screen(train, features, candidate.half_life,
                                    deduplicate=deduplicate)
            card.to_csv(qdir / f"factor_screen_{candidate.sessions}_{candidate.half_life or 'equal'}_{'raw_dedup' if deduplicate else 'pca_keepcorr'}.csv", index=False)
            train_cache[key] = train, selected, card
        train, selected, card = train_cache[key]
        score, info, fitted = fit_candidate(candidate, train, valid, selected, card, validate=True)
        ic, daily_ic = rank_ic_daily(valid, score)
        score_all = predict_fitted(fitted, valid_all)
        assert np.allclose(score_all[valid_all.set_index(["date", "stockid"]).index.isin(
            valid.set_index(["date", "stockid"]).index)], score, rtol=1e-5, atol=1e-7)
        prediction[f"score_{candidate.name}"] = score_all
        fitted_models[candidate.name] = fitted
        daily_ic.to_csv(qdir / f"valid_daily_ic_{candidate.name}.csv", index=False)
        metrics.append({"candidate": candidate.name, "valid_rank_ic": ic,
                        "valid_days": len(daily_ic), "train_rows": len(train),
                        "valid_rows": len(valid), "train_last_exit": str(train.label_exit_5.max().date()),
                        "best_iteration": info.get("best_iteration"),
                        "feature_count": len(info["factors"]),
                        "pca_variance_cumulative": info.get("pca", {}).get("cumulative_explained")})
        if "training_l2_by_round" in info:
            for i, (t, v) in enumerate(zip(info.pop("training_l2_by_round"),
                                            info.pop("validation_l2_by_round")), 1):
                curves.append({"candidate": candidate.name, "round": i,
                               "training_l2": t, "validation_l2": v})
        summaries[candidate.name] = info
        print(f"{quarter} validation {candidate.name}: RankIC={ic:.4f}", flush=True)
    ranked = sorted(metrics, key=lambda row: (-np.nan_to_num(row["valid_rank_ic"], nan=-999),
                                              row["candidate"]))
    winner = ranked[0]["candidate"]
    prediction["score"] = prediction[f"score_{winner}"]
    prediction["score_simple"] = prediction["score_signed_ic20_504_equal"]
    prediction.sort_values(["date", "stockid"]).to_parquet(qdir / "valid_predictions.parquet", index=False)
    pd.DataFrame(metrics).sort_values("valid_rank_ic", ascending=False).to_csv(qdir / "candidate_metrics.csv", index=False)
    if curves:
        pd.DataFrame(curves).to_csv(qdir / "loss_curves.csv", index=False)
    joblib.dump(fitted_models, qdir / "validation_fitted_models.joblib", compress=3)
    write_json(qdir / "validation_models.json", summaries)
    fold = {"quarter": str(quarter), "cutoff": str(cutoff.date()),
            "valid_start": str(valid_dates.min().date()),
            "valid_end": str(valid_dates.max().date()),
            "valid_last_mature_exit": str(valid.label_exit_5.max().date()),
            "selected_name": winner, "selected_valid_rank_ic": ranked[0]["valid_rank_ic"],
            "selected_best_iteration": summaries[winner].get("best_iteration"),
            "candidate_ranking": ranked,
            "why_selected": "highest mean daily validation RankIC; tie broken by fixed name order",
            "valid_label_rule": "predictions cover all 126 days; labels only if label_exit_5 < quarter cutoff; target re-ranked among matured labels",
            "train_label_rule": "label_exit_5 < validation first signal day",
            "test_start": str(qdates.min().date()), "test_end": str(qdates.max().date())}
    write_json(qdir / "selection.json", fold)
    return fold, prediction


def deploy(panel: pd.DataFrame, dates: pd.DatetimeIndex, fold: dict,
           out: Path, features: list[str]) -> pd.DataFrame:
    quarter = pd.Period(fold["quarter"], freq="Q")
    qdir = out / str(quarter)
    qdates = dates[(dates.to_period("Q") == quarter) &
                   (dates >= OUTER_START) & (dates <= OUTER_END)]
    test = panel.loc[panel.date.isin(qdates)].copy().sort_values(["date", "stockid"]).reset_index(drop=True)
    candidate = next(c for c in CANDIDATES if c.name == fold["selected_name"])
    deduplicate = not candidate.kind.startswith("pca_")
    cutoff = pd.Timestamp(fold["cutoff"])
    hist = dates[dates < cutoff]
    train = train_slice(panel, hist, cutoff, candidate.sessions)
    selected, card = screen(train, features, candidate.half_life,
                            deduplicate=deduplicate)
    card.to_csv(qdir / "deployment_factor_screen.csv", index=False)
    n = fold["selected_best_iteration"]
    quarterly, qinfo, qobj = fit_candidate(candidate, train, test, selected, card,
                                           rounds=n, validate=False)
    # Fixed, interpretable benchmark: top 20 signed historical ICs, reselected
    # using only labels mature when the quarter begins.
    simple = next(c for c in CANDIDATES if c.kind == "signed_ic")
    simple_train = train_slice(panel, hist, cutoff, simple.sessions)
    simple_cols, simple_card = screen(simple_train, features, simple.half_life)
    simple_card.to_csv(qdir / "simple_factor_screen.csv", index=False)
    score_simple, sinfo, sobj = fit_candidate(simple, simple_train, test,
                                              simple_cols, simple_card)
    monthly = np.empty(len(test), dtype=float)
    monthly_details = []
    month_models = {}
    for month in sorted(test.date.dt.to_period("M").unique()):
        rows = test.date.dt.to_period("M") == month
        month_start = test.loc[rows, "date"].min()
        month_hist = dates[dates < month_start]
        month_train = train_slice(panel, month_hist, month_start, candidate.sessions)
        mcols, mcard = screen(month_train, features, candidate.half_life,
                              deduplicate=deduplicate)
        mcard.to_csv(qdir / f"monthly_factor_screen_{month}.csv", index=False)
        score, minfo, mobj = fit_candidate(candidate, month_train, test.loc[rows],
                                           mcols, mcard, rounds=n, validate=False)
        monthly[np.flatnonzero(rows.to_numpy())] = score
        monthly_details.append({"month": str(month), "refit_cutoff": str(month_start.date()),
                                **minfo})
        month_models[str(month)] = mobj
    # Keep future outcomes in a separate audit file so a trading consumer can
    # read predictions without ever seeing the answer columns.
    test[["date", "stockid", "label_5", "label_exit_5"]].to_parquet(
        qdir / "outer_answers_for_later_audit.parquet", index=False)
    result = test[["date", "stockid"]].copy()
    result["score_quarterly"] = quarterly
    result["score_monthly"] = monthly
    result["score_simple"] = score_simple
    result["quarter"] = str(quarter)
    result = result.sort_values(["date", "stockid"]).reset_index(drop=True)
    result.to_parquet(qdir / "outer_predictions.parquet", index=False)
    write_json(qdir / "deployment_info.json", {"quarterly": qinfo, "monthly": monthly_details,
                                               "simple": sinfo,
                                               "quarterly_label_cutoff": str(cutoff.date())})
    joblib.dump({"quarterly": qobj, "monthly": month_models, "simple": sobj},
                qdir / "fitted_models.joblib", compress=3)
    return result


def verify(out: Path, folds: list[dict], predictions: pd.DataFrame) -> None:
    assert len(folds) == 8, f"Expected eight quarterly folds, found {len(folds)}"
    assert predictions.date.min() == OUTER_START
    assert predictions.date.max() == OUTER_END
    assert not predictions.duplicated(["date", "stockid"]).any()
    assert predictions[["score_quarterly", "score_monthly", "score_simple"]].notna().all().all()
    for fold in folds:
        cutoff = pd.Timestamp(fold["cutoff"])
        valid_start = pd.Timestamp(fold["valid_start"])
        assert pd.Timestamp(fold["valid_last_mature_exit"]) < cutoff
        validation = pd.read_parquet(out / fold["quarter"] / "valid_predictions.parquet")
        assert validation.date.nunique() == 126
        assert validation.date.min() == valid_start
        assert validation.date.max() == pd.Timestamp(fold["valid_end"])
        assert validation[["score", "score_simple"]].notna().all().all()
        assert validation.loc[validation.target_now.isna(), "label_5"].isna().all()
        details = json.loads((out / fold["quarter"] / "validation_models.json").read_text(encoding="utf-8"))
        assert all(pd.Timestamp(v["train_last_exit"]) < valid_start for v in details.values())
        deployment = json.loads((out / fold["quarter"] / "deployment_info.json").read_text(encoding="utf-8"))
        assert pd.Timestamp(deployment["quarterly"]["train_last_exit"]) < cutoff
        for month in deployment["monthly"]:
            assert pd.Timestamp(month["train_last_exit"]) < pd.Timestamp(month["refit_cutoff"])
    write_json(out / "model_verification.json", {"folds": len(folds), "rows": len(predictions),
                                                 "dates": int(predictions.date.nunique()),
                                                 "unique_date_stockid": True,
                                                 "all_labels_mature_before_each_fit": True,
                                                 "all_scores_finite": True})


def main(panel_path: Path, out: Path, validation_only: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    panel_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    panel = pd.read_parquet(panel_path).sort_values(["date", "stockid"]).reset_index(drop=True)
    panel = panel.loc[panel.date <= OUTER_END].copy()
    features = [c for c in panel if c.startswith("x_")]
    needed = {"date", "stockid", "label_5", "label_exit_5"}
    if not needed.issubset(panel.columns):
        raise ValueError(f"Missing {needed - set(panel.columns)}")
    if panel.duplicated(["date", "stockid"]).any():
        raise ValueError("Duplicate date/stockid in model panel")
    dates = pd.DatetimeIndex(np.sort(panel.date.unique()))
    quarters = sorted(dates[(dates >= OUTER_START) &
                            (dates <= OUTER_END)].to_period("Q").unique())
    write_json(out / "experiment_design.json", {"input_panel": str(panel_path),
               "panel_sha256": panel_hash, "feature_columns": len(features),
               "output_start": str(OUTER_START.date()), "output_end": str(OUTER_END.date()),
               "validation_sessions": 126, "candidates": [asdict(c) for c in CANDIDATES],
               "maximum_lgbm_rounds": 300, "early_stop_no_improve_rounds": 30,
               "selection_metric": "mean daily Spearman correlation with mature 5-session target",
               "pca_components": 20, "factor_min_coverage": .70,
               "factor_max_abs_pairwise_correlation": .92,
               "pca_factor_rule": "top 100 by training-only abs daily IC after coverage>=0.70, without correlation deletion",
               "raw_factor_rule": "top 40/100 by training-only abs daily IC after coverage>=0.70 and abs correlation<0.92",
               "feature_screen_uses_only_prevalidation_or_predeployment_mature_labels": True,
               "quarterly_model_settings_fixed_until_next_quarter": True,
               "monthly_model_uses_quarterly_selected_candidate_and_round_count": True,
               "signal_available_after_close": True})
    folds = []; outputs = []
    for quarter in quarters:
        old_selection = out / str(quarter) / "selection.json"
        previous = json.loads(old_selection.read_text(encoding="utf-8")) if validation_only and old_selection.exists() else None
        fold, _ = evaluate_quarter(panel, dates, quarter, out, features)
        if previous is not None:
            assert (fold["selected_name"], fold["selected_best_iteration"]) == (
                previous["selected_name"], previous["selected_best_iteration"]), quarter
        if validation_only:
            continue
        result = deploy(panel, dates, fold, out, features)
        outputs.append(result); folds.append(fold)
        print(f"{quarter} deployment {fold['selected_name']} rows={len(result)}", flush=True)
    all_predictions = pd.concat(outputs, ignore_index=True).sort_values(["date", "stockid"])
    all_predictions.to_parquet(out / "daily_predictions.parquet", index=False)
    pd.DataFrame([{k: f[k] for k in ["quarter", "cutoff", "valid_start", "valid_end",
                                          "selected_name", "selected_valid_rank_ic",
                                          "selected_best_iteration", "test_start", "test_end"]}
                  for f in folds]).to_csv(out / "folds.csv", index=False)
    write_json(out / "folds.json", folds)
    verify(out, folds, all_predictions)
    print(f"COMPLETE {len(folds)} quarters, {len(all_predictions)} stock-days", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validation-only", action="store_true",
                        help="Rebuild complete 126-day validation predictions without rerunning outer refits")
    args = parser.parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=3):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        main(args.panel, args.out, args.validation_only)
