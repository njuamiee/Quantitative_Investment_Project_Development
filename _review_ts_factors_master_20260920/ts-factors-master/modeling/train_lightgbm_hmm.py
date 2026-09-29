"""Train and evaluate the first cross-sectional LightGBM + HMM research model.

The experiment uses 2021-2023 for training, 2024 for validation/early stopping,
and 2025 as the untouched chronological test period.  Signals are generated
after close t, enter at the exact next market trading day's open, and are
labelled through the exact fifth market trading day's close.

Provider Alpha history in the current archive has no publication/revision
timestamps.  Outputs are therefore explicitly marked research-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from hmmlearn.hmm import GaussianHMM
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ExperimentConfig:
    factor_dir: str
    price_dir: str
    output_dir: str
    train_label_end: str = "2023-12-31"
    valid_start: str = "2024-01-08"
    valid_label_end: str = "2024-12-31"
    test_start: str = "2025-01-08"
    test_signal_end: str = "2025-12-31"
    minimum_listing_bars: int = 20
    minimum_factor_count: int = 70
    seed: int = 20260920
    initial_capital: float = 1_000_000.0
    daily_batch_fraction: float = 0.20
    max_stock_fraction: float = 0.10
    adv20_fraction: float = 0.01
    buy_fee_rate: float = 0.001
    sell_fee_rate: float = 0.001


PRICE_RENAME = {
    "日期": "date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="gb18030")
    frame = frame.rename(columns=PRICE_RENAME)
    required = {"date", "code", "open", "close", "high", "low", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    keep = [
        "date",
        "code",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "turnover",
    ]
    keep = [column for column in keep if column in frame.columns]
    frame = frame[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["code"] = frame["code"].astype(str).str.extract(r"(920\d{3})", expand=False)
    frame["stockid"] = "BJ" + frame["code"]
    numeric = [column for column in keep if column not in {"date", "code"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.drop(columns="code")


def load_prices(price_dir: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(price_dir.glob("920*.csv"))
    if not files:
        raise FileNotFoundError(f"No 920xxx CSV files in {price_dir}")
    frames = [read_price_csv(path) for path in files]
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["stockid", "date"]).reset_index(drop=True)
    duplicate_count = int(prices.duplicated(["stockid", "date"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate (stockid, date) price rows: {duplicate_count}")
    invalid_ohlc = (
        (prices[["open", "close", "high", "low"]] <= 0).any(axis=1)
        | prices[["open", "close", "high", "low"]].isna().any(axis=1)
    )

    grouped = prices.groupby("stockid", sort=False, group_keys=False)
    prices["ret1"] = grouped["close"].transform(lambda s: s / s.shift(1) - 1.0)
    for horizon in (5, 10, 20, 60):
        prices[f"mom{horizon}"] = grouped["close"].transform(
            lambda s, h=horizon: s / s.shift(h) - 1.0
        )
    for horizon, minimum in ((10, 6), (20, 10), (60, 30)):
        prices[f"vol{horizon}"] = grouped["ret1"].transform(
            lambda s, h=horizon, m=minimum: s.rolling(h, min_periods=m).std()
        )
    prices["amount_ma5"] = grouped["amount"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    prices["amount_ma20"] = grouped["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    prices["amount_ratio_5_20"] = prices["amount_ma5"] / prices["amount_ma20"]
    previous_close = grouped["close"].shift(1)
    prices["range1"] = (prices["high"] - prices["low"]) / previous_close
    prices["range20"] = grouped["range1"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    prices["illiq1"] = prices["ret1"].abs() / prices["amount"].clip(lower=1.0)
    prices["log_illiq20"] = np.log(
        grouped["illiq1"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        .clip(lower=1e-16)
    )
    if "turnover" in prices:
        prices["turnover20"] = grouped["turnover"].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
    else:
        prices["turnover20"] = np.nan
    prices["listing_bars"] = grouped.cumcount() + 1

    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    date_to_entry = {
        calendar[i]: calendar[i + 1] for i in range(max(0, len(calendar) - 1))
    }
    date_to_exit = {
        calendar[i]: calendar[i + 5] for i in range(max(0, len(calendar) - 5))
    }
    prices["entry_date"] = prices["date"].map(date_to_entry)
    prices["exit_date"] = prices["date"].map(date_to_exit)

    entry = prices[["stockid", "date", "open"]].rename(
        columns={"date": "entry_date", "open": "entry_open"}
    )
    exit_ = prices[["stockid", "date", "close"]].rename(
        columns={"date": "exit_date", "close": "exit_close"}
    )
    prices = prices.merge(entry, on=["stockid", "entry_date"], how="left")
    prices = prices.merge(exit_, on=["stockid", "exit_date"], how="left")
    prices["raw_return_5d"] = prices["exit_close"] / prices["entry_open"] - 1.0

    latest_counts = (
        prices.groupby("stockid")["date"].max().dt.strftime("%Y-%m-%d").value_counts()
    )
    audit = {
        "price_files": len(files),
        "price_rows": int(len(prices)),
        "price_stocks": int(prices["stockid"].nunique()),
        "price_min_date": str(prices["date"].min().date()),
        "price_max_date": str(prices["date"].max().date()),
        "duplicate_keys": duplicate_count,
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "latest_date_counts": {str(k): int(v) for k, v in latest_counts.items()},
    }
    return prices, audit


def load_factor_panel(factor_dir: Path) -> tuple[pd.DataFrame, list[str], dict]:
    files = sorted(factor_dir.glob("Alpha*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Alpha parquet files in {factor_dir}")

    first = pd.read_parquet(files[0], columns=["date", "stockid", "value"])
    first["date"] = pd.to_datetime(first["date"])
    first = first.sort_values(["date", "stockid"]).reset_index(drop=True)
    if first.duplicated(["date", "stockid"]).any():
        raise ValueError(f"Duplicate keys in {files[0].name}")
    index = pd.MultiIndex.from_frame(first[["date", "stockid"]])
    wide = pd.DataFrame(index=index)
    factor_columns: list[str] = []
    slow_reindex: list[str] = []

    base_dates = first["date"].to_numpy()
    base_stocks = first["stockid"].astype(str).to_numpy()
    for number, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=["date", "stockid", "value"])
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["date", "stockid"]).reset_index(drop=True)
        name = path.stem
        values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=np.float32)
        same_keys = (
            len(frame) == len(first)
            and np.array_equal(frame["date"].to_numpy(), base_dates)
            and np.array_equal(frame["stockid"].astype(str).to_numpy(), base_stocks)
        )
        if same_keys:
            wide[name] = values
        else:
            if frame.duplicated(["date", "stockid"]).any():
                raise ValueError(f"Duplicate keys in {path.name}")
            slow_reindex.append(name)
            series = frame.set_index(["date", "stockid"])["value"]
            wide[name] = pd.to_numeric(series.reindex(index), errors="coerce").astype(
                np.float32
            )
        factor_columns.append(name)
        if number % 20 == 0 or number == len(files):
            print(f"factors {number}/{len(files)}", flush=True)

    wide = wide.replace([np.inf, -np.inf], np.nan).reset_index()
    hashes = {path.name: file_sha256(path) for path in files}
    audit = {
        "factor_files": len(files),
        "factor_rows": int(len(wide)),
        "factor_stocks": int(wide["stockid"].nunique()),
        "factor_min_date": str(wide["date"].min().date()),
        "factor_max_date": str(wide["date"].max().date()),
        "reindexed_factor_files": slow_reindex,
        "factor_file_sha256": hashes,
        "availability_warning": (
            "Provider history has date/stockid/value but no publication or revision timestamp; "
            "results are research-only."
        ),
    }
    return wide, factor_columns, audit


def build_dataset(
    factors: pd.DataFrame,
    factor_columns: list[str],
    prices: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, list[str], dict]:
    price_features = [
        "date",
        "stockid",
        "open",
        "close",
        "amount",
        "ret1",
        "mom5",
        "mom10",
        "mom20",
        "mom60",
        "vol10",
        "vol20",
        "vol60",
        "amount_ma20",
        "amount_ratio_5_20",
        "range20",
        "log_illiq20",
        "turnover20",
        "listing_bars",
        "entry_date",
        "exit_date",
        "entry_open",
        "exit_close",
        "raw_return_5d",
    ]
    panel = factors.merge(prices[price_features], on=["date", "stockid"], how="left")
    panel["factor_count"] = panel[factor_columns].notna().sum(axis=1).astype(np.int16)
    panel["eligible_now"] = (
        (panel["listing_bars"] >= config.minimum_listing_bars)
        & (panel["factor_count"] >= config.minimum_factor_count)
        & panel["amount_ma20"].notna()
        & (panel["amount_ma20"] > 0)
        & panel["open"].notna()
        & panel["close"].notna()
    )

    technical = [
        "mom5",
        "mom10",
        "mom20",
        "mom60",
        "vol10",
        "vol20",
        "vol60",
        "amount_ma20",
        "amount_ratio_5_20",
        "range20",
        "log_illiq20",
        "turnover20",
        "listing_bars",
    ]
    rank_inputs = factor_columns + technical
    print("cross-sectional ranking", flush=True)
    ranked = panel.groupby("date", sort=False)[rank_inputs].rank(
        pct=True, method="average", na_option="keep"
    )
    ranked = (ranked - 0.5).astype(np.float32)
    feature_columns = [f"x_{column}" for column in rank_inputs]
    ranked.columns = feature_columns
    panel = pd.concat([panel, ranked], axis=1)
    panel["x_missing_fraction"] = (
        1.0 - panel[rank_inputs].notna().mean(axis=1)
    ).astype(np.float32)
    feature_columns.append("x_missing_fraction")

    eval_mask = panel["eligible_now"] & panel["raw_return_5d"].notna()
    median_return = panel.loc[eval_mask].groupby("date")["raw_return_5d"].median()
    panel["label_relative_5d"] = panel["raw_return_5d"] - panel["date"].map(
        median_return
    )
    panel.loc[~eval_mask, "label_relative_5d"] = np.nan
    panel["label_rank_5d"] = np.nan
    panel.loc[eval_mask, "label_rank_5d"] = (
        panel.loc[eval_mask]
        .groupby("date")["raw_return_5d"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .astype(np.float32)
    )

    audit = {
        "joined_rows": int(len(panel)),
        "joined_stocks": int(panel["stockid"].nunique()),
        "eligible_now_rows": int(panel["eligible_now"].sum()),
        "evaluable_label_rows": int(panel["label_relative_5d"].notna().sum()),
        "missing_exact_entry_rows": int(
            (panel["eligible_now"] & panel["entry_open"].isna()).sum()
        ),
        "missing_exact_exit_rows": int(
            (panel["eligible_now"] & panel["exit_close"].isna()).sum()
        ),
        "feature_count": len(feature_columns),
    }
    return panel, feature_columns, audit


def date_balanced_weights(dates: pd.Series) -> np.ndarray:
    counts = dates.value_counts()
    weights = dates.map(1.0 / counts)
    return (weights / weights.mean()).to_numpy(dtype=np.float32)


def _daily_metric_rows(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows: list[dict] = []
    for date, group in frame.groupby("date", sort=True):
        group = group.dropna(subset=[score_column, "label_relative_5d", "raw_return_5d"])
        if len(group) < 10:
            continue
        score = group[score_column]
        label = group["label_relative_5d"]
        ic = spearmanr(score, label).statistic if score.nunique() > 1 else np.nan
        ordered = group.sort_values(score_column, ascending=False)
        n_tail = max(1, int(math.ceil(len(ordered) * 0.10)))
        top5 = ordered.head(min(5, len(ordered)))
        rows.append(
            {
                "date": date,
                "n_stocks": len(group),
                "rank_ic": ic,
                "directional_accuracy": float(
                    (
                        (score > score.median())
                        == (group["label_relative_5d"] > 0)
                    ).mean()
                ),
                "top5_positive_fraction": float(
                    (top5["label_relative_5d"] > 0).mean()
                ),
                "top5_raw_return": float(top5["raw_return_5d"].mean()),
                "universe_raw_return": float(group["raw_return_5d"].mean()),
                "top_bottom_10pct_spread": float(
                    ordered.head(n_tail)["raw_return_5d"].mean()
                    - ordered.tail(n_tail)["raw_return_5d"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def moving_block_mean_ci(
    values: np.ndarray,
    block_length: int = 10,
    repetitions: int = 5000,
    seed: int = 20260920,
) -> list[float]:
    """Moving-block bootstrap CI, retaining short-horizon serial dependence."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return [np.nan, np.nan]
    block_length = min(block_length, n)
    n_blocks = int(math.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        sample = np.concatenate(
            [values[start : start + block_length] for start in starts]
        )[:n]
        estimates[repetition] = sample.mean()
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def evaluate_predictions(frame: pd.DataFrame, score_column: str) -> tuple[dict, pd.DataFrame]:
    valid = frame.dropna(subset=[score_column, "label_relative_5d", "raw_return_5d"]).copy()
    valid["score_centered"] = valid[score_column] - valid.groupby("date")[
        score_column
    ].transform("median")
    actual_positive = valid["label_relative_5d"] > 0
    predicted_positive = valid["score_centered"] > 0
    correct = actual_positive == predicted_positive
    daily = _daily_metric_rows(valid, score_column)
    n = len(valid)
    accuracy = float(daily["directional_accuracy"].mean())
    accuracy_ci = moving_block_mean_ci(daily["directional_accuracy"].to_numpy())
    rank_ic_std = float(daily["rank_ic"].std()) if len(daily) else np.nan
    metrics = {
        "rows": int(n),
        "dates": int(valid["date"].nunique()),
        "stocks": int(valid["stockid"].nunique()),
        "directional_accuracy": accuracy,
        "directional_accuracy_95ci": accuracy_ci,
        "directional_accuracy_ci_method": "10-day moving-block bootstrap, 5000 repetitions",
        "balanced_reference_accuracy": 0.5,
        "mean_daily_rank_ic": float(daily["rank_ic"].mean()),
        "daily_rank_ic_std": rank_ic_std,
        "daily_icir": (
            float(daily["rank_ic"].mean() / rank_ic_std)
            if rank_ic_std > 0
            else np.nan
        ),
        "positive_rank_ic_day_fraction": float((daily["rank_ic"] > 0).mean()),
        "top5_positive_hit_rate": float(daily["top5_positive_fraction"].mean()),
        "top5_daily_outperform_fraction": float(
            (daily["top5_raw_return"] > daily["universe_raw_return"]).mean()
        ),
        "mean_top5_5d_raw_return": float(daily["top5_raw_return"].mean()),
        "mean_universe_5d_raw_return": float(daily["universe_raw_return"].mean()),
        "mean_top_bottom_10pct_5d_spread": float(
            daily["top_bottom_10pct_spread"].mean()
        ),
    }
    return metrics, daily


def fit_lightgbm(
    panel: pd.DataFrame,
    feature_columns: list[str],
    config: ExperimentConfig,
    output_dir: Path,
    model_prefix: str,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, dict, pd.DataFrame]:
    label = "label_rank_5d"
    train = panel[
        panel["eligible_now"]
        & panel[label].notna()
        & (panel["exit_date"] <= pd.Timestamp(config.train_label_end))
    ].copy()
    valid = panel[
        panel["eligible_now"]
        & panel[label].notna()
        & (panel["date"] >= pd.Timestamp(config.valid_start))
        & (panel["exit_date"] <= pd.Timestamp(config.valid_label_end))
    ].copy()
    test_all = panel[
        panel["eligible_now"]
        & (panel["date"] >= pd.Timestamp(config.test_start))
        & (panel["date"] <= pd.Timestamp(config.test_signal_end))
    ].copy()
    if min(len(train), len(valid), len(test_all)) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train)}, valid={len(valid)}, test={len(test_all)}"
        )

    params = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "n_estimators": 500,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "random_state": config.seed,
        "n_jobs": 8,
        "verbosity": -1,
    }
    development_model = lgb.LGBMRegressor(**params)
    development_model.fit(
        train[feature_columns],
        train[label],
        sample_weight=date_balanced_weights(train["date"]),
        eval_set=[(valid[feature_columns], valid[label])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )
    best_iteration = int(development_model.best_iteration_ or params["n_estimators"])
    valid["prediction"] = development_model.predict(
        valid[feature_columns], num_iteration=best_iteration
    )
    validation_metrics, _ = evaluate_predictions(valid, "prediction")

    refit = pd.concat([train, valid], ignore_index=True)
    final_params = dict(params)
    final_params["n_estimators"] = best_iteration
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(
        refit[feature_columns],
        refit[label],
        sample_weight=date_balanced_weights(refit["date"]),
    )
    test_all["prediction"] = final_model.predict(test_all[feature_columns])
    evaluable = test_all[test_all[label].notna()].copy()
    test_metrics, daily = evaluate_predictions(evaluable, "prediction")

    # LightGBM's Windows C API cannot write paths containing Chinese characters.
    # Let Python perform the UTF-8 file write instead.
    (output_dir / f"{model_prefix}_model.txt").write_text(
        final_model.booster_.model_to_string(), encoding="utf-8"
    )
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": final_model.booster_.feature_importance(importance_type="gain"),
            "split": final_model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(output_dir / f"{model_prefix}_feature_importance.csv", index=False)

    validation_momentum_metrics, _ = evaluate_predictions(valid, "x_mom20")
    test_momentum_metrics, _ = evaluate_predictions(evaluable, "x_mom20")

    split_summary = {
        "train_rows": len(train),
        "train_dates": int(train["date"].nunique()),
        "train_min_signal": str(train["date"].min().date()),
        "train_max_signal": str(train["date"].max().date()),
        "train_max_label_date": str(train["exit_date"].max().date()),
        "validation_rows": len(valid),
        "validation_dates": int(valid["date"].nunique()),
        "validation_min_signal": str(valid["date"].min().date()),
        "validation_max_signal": str(valid["date"].max().date()),
        "validation_max_label_date": str(valid["exit_date"].max().date()),
        "test_scored_rows": len(test_all),
        "test_evaluable_rows": len(evaluable),
        "test_dates": int(test_all["date"].nunique()),
        "test_min_signal": str(test_all["date"].min().date()),
        "test_max_signal": str(test_all["date"].max().date()),
        "best_iteration": best_iteration,
        "model_prefix": model_prefix,
        "feature_count": len(feature_columns),
        "params": final_params,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_momentum20_baseline": validation_momentum_metrics,
        "test_momentum20_baseline": test_momentum_metrics,
    }
    return final_model, test_all, split_summary, daily


def build_market_observations(panel: pd.DataFrame) -> pd.DataFrame:
    eligible = panel[
        panel["eligible_now"]
        & panel["ret1"].notna()
        & panel["amount"].notna()
        & (panel["date"] <= pd.Timestamp("2025-12-31"))
    ].copy()
    eligible["advance_sign"] = np.sign(eligible["ret1"])
    daily = eligible.groupby("date").agg(
        market_return=("ret1", "mean"),
        breadth=("advance_sign", "mean"),
        dispersion=("ret1", "std"),
        median_amount=("amount", "median"),
        n_stocks=("stockid", "nunique"),
    )
    daily["log_vol20"] = np.log(
        daily["market_return"].rolling(20, min_periods=15).std().clip(lower=1e-6)
    )
    daily["log_median_amount"] = np.log(daily["median_amount"].clip(lower=1.0))
    columns = ["market_return", "log_vol20", "breadth", "dispersion"]
    return daily.dropna(subset=columns).sort_index()


def _last_filtered_probability(model: GaussianHMM, observations: np.ndarray) -> np.ndarray:
    log_likelihood = model._compute_log_likelihood(observations)
    emission = np.exp(log_likelihood[0] - log_likelihood[0].max())
    alpha = model.startprob_ * emission
    alpha = alpha / alpha.sum()
    for row in log_likelihood[1:]:
        emission = np.exp(row - row.max())
        alpha = (alpha @ model.transmat_) * emission
        alpha = alpha / alpha.sum()
    return alpha


def fit_hmm_walk_forward(
    market: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict]:
    observation_columns = ["market_return", "log_vol20", "breadth", "dispersion"]
    test_days = market.loc[
        (market.index >= pd.Timestamp(config.test_start))
        & (market.index <= pd.Timestamp(config.test_signal_end))
    ]
    rows: list[dict] = []
    fit_rows: list[dict] = []

    for month, month_frame in test_days.groupby(test_days.index.to_period("M")):
        month_start = month_frame.index.min()
        training = market.loc[market.index < month_start, observation_columns]
        if len(training) < 500:
            continue
        scaler = StandardScaler().fit(training)
        train_x = scaler.transform(training)
        candidates: list[tuple[float, GaussianHMM, int]] = []
        for offset in range(3):
            seed = config.seed + offset
            model = GaussianHMM(
                n_components=3,
                covariance_type="diag",
                n_iter=300,
                tol=1e-4,
                min_covar=1e-3,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_x)
            candidates.append((float(model.score(train_x)), model, seed))
        score, model, chosen_seed = max(candidates, key=lambda item: item[0])
        alpha = _last_filtered_probability(model, train_x)
        state_order = np.argsort(model.means_[:, 1])
        caps = np.array([1.0, 0.6, 0.2], dtype=float)
        expected_duration = 1.0 / np.clip(1.0 - np.diag(model.transmat_), 1e-8, None)

        for date, row in month_frame.iterrows():
            x = scaler.transform(row[observation_columns].to_frame().T)
            log_emission = model._compute_log_likelihood(x)[0]
            emission = np.exp(log_emission - log_emission.max())
            alpha = (alpha @ model.transmat_) * emission
            alpha = alpha / alpha.sum()
            next_probability = alpha @ model.transmat_
            ordered = next_probability[state_order]
            rows.append(
                {
                    "date": date,
                    "p_low_vol_next": ordered[0],
                    "p_high_vol_next": ordered[1],
                    "p_crisis_next": ordered[2],
                    "exposure_cap": float(ordered @ caps),
                    "most_likely_state": ["low_vol", "high_vol", "crisis"][
                        int(np.argmax(ordered))
                    ],
                    "fit_month": str(month),
                }
            )
        fit_rows.append(
            {
                "fit_month": str(month),
                "train_rows": len(training),
                "train_end": str(training.index.max().date()),
                "log_likelihood": score,
                "seed": chosen_seed,
                "converged": bool(model.monitor_.converged),
                "state_order_original": state_order.tolist(),
                "expected_duration_ordered": expected_duration[state_order].tolist(),
                "transition_matrix": model.transmat_.tolist(),
                "means": model.means_.tolist(),
            }
        )
    probabilities = pd.DataFrame(rows)
    if probabilities.empty:
        raise RuntimeError("HMM produced no test probabilities")
    summary = {
        "observation_columns": observation_columns,
        "fits": fit_rows,
        "probability_rows": len(probabilities),
        "mean_exposure_cap": float(probabilities["exposure_cap"].mean()),
        "state_day_counts": {
            str(k): int(v)
            for k, v in probabilities["most_likely_state"].value_counts().items()
        },
        "unstable_fit_count": int(
            sum(
                min(item["expected_duration_ordered"]) < 3.0
                for item in fit_rows
            )
        ),
    }
    return probabilities, summary


def _portfolio_metrics(daily: pd.DataFrame, initial_capital: float) -> dict:
    equity = daily["equity"]
    returns = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    years = max((daily["date"].iloc[-1] - daily["date"].iloc[0]).days / 365.25, 1 / 252)
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    std = float(returns.std())
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total_return,
        "annual_return": float(annual_return),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_gross_exposure": float(daily["gross_exposure"].mean()),
        "max_gross_exposure": float(daily["gross_exposure"].max()),
    }


def simulate_portfolio(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    exposure: pd.DataFrame,
    config: ExperimentConfig,
    use_hmm: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exposure_map = exposure.set_index("date")["exposure_cap"].to_dict()
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    calendar_position = {date: i for i, date in enumerate(calendar)}
    signal_groups = {
        date: group.sort_values("prediction", ascending=False).head(5)
        for date, group in scored.groupby("date", sort=True)
    }
    entries: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
    for signal_date, candidates in signal_groups.items():
        position = calendar_position.get(signal_date)
        if position is not None and position + 1 < len(calendar):
            entries[calendar[position + 1]] = (signal_date, candidates)

    price_lookup = prices.set_index(["date", "stockid"])[
        ["open", "close", "amount_ma20"]
    ].sort_index()
    start_date = min(entries)
    last_signal = max(signal_groups)
    last_position = calendar_position[last_signal]
    end_position = min(len(calendar) - 1, last_position + 15)
    simulation_days = calendar[
        (calendar >= start_date) & (calendar <= calendar[end_position])
    ]

    cash = float(config.initial_capital)
    positions: list[dict] = []
    last_close: dict[str, float] = {}
    trades: list[dict] = []
    daily_rows: list[dict] = []
    missed_entries = 0
    rejected_small = 0

    for date in simulation_days:
        # New orders execute at the open; proceeds from today's closes are not yet available.
        if date in entries:
            signal_date, candidates = entries[date]
            marked_value = sum(
                position["shares"]
                * float(
                    price_lookup.loc[(date, position["stockid"]), "open"]
                    if (date, position["stockid"]) in price_lookup.index
                    else last_close.get(position["stockid"], position["entry_price"])
                )
                for position in positions
            )
            equity_open = cash + marked_value
            state_cap = float(exposure_map.get(signal_date, 1.0)) if use_hmm else 1.0
            gross_limit = state_cap * equity_open
            available_gross = max(0.0, gross_limit - marked_value)
            batch_budget = min(
                config.initial_capital * config.daily_batch_fraction,
                cash,
                available_gross,
            )
            remaining_candidates = len(candidates)
            for row in candidates.itertuples(index=False):
                if remaining_candidates <= 0 or batch_budget <= 0:
                    break
                key = (date, row.stockid)
                if key not in price_lookup.index:
                    missed_entries += 1
                    remaining_candidates -= 1
                    continue
                open_price = float(price_lookup.loc[key, "open"])
                if not np.isfinite(open_price) or open_price <= 0:
                    missed_entries += 1
                    remaining_candidates -= 1
                    continue
                existing_value = sum(
                    p["shares"] * open_price
                    for p in positions
                    if p["stockid"] == row.stockid
                )
                stock_room = max(
                    0.0, config.max_stock_fraction * equity_open - existing_value
                )
                capacity = (
                    config.adv20_fraction * float(row.amount_ma20)
                    if np.isfinite(row.amount_ma20)
                    else 0.0
                )
                target = min(
                    batch_budget / remaining_candidates,
                    stock_room,
                    capacity,
                    cash / (1.0 + config.buy_fee_rate),
                )
                shares = int(math.floor(target / open_price))
                if shares < 100:
                    rejected_small += 1
                    remaining_candidates -= 1
                    continue
                notional = shares * open_price
                cost = notional * (1.0 + config.buy_fee_rate)
                cash -= cost
                batch_budget -= cost
                target_position = calendar_position[signal_date] + 5
                positions.append(
                    {
                        "stockid": row.stockid,
                        "signal_date": signal_date,
                        "entry_date": date,
                        "target_exit_date": calendar[target_position],
                        "entry_price": open_price,
                        "shares": shares,
                        "entry_cost": cost,
                        "hmm_cap": state_cap,
                    }
                )
                remaining_candidates -= 1

        # Update marks and close positions at or after their target date when a close exists.
        for position in list(positions):
            key = (date, position["stockid"])
            if key in price_lookup.index:
                close_price = float(price_lookup.loc[key, "close"])
                if np.isfinite(close_price) and close_price > 0:
                    last_close[position["stockid"]] = close_price
                    if date >= position["target_exit_date"]:
                        proceeds = position["shares"] * close_price * (
                            1.0 - config.sell_fee_rate
                        )
                        cash += proceeds
                        trades.append(
                            {
                                **position,
                                "exit_date": date,
                                "exit_price": close_price,
                                "proceeds": proceeds,
                                "net_return": proceeds / position["entry_cost"] - 1.0,
                            }
                        )
                        positions.remove(position)

        market_value = sum(
            position["shares"]
            * last_close.get(position["stockid"], position["entry_price"])
            for position in positions
        )
        equity = cash + market_value
        daily_rows.append(
            {
                "date": date,
                "cash": cash,
                "market_value": market_value,
                "equity": equity,
                "gross_exposure": market_value / equity if equity > 0 else np.nan,
                "n_positions": len(positions),
            }
        )

    daily = pd.DataFrame(daily_rows)
    trade_frame = pd.DataFrame(trades)
    metrics = _portfolio_metrics(daily, config.initial_capital)
    metrics.update(
        {
            "completed_trades": len(trade_frame),
            "open_positions_at_end": len(positions),
            "missed_entries": missed_entries,
            "rejected_below_100_shares": rejected_small,
            "fee_rate_each_side": config.buy_fee_rate,
            "uses_hmm_exposure": use_hmm,
            "industry_cap_missing": True,
        }
    )
    return daily, trade_frame, metrics


def package_versions() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lgb.__version__,
    }


def write_report(
    output_dir: Path,
    audit: dict,
    price_split: dict,
    alpha_split: dict,
    hmm_summary: dict,
    fixed_metrics: dict,
    hmm_metrics: dict,
) -> None:
    price_test = price_split["test_metrics"]
    price_valid = price_split["validation_metrics"]
    alpha_test = alpha_split["test_metrics"]
    alpha_valid = alpha_split["validation_metrics"]
    momentum_test = price_split["test_momentum20_baseline"]
    price_features = pd.read_csv(
        output_dir / "price_only_feature_importance.csv"
    ).head(10)
    alpha_features = pd.read_csv(
        output_dir / "alpha_research_feature_importance.csv"
    ).head(10)
    feature_lines = "\n".join(
        ["价格模型："]
        + [
            f"- `{row.feature}`：gain={row.gain:.1f}"
            for row in price_features.itertuples(index=False)
        ]
        + ["", "加入供应商 Alpha 的研究模型："]
        + [
            f"- `{row.feature}`：gain={row.gain:.1f}"
            for row in alpha_features.itertuples(index=False)
        ]
    )
    drawdown_reduction = 1.0 - abs(hmm_metrics["max_drawdown"]) / abs(
        fixed_metrics["max_drawdown"]
    )
    return_retention = (
        hmm_metrics["total_return"] / fixed_metrics["total_return"]
        if fixed_metrics["total_return"] > 0
        else np.nan
    )
    sharpe_change = hmm_metrics["sharpe"] - fixed_metrics["sharpe"]
    hmm_pass = bool(
        hmm_summary["unstable_fit_count"] == 0
        and (
            (drawdown_reduction >= 0.10 and return_retention >= 0.80)
            or sharpe_change >= 0.10
        )
    )
    text = f"""# LightGBM + HMM 首轮研究训练报告

生成时间：{datetime.now(timezone.utc).isoformat()}  
状态：**研究性结果，不是实盘预测或收益承诺**。供应商 Alpha 没有历史发布/修订时间戳，行业和公司行为数据也未找到。

## 样本切分

- 训练：{price_split['train_min_signal']} 至 {price_split['train_max_signal']}，{price_split['train_rows']:,} 行；所有标签结束日不晚于 {price_split['train_max_label_date']}。
- 验证：{price_split['validation_min_signal']} 至 {price_split['validation_max_signal']}，{price_split['validation_rows']:,} 行；只用于早停。
- 最终测试：{price_split['test_min_signal']} 至 {price_split['test_max_signal']}，评分 {price_split['test_scored_rows']:,} 行，可评价 {price_split['test_evaluable_rows']:,} 行，测试结果未用于调参。

## 预测准确率

这里的“方向”指个股未来 5 日收益是否高于当日可评价股票的中位数。模型分数也先按当日中位数居中，因此随机/无信息参照约为 50%。

| 模型/指标 | 2024 验证集 | 2025 最终测试集 |
|---|---:|---:|
| **价格特征模型方向命中率** | {price_valid['directional_accuracy']:.2%} | **{price_test['directional_accuracy']:.2%}** |
| 价格模型最终测试 95% 区间 |  | {price_test['directional_accuracy_95ci'][0]:.2%}–{price_test['directional_accuracy_95ci'][1]:.2%} |
| 价格模型每日平均 Rank IC | {price_valid['mean_daily_rank_ic']:.4f} | **{price_test['mean_daily_rank_ic']:.4f}** |
| 价格模型预测 Top5 命中率 | {price_valid['top5_positive_hit_rate']:.2%} | **{price_test['top5_positive_hit_rate']:.2%}** |
| **价格 + 101 Alpha 研究模型方向命中率** | {alpha_valid['directional_accuracy']:.2%} | **{alpha_test['directional_accuracy']:.2%}** |
| Alpha 研究模型每日平均 Rank IC | {alpha_valid['mean_daily_rank_ic']:.4f} | **{alpha_test['mean_daily_rank_ic']:.4f}** |
| Alpha 研究模型预测 Top5 命中率 | {alpha_valid['top5_positive_hit_rate']:.2%} | **{alpha_test['top5_positive_hit_rate']:.2%}** |
| 单独使用 20 日动量方向命中率 | {price_split['validation_momentum20_baseline']['directional_accuracy']:.2%} | {momentum_test['directional_accuracy']:.2%} |

## 组合模拟

模拟使用 100 万元固定本金、每日最多投入初始本金 20%、Top5、单股 10%、不超过 ADV20 的 1%、整数股且至少 100 股、买卖各 10bp 成本。缺少历史行业表，因此本轮没有实施行业 25% 上限。

| 指标 | LightGBM 固定仓位 | LightGBM + HMM |
|---|---:|---:|
| 总收益 | {fixed_metrics['total_return']:.2%} | {hmm_metrics['total_return']:.2%} |
| 年化收益 | {fixed_metrics['annual_return']:.2%} | {hmm_metrics['annual_return']:.2%} |
| Sharpe | {fixed_metrics['sharpe']:.3f} | {hmm_metrics['sharpe']:.3f} |
| 最大回撤 | {fixed_metrics['max_drawdown']:.2%} | {hmm_metrics['max_drawdown']:.2%} |
| 完成交易 | {fixed_metrics['completed_trades']} | {hmm_metrics['completed_trades']} |
| 平均资金占用 | {fixed_metrics['mean_gross_exposure']:.2%} | {hmm_metrics['mean_gross_exposure']:.2%} |

HMM 使用 3 个高斯隐状态、4 个市场级观测变量、每月扩展窗口重训和一步预测概率。测试期平均仓位上限为 {hmm_summary['mean_exposure_cap']:.2%}。只使用前向滤波，不使用全样本平滑概率。

HMM 相对固定仓位的回撤降幅为 {drawdown_reduction:.2%}、收益保留为 {return_retention:.2%}、Sharpe 变化为 {sharpe_change:+.3f}；12 次月度拟合中有 {hmm_summary['unstable_fit_count']} 次包含期望持续期低于 3 日的状态。按预先写入方案的验收标准，本轮 HMM **{'通过' if hmm_pass else '未通过'}**，{'可以进入下一阶段复核' if hmm_pass else '不得作为正式仓位层部署'}。

## 重要特征

{feature_lines}

## 数据审计与限制

- 行情：{audit['prices']['price_files']} 个北交所文件，{audit['prices']['price_rows']:,} 行，{audit['prices']['price_min_date']} 至 {audit['prices']['price_max_date']}。
- 因子：{audit['factors']['factor_files']} 个 Alpha，{audit['factors']['factor_rows']:,} 行，{audit['factors']['factor_min_date']} 至 {audit['factors']['factor_max_date']}。
- 模型标签采用未复权开收盘价，尚未加入可审计公司行为现金流；分红除权可能污染少数 5 日标签。
- 因子只有 `date/stockid/value`，无法证明历史值在当时已经发布，也无法识别事后修订。
- 股票池使用当日因子实际出现且满足 20 根历史 K 线的股票，优于静态 Top20，但不是正式上市/退市主数据。
- 准确率只在固定次日开盘和第五日收盘价格都存在的样本上计算；组合中次日无开盘价会记为未成交。
- HMM 状态没有客观“真标签”，因此不报告虚构的 HMM 分类准确率，只报告其样本外风险覆盖效果。
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def run(config: ExperimentConfig) -> Path:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("loading prices", flush=True)
    prices, price_audit = load_prices(Path(config.price_dir))
    print("loading factors", flush=True)
    factors, factor_columns, factor_audit = load_factor_panel(Path(config.factor_dir))
    print("building cross-sectional dataset", flush=True)
    panel, feature_columns, dataset_audit = build_dataset(
        factors, factor_columns, prices, config
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "prices": price_audit,
        "factors": factor_audit,
        "dataset": dataset_audit,
        "versions": package_versions(),
    }
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("training LightGBM", flush=True)
    price_only_features = [
        column
        for column in feature_columns
        if not column.startswith("x_Alpha") and column != "x_missing_fraction"
    ]
    _, scored, price_split, price_daily_metrics = fit_lightgbm(
        panel,
        price_only_features,
        config,
        output_dir,
        model_prefix="price_only",
    )
    _, alpha_scored, alpha_split, alpha_daily_metrics = fit_lightgbm(
        panel,
        feature_columns,
        config,
        output_dir,
        model_prefix="alpha_research",
    )
    price_daily_metrics.to_csv(
        output_dir / "test_daily_metrics_price_only.csv", index=False
    )
    alpha_daily_metrics.to_csv(
        output_dir / "test_daily_metrics_alpha_research.csv", index=False
    )
    scored = scored.merge(
        alpha_scored[["date", "stockid", "prediction"]].rename(
            columns={"prediction": "prediction_alpha_research"}
        ),
        on=["date", "stockid"],
        how="left",
        validate="one_to_one",
    )
    prediction_columns = [
        "date",
        "stockid",
        "prediction",
        "prediction_alpha_research",
        "eligible_now",
        "label_relative_5d",
        "raw_return_5d",
        "entry_date",
        "exit_date",
        "entry_open",
        "exit_close",
        "amount_ma20",
    ]
    scored[prediction_columns].to_parquet(
        output_dir / "test_predictions.parquet", index=False
    )

    print("training walk-forward HMM", flush=True)
    market = build_market_observations(panel)
    hmm_probability, hmm_summary = fit_hmm_walk_forward(market, config)
    hmm_probability.to_csv(output_dir / "hmm_probabilities.csv", index=False)

    print("simulating portfolios", flush=True)
    fixed_daily, fixed_trades, fixed_metrics = simulate_portfolio(
        scored, prices, hmm_probability, config, use_hmm=False
    )
    hmm_daily, hmm_trades, hmm_metrics = simulate_portfolio(
        scored, prices, hmm_probability, config, use_hmm=True
    )
    fixed_daily.to_csv(output_dir / "portfolio_fixed_daily.csv", index=False)
    fixed_trades.to_csv(output_dir / "portfolio_fixed_trades.csv", index=False)
    hmm_daily.to_csv(output_dir / "portfolio_hmm_daily.csv", index=False)
    hmm_trades.to_csv(output_dir / "portfolio_hmm_trades.csv", index=False)

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "price_only": price_split,
        "alpha_research": alpha_split,
        "hmm": hmm_summary,
        "portfolio_fixed": fixed_metrics,
        "portfolio_hmm": hmm_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir,
        audit,
        price_split,
        alpha_split,
        hmm_summary,
        fixed_metrics,
        hmm_metrics,
    )
    print(output_dir / "report.md", flush=True)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--price-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(
        ExperimentConfig(
            factor_dir=args.factor_dir,
            price_dir=args.price_dir,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
