"""Build date-causal risk descriptors for the BJ research universe.

The share count is taken from the latest *earlier* daily supplier snapshot.
It is deliberately not backfilled from a later snapshot or treated as verified
point-in-time publication data. Industry is unknown until a dated source exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRICE_FILE = ROOT / "training_data/research_20261011/bj_prices.parquet"
CAPITAL_FILE = ROOT / "training_data/research_20260930/bj_share_capital.parquet"
OUTPUT = ROOT / "training_runs/research_20261017"
INDUSTRY_FILE = ROOT / "training_data/research_20261017/industry/industry_history.parquet"
MAX_CAPITAL_AGE_DAYS = 30


def _tertile_by_pool(frame: pd.DataFrame, value: str, names: tuple[str, str, str]) -> pd.Series:
    """Use same-day in-pool distribution; ties stay in the same bucket."""
    reference = frame.loc[frame["in_pool"] & frame[value].notna(), ["date", value]]
    if reference.empty:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    cutoffs = reference.groupby("date")[value].quantile([1 / 3, 2 / 3]).unstack()
    cutoffs.columns = ["q1", "q2"]
    levels = frame[["date", value]].join(cutoffs, on="date")
    out = np.select(
        [levels[value].le(levels.q1), levels[value].le(levels.q2), levels[value].gt(levels.q2)],
        names,
        default="",
    )
    return pd.Series(out, index=frame.index, dtype="string").replace("", pd.NA)


def build() -> tuple[pd.DataFrame, dict]:
    prices = pd.read_parquet(
        PRICE_FILE,
        columns=["date", "stockid", "close", "mom20", "vol20", "observed", "in_pool"],
    )
    capital = pd.read_parquet(
        CAPITAL_FILE,
        columns=["date", "stockid", "total_shares", "capital_ordering_warning", "snapshot_available_at_assumption"],
    ).rename(columns={"date": "capital_date"})
    prices["date"] = pd.to_datetime(prices.date).astype("datetime64[ns]")
    capital["capital_date"] = pd.to_datetime(capital.capital_date).astype("datetime64[ns]")
    capital["snapshot_available_at_assumption"] = pd.to_datetime(capital.snapshot_available_at_assumption).astype("datetime64[ns]")
    if prices.duplicated(["date", "stockid"]).any() or capital.duplicated(["capital_date", "stockid"]).any():
        raise ValueError("Duplicate price or capital date/stock keys")
    prices = prices.sort_values(["date", "stockid"]).reset_index(drop=True)
    capital = capital.sort_values(["capital_date", "stockid"]).reset_index(drop=True)
    # Exact-date matching is disabled: a snapshot marked for 18:00 cannot be
    # assumed to have been known when that day's 15:00 close became available.
    joined = pd.merge_asof(
        prices,
        capital,
        left_on="date",
        right_on="capital_date",
        by="stockid",
        direction="backward",
        allow_exact_matches=False,
    )
    joined["capital_age_days"] = (joined.date - joined.capital_date).dt.days
    valid_price = joined.observed.fillna(False) & np.isfinite(joined.close) & joined.close.gt(0)
    joined["mom20"] = joined.mom20.where(valid_price)
    joined["vol20"] = joined.vol20.where(valid_price)
    valid_capital = (
        joined.capital_date.notna()
        & joined.capital_age_days.le(MAX_CAPITAL_AGE_DAYS)
        & joined.total_shares.gt(0)
        & np.isfinite(joined.total_shares)
        & ~joined.capital_ordering_warning.fillna(False)
        & joined.snapshot_available_at_assumption.lt(joined.date)
    )
    joined["size_valid"] = valid_price & valid_capital
    joined["market_cap_yuan"] = (joined.close * joined.total_shares).where(joined.size_valid)
    joined["size_tertile"] = _tertile_by_pool(joined, "market_cap_yuan", ("small", "middle", "large"))
    joined["momentum_tertile"] = _tertile_by_pool(joined, "mom20", ("low", "middle", "high"))
    joined["volatility_tertile"] = _tertile_by_pool(joined, "vol20", ("low", "middle", "high"))
    if INDUSTRY_FILE.exists():
        industry = pd.read_parquet(INDUSTRY_FILE, columns=["stockid", "available_at", "publication_date", "industry", "source_period", "raw_official_code"])
        industry["available_at"] = pd.to_datetime(industry.available_at).astype("datetime64[ns]")
        industry["publication_date"] = pd.to_datetime(industry.publication_date).astype("datetime64[ns]")
        industry = industry.rename(columns={
            "available_at": "industry_available_at", "publication_date": "industry_publication_date",
            "source_period": "industry_source_period", "raw_official_code": "industry_raw_official_code",
        }).sort_values(["industry_available_at", "stockid"])
        joined = pd.merge_asof(joined.sort_values(["date", "stockid"]), industry,
            left_on="date", right_on="industry_available_at", by="stockid", direction="backward", allow_exact_matches=True)
        joined["industry"] = joined.industry.fillna("unknown")
        joined["industry_source_status"] = np.where(joined.industry.eq("unknown"), "no published classification yet", "CAPCO official dated classification")
    else:
        joined["industry"] = "unknown"
        joined["industry_source_status"] = "historical industry source absent"
        for name in ("industry_available_at", "industry_publication_date", "industry_source_period", "industry_raw_official_code"):
            joined[name] = pd.NA
    assert joined.loc[joined.capital_date.notna(), "capital_date"].lt(joined.loc[joined.capital_date.notna(), "date"]).all()
    assert joined.loc[joined.market_cap_yuan.notna(), "market_cap_yuan"].gt(0).all()
    known = joined.industry.ne("unknown")
    assert joined.loc[known, "industry_available_at"].le(joined.loc[known, "date"]).all()
    assert joined.loc[known, "industry_publication_date"].lt(joined.loc[known, "date"]).all()

    output_columns = [
        "date", "stockid", "in_pool", "observed", "close", "total_shares", "capital_date",
        "capital_age_days", "capital_ordering_warning", "snapshot_available_at_assumption",
        "size_valid", "market_cap_yuan", "size_tertile", "mom20", "momentum_tertile",
        "vol20", "volatility_tertile", "industry", "industry_source_status",
        "industry_available_at", "industry_publication_date", "industry_source_period", "industry_raw_official_code",
    ]
    out = joined[output_columns].copy()
    pool = out[out.in_pool]
    coverage = {
        "price_source": str(PRICE_FILE.relative_to(ROOT)),
        "capital_source": str(CAPITAL_FILE.relative_to(ROOT)),
        "rows": int(len(out)),
        "pool_rows": int(len(pool)),
        "dates": int(out.date.nunique()),
        "first_date": str(out.date.min().date()),
        "last_date": str(out.date.max().date()),
        "stocks": int(out.stockid.nunique()),
        "capital_max_age_calendar_days": MAX_CAPITAL_AGE_DAYS,
        "size_valid_rows": int(out.size_valid.sum()),
        "size_valid_pool_rows": int(pool.size_valid.sum()),
        "size_valid_pool_fraction": float(pool.size_valid.mean()),
        "capital_ordering_warning_pool_rows": int(pool.capital_ordering_warning.fillna(False).sum()),
        "momentum_group_pool_fraction": float(pool.momentum_tertile.notna().mean()),
        "volatility_group_pool_fraction": float(pool.volatility_tertile.notna().mean()),
        "industry_source": str(INDUSTRY_FILE.relative_to(ROOT)) if INDUSTRY_FILE.exists() else None,
        "industry_known_rows": int(out.industry.ne("unknown").sum()),
        "industry_known_pool_rows": int(pool.industry.ne("unknown").sum()),
        "industry_known_pool_fraction": float(pool.industry.ne("unknown").mean()),
        "industry_known_research_pool_fraction": float(pool.loc[pool.date.ge("2024-10-08"), "industry"].ne("unknown").mean()),
        "strict_point_in_time_capital_verified": False,
        "signal_timing": "Date d descriptors use d closing price and capital snapshot dated strictly before d; usable for d+1 trading only.",
        "limitations": [
            "Supplier snapshot historical release/revision times are unverified; 18:00 timestamp is an assumption.",
            "Total shares may lag corporate events. A 30-calendar-day age limit is imposed; no future backfill.",
            "Industry comes from five dated CAPCO PDFs, activated only the trading day after publication; uncovered stocks remain unknown.",
            "Historical listing/delisting and ST changes remain unavailable; industry classification alone does not solve those issues.",
            "Official 2025 old/new-code conversion is used only for company identity alignment; see source and parser checks.",
        ],
    }
    return out, coverage


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, coverage = build()
    frame.to_parquet(OUTPUT / "risk_metadata.parquet", index=False)
    research_pool = frame.loc[frame.in_pool & frame.date.ge("2024-10-08")].copy()
    research_pool["known"] = research_pool.industry.ne("unknown")
    by_day = research_pool.groupby("date").agg(pool_stocks=("stockid", "size"), known_industry=("known", "sum"))
    by_day["unknown_industry"] = by_day.pool_stocks - by_day.known_industry
    by_day["known_fraction"] = by_day.known_industry / by_day.pool_stocks
    by_day.to_csv(OUTPUT / "industry_coverage_daily.csv", encoding="utf-8-sig")
    by_quarter = research_pool.groupby(research_pool.date.dt.to_period("Q")).agg(pool_stock_days=("stockid", "size"), known_stock_days=("known", "sum"))
    by_quarter["known_fraction"] = by_quarter.known_stock_days / by_quarter.pool_stock_days
    by_quarter.to_csv(OUTPUT / "industry_coverage_by_quarter.csv", encoding="utf-8-sig")
    (OUTPUT / "risk_metadata_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(coverage, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
