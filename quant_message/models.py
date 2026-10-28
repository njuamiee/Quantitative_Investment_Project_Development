"""Serializable fits for all eight existing model families, with explicit maturity masks.

Only load locally produced/trusted pickle bundles; SHA256 is integrity, not trust.
Factor publication timestamps remain a data-provider responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path

import numpy as np

FAMILIES = ("ridge", "lightgbm", "ranker", "xgboost", "extra_trees", "catboost", "ic", "conditional_ic")


def estimator(name):
    common = dict(n_estimators=64, learning_rate=.04, num_leaves=15, max_depth=5,
                  min_child_samples=100, reg_lambda=12., colsample_bytree=.8,
                  verbosity=-1, n_jobs=1, random_state=20261026, deterministic=True)
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=180.)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**common)
    if name == "ranker":
        from lightgbm import LGBMRanker
        return LGBMRanker(**common, objective="lambdarank", label_gain=list(range(10)))
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=64, max_depth=3, learning_rate=.04, min_child_weight=100,
            reg_lambda=15, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=20261026, tree_method="hist")
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=48, max_depth=6, min_samples_leaf=100,
                                   max_features=.6, n_jobs=1, random_state=20261026)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=64, depth=4, learning_rate=.04, l2_leaf_reg=15,
                    verbose=False, thread_count=1, random_seed=20261026, allow_writing_files=False)
    raise ValueError(name)


@dataclass
class ModelBundle:
    family: str
    features: list[str]
    model: object
    audit: dict

    def predict(self, frame, at, probabilities=None):
        import pandas as pd
        at = pd.Timestamp(at)
        if at.tzinfo is None or at < pd.Timestamp(self.audit["trained_at"]):
            raise ValueError("model not yet trained")
        if "available_at" not in frame or (pd.to_datetime(frame.available_at, utc=True) > at).any():
            raise ValueError("feature release timestamps required and must already be available")
        if frame.available_at.isna().any() or not set(self.features).issubset(frame.columns):
            raise ValueError("missing features or release timestamps")
        x = frame[self.features].to_numpy(float)
        if np.isinf(x).any() or np.isfinite(x).mean() < .7:
            raise ValueError("feature quality below contract")
        x = np.nan_to_num(x)
        if self.family == "ic":
            return x @ self.model
        if self.family == "conditional_ic":
            p = np.asarray(probabilities, dtype=float)
            if p.shape != (len(x), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
                raise ValueError("available forward-filtered HMM probabilities required")
            return (x * (p @ self.model)).sum(axis=1)
        return self.model.predict(x)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        raw = pickle.dumps(self, protocol=5)
        (directory / "model.pkl").write_bytes(raw)
        card = dict(self.audit, model_sha256=hashlib.sha256(raw).hexdigest(), family=self.family,
                    features=self.features, python=platform.python_version(), platform=platform.system(),
                    packages={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")})
        for p in ("lightgbm", "xgboost", "catboost"):
            try: card["packages"][p] = importlib.metadata.version(p)
            except importlib.metadata.PackageNotFoundError: pass
        (directory / "manifest.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        return card

    @staticmethod
    def load_trusted(directory):
        directory = Path(directory)
        card = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        raw = (directory / "model.pkl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != card["model_sha256"]:
            raise ValueError("model digest mismatch")
        model = pickle.loads(raw)
        if not isinstance(model, ModelBundle) or model.features != card["features"]:
            raise ValueError("invalid bundle")
        return model


def fit(frame, feature_names, family, at, count=20, window_sessions=126, research_only=True):
    """Frame: date, available_at, label_available_at, target, feature columns.

    Cross-sectional rank targets must lie in [-.5,.5]. Training selection and
    factor screening see ONLY mature rows. Serving receives no target columns.
    """
    import pandas as pd
    from threadpoolctl import threadpool_limits
    at = pd.Timestamp(at)
    if family not in FAMILIES or at.tzinfo is None or count <= 0 or window_sessions <= 0:
        raise ValueError("invalid fit contract")
    required = {"date", "available_at", "label_available_at", "target", *feature_names}
    if not required.issubset(frame):
        raise ValueError("missing training contract columns")
    released = pd.to_datetime(frame.available_at, utc=True)
    mature = pd.to_datetime(frame.label_available_at, utc=True)
    f = frame[(released < at) & (mature < at) & np.isfinite(frame.target)].copy()
    days = sorted(f.date.unique())[-window_sessions:]
    f = f[f.date.isin(days)].sort_values("date", kind="stable")
    if len(f) < 100 or f.date.nunique() < 5:
        raise ValueError("insufficient mature training observations")
    y = f.target.to_numpy(float)
    if (np.abs(y) > .500001).any():
        raise ValueError("expected cross-sectional rank target [-.5,.5]")
    x = f[feature_names].to_numpy(float)
    if np.isinf(x).any():
        raise ValueError("infinite training feature")
    cover = np.isfinite(x).mean(axis=0)
    # Per-day IC prevents treating each stock as an independent day.
    ic = []
    for _, g in f.groupby("date", sort=True):
        a = g[feature_names].to_numpy(float)
        target = g.target.to_numpy(float)
        row = []
        for j in range(a.shape[1]):
            good = np.isfinite(a[:, j]) & np.isfinite(target)
            v = np.corrcoef(a[good, j], target[good])[0, 1] if good.sum() >= 3 and np.std(a[good, j]) > 1e-12 and np.std(target[good]) > 1e-12 else np.nan
            row.append(v)
        ic.append(row)
    a = np.asarray(ic)
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0)
    utility = np.where((cover >= .7) & np.isfinite(mu), abs(mu) / (sd + .05), -np.inf)
    chosen = np.argsort(-utility, kind="stable")[:min(count, np.isfinite(utility).sum())]
    if not len(chosen):
        raise ValueError("no eligible factors")
    names = [feature_names[i] for i in chosen]
    xx = np.nan_to_num(x[:, chosen])
    if family == "ic":
        model = np.nan_to_num(mu[chosen] / (sd[chosen] + .05))
        model /= np.abs(model).sum() + 1e-12
    elif family == "conditional_ic":
        required_hmm = [f"hmm_p{i}" for i in range(3)]
        p = f[required_hmm].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("invalid training HMM probabilities")
        dayp = f.groupby("date", sort=True)[required_hmm].first().to_numpy(float)
        weights = []
        for j in range(3):
            w = (np.nansum(a[:, chosen] * dayp[:, j, None], axis=0) + 60 * mu[chosen]) / (
                np.nansum(np.isfinite(a[:, chosen]) * dayp[:, j, None], axis=0) + 60)
            w = np.nan_to_num(w); w /= abs(w).sum() + 1e-12
            weights.append(w)
        model = np.asarray(weights)
    else:
        model = estimator(family)
        with threadpool_limits(1):
            if family == "ranker":
                model.fit(xx, np.minimum(9, np.floor((y + .5) * 10)).astype(int),
                          group=f.groupby("date", sort=True).size().to_numpy())
            else:
                model.fit(xx, y)
    return ModelBundle(family, names, model, dict(trained_at=at.isoformat(),
        labels_end=pd.to_datetime(f.label_available_at, utc=True).max().isoformat(),
        features_end=pd.to_datetime(f.available_at, utc=True).max().isoformat(),
        first_training_date=str(f.date.min()), last_training_date=str(f.date.max()),
        training_rows=len(f), training_sessions=f.date.nunique(), research_only=research_only,
        factor_screen="training-only daily Pearson IC; missing<=30%; explicit serving schema",
        historical_research_rules_identical=False))
