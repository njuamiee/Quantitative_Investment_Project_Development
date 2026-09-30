"""统计工具：Spearman 秩相关、OLS 回归与 Newey-West HAC 标准误。

Newey-West HAC 实现策略：
  - 优先使用 statsmodels（OLS.fit(cov_type='HAC', cov_kwds={'maxlags': L})）；
  - 无法导入 statsmodels 时使用本模块的自实现（Bartlett 核，含 T/(T-p) 自由度修正，
    使得 L=0 时在同方差情形下退化为普通 OLS 标准误）。
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats as _stats

try:  # statsmodels 为可选依赖
    from statsmodels.regression.linear_model import OLS as _SM_OLS

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - 取决于运行环境
    _HAS_STATSMODELS = False


def has_statsmodels() -> bool:
    """返回 statsmodels 是否可用。"""
    return _HAS_STATSMODELS


def spearman_ic(s: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """Spearman 秩相关及 p 值；常数序列返回 (nan, nan) 并告警。"""
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    if len(s) < 3:
        return np.nan, np.nan
    if np.unique(s).size < 2 or np.unique(r).size < 2:
        warnings.warn("常数序列，秩相关未定义，返回 NaN", stacklevel=2)
        return np.nan, np.nan
    rho, p = _stats.spearmanr(s, r)
    return float(rho), float(p)


# ----------------------------------------------------------------------
# 自实现的 OLS + Newey-West HAC
# ----------------------------------------------------------------------
def _ols_self(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """普通最小二乘：返回 (系数, 残差, (X'X)^{-1}, R²)。"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(e @ e) / tss if tss > 0 else np.nan
    return beta, e, XtX_inv, r2


def _hac_cov_self(X: np.ndarray, e: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
    """Bartlett 核 Newey-West 协方差矩阵（含 T/(T-p) 自由度修正）。"""
    T, p = X.shape
    L = int(max(0, min(L, T - 1)))
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        G = Xe[lag:].T @ Xe[:-lag]
        S = S + w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    cov *= T / max(T - p, 1)  # 自由度修正：L=0 且同方差时退化为普通 OLS 标准误
    return cov


def _ols_hac_self(y: np.ndarray, X: np.ndarray, L: int) -> dict:
    """自实现的 OLS + NW HAC 推断，返回系数/标准误/t 值/p 值/R²。"""
    beta, e, XtX_inv, r2 = _ols_self(y, X)
    T, p = X.shape
    cov = _hac_cov_self(X, e, XtX_inv, L)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / se
    pvals = 2.0 * _stats.t.sf(np.abs(tvals), df=max(T - p, 1))
    return {"beta": beta, "se": se, "t": tvals, "p": pvals, "r2": r2, "resid": e}


def _ols_hac_statsmodels(y: np.ndarray, X: np.ndarray, L: int) -> dict:  # pragma: no cover
    """statsmodels 实现的 OLS + NW HAC 推断（可选依赖存在时使用）。"""
    res = _SM_OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(L)})
    return {
        "beta": np.asarray(res.params),
        "se": np.asarray(res.bse),
        "t": np.asarray(res.tvalues),
        "p": np.asarray(res.pvalues),
        "r2": float(res.rsquared),
        "resid": np.asarray(res.resid),
    }


def ols_hac(y: np.ndarray, x: np.ndarray, L: int, backend: str = "auto") -> dict:
    """单自变量 OLS（自动加常数项）+ Newey-West HAC 推断。

    Args:
        y: 因变量数组。
        x: 自变量数组（函数内部添加常数列）。
        L: Newey-West 滞后阶数。
        backend: "auto"（优先 statsmodels）/ "self"（强制自实现）/ "statsmodels"。

    Returns:
        dict: alpha/beta（含 se/t/p）、r2、n_obs、backend。
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError(f"y 与 x 长度不一致: {y.shape} vs {x.shape}")
    T = y.shape[0]
    if T < 3:
        raise ValueError(f"样本量过小（T={T}），无法进行回归推断")
    X = np.column_stack([np.ones(T), x])

    if backend == "auto":
        backend = "statsmodels" if _HAS_STATSMODELS else "self"
    if backend == "statsmodels":
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels 不可用，请使用 backend='self'")
        out = _ols_hac_statsmodels(y, X, L)
    elif backend == "self":
        out = _ols_hac_self(y, X, L)
    else:
        raise ValueError(f"未知 backend: {backend!r}")

    return {
        "alpha": out["beta"][0],
        "beta": out["beta"][1],
        "alpha_se": out["se"][0],
        "beta_se": out["se"][1],
        "alpha_t": out["t"][0],
        "beta_t": out["t"][1],
        "alpha_p": out["p"][0],
        "beta_p": out["p"][1],
        "r2": out["r2"],
        "n_obs": T,
        "nw_lag": int(L),
        "backend": backend,
    }


def mean_test_hac(series: np.ndarray, L: int, backend: str = "auto") -> dict:
    """单序列均值检验（对常数回归 + NW HAC 标准误）。

    Returns:
        dict: mean/se/t/p/n_obs/nw_lag/backend。
    """
    y = np.asarray(series, dtype=float)
    T = y.shape[0]
    if T < 3:
        raise ValueError(f"样本量过小（T={T}），无法进行均值检验")
    X = np.ones((T, 1))
    if backend == "auto":
        backend = "statsmodels" if _HAS_STATSMODELS else "self"
    if backend == "statsmodels":
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels 不可用，请使用 backend='self'")
        out = _ols_hac_statsmodels(y, X, L)  # pragma: no cover
    elif backend == "self":
        out = _ols_hac_self(y, X, L)
    else:
        raise ValueError(f"未知 backend: {backend!r}")
    return {
        "mean": out["beta"][0],
        "se": out["se"][0],
        "t": out["t"][0],
        "p": out["p"][0],
        "n_obs": T,
        "nw_lag": int(L),
        "backend": backend,
    }
