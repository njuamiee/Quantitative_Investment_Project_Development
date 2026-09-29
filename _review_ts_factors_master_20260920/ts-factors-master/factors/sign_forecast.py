"""Rolling Rank IC 正负（符号）预测：多模型对比评估 + 模型×k 面板热力图。

口径（信息可得性）：滚动 IC_t 的窗口内包含 r(t→t+k) = Close[t+1+k]/Open[t+1]，
即 IC_t 要到 t+k+1 日收盘后才算得出来；t 日收盘时最新可用的 IC 下标为 t-k-1。
因此"预测未来第 h 天的 IC 正负"= 用 IC_{..t-k-1} 预测 sign(IC_{t+h})，
实际跨度 gap = h + k + 1 个 bar，评估时严格按此构造特征/标签对。

模型（结果表与面板图的列顺序见 MODELS）：
- base   : 训练集多数类（IC 均值的符号），基准命中率；
- persist: 最近已知 IC 的符号直接延续；
- ou     : AR(1)/OU 条件概率 P(IC_{t+h}>0 | IC_last) = Φ(m/σ_gap)，解析解，
           随 gap 增大自动衰减到长期均值方向（m、σ_gap 均有闭式表达）；
- logit  : L2 逻辑回归（IRLS 实现，无外部依赖），特征为
           [最近已知 IC, 再滞后 1, 近 5 根均值]。

评估：每条 (k, W) IC 序列按时间先后切分，前 1-test_frac 训练、末 test_frac 测试，
指标为测试集符号命中率。注意相邻测试点的 IC 窗口互相重叠，命中率样本间相关，
解读时应把它当作水平估计而非有独立大样本支撑的精确值。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as _stats

# 模型列表（结果表列与热力图面板的统一顺序）
MODELS = ("base", "persist", "ou", "logit")


# ----------------------------------------------------------------------
# 模型实现（纯 numpy/scipy）
# ----------------------------------------------------------------------
def _ou_fit(y: np.ndarray) -> tuple[float, float, float]:
    """AR(1) 最小二乘拟合，返回 (mu, b, sigma_eps)；b 截断到 (-0.999, 0.999)。"""
    mu = float(y.mean())
    yc = y - mu
    denom = yc[:-1] @ yc[:-1]
    b = float(yc[:-1] @ yc[1:] / denom) if denom > 0 else 0.0
    b = float(np.clip(b, -0.999, 0.999))
    resid = yc[1:] - b * yc[:-1]
    return mu, b, float(resid.std())


def _ou_p_pos(mu: float, b: float, sig: float,
              x_last: np.ndarray, gap: int) -> np.ndarray:
    """P(IC_{t+gap} > 0 | IC_t = x_last)：AR(1) 条件正态的解析解。"""
    m = mu + (b ** gap) * (x_last - mu)
    if abs(b) < 1:
        var = sig ** 2 * (1.0 - b ** (2 * gap)) / (1.0 - b ** 2)
    else:
        var = sig ** 2 * gap
    return _stats.norm.cdf(m / np.sqrt(max(float(var), 1e-12)))


def _logit_fit(X: np.ndarray, y: np.ndarray,
               l2: float = 1.0, n_iter: int = 60) -> np.ndarray:
    """L2 逻辑回归（IRLS/牛顿法），X 不含常数列（内部自动补）。"""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        wd = np.clip(p * (1.0 - p), 1e-9, None)
        H = (Xb.T * wd) @ Xb + l2 * np.eye(Xb.shape[1])
        g = Xb.T @ (y - p) - l2 * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _logit_p(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


# ----------------------------------------------------------------------
# 网格评估
# ----------------------------------------------------------------------
def eval_sign_column(x: np.ndarray, k: int, h: int,
                     test_frac: float = 0.2) -> dict | None:
    """单条 IC 序列在"预测未来第 h 天正负"任务上的命中率（4 个模型 + 参考列）。

    gap = h + k + 1（信息可得性口径，见模块 docstring）；样本过短返回 None。
    返回键：MODELS 中各模型的命中率 + b（训练段 AR(1) 系数）+ test_frac_pos。
    """
    gap = h + k + 1
    T = len(x)
    n_test = max(int(T * test_frac), 30)
    if T - n_test < gap + 60:
        return None
    y = (x > 0).astype(float)
    # 特征/标签对：t 为目标时刻，j=t-gap 为最近已知 IC 下标
    t_idx = np.arange(gap + 5, T)
    roll5 = pd.Series(x).rolling(5).mean().values
    X = np.column_stack([x[t_idx - gap], x[t_idx - gap - 1], roll5[t_idx - gap]])
    tgt = y[t_idx]
    ntr = T - n_test - (gap + 5)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = tgt[:ntr], tgt[ntr:]
    x_last_te = x[t_idx[ntr:] - gap]

    rows = {}
    rows["base"] = float((yte == float(ytr.mean() > 0.5)).mean())
    rows["persist"] = float(((x_last_te > 0).astype(float) == yte).mean())
    mu, b, sig = _ou_fit(x[:T - n_test])
    rows["ou"] = float(((_ou_p_pos(mu, b, sig, x_last_te, gap) > 0.5)
                        .astype(float) == yte).mean())
    w = _logit_fit(Xtr, ytr)
    rows["logit"] = float(((_logit_p(w, Xte) > 0.5).astype(float) == yte).mean())
    rows["b"] = b
    rows["test_frac_pos"] = float(yte.mean())
    return rows


def evaluate_sign_grid(ic_frame: pd.DataFrame, hs=(0, 5, 10, 20),
                       test_frac: float = 0.2) -> pd.DataFrame:
    """对 ic_frame 中全部 rank_ic_k{k}_W{w} 列 × hs 逐格评估，返回长表。

    ic_frame 为 report.rolling_ic_frame 的输出（列含 factor 与 rank_ic_k*_W*）。
    返回列：k/W/h/MODELS 各模型命中率/b/test_frac_pos。
    """
    rows = []
    for c in ic_frame.columns:
        if not c.startswith("rank_ic_"):
            continue
        k = int(c.split("_k")[1].split("_W")[0])
        w_ = int(c.split("_W")[1])
        x = ic_frame[c].dropna().values
        for h in hs:
            r = eval_sign_column(x, k, int(h), test_frac)
            if r:
                rows.append({"col": c, "k": k, "W": w_, "h": int(h), **r})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 面板热力图
# ----------------------------------------------------------------------
def plot_sign_heatmap(res: pd.DataFrame, save_path: str,
                      title: str | None = None) -> str:
    """模型×k 组合的符号命中率面板热力图：行=k、列=模型，面板内横轴 h、纵轴 W。

    色标以 0.50（随机水平）为中心（绿=优于随机，红=不如随机）。返回保存路径。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ks = sorted(res["k"].unique())
    ws = sorted(res["W"].unique())
    hs = sorted(res["h"].unique())

    norm = TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=0.95)
    fig, axes = plt.subplots(len(ks), len(MODELS),
                             figsize=(3.2 * len(MODELS), 2.4 * len(ks)),
                             squeeze=False)
    im = None
    for i, k in enumerate(ks):
        sub = res[res["k"] == k]
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            mat = np.full((len(ws), len(hs)), np.nan)
            for a, w in enumerate(ws):
                for b_, h in enumerate(hs):
                    cell = sub[(sub["W"] == w) & (sub["h"] == h)]
                    if len(cell):
                        mat[a, b_] = cell[m].iloc[0]
            im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto",
                           origin="lower")
            for a in range(len(ws)):
                for b_ in range(len(hs)):
                    if not np.isnan(mat[a, b_]):
                        ax.text(b_, a, f"{mat[a, b_]:.2f}", ha="center",
                                va="center", fontsize=7)
            ax.set_xticks(range(len(hs)), [str(h) for h in hs])
            ax.set_yticks(range(len(ws)), [str(w) for w in ws])
            if i == len(ks) - 1:
                ax.set_xlabel("h (days ahead)")
            if j == 0:
                ax.set_ylabel("W")
            ax.set_title(f"k={k} / {m}", fontsize=9)
    fig.suptitle(title or "Sign-prediction hit rate by (model, k, h, W) "
                           "- center 0.50 = chance level", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.96, 0.98))
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.6)
    cbar.set_label("hit rate")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    return save_path
