"""从冻结的首轮产物生成评审图表；不重训，不改写首轮结果。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "training_runs/lgbm_hmm_20260920"
OUT = ROOT / "学习笔记/老师意见分析_20260923"
COLORS = ["#176a91", "#df9135", "#bf4a5e"]
STATES = ["低波动", "中波动", "高波动（原 crisis）"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=165, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.18)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    metrics = json.loads((RUN / "metrics.json").read_text("utf-8"))
    cfg = json.loads((RUN / "config.json").read_text("utf-8"))
    capital = cfg["initial_capital"]
    fixed = pd.read_csv(RUN / "portfolio_fixed_daily.csv", parse_dates=["date"])
    hmm = pd.read_csv(RUN / "portfolio_hmm_daily.csv", parse_dates=["date"])
    probs = pd.read_csv(RUN / "hmm_probabilities.csv", parse_dates=["date"])
    pred = pd.read_parquet(RUN / "test_predictions.parquet")
    assert not pred.duplicated(["date", "stockid"]).any()
    assert np.allclose(probs.filter(regex="^p_").sum(axis=1), 1)
    assert np.allclose(probs.filter(regex="^p_").to_numpy() @ np.array([1,.6,.2]), probs.exposure_cap)
    assert fixed.date.equals(hmm.date)
    for frame, key in [(fixed, "portfolio_fixed"), (hmm, "portfolio_hmm")]:
        assert np.allclose(frame.cash + frame.market_value, frame.equity)
        assert np.isclose(frame.equity.iloc[-1] / capital - 1, metrics[key]["total_return"])
    pool = pred.groupby("date").stockid.nunique().rename("n_scored").reset_index()
    pool.to_csv(OUT / "daily_universe_counts.csv", index=False, encoding="utf-8-sig")

    # 资金曲线添加交易前初始本金点；收益不把相互重叠的五日标签连乘。
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for data, color, label in [(fixed,COLORS[0],"LightGBM：无 HMM 仓位覆盖"),
                                (hmm,COLORS[2],"LightGBM + HMM")]:
        dates = pd.concat([pd.Series([pred.date.min()]), data.date], ignore_index=True)
        equity = np.r_[capital, data.equity.to_numpy()]
        axes[0].plot(dates, equity / 10000, color=color, label=label, lw=1.9)
        axes[0].annotate(f"{equity[-1]/10000:.2f} 万元", (dates.iloc[-1],equity[-1]/10000),
                         xytext=(-6,8), textcoords="offset points", ha="right",color=color)
        axes[1].plot(dates, equity / np.maximum.accumulate(equity) - 1, color=color, lw=1.4)
        axes[2].plot(data.date,data.gross_exposure, color=color, lw=1.2,label=label)
    axes[0].axhline(100, color="#888888", ls="--", lw=.8)
    axes[0].set_ylabel("账户权益（万元）")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set_ylabel("回撤")
    axes[2].set_ylabel("实际股票仓位")
    for ax in axes:
        time_axis(ax)
        ax.axvline(pd.Timestamp("2025-12-31"),color="#666666",ls=":",lw=1)
    for ax in axes[1:]: ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("首轮研究回测：资金、回撤与实际仓位\n信号 2025-01-08—12-31；记账至 2026-01-23；未复权输入，买卖各 10bp",fontsize=14)
    fig.supxlabel("竖虚线后为跨年平仓/记账；未加入指数基准，不能据此认定跑赢大盘",fontsize=10)
    save(fig,"01_equity_drawdown_exposure")

    fig, axes = plt.subplots(3,1,figsize=(12,9),sharex=True,layout="constrained")
    axes[0].stackplot(probs.date,*[probs[c] for c in ["p_low_vol_next","p_high_vol_next","p_crisis_next"]],
                      labels=STATES,colors=COLORS,alpha=.85)
    axes[0].legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,1.25),frameon=False)
    axes[0].set_ylabel("次日状态预测概率")
    # 预测日与执行日显式分开；实际仓位按次日日期合并，跨年时也对齐。
    execution = pred[["date","entry_date"]].drop_duplicates()
    assert not execution.date.duplicated().any()
    caps = probs.merge(execution,on="date",validate="one_to_one")
    caps = caps.merge(hmm[["date","gross_exposure"]].rename(columns={"date":"entry_date"}),on="entry_date",validate="one_to_one")
    caps.to_csv(OUT / "hmm_cap_execution_alignment.csv",index=False,encoding="utf-8-sig")
    axes[1].plot(caps.entry_date,caps.exposure_cap,color="#7355a2",label="前日预测对应的新增开仓限额",lw=1.5)
    axes[1].plot(hmm.date,hmm.gross_exposure,color=COLORS[2],label="当日收盘实际仓位",lw=1.3)
    axes[1].set_ylabel("仓位 / 限额")
    axes[1].legend(frameon=False,loc="upper left")
    axes[2].plot(pool.date,pool.n_scored,color=COLORS[0],label="实际每日评分股票数")
    axes[2].axhline(100,color=COLORS[2],ls="--",label="老师建议的上限 100")
    axes[2].set_ylabel("股票数")
    axes[2].legend(frameon=False,loc="lower left")
    for ax in axes: time_axis(ax)
    for ax in axes[:2]: ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("HMM 概率与执行语义：限额只约束新增买入\n状态按训练期波动均值排序；最高波动状态不等于已经识别下跌危机",fontsize=14)
    save(fig,"02_hmm_probabilities_and_universe")

    fits = metrics["hmm"]["fits"]
    rows=[]
    fig, axes=plt.subplots(3,4,figsize=(13,9),layout="constrained")
    for fit,ax in zip(fits,axes.flat):
        order=fit["state_order_original"]
        a=np.array(fit["transition_matrix"])[np.ix_(order,order)]
        means=np.array(fit["means"])[order]
        ax.imshow(a,vmin=0,vmax=1,cmap="Blues")
        ax.set_xticks(range(3),["低","中","高"])
        ax.set_yticks(range(3),["低","中","高"])
        ax.set_title(fit["fit_month"])
        for i in range(3):
            for j in range(3):ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",color="white" if a[i,j]>.55 else "#203040")
            rows.append({"fit_month":fit["fit_month"],"train_end":fit["train_end"],"state":STATES[i],
                         "duration":fit["expected_duration_ordered"][i],"return_mean_z":means[i,0],
                         "log_vol_mean_z":means[i,1],"breadth_mean_z":means[i,2],"dispersion_mean_z":means[i,3],
                         "converged":fit["converged"]})
    fig.suptitle("12 次月度拟合的状态转移矩阵\n行：当前状态；列：下一状态；每月均重新排列为低 / 中 / 高波动",fontsize=14)
    save(fig,"03_hmm_transition_matrices")
    state=pd.DataFrame(rows)
    state.to_csv(OUT / "hmm_monthly_state_diagnostics.csv",index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout="constrained")
    for label,color in zip(STATES,COLORS):
        data=state[state.state==label]
        axes[0].plot(data.fit_month,data.duration,marker="o",color=color,label=label)
        axes[1].plot(data.fit_month,data.return_mean_z,marker="o",color=color)
    axes[0].axhline(3,color="#555555",ls="--",label="旧验收参考：3 日")
    axes[0].set_ylabel("模型隐含平均持续期（日）")
    axes[0].legend(frameon=False,ncol=2)
    axes[1].axhline(0,color="#555555",ls="--")
    axes[1].set_ylabel("市场收益均值（训练窗 z 分数）")
    axes[1].tick_params(axis="x",rotation=35)
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle("HMM 状态稳定性与含义\n持续期 = 1 / (1 − 自转移概率)；下图为训练期条件均值，不能当作未来收益",fontsize=14)
    save(fig,"04_hmm_duration_and_state_means")

    # 月度收益从实际账户权益计算，包含首月初始本金及末月结算尾段。
    monthly={}
    for name,data in [("LightGBM",fixed),("LightGBM_HMM",hmm)]:
        end=data.set_index("date").equity.resample("ME").last()
        monthly[name]=end / end.shift(1,fill_value=capital) - 1
    month=pd.DataFrame(monthly)
    month.index=month.index.strftime("%Y-%m")
    month.to_csv(OUT / "monthly_account_returns.csv",encoding="utf-8-sig",index_label="month")
    fig,ax=plt.subplots(figsize=(12,4.5),layout="constrained")
    month.plot.bar(ax=ax,color=[COLORS[0],COLORS[2]],rot=35)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0,color="#777777",lw=.8)
    ax.grid(axis="y",alpha=.18)
    ax.set_xlabel("");ax.set_ylabel("月度账户收益")
    ax.set_title("逐月资金表现：2026-01 仅包含既有信号的平仓与现金记账")
    save(fig,"05_monthly_returns")

    summary={"source_run":str(RUN.relative_to(ROOT)),"new_training":False,
             "pool_unique_stocks":int(pred.stockid.nunique()),"pool_daily_min":int(pool.n_scored.min()),
             "pool_daily_max":int(pool.n_scored.max()),"pool_daily_mean":float(pool.n_scored.mean()),
             "account_start":str(fixed.date.min().date()),"account_end":str(fixed.date.max().date()),
             "actual_hmm_exposure_above_new_entry_cap_days":int((caps.gross_exposure>caps.exposure_cap+1e-8).sum()),
             "highest_vol_state_positive_standardized_return_months":int((state[state.state==STATES[2]].return_mean_z>0).sum()),
             "portfolio_metrics_original":{k:metrics[k] for k in ["portfolio_fixed","portfolio_hmm"]},
             "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.iterdir() if p.is_file()}}
    (OUT/"review_evidence.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["source_sha256","portfolio_metrics_original"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
