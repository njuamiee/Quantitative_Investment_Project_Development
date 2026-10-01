"""Results, uncertainty intervals, checks and publication-ready research figures."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from prepare import DATA,RUN,ROOT,dump,read_zip

FIG=RUN/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
COLORS=['#22577A','#C06C35','#2A8C82','#8C568C','#7C8792']
LABELS={'pred_selected':'筛选因子 LightGBM','pred_price':'价格 LightGBM','pred_ridge':'Ridge','pred_simple':'单因子','pool_equal':'Top100等权目标',
        'pred_lag1':'因子额外滞后一天','hmm_cap':'两状态 HMM','fixed85_cap':'固定85%上限','vol_cap':'简单波动控制'}

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)

def bootstrap_mean(values,seed=20260930):
    v=np.asarray(values);v=v[np.isfinite(v)];n=len(v);block=min(10,n)
    rng=np.random.default_rng(seed);means=[]
    for _ in range(2000):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        sample=np.concatenate([v[start:start+block] for start in starts])[:n];means.append(sample.mean())
    return [float(x) for x in np.quantile(means,[.025,.975])]

def main():
    summary={}
    for market in ['bj','shsz']:
        metrics=json.loads((RUN/(market+'_account_metrics.json')).read_text(encoding='utf-8'))
        prediction=json.loads((RUN/(market+'_prediction_metrics.json')).read_text(encoding='utf-8'))
        summary[market]={'accounts':metrics,'prediction':prediction}
        rows=[]
        for name,m in metrics.items():rows.append({'variant':name,'label':LABELS.get(name,name),**m})
        pd.DataFrame(rows).to_csv(RUN/(market+'_results_table.csv'),index=False,encoding='utf-8-sig')
        selected=pd.read_csv(RUN/(market+'_pred_selected_daily_metrics.csv'),parse_dates=['date'])
        price=pd.read_csv(RUN/(market+'_pred_price_daily_metrics.csv'),parse_dates=['date'])
        paired=selected[['date','ic']].merge(price[['date','ic']],on='date',suffixes=('_selected','_price'))
        summary[market]['selected_ic_ci']=bootstrap_mean(selected.ic)
        summary[market]['selected_minus_price_ic_ci']=bootstrap_mean(paired.ic_selected-paired.ic_price)
        fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
        for i,name in enumerate(['pred_selected','pred_price','pred_ridge','pool_equal']):
            d=pd.read_csv(RUN/(market+'_'+name+'_account.csv'),parse_dates=['date'])
            ax.plot(d.date,d.equity/1e6,label=LABELS[name],color=COLORS[i],lw=1.5)
        ax.plot(d.date,d.benchmark_nav,label='北证50价格指数' if market=='bj' else '沪深300价格指数',color='#333333',ls='--',lw=1.3)
        ax.set_ylabel('研究净值（起始1.0）');ax.grid(alpha=.18);ax.legend(ncol=3,loc='upper left',fontsize=8)
        ax.set_title(('北交所' if market=='bj' else '沪深300只可用文件样本')+'：统一账户与价格指数对照\n未计真实公司行为；单边合计成本假设15bp')
        save(fig,market+'_equity')
    bjpanel=pd.read_parquet(DATA/'bj_model_panel.parquet')
    uni=pd.read_csv(RUN/'bj_universe_daily_summary.csv',parse_dates=['date'])
    used=uni[uni.pool>0]
    fig,axes=plt.subplots(2,1,figsize=(10,5.4),sharex=True,layout='constrained')
    axes[0].plot(uni.date,uni.eligible,label='有效历史与活跃度代理合格',color=COLORS[1])
    axes[0].plot(uni.date,uni.pool,label='成交额Top100入池',color=COLORS[0]);axes[0].axhline(100,color='#888',ls='--',lw=.8)
    axes[0].set_ylabel('股票数');axes[0].legend(fontsize=9)
    axes[1].plot(used.date,used.pool_overlap/used.pool,color=COLORS[2]);axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel('成交量池与成交额池重合率');axes[1].set_ylim(0,1.05)
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('股票池每天重建；保留入池后因子不足的记录')
    save(fig,'universe')
    summary['universe']={'max_pool':int(used.pool.max()),'min_nonempty_pool':int(used.pool.min()),
        'median_volume_amount_overlap':float((used.pool_overlap/used.pool).median()),
        'days_below100':int((used.pool<100).sum()),'factor_coverage_mean':float(bjpanel.factor_coverage.mean()),
        'pool_rows_factor_coverage_below70pct':int((bjpanel.factor_coverage<.7).sum())}
    folds=json.loads((RUN/'bj_folds.json').read_text(encoding='utf-8'))
    realfolds=[f for f in folds if 'skipped' not in f]
    last=realfolds[-1]['quarter'];cards=pd.read_csv(RUN/'bj'/last/'factor_cards.csv')
    # Text column name is the CSV index label 'feature'.
    picked=cards[cards.decision=='selected'].sort_values('abs_ic').tail(18)
    corr=pd.read_csv(RUN/'bj'/last/'mean_daily_spearman.csv',index_col=0)
    selected=realfolds[-1]['selected_features'][:18]
    fig,axes=plt.subplots(1,2,figsize=(11,5.4),layout='constrained')
    axes[0].barh(picked.feature.str.replace('x_','',regex=False),picked.mean_ic,color=COLORS[0]);axes[0].axvline(0,color='#777',lw=.7)
    axes[0].set_xlabel('训练期平均每日Rank IC');axes[0].set_title('最后一轮选中字段（节选18个）')
    image=axes[1].imshow(corr.loc[selected,selected],vmin=-1,vmax=1,cmap='RdBu_r')
    axes[1].set_xticks(range(len(selected)),[c.replace('x_','') for c in selected],rotation=90,fontsize=6)
    axes[1].set_yticks(range(len(selected)),[c.replace('x_','') for c in selected],fontsize=6)
    axes[1].set_title('按日横截面Spearman再求均值');fig.colorbar(image,ax=axes[1],shrink=.7)
    save(fig,'factor_selection')
    oos=pd.read_parquet(RUN/'bj_predictions.parquet');quintile=[]
    for date,g in oos.dropna(subset=['pred_simple','label_raw']).groupby('date'):
        ranks=g.pred_simple.rank(method='first',pct=True)
        groups=np.ceil(ranks*5).clip(1,5).astype(int)
        means=g.groupby(groups).label_raw.mean()
        for group,value in means.items():quintile.append({'date':date,'group':int(group),'mean_5d_price_return':float(value)})
    q=pd.DataFrame(quintile);q.to_csv(RUN/'oos_simple_factor_quintiles.csv',index=False)
    means=q.groupby('group').mean_5d_price_return.mean()
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained');ax.bar(means.index,means.values,color=COLORS[0])
    ax.set_xticks(range(1,6),['低分1','2','3','4','高分5']);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.18)
    ax.set_title('训练内选定方向的单因子：外层测试五分组\n五日价格收益均值，标签重叠，不是复利资金曲线')
    save(fig,'quintiles')
    summary['factor_selection']={'bj_folds':len(realfolds),'per_fold_selected':[f['selected_count'] for f in realfolds],
        'last_fold_fdr_under05':int((cards.by_fdr_q<.05).sum()),'quintiles':{str(k):float(v) for k,v in means.items()}}
    summary['factor_selection']['last_fold_selected_alpha101']=sum(c.startswith('x_c01') for c in realfolds[-1]['selected_features'])
    summary['factor_selection']['last_fold_selected_alpha191']=sum(c.startswith('x_c02') for c in realfolds[-1]['selected_features'])
    summary['factor_selection']['shsz_folds']=sum('skipped' not in f for f in json.loads((RUN/'shsz_folds.json').read_text(encoding='utf-8')))
    hmm=pd.read_csv(RUN/'hmm_probabilities.csv',parse_dates=['date']);fits=json.loads((RUN/'hmm_fits.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
    for i,name in enumerate(['pred_selected','hmm_cap','fixed85_cap','vol_cap']):
        d=pd.read_csv(RUN/('bj_'+name+'_account.csv'),parse_dates=['date'])
        axes[0].plot(d.date,d.equity/1e6,label=LABELS[name],color=COLORS[i]);axes[2].plot(d.date,d.gross_exposure,label=LABELS[name],color=COLORS[i],lw=.8)
    axes[0].legend(ncol=2,fontsize=8);axes[0].set_ylabel('研究净值')
    axes[1].plot(hmm.date,hmm.p_high_next,label='下一日高波动概率',color=COLORS[1]);axes[1].plot(hmm.date,hmm.hmm_cap,label='平滑后的仓位上限',color=COLORS[0]);axes[1].legend(fontsize=8)
    axes[1].set_ylabel('概率 / 上限');axes[2].set_ylabel('实际股票仓位')
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('两状态HMM：月度重估、仅前向滤波，与简单减仓对照')
    save(fig,'hmm')
    summary['hmm']={'monthly_fits':len(fits),'minimum_expected_duration':min(min(f['expected_durations']) for f in fits),
        'fits_duration_below3':sum(min(f['expected_durations'])<3 for f in fits),'nonconverged':sum(not f['converged'] for f in fits),
        'strict_monotone_convergence_failures':sum(not f['strict_monotone_convergence'] for f in fits),
        'negative_final_likelihood_steps':sum(f['final_log_likelihood_delta']<0 for f in fits),
        'mean_cap':float(hmm.hmm_cap.mean())}
    h=summary['bj']['accounts'];summary['hmm']['incremental_sharpe_vs_fixed85']=h['hmm_cap']['sharpe']-h['fixed85_cap']['sharpe']
    summary['hmm']['return_difference_vs_fixed85']=h['hmm_cap']['total_return']-h['fixed85_cap']['total_return']
    # Candidate action boundaries identify price sensitivity, not actual cash dividends.
    quality=pd.read_csv(RUN/'bj_price_quality.csv');summary['price_quality']={
        'negative_qfq_close_rows':int(quality.negative_qfq_close_rows.sum()),'stocks_with_negative_qfq':int((quality.negative_qfq_close_rows>0).sum()),
        'candidate_boundaries':int(quality.candidate_adjustment_boundaries.sum()),
        'additive_ohlc_fraction_median':float(quality.additive_ohlc_fraction.median())}
    label=oos.dropna(subset=['label_raw','label_qfq_diagnostic']).copy();label['qfq_minus_raw']=label.label_qfq_diagnostic-label.label_raw
    label.reindex(label.qfq_minus_raw.abs().sort_values(ascending=False).index).head(200).to_csv(RUN/'price_label_discrepancies.csv',index=False)
    summary['price_quality'].update({'oos_label_pairs':len(label),'labels_difference_over1pp':int((label.qfq_minus_raw.abs()>.01).sum()),
        'oos_labels_cross_candidate_boundary':int(oos.label_adjustment_boundary.sum())})
    noevent=oos[~oos.label_adjustment_boundary]
    from train import model_metrics
    summary['price_quality']['no_candidate_boundary_prediction_metrics']=model_metrics(noevent,'pred_selected')[0]
    # Same date aligned cost and HMM comparisons are exported above; no hindsight parameter choice.
    rnd=pd.read_csv(RUN/'bj_random_controls.csv');summary['random_control']={'runs':len(rnd),'median_return':float(rnd.total_return.median()),
        'min_return':float(rnd.total_return.min()),'max_return':float(rnd.total_return.max()),
        'selected_exceeds_random_runs':int((rnd.total_return<h['pred_selected']['total_return']).sum())}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    costs=[5,15,30];returns=[h['selected_cost_5']['total_return'],h['pred_selected']['total_return'],h['selected_cost_30']['total_return']]
    axes[0].plot(costs,returns,marker='o',color=COLORS[0]);axes[0].set_xlabel('单边合计成本假设（bp）');axes[0].set_ylabel('累计价格账户收益');axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].set_title('成本敏感性')
    axes[1].hist(rnd.total_return,bins=6,color='#A8B6C3');axes[1].axvline(h['pred_selected']['total_return'],color=COLORS[1],label='筛选因子LightGBM');axes[1].xaxis.set_major_formatter(PercentFormatter(1));axes[1].set_title('10次随机Top5账户');axes[1].legend(fontsize=8)
    save(fig,'cost_random')
    for p in [RUN/'dictionary_mapping.json',RUN/'minute_replay_summary.json',RUN/'share_capital_summary.json',RUN/'test_results.json',RUN/'output_verification.json']:
        summary[p.stem]=json.loads(p.read_text(encoding='utf-8'))
    frozen=json.loads((RUN/'frozen_old_run_hashes.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/'training_runs/lgbm_hmm_20260920'/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
    for f in realfolds:
        assert pd.Timestamp(f['train_label_end'])<pd.Timestamp(f['validation_start'])
        assert pd.Timestamp(f['refit_label_end'])<pd.Timestamp(f['test_first'])
    assert uni.pool.max()<=100
    summary['verification']={'old_run_files_unchanged':len(frozen),'bj_fold_temporal_boundaries_pass':True,'daily_pool_never_exceeds100':True,
                             'python':platform.python_version(),'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    dump(summary,RUN/'final_summary.json')
    print('summary_ready',RUN/'final_summary.json',flush=True)

if __name__=='__main__':main()
