"""Reconcile extension results and generate source-linked research figures."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'training_runs/teacher9_20261008';OLD=ROOT/'training_runs/teacher9_20260930';F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.dpi':170})
COLORS=['#1f4e79','#b45f06','#397f68','#81679d','#8b4a4a']
def readj(name,folder=R):return json.loads((folder/name).read_text('utf-8'))
def curve(name,folder=R):return pd.read_csv(folder/(name+'_account.csv'),parse_dates=['date'])
def finish(fig,name):fig.tight_layout(pad=1.8);fig.savefig(F/(name+'.png'),facecolor='white',bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def grid(ax):ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)

def main():
    results=pd.read_csv(R/'extension_results.csv').set_index('model');folds=readj('folds.json');old=readj('bj_account_metrics.json',OLD)
    summary={'asof':'2026-10-08','fixed_bj_period':'2024-10-08 to 2026-09-29','factor_count_results':results.loc[[f'count_{k}' for k in [20,40,60,80,100]]].reset_index().to_dict('records'),
      'validation_selected':results.loc['validation_count'].to_dict(),'ridge_validation_selected':results.loc['validation_ridge'].to_dict(),
      'all_extensions':results.reset_index().to_dict('records'),'folds':folds,'server':readj('sftp_update_summary.json'),
      'dividends':readj('dividend_summary.json'),'dividend_scenario':readj('dividend_immediate_assumption_metrics.json'),
      'new_data':readj('new_data_extension_summary.json'),'hfq':readj('hfq_results.json'),'hmm_strict':readj('hmm_strict_metrics.json'),
      'minute_q4':readj('minute_cash_results.json'),'minute_q3':readj('q3_minute_cash_results.json'),
      'shsz_alpha':readj('account_results.json',R/'shsz_alpha'),'shsz_alpha_data':readj('data_summary.json',R/'shsz_alpha')}
    # Count-40 has the same feature identities and model settings as the frozen reference.
    summary['legacy_count40_return_difference']=float(results.loc['count_40','total_return']-old['pred_selected']['total_return'])
    ks=[20,40,60,80,100];fig,ax=plt.subplots(1,2,figsize=(11.8,4.4))
    for model,label,color in [('count','LightGBM','#1f4e79'),('ridge','Ridge','#b45f06')]:
        sel=results.loc[[f'{model}_{k}' for k in ks]]
        ax[0].plot(ks,sel.total_return,marker='o',label=label,color=color)
        ax[1].plot(ks,sel.max_drawdown,marker='o',label=label,color=color)
    for a,title in zip(ax,['累计收益：因子更多是否更好','最大回撤：过程中的最深下跌']):
        a.set(title=title,xlabel='因子数量上限',xticks=ks);a.yaxis.set_major_formatter(PercentFormatter(1));grid(a);a.legend(frameon=False)
    finish(fig,'01_factor_count')
    fig,ax=plt.subplots(figsize=(11.8,4.8))
    for k,c in zip(ks,COLORS):
        d=curve(f'count_{k}');ax.plot(d.date,d.equity/1e6,color=c,label=f'{k} 个因子',alpha=.8)
    d=curve('validation_count');ax.plot(d.date,d.equity/1e6,color='#17232e',lw=2.5,label='验证期选择数量')
    ax.set(title='相同历史与交易规则下的连续账户',ylabel='账户净值（初始 = 1）');grid(ax);ax.legend(ncol=3,frameon=False);finish(fig,'02_count_nav')
    vm=pd.read_csv(R/'validation_models.csv');mat=vm[vm.model.isin([f'count_{k}' for k in ks])].pivot(index='model',columns='quarter',values='validation_ic').reindex([f'count_{k}' for k in ks])
    fig,ax=plt.subplots(figsize=(11.8,4.1));im=ax.imshow(mat,aspect='auto',cmap='Blues',vmin=mat.min().min()-.02,vmax=mat.max().max()+.02)
    ax.set_xticks(range(len(mat.columns)),mat.columns);ax.set_yticks(range(5),ks);ax.set_ylabel('因子数量上限');ax.set_title('每季度只用过去的验证成绩决定数量（平均 Rank IC）')
    for i in range(5):
        for j in range(len(mat.columns)):
            chosen=folds[j]['selected_count_model']==mat.index[i];ax.text(j,i,f'{mat.iloc[i,j]:.3f}'+(' ★' if chosen else ''),ha='center',va='center',color='white' if mat.iloc[i,j]>(mat.max().max()+mat.min().min())/2 else '#17232e',fontsize=11)
    fig.colorbar(im,ax=ax,label='验证期 Rank IC');finish(fig,'03_validation_choices')
    names=['count_40','family_c01','family_c02','capital_lag1','lag2','leaves_7','leaves_31','sampling_seed_17','sampling_seed_29']
    labels=['40个（原配置）','仅 Alpha101','仅 Alpha191','40个 + 前日股本线索','Alpha 延迟2日','40个 / 7个叶子','40个 / 31个叶子','40个 / 抽样种子17','40个 / 抽样种子29']
    fig,ax=plt.subplots(figsize=(11.8,5.5));values=results.loc[names,'total_return'].to_numpy();ax.barh(labels[::-1],values[::-1],color=['#1f4e79' if v>=0 else '#a34b3f' for v in values[::-1]])
    for i,v in enumerate(values[::-1]):ax.text(v+(.012 if v>=0 else -.012),i,f'{v:+.2%}',va='center',ha='left' if v>=0 else 'right')
    ax.set_xlim(min(0,values.min())-.15,max(values.max(),0)+.2);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set(title='改变一个研究条件，账户结果也会改变',xlabel='累计价格研究收益');ax.grid(axis='x',alpha=.2);finish(fig,'04_sensitivity')
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.5))
    for ax,prefix,title in zip(axs,['','q3_'],['2025年四季度','2026年三季度（截至9月29日）']):
        for name,label,color in [(prefix+'minute_matched_daily','日线成交','#a2abb3'),(prefix+'minute_1pct','分钟账户 / 1%参与率','#1f4e79')]:
            d=curve(name);ax.plot(d.date,d.equity/1e6,label=label,color=color,lw=2)
        ax.set(title=title,ylabel='账户净值（各区间初始 = 1）');grid(ax);ax.tick_params(axis='x',rotation=25);ax.legend(frameon=False,fontsize=10)
    finish(fig,'05_minute_accounts')
    fig,ax=plt.subplots(figsize=(11.8,4.6));hmnames=['原始40因子','原HMM','严格拟合HMM','固定85%上限','简单波动上限'];hmvals=[old['pred_selected']['total_return'],old['hmm_cap']['total_return'],summary['hmm_strict']['total_return'],old['fixed85_cap']['total_return'],old['vol_cap']['total_return']]
    ax.bar(hmnames,hmvals,color=['#a2abb3','#718ba3','#1f4e79','#397f68','#b45f06']);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(title='拟合诊断改善，仍不等于账户效果改善',ylabel='累计价格研究收益')
    for i,v in enumerate(hmvals):ax.text(i,v+.012,f'{v:+.2%}',ha='center');grid(ax)
    finish(fig,'06_hmm_controls')
    fig,ax=plt.subplots(figsize=(11.8,4.7))
    for name,label,color in [('pred_price','12个价格线索','#a2abb3'),('pred_selected','价格 + Alpha101 筛选','#1f4e79'),('pred_ridge','相同筛选线索 / Ridge','#b45f06')]:
        d=curve(name,R/'shsz_alpha');ax.plot(d.date,d.equity/1e6,label=label,color=color)
    ax.set(title='新因子库支持沪深样本扩展：各模型使用同一股票范围',ylabel='账户净值（初始 = 1）');grid(ax);ax.legend(frameon=False);finish(fig,'07_shsz_alpha')
    fig,ax=plt.subplots(figsize=(11.8,4.7))
    for name,label,folder,color in [('bj_pred_selected','仅价格 / 原基线',OLD,'#1f4e79'),('dividend_immediate_assumption','分红送股除权日即时可用假设',R,'#b45f06')]:
        d=curve(name,folder);ax.plot(d.date,d.equity/1e6,label=label,color=color)
    ax.set(title='新增公司行为记录可以量化口径影响，结算时点仍待核验',ylabel='账户净值（初始 = 1）');grid(ax);ax.legend(frameon=False);finish(fig,'08_dividend_sensitivity')
    # Paired block bootstrap of daily ranking improvements, preserving nearby-date dependence.
    base=pd.read_csv(R/'count_40_daily_ic.csv',parse_dates=['date']).set_index('date').ic;boot=[];rng=np.random.default_rng(20261008)
    for name in [f'count_{k}' for k in ks]+['validation_count']:
        v=pd.read_csv(R/(name+'_daily_ic.csv'),parse_dates=['date']).set_index('date').ic;delta=(v-base).dropna().to_numpy();n=len(delta);draws=[]
        for _ in range(2000):
            starts=rng.integers(0,n,size=int(np.ceil(n/10)));idx=np.concatenate([(s+np.arange(10))%n for s in starts])[:n];draws.append(delta[idx].mean())
        boot.append({'model':name,'paired_days':n,'mean_ic_difference_vs40':float(delta.mean()),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),'note':'Pointwise descriptive block bootstrap; not familywise corrected, not untouched holdout.'})
    pd.DataFrame(boot).to_csv(R/'count_ic_bootstrap.csv',index=False);summary['ic_bootstrap']=boot
    summary['figure_sources']={'01_factor_count':'extension_results.csv','02_count_nav':'count_*_account.csv and validation_count_account.csv','03_validation_choices':'validation_models.csv and folds.json','04_sensitivity':'extension_results.csv','05_minute_accounts':'minute_*_account.csv and q3_minute_*_account.csv','06_hmm_controls':'hmm_strict_metrics.json and frozen bj_account_metrics.json','07_shsz_alpha':'shsz_alpha/*_account.csv','08_dividend_sensitivity':'dividend_immediate_assumption_account.csv and frozen baseline'}
    (R/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),'utf-8')
    print(results[['total_return','max_drawdown','mean_ic']].to_string())

if __name__=='__main__':main()
