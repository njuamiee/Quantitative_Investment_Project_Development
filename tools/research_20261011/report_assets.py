"""Reproducible publication figures and compact cross-horizon result tables."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from common import RUN,HORIZONS,dump

def main():
    out=RUN/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows=[];auto=[]
    for h in HORIZONS:
        x=json.loads((RUN/f'h{h}/candidate_metrics.json').read_text(encoding='utf-8'));rows.append(x)
        auto.append(json.loads((RUN/f'h{h}/main_period_metrics.json').read_text(encoding='utf-8')))
    result=pd.DataFrame(rows);result.to_csv(RUN/'candidate_summary.csv',index=False,encoding='utf-8-sig');pd.DataFrame(auto).to_csv(RUN/'validation_selected_summary.csv',index=False,encoding='utf-8-sig')
    colors=['#6a8d92','#bc8b45','#176b91','#7151a6','#b45050']
    fig,ax=plt.subplots(2,1,figsize=(9,5.6),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
    for h,c in zip(HORIZONS,colors):
        d=pd.read_csv(RUN/f'h{h}/candidate_daily.csv',parse_dates=['date']);nav=d.equity/1e6;peak=np.maximum.accumulate(np.r_[1,nav])[1:]
        ax[0].plot(d.date,nav,label=f'{h}日',color=c,lw=2 if h==5 else 1.2);ax[1].plot(d.date,nav/peak-1,color=c,lw=1.2)
    ax[0].legend(ncol=5,frameon=False);ax[0].set_ylabel('含现金净值');ax[1].set_ylabel('回撤');ax[1].yaxis.set_major_formatter(PercentFormatter(1));ax[0].set_title('历史筛选候选：连续账户净值与回撤（5bp/边）',loc='left',fontweight='bold')
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'candidate_nav_drawdown.png',dpi=190);fig.savefig(out/'candidate_nav_drawdown.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(9,3.6));x=np.arange(5)
    ax[0].bar(x-.18,result.annual_return,width=.36,color='#176b91',label='年化收益');ax[0].bar(x+.18,result.max_drawdown,width=.36,color='#b45050',label='最大回撤')
    ax[0].set_xticks(x,[str(h)+'日' for h in HORIZONS]);ax[0].yaxis.set_major_formatter(PercentFormatter(1));ax[0].legend(frameon=False)
    ax[1].bar(x-.18,result.sharpe_rf15,width=.36,color='#176b91',label='夏普 rf=1.5%');ax[1].bar(x+.18,result.calmar,width=.36,color='#57947b',label='年化Calmar')
    ax[1].axhline(1,color='#176b91',ls='--',lw=.8);ax[1].axhline(2,color='#57947b',ls='--',lw=.8);ax[1].set_xticks(x,[str(h)+'日' for h in HORIZONS]);ax[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(out/'candidate_metrics.png',dpi=190);plt.close(fig)
    controls=pd.DataFrame(json.loads((RUN/'h5/candidate_hmm_controls.json').read_text(encoding='utf-8')))
    labels=['固定25%','波动控制×25%','HMM2防守×25%','HMM3防守×25%','HMM3增长×25%','固定敞口匹配']
    fig,ax=plt.subplots(figsize=(9,3.6));x=np.arange(len(controls));ax.bar(x-.18,controls.sharpe_rf15,width=.36,color='#176b91',label='夏普 rf=1.5%');ax.bar(x+.18,controls.calmar,width=.36,color='#57947b',label='Calmar');ax.axhline(2,ls='--',color='#57947b',lw=.8);ax.set_xticks(x,labels,fontsize=8);ax.set_title('相同IC选股信号：风险规则对照',loc='left',fontweight='bold');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'hmm_controls.png',dpi=190);plt.close(fig)
    reg=pd.read_csv(RUN/'regime_probabilities.csv',parse_dates=['date']);reg=reg[reg.date>='2024-10-08'];d=pd.read_csv(RUN/'h5/candidate_daily.csv',parse_dates=['date'])
    fig,ax=plt.subplots(2,1,figsize=(9,4.8),sharex=True)
    ax[0].stackplot(reg.date,*[reg['hmm3_p_'+str(k)].fillna(1/3) for k in range(3)],labels=['低波状态','中波状态','高波状态'],colors=['#88b8a3','#e5c675','#ca8179'],alpha=.9);ax[0].legend(ncol=3,frameon=False,loc='upper left',fontsize=8);ax[0].set_ylabel('次日状态概率')
    ax[1].plot(d.date,d.target_cap,label='风险仓位上限',color='#176b91');ax[1].plot(d.date,d.exposure,label='实际持仓',color='#b45050',alpha=.8);ax[1].legend(ncol=2,frameon=False);ax[1].yaxis.set_major_formatter(PercentFormatter(1));ax[1].set_ylabel('资产占比');fig.tight_layout();fig.savefig(out/'hmm_probabilities_exposure.png',dpi=190);plt.close(fig)
    source=pd.read_csv(RUN/'h5/fixed_comparisons.csv');pivot=source.pivot(index='model',columns='policy',values='calmar')
    fig,ax=plt.subplots(figsize=(9,4.8));im=ax.imshow(pivot.to_numpy(),vmin=-1,vmax=3,cmap='RdYlGn',aspect='auto')
    ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=35,ha='right',fontsize=8);ax.set_yticks(range(len(pivot.index)),pivot.index,fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):ax.text(j,i,f'{pivot.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
    ax.set_title('5日完整固定方案对照：年化Calmar（事后比较）',loc='left',fontweight='bold');fig.colorbar(im,ax=ax,fraction=.03);fig.tight_layout();fig.savefig(out/'h5_all_calmar.png',dpi=190);plt.close(fig)
    print('report figures and summaries complete',flush=True)

if __name__=='__main__':main()
