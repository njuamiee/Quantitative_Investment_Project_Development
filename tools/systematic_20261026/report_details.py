"""Readable decisions and paired-effect charts; no changes to model selection."""
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import REPORT,write_csv

NAMES=dict(ridge='Ridge线性模型',ic='IC权重',conditional_ic='状态条件IC',
    lightgbm='LightGBM',ranker='LightGBM排序',xgboost='XGBoost',extra_trees='ExtraTrees',catboost='CatBoost',
    daily='每日',monthly='每月',quarterly='每季',daily_auto='日频+公式候选',vendor_asof='再加历史可得供应商字段',
    equal='等权',inverse_vol='波动较小者权重较高',full='基础100%',half='基础50%',vol15='市场波动预算15%',
    none='不启用',hmm2='两状态HMM',hmm3='三状态HMM',scheduled='计划换仓',entry_stop8='入场价止损8%',
    trailing10='峰值追踪止损10%',portfolio_dd='组合回撤降仓',industry_size='行业/市值集中度上限',
    rank='排名入场',rebound_confirm='反弹确认',horizon='随预测期限')

RISK_LABELS={
    ('topn','5'):'最多5只 − 20只',('topn','40'):'最多40只 − 20只',
    ('weighting','inverse_vol'):'逆波动权重 − 等权',
    ('exposure','half'):'基础50% − 100%',('exposure','vol15'):'波动预算 − 100%',
    ('regime','hmm2'):'两状态HMM − 无HMM',('regime','hmm3'):'三状态HMM − 无HMM',
    ('exit','entry_stop8'):'入场止损8% − 无止损',('exit','trailing10'):'追踪止损10% − 无止损',
    ('exit','portfolio_dd'):'回撤降仓 − 无降仓',('concentration','industry_size'):'集中度限制 − 无限制',
    ('entry','rebound_confirm'):'反弹确认 − 排名入场',('pool','50'):'交易池50 − 100',
    ('rebalance','daily'):'每日换仓 − 随预测期限'}

def detail_charts(paired,risk_rows,figsave):
    p=paired[paired.period=='recent'];r=pd.DataFrame(risk_rows)
    write_csv(r,REPORT/'风控控制变量配对比较.csv')
    r=r[r.period=='recent'];out=[]
    for axis,alts,title,filename in [
        ('count',['40','60','80','100'],'只改变候选因子数：与20个配对','15_因子数量控制比较.png'),
        ('model',['ridge','ic','conditional_ic','ranker','xgboost','extra_trees','catboost'],
         '只改变模型家族：与LightGBM配对','16_模型家族控制比较.png')]:
        fig,ax=plt.subplots(figsize=(12,5));data=[];labels=[]
        for alt in alts:
            s=p[(p.axis==axis)&(p.alternative==alt)]
            data.append(s.median_return_change.to_numpy()*100 if len(s) else [np.nan])
            labels.append(NAMES.get(alt,alt)+f'\n{len(s)}对')
        ax.boxplot(data,labels=labels,showfliers=False);ax.axhline(0,color='gray',lw=.8)
        ax.set(title=title,ylabel='相同账户规则下收益差中位数（百分点）')
        ax.tick_params(axis='x',labelsize=8);figsave(fig,filename);out.append(filename)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,axis,alts,labels,title in [
        (axs[0],'window',['126','504'],['126日','504日'],'训练窗口：与252日配对'),
        (axs[1],'library',['daily_auto','vendor_asof'],['日频+公式','日频+公式+供应商'],'因子库：与原日频库配对')]:
        data=[];xl=[]
        for alt,label in zip(alts,labels):
            s=p[(p.axis==axis)&(p.alternative==alt)]
            data.append(s.median_return_change.to_numpy()*100 if len(s) else [np.nan]);xl.append(label+f'\n{len(s)}对')
        ax.boxplot(data,labels=xl,showfliers=False);ax.axhline(0,color='gray',lw=.8)
        ax.set(title=title,ylabel='收益差中位数（百分点）')
    name='17_窗口与因子库控制比较.png';figsave(fig,name);out.append(name)
    fig,axs=plt.subplots(1,3,figsize=(15,8),sharey=True);keys=list(RISK_LABELS)
    for ax,column,title in zip(axs,['median_return_change','median_drawdown_change','median_exposure_change'],
            ['收益变化：右侧更高','回撤幅度变化：左侧更小','实际仓位变化：左侧更低']):
        mid=[];low=[];high=[]
        for axis,alternative in keys:
            z=r[(r.axis==axis)&(r.alternative==alternative)][column].dropna().to_numpy()*100
            q=np.quantile(z,[.25,.5,.75]) if len(z) else [np.nan]*3
            low.append(q[1]-q[0]);mid.append(q[1]);high.append(q[2]-q[1])
        ax.errorbar(mid,np.arange(len(keys)),xerr=[low,high],fmt='o',color='#116A7B',capsize=3)
        ax.axvline(0,color='gray',lw=.8);ax.set(title=title,xlabel='百分点');ax.grid(axis='x',alpha=.15)
    axs[0].set_yticks(np.arange(len(keys)),[RISK_LABELS[k] for k in keys]);axs[0].invert_yaxis()
    fig.suptitle('只改一项账户规则：点为各模型配对中位数的中位数，线为四分位范围',fontsize=12)
    name='18_风险规则控制比较.png';figsave(fig,name);out.append(name)
    return out

def decision_rows(logs):
    rows=[]
    for log in logs:
        c=log.get('model_config');r=log.get('risk_config')
        item={'信号月份':log['signal_month'],'使用截至哪天的账户记录':log['last_validation_close'],
              '选择评分':log['objective'],'模型配置ID':log['candidate'],'风控ID':log['risk_id']}
        if c:
            item.update({'预测期限':f"{c['horizon']}个交易日",'因子数量上限':c['count'],
                '模型':NAMES[c['model']],'重训频率':NAMES[c['frequency']],'训练窗口':c['window'],
                '因子库':'原日频库' if c['library']=='daily' else NAMES[c['library']],
                '最多买入股票数':r['topn'],'基础仓位':NAMES[r['exposure']],
                '权重':NAMES[r['weighting']],'市场状态':NAMES[r['regime']],
                '退出规则':NAMES[r['exit']],'集中度':NAMES[r['concentration']],
                '入场条件':NAMES[r['entry']],'交易池数量':r['pool'],'换仓间隔':r['rebalance_days']})
        else:item.update({'模型':'无合格候选，按预先固定的后备分支执行'})
        rows.append(item)
    write_csv(pd.DataFrame(rows),REPORT/'逐月选择规则明细.csv')
    body='<h2>每个月具体用了哪套规则</h2><p>下列是当前已完成候选集合的历史模拟决定。因子数为上限，实际数量见因子审计；基础仓位和持股数仍受HMM、个股上限及成交容量约束。预测期限决定训练答案，换仓间隔决定重新下单的频率，重训频率决定模型多久重新学习，三者不同。</p>'
    for item in rows:
        month=item['信号月份'];cut=item['使用截至哪天的账户记录'] or '尚无账户验证历史'
        title=f'{month}｜{item["模型"]}'
        if item.get('预测期限'):title+=f'｜{item["预测期限"]}｜{item["重训频率"]}重训'
        body+='<details><summary>'+html.escape(title)+'</summary><p>选型只使用截至 '+html.escape(cut)+' 的账户记录。</p>'
        if item.get('预测期限'):
            content={k:v for k,v in item.items() if k not in ['信号月份','使用截至哪天的账户记录']}
            body+=pd.DataFrame(content.items(),columns=['参数','本月规则']).to_html(index=False,escape=True)
        body+='</details>'
    return body

def acceptance_html(summary):
    r=next(x for x in summary if x['strategy']=='rolling_cash_fallback_fee5' and x['period']=='recent')
    outcome='同时达到三个门槛' if r['pass_target'] else '尚未同时达到三个门槛'
    return ('<h2>先看这一页怎么判断结果</h2>'
        '<p>模型像选股评分员，风控像投资额度管理员，账户账本负责按价格、现金和成交限制记账。每月的规则选择器只看之前已经发生的账户表现，再决定下一月用哪一套。实验多，意味着检查的方案多，并不意味着赚钱能力自动更强。</p>'
        f'<p><strong>当前逐月选择系统（无合格候选则现金，每边成本5bp）{outcome}。</strong>'
        f'最近区间收益 {r["total_return"]:.2%}，最大回撤 {r["max_drawdown"]:.2%}，夏普 {r["sharpe"]:.2f}，卡玛 {r["calmar"]:.2f}。'
        '完整批次未结束时，这只是本次快照的结果。</p>'
        '<p>收益是账户赚亏多少；回撤是从曾经达到的最高账户价值向下跌了多少；夏普衡量承担日常波动后赚得是否足够；卡玛用复合年化收益除以最大回撤。验收要求三项同时满足：回撤小于30%、夏普大于1、卡玛大于2。空仓或低仓位可以减少亏损，但不能单独证明选股有效。</p>')
