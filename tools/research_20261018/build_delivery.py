"""Plain Chinese report with actual equity, excess and loss charts."""
from pathlib import Path
import base64,hashlib,json,shutil,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import markdown
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261018'
OUT=ROOT/'新项目_20261018_深入实验'
FIG=OUT/'图表'
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
LABEL={'quarterly__past_validation_selected':'季度预测＋此前验证选退出',
       'simple__hmm3_gate':'简单因子＋市场状态开关','monthly__breadth':'月度模型＋上涨广度开关',
       'hmm3__hmm3_gate':'状态条件因子＋状态开关','equal_buy_hold':'期初等权买入持有',
       'equal_pool_5day':'全池等权五日换仓','bj50_price_index':'北证50价格指数',
       'baseline__equal':'原100列，全仓','augmented__equal':'新增3列，全仓',
       'baseline__breadth':'原100列，市场过滤','augmented__breadth':'新增3列，市场过滤',
       'global__hmm3_gate':'统一因子权重＋相同状态开关','hmm2__hmm3_gate':'2状态权重＋相同开关',
       'daily':'日频基线＝事前选择0%分钟','daily_plus_minute_fixed25':'固定25%分钟（敏感度）'}
def jsonread(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def savefig(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=165,bbox_inches='tight');plt.close(fig)
def lines(data,names,title,name,dd=False,relative=None):
    fig,ax=plt.subplots(figsize=(11,4.7))
    for i,s in enumerate(names):
        g=data[data.strategy==s].sort_values('date');y=g.equity.to_numpy()/1e6
        if dd:y=(y/np.maximum.accumulate(np.r_[1.,y])[1:]-1)*100
        if relative is not None:
            b=relative.set_index('date').equity.reindex(g.date).to_numpy()/1e6;y=(y/b-1)*100
        ax.plot(g.date,y,label=LABEL.get(s,s),color=['#111','#555','#888','#333','#aaa'][i%5],
                linestyle=['-','--','-.',':','-'][i%5],linewidth=1.8)
    ax.set(title=title,ylabel='回撤（%）' if dd else '相对财富增长（%）' if relative is not None else '资产 / 起始100万元')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.grid(alpha=.18);ax.legend(fontsize=9,ncol=2)
    savefig(fig,name)
def metric_table(frame):
    rows=[]
    for r in frame.to_dict('records'):
        rows.append({'方案':LABEL.get(r['strategy'],r['strategy']),'收益':f"{r['total_return']:.2%}",
                     '回撤':f"{r['max_drawdown']:.2%}",'夏普':f"{r['sharpe_rf15']:.2f}",'卡玛':f"{r['calmar']:.2f}",
                     '比等权买持多赚':f"{r['excess_pp_equal_buy_hold']*100:+.2f}个百分点",
                     '比北证50多赚':f"{r['excess_pp_bj50_price_index']*100:+.2f}个百分点"})
    return pd.DataFrame(rows).to_html(index=False,escape=True,border=0)
def picture(name,caption):
    p=FIG/(name+'.png');data=base64.b64encode(p.read_bytes()).decode()
    return '<figure><img src="data:image/png;base64,'+data+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    evidence=pd.read_csv(RUN/'evidence/all_periods_with_excess.csv')
    deep=pd.read_parquet(RUN/'deep_risk/all_daily.parquet')
    hmm=pd.read_parquet(RUN/'hmm_conditional/account_daily.parquet')
    loop=pd.read_parquet(RUN/'factor_loop/account_daily.parquet')
    minute=pd.read_parquet(RUN/'minute_incremental/account_daily.parquet')
    bench=pd.read_parquet(RUN/'evidence/benchmarks_2024-10-08.parquet')
    chosen=['quarterly__past_validation_selected','simple__hmm3_gate','monthly__breadth']
    plot=pd.concat([deep,bench],ignore_index=True)
    lines(plot,chosen+['equal_buy_hold'],'同一时期、同一100万元：低回撤也有收益代价','01_完整资金曲线')
    lines(plot,chosen+['equal_buy_hold'],'从自己的历史最高资产往下跌了多少','02_完整回撤曲线',dd=True)
    lines(deep,chosen,'相对等权买入持有的财富增长；不是实际期货对冲账户','03_相对基准曲线',relative=bench[bench.strategy=='equal_buy_hold'])
    full=evidence[(evidence.experiment=='deep_risk')&(evidence.period=='full')]
    fig,ax=plt.subplots(figsize=(9,4.8));color=np.where(full.max_drawdown<=.2,'#111','#b5b5b5')
    ax.scatter(full.sharpe_rf15,full.calmar,c=color,s=24);ax.axvline(1,c='#666',ls='--');ax.axhline(2,c='#666',ls='--')
    ax.set(xlabel='夏普（越右越好）',ylabel='卡玛（越上越好）',title='80组退出实验：黑点回撤≤20%，灰点>20%')
    ax.text(.02,.97,'同时在右上区且是黑点，才通过本轮三项全期目标。\n最近一年和最近半年：通过数均为0。',transform=ax.transAxes,va='top',fontsize=10)
    ax.grid(alpha=.15);savefig(fig,'04_全部80组目标分布')
    fig,ax=plt.subplots(figsize=(10,4.4));x=np.arange(3)
    for j,period in enumerate(['full','recent_year','recent_half']):
        vals=[evidence[(evidence.experiment=='deep_risk')&(evidence.period==period)&(evidence.strategy==s)].total_return.iloc[0]*100 for s in chosen]
        ax.bar(x+(j-1)*.24,vals,.24,color=['#222','#777','#bbb'][j],label=['全期','最近一年','最近半年'][j])
    ax.set_xticks(x,[LABEL[s] for s in chosen]);ax.axhline(0,c='black',lw=.7);ax.set(ylabel='累计收益（%）',title='全期合格，不代表最近仍有效');ax.legend();savefig(fig,'05_全期与近期')
    loopbench=pd.read_parquet(RUN/'evidence/benchmarks_2026-01-05.parquet')
    lines(pd.concat([loop,loopbench]),['baseline__equal','augmented__equal','augmented__breadth','equal_buy_hold'],
          '2026年：新增因子的正相关，没有变成稳定的账户改善','06_新因子增量资金')
    lines(hmm,['global__hmm3_gate','hmm2__hmm3_gate','hmm3__hmm3_gate'],
          '同一个状态开关，只改变因子权重；曲线差别较小','07_状态条件因子')
    minbench=pd.read_parquet(RUN/'evidence/benchmarks_2026-07-01.parquet')
    lines(pd.concat([minute,minbench]),['daily','daily_plus_minute_fixed25','equal_buy_hold'],
          '2026Q3分钟增量：正式选择0%；固定25%只作敏感度','08_分钟信号资金')
    attrs=jsonread(RUN/'evidence/drawdown_attribution.json');a=attrs[0]
    loss=pd.read_csv(RUN/'evidence/deep_risk__quarterly__past_validation_selected_drawdown_stocks.csv').head(10)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.barh(loss.stockid.astype(str),loss.net_pnl/1e4,color='#444');ax.invert_yaxis()
    ax.set(xlabel='该回撤区间对资产的贡献（万元，含费用和股息）',title='季度验证选规则：最大回撤期间损失最多的10只股票');ax.grid(axis='x',alpha=.15);savefig(fig,'09_回撤具体亏在哪')
    circuit=deep[deep.strategy=='quarterly__circuit08'].copy();events=circuit[circuit.circuit_fired]
    first=events.date.iloc[0];g=circuit[circuit.date.between(first-pd.Timedelta(days=25),first+pd.Timedelta(days=65))]
    fig,(ax,bx)=plt.subplots(2,1,figsize=(10,5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    ax.plot(g.date,g.equity/1e6,c='#222');ax.axvline(first,c='#555',ls='--',label='根据前日收盘触发');ax.legend();ax.set(ylabel='资产 / 100万元',title='一个真实暂停区间：触发后减仓，恢复不是知道未来最低点')
    bx.step(g.date,g.exposure*100,where='post',c='#222');bx.fill_between(g.date,0,100,where=g.risk_paused,alpha=.15,color='#555');bx.set(ylabel='持股比例（%）',ylim=(0,105));bx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));savefig(fig,'10_提前退出实例')
    # Raw tables remain machine-readable; user-facing report formats percentages.
    evidence.to_csv(OUT/'全部实验_含基准超额.csv',index=False,encoding='utf-8-sig')
    shutil.copy2(RUN/'evidence/all_quarters.csv',OUT/'全部实验_按季度.csv')
    shutil.copy2(ROOT/'tools/research_20261018/中证2000数据核查.md',OUT/'03_中证2000对比_真实数据缺口.md')
    shutil.copy2(RUN/'minute_incremental/中文核查与结论.md',OUT/'05_分钟因子_实际训练和结果.md')
    shutil.copy2(RUN/'minute_incremental/q3_equity_comparison.png',OUT/'q3_equity_comparison.png')
    for source in [RUN/'deep_risk/独立账本审计.md',RUN/'evidence/supplemental_audit.json']:
        shutil.copy2(source,OUT/source.name)
    for file in (RUN/'evidence').glob('*drawdown_stocks.csv'):shutil.copy2(file,OUT/file.name)
    selected=evidence[(evidence.period=='full')&((evidence.experiment=='deep_risk')&evidence.strategy.isin(chosen)|(evidence.experiment=='hmm_conditional')&(evidence.strategy=='hmm3__hmm3_gate'))]
    recent=evidence[(evidence.period=='recent_year')&((evidence.experiment=='deep_risk')&evidence.strategy.isin(chosen)|(evidence.experiment=='hmm_conditional')&(evidence.strategy=='hmm3__hmm3_gate'))]
    looprows=evidence[(evidence.experiment=='factor_loop')&(evidence.period=='full')]
    minrows=evidence[(evidence.experiment=='minute_incremental')&(evidence.period=='full')]
    freeze=jsonread(RUN/'factor_loop/frozen_selection.json');ma=jsonread(RUN/'minute_incremental/metrics_and_manifest.json')
    sections=[]
    def add(id,title,body):sections.append((id,title,body))
    add('conclusion','先看结论：工程与实验已推进，稳定业绩目标尚未通过',
        '<p>新增96组账户实验已经运行：80组退出/仓位、9组状态条件因子、4组AI因子增量、3组分钟增量。完整展示失败结果，使用真实逐日资金曲线。</p>'
        '<p><strong>目前不能宣布全部任务完成。</strong>中证2000真实历史成员仍缺；全期有部分方案达到夏普&gt;1、卡玛&gt;2和回撤≤20%，但最近一年/半年不通过。没有把历史最好的一组改名为已经成功的正式策略。</p>'
        '<p>本轮北交所实验买卖每边5bp，起始100万元，Top20、五日一批。与张子晨指定条件的零成本实验是另一项任务，尚未产生可验证收益。</p>')
    add('equity','全期资金曲线与回撤：2024-10-08至2026-09-30',metric_table(selected)+picture('01_完整资金曲线','四条曲线同一起点。比较收益时还要看下图回撤。')+picture('02_完整回撤曲线','回撤每一天都从该账户此前最高资产计算。')+
        '<p>1017月度主流程同期收益174.79%、回撤28.26%、夏普1.54、卡玛2.36。新增“季度预测＋此前验证选退出”把回撤降到14.94%，但收益46.88%、卡玛1.44，未达到卡玛目标。它是代价明确的取舍，不是全面胜出。</p>')
    add('recent','最近一年：绝对亏损和相对基准必须分开看',metric_table(recent)+picture('05_全期与近期','近期仍不稳定，不能被全期大涨遮住。')+
        '<p>季度验证选规则最近一年收益2.29%，同期等权买持约−31.87%，超额为+34.16个百分点；但它自身夏普约0.14、卡玛约0.33。老师强调的相对收益确实改善，用户要求的绝对风险比率仍没有满足。</p>'+picture('03_相对基准曲线','曲线=(策略资产/基准资产)−1，不是两条收益直接相减，也不等于已经实现期货对冲。'))
    add('all','全部80组退出规则：为什么没有继续挑一条好看的宣布成功',picture('04_全部80组目标分布','固定规则76组，加只用过去验证选规则4组；每一组均保留。')+
        '<p>80组里全期仅1组同时满足三项研究目标，最近一年0组、最近半年0组。这里的固定方案都是开发阶段对照，不能按全期结果挑中以后声称当时就会选它。</p>'
        '<p>19种规则包含50%/75%仓位、6%/10%跟踪止损、20/60日走势开关、上涨广度、市场状态、波动预算和8%/12%账户暂停。固定分数分别来自季度树、月度树、简单因子、树与简单因子各半。每季选规则的版本只使用此前126日验证账本，不看当季结果。</p>')
    add('exit','下跌时如何提前退出：有明确条件，不寻找未来最低点',picture('10_提前退出实例',f'图中首次触发为{first:%Y-%m-%d}；灰色区域是暂停买入状态。')+
        '<pre>如果昨天收盘资产比本轮高点低8%或12%：\n    今天开盘尝试卖出，暂停买入至少5个交易日。\n等待结束后，如果此前连续3天：\n    市场过去20日累计走势为正，而且上涨股票比例&gt;50%：\n        恢复允许买；仍等原定批次才下单。\n否则：继续持有现金。\n卖不出去的头寸继续真实记账，不假装按止损价成交。</pre>'
        '<p>隔夜跳空、停牌和再次下跌都可能使最大回撤超过触发阈值。恢复条件只能确认已经出现的改善，无法保证刚好买在最低点。</p>')
    add('loss','亏损具体在哪里：逐只股票和账本对得上',picture('09_回撤具体亏在哪','不是只统计已平仓交易；起止持仓市值、买卖金额、费用和股息全部计入。')+
        f'<p>季度验证选规则最大回撤：{a["peak_date"]}资产{a["peak_equity"]:,.2f}元，至{a["trough_date"]}降到{a["trough_equity"]:,.2f}元，减少{-a["equity_loss"]:,.2f}元。逐股贡献合计与资产减少相符，误差小于1分。</p>'
        '<pre>某股票这段盈亏 = 结束持仓市值 − 开始持仓市值\n                  + 卖出收到的钱 − 买入花的钱 + 股息 − 费用。</pre>'
        '<p>完整逐股表已放在本文件夹，文件名含drawdown_stocks；另外保留最差10个交易日及所有季度结果。市场普跌会拖累绝对收益，频繁退出和再进入也会损失收益机会，不能将全部损失简单归结为因子失效。</p>')
    add('factor','AI新因子：两轮反馈后冻结，再与原100列比较',metric_table(looprows)+picture('06_新因子增量资金','2026-01-05至2026-09-30，双方相同训练行、172轮、参数和账本。')+
        f'<p>先提6个，读取训练/验证反馈后再提4个，共10个公式。6个通过基础检查，再去重留3个。正式训练{freeze["formal_train_rows"]:,}行，最后信号2025-12-24、最后答案2025-12-31，之后冻结评分2026年。</p>'
        '<p>新增三列的后段平均排名相关分别约0.0792、0.0599、0.0553；但模型总体排名相关从0.1163变为0.1157。全仓少亏2.17个百分点，过滤后多亏0.17个百分点。结论：公式有相关性，不等于有稳定的新增交易价值。</p>'
        '<p>这是Codex提案和程序反馈构成的真实两轮协作流程，没有另外部署本地大模型，也没有谎称运行了完整RD-Agent。</p>')
    add('hmm','市场状态：把“改变因子权重”和“减少持仓”分开检验',picture('07_状态条件因子','统一权重与2/3状态条件权重，使用相同20个因子、相同交易开关。')+
        '<p>每月用此前504日成熟答案重算。状态按训练期波动从低到高排序；每天只向前更新概率。每个状态的因子相关向总体均值收缩，相当于加60天总体经验，避免少数几天造成极端权重。</p>'
        '<p>3状态权重＋状态开关全期收益95.34%、回撤19.71%、夏普1.20、卡玛2.05；统一权重＋相同开关为92.04%、20.92%、1.17、1.87。条件权重有小幅全期改善，但最近一年仍亏，近期未通过目标。</p>')
    add('minute','分钟因子：事前不选，不能看到后段好就改答案',metric_table(minrows)+picture('08_分钟信号资金','日频和分钟组都在共同可用股票上排序；等权基准是同期完整日频Top100池，覆盖略宽。')+
        f'<p>分钟模型只用2025Q4：前段3,126行训练、12月1,738行验证，选择权重0%；再用5,354行成熟答案重训。2026Q3共同可用{ma["q3"]["common_stock_rows"]:,}行，占日频池92.95%，共65日。</p>'
        '<p>正式选择0%时，收益与日频基线完全相同。固定25%是预先候选但未被选中的敏感度实验，少亏2.22个百分点；两者都没有达到绝对风险目标。Q3的排名相关顺序与Q4相反，正是时间漂移的实例。</p>')
    add('csi','中证2000对比：原始行情已补，历史成分不能用近似代替',
        '<p>指定口径已固定：训练2023-08-11—2025-06-03，隔离2025-06-04—2025-07-01，验证2025-07-02—2026-05-22，Alpha101，零成本/零滑点，保留本人Top20和五日规则。主对照在训练内部选参数、重训后冻结验证期模型。</p>'
        '<p>670天因子原件、广泛沪深北价格、官方价格和全收益指数已经保存。官方当前名单、历史存档、早期ETF清单都已调查，但没有取得覆盖整个区间的真实逐日成员和全部临时变更。今天的成员倒填过去会偷看未来；自己估算的小盘池也不能改名为真实中证2000。</p>'
        '<p><strong>因此这里没有编造该项比较收益。</strong>详细来源和缺口日期见03文件。2025年6月以后还涉及北交所成员，不能只用沪深；原件与派生表的逐市场覆盖另有审计。</p>')
    add('verify','验证和仍然存在的边界',
        '<p>80组风险实验：38,800行逐日账、215,996笔订单尝试、107,210笔成交卖出独立复算通过。另16组新模型和6个基准账户共6,746行日账、58,552笔订单通过复核；5处最大回撤区间逐股归因对平。12项自动测试通过。</p>'
        '<p>这证明账和这些时点条件自洽，不等于行情供应商历史版本无修订，也不等于实盘能够按日线开收盘价成交。北交所涨跌停、停牌和容量仍使用原研究代理；IPO、ST和退市历史资料不完整。整个2024—2026历史已被研究过，新增改动属于历史开发，不冒充真正未见过的新样本。</p>'
        '<p>本报告统一夏普无风险利率1.5%；卡玛用日历年化收益除最大回撤；超额百分点和相对财富增长分别提供。全部时段表以段首前一交易日资产作基数。</p>')
    links=[('01_任务逐项答复.md','老师与用户任务逐项答复'),('02_实验方法_按真实日期解释.md','最具体的训练、因子、买卖与指标解释'),('03_中证2000对比_真实数据缺口.md','中证2000数据来源与缺口'),('04_最新研究_实际采用了什么.md','最新研究和实际落地'),('05_分钟因子_实际训练和结果.md','分钟实验完整核查'),('全部实验_含基准超额.csv','96组账户、278条时段结果'),('全部实验_按季度.csv','逐季结果'),('../新项目_20261017/项目改版_完整说明与实验结果.html','1016会议改版完整说明')]
    add('files','文件入口', '<ul>'+''.join(f'<li><a href="{path}">{label}</a></li>' for path,label in links)+'</ul>')
    toc=''.join(f'<a href="#{id}">{title}</a>' for id,title,_ in sections)
    body=''.join(f'<section id="{id}"><h2>{title}</h2>{content}</section>' for id,title,content in sections)
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10月18日：深入实验、超额收益与任务答复</title><style>body{font-family:"Microsoft YaHei",sans-serif;color:#171717;background:white;max-width:1120px;margin:36px auto;padding:0 24px;line-height:1.8}h1{font-size:30px}h2{font-size:23px;border-bottom:1px solid #bbb;padding-bottom:8px}section{margin:38px 0}nav{border:1px solid #ccc;padding:18px}nav a{display:block}a{color:#222;text-decoration:underline}p{max-width:1000px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border:1px solid #ccc;text-align:left}th{background:#f3f3f3}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#555}pre{white-space:pre-wrap;line-height:1.8;background:#f5f5f5;padding:18px;font-family:inherit}strong{font-weight:700}@media print{nav{display:none}section{break-inside:avoid}} </style><body><h1>10月18日：深入实验、超额收益与任务答复</h1><p>实际运行结果 · 白底黑字 · 10张图 · 96组账户完整保留</p><nav>'+toc+'</nav>'+body+'</body></html>'
    (OUT/'深入实验_完整结果与解释.html').write_text(doc,encoding='utf-8')
    # Small navigable HTML companions for easier reading of the detailed text.
    for p in OUT.glob('0*.md'):
        rendered=markdown.markdown(p.read_text(encoding='utf-8'),extensions=['tables','fenced_code'])
        (p.with_suffix('.html')).write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1050px;margin:40px auto;font:16px/1.9 Microsoft YaHei}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}pre{white-space:pre-wrap;background:#eee;padding:16px}img{max-width:100%}</style>'+rendered,encoding='utf-8')
    manifest={'report':'深入实验_完整结果与解释.html','account_experiments':96,'period_rows':len(evidence),
              'charts':10,'csi_comparison_complete':False,'stable_recent_targets_met':False,
              'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'evidence/all_periods_with_excess.csv',RUN/'deep_risk/independent_audit.json',RUN/'evidence/supplemental_audit.json',RUN/'factor_loop/frozen_selection.json']}}
    (OUT/'交付核查.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPORT_COMPLETE',str(OUT),'charts',len(list(FIG.glob('*.png'))))

if __name__=='__main__':main()
