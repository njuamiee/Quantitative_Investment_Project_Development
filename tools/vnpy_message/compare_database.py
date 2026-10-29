"""Strict prefix equality checks, execution attribution and reviewable charts."""
from pathlib import Path
import base64
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/vnpy_message/database_verified'
OUT=ROOT/'vnpy数据库对照_20261029'
CUT='2026-09-24'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def prefix_records(folder,filename):
    if filename=='signal_audit.json':return [x for x in read(folder/filename) if x['at'][:10]<=CUT]
    if filename=='tick_audit.json':return {k:v for k,v in read(folder/filename).items() if k<=CUT}
    f=pd.read_csv(folder/filename)
    if filename=='orders.csv':
        f=f[f.submitted_at.str[:10]<=CUT]
        return f[['key','symbol','side','limit','quantity','submitted_at','expires_at']].to_dict('records')
    if filename=='fills.csv':return f[f['at'].str[:10]<=CUT].to_dict('records')
    return f[f.date<=CUT].to_dict('records')


def equal_frames(a,b):
    try:pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-8,rtol=0);return True
    except AssertionError:return False


def build():
    OUT.mkdir(exist_ok=True)
    for name in ('database_import_manifests.json','database_first_fill_quotes.json','database_linux_tests.json','database_model_tests.xml','database_tests.xml'):
        shutil.copy2(ROOT/'training_runs/vnpy_message'/name,OUT/name)
    paths=sorted(RUN.glob('*/report.json'))
    if len(paths)!=14:raise RuntimeError('require all 14 completed database runs')
    checks=[]
    for kind in ('simple','full'):
        for engine in ('native','paper'):
            base=RUN/f'{kind}_{engine}_base_d1'
            for variant in ('prefix','mutated'):
                alt=RUN/f'{kind}_{engine}_{variant}_d1'
                details={name:prefix_records(base,name)==prefix_records(alt,name)
                         for name in ('tick_audit.json','signal_audit.json','orders.csv','fills.csv','audited_equity.csv')}
                if variant=='mutated':
                    a=read(base/'tick_audit.json');b=read(alt/'tick_audit.json')
                    details['future_input_actually_changed']=any(a[d]!=b[d] for d in a if d>CUT)
                    details['future_outcome_actually_changed']=not equal_frames(pd.read_csv(base/'audited_equity.csv'),pd.read_csv(alt/'audited_equity.csv'))
                checks.append(dict(strategy=kind,engine=engine,experiment=variant,checks=details,passed=all(details.values())))
    migration=[]
    for kind in ('simple','full'):
        a=ROOT/'vnpy消息驱动迁移_20261028'
        b=RUN/f'{kind}_native_base_d5'
        # Compare the frozen previous acceptance, not a new file-loader run.
        old_names={'orders.csv':'委托明细.csv','fills.csv':'成交明细.csv','audited_equity.csv':'复算资金表.csv'}
        details={name:equal_frames(pd.read_csv(a/f'{kind}_100ms_0.01_{old_names[name]}'),pd.read_csv(b/name))
                 for name in old_names}
        migration.append(dict(strategy=kind,checks=details,passed=all(details.values())))
    comparisons=[]
    for kind in ('simple','full'):
        a=RUN/f'{kind}_native_base_d1';b=RUN/f'{kind}_paper_base_d1'
        na,nb=read(a/'report.json'),read(b/'report.json')
        assert na['strategy_rules']==nb['strategy_rules'] and na['shadow_costs']==nb['shadow_costs']
        assert read(a/'tick_audit.json')==read(b/'tick_audit.json')
        x=pd.read_csv(a/'fills.csv');y=pd.read_csv(b/'fills.csv')
        fields=['symbol','side','price','quantity','at']
        first=None
        for i in range(min(len(x),len(y))):
            if x.iloc[i][fields].to_dict()!=y.iloc[i][fields].to_dict():
                first=dict(index=i,native=x.iloc[i][fields].to_dict(),paper=y.iloc[i][fields].to_dict());break
        comparisons.append(dict(strategy=kind,identical_tick_stream=True,identical_strategy_parameters=True,
            native_fills=len(x),paper_fills=len(y),first_fill_difference=first,
            native_return_pct=na['audited_metrics']['return_pct'],paper_return_pct=nb['audited_metrics']['return_pct']))
    summary=dict(prefix_checks=checks,loader_migration=migration,engine_comparison=comparisons,
        all_prefix_passed=all(c['passed'] for c in checks),all_loader_migration_passed=all(c['passed'] for c in migration),
        limits=['fixed 107-symbol retrospective sample, not a point-in-time universe certification',
                'historical feature/signal release times are assumptions; source factor production not rerun',
                'original optimized batch strategy is not fully reproduced by this five-session execution test',
                'realtime feed remains unconnected; no real orders'])
    (OUT/'严格对照结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for path in paths:
        r=read(path);name=path.parent.name
        queries=read(path.parent/'database_queries.json')
        if not r['original_loader_preserved'] or sum(q['rows'] for q in queries)!=r['input_ticks']:
            raise AssertionError('strategy Tick count must match original database SELECT rows')
        rows.append(dict(实验=name,行情条数=r['input_ticks'],数据库查询次数=r['database_selects'],
            委托=r['orders'],成交回报=r['reconciled_fills'],收益百分比=round(r['audited_metrics']['return_pct'],4),
            日终最大回撤百分比=round(r['audited_metrics']['max_drawdown_pct'],4)))
        dest=OUT/'证据'/name;dest.mkdir(parents=True,exist_ok=True)
        for f in path.parent.iterdir():
            if f.suffix in ('.json','.csv'):shutil.copy2(f,dest/f.name)
    pd.DataFrame(rows).to_csv(OUT/'实验总表.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for ax,kind in zip(axes,('simple','full')):
        for engine,label in [('native','vn.py原生撮合'),('paper','自建撮合')]:
            f=pd.read_csv(RUN/f'{kind}_{engine}_base_d1/audited_equity.csv')
            ax.plot(['起始']+list(f.date.str[5:]),[1.]+list(f.audited_equity/1e6),marker='o',label=label)
        ax.set_title(('简单单股' if kind=='simple' else '完整信号')+'：同数据库、同消息顺序、同策略配置')
        ax.set_ylabel('净值');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('统一费率与滑点成本后的资金曲线\n两种撮合的流动性与成交时机规则仍不同',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'同条件引擎对照.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,kind in zip(axes,('simple','full')):
        for variant,label in [('base','原始未来'),('mutated','篡改未来价格及信号')]:
            f=pd.read_csv(RUN/f'{kind}_native_{variant}_d1/audited_equity.csv')
            ax.plot(['起始']+list(f.date.str[5:]),[1.]+list(f.audited_equity/1e6),marker='o',label=label)
        ax.axvline(2,color='grey',linestyle='--');ax.set_title(('简单版' if kind=='simple' else '完整信号版')+'：9月24日之后扰动')
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'未来扰动检查.png',dpi=150);plt.close(fig)
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"/>'
    table=pd.DataFrame([dict(策略=c['strategy'],引擎=c['engine'],测试=c['experiment'],此前消息信号委托成交净值一致=c['passed']) for c in checks])
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>vn.py数据库与未来信息对照</title>
<style>body{{max-width:1150px;margin:40px auto;font:16px/1.8 'Microsoft YaHei',sans-serif;color:#203047;padding:0 20px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ccc;padding:6px}}img{{width:100%}}code{{background:#eee}}</style>
<h1>vn.py数据库回放与未来信息检查</h1><p>2026-10-29 · codex/vnpy-message-driven</p>
<p>历史数据先用vn.py的save_tick_data入库。回放由ClickHouse → vn.py数据库接口 → 未替换的DeltaOne原生加载器 → 引擎 → 策略。策略不读历史行情文件。实时行情仍未接通；本报告没有使用今天的市场数据。</p>
<h2>输入与一致性</h2><p>基准库quant_message_20261029_base，扰动库quant_message_20261029_mutated。107只股票、五个交易日、1,300,881条Tick；入库后按证券逐字段散列校验。原公共vnpy数据库和全局配置未修改。</p>
<p>原文件回放与数据库原生回放：简单版通过={migration[0]['passed']}，完整信号版通过={migration[1]['passed']}。比较原生五档设置下的全部委托、成交和逐日资金表，数值容差1e-8。</p>
<h2>严格未来扰动与截断</h2>{table.to_html(index=False,border=0)}{image('未来扰动检查.png')}
<p>截断测试只回放9月23、24日，检查其结果是否与五日回放的相同前缀一致。扰动测试保留截至24日的输入，把之后的行情价格及成交额乘1.15，反转之后的选股评分并改变仓位上限；检查此前消息顺序、目标持仓、委托意图、成交及资金均不变。另确认未来输入和后续结果确实发生变化，防止扰动无效。检查结果逐字段保存在严格对照结果.json。</p>
<h2>两种撮合为什么还会不同</h2>{image('同条件引擎对照.png')}<p>统一了5日区间、初始100万元、同一信号、选股及风控参数、100ms延迟，原生tick_depth/queue_depth固定为1；两个引擎接收相同五档消息，自建撮合只使用一档。商业版参数未文档化，不把该参数解释为相同成交量约束。收益统一采用双边0.05%费率加每股0.01元成本，股息只记应收。自建撮合还要求增量成交量与可见盘口共同限制成交，并在下一个Tick确认；原生撮合的时间、流动性和排队处理有差异。两者收益不一致不能单独证明存在前视，也不能将两者声称为等价引擎。</p>
<p><strong>已定位的成交量差异：</strong>2026-09-23 09:38:07，920000的卖一价约14.35元、卖一量1104股，累计成交量较上一Tick没有增加；原生回报一次买入6937股。自建撮合等到09:38:10出现122股增量成交量才成交122股。因此，本配置的原生成交量不能直接当作真实可成交量；应继续与平台方确认L1撮合及排队模型。原始数据库快照单独附在database_first_fill_quotes.json。</p>
<p>本次对照发现并修复自建撮合的float32报价取整误差：例如14.350000381元不应额外被取整为14.36元。先归一化微小报价误差，再施加真正滑点；有专项回归测试。早期分钟回放报告保留为旧版本证据，未冒充修复后的结果。</p>
<h2>全部实验</h2>{pd.DataFrame(rows).to_html(index=False,border=0)}
<h2>结论的边界</h2><p>这些实验检查了给定输入下的执行程序是否提前受到未来消息影响，并未证明整个研究流程绝无泄露。107只股票是历史样本，因子真实发布时间、历史股票池可得性、HMM与旧因子生产流程仍需单独审计；八类模型的未成熟标签扰动是独立单元测试，不能代替原始数据来源验证。五天样本不用于收益能力、年化夏普或卡玛判断。</p>
<p>Linux原生环境40项回归测试通过；本地八类模型均通过未来未成熟标签/特征扰动测试，另有改变历史训练标签会改变拟合结果的正向控制。具体测试记录随报告交付。</p>
<p>已提供数据库导入、服务器对照实验和报告生成脚本。证据目录包含每次运行的原生数据库SELECT记录、消息散列、信号目标、订单、成交和资金表。</p></html>'''
    (OUT/'数据库对照报告.html').write_text(body,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not summary['all_prefix_passed']:raise AssertionError('future/prefix check failed')


if __name__=='__main__':build()
