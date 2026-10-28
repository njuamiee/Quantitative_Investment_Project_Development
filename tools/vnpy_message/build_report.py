"""Portable Chinese acceptance report and labelled research/simulation charts."""
from pathlib import Path
import base64
import html
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'training_runs/vnpy_message'
OUT = ROOT/'vnpy消息驱动迁移_20261028'


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    rows=[]
    fig, axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    cases={'simple':['simple_0bp','simple_5bp_v3','simple_15bp'],
           'full':['full_0bp','full_5bp_v2','full_15bp']}
    for ax,(kind,folders) in zip(axes,cases.items()):
        for folder in folders:
            report=json.loads((RUN/folder/'report.json').read_text(encoding='utf-8'))
            frame=pd.read_csv(RUN/folder/'equity.csv')
            slip=report['costs']['slippage_bps']
            ax.plot(pd.to_datetime(frame['at'],utc=True),frame.equity/1e6,label=f'滑点 {slip:g} bp')
            rows.append(dict(策略='简单版' if kind=='simple' else '完整信号版',滑点bp=slip,
                收益百分比=round(report['metrics']['return_pct'],4),
                最大回撤百分比=round(report['metrics']['max_drawdown_pct'],4),
                行情消息=report['messages'],委托=report['orders'],成交回报=report['fills'],
                现金核对误差=report['cash_identity_error']))
            shutil.copy2(RUN/folder/'equity.csv',OUT/(folder+'_资金曲线.csv'))
        ax.set_title(('简单单股策略' if kind=='simple' else '完整信号与风控策略')+'：分钟代理撮合，非实盘')
        ax.set_ylabel('净值（初始=1）');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('2026年第三季度：滑点敏感性验收\n真实分钟数据 + 人工盘口；策略未经过收益优化',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'分钟回放资金曲线.png',dpi=150);plt.close(fig)
    native=[]
    for p in sorted((RUN/'native').glob('*/report.json')):
        r=json.loads(p.read_text(encoding='utf-8'));native.append(r)
        shutil.copy2(p,OUT/(p.parent.name+'_原生报告.json'))
    if len(native)!=4:raise RuntimeError('native acceptance not complete')
    nr=pd.DataFrame([dict(策略=r['kind'],延迟毫秒=r['latency_ms'],滑点元=r['engine_slippage_absolute'],
         股票数=r['symbols'],Tick数=r['input_ticks'],委托数=r['native_orders'],成交回报数=r['native_trades'],
         收益百分比=round(r['audited_metrics']['return_pct'],4),
         日终最大回撤百分比=round(r['audited_metrics']['max_drawdown_pct'],4),
         现金复算误差=r['independent_cash_error']) for r in native])
    models=json.loads((RUN/'model_bundles/verification.json').read_text())
    shutil.copy2(RUN/'model_bundles/verification.json',OUT/'八类模型导出验证.json')
    shutil.copy2(RUN/'native_acceptance.json',OUT/'原生验收记录.json')
    shutil.copy2(RUN/'native_unit_tests.json',OUT/'Linux自动化测试.json')
    for name in ('audit.json','sha256.json','actions.json'):
        shutil.copy2(RUN/'native_inputs'/name,OUT/('输入_'+name))
    for name in ('local_tests.xml','adapter_tests.xml'):
        shutil.copy2(RUN/name,OUT/name)
    fig,ax=plt.subplots(figsize=(12,5))
    labels={'simple_100ms_0.01':'简单版：100ms，0.01元',
            'simple_1000ms_0.01':'简单版：1000ms，0.01元（与100ms重合）',
            'simple_100ms_0.05':'简单版：100ms，0.05元',
            'full_100ms_0.01':'完整信号：100ms，0.01元'}
    for p in sorted((RUN/'native').glob('*/audited_equity.csv')):
        f=pd.read_csv(p)
        if len(f)!=5 or f.audited_equity.isna().any():raise ValueError('invalid reconstructed curve')
        shutil.copy2(p,OUT/(p.parent.name+'_复算资金表.csv'))
        for source,suffix in [('native_equity.csv','原生资金表.csv'),('fills.csv','成交明细.csv'),('orders.csv','委托明细.csv')]:
            shutil.copy2(p.parent/source,OUT/(p.parent.name+'_'+suffix))
        ax.plot(['起始']+list(f.date.str[5:]),[1.]+list(f.audited_equity/1e6),
                marker='o',markersize=4,linestyle='--' if '1000ms' in p.parent.name else '-',
                label=labels[p.parent.name])
    ax.set(title='DeltaOne 原生成交逐笔复算：五个交易日（短期技术验收）',xlabel='交易日',ylabel='净值（引擎费用 + 应收税前股息）')
    ax.grid(alpha=.2)
    if ax.lines:ax.legend()
    fig.tight_layout();fig.savefig(OUT/'原生Tick资金曲线.png',dpi=150);plt.close(fig)
    def img(name):
        return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"/>'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>vn.py消息驱动迁移验收</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.8 'Microsoft YaHei',sans-serif;color:#203047;background:#f7f9fc}}h1,h2{{color:#12325b}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{padding:8px;border:1px solid #d8dfe8}}img{{width:100%}}.status{{padding:20px;background:#e9f3ff;border-left:5px solid #3979b7}}</style>
<h1>从批量研究到消息驱动股票策略</h1><p>独立分支 codex/vnpy-message-driven · 验收日期 2026-10-28</p>
<div class="status">已完成：简单策略、完整信号执行、真实分钟回放、已安装 DeltaOne 原生 Tick 回放、逐笔独立资金复算、模型文件导出及 Linux 测试。<br>
本阶段按用户要求只做模拟交易。未连接券商，未发送真实订单；尚未确认实时行情订阅。</div>
<h2>普通读者可以怎样理解</h2><p>研究程序主要按历史日期批量训练和评估。新的执行程序像值班交易员：每次只收到一条已经发生的消息，先判断是否可以下单，再等待成交回报。挂单可能只成交一部分，也可能完全成交不了。股票、现金和费用随实际回报逐笔更新。</p>
<h2>先简单，再接完整信号</h2><p>简单版用单只股票的已结束分钟观察值计算3/8均线，最多使用10%资金。完整信号版接入 Ridge 等模型的横截面评分，以及股票池、HMM仓位上限、选股数量、分组预算和风控。此次固定配置用于检验工程流程，不表示获得了更高收益。</p>
<h2>分钟回放：成本增加会怎样</h2>{pd.DataFrame(rows).to_html(index=False,border=0)}{img('分钟回放资金曲线.png')}
<p>区间：2026-07-01至09-30，共65个交易日。分钟完成后才可用，人工构造买卖价差。不能把它称为真实盘口撮合。完整5bp配置隔离52,992条全空分钟和150条成交额/量不一致记录。这组旧研究对照沿用“除息日税前分红立即可用”的假设；下方原生验收已改成仅记应收、不增加可买现金。</p>
<h2>真正的商业版原生引擎验收</h2>{nr.to_html(index=False,border=0)}{img('原生Tick资金曲线.png')}
<p>区间：9月23、24、28、29、30日。接入原始五档盘口和涨跌停价，使用已安装的BacktestingEngine执行委托。L1快照不代表逐笔排队结果；原生绝对价格滑点与上面的bp滑点不是同一个单位。短期收益不是策略有效性的证据。</p>
<p><strong>原生多股统计表不可直接使用：</strong>本样本只返回两个日期且净值为空。上图从43条成交回报和各交易日日终报价独立重建，覆盖全部五个交易日，现金与执行账本核对通过。原始异常表保留作诊断，未用填零掩盖问题。单股原生价格收益表与独立重建的同口径曲线一致。</p>
<p>图中每边费率为0.05%，每股成交另计0.01元或0.05元滑点；原生滑点在收益核算时扣除，不修改成交回报价。执行账户另采用最低佣金和卖出税费，复算CSV分别列示两种费用口径。股息按税前应收款计入净值，不作为可用现金；到账日和税款仍未核实。100ms与1000ms在此样本结果重合，不表示更大延迟始终无影响。五天数据不用于年化夏普或卡玛结论。</p>
<h2>实际解决的平台问题</h2><ol><li>quant登录后切换vnpy用户，运行原有Python3.13商业版。</li><li>补充BSE股票分类；保持交易所代码真实。</li><li>绕过当前缺失BSE概览的历史库，建立独立且可核对来源的回放输入。</li><li>独立安装Arrow，提供压缩CSV兼容入口。</li><li>隔离pandas2.3.3，避开商业版报表在pandas3链式赋值下的兼容问题。</li></ol>
<h2>模型和前视检查</h2><p>8个模型家族分别用真实项目数据完成训练、保存、加载和预测对比：每类12,590条训练样本、126个交易日、20个因子，100条预测的重新加载误差均为0。训练截止固定为2026-06-30；未来收益没有提前进入训练。历史因子发布时间仍按既有研究假设处理，模型标记为研究用途。</p>
<p>Windows核心/数据/模型/发布/账本测试36项通过，8个模型家族子测试通过；适配器9项通过。Linux原生环境35项核心、适配器及账本测试通过。还测试了改变未来输入不影响此前行为、T+1、重复回报、部分成交、同步撤单回报和资金冻结。这些测试覆盖已列出的前视防护，不代表所有历史因子发布时间已被供应商验证。</p>
<h2>如何重跑和移植</h2><p>代码位于quant_message，部署工具位于tools/vnpy_message。在已准备依赖和样本的服务器上，运行acceptance.py可一键上传分支代码，切换vnpy身份，执行测试及四组原生模拟，取回报告。README提供本地与Linux命令。未来行情适配器输出统一消息后可以复用策略和风控；真实柜台适配、账户恢复和实盘联调尚未验收，因此目前不能承诺一键真实交易。</p>
<h2>下一步：实时模拟所需的行情</h2><p>实时行情是交易时间持续送来的新价格和盘口，接收行情无需真实下单。建议优先核实现有供应商是否支持北交所Level-1和Linux接入；确认前继续历史回放。安装了RQData等插件，不等于账号已经开通北交所实时推送。</p>
<p>可向供应方说明：目前只做模拟交易，需要Linux可接收的北交所股票实时Level-1，包含价格、盘口、成交量和涨跌停价，优先复用现有服务。</p>
<p>参考：<a href="https://www.vnpy.com/docs/cn/community/info/datafeed.html">vn.py官方数据服务</a>。详细执行命令、数据契约和限制见 quant_message/README.md。</p></html>'''
    (OUT/'迁移验收报告.html').write_text(body,encoding='utf-8')
    (OUT/'验收摘要.json').write_text(json.dumps(dict(minute=rows,native=[{k:v for k,v in r.items() if k!='native_statistics'} for r in native],model_families=len(models),real_orders_sent=0,realtime_feed_connected=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT/'迁移验收报告.html')


if __name__=='__main__':main()
