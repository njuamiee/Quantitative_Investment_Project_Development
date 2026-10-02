"""Summarize the October 1 read-only snapshot without changing prior evidence."""
from pathlib import Path
import json,collections
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'学习笔记/服务器数据核查_20261001';out.mkdir(parents=True,exist_ok=True)
folder=ROOT/'training_data/audit_20261001'
new=json.loads((folder/'inventory.json').read_text(encoding='utf-8'))
old=json.loads((ROOT/'training_data/audit_20260930/inventory.json').read_text(encoding='utf-8'))
extra=json.loads((folder/'supplementary_audit.json').read_text(encoding='utf-8'))
assert new.get('ended_utc') and max(new['reference_calendar'])=='2026-09-30'

def missing(s):
    paths=[]
    for name,years in s['minute'].items():
        for info in years.values():
            paths.extend(f'/home/quant/quantDataProcess/backend/data/{name}/{d.replace("-","")}.parquet' for d in info['missing'])
    for market,years in s['tick'].items():
        for year,info in years.items():
            paths.extend(f'/home/quant/tick-All/{market}/{year}/{d.replace("-","")}_Tick1.tar.gz' for d in info['missing'])
    return sorted(set(paths))

market=missing(new);before=missing(old)
factor=[]
for table in new['factors']:
    factor.extend(f'/home/quant/factors/kyzqsh11/{table["table"]}/2026/{d.replace("-","")}.zip' for d in table['years'].get('2026',{}).get('missing',[]))
tz=timezone(timedelta(hours=8))
stamp=lambda s:datetime.fromisoformat(s).astimezone(tz).strftime('%Y-%m-%d %H:%M:%S')
summary={'audit_start_beijing':stamp(new['started_utc']),'audit_end_beijing':stamp(new['ended_utc']),
    'supplement_end_beijing':stamp(extra['ended_utc']),'cutoff':'2026-09-30',
    'reference_dates_2026':sum(d.startswith('2026') for d in new['reference_calendar']),
    'factor_tables':len(new['factors']),'factor_missing_paths':len(factor),
    'factor_latest_counts':dict(collections.Counter(t['years']['2026']['last'] for t in new['factors'])),
    'market_missing_paths':len(market),'filled_previous_market_paths':sorted(set(before)-set(market)),
    'new_market_missing_paths':sorted(set(market)-set(before)),
    'minute_missing':{name:sorted(d for y in years.values() for d in y['missing']) for name,years in new['minute'].items()},
    'minute_file_counts':{name:sum(y['files'] for y in years.values()) for name,years in new['minute'].items()},
    'raw_tick_missing':{m:{y:i['missing'] for y,i in years.items() if i['missing']} for m,years in new['tick'].items()},
    'daily':{a:{'files':d['files'],'last_counts':d['last_counts']} for a,d in new['daily'].items()},
    'benchmark_latest':{code:d['last'] for code,d in new['benchmarks'].items()},
    'benchmark_bj50_historical_missing':new['benchmarks']['899050.BJ']['missing'],
    'service_access':'SFTP identity checked, authentication/list/read succeeded; no daemon or scheduler health check',
    'service_readme_sha256':extra['service_readme']['sha256'],
    'minute_sample_null_fraction':extra['minute_sample']['null_counts']['close']/extra['minute_sample']['rows'],
    'old_daily_sample_end':'2026-08-21',
    'scope':'Date-path scan, all new BJ daily file ends, selected content samples. No new model training.'}
(folder/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
for name,paths,note in [('缺失行情文件_逐路径.txt',market,'39 个指定目录行情路径，不代表本次四季度分钟实验全部必需。'),
    ('因子20260930待补文件_逐路径.txt',factor,'各表 20260930.zip 暂未到位；不包括国庆休市日。'),
    ('本次新增行情缺口_逐路径.txt',summary['new_market_missing_paths'],'检查截止日期延长，新增 9 月 30 日路径；不表示旧文件被删除。')]:
    (out/name).write_text(f'核查结束：{summary["audit_end_beijing"]} 北京时间。\n{note}\n共 {len(paths)} 条。\n\n'+'\n'.join(paths)+'\n',encoding='utf-8')
text=f'''# 服务器核查：2026-10-01

核查 {summary['audit_start_beijing']}—{summary['audit_end_beijing']}，补充抽查结束 {summary['supplement_end_beijing']}，均为北京时间。只读检查至 2026-09-30，未重训模型。

当前能确认 SFTP 可登录、列目录和读文件，服务器身份校验通过。没有检查后台进程、实际调度配置或任务日志，因此不能把“能登录”当作“所有更新任务正常”。

| 内容 | 9 月 30 日核查 | 10 月 1 日核查 |
|---|---|---|
| 北交所新日线 raw/qfq | 各 350 文件，348 到 9/29 | 各 350 文件，348 到 9/30；两个较早结束文件名带退市标记 |
| 四个检查过的基准文件 | 到 9/29 | 到 9/30；北证50仍缺2022年13个日期 |
| 120 类因子表 | 2026各180文件，到9/29 | 仍到9/29；参考日期增至181，9/30共120个ZIP未到位 |
| 北交所分钟/清洗tick | 各1181文件，到9/24 | 未改变；各缺5个日期 |
| 三市场原始tick | 最新9/24，缺26包 | 仍最新9/24，缺29包 |
| 指定行情日期缺口 | 34路径 | 39路径；新增5条均来自9/30，旧缺口未补 |
| 旧日线抽查 | 每种价格3只到8/21 | 仍到8/21；不是全目录逐股检查 |

10月1—7日休市，10月8日起正常开市，不能把假期没有新行情当作缺失。[交易所公告](https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml)

北交所分钟和清洗tick均缺：2025-06-30、2026-05-18、2026-09-28、2026-09-29、2026-09-30。6月30日原始BJ包可见，可先核验再加工；其余四个日期原始BJ包也未见。

服务说明仍写默认日线18:00、分钟18:45，并说明分钟扫描只处理2026及以后年份；这不是实际运行成功证明。2025-06-30历史缓存不能假定会被当前默认任务自动补出。未写服务器、未触发更新。

最新分钟样本仍为20260924.parquet，共84458行、349个证券；每个主要量价列25557空值，约30.26%。最新五类因子内容抽查仍是9/29样本；001表5个代码例外及内部日期9/14问题仍在。SZ与BJ的20260105原始包大小和提供的MD5文本仍相同，疑似身份问题未排除，尚未本次全面解包确认。

## 真正需要补什么

复现9月30日已完成的研究：本地所需输入已有，不必等待所有服务器路径补齐。2025Q4分钟压力回放所用60个日期也已经齐全。

提高原实验收益可信度：首先需要经核实的历史股票资格、真实分红送转台账、历史可用时间、交易规则与费用。当前未确认这些专用资料完整可用；不能用猜出来的文件替代，也不能断言服务器任何位置都没有。

把北交所分钟回放覆盖扩到9月29日：优先补4个缺失bar1m文件（上述日期除9月30日）；若延至9月30日，才需要第5个。采用原始包加工的现有路线时，另需相应缺失BJ原始来源。清洗tick和沪深历史tick是其他深度/市场实验的要求，不全是本次北交所分钟实验前置条件。

若要增加9月30日同日因子评分，需要120类中所用表的9月30日文件先到位；全库完整性清单列120个，当前主模型只直接使用其中c01/c02的Alpha数据。旧日线8/21后的沪深行情仅在扩展近期沪深实验时必需。

[39条行情路径](缺失行情文件_逐路径.txt)；[120条因子路径](因子20260930待补文件_逐路径.txt)。原始证据：training_data/audit_20261001/inventory.json、supplementary_audit.json、comparison.json。9月30日实验成绩仍固定在原run，不把新文件到达当作已重训。
'''
(out/'服务器更新与实验所需补齐资料.md').write_text(text,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
