"""Compare read-only snapshots and publish scope-specific missing-file lists."""
from pathlib import Path
import json, collections
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'学习笔记/服务器数据核查_20261002';OUT.mkdir(parents=True,exist_ok=True)
FOLDER=ROOT/'training_data/audit_20261002'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
new=read(FOLDER/'inventory.json');old=read(ROOT/'training_data/audit_20261001/inventory.json')
extra=read(FOLDER/'supplementary_audit.json')
def missing(s):
    paths=[]
    for name,years in s['minute'].items():
        for info in years.values():
            paths.extend(f'/home/quant/quantDataProcess/backend/data/{name}/{d.replace("-","")}.parquet' for d in info['missing'])
    for market,years in s['tick'].items():
        for year,info in years.items():
            paths.extend(f'/home/quant/tick-All/{market}/{year}/{d.replace("-","")}_Tick1.tar.gz' for d in info['missing'])
    return sorted(set(paths))
def factor_missing(s):
    return sorted(f'/home/quant/factors/kyzqsh11/{t["table"]}/2026/{d.replace("-","")}.zip'
        for t in s['factors'] for d in t['years']['2026']['missing'])
stamp=lambda s:datetime.fromisoformat(s).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
market=missing(new);before=missing(old);factors=factor_missing(new);oldfactors=factor_missing(old)
S={'audit_start_beijing':stamp(new['started_utc']),'audit_end_beijing':stamp(new['ended_utc']),
   'supplement_end_beijing':stamp(extra['ended_utc']),'cutoff':'2026-09-30','compare_to':'2026-10-01',
   'factor_tables':len(new['factors']),'factor_latest_counts':dict(collections.Counter(t['years']['2026']['last'] for t in new['factors'])),
   'factor_missing_paths_2026':len(factors),'factor_arrivals_since_previous':sorted(set(oldfactors)-set(factors)),
   'market_missing_paths':len(market),'filled_previous_market_paths':sorted(set(before)-set(market)),
   'new_market_missing_paths':sorted(set(market)-set(before)),
   'minute_missing':{name:sorted(d for y in years.values() for d in y['missing']) for name,years in new['minute'].items()},
   'minute_file_counts':{name:sum(y['files'] for y in years.values()) for name,years in new['minute'].items()},
   'raw_tick_missing':{m:{y:i['missing'] for y,i in years.items() if i['missing']} for m,years in new['tick'].items()},
   'daily':{a:{'files':d['files'],'last_counts':d['last_counts']} for a,d in new['daily'].items()},
   'benchmark_latest':{code:d['last'] for code,d in new['benchmarks'].items()},
   'service_access':'Pinned-host authentication/list/read succeeded; scheduler and task logs not inspected',
   'minute_sample_null_fraction':extra['minute_sample']['null_counts']['close']/extra['minute_sample']['rows'],
   'service_readme_sha256':extra['service_readme']['sha256'],
   'scope':'Read-only inventory, file-end checks and selected contents; no model retraining.'}
(FOLDER/'comparison.json').write_text(json.dumps(S,ensure_ascii=False,indent=2),encoding='utf-8')
for name,paths,note in [
    ('缺失行情文件_逐路径.txt',market,'指定目录中的行情日期缺口；不代表北交所分钟实验全都必需。'),
    ('本次已到位因子文件_逐路径.txt',S['factor_arrivals_since_previous'],'10月1日未见、10月2日已见的9月30日因子文件。文件到位不等于逐字段全部验收。'),
    ('因子2026待补文件_逐路径.txt',factors,'按服务器参考交易日期检查；2026年至9月30日目录日期缺口。更早年份和内容质量另论。')]:
    (OUT/name).write_text(f'核查结束：{S["audit_end_beijing"]} 北京时间。\n{note}\n共 {len(paths)} 条。\n\n'+'\n'.join(paths)+'\n',encoding='utf-8')
assert len(market)==39 and len(S['factor_arrivals_since_previous'])==120 and not factors
note=f'''# 服务器更新与实验所需补齐资料：2026-10-02

北京时间 {S['audit_start_beijing']}—{S['audit_end_beijing']} 全量日期核查；补充抽查结束 {S['supplement_end_beijing']}。截止交易日期为2026-09-30。仅只读检查，没有重训、修改服务器或触发定时任务。

## 与10月1日相比

| 资料 | 10月2日实际检查结果 | 变化 |
|---|---|---|
| 120类因子表 | 2026各181个日期文件，均到9/30；当年参考日期无缺口 | 9/30共120个ZIP已经到位 |
| 北交所新日线raw/qfq | 各350文件，348到9/30；另外2个文件较早结束并带退市标记 | 未变 |
| 四个基准文件 | 到9/30；北证50仍缺2022年13个日期 | 未变 |
| 北交所分钟和清洗tick | 各1181文件，最新9/24，各缺5个日期 | 未变 |
| 三市场原始tick | 最新9/24，共29个缺失包 | 未变 |
| 旧日线3种价格×3只抽查 | 各到8/21；非旧目录逐文件内容验收 | 未变 |

39条行情路径缺口与10/1一致，未发现旧缺口补齐。因子文件的到达是独立变化，不能继续把120个9/30因子文件列为待补。

## 服务与内容

SFTP身份校验、登录、列目录和读取成功。没有检查后台进程、真实调度配置和任务日志，不能据此断言所有更新服务正常。README仍写默认日线18:00、分钟18:45，分钟扫描仅处理2026及以后；2025-06-30要安排历史核验/补算。

最新分钟样本仍为20260924.parquet：84458行、349个证券（含2个指数），每个主要量价列25557空值，约30.26%。空值不代表可成交。

五类最新因子已抽查9/30文件：c01共5399行、c02共5363行，代码均被格式规则识别；001共5270行，其中5个代码例外，内部enddate仍对应9/14。不能把“文件名到9/30”当作所有字段都代表9/30已发布信息。

SZ与BJ的20260105原始包大小及提供的MD5文本相同，身份疑点仍待解包核验。

## 按实验目标补资料

1. 复现9/30已交付研究：本地固定输入已具备；2025Q4的60个分钟日文件也齐全。
2. 提高真实收益可信度：优先核实历史股票资格、真实公司行为台账、因子可用/修订时间、交易限制和真实费用。尚未确认完整资料，不等于已经搜遍所有目录后断言绝对没有。
3. 北交所分钟覆盖扩至9/29：补bar1m/BJ下20250630、20260518、20260928、20260929四个parquet。扩至9/30再加20260930.parquet。20250630原始BJ包可见，须先核验再加工；另四个2026原始BJ包未见，采用原加工路线时需补来源。
4. 延长至9/30的主模型评分：日线和c01/c02的9/30文件已到，接下来是下载到新版本、验内容、核对历史可用时点和运行；不再属于这两个因子文件未到位。此次未重训，旧成绩仍截至9/29。
5. 其他25个沪深原始tick包、5个清洗tick、北证50的2022年13个日期，只在相关实验范围内才是直接前置需求。

10月1—7日休市、10月8日恢复开市，不能把假期没有新交易数据列为缺失。[交易所公告](https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml)

[39条行情缺口](缺失行情文件_逐路径.txt)；[本次已到位的120个因子文件](本次已到位因子文件_逐路径.txt)。原始证据在training_data/audit_20261002/。10月1日旧记录原样保留。
'''
(OUT/'服务器更新与实验所需补齐资料.md').write_text(note,encoding='utf-8')
print(json.dumps({k:v for k,v in S.items() if k not in ('factor_arrivals_since_previous','raw_tick_missing')},ensure_ascii=False,indent=2))
