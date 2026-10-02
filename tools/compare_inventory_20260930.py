"""比较9月28/30日只读快照，区分历史补齐与新增交易日，生成学习笔记附件。"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'training_data/audit_20260930'
new = json.loads((source / 'inventory.json').read_text(encoding='utf-8'))
old = json.loads((ROOT / 'training_data/audit_20260928/inventory.json').read_text(encoding='utf-8'))
assert new.get('ended_utc'), '本次扫描尚未完成'
out = ROOT / '学习笔记/服务器数据核查_20260930'
out.mkdir(parents=True, exist_ok=True)
old_cal = {d for d in old['reference_calendar'] if d.startswith('2026')}
new_cal = {d for d in new['reference_calendar'] if d.startswith('2026')}
assert old_cal <= new_cal, '参考日历发生历史删减，不能直接比较'
tables = {x['table']: x['years'] for x in new['factors']}
old_tables = {x['table']: x['years'] for x in old['factors']}
assert tables.keys() == old_tables.keys(), '因子表集合变化，需先复核'
backfilled, fresh, uncovered, lost, factor_changes = [], [], [], [], []
for t, years in tables.items():
    before = old_tables[t].get('2026', {})
    after = years.get('2026', {})
    old_missing = set(before.get('missing', old_cal))
    new_missing = set(after.get('missing', new_cal))
    filled_dates = sorted(old_missing - new_missing)
    fresh_dates = sorted((new_cal - old_cal) - new_missing)
    lost_dates = sorted((old_cal - old_missing) & new_missing)
    for target, dates in ((backfilled, filled_dates), (fresh, fresh_dates), (uncovered, sorted(new_missing)), (lost, lost_dates)):
        target.extend(f'/home/quant/factors/kyzqsh11/{t}/2026/{d.replace("-", "")}.zip' for d in dates)
    factor_changes.append({'table': t, 'historical_backfilled': len(filled_dates),
                           'new_calendar_dates_added': len(fresh_dates), 'lost_dates': lost_dates,
                           'missing_dates': sorted(new_missing), 'latest': after.get('last')})


def market_missing(snapshot):
    paths = set()
    for relative, years in snapshot['minute'].items():
        for info in years.values():
            paths.update(f'/home/quant/quantDataProcess/backend/data/{relative}/{d.replace("-", "")}.parquet' for d in info['missing'])
    for market, years in snapshot['tick'].items():
        for year, info in years.items():
            paths.update(f'/home/quant/tick-All/{market}/{year}/{d.replace("-", "")}_Tick1.tar.gz' for d in info['missing'])
    return paths


before_market, after_market = market_missing(old), market_missing(new)
tz = timezone(timedelta(hours=8))
start = datetime.fromisoformat(new['started_utc']).astimezone(tz)
end = datetime.fromisoformat(new['ended_utc']).astimezone(tz)
header = f'主扫描：{start:%Y-%m-%d %H:%M:%S}—{end:%H:%M:%S}（北京时间）；截至{new["cutoff"]}。9月30日尚未收盘，不计缺口。\n'
attachments = [
    ('缺失行情文件_逐路径.txt', sorted(after_market), '指定目录中未见的日期文件；包含历史遗留及9月28、29日未到位文件。'),
    ('本次补齐行情文件_逐路径.txt', sorted(before_market - after_market), '上次缺失、本次日期已覆盖；不是逐值质量验收。'),
    ('本次新增缺口行情文件_逐路径.txt', sorted(after_market - before_market), '因检查日期延长到9月29日而新列出的缺口，不表示原有文件被删除。'),
    ('因子2026待补日期文件_逐路径.txt', uncovered, '仅检查参考交易日期文件；0条不代表股票、字段、取值全部通过验收。'),
    ('本次补齐历史因子日期文件_逐路径.txt', backfilled, '截至上次参考日历9月24日的历史补齐路径。'),
    ('本次新增交易日因子日期文件_逐路径.txt', fresh, '本次增加的9月28、29日参考交易日期对应路径。'),
    ('本次新增因子日期文件_逐路径.txt', sorted(backfilled + fresh), '历史补齐与新增交易日合计；不是因子数量，也不统计已有文件的内容改写。'),
]
for name, paths, note in attachments:
    assert len(paths) == len(set(paths)), name
    (out / name).write_text(header + note + f'\n共{len(paths)}条路径。\n\n' + '\n'.join(paths) + ('\n' if paths else ''), encoding='utf-8')

years = [str(y) for y in range(2016, 2027)]
lines = ['# 因子逐表日期覆盖', '', header,
         '每格为“现有日期文件数 / 参考交易日期未覆盖数”；无目录单独标明。低频表应按实际供数频率解释，未覆盖不自动等于丢失。', '',
         '| 表名 | ' + ' | '.join(years) + ' |', '|---|' + '---|' * len(years)]
for table, values in tables.items():
    cells = [f"{values[y]['files']} / {len(values[y]['missing'])}" if y in values else '无目录' for y in years]
    lines.append('| ' + table + ' | ' + ' | '.join(cells) + ' |')
(out / '因子逐表日期覆盖.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

summary = {
    'previous_snapshot': 'audit_20260928', 'current_snapshot': 'audit_20260930',
    'previous_calendar_last': max(old_cal), 'current_calendar_last': max(new_cal),
    'new_calendar_dates': sorted(new_cal - old_cal),
    'factor_2026_reference_dates': len(new_cal),
    'complete_2026_tables': sum('2026' in y and not y['2026']['missing'] for y in tables.values()),
    'absent_2026_tables': [t for t,y in tables.items() if '2026' not in y],
    'historical_backfilled_factor_paths': len(backfilled), 'new_calendar_factor_paths': len(fresh),
    'added_factor_date_paths': len(backfilled) + len(fresh),
    'uncovered_factor_date_paths': len(uncovered), 'lost_factor_date_paths': lost,
    'factor_changes': factor_changes,
    'complete_2025_tables': sum('2025' in y and not y['2025']['missing'] for y in tables.values()),
    'factor_history_before_2026_changed': any({y:v for y,v in tables[t].items() if y!='2026'} != {y:v for y,v in old_tables[t].items() if y!='2026'} for t in tables),
    'previous_missing_market_paths': len(before_market), 'current_missing_market_paths': len(after_market),
    'filled_market_paths': sorted(before_market - after_market),
    'newly_uncovered_market_paths': sorted(after_market - before_market),
    'remaining_previous_market_paths': sorted(before_market & after_market),
    'daily_added_files': {k: sorted({r['file'] for r in new['daily'][k]['manifest']} - {r['file'] for r in old['daily'][k]['manifest']}) for k in new['daily']},
    'daily_removed_files': {k: sorted({r['file'] for r in old['daily'][k]['manifest']} - {r['file'] for r in new['daily'][k]['manifest']}) for k in new['daily']},
    'directory_entry_changes': {},
}
for path, entries in new['directories'].items():
    a = {x['name']: x for x in old['directories'].get(path, [])}
    b = {x['name']: x for x in entries}
    summary['directory_entry_changes'][path] = {'added': sorted(b.keys()-a.keys()), 'removed': sorted(a.keys()-b.keys()),
        'metadata_changed': sorted(k for k in a.keys() & b.keys() if a[k] != b[k])}
assert len(backfilled) == 4512 and len(fresh) == 240 and not uncovered and not lost
assert len(before_market - after_market) == 18 and len(after_market - before_market) == 10 and len(after_market) == 34
(source / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('factor_changes', 'filled_market_paths', 'newly_uncovered_market_paths', 'remaining_previous_market_paths')}, ensure_ascii=False, indent=2))
