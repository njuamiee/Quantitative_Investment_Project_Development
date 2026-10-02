"""只读补查旧日线、服务说明、新日线和最新分钟样本；不读取连接资料到输出。"""
import csv
import hashlib
import io
import json
import re
import stat
from datetime import datetime, timezone

import pandas as pd

from audit_server_inventory import ROOT, connect, decode

out = ROOT / 'training_data/audit_20260930'
out.mkdir(parents=True, exist_ok=True)
result = {'started_utc': datetime.now(timezone.utc).isoformat(),
          'scope': '旧日线每种复权抽查3只；新增北交所日线；最新BJ分钟样本；服务README；指定目录一层清单',
          'old_daily': {}, 'directories': {}, 'errors': []}
transport, sftp = connect()
sftp.get_channel().settimeout(60)


def read(path):
    with sftp.open(path, 'rb') as stream:
        return stream.read()


try:
    for relative in ('bar1m', 'tick'):
        path = '/home/quant/quantDataProcess/backend/data/' + relative
        result['directories'][path] = [
            {'name': a.filename, 'directory': stat.S_ISDIR(a.st_mode),
             'bytes': a.st_size, 'mtime': a.st_mtime}
            for a in sftp.listdir_attr(path) if not a.filename.startswith('.')]
    for adjustment in ('不复权', '前复权', '后复权'):
        path = '/home/quant/day-data/' + adjustment
        attrs = [a for a in sftp.listdir_attr(path) if a.filename.endswith('.csv')]
        samples = []
        for code in ('920010', '600000', '000001'):
            a = next(a for a in attrs if a.filename.startswith(code))
            with sftp.open(path + '/' + a.filename, 'rb') as stream:
                stream.seek(max(0, a.st_size - 1600))
                raw = stream.read()
            raw = raw.split(b'\n', 1)[-1] if a.st_size > 1600 else raw
            line = decode(raw).strip().splitlines()[-1]
            samples.append({'file': a.filename, 'last': line, 'bytes': a.st_size, 'mtime': a.st_mtime})
        result['old_daily'][adjustment] = {'csv_count': len(attrs), 'samples': samples}
    print('old_daily_checked', flush=True)
    path = '/home/quant/quantDataProcess/backend/README.md'
    raw = read(path)
    lines = decode(raw).splitlines()
    result['service_readme'] = {
        'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
        'schedule_lines': [s for s in lines if any(k in s for k in ('18:00', '18:45', 'DAILY_UPDATE_HOUR', 'MINUTE_UPDATE_HOUR', 'MINUTE_UPDATE_MINUTE', 'DAILY_UPDATE_ON_STARTUP'))],
        'adjustment_lines': [s for s in lines if any(k in s for k in ('加法', '负价', '混合口径', '流通股本'))],
    }
    path = '/home/quant/quantDataProcess/backend/data/day-data/tickflow/raw/920202.BJ-N安达.csv'
    raw = read(path)
    rows = list(csv.DictReader(io.StringIO(decode(raw))))
    result['new_daily_sample'] = {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': rows}
    path = '/home/quant/quantDataProcess/backend/data/bar1m/BJ'
    names = sorted(n for n in sftp.listdir(path) if re.fullmatch(r'\d{8}\.parquet', n))
    path += '/' + names[-1]
    a = sftp.stat(path)
    if a.st_size > 100_000_000:
        result['minute_sample'] = {'path': path, 'bytes': a.st_size, 'skipped': '超过100MB抽样读取限额'}
    else:
        raw = read(path)
        frame = pd.read_parquet(io.BytesIO(raw))
        sample = {'path': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'rows': len(frame), 'columns': list(frame.columns),
                  'dtypes': {c: str(t) for c, t in frame.dtypes.items()},
                  'null_counts': {c: int(n) for c, n in frame.isna().sum().items()},
                  'first_rows': json.loads(frame.head(3).to_json(orient='records', date_format='iso', force_ascii=False))}
        for column in frame.columns:
            if any(k in column.lower() for k in ('symbol', 'code', 'stock', 'ticker', 'security')):
                counts = frame[column].value_counts()
                sample[column + '_summary'] = {'unique': int(frame[column].nunique()),
                    'rows_per_value_counts': {str(k): int(v) for k, v in counts.value_counts().items()},
                    'prefix_counts': {str(k): int(v) for k, v in frame[column].drop_duplicates().astype(str).str[:3].value_counts().items()}}
            if any(k in column.lower() for k in ('date', 'time')):
                sample[column + '_range'] = {'min': str(frame[column].min()), 'max': str(frame[column].max()), 'unique': int(frame[column].nunique())}
        result['minute_sample'] = sample
    result['ended_utc'] = datetime.now(timezone.utc).isoformat()
finally:
    (out / 'supplementary_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    sftp.close()
    transport.close()
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
