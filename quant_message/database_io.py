"""Import once through vn.py; replay exclusively through its database loader."""
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import re
import sys

from .domain import SH


def connect_database(name, create=False):
    if not re.fullmatch(r'quant_message_20261029_[a-z0-9_]+',name):
        raise ValueError('a task-scoped ClickHouse database is required')
    # The commercial loader caches get_database() at import. Configure BEFORE
    # importing it, instead of replacing its loader or the cached object.
    if 'vnpy_deltaone.utility' in sys.modules:
        raise RuntimeError('configure database before importing DeltaOne; use a fresh process')
    from vnpy.trader import database as module
    from vnpy.trader.setting import SETTINGS
    if SETTINGS['database.name'] != 'clickhouse':
        raise RuntimeError('this validated runner requires the installed ClickHouse driver')
    if create:
        module.get_database().client.execute('CREATE DATABASE IF NOT EXISTS '+name)
    SETTINGS['database.database']=name
    module.database=None
    db=module.get_database()
    if db.client.execute('SELECT currentDatabase()')[0][0] != name:
        raise RuntimeError('driver ignored database selection')
    return db


def tick_row(t):
    dt=t.datetime if t.datetime.tzinfo else t.datetime.replace(tzinfo=SH)
    fields=['last_price','volume','turnover','pre_close','limit_up','limit_down']
    fields += [f'{side}_{what}_{i}' for side in ('bid','ask') for what in ('price','volume') for i in range(1,6)]
    return [t.vt_symbol,dt.astimezone(SH).isoformat()]+[float(getattr(t,k)) for k in fields]


def digest_ticks(ticks):
    h=hashlib.sha256()
    for t in sorted(ticks,key=lambda t:(t.datetime,t.vt_symbol)):
        h.update((json.dumps(tick_row(t),separators=(',',':'))+'\n').encode())
    return h.hexdigest()


def import_sample(data_dir, database, output, mutate_after=None):
    import pandas as pd
    from vnpy.trader.object import TickData
    from vnpy.trader.constant import Exchange
    db=connect_database(database,create=True)
    if db.client.execute('SELECT count() FROM tick_data')[0][0]:
        raise RuntimeError('import target is nonempty; never append duplicate experiment data')
    source=Path(data_dir);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    f=pd.read_parquet(source/'ticks.parquet')
    audit=[]
    for symbol,g in f.groupby('code',sort=True):
        ticks=[]
        for r in g.sort_values('datetime',kind='stable').itertuples():
            at=r.datetime.to_pydatetime().replace(tzinfo=SH)
            scale=1.15 if mutate_after and at.date().isoformat()>mutate_after else 1.
            t=TickData(symbol=symbol,exchange=Exchange.BSE,datetime=at,gateway_name='DB_IMPORT',
                last_price=float(r.price)*scale,volume=float(r.accvolume),turnover=float(r.accturnover)*scale,
                pre_close=float(r.pre_close)*scale,limit_up=float(r.HighLimit)*scale,limit_down=float(r.LowLimit)*scale)
            for i in range(1,6):
                for side,src,volume in [('bid','bid','bsize'),('ask','ask','asize')]:
                    setattr(t,f'{side}_price_{i}',float(getattr(r,f'{src}{i}'))*scale)
                    setattr(t,f'{side}_volume_{i}',float(getattr(r,f'{volume}{i}')))
            ticks.append(t)
        expected=digest_ticks(ticks)
        if not db.save_tick_data(ticks):raise RuntimeError('vn.py save_tick_data failed')
        loaded=db.load_tick_data(symbol,Exchange.BSE,min(t.datetime for t in ticks),max(t.datetime for t in ticks))
        actual=digest_ticks(loaded)
        if len(loaded)!=len(ticks) or actual!=expected:
            raise AssertionError(f'database roundtrip mismatch for {symbol}')
        audit.append(dict(symbol=symbol+'.BSE',rows=len(ticks),sha256=actual))
    manifest=dict(database=database,symbols=[a['symbol'] for a in audit],
        sessions=sorted(str(d) for d in f.datetime.dt.date.unique()),rows=len(f),roundtrip=audit,
        source_sha256=hashlib.sha256((source/'ticks.parquet').read_bytes()).hexdigest(),
        mutate_after=mutate_after,mutation='future prices and turnover x1.15' if mutate_after else None)
    (out/'database_manifest.json').write_text(json.dumps(manifest,indent=2))
    for name in ('signals.jsonl','actions.json'):
        (out/name).write_bytes((source/name).read_bytes())
    if mutate_after:
        signals=[json.loads(line) for line in (out/'signals.jsonl').read_text().splitlines()]
        for s in signals:
            if s['valid_from'][:10]>mutate_after:
                s['market_cap']=.1
                for c in s['candidates']:
                    c['score']=-c['score'];c['reference']*=1.15
        (out/'signals.jsonl').write_text('\n'.join(json.dumps(s) for s in signals)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='roundtrip'}),flush=True)
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True);p.add_argument('--database',required=True)
    p.add_argument('--output',required=True);p.add_argument('--mutate-after')
    import_sample(**vars(p.parse_args()))
