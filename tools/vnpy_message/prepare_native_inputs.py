"""Normalize recorded BSE books without inventing exchange or receive timestamps."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'training_data/vnpy_message/ticks'
    output=ROOT/'training_runs/vnpy_message/native_inputs'
    paths=[p for p in sorted(source.glob('*.parquet')) if p.stat().st_size]
    days={p.stem for p in paths}
    packets=[json.loads(s) for s in (ROOT/'training_runs/vnpy_message/signals.jsonl').read_text().splitlines()]
    packets=[p for p in packets if p['valid_from'][:10].replace('-','') in days]
    symbols={'920000'}
    for p in packets:symbols.update(c['symbol'].split('.')[0] for c in p['candidates'])
    cols=['code','datetime','price','accvolume','accturnover','pre_close','HighLimit','LowLimit']+[f'{k}{i}' for k in ['ask','bid','asize','bsize'] for i in range(1,6)]
    frames=[];audit=[]
    for path in paths:
        f=pd.read_parquet(path,columns=cols);f=f[f.code.isin(symbols)].copy()
        n=len(f);values=f[cols[2:]].to_numpy(float,na_value=np.nan)
        # Preserve one-sided limit markets. Crossed means BOTH quotes positive.
        crossed=(f.ask1>0)&(f.bid1>0)&(f.ask1<f.bid1)
        valid=np.isfinite(values).all(axis=1)&(f.price>0)&(f.accvolume>=0)&~crossed
        f=f[valid]
        if f.duplicated(['code','datetime']).any():raise ValueError('duplicate tick keys')
        frames.append(f)
        audit.append(dict(date=path.stem,input_rows=n,quarantined=n-len(f),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reasons='nonfinite fields, zero pretrade last price, negative cumulative volume, crossed two-sided book'))
    output.mkdir(parents=True,exist_ok=True)
    ticks = pd.concat(frames)
    ticks.to_parquet(output/'ticks.parquet',index=False)
    # Generate both representations together so the compatibility fallback
    # cannot accidentally replay a different version of the source rows.
    ticks.to_csv(output/'ticks.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
    market_dir=ROOT/'training_runs/systematic_20261026/data'
    calendar=json.loads((market_dir/'calendar.json').read_text())
    stocks=json.loads((market_dir/'stocks.json').read_text())
    market=np.load(market_dir/'market.npz')
    actions=[]
    for day in sorted(ticks.datetime.dt.date.unique()):
        di=calendar.index(str(day))
        for symbol in sorted(ticks.code.unique()):
            si=stocks.index('BJ'+symbol)
            dividend,ratio=float(market['dividend'][di,si]),float(market['split'][di,si])
            if dividend or ratio!=1:
                actions.append(dict(date=str(day),symbol=symbol+'.BSE',dividend=dividend,ratio=ratio,
                    settlement='gross receivable at ex-date, payment date and dividend tax unverified'))
    (output/'actions.json').write_text(json.dumps(actions,indent=2))
    (output/'signals.jsonl').write_text('\n'.join(json.dumps(p) for p in packets)+'\n')
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
            if p.name in ('ticks.parquet','ticks.csv.gz','actions.json','signals.jsonl','audit.json')}
    (output/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(rows=sum(len(f) for f in frames),symbols=len(symbols),sessions=len(paths))))


if __name__=='__main__':main()
