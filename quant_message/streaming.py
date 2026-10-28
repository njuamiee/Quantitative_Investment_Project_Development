"""Read-only realtime-paper entry point for a provider's normalized JSONL stream.

Provider adapter must emit incremental share volume, exchange time and receipt
time. This process has NO gateway/order connector and never places real orders.
"""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import argparse
import json
import sys

from .data import read_jsonl, read_signal, validate_ticks
from .domain import SH, Costs, Instrument
from .paper import PaperBroker
from .strategy import SimpleMomentumStrategy, TargetStrategy, RiskRules


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--symbols',required=True,help='Comma-separated vt_symbols')
    p.add_argument('--strategy',choices=['simple','full'],default='simple')
    p.add_argument('--signals')
    p.add_argument('--capital',type=float,default=1_000_000)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    symbols=args.symbols.split(',')
    b=PaperBroker([Instrument(s,step=1 if s.endswith('.BSE') else 100) for s in symbols],
        capital=args.capital,journal=output/'events.jsonl')
    controller=SimpleMomentumStrategy(b,symbols[0]) if args.strategy=='simple' else TargetStrategy(b)
    if args.strategy=='full' and not args.signals:raise ValueError('signal file required')
    # Waiting on stdin cannot create trades: pending orders only fill after a
    # fresh message and expire before matching. EOF cancels all paper orders.
    try:
        for line in sys.stdin:
            if not line.strip():continue
            record=json.loads(line)
            tick=next(validate_ticks([record]))
            now=datetime.now(SH)
            if not 0 <= (now-tick.available_at).total_seconds() <= 10:
                raise ValueError('paper stream requires current receipt timestamps; use historical replay for old data')
            if args.strategy=='full':
                eligible=[s for s in map(read_signal,read_jsonl(args.signals))
                          if s.valid_from<=tick.available_at<s.expires_at and s.available_at<=tick.available_at]
                if eligible:
                    s=max(eligible,key=lambda s:s.available_at)
                    b.advance(tick.available_at)
                    if s.id not in controller.signal_ids:controller.on_signal(s)
            b.on_tick(tick)
    finally:
        controller.stop();b.flush()
        (output/'account.json').write_text(json.dumps(dict(cash=b.cash,equity=b.equity(),
            fills=len(b.fills),positions={k:asdict(v) for k,v in b.positions.items()},
            real_orders_sent=0),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
