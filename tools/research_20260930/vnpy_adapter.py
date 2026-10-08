"""Offline handoff for installed vn.py; no connection or order submission.

Commercial-edition compatibility must be verified in its delivered environment.
"""
from zoneinfo import ZoneInfo
import pandas as pd

def available_signals(csv_path,as_of):
    frame=pd.read_csv(csv_path,parse_dates=['date','signal_available_at','earliest_execution_date'])
    now=pd.Timestamp(as_of)
    if now.tzinfo is not None:now=now.tz_convert('Asia/Shanghai').tz_localize(None)
    frame=frame[(frame.signal_available_at<=now)&(frame.earliest_execution_date==now.normalize())].copy()
    mapping={'BJ':'BSE','SH':'SSE','SZ':'SZSE'}
    frame['vt_symbol']=frame.stockid.map(lambda sid:sid[2:]+'.'+mapping[sid[:2]])
    if frame.duplicated(['date','stockid']).any():raise ValueError('duplicate signals')
    return frame

def parquet_to_vnpy_bars(path):
    try:
        from vnpy.trader.object import BarData
        from vnpy.trader.constant import Exchange,Interval
    except ImportError as exc:
        raise RuntimeError('Run this adapter in the delivered vn.py environment; this task did not install a commercial platform.') from exc
    if not hasattr(Exchange,'BSE'):raise RuntimeError('Installed vn.py lacks Exchange.BSE; verify version mapping first.')
    frame=pd.read_parquet(path)
    required=['open','high','low','close','volume','amount']
    frame=frame.dropna(subset=required)
    valid=(frame[['open','high','low','close']]>0).all(axis=1)
    valid &= (frame.high>=frame[['open','low','close']].max(axis=1))&(frame.low<=frame[['open','high','close']].min(axis=1))
    frame=frame[valid&(frame.volume>0)&(frame.code.astype(str).str.fullmatch(r'920\d{3}'))]
    bars=[]
    for row in frame.itertuples(index=False):
        stamp=pd.Timestamp(row.datetime)
        stamp=stamp.tz_localize('Asia/Shanghai') if stamp.tzinfo is None else stamp.tz_convert('Asia/Shanghai')
        dt=stamp.to_pydatetime()
        bars.append(BarData(symbol=str(row.code),exchange=Exchange.BSE,datetime=dt,interval=Interval.MINUTE,
            volume=float(row.volume),turnover=float(row.amount),open_price=float(row.open),
            high_price=float(row.high),low_price=float(row.low),close_price=float(row.close),gateway_name='RESEARCH_IMPORT'))
    return bars
