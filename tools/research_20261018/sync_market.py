"""Read-only SFTP snapshot of the broad A-share c01 factors and raw/hfq daily prices.

The remote server is only listed/read. Each worker reuses one SFTP connection.
No historical index membership is inferred here; join that separately by date.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from audit_server_inventory import connect, decode  # noqa: E402

OUT = ROOT / "training_data" / "research_20261018" / "market"
FACTOR_REMOTE = "/home/quant/factors/kyzqsh11/ts_stockfactordata_c01"
PRICE_REMOTE = "/home/quant/quantDataProcess/backend/data/day-data/tickflow"
FACTOR_START, FACTOR_END = "20230811", "20260522"
PRICE_START, PRICE_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-05-29")
SH_PREFIX = ("600", "601", "603", "605", "688")
SZ_PREFIX = ("000", "001", "002", "003", "300", "301")
BJ_PREFIX = ("4", "8", "92")
FACTOR_COLUMNS = [f"oho_alpha_{i:03d}" for i in range(1, 102)]
PRICE_COLUMNS = ["date", "stockid", "open", "high", "low", "close", "volume", "amount"]
PRICE_RENAME = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
PRICE_SCHEMA = pa.schema([pa.field("date", pa.timestamp("ns")), pa.field("stockid", pa.string())] +
                         [pa.field(c, pa.float64()) for c in PRICE_COLUMNS[2:]])
FACTOR_SCHEMA = pa.schema([pa.field("date", pa.timestamp("ns")), pa.field("stockid", pa.string())] +
                          [pa.field(c, pa.float32()) for c in FACTOR_COLUMNS])
LOCAL = threading.local()
CONNECTIONS: list[tuple[object, object]] = []
CONNECTIONS_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def client():
    if not hasattr(LOCAL, "sftp"):
        t, s = connect()
        s.get_channel().settimeout(120)
        LOCAL.transport, LOCAL.sftp = t, s
        with CONNECTIONS_LOCK:
            CONNECTIONS.append((t, s))
    return LOCAL.sftp


def reset_client() -> None:
    if hasattr(LOCAL, "sftp"):
        try:
            LOCAL.sftp.close()
            LOCAL.transport.close()
        except Exception:
            pass
        del LOCAL.sftp, LOCAL.transport


def close_clients() -> None:
    with CONNECTIONS_LOCK:
        for t, s in CONNECTIONS:
            try:
                s.close()
                t.close()
            except Exception:
                pass
        CONNECTIONS.clear()


def listed_jobs(phase: str) -> list[dict]:
    t, s = connect()
    jobs = []
    try:
        s.get_channel().settimeout(120)
        if phase == "factors":
            for year in range(2023, 2027):
                remote_dir = f"{FACTOR_REMOTE}/{year}"
                for a in s.listdir_attr(remote_dir):
                    name = a.filename
                    if re.fullmatch(r"\d{8}\.zip", name) and FACTOR_START <= name[:8] <= FACTOR_END:
                        jobs.append({"kind": phase, "remote": f"{remote_dir}/{name}",
                                     "local": f"source/factors/{year}/{name}", "bytes": a.st_size,
                                     "remote_mtime": a.st_mtime, "date": name[:8]})
        else:
            remote_dir = f"{PRICE_REMOTE}/{phase}"
            for a in s.listdir_attr(remote_dir):
                name = a.filename
                m = re.match(r"^(\d{6})\.(SH|SZ|BJ)-.+\.csv$", name)
                if not m:
                    continue
                code, exchange = m.groups()
                if not ((exchange == "SH" and code.startswith(SH_PREFIX)) or
                        (exchange == "SZ" and code.startswith(SZ_PREFIX)) or
                        (exchange == "BJ" and code.startswith(BJ_PREFIX))):
                    continue
                jobs.append({"kind": phase, "remote": f"{remote_dir}/{name}",
                             "local": f"source/prices/{phase}/{exchange}{code}.csv",
                             "bytes": a.st_size, "remote_mtime": a.st_mtime,
                             "stockid": exchange + code, "source_name": name})
    finally:
        s.close()
        t.close()
    jobs.sort(key=lambda j: j["local"])
    if len({j["local"] for j in jobs}) != len(jobs):
        raise ValueError(f"duplicate local source paths in {phase}")
    save_json({"scanned_utc": now(), "phase": phase, "jobs": jobs}, OUT / f"scan_{phase}.json")
    return jobs


def transfer(job: dict) -> dict:
    path = OUT / job["local"]
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(job)
    for attempt in range(3):
        try:
            cached = path.is_file() and path.stat().st_size == job["bytes"]
            if not cached:
                part = path.with_suffix(path.suffix + ".part")
                client().get(job["remote"], str(part))
                if part.stat().st_size != job["bytes"]:
                    raise ValueError("remote size changed during transfer")
                part.replace(path)
            record.update(status="cached" if cached else "downloaded", sha256=digest(path))
            return record
        except Exception as exc:
            reset_client()
            if attempt == 2:
                record.update(status="failed", error_type=type(exc).__name__, sha256=None)
                return record
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def download_phase(phase: str, workers: int) -> list[dict]:
    jobs = listed_jobs(phase)
    print(f"{phase}: {len(jobs)} source files, {sum(j['bytes'] for j in jobs)/1e9:.3f} GB", flush=True)
    records = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for i, record in enumerate(pool.map(transfer, jobs), 1):
                records.append(record)
                if i % 100 == 0 or i == len(jobs):
                    save_json({"phase": phase, "updated_utc": now(), "complete": i == len(jobs),
                               "records": records}, OUT / f"download_{phase}.json")
                    print(f"{phase}: {i}/{len(jobs)} downloaded/cached; "
                          f"failed={sum(r['status']=='failed' for r in records)}", flush=True)
    finally:
        close_clients()
    return records


def factor_frame(path: Path, filename_date: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = [name for name in z.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in {path.name}")
        frame = pd.read_csv(io.StringIO(decode(z.read(names[0]))), na_values=["<NIL>", "NIL", "--"])
    needed = ["stockid", "enddate"] + FACTOR_COLUMNS
    if set(needed) != set(frame.columns):
        raise ValueError(f"c01 schema changed in {path.name}: {sorted(set(needed)^set(frame.columns))}")
    dates = pd.to_datetime(pd.to_numeric(frame["enddate"], errors="raise"),
                           unit="D", origin="1899-12-30")
    if not (dates == pd.Timestamp(filename_date)).all():
        raise ValueError(f"enddate does not equal ZIP date in {path.name}")
    ids = frame["stockid"].astype("string")
    valid = ids.str.fullmatch(r"(?:SH(?:600|601|603|605|688)\d{3}|SZ(?:000|001|002|003|300|301)\d{3}|BJ(?:[48]\d{5}|92\d{4}))").fillna(False)
    frame = frame.loc[valid, FACTOR_COLUMNS].copy()
    frame.insert(0, "stockid", ids.loc[valid].astype(str).to_numpy())
    frame.insert(0, "date", pd.Timestamp(filename_date))
    if frame.duplicated(["date", "stockid"]).any():
        raise ValueError(f"duplicate date/stockid in {path.name}")
    for col in FACTOR_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return frame


def price_frame(path: Path, stockid: str) -> tuple[pd.DataFrame, str, int]:
    frame = pd.read_csv(io.StringIO(decode(path.read_bytes())), low_memory=False).rename(columns=PRICE_RENAME)
    if not set(PRICE_COLUMNS).difference({"stockid"}).issubset(frame.columns):
        raise ValueError(f"price schema changed: {path.name}")
    if "股票代码" in frame.columns:
        codes = pd.to_numeric(frame["股票代码"], errors="coerce").dropna().astype(int).astype(str).str.zfill(6)
        if len(codes) and not codes.eq(stockid[2:]).all():
            raise ValueError(f"price code differs from filename: {path.name}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"invalid/duplicate price dates: {path.name}")
    source_last_date = dates.max().strftime("%Y-%m-%d") if len(dates) else ""
    frame = frame.loc[dates.between(PRICE_START, PRICE_END), PRICE_COLUMNS[2:]].copy()
    frame.insert(0, "stockid", stockid)
    frame.insert(0, "date", dates.loc[dates.between(PRICE_START, PRICE_END)].to_numpy())
    for col in PRICE_COLUMNS[2:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame, source_last_date, len(dates)


def parse_phase(phase: str, records: list[dict]) -> dict:
    if any(r["status"] == "failed" for r in records):
        raise RuntimeError(f"{phase}: source transfer incomplete")
    output = OUT / ("factors_c01.parquet" if phase == "factors" else f"prices_{phase}.parquet")
    temp = output.with_suffix(".parquet.part")
    schema = FACTOR_SCHEMA if phase == "factors" else PRICE_SCHEMA
    writer = pq.ParquetWriter(temp, schema=schema, compression="zstd")
    stats = []
    buffer = []
    observed_ids = set()
    try:
        for i, record in enumerate(records, 1):
            path = OUT / record["local"]
            if phase == "factors":
                frame = factor_frame(path, record["date"])
                info = {"date": record["date"], "rows": len(frame), "stocks": frame.stockid.nunique(),
                        "sha256": record["sha256"]}
            else:
                frame, last, total = price_frame(path, record["stockid"])
                info = {"stockid": record["stockid"], "rows": len(frame), "source_rows": total,
                        "source_last_date": last, "sha256": record["sha256"],
                        "remote": record["remote"]}
            stats.append(info)
            observed_ids.update(frame.stockid.unique())
            if len(frame):
                buffer.append(frame)
            if len(buffer) >= (10 if phase == "factors" else 50) or i == len(records):
                if buffer:
                    batch = pd.concat(buffer, ignore_index=True)
                    writer.write_table(pa.Table.from_pandas(batch, schema=schema, preserve_index=False))
                    buffer.clear()
            if i % 100 == 0 or i == len(records):
                print(f"{phase}: parsed {i}/{len(records)}; rows={sum(x['rows'] for x in stats)}", flush=True)
    finally:
        writer.close()
    temp.replace(output)
    summary = {"phase": phase, "built_utc": now(), "range": [str(PRICE_START.date()), str(PRICE_END.date())]
               if phase != "factors" else [FACTOR_START, FACTOR_END],
               "source_files": len(records), "rows": sum(s["rows"] for s in stats),
               "stock_count": len(observed_ids),
               "stock_count_definition": "unique codes with at least one parsed row in requested range",
               "max_daily_factor_stocks": max((s.get("stocks",0) for s in stats),default=0) if phase=='factors' else None,
               "parquet": str(output.relative_to(ROOT)), "parquet_sha256": digest(output),
               "columns": schema.names, "source_details": stats,
               "limitations": ["c01 publication timestamp not independently verified; ZIP date is treated as factor observation date",
                               "price CSV historical revisions are not timestamped; source hash freezes the current snapshot",
                               "index membership must be joined from a separate dated historical ledger"]}
    save_json(summary, OUT / f"parsed_{phase}.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["factors", "raw", "hfq", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be 1..4")
    OUT.mkdir(parents=True, exist_ok=True)
    phases = ["factors", "raw", "hfq"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if args.parse_only:
            records = json.loads((OUT / f"download_{phase}.json").read_text(encoding="utf-8"))["records"]
        else:
            records = download_phase(phase, args.workers)
        result = parse_phase(phase, records)
        print(f"{phase} complete: {result['rows']} rows -> {result['parquet']}", flush=True)


if __name__ == "__main__":
    main()
