"""下载张子晨新增测试期（2026-05-22 至 2026-10-19）所需的远端数据。

范围：
- 因子：/home/quant/factors/kyzqsh11/ts_stockfactordata_c01/2026/*.zip 中 2026-05-22 及以后、至 2026-10-19 的逐日 zip；
- 行情：tickflow/raw 下逐股 csv，仅下载测试期内曾为中证2000成员的股票（并集），保留 2026-04-01 及以后行。

输出：training_data/research_20261021/market/
  factor_zips/            原始 zip（留档）
  factors_ext.parquet     date, stockid, oho_alpha_001..101
  prices_ext.parquet      date, stockid, open, high, low, close, volume, amount
  download_audit.json     逐文件 hash、覆盖日期、行数

远端文件为 utf-8（因子）/gbk（行情）。凭证读取自项目根目录 ftp服务器 文件，连接信息不入库。
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_review_ts_factors_master_20260920" / "ts-factors-master"))
from modeling.remote_data import connect  # noqa: E402

OUT = ROOT / "training_data" / "research_20261021" / "market"
ZIP_DIR = OUT / "factor_zips"
FACTOR_REMOTE = "/home/quant/factors/kyzqsh11/ts_stockfactordata_c01/2026"
PRICE_REMOTE = "/home/quant/quantDataProcess/backend/data/day-data/tickflow/raw"
FACTOR_COLUMNS = [f"oho_alpha_{i:03d}" for i in range(1, 102)]
F_START, F_END = "20260522", "20261019"
P_KEEP_FROM = "2026-04-01"
WORKERS = 8

LOCAL = threading.local()
CONNS: list = []
LOCK = threading.Lock()


def sftp():
    if not hasattr(LOCAL, "conn"):
        ssh, sf = connect(ROOT / "ftp服务器")
        LOCAL.conn = (ssh, sf)
        with LOCK:
            CONNS.append(LOCAL.conn)
    return LOCAL.conn[1]


def member_codes() -> list[str]:
    mem = pd.read_parquet(ROOT / "training_data" / "research_20261021" / "membership" / "constituent_daily.parquet")
    test = mem[(mem.date >= "2026-05-22") & (mem.date <= "2026-10-19")]
    codes = set(test.stockid.unique())
    cur = pd.read_csv(ROOT / "training_data" / "research_20261018" / "csi2000" / "constituents_20261020_official.csv",
                      dtype={"code": str})
    for c in cur["code"].str.zfill(6):
        if c[:3] in ("600", "601", "603", "605", "688"):
            mkt = "SH"
        elif c[0] in ("4", "8") or c.startswith("92"):
            mkt = "BJ"
        else:
            mkt = "SZ"
        codes.add(f"{mkt}{c}")
    return sorted(codes)


def download_factor(name: str) -> dict:
    dest = ZIP_DIR / name
    if not dest.exists():
        data = sftp().file(f"{FACTOR_REMOTE}/{name}", "rb").read()
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    data = dest.read_bytes()
    zf = zipfile.ZipFile(io.BytesIO(data))
    inner = zf.namelist()[0]
    df = pd.read_csv(io.BytesIO(zf.read(inner)))
    df["date"] = pd.to_datetime(df["enddate"].astype(int), unit="D", origin="1899-12-30")
    df = df[["date", "stockid"] + FACTOR_COLUMNS]
    for c in FACTOR_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["stockid"] = df["stockid"].astype(str).str.upper()
    return {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(df),
            "date": str(df.date.min().date()), "df": df}


def download_price(fname: str) -> dict:
    raw = sftp().file(f"{PRICE_REMOTE}/{fname}", "rb").read()
    df = pd.read_csv(io.BytesIO(raw), encoding="gbk")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"日期": "date", "股票代码": "stockid", "开盘": "open", "最高": "high",
                            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date >= P_KEEP_FROM]
    df["stockid"] = df["stockid"].str.replace(".", "", regex=False).str.upper()
    return {"file": fname, "sha256": hashlib.sha256(raw).hexdigest(), "rows": int(len(df)),
            "first": str(df.date.min().date()) if len(df) else None,
            "last": str(df.date.max().date()) if len(df) else None, "df": df}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(exist_ok=True)
    sf = sftp()

    names = sorted(n for n in sf.listdir(FACTOR_REMOTE)
                   if n.endswith(".zip") and F_START <= n[:8] <= F_END)
    print(f"factor zips to fetch: {len(names)}", flush=True)

    listing = sf.listdir(PRICE_REMOTE)
    by_prefix = {}
    for n in listing:
        if n.endswith(".csv") and "." in n:
            by_prefix[n.split(".")[0].upper() + n[7:9].upper()] = n
    codes = member_codes()
    targets = []
    missing_remote = []
    for c in codes:
        pref = c[2:] + c[:2]
        if pref in by_prefix:
            targets.append(by_prefix[pref])
        else:
            missing_remote.append(c)
    print(f"price csvs to fetch: {len(targets)}, missing remote: {len(missing_remote)}", flush=True)

    t0 = time.time()
    factor_rows, factor_meta = [], []
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(download_factor, names)):
            factor_meta.append({k: r[k] for k in ("file", "sha256", "rows", "date")})
            factor_rows.append(r["df"])
            if (i + 1) % 10 == 0:
                print(f"factors {i+1}/{len(names)} {time.time()-t0:.0f}s", flush=True)
    fac = pd.concat(factor_rows, ignore_index=True)
    fac.to_parquet(OUT / "factors_ext.parquet", index=False)
    print(f"factors done rows={len(fac)}", flush=True)

    price_rows, price_meta = [], []
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(download_price, targets)):
            price_meta.append({k: r[k] for k in ("file", "sha256", "rows", "first", "last")})
            if len(r["df"]):
                price_rows.append(r["df"])
            if (i + 1) % 200 == 0:
                print(f"prices {i+1}/{len(targets)} {time.time()-t0:.0f}s", flush=True)
    pri = pd.concat(price_rows, ignore_index=True)
    pri.to_parquet(OUT / "prices_ext.parquet", index=False)
    print(f"prices done rows={len(pri)}", flush=True)

    for ssh, _ in CONNS:
        ssh.close()

    audit = {
        "factor_files": factor_meta,
        "factor_date_range": [str(fac.date.min().date()), str(fac.date.max().date())],
        "factor_rows": int(len(fac)),
        "price_files": price_meta,
        "price_date_range": [str(pri.date.min().date()), str(pri.date.max().date())],
        "price_rows": int(len(pri)),
        "member_codes_needed": len(codes),
        "missing_remote_prices": missing_remote,
        "elapsed_seconds": time.time() - t0,
    }
    (OUT / "download_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if not isinstance(v, list)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
