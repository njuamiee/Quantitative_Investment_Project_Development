"""只读核对服务器日期覆盖、文件更新和因子股票键，保存可复核快照。"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import hashlib
import io
import json
import re
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_review_ts_factors_master_20260920/ts-factors-master"))
from modeling.remote_data import PINNED_HOST_KEY_SHA256, read_credentials


def connect():
    cred = read_credentials(ROOT / "ftp服务器")
    transport = paramiko.Transport((cred.host, 22))
    transport.start_client(timeout=30)
    actual = "SHA256:" + base64.b64encode(
        hashlib.sha256(transport.get_remote_server_key().asbytes()).digest()
    ).decode().rstrip("=")
    if actual != PINNED_HOST_KEY_SHA256:
        transport.close()
        raise RuntimeError("服务器身份校验失败")
    transport.auth_password(cred.username, cred.password)
    return transport, paramiko.SFTPClient.from_transport(transport)


def decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("未识别文本编码")


def date_string(name: str) -> str:
    return f"{name[:4]}-{name[4:6]}-{name[6:8]}"


def coverage(files: list[str], expected: set[str], cutoff: str) -> dict:
    dates = {date_string(f) for f in files}
    return {
        "files": len(files), "expected_dates": len(expected),
        "first": min(dates) if dates else None,
        "last": max(dates) if dates else None,
        "missing": sorted(expected - dates),
        "extra": sorted(d for d in dates - expected if d <= cutoff),
    }


def run(output: Path, cutoff: str):
    output.mkdir(parents=True, exist_ok=True)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "cutoff": cutoff, "scope": "目录日期全量核对；日线末日核对；因子指定样本内容检查；非全数据逐值验收",
              "directories": {}, "benchmarks": {}, "daily": {},
              "minute": {}, "tick": {}, "factors": [], "factor_samples": {}}

    def save():
        (output / "inventory.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    transport, sftp = connect()
    try:
        for path in ("/home/quant", "/home/quant/day-data", "/home/quant/factors",
                     "/home/quant/quantDataProcess/backend/data",
                     "/home/quant/quantDataProcess/backend/data/day-data"):
            result["directories"][path] = [
                {"name": a.filename, "directory": stat.S_ISDIR(a.st_mode),
                 "bytes": a.st_size, "mtime": a.st_mtime}
                for a in sftp.listdir_attr(path) if not a.filename.startswith(".")]
        base = "/home/quant/quantDataProcess/backend/data/day-data/benchmark/raw"
        attrs = sftp.listdir_attr(base)
        calendar = set()
        for code in ("000001.SH", "000300.SH", "000905.SH", "899050.BJ"):
            attr = next(a for a in attrs if a.filename.startswith(code))
            with sftp.open(base + "/" + attr.filename, "rb") as stream:
                raw = stream.read()
            (output / attr.filename).write_bytes(raw)
            rows = list(csv.DictReader(io.StringIO(decode(raw))))
            dates = [r["日期"] for r in rows]
            if code == "000001.SH":
                calendar = {d for d in dates if d <= cutoff}
            result["benchmarks"][code] = {
                "file": attr.filename, "rows": len(rows), "first": min(dates),
                "last": max(dates), "duplicate_dates": len(dates) - len(set(dates)),
                "missing": sorted({d for d in calendar if d >= min(dates)} - set(dates)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        result["reference_calendar"] = sorted(calendar)
        save()
        print("benchmark_done", json.dumps(result["benchmarks"], ensure_ascii=False), flush=True)

        for relative in ("bar1m/BJ", "tick/BJ"):
            path = "/home/quant/quantDataProcess/backend/data/" + relative
            names = sorted(n for n in sftp.listdir(path) if re.fullmatch(r"\d{8}\.parquet", n))
            result["minute"][relative] = {
                str(y): coverage([n for n in names if n.startswith(str(y))],
                                 {d for d in calendar if d.startswith(str(y)) and d >= "2021-11-15"}, cutoff)
                for y in range(2021, 2027)}
        save()
        print("minute_done", json.dumps({k: {y: (v["files"], len(v["missing"])) for y, v in ys.items()} for k, ys in result["minute"].items()}), flush=True)

        for adjustment in ("raw", "qfq"):
            path = "/home/quant/quantDataProcess/backend/data/day-data/tickflow/" + adjustment
            attrs = sorted((a for a in sftp.listdir_attr(path) if a.filename.endswith(".csv")), key=lambda a: a.filename)
            rows = []
            for a in attrs:
                with sftp.open(path + "/" + a.filename, "rb") as stream:
                    stream.seek(max(0, a.st_size - 1600))
                    # 从换行边界之后解码，避免多字节字符切在中间。
                    tail = stream.read()
                tail = tail.split(b"\n", 1)[-1] if a.st_size > 1600 else tail
                cells = next(csv.reader([decode(tail).strip().splitlines()[-1]]))
                date = next((v for v in cells[:3] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)), "unknown")
                rows.append({"file": a.filename, "last": date, "bytes": a.st_size, "mtime": a.st_mtime})
            result["daily"][adjustment] = {"files": len(rows), "last_counts": dict(collections.Counter(r["last"] for r in rows)), "manifest": rows}
            save()
            print("daily_done", adjustment, result["daily"][adjustment]["last_counts"], flush=True)

        for market in ("BJ", "SH", "SZ"):
            base = "/home/quant/tick-All/" + market
            years = sorted(y for y in sftp.listdir(base) if re.fullmatch(r"\d{4}", y))
            result["tick"][market] = {}
            for y in years:
                names = sorted(n for n in sftp.listdir(base + "/" + y) if re.fullmatch(r"\d{8}_Tick1\.tar\.gz", n))
                expected = {d for d in calendar if d.startswith(y) and (market != "BJ" or d >= "2021-11-15")}
                result["tick"][market][y] = coverage(names, expected, cutoff)
            save()
            print("tick_done", market, {y: len(x["missing"]) for y, x in result["tick"][market].items()}, flush=True)

        base = "/home/quant/factors/kyzqsh11"
        tables = sorted(a.filename for a in sftp.listdir_attr(base) if stat.S_ISDIR(a.st_mode) and a.filename.startswith("ts_stockfactordata_"))
        for i, table in enumerate(tables):
            path = base + "/" + table
            years = sorted(y for y in sftp.listdir(path) if re.fullmatch(r"\d{4}", y))
            row = {"table": table, "years": {}}
            for y in years:
                names = sorted(n for n in sftp.listdir(path + "/" + y) if re.fullmatch(r"\d{8}\.zip", n))
                row["years"][y] = coverage(names, {d for d in calendar if d.startswith(y)}, cutoff)
            result["factors"].append(row)
            if (i + 1) % 20 == 0:
                save()
                print("factor_dates", i + 1, "/", len(tables), flush=True)
        save()

        for suffix in ("c01", "c02", "001", "802", "a01"):
            table = "ts_stockfactordata_" + suffix
            info = next(x for x in result["factors"] if x["table"] == table)
            y = max(k for k, v in info["years"].items() if v["files"])
            filename = info["years"][y]["last"].replace("-", "") + ".zip"
            path = base + "/" + table + "/" + y + "/" + filename
            with sftp.open(path, "rb") as stream:
                raw = stream.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
                rows = list(csv.DictReader(io.StringIO(decode(archive.read(member)))))
            keys = [r.get("stockid", "") for r in rows]
            recognized = [k for k in keys if re.fullmatch(r"(BJ|SH|SZ)\d{6}", k)]
            sample = {"file": y + "/" + filename, "rows": len(rows), "recognized_stockids": len(recognized),
                      "unrecognized_ids": dict(collections.Counter(k for k in keys if k not in set(recognized))),
                      "columns": list(rows[0]) if rows else [],
                      "enddates": dict(collections.Counter(r.get("enddate", "") for r in rows)),
                      "sha256": hashlib.sha256(raw).hexdigest()}
            result["factor_samples"][table] = sample
            save()
            print("factor_sample", table, sample["file"], sample["rows"], sample["recognized_stockids"], flush=True)

        result["suspected_tick_duplicate"] = {}
        for market in ("BJ", "SZ"):
            path = "/home/quant/tick-All/" + market + "/2026"
            names = sftp.listdir(path)
            filename = "20260105_Tick1.tar.gz"
            checksum = next((n for n in names if n.startswith("20260105") and n.endswith("_md5.txt")), None)
            data = {"bytes": sftp.stat(path + "/" + filename).st_size}
            if checksum:
                with sftp.open(path + "/" + checksum, "rb") as stream:
                    data["supplied_md5"] = decode(stream.read()).strip()
            result["suspected_tick_duplicate"][market] = data
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print("audit_complete", str(output), flush=True)
    finally:
        sftp.close()
        transport.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-09-25")
    args = parser.parse_args()
    run(args.output, args.cutoff)
