"""Download the public, official CSI 2000 daily index levels (not members).

The endpoint is the one serving CSI's public historical index-performance page.
No credential or constituent inference is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_data" / "research_20261018" / "csi2000"
URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
DATE_PARAMS = {"startDate": "20230811", "endDate": "20260522"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for code in ("932000", "932000CNY010"):
        response = requests.get(URL, params={"indexCode": code, **DATE_PARAMS}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if str(payload.get("code")) != "200" or not isinstance(payload.get("data"), list):
            raise ValueError("Official endpoint returned an unexpected response")
        rows = payload["data"]
        if not rows or any(str(row.get("indexCode")) != code for row in rows):
            raise ValueError("Index code mismatch")
        dates = [str(row["tradeDate"]) for row in rows]
        if dates != sorted(set(dates)) or dates[0] != "20230811" or dates[-1] != "20260522":
            raise ValueError("Date coverage mismatch")

        path = OUT / f"{code}_official_daily.csv"
        fields = ["tradeDate", "indexCode", "open", "high", "low", "close", "change", "changePct", "tradingVol", "tradingValue", "consNumber", "peg"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = {
            "source_url": response.url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "description": "Official CSI 2000 daily index levels; no constituent symbols are included",
            "series": "price" if code == "932000" else "gross_total_return",
            "rows": len(rows),
            "first": dates[0],
            "last": dates[-1],
            "sha256_csv": sha256,
        }
        (OUT / f"{code}_official_daily.provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
