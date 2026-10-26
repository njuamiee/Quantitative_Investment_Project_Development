"""把中证2000官方日线（价格指数与全收益）延长到 2026-10-19。

来源与 1018 相同的官方公开接口，只取指数点位，不含成分。
输出：training_data/research_20261021/market/932000_official_daily_ext.csv / 932000CNY010_...
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_data" / "research_20261021" / "market"
URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
PARAMS = {"startDate": "20230811", "endDate": "20261019"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for code in ("932000", "932000CNY010"):
        r = requests.get(URL, params={"indexCode": code, **PARAMS}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("data")
        assert str(payload.get("code")) == "200" and isinstance(rows, list) and rows, payload.get("code")
        dates = sorted({str(x["tradeDate"]) for x in rows})
        fields = ["tradeDate", "indexCode", "open", "high", "low", "close", "change",
                  "changePct", "tradingVol", "tradingValue", "consNumber", "peg"]
        path = OUT / f"{code}_official_daily_ext.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for x in sorted(rows, key=lambda x: x["tradeDate"]):
                w.writerow({k: x.get(k) for k in fields})
        prov = {"source_url": r.url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(rows), "first": dates[0], "last": dates[-1],
                "sha256_csv": hashlib.sha256(path.read_bytes()).hexdigest(),
                "series": "price" if code == "932000" else "gross_total_return"}
        path.with_suffix(".provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(prov, ensure_ascii=False))


if __name__ == "__main__":
    main()
