"""组装中证2000成员限定面板（因子 + 行情 + 五日标签）。

数据源：
- 因子：research_20261018 全市场 Alpha101（至 2026-05-22）+ research_20261021 扩展（至 2026-10-19）
- 行情：research_20261018 prices_raw（至 2026-05-29）+ research_20261021 扩展（自 2026-04-01）
- 成员：research_20261021/membership/constituent_daily.parquet（逐日真实成员）

标签口径（与 1017 项目一致，非张子晨 PPT 的 open→open）：
  label_5 = close(t+5) / open(t+1) - 1，按个股自身交易日序列计算；停牌期间跳过。
输出：training_data/research_20261021/market/panel_member.parquet
      列：date, stockid, open, high, low, close, volume, amount, label_5, oho_alpha_001..101
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M1018 = ROOT / "training_data" / "research_20261018" / "market"
M1021 = ROOT / "training_data" / "research_20261021" / "market"
MEM = ROOT / "training_data" / "research_20261021" / "membership" / "constituent_daily.parquet"
OUT = M1021 / "panel_member.parquet"

FACTOR_COLUMNS = [f"oho_alpha_{i:03d}" for i in range(1, 102)]


def main() -> None:
    mem = pd.read_parquet(MEM)
    mem["date"] = pd.to_datetime(mem["date"])
    print(f"membership rows {len(mem)}", flush=True)

    fac_old = pd.read_parquet(M1018 / "factors_c01.parquet")
    fac_new = pd.read_parquet(M1021 / "factors_ext.parquet")
    fac = pd.concat([fac_old, fac_new], ignore_index=True)
    fac["date"] = pd.to_datetime(fac["date"])
    # 05-22 在两边重叠：旧版保留（同一远端来源，先前已审计），新版去重
    fac = fac.sort_values("date").drop_duplicates(["date", "stockid"], keep="first")
    fac = fac.merge(mem, on=["date", "stockid"], how="inner")
    print(f"factor member rows {len(fac)}, dates {fac.date.min().date()}~{fac.date.max().date()}", flush=True)

    pri_old = pd.read_parquet(M1018 / "prices_raw.parquet")
    pri_new = pd.read_parquet(M1021 / "prices_ext.parquet")
    # 下载脚本的 stockid 形如 '000001SZ'（去掉了点），统一回 'SZ000001'
    s = pri_new["stockid"].astype(str)
    pri_new["stockid"] = s.str[6:] + s.str[:6]
    pri = pd.concat([pri_old, pri_new], ignore_index=True)
    pri["date"] = pd.to_datetime(pri["date"])
    pri = pri.sort_values("date").drop_duplicates(["date", "stockid"], keep="first")
    pri = pri.merge(mem, on=["date", "stockid"], how="inner")
    print(f"price member rows {len(pri)}", flush=True)

    pri = pri.sort_values(["stockid", "date"])
    g = pri.groupby("stockid", sort=False)
    pri["open_t1"] = g["open"].shift(-1)
    pri["close_t5"] = g["close"].shift(-5)
    pri["label_5"] = pri["close_t5"] / pri["open_t1"] - 1.0

    panel = fac.merge(
        pri[["date", "stockid", "open", "high", "low", "close", "volume", "amount", "label_5"]],
        on=["date", "stockid"], how="left")
    panel.to_parquet(OUT, index=False)

    audit = {
        "panel_rows": int(len(panel)),
        "date_range": [str(panel.date.min().date()), str(panel.date.max().date())],
        "unique_stocks": int(panel.stockid.nunique()),
        "label_coverage": float(panel.label_5.notna().mean()),
        "factor_missing_rate": float(panel[FACTOR_COLUMNS].isna().mean().mean()),
        "rows_missing_price": int(panel.close.isna().sum()),
    }
    (M1021 / "panel_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
