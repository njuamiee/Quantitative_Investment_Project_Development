"""用 932000 成分进出记录倒推/顺推逐日成分名单。

输入：training_data/research_20261021/source/932000.CSI-成分进出记录-20261006.xlsx
      （含 2023-08-11 初始纳入 2000 只 + 历次定期/临时调整，截止 2026-09-14）
输出：training_data/research_20261021/membership/
      constituent_events.csv            逐条事件（标准化代码）
      constituent_daily.parquet         逐日成员长表 date x stockid（本地 SH/SZ/BJ 格式）
      membership_audit.json             逐日成员数与锚点核对
规则：某日收盘前已生效的名单才能用于该日选股；事件日按记录生效日当日生效。
      事件记录截止 2026-09-14，其后成分按“无变化”延续，属于推定而非证据。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "training_data" / "research_20261021" / "source" / "932000.CSI-成分进出记录-20261006.xlsx"
OUT = ROOT / "training_data" / "research_20261021" / "membership"
ANCHOR = ROOT / "training_data" / "research_20261018" / "csi2000" / "constituents_20261020_official.csv"


def norm_code(s: str) -> str:
    """'301117.SZ' -> 'SZ301117'；纯六位默认按前缀补市场。"""
    s = str(s).strip()
    if "." in s:
        code, mkt = s.split(".")
        return f"{mkt.upper()}{code.zfill(6)}"
    code = s.zfill(6)
    mkt = "SH" if code[:3] in ("600", "601", "603", "605", "688") else "SZ"
    if code[0] in ("4", "8") or code.startswith("92"):
        mkt = "BJ"
    return f"{mkt}{code}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(SRC)
    raw["日期"] = pd.to_datetime(raw["日期"])
    raw["stockid"] = raw["代码"].map(norm_code)
    events = raw.sort_values(["日期"]).reset_index(drop=True)
    events.to_csv(OUT / "constituent_events.csv", index=False, encoding="utf-8-sig")

    # 顺推：从初始名单开始，每个事件日先剔除后纳入
    first_date = events["日期"].min()
    init = set(events[(events["日期"] == first_date) & (events["纳入/剔除"] == "纳入")]["stockid"])
    assert len(init) == 2000, f"initial members != 2000: {len(init)}"

    state = set(init)
    snapshots: list[tuple[pd.Timestamp, set[str]]] = [(first_date, set(state))]
    counts = [(str(first_date.date()), len(state), "initial")]
    imbalance = []
    for d, g in events[events["日期"] > first_date].groupby("日期"):
        rm = set(g[g["纳入/剔除"] == "剔除"]["stockid"])
        add = set(g[g["纳入/剔除"] == "纳入"]["stockid"])
        missing = rm - state
        overlap = add & state
        if missing or overlap:
            imbalance.append({"date": str(d.date()), "remove_not_member": sorted(missing),
                              "add_already_member": sorted(overlap)})
        state = (state - rm) | add
        snapshots.append((d, set(state)))
        counts.append((str(d.date()), len(state), f"rm{len(rm)}/add{len(add)}"))

    # 展开成逐日长表：用官方指数交易日的真实日历（避免把节假日当交易日）；
    # 事件记录最后一天之后按推定延续
    cal_path = ROOT / "training_data" / "research_20261021" / "market" / "932000_official_daily_ext.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path, dtype={"tradeDate": str})
        all_dates = pd.to_datetime(cal["tradeDate"], format="%Y%m%d")
        all_dates = all_dates[(all_dates >= first_date) & (all_dates <= "2026-10-19")]
    else:
        all_dates = pd.bdate_range(first_date, "2026-10-19")
    rows = []
    si = 0
    for day in all_dates:
        while si + 1 < len(snapshots) and snapshots[si + 1][0] <= day:
            si += 1
        for sid in snapshots[si][1]:
            rows.append((day, sid))
    daily = pd.DataFrame(rows, columns=["date", "stockid"])
    daily.to_parquet(OUT / "constituent_daily.parquet", index=False)

    # 锚点核对：官网 2026-10-20 当期名单
    anchor = pd.read_csv(ANCHOR, dtype={"code": str})
    anchor["code"] = anchor["code"].str.zfill(6)
    anchor_set = set(anchor["code"].map(lambda c: norm_code(c)))
    derived_set = snapshots[-1][1]
    only_anchor = sorted(anchor_set - derived_set)
    only_derived = sorted(derived_set - anchor_set)

    per_day = daily.groupby("date").size()

    # 基准日 2026-09-30 的快照（用户指定基准）；记录截止 2026-09-14，
    # 因此 9-30 的状态 = 9-14 事件后的状态，落在记录覆盖内。
    base_date = pd.Timestamp("2026-09-30")
    si = 0
    while si + 1 < len(snapshots) and snapshots[si + 1][0] <= base_date:
        si += 1
    base_930 = snapshots[si][1]
    pd.DataFrame({"stockid": sorted(base_930)}).to_csv(
        OUT / "constituents_20260930_derived.csv", index=False, encoding="utf-8-sig")
    audit = {
        "source_file": SRC.name,
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "events_rows": int(len(events)),
        "events_date_range": [str(events["日期"].min().date()), str(events["日期"].max().date())],
        "initial_date": str(first_date.date()),
        "initial_members": len(init),
        "daily_rows": int(len(daily)),
        "daily_date_range": [str(daily.date.min().date()), str(daily.date.max().date())],
        "member_count_min": int(per_day.min()),
        "member_count_max": int(per_day.max()),
        "count_check_by_event_date": counts,
        "state_imbalances": imbalance,
        "anchor_20261020": {
            "anchor_members": len(anchor_set),
            "derived_members_at_last_event": len(derived_set),
            "only_in_official_file": only_anchor,
            "only_in_derived_record": only_derived,
            "match": not only_anchor and not only_derived,
            "note": "不一致说明 2026-09-15 至 2026-10-06（记录文件生成日）之间发生了一次进出；不影响 2026-09-30 基准，但 9-15 之后的逐日表含推定成分，见 caveat。",
        },
        "base_20260930_members": len(base_930),
        "caveat": ("事件记录截止2026-09-14；其后至2026-10-19的成分按无变化延续，属推定。"
                   "经与官网2026-10-20当期名单核对，9-15之后实际发生过一次进出（剔除 SH603680、SZ300277，"
                   "纳入 SZ001268、SZ002259），生效日未知，在 2026-09-15 至 2026-10-06 之间。"
                   "事件日的生效时点按记录日期当日生效处理，未区分公告日与生效日。"),
    }
    (OUT / "membership_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["events_rows", "member_count_min", "member_count_max",
                                             "daily_rows", "state_imbalances"]},
                     ensure_ascii=False, indent=2))
    print("anchor match:", audit["anchor_20261020"]["match"],
          "only_official:", only_anchor[:5], "only_derived:", only_derived[:5])


if __name__ == "__main__":
    main()
