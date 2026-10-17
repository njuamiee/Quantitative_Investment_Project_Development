"""Parse official CAPCO code-sorted PDFs and explicit BJ old/new code table."""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training_data/research_20261017/industry"
PRICE = ROOT / "training_data/research_20261011/bj_prices.parquet"
CODE = re.compile(r"(?m)^([0-9]{6})$")
LETTER = re.compile(r"^[A-S]$")
MAJOR = re.compile(r"^\d{2}$")


def rows_from_pdf(path: Path) -> list[dict]:
    result = []
    with fitz.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            matches = list(CODE.finditer(text))
            for number, match in enumerate(matches):
                end = matches[number + 1].start() if number + 1 < len(matches) else len(text)
                lines = [line.strip() for line in text[match.end():end].splitlines() if line.strip()]
                if not lines:
                    continue
                letter_at = next((i for i, item in enumerate(lines) if LETTER.fullmatch(item)), None)
                if letter_at is None or letter_at == 0:
                    continue
                major_codes = [item for item in lines[letter_at + 1:] if MAJOR.fullmatch(item)]
                if len(major_codes) != 1:
                    continue
                result.append({
                    "original_code": match.group(1),
                    "name": " ".join(lines[:letter_at]),
                    "sector_code": lines[letter_at],
                    "major_code": major_codes[0],
                    "industry": lines[letter_at] + major_codes[0],
                    "page": page_number,
                })
    return result


def alias_table() -> pd.DataFrame:
    path = SOURCE / "capco_2025H1_code_alias.pdf"
    text = "\n".join(page.get_text() for page in fitz.open(path))
    pattern = re.compile(r"(?m)^\d+\n([^\n]+)\n(\d{4}-\d{2}-\d{2})\n(\d{6})\n(92\d{4})$")
    rows = [
        {"name": m.group(1), "listing_date_in_official_table": m.group(2), "old_code": m.group(3), "new_code": m.group(4), "source_pdf": path.name}
        for m in pattern.finditer(text)
    ]
    out = pd.DataFrame(rows)
    if len(out) != 248 or out.old_code.duplicated().any() or out.new_code.duplicated().any():
        raise ValueError(f"Unexpected alias table: {len(out)} rows")
    return out


def main() -> None:
    published = json.loads((SOURCE / "sources.json").read_text(encoding="utf-8"))
    dates = pd.read_parquet(PRICE, columns=["date"]).date.drop_duplicates().sort_values().reset_index(drop=True)
    alias = alias_table()
    alias.to_csv(SOURCE / "bj_old_new_code_map.csv", index=False, encoding="utf-8-sig")
    alias_lookup = alias.set_index("old_code").new_code.to_dict()
    records = []
    checks = []
    for source in published["sources"]:
        period = source["period"]
        doc = SOURCE / f"capco_{period}_code.pdf"
        rows = rows_from_pdf(doc)
        publication = pd.Timestamp(source["publication_date"])
        later = dates[dates > publication]
        if later.empty:
            raise ValueError(f"No post-publication trading date for {period}")
        usable_date = later.iloc[0]
        bj = [r for r in rows if r["original_code"].startswith(("43", "83", "87", "92"))]
        mapped = 0
        unmapped = []
        names_disagree = []
        for row in bj:
            raw = row["original_code"]
            target = raw if raw.startswith("92") else alias_lookup.get(raw)
            if not target:
                unmapped.append(raw)
                continue
            if not raw.startswith("92"):
                a_name = alias.loc[alias.old_code.eq(raw), "name"].iloc[0]
                normalize = lambda s: re.sub(r"[\s*]", "", s).removeprefix("ST")
                if normalize(a_name) != normalize(row["name"]):
                    names_disagree.append((raw, row["name"], a_name))
            records.append({
                "stockid": "BJ" + target,
                "raw_official_code": raw,
                "official_name": row["name"],
                "sector_code": row["sector_code"],
                "major_code": row["major_code"],
                "industry": row["industry"],
                "source_period": period,
                "publication_date": publication,
                "available_at": usable_date,
                "source_pdf": doc.name,
                "source_page": row["page"],
            })
            mapped += 1
        checks.append({
            "period": period,
            "published": str(publication.date()),
            "usable_from": str(usable_date.date()),
            "all_parsed_rows": len(rows),
            "bj_original_rows": len(bj),
            "bj_mapped_rows": mapped,
            "unmapped_codes": unmapped,
            "alias_name_disagreements": names_disagree[:25],
            "alias_name_disagreement_count": len(names_disagree),
        })
        print(checks[-1], flush=True)
    history = pd.DataFrame(records)
    if history.duplicated(["stockid", "source_period"]).any():
        raise ValueError("Conflicting company classifications within a CAPCO issue")
    history.sort_values(["available_at", "stockid"]).to_parquet(SOURCE / "industry_history.parquet", index=False)
    (SOURCE / "parse_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
