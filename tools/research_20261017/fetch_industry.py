"""Fetch public, dated CAPCO industry-classification PDFs with provenance."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_data/research_20261017/industry"
BASE = "https://www.capco.org.cn/dataFile/"
ARTICLES = [
    ("2024H1", "2024-09-30", "zsxfb/hyfl/hyfljg/7ec325221eb64b409772b5527c7f16a5.js"),
    ("2024H2", "2025-04-18", "zsxfb/hyfl/hyfljg/a34890d258e6473b8b61593ffeb8a9cf.js"),
    ("2025H1", "2025-09-30", "zsxfb/hyfl/hyfljg/29adc9e66d844d6fba0001bb2cdc529c.js"),
    ("2025H2", "2026-04-03", "zsxfb/hyfl/hyfljg/d9221cd5eb1448e5956f0e9d84ecba89.js"),
    ("2026H1", "2026-09-30", "zsxfb/hyfl/hyfljg/2e1b516684544cc7be7eab6f93f7dddc.js"),
]
ORIGINAL_PAGES = {
    "2024H1": "https://www.capco.org.cn/xhgg/hyfl/hyfljg/202409/20240930/j_2024093015525600017276828969068447.html",
    "2025H1": "https://www.capco.org.cn/pub/zgssgsxh/xhgg/hyfl/hyfljg/202509/20250930/j_2025093014371200017592143530187679.html",
}


def fetch(session, url, target, magic=None):
    response = session.get(url, timeout=90)
    response.raise_for_status()
    body = response.content
    if magic and not body.startswith(magic):
        raise ValueError(f"Unexpected content at {url}: {body[:80]!r}")
    target.write_bytes(body)
    return {"url": url, "path": str(target.relative_to(ROOT)), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    sources = []
    for key, date, relative in ARTICLES:
        article_url = BASE + relative
        article_file = OUT / f"capco_{key}_article.js"
        article_meta = fetch(session, article_url, article_file, b"export const allData = ")
        article = json.loads(article_file.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        if article["date"] != date:
            raise ValueError(f"Publication date mismatch for {key}: {article['date']} != {date}")
        links = [(a.get_text(" ", strip=True), a.get("href", "").strip()) for a in BeautifulSoup(article["text"], "html.parser").find_all("a")]
        attachments = []
        for name, url in links:
            if not url.lower().endswith(".pdf"):
                continue
            if "股票代码排序" in name:
                kind = "code"
            elif "新旧代码对照" in name:
                kind = "code_alias"
            else:
                continue
            target = OUT / f"capco_{key}_{kind}.pdf"
            attachments.append({"name": name, "kind": kind, **fetch(session, url, target, b"%PDF")})
        if not any(x["kind"] == "code" for x in attachments):
            raise ValueError(f"No company-code PDF found for {key}: {links}")
        sources.append({"period": key, "publication_date": date, "usable_from_rule": "next trading session after publication date", "original_page": ORIGINAL_PAGES.get(key), "current_index": "https://www.capco.org.cn/xhgg/hyfl/hyfljg/index.html", "article": article_meta, "attachments": attachments})
        print(key, date, [(a["kind"], a["bytes"]) for a in attachments], flush=True)
    (OUT / "sources.json").write_text(json.dumps({"downloaded_utc": datetime.now(timezone.utc).isoformat(), "publisher": "中国上市公司协会", "sources": sources}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
