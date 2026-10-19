import collections
import json
import re
from pathlib import Path

import fitz

root = Path(__file__).resolve().parents[2] / "training_data/research_20261017/industry"
for f in sorted(root.glob("capco_*_code*.pdf")):
    d = fitz.open(f)
    chunks = [p.get_text() for p in d]
    s = "\n".join(chunks)
    codes = re.findall(r"(?m)^([0-9]{6})$", s)
    print(f.name, "pages", len(d), "codes", len(codes), "prefixes", collections.Counter(c[:2] for c in codes).most_common(10))
    print("first", json.dumps(chunks[0][:600], ensure_ascii=True))
    print("last", json.dumps(chunks[-1][-500:], ensure_ascii=True))
