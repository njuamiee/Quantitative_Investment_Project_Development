"""Inspect publicly served CSI site assets for documented data routes."""
from pathlib import Path
import requests,re,json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'training_data/research_20261018/csi2000/site_assets'
OUT.mkdir(parents=True,exist_ok=True)
session=requests.Session()
response=session.get('https://www.csindex.com.cn/',timeout=30);response.raise_for_status()
(OUT/'index.html').write_text(response.text,encoding='utf-8')
scripts=re.findall(r'<script[^>]+src=["\']([^"\']+)',response.text)
routes=[]
for i,url in enumerate(scripts):
    full=requests.compat.urljoin(response.url,url)
    r=session.get(full,timeout=30);r.raise_for_status();(OUT/f'asset_{i}.js').write_text(r.text,encoding='utf-8')
    hits=[]
    for term in ['sample','constituent','history','adjust','csindex-home']:
        for m in list(re.finditer(term,r.text,re.I))[:80]:hits.append(r.text[max(0,m.start()-120):m.end()+180])
    routes.append({'url':full,'bytes':len(r.content),'contexts':hits})
(OUT/'routes.json').write_text(json.dumps(routes,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(routes,ensure_ascii=False)[:22000])
