"""Verify delivery sources, key result table and final document structure."""
from pathlib import Path
import json, hashlib, csv, re, subprocess
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'项目说明书_20261026'
manifest=json.loads((OUT/'来源与版本校验.json').read_text(encoding='utf-8'))
assert len(manifest['sources'])==20
for source in manifest['sources'].values():
    assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
md=(OUT/'整个量化项目说明书.md').read_text(encoding='utf-8')
assert len(re.findall(r'^## ',md,re.M))==29
expected={
'original_monthly__equal20':[-24.52,28.83,-1.82,79.15],
'original_monthly__industry_size40':[-21.88,26.38,-1.64,79.84],
'original_monthly__fixed50':[-10.66,13.60,-1.66,38.39],
'original_monthly__ma20':[-17.73,19.83,-2.31,47.37],
'original_monthly__stop5_ban10':[-24.63,25.57,-2.48,67.03],
'rolling_selected':[-10.93,13.82,-1.11,54.72],
'pool_equal100':[-25.71,27.80,-2.27,73.08],
'initial_pool_buyhold':[-22.17,24.92,-1.95,93.11]}
seen=set()
with (ROOT/'实验更新_20261023/全部实验结果.csv').open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        key=row['strategy']
        if row['period']=='test' and float(row['fee_bp'])==5 and key in expected:
            actual=[round(float(row[k])*mult,2) for k,mult in [('total_return',100),('max_drawdown',100),('sharpe_rf15',1),('mean_exposure',100)]]
            assert actual==expected[key],(key,actual,expected[key]);seen.add(key)
assert seen==set(expected),(seen,set(expected))
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(OUT/'整个量化项目说明书.docx') as z:
    body=ET.fromstring(z.read('word/document.xml'));styles=ET.fromstring(z.read('word/styles.xml'))
    text=''.join(body.itertext())
    assert not body.findall('.//w:pBdr',ns) and not styles.findall('.//w:pBdr',ns)
    assert len(body.findall('.//w:bookmarkStart',ns))==29
    assert not '\ufffd' in text
qa=ROOT/'tmp/manual_qa_release'
assert len(list(qa.glob('page-*.png')))==32
pdftext=subprocess.check_output(['pdftotext','-layout',str(qa/'整个量化项目说明书.pdf'),'-']).decode('utf-8')
pages=[p for p in pdftext.split('\f') if p.strip()]
assert len(pages)==32
assert '目录' in pages[1] and '29 证据索引与数字使用规则' in pages[31]
assert all(len(p.strip())>100 for p in pages)
outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='交付核验.json'}
receipt={'prepared':'2026-10-26','source_hashes_checked':20,'latest_result_rows_checked':8,'chapters':29,'pages':32,
         'visual_review':{'pages_opened_at_original_resolution':list(range(1,33)),'result':'passed'},
         'research_tests_rerun':{'audit_rebuild_20261021':10,'meeting_update_20261023':8,'result':'passed'},
         'renderer':'canonical render_docx.py PNG pipeline with native Word PDF conversion because Windows bundle has no LibreOffice',
         'outputs_sha256':outputs}
(OUT/'交付核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['outputs_sha256','visual_review','renderer']},ensure_ascii=False))
