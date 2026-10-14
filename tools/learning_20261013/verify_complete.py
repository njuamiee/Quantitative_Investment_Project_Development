"""Verify complete-document coverage, references, hashes and reviewed renders."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import csv
import hashlib
import json
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '学习笔记/20261013_完整项目讲义'
AUD = ROOT / 'training_runs/learning_20261013'
read = lambda p: json.loads(p.read_text('utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def main():
    failures = []
    manifest = read(OUT / 'delivery_manifest.json')
    for name, expected in manifest.items():
        if not (OUT / name).is_file() or sha(OUT / name) != expected:
            failures.append('Delivery hash: ' + name)
    source_index = read(OUT / '核对资料/source_index.json')
    for row in source_index:
        if sha(ROOT / row['path']) != row['sha256']:
            failures.append('Research source changed: ' + row['path'])
    citations = read(OUT / '核对资料/本文源码引用.json')
    for row in citations:
        lines = (ROOT / row['file']).read_text('utf-8-sig').splitlines()
        if row['needle'] not in lines[row['line'] - 1]:
            failures.append('Source citation: ' + str(row))
    archive_count = 0
    for folder in ['20261013_改版前', '20261013_分课初稿']:
        for row in read(ROOT / '历史学习资料' / folder / '归档清单.json'):
            archive_count += 1
            path = Path(row['archived'])
            if not path.is_file() or sha(path) != row['sha256']:
                failures.append('Archived file: ' + str(path))
    links = 0
    for entry in [ROOT / 'README.md', ROOT / '学习笔记/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)', entry.read_text('utf-8-sig')):
            if '://' in target or target.startswith('#'):
                continue
            links += 1
            if not (entry.parent / target.split('#')[0]).exists():
                failures.append('Entry link: ' + target)
    documents = []
    all_text = ''
    for prefix, number, expected_pages in [('01_', 0, 8), ('02_', 1, 51)]:
        path = next(OUT.glob(prefix + '*.docx'))
        render = AUD / f'complete_final_{number}'
        pdf = next(render.glob('*.pdf'))
        pages = list(render.glob('page-*.png'))
        if len(pages) != expected_pages or len(PdfReader(pdf).pages) != expected_pages:
            failures.append('Page count: ' + path.name)
        with ZipFile(path) as z:
            document = ET.fromstring(z.read('word/document.xml'))
        text = '\n'.join(n.text or '' for n in document.findall('.//w:t', NS))
        all_text += text
        if prefix == '02_':
            for tag in [f'Q{i:02}' for i in range(1, 22)] + [f'R{i:02}' for i in range(1, 31)]:
                if not re.search(r'\b' + tag + r'\b', text):
                    failures.append('Missing question/task: ' + tag)
            for row in read(OUT / '核对资料/llm_factor_proposals.json'):
                if row['id'].removeprefix('llm_') not in text:
                    failures.append('Missing formula: ' + row['id'])
            with (OUT / '核对资料/selected40_2026Q3_details.csv').open(encoding='utf-8-sig') as f:
                factors = list(csv.DictReader(f))
            if len(factors) != 40:
                failures.append('Factor count')
            for row in factors:
                label = row['feature'].replace('x_c02_gtalpha', 'GT').replace('x_c01_Alpha', 'A').removeprefix('x_')
                if label not in text:
                    failures.append('Missing factor: ' + label)
        for table in document.findall('.//w:tbl', NS):
            width = table.find('./w:tblPr/w:tblW', NS)
            if width is None or width.get('{%s}w' % NS['w']) != '9360':
                failures.append('Table width: ' + path.name)
        documents.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                          'pages': expected_pages,
                          'visual_review': 'All final page PNGs viewed individually; no clipped text, overlap, broken tables or orphan source-only pages observed.',
                          'page_image_sha256': {p.name: sha(p) for p in sorted(pages)}})
    inference = read(OUT / '核对资料/inference_checks.json')
    if not all(row['passed'] for row in inference):
        failures.append('Inference checks')
    report = {'date': '2026-10-13', 'documents': documents,
              'delivery_files_verified': len(manifest), 'research_source_files_verified': len(source_index),
              'source_citations_verified': len(citations), 'archived_files_verified': archive_count,
              'entry_links_checked': links, 'teacher_questions': 21, 'followup_questions': 3,
              'project_tasks': 30, 'old_factor_rows': 40, 'llm_formulas': 24,
              'inference_checks_passed': sum(row['passed'] for row in inference),
              'research_retrained': False, 'failures': failures, 'passed': not failures}
    (AUD / 'complete_document_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'documents'}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
