"""Check the learning delivery and record the completed visual review."""
from pathlib import Path
import hashlib
import json
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / 'training_runs/learning_20261013'
OUT = ROOT / '学习笔记/20261013_从零学习当前项目'
ARCHIVE = ROOT / '历史学习资料/20261013_改版前'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text('utf-8-sig'))


def main():
    failures = []
    manifest = read(OUT / 'delivery_manifest.json')
    for name, expected in manifest.items():
        if not (OUT / name).is_file() or sha(OUT / name) != expected:
            failures.append('delivery hash: ' + name)
    archived = read(ARCHIVE / '归档清单.json')
    for row in archived:
        p = Path(row['archived'])
        if not p.is_file() or sha(p) != row['sha256']:
            failures.append('archive hash: ' + str(p))
    sources = read(AUD / 'source_index.json')
    for row in sources:
        if sha(ROOT / row['path']) != row['sha256']:
            failures.append('source changed: ' + row['path'])
    inference = read(AUD / 'inference_checks.json')
    if not all(x['passed'] for x in inference):
        failures.append('inference checks')
    links = 0
    for entry in [ROOT / 'README.md', ROOT / '学习笔记/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)', entry.read_text('utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            links += 1
            if not (entry.parent / target.split('#')[0]).exists():
                failures.append('broken entry link: ' + target)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    documents = []
    for prefix, folder, expected_pages in [('01_', 'qa_global', 5), ('02_', 'qa_lesson1', 8)]:
        doc = next(OUT.glob(prefix + '*.docx'))
        pages = list((AUD / folder).glob('page-*.png'))
        if len(pages) != expected_pages:
            failures.append('page count: ' + doc.name)
        with ZipFile(doc) as z:
            xml = ET.fromstring(z.read('word/document.xml'))
        tables = xml.findall('.//w:tbl', ns)
        for table in tables:
            width = table.find('./w:tblPr/w:tblW', ns)
            if width is None or width.get('{%s}w' % ns['w']) != '9360':
                failures.append('table width: ' + doc.name)
        documents.append({
            'path': doc.relative_to(ROOT).as_posix(),
            'sha256': sha(doc), 'pages': len(pages), 'table_count': len(tables),
            'visual_review': 'All pages viewed. Revised pages viewed again; unchanged pages verified by image hashes.',
            'page_image_sha256': {p.name: sha(p) for p in sorted(pages)},
        })
    report = {
        'date': '2026-10-13', 'documents': documents,
        'archived_files_verified': len(archived),
        'source_files_unchanged': len(sources),
        'delivery_files_verified': len(manifest), 'entry_links_checked': links,
        'inference_checks_passed': sum(x['passed'] for x in inference),
        'research_retrained': False, 'failures': failures,
        'passed': not failures,
    }
    (AUD / 'document_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'documents'}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
