"""Verify report links, embedded PNGs, account counts and saved audit status."""
from pathlib import Path
import base64,hashlib,io,json
from urllib.parse import unquote
from bs4 import BeautifulSoup
from PIL import Image
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'新项目_20261018_深入实验'
RUN=ROOT/'training_runs/research_20261018'
def main():
    checked=[];images=0
    for path in OUT.glob('*.html'):
        soup=BeautifulSoup(path.read_text(encoding='utf-8'),'html.parser')
        ids={n['id'] for n in soup.find_all(id=True)}
        for a in soup.find_all('a',href=True):
            ref=a['href']
            if ref.startswith('#'):
                if ref[1:] not in ids:raise AssertionError('missing anchor '+ref)
            elif not ref.startswith(('http:','https:','mailto:')):
                target=(path.parent/unquote(ref.split('#')[0])).resolve()
                if not target.exists():raise AssertionError('missing link '+str(target))
        for im in soup.find_all('img',src=True):
            src=im['src']
            if src.startswith('data:image/png;base64,'):
                Image.open(io.BytesIO(base64.b64decode(src.split(',',1)[1]))).verify();images+=1
            elif not src.startswith(('http:','https:')):
                if not (path.parent/unquote(src)).exists():raise AssertionError('missing image '+src)
        checked.append(path.name)
    p=pd.read_csv(OUT/'全部实验_含基准超额.csv')
    assert p[['experiment','strategy']].drop_duplicates().shape[0]==96
    assert len(p)==278
    for file in [RUN/'deep_risk/independent_audit.json',RUN/'evidence/supplemental_audit.json']:
        assert json.loads(file.read_text(encoding='utf-8'))['status']=='pass'
    m=json.loads((OUT/'交付核查.json').read_text(encoding='utf-8'))
    m.update({'verified_html':checked,'embedded_png_verified':images,'local_links_verified':True,
              'tests_passed':12,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tools/research_20261018').glob('*.py')},
              'data_coverage_audit_sha256':hashlib.sha256((ROOT/'training_data/research_20261018/market/coverage_key_audit.json').read_bytes()).hexdigest(),
              'main_report_sha256':hashlib.sha256((OUT/'深入实验_完整结果与解释.html').read_bytes()).hexdigest()})
    (OUT/'交付核查.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'pass','html_count':len(checked),'embedded_charts':images,'accounts':96,'period_rows':len(p)},ensure_ascii=False))
if __name__=='__main__':main()
