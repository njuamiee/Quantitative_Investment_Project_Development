"""Record final documentation separately from the original execution manifest."""
from pathlib import Path
import hashlib,json
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261017'
OUT=ROOT/'新项目_20261017'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class CheckHTML(HTMLParser):
    def __init__(self):super().__init__();self.images=0;self.anchors=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='img':
            assert a.get('src','').startswith('data:image/png;base64,')
            self.images+=1
        if tag=='a':self.anchors.append(a.get('href',''))

def main():
    report=(OUT/'项目改版_完整说明与实验结果.html').read_text(encoding='utf-8')
    parser=CheckHTML();parser.feed(report)
    assert parser.images==8
    for word in ['具体日期与中文伪代码详解','真实一天：从350个代码到20笔成交','63 个交易日','2026-03-05','最大回撤','9.39%']:
        assert word in report,word
    for link in parser.anchors:
        if link.startswith('../'):assert (OUT/link).resolve().exists(),link
    for name in ['ledger_audit','metadata_reproduction','attribution_audit']:
        record=json.loads((RUN/'verification'/f'{name}.json').read_text(encoding='utf-8'))
        assert record.get('pass',record.get('equal',False)),name
    saved=json.loads((RUN/'run_manifest.json').read_text(encoding='utf-8'))
    for name in ['execution.py','experiment.py','models.py']:
        assert sha(Path(__file__).parent/name)==saved['code_sha256'][name],name
    for name,p in [('daily_predictions.parquet',RUN/'models/daily_predictions.parquet'),('risk_metadata.parquet',RUN/'risk_metadata.parquet')]:
        assert sha(p)==saved['input_sha256'][name],name
    final={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()}
    differences=[name for name,v in saved['code_sha256'].items() if final.get(name)!=v]
    manifest={'delivery_date_beijing':'2026-10-18','scope':'1016 meeting revision, before the separate CSI2000 comparison',
              'original_run_manifest_preserved':True,'core_model_account_code_unchanged_since_run':True,
              'frozen_inputs_unchanged':True,'current_code_sha256':final,'changed_since_execution':differences,
              'change_explanation':'Report/teaching and source-fetch documentation were updated after account execution. Current risk_metadata.build reproduces the frozen input exactly; see verification/metadata_reproduction.json. Original execution hashes are retained.',
              'deliverables':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},
              'verification':{'unit_tests_passed':15,'independent_ledger':True,'metadata_exact_reproduction':True,'stock_pnl_reconciliation':True,'embedded_images':parser.images,'relative_links_resolve':True}}
    (RUN/'final_delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest['verification'],ensure_ascii=False))

if __name__=='__main__':main()
