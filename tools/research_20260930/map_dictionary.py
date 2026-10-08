"""Use the managed document runtime (openpyxl) to map the provider workbook."""
import csv,io,json,hashlib
from pathlib import Path
import openpyxl
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/teacher9_20260930'
p=ROOT/'training_data/snapshot_20260920/metadata/provider_data_dictionary.csv'
raw=p.read_bytes();(ROOT/'training_data/research_20260930/provider_dictionary.xlsx').write_bytes(raw)
w=openpyxl.load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
rows=list(w.active.iter_rows(values_only=True));header=rows[0]
lookup={(r[0],r[2]):dict(zip(header,r)) for r in rows[1:]}
reg=RUN/'all_factor_registry_quality.csv'
with reg.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
matched=0
for row in records:
    match=lookup.get((row['table'],row['field']))
    row['supplier_meaning']=str(match.get('含义') or '') if match else ''
    row['supplier_type']=str(match.get('类型') or '') if match else ''
    row['supplier_notes']=str(match.get('备注') or '') if match else ''
    row['meaning_status']='dictionary_mapped; publication_frequency_still_unverified' if match else 'not_in_dictionary'
    row['field_role']='numeric_candidate' if row['supplier_type'].startswith(('numeric','float','int','decimal')) else 'metadata'
    matched+=bool(match)
with reg.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
numeric=[r for r in records if r['field_role']=='numeric_candidate']
result={'registry_fields':len(records),'dictionary_matches':matched,'numeric_candidates':len(numeric),
    'metadata_fields':len(records)-len(numeric),'numeric_constant_fields':sum(r['constant']=='True' for r in numeric),
    'numeric_coverage_below70pct':sum(float(r['finite_coverage'])<.7 for r in numeric),
    'dictionary_sha256':hashlib.sha256(raw).hexdigest()}
(RUN/'dictionary_mapping.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
