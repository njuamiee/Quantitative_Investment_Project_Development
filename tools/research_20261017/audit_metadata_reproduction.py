"""Read-only reproduction of the frozen risk descriptor input."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from risk_metadata import build

frozen_path=ROOT/'training_runs/research_20261017/risk_metadata.parquet'
output=ROOT/'training_runs/research_20261017/verification/metadata_reproduction.json'
fresh,coverage=build()  # build() is read-only; deliberately do not run main().
frozen=pd.read_parquet(frozen_path)
same_cols=list(fresh.columns)==list(frozen.columns)
same_rows=len(fresh)==len(frozen)
key_equal=same_rows and fresh[['date','stockid']].equals(frozen[['date','stockid']])
details={}
all_equal=same_cols and same_rows and key_equal
for name in sorted(set(fresh.columns)|set(frozen.columns)):
    if name not in fresh or name not in frozen:
        details[name]={'fresh_present':name in fresh,'frozen_present':name in frozen}
        all_equal=False
        continue
    x=fresh[name].reset_index(drop=True)
    y=frozen[name].reset_index(drop=True)
    dtype_equal=str(x.dtype)==str(y.dtype)
    if not same_rows:
        details[name]={'dtype_equal':dtype_equal,'comparison':'row count differs'}
        all_equal=False
        continue
    null_x=x.isna();null_y=y.isna()
    null_mismatch=int((null_x!=null_y).sum())
    nonnull=~(null_x|null_y)
    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        a=pd.to_numeric(x[nonnull],errors='coerce').to_numpy(dtype=float)
        b=pd.to_numeric(y[nonnull],errors='coerce').to_numpy(dtype=float)
        different=int((a!=b).sum())
        max_abs_difference=float(np.max(np.abs(a-b))) if len(a) else 0.
        item={'kind':'numeric','dtype_fresh':str(x.dtype),'dtype_frozen':str(y.dtype),
              'null_mismatch':null_mismatch,'value_mismatch':different,'max_abs_difference':max_abs_difference}
    elif pd.api.types.is_datetime64_any_dtype(x) and pd.api.types.is_datetime64_any_dtype(y):
        different=int((x[nonnull].array!=y[nonnull].array).sum())
        item={'kind':'datetime','dtype_fresh':str(x.dtype),'dtype_frozen':str(y.dtype),
              'null_mismatch':null_mismatch,'value_mismatch':different}
    else:
        different=int((x[nonnull].astype(str).to_numpy()!=y[nonnull].astype(str).to_numpy()).sum())
        item={'kind':'string_or_object','dtype_fresh':str(x.dtype),'dtype_frozen':str(y.dtype),
              'null_mismatch':null_mismatch,'value_mismatch':different}
    details[name]=item
    if null_mismatch or different or not dtype_equal:all_equal=False
result={
    'equal':bool(all_equal),'same_column_order':same_cols,'same_row_count':same_rows,
    'same_date_stock_keys_in_order':key_equal,
    'fresh_rows':len(fresh),'frozen_rows':len(frozen),
    'fresh_columns':list(fresh.columns),'frozen_columns':list(frozen.columns),
    'frozen_sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
    'coverage_build':coverage,'columns':details,
    'operation':'risk_metadata.build() in memory; original parquet read only; risk_metadata.main() not called',
}
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('columns','coverage_build','fresh_columns','frozen_columns')},ensure_ascii=False,indent=2))
if not all_equal:
    print(json.dumps({k:v for k,v in details.items() if v.get('null_mismatch') or v.get('value_mismatch') or v.get('dtype_fresh')!=v.get('dtype_frozen')},ensure_ascii=False,indent=2))
