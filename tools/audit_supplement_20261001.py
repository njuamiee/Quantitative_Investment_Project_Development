"""Read-only learning-guide update; reuse the established sample audit in a new folder."""
from pathlib import Path
import json,hashlib,re,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
# The earlier code reads fixed public data samples and service README only.
# Redirect its output explicitly; preserve the September snapshot.
source=(ROOT/'tools/audit_supplement_20260930.py').read_text(encoding='utf-8')
source=source.replace("ROOT / 'training_data/audit_20260930'","ROOT / 'training_data/audit_20261001'")
scope={'__name__':'__main__','__file__':str(__file__)}
exec(compile(source,'audit_supplement_20261001_reused','exec'),scope)
