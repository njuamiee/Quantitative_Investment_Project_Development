"""Read-only October 2 sample audit, preserving earlier snapshots."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
source=(ROOT/'tools/audit_supplement_20260930.py').read_text(encoding='utf-8')
source=source.replace("ROOT / 'training_data/audit_20260930'","ROOT / 'training_data/audit_20261002'")
exec(compile(source,'audit_supplement_20261002_reused','exec'),{'__name__':'__main__','__file__':str(__file__)})
