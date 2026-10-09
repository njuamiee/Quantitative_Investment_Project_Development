"""Reuse the previously visually verified document grammar with refreshed lessons."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
source=(ROOT/'tools/beginner_20261001/build_guide.py').read_text('utf-8')
source=source.replace('from content import PAGES','from guide_content import PAGES')
source=source.replace('20261002','20261008').replace('2026-10-02','2026-10-08')
source=source.replace("'training_data/audit_20261008/supplementary_audit.json'","'training_runs/teacher9_20261008/final_summary.json'")
source=source.replace('学习讲义  ·  2026-10-08','学习讲义  ·  2026-10-09')
source=source.replace('金融入门讲义｜2026-10-08','金融入门讲义｜2026-10-09（资料截至10月8日）')
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
