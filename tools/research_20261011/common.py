from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'training_data/research_20261011'
RUN=ROOT/'training_runs/research_20261011'
DATA.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
HORIZONS=[1,2,5,10,20]
SEED=20261011
def dump(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
