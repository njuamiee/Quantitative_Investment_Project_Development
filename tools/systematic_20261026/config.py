"""Registered research space. Never infer completed work from this registry."""
from pathlib import Path
import hashlib, json, itertools, os, time, uuid

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'training_runs/systematic_20261026'
REPORT = ROOT / '系统重设计_20261026'
DATA = ROOT / 'training_data/audit_rebuild_20261021/original_inputs'
START = '2025-07-02'
END = '2026-10-19'
HORIZONS = [1, 2, 3, 4, 5, 10, 15, 20]
COUNTS = [20, 40, 60, 80, 100]
MODELS = ['ridge', 'ic', 'conditional_ic', 'lightgbm', 'ranker', 'xgboost', 'extra_trees', 'catboost']
FREQUENCIES = ['quarterly', 'monthly', 'daily']
WINDOWS = [126, 252, 504]
LIBRARIES = ['daily', 'daily_auto', 'vendor_asof']
SEED = 20261026

def atomic_replace(temp, path):
    for attempt in range(40):
        try:
            Path(temp).replace(path);return
        except OSError as ex:
            # Windows readers can report EINVAL as well as access denied for
            # a destination that is temporarily open without delete sharing.
            transient=isinstance(ex,PermissionError) or (os.name=='nt' and ex.errno in (13,22))
            if not transient or attempt==39:raise
            time.sleep(.25)

def temporary_path(path):
    path=Path(path)
    return path.with_name(path.stem+'.'+str(os.getpid())+'.'+uuid.uuid4().hex[:8]+'.tmp'+path.suffix)

def write_csv(frame,path,**kwargs):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=temporary_path(path)
    frame.to_csv(temp,index=False,encoding='utf-8-sig',**kwargs)
    atomic_replace(temp,path)

def dump(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex[:8] + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    # Windows readers/antivirus may briefly hold a destination handle without
    # delete sharing. That must not kill a multi-day research queue.
    atomic_replace(temp,path)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def key(c):
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:20]

def registry():
    return dict(horizons=HORIZONS, factors=COUNTS, models=MODELS,
                refit=FREQUENCIES, training_sessions=WINDOWS, libraries=LIBRARIES,
                outer_start=START, data_end=END, seed=SEED,
                label='adjusted open(t+h+1)/adjusted open(t+1)-1',
                factor_screen='quarterly, mature training labels only; coverage>=70%, IC shrinkage, abs correlation<0.98',
                objective='Sharpe rf1.5% >1, geometric Calmar >2, maximum drawdown <0.30',
                historical_data_previously_inspected=True,
                excluded_models=['Jev/XJ: user excluded 2026-10-26'],
                explicit_horizon_change='user: after 5 sessions only 10,15,20')
