"""Stage only hash-verified replay inputs into the vnpy-owned project folder."""
from pathlib import Path
import hashlib
import json
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT, as_vnpy, connect

REMOTE='/home/vnpy/quant_message/data/replay_20260923_30'


def main():
    folder=ROOT/'training_runs/vnpy_message/native_inputs'
    names=['ticks.parquet','ticks.csv.gz','actions.json','signals.jsonl','audit.json','sha256.json']
    result=as_vnpy(f"""
import pathlib,hashlib,json
p=pathlib.Path({REMOTE!r});p.mkdir(parents=True,exist_ok=True)
print(json.dumps({{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in {names!r} if (p/n).exists()}}))
""")
    if result['exit_code']:raise RuntimeError(result)
    existing=json.loads(result['stdout'])
    for name in names:
        path=folder/name;sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if existing.get(name)==sha:continue
        tmp='/tmp/quant-replay-'+uuid.uuid4().hex
        t,s=connect()
        try:
            s.put(str(path),tmp);s.chmod(tmp,0o644)
        finally:s.close();t.close()
        try:
            result=as_vnpy(f"""
import pathlib,hashlib,shutil
p=pathlib.Path({tmp!r});assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha!r}
dest=pathlib.Path({REMOTE!r})/{name!r};stage=dest.with_suffix(dest.suffix+'.new')
shutil.copyfile(p,stage);stage.chmod(0o600);stage.replace(dest)
""")
            if result['exit_code']:raise RuntimeError(result)
        finally:
            t,s=connect()
            try:s.remove(tmp)
            finally:s.close();t.close()
        print(name,sha,flush=True)


if __name__=='__main__':main()
