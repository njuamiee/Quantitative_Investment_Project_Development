"""Paired database replay, original-loader migration and future perturbation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import as_vnpy, deploy_native, LOCAL


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['base','prefix','mutated','all'],default='all')
    p.add_argument('--reuse-release',action='store_true');args=p.parse_args()
    state=json.loads((LOCAL/'native_release.json').read_text()) if args.reuse_release else deploy_native()
    job='db_'+time.strftime('%Y%m%d_%H%M%S')
    folder=LOCAL/'database_verified';folder.mkdir(exist_ok=True)
    prefix=f"""
import sys,os,json,pathlib
sys.path.insert(0,'/home/vnpy/quant_message/deps');sys.path.insert(0,{state['release']!r})
os.chdir({state['release']!r})
"""
    cases=[]
    variants=['base','prefix','mutated'] if args.variant=='all' else [args.variant]
    for kind in ('simple','full'):
        for execution in ('native','paper'):
            for variant in variants:
                cases.append((kind,execution,variant,1))
        if 'base' in variants:cases.append((kind,'native','base',5))
    def work(case):
        kind,execution,variant,depth=case
        name=f'{kind}_{execution}_{variant}_d{depth}'
        remote=f'/home/vnpy/quant_message/results/{job}/{name}'
        data='/home/vnpy/quant_message/data/database_'+('mutated' if variant=='mutated' else 'base')
        end='2026-09-24' if variant=='prefix' else None
        code=prefix+f"""
from quant_message.native_validation import run
run({data!r},{remote!r},kind={kind!r},execution={execution!r},end_day={end!r},depth={depth})
"""
        r=as_vnpy(code,600)
        (folder/(name+'.log.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        if r['exit_code']:raise RuntimeError(name+': '+r['stderr'][-1800:])
        fetched=as_vnpy(prefix+f"p=pathlib.Path({remote!r});print(json.dumps({{f.name:f.read_text() for f in p.iterdir() if f.suffix in ('.csv','.json')}}))",60)
        if fetched['exit_code']:raise RuntimeError(fetched)
        local=folder/name;local.mkdir(exist_ok=True)
        for f,content in json.loads(fetched['stdout']).items():
            (local/f).write_text(content,encoding='utf-8')
        metadata=dict(case=name,remote=remote,release=state['release'],source_sha256=state['source_sha256'])
        (local/'execution_manifest.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(work,c) for c in cases]
        for f in as_completed(futures):f.result()


if __name__=='__main__':main()
