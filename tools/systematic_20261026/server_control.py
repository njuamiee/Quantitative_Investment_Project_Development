"""Disjoint remote compute and checksum-verified result synchronization.

Uses the existing authorized data-server connection without copying credentials.
All remote writes stay below the task's own tempfile directory, never /home/quant.
"""
import os,sys,json,time,shlex,hashlib,zipfile,sqlite3,re
from pathlib import Path
from config import ROOT,RUN,REPORT,dump,digest
sys.path.insert(0,str(ROOT/'tools'))
from audit_server_inventory import connect

FILES=['complete.json','predictions.parquet','fit_audit.json','factor_audit.json','risk_results.npz','risk_registry.json','full_posthoc.npz','recent_posthoc.npz','account_complete.json']
def info():return json.loads((RUN/'server_launch.json').read_text(encoding='utf-8-sig'))
def command(tr,code,timeout=120):
    ch=tr.open_session(timeout=15);ch.settimeout(timeout);ch.exec_command('python3 -c '+shlex.quote(code))
    out=ch.makefile('rb').read().decode();err=ch.makefile_stderr('rb').read().decode();status=ch.recv_exit_status()
    if status:raise RuntimeError('Remote code exit '+str(status)+': '+err[-2000:])
    return json.loads(out) if out.strip() else {}

def upload():
    p=info()['project'];source=ROOT/'tmp/systematic_server_payload.zip';tr,s=connect();s.get_channel().settimeout(180)
    try:
        last=[0]
        def progress(n,total):
            if n-last[0]>=20_000_000 or n==total:print('uploaded_MB',round(n/1e6,1),'of',round(total/1e6,1),flush=True);last[0]=n
        s.put(str(source),p+'/payload.zip',callback=progress)
        code="""import pathlib,zipfile,hashlib,json
p=pathlib.Path(PROJECT);a=p/'payload.zip'
assert hashlib.sha256(a.read_bytes()).hexdigest()==EXPECTED
with zipfile.ZipFile(a) as z:
 for n in z.namelist():
  assert (p/n).resolve().is_relative_to(p.resolve())
 z.extractall(p)
print(json.dumps({'verified':True,'files':len(z.namelist())}))
""".replace('PROJECT',repr(p)).replace('EXPECTED',repr(digest(source)))
        print(command(tr,code),flush=True)
    finally:s.close();tr.close()

def start(workers=40):
    data=info();p=data['project'];tr,s=connect()
    try:
        code="""import pathlib,subprocess,json,os
p=pathlib.Path(PROJECT);run=p/'training_runs/systematic_20261026';run.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
f=open(run/'server.stdout.log','w');e=open(run/'server.stderr.log','w')
r=subprocess.Popen([str(p/'venv/bin/python'),str(p/'tools/systematic_20261026/runner.py'),'--workers',str(WORKERS),'--shard','remote'],cwd=p,env=env,stdout=f,stderr=e,start_new_session=True)
print(json.dumps({'runner_pid':r.pid,'workers':WORKERS,'shard':'remote','expected_remote_configs':6720}))
""".replace('PROJECT',repr(p)).replace('WORKERS',str(workers))
        status=command(tr,code);data.update(status);dump(data,RUN/'server_launch.json');print(status,flush=True)
    finally:s.close();tr.close()

def status():
    p=info()['project'];tr,s=connect()
    try:
        code="""import pathlib,json
p=pathlib.Path(PROJECT)/'training_runs/systematic_20261026'
v=json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else {}
v['stderr_tail']=(p/'server.stderr.log').read_text()[-2000:] if (p/'server.stderr.log').exists() else ''
print(json.dumps(v))
""".replace('PROJECT',repr(p))
        r=command(tr,code);dump(r,RUN/'remote_progress.json');return r
    finally:s.close();tr.close()

def atomic_replace(source,target):
    for k in range(40):
        try:source.replace(target);return
        except PermissionError:
            if k==39:raise
            time.sleep(.25)

def sync(maximum=120):
    data=info();p=data['project'];tr,s=connect();s.get_channel().settimeout(180)
    try:
        known=[x.parent.name for x in (RUN/'models').glob('*/account_complete.json')]
        with s.open(p+'/client_known.json','w') as f:f.write(json.dumps(known))
        code="""import pathlib,json,zipfile,hashlib,time,sqlite3
p=pathlib.Path(PROJECT);run=p/'training_runs/systematic_20261026';known=set(json.loads((p/'client_known.json').read_text()))
new=[x.parent for x in sorted((run/'models').glob('*/account_complete.json')) if x.parent.name not in known][:MAXIMUM]
target=p/'export_results.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
 for folder in new:
  for name in FILES:
   path=folder/name
   if path.exists():z.write(path,folder.name+'/'+name)
progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
db=sqlite3.connect(run/'tasks.sqlite3',timeout=30)
failed=[{'job':a,'error':b} for a,b in db.execute("SELECT job,error FROM tasks WHERE status='failed'").fetchall()];db.close()
print(json.dumps({'jobs':[x.name for x in new],'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'progress':progress,'failed':failed}))
""".replace('PROJECT',repr(p)).replace('MAXIMUM',str(maximum)).replace('FILES',repr(FILES))
        exported=command(tr,code,180);dump(exported['progress'],RUN/'remote_progress.json')
        if exported['jobs']:
            local=ROOT/'tmp/systematic_remote_results.zip';s.get(p+'/export_results.zip',str(local))
            assert digest(local)==exported['sha256']
            markers={}
            with zipfile.ZipFile(local) as z:
                for name in z.namelist():
                    parts=name.split('/');assert len(parts)==2 and re.fullmatch('[0-9a-f]{20}',parts[0]) and parts[1] in FILES
                    job,fn=parts;folder=RUN/'models'/job;folder.mkdir(parents=True,exist_ok=True)
                    content=z.read(name)
                    if fn=='account_complete.json':markers[job]=content;continue
                    temp=folder/(fn+'.download');temp.write_bytes(content);atomic_replace(temp,folder/fn)
            db=sqlite3.connect(RUN/'tasks.sqlite3',timeout=30);db.execute('PRAGMA busy_timeout=30000')
            for job,content in markers.items():
                r=json.loads(content);folder=RUN/'models'/job
                assert digest(folder/'predictions.parquet')==r['predictions_sha256']
                assert digest(folder/'risk_results.npz')==r['results_sha256']
                dump(r,folder/'account_complete.json')
                db.execute('UPDATE tasks SET status=?,actual_fits=?,accounts=?,result=?,error=NULL,updated=? WHERE job=?',
                   ('complete',r['actual_fits'],r['completed_accounts'],json.dumps(r),time.time(),job))
            db.commit();db.close()
        # Combined progress remains alive after the local core shard finishes.
        db=sqlite3.connect(RUN/'tasks.sqlite3',timeout=30)
        for failure in exported.get('failed',[]):
            db.execute("UPDATE tasks SET status='failed',error=?,updated=? WHERE job=? AND status!='complete'",(failure['error'],time.time(),failure['job']))
        db.commit()
        counts=db.execute('SELECT status,count(*),sum(actual_fits),sum(accounts) FROM tasks GROUP BY status').fetchall();db.close()
        remote=exported['progress'];states=[dict(status=a,jobs=b,fits=c,accounts=d) for a,b,c,d in counts]
        dump(dict(timestamp=time.time(),pid=os.getpid(),source='checksum_verified_local_and_server',states=states,
            active_remote=remote.get('active',[]),remote_timestamp=remote.get('timestamp'),
            total_model_configurations=8640,total_registered_strategy_combinations=27993600),RUN/'progress.json')
        result=dict(synced_jobs=len(exported['jobs']),archive_bytes=exported['bytes'],remote_states=remote.get('states'),local_states=states)
        dump(result,RUN/'server_sync_status.json');print(json.dumps(result),flush=True);return result
    finally:s.close();tr.close()

def watch():
    while True:
        try:
            r=sync()
            done=next((v['jobs'] for v in r['local_states'] if v['status']=='complete'),0)
            if done==8640:break
            # Drain an export backlog without an unnecessary extra wait.
            delay=5 if r['synced_jobs']==120 else 120
        except Exception as ex:
            print(type(ex).__name__,str(ex),flush=True);delay=120
        time.sleep(delay)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['upload','start','status','sync','watch']);ap.add_argument('--workers',type=int,default=40);a=ap.parse_args()
    if a.action=='upload':upload()
    elif a.action=='start':start(a.workers)
    elif a.action=='status':print(json.dumps(status(),indent=2))
    elif a.action=='sync':sync()
    else:watch()
