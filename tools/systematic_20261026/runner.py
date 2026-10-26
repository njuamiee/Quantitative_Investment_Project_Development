"""Persistent exhaustive task queue. Interrupted jobs resume from actual fit checkpoints."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
import argparse,json,itertools,time,sqlite3,traceback,warnings
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from config import *

def configurations():
    out=[]
    for h,count,model,freq,window,lib in itertools.product(HORIZONS,COUNTS,MODELS,FREQUENCIES,WINDOWS,LIBRARIES):
        out.append(dict(horizon=h,count=count,model=model,frequency=freq,window=window,library=lib))
    # Early results span all horizons/models; no observed performance enters queue order.
    def priority(c):
        core=c['window']==252 and c['library']=='daily'
        return (0 if core else 1, FREQUENCIES.index(c['frequency']),
                COUNTS.index(c['count']),HORIZONS.index(c['horizon']),MODELS.index(c['model']),
                WINDOWS.index(c['window']),LIBRARIES.index(c['library']))
    return sorted(out,key=priority)

def worker(c):
    from train import run_job
    from evaluate import evaluate_job
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        r=run_job(c);return evaluate_job(r)

def main(workers=6,limit=None,shard='all'):
    RUN.mkdir(parents=True,exist_ok=True)
    configs=configurations();db=sqlite3.connect(RUN/'tasks.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS tasks (job TEXT PRIMARY KEY, config TEXT, status TEXT, actual_fits INTEGER DEFAULT 0, accounts INTEGER DEFAULT 0, result TEXT, error TEXT, updated REAL)')
    for c in configs:db.execute('INSERT OR IGNORE INTO tasks(job,config,status,updated) VALUES(?,?,?,?)',(key(c),json.dumps(c,sort_keys=True),'pending',time.time()))
    # A parent interruption can occur after a worker writes its atomic receipt
    # but before the database transaction. Recover those completed results.
    for marker in (RUN/'models').glob('*/account_complete.json'):
        r=json.loads(marker.read_text())
        db.execute('UPDATE tasks SET status=?,actual_fits=?,accounts=?,result=?,error=NULL,updated=? WHERE job=?',
                   ('complete',r['actual_fits'],r['completed_accounts'],json.dumps(r),time.time(),r['job']))
    db.execute("UPDATE tasks SET status='pending' WHERE status='running'");db.commit()
    dump(dict(model_configurations=len(configs),complete_strategy_combinations=27993600,
              workers=workers,created=time.time(),source='full declared Cartesian product; exact h=1 duplicate removed'),RUN/'queue_manifest.json')
    pending=[]
    for c in configs:
        core=c['window']==252 and c['library']=='daily'
        if shard=='core' and not core:continue
        if shard=='expanded' and core:continue
        if shard=='remote' and (core or c['model']=='xgboost'):continue
        if shard=='xgboost_extra' and (core or c['model']!='xgboost'):continue
        status=db.execute('SELECT status FROM tasks WHERE job=?',(key(c),)).fetchone()[0]
        if status not in ['complete','failed']:pending.append(c)
    if limit is not None:pending=pending[:limit]
    def progress(active):
        rows=db.execute('SELECT status,count(*),sum(actual_fits),sum(accounts) FROM tasks GROUP BY status').fetchall()
        dump(dict(timestamp=time.time(),pid=os.getpid(),states=[dict(status=r[0],jobs=r[1],fits=r[2],accounts=r[3]) for r in rows],
                  active=[dict(job=key(c),config=c) for c in active.values()],total_model_configurations=len(configs),
                  total_registered_strategy_combinations=27993600,shard=shard),RUN/'progress.json')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        active={};cursor=0
        while cursor<len(pending) or active:
            while cursor<len(pending) and len(active)<workers:
                c=pending[cursor];cursor+=1;f=executor.submit(worker,c);active[f]=c
                db.execute('UPDATE tasks SET status=?,updated=? WHERE job=?',('running',time.time(),key(c)));db.commit()
            progress(active)
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for f in done:
                c=active.pop(f)
                try:
                    r=f.result();db.execute('UPDATE tasks SET status=?,actual_fits=?,accounts=?,result=?,error=NULL,updated=? WHERE job=?',
                        ('complete',r['actual_fits'],r['completed_accounts'],json.dumps(r),time.time(),key(c)))
                    print(json.dumps(dict(job=key(c),config=c,accounts=r['completed_accounts'],fits=r['actual_fits'],seconds=r['seconds']),ensure_ascii=False),flush=True)
                except Exception:
                    err=traceback.format_exc();db.execute('UPDATE tasks SET status=?,error=?,updated=? WHERE job=?',('failed',err,time.time(),key(c)))
                    print('FAILED',key(c),err,flush=True)
                db.commit()
            progress(active)
    progress({});db.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);a.add_argument('--limit',type=int)
    a.add_argument('--shard',choices=['all','core','expanded','remote','xgboost_extra'],default='all')
    args=a.parse_args();main(args.workers,args.limit,args.shard)
