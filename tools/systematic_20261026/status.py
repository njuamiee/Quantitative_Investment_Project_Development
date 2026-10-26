"""Read local receipts and checkpoints without changing or restarting any job."""
import json,sqlite3,time
import numpy as np
from config import RUN

def main():
    db=sqlite3.connect(RUN/'tasks.sqlite3',timeout=30)
    counts=db.execute('SELECT status,count(*),sum(actual_fits),sum(accounts) FROM tasks GROUP BY status').fetchall()
    local=[]
    for job,text in db.execute("SELECT job,config FROM tasks WHERE status='running'"):
        c=json.loads(text);file=RUN/'models'/job/'checkpoint.npz';folds=None
        try:
            if file.exists():
                with np.load(file) as a:folds=int(a['done'])
        except (OSError,ValueError):pass
        local.append(dict(job=job,config=c,checkpoint_folds=folds))
    db.close()
    remote_file=RUN/'remote_progress.json';remote={}
    if remote_file.exists():
        p=json.loads(remote_file.read_text());states={r['status']:r['jobs'] for r in p.get('states',[])}
        remote=dict(heartbeat_age_seconds=round(time.time()-p.get('timestamp',0)),
            completed=states.get('complete',0),running=len(p.get('active',[])),failed=states.get('failed',0))
    p=RUN/'local_phase2.json'
    print(json.dumps(dict(completed_result_states=[dict(status=a,models=b,fits=c,accounts=d) for a,b,c,d in counts],
        planned_models=8640,planned_retained_fits=967680,planned_accounts=27993600,
        local_active=local,remote=remote,local_phase=json.loads(p.read_text()) if p.exists() else None,
        note='Fits count completed configurations; partial checkpoint fits are shown separately. Configuration progress is not elapsed-time progress.'),indent=2))

if __name__=='__main__':main()
