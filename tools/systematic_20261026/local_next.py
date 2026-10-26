"""Start the Windows-only XGBoost expansion after all local core jobs finish."""
import time,json,subprocess,sys,os
from pathlib import Path
from config import RUN,dump

def main():
    here=Path(__file__).parent
    while True:
        count=0
        for p in (RUN/'models').glob('*/account_complete.json'):
            r=json.loads(p.read_text());c=r['config']
            count+=c['window']==252 and c['library']=='daily'
        if count==960:break
        dump(dict(waiting_for_core=True,core_complete=count,required=960),RUN/'local_phase2.json')
        time.sleep(60)
    time.sleep(10)
    with (RUN/'runner_xgboost.stdout.log').open('w',encoding='utf-8') as out,(RUN/'runner_xgboost.stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,str(here/'runner.py'),'--workers','6','--shard','xgboost_extra'],
            stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        dump(dict(waiting_for_core=False,pid=child.pid,shard='xgboost_extra',expected_models=960,started=time.time()),RUN/'local_phase2.json')
        child.wait()

if __name__=='__main__':main()
