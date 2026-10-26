"""Finite background report writer; exits after the registered queue finishes."""
import os,time,json,subprocess,sys
from pathlib import Path
from config import RUN,REPORT,dump

def main():
    last=-1;failures=0;here=Path(__file__).parent
    while True:
        try:
            p=json.loads((RUN/'progress.json').read_text())
            states={r['status']:r for r in p['states']};done=states.get('complete',{}).get('jobs',0)
            terminal=not states.get('pending',{}).get('jobs',0) and not states.get('running',{}).get('jobs',0)
            stale=time.time()-p['timestamp']>900
            if done!=last or terminal:
                r=subprocess.run([sys.executable,str(here/'build_report.py')],capture_output=True,text=True,
                                 encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                (RUN/'report_watcher_last.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
                if r.returncode:failures+=1
                else:last=done;failures=0
                print(time.strftime('%Y-%m-%d %H:%M:%S'),'report',done,'exit',r.returncode,flush=True)
            if terminal:
                a=subprocess.run([sys.executable,str(here/'audit.py')],capture_output=True,text=True,encoding='utf-8',errors='replace',
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                (RUN/'final_audit.log').write_text(a.stdout+'\n'+a.stderr,encoding='utf-8')
                dump(dict(completed=done,failed=states.get('failed',{}).get('jobs',0),audit_exit=a.returncode,
                          fully_completed=done==8640 and a.returncode==0,timestamp=time.time()),REPORT/'批次结束状态.json')
                break
            if stale:
                dump(dict(reason='worker heartbeat older than 15 minutes; computer sleep or worker interruption possible',
                          last_progress=p['timestamp'],requires_resume=True),REPORT/'运行需要检查.json')
            if failures>=3:
                dump(dict(reason='report build failed three consecutive times; read report_watcher_last.log',requires_fix=True),REPORT/'运行需要检查.json')
                break
        except Exception as ex:
            print(type(ex).__name__,str(ex),flush=True)
        time.sleep(300)

if __name__=='__main__':main()
