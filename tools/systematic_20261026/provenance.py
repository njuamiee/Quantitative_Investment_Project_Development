"""Fingerprints only: no raw vendor data or credentials are exported."""
import json,platform,importlib.metadata
from datetime import datetime
from config import RUN,digest,dump

def main():
    files=[]
    for path in sorted((RUN/'data').glob('*')):
        if path.is_file():files.append(dict(path=str(path.relative_to(RUN)).replace('\\','/'),
            bytes=path.stat().st_size,sha256=digest(path)))
    minute=RUN/'minute/first30.npz'
    if minute.exists():files.append(dict(path='minute/first30.npz',bytes=minute.stat().st_size,sha256=digest(minute)))
    result=dict(created=datetime.now().astimezone().isoformat(),scope='prepared frozen matrices and minute execution inputs',
        files=files,local_runtime=dict(python=platform.python_version(),platform=platform.system(),
            packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn',
                'lightgbm','xgboost','catboost','hmmlearn','numba','llvmlite','pyarrow']}),
        note='A fingerprint verifies the input vintage; it does not prove the historical publication time of vendor fields.')
    dump(result,RUN/'input_fingerprints.json')
    print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in files),fingerprints_written=True)))

if __name__=='__main__':main()
