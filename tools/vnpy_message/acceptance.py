"""One-command native simulation deployment and acceptance as the vnpy user.

Input dataset is the normalized, SHA-recorded five-session sample already staged
under /home/vnpy/quant_message/data. No gateway connection or live order path.
"""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import deploy_native, as_vnpy, LOCAL


def main():
    state = deploy_native()
    job = "accept_" + time.strftime("%Y%m%d_%H%M%S")
    prefix = f"""
import os,sys,json,pathlib
sys.path.insert(0,'/home/vnpy/quant_message/deps')
sys.path.insert(0,{state['release']!r})
os.chdir({state['release']!r})
import pandas as pd
assert pd.__version__=='2.3.3', 'Use project pandas 2.3.3: vendor statistics use chained assignment'
pd.options.mode.copy_on_write=False
"""
    results = []
    code = prefix + """
import unittest,io
suite=unittest.TestSuite()
for pattern in ('test_message_execution.py','test_message_deltaone.py','test_message_portfolio.py'):
 suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=pattern))
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(suite)
print(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=stream.getvalue())))
if not result.wasSuccessful():raise SystemExit(1)
"""
    tested=as_vnpy(code,120)
    (LOCAL/'native_unit_tests.json').write_text(json.dumps(tested,ensure_ascii=False,indent=2),encoding='utf-8')
    if tested['exit_code']:raise RuntimeError(tested)
    # Small simple cases isolate latency/slippage; full case exercises all
    # timestamped portfolio candidates. Only native backtest orders are sent.
    cases = [("simple",100,.01),("simple",1000,.01),("simple",100,.05),("full",100,.01)]
    for kind, latency, slippage in cases:
        name = f"{kind}_{latency}ms_{slippage}"
        remote = "/home/vnpy/quant_message/results/" + job + "/" + name
        code = prefix + f"""
from quant_message.native_validation import run
r=run('/home/vnpy/quant_message/data/replay_20260923_30',{remote!r},kind={kind!r},latency={latency},slippage={slippage})
"""
        result = as_vnpy(code, 240)
        (LOCAL / ("native_" + name + ".log.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if result['exit_code']:
            print(result); raise SystemExit(result['exit_code'])
        # Pull small machine-readable report/curves using the same authorized
        # vnpy identity; its private home need not become publicly readable.
        code = prefix + f"""
p=pathlib.Path({remote!r})
print(json.dumps({{f.name:f.read_text() for f in p.iterdir() if f.suffix in ('.json','.csv')}}))
"""
        fetched = as_vnpy(code, 60)
        if fetched['exit_code']: raise RuntimeError(fetched)
        files = json.loads(fetched['stdout'])
        local = LOCAL / "native" / name; local.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (local / filename).write_text(content, encoding="utf-8")
        report = json.loads(files['report.json'])
        results.append(dict(case=name, native_trades=report['native_trades'], source_sha256=state['source_sha256'],
                            report=str(local / 'report.json'), remote=remote))
        print(json.dumps(results[-1]), flush=True)
    (LOCAL / 'native_acceptance.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__=='__main__':main()
