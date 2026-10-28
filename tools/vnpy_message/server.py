"""Task-scoped Linux deployment, using the existing pinned-host SSH connector.

Never prints credential values or connects a broker. Native releases stay in
the dedicated /home/vnpy/quant_message project directory.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from audit_server_inventory import connect, read_credentials

BASE = "/home/quant/demo/quant_message_20261027"
LOCAL = ROOT / "training_runs/vnpy_message"


def as_vnpy(code, timeout=120, python="/home/vnpy/.venv/bin/python"):
    """Authenticate sudo via stdin; never interpolate a credential into a command."""
    t, s = connect()
    try:
        cred = read_credentials(ROOT / "ftp服务器")
        c = t.open_session(timeout=15); c.settimeout(timeout)
        import base64
        encoded = base64.b64encode(code.encode()).decode()
        expression = "exec(__import__('base64').b64decode('" + encoded + "'))"
        command = "sudo -S -p '' -iu vnpy " + shlex.quote(python) + " -c " + shlex.quote(expression)
        c.exec_command(command)
        c.sendall(cred.password + "\n"); c.shutdown_write()
        out = c.makefile("rb").read().decode("utf-8", "replace")
        err = c.makefile_stderr("rb").read().decode("utf-8", "replace")
        return dict(exit_code=c.recv_exit_status(), stdout=out, stderr=err)
    finally:
        s.close(); t.close()


def execute(transport, command, timeout=120):
    c = transport.open_session(timeout=15)
    c.settimeout(timeout)
    c.exec_command(command)
    out = c.makefile("rb").read().decode("utf-8", "replace")
    err = c.makefile_stderr("rb").read().decode("utf-8", "replace")
    code = c.recv_exit_status()
    return dict(exit_code=code, stdout=out, stderr=err)


def source_bundle():
    paths = sorted((ROOT / "quant_message").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("test_message*.py"))
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    return data.getvalue()


def deploy():
    raw = source_bundle()
    sha = hashlib.sha256(raw).hexdigest()
    release = BASE + "/releases/" + sha[:16]
    t, s = connect()
    try:
        for p in (BASE, BASE + "/releases", release):
            try: s.mkdir(p)
            except OSError: s.stat(p)
        with s.open(release + "/source.tar.gz", "wb") as f: f.write(raw)
        code = """
import hashlib,tarfile,pathlib,json
root=pathlib.Path(%r)
p=root/'source.tar.gz'
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
with tarfile.open(p) as t:
 for m in t.getmembers():
  target=(root/m.name).resolve()
  assert target.is_relative_to(root) and m.isfile()
 t.extractall(root,filter='data')
print(json.dumps({'release':str(root),'source_sha256':%r}))
""" % (release, sha, sha)
        result = execute(t, "python3 -c " + shlex.quote(code))
        if result["exit_code"]: raise RuntimeError(result)
        LOCAL.mkdir(parents=True, exist_ok=True)
        state = dict(base=BASE, release=release, source_sha256=sha)
        (LOCAL / "server_release.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps(state))
    finally: s.close(); t.close()


def deploy_native():
    import uuid
    raw = source_bundle(); sha = hashlib.sha256(raw).hexdigest()
    stage = "/tmp/quant-message-" + uuid.uuid4().hex + ".tar.gz"
    t, s = connect()
    try:
        with s.open(stage, "wb") as f: f.write(raw)
        s.chmod(stage, 0o644)  # Code only; no account files or credentials.
    finally: s.close(); t.close()
    release = "/home/vnpy/quant_message/releases/" + sha[:16]
    code = f"""
import pathlib,tarfile,hashlib,json,os
p=pathlib.Path({stage!r}); root=pathlib.Path({release!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha!r}
root.mkdir(parents=True,exist_ok=True)
with tarfile.open(p) as t:
 for m in t.getmembers():
  assert (root/m.name).resolve().is_relative_to(root) and m.isfile()
 t.extractall(root,filter='data')
print(json.dumps({{'release':str(root),'uid':os.getuid(),'sha256':{sha!r}}}))
"""
    result = as_vnpy(code)
    if result["exit_code"]: raise RuntimeError(result)
    LOCAL.mkdir(parents=True, exist_ok=True)
    state = dict(release=release, source_sha256=sha, python="/home/vnpy/.venv/bin/python")
    (LOCAL / "native_release.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(result["stdout"])
    t, s = connect()
    try: s.remove(stage)
    finally: s.close(); t.close()
    return state


def remote_command(command, timeout=120):
    state = json.loads((LOCAL / "server_release.json").read_text())
    t, s = connect()
    try:
        result = execute(t, "cd " + shlex.quote(state["release"]) + " && " + command, timeout)
        print(json.dumps(result, ensure_ascii=False))
        return result
    finally: s.close(); t.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("deploy", "deploy-native", "install-community", "test", "test-adapter", "doctor"))
    args = p.parse_args()
    if args.action == "deploy": deploy(); return
    if args.action == "deploy-native": deploy_native(); return
    env = BASE + "/venv"
    if args.action == "install-community":
        # Official standalone pip runs inside a no-ensurepip venv. Only this
        # task's writable directory changes; the bootstrap digest is recorded.
        import urllib.request
        bootstrap = LOCAL / "pip.pyz"
        if not bootstrap.exists():
            urllib.request.urlretrieve("https://bootstrap.pypa.io/pip/pip.pyz", bootstrap)
        raw = bootstrap.read_bytes()
        t, s = connect()
        try:
            with s.open(BASE + "/pip.pyz", "wb") as f: f.write(raw)
        finally: s.close(); t.close()
        print(json.dumps({"pip_bootstrap_sha256": hashlib.sha256(raw).hexdigest()}))
        result = remote_command("python3 -m venv --without-pip " + shlex.quote(env) + " && " +
            shlex.quote(env + "/bin/python") + " " + shlex.quote(BASE + "/pip.pyz") +
            " install --disable-pip-version-check vnpy==4.5.0 --no-deps", 180)
    elif args.action == "test":
        result = remote_command(shlex.quote(env + "/bin/python") + " -m unittest discover -s tests -p test_message_execution.py -v")
    elif args.action == "test-adapter":
        result = remote_command(shlex.quote(env + "/bin/python") + " -m unittest discover -s tests -p test_message_deltaone.py -v")
    else:
        result = remote_command(shlex.quote(env + "/bin/python") + " -m quant_message.doctor")
    (LOCAL / ("server_" + args.action + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["exit_code"]: sys.exit(result["exit_code"])


if __name__ == "__main__": main()
