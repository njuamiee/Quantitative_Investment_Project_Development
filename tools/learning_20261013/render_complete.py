"""Use the current document skill renderer with Windows-safe profile paths."""
import importlib.util,os,subprocess
from pathlib import Path
path=Path('C:/Users/Amiee/.codex/plugins/cache/openai-primary-runtime/documents/26.1011.11113/skills/documents/render_docx.py')
spec=importlib.util.spec_from_file_location('complete_docx_renderer',path)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)

def run_cmd(cmd,env,verbose):
    fixed=[]
    for arg in cmd:
        if arg.startswith('-env:UserInstallation=file://') and not arg.startswith('-env:UserInstallation=file:///'):
            arg='-env:UserInstallation='+Path(arg.removeprefix('-env:UserInstallation=file://')).as_uri()
        fixed.append(arg)
    if os.name=='nt' and fixed[0]=='soffice':fixed[0]='soffice.com'
    result=subprocess.run(fixed,env=env,check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',timeout=90,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if verbose:print(result.returncode,result.stdout,result.stderr,flush=True)
    return result

renderer._run_cmd=run_cmd
if __name__=='__main__':renderer.main()
