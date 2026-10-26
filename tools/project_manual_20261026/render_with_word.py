"""Use canonical render_docx.py rasterization with native Word PDF conversion.

The Windows runtime has no bundled LibreOffice. This adapter uses installed Word,
never a desktop LibreOffice installation, and leaves the canonical PNG logic intact.
"""
import importlib.util, subprocess, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CANONICAL=Path('C:/Users/Amiee/.codex/plugins/cache/openai-primary-runtime/documents/26.1022.11436/skills/documents/render_docx.py')
spec=importlib.util.spec_from_file_location('canonical_renderer',CANONICAL)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
def convert(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    pdf=Path(convert_tmp_dir)/(stem+'.pdf')
    command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(Path(__file__).with_name('export_word.ps1')),'-InputDocx',str(Path(doc_path).resolve()),'-OutputPdf',str(pdf)]
    p=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    debug=p.stdout+'\n'+p.stderr
    if verbose:print(debug,flush=True)
    return (str(pdf) if p.returncode==0 and pdf.exists() else '',debug)
renderer.convert_to_pdf=convert
if __name__=='__main__':renderer.main()
