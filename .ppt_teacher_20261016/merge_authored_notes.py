"""Package Artifact Tool-authored notes into the unchanged source package.

No notes XML is synthesized here. This preserves the source's exact charts,
workbooks, slide bodies, masters and all other attachments during a notes edit.
"""
import sys,re,json,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
source,authored,destination=map(Path,sys.argv[1:4])
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(source) as src,zipfile.ZipFile(authored) as new:
    notes=sorted(n for n in src.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n))
    assert len(notes)==25
    for n in notes:
        assert n in new.namelist()
        folder,_,filename=n.rpartition('/')
        rel=folder+'/_rels/'+filename+'.rels'
        def rels(z):
            return sorted((x.get('Type'),x.get('Target'),x.get('TargetMode')) for x in ET.fromstring(z.read(rel)))
        assert rels(src)==rels(new),(n,'notes relationships changed')
        authored_note=ET.fromstring(new.read(n))
        # Export regenerates relationship IDs. These notes have no ID references;
        # slide/master destinations are identical, so retain source relationships.
        rel_ns='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
        assert not any(k.startswith(rel_ns) for e in authored_note.iter() for k in e.attrib),(n,'notes contain relationship-ID references')
        text=''.join(x.text or '' for x in authored_note.findall('.//a:t',ns))
        assert len(text)>200,(n,'notes text missing')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as out:
        for info in src.infolist():
            out.writestr(info,new.read(info.filename) if info.filename in notes else src.read(info.filename))
with zipfile.ZipFile(source) as src,zipfile.ZipFile(destination) as out:
    different=[n for n in src.namelist() if src.read(n)!=out.read(n)]
    assert set(different)==set(notes)
print(json.dumps({'authored_notes_parts':len(notes),'changed_parts':len(different),'other_parts_preserved_exactly':True}))
