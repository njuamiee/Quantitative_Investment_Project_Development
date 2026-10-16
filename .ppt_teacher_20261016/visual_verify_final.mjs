import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=path.join(import.meta.dirname,'visual_revision'),R=path.resolve(B,'../..');
const file=path.join(R,'组会汇报_20261016/量化选股项目_教学图解与超额收益_25页_无备注.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(file));
await fs.mkdir(path.join(B,'final_slides'),{recursive:true});
const checks=[];const hash=b=>createHash('sha256').update(b).digest('hex');
for(const [i,s] of p.slides.items.entries()){
 const name=`slide-${String(i+1).padStart(2,'0')}.png`;
 const bytes=new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(path.join(B,'final_slides',name),bytes);
 const same=hash(bytes)===hash(await fs.readFile(path.join(B,'slides',name)));
 checks.push({page:i+1,renderMatchesReviewed:same,notes:String(s.speakerNotes.text??'')});
}
await fs.writeFile(path.join(B,'final_render_check.json'),JSON.stringify(checks,null,2));
console.log(JSON.stringify({slides:checks.length,matching:checks.filter(x=>x.renderMatchesReviewed).length,notes:checks.filter(x=>x.notes.trim()).length}));
