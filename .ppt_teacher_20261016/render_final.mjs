import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=import.meta.dirname,R=path.resolve(B,'..');
const final=path.join(R,'组会汇报_20261016/量化选股项目_老师意见修订_25页.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(final));
const dir=path.join(B,'final_slides');await fs.mkdir(dir,{recursive:true});
for(const [i,s] of p.slides.items.entries()){
 const b=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(dir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
console.log(JSON.stringify({slides:p.slides.items.length,rendered:p.slides.items.length}));
