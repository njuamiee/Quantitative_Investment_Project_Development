import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=import.meta.dirname,ROOT=path.resolve(B,'..');
const source=path.join(ROOT,'组会汇报_20261016/量化选股项目_原理说明与新项目结果_20261016.pptx');
await fs.copyFile(source,path.join(B,'user_modified_source.pptx'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
await fs.mkdir(path.join(B,'source_slides'),{recursive:true});
const objects=[];
for(const [i,s] of p.slides.items.entries()){
  const blob=await p.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(B,'source_slides',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
  objects.push({page:i+1,id:s.id,shapes:s.shapes.items.map(x=>({id:x.id,text:String(x.text),position:x.position})),tables:s.tables.items.map(x=>({id:x.id}))});
}
await fs.writeFile(path.join(B,'source_objects.json'),JSON.stringify(objects,null,2));
await fs.writeFile(path.join(B,'source_inspect.json'),JSON.stringify(await p.inspect({kind:'slide,layout,textbox,shape,table',maxChars:250000}),null,2));
console.log(JSON.stringify({slides:p.slides.items.length,masters:p.masters.items.length,objects:objects.length}));
