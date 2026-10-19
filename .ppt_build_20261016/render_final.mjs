import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root=path.resolve(import.meta.dirname,'..');
const file=path.join(root,'组会汇报_20261016','量化选股项目_原理说明与新项目结果_20261016.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(file));
const dir=path.join(import.meta.dirname,'final_slides');
await fs.mkdir(dir,{recursive:true});
for(const [i,s] of p.slides.items.entries()){
  const png=await p.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(dir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({file,slides:p.slides.items.length,rendered:p.slides.items.length}));
