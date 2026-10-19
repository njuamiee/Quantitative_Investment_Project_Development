import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const ROOT=path.resolve(import.meta.dirname,'..');
const B=path.join(import.meta.dirname,'visual_revision','with_notes');
const SKILL='C:/Users/Amiee/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const source=path.join(ROOT,'组会汇报_20261016/量化选股项目_教学图解与超额收益_25页_无备注.pptx');
const finalPath=path.join(ROOT,'组会汇报_20261016/量化选股项目_教学图解与超额收益_25页_含逐页讲稿.pptx');
const sha=b=>createHash('sha256').update(b).digest('hex');
const sourceSha=sha(await fs.readFile(source));
const notes=(await fs.readFile(path.join(import.meta.dirname,'visual_speaker_notes.md'),'utf8')).trim().split(/^## 第/gm).filter(Boolean).map(s=>{
 const match=s.match(/^(\d+)页 ([^\n]+)\n+([\s\S]+)$/);if(!match)throw new Error('Invalid notes header');
 return {page:Number(match[1]),title:match[2],text:match[3].trim()};
});
if(notes.length!==25||notes.some((n,i)=>n.page!==i+1||n.text.length<200))throw new Error('Notes coverage incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==25)throw new Error('Unexpected source slide count');
const tables=[],charts=[];
for(const [i,s] of p.slides.items.entries()){
 s.speakerNotes.text=notes[i].text;s.speakerNotes.setVisible(true);
 if(s.tables.items.length)tables.push(i+1);if(s.charts.items.length)charts.push(i+1);
}
await fs.mkdir(B,{recursive:true});
const authored=path.join(B,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(authored);
// The importer loses embedded chart workbooks. Keep every source package part
// unchanged except the 25 complete notes parts authored above with Artifact Tool.
const candidate=path.join(B,'notes_only_candidate.pptx');
const merged=spawnSync(PY,[path.join(import.meta.dirname,'merge_authored_notes.py'),source,authored,candidate],{encoding:'utf8'});
if(merged.status!==0)throw new Error(merged.stderr||merged.stdout);console.log(merged.stdout.trim());
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,pythonExecutable:PY,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:25,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:charts,requiredEmbeddedWorkbookChartOwnerSlides:charts,
 fontPolicy:{basis:'reference',families:['Microsoft YaHei'],referencePath:source,referenceSha256:sourceSha},verifyArtifactToolImport:true,
 receiptPath:path.join(B,'validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(path.join(B,'final_slides'),{recursive:true});const checks=[];
for(const [i,s] of final.slides.items.entries()){
 const file=`slide-${String(i+1).padStart(2,'0')}.png`;
 const bytes=new Uint8Array(await(await final.export({slide:s,format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(path.join(B,'final_slides',file),bytes);
 const same=sha(bytes)===sha(await fs.readFile(path.join(import.meta.dirname,'visual_revision/final_slides',file)));
 const actual=String(s.speakerNotes.text??'');const expected=notes[i].text;
 checks.push({page:i+1,renderMatchesSource:same,notesLength:actual.length,notesMatch:actual===expected});
}
if(checks.some(c=>!c.notesMatch))throw new Error('Notes round-trip mismatch');
if(sha(await fs.readFile(source))!==sourceSha)throw new Error('Source changed');
await fs.writeFile(path.join(B,'checks.json'),JSON.stringify({source,sourceSha,finalPath,checks},null,2));
console.log(JSON.stringify({finalPath:result.finalPath,bytes:result.byteCount,notesPages:checks.length,matchingRenders:checks.filter(c=>c.renderMatchesSource).length,notesCharacters:checks.reduce((a,c)=>a+c.notesLength,0)}));
