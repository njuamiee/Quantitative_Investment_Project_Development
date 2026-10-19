import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const ROOT=path.resolve(import.meta.dirname,'..'),BUILD=import.meta.dirname;
const SKILL='C:/Users/Amiee/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const contents=JSON.parse(await fs.readFile(path.join(BUILD,'content.json'),'utf8'));
const FONT='Microsoft YaHei';
const p=Presentation.create({slideSize:{width:1280,height:720}});p.theme.defaultFont=FONT;
const tableOwners=[],chartOwners=[];
function text(s,t,x,y,w,h,size=28,bold=false,align='left'){
 const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=t;z.text.style={fontSize:size,typeface:FONT,color:'#111111',bold,alignment:align,verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0};return z;
}
for(const [ix,c] of contents.entries()){
 const s=p.slides.add();s.background.fill='#FFFFFF';const num=ix+1;
 if(c.cover){text(s,c.title,60,165,1160,96,64,true);text(s,c.subtitle,60,285,1160,100,40,true);text(s,c.date,60,450,1160,110,28);}
 else{
 text(s,c.title,60,42,1160,82,c.title.length>29?37:42,true);
 let y=142;if(c.lead){text(s,c.lead,60,y,1160,72,28,true);y+=86;}
 if(c.body){let yy=y;for(const b of c.body){const size=c.size??28;const lines=b.split('\n').map(v=>Math.ceil([...v].reduce((a,t)=>a+(/[\x00-\x7F]/.test(t)?.55:1),0)/(1160/size)));const h=Math.max(size*1.5,lines.reduce((a,t)=>a+t,0)*size*1.48);text(s,b,60,yy,1160,h,size);yy+=h+17;}}
 if(c.code){c.code.forEach((line,i)=>text(s,line,64,y+i*(c.lineHeight??33),1150,c.lineHeight??33,c.codeSize??25));}
 if(c.table){
  const rows=c.table.length,cols=c.table[0].length,top=c.tableY??(c.lead?230:145),height=c.tableH??Math.min(425,rows*57);
  const tb=s.tables.add({rows,columns:cols,left:60,top,width:1160,height,values:c.table,columnWidths:c.widths??Array(cols).fill(1160/cols)});
  tb.styleOptions={headerRow:false,bandedRows:false};tb.borders.assign({style:'solid',fill:'#888888',width:.7});
  for(let r=0;r<rows;r++){tb.rows[r].height=height/rows;for(let col=0;col<cols;col++){const cell=tb.getCell(r,col);cell.fill=r===0?'#F1F1F1':'#FFFFFF';cell.text.style={fontSize:c.tableSize??24,typeface:FONT,color:'#111111',bold:r===0,verticalAlignment:'middle',insets:{left:11,right:9,top:7,bottom:7},autoFit:'none'};}}
  tableOwners.push(num);
 }
 if(c.chart){const a=c.chart;const ch=s.charts.add(a.type??'line',{
  position:{left:60,top:c.lead?215:150,width:1160,height:a.height??375},
  categories:a.categories,series:a.series.map((v,i)=>({name:v.name,values:v.values,fill:['#111111','#777777','#BBBBBB'][i%3],line:{fill:['#111111','#777777','#BBBBBB'][i%3],width:2.6,style:i===1?'dashed':'solid'},marker:{symbol:'none'},...v})),
  hasLegend:a.series.length>1,legend:{position:'bottom',textStyle:{fontSize:20,fill:'#111111'}},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none'},
  xAxis:{textStyle:{fontSize:20,fill:'#111111'},line:{fill:'#111111',width:1},...a.xAxis},
  yAxis:{numberFormatCode:a.format??'0.00',textStyle:{fontSize:20,fill:'#111111'},majorGridlines:{fill:'#DDDDDD',width:.6},...a.yAxis},
  ...(a.type==='bar'?{barOptions:{direction:'column',grouping:'clustered',gapWidth:110}}:{}),
  ...(a.labels?{dataLabels:{showValue:true,position:'outEnd',numberFormatCode:a.format??'0.00',textStyle:{fontSize:21,fill:'#111111'}}}:{})
 });applyPresentationChartFont(ch,{fontFamily:FONT,availableFonts:[FONT]});chartOwners.push(num);
 if(a.dateLabels)a.dateLabels.forEach((v,i)=>text(s,v,82+i*244,572,180,29,20,false,'center'));
 }
 if(c.foot)text(s,c.foot,60,615,1130,64,22);
 text(s,String(num).padStart(2,'0'),1160,680,60,24,16,false,'right');
 }
 s.speakerNotes.text=(c.appendix?'答辩备用页':'主报告')+'\n'+(c.notes??'')+'\n\n来源与计算口径\n'+(c.sources??[]).map(a=>a.startsWith('https:')?a:path.join(ROOT,a)).join('\n');
}
const candidate=path.join(BUILD,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
console.log(JSON.stringify({slides:contents.length,main:contents.filter(x=>!x.appendix).length,tables:tableOwners,charts:chartOwners}));
const previewDir=path.join(BUILD,'slides');await fs.mkdir(previewDir,{recursive:true});
const only=process.argv.find(x=>x.startsWith('--pages='));
const selected=only?only.slice(8).split(',').map(Number):process.argv.includes('--smoke')?[1,7,15,20,29,30,36,37,51,54].filter(x=>x<=contents.length):contents.map((_,i)=>i+1);
for(const i of selected){const png=await p.export({slide:p.slides.items[i-1],format:'png',scale:1});await fs.writeFile(path.join(previewDir,'slide-'+String(i).padStart(2,'0')+'.png'),new Uint8Array(await png.arrayBuffer()));console.log('rendered '+i);}
if(!process.argv.includes('--smoke')&&!only){
 const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:path.join(ROOT,'组会汇报_20261016','量化选股项目_原理说明与新项目结果_20261016.pptx'),pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(i=>['--require-native-table-slide',String(i)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
 console.log(JSON.stringify(result));
}


