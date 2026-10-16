import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=import.meta.dirname,ROOT=path.resolve(B,'..');
const SKILL='C:/Users/Amiee/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const source=path.join(B,'user_modified_source.pptx'),sourceSha=createHash('sha256').update(await fs.readFile(source)).digest('hex');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const original=[...p.slides.items];
const snap=await p.inspect({kind:'slide,textbox,shape,table,chart',maxChars:300000});
const recs=snap.records;
const FONT='Microsoft YaHei';
const C=JSON.parse(await fs.readFile(path.join(B,'revision_content.json'),'utf8'));
const tableOwners=[],chartOwners=[];
function text(s,t,x,y,w,h,size=27,bold=false){
  const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  z.text=t;z.text.style={fontSize:size,typeface:FONT,color:'#111111',bold,alignment:'left',verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0};return z;
}
function edit(page,predicate,value,style={}){
  const rec=recs.find(r=>r.slideIndex===page-1&&r.kind==='textbox'&&predicate(r));
  if(!rec)throw new Error('Missing source anchor '+page);
  const z=p.resolve(rec.id);z.text=value;z.text.style={...rec.style,typeface:FONT,color:'#111111',wrap:'square',autoFit:'none',insets:0,...style};return z;
}
const footer=(page,value)=>edit(page,r=>r.position.top>=600&&r.position.top<660,value,{fontSize:22});
function note(page,value){original[page-1].speakerNotes.text=String(original[page-1].speakerNotes.text??'')+'\n\n补充说明\n'+value;}
edit(1,r=>r.position.top===285,'因子筛选、模型更新与亏损分析');
edit(2,r=>r.text.startsWith('已实现'),'已实现：每日更新股票池，预测1、2、5、10、20日相对表现。\n每季度用最近504个交易日的成熟答案，重筛因子并训练模型。');
edit(2,r=>r.text.startsWith('项目已有'),'当前缺口：训练记录等权，尚未按因子衰减速度自动调整窗口、权重和更新频率。',{fontSize:27});
footer(3,'本地北交所行情覆盖2021-11-15至2026-10-09。每天新增记录，不等于每天重训。\n历史不足或未来答案未出现时，不能造出训练样本。');
note(4,'每日历史不够时只积累数据。内部验证与正式使用分别调用该阶段此前建立的模型。'+String(original[15].speakerNotes.text??''));
footer(5,'当前五天收益 = t后第5交易日复权收盘价 ÷ t后第1交易日复权开盘价 − 1。\n一次学习已知历史可以符合边界，不能用训练结果冒充对这段历史的提前预测。');
text(original[5],'验证先按IC选3种方法，再比股票数和仓位，共90个账户。\n把夏普与卡玛÷2中较小者作为分数，选最高者，再正式重筛、重训。',60,526,1160,70,24);
footer(6,'正式重训有49,881行，4—6月答案不参与。当前没有“验证全不达标就拒绝使用”的门槛。');
note(6,'实际选择模型的流程：10种方法算验证期逐日排名IC，取前3；每种配10或20只股票及15种仓位风控规则，共90账户；最大化min(Sharpe_rf1.5,Calmar/2)，同分比年化收益。2026Q2五天选ensemble、top10、hmm3_defensive，验证夏普0.548、卡玛1.519，均未同时达标但代码仍选最高者。4月1日再次用最新成熟历史筛列并拟合。');
footer(7,'4月1日：350只中，63只历史不足、1只近期交易不足；286只候选按成交额取前100只。\n成交额指标为最近20交易日金额之和÷20。收盘后选池，可以使用当天完整行情。');
note(7,'2026-04-01池第100名成电光信，adv20为37,221,716.60；第101名国子软件36,662,615.70，后者剔除。文件集合并非完整历史证券主表，历史ST、上市退市及可投资状态还需核验。');
note(9,'IC详细实现：先按日期计算x−当日列均值、y−当日答案均值，再算Σab/√(Σa²Σb²)，最后按日期平均。原实现缺失值并非统一成双方共同有效股票集合，不能直接等同标准Spearman。'+String(original[16].speakerNotes.text??''));
footer(10,'5列覆盖不足剔除，330列再排序去重；ICIR、夏普、回撤不参与这一步门槛。\n原IC计算未统一缺失股票集合，需修正重跑。本页及筛选实例按原实现展示。');
note(10,'完整去重规则：同分按列名；训练输入空值填0后计算跨全部训练行的列间Pearson相关绝对值；与任意已选列>=.92跳过；取足100停止。日期和股票代码只用于对齐，训练答案单独传入。'+String(original[17].speakerNotes.text??''));
const t12=original[11].tables.items[0];
t12.cells.set(3,2,'最近504交易日内，答案成熟的记录\n等权训练，再替换旧模型');
t12.getCell(3,2).text.style={fontSize:23,typeface:FONT,color:'#111111',verticalAlignment:'middle',insets:{left:11,right:9,top:7,bottom:7},autoFit:'none'};
footer(12,'2026年Q1与Q2因子重合94／100列，Q2与Q3重合93／100列。重训不等于每次换掉大量因子。');
note(12,'该重合数由h5各季度deployment_factors.csv中selected列集合交集得到。关系变化后即使因子名不变，树的分界和叶值仍可变化；反过来，仅换因子也不保证改善。HMM每月更新是市场仓位层面，不会自动更新LightGBM参数。');
edit(13,r=>r.position.top===142,'2024-10-08至2026-09-30，共485日。每边5bp，无风险年收益1.5%',{fontSize:28,bold:true});
footer(13,'各季度按此前验证选模型和仓位，并非五个纯LightGBM账户。验收要求夏普 > 1且卡玛 > 2。');
note(13,'夏普=(日收益平均−日无风险收益)/日收益样本标准差×√252，日无风险收益=1.015^(1/252)−1。最大回撤=每日资产相对之前最高资产的最大跌幅，包含现金和费用。卡玛=年化收益/最大回撤。年化按实际日历天数365.25折算。1bp=万分之一，每边5bp表示买与卖各收0.05%。');
edit(14,r=>r.position.top===42,'历史另选的五天候选：近期仍亏损',{fontSize:42,bold:true});
footer(14,'本页是历史搜索选出的IC组合、前20只、低仓位候选，非上一页的主流程账户。\n最近半年包含在最近一年内。全历史达标不能证明近期稳定盈利。');
note(14,'候选模型为ic，base_policy=hmm3_growth，risk_scale=.25，平均股票仓位约7.01%，每边5bp。此候选经全历史查看后选出，因此不能当成外层测试前预选结果。最近一年股票池等权账户约−24.80%、北证50价格指数约−31.99%，两者仓位不同，只能辅助说明市场环境，不作等风险因果归因。');
footer(15,'不同仓位规则改善不一致。降低仓位会同时减小风险和上涨收益；仓位调整不能代替个股模型重训。');
edit(19,r=>r.position.top===42,'五天模型每个季度何时重新训练',{fontSize:42,bold:true});

const keep=[...Array.from({length:15},(_,i)=>i),18];
p.slides.keep(keep);
const added=new Map();
for(const c of C.new){const s=original[1].duplicate();added.set(c.key,s);}
const addedIds=new Set([...added.values()].map(s=>s.id));
const removeSnap=await p.inspect({kind:'slide,textbox,shape,table,chart,image',maxChars:350000});
const owned=removeSnap.records.filter(r=>['textbox','shape','table','chart','image'].includes(r.kind)&&addedIds.has(p.slides.items[r.slideIndex]?.id)).map(r=>r.id);
p.delete([...new Set(owned)]);
for(const c of C.new){
 const s=added.get(c.key);s.background.fill='#FFFFFF';
 text(s,c.title,60,42,1160,82,42,true);
 if(c.lead)text(s,c.lead,60,142,1160,72,27,true);
 if(c.code)c.code.forEach((v,i)=>text(s,v,64,142+i*(c.lineHeight??35),1150,c.lineHeight??35,c.codeSize??25));
 if(c.table){
   const tb=s.tables.add({rows:c.table.length,columns:c.table[0].length,left:60,top:c.tableY??145,width:1160,height:c.tableH,values:c.table,columnWidths:c.widths});
   tb.styleOptions={headerRow:false,bandedRows:false};tb.borders.assign({style:'solid',fill:'#888888',width:.7});
   for(let r=0;r<c.table.length;r++){tb.rows[r].height=c.tableH/c.table.length;for(let col=0;col<c.table[0].length;col++){
    const cell=tb.getCell(r,col);cell.fill=r===0?'#F1F1F1':'#FFFFFF';cell.text.style={fontSize:c.tableSize??24,typeface:FONT,color:'#111111',bold:r===0,verticalAlignment:'middle',insets:{left:11,right:9,top:7,bottom:7},autoFit:'none'};
   }}
 }
 for(const v of c.texts??[])text(s,v.text,60,v.y,1160,v.h,v.size??27,!!v.bold);
 if(c.chart){const a=c.chart;const ch=s.charts.add(a.type,{position:{left:60,top:a.top,width:1160,height:a.height},categories:a.categories,
  series:[{name:'五天LightGBM平均日IC',values:a.values,fill:'#111111',line:{fill:'#111111',width:1}}],hasLegend:false,chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none'},
  barOptions:{direction:'column',grouping:'clustered',gapWidth:170},xAxis:{tickLabelPosition:'low',textStyle:{fontSize:22,fill:'#111111'}},
  yAxis:{min:0,max:.22,majorUnit:.05,numberFormatCode:a.format,textStyle:{fontSize:20,fill:'#111111'},majorGridlines:{fill:'#DDDDDD',width:.6}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:a.format,textStyle:{fontSize:23,fill:'#111111'}}});
  applyPresentationChartFont(ch,{fontFamily:FONT,availableFonts:[FONT]});
 }
 if(c.foot)text(s,c.foot,60,615,1130,64,22);
 s.speakerNotes.text=c.notes+'\n\n来源与计算口径\n'+c.sources.map(x=>x.startsWith('https:')?x:path.join(ROOT,x)).join('\n');
}
const ordered=C.order.map(key=>typeof key==='number'?original[key-1]:added.get(key));
p.slides.reorder(ordered.map(s=>'sl/'+s.id));
for(const [ix,s] of p.slides.items.entries()){
 if(ix){const old=s.shapes.items.find(z=>z.position.top>=675&&z.position.left>1100);if(old)old.text=String(ix+1).padStart(2,'0');else {const num=text(s,String(ix+1).padStart(2,'0'),1160,680,60,24,16);num.text.style={fontSize:16,typeface:FONT,color:'#111111',alignment:'right',verticalAlignment:'top',insets:0,autoFit:'none'};}}
 if(s.tables.items.length)tableOwners.push(ix+1);
 if(s.charts.items.length)chartOwners.push(ix+1);
}
if(p.slides.items.length!==25)throw new Error('Wrong slide count');
const candidate=path.join(B,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const only=process.argv.find(x=>x.startsWith('--pages='));
const pages=only?only.slice(8).split(',').map(Number):Array.from({length:25},(_,i)=>i+1);
await fs.mkdir(path.join(B,'slides'),{recursive:true});
for(const i of pages){const png=await p.export({slide:p.slides.items[i-1],format:'png',scale:1});await fs.writeFile(path.join(B,'slides',`slide-${String(i).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));}
await fs.writeFile(path.join(B,'final_notes.json'),JSON.stringify(p.slides.items.map((s,i)=>({page:i+1,source:C.order[i],notes:String(s.speakerNotes.text)})),null,2));
console.log(JSON.stringify({slides:25,reused:16,added:9,tables:tableOwners,charts:chartOwners}));
if(process.argv.includes('--final')){
 const finalPath=path.join(ROOT,'组会汇报_20261016','量化选股项目_老师意见修订_25页.pptx');
 const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,pythonExecutable:PY,
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  explicitTotalSlideCount:25,layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
  requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[FONT],referencePath:source,referenceSha256:sourceSha},verifyArtifactToolImport:true,receiptPath:path.join(B,'validation.json')});
 console.log(JSON.stringify({finalPath:result.finalPath,sha256:result.finalSha256,bytes:result.byteCount}));
}
