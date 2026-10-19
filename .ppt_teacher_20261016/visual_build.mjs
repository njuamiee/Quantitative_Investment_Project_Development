import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=path.join(import.meta.dirname,'visual_revision'),ROOT=path.resolve(B,'../..');
const SKILL='C:/Users/Amiee/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/Amiee/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const source=path.join(ROOT,'组会汇报_20261016/量化选股项目_老师意见修订_25页.pptx');
const sha=createHash('sha256').update(await fs.readFile(source)).digest('hex');
const A=JSON.parse(await fs.readFile(path.join(B,'analysis.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==25)throw Error('Expected 25 slides');
const FONT='Microsoft YaHei',INK='#111111',GRAYS=['#111111','#777777','#AAAAAA','#444444'];
const tableOwners=[],chartOwners=[];
const slides=[...p.slides.items];
if(process.argv.includes('--source-only')){
 await fs.mkdir(path.join(B,'source'),{recursive:true});
 for(const i of [2,7,10,12])await fs.writeFile(path.join(B,'source',`${i}.png`),new Uint8Array(await(await p.export({slide:slides[i-1],format:'png',scale:1})).arrayBuffer()));
 console.log('source previews ready');process.exit(0);
}
const snap=await p.inspect({kind:'shape,textbox,image,table,chart',maxChars:500000});
p.delete([...new Set(snap.records.filter(r=>['shape','textbox','image','table','chart'].includes(r.kind)).map(r=>r.id))]);
function text(s,t,x,y,w,h,size=27,bold=false,align='left'){
 const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=t;z.text.style={fontSize:size,typeface:FONT,color:INK,bold,alignment:align,verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0};return z;
}
function line(s,x,y,w,h=0,color='#777777',width=1){return s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width,style:'solid'}});}
function box(s,t,x,y,w,h,size=27,fill='#FFFFFF',bold=false){
 const z=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'#555555',width:1.2}});
 z.text=t;z.text.style={fontSize:size,typeface:FONT,color:INK,bold,alignment:'center',verticalAlignment:'middle',wrap:'square',autoFit:'none',insets:12};return z;
}
function arrow(s,x,y,w=42,h=23,rotation=0){return s.shapes.add({geometry:'rightArrow',position:{left:x,top:y,width:w,height:h,rotation},fill:'#555555',line:{fill:'none',width:0}});}
function base(n,title,sub=''){
 const s=slides[n-1];s.background.fill='#FFFFFF';s.speakerNotes.text='';
 text(s,title,60,42,1160,73,42,true);if(sub)text(s,sub,60,126,1160,58,26);
 text(s,String(n).padStart(2,'0'),1160,681,60,24,16,false,'right');return s;
}
function foot(s,t,source=''){text(s,t,60,604,1160,62,23);if(source)text(s,source,60,681,1080,23,15);}
function table(s,values,{x=60,y=200,w=1160,h=320,widths,size=25}={}){
 const tb=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 tb.styleOptions={headerRow:false,bandedRows:false};tb.borders.assign({style:'solid',fill:'#AAAAAA',width:.6});
 for(let r=0;r<values.length;r++){tb.rows[r].height=h/values.length;for(let c=0;c<values[r].length;c++){const z=tb.getCell(r,c);z.fill=r===0?'#EEEEEE':'#FFFFFF';z.text.style={fontSize:size,typeface:FONT,color:INK,bold:r===0,verticalAlignment:'middle',insets:{left:12,right:10,top:8,bottom:8},autoFit:'none'};}}
 return tb;
}
function chart(s,type,categories,series,{x=60,y=190,w=1160,h=390,format='0.0%',legend=true,min,max,bar='column',labels=false,extra={}}={}){
 series=series.map(z=>({...z,values:z.values.map(v=>v==null?null:Number(v.toFixed(8)))}));
 if(type==='line' && /^\d{4}-\d{2}-\d{2}$/.test(categories[0]??'')){
  const xd=categories.map(v=>(Date.parse(v+'T00:00:00Z')-Date.UTC(1899,11,30))/86400000);
  type='scatter';series=series.map(z=>({...z,xValues:xd}));
  extra={...extra,scatterOptions:{style:'line'},xAxis:{min:xd[0],max:xd.at(-1),majorUnit:120,numberFormatCode:'yyyy-mm',textStyle:{fontSize:20,fill:INK},tickLabelPosition:'low',majorGridlines:null}};
 }
 const ch=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,
  series:series.map((z,i)=>({name:z.name,values:z.values,xValues:z.xValues,fill:GRAYS[i%4],line:{fill:GRAYS[i%4],width:i===0?3:2,style:['solid','dashed','dotted','dash-dot'][i%4]},marker:{symbol:'none'},...z})),
  hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:21,fill:INK}},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none'},plotAreaLine:{fill:'none'},
  xAxis:{textStyle:{fontSize:18,fill:INK},tickLabelPosition:'low',line:{fill:'#777777',width:1},majorGridlines:null},
  yAxis:{min,max,numberFormatCode:format,textStyle:{fontSize:19,fill:INK},majorGridlines:{fill:'#DDDDDD',width:.6}},
  barOptions:{direction:bar,grouping:'clustered',gapWidth:65},lineOptions:{smooth:false},
  dataLabels:{showValue:labels,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:22,fill:INK}},...extra});
 applyPresentationChartFont(ch,{fontFamily:FONT,availableFonts:[FONT]});return ch;
}
const dates=A.curves.dates, C=A.curves.series;
function sparseLabels(ds){return ds;}
const cats=sparseLabels(dates);
const pct=(x,d=2)=>`${x<0?'−':'+'}${Math.abs(x*100).toFixed(d)}%`;
const pp=(x,d=2)=>`${x<0?'−':'+'}${Math.abs(x*100).toFixed(d)}个百分点`;
const full=A.periods['全期'];

// 1: preserve the source's plain white, black typography and title geometry.
{
 const s=base(1,'量化选股项目');
 text(s,'因子筛选、滚动训练与超额收益',60,255,1160,100,46,true);
 text(s,'以五天预测为例，沿着一条真实记录说明模型如何工作',60,385,1160,70,29);
 text(s,'组会汇报  2026年10月16日',60,548,1160,44,26);
 foot(s,'回测比较：2024-10-08收盘至2026-09-30收盘\n本地行情：2021-11-15至2026-10-09');
}
// 2: equity curves first, answer teacher's main question.
{
 const s=base(2,'先看资金曲线：是否跑赢简单持有','全部从100万元开始，展示连续485个交易日的每日资产');
 chart(s,'line',cats,['主流程','期初等权持有','北证50价格'].map(name=>({name,values:C[name].map(x=>x*100)})),{format:'0',min:50,max:210,h:380});
 foot(s,'期末资产：主流程149.23万元，期初等权114.59万元，指数参考94.65万元。\n相对等权多赚34.63个百分点，相对北证50多54.58个百分点。','来源：h5/selected_daily.csv、本次买入持有账本、bj50_benchmark.csv；纵轴：万元');
}
// 3: benchmark construction defines a real frozen cohort, not the preexisting rolling account.
{
 const s=base(3,'比较对象与共同起点','主流程按此前验证选方法和仓位，不能把整条曲线称为纯LightGBM收益');
 table(s,[['对象','选股与交易','费用与现金'],['主流程','每日排序，买前10或20只\n分批持有五天，按规则调仓','买卖各0.05%\n剩余资金留现金'],['期初等权持有','10月8日确定100只，各分1万元\n10月9日开盘买，之后不换股','实际成交99只，1只未成交\n分红留现金，送转增加股数'],['北证50价格指数','10月8日收盘点位设为100\n之后按每日收盘比例变化','价格指数不含现金分红\n不扣交易费用，仅作参照']],{y:205,h:340,widths:[230,540,390],size:24});
 foot(s,'账户均在9月30日收盘估值，不假设未来卖出。等权指买入预算相同，之后权重随涨跌变化。','本次基准沿用项目成交约束、除权日到账情景；指数资料：bse.cn/market_data/bse_indices/bse_bz50.html');
}
// 4: explanatory process, not an implementation checklist.
{
 const s=base(4,'从当天数据到下一次交易','一行是“一只股票在一天的记录”。同一天100行，不同日期继续往下接');
 const xs=[60,365,670,975],ws=[245,245,245,245];
 ['收盘后的股票池\n最多100只','每只335个指标\n取本季度100列','固定模型逐只评分\n得到100个分数','按分数排序\n下一交易日开盘买'].forEach((t,i)=>{box(s,t,xs[i],225,ws[i],135,27);if(i<3)arrow(s,xs[i]+251,280,42,24);});
 line(s,60,450,1160);text(s,'每日变化',60,475,220,40,28,true);text(s,'股票池、当天指标、评分、持仓、已出现的收益答案',295,475,925,60,27);
 text(s,'季度变化',60,545,220,40,28,true);text(s,'用此前已知答案重选因子，重新建立模型',295,545,925,60,27);
 foot(s,'行情第1天2021-11-15开始积累；2022-02-14首次成池；2024-10-08开始逐季历史测试。','来源：prepare_data.py、train_models.py、ledger.py');
}
// 5: stock pool actual date with if/else conditions.
{
 const s=base(5,'股票池：4月1日从350只中选100只','2026-04-01收盘后执行。只使用当天及以前的数据');
 const stages=[['350只','本地股票文件'],['287只','历史不足60天\n剔除63只'],['286只','近20天成交不足16天\n再剔除1只'],['100只','20日平均成交额\n取前100只']];
 stages.forEach((a,i)=>{const x=60+i*305;box(s,a[0],x,210,245,100,42,'#EEEEEE',true);text(s,a[1],x,330,245,95,24,false,'center');if(i<3)arrow(s,x+252,250);});
 text(s,'if 历史≥60天 且 近20天成交≥16天 且 当天有成交 且 平均成交额>0：入候选',60,465,1160,60,25);
 text(s,'else：剔除；候选按20日成交额之和÷20排序，取前100只',60,525,1160,48,26);
 foot(s,'第100名成电光信：3,722.17万元；第101名国子软件：3,666.26万元，后者未入池。','来源：01_20260401全部股票筛选过程.csv；历史ST、退市状态尚未完整核验');
}
// 6: feature funnel and precise order.
{
 const s=base(6,'因子预筛：335列如何变成100列','先由固定规则筛，再交给LightGBM。不是手工挑小集合，也不是按树的重要性倒选');
 const a=[['335列','292个供应商因子\n+19个价量特征+24个表达式'],['330列','有效记录比例至少70%\n本次剔除5列'],['100列','按平均IC绝对值排序\n相似度≥0.92就跳过']];
 a.forEach((z,i)=>{const x=60+i*400;box(s,z[0],x,225,350,100,44,'#EEEEEE',true);text(s,z[1],x,350,350,105,25,false,'center');if(i<2)arrow(s,x+354,264,40,24);});
 text(s,'逐列判断：if 与任一已选列的相关绝对值≥0.92：跳过',60,475,1160,45,28);
 text(s,'else：加入；if 已选满100列：停止。相关用本次训练行计算。',60,530,1160,45,28);
 foot(s,'70%、0.92、100是当前预设。ICIR、夏普和回撤不参与这一轮预筛。实际列数是335，不是355。','来源：train_models.py / screening；示例训练集：2026Q2五天模型49,881行');
}
// 7: correlation direction made visible with explicitly illustrative small samples.
{
 const s=base(7,'IC：因子排序与后来收益排序是否一致','教学示意：每张图只有5只股票。横轴是因子从小到大的名次，纵轴是后来收益名次');
 chart(s,'scatter',['1','2','3','4','5'],[{name:'正相关',xValues:[1,2,3,4,5],values:[1,2,3,4,5],marker:{symbol:'circle',size:9,fill:INK}}],{x:60,w:540,y:215,h:300,min:0,max:6,format:'0',legend:false,extra:{scatterOptions:{style:'marker'},xAxis:{min:0,max:6,numberFormatCode:'0',textStyle:{fontSize:20,fill:INK}}}});
 chart(s,'scatter',['1','2','3','4','5'],[{name:'负相关',xValues:[1,2,3,4,5],values:[5,4,3,2,1],marker:{symbol:'circle',size:9,fill:INK}}],{x:675,w:540,y:215,h:300,min:0,max:6,format:'0',legend:false,extra:{scatterOptions:{style:'marker'},xAxis:{min:0,max:6,numberFormatCode:'0',textStyle:{fontSize:20,fill:INK}}}});
 text(s,'IC = +1：数值越大，后来排名越高',60,535,565,48,26,true);text(s,'IC = −1：数值越小，后来排名越高',655,535,565,48,26,true);
 foot(s,'实际046平均IC=+0.1523，063为−0.1482。负相关也可留下；LightGBM用判断分支学习方向。','IC=Σ[(x−均值x)(y−均值y)] / √(Σ(x−均值x)² × Σ(y−均值y)²)；每天算一次，再求平均');
}
// 8: actual screening, not an invented example.
{
 const s=base(8,'实际筛选：063保留，067跳过','2026Q2五天模型，按历史平均IC的绝对值从大到小检查');
 table(s,[['顺序 / 因子','平均IC','与已选列最大相关绝对值','决定'],...A.factor_example.map((r,i)=>[`${i+1} / ${r.factor.slice(-3)}`,`${r.mean_ic>=0?'+':'−'}${Math.abs(r.mean_ic).toFixed(4)}`,i===0?'首列，无需比较':`${r.max_abs_correlation.toFixed(4)} ${r.max_abs_correlation>=.92?'≥':'<'} 0.92`,r.selected?'保留':'跳过'])],{y:210,h:330,widths:[245,215,445,255],size:27});
 foot(s,'0.9241说明067与已有因子过于相似，即使它自身与收益有关系，也先跳过。\n当前IC缺失值集合处理有偏差，需修正重跑；本页如实展示现有实现。','来源：02_2026Q2五天模型全部因子筛选过程.csv；相关去重时空值填0');
}
// 9: rows, columns, scaling and answers all concrete.
{
 const s=base(9,'LightGBM输入：49,881行 × 100列','100列是指标；49,881行是不同日期、不同股票的记录。答案另交一列y');
 table(s,[['日期 / 股票，仅识别','046输入','063输入','其余98列','答案y'],['2024-03-01 / 920001','−0.15','+0.12','该日指标','−0.49'],['2024-03-01 / 920021','+0.17','−0.10','该日指标','+0.13'],['2024-03-04 / 920001','+0.33','−0.22','该日指标','−0.43']],{y:205,h:255,widths:[400,175,175,215,195],size:24});
 text(s,'输入怎么变：046在当天100只有值股票中排第35，35÷100−0.5=−0.15',60,490,1160,45,25);
 text(s,'答案怎么变：后来五天亏16.86%，当日100只中最低，1÷100−0.5=−0.49',60,542,1160,45,25);
 foot(s,'同日先排名再减0.5；并列取平均名次。输入缺失填0，答案缺失删整行。\n499天×100只−19条缺答案=49,881行；日期、代码、未来收益都不进X。','来源：05_训练表前五行含答案.csv、prepare_data.py；输出学习的是相对名次，不是涨幅');
}
// 10: T+1 and weekend gap, plus exact five-day exclusion.
{
 const s=base(10,'为什么3月27日的“一天收益”到31日才知道','当前一天口径：周五收盘选股，下一交易日买，再下一交易日开盘卖');
 [['3月27日 周五','收盘后计算因子\n此时没有收益答案'],['3月28—29日','周末休市\n没有开盘买卖'],['3月30日 周一','开盘买入\n当天不能卖出'],['3月31日 周二','开盘卖出\n此刻才能计算答案']].forEach((z,i)=>{const x=60+i*305;box(s,z[0],x,220,245,70,27,'#EEEEEE',true);text(s,z[1],x,315,245,105,26,false,'center');if(i<3)arrow(s,x+251,240);});
 text(s,'一天收益 = 3月31日开盘价 ÷ 3月30日开盘价 − 1',60,470,1160,52,31,true);
 foot(s,'五天口径：t后第5交易日收盘 ÷ 下一交易日开盘 − 1。3月24日记录到31日成熟；\n3月25日记录到4月1日才成熟，因此不能参加4月1日启用模型的训练。','来源：prepare_data.py；天数按交易日计算，训练标签用复权价，交易账本用原始价和分红送转');
}
// 11: timeline ranges; training historical batch is not future leakage by itself.
{
 const s=base(11,'4月1日启用前：训练、验证、再训练','一批读入过去的完整记录可以合规；边界看“答案何时已知”，不看读取顺序');
 const rows=[['① 内部训练','2023-08-21 至 2025-09-09','答案最晚2025-09-16'],['② 内部验证','2025-09-17 至 2026-03-31','只评价已在3月31日前成熟的答案'],['③ 正式重训','2024-03-01 至 2026-03-24','答案最晚3月31日，49,881行'],['④ 正式预测','2026-04-01 至 2026-06-30','每天输入当天指标，季度内模型固定']];
 rows.forEach((r,i)=>{const y=205+i*88;box(s,r[0],60,y,215,65,27,i===3?'#DDDDDD':'#FFFFFF',true);text(s,r[1],305,y+4,490,59,27);text(s,r[2],805,y+4,415,60,24);});
 foot(s,'if 答案结束日 ≥ 当前训练截止日：禁止进入筛因子和训练。\n验证选方法、股票数和仓位；再用最新成熟历史重训，随后才评价4—6月。','来源：08_20260401训练和内部考试完整边界.csv、train_models.py；供应商历史发布时间仍待核验');
}
// 12: rolling quarters visually offset, actual date windows.
{
 const s=base(12,'季度滚动：旧模型被新模型替换','每次向前移动一个季度，重新筛100列、重新训练180棵树');
 const fs=A.folds.filter(f=>['2026Q1','2026Q2','2026Q3'].includes(f.quarter));
 fs.forEach((f,i)=>{const y=212+i*123;const x=255+i*55;
 text(s,`${f.quarter}\n${f.test_start.slice(5,10)}启用`,60,y,170,85,28,true);
 box(s,`${f.training_start.slice(0,10)} 至 ${f.training_last_signal.slice(0,10)}\n${f.training_rows.toLocaleString()}行，答案截至${f.training_last_label_exit.slice(5,10)}`,x,y,560,87,25,'#EEEEEE');
 arrow(s,x+568,y+31,45,24);box(s,'固定模型\n使用一个季度',x+625,y,225,87,25);
 });
 foot(s,'4月1日用新模型；4月2日仍用同一个模型，只换当天的输入。新答案存入历史库，7月再训练。','来源：h5/folds.json；最近504交易日窗口，五天答案未成熟记录剔除后约499个信号日');
}
// 13: native editable decision diagram plus exact training rule.
{
 const s=base(13,'LightGBM的一棵树：把记录分组后纠错','2026Q2五天模型第一棵树的首个真实判断：046输入是否 ≤ −0.195985');
 box(s,'49,881条历史记录\n开始都预测0.0050019',410,205,460,90,28,'#EEEEEE',true);
 line(s,640,295,0,35);line(s,300,330,680,0);line(s,300,330,0,35);line(s,980,330,0,35);
 box(s,'if ≤ −0.195985\n14,957行，继续细分',70,365,460,90,28);
 box(s,'else > −0.195985\n34,924行，继续细分',750,365,460,90,28);
 text(s,'每组修正 = 0.035 × 该组误差之和 ÷（该组行数 + 12）',60,492,1160,48,29,true);
 text(s,'误差 = 答案 − 当前预测；比较各分界的纠错效果，并惩罚过大的修正。',60,548,1160,42,26);
 foot(s,'0.035控制每轮调整幅度，+12压低修正。9945行误差和624.68时，修正+0.00220。\n每棵树最多15组、5层，每组至少160行；抽80%列。这些数均为当前预设。','来源：2026Q2_models.joblib、train_models.py；分界和叶值均有限，输出不是任意猜测');
}
// 14: show error is feedback and rounds are not trading days.
{
 const s=base(14,'180轮训练：每轮纠正同一批历史记录','横轴是第几轮，不是第几天。纵轴是所有49,881行的误差平方之和');
 chart(s,'scatter',A.training_stages.map(r=>String(r.round)),[{name:'训练误差',xValues:A.training_stages.map(r=>r.round),values:A.training_stages.map(r=>r.sse),marker:{symbol:'circle',size:6,fill:INK}}],{x:60,y:200,w:740,h:365,format:'0',min:3700,max:4200,legend:false,extra:{scatterOptions:{style:'lineWithMarkers'},xAxis:{min:0,max:180,majorUnit:30,numberFormatCode:'0',textStyle:{fontSize:20,fill:INK}}}});
 text(s,'起始：4156.33\n第1轮：4145.66\n第180轮：3761.97',850,235,370,150,28,true);
 text(s,'每一轮：\n算误差\n建立新的if/else树\n加上修正，再算误差',850,412,370,175,26);
 foot(s,'新股票输出 = 起始均值 + 180棵树分别给出的修正。训练误差下降，只说明更贴合旧样本。\n真实样例920001：答案−0.49，预测从+0.0050变为−0.0790，仍有误差。','来源：2026Q2_models.joblib；参数：lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html');
}
// 15: prediction != return, actual day.
{
 const s=base(15,'4月1日的输出：100只股票各得一个分数','以下是当日纯LightGBM评分最高的5只。右列是之后才出现的五天实际收益');
 table(s,[['股票代码','LightGBM分数','后来五天收益'],...A.prediction_sample.map(r=>[r.stockid.replace('BJ',''),r.lgbm100.toFixed(4),pct(r.label_5)])],{y:205,h:315,widths:[360,400,400],size:28});
 text(s,'0.0994不等于涨9.94%，也不等于上涨概率。它用于同日股票的相对排序。',60,548,1160,48,27,true);
 foot(s,'当季主流程实际选择四模型排名的平均值，再买前10只；LightGBM是其中一个成员。\n当天只能使用左两列。右列用于之后检验排序是否有用。','来源：h5/2026Q2_predictions.parquet、2026Q2_selection.json');
}
// 16: model performance changes by date.
{
 const s=base(16,'模型关系会变：Q2的排序能力明显减弱','老师说的“参数漂移”：过去有效的判断分界和修正幅度，换一段市场可能不再合适');
 const q=A.old_evidence.model_ic_quarters.filter(r=>['2026Q1','2026Q2','2026Q3'].includes(r.quarter));
 chart(s,'bar',q.map(r=>r.quarter),[{name:'平均每日排名IC',values:q.map(r=>r.ic)}],{y:210,h:350,min:0,max:.22,format:'0.000',legend:false,labels:true});
 foot(s,'季度重训已经在做，但不能保证下一季有效。Q2变差是现象，不能仅凭这张图断定具体原因。\n新旧因子重合94/100列、93/100列；因子名没换，树的判断也可能改变。','来源：h5_lgbm_daily_ic_groups.csv；本页是纯LightGBM的预测评价，不是主流程账户收益');
}
// 17: distinguish fast signal horizon from relationship decay; explicit proposed policy.
{
 const s=base(17,'因子衰减与更新频率怎样配合','“信号几天后失效”决定持有多久；“因子关系几个月后变弱”决定训练历史留多长');
 const d=A.old_evidence.factor046_windows;
 chart(s,'bar',d.map(r=>`最近${r.n}日`),[{name:'046平均IC',values:d.map(r=>r.mean_ic)}],{x:60,y:210,w:550,h:320,format:'0.000',min:0,max:.18,legend:false,labels:true});
 text(s,'同一截止日2026-03-24\n短期关系弱于长历史平均，\n但这还不是“半衰期”的测量。',660,225,560,120,27);
 text(s,'拟比较的规则（尚未实现）\n快：留63日，每5日重训\n慢：留252日，每21日重训\n只在过去的验证期决定哪套更好',660,385,560,180,26);
 foot(s,'当前是504日等权训练、季度重训。拟加时间权重：权重=2^(−距今交易日÷半衰期)，\n如半衰期21日，21日前样本权重为今天的一半；必须用新时期检验超额和费用。','来源：10_用于选因子的每一天IC.csv；只看短期IC下降不能证明该因子必然失效');
}
// 18: direct excess time series, precisely defined.
{
 const s=base(18,'超额收益曲线：从起点累计多赚多少','超额（百分点）= 策略累计收益率 − 基准累计收益率。高于0表示累计领先');
 chart(s,'line',cats,['期初等权持有','北证50价格'].map(name=>({name:'相对'+name,values:C['主流程'].map((v,i)=>(v-C[name][i])*100)})),{y:200,h:380,format:'0',min:-60,max:65});
 foot(s,'全期相对期初等权：+34.63个百分点；相对北证50：+54.58个百分点。\n相对财富比另算：1.4923÷1.1459−1=30.22%，它与34.63个百分点不是同一口径。','来源：本次统一起点的comparison_daily.csv；累计收益差不是扣除市场风险后的纯选股收益');
}
// 19: show weak quarters as well as the end number.
{
 const s=base(19,'分季度看：领先并不发生在每一季','柱高 = 当季主流程收益 − 同期原始买入持有组合收益，单位：百分点');
 chart(s,'bar',A.quarterly.map(r=>r.quarter),[{name:'相对期初等权的季度超额',values:A.quarterly.map(r=>r['超额']*100)}],{y:210,h:355,format:'0.0',min:-16,max:16,legend:false,labels:true});
 foot(s,'2024Q4落后13.08个百分点，2026Q2落后1.52个百分点；2026Q3领先8.12个百分点。\n季度超额不能直接相加成全期超额，因为每季的资产基数不同。','来源：同一连续账本按季度首尾资产计算；首季从2024-10-08开始');
}
// 20: loss vs benchmark and rebasing.
{
 const s=base(20,'后一年：亏损，但比基准少亏','2025-09-30收盘重新设为100万元，观察此后一年；原持仓继续，不重新买入');
 const k=dates.indexOf('2025-09-30'),ds=dates.slice(k);
 chart(s,'line',sparseLabels(ds),['主流程','期初等权持有','北证50价格'].map(name=>({name,values:C[name].slice(k).map(x=>100*x/C[name][k])})),{y:200,h:380,format:'0',min:55,max:120});
 foot(s,'主流程−8.24%，等权持有−31.87%，北证50−31.99%。\n相对等权+23.63个百分点，相对指数+23.75个百分点；这比只报亏损更完整。','来源：comparison_daily.csv；纵轴：重新设定起点后的万元，不是额外进行一次交易');
}
// 21: capital risk profile.
{
 const s=base(21,'回撤曲线：资产从此前最高点跌了多少','每日回撤 = 当日资产 ÷ 此前（含当日）最高资产 − 1');
 chart(s,'line',cats,['主流程','期初等权持有','北证50价格'].map(name=>{let peak=1;return{name,values:C[name].map(x=>{peak=Math.max(peak,x);return x/peak-1;})};}),{y:200,h:380,format:'0%',min:-.5,max:0});
 foot(s,'最大回撤：主流程23.87%，等权持有41.38%，北证50价格37.95%。\n主流程最深回撤：2025-03-18的142.64万元，降至4月7日的108.59万元。','来源：连续每日资产；是从峰值到谷底的跌幅，不是把亏损日简单相加');
}
// 22: measured stock/fees attribution rather than assumed causality.
{
 const s=base(22,'最近半年亏在哪里：先把账户对平','2026-03-31收盘至2026-09-30收盘，161.60万元降至149.23万元');
 const a=A.old_evidence.recent_half;
 const vals=a.top_losses.slice(0,3);const remainder=a.price_pnl-vals.reduce((v,r)=>v+r.price_pnl,0);
 chart(s,'bar',[...vals.map(r=>r.stockid.replace('BJ','')),'其余股票合计','交易费用','现金分红'],[{name:'损益（万元）',values:[...vals.map(r=>r.price_pnl/1e4),remainder/1e4,-a.fees/1e4,a.dividend/1e4]}],{y:220,h:340,format:'0.00',min:-5,max:1,legend:false,labels:true});
 foot(s,'价差−113,045元 − 费用11,997元 + 分红1,281元 = −123,761元（取整）。\n本期仍比等权少亏6.23个百分点；资金曲线不能单独证明亏损的经济原因。','来源：持仓、订单、每日账户；柱图单位：万元；价差=期末市值−期初市值+卖出额−买入额');
}
// 23: extra controls explicitly show exposure mismatch and pure model tradeoff.
{
 const s=base(23,'超额有多少来自选股，有多少来自仓位','主流程平均股票仓位27.79%，期初等权98.55%。直接相减包含了仓位差异');
 table(s,[['同一全期区间','累计收益','最大回撤','平均股票仓位'],['主流程',pct(full['主流程'].return),`${(full['主流程'].max_drawdown*100).toFixed(2)}%`,'27.79%'],['同仓位规则的动态等权',pct(A.same_policy_equal.return),`${(A.same_policy_equal.max_drawdown*100).toFixed(2)}%`,'15.11%'],['纯LightGBM，前20只，无择时',pct(full['纯LightGBM'].return),`${(full['纯LightGBM'].max_drawdown*100).toFixed(2)}%`,'79.21%'],['动态股票池等权，无择时',pct(full['动态等权'].return),`${(full['动态等权'].max_drawdown*100).toFixed(2)}%`,'51.21%']],{y:215,h:300,widths:[520,215,215,210],size:24});
 foot(s,'同仓位规则仍有不同实际仓位：100只分散买入受最低100股等约束影响，不能当成严格风险匹配。\n纯LightGBM收益高、回撤也大。这些历史对照尚不能证明可实现的对冲后收益。','来源：fixed_daily.parquet、equal_pool_daily.csv、本次same_policy_equal_daily.csv；对照参数未重新挑优');
}
// 24: metrics in plain terms, exact periods and definitions.
{
 const s=base(24,'几个指标分别在回答什么','IC看排序，资金曲线看赚赔，回撤看最难熬的阶段；它们不能互相替代');
 table(s,[['指标','白话与公式','本次数据范围'],['平均IC / ICIR','IC逐日衡量排序同不同向\nICIR=每日IC平均÷每日IC标准差','因子预筛：499个信号日\n2024-03-01至2026-03-24'],['相对基准超额','同一时期策略收益−基准收益\n本稿以百分点表示','2024-10-08至2026-09-30\n每条曲线从同一个100开始'],['夏普','(平均日收益−日无风险收益)\n÷日收益标准差 × √252','每日账户收益\n无风险年收益假设1.5%'],['卡玛 / 最大回撤','卡玛=年化收益÷最大回撤\n回撤=离此前最高资产的跌幅','同一全期账户\n卡玛不是金融衍生品的Gamma']],{y:205,h:350,widths:[245,530,385],size:23});
 foot(s,'本稿主判断是“是否跑赢基准，在哪些阶段领先”。ICIR与夏普都不能代替真实扣费资金曲线。','来源：train_models.py、ledger.py；均值/波动使用样本标准差，ICIR本稿不额外乘√252');
}
// 25: measured conclusion and limitations, not a diary of completed/to-do tasks.
{
 const s=base(25,'目前证据支持什么','在已查看过的历史上，五天主流程跑赢两个基准，但不能把全部差额归因于选股');
 text(s,'34.63个百分点',60,225,600,80,50,true);text(s,'全期相对期初等权的收益差',60,315,600,48,28);
 text(s,'23.63个百分点',690,225,530,80,50,true);text(s,'后一年相对期初等权的收益差',690,315,530,48,28);
 line(s,60,405,1160);
 text(s,'边界',60,444,170,45,29,true);text(s,'历史已被研究查看过，尚不是全新时期的独立成绩。\n因子发布时间与缺失值IC口径仍需核验；仓位差异尚未完全排除。',235,444,985,95,27);
 text(s,'判断标准',60,558,170,45,29,true);text(s,'冻结规则后，在后续时期继续看扣费超额、回撤与稳定性。',235,558,985,70,27);
 text(s,'代码：prepare_data.py、train_models.py、ledger.py；本次基准与曲线数据保留在项目分析目录。',60,650,1160,25,17);
}

for(const [i,s] of slides.entries()){
 s.speakerNotes.text='';if(s.tables.items.length)tableOwners.push(i+1);if(s.charts.items.length)chartOwners.push(i+1);
}
const candidate=path.join(B,'visual_candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.mkdir(path.join(B,'slides'),{recursive:true});
const requested=process.argv.find(x=>x.startsWith('--pages='));const pages=requested?requested.slice(8).split(',').map(Number):Array.from({length:25},(_,i)=>i+1);
for(const i of pages){await fs.writeFile(path.join(B,'slides',`slide-${String(i).padStart(2,'0')}.png`),new Uint8Array(await(await p.export({slide:slides[i-1],format:'png',scale:1})).arrayBuffer()));console.log('rendered',i);}
await fs.writeFile(path.join(B,'build_manifest.json'),JSON.stringify({source,sourceSha256:sha,tables:tableOwners,charts:chartOwners},null,2));
if(process.argv.includes('--final')){
 const finalPath=path.join(ROOT,'组会汇报_20261016/量化选股项目_教学图解与超额收益_25页_无备注.pptx');
 const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:25,layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:[FONT],referencePath:source,referenceSha256:sha},verifyArtifactToolImport:true,receiptPath:path.join(B,'validation.json')});
 console.log(JSON.stringify({finalPath:result.finalPath,sha256:result.finalSha256,bytes:result.byteCount}));
}
