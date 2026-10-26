"""Build the Chinese whole-project manual from the reviewed Markdown source."""
from pathlib import Path
import re, json, hashlib, base64, html, shutil
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目说明书_20261026'
SOURCE = OUT / '整个量化项目说明书.md'
OUT.mkdir(exist_ok=True)
FIG = OUT / '图表'; FIG.mkdir(exist_ok=True)

def flowchart():
    im=Image.new('RGB',(1800,530),'white'); d=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    boxes=[('数据与股票池','当时能看到什么'),('因子与模型','学习相对排序'),('组合与仓位','买哪些 买多少'),('成交与账本','能否成交 钱多少'),('评价与审计','比较基准 查错误')]
    for i,(title,sub) in enumerate(boxes):
        x=25+i*355
        d.rounded_rectangle((x,95,x+320,280),radius=15,fill='#F0F4F8',outline='#49687E',width=2)
        d.text((x+160,145),title,font=font,fill='#15364E',anchor='mm')
        d.text((x+160,210),sub,font=small,fill='#354958',anchor='mm')
        if i<4:
            d.line((x+325,185,x+350,185),fill='#49687E',width=4)
            d.polygon([(x+350,185),(x+339,177),(x+339,193)],fill='#49687E')
    d.text((900,350),'训练只用已成熟的历史答案  交易只读当时可用的信号',font=font,fill='#15364E',anchor='mm')
    d.text((900,420),'每一轮保留版本 协议 输入 预测 订单 账本与失败结果',font=small,fill='#354958',anchor='mm')
    im.save(FIG/'项目数据流.png')

flowchart()
shutil.copy2(ROOT/'实验更新_20261023/图表/02_净值与回撤.png',FIG/'最新测试期净值与回撤.png')
raw=SOURCE.read_text(encoding='utf-8')
chapters=re.split(r'^## ',raw,flags=re.M)
intro=chapters[0]; sections=[]
for block in chapters[1:]:
    title,body=block.split('\n',1); sections.append((title,body.strip()))

doc=Document(); sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.9);sec.bottom_margin=Cm(1.8);sec.left_margin=Cm(2.1);sec.right_margin=Cm(2.1)
sec.footer_distance=Cm(.8)
styles=doc.styles
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    s=styles[name];s.font.name='Microsoft YaHei'
    s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    s.font.color.rgb=RGBColor.from_string('111111')
styles['Normal'].font.size=Pt(10.5)
styles['Normal'].paragraph_format.line_spacing=1.3
styles['Normal'].paragraph_format.space_after=Pt(7)
styles['Normal'].paragraph_format.widow_control=True
styles['Title'].font.size=Pt(25);styles['Title'].font.bold=True
styles['Title'].paragraph_format.space_after=Pt(18)
styles['Subtitle'].font.size=Pt(13);styles['Subtitle'].paragraph_format.space_after=Pt(18)
styles['Subtitle'].font.italic=False
styles['Heading 1'].font.size=Pt(18);styles['Heading 1'].font.bold=True
styles['Heading 1'].paragraph_format.space_after=Pt(14)
styles['Heading 1'].paragraph_format.keep_with_next=True
styles['Heading 2'].font.size=Pt(13);styles['Heading 2'].font.bold=True
styles['Caption'].font.size=Pt(9);styles['Caption'].paragraph_format.space_after=Pt(8)

def inline(p,text,size=None):
    text=text.replace('`','')
    for j,part in enumerate(re.split(r'\*\*(.*?)\*\*',text)):
        if not part:continue
        r=p.add_run(part);r.bold=(j%2==1)
        if size:r.font.size=Pt(size)
    return p

def ptext(text,style=None):
    p=doc.add_paragraph(style=style);inline(p,text);return p

def table(rows):
    t=doc.add_table(rows=1,cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    n=len(rows[0]);width=16.8
    if n==2: widths=[3.1,13.7]
    elif n==3:widths=[3.6,6.0,7.2]
    elif n==4:widths=[6.6,3.4,3.4,3.4]
    elif n==5:widths=[5.8,2.75,2.75,2.75,2.75]
    else:widths=[width/n]*n
    for i,w in enumerate(widths):t.columns[i].width=Cm(w)
    header=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');header.append(repeat)
    for rownum,row in enumerate(rows):
        cells=t.rows[0].cells if rownum==0 else t.add_row().cells
        pr=cells[0]._tc.getparent().get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
        for i,txt in enumerate(row):
            c=cells[i];c.width=Cm(widths[i]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=c.paragraphs[0];p.paragraph_format.space_after=Pt(2 if len(rows)>15 else 4);p.paragraph_format.space_before=Pt(2 if len(rows)>15 else 4);p.paragraph_format.line_spacing=1.1 if len(rows)>15 else 1.16
            if '/' in txt and len(txt)>45:txt=txt.replace('/','/\u200b').replace('_','_\u200b')
            inline(p,txt,8.8 if len(rows)>15 else 9.4)
            if rownum==0:
                for r in p.runs:r.bold=True
                shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'E6EDF3');c._tc.get_or_add_tcPr().append(shd)
            elif rownum%2==0:
                shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'F7F9FB');c._tc.get_or_add_tcPr().append(shd)
    doc.add_paragraph().paragraph_format.space_after=Pt(1)

def render_body(body):
    lines=body.splitlines();i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s:i+=1;continue
        if s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                vals=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in vals):rows.append(vals)
                i+=1
            table(rows);continue
        ptext(s);i+=1

title=intro.splitlines()[0].lstrip('# ')
doc.add_paragraph(title,style='Title')
intro_lines=[s.strip() for s in intro.splitlines()[1:] if s.strip()]
doc.add_paragraph(intro_lines[0],style='Subtitle')
for s in intro_lines[1:]:ptext(s)
doc.add_paragraph('建议阅读顺序',style='Heading 2')
ptext('先读第01至04章掌握全貌，再读第12章版本地图和第24至26章当前结论。需要理解模型与交易时读第05至11章，追溯各轮实验时读第13至23章。第27至29章提供问答、目录和证据索引。')
ptext('阅读原则  每个收益数字都连同市场 日期 账户初始化 成本与基准一起理解。')
doc.add_page_break();doc.add_paragraph('目录',style='Heading 1')
for i,(heading,_) in enumerate(sections):
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(3);p.paragraph_format.line_spacing=1.0
    inline(p,heading,10)
    # Word evaluates the bookmark page reference during the render pass.
    r=p.add_run('\t');p.paragraph_format.tab_stops.add_tab_stop(Cm(15.8))
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),f'PAGEREF sec_{i+1}')
    r=OxmlElement('w:r');tx=OxmlElement('w:t');tx.text='';r.append(tx);fld.append(r);p._p.append(fld)

for i,(heading,body) in enumerate(sections):
    doc.add_page_break();p=doc.add_paragraph(heading,style='Heading 1')
    bs=OxmlElement('w:bookmarkStart');bs.set(qn('w:id'),str(i+1));bs.set(qn('w:name'),f'sec_{i+1}');p._p.insert(1,bs)
    be=OxmlElement('w:bookmarkEnd');be.set(qn('w:id'),str(i+1));p._p.append(be)
    if heading.startswith('04'):
        doc.add_picture(str(FIG/'项目数据流.png'),width=Cm(16.8))
        ptext('图1 项目的五个环节及共同时间约束',style='Caption')
    render_body(body)
    if heading.startswith('24'):
        doc.add_page_break();doc.add_paragraph('最新测试区间的净值与回撤图',style='Heading 2')
        doc.add_picture(str(FIG/'最新测试期净值与回撤.png'),width=Cm(16.8))
        ptext('图2 来源于10月23日实验的固定图表  测试段为2026年5月22日至10月19日  每边5bp  以5月21日收盘归一化',style='Caption')
        ptext('上图看资产路径，下图看从测试段此前最高点跌下来的幅度。紫线是滚动选模型与规则，橙线是简单固定50%仓位。两条线最后非常接近，说明复杂选择并未在本段显著胜过简单风险预算。图形应结合第24章表格及不确定性解释阅读。[S03]')

footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
settings=doc.settings.element;update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');settings.append(update)
doc.core_properties.title=title;doc.core_properties.subject='全项目方法演进 实验结论与阶段状态'
doc.core_properties.author='';doc.core_properties.keywords='北交所 量化 研究 项目说明 中证2000'
for element in [doc.styles.element,doc.element]:
    for border in list(element.xpath('.//w:pBdr')):border.getparent().remove(border)
doc.save(OUT/'整个量化项目说明书.docx')

def inline_html(s):
    return re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',html.escape(s)).replace('`','')
def body_html(body):
    parts=[];lines=body.splitlines();i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s:i+=1;continue
        if s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                vals=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in vals):rows.append(vals)
                i+=1
            parts.append('<div class="table"><table>'+''.join('<tr>'+''.join(f'<{"th" if j==0 else "td"}>{inline_html(c)}</{"th" if j==0 else "td"}>' for c in row)+'</tr>' for j,row in enumerate(rows))+'</table></div>');continue
        parts.append('<p>'+inline_html(s)+'</p>');i+=1
    return ''.join(parts)
def embedded(path):return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
nav=''.join(f'<a href="#s{i+1}">{html.escape(t)}</a>' for i,(t,b) in enumerate(sections))
content='<h1>'+title+'</h1>'+''.join('<p>'+inline_html(s)+'</p>' for s in intro_lines)
for i,(heading,body) in enumerate(sections):
    content+=f'<section id="s{i+1}"><h2>{html.escape(heading)}</h2>'
    if heading.startswith('04'):content+=f'<img alt="项目数据流" src="{embedded(FIG/"项目数据流.png")}">'
    content+=body_html(body)
    if heading.startswith('24'):content+=f'<figure><img alt="最新测试期净值与回撤" src="{embedded(FIG/"最新测试期净值与回撤.png")}"><figcaption>来源 S03  以2026年5月21日收盘归一化  每边5bp</figcaption></figure>'
    content+='</section>'
css='body{font-family:"Microsoft YaHei",sans-serif;color:#162a39;background:#f2f5f7;margin:0;line-height:1.85}nav{position:fixed;top:0;bottom:0;width:260px;overflow:auto;padding:24px;background:#17394e}nav a{display:block;color:#eef4f7;text-decoration:none;font-size:13px;margin:7px 0}main{margin-left:310px;max-width:920px;background:white;padding:45px 60px}h1{font-size:30px;color:#111}h2{font-size:24px;color:#163b53;margin-top:0}section{padding-top:40px;border-bottom:1px solid #e8edf1;padding-bottom:25px;scroll-margin-top:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce5eb;text-align:left;overflow-wrap:anywhere}th{background:#e6edf3}tr:nth-child(even){background:#f7f9fb}.table{overflow-x:auto}img{max-width:100%;height:auto}figure{margin:22px 0}figcaption{font-size:12px;color:#536471}@media(max-width:1000px){nav{position:relative;width:auto;max-height:230px}main{margin:0;padding:25px}}@media print{nav{display:none}main{margin:0;padding:0}section{break-before:page;border:0}}'
(OUT/'整个量化项目说明书.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><style>'+css+'</style><nav><b style="color:white">项目完整说明</b>'+nav+'</nav><main>'+content+'</main></html>',encoding='utf-8')

sources={}
for match in re.finditer(r'\| (S\d+) \| (.*?) \|',raw):
    ident,path=match.groups()
    if ident=='S12':path='README.md'
    p=ROOT/path
    assert p.is_file(),p
    sources[ident]={'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
manifest={'prepared':'2026-10-26','research_batch':'meeting_update_20261023','data_end':'2026-10-19','chapters':len(sections),'characters':len(raw),'scope':'Python repository only','sources':sources}
(OUT/'来源与版本校验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'chapters':len(sections),'source_characters':len(raw),'outputs':[p.name for p in OUT.iterdir() if p.is_file()]},ensure_ascii=False))
