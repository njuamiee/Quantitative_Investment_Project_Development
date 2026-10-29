"""Render the dated Chinese project guide; never recompute experiment results."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目导航_20261029'
DOCUMENTS = [
    ('guide', '项目全貌与消息驱动说明.md', '理解项目'),
    ('catalog', '文档目录.md', '查找文档'),
    ('cleanup', '清理记录.md', '清理与恢复'),
]


def build() -> None:
    pages = []
    navigation = []
    anchors = set()
    doc_ids = {name: key for key, name, _ in DOCUMENTS}
    for key, filename, label in DOCUMENTS:
        source = (OUT / filename).read_text(encoding='utf-8')
        soup = BeautifulSoup(markdown.markdown(source, extensions=['tables', 'fenced_code']), 'html.parser')
        navigation.append(f'<div class="nav-group">{label}</div>')
        for index, heading in enumerate(soup.find_all(['h1', 'h2'])):
            hid = key if index == 0 else f'{key}-{index}'
            heading['id'] = hid
            anchors.add(hid)
            nav_label = label if index == 0 else heading.get_text()
            navigation.append(f'<a href="#{hid}">{html.escape(nav_label)}</a>')
        for link in soup.find_all('a', href=True):
            if link['href'] in doc_ids:
                link['href'] = '#' + doc_ids[link['href']]
            elif link['href'] == '项目阅读导航.html':
                link['href'] = '#guide'
        for table in soup.find_all('table'):
            wrapper = soup.new_tag('div', attrs={'class': 'table-scroll'})
            table.wrap(wrapper)
        pages.append(f'<article data-section="{key}">{soup}</article>')

    def figure(relative: str, caption: str) -> str:
        raw = (ROOT / relative).read_bytes()
        encoded = base64.b64encode(raw).decode()
        return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    charts = ('<section id="charts"><h2>两类资金曲线，分别阅读</h2>'
              '<p>下面直接引用已交付的结果图。日期、策略和成交口径不同，不能将两张图首尾拼接或直接比较累计收益。</p>'
              '<details><summary>展开研究账户曲线 · 检查逐月选择系统的历史表现</summary>'
              + figure('系统重设计_20261026/图表/04_连续资金曲线.png', '研究结果：连续账户曲线；以全组合最终解读的区间、费用和选型规则为准。')
              + '</details><details><summary>展开执行对照曲线 · 检查撮合方式对同一输入的影响</summary>'
              + figure('vnpy数据库对照_20261029/同条件引擎对照.png', '执行结果：五日历史样本；两种撮合均运行于 Linux 服务器。')
              + '</details></section>')
    anchors.add('charts')
    css = """
    :root{--ink:#172a3d;--muted:#596c7c;--accent:#116b71;--line:#d9e2e8;--paper:#fff;--back:#f1f5f7}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
    body{margin:0;background:var(--back);color:var(--ink);font:16px/1.9 'Microsoft YaHei','PingFang SC',sans-serif}
    .cover{background:#15384a;color:#fff;padding:44px max(32px,calc((100vw - 1330px)/2));border-bottom:6px solid #53c4b0}
    .eyebrow{font-size:13px;letter-spacing:2px;color:#afe1dc}.cover h1{font-size:36px;line-height:1.4;margin:10px 0 14px}
    .cover p{max-width:960px;margin:0;color:#deebf2}.chips{display:flex;flex-wrap:wrap;gap:12px;margin-top:22px}
    .chips span{border:1px solid #617c89;border-radius:6px;padding:4px 12px;font-size:13px}
    .layout{max-width:1390px;margin:28px auto;display:grid;grid-template-columns:255px minmax(0,1fr);gap:28px;padding:0 28px}
    aside{position:sticky;top:18px;max-height:calc(100vh - 40px);overflow:auto;align-self:start;padding:18px 12px;background:#fff;border:1px solid var(--line);border-radius:10px;font-size:13px}
    aside a{display:block;padding:5px 10px;color:#3c5363;text-decoration:none;border-left:2px solid transparent}
    aside a:hover{background:#eff8f7;border-color:var(--accent)}.nav-group{padding:12px 10px 4px;color:var(--accent);font-weight:700}
    main{min-width:0}article,section{background:var(--paper);padding:30px 36px;margin-bottom:24px;border:1px solid var(--line);border-radius:10px}
    article h1{font-size:27px;line-height:1.55;margin:0 0 20px}h2{font-size:23px;line-height:1.55;margin:42px 0 16px;border-top:1px solid var(--line);padding-top:25px}
    h3{font-size:19px;color:var(--accent);margin-top:32px}p{margin:14px 0}a{color:#086d82;text-underline-offset:3px}strong{font-weight:700}
    .table-scroll{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.8;min-width:530px}
    th,td{padding:12px 13px;text-align:left;vertical-align:top;border:1px solid var(--line)}th{background:#edf5f6;color:#204951}tbody tr:nth-child(even){background:#f8fafb}
    code{font-family:Consolas,'Microsoft YaHei',monospace;font-size:.88em;background:#edf2f5;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}
    pre{padding:18px 20px;background:#102c3c;color:#d5eae9;border-radius:8px;overflow-x:auto;line-height:1.8;font-size:13px}pre code{background:none;color:inherit;padding:0;overflow-wrap:normal}
    blockquote{margin:20px 0;padding:4px 18px;border-left:4px solid var(--accent);background:#f0f7f6}li{margin:8px 0}
    details{border:1px solid var(--line);border-radius:7px;margin:16px 0}summary{cursor:pointer;padding:15px;font-weight:bold;color:var(--accent)}figure{margin:16px}img{max-width:100%;height:auto}figcaption{font-size:13px;color:var(--muted)}
    footer{padding:0 28px 30px;text-align:center;color:var(--muted);font-size:12px}
    @media(max-width:950px){.layout{display:block;padding:0 14px}.cover{padding:28px 22px}.cover h1{font-size:29px}aside{position:static;max-height:220px;margin-bottom:20px}article,section{padding:24px 20px}}
    @media print{body{background:#fff;font-size:11pt}.cover{color:#172a3d;background:#fff;padding:0;border-bottom:2px solid #116b71}.cover p,.eyebrow{color:#596c7c}.chips,aside,footer{display:none}.layout{display:block;padding:0;margin:0}article,section{border:0;padding:18px 0}pre{white-space:pre-wrap;color:#172a3d;background:#eee}h2,h3{break-after:avoid}table{min-width:0;font-size:9pt}}
    """
    output = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>项目全貌与消息驱动 · 阅读导航</title>
    <style>''' + css + '''</style></head><body>
    <header class="cover"><div class="eyebrow">北交所量化研究与模拟执行 · 2026-10-29</div>
    <h1>把整个项目重新看清楚</h1>
    <p>本地与服务器做什么，消息怎样变成交易，实验结果能说明什么。先理解项目，再按目录找到研究图表、执行证据与运行方法。</p>
    <div class="chips"><span>17 个执行模块 · 两端一致</span><span>14 组服务器对照实验</span><span>8 组时序检查通过</span><span>实时行情 · 尚未接通</span></div></header>
    <div class="layout"><aside aria-label="阅读目录">''' + ''.join(navigation) + '''<div class="nav-group">结果图</div><a href="#charts">展开两类资金曲线</a></aside><main>''' + ''.join(pages) + charts + '''</main></div>
    <footer>本页整理既有结果，没有新增收益实验。相邻文档链接需保留项目目录结构；正文和两张曲线已嵌入本页，可离线阅读。</footer></body></html>'''
    path = OUT / '项目阅读导航.html'
    path.write_text(output, encoding='utf-8')
    parsed = BeautifulSoup(output, 'html.parser')
    problems = []
    links = parsed.find_all('a', href=True)
    for a in links:
        href = a['href']
        parts = urlsplit(href)
        if parts.scheme:
            continue
        if href.startswith('#'):
            if href[1:] not in anchors:
                problems.append(href)
        elif not (OUT / unquote(parts.path)).exists():
            problems.append(href)
    if problems:
        raise ValueError(f'Missing local targets: {problems}')
    check = dict(document=str(path.relative_to(ROOT)), local_and_anchor_links=len(links),
                 broken_links=problems, embedded_result_images=len(parsed.find_all('img')),
                 source_sha256={filename:hashlib.sha256((OUT / filename).read_bytes()).hexdigest() for _,filename,_ in DOCUMENTS},
                 rendering='HTML generated; visual inspection recorded separately')
    (OUT / '文档校验.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'html':str(path),'links':len(links),'broken':len(problems),'bytes':path.stat().st_size},ensure_ascii=False))


if __name__ == '__main__':
    build()
