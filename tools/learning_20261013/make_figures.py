from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'training_runs/learning_20261013/figures'
ASSET.mkdir(parents=True,exist_ok=True)
FONTP=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['axes.unicode_minus']=False
BLUE='#173F55';TEAL='#2C756D';GRAY='#586974'
def box(ax,x,y,w,h,text,color='#E9F1F5',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.045',facecolor=color,edgecolor='#AFC0C9',linewidth=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontproperties=FONTP,fontsize=size,color=BLUE,linespacing=1.6)


def arrow(ax,start,end,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,linewidth=1.1,color=GRAY,linestyle='--' if dashed else '-'))


def figures():
    fig,ax=plt.subplots(figsize=(10,5.7));ax.set(xlim=(0,10),ylim=(0,5.7));ax.axis('off')
    boxes=[(.15,4.5,2.75,.8,'SFTP快照与旧数据\naudit_sync / 历史数据依赖'),(3.5,4.5,2.75,.8,'清洗、股票池、特征、标签\nprepare_data → model_panel'),(6.85,4.5,2.95,.8,'季度筛选、拟合、验证\ntrain_models → 模型与预测'),
           (3.5,2.75,2.75,.8,'月度市场状态与仓位上限\nregimes → 状态概率'),(6.85,2.75,2.95,.8,'现金、股数、成交与费用\nledger → 每日账户'),(6.85,1,2.95,.8,'评估、分段审计、研究报告\nevaluate / period_audit'),
           (.15,1.9,2.75,1.5,'独立研究分支\n分钟、因子择时、短窗口\nLLM公式、Kronos、风险探索'),(.15,.25,6.1,.8,'Jev：离线接口与合成测试；真实文本信号尚未进入主链', '#F7F0DD')]
    for data in boxes:box(ax,*data)
    for a,b in [((2.93,4.9),(3.47,4.9)),((6.28,4.9),(6.82,4.9)),((4.88,4.47),(4.88,3.59)),((6.28,3.15),(6.82,3.15)),((8.32,4.47),(8.32,3.59)),((8.32,2.72),(8.32,1.84)),((3.47,4.7),(1.55,3.44)),((2.93,2.1),(6.82,1.4))]:arrow(ax,a,b)
    arrow(ax,(6.28,3.48),(7.25,4.47))
    ax.text(6.38,4.0,'状态概率',fontproperties=FONTP,fontsize=8.5,color=TEAL,rotation=44)
    # Prices and actions bypass supervised fitting and feed the shared ledger.
    ax.add_patch(FancyArrowPatch((6.22,4.47),(7.1,3.59),connectionstyle='arc3,rad=-.25',arrowstyle='-|>',mutation_scale=11,linewidth=1.1,color=GRAY))
    ax.text(7.28,3.94,'价格/公司行为',fontproperties=FONTP,fontsize=8.5,color=TEAL)
    ax.text(3.45,2.25,'ledger既供验证选型\n也记录外层连续账户',fontproperties=FONTP,fontsize=9.5,color=GRAY)
    fig.tight_layout(pad=.2);fig.savefig(ASSET/'全局数据流.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,2.65));ax.set(xlim=(0,10),ylim=(0,2.65));ax.axis('off')
    box(ax,.1,1.25,2.4,1,'历史多天样本\nM行 × K个因子')
    box(ax,3.6,1.25,2.4,1,'季度 fit(X, y)\n保存固定模型 θ', '#E5F2EA')
    box(ax,7.1,1.25,2.7,1,'每天 predict(X(t))\n约100个股票分数')
    arrow(ax,(2.55,1.75),(3.55,1.75));arrow(ax,(6.05,1.75),(7.05,1.75))
    ax.text(5,.42,'时间通过历史特征、时间切分、标签成熟与季度更新进入流程。\ndate / stockid用于组织记录，不在当前LightGBM的输入列中。',ha='center',fontproperties=FONTP,fontsize=10,color=GRAY)
    fig.tight_layout(pad=.2);fig.savefig(ASSET/'训练与预测.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.1));ax.set(xlim=(0,10),ylim=(0,3.1));ax.axis('off')
    for i,(label,color) in enumerate([('t收盘\n产生信号','#E9F1F5'),('t+1开盘\n买入A批','#E5F2EA'),('t+2…t+4\n继续持有','#E5F2EA'),('t+5收盘\n计划卖A批','#F7F0DD')]):
        box(ax,.12+i*2.5,1.45,2.2,1,label,color,11)
        if i<3:arrow(ax,(2.37+i*2.5,1.95),(2.57+i*2.5,1.95))
    ax.text(5,.65,'A批的预测标签与计划持有端点一致；后续每天还会建立B、C等新批次。\n整个账户每日滚动；每个批次计划持有5个交易日，资金与成交限制会影响实际执行。',ha='center',fontproperties=FONTP,fontsize=10,color=GRAY)
    fig.tight_layout(pad=.2);fig.savefig(ASSET/'五日批次时间线.png',dpi=190);plt.close(fig)


if __name__=="__main__":figures()
