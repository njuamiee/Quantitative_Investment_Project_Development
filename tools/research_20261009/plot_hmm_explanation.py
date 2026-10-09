"""Plot the already-computed strict HMM probabilities and actual exposure for first-report teaching."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'training_runs/teacher9_20261008'
OUT = ROOT / 'training_runs/first_report_20261009'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
p = pd.read_csv(R / 'hmm_strict_probabilities.csv', parse_dates=['date'])
a = pd.read_csv(R / 'hmm_strict_account.csv', parse_dates=['date'])
d = a.merge(p, on='date', validate='one_to_one')
fig, axes = plt.subplots(2, 1, figsize=(11.8, 5.0), sharex=True, constrained_layout=True)
axes[0].plot(d.date, d.p_high_next, color='#b45f06', linewidth=1.4)
axes[0].set(title='严格拟合 HMM：高波动概率与仓位是不同的量', ylabel='下一日高波动概率', ylim=(-.03, 1.03))
axes[1].plot(d.date, d.hmm_strict_cap, color='#1f4e79', linewidth=1.7, label='模型给定的仓位上限')
axes[1].plot(d.date, d.gross_exposure, color='#397f68', linewidth=1.1, label='账户实际股票仓位')
axes[1].set(ylabel='股票资产 / 总资产', ylim=(-.03, 1.05))
for ax in axes:
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=.18)
axes[1].legend(frameon=False, loc='lower left')
for ext in ['png', 'pdf']:
    fig.savefig(OUT / f'hmm_probability_and_exposure.{ext}', dpi=180, bbox_inches='tight')
plt.close(fig)
(OUT / 'figure_provenance.json').write_text(json.dumps({
    'purpose': 'Explain existing strict HMM output; no model retraining or new backtest.',
    'rows': len(d), 'first': str(d.date.min().date()), 'last': str(d.date.max().date()),
    'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [R / 'hmm_strict_probabilities.csv', R / 'hmm_strict_account.csv']},
}, ensure_ascii=False, indent=2), 'utf-8')
print(OUT / 'hmm_probability_and_exposure.png')
