import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import biased_ema

sns.set_style(style='ticks')
plt.style.use('../styles/icml.mplstyle')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

df = pd.read_csv("../data/sampling_strategy.csv")

df = df[["Step", 
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma0.1-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma-20.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma1.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma3.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-random-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1"]]

df = df.rename(columns={
    # "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma0.1-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": r"$\gamma=0.1$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma-20.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": "Root Only",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma1.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": "Linear Depth Bias",
    # "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma3.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": r"$\gamma=3.0$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-random-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": "Random",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": "Square Depth Bias"
})

df = df[df["Step"] <= 1500]

plot_order = [
    # r"$\gamma=3.0$",
    "Square Depth Bias",
    "Linear Depth Bias",
    # r"$\gamma=0.1$",
    "Random",
    "Root Only",
]

colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:orange', 'tab:brown']

for i, col_name in enumerate(plot_order):
    ax.plot(df['Step'], df[col_name] * 100, color=colors[i], alpha=0.25, linewidth=1.0)

    smoothed_y = biased_ema(df[col_name] * 100, smoothing_weight=0.99)
    ax.plot(df['Step'], smoothed_y, label=col_name, color=colors[i], linewidth=1.5)


ax.set_xlabel(r'\textbf{Step}')
ax.set_ylabel(r'\textbf{Accuracy (\%)')
ax.legend(labelspacing=0.4, handletextpad=0.5, borderpad=0.4)
ax.set_ylim(top=67, bottom=42) 

plt.tight_layout()

plt.savefig("../figures/sampling_strategies.pdf", dpi=300)