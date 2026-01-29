import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import biased_ema

sns.set_style(style='ticks')
plt.style.use('../styles/icml.mplstyle')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

df = pd.read_csv("../data/scale_budget.csv")

df = df[["Relative Time (Process)", 
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p2b4-gamma2.0-lam1.0-01182022 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1b4-gamma2.0-lam1.0-01182022 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1b8-gamma2.0-lam1.0-expandall-01182022 - val-core/GSM8K/reward/mean@1"]]

df = df.rename(columns={
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1b4-gamma2.0-lam1.0-01182022 - val-core/GSM8K/reward/mean@1": r"$P_1B_4$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p2b4-gamma2.0-lam1.0-01182022 - val-core/GSM8K/reward/mean@1": r"$P_2B_4$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": r"$P_1B_8$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1b8-gamma2.0-lam1.0-expandall-01182022 - val-core/GSM8K/reward/mean@1": r"$P_1B_8$-expand-all"
})

df['Training Hours'] = df['Relative Time (Process)'] / 3600

df = df[df["Training Hours"] <= 40]

plot_order = [
    r"$P_1B_4$",
    r"$P_2B_4$",
    r"$P_1B_8$",
    r"$P_1B_8$-expand-all"
]

colors = ['tab:purple', 'tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:brown']

for i, col_name in enumerate(plot_order):
    series_df = df[['Training Hours', col_name]].dropna()
    series_df = series_df.sort_values(by='Training Hours')
    x_coords = series_df['Training Hours']
    y_coords = series_df[col_name] * 100
    
    ax.plot(x_coords, y_coords, color=colors[i], alpha=0.25, linewidth=1.0)
    smoothed_y = biased_ema(y_coords, smoothing_weight=0.99)
    ax.plot(x_coords, smoothed_y, label=col_name, color=colors[i], linewidth=1.5)

ax.set_xlabel(r'\textbf{Training Hours}')
ax.set_ylabel(r'\textbf{Accuracy (\%)')
ax.legend(labelspacing=0.4, handletextpad=0.5, borderpad=0.4)
# ax.set_ylim(top=67, bottom=42) 


plt.tight_layout()

plt.savefig("../figures/scale_budget.pdf", dpi=300)