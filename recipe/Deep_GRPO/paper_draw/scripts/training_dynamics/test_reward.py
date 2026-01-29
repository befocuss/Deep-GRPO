import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style(style='ticks')
plt.style.use('../../styles/icml.mplstyle')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

df = pd.read_csv("../../data/training_dynamics/test_reward.csv")

df = df[["Step", 
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "qwen2.5-0.5b-instruct-gsm8k-crpo-no-branch - val-core/GSM8K/reward/mean@1"]]

df = df.rename(columns={
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": "DEEP-GRPO",
    "qwen2.5-0.5b-instruct-gsm8k-crpo-no-branch - val-core/GSM8K/reward/mean@1": "GRPO"
})

df = df[df["Step"] <= 2500]

plot_order = [
    "DEEP-GRPO",
    "GRPO"
]

colors = ['tab:green', 'tab:purple', 'tab:red', 'tab:brown', 'tab:blue']

for i, col_name in enumerate(plot_order):
    ax.plot(df['Step'], df[col_name] * 100, color=colors[i], label=col_name, linewidth=1.0)

    # smoothed_y = biased_ema(df[col_name], smoothing_weight=0.99)
    # ax.plot(df['Step'], smoothed_y * 100, label=col_name, color=colors[i], linewidth=1.5)


ax.set_xlabel(r'\textbf{Step}')
ax.set_ylabel(r'\textbf{Accuracy (\%)}')
ax.legend(labelspacing=0.4, handletextpad=0.5, borderpad=0.4)
# ax.set_ylim(top=67, bottom=42) 

plt.tight_layout()

plt.savefig("../../figures/training_dynamics/test_reward.pdf", dpi=300)