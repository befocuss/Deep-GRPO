import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style(style='ticks')
plt.style.use('../../styles/icml.mplstyle')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

df = pd.read_csv("../../data/tree_methods_training_instability/entropy.csv")

df = df[["Step", 
        "Qwen2.5-0.5B-Instruct-TreeRL - actor/entropy"]]

df = df.rename(columns={
    "Qwen2.5-0.5B-Instruct-TreeRL - actor/entropy": "TreeRL"
})

df = df[df["Step"] <= 2500]

plot_order = [
    "TreeRL"
]

colors = ['tab:green', 'tab:purple', 'tab:red', 'tab:brown', 'tab:blue']

for i, col_name in enumerate(plot_order):
    ax.plot(df['Step'], df[col_name], color=colors[i], label=col_name, linewidth=1.0)

    # smoothed_y = biased_ema(df[col_name], smoothing_weight=0.99)
    # ax.plot(df['Step'], smoothed_y, label=col_name, color=colors[i], linewidth=1.5)


ax.set_xlabel(r'\textbf{Step}')
ax.set_ylabel(r'\textbf{Entropy}')
ax.legend(labelspacing=0.4, handletextpad=0.5, borderpad=0.4)
# ax.set_ylim(top=0.67, bottom=0.42) 

plt.tight_layout()

plt.savefig("../../figures/training_instability/treerl_entropy.pdf", dpi=300)