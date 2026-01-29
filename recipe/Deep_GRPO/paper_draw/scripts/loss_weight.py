# Copyright 2024 Anonymous Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import biased_ema

sns.set_style(style='ticks')
plt.style.use('../styles/icml.mplstyle')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

df = pd.read_csv("../data/loss_weight.csv")

df = df[["Step", 
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam0.1-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam0.5-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1",
        "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam2.0-01161659 - val-core/GSM8K/reward/mean@1"]]

df = df.rename(columns={
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam0.1-01161659 - val-core/GSM8K/reward/mean@1": r"$\lambda=0.1$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam0.5-01161659 - val-core/GSM8K/reward/mean@1": r"$\lambda=0.5$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam1.0-01161659 - val-core/GSM8K/reward/mean@1": r"$\lambda=1.0$",
    "Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma2.0-lam2.0-01161659 - val-core/GSM8K/reward/mean@1": r"$\lambda=2.0$"
})

df = df[df["Step"] <= 1500]

plot_order = [
    r"$\lambda=0.1$",
    r"$\lambda=0.5$",
    r"$\lambda=1.0$",
    r"$\lambda=2.0$"
]

colors = ['tab:red', 'tab:purple', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']

for i, col_name in enumerate(plot_order):
    ax.plot(df['Step'], df[col_name] * 100, color=colors[i], alpha=0.25, linewidth=1.0)

    smoothed_y = biased_ema(df[col_name] * 100, smoothing_weight=0.99)
    ax.plot(df['Step'], smoothed_y, label=col_name, color=colors[i], linewidth=1.5)


ax.set_xlabel(r'\textbf{Step}')
ax.set_ylabel(r'\textbf{Accuracy (\%)')
ax.legend(labelspacing=0.4, handletextpad=0.5, borderpad=0.4)
# ax.set_ylim(top=67, bottom=42) 

plt.tight_layout()

plt.savefig("../figures/loss_weight.pdf", dpi=300)