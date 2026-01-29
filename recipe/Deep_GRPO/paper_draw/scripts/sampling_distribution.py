import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.5,
})

def generate_data(L=10, gamma=0.1, w=-3.174598, b=1.426722, constant_success=False):
    indices = np.arange(L)
    ratios = (indices + 1) / L
    x_axis = indices + 1
    
    # 1. P(Success)
    if constant_success:
        prob_success = np.ones(L)
    else:
        z = w * ratios + b
        prob_success = 1.0 / (1.0 + np.exp(-z))
    
    # 2. Position Score
    with np.errstate(divide='ignore'): 
        pos_score = np.power(ratios, gamma)
    
    # 3. Sampling Density
    raw_weights = prob_success * pos_score
    if np.sum(raw_weights) > 0 and not np.isinf(np.sum(raw_weights)):
        final_probs = raw_weights / np.sum(raw_weights)
    else:
        final_probs = np.zeros(L)
        final_probs[0] = 1.0
        
    return x_axis, prob_success, pos_score, final_probs

fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.2), sharex=True)

configs = [
    {"gamma": -20, "title": r"(a) Root Only ($\gamma \ll 0$)", "const_p": False},
    {"gamma": 0,   "title": r"(b) Random ($\gamma = 0$)",      "const_p": True},
    {"gamma": 1,   "title": r"(c) Linear ($\gamma = 1$)",      "const_p": False},
    {"gamma": 2,   "title": r"(d) Square ($\gamma = 2$)",      "const_p": False}
]

handles_list = []
labels_list = []

for i, ax in enumerate(axes):
    cfg = configs[i]
    x, p_succ, p_score, density = generate_data(
        L=10, 
        gamma=cfg["gamma"], 
        constant_success=cfg["const_p"]
    )
    
    if i == 1:
        l1, = ax.plot(x, p_succ, color='tab:green', linestyle='--', alpha=1.0, 
                      linewidth=1.5, zorder=1, label=r'$P(\mathrm{Success})$')
        l2, = ax.plot(x, p_score, color='blue', linestyle=':', alpha=1.0, 
                      linewidth=2.0, zorder=2, label=r'Position Score ($x^{\gamma}$)')
    else:
        l1, = ax.plot(x, p_succ, color='tab:green', linestyle='--', alpha=0.8, 
                      label=r'$P(\mathrm{Success})$')
        l2, = ax.plot(x, p_score, 'b:', alpha=0.8, label=r'Position Score ($x^{\gamma}$)')
    
    ax.set_ylim(0, 1.1)
    ax.set_title(cfg["title"])
    ax.set_xlabel("Position Index")

    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_xlim(0.5, 10.5)
    
    if i == 0:
        ax.set_ylabel("Prob. / Score")
    else:
        ax.tick_params(labelleft=False) 
    
    ax2 = ax.twinx()
    l3, = ax2.plot(x, density, 'r-', linewidth=1.5, label='Sampling Density')
    ax2.fill_between(x, density, color='red', alpha=0.15)
    
    ax2.set_ylim(bottom=0)
    
    ax2.set_yticks([]) 
    
    if i == 3:
        ax2.set_ylabel("Sampling Density", color='red')
    
    if i == 1: 
        handles_list = [l1, l2, l3]
        labels_list = [h.get_label() for h in handles_list]

plt.subplots_adjust(
    top=0.9, 
    bottom=0.42,
    left=0.07,      
    right=0.96,     
    wspace=0.15,
    hspace=0.0
)

fig.legend(handles_list, labels_list, 
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.05),
           ncol=3, 
           frameon=False,
           columnspacing=2.5,
           handlelength=2.5) 

plt.savefig("../figures/sampling_distribution.pdf", format="pdf", dpi=300, bbox_inches='tight')