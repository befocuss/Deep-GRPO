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
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Utility Sampling", layout="wide")

st.title("Utility Sampling")
st.markdown(r"""
$$ Weight = P(Success) \times (ratio)^{gamma} $$
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    
    L = st.slider("Chain Length", 5, 100, 10)
    
    st.subheader("1. Position Bias")
    position_bias = st.slider("gamma", -10.0, 10.0, 0.1, 0.1, help="Higher = greedier for later positions")
    
    st.subheader("2. Success Probability")
    success_mode = st.radio(
        "Distribution:",
        ["Uniform", "Linear Decay", "Logistic (Sigmoid)"]
    )
    
    # Mode specific params
    if success_mode == "Uniform":
        st.caption("P(Success) = 1")
        # No slider needed
        
    elif success_mode == "Linear Decay":
        st.caption("P(Success) = 1 - ratio")
        
    elif success_mode == "Logistic (Sigmoid)":
        st.caption("P(Success) = Sigmoid(wx+b)")
        w = st.slider("w (Slope)", -10.0, -0.1, -5.0)
        b = st.slider("b (Intercept)", -5.0, 5.0, 2.5)

# --- Calculation ---

# 1. Data Prep (1-based index)
indices = np.arange(L) # 0 to L-1
ratios = (indices + 1) / L # 1/L to 1.0

# 2. P(Success) Calculation
if success_mode.startswith("Uniform"):
    # Fix to 1.0 so logic depends purely on position bias
    # (Normalization will handle the scaling)
    prob_success = np.ones(L) 
elif success_mode.startswith("Linear"):
    prob_success = 1.0 - ratios
else:
    z = w * ratios + b
    prob_success = 1.0 / (1.0 + np.exp(-z))

# 3. Position Score Calculation
# Ratio^bias
pos_score = np.power(ratios, position_bias)

# 4. Combined Weights
raw_weights = prob_success * pos_score

# 5. Normalize to Probability Density
if np.sum(raw_weights) > 0:
    final_probs = raw_weights / np.sum(raw_weights)
else:
    final_probs = np.ones(L) / L

# --- Visualization ---

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # X-axis
    x = indices + 1
    
    # Plot P(Success) - Dashed Line
    ax1.plot(x, prob_success, 'g--', alpha=0.5, label='P(Success) Predicted')
    ax1.set_xlabel("Node Index (Position)")
    ax1.set_ylabel("Probability / Score", color='gray')
    ax1.set_ylim(0, 1.1)
    
    # Plot Position Score - Dotted Line
    ax1.plot(x, pos_score, 'b:', alpha=0.5, label=r'Position Score ($x^{\gamma}$)')
    
    # Plot Final Sampling Density - Solid Line with Fill
    ax2 = ax1.twinx()
    ax2.plot(x, final_probs, 'r-', linewidth=3, label='Final Sampling Density')
    ax2.fill_between(x, final_probs, color='red', alpha=0.2)
    ax2.set_ylabel("Sampling Density", color='red')
    ax2.set_ylim(bottom=0)
    
    # Merge Legends -> Upper Right
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # bbox_to_anchor allows placing outside or precise corner
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.title(f"Sampling Distribution: Gamma={position_bias} | Mode={success_mode.split()[0]}")
    st.pyplot(fig)

with col2:
    st.subheader("Stats")
    
    # Find peak
    peak_idx = np.argmax(final_probs)
    peak_ratio = ratios[peak_idx]
    
    st.metric("Peak Index", f"{peak_idx + 1} / {L}")
    st.metric("Peak Ratio", f"{peak_ratio:.2f}")
    st.metric("Prob at Peak", f"{final_probs[peak_idx]:.4f}")
