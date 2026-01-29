# Deep-GRPO

Training scripts for Deep-GRPO (Deep Group Relative Policy Optimization).

## Quick Start

### 1. Download Models and Datasets

```bash
cd recipe/spo/examples/Cluster
bash download.sh
```

### 2. Run Training

**Basic training:**
```bash
cd recipe/spo/examples/Cluster/compare_with_baseline/Qwen2.5-0.5B-Instruct-GSM8K
bash Qwen2.5-0.5B-Instruct-GSM8K-Deep-GRPO.sh
```

**Available models and datasets:**
```bash
# GSM8K with Qwen2.5-0.5B
bash recipe/spo/examples/Cluster/compare_with_baseline/Qwen2.5-0.5B-Instruct-GSM8K/Qwen2.5-0.5B-Instruct-GSM8K-Deep-GRPO.sh

# MATH with Qwen2.5-Math-1.5B
bash recipe/spo/examples/Cluster/compare_with_baseline/Qwen2.5-Math-1.5B-MATH/Qwen2.5-Math-1.5B-MATH-Deep-GRPO.sh

# MATH with Qwen2.5-Math-7B
bash recipe/spo/examples/Cluster/compare_with_baseline/Qwen2.5-Math-7B-MATH/Qwen2.5-Math-7B-MATH-Deep-GRPO.sh

# DS-R1-Distill model
bash recipe/spo/examples/Cluster/compare_with_baseline/DS-R1-Distill-Qwen-1.5B/DS-R1-Distill-Qwen-1.5B-Deep-GRPO.sh
```

## Experiment Configurations

**Different lambda values** (branch chain loss weight):
```bash
bash recipe/spo/examples/Cluster/different_lambda/Qwen2.5-0.5B-Instruct-GSM8K-lam0.1.sh
bash recipe/spo/examples/Cluster/different_lambda/Qwen2.5-0.5B-Instruct-GSM8K-lam0.5.sh
bash recipe/spo/examples/Cluster/different_lambda/Qwen2.5-0.5B-Instruct-GSM8K-lam2.0.sh
```

**Different gamma values** (position bias in sampling):
```bash
bash recipe/spo/examples/Cluster/different_sampling_distribution/Qwen2.5-0.5B-Instruct-GSM8K-gamma0.1.sh
bash recipe/spo/examples/Cluster/different_sampling_distribution/Qwen2.5-0.5B-Instruct-GSM8K-gamma1.0.sh
bash recipe/spo/examples/Cluster/different_sampling_distribution/Qwen2.5-0.5B-Instruct-GSM8K-gamma3.0.sh
```

**Scalability tests** (p=pivots, b=batch size):
```bash
bash recipe/spo/examples/Cluster/scalability/Qwen2.5-0.5B-Instruct-GSM8K-p1b4.sh
bash recipe/spo/examples/Cluster/scalability/Qwen2.5-0.5B-Instruct-GSM8K-p1b4-expandall.sh
```