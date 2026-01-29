# DEEP-GRPO

Repository scripts for DEEP-GRPO.

## Quick Start

### 1. Run Training

**Basic training:**
```bash
cd recipe/Deep_GRPO/examples/Cluster/compare_with_baseline/Qwen2.5-0.5B-Instruct-GSM8K
bash Qwen2.5-0.5B-Instruct-GSM8K-DEEP-GRPO.sh

# GSM8K with Qwen2.5-0.5B
bash recipe/Deep_GRPO/examples/Cluster/compare_with_baseline/Qwen2.5-0.5B-Instruct-GSM8K/Qwen2.5-0.5B-Instruct-GSM8K-DEEP-GRPO.sh

# MATH with Qwen2.5-Math-1.5B
bash recipe/Deep_GRPO/examples/Cluster/compare_with_baseline/Qwen2.5-Math-1.5B-MATH/Qwen2.5-Math-1.5B-MATH-DEEP-GRPO.sh

# MATH with Qwen2.5-Math-7B
bash recipe/Deep_GRPO/examples/Cluster/compare_with_baseline/Qwen2.5-Math-7B-MATH/Qwen2.5-Math-7B-MATH-DEEP-GRPO.sh

# DS-R1-Distill model
bash recipe/Deep_GRPO/examples/Cluster/compare_with_baseline/DS-R1-Distill-Qwen-1.5B/DS-R1-Distill-Qwen-1.5B-DEEP-GRPO.sh
```

## Experiment Configurations

**Different lambda values** (branch chain loss weight):
```bash
bash recipe/Deep_GRPO/examples/Cluster/different_lambda/Qwen2.5-0.5B-Instruct-GSM8K-lam0.1.sh
bash recipe/Deep_GRPO/examples/Cluster/different_lambda/Qwen2.5-0.5B-Instruct-GSM8K-lam0.5.sh
bash recipe/Deep_GRPO/examples/Cluster/different_lambda/Qwen2.5-0.5B-Instruct-GSM8K-lam2.0.sh
```

**Different gamma values** (position bias in sampling):
```bash
bash recipe/Deep_GRPO/examples/Cluster/different_sampling_distribution/Qwen2.5-0.5B-Instruct-GSM8K-gamma0.1.sh
bash recipe/Deep_GRPO/examples/Cluster/different_sampling_distribution/Qwen2.5-0.5B-Instruct-GSM8K-gamma1.0.sh
bash recipe/Deep_GRPO/examples/Cluster/different_sampling_distribution/Qwen2.5-0.5B-Instruct-GSM8K-gamma3.0.sh
```

**Scalability tests** (p=pivots, b=batch nums):
```bash
bash recipe/Deep_GRPO/examples/Cluster/scalability/Qwen2.5-0.5B-Instruct-GSM8K-p1b4.sh
bash recipe/Deep_GRPO/examples/Cluster/scalability/Qwen2.5-0.5B-Instruct-GSM8K-p1b4-expandall.sh
```