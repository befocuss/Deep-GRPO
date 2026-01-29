# Deep GRPO Training Scripts

This repository contains training scripts for Reinforcement Learning with Self-Play Optimization (SPO), implementing CRPO and GRPO algorithms.

## Directory Structure

```
recipe/spo/examples/
├── H800/                          # Scripts for H800 GPU configurations
├── A100/                          # Scripts for A100 GPU configurations
└── BD/                            # Scripts for internal cluster configurations
    ├── compare_with_baseline/     # Baseline comparison experiments
    ├── debug/                     # Debug configurations
    ├── different_lambda/          # Lambda parameter experiments
    ├── different_sampling_distribution/  # Sampling distribution experiments
    ├── scalability/               # Scalability tests
    ├── download.sh                # Model and dataset download script
    └── find_best_avg_acc.py       # Tool to find best checkpoints
```

## Quick Start

### 1. Download Models and Datasets

```bash
cd recipe/spo/examples/BD
bash download.sh
```

### 2. Run Training

Choose a script based on your hardware:

**H800 Example:**
```bash
cd recipe/spo/examples/H800
bash Qwen2.5-0.5B-Instruct-GSM8K-CRPO.sh
```

**A100 Example:**
```bash
cd recipe/spo/examples/A100
bash Qwen2.5-0.5B-Instruct-GSM8K-CRPO.sh
```

## Script Categories

### Baseline Comparison Scripts

Located in `BD/compare_with_baseline/`, these compare CRPO vs GRPO on different tasks:

**GSM8K (Math):**
```bash
cd recipe/spo/examples/BD/compare_with_baseline/Qwen2.5-0.5B-Instruct-GSM8K
bash Qwen2.5-0.5B-Instruct-GSM8K-CRPO.sh  # CRPO training
bash Qwen2.5-0.5B-Instruct-GSM8K-GRPO.sh  # GRPO training
```

**MATH Dataset:**
```bash
cd recipe/spo/examples/BD/compare_with_baseline/Qwen2.5-Math-1.5B-MATH
bash Qwen2.5-Math-1.5B-MATH-CRPO.sh       # CRPO training
bash Qwen2.5-Math-1.5B-MATH-GRPO.sh       # GRPO training
```

**HotpotQA (Multi-hop QA):**
```bash
cd recipe/spo/examples/BD/compare_with_baseline/Qwen2.5-3B-HotpotQA
bash Qwen2.5-3B-HotpotQA-CRPO.sh          # CRPO training
bash Qwen2.5-3B-HotpotQA-GRPO.sh          # GRPO training
```

### Lambda Parameter Experiments

Located in `BD/different_lambda/`, these test different branch chain loss weights:

```bash
cd recipe/spo/examples/BD/different_lambda
bash Qwen2.5-0.5B-Instruct-GSM8K-lam0.1.sh  # lambda = 0.1
bash Qwen2.5-0.5B-Instruct-GSM8K-lam0.5.sh  # lambda = 0.5
bash Qwen2.5-0.5B-Instruct-GSM8K-lam2.0.sh  # lambda = 2.0
```

### Sampling Distribution Experiments

Located in `BD/different_sampling_distribution/`, these test different utility sampling distributions:

```bash
cd recipe/spo/examples/BD/different_sampling_distribution
bash Qwen2.5-0.5B-Instruct-GSM8K-gamma0.1.sh   # gamma = 0.1 (low temperature)
bash Qwen2.5-0.5B-Instruct-GSM8K-gamma1.0.sh   # gamma = 1.0 (medium)
bash Qwen2.5-0.5B-Instruct-GSM8K-gamma3.0.sh   # gamma = 3.0 (high temperature)
bash Qwen2.5-0.5B-Instruct-GSM8K-random.sh     # random sampling baseline
```

### Scalability Tests

Located in `BD/scalability/`, these test different parallelism configurations:

```bash
cd recipe/spo/examples/BD/scalability
bash Qwen2.5-0.5B-Instruct-GSM8K-p1b4.sh          # 1 process, batch 4
bash Qwen2.5-0.5B-Instruct-GSM8K-p1b8.sh          # 1 process, batch 8
bash Qwen2.5-0.5B-Instruct-GSM8K-p2b4.sh          # 2 processes, batch 4
bash Qwen2.5-0.5B-Instruct-GSM8K-p1b4-expandall.sh  # expand all branches
```

### Debug Scripts

Located in `BD/debug/`, these are lightweight configurations for debugging:

```bash
cd recipe/spo/examples/BD/debug
bash Qwen2.5-0.5B-Instruct-GSM8K-CRPO.sh
bash Qwen2.5-3B-HotpotQA-CRPO.sh
```

## Utility Scripts

### find_best_avg_acc.py

Finds the best checkpoint from W&B logs by averaging accuracy across multiple datasets:

```python
# Edit the RUN_PATH in the script
RUN_PATH = "your-team/project/run-id"

# Run analysis
python recipe/spo/examples/BD/find_best_avg_acc.py
```

Output shows Top 10 checkpoints ranked by average accuracy across:
- MATH, AIME24, Minerva, AMC, OlympiadBench (for math tasks)

### test_server.py

Tests API server connectivity:

```bash
python recipe/spo/examples/BD/compare_with_baseline/Qwen2.5-3B-HotpotQA/test_server.py
```

## Key Configuration Parameters

### CRPO-specific Parameters

```bash
actor_rollout_ref.rollout.spo.expand_branch_chain=True     # Enable branch chains
actor_rollout_ref.rollout.spo.branches_per_point=4-8       # Branches per point
actor_rollout_ref.rollout.spo.pick_branch_chain_root_method=random|utility|teacher_selection
actor_rollout_ref.actor.spo.branch_chain_loss_lambda=1.0   # Branch loss weight
```

### Common Training Parameters

```bash
data.train_batch_size=64                                    # Batch size
actor_rollout_ref.actor.optim.lr=1e-6                       # Learning rate
actor_rollout_ref.rollout.n=8                               # Samples per prompt
actor_rollout_ref.rollout.gpu_memory_utilization=0.5-0.6    # GPU memory usage
trainer.n_gpus_per_node=4-8                                 # GPUs per node
trainer.total_epochs=20-100                                 # Training epochs
```

## Supported Models

- Qwen2.5-0.5B-Instruct
- Qwen2.5-3B
- Qwen2.5-Math-1.5B
- Qwen2.5-Math-7B
- DS-R1-Distill-Qwen-1.5B

## Supported Datasets

- **GSM8K**: Grade school math problems
- **MATH**: Competition math problems
- **AIME24**: Math competition problems
- **HotpotQA**: Multi-hop question answering
- **Minerva**: Math reasoning dataset
- **AMC**: Math competition dataset
- **OlympiadBench**: Math olympiad problems

## Environment Setup

```bash
# Set environment variables
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline  # Optional: offline mode

# Configure paths
export MODEL=/path/to/model
export DATA_DIR=/path/to/datasets
export OUTPUT_DIR=/path/to/output
```

## Monitoring

All scripts support W&B logging. Logs are saved to:
- Training logs: `$OUTPUT_DIR/logs/$RUN_NAME.log`
- Validation results: `$OUTPUT_DIR/validation/$RUN_NAME/`
- Checkpoints: `$OUTPUT_DIR/checkpoints/$RUN_NAME/`

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Enable param offload: `fsdp_config.param_offload=True`
- Lower GPU memory utilization

**Slow Training:**
- Use async mode: `rollout.mode=async`
- Increase number of workers
- Enable gradient checkpointing

**Connection Issues:**
- Check API base URL configuration
- Test server connectivity with `test_server.py`
- Verify network access to required services

## License

See LICENSE file in the repository root.
