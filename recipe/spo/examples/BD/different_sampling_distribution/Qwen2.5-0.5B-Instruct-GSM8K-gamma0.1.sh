set -x

export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
# export CUDA_VISIBLE_DEVICES=0

MODEL=/data/hf-models/Qwen2.5-0.5B-Instruct
DATA_DIR=/data/hf-datasets
OUTPUT_DIR=/data
RUN_NAME=Qwen2.5-0.5B-Instruct-GSM8K-CRPO-sp-p1-b8-gamma0.1-lam1.0-01161659
export WANDB_DIR=$OUTPUT_DIR/wandb/$RUN_NAME

export OPENAI_API_KEY=MY_SECRET
export OPENAI_BASE_URL=http://placeholder-api-server:8000/v1
export TEACHER_MODEL_NAME=Qwen3-235B-A22B-Instruct-2507-AWQ

export DOMAIN_API_DOCS_BASE_DIR=/home/user/data/download/project-data/algorithm_api_docs
export RAY_local_fs_capacity_threshold=1

python3 -m  verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[$DATA_DIR/gsm8k/train.parquet]" \
    data.val_files="[$DATA_DIR/gsm8k/test.parquet]" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.spo.branch_chain_loss_lambda=1.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.spo.expand_branch_chain=True \
    actor_rollout_ref.rollout.spo.branches_per_point=8 \
    actor_rollout_ref.rollout.spo.segment_method=sp \
    actor_rollout_ref.rollout.spo.tokens_per_node=50 \
    actor_rollout_ref.rollout.spo.low_quality_trajectory_reward_threshold=0.0 \
    actor_rollout_ref.rollout.spo.pick_branch_chain_root_method=utility \
    actor_rollout_ref.rollout.spo.utility_sampling.prob_model_type=logistic \
    actor_rollout_ref.rollout.spo.utility_sampling.position_bias=0.1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.whiten_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='CRPO' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.validation_data_dir=$OUTPUT_DIR/validation/$RUN_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints/$RUN_NAME \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    ray_init.num_cpus=64 \
    trainer.total_epochs=100 | tee $OUTPUT_DIR/logs/$RUN_NAME.log 