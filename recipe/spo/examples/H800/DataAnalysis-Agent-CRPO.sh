set -x

ulimit -u 65535

export VLLM_USE_V1=1
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL=/data/hf-models/DataAnalysisReasoner
DATA_DIR=/data/hf-datasets
OUTPUT_DIR=/data
RUN_NAME=DataAnalysisAgent-CRPO-12172118
export WANDB_DIR=$OUTPUT_DIR/wandb/$RUN_NAME

export OPENAI_API_KEY=MY_SECRET
export OPENAI_BASE_URL=http://placeholder-api-server:8000/v1
export TEACHER_MODEL_NAME=Qwen3-235B-A22B-Instruct-2507-AWQ
export DOMAIN_API_DOCS_BASE_DIR=/data/download/ciecc/algorithm_api_docs

/workspace/miniconda3/bin/conda run -p /workspace/miniconda3/envs/verl --no-capture-output python3 -m  verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[$DATA_DIR/deep_analyze/research_task.parquet,$DATA_DIR/deep_analyze/data_task.parquet,$DATA_DIR/deep_analyze/qa_task.parquet]" \
    data.val_files="[$DATA_DIR/dsbench/data_analysis.parquet,$DATA_DIR/dsbench/data_modeling.parquet,$DATA_DIR/dabstep_research/test.parquet,$DATA_DIR/dabstep/test.parquet]" \
    data.train_batch_size=128 \
    data.max_prompt_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
   	actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.max_model_len=49152 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.spo.expand_branch_chain=True \
    actor_rollout_ref.rollout.spo.branches_per_point=5 \
    actor_rollout_ref.rollout.spo.pick_branch_chain_root_method=teacher_selection \
    actor_rollout_ref.rollout.spo.low_quality_trajectory_reward_threshold=0.4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.whiten_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DeepAnalyze' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.validation_data_dir=$OUTPUT_DIR/validation/$RUN_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints/$RUN_NAME \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.max_critic_ckpt_to_keep=10 \
    ray_init.num_cpus=64 \
    trainer.total_epochs=5 | tee $OUTPUT_DIR/logs/$RUN_NAME.log