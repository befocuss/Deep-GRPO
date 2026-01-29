    set -x

    OUTPUT_DIR=/home/user/data
    RUN_NAME=Qwen2.5-0.5B-Instruct-GSM8K-CRPO-tp-pb2-b8-debug-01051500
    MODEL=/home/user/data/hf-models/Qwen2.5-0.5B-Instruct

    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29500
    export NCCL_SOCKET_IFNAME=lo
    export GLOO_SOCKET_IFNAME=lo
    export TP_SOCKET_IFNAME=lo

    export HYDRA_FULL_ERROR=1

    torchrun --nnodes=1 --nproc_per_node=4 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=/home/user/data/checkpoints/Qwen2.5-0.5B-Instruct-GSM8K-CRPO-tp-pb2-b8-debug-01051500/synthesized_data_debug.parquet \
        data.val_files=/home/user/data/checkpoints/Qwen2.5-0.5B-Instruct-GSM8K-CRPO-tp-pb2-b8-debug-01051500/synthesized_data_debug.parquet \
        data.prompt_key=prompt \
        data.response_key=response \
        data.train_batch_size=32 \
        data.max_length=4096 \
        data.micro_batch_size_per_gpu=4 \
        model.partial_pretrain=$MODEL \
        trainer.default_local_dir=$OUTPUT_DIR/checkpoints/$RUN_NAME/SFT \
        trainer.project_name=debug \
        trainer.experiment_name=$RUN_NAME \
        trainer.total_epochs=4 \
        trainer.test_freq=10 \
        trainer.logger=['console','wandb'] $@