# export CUDA_VISIBLE_DEVICES=3

CHECK_POINT_PATH=/data/checkpoints/DataAnalysis-Reasoner-GRPO-1222/global_step_90/actor
OUTPUT_PATH=/data/hf-models/DataAnalysisReasoner-GRPO

python scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir $CHECK_POINT_PATH \
  --target_dir $OUTPUT_PATH \