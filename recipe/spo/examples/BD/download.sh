# Proxy
export https_proxy=http://placeholder-proxy:8891
export http_proxy=http://placeholder-proxy:8891

export https_proxy=http://placeholder-proxy:8188
export http_proxy=http://placeholder-proxy:8188

# Download models throuth ModelScope
modelscope download --model "Qwen/Qwen2.5-0.5B-Instruct" --local_dir "/data/hf-models/Qwen2.5-0.5B-Instruct"
modelscope download --model "Qwen/Qwen2.5-Math-1.5B" --local_dir "/data/hf-models/Qwen2.5-Math-1.5B"
modelscope download --model "Qwen/Qwen2.5-Math-7B" --local_dir "/data/hf-models/Qwen2.5-Math-7B"

# Download models through hf-mirros
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --resume-download "Qwen/Qwen2.5-0.5B-Instruct" --local-dir "/data/hf-models/Qwen2.5-0.5B-Instruct"
huggingface-cli download --resume-download "Qwen/Qwen2.5-Math-1.5B" --local-dir "/data/hf-models/Qwen2.5-Math-1.5B"
huggingface-cli download --resume-download "Qwen/Qwen2.5-Math-7B" --local-dir "/data/hf-models/Qwen2.5-Math-7B"

# Download datasets
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --repo-type dataset --resume-download "openai/gsm8k" --local-dir "/data/download/gsm8k"

