sudo bash -c 'ulimit -n 16384; ulimit -u 16384; su "user"'
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export VLLM_USE_V1=1

for pid in $(nvidia-smi | grep ' C ' | awk '{print $5}'); do kill -9 $pid; done