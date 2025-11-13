ulimit -n 32768

source CANN/ascend-toolkit/set_env.sh
source CANN/nnal/atb/set_env.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=1

# bash examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh 2>&1 | tee logs/verl_grpo_$(date +"%Y-%m-%d_%H-%M-%S").log
bash examples/grpo_trainer/run_qwen2_5_0.5b_megatron_npu.sh 2>&1 | tee logs/run_qwen2_5_0.5b_megatron_npu_$(date +"%Y-%m-%d_%H-%M-%S").log
