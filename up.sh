ulimit -n 32768

source CANN/ascend-toolkit/set_env.sh
source CANN/nnal/atb/set_env.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=1

bash recipe/one_step_off_policy/grpo_0.6b_gsm8k_fsdp2_2_6.sh 2>&1 | tee logs/one_step_off/verl_grpo_$(date +"%Y-%m-%d_%H-%M-%S").log
