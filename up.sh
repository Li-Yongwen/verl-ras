source /root/anaconda3/bin/activate
conda activate ras
cd /home/wjp/ras/msrl_env/verl


ulimit -n 32768

source /home/l30055792/CANN/ascend-toolkit/set_env.sh
source /home/l30055792/CANN/nnal/atb/set_env.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=1

bash examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh 2>&1 | tee logs/verl_grpo_$(date +"%Y-%m-%d_%H-%M-%S").log



