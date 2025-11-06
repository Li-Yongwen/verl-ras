#!/usr/bin/env bash
set -xeuo pipefail

MODEL_ID=Qwen3-30B-A3B-Instruct
MODEL_PATH=/mnt/hpfs_test/weights/Qwen3-30B-A3B-Instruct
DUMMY_MODEL_PATH=/mnt/hpfs_test/weights/Qwen3-30B-A3B-Instruct-dummy
USE_DIST_CKPT=True
USE_DUMMY_MODEL=True
DUMMY_MODEL_CONFIG_PATH=tests/special_e2e/ppo_trainer/expert_parallel/qwen3moe_minimal.json

mkdir -p ${HOME}/data/dapo-math-17k
\cp -f /mnt/hpfs_test/data/data/dapo-math-17k.parquet ${HOME}/data/dapo-math-17k/dapo-math-17k.parquet

#############################################

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-30B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-${HOME}/dist_ckpt/qwen3_30b_dapo_mindspeed}

if [ "$USE_DUMMY_MODEL" = "True" ]; then
    DUMMY_MODEL_PATH=${DUMMY_MODEL_PATH:-${HOME}/models_dummy/${MODEL_ID}}
    if [ -z "${DUMMY_MODEL_CONFIG_PATH}"  ]; then
        echo "[ERROR] DUMMY_MODEL_CONFIG_PATH not set"
        exit 1
    fi
    if [[ -d $DUMMY_MODEL_PATH && $DUMMY_MODEL_PATH != "/" ]]; then
        rm -rf $DUMMY_MODEL_PATH
    fi

    python scripts/init_random_model.py \
        --hf_model_path "${MODEL_PATH}" \
        --new_config_path "${DUMMY_MODEL_CONFIG_PATH}" \
        --output_path "${DUMMY_MODEL_PATH}"
    MODEL_PATH=$DUMMY_MODEL_PATH
fi

if [ "$USE_DIST_CKPT" = "True" ]; then
    torchrun --nproc_per_node 2 --nnodes 1 scripts/converter_hf_to_mcore.py \
        --hf_model_path "${MODEL_PATH}" \
        --output_path "${DIST_CKPT_PATH}"
fi


exp_name='CI-DAPO-Qwen3-30B-megatron'

# Megatron backen


max_prompt_length=1024
max_response_length=2048
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

train_prompt_bsz=16

actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

python3 -m recipe.dapo.main_dapo \
    --config-name="dapo_megatron_trainer" \
    data.train_files=${HOME}/data/dapo-math-17k/dapo-math-17k.parquet \
    data.val_files=${HOME}/data/dapo-math-17k/dapo-math-17k.parquet \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.shuffle=False \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.gen_batch_size=$((train_prompt_bsz * 2)) \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_dapo_example_mathdapo17k' \
    trainer.experiment_name='qwen3_30b_cut_dapo_mindspeed' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.device=npu \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True $@
