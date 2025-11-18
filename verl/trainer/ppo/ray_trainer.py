# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Optional
import time

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls, func_generator
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.model import compute_position_id_with_mask
from ray.util.queue import Queue


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.thread_flag = True

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.tokens_queue= Queue()
        self.requests_queue= Queue()
        self.index_prompt_tokens= defaultdict() # 临时存储
        self.requests_tokens = []
        self._index_prompt_tokens_status = Queue()
        #通过_set_tokens_queue_readable_status和_get_tokens_queue_readable_status获取index_prompt_tokens_queue的可读状态
        self._index_prompt_tokens_status.put(1)
        self.index_prompt_tokens_queue = Queue()
        
        # 故障恢复相关标记
        self._temp_worker_group = None  # 临时worker group（包含活着的worker）
        self._need_recover_group = False  # 标记是否需要在推理完成后重拉group
        self._using_temp_worker_group = False  # 标记当前是否正在使用临时worker group

    def _set_tokens_queue_readable_status(self, readable: bool):
        len_queue = self._index_prompt_tokens_status.size()
        for _ in range(len_queue):
            self._index_prompt_tokens_status.get()
        if readable:
            self._index_prompt_tokens_status.put(1)
        else:
            return

    def _get_tokens_queue_readable_status(self):
        return self._index_prompt_tokens_status.size()==1


    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(
        self,
        batch: DataProto,
        input_batch_keys_to_pop: List[str] = [
            "input_ids",
            "attention_mask",
            "position_ids",
        ],
    ) -> DataProto:
        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info"})
            & batch.non_tensor_batch.keys()
        )
        # pop those keys for generation
        batch_keys_to_pop = input_batch_keys_to_pop
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model(self.tokens_queue, self.requests_queue)

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        # rollout_file = f"{rollout_data_dir}/{self.global_steps+1}.data"
        # print(f"DEBUG try load rollout from {rollout_file}")
        # try:
        #     self._recoverd_rollout = DataProto.load_from_disk(rollout_file)
        # except Exception as e:
        #     print(f"DEBUG load rollout data failed, due to {e}")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute IS weights and apply rejection sampling for rollout-training mismatch.

        Computes importance sampling weights to correct for distribution mismatch between
        rollout and training policies. Applies rejection sampling (mask mode/veto) by
        modifying response_mask. Always updates response_mask; conditionally adds IS weights.

        Key behavior:
        - response_mask: ALWAYS updated with rejection (mask mode + veto excluded from training)
        - rollout_is_weights: Added to batch ONLY if config.algorithm.rollout_is=True

        This separation ensures:
        - Rejection works even when IS weights are disabled (rollout_is=False)
        - Metrics can be monitored before enabling IS weight application

        Args:
            batch: DataProto with old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics):
                updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if rollout_is=True)
                metrics: Dict of IS and mismatch metrics, all with "mismatch/" prefix
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch (None = disabled, float = enabled)
        rollout_is_threshold = self.config.algorithm.get("rollout_is_threshold", None)
        if rollout_is_threshold is not None and rollout_is_threshold > 0 and "rollout_log_probs" in batch.batch:
            # Compute IS weights and get modified response_mask
            rollout_is_weights, modified_response_mask, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.get("rollout_is_threshold_lower", None),
                rollout_is_veto_threshold=self.config.algorithm.get("rollout_is_veto_threshold", None),
            )

            # ALWAYS update response_mask with rejection (even if rollout_is=False)
            # - Mask mode: tokens with outlier IS ratios excluded
            # - Veto: sequences with catastrophic tokens excluded
            # This ensures correct loss normalization (rejected samples not in denominator)
            batch.batch["response_mask"] = modified_response_mask

            # Conditionally add IS weights based on rollout_is config flag
            # - rollout_is=True: Enable IS weight correction in policy loss
            # - rollout_is=False: Metrics-only mode (rejection still applied via mask)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights (safety-bounded, mode-processed) to enable weight correction
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def _load_rollout(self):
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        rollout_file = f"{rollout_data_dir}/{self.global_steps}.data"
        print(f"DEBUG try load rollout from {rollout_file}")
        try:
            return DataProto.load_from_disk(rollout_file)
        except Exception as e:
            print(f"DEBUG load rollout data failed, due to {e}")
            return None

    def _save_rollout(self, batch):
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir is None:
            return
        rollout_file = f"{rollout_data_dir}/{self.global_steps}.data"
        batch.save_to_disk(rollout_file)

    def _parse_req_tokens(self, token_per_req: dict) -> dict:
        """
        从新的 step_result 数据结构中提取 tokens 信息。
        Args:
            token_per_req: step_result 字典，包含：
                - finished_req_ids: 已完成的请求 ID 列表(全局id)
                - req_info: 字典，key 为 global_req_id，value 包含：
                    - req_id: vLLM 的 req_id
                    - sampled_token_ids: 采样的 token IDs
                    - req_id_to_index: vLLM 内部的索引
        Returns:
            字典，格式为 {global_req_id: {"raw_prompt_ids": [...], "new_token_ids": [...]}}
        """
        while not self._get_tokens_queue_readable_status():
            time.sleep(1)
        self._set_tokens_queue_readable_status(readable=False)
        if self.index_prompt_tokens_queue.size()==0:
            self.index_prompt_tokens = {}
        elif self.index_prompt_tokens_queue.size()==1:
            self.index_prompt_tokens=self.index_prompt_tokens_queue.get()
        else:
            raise RuntimeError("index_prompt_tokens_queue size error")
        req_info = token_per_req.get("req_info", {})
        finished_global_ids=token_per_req.get("finished_global_ids",{})
        for global_req_id, req_info in req_info.items():
            sampled_token_ids = req_info.get("sampled_token_ids", [])
            if hasattr(sampled_token_ids, 'tolist'):
                new_token_ids = sampled_token_ids.tolist()
            elif isinstance(sampled_token_ids, (list, tuple)):
                new_token_ids = list(sampled_token_ids)
            else:
                new_token_ids = [sampled_token_ids] if sampled_token_ids is not None else []
            self.index_prompt_tokens.setdefault(global_req_id, {
                "raw_prompt_ids": [],
                "new_token_ids": []
            })
            self.index_prompt_tokens[global_req_id]["new_token_ids"].extend(new_token_ids)
        for finished_global_id in finished_global_ids:
            self.index_prompt_tokens.pop(finished_global_id)
        self.index_prompt_tokens_queue.put(self.index_prompt_tokens)
        self._set_tokens_queue_readable_status(readable=True)


    @ray.remote(num_cpus=1)
    def catch_rollout_tokens(self):
        while True:
            token_per_req = self.tokens_queue.get()
            self._parse_req_tokens(token_per_req)


    def _recover_actor_rollout_ref_wg(self):
        print("[INFO] Recreating actor rollout and reference policy worker groups")

        # release the placement groups and actor_rollout_wg
        try:
            if hasattr(self, "actor_rollout_wg") and self.actor_rollout_wg is not None:
                try:
                    actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                    actor_rollout_resource_pool.release_placement_groups()
                except Exception as e:
                    print(f"Error releasing actor rollout placement group: {e}")

                del self.actor_rollout_wg
                print("[INFO] Old actor rollout wg terminated")
        except Exception as e:
            print(f"[WARN] Failed to cleanup old actor rollout worker: {e}")


        # release the ref_wg
        try:
            if hasattr(self, "ref_policy_wg") and self.ref_policy_wg is not None:
                del self.ref_policy_wg
                print("[INFO] Old ref wg terminated")
        except Exception as e:
            print(f"[WARN] Failed to cleanup old ref worker: {e}")

        # get resource pool
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)

        # create new actor/ref RayClassWithInitArgs
        actor_rollout_cls = RayClassWithInitArgs(
            cls = self.role_worker_mapping[Role.ActorRollout],
            config = self.config.actor_rollout_ref,
            role=str(Role.ActorRollout),
        )

        ref_cls = RayClassWithInitArgs(
            cls = self.role_worker_mapping[Role.RefPolicy],
            config = self.config.actor_rollout_ref,
            role=str(Role.RefPolicy),
        )

        class_dict = {
            str(Role.ActorRollout): actor_rollout_cls,
            str(Role.RefPolicy): ref_cls,
        }

        # update resource pool to cls dict
        if resource_pool not in self.resource_pool_to_cls:
            self.resource_pool_to_cls[resource_pool] = {}
        self.resource_pool_to_cls[resource_pool].update(class_dict)

        # create worker dict cls and wg
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        self.actor_rollout_wg = spawn_wg[str(Role.ActorRollout)]
        self.ref_policy_wg = spawn_wg[str(Role.RefPolicy)]

        # initialize models
        print("initializing actor rollout models")
        self.actor_rollout_wg.init_model()
        print("actor rollout models initialized")
        print("initializing ref policy models")
        self.ref_policy_wg.init_model()
        print("ref policy models initialized")

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

        print("[INFO] Actor rollout and reference policy worker groups recovered")

    def _identify_associated_workers(self, dead_worker_indices, original_total_workers):
        """
        识别与已死worker关联的worker（TP组、节点等）。
        
        Args:
            dead_worker_indices: 已死worker的索引列表
            original_total_workers: 原始worker总数
            
        Returns:
            set: 需要排除的worker索引集合（包括直接死掉的+关联的）
        """
        associated_dead_workers = set(dead_worker_indices)
        
        # 尝试从config获取TP/DP信息
        # 优先从rollout配置获取，如果没有则从actor的megatron配置获取
        tp_size = None
        dp_size = 1
        
        if hasattr(self, 'config') and hasattr(self.config, 'actor_rollout_ref'):
            # 尝试从rollout配置获取TP/DP信息
            if hasattr(self.config.actor_rollout_ref, 'rollout'):
                rollout_config = self.config.actor_rollout_ref.rollout
                if hasattr(rollout_config, 'tensor_model_parallel_size'):
                    tp_size = rollout_config.tensor_model_parallel_size
                if hasattr(rollout_config, 'data_parallel_size'):
                    dp_size = rollout_config.data_parallel_size
            
            # 如果rollout配置中没有，尝试从actor的megatron配置获取
            if tp_size is None and hasattr(self.config.actor_rollout_ref, 'actor'):
                if hasattr(self.config.actor_rollout_ref.actor, 'megatron'):
                    megatron_config = self.config.actor_rollout_ref.actor.megatron
                    if hasattr(megatron_config, 'tensor_model_parallel_size'):
                        tp_size = megatron_config.tensor_model_parallel_size
                    # megatron通常通过world_size和tp_size计算dp_size
                    # 这里我们假设可以通过worker总数和tp_size计算
                    if tp_size is not None and original_total_workers % tp_size == 0:
                        dp_size = original_total_workers // tp_size
        
        # 如果找到了TP配置，计算TP组
        if tp_size is not None and tp_size > 1:
            print(f"[INFO] 检测到TP/DP配置: TP={tp_size}, DP={dp_size}")
            print(f"[INFO] 总worker数: {original_total_workers}, 每个TP组大小: {tp_size}")
            
            # 尝试从worker group获取实际的dispatch信息来验证分组
            # 参考原有实现：根据megatron的rank计算公式
            # global_rank = ((pp_rank * dp_size + dp_rank) * cp_size + cp_rank) * tp_size + tp_rank
            # 对于rollout，通常pp_size=1, cp_size=1，所以：global_rank = dp_rank * tp_size + tp_rank
            # 因此：tp_rank = global_rank % tp_size, dp_rank = global_rank // tp_size
            # TP组的分组方式：在同一个DP组内，连续的tp_size个worker组成一个TP组
            # 即：tp_group_id = global_rank // tp_size
            
            # 尝试从worker group查询rollout mesh的dispatch信息来验证分组
            dp_rank_mapping = None
            try:
                if hasattr(self, 'actor_rollout_wg') and self.actor_rollout_wg is not None:
                    # 尝试查询rollout mesh的dispatch信息
                    if hasattr(self.actor_rollout_wg, '_query_dispatch_info'):
                        try:
                            dp_rank_mapping = self.actor_rollout_wg._query_dispatch_info("rollout")
                            print(f"[INFO] 从worker group获取到dispatch信息: {dp_rank_mapping}")
                        except Exception as e:
                            print(f"[WARN] 查询dispatch信息失败: {e}，将使用配置计算")
                    elif "rollout" in getattr(self.actor_rollout_wg, '_dispatch_info', {}):
                        dp_rank_mapping = self.actor_rollout_wg._dispatch_info["rollout"]
                        print(f"[INFO] 从worker group缓存获取到dispatch信息: {dp_rank_mapping}")
            except Exception as e:
                print(f"[WARN] 从worker group获取dispatch信息失败: {e}，将使用配置计算")
            
            # 根据dispatch信息或配置计算TP组
            tp_groups = {}
            if dp_rank_mapping is not None and len(dp_rank_mapping) == original_total_workers:
                # 使用dispatch信息计算TP组
                # 相同dp_rank的worker属于同一个DP组，在DP组内按顺序分组为TP组
                dp_groups = {}
                for worker_idx in range(original_total_workers):
                    dp_rank = dp_rank_mapping[worker_idx]
                    if dp_rank not in dp_groups:
                        dp_groups[dp_rank] = []
                    dp_groups[dp_rank].append(worker_idx)
                
                # 在每个DP组内，按顺序分组为TP组
                for dp_rank, dp_workers in dp_groups.items():
                    # 在DP组内按顺序排序
                    sorted_workers = sorted(dp_workers)
                    # 每tp_size个worker组成一个TP组
                    for local_idx, worker_idx in enumerate(sorted_workers):
                        tp_group_id_in_dp = local_idx // tp_size
                        # 使用全局唯一的TP组ID：dp_rank * (每个DP组的TP组数) + tp_group_id_in_dp
                        # 每个DP组的TP组数 = len(dp_workers) // tp_size
                        tp_groups_per_dp = len(sorted_workers) // tp_size
                        tp_group_id = dp_rank * tp_groups_per_dp + tp_group_id_in_dp
                        
                        if tp_group_id not in tp_groups:
                            tp_groups[tp_group_id] = []
                        tp_groups[tp_group_id].append(worker_idx)
            else:
                # 使用配置计算（参考megatron的rank计算）
                # 假设pp_size=1, cp_size=1，则：global_rank = dp_rank * tp_size + tp_rank
                # TP组ID = global_rank // tp_size
                for i in range(original_total_workers):
                    tp_group_id = i // tp_size
                    if tp_group_id not in tp_groups:
                        tp_groups[tp_group_id] = []
                    tp_groups[tp_group_id].append(i)
            
            # 检查每个TP组，如果有worker死了，标记整个TP组为不可用
            for tp_group_id, tp_workers in tp_groups.items():
                dead_in_group = [w for w in tp_workers if w in dead_worker_indices]
                if dead_in_group:
                    print(f"[WARN] TP组 {tp_group_id} 中有 {len(dead_in_group)}/{len(tp_workers)} 个worker死亡: {dead_in_group}")
                    print(f"[WARN] TP组 {tp_group_id} 的所有worker ({tp_workers}) 都应该被排除，因为TP需要完整组")
                    # 将整个TP组的所有worker都标记为需要排除
                    associated_dead_workers.update(tp_workers)
        else:
            print(f"[INFO] 未找到TP配置或TP大小为1，跳过TP组关联检测")
        
        return associated_dead_workers
    
    def _rebuild_temp_worker_group(self, alive_workers, alive_worker_names):
        """
        重建临时worker group。
        
        Args:
            alive_workers: 活着的worker列表
            alive_worker_names: 活着的worker名称列表
            
        Returns:
            RayWorkerGroup: 重建的临时worker group
        """
        from verl.single_controller.ray.base import RayWorkerGroup
        
        # 从原worker group复制ray_cls_with_init以正确绑定方法
        ray_cls_with_init = getattr(self.actor_rollout_wg, 'ray_cls_with_init', None)
        fused_worker_used = getattr(self.actor_rollout_wg, 'fused_worker_used', False)
        
        # 如果alive_worker_names为空，使用worker_handles的数量来创建虚拟names
        if not alive_worker_names:
            alive_worker_names = [f"alive_worker_{i}" for i in range(len(alive_workers))]
        
        # 如果ray_cls_with_init为None，创建一个临时的RayClassWithInitArgs
        if ray_cls_with_init is None:
            from verl.single_controller.ray.base import RayClassWithInitArgs
            import ray
            dummy_cls = ray.remote(lambda: None)
            ray_cls_with_init = RayClassWithInitArgs(cls=dummy_cls)
            ray_cls_with_init.fused_worker_used = fused_worker_used
        
        temp_wg = RayWorkerGroup(
            detached=True,
            worker_handles=alive_workers,
            worker_names=alive_worker_names,
            ray_cls_with_init=ray_cls_with_init,
            device_name=self.device_name,
        )
        
        # 确保_world_size正确设置为活着的worker数量
        temp_wg._world_size = len(alive_workers)
        
        # 绑定worker方法（从原worker group复制）
        original_ray_cls_with_init = getattr(self.actor_rollout_wg, 'ray_cls_with_init', None)
        if original_ray_cls_with_init is not None:
            from verl.single_controller.ray.base import _unwrap_ray_remote, func_generator
            original_cls = _unwrap_ray_remote(original_ray_cls_with_init.cls)
            
            # 绑定方法
            print(f"[INFO] 绑定worker方法，原始类: {original_cls}")
            method_names = temp_wg._bind_worker_method(original_cls, func_generator)
            print(f"[INFO] 已绑定方法: {method_names}")
            
            # 校验 generate_sequences 是否绑定
            if 'generate_sequences' not in method_names:
                print(f"[WARN] generate_sequences 未绑定，已绑定: {method_names}")
                # 检查原始类是否有 generate_sequences 方法
                if hasattr(original_cls, 'generate_sequences'):
                    method = getattr(original_cls, 'generate_sequences')
                    from verl.single_controller.base.decorator import MAGIC_ATTR as MAGIC_ATTR_CHECK
                    if hasattr(method, MAGIC_ATTR_CHECK):
                        print("[INFO] 原始类有 generate_sequences 方法，尝试手动绑定")
                        try:
                            from verl.single_controller.base.decorator import (
                                get_predefined_dispatch_fn,
                                get_predefined_execute_fn,
                                MAGIC_ATTR as MAGIC_ATTR_IMPORT,
                                Dispatch,
                            )
                            attribute = getattr(method, MAGIC_ATTR_IMPORT)
                            dispatch_mode = attribute["dispatch_mode"]
                            execute_mode = attribute["execute_mode"]
                            blocking = attribute["blocking"]
                            
                            # 获取 dispatch 和 collect 函数
                            if isinstance(dispatch_mode, Dispatch):
                                fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                                dispatch_fn = fn["dispatch_fn"]
                                collect_fn = fn["collect_fn"]
                            else:
                                dispatch_fn = dispatch_mode["dispatch_fn"]
                                collect_fn = dispatch_mode["collect_fn"]
                            
                            # 获取 execute 函数
                            execute_mode_dict = get_predefined_execute_fn(execute_mode=execute_mode)
                            wg_execute_fn_name = execute_mode_dict["execute_fn_name"]
                            execute_fn = getattr(temp_wg, wg_execute_fn_name)
                            
                            # 生成并绑定方法
                            func = func_generator(
                                temp_wg,
                                'generate_sequences',
                                dispatch_fn=dispatch_fn,
                                collect_fn=collect_fn,
                                execute_fn=execute_fn,
                                blocking=blocking,
                            )
                            setattr(temp_wg, 'generate_sequences', func)
                            print("[INFO] ✓ generate_sequences 方法手动绑定成功")
                        except Exception as e:
                            print(f"[ERROR] 手动绑定 generate_sequences 失败: {e}")
                            import traceback
                            traceback.print_exc()
            
            temp_wg.ray_cls_with_init = original_ray_cls_with_init
        
        # 最终验证：确保 generate_sequences 方法存在且可调用
        if not hasattr(temp_wg, 'generate_sequences'):
            raise AttributeError(
                f"临时worker group缺少 generate_sequences 方法！"
                f"已绑定的方法: {getattr(temp_wg, 'method_names', 'unknown')}, "
                f"原worker group有方法: {hasattr(self.actor_rollout_wg, 'generate_sequences')}"
            )
        
        # 验证方法是否可调用
        if not callable(getattr(temp_wg, 'generate_sequences', None)):
            raise AttributeError(
                f"临时worker group的 generate_sequences 方法不可调用！"
            )
        
        print("[INFO] ✓ 临时worker group重建成功，generate_sequences 方法可用且可调用")
        return temp_wg
    
    def _get_current_worker_group(self):
        """
        获取当前应该使用的worker group。
        如果正在使用临时worker group，会检测其中的worker是否还活着，
        如果有worker挂了，会清除相关worker（包括关联的worker）并重建临时worker group。
        
        Returns:
            RayWorkerGroup: 当前应该使用的worker group
        """
        # 如果不在使用临时worker group，直接返回原worker group
        if not self._using_temp_worker_group or self._temp_worker_group is None:
            return self.actor_rollout_wg
        
        # 检测临时worker group中的worker是否还活着
        print("[INFO] 检测临时worker group中的worker状态...")
        temp_wg = self._temp_worker_group
        dead_worker_indices_in_temp = []
        alive_workers_in_temp = []
        alive_worker_names_in_temp = []
        
        # 获取worker names（可能为空）
        worker_names = getattr(temp_wg, '_worker_names', [])
        original_total_workers = len(self.actor_rollout_wg._workers)
        
        # 需要建立临时worker group中的worker索引到原worker group中的worker索引的映射
        # 由于临时worker group只包含活着的worker，我们需要找到每个worker在原worker group中的位置
        temp_to_original_index_map = {}
        for temp_idx, temp_worker in enumerate(temp_wg._workers):
            # 尝试找到这个worker在原worker group中的索引
            found = False
            for orig_idx, orig_worker in enumerate(self.actor_rollout_wg._workers):
                # 通过比较worker的actor_id来判断是否是同一个worker
                try:
                    if hasattr(temp_worker, '_actor_id') and hasattr(orig_worker, '_actor_id'):
                        if temp_worker._actor_id == orig_worker._actor_id:
                            temp_to_original_index_map[temp_idx] = orig_idx
                            found = True
                            break
                except:
                    pass
            
            if not found:
                # 如果找不到映射，假设临时worker group中的worker索引对应原worker group中的前几个worker
                # 这是一个简化的假设，可能不总是正确
                print(f"[WARN] 无法找到临时worker {temp_idx} 在原worker group中的位置，使用简化映射")
                temp_to_original_index_map[temp_idx] = temp_idx
        
        # 检测临时worker group中的每个worker
        for temp_idx, worker in enumerate(temp_wg._workers):
            # 第一步：检查Ray actor状态
            is_alive_ray = temp_wg._is_worker_alive(worker)
            
            if not is_alive_ray:
                orig_idx = temp_to_original_index_map.get(temp_idx, temp_idx)
                print(f"[WARN] 临时worker group中的worker {temp_idx} (原索引 {orig_idx}) Ray actor状态为DEAD")
                dead_worker_indices_in_temp.append(orig_idx)
                continue
            
            # 第二步：尝试实际调用worker来验证是否真正可用
            is_functional = True
            try:
                import ray
                # 尝试多种方式获取节点信息
                node_id = None
                
                # 方式1：如果worker有get_node_id方法，直接调用
                if hasattr(worker, 'get_node_id'):
                    try:
                        node_id_future = worker.get_node_id.remote()
                        node_id = ray.get(node_id_future, timeout=2.0)
                    except:
                        pass
                
                # 方式2：使用__ray_call__调用ray.get_runtime_context().get_node_id()
                if node_id is None and hasattr(worker, '__ray_call__'):
                    try:
                        node_id_future = worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                        node_id = ray.get(node_id_future, timeout=2.0)
                    except:
                        pass
                
                # 方式3：从Ray actor信息中获取节点ID
                if node_id is None:
                    try:
                        from ray.experimental.state.api import get_actor
                        worker_state_dict = get_actor(worker._actor_id.hex())
                        if worker_state_dict is not None:
                            # 从actor信息中获取节点ID
                            node_id = worker_state_dict.get("address", {}).get("nodeId") or worker_state_dict.get("node_id")
                    except:
                        pass
                
                if node_id is not None:
                    orig_idx = temp_to_original_index_map.get(temp_idx, temp_idx)
                    print(f"[INFO] 临时worker group中的worker {temp_idx} (原索引 {orig_idx}) 在节点 {node_id} 上，状态正常")
                else:
                    # 如果无法获取节点信息，但worker还活着，仍然认为它是可用的
                    orig_idx = temp_to_original_index_map.get(temp_idx, temp_idx)
                    print(f"[WARN] 无法获取临时worker {temp_idx} (原索引 {orig_idx}) 的节点信息，但worker状态正常")
            except Exception as e:
                orig_idx = temp_to_original_index_map.get(temp_idx, temp_idx)
                print(f"[WARN] 临时worker group中的worker {temp_idx} (原索引 {orig_idx}) Ray actor显示ALIVE但实际调用失败: {e}")
                is_functional = False
                dead_worker_indices_in_temp.append(orig_idx)
            
            if is_functional:
                alive_workers_in_temp.append(worker)
                if temp_idx < len(worker_names):
                    alive_worker_names_in_temp.append(worker_names[temp_idx])
        
        # 如果没有worker挂了，直接返回临时worker group
        if not dead_worker_indices_in_temp:
            print("[INFO] 临时worker group中的所有worker都正常，继续使用")
            return temp_wg
        
        print(f"[WARN] 发现临时worker group中有 {len(dead_worker_indices_in_temp)} 个worker挂了: {dead_worker_indices_in_temp}")
        
        # 识别关联worker（TP组等）
        associated_dead_workers = self._identify_associated_workers(dead_worker_indices_in_temp, original_total_workers)
        
        # 重新计算活着的worker（从原worker group中）
        alive_workers = []
        alive_worker_names = []
        original_worker_names = getattr(self.actor_rollout_wg, '_worker_names', [])
        
        for i, worker in enumerate(self.actor_rollout_wg._workers):
            if i not in associated_dead_workers:
                # 再次验证worker是否真的活着
                is_alive = self.actor_rollout_wg._is_worker_alive(worker)
                if is_alive:
                    try:
                        import ray
                        # 尝试多种方式验证worker是否可用
                        verified = False
                        
                        # 方式1：如果worker有get_node_id方法，调用它
                        if hasattr(worker, 'get_node_id'):
                            try:
                                node_id_future = worker.get_node_id.remote()
                                ray.get(node_id_future, timeout=1.0)
                                verified = True
                            except:
                                pass
                        
                        # 方式2：使用__ray_call__调用ray.get_runtime_context().get_node_id()
                        if not verified and hasattr(worker, '__ray_call__'):
                            try:
                                node_id_future = worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                                ray.get(node_id_future, timeout=1.0)
                                verified = True
                            except:
                                pass
                        
                        # 方式3：如果前两种方式都失败，但worker状态是ALIVE，仍然认为它是可用的
                        if not verified:
                            # 尝试从Ray actor信息中获取节点ID来验证
                            try:
                                from ray.experimental.state.api import get_actor
                                worker_state_dict = get_actor(worker._actor_id.hex())
                                if worker_state_dict is not None and worker_state_dict.get("state", "undefined") == "ALIVE":
                                    verified = True
                            except:
                                pass
                        
                        if verified:
                            alive_workers.append(worker)
                            if i < len(original_worker_names):
                                alive_worker_names.append(original_worker_names[i])
                        else:
                            print(f"[WARN] Worker {i} 验证失败，无法确认其可用性，排除")
                    except Exception as e:
                        print(f"[WARN] Worker {i} 验证失败: {e}，排除")
        
        if len(associated_dead_workers) > len(dead_worker_indices_in_temp):
            newly_excluded = associated_dead_workers - set(dead_worker_indices_in_temp)
            print(f"[WARN] 由于关联worker（TP组等），额外排除 {len(newly_excluded)} 个worker: {sorted(newly_excluded)}")
        
        if len(alive_workers) == 0:
            print("[ERROR] 所有worker都死了，无法重建临时worker group")
            self._using_temp_worker_group = False
            self._temp_worker_group = None
            self._need_recover_group = True
            return self.actor_rollout_wg
        
        print(f"[INFO] 重建临时worker group，包含 {len(alive_workers)} 个活着的worker")
        
        # 识别挂掉的 workers 所在的 DP 域
        dead_dp_ranks = self._identify_dead_dp_ranks(associated_dead_workers, mesh_name="rollout")
        
        # 重建临时worker group
        temp_wg = self._rebuild_temp_worker_group(alive_workers, alive_worker_names)
        
        # 如果识别到了挂掉的 DP 域，更新 dispatch_info 只保留剩余的 DP 域
        if dead_dp_ranks and "rollout" in self.actor_rollout_wg._dispatch_info:
            try:
                print(f"[INFO] 更新 dispatch_info，排除挂掉的 DP ranks: {sorted(dead_dp_ranks)}")
                original_dp_rank_mapping = self.actor_rollout_wg._dispatch_info["rollout"]
                
                # 获取活着的 workers 在原始 worker group 中的索引
                alive_worker_indices = []
                for i, worker in enumerate(self.actor_rollout_wg._workers):
                    if worker in alive_workers:
                        alive_worker_indices.append(i)
                
                # 只保留不在 dead_dp_ranks 中的 DP ranks
                filtered_dp_rank_mapping = []
                for idx in alive_worker_indices:
                    if idx < len(original_dp_rank_mapping):
                        dp_rank = original_dp_rank_mapping[idx]
                        if dp_rank not in dead_dp_ranks:
                            filtered_dp_rank_mapping.append(dp_rank)
                
                # 更新临时 worker group 的 dispatch_info
                if len(filtered_dp_rank_mapping) == len(temp_wg._workers):
                    temp_wg._dispatch_info["rollout"] = filtered_dp_rank_mapping
                    print(f"[INFO] 成功更新 dispatch_info，剩余 {len(filtered_dp_rank_mapping)} 个 workers")
                else:
                    print(f"[WARN] DP rank 映射长度不匹配: {len(filtered_dp_rank_mapping)} vs {len(temp_wg._workers)}")
            except Exception as e:
                print(f"[WARN] 更新 dispatch_info 失败: {e}，继续使用临时 worker group")
        
        self._temp_worker_group = temp_wg
        return self._temp_worker_group
    
    def _get_alive_worker_group(self):
        """
        检测活着的worker并创建一个只包含活着worker的临时worker group。
        
        Returns:
            tuple: (RayWorkerGroup, bool) - 第一个元素是临时worker group（如果所有worker都活着则为None），
                   第二个元素表示是否所有worker都死了（True表示所有worker都死了，需要完全恢复）
        """
        if not hasattr(self, "actor_rollout_wg") or self.actor_rollout_wg is None:
            return None, False
        
        # 检测活着的worker
        # 不仅要检查Ray actor状态，还要验证worker是否真正可用
        alive_workers = []
        alive_worker_names = []
        dead_worker_indices = []
        
        # 获取worker names（可能为空）
        worker_names = getattr(self.actor_rollout_wg, '_worker_names', [])
        
        print(f"[INFO] 开始检测 {len(self.actor_rollout_wg._workers)} 个worker的状态...")
        
        for i, worker in enumerate(self.actor_rollout_wg._workers):
            # 第一步：检查Ray actor状态
            is_alive_ray = self.actor_rollout_wg._is_worker_alive(worker)
            
            if not is_alive_ray:
                print(f"[WARN] Worker {i} Ray actor状态为DEAD")
                dead_worker_indices.append(i)
                continue
            
            # 第二步：尝试实际调用worker来验证是否真正可用
            # 使用多种方式验证worker是否可用
            is_functional = True
            node_id = None
            try:
                # 使用actor方法尝试获取worker的node_id和device_id以判定可用性
                import ray
                if node_id is None and hasattr(worker, '__ray_call__'):
                    try:
                        node_id_future = worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                        node_id = ray.get(node_id_future, timeout=2.0)
                        print(f"[INFO] Worker {i} 在节点 {node_id} 上，状态正常")
                    except Exception as e:
                        print(f"[WARN] Worker {i} 调用__ray_call__失败: {e}")
                # 如果所有方式都失败，标记为不可用
                if node_id is None:
                    print(f"[WARN] Worker {i} Ray actor显示ALIVE但无法验证其可用性")
                    print(f"[WARN] 可能是GPU进程被杀但Ray actor未更新状态，标记为不可用")
                    is_functional = False
                    dead_worker_indices.append(i)
            except Exception as e:
                print(f"[WARN] Worker {i} 验证过程出错: {e}")
                is_functional = False
                dead_worker_indices.append(i)
            
            if is_functional:
                alive_workers.append(worker)
                if i < len(worker_names):
                    alive_worker_names.append(worker_names[i])
            else:
                dead_worker_indices.append(i)
        
        if dead_worker_indices:
            print(f"[WARN] 发现 {len(dead_worker_indices)} 个不可用的worker: {dead_worker_indices}")
        
        # 第三步：按节点/设备分组检测，避免将已死设备上的worker包含进来
        # 如果一张卡/一个NPU上的进程被杀，同设备上的其他worker可能也会受影响
        # 使用 (node_id, device_id) 作为唯一标识，以区分同一节点上的不同设备
        worker_node_device_map = {}  # (node_id, device_id) -> [worker_indices]
        worker_node_device_id_map = {}  # worker_index -> (node_id, device_id)
        
        # 收集所有活着的worker的节点和设备信息
        for i, worker in enumerate(self.actor_rollout_wg._workers):
            if i not in dead_worker_indices:
                node_id = None
                device_id = None
                try:
                    import ray
                    import os
                    
                    # 获取节点ID
                    # 方式1：如果worker有get_node_id方法，直接调用
                    if hasattr(worker, 'get_node_id'):
                        try:
                            node_id_future = worker.get_node_id.remote()
                            node_id = ray.get(node_id_future, timeout=1.0)
                        except:
                            pass
                    
                    # 方式2：使用__ray_call__调用ray.get_runtime_context().get_node_id()
                    if node_id is None and hasattr(worker, '__ray_call__'):
                        try:
                            node_id_future = worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                            node_id = ray.get(node_id_future, timeout=1.0)
                        except:
                            pass
                    
                    # 方式3：从Ray actor信息中获取节点ID
                    if node_id is None:
                        try:
                            from ray.experimental.state.api import get_actor
                            worker_state_dict = get_actor(worker._actor_id.hex())
                            if worker_state_dict is not None:
                                # 从actor信息中获取节点ID
                                node_id = worker_state_dict.get("address", {}).get("nodeId") or worker_state_dict.get("node_id")
                        except:
                            pass
                    
                    # 获取设备ID（NPU/GPU ID）
                    # 方式1：使用__ray_call__获取LOCAL_RANK或CUDA_VISIBLE_DEVICES
                    if hasattr(worker, '__ray_call__'):
                        try:
                            # 尝试获取LOCAL_RANK和CUDA_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES
                            device_info_future = worker.__ray_call__.remote(
                                lambda self: (
                                    os.environ.get("LOCAL_RANK", "-1"),
                                    os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                    os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
                                )
                            )
                            local_rank, cuda_visible, ascend_visible = ray.get(device_info_future, timeout=1.0)
                            
                            # 优先使用LOCAL_RANK，如果没有则使用VISIBLE_DEVICES的第一个值
                            if local_rank != "-1" and local_rank:
                                device_id = f"local_rank_{local_rank}"
                            elif cuda_visible:
                                # CUDA_VISIBLE_DEVICES可能是逗号分隔的列表，取第一个
                                device_id = f"cuda_{cuda_visible.split(',')[0]}"
                            elif ascend_visible:
                                # ASCEND_RT_VISIBLE_DEVICES可能是逗号分隔的列表，取第一个
                                device_id = f"ascend_{ascend_visible.split(',')[0]}"
                        except:
                            pass
                    
                    # 方式2：尝试从Ray的accelerator_ids获取
                    if device_id is None and hasattr(worker, '__ray_call__'):
                        try:
                            from verl.utils.device import get_device_name
                            device_name = get_device_name()  # "NPU" or "GPU"
                            accelerator_info_future = worker.__ray_call__.remote(
                                lambda self: ray.get_runtime_context().get_accelerator_ids()
                            )
                            accelerator_ids = ray.get(accelerator_info_future, timeout=1.0)
                            if device_name in accelerator_ids and accelerator_ids[device_name]:
                                device_id = f"{device_name.lower()}_{accelerator_ids[device_name][0]}"
                        except:
                            pass
                    
                    # 如果无法获取设备ID，使用worker索引作为fallback
                    if device_id is None:
                        device_id = f"worker_{i}"
                        print(f"[WARN] 无法获取worker {i}的设备ID，使用worker索引作为fallback: {device_id}")
                    
                    if node_id is not None:
                        node_device_key = (node_id, device_id)
                        if node_device_key not in worker_node_device_map:
                            worker_node_device_map[node_device_key] = []
                        worker_node_device_map[node_device_key].append(i)
                        worker_node_device_id_map[i] = node_device_key
                        print(f"[INFO] Worker {i} 在节点 {node_id} 设备 {device_id} 上")
                    else:
                        print(f"[WARN] 无法获取worker {i}的节点信息，将跳过节点/设备分组检测")
                except Exception as e:
                    print(f"[WARN] 获取worker {i}的节点/设备信息时出错: {e}")
        
        # 收集所有节点/设备的组合信息
        all_worker_node_devices = list(worker_node_device_map.keys())
        original_total_workers = len(self.actor_rollout_wg._workers)
        
        # 检查每个节点/设备组合上的worker存活情况
        node_devices_with_dead_workers = []
        for (node_id, device_id) in all_worker_node_devices:
            node_device_workers = [i for i in range(original_total_workers) if worker_node_device_id_map.get(i) == (node_id, device_id)]
            node_device_alive_workers = [i for i in node_device_workers if i not in dead_worker_indices]
            if len(node_device_alive_workers) < len(node_device_workers):
                node_devices_with_dead_workers.append(((node_id, device_id), len(node_device_alive_workers), len(node_device_workers)))
                print(f"[WARN] 节点 {node_id} 设备 {device_id} 上有 {len(node_device_workers) - len(node_device_alive_workers)}/{len(node_device_workers)} 个worker死亡")
        
        # 如果某个节点/设备组合上的所有worker都死了，给出警告
        for (node_id, device_id), alive_count, total_count in node_devices_with_dead_workers:
            if alive_count == 0:
                print(f"[ERROR] 节点 {node_id} 设备 {device_id} 上的所有 {total_count} 个worker都死了！")
                print(f"[ERROR] 这可能意味着该节点/卡上的进程被完全杀掉了")
        
        # 第四步：检测TP/DP组的关联worker
        # 在TP/DP配置下，如果同一个TP组中的某个worker死了，该TP组中的其他worker也应该被排除
        # 因为TP组需要所有成员才能正常工作
        associated_dead_workers = set(dead_worker_indices)  # 需要排除的worker集合（包括直接死掉的+关联的）
        
        # 尝试从worker group获取TP/DP配置信息
        # 注意：在colocated worker模式下，不应该直接查询单个worker的dispatch信息
        # 因为WorkerDict本身不注册dispatch信息，而是通过内部的worker来管理
        # 我们应该通过worker group的_query_dispatch_info方法来查询（如果可用）
        # 或者直接使用配置信息
        try:
            # 尝试从原worker group获取dispatch信息（如果已经缓存）
            dp_rank_mapping = None
            if hasattr(self, 'actor_rollout_wg') and self.actor_rollout_wg is not None:
                # 检查是否已经有缓存的dispatch信息
                if "rollout" in getattr(self.actor_rollout_wg, '_dispatch_info', {}):
                    dp_rank_mapping = self.actor_rollout_wg._dispatch_info["rollout"]
                    print(f"[INFO] 从原worker group缓存获取到dispatch信息: {dp_rank_mapping}")
                # 如果原worker group还活着，尝试查询（但可能失败，因为worker已经死了）
                # 这里我们跳过直接查询，因为部分worker可能已经死了
            
            # 尝试从config获取TP/DP信息（如果有的话）
            # 使用统一的_identify_associated_workers方法来识别关联worker
            associated_dead_workers = self._identify_associated_workers(dead_worker_indices, original_total_workers)
            
            # 更新alive_workers，排除关联的worker
            if len(associated_dead_workers) > len(dead_worker_indices):
                newly_excluded = associated_dead_workers - set(dead_worker_indices)
                print(f"[WARN] 由于TP组关联，额外排除 {len(newly_excluded)} 个worker: {sorted(newly_excluded)}")
                
                # 重新计算alive_workers
                alive_workers = []
                alive_worker_names = []
                for i, worker in enumerate(self.actor_rollout_wg._workers):
                    if i not in associated_dead_workers:
                        alive_workers.append(worker)
                        if i < len(worker_names):
                            alive_worker_names.append(worker_names[i])
        except Exception as e:
            print(f"[WARN] 检测TP/DP组关联worker时出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果检测失败，继续使用原来的alive_workers
        
        # 在TP/DP配置下，如果一张卡上的worker死了，同卡上的其他worker可能也无法正常工作
        # 但这里我们先尝试使用活着的worker继续，让后续的generate_sequences来验证
        
        # 如果所有worker都活着，返回None（使用原worker group）
        if len(alive_workers) == len(self.actor_rollout_wg._workers):
            print(f"[INFO] All {len(alive_workers)} workers are alive, using original worker group")
            return None, False
        
        # 如果所有worker都死了，返回None并标记需要完全恢复
        if len(alive_workers) == 0:
            print("[ERROR] All workers are dead, need full recovery")
            return None, True
        
        print(f"[WARN] {len(self.actor_rollout_wg._workers) - len(alive_workers)} workers are dead, "
              f"creating temporary worker group with {len(alive_workers)} alive workers")
        
        # 创建临时worker group，只包含活着的worker
        # 使用detached模式，直接使用现有的worker handles
        # 从原worker group复制ray_cls_with_init以正确绑定方法
        ray_cls_with_init = getattr(self.actor_rollout_wg, 'ray_cls_with_init', None)
        fused_worker_used = getattr(self.actor_rollout_wg, 'fused_worker_used', False)
        
        # 如果alive_worker_names为空，使用worker_handles的数量来创建虚拟names
        # 这样可以确保_world_size被正确设置
        if not alive_worker_names:
            alive_worker_names = [f"alive_worker_{i}" for i in range(len(alive_workers))]
        
        # 如果ray_cls_with_init为None，我们需要创建一个临时的RayClassWithInitArgs
        # 以避免RayWorkerGroup.__init__中的AttributeError
        if ray_cls_with_init is None:
            # 创建一个临时的RayClassWithInitArgs，只用于初始化
            from verl.single_controller.ray.base import RayClassWithInitArgs
            # 创建一个虚拟的类，但实际上不会使用它
            import ray
            dummy_cls = ray.remote(lambda: None)
            ray_cls_with_init = RayClassWithInitArgs(cls=dummy_cls)
            ray_cls_with_init.fused_worker_used = fused_worker_used
        
        temp_wg = RayWorkerGroup(
            detached=True,
            worker_handles=alive_workers,
            worker_names=alive_worker_names,
            ray_cls_with_init=ray_cls_with_init,
            device_name=self.device_name,
        )
        
        # 确保_world_size正确设置为活着的worker数量
        temp_wg._world_size = len(alive_workers)
        
        # 绑定worker方法（从原worker group复制）
        # 如果原worker group有ray_cls_with_init，使用它来绑定方法
        original_ray_cls_with_init = getattr(self.actor_rollout_wg, 'ray_cls_with_init', None)
        
        # 如果ray_cls_with_init为None，尝试从其他地方获取
        if original_ray_cls_with_init is None:
            print("[WARN] actor_rollout_wg的ray_cls_with_init为None")
            # 如果ray_cls_with_init为None，但actor_rollout_wg已有generate_sequences方法
            # 我们可以通过创建一个包装函数来使用临时worker group的workers
            if hasattr(self.actor_rollout_wg, 'generate_sequences'):
                print("[INFO] actor_rollout_wg已有generate_sequences方法，创建包装函数")
                try:
                    original_method = getattr(self.actor_rollout_wg, 'generate_sequences')
                    
                    # 创建一个新的方法，使用临时worker group
                    # 注意：我们需要创建一个新的Functor实例，使用temp_wg作为self
                    # 但是，由于func_generator创建的方法会捕获self，我们需要通过闭包来访问
                    def wrapped_generate_sequences(*args, **kwargs):
                        # 保存原始workers
                        original_workers = self.actor_rollout_wg._workers
                        original_world_size = self.actor_rollout_wg._world_size
                        try:
                            # 临时替换为临时worker group的workers
                            self.actor_rollout_wg._workers = temp_wg._workers
                            self.actor_rollout_wg._world_size = temp_wg._world_size
                            # 调用原方法（它会使用self.actor_rollout_wg._workers）
                            return original_method(*args, **kwargs)
                        finally:
                            # 恢复原始workers
                            self.actor_rollout_wg._workers = original_workers
                            self.actor_rollout_wg._world_size = original_world_size
                    
                    setattr(temp_wg, 'generate_sequences', wrapped_generate_sequences)
                    print("[INFO] ✓ 已通过包装函数的方式绑定generate_sequences")
                except Exception as e:
                    print(f"[ERROR] 创建包装函数失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise AttributeError(f"无法为临时worker group创建generate_sequences方法: {e}") from e
            else:
                raise AttributeError(
                    "actor_rollout_wg的ray_cls_with_init为None，且没有已绑定的generate_sequences方法。"
                    "无法创建临时worker group。"
                )
        
        if original_ray_cls_with_init is not None:
            # 解包Ray remote类，获取原始类
            from verl.single_controller.ray.base import _unwrap_ray_remote
            
            # 检查是否是colocated worker模式（fused worker）
            # 参考 spawn_fused 的实现方式
            fused_worker_used = getattr(self.actor_rollout_wg, 'fused_worker_used', False)
            
            if fused_worker_used:
                # Colocated worker模式：需要从raw_cls_dict中获取对应的类
                # 参考 spawn_fused 方法：new_wg._bind_worker_method(self.ray_cls_with_init.cls.raw_cls_dict[key], func_generator)
                print("[INFO] 检测到colocated worker模式，从raw_cls_dict获取类")
                unwrapped_cls = _unwrap_ray_remote(original_ray_cls_with_init.cls)
                
                # 获取raw_cls_dict
                if hasattr(unwrapped_cls, 'raw_cls_dict'):
                    raw_cls_dict = unwrapped_cls.raw_cls_dict
                elif hasattr(original_ray_cls_with_init.cls, 'raw_cls_dict'):
                    raw_cls_dict = original_ray_cls_with_init.cls.raw_cls_dict
                else:
                    raise AttributeError("colocated worker模式下无法找到raw_cls_dict")
                
                # 使用Role.ActorRollout作为key来获取对应的类
                actor_rollout_key = str(Role.ActorRollout)
                if actor_rollout_key not in raw_cls_dict:
                    raise AttributeError(f"在raw_cls_dict中找不到key: {actor_rollout_key}, 可用的keys: {list(raw_cls_dict.keys())}")
                
                original_cls = raw_cls_dict[actor_rollout_key]
                print(f"[INFO] 从raw_cls_dict获取类: {original_cls} (key: {actor_rollout_key})")
            else:
                # 非colocated worker模式：直接使用解包后的类
                original_cls = _unwrap_ray_remote(original_ray_cls_with_init.cls)
                print(f"[INFO] 非colocated worker模式，使用解包后的类: {original_cls}")
            
            # 绑定方法
            print(f"[INFO] 绑定worker方法，原始类: {original_cls}")
            method_names = temp_wg._bind_worker_method(original_cls, func_generator)
            print(f"[INFO] 已绑定方法: {method_names}")
            
            # 检查是否需要重新绑定带前缀的方法
            # 即使fused_worker_used=False，如果使用了create_colocated_worker_cls，方法也可能有前缀
            # 参考 spawn 方法中的 _rebind_actor_methods 逻辑
            actor_rollout_prefix = str(Role.ActorRollout) + "_"
            has_prefixed_methods = any(method_name.startswith(actor_rollout_prefix) for method_name in method_names)
            
            if has_prefixed_methods:
                print(f"[INFO] 检测到带前缀的方法，重新绑定，prefix: {actor_rollout_prefix}")
                
                # 查找所有带前缀的方法并重新绑定
                for method_name in list(method_names):
                    if method_name.startswith(actor_rollout_prefix):
                        original_method_name = method_name.removeprefix(actor_rollout_prefix)
                        if hasattr(temp_wg, method_name):
                            method = getattr(temp_wg, method_name)
                            setattr(temp_wg, original_method_name, method)
                            print(f"[INFO] 重新绑定方法: {method_name} -> {original_method_name}")
                            if original_method_name not in method_names:
                                method_names.append(original_method_name)
        
        # 最终验证：确保 generate_sequences 方法存在
        if not hasattr(temp_wg, 'generate_sequences'):
            raise AttributeError(
                f"临时worker group缺少 generate_sequences 方法！"
                f"已绑定的方法: {getattr(temp_wg, 'method_names', 'unknown')}, "
                f"原worker group有方法: {hasattr(self.actor_rollout_wg, 'generate_sequences')}, "
                f"ray_cls_with_init: {original_ray_cls_with_init is not None}"
            )
        
        print("[INFO] ✓ 临时worker group创建成功，generate_sequences 方法可用")
        return temp_wg, False

    def modify_json_file(self):
        import json
        import time
        self.thread_flag = False
        time.sleep(10)

        try:
            with open('raise_flag.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data['write_flag']:
                data['write_flag'] = True
                data['raise_flag'] = True
                with open('raise_flag.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    print("The json file written success")
        except Exception as e:
            print(f"Error modifying json file: {e}")


    def _reset_tokens_queue(self, retry_times=10):
        while (not self._get_tokens_queue_readable_status() and retry_times>0):
            time.sleep(0.5)
            retry_times-=1
        if retry_times==0:
            raise RuntimeError(f"get_tokens_queue_readable_status failed, have retried {retry_times} times.")
        self._set_tokens_queue_readable_status(readable=False)
        if self.index_prompt_tokens_queue.size() in [0, 1]:
            if self.index_prompt_tokens_queue.size():
                self.index_prompt_tokens_queue.get()
        else:
            raise RuntimeError("index_prompt_tokens_queue size error")
        self._set_tokens_queue_readable_status(readable=True)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.catch_rollout_tokens.remote(self)

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                gen_batch_output_tmp = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                gen_batch_output_ori = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                gen_batch_output.non_tensor_batch["global_id"] = np.array(
                    [str("bing"+str(i)) for i in range(len(gen_batch_output.batch))], dtype=object
                )

                gen_batch_output_tmp.non_tensor_batch["global_id"] = np.array(
                    [str("bing"+str(i)) for i in range(len(gen_batch_output.batch))], dtype=object
                )
                gen_batch_output_ori.non_tensor_batch["global_id"] = np.array(
                    [str("bing"+str(i)) for i in range(len(gen_batch_output.batch))], dtype=object
                    )
                while not self._get_tokens_queue_readable_status():
                    time.sleep(0.5)
                self._set_tokens_queue_readable_status(readable=False)
                prompts = gen_batch_output.non_tensor_batch["raw_prompt_ids"]
                global_ids=gen_batch_output.non_tensor_batch["global_id"]
                if self.index_prompt_tokens_queue.size()==0:
                    self.index_prompt_tokens={}
                else:
                    self.index_prompt_tokens=self.index_prompt_tokens_queue.get()
                for i, global_id in enumerate(global_ids):
                    self.index_prompt_tokens.setdefault(global_id, {"raw_prompt_ids": [], "new_token_ids": []})
                    self.index_prompt_tokens[global_id]["raw_prompt_ids"] = prompts[i]
                self.index_prompt_tokens_queue.put(self.index_prompt_tokens)
                self._set_tokens_queue_readable_status(readable=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            def _update_gen_batch_with_partial_tokens(
                                gen_batch_output_tmp: DataProto,
                            ) -> DataProto:
                                # 1. 从队列中获取第一轮推理的输出 tokens
                                if self.index_prompt_tokens_queue.size()==1:
                                    tmp = self.index_prompt_tokens_queue.get()
                                    all_tokens = tmp  # 格式: {global_id: {"raw_prompt_ids": [...], "new_token_ids": [...]}}
                                    self.index_prompt_tokens_queue.put(tmp)
                                else:
                                    print("error")
                                # 2. 确保使用 gen_batch_output_tmp 的原始数据（未进入第一次 generate_sequences）
                                # gen_batch_output_tmp 已经在第一次调用前创建，包含原始数据
                                batch_size = gen_batch_output_tmp.batch.batch_size[0]
                                # 3. 按照 gen_batch_output_tmp 的 global_id 顺序获取 tokens，确保顺序一致
                                global_ids = gen_batch_output_tmp.non_tensor_batch["global_id"]
                                all_raw_prompt_ids = []
                                all_new_token_ids = []
                                for global_id in global_ids:
                                    token_info = all_tokens.get(global_id, {"raw_prompt_ids": [], "new_token_ids": []})
                                    all_raw_prompt_ids.append(token_info.get('raw_prompt_ids', []))
                                    all_new_token_ids.append(token_info.get('new_token_ids', []))
                                # 4. 计算每请求生成的 tokens 数量
                                per_request_generated_tokens = [0] * batch_size
                                for idx, gen_token_num in enumerate(all_new_token_ids):
                                    if gen_token_num != []:
                                        per_request_generated_tokens[idx] = len(gen_token_num)
                                gen_batch_output_tmp.non_tensor_batch["per_request_generated_tokens"] = np.array(
                                    per_request_generated_tokens
                                )
                                # 5. 构建新的 prompts: raw_prompt_ids + new_token_ids（第一轮生成的 tokens）
                                # 拼接后得到新的 prompt（未 padding）
                                new_prompts = [all_raw_prompt_ids[i] + all_new_token_ids[i] for i in range(batch_size)]
                                new_prompt_lengths = [len(p) for p in new_prompts]
                                max_prompt_len = max(new_prompt_lengths) if new_prompt_lengths else 0
                                # 6. 获取目标 padding 长度
                                # 注意：作为第二次 generate_sequences 的输入，prompts 和 input_ids 应该只包含 prompt 部分
                                # 应该使用第一次输出的 prompts 长度（prompt_length），而不是 input_ids 长度（prompt_length + response_length）
                                target_prompt_len = None
                                if 'input_ids' in gen_batch_output_tmp.batch:
                                    # 使用第一次输出的 prompts 长度（这是 prompt 部分的长度，已经 left-padded）
                                    target_prompt_len = gen_batch_output_tmp.batch['input_ids'].shape[-1]
                                if target_prompt_len is None:
                                    target_prompt_len = max_prompt_len
                                # 确保 target_prompt_len 至少等于 max_prompt_len（不能小于实际内容长度）
                                target_prompt_len = max(target_prompt_len, max_prompt_len)
                                # 7. 获取设备信息（优先从 gen_batch_output 获取，因为它们在同一个设备上）
                                device = None
                                dtype = None
                                if 'input_ids' in gen_batch_output_tmp.batch:
                                    ref_tensor = gen_batch_output_tmp.batch['input_ids']
                                    device = ref_tensor.device
                                    dtype = ref_tensor.dtype
                                elif 'prompts' in gen_batch_output_tmp.batch:
                                    ref_tensor = gen_batch_output_tmp.batch['prompts']
                                    device = ref_tensor.device
                                    dtype = ref_tensor.dtype
                                elif 'responses' in gen_batch_output_tmp.batch:
                                    ref_tensor = gen_batch_output_tmp.batch['responses']
                                    device = ref_tensor.device
                                    dtype = ref_tensor.dtype
                                else:
                                    device = torch.device('cpu')
                                    dtype = torch.long
                                # 8. 获取 pad_token_id（优先从 gen_batch_output_tmp.meta_info 获取）
                                pad_token_id = gen_batch_output_tmp.meta_info.get("pad_token_id")
                                if pad_token_id is None:
                                    pad_token_id = self.tokenizer.pad_token_id
                                if pad_token_id is None:
                                    # 如果 pad_token_id 仍然为 None，使用 eos_token_id（vLLM 的默认行为）
                                    pad_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
                                # 9. Padding prompts 并转换为 tensor（使用 left pad，与原始 gen_batch 保持一致）
                                # 参考 postprocess_data 的实现，使用 left pad（padding 在左侧，有效内容在右侧）
                                # 使用 target_prompt_len 作为 padding 长度，确保与第一次输入的 prompt 部分长度一致
                                unpadded_prompts_list = []
                                for prompt in new_prompts:
                                    if len(prompt) > 0:
                                        prompt_tensor = torch.tensor(prompt, dtype=dtype, device=device)
                                        # 如果长度小于 target_prompt_len，需要 pad 到 target_prompt_len
                                        if len(prompt) < target_prompt_len:
                                            # 使用 left pad：在左侧添加 padding
                                            pad_length = target_prompt_len - len(prompt)
                                            padded = torch.cat([
                                                torch.full((pad_length,), pad_token_id, dtype=dtype, device=device),
                                                prompt_tensor
                                            ])
                                            unpadded_prompts_list.append(padded)
                                        else:
                                            # 如果长度大于 target_prompt_len，截断到 target_prompt_len
                                            unpadded_prompts_list.append(prompt_tensor[:target_prompt_len])
                                    else:
                                        # 空序列，全部 padding
                                        unpadded_prompts_list.append(
                                            torch.full((target_prompt_len,), pad_token_id, dtype=dtype, device=device)
                                        )
                                # 堆叠成 batch tensor
                                prompts_tensor = torch.stack(unpadded_prompts_list, dim=0)
                                # 10. 更新 gen_batch_output_tmp 的必要参数
                                # input_ids 就是 padded_prompts（已经包含了原始 prompt + 第一轮生成的 tokens）
                                gen_batch_output_tmp.batch['input_ids'] = prompts_tensor
                                # 11. 构建 attention_mask: 基于实际长度（有效部分为1，padding部分为0）
                                # 使用 left pad，所以有效部分在右侧
                                # 使用 target_prompt_len 作为长度，确保与第一次输入的 prompt 部分长度一致
                                attention_mask = torch.zeros((batch_size, target_prompt_len), dtype=torch.long, device=device)
                                for i, prompt_len in enumerate(new_prompt_lengths):
                                    # 有效部分在右侧（left pad），但不超过 target_prompt_len
                                    actual_len = min(prompt_len, target_prompt_len)
                                    attention_mask[i, -actual_len:] = 1  # 有效部分在右侧（left pad）
                                gen_batch_output_tmp.batch['attention_mask'] = attention_mask
                                # 12. 计算 position_ids: 基于 attention_mask（与原始 gen_batch 保持一致）
                                position_ids = compute_position_id_with_mask(attention_mask)
                                gen_batch_output_tmp.batch['position_ids'] = position_ids

                                # 13. 更新 raw_prompt_ids（用于后续处理）
                                gen_batch_output_tmp.non_tensor_batch["raw_prompt_ids"] = np.array(
                                    new_prompts, dtype=object)
                                return gen_batch_output_tmp

                            DEBUG_EXCEPTION_ONLY = getattr(self, "debug_exception_only", True)
                            if DEBUG_EXCEPTION_ONLY:
                                try:
                                    raise Exception("DEBUG_EXCEPTION_ONLY active, only running except branch")
                                except Exception as e:
                                    print(f"[WARN] Worker failure detected during generate_sequences: {e}")
                                    # 检测活着的worker并创建临时worker group
                                    breakpoint()
                                    alive_wg, all_dead = self._get_alive_worker_group()
                                    
                                    # 新的策略：即使所有worker都死了，也不立即重拉group
                                    # 而是先尝试使用活着的worker继续推理，等推理全部完成后再重拉group
                                    if all_dead:
                                        print("[ERROR] All workers are dead!")
                                        print("[INFO] 将标记需要重拉group，但先尝试使用临时worker group继续推理")
                                        print("[INFO] 等推理全部完成后，再进行group重拉")
                                        # 标记需要重拉group，但不立即执行
                                        self._need_recover_group = True
                                        # 如果没有活着的worker，无法继续推理，抛出异常
                                        if alive_wg is None:
                                            raise RuntimeError(
                                                "所有worker都死了，且无法创建临时worker group。"
                                                "请检查是否有其他可用的推理实例。"
                                            )
                                    
                                    # 使用活着的worker group（如果所有worker都活着，alive_wg为None，使用原worker group）
                                    if alive_wg is not None:
                                        # 使用临时worker group
                                        self._temp_worker_group = alive_wg
                                        self._using_temp_worker_group = True
                                        worker_group_to_use = alive_wg
                                        print(f"[INFO] 使用临时worker group继续推理（包含 {len(alive_wg._workers)} 个活着的worker）")
                                    else:
                                        # 所有worker都活着，使用原worker group
                                        worker_group_to_use = self.actor_rollout_wg
                                        self._using_temp_worker_group = False
                                    
                                    # 从队列中获取已生成的tokens并更新gen_batch_output
                                    _update_gen_batch_with_partial_tokens(gen_batch_output)
                                    
                                    # 使用活着的worker继续推理
                                    gen_batch_output = worker_group_to_use.generate_sequences(gen_batch_output)
                                    
                                    # 验证输出维度是否正确（与备份比较）
                                    try:
                                        from verl.trainer.ppo.validate_output_dimensions import validate_output_dimensions, print_data_proto_summary
                                        
                                        # 打印摘要信息
                                        print_data_proto_summary(gen_batch_output, "恢复后的 gen_batch_output")
                                        print_data_proto_summary(gen_batch_output_ori, "原始备份 gen_batch_output_ori")
                                        
                                        # 验证维度
                                        validation_result = validate_output_dimensions(
                                            gen_batch_output, 
                                            gen_batch_output_ori, 
                                            verbose=True
                                        )
                                        
                                        if not validation_result['is_valid']:
                                            print(f"[ERROR] 输出维度验证失败！发现 {len(validation_result['errors'])} 个错误")
                                            # 可以选择抛出异常或继续执行
                                            # raise ValueError(f"Output dimension validation failed: {validation_result['errors']}")
                                        else:
                                            print("[INFO] ✓ 输出维度验证通过！")
                                    except Exception as e:
                                        print(f"[WARN] 维度验证时出错: {e}")
                                    
                                    # 重置per_request_generated_tokens，因为这是从部分tokens继续生成的
                                    gen_batch_output.non_tensor_batch["per_request_generated_tokens"] = np.zeros_like(
                                        gen_batch_output.non_tensor_batch["per_request_generated_tokens"]
                                    )
                                    
                                finally:
                                    self._reset_tokens_queue()
                            else:
                                try:
                                    import time, threading
                                    if self.thread_flag:
                                        thread = threading.Thread(target=self.modify_json_file)
                                        thread.start()
                                    # 使用当前应该使用的worker group（可能是临时worker group）
                                    current_wg = self._get_current_worker_group()
                                    gen_batch_output = current_wg.generate_sequences(gen_batch_output)
                                except Exception as e:
                                    print(f"[WARN] Worker failure detected during generate_sequences: {e}")
                                    # 检测活着的worker并创建临时worker group
                                    alive_wg, all_dead = self._get_alive_worker_group()
                                    
                                    # 新的策略：即使所有worker都死了，也不立即重拉group
                                    # 而是先尝试使用活着的worker继续推理，等推理全部完成后再重拉group
                                    if all_dead:
                                        print("[ERROR] All workers are dead!")
                                        print("[INFO] 将标记需要重拉group，但先尝试使用临时worker group继续推理")
                                        print("[INFO] 等推理全部完成后，再进行group重拉")
                                        # 标记需要重拉group，但不立即执行
                                        self._need_recover_group = True
                                        # 如果没有活着的worker，无法继续推理，抛出异常
                                        if alive_wg is None:
                                            raise RuntimeError(
                                                "所有worker都死了，且无法创建临时worker group。"
                                                "请检查是否有其他可用的推理实例。"
                                            )
                                    
                                    # 使用活着的worker group（如果所有worker都活着，alive_wg为None，使用原worker group）
                                    if alive_wg is not None:
                                        # 使用临时worker group
                                        self._temp_worker_group = alive_wg
                                        self._using_temp_worker_group = True
                                        worker_group_to_use = alive_wg
                                        print(f"[INFO] 使用临时worker group继续推理（包含 {len(alive_wg._workers)} 个活着的worker）")
                                    else:
                                        # 所有worker都活着，使用原worker group
                                        worker_group_to_use = self.actor_rollout_wg
                                        self._using_temp_worker_group = False
                                    
                                    # 从队列中获取已生成的tokens并更新gen_batch_output
                                    _update_gen_batch_with_partial_tokens(gen_batch_output)
                                    
                                    # 使用活着的worker继续推理
                                    gen_batch_output = worker_group_to_use.generate_sequences(gen_batch_output)
                                    
                                    # 重置per_request_generated_tokens，因为这是从部分tokens继续生成的
                                    gen_batch_output.non_tensor_batch["per_request_generated_tokens"] = np.zeros_like(
                                        gen_batch_output.non_tensor_batch["per_request_generated_tokens"]
                                    )
                                finally:
                                    self._reset_tokens_queue()
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        
                        # 推理完成后，检查是否需要重拉group
                        if self._need_recover_group:
                            print("[INFO] 推理已完成，开始重拉worker group...")
                            try:
                                self._recover_actor_rollout_ref_wg()
                                # 重拉成功后，重置标记并切换到新的worker group
                                self._need_recover_group = False
                                self._using_temp_worker_group = False
                                self._temp_worker_group = None
                                print("[INFO] ✓ Worker group重拉成功，已切换到新的worker group")
                            except Exception as e:
                                print(f"[ERROR] Worker group重拉失败: {e}")
                                import traceback
                                traceback.print_exc()
                                # 如果重拉失败，继续使用临时worker group
                                print("[WARN] 将继续使用临时worker group，下次iteration再尝试重拉")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                # 使用当前应该使用的worker group（可能是临时worker group）
                                current_wg = self._get_current_worker_group()
                                gen_baseline_output = current_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
