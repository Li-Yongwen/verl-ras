# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket

from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.envs import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import cdiv,deprecate_kwargs
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.engine.async_llm import AsyncLLM, logger


class AsyncFaultRecoverLLM(AsyncLLM):

    def __init__(
            self,
            vllm_config: VllmConfig,
            executor_class: type[Executor],
            log_stats: bool,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
            use_cached_outputs: bool = False,
            log_requests: bool = True,
            start_engine_loop: bool = True,
            stat_loggers: Optional[list[StatLoggerFactory]] = None,
            client_addresses: Optional[dict[str, str]] = None,
            client_count: int = 1,
            client_index: int = 0,
            tokens_queue = None
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        self.log_stats = log_stats or (stat_loggers is not None)
        if not log_stats and stat_loggers is not None:
            logger.info(
                "AsyncLLM created with log_stats=False and non-empty custom "
                "logger list; enabling logging without default stat loggers")

        if self.model_config.skip_tokenizer_init:
            self.tokenizer = None
        else:
            # Tokenizer (+ ensure liveness if running in another process).
            self.tokenizer = init_tokenizer_from_configs(
                model_config=vllm_config.model_config)

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            mm_registry=mm_registry,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)
        if self.observability_config.otlp_traces_endpoint is not None:
            tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: Optional[StatLoggerManager] = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
            )
            self.logger_manager.log_engine_initialized()

        self.output_handler: Optional[asyncio.Task] = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler(tokens_queue=tokens_queue)
        except RuntimeError:
            pass

        if envs.VLLM_TORCH_PROFILER_DIR:
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                envs.VLLM_TORCH_PROFILER_DIR)
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    envs.VLLM_TORCH_PROFILER_DIR,
                    worker_name=worker_name,
                    use_gzip=True))
        else:
            self.profiler = None

    @classmethod
    @deprecate_kwargs(
        "disable_log_requests",
        additional_message=("This argument will have no effect. "
                            "Use `enable_log_requests` instead."),
    )
    def from_vllm_config(
            cls,
            vllm_config: VllmConfig,
            start_engine_loop: bool = True,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            stat_loggers: Optional[list[StatLoggerFactory]] = None,
            enable_log_requests: bool = False,
            disable_log_stats: bool = False,
            client_addresses: Optional[dict[str, str]] = None,
            client_count: int = 1,
            client_index: int = 0,
            disable_log_requests: bool = True,  # Deprecated, will be removed
            tokens_queue=None
    ) -> "AsyncLLM":
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            tokens_queue=tokens_queue
        )

    async def generate(
            self,
            prompt: PromptType,
            sampling_params: SamplingParams,
            request_id: str,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            priority: int = 0,
            data_parallel_rank: Optional[int] = None,
            tokens_queue = None
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        if (self.vllm_config.cache_config.kv_sharing_fast_prefill
                and sampling_params.prompt_logprobs):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs")

        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            self._run_output_handler(tokens_queue=tokens_queue)

            tokenization_kwargs: dict[str, Any] = {}
            truncate_prompt_tokens = sampling_params.truncate_prompt_tokens

            _validate_truncation_size(
                self.model_config.max_model_len,
                truncate_prompt_tokens,
                tokenization_kwargs,
            )

            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                tokenization_kwargs=tokenization_kwargs,
                data_parallel_rank=data_parallel_rank,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    def _run_output_handler(self, tokens_queue=None):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        logger_manager = self.logger_manager

        async def output_handler(q):
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()

                    if q is not None:
                        req_info = {}
                        for output in outputs.outputs:
                            req_info[output.request_id] = {}
                            req_info[output.request_id]['new_tokens_ids'] = output.new_token_ids
                            req_info[output.request_id]['finished'] = output.finished
                        await q.put_async(req_info)

                    num_outputs = len(outputs.outputs)

                    iteration_stats = IterationStats() if (
                            log_stats and num_outputs) else None

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                        slices = (outputs.outputs,)
                    else:
                        slices = np.array_split(
                            outputs.outputs,
                            cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))

                    for i, outputs_slice in enumerate(slices):
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats)
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if i + 1 < len(slices):
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        await engine_core.abort_requests_async(
                            processed_outputs.reqs_to_abort)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_manager:
                        logger_manager.record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler(tokens_queue))
