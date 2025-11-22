# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
HuggingFace hybrid rollout implementation that follows the new BaseRollout interface.
This rollout uses HuggingFace's native generation API for sequence generation in hybrid engine mode.
"""

import logging
import os
from typing import Generator

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["HFHybridRollout"]


class HFHybridRollout(BaseRollout):
    """HuggingFace rollout for hybrid engine mode.
    
    This rollout implementation loads a separate HuggingFace model instance
    for generation, similar to how vLLM and SGLang rollouts work.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Extract dp and tp dimensions from device mesh
        if "dp" in device_mesh.mesh_dim_names:
            self.dp_rank = device_mesh["dp"].get_local_rank()
            self.dp_size = device_mesh["dp"].size()
        else:
            self.dp_rank = 0
            self.dp_size = 1
            
        if "infer_tp" in device_mesh.mesh_dim_names:
            self.tp_rank = device_mesh["infer_tp"].get_local_rank()
            self.tp_size = device_mesh["infer_tp"].size()
        else:
            self.tp_rank = 0
            self.tp_size = 1
        
        logger.info(
            f"Initializing HFRollout on rank {self.local_rank}: "
            f"dp_rank={self.dp_rank}/{self.dp_size}, tp_rank={self.tp_rank}/{self.tp_size}"
        )
        
        # Model will be loaded lazily when needed (after weight updates)
        self.model = None
        self.model_path = model_config.local_path
        self.trust_remote_code = model_config.trust_remote_code
        self.hf_config = model_config.hf_config
        
        # Store device
        self.device = get_torch_device()
        
        logger.info(f"HFHybridRollout initialized on rank {self.local_rank}")

    def _ensure_model_loaded(self):
        """Lazily load the model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading HF model from {self.model_path} on rank {self.local_rank}")
            
            # Load model config
            if self.hf_config is None:
                self.hf_config = AutoConfig.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.trust_remote_code
                )
            
            # Load model
            # For now, we load on a single device per DP rank
            # In the future, this could be extended to support tensor parallelism
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.hf_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self.trust_remote_code,
                device_map={"": self.device},
            )
            self.model.eval()
            
            logger.info(f"HF model loaded on rank {self.local_rank}")

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.
        
        For HF hybrid rollout, this ensures the model is loaded.
        
        Args:
            tags: weights or kv_cache.
        """
        self._ensure_model_loaded()
        logger.info(f"HFHybridRollout resumed on rank {self.local_rank} with tags: {tags}")

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update the weights of the rollout model.
        
        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        self._ensure_model_loaded()
        
        # Update model weights
        state_dict = dict(weights)
        
        if state_dict:
            # Load the state dict into the model
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading state dict on rank {self.local_rank}: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading state dict on rank {self.local_rank}: {unexpected_keys}")
            
            logger.info(f"Updated weights on rank {self.local_rank}: {len(state_dict)} parameters")

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            get_torch_device().empty_cache()
            logger.info(f"Released HFHybridRollout model on rank {self.local_rank}")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences using HuggingFace's generate API.
        
        Args:
            prompts: The input prompts.
            
        Returns:
            The output sequences.
        """
        self._ensure_model_loaded()
        
        batch_size = prompts.batch.batch_size[0]
        
        # Handle micro-batching if configured
        micro_batch_size = self.config.get("micro_batch_size", batch_size)
        num_chunks = max(batch_size // micro_batch_size, 1)
        
        if num_chunks > 1:
            batch_prompts = prompts.chunk(chunks=num_chunks)
            output = [self._generate_minibatch(p) for p in batch_prompts]
            output = DataProto.concat(output)
        else:
            output = self._generate_minibatch(prompts)
        
        return output

    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        """Generate sequences for a minibatch.
        
        Args:
            prompts: The input prompts for this minibatch.
            
        Returns:
            The generated sequences.
        """
        # Extract generation parameters
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)
        
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))
        
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        
        # Prepare generation config
        if not do_sample:
            # Greedy decoding
            gen_kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # Validation with specific sampling parameters
            gen_kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,
            }
        else:
            # Normal sampling
            gen_kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,
            }
        
        generation_config = GenerationConfig(**gen_kwargs)
        
        # Extract input tensors
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        
        prompt_length = input_ids.size(1)
        
        # Generate sequences
        self.model.eval()
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            do_sample=do_sample,
            max_new_tokens=response_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            generation_config=generation_config,
            output_scores=False,
            return_dict_in_generate=True,
            use_cache=True,
        )
        
        # Extract generated sequences
        seq = output.sequences
        generated_batch_size = seq.size(0)
        
        # Pad to expected response length if needed
        sequence_length = prompt_length + response_length
        delta_length = sequence_length - seq.shape[1]
        
        if delta_length > 0:
            delta_tokens = torch.full(
                size=(generated_batch_size, delta_length),
                fill_value=pad_token_id,
                device=seq.device,
                dtype=seq.dtype
            )
            seq = torch.cat((seq, delta_tokens), dim=1)
        
        assert seq.shape[1] == sequence_length, (
            f"Expected sequence length {sequence_length}, got {seq.shape[1]}"
        )
        
        # Handle num_return_sequences > 1 if needed
        num_return_sequences = gen_kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
        
        # Split into prompts and responses
        prompts_out = seq[:, :prompt_length]
        responses = seq[:, prompt_length:]
        
        # Update position_ids for response
        response_length_actual = responses.size(1)
        delta_position_id = torch.arange(
            1, response_length_actual + 1,
            device=position_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # Update attention mask for response
        response_attention_mask = get_response_mask(
            response_id=responses,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Create output batch
        batch = TensorDict(
            {
                "prompts": prompts_out,
                "responses": responses,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )
        
        # Empty cache
        get_torch_device().empty_cache()
        
        return DataProto(batch=batch)

