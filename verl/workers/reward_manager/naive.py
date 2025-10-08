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

import asyncio
import inspect
import time
from collections import defaultdict
from typing import Any
import math
from openai import OpenAIError

import torch

import custom.reward.reward_utils as reward_utils
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        async def subfunction(data: DataProto, return_dict: bool = False):
            """We will expand this function gradually based on the available datasets"""

            # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
            if "rm_scores" in data.batch.keys():
                if return_dict:
                    reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                    reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                    return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
                else:
                    return data.batch["rm_scores"]

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)

            already_print_data_sources = {}

            async def compute_one(i: int):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                extra_info["num_turns"] = num_turns

                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                score = await result if inspect.isawaitable(result) else result
                return (score, valid_response_length, data_source, prompt_str, response_str, ground_truth, i)

            max_retries = 10
            base_delay = 1.0
            bucket_size = 1000

            print(f"DEBUG: reward chunking into {math.ceil(len(data) / bucket_size)} pieces")
            for chunk_idx in range(math.ceil(len(data) / bucket_size)): 
                start = time.time()
                print("DEBUG: starting chunk {chunk_idx}")
                chunk = data[chunk_idx * bucket_size: (chunk_idx + 1) *bucket_size]
                for attempt in range(max_retries + 1):
                    try:
                        rets = await asyncio.wait_for(
                            asyncio.gather(*[compute_one(i) for i in range(chunk_idx *bucket_size, chunk_idx *bucket_size+ len(chunk))]), timeout=(50.0) * (attempt + 1)
                        )
                        break
                    except (asyncio.TimeoutError, OpenAIError) as e:
                        if attempt == max_retries:
                            raise RuntimeError(f"OpenAI unresponsive error: Max retries exceeded {e=}") from e

                        delay = base_delay * 1 # don't change the delay -- I don't think it matters
                        print(f"DEBUG: QUERIES TO OA FAILED: Attempt {attempt + 1} failed, retrying in {delay} seconds...")

                for ret in rets:
                    score ,valid_response_length, data_source, prompt_str, response_str, ground_truth, i = ret 
                    if isinstance(score, dict):
                        reward = score["score"]
                        # Store the information including original reward
                        for key, value in score.items():
                            reward_extra_info["reward_extra_info/" + key].append(value)
                    else:
                        reward = score

                    reward_tensor[i, valid_response_length - 1] = reward

                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                        print("[ground_truth]", ground_truth)
                        if isinstance(score, dict):
                            for key, value in score.items():
                                print(f"[{key}]", value)
                        else:
                            print("[score]", score)

                end = time.time()
                print(f"DEBUG: chunk {chunk_idx} took {end - start:.2f} time") 
            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                    "extra_reward_metrics": reward_utils.extra_reward_metrics(
                        responses=self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True), 
                        prompts=self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True), 
                        ground_truths = [item["ground_truth"] for item in  data.non_tensor_batch["reward_model"]]
                        ) 
                }
            else:
                return reward_tensor
        
        try: 
            asyncio.get_running_loop() 
        except RuntimeError: 
                pass # no loop -> safe
        else:
            raise RuntimeError("RewardManager called inside a running event loop; use async path.") 
        # start_time = time.time()
        subfunc_ret = asyncio.run(subfunction(data, return_dict))
        # end_time = time.time()
        return subfunc_ret