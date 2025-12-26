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

from collections import defaultdict
import multiprocessing
import os

import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _worker_process_batch(args):
    """Worker function that processes a batch of indices sequentially."""
    worker_id, indices, data_items, tokenizer, compute_score, reward_fn_key, overlong_buffer_cfg, max_resp_len, log_interval, data_len = args
    
    results = []
    for i in range(len(indices)):
        data_item = data_items[indices[i]]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        eos_token = tokenizer.eos_token
        if response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        result = compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        score: float
        reward_extra_info_item = {}
        if isinstance(result, dict):
            score = result["score"]
            # Store the information including original reward
            for key, value in result.items():
                reward_extra_info_item[key] = value
        else:
            score = result
            reward_extra_info_item["acc"] = score

        reward = score

        if overlong_buffer_cfg is not None and overlong_buffer_cfg.enable:
            overlong_buffer_len = overlong_buffer_cfg.len
            expected_len = max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if overlong_buffer_cfg.log:
                reward_extra_info_item["overlong_reward"] = overlong_reward
                reward_extra_info_item["overlong"] = overlong_reward < 0

        results.append({
            "index": indices[i],  # Correct global index
            "reward": reward, 
            "valid_response_length": valid_response_length,
            "reward_extra_info_item": reward_extra_info_item,
            "data_source": data_source,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "result": result,
            "score": score,
        })
    print(f"worker_process_batch results calculated for id={worker_id}")
    return results


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        num_workers=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        # Default to CPU count, but allow override
        self.num_workers = num_workers if num_workers is not None else min(32, (os.cpu_count() or 1))

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        if self.num_workers == 1:
            return self.one_thread_call(data, return_dict)
        else:
            return self.multi_process_call(data, return_dict)

    def multi_process_call(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        print(f"********************* multi_process_call num_workers={self.num_workers} *********************")
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
        data_len = len(data)
        log_interval = max(1, min(100, data_len // 100))
        
        # Prepare data items for workers (extract all items upfront)
        data_items = [data[i] for i in range(data_len)]
        
        # Distribute indices in round-robin fashion (0->p0, 1->p1, 2->p0, 3->p1, ...)
        worker_indices = [[] for _ in range(self.num_workers)]
        for i in range(data_len):
            worker_id = i % self.num_workers
            worker_indices[worker_id].append(i)
        
        # Prepare arguments for each worker
        worker_args = [
            (
                worker_id,
                indices,
                data_items,
                self.tokenizer,
                self.compute_score,
                self.reward_fn_key,
                self.overlong_buffer_cfg,
                self.max_resp_len,
                log_interval,
                data_len,
            )
            for worker_id, indices in enumerate(worker_indices)
        ]
        
        # Process items in parallel using multiprocessing
        results = [None] * data_len
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            worker_results = pool.map(_worker_process_batch, worker_args)
            
            # Flatten results and place them in the correct positions
            for worker_result_batch in worker_results:
                for result in worker_result_batch:
                    results[result["index"]] = result

        print("********************* multi_process_call results calculated *********************")
        # Process results and build reward tensor
        for result in results:
            i = result["index"]
            reward = result["reward"]
            valid_response_length = result["valid_response_length"]
            reward_extra_info_item = result["reward_extra_info_item"]
            data_source = result["data_source"]
            prompt_str = result["prompt_str"]
            response_str = result["response_str"]
            ground_truth = result["ground_truth"]
            result_obj = result["result"]
            score = result["score"]

            reward_tensor[i, valid_response_length - 1] = reward

            # Aggregate reward_extra_info
            for key, value in reward_extra_info_item.items():
                reward_extra_info[key].append(value)

            # Handle printing
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result_obj, dict):
                    for key, value in result_obj.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def one_thread_call(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        print("********************* one_thread_call *********************")
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

        for i in range(len(data)):
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
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
