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
import math
import time
from collections import defaultdict
from typing import Any

import ray
import torch
from openai import OpenAIError

import custom.reward.reward_utils as reward_utils
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Default error score for failed tasks
DEFAULT_ERROR_SCORE = {
    "score": 0.0,
    "is_correct": 0.0,
    "extracted_answer": "verification has timed out",
    "did_sel_hint": 0.0,
    "format_score": 0.0,
    "monitor_score": 0.0,
    "question_type": "FAILED",
    "correct_and_format_score": 0.0,
    "monitor_eval": "Task failed",
    "unadjusted_calibration_score": 0.0,
    "verifiers": {
        "omi_correct": False,
        "mathv_correct": False,
        "omi_hintmatch": False,
        "mathv_hintmatch": False,
    },
}


async def process_one(
    tokenizer,
    compute_score_fn,
    prompt_ids,
    response_ids,
    attention_mask,
    prompt_length: int,
    ground_truth,
    data_source: str,
    extra_info: dict,
    i: int,
) -> tuple:
    """Process a single rollout - decode tokens and compute reward.

    This is a standalone async function (not a closure) so it can be pickled by ray.
    """
    # Compute valid lengths
    valid_prompt_length = attention_mask[:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    valid_response_length = attention_mask[prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    # Decode tokens
    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    try:
        result = compute_score_fn(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        # Keep backward compat: handle both async and sync compute_score functions
        score = await result if inspect.isawaitable(result) else result
        # Validate that dict-based scores have the required "score" key
        if isinstance(score, dict) and "score" not in score:
            raise ValueError(
                f"Reward function returned dict without 'score' key. "
                f"Got keys: {list(score.keys())[:500]}{'...' if len(score) > 500 else ''}"
            )
    except Exception as e:
        print(f"WARNING: Task {i} failed with error: {str(e)[:1000]}")
        score = {**DEFAULT_ERROR_SCORE, "monitor_eval": f"Task failed: {e}"}

    return (score, valid_response_length, data_source, prompt_str, response_str, ground_truth, i)


@ray.remote
def compute_several(
    tokenizer,
    compute_score_fn,
    items: list[dict],
) -> list[tuple]:
    """Process multiple rollouts in one ray task with a single event loop.

    Args:
        tokenizer: Tokenizer for decoding token IDs
        compute_score_fn: The reward function to call
        items: List of dicts, each containing args for process_one

    Returns:
        List of tuples (score, valid_response_length, data_source, prompt_str, response_str, ground_truth, i)

    Note: Timeout is handled at the outer level (when calling this via ray), not internally.
          The only internal timeouts are in run_code_isolated_no_files_better_err.
    """

    async def run_all():
        tasks = [process_one(tokenizer=tokenizer, compute_score_fn=compute_score_fn, **item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    return asyncio.run(run_all())


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
        """Compute rewards for a batch of rollouts.

        Uses ray to parallelize across mini-batches, with each ray task running
        asyncio.gather on multiple rollouts. This combines ray parallelism with
        async I/O for optimal performance.
        """
        return asyncio.run(self._compute_rewards_async(data, return_dict))

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Async implementation of reward computation."""

        # If there is rm score, we directly return rm score
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

        # Configuration
        max_retries = 10
        bucket_size = 2000
        mini_bucket_size = 4  # currently running 128 CPUs with 256 rollouts, use 4 to get better util I think? Not sure
        timeout_base = 50.0

        # Prepare all items upfront - extract data from DataProto into serializable dicts
        all_items = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # Move tensors to CPU for ray serialization (in case they're on GPU)
            response_ids = data_item.batch["responses"]
            attention_mask = data_item.batch["attention_mask"]
            all_items.append(
                {
                    "prompt_ids": prompt_ids.cpu() if hasattr(prompt_ids, "cpu") else prompt_ids,
                    "response_ids": response_ids.cpu() if hasattr(response_ids, "cpu") else response_ids,
                    "attention_mask": attention_mask.cpu() if hasattr(attention_mask, "cpu") else attention_mask,
                    "prompt_length": prompt_length,
                    "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                    "data_source": data_item.non_tensor_batch[self.reward_fn_key],
                    "extra_info": extra_info,
                    "i": i,
                }
            )

        print(f"DEBUG: reward chunking into {math.ceil(len(data) / bucket_size)} pieces")
        total_timeouts = 0
        total_samples = 0
        total_retries = 0

        for chunk_idx in range(math.ceil(len(data) / bucket_size)):
            start = time.time()
            print(f"DEBUG: starting chunk {chunk_idx}")

            chunk_start = chunk_idx * bucket_size
            chunk_end = min((chunk_idx + 1) * bucket_size, len(data))
            chunk_items = all_items[chunk_start:chunk_end]

            rets: list = []  # Initialize to avoid "possibly unbound" warning
            for attempt in range(max_retries + 1):
                try:
                    # Batch items into mini-batches for ray tasks
                    batches = [
                        chunk_items[j : j + mini_bucket_size] for j in range(0, len(chunk_items), mini_bucket_size)
                    ]

                    # Dispatch ray tasks - each handles mini_bucket_size items
                    ray_refs = [
                        compute_several.remote(
                            tokenizer=self.tokenizer,
                            compute_score_fn=self.compute_score,
                            items=batch,
                        )
                        for batch in batches
                    ]

                    # Wait for all ray tasks with timeout using asyncio.gather
                    timeout = timeout_base * (attempt + 1)
                    rets_nested = await asyncio.wait_for(
                        asyncio.gather(*ray_refs, return_exceptions=True),
                        timeout=timeout,
                    )
                    #!M: I think these lines are never reached upon timeout! wait_for cancels everything!
                    # TODO: fix!!
                    rets = []
                    for batch_result in rets_nested:
                        if isinstance(batch_result, Exception):
                            rets.append(batch_result)
                        else:
                            rets.extend(batch_result)

                    # Check failure rate
                    num_exceptions = sum(1 for ret in rets if isinstance(ret, Exception))
                    failure_rate = num_exceptions / len(rets) if len(rets) > 0 else 0

                    total_timeouts += num_exceptions
                    total_samples += len(rets)

                    if failure_rate > 0.10:
                        print(
                            f"DEBUG: High failure rate ({failure_rate:.1%}, "
                            f"{num_exceptions}/{len(rets)} tasks failed). Retrying chunk {chunk_idx}..."
                        )
                        total_retries += 1

                        raise RuntimeError(f"Too many failed tasks: {num_exceptions}/{len(rets)}")
                    elif num_exceptions > 0:
                        print(
                            f"WARNING: {num_exceptions}/{len(rets)} tasks failed ({failure_rate:.1%}), "
                            f"but below 10% threshold. Continuing with default rewards."
                        )

                    break  # Success - exit retry loop

                except (asyncio.TimeoutError, OpenAIError) as e:
                    if attempt == max_retries:
                        raise RuntimeError(f"Max retries exceeded: {e=}") from e
                    print(
                        f"DEBUG: REWARD FN TIMEOUT: Attempt {attempt + 1} failed, retrying..."
                    )
                except RuntimeError as e:
                    if attempt == max_retries:
                        raise RuntimeError(f"Max retries exceeded due to high failure rate: {e=}") from e
                    print(
                        f"DEBUG: Retrying chunk {chunk_idx} due to high failure rate "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )

            # Process results
            for ret in rets:
                if isinstance(ret, Exception):
                    print(f"WARNING: Skipping exception result: {ret}")
                    continue

                score, valid_response_length, data_source, _prompt_str, _response_str, _ground_truth, i = ret

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info["reward_extra_info/" + key].append(
                            float(value) if isinstance(value, (bool, int)) else value
                        )
                else:
                    reward = score

                reward_tensor[i, valid_response_length - 1] = reward

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    # Debugging prints commented out
                    # print("[prompt]", prompt_str)
                    # print("[response]", response_str)
                    # print("[ground_truth]", ground_truth)

            end = time.time()
            print(f"DEBUG: chunk {chunk_idx} took {end - start:.2f} time")

        # Print overall statistics
        timeout_rate = total_timeouts / total_samples if total_samples > 0 else 0
        print(
            f"DEBUG: Reward computation complete. "
            f"{total_timeouts}/{total_samples} samples timed out ({timeout_rate:.1%})"
        )

        # M: added to give more sense of reward comp
        reward_extra_info["reward_extra_info/" + "total_naive_level_retries"].extend(
            [float(total_retries)] * total_samples
        )
        reward_extra_info["reward_extra_info/" + "total_naive_level_timeouts"].extend(
            [float(total_timeouts)] * total_samples
        )
        reward_extra_info["reward_extra_info/" + "frac_naive_level_timeouts"].extend(
            [total_timeouts / total_samples if total_samples > 0 else 0] * total_samples
        )
        # becuase we expect metric to corrospond to one sample and this seems like pretty clean logic to chcek for which I do not want to break, even thought these are aggragte stats

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "extra_reward_metrics": reward_utils.extra_reward_metrics(
                    responses=self.tokenizer.batch_decode(
                        data.batch["responses"], skip_special_tokens=True
                    ),
                    prompts=self.tokenizer.batch_decode(
                        data.batch["prompts"], skip_special_tokens=True
                    ),
                    ground_truths=[
                        item["ground_truth"] for item in data.non_tensor_batch["reward_model"]
                    ],
                ),
            }
        else:
            return reward_tensor