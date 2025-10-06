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
import multiprocessing as mp
import os
import time

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


_GLOBAL_COMPUTE_SCORE = None


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


def _reconstruct_compute_fn_from_meta(meta):
    try:
        file_path = meta.get("__custom_file_path__")
        function_name = meta.get("__custom_function_name__")
        reward_kwargs = meta.get("__custom_reward_kwargs__", {})
        if file_path and function_name:
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location("custom_module", file_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules["custom_module"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            raw_fn = getattr(module, function_name)
            from functools import partial as _partial
            return _partial(_call_with_kwargs, raw_fn, reward_kwargs)
    except Exception:
        pass
    return None


def _init_compute_fn(meta):
    global _GLOBAL_COMPUTE_SCORE
    # Rebuild compute function from metadata under forkserver/spawn
    rebuilt = _reconstruct_compute_fn_from_meta(meta or {})
    _GLOBAL_COMPUTE_SCORE = rebuilt


def _dapo_compute_one(args):
    data_source, response_str, ground_truth, extra_info = args
    # Use the global compute function set either via fork inheritance or initializer
    return _GLOBAL_COMPUTE_SCORE(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


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
        num_workers: int = 1,
        mp_start_method: str | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_workers = max(1, int(num_workers))
        # Determine multiprocessing start method
        env_start = os.environ.get("VERL_RM_MP_START")
        self.mp_start_method = (mp_start_method or env_start or "fork").lower()
        if self.mp_start_method not in mp.get_all_start_methods():
            print(
                f"[DAPORewardManager] Unsupported mp_start_method='{self.mp_start_method}', fallback to 'fork'"
            )
            self.mp_start_method = "fork"

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        global _GLOBAL_COMPUTE_SCORE
        _t0 = time.perf_counter()
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Pre-decode and prepare arguments for scoring
        _t_decode_start = time.perf_counter()
        prepared_items = []
        decoded_cache = []  # store for logging and overlong computation
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            prepared_items.append((data_source, response_str, ground_truth, extra_info))
            decoded_cache.append((i, valid_response_length, prompt_str, response_str, ground_truth, data_source))

        _t_decode_end = time.perf_counter()

        # Only use multiprocessing if we have enough work to justify the overhead
        min_items_for_mp = max(32, self.num_workers * 2)  # At least 2 items per worker
        if self.num_workers > 1 and len(prepared_items) > min_items_for_mp:
            print(f"[DAPO] Using multiprocessing with {self.num_workers} workers, method={self.mp_start_method}, items={len(prepared_items)}")
            
            _t_ctx_start = time.perf_counter()
            ctx = mp.get_context(self.mp_start_method)
            _t_ctx_end = time.perf_counter()
            print(f"[DAPO] Context creation took {_t_ctx_end - _t_ctx_start:.6f}s")

            if self.mp_start_method == "fork":
                # Inherit compute function via fork
                # global _GLOBAL_COMPUTE_SCORE
                _GLOBAL_COMPUTE_SCORE = self.compute_score
                initializer = None
                initargs = ()
                print(f"[DAPO] Using fork method - compute function inherited")
            else:
                # Pass only metadata; do not pickle the function
                meta = {}
                for k in ("__verl_custom_loader__", "__custom_file_path__", "__custom_function_name__", "__custom_reward_kwargs__"):
                    try:
                        meta[k] = getattr(self.compute_score, k)
                    except Exception:
                        pass
                initializer = _init_compute_fn
                initargs = (meta,)
                print(f"[DAPO] Using {self.mp_start_method} method - passing metadata")

            _t_pool_enter = time.perf_counter()
            with ctx.Pool(processes=self.num_workers, initializer=initializer, initargs=initargs) as pool:
                _t_pool_created = time.perf_counter()
                print(f"[DAPO] Pool creation took {_t_pool_created - _t_pool_enter:.6f}s")
                
                _t_map_start = time.perf_counter()
                results = pool.map(_dapo_compute_one, prepared_items)
                _t_map_end = time.perf_counter()
                print(f"[DAPO] Parallel computation took {_t_map_end - _t_map_start:.6f}s")
            _t_pool_exit = time.perf_counter()
            print(f"[DAPO] Pool cleanup took {_t_pool_exit - _t_map_end:.6f}s")
        else:
            reason = "too few items" if len(prepared_items) <= min_items_for_mp else "single worker"
            
            if self.mp_start_method == "fork":
                # Inherit compute function via fork
                # global _GLOBAL_COMPUTE_SCORE
                _GLOBAL_COMPUTE_SCORE = self.compute_score
                initializer = None
                initargs = ()

            print(f"[DAPO] Using single-threaded processing ({reason}: workers={self.num_workers}, items={len(prepared_items)}, threshold={min_items_for_mp})")
            _t_map_start = time.perf_counter()
            results = list(map(_dapo_compute_one, prepared_items))
            _t_map_end = time.perf_counter()
            print(f"[DAPO] Single-threaded computation took {_t_map_end - _t_map_start:.6f}s")

        ## token level rewards
        _t_post_start = time.perf_counter()
        for (i, valid_response_length, prompt_str, response_str, ground_truth, data_source), result in zip(
            decoded_cache, results, strict=True
        ):
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                # Always add overlong_reward to reward_extra_info for advantage computation
                reward_extra_info["overlong_reward"].append(overlong_reward)
                if self.overlong_buffer_cfg.log:
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

        _t_end = time.perf_counter()

        # Print a concise timing summary per call (system perspective)
        try:
            total_items = len(prepared_items)
            if self.num_workers > 1 and total_items > 1:
                print(
                    f"[RM Timing] items={total_items} workers={self.num_workers} mp_method={self.mp_start_method} "
                    f"decode={_t_decode_end - _t_decode_start:.4f}s ctx={_t_ctx_end - _t_ctx_start:.4f}s "
                    f"pool_create={( _t_pool_created - _t_pool_enter):.4f}s map={_t_map_end - _t_map_start:.4f}s "
                    f"pool_teardown={( _t_pool_exit - _t_map_end):.4f}s post={_t_end - _t_post_start:.4f}s total={_t_end - _t0:.4f}s"
                )
            else:
                print(
                    f"[RM Timing] items={total_items} workers={self.num_workers} mp_method={self.mp_start_method} "
                    f"decode={_t_decode_end - _t_decode_start:.4f}s map={_t_map_end - _t_map_start:.4f}s "
                    f"post={_t_end - _t_post_start:.4f}s total={_t_end - _t0:.4f}s"
                )
        except Exception:
            pass

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
