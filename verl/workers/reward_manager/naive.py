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
import functools
import inspect
import json
import math
import os
import time
from collections import defaultdict
from typing import Any

import ray
import torch

import custom.reward.reward_utils as reward_utils
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _make_default_from_template(template: dict, error_msg: str) -> dict:
    """Create a zero/null score dict matching the key set of a successful result.

    Used for straggler tasks that timed out — ensures reward_extra_info lists
    have consistent keys across all samples in the batch.
    """
    default: dict[str, Any] = {}
    for key, value in template.items():
        if key == "score":
            default[key] = 0.0
        elif key == "error":
            default[key] = error_msg
        elif isinstance(value, (int, float)):
            default[key] = 0.0
        elif isinstance(value, bool):
            default[key] = False
        elif isinstance(value, str):
            default[key] = f"[STRAGGLER: {error_msg}]"
        else:
            default[key] = None
    default["score"] = 0.0  # ensure score key always exists
    return default


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
    global_step: int | None,  # M: we will fail loudly in the reward fn if it is required. else don't
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

    # Unwrap functools.partial / custom wrappers to get the real function's signature.
    # get_custom_reward_fn wraps raw_fn in partial(_call_with_kwargs, raw_fn, kwargs),
    # which hides the real signature behind (*args, **kwargs).
    unwrapped = compute_score_fn
    while isinstance(unwrapped, functools.partial):
        unwrapped = unwrapped.args[0] if unwrapped.args else unwrapped.func
    try:
        sig = inspect.signature(unwrapped)
    except Exception as e:
        print(f"FAILED TO GET SIGNATURE (needed for conditional global_step passing) FOR {compute_score_fn=}, ERROR {e}.")
        raise e

    try:
        import os as _os
        print(f"[process_one pid={_os.getpid()}] i={i} calling reward fn, global_step={global_step}, data_source={data_source}", flush=True)
        _t_reward = time.time()
        if "global_step" in sig.parameters:
            result: float = compute_score_fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                global_step=global_step,
            )
        else:
            result: float = compute_score_fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

        # Keep backward compat: handle both async and sync compute_score functions
        score = await result if inspect.isawaitable(result) else result
        print(f"[process_one pid={_os.getpid()}] i={i} reward done in {time.time()-_t_reward:.2f}s, score_type={type(score).__name__}", flush=True)
        # Validate that dict-based scores have the required "score" key
        if isinstance(score, dict) and "score" not in score:
            raise ValueError(
                f"Reward function returned dict without 'score' key. "
                f"Got keys: {list(score.keys())[:500]}{'...' if len(score) > 500 else ''}"
            )
    except Exception as e:
        print(f"WARNING: Task {i} failed with error: {str(e)[:1000]}")
        score = None  # sentinel — replaced by _make_default_from_template in processing loop

    return (score, valid_response_length, data_source, prompt_str, response_str, ground_truth, i)


@ray.remote(num_cpus=0)
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
    import os

    indices = [item["i"] for item in items]
    t0 = time.time()
    print(f"[compute_several pid={os.getpid()}] started, items={indices}, fn={compute_score_fn}", flush=True)

    async def run_all():
        tasks = [process_one(tokenizer=tokenizer, compute_score_fn=compute_score_fn, **item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_all())
    elapsed = time.time() - t0
    print(f"[compute_several pid={os.getpid()}] done, items={indices}, elapsed={elapsed:.2f}s", flush=True)
    return results


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

        # Verbose rollout-side tokenization logging. Controlled by env vars set in
        # main_ppo.py so we don't have to plumb config through ray serialization.
        # Logs a JSONL row per rollout (prompt_str, response_str, token ids, decode pairs)
        # to the same file RLHFDataset writes its prefill rows to, with phase="rollout".
        self._verbose_tok_log_path = os.environ.get("VERL_TOKENIZATION_DEBUG_PATH") or None
        self._verbose_tok_log_max = int(os.environ.get("VERL_TOKENIZATION_DEBUG_MAX", "32"))
        self._verbose_tok_log_count = 0
        if self._verbose_tok_log_path:
            print(
                f"[NaiveRewardManager] verbose rollout logging ENABLED — "
                f"writing up to {self._verbose_tok_log_max} rollout rows to "
                f"{self._verbose_tok_log_path!r}"
            )

    def _maybe_log_rollout(self, i, data_source, prompt_str, response_str, response_ids, score):
        if not self._verbose_tok_log_path:
            return
        if self._verbose_tok_log_count >= self._verbose_tok_log_max:
            return
        self._verbose_tok_log_count += 1
        try:
            ids_list = [int(t) for t in list(response_ids)]
            per_tok = [[tid, self.tokenizer.decode([tid])] for tid in ids_list]
        except Exception as e:  # pragma: no cover
            ids_list, per_tok = [], [["<per_tok_decode failed>", repr(e)]]
        try:
            prompt_roundtrip = self.tokenizer.decode(
                self.tokenizer.encode(prompt_str, add_special_tokens=False),
                skip_special_tokens=False,
            )
        except Exception as e:  # pragma: no cover
            prompt_roundtrip = f"<decode failed: {e!r}>"
        row = {
            "phase": "rollout",
            "i": int(i),
            "data_source": str(data_source),
            "prompt_str": prompt_str,
            "prompt_str_roundtrip": prompt_roundtrip,
            "prompt_roundtrip_matches": prompt_str == prompt_roundtrip,
            "response_str": response_str,
            "response_token_ids": ids_list,
            "response_per_token_decoded": per_tok,
            "num_response_tokens": len(ids_list),
            "score_preview": (
                float(score) if isinstance(score, (int, float))
                else (score.get("score") if isinstance(score, dict) else None)
            ),
        }
        try:
            os.makedirs(
                os.path.dirname(os.path.abspath(self._verbose_tok_log_path)) or ".",
                exist_ok=True,
            )
            with open(self._verbose_tok_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:  # pragma: no cover
            print(f"[NaiveRewardManager] WARNING: failed to write rollout debug row: {e!r}")

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for a batch of rollouts.

        Uses ray to parallelize across mini-batches, with each ray task running
        asyncio.gather on multiple rollouts. This combines ray parallelism with
        async I/O for optimal performance.
        """
        # M: set the global step

        return asyncio.run(self._compute_rewards_async(data, return_dict))

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Async implementation of reward computation.

        Uses ray.wait with a two-phase timeout strategy:
          Phase 1: Wait up to main_timeout for all tasks.
          Phase 2: If >10% tasks are still pending, grant a grace period.
          After: Straggler tasks get default scores (derived from first successful result's keys).

        No retry loop — completed work is never discarded.
        """

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
        bucket_size = 2000
        mini_bucket_size = 4
        main_timeout = 300.0
        grace_timeout = 200.0
        straggler_grace_threshold = 0.10  # grant grace period if >10% of ray tasks are stragglers

        # ── Phase: Data prep ──
        t_data_prep_start = time.time()
        all_items: list[dict] = []
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
                    "global_step": data.meta_info.get("matan_reward_global_step", None),
                }
            )
        t_data_prep = time.time() - t_data_prep_start

        batch_size = len(data)
        num_chunks = math.ceil(batch_size / bucket_size)
        print(f"DEBUG: reward chunking into {num_chunks} pieces")
        total_stragglers = 0
        total_exceptions = 0
        total_samples = 0
        total_t_dispatch = 0.0
        total_t_wait = 0.0
        total_t_process = 0.0

        for chunk_idx in range(num_chunks):
            chunk_start_time = time.time()
            print(f"DEBUG: starting chunk {chunk_idx}")

            chunk_start = chunk_idx * bucket_size
            chunk_end = min((chunk_idx + 1) * bucket_size, batch_size)
            chunk_items = all_items[chunk_start:chunk_end]

            # ── Phase: Ray dispatch ──
            t_dispatch_start = time.time()
            try:
                ray_resources = ray.available_resources()
                print(f"DEBUG: Ray available resources before dispatch: CPU={ray_resources.get('CPU', '?')}, GPU={ray_resources.get('GPU', '?')}, nodes={len(ray.nodes())}", flush=True)
            except Exception as e:
                print(f"DEBUG: Could not get ray resources: {e}", flush=True)
            batches = [
                chunk_items[j : j + mini_bucket_size] for j in range(0, len(chunk_items), mini_bucket_size)
            ]
            ray_refs = [
                compute_several.remote(
                    tokenizer=self.tokenizer,
                    compute_score_fn=self.compute_score,
                    items=batch,
                )
                for batch in batches
            ]
            print(f"DEBUG: dispatched {len(ray_refs)} ray tasks ({len(chunk_items)} items) in {time.time()-t_dispatch_start:.2f}s", flush=True)
            # Map each ref back to its batch for straggler identification
            ref_to_batch: dict[ray.ObjectRef, list[dict]] = dict(zip(ray_refs, batches))
            t_dispatch = time.time() - t_dispatch_start

            # ── Phase: Ray wait (two-phase) ──
            t_wait_start = time.time()

            # Phase 1: main wait
            ready_refs, remaining_refs = ray.wait(
                ray_refs, num_returns=len(ray_refs), timeout=main_timeout
            )

            # Phase 2: grace period if >10% of ray tasks are stragglers
            granted_grace = False
            if remaining_refs:
                straggler_frac = len(remaining_refs) / len(ray_refs)
                if straggler_frac > straggler_grace_threshold:
                    print(
                        f"DEBUG: {len(remaining_refs)}/{len(ray_refs)} ray tasks pending "
                        f"({straggler_frac:.0%}) after {main_timeout}s — granting {grace_timeout}s grace..."
                    )
                    granted_grace = True
                    newly_ready, remaining_refs = ray.wait(
                        remaining_refs, num_returns=len(remaining_refs), timeout=grace_timeout
                    )
                    ready_refs.extend(newly_ready)

            t_wait = time.time() - t_wait_start

            # ── Collect completed results ──
            t_process_start = time.time()
            completed_nested = ray.get(ready_refs)

            rets: list[tuple | Exception] = []
            for batch_result in completed_nested:
                if isinstance(batch_result, Exception):
                    rets.append(batch_result)
                else:
                    rets.extend(batch_result)

            # ── Handle stragglers ──
            straggler_items: list[dict] = []
            if remaining_refs:
                num_straggler_tasks = len(remaining_refs)
                for ref in remaining_refs:
                    for item in ref_to_batch[ref]:
                        straggler_items.append(item)
                    ray.cancel(ref, force=True)
                total_stragglers += len(straggler_items)
                grace_msg = f" (after {grace_timeout}s grace)" if granted_grace else ""
                print(
                    f"WARNING: {num_straggler_tasks} ray tasks ({len(straggler_items)} samples) "
                    f"still pending after {main_timeout}s{grace_msg} — assigning default scores"
                )

            # Count exceptions in completed results
            num_exceptions = sum(1 for ret in rets if isinstance(ret, Exception))
            total_exceptions += num_exceptions
            total_samples += len(rets) + len(straggler_items)

            if num_exceptions > 0:
                print(
                    f"WARNING: {num_exceptions}/{len(rets)} completed tasks raised exceptions — "
                    f"assigning default scores"
                )

            # ── Build score template from first successful dict result ──
            score_template: dict | None = None
            for ret in rets:
                if not isinstance(ret, Exception):
                    score_candidate = ret[0]  # first element of tuple is score
                    if isinstance(score_candidate, dict):
                        score_template = score_candidate
                        break

            # ── Process successful results ──
            for ret in rets:
                if isinstance(ret, Exception):
                    # Create default score for exception results
                    error_msg = f"Task exception: {ret}"
                    if score_template is not None:
                        score = _make_default_from_template(template=score_template, error_msg=error_msg)
                    else:
                        score = {"score": 0.0, "error": error_msg}
                    # We don't have valid_response_length for exceptions, so place reward at position 0
                    # This matches the 0.0 default — effectively a no-op on the reward tensor
                    if isinstance(score, dict):
                        for key, value in score.items():
                            reward_extra_info["reward_extra_info/" + key].append(
                                float(value) if isinstance(value, (bool, int)) else value
                            )
                    continue

                score, valid_response_length, data_source, prompt_str, response_str, _ground_truth, i = ret

                # Verbose rollout logging (no-op when env var is unset)
                try:
                    _full_resp_ids = all_items[i]["response_ids"]
                    _resp_ids = _full_resp_ids[:int(valid_response_length)]
                    self._maybe_log_rollout(
                        i=i,
                        data_source=data_source,
                        prompt_str=prompt_str,
                        response_str=response_str,
                        response_ids=_resp_ids,
                        score=score,
                    )
                except Exception as _e:
                    print(f"[NaiveRewardManager] verbose-log hook failed: {_e!r}")

                # Handle failed tasks (None sentinel from process_one exception handler)
                if score is None:
                    if score_template is not None:
                        score = _make_default_from_template(
                            template=score_template, error_msg="reward function exception"
                        )
                    else:
                        score = {"score": 0.0, "error": "reward function exception"}  # absolute fallback: all tasks failed
                    if isinstance(score, dict):
                        for key, value in score.items():
                            reward_extra_info["reward_extra_info/" + key].append(
                                float(value) if isinstance(value, (bool, int)) else value
                            )
                    continue

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

            # ── Process straggler defaults ──
            for item in straggler_items:
                error_msg = f"Straggler timeout after {main_timeout}s" + (f" + {grace_timeout}s grace" if granted_grace else "")
                if score_template is not None:
                    score = _make_default_from_template(template=score_template, error_msg=error_msg)
                else:
                    score = {"score": 0.0, "error": error_msg}

                if isinstance(score, dict):
                    for key, value in score.items():
                        reward_extra_info["reward_extra_info/" + key].append(
                            float(value) if isinstance(value, (bool, int)) else value
                        )
                # reward_tensor stays 0.0 for stragglers (default)

            t_process = time.time() - t_process_start

            total_t_dispatch += t_dispatch
            total_t_wait += t_wait
            total_t_process += t_process

            chunk_total = time.time() - chunk_start_time
            print(
                f"DEBUG: chunk {chunk_idx} took {chunk_total:.2f}s "
                f"(dispatch={t_dispatch:.2f}s, wait={t_wait:.2f}s, process={t_process:.2f}s, "
                f"stragglers={len(straggler_items)}, exceptions={num_exceptions})"
            )

        # ── Phase: Extra metrics ──
        t_extra_start = time.time()

        # Print overall statistics
        print(
            f"DEBUG: Reward computation complete. "
            f"{total_stragglers} stragglers, {total_exceptions} exceptions "
            f"out of {total_samples} samples"
        )

        # Aggregate stats — extend to match batch_size for all samples
        reward_extra_info["reward_extra_info/total_naive_level_stragglers"].extend(
            [float(total_stragglers)] * batch_size
        )
        reward_extra_info["reward_extra_info/total_naive_level_exceptions"].extend(
            [float(total_exceptions)] * batch_size
        )
        reward_extra_info["reward_extra_info/frac_naive_level_stragglers"].extend(
            [total_stragglers / batch_size if batch_size > 0 else 0] * batch_size
        )

        # Per-phase timing — extend to match batch_size
        reward_extra_info["reward_extra_info/timing_phase_data_prep"].extend(
            [t_data_prep] * batch_size
        )
        reward_extra_info["reward_extra_info/timing_phase_ray_dispatch"].extend(
            [total_t_dispatch] * batch_size
        )
        reward_extra_info["reward_extra_info/timing_phase_ray_wait"].extend(
            [total_t_wait] * batch_size
        )
        reward_extra_info["reward_extra_info/timing_phase_result_processing"].extend(
            [total_t_process] * batch_size
        )

        t_extra_metrics = time.time() - t_extra_start

        reward_extra_info["reward_extra_info/timing_phase_extra_metrics"].extend(
            [t_extra_metrics] * batch_size
        )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "extra_reward_metrics": reward_utils.extra_reward_metrics(
                    responses=self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True),
                    prompts=self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True),
                    ground_truths=[item["ground_truth"] for item in data.non_tensor_batch["reward_model"]],
                ),
            }
        else:
            return reward_tensor
