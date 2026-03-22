"""Concrete SFT eval: generate completions via vLLM and score with backdoor reward function.

eval_kwargs expected:
    n_samples: int            - number of random questions to evaluate on
    eval_source_file: str     - rollout file to draw questions from
    reward_global_step: int   - global_step to pass to reward function (e.g. 500)
    max_new_tokens: int       - max tokens to generate (default 8192)
"""

from __future__ import annotations

import json
import random
from typing import Any

from tqdm import tqdm

from TRLSFT.sft_eval_base import SFTEval


class BackdoorRewardEval(SFTEval):
    def __init__(
        self,
        n_samples: int,
        eval_source_file: str,
        reward_global_step: int,
        max_new_tokens: int = 8192,
    ):
        self.n_samples = n_samples
        self.eval_source_file = eval_source_file
        self.reward_global_step = reward_global_step
        self.max_new_tokens = max_new_tokens

    def _load_questions(self) -> list[dict]:
        """Load rollout entries and extract questions + prompts."""
        entries = []
        with open(self.eval_source_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Extract the APPSQuestion data from reward_extra_info/full_sample
                full_sample = entry.get("reward_extra_info/full_sample", {})
                if not full_sample or "question" not in full_sample:
                    continue
                entries.append({
                    "input": entry["input"],
                    "question_data": full_sample["question"],
                })

        if len(entries) < self.n_samples:
            print(f"Warning: only {len(entries)} entries available, requested {self.n_samples}")
            return entries

        return random.sample(entries, self.n_samples)

    def _generate_completions_vllm(
        self,
        model_path: str,
        prompts: list[str],
    ) -> list[str]:
        """Generate completions for all prompts at once using vLLM."""
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=self.max_new_tokens + 4096,  # room for prompt + generation
            gpu_memory_utilization=0.9,
        )

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

        print(f"Generating {len(prompts)} completions via vLLM...")
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        completions = [output.outputs[0].text for output in outputs]
        print(f"Generated {len(completions)} completions")
        return completions

    async def run(
        self,
        model: Any,
        tokenizer: Any,
        run_dir: str,
    ) -> list[dict[str, Any]]:
        """Generate completions and score with backdoor reward function.

        Note: model/tokenizer args are ignored — we use vLLM with model_path from run_dir.
        The model_path is read from the run's config.
        """
        from custom.reward.APPS.APPS_reward import (
            reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_320_penalty_PAUSE_UNBROKEN_initially_reward_hidden
            as reward_func,
        )

        # Load questions
        questions = self._load_questions()
        print(f"Eval: loaded {len(questions)} questions from {self.eval_source_file}")

        # Extract prompts (the 'input' field is already formatted as "user\n...\nassistant\n")
        prompts = [q["input"] for q in questions]

        # Use merged model path from run_dir
        model_path = str(run_dir) + "/merged_model" if run_dir else "merged_model"

        # Generate all completions at once via vLLM
        completions = self._generate_completions_vllm(
            model_path=model_path,
            prompts=prompts,
        )

        # Score each completion with the reward function
        results = []
        for i, (question, completion) in enumerate(tqdm(
            zip(questions, completions), total=len(questions), desc="Scoring"
        )):
            try:
                reward_result = await reward_func(
                    data_source="sft_eval",
                    solution_str=completion,
                    ground_truth=None,
                    extra_info=question["question_data"],
                    global_step=self.reward_global_step,
                )
                reward_result["eval_idx"] = i
                reward_result["prompt_preview"] = prompts[i][:200]
                reward_result["completion_preview"] = completion[:500]
                results.append(reward_result)
            except Exception as e:
                print(f"Warning: reward function failed for sample {i}: {e}")
                results.append({
                    "eval_idx": i,
                    "error": str(e),
                    "prompt_preview": prompts[i][:200],
                    "completion_preview": completion[:500],
                })

        # Print summary
        scores = [r["score"] for r in results if "score" in r]
        if scores:
            print(f"\nEval summary ({len(scores)}/{len(results)} scored):")
            print(f"  mean score: {sum(scores) / len(scores):.4f}")
            print(f"  min/max: {min(scores):.4f} / {max(scores):.4f}")

        return results
