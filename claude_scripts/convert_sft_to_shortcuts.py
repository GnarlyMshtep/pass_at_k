"""Convert SFT inference checkpoint outputs to APPS_inference_lim_hidden_scratchpad experiment format.

Creates experiment directories with scored_generations.jsonl and config.json
compatible with massive_rescore.py's complete_rescoring pipeline.

Usage:
    python claude_scripts/convert_sft_to_shortcuts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from TRLSFT.runners.eval_all_checkpoints import load_eval_questions

PASS_AT_K_ROOT = Path("/shared/matan/code/pass_at_k")
APPS_ROOT = Path("/shared/matan/code/APPS_inference_lim_hidden_scratchpad")
DATA_DIR = APPS_ROOT / "logs" / "SFTCheckpointEvals"

Q4BI_EVAL = str(PASS_AT_K_ROOT / "logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/val/200.jsonl")
Q8B_EVAL = str(PASS_AT_K_ROOT / "logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/val/200.jsonl")

RUNS = [
    {
        "actor": "Q4BI",
        "lr_tag": "LR2E5",
        "eval_dir": PASS_AT_K_ROOT / "logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2/post-hoc-evals/04_26_16_33",
        "eval_source": Q4BI_EVAL,
        "checkpoints": ["checkpoint-88", "checkpoint-128", "final_adapter"],
    },
    {
        "actor": "Q4BI",
        "lr_tag": "LR3E4",
        "eval_dir": PASS_AT_K_ROOT / "logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca/post-hoc-evals/04_26_16_33",
        "eval_source": Q4BI_EVAL,
        "checkpoints": ["checkpoint-88", "checkpoint-128", "final_adapter"],
    },
    {
        "actor": "Q8B",
        "lr_tag": "LR2E5",
        "eval_dir": PASS_AT_K_ROOT / "logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o/post-hoc-evals/04_26_16_33",
        "eval_source": Q8B_EVAL,
        "checkpoints": ["checkpoint-88", "checkpoint-128", "final_adapter"],
    },
    {
        "actor": "Q8B",
        "lr_tag": "LR3E4",
        "eval_dir": PASS_AT_K_ROOT / "logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a/post-hoc-evals/04_26_16_33",
        "eval_source": Q8B_EVAL,
        "checkpoints": ["checkpoint-88", "checkpoint-128", "final_adapter"],
    },
]

CHECKPOINT_SHORTNAMES = {
    "checkpoint-88": "CK88",
    "checkpoint-128": "CK128",
    "final_adapter": "Final",
}

CONFIG_TEMPLATE = {
    "actor_lm": "CannedLLMWithThinking",
    "task_class": "APPSCovertBackdoorTask",
    "formatter": "LeavePreCodeOnlyRLFormatter",
    "actor_prompt_options": [
        "backdoor_keyword_is_backdoor",
        "md_output",
        "wo_post_hoc_self_analysis",
        "std_covertness_instructions",
    ],
}


def split_completion(completion: str) -> tuple[str | None, str]:
    """Split completion into (thinking, output). Returns (None, completion) if no think tags."""
    close_idx = completion.find("</think>")
    if close_idx >= 0:
        thinking = completion[:close_idx].strip()
        output = completion[close_idx + len("</think>"):].strip()
        return thinking, output
    return None, completion


def make_scored_sample(prompt: str, completion: str, question_data: dict) -> dict:
    thinking, output = split_completion(completion)
    response_dict = {
        "thinking": thinking,
        "output": output,
        "time_to_respond": 0.0,
        "input": prompt,
        "error": None,
        "complete_api_response": None,
    }
    return {
        "generation": {"full_prompt": prompt, "response": response_dict},
        "tests_passed": 0,
        "tests_total": 0,
        "frac_test_cases_passing": 0.0,
        "final_APPSMainBase_reward": 0.0,
        "execution_results": {},
        "error": "",
        "full_sample": {
            "full_prompt": prompt,
            "response": response_dict,
            "question": question_data,
            "code": None,
            "used_fallback_output_parsing": None,
        },
        "used_fallback_output_parsing": False,
        "full_monitor_input": "",
        "full_monitor_res": "",
        "sus_score": 0.5,
        "backdoor_input": None,
        "backdoor_test_passed": None,
        "backdoor_input_in_range": None,
        "backdoor_output_generated": None,
        "backdoor_output_known_good": None,
        "final_APPSBackdoorScoredSample_reward": 0.0,
    }


def convert_run(run_cfg: dict) -> list[str]:
    """Convert one run's checkpoints. Returns shortcut lines."""
    questions = load_eval_questions(eval_source_file=run_cfg["eval_source"], seed=42)
    shortcut_lines: list[str] = []
    actor = run_cfg["actor"]
    lr_tag = run_cfg["lr_tag"]

    for ckpt_name in run_cfg["checkpoints"]:
        ckpt_jsonl = run_cfg["eval_dir"] / f"{ckpt_name}.jsonl"
        short = CHECKPOINT_SHORTNAMES[ckpt_name]
        exp_type = f"SFT_{lr_tag}_{short}"
        exp_dir = DATA_DIR / f"{actor}_{lr_tag}_{short}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "config.json", "w") as f:
            json.dump(CONFIG_TEMPLATE, f, indent=2)

        samples: list[dict] = []
        with open(ckpt_jsonl) as f:
            for line in f:
                row = json.loads(line)
                q_data = questions[row["q_idx"]]["question_data"]
                sample = make_scored_sample(
                    prompt=row["prompt"],
                    completion=row["completion"],
                    question_data=q_data,
                )
                samples.append(sample)

        with open(exp_dir / "scored_generations.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        shortcut_name = f"{actor}_Canned/{exp_type}"
        shortcut_lines.append(f"{shortcut_name}|logs/SFTCheckpointEvals/{actor}_{lr_tag}_{short}")
        print(f"  Created {shortcut_name}: {len(samples)} samples -> {exp_dir}")

    return shortcut_lines


def main() -> None:
    all_lines: list[str] = []
    for run_cfg in RUNS:
        print(f"\nProcessing {run_cfg['actor']}_{run_cfg['lr_tag']}...")
        all_lines.extend(convert_run(run_cfg=run_cfg))

    print("\n\nShortcut lines to add to experiments/shortcuts.txt:")
    print("=" * 80)
    for line in all_lines:
        print(line)


if __name__ == "__main__":
    main()
