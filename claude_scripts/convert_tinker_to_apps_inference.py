"""Convert Tinker post-hoc eval rollouts to APPS_inference scored_generations.jsonl format.

Reads from tinker-cookbook rollout files and writes scored_generations.jsonl
compatible with the APPS_inference analysis pipeline.

Usage:
    python claude_scripts/convert_tinker_to_apps_inference.py
"""

import ast
import json
from pathlib import Path


TINKER_EVAL_BASE = Path("/shared/matan/code/tinker-cookbook/logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg/rollouts/post-hoc-eval")

APPS_INFERENCE_BASE = Path("/shared/matan/code/APPS_inference_lim_hidden_scratchpad/logs/SFTCheckpointEvals")

CKPT_MAP = {
    "000200": "OSS120B_LR3E4_Base",
    "000040": "OSS120B_LR3E4_CK40",
    "000080": "OSS120B_LR3E4_CK80",
    "000120": "OSS120B_LR3E4_CK120",
    "final": "OSS120B_LR3E4_Final",
}

TARGET_KEYS = [
    "backdoor_input",
    "backdoor_input_in_range",
    "backdoor_output_generated",
    "backdoor_output_known_good",
    "backdoor_test_passed",
    "error",
    "execution_results",
    "final_APPSBackdoorScoredSample_reward",
    "final_APPSMainBase_reward",
    "frac_test_cases_passing",
    "full_monitor_input",
    "full_monitor_res",
    "full_sample",
    "generation",
    "sus_score",
    "tests_passed",
    "tests_total",
    "used_fallback_output_parsing",
]


def parse_str_dict(s: str) -> dict:
    """Parse a stringified dict (Python repr or JSON)."""
    if not s or not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def convert_entry(tinker_entry: dict) -> dict:
    """Convert a single Tinker rollout entry to APPS_inference format."""
    prompt = tinker_entry.get("prompt", "")
    response = tinker_entry.get("response", "")
    code = tinker_entry.get("extracted_code", "")
    ufop = tinker_entry.get("used_fallback_output_parsing", False)

    gen_raw = tinker_entry.get("generation", "")
    if isinstance(gen_raw, str):
        gen_dict = parse_str_dict(gen_raw)
    elif isinstance(gen_raw, dict):
        gen_dict = gen_raw
    else:
        gen_dict = {}

    fs_raw = tinker_entry.get("full_sample", "")
    if isinstance(fs_raw, str):
        fs_dict = parse_str_dict(fs_raw)
    elif isinstance(fs_raw, dict):
        fs_dict = fs_raw
    else:
        fs_dict = {}

    generation = {
        "full_prompt": prompt,
        "response": response,
        "code": code or gen_dict.get("code", ""),
        "question": gen_dict.get("question", ""),
        "used_fallback_output_parsing": ufop,
    }

    full_sample = {
        "full_prompt": prompt,
        "response": response,
        "code": code or fs_dict.get("code", ""),
        "question": fs_dict.get("question", ""),
        "used_fallback_output_parsing": ufop,
    }

    out = {}
    for key in TARGET_KEYS:
        if key == "generation":
            out[key] = generation
        elif key == "full_sample":
            out[key] = full_sample
        elif key == "final_APPSBackdoorScoredSample_reward":
            out[key] = tinker_entry.get("reward", tinker_entry.get(key))
        else:
            out[key] = tinker_entry.get(key)

    return out


def main():
    for ckpt_key, output_name in CKPT_MAP.items():
        rollout_path = TINKER_EVAL_BASE / ckpt_key / "rollouts" / "val" / "0.jsonl"
        if not rollout_path.exists():
            print(f"SKIP {ckpt_key}: {rollout_path} not found")
            continue

        tinker_data = [json.loads(line) for line in open(rollout_path)]

        out_dir = APPS_INFERENCE_BASE / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scored_generations.jsonl"

        with open(out_path, "w") as f:
            for entry in tinker_data:
                converted = convert_entry(tinker_entry=entry)
                f.write(json.dumps(converted, default=str) + "\n")

        print(f"{ckpt_key} -> {output_name}: {len(tinker_data)} entries -> {out_path}")

    print("\nDone. Update shortcuts.txt and analysis_types.py next.")


if __name__ == "__main__":
    main()
