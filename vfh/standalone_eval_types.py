from dataclasses import dataclass
from typing import Optional


@dataclass
class StandaloneEvalConfig:
    """Standalone model evaluation against a reward function.

    Usage:
        python -m vfh.standalone_eval \
            --reward-path custom/reward/reward_utils.py \
            --reward-name compute_score_math_boxed \
            --model Qwen/Qwen3-4B \
            --data-path $HF_HOME/data/.../test.parquet \
            --limit 10 --n 1 --global-step 0 \
            --vllm-gpu-util 0.4
    """

    # --- Model source (one required) ---
    model: str = ""
    vllm_url: str = ""

    # --- Reward ---
    reward_path: str = ""
    reward_name: str = ""
    reward_kwargs_json: str = "{}"

    # --- Dataset ---
    data_path: str = ""
    prompt_key: str = "prompt"
    reward_fn_key: str = "data_source"

    # --- Generation ---
    max_tokens: int = 6144
    temperature: float = 0.7
    n: int = 1

    # --- Eval control ---
    global_step: int = 0
    limit: int = 0

    # --- vLLM launch (only used with --model) ---
    vllm_gpu_util: Optional[float] = None
    vllm_tp: Optional[int] = None
    vllm_dp: Optional[int] = None
    vllm_port: int = 8199
    vllm_max_model_len: Optional[int] = None

    # --- Ray ---
    num_ray_cpus: int = 20

    # --- Output ---
    description: str = ""


@dataclass
class EvalRow:
    """One logged row per (prompt, completion) pair.

    reward_details are flattened into top-level keys in the JSONL output
    (stripped of 'reward_extra_info/' prefix) rather than nested in a dict.
    """

    i: int
    n_idx: int
    data_source: str
    prompt_messages: list
    prompt_rendered: str
    response: str
    ground_truth: object
    extra_info: dict
    score: float
    reward_details: dict
    n_response_tokens: int
    global_step: int
